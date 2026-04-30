import unittest
from argparse import Namespace
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.rolling_train import (
    _compute_rolling_window_indices,
    _compute_validation_topk_summary,
    _prepare_opportunity_training_context,
    _run_lgbm_window,
    generate_prediction_bundle,
)
from src.feature_selection import apply_cross_sectional_rank, apply_feature_transforms
from src.rolling_types import RollingPaths, RollingRuntimeData


class RollingTrainTest(unittest.TestCase):
    def test_compute_validation_topk_summary_aggregates_daily_topk_returns(self):
        predictions = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2], dtype=np.float32)
        labels = pd.Series([0.10, -0.05, -0.10, 0.03, 0.02, -0.04], dtype=float)
        dates = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ]
        )

        summary = _compute_validation_topk_summary(predictions, labels, dates, topk=2)

        self.assertEqual(summary["valid_topk_days"], 2)
        self.assertAlmostEqual(float(summary["valid_top1_label_mean"]), 0.065, places=8)
        self.assertAlmostEqual(float(summary["valid_top1_positive_rate"]), 1.0, places=8)
        self.assertAlmostEqual(float(summary["valid_topk_label_mean"]), 0.025, places=8)
        self.assertAlmostEqual(float(summary["valid_topk_label_median"]), 0.025, places=8)
        self.assertAlmostEqual(float(summary["valid_topk_min_label_mean"]), -0.015, places=8)
        self.assertAlmostEqual(float(summary["valid_topk_positive_rate"]), 0.75, places=8)
        self.assertAlmostEqual(float(summary["valid_topk_excess_mean"]), 0.031666667, places=8)

    def test_compute_validation_topk_summary_handles_empty_input(self):
        summary = _compute_validation_topk_summary(
            np.array([], dtype=np.float32),
            pd.Series(dtype=float),
            pd.Series(dtype="datetime64[ns]"),
            topk=3,
        )

        self.assertEqual(summary["valid_topk_days"], 0)
        self.assertTrue(np.isnan(summary["valid_topk_label_mean"]))

    def test_compute_rolling_window_indices_applies_label_embargo(self):
        calendar = pd.Series(pd.to_datetime([f"2024-01-{day:02d}" for day in range(1, 9)]))

        bounds = _compute_rolling_window_indices(
            calendar,
            pd.Timestamp("2024-01-08"),
            train_days=2,
            valid_days=2,
            label_embargo_days=3,
        )

        self.assertEqual(bounds, (0, 1, 2, 3))
        self.assertLess(3 + 2 + 1, int(calendar.searchsorted(pd.Timestamp("2024-01-08"))))

    def test_prepare_opportunity_training_context_requires_industry_groups(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-01"])
        runtime_data = RollingRuntimeData(
            factor_frame=pd.DataFrame({"date": dates, "symbol": ["A", "B"], "f1": [1.0, 2.0]}),
            dt_index=pd.Series(dates),
            y=np.array([0.01, -0.01], dtype=np.float32),
            backtest_y=np.array([0.01, -0.01], dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-01"])),
            test_start=pd.Timestamp("2024-01-01"),
            test_end=pd.Timestamp("2024-01-01"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-01"])),
            selected_feature_names=["f1"],
            selected_feature_sources=["f1"],
            finite_feature_mask=np.array([True, True]),
            lookback=20,
            batch_size=2,
        )
        cfg = {
            "data": {"source": "tushare"},
            "label": {"opportunity": {"mode": "industry_excess"}},
        }

        with patch(
            "src.rolling_train.load_instrument_industry_groups",
            side_effect=ValueError("Industry group mapping unavailable"),
        ) as load_groups:
            with self.assertRaisesRegex(ValueError, "Industry group mapping unavailable"):
                _prepare_opportunity_training_context(cfg, runtime_data, signal_horizon=10)
        self.assertTrue(load_groups.call_args.kwargs["required"])

    def test_lgbm_window_reuses_precomputed_training_rank_frame(self):
        dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
        factor_frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["A", "B", "C"] * 3,
                "f1": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0],
                "f2": [3.0, 2.0, 1.0, 6.0, 4.0, 2.0, 9.0, 6.0, 3.0],
            }
        )
        ranked_training_feature_frame = apply_cross_sectional_rank(
            factor_frame[["f1", "f2"]],
            pd.Series(dates),
        )
        runtime_data = RollingRuntimeData(
            factor_frame=factor_frame,
            dt_index=pd.Series(dates),
            y=np.array([0.01, 0.02, 0.03, 0.02, 0.04, 0.06, 0.03, 0.06, 0.09], dtype=np.float32),
            backtest_y=np.zeros(9, dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            test_start=pd.Timestamp("2024-01-03"),
            test_end=pd.Timestamp("2024-01-03"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-03"])),
            selected_feature_names=["f1", "f2"],
            selected_feature_sources=["f1", "f2"],
            finite_feature_mask=np.array([True] * 9),
            lookback=20,
            batch_size=3,
            ranked_training_feature_frame=ranked_training_feature_frame,
        )
        cfg = {
            "data": {"source": "tushare"},
            "features": {
                "transforms": {
                    "cross_sectional_rank": True,
                    "cross_sectional_rank_exclude_columns": [],
                }
            },
            "strategy": {"topk": 1},
            "lgbm": {"early_stop": 0},
        }
        captured: dict[str, pd.DataFrame] = {}

        class FakeNativeLGBM:
            def __init__(self, **kwargs):
                self.feature_names = []

            def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
                self.feature_names = X_train.columns.tolist()
                captured["X_train"] = X_train.copy()
                captured["X_valid"] = X_valid.copy()
                return self

            def predict(self, X):
                captured["last_predict_X"] = X.copy()
                return np.linspace(0.0, 1.0, len(X), dtype=np.float32)

            def save_feature_importance(self, save_path):
                path = Path(save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.get_feature_importance_frame("gain").to_csv(path, index=False)
                return path

            def get_feature_importance_frame(self, importance_type="gain"):
                return pd.DataFrame({"feature": self.feature_names, importance_type: [1.0] * len(self.feature_names)})

            def save_training_history(self, save_path):
                return None

            def get_training_summary(self):
                return {"num_iterations": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = RollingPaths(
                results_dir=base,
                models_dir=base / "models",
                importance_dir=base / "feature_importance",
                training_history_dir=base / "training_history",
                prediction_artifact_dir=base / "prediction_artifacts",
            )
            for path in [paths.results_dir, paths.importance_dir, paths.training_history_dir]:
                path.mkdir(parents=True, exist_ok=True)
            with patch("src.models.pure_lightgbm.NativeLGBM", FakeNativeLGBM):
                with patch("src.rolling_train.apply_feature_transforms", wraps=apply_feature_transforms) as transforms:
                    pred_series, importance_df, training_summary = _run_lgbm_window(
                        cfg,
                        runtime_data,
                        train_mask=(runtime_data.dt_index == pd.Timestamp("2024-01-01")).to_numpy(),
                        valid_mask=(runtime_data.dt_index == pd.Timestamp("2024-01-02")).to_numpy(),
                        test_mask=(runtime_data.dt_index == pd.Timestamp("2024-01-03")).to_numpy(),
                        current_test_start=pd.Timestamp("2024-01-03"),
                        current_test_end=pd.Timestamp("2024-01-03"),
                        train_start=pd.Timestamp("2024-01-01"),
                        train_end=pd.Timestamp("2024-01-01"),
                        valid_start=pd.Timestamp("2024-01-02"),
                        valid_end=pd.Timestamp("2024-01-02"),
                        signal_horizon=5,
                        paths=paths,
                        load_models=False,
                        save_models=False,
                    )

        expected_train = pd.DataFrame(
            {
                "f1": [1.0 / 3.0, 2.0 / 3.0, 1.0],
                "f2": [1.0, 2.0 / 3.0, 1.0 / 3.0],
            }
        )
        pd.testing.assert_frame_equal(captured["X_train"], expected_train)
        pd.testing.assert_frame_equal(captured["X_valid"], expected_train)
        pd.testing.assert_frame_equal(captured["last_predict_X"], expected_train)
        self.assertEqual(transforms.call_count, 0)
        self.assertIsNotNone(pred_series)
        self.assertIsNotNone(importance_df)
        self.assertEqual(training_summary["num_iterations"], 1)

    def test_generate_prediction_bundle_supports_formula_score_model(self):
        dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
        runtime_data = RollingRuntimeData(
            factor_frame=pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ["A", "B", "C"] * 3,
                    "good": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
                    "bad": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                }
            ),
            dt_index=pd.Series(dates),
            y=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32),
            backtest_y=np.zeros(9, dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            test_start=pd.Timestamp("2024-01-03"),
            test_end=pd.Timestamp("2024-01-03"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-03"])),
            selected_feature_names=["good", "bad"],
            selected_feature_sources=["good", "bad"],
            finite_feature_mask=np.array([True] * 9),
            lookback=20,
            batch_size=3,
        )
        cfg = {
            "data": {"source": "tushare"},
            "universe": "csi300",
            "features": {
                "transforms": {
                    "cross_sectional_rank": False,
                    "cross_sectional_rank_exclude_columns": [],
                }
            },
            "model": {"name": "formula_score"},
            "formula_score": {"mode": "rank_ic_weighted", "min_abs_rank_ic": 0.0},
            "strategy": {"topk": 1},
            "backtest": {"rebalance_freq": 1},
        }
        args = Namespace(torch_gpu=-1, load_models=False, save_models=False, rebalance_freq=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            paths = RollingPaths(
                results_dir=base,
                models_dir=base / "models",
                importance_dir=base / "feature_importance",
                training_history_dir=base / "training_history",
                prediction_artifact_dir=base / "prediction_artifacts",
            )

            bundle = generate_prediction_bundle(
                cfg,
                args,
                runtime_data,
                paths,
                retrain_step=1,
                train_days=1,
                valid_days=1,
                signal_horizon=10,
                label_embargo_days=0,
                model_name="formula_score",
            )

        values = bundle.final_predictions.to_numpy(dtype=float)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertEqual(bundle.metadata["model_name"], "formula_score")
        self.assertEqual(bundle.metadata["label_embargo_days"], 0)
        self.assertEqual(bundle.training_summary_records[0]["label_embargo_days"], 0)
        self.assertEqual(bundle.training_summary_records[0]["formula_score_mode"], "rank_ic_weighted")


if __name__ == "__main__":
    unittest.main()
