import unittest
from argparse import Namespace
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.rolling_train import _compute_validation_topk_summary, generate_prediction_bundle
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
        args = Namespace(gpu=-1, load_models=False, save_models=False, rebalance_freq=None)
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
                model_name="formula_score",
            )

        values = bundle.final_predictions.to_numpy(dtype=float)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertEqual(bundle.metadata["model_name"], "formula_score")
        self.assertEqual(bundle.training_summary_records[0]["formula_score_mode"], "rank_ic_weighted")


if __name__ == "__main__":
    unittest.main()
