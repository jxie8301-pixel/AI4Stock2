import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import main as main_module


class _FakeLGBM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X_train, y_train, X_valid, y_valid, valid_dates=None):
        return None

    def predict(self, X_test):
        return np.array([1.0, -1.0], dtype=np.float32)

    def save_feature_importance(self, save_path):
        Path(save_path).write_text("feature,gain\nf1,1.0\n", encoding="utf-8")


class PipelineLabelSemanticsTest(unittest.TestCase):
    def test_native_pipeline_backtest_always_uses_1d_realized_returns(self):
        factor_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-04",
                    ]
                ),
                "symbol": ["A", "B", "A", "B", "A", "B"],
                "f1": [1.0, 2.0, 1.1, 1.9, 1.2, 1.8],
                "label": [0.20, -0.20, 0.10, -0.10, 0.30, -0.30],
                "label_1d": [0.02, -0.02, 0.01, -0.01, 0.03, -0.03],
            }
        )
        expected_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2024-01-04", "2024-01-04"]), ["A", "B"]],
            names=["datetime", "instrument"],
        )
        expected_backtest_labels = pd.Series([0.03, -0.03], index=expected_index, dtype=float)
        captured: dict[str, pd.Series] = {}

        def fake_backtest(preds, labels, **kwargs):
            captured["backtest_labels"] = labels.sort_index()
            return pd.DataFrame(
                {
                    "net_return": [0.0],
                    "turnover": [0.0],
                    "bench": [0.0],
                },
                index=pd.to_datetime(["2024-01-04"]),
            )

        def fake_benchmark(labels):
            captured["benchmark_labels"] = labels.sort_index()
            return pd.Series([0.0], index=pd.to_datetime(["2024-01-04"]))

        cfg = {
            "features": {"lookback": 2},
            "model": {"batch_size": 2},
            "label": {"signal_horizon": 10},
            "time": {
                "train": ["2024-01-02", "2024-01-02"],
                "valid": ["2024-01-03", "2024-01-03"],
                "test": ["2024-01-04", "2024-01-04"],
            },
            "strategy": {"topk": 1, "n_drop": 0},
            "backtest": {
                "cost": {"buy": 0.0, "sell": 0.0},
                "min_cost": 0.0,
                "account": 1000.0,
                "risk_degree": 1.0,
                "slippage": 0.0,
            },
            "native": {"universe_dir": "data/universes"},
            "universe": "all",
        }
        args = Namespace(
            load_model=None,
            save_model=None,
            skip_backtest=False,
            gpu=-1,
            trace_dates=None,
            trace_backtest=False,
            trace_top_days=5,
            rebalance_freq=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            with (
                patch.object(main_module, "get_native_factor_store_dir", return_value=Path(tmpdir) / "store"),
                patch.object(main_module, "load_factor_store_metadata", return_value={"feature_names": ["f1"]}),
                patch.object(main_module, "resolve_selected_features", return_value=([0], ["f1"])),
                patch.object(main_module, "load_factor_frame", return_value=factor_frame.copy()),
                patch.object(
                    main_module,
                    "compute_finite_feature_mask_frame",
                    return_value=np.ones(len(factor_frame), dtype=bool),
                ),
                patch.object(main_module, "apply_feature_transforms", side_effect=lambda df, dates, cfg: df),
                patch.object(main_module, "get_lgbm_config", return_value={}),
                patch("src.models.pure_lightgbm.NativeLGBM", _FakeLGBM),
                patch("src.native_backtest.run_native_backtest", side_effect=fake_backtest),
                patch(
                    "src.evaluate.compute_signal_metrics",
                    return_value=({"IC_mean": 0.0}, pd.Series(dtype=float)),
                ),
                patch("src.evaluate.plot_ic_series", return_value=None),
                patch("src.evaluate.compute_portfolio_metrics", side_effect=lambda portfolio_metric: ({}, portfolio_metric[0])),
                patch("src.evaluate.build_cross_section_benchmark", side_effect=fake_benchmark),
                patch("src.evaluate.build_period_summary", return_value=pd.DataFrame()),
                patch("src.evaluate.plot_cumulative_return", return_value=None),
                patch("src.evaluate.plot_drawdown", return_value=None),
                patch("src.evaluate.plot_monthly_heatmap", return_value=None),
                patch("src.evaluate.save_monthly_report", return_value=None),
                patch("src.evaluate.save_period_summary", return_value=None),
                patch("src.evaluate.print_metrics", return_value=None),
            ):
                main_module.run_native_pipeline(cfg, args, results_dir, "lgbm")

        pd.testing.assert_series_equal(
            captured["backtest_labels"],
            expected_backtest_labels,
            check_names=False,
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            captured["benchmark_labels"],
            expected_backtest_labels,
            check_names=False,
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
