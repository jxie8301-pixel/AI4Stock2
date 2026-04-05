import unittest
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import (
    align_benchmark_to_report_index,
    align_prediction_label_pairs,
    build_benchmark_series,
    build_rebalance_period_summary,
    build_period_summary,
    compute_portfolio_metrics,
)
from src.label_utils import sanitize_label_array, sanitize_label_series
from src.model_config import get_lgbm_config
from src.models.pure_lightgbm import (
    NativeLGBM,
    _build_ranking_relevance_labels,
    _compute_time_decay_weights,
    _compute_ranking_groups,
    _daily_ic_metric,
    _daily_rank_ic_metric_from_labels,
)
from src.models.pure_pytorch_lstm import NativeLSTMTrainer, NativeStockDataset, compute_daily_ic


class NativeModelMetricsTest(unittest.TestCase):
    def test_compute_daily_ic_uses_mean_of_daily_cross_sections(self):
        predictions = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )

        daily_ic = compute_daily_ic(predictions, labels, dates)

        self.assertAlmostEqual(daily_ic, 0.0, places=8)

    def test_lightgbm_daily_ic_metric_matches_lstm_metric(self):
        predictions = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )
        dataset = lgb.Dataset(np.zeros((4, 1), dtype=np.float32), label=labels)

        metric_name, metric_value, higher_is_better = _daily_ic_metric(predictions, dataset, dates)

        self.assertEqual(metric_name, "daily_ic")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.0, places=8)

    def test_native_lgbm_defaults_to_stable_eval_metric(self):
        mse_model = NativeLGBM(loss="mse")
        mae_model = NativeLGBM(loss="mae")
        huber_model = NativeLGBM(loss="huber")
        rank_model = NativeLGBM(loss="rank_xendcg")
        custom_model = NativeLGBM(loss="mse", eval_metric="rmse")

        self.assertEqual(mse_model.params["metric"], "l2")
        self.assertEqual(mae_model.params["metric"], "l1")
        self.assertEqual(huber_model.params["metric"], "rmse")
        self.assertEqual(huber_model.params["objective"], "huber")
        self.assertEqual(rank_model.params["metric"], "ndcg")
        self.assertEqual(rank_model.params["objective"], "rank_xendcg")
        self.assertEqual(custom_model.params["metric"], "rmse")

    def test_daily_rank_ic_metric_uses_cross_sectional_spearman_mean(self):
        predictions = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )

        metric_name, metric_value, higher_is_better = _daily_rank_ic_metric_from_labels(predictions, labels, dates)

        self.assertEqual(metric_name, "daily_rank_ic")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.0, places=8)

    def test_native_lgbm_custom_early_stopping_metric_disables_builtin_metric(self):
        model = NativeLGBM(loss="huber", early_stopping_metric="daily_rank_ic")

        self.assertEqual(model.params["metric"], "None")
        self.assertEqual(model.early_stopping_metric, "daily_rank_ic")

    def test_native_lgbm_rejects_unknown_early_stopping_metric(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_metric must be one of"):
            NativeLGBM(loss="huber", early_stopping_metric="demo")

    def test_native_lgbm_can_export_feature_importance_csv(self):
        model = NativeLGBM(loss="mse", early_stop=0, num_threads=1)
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])

        model.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.csv"
            model.save_feature_importance(save_path)
            exported = pd.read_csv(save_path)

        self.assertEqual(exported.columns.tolist(), ["feature", "gain"])
        self.assertEqual(set(exported["feature"]), {"f1", "f2"})

    def test_compute_time_decay_weights_emphasizes_recent_rows(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-31"])

        weights = _compute_time_decay_weights(dates, half_life=10)

        self.assertEqual(weights.shape[0], 3)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))

    def test_native_lgbm_accepts_time_decay_training_weights(self):
        model = NativeLGBM(loss="mse", early_stop=0, num_threads=1, train_weight_half_life=10)
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"])

        model.fit(X_train, y_train, train_dates=train_dates)

        self.assertIsNotNone(model.model)

    def test_build_ranking_relevance_labels_is_cross_sectional(self):
        labels = np.array([0.10, 0.00, -0.10, 0.20, 0.10, 0.00], dtype=np.float32)
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

        rel = _build_ranking_relevance_labels(labels, dates, num_bins=3)
        groups = _compute_ranking_groups(dates)

        self.assertEqual(groups.tolist(), [3, 3])
        self.assertEqual(rel.tolist()[:3], [2, 1, 0])
        self.assertEqual(rel.tolist()[3:], [2, 1, 0])

    def test_native_lgbm_accepts_ranking_objective(self):
        model = NativeLGBM(loss="rank_xendcg", early_stop=0, num_threads=1, ranking_num_bins=3)
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.3, 0.2])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])

        model.fit(X_train, y_train, train_dates=train_dates)

        self.assertIsNotNone(model.model)
        preds = model.predict(X_train)
        self.assertEqual(preds.shape[0], len(X_train))

    def test_native_lgbm_custom_early_stopping_records_rank_metrics(self):
        model = NativeLGBM(
            loss="rank_xendcg",
            early_stop=3,
            num_boost_round=20,
            num_threads=1,
            ranking_num_bins=3,
            early_stopping_metric="daily_rank_ic",
            log_evaluation_period=1000,
        )
        X_train = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "f2": [2.0, 1.0, 0.0, 1.5, 0.5, -0.5],
            }
        )
        y_train = pd.Series([0.0, 0.1, 0.3, -0.1, 0.2, 0.4])
        train_dates = pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"]
        )
        X_valid = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "f2": [1.8, 0.8, -0.2, 1.3, 0.3, -0.7],
            }
        )
        y_valid = pd.Series([0.0, 0.05, 0.2, -0.2, 0.1, 0.3])
        valid_dates = pd.to_datetime(
            ["2024-01-03", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04", "2024-01-04"]
        )

        model.fit(
            X_train,
            y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            train_dates=train_dates,
            valid_dates=valid_dates,
        )

        history = model.get_training_history_frame()
        self.assertIn("valid_daily_rank_ic", history.columns)
        self.assertIn("valid_daily_ic", history.columns)
        self.assertIsNotNone(model.best_iteration_)

    def test_native_lgbm_custom_early_stopping_requires_valid_dates(self):
        model = NativeLGBM(loss="huber", early_stop=5, num_threads=1, early_stopping_metric="daily_rank_ic")
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])
        X_valid = pd.DataFrame({"f1": [0.5, 1.5], "f2": [1.0, 0.0]})
        y_valid = pd.Series([0.05, 0.15])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"])

        with self.assertRaisesRegex(ValueError, "valid_dates is required"):
            model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, train_dates=train_dates)

    def test_get_lgbm_config_uses_dedicated_block(self):
        cfg = {
            "model": {"early_stop": 12, "n_jobs": 4, "loss": "pearson"},
            "lgbm": {"loss": "huber", "num_threads": 6},
        }

        lgbm_cfg = get_lgbm_config(cfg)

        self.assertEqual(lgbm_cfg["loss"], "huber")
        self.assertEqual(lgbm_cfg["num_threads"], 6)
        self.assertEqual(lgbm_cfg["early_stop"], 12)

    def test_align_prediction_label_pairs_drops_nan_rows(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-02"), "B"),
                (pd.Timestamp("2024-01-03"), "A"),
            ],
            names=["datetime", "instrument"],
        )
        predictions = pd.Series([1.0, 2.0, 3.0], index=index)
        labels = pd.Series([0.1, np.nan, 0.3], index=index)

        aligned_preds, aligned_labels = align_prediction_label_pairs(predictions, labels)

        self.assertEqual(aligned_preds.index.tolist(), [index[0], index[2]])
        self.assertEqual(aligned_labels.index.tolist(), [index[0], index[2]])

    def test_compute_portfolio_metrics_preserves_raw_returns(self):
        report = pd.DataFrame(
            {
                "return": [1.0, -0.5],
                "turnover": [0.2, 0.4],
                "bench": [0.0, 0.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

        portfolio_metrics, metric_report = compute_portfolio_metrics((report, None))

        self.assertEqual(metric_report["return"].tolist(), [1.0, -0.5])
        self.assertAlmostEqual(
            portfolio_metrics["annualized_return"]["risk"],
            np.mean([1.0, -0.5]) * 242,
            places=8,
        )
        self.assertAlmostEqual(
            portfolio_metrics["annualized_volatility"]["risk"],
            np.std([1.0, -0.5], ddof=1) * np.sqrt(242),
            places=8,
        )
        self.assertAlmostEqual(portfolio_metrics["daily_win_rate"]["risk"], 0.5, places=8)
        self.assertAlmostEqual(portfolio_metrics["monthly_win_rate"]["risk"], 0.0, places=8)
        self.assertEqual(portfolio_metrics["profitable_month_count"], 0)
        self.assertEqual(portfolio_metrics["total_month_count"], 1)
        self.assertEqual(portfolio_metrics["profitable_month_summary"], "0 / 1 = 0.00%")
        self.assertAlmostEqual(portfolio_metrics["turnover_mean"]["risk"], 0.3, places=8)

    def test_compute_portfolio_metrics_converts_qlib_gross_return_to_net(self):
        report = pd.DataFrame(
            {
                "account": [100.0, 101.0],
                "return": [0.02, 0.01],
                "turnover": [0.1, 0.2],
                "cost": [0.005, 0.002],
                "value": [95.0, 96.0],
                "cash": [5.0, 5.0],
                "bench": [0.0, 0.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

        portfolio_metrics, metric_report = compute_portfolio_metrics((report, None))

        self.assertEqual(metric_report["return"].tolist(), [0.015, 0.008])
        self.assertEqual(metric_report["gross_return"].tolist(), [0.02, 0.01])
        self.assertAlmostEqual(
            portfolio_metrics["annualized_return"]["risk"],
            np.mean([0.015, 0.008]) * 242,
            places=8,
        )
        self.assertAlmostEqual(portfolio_metrics["daily_win_rate"]["risk"], 1.0, places=8)
        self.assertAlmostEqual(portfolio_metrics["monthly_win_rate"]["risk"], 1.0, places=8)
        self.assertEqual(portfolio_metrics["profitable_month_count"], 1)
        self.assertEqual(portfolio_metrics["total_month_count"], 1)
        self.assertEqual(portfolio_metrics["profitable_month_summary"], "1 / 1 = 100.00%")

    def test_build_benchmark_series_supports_close_file_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "csi300.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    "close": [100.0, 110.0, 121.0],
                }
            ).to_csv(path, index=False)

            benchmark, benchmark_name = build_benchmark_series(
                pd.Series(dtype=float),
                {
                    "mode": "file",
                    "path": str(path),
                    "date_column": "date",
                    "value_column": "close",
                    "value_type": "close",
                    "name": "CSI300",
                },
            )

        self.assertEqual(benchmark_name, "CSI300")
        self.assertTrue(np.isnan(float(benchmark.iloc[0])))
        self.assertAlmostEqual(float(benchmark.iloc[1]), 0.10, places=8)
        self.assertAlmostEqual(float(benchmark.iloc[2]), 0.10, places=8)

    def test_build_benchmark_series_rejects_empty_file_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "csi300.csv"
            pd.DataFrame({"date": [], "close": []}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "returned no usable rows"):
                build_benchmark_series(
                    pd.Series(dtype=float),
                    {
                        "mode": "file",
                        "path": str(path),
                        "date_column": "date",
                        "value_column": "close",
                        "value_type": "close",
                        "name": "CSI300",
                    },
                )

    def test_align_benchmark_to_report_index_rejects_no_overlap(self):
        benchmark = pd.Series(
            [0.01, 0.02],
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
            dtype=float,
        )

        with self.assertRaisesRegex(ValueError, "has no overlap with backtest period"):
            align_benchmark_to_report_index(
                benchmark,
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                benchmark_name="CSI300",
            )

    def test_compute_portfolio_metrics_adds_benchmark_and_excess_fields(self):
        report = pd.DataFrame(
            {
                "return": [0.02, 0.01],
                "bench": [0.01, 0.00],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        report.attrs["benchmark_name"] = "CSI300"

        portfolio_metrics, _ = compute_portfolio_metrics((report, None))

        self.assertEqual(portfolio_metrics["benchmark_name"], "CSI300")
        self.assertAlmostEqual(
            portfolio_metrics["benchmark_annualized_return"]["risk"],
            np.mean([0.01, 0.00]) * 242,
            places=8,
        )
        self.assertAlmostEqual(
            portfolio_metrics["excess_annualized_return"]["risk"],
            np.mean([0.01, 0.01]) * 242,
            places=8,
        )
        self.assertIn("excess_information_ratio", portfolio_metrics)

    def test_compute_portfolio_metrics_adds_rebalance_win_rate(self):
        report = pd.DataFrame(
            {
                "return": [0.01, 0.01, -0.02, 0.0],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )
        report.attrs["rebalance_freq"] = 2

        portfolio_metrics, _ = compute_portfolio_metrics((report, None))

        self.assertAlmostEqual(portfolio_metrics["rebalance_win_rate"]["risk"], 0.5, places=8)

    def test_build_rebalance_period_summary_groups_by_fixed_trading_windows(self):
        report = pd.DataFrame(
            {
                "return": [0.10, -0.05, 0.02],
                "turnover": [0.01, 0.03, 0.02],
                "bench": [0.00, 0.00, 0.01],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

        summary = build_rebalance_period_summary(report, rebalance_freq=2)

        self.assertEqual(summary["period"].tolist(), ["rebalance_001", "rebalance_002"])
        self.assertAlmostEqual(summary.iloc[0]["return"], (1.10 * 0.95) - 1.0, places=8)
        self.assertEqual(summary.iloc[0]["days"], 2)
        self.assertEqual(summary.iloc[1]["days"], 1)

    def test_build_period_summary_includes_win_rate_and_turnover(self):
        report = pd.DataFrame(
            {
                "return": [0.10, -0.05, 0.02],
                "turnover": [0.01, 0.03, 0.02],
                "bench": [0.00, 0.00, 0.01],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"]),
        )

        summary = build_period_summary(report, freq="ME")

        self.assertEqual(summary["period"].tolist(), ["2024-01", "2024-02"])
        self.assertAlmostEqual(summary.iloc[0]["return"], (1.10 * 0.95) - 1.0, places=8)
        self.assertAlmostEqual(summary.iloc[0]["win_rate"], 0.5, places=8)
        self.assertAlmostEqual(summary.iloc[0]["avg_turnover"], 0.02, places=8)
        self.assertEqual(int(summary.iloc[0]["days"]), 2)

    def test_train_epoch_returns_zero_for_empty_loader(self):
        trainer = NativeLSTMTrainer(d_feat=3, hidden_size=4, num_layers=1, device="cpu")
        dataset = TensorDataset(
            torch.empty((0, 5, 3), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=4, drop_last=True)

        loss = trainer.train_epoch(loader)

        self.assertEqual(loss, 0.0)

    def test_native_stock_dataset_preserves_raw_label_values(self):
        features = np.zeros((3, 2), dtype=np.float32)
        labels = np.array([0.0, 0.25, 0.0], dtype=np.float32)
        symbols = np.array([1, 1, 1], dtype=np.int32)
        mask = np.array([False, True, False], dtype=bool)
        dates = np.array(
            ["2024-01-02", "2024-01-03", "2024-01-04"],
            dtype="datetime64[ns]",
        )

        dataset = NativeStockDataset(
            features,
            labels,
            symbols,
            mask,
            lookback=2,
            full_dates=dates,
        )

        _, label = dataset[0]

        self.assertAlmostEqual(float(label), 0.25, places=8)

    def test_sanitize_label_array_masks_unrealistic_returns(self):
        labels = np.array([0.02, 0.31, -0.4, np.inf], dtype=np.float32)

        cleaned = sanitize_label_array(labels, abs_cap=0.35)

        self.assertTrue(np.isfinite(cleaned[0]))
        self.assertTrue(np.isfinite(cleaned[1]))
        self.assertTrue(np.isnan(cleaned[2]))
        self.assertTrue(np.isnan(cleaned[3]))

    def test_sanitize_label_series_masks_unrealistic_returns(self):
        labels = pd.Series([0.01, 1.2, -0.5], index=["a", "b", "c"])

        cleaned = sanitize_label_series(labels, abs_cap=0.35)

        self.assertAlmostEqual(cleaned.loc["a"], 0.01, places=8)
        self.assertTrue(np.isnan(cleaned.loc["b"]))
        self.assertTrue(np.isnan(cleaned.loc["c"]))


if __name__ == "__main__":
    unittest.main()
