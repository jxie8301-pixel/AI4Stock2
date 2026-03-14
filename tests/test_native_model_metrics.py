import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import align_prediction_label_pairs, compute_portfolio_metrics
from src.label_utils import sanitize_label_array, sanitize_label_series
from src.model_config import get_lgbm_config
from src.models.pure_lightgbm import NativeLGBM, _daily_ic_metric
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
        custom_model = NativeLGBM(loss="mse", eval_metric="rmse")

        self.assertEqual(mse_model.params["metric"], "l2")
        self.assertEqual(mae_model.params["metric"], "l1")
        self.assertEqual(huber_model.params["metric"], "rmse")
        self.assertEqual(huber_model.params["objective"], "huber")
        self.assertEqual(custom_model.params["metric"], "rmse")

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
