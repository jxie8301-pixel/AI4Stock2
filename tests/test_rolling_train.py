import unittest

import numpy as np
import pandas as pd

from src.rolling_train import _compute_validation_topk_summary


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


if __name__ == "__main__":
    unittest.main()
