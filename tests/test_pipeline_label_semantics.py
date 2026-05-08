import unittest

import numpy as np
import pandas as pd

from src.label_utils import compute_opportunity_sample_weights, transform_training_label_series


class PipelineLabelSemanticsTest(unittest.TestCase):
    def test_training_label_transform_is_training_only_utility(self):
        labels = pd.Series([0.20, -0.20], name="label", dtype=float)
        dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-02"]))
        cfg = {
            "label": {
                "train_transform": {"mode": "profit_tanh", "neutral_band": 0.05},
            },
        }

        transformed = transform_training_label_series(labels, dates, cfg)

        self.assertEqual(len(transformed), 2)
        self.assertNotEqual(transformed.astype(float).tolist(), labels.astype(float).tolist())

    def test_opportunity_sample_weights_can_date_normalize(self):
        labels = pd.Series([0.03, -0.01, 0.02, -0.04], index=[10, 11, 12, 13], name="label", dtype=float)
        dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]))

        weights = compute_opportunity_sample_weights(
            labels,
            dates,
            opportunity_cfg={"mode": "positive", "threshold": 0.0, "neutral_band": 0.005},
            sample_weight_mode="opportunity_distance",
            sample_weight_scale=0.01,
            sample_weight_date_normalize=True,
        )

        self.assertEqual(len(weights), 4)
        self.assertEqual(weights.index.tolist(), [10, 11, 12, 13])
        self.assertTrue(np.isfinite(weights.to_numpy(dtype=float)).all())
        for date in dates.unique():
            day_weights = weights.iloc[(dates == date).to_numpy()]
            self.assertAlmostEqual(float(day_weights.mean()), 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
