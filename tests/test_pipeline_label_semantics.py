import unittest

import numpy as np
import pandas as pd

from src.label_utils import compute_opportunity_sample_weights, transform_training_label_series
from src.rolling_runtime import build_label_series
from src.rolling_types import RollingRuntimeData


class PipelineLabelSemanticsTest(unittest.TestCase):
    def test_rolling_backtest_labels_use_1d_realized_returns(self):
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
            }
        )
        signal_labels = np.array([0.20, -0.20, 0.10, -0.10, 0.30, -0.30], dtype=np.float32)
        backtest_labels = np.array([0.02, -0.02, 0.01, -0.01, 0.03, -0.03], dtype=np.float32)
        runtime_data = RollingRuntimeData(
            factor_frame=factor_frame,
            dt_index=pd.to_datetime(factor_frame["date"]),
            y=signal_labels,
            backtest_y=backtest_labels,
            full_calendar=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])),
            test_start=pd.Timestamp("2024-01-04"),
            test_end=pd.Timestamp("2024-01-04"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-04"])),
            selected_feature_names=["f1"],
            selected_feature_sources=["f1"],
            finite_feature_mask=np.ones(len(factor_frame), dtype=bool),
            lookback=2,
            batch_size=2,
        )

        signal_series, backtest_series = build_label_series(runtime_data)
        expected_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(["2024-01-04", "2024-01-04"]), ["A", "B"]],
            names=["datetime", "instrument"],
        )

        pd.testing.assert_series_equal(
            signal_series,
            pd.Series([0.30, -0.30], index=expected_index, name="label", dtype=np.float32),
            check_dtype=False,
        )
        pd.testing.assert_series_equal(
            backtest_series,
            pd.Series([0.03, -0.03], index=expected_index, name="label", dtype=np.float32),
            check_dtype=False,
        )

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
