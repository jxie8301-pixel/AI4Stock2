import unittest

import numpy as np
import pandas as pd

from src.gen_feature import (
    _rolling_rank_pct,
    _rolling_regression_stats,
    build_open_to_open_label,
    build_open_to_open_labels,
)


def _reference_rank_pct(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out: list[float] = []
    for end in range(len(values)):
        window_values = values[max(0, end - window + 1) : end + 1]
        last = window_values[-1]
        if np.isnan(last):
            out.append(np.nan)
            continue
        valid = window_values[np.isfinite(window_values)]
        if valid.size == 0:
            out.append(np.nan)
            continue
        less = float(np.sum(valid < last))
        equal = float(np.sum(valid == last))
        out.append((less + (equal + 1.0) / 2.0) / float(valid.size))
    return pd.Series(out, index=series.index, dtype=float)


def _reference_regression_stats(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = series.to_numpy(dtype=float)
    slopes: list[float] = []
    rsquares: list[float] = []
    residuals: list[float] = []

    for end in range(len(values)):
        window_values = values[max(0, end - window + 1) : end + 1]
        valid = window_values[np.isfinite(window_values)]
        if valid.size < 2:
            slopes.append(np.nan)
            rsquares.append(np.nan)
            residuals.append(np.nan)
            continue

        x = np.arange(1, valid.size + 1, dtype=np.float64)
        xm = x.mean()
        ym = valid.mean()
        denom = np.sum((x - xm) ** 2)
        if np.isclose(denom, 0.0):
            slopes.append(np.nan)
            rsquares.append(np.nan)
            residuals.append(np.nan)
            continue

        beta = float(np.sum((x - xm) * (valid - ym)) / denom)
        y_std = valid.std(ddof=0)
        if np.isclose(x.std(ddof=0), 0.0) or np.isclose(y_std, 0.0):
            rsq = np.nan
        else:
            rsq = float(np.corrcoef(x, valid)[0, 1] ** 2)
        alpha = ym - beta * xm

        slopes.append(beta)
        rsquares.append(rsq)
        residuals.append(float(valid[-1] - (alpha + beta * x[-1])))

    return (
        pd.Series(slopes, index=series.index, dtype=float),
        pd.Series(rsquares, index=series.index, dtype=float),
        pd.Series(residuals, index=series.index, dtype=float),
    )


class GenFeatureCoreTest(unittest.TestCase):
    def test_rolling_rank_pct_matches_reference(self):
        series = pd.Series([1.0, 2.0, 2.0, np.nan, 3.0, 1.5, 1.5, 4.0], dtype=float)

        actual = _rolling_rank_pct(series, 4)
        expected = _reference_rank_pct(series, 4)

        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True, atol=1e-12, rtol=0.0)

    def test_rolling_regression_stats_match_reference(self):
        series = pd.Series([10.0, 11.0, np.nan, 14.0, 15.0, 16.0, np.nan, 18.0], dtype=float)

        actual = _rolling_regression_stats(series, 5)
        expected = _reference_regression_stats(series, 5)

        for actual_series, expected_series in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_series.to_numpy(),
                expected_series.to_numpy(),
                equal_nan=True,
                atol=1e-10,
                rtol=1e-10,
            )

    def test_build_open_to_open_labels_matches_single_horizon_builder(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-08",
                        "2024-01-09",
                    ]
                ),
                "open": [10.0, 10.5, 11.0, 10.8, 11.5, 12.0],
                "high": [10.2, 10.8, 11.1, 11.0, 11.7, 12.2],
                "low": [9.9, 10.2, 10.9, 10.7, 11.3, 11.9],
                "close": [10.1, 10.6, 10.95, 10.9, 11.6, 12.1],
                "volume": [100, 110, 120, 130, 140, 150],
                "amount": [1000, 1100, 1200, 1300, 1400, 1500],
            }
        )

        labels = build_open_to_open_labels(df, [1, 3])

        pd.testing.assert_series_equal(labels["label_1d"], build_open_to_open_label(df, horizon_days=1))
        pd.testing.assert_series_equal(labels["label_3d"], build_open_to_open_label(df, horizon_days=3))
        pd.testing.assert_series_equal(labels["label"], labels["label_1d"])


if __name__ == "__main__":
    unittest.main()
