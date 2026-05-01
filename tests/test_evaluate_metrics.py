from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.evaluate import (
    align_prediction_label_pairs,
    compute_signal_metrics,
    safe_cross_sectional_corr,
)


def test_compute_signal_metrics_matches_daily_cross_sectional_corr_with_ties_and_nans() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("2024-01-02", "A"),
            ("2024-01-02", "B"),
            ("2024-01-02", "C"),
            ("2024-01-02", "D"),
            ("2024-01-03", "A"),
            ("2024-01-03", "B"),
            ("2024-01-03", "C"),
            ("2024-01-04", "A"),
            ("2024-01-04", "B"),
            ("2024-01-04", "C"),
            ("2024-01-05", "A"),
            ("2024-01-05", "B"),
        ],
        names=["datetime", "instrument"],
    )
    predictions = pd.Series(
        [1.0, 2.0, 2.0, np.nan, 3.0, 3.0, 3.0, 1.0, 2.0, 4.0, 5.0, np.nan],
        index=index,
    )
    labels = pd.Series(
        [1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 0.0, np.nan, 2.0],
        index=index,
    )

    metrics, daily_ic = compute_signal_metrics(predictions, labels)
    aligned_preds, aligned_labels = align_prediction_label_pairs(predictions, labels)
    frame = pd.DataFrame({"pred": aligned_preds, "label": aligned_labels})
    expected_daily_ic = frame.groupby(level=0).apply(
        lambda group: safe_cross_sectional_corr(group["pred"], group["label"], method="pearson"),
        include_groups=False,
    )
    expected_daily_rank_ic = frame.groupby(level=0).apply(
        lambda group: safe_cross_sectional_corr(group["pred"], group["label"], method="spearman"),
        include_groups=False,
    )

    assert_series_equal(daily_ic, expected_daily_ic, check_names=False)
    assert np.isclose(metrics["IC_mean"], expected_daily_ic.mean())
    assert np.isclose(metrics["IC_std"], expected_daily_ic.std())
    assert np.isclose(metrics["IC_win_rate"], float((expected_daily_ic.dropna() > 0).mean()))
    assert np.isclose(metrics["Rank_IC_mean"], expected_daily_rank_ic.mean())
    assert np.isclose(metrics["Rank_IC_std"], expected_daily_rank_ic.std())
    assert np.isclose(metrics["Rank_IC_win_rate"], float((expected_daily_rank_ic.dropna() > 0).mean()))
