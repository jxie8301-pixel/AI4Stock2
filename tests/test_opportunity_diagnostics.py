from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.opportunity_diagnostics import (
    build_score_bucket_report,
    build_topk_opportunity_daily_report,
    build_yearly_score_bucket_report,
    save_opportunity_diagnostics,
    summarize_score_bucket_report,
)


def _sample_series() -> tuple[pd.Series, pd.Series, pd.Series]:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "C"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-03"), "C"),
        ],
        names=["datetime", "instrument"],
    )
    predictions = pd.Series([0.9, 0.8, 0.8, 0.2, 0.4, 0.3], index=index, name="prediction")
    labels = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, 0.0], index=index, name="label")
    opportunity_labels = pd.Series([1.0, 0.0, float("nan"), 0.0, 1.0, 1.0], index=index)
    return predictions, labels, opportunity_labels


def test_topk_opportunity_daily_report_preserves_stable_tie_order() -> None:
    predictions, labels, opportunity_labels = _sample_series()

    report = build_topk_opportunity_daily_report(
        predictions,
        labels,
        topk=2,
        opportunity_labels=opportunity_labels,
    )

    assert list(report["datetime"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    first = report.iloc[0]
    assert int(first["universe_count"]) == 3
    assert int(first["selected_count"]) == 2
    assert first["selected_prediction_mean"] == pytest.approx(0.85)
    assert first["selected_label_mean"] == pytest.approx(-0.05)
    assert first["selected_positive_rate"] == pytest.approx(0.5)
    assert first["universe_label_mean"] == pytest.approx(pd.Series([0.1, -0.2, 0.3]).mean())
    assert first["universe_positive_rate"] == pytest.approx(2 / 3)
    assert first["selected_opportunity_rate"] == pytest.approx(0.5)
    assert first["universe_opportunity_rate"] == pytest.approx(0.5)


def test_bucket_reports_and_summary_include_opportunity_rates() -> None:
    predictions, labels, opportunity_labels = _sample_series()

    bucket_report = build_score_bucket_report(
        predictions,
        labels,
        opportunity_labels=opportunity_labels,
        n_buckets=3,
    )
    yearly_report = build_yearly_score_bucket_report(
        predictions,
        labels,
        opportunity_labels=opportunity_labels,
        n_buckets=3,
    )
    summary = summarize_score_bucket_report(bucket_report)

    assert list(bucket_report.columns) == [
        "bucket",
        "count",
        "prediction_mean",
        "label_mean",
        "positive_rate",
        "opportunity_rate",
    ]
    assert bucket_report["count"].sum() == 6
    pd.testing.assert_frame_equal(bucket_report.assign(year=2024)[yearly_report.columns], yearly_report)
    assert summary["bucket_count"] == 3
    assert summary["return_monotonicity_spearman"] is not None
    assert summary["opportunity_rate_monotonicity_spearman"] is not None


def test_bucket_report_matches_qcut_on_integer_quantile_edges() -> None:
    dates = [pd.Timestamp("2024-01-02")] * 13
    instruments = [f"S{i:02d}" for i in range(13)]
    index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    predictions = pd.Series(np.arange(13, 0, -1, dtype=float), index=index)
    labels = pd.Series(np.linspace(-0.06, 0.06, 13), index=index)

    bucket_report = build_score_bucket_report(predictions, labels, n_buckets=10)

    assert bucket_report["count"].tolist() == [2, 1, 1, 1, 2, 1, 1, 1, 1, 2]


def test_save_opportunity_diagnostics_writes_expected_artifacts(tmp_path) -> None:
    predictions, labels, opportunity_labels = _sample_series()

    paths = save_opportunity_diagnostics(
        tmp_path,
        predictions=predictions,
        labels=labels,
        topk=2,
        opportunity_labels=opportunity_labels,
        opportunity_mode="absolute",
        opportunity_threshold=0.0,
        n_buckets=3,
    )

    for path in paths.values():
        assert pd.io.common.file_exists(path)
    summary = json.loads((tmp_path / "native_buyability_summary.json").read_text(encoding="utf-8"))
    assert summary["opportunity_mode"] == "absolute"
    assert summary["topk_summary"]["date_count"] == 2
    assert summary["bucket_summary"]["bucket_count"] == 3
