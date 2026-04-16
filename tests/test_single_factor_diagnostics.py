from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from src.single_factor_diagnostics import (
    build_single_factor_detail_frames,
    build_segmented_single_factor_diagnostics,
    build_single_factor_diagnostics,
    compute_feature_diagnostics,
    normalize_segments,
    save_single_factor_diagnostics,
)


def _build_test_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    instruments = ["A", "B", "C", "D"]
    label_map = {
        "A": 0.04,
        "B": 0.02,
        "C": -0.01,
        "D": -0.03,
    }
    signal_map = {
        "A": 4.0,
        "B": 3.0,
        "C": 2.0,
        "D": 1.0,
    }
    inverse_map = {
        "A": 1.0,
        "B": 2.0,
        "C": 3.0,
        "D": 4.0,
    }
    for date in dates:
        for instrument in instruments:
            rows.append(
                {
                    "date": date,
                    "symbol": instrument,
                    "label": label_map[instrument],
                    "signal": signal_map[instrument],
                    "inverse": inverse_map[instrument],
                    "flat": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_compute_feature_diagnostics_detects_strong_positive_signal() -> None:
    frame = _build_test_frame().sort_values(["date", "symbol"]).reset_index(drop=True)
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(frame["date"]), frame["symbol"].astype(str)],
        names=["datetime", "instrument"],
    )
    feature = pd.Series(frame["signal"].to_numpy(dtype=float), index=index, dtype=float)
    labels = pd.Series(frame["label"].to_numpy(dtype=float), index=index, dtype=float)

    metrics = compute_feature_diagnostics(feature, labels, quantile_bins=2)

    assert metrics["coverage_pct"] == 1.0
    assert metrics["effective_date_count"] == 4
    assert metrics["rank_ic_mean"] > 0.99
    assert metrics["monotonicity_mean"] > 0.99
    assert metrics["top_bottom_spread_mean"] > 0.0
    assert metrics["suggested_direction"] == 1


def test_build_single_factor_diagnostics_preserves_negative_direction() -> None:
    frame = _build_test_frame()

    summary = build_single_factor_diagnostics(
        frame,
        feature_names=["signal", "inverse", "flat"],
        label_column="label",
        quantile_bins=2,
    )
    by_feature = summary.set_index("feature")

    assert by_feature.loc["signal", "rank_ic_mean"] > 0.99
    assert by_feature.loc["signal", "suggested_direction"] == 1
    assert by_feature.loc["signal", "monthly_rank_ic_directional_hit_rate"] == 1.0
    assert by_feature.loc["inverse", "rank_ic_mean"] < -0.99
    assert by_feature.loc["inverse", "suggested_direction"] == -1
    assert by_feature.loc["inverse", "monthly_rank_ic_directional_hit_rate"] == 1.0
    assert by_feature.loc["flat", "effective_date_count"] == 0
    assert pd.isna(by_feature.loc["flat", "rank_ic_mean"])
    assert by_feature.loc["signal", "feature_group"] == "alpha158"


def test_build_single_factor_detail_frames_exports_daily_and_yearly_artifacts() -> None:
    frame = _build_test_frame()

    details = build_single_factor_detail_frames(
        frame,
        feature_names=["signal", "inverse"],
        label_column="label",
        quantile_bins=2,
    )

    assert set(details.bucket_return_daily["feature"]) == {"signal", "inverse"}
    assert set(details.bucket_return_daily["bucket"]) == {0, 1}
    assert set(details.top_bottom_spread_daily["feature"]) == {"signal", "inverse"}
    signal_spread = details.top_bottom_spread_daily[
        details.top_bottom_spread_daily["feature"] == "signal"
    ]["top_bottom_spread"]
    inverse_spread = details.top_bottom_spread_daily[
        details.top_bottom_spread_daily["feature"] == "inverse"
    ]["top_bottom_spread"]
    assert bool((signal_spread > 0).all())
    assert bool((inverse_spread < 0).all())
    assert set(details.rank_ic_monthly["feature"]) == {"signal", "inverse"}
    assert details.feature_missing_by_year["feature_coverage_pct"].min() == 1.0


def test_build_segmented_single_factor_diagnostics_detects_direction_flip() -> None:
    rows: list[dict[str, object]] = []
    instruments = ["A", "B", "C", "D"]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    positive = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    negative = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    labels = {"A": 0.04, "B": 0.02, "C": -0.01, "D": -0.03}
    for date in dates[:2]:
        for instrument in instruments:
            rows.append(
                {
                    "date": date,
                    "symbol": instrument,
                    "label": labels[instrument],
                    "signal": positive[instrument],
                }
            )
    for date in dates[2:]:
        for instrument in instruments:
            rows.append(
                {
                    "date": date,
                    "symbol": instrument,
                    "label": labels[instrument],
                    "signal": negative[instrument],
                }
            )
    frame = pd.DataFrame(rows)

    comparison, segments = build_segmented_single_factor_diagnostics(
        frame,
        feature_names=["signal"],
        segments=[
            ("seg_a", "2024-01-01", "2024-01-31"),
            ("seg_b", "2024-02-01", "2024-02-29"),
        ],
        label_column="label",
        quantile_bins=2,
    )

    assert set(segments) == {"seg_a", "seg_b"}
    assert comparison.iloc[0]["feature"] == "signal"
    assert comparison.iloc[0]["suggested_direction__seg_a"] == 1
    assert comparison.iloc[0]["suggested_direction__seg_b"] == -1
    assert bool(comparison.iloc[0]["direction_flip"]) is True


def test_normalize_segments_ignores_embedded_whitespace_in_dates() -> None:
    segments = normalize_segments(
        [
            ("seg_a", "2023-01-01", "2023-12-\n  31"),
        ]
    )

    assert segments[0][0] == "seg_a"
    assert str(segments[0][1].date()) == "2023-01-01"
    assert str(segments[0][2].date()) == "2023-12-31"


def test_save_single_factor_diagnostics_renders_segment_comparison_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "feature": "signal",
                "rank_ic_mean": 0.1,
                "rank_ic_ir": 0.2,
                "coverage_pct": 1.0,
                "monotonicity_mean": 0.3,
                "monthly_rank_ic_directional_hit_rate": 0.6,
                "rank_ic_abs_mean": 0.1,
            }
        ]
    )
    segment_comparison = pd.DataFrame(
        [
            {
                "feature": "signal",
                "direction_flip": True,
                "best_segment_by_abs_rank_ic": "seg_a",
                "worst_segment_by_abs_rank_ic": "seg_b",
                "segment_rank_ic_abs_max": 0.2,
                "segment_rank_ic_mean_range": 0.1,
                "segment_monthly_directional_hit_mean": 0.65,
            }
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        save_single_factor_diagnostics(
            summary,
            output_dir=tmpdir,
            config_snapshot={"features": {"profile": "x"}},
            metadata={"feature_profile": "x"},
            top_n=10,
            segment_comparison=segment_comparison,
            segment_summaries=None,
            detail_frames=build_single_factor_detail_frames(
                _build_test_frame(),
                feature_names=["signal"],
                label_column="label",
                quantile_bins=2,
            ),
        )
        readme = (Path(tmpdir) / "README.md").read_text(encoding="utf-8")
        assert (Path(tmpdir) / "single_factor_bucket_return_daily.csv").exists()
        assert (Path(tmpdir) / "single_factor_top_bottom_spread_daily.csv").exists()
        assert (Path(tmpdir) / "single_factor_rank_ic_monthly.csv").exists()
        assert (Path(tmpdir) / "single_factor_missing_by_year.csv").exists()

    assert "Segment Comparison" in readme
    assert "Detailed Artifacts" in readme
    assert "direction_flip" in readme
    assert "best_segment_by_abs_rank_ic" in readme
    assert "seg_a" in readme
