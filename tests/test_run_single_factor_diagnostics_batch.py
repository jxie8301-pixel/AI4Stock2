from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from run_single_factor_diagnostics_batch import (
    _filter_summary_for_features,
    _resolve_incremental_feature_names,
    _write_single_factor_subset_artifacts,
)


def test_resolve_incremental_feature_names_preserves_case_order() -> None:
    incremental = _resolve_incremental_feature_names(
        ["base_a", "layer_b", "base_c", "layer_d"],
        ["base_a", "base_c", "missing_from_case"],
    )

    assert incremental == ["layer_b", "layer_d"]


def test_filter_summary_for_features_keeps_sorted_incremental_rows_only() -> None:
    summary = pd.DataFrame(
        [
            {"feature": "base_a", "rank_ic_abs_mean": 0.09},
            {"feature": "layer_b", "rank_ic_abs_mean": 0.07},
            {"feature": "layer_c", "rank_ic_abs_mean": 0.03},
        ]
    )

    filtered = _filter_summary_for_features(summary, ["layer_c", "layer_b"])

    assert filtered["feature"].tolist() == ["layer_b", "layer_c"]


def test_write_single_factor_subset_artifacts_exports_incremental_csvs() -> None:
    summary = pd.DataFrame(
        [
            {
                "feature": "layer_strong",
                "rank_ic_mean": -0.04,
                "rank_ic_abs_mean": 0.04,
                "rank_ic_ir": -0.6,
                "coverage_pct": 1.0,
            },
            {
                "feature": "layer_weak",
                "rank_ic_mean": 0.01,
                "rank_ic_abs_mean": 0.01,
                "rank_ic_ir": 0.2,
                "coverage_pct": 1.0,
            },
        ]
    )
    segment_comparison = pd.DataFrame(
        [
            {"feature": "layer_strong", "direction_flip": False},
            {"feature": "base_a", "direction_flip": True},
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = _write_single_factor_subset_artifacts(
            summary,
            output_dir=tmpdir,
            prefix="incremental",
            top_n=1,
            segment_comparison=segment_comparison,
            feature_names=["layer_strong", "layer_weak"],
        )

        summary_path = Path(artifacts["incremental_summary_csv"])
        top_abs_path = Path(artifacts["incremental_top_abs_rankic_csv"])
        segment_path = Path(artifacts["incremental_segment_comparison_csv"])

        assert summary_path.exists()
        assert top_abs_path.exists()
        assert segment_path.exists()
        assert pd.read_csv(summary_path)["feature"].tolist() == ["layer_strong", "layer_weak"]
        assert pd.read_csv(top_abs_path)["feature"].tolist() == ["layer_strong"]
        assert pd.read_csv(segment_path)["feature"].tolist() == ["layer_strong"]
