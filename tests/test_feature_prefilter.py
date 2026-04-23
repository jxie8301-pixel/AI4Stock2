from __future__ import annotations

import pandas as pd

from src.feature_prefilter import (
    build_robust_feature_summary,
    prefilter_feature_summary,
    prune_correlated_features,
    prune_exact_duplicate_features,
)


def test_prefilter_feature_summary_applies_thresholds() -> None:
    summary = pd.DataFrame(
        [
            {
                "feature": "good_a",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.08,
                "monthly_rank_ic_positive_rate": 0.60,
                "monthly_rank_ic_directional_hit_rate": 0.60,
                "monotonicity_mean": 0.10,
            },
            {
                "feature": "good_b",
                "coverage_pct": 0.97,
                "rank_ic_mean": 0.01,
                "rank_ic_ir": 0.20,
                "monthly_rank_ic_positive_rate": 0.55,
                "monthly_rank_ic_directional_hit_rate": 0.55,
                "monotonicity_mean": 0.03,
            },
            {
                "feature": "bad_cov",
                "coverage_pct": 0.70,
                "rank_ic_mean": 0.10,
                "rank_ic_ir": 0.50,
                "monthly_rank_ic_positive_rate": 0.90,
                "monthly_rank_ic_directional_hit_rate": 0.90,
                "monotonicity_mean": 0.20,
            },
        ]
    )

    kept, dropped = prefilter_feature_summary(
        summary,
        min_coverage_pct=0.95,
        min_abs_rank_ic=0.02,
        min_abs_rank_ic_ir=0.10,
        min_monthly_positive_rate=0.45,
    )

    assert kept["feature"].tolist() == ["good_b", "good_a"]
    assert dropped["feature"].tolist() == ["bad_cov"]


def test_prune_correlated_features_prefers_higher_ranked_feature() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "symbol": ["A", "B", "A", "B"],
            "LGBM_bp": [1.0, 2.0, 1.2, 2.2],
            "TEMP_bp_clone": [1.0, 2.0, 1.2, 2.2],
            "TECH_other": [2.0, 1.0, 1.0, 2.0],
        }
    )
    candidates = pd.DataFrame(
        [
            {
                "feature": "LGBM_bp",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.06,
                "rank_ic_ir": 0.30,
                "monthly_rank_ic_positive_rate": 0.60,
                "monotonicity_mean": 0.10,
            },
            {
                "feature": "TEMP_bp_clone",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.06,
                "rank_ic_ir": 0.30,
                "monthly_rank_ic_positive_rate": 0.60,
                "monotonicity_mean": 0.10,
            },
            {
                "feature": "TECH_other",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.12,
                "monthly_rank_ic_positive_rate": 0.55,
                "monotonicity_mean": 0.02,
            },
        ]
    )

    kept, dropped = prune_correlated_features(
        frame,
        candidates,
        corr_threshold=0.99,
        use_cross_sectional_rank=False,
    )

    assert "LGBM_bp" in kept["feature"].tolist()
    assert "TEMP_bp_clone" not in kept["feature"].tolist()
    assert dropped.iloc[0]["feature"] == "TEMP_bp_clone"
    assert dropped.iloc[0]["dropped_by"] == "LGBM_bp"


def test_prefilter_feature_summary_can_apply_segment_stability_rules() -> None:
    summary = pd.DataFrame(
        [
            {
                "feature": "stable_a",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.20,
                "monthly_rank_ic_positive_rate": 0.55,
                "monthly_rank_ic_directional_hit_rate": 0.60,
                "monotonicity_mean": 0.10,
                "direction_flip": False,
                "segment_monthly_directional_hit_mean": 0.58,
                "segment_rank_ic_mean_range": 0.08,
                "segment_rank_ic_abs_max": 0.06,
            },
            {
                "feature": "flip_b",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.04,
                "rank_ic_ir": 0.25,
                "monthly_rank_ic_positive_rate": 0.55,
                "monthly_rank_ic_directional_hit_rate": 0.60,
                "monotonicity_mean": 0.08,
                "direction_flip": True,
                "segment_monthly_directional_hit_mean": 0.57,
                "segment_rank_ic_mean_range": 0.16,
                "segment_rank_ic_abs_max": 0.10,
            },
        ]
    )

    kept, dropped = prefilter_feature_summary(
        summary,
        min_coverage_pct=0.95,
        min_abs_rank_ic=0.02,
        min_abs_rank_ic_ir=0.10,
        min_monthly_positive_rate=0.45,
        min_segment_directional_hit_mean=0.55,
        max_segment_rank_ic_mean_range=0.14,
        exclude_direction_flip=True,
    )

    assert kept["feature"].tolist() == ["stable_a"]
    assert dropped["feature"].tolist() == ["flip_b"]


def test_prune_exact_duplicate_features_keeps_higher_priority_representative() -> None:
    candidates = pd.DataFrame(
        [
            {
                "feature": "CORR20",
                "coverage_pct": 0.99,
                "rank_ic_mean": -0.05,
                "rank_ic_ir": -0.30,
                "monthly_rank_ic_positive_rate": 0.30,
                "monthly_rank_ic_directional_hit_rate": 0.70,
                "monotonicity_mean": -0.15,
            },
            {
                "feature": "TEMP_corr_cv_20",
                "coverage_pct": 0.99,
                "rank_ic_mean": -0.05,
                "rank_ic_ir": -0.30,
                "monthly_rank_ic_positive_rate": 0.30,
                "monthly_rank_ic_directional_hit_rate": 0.70,
                "monotonicity_mean": -0.15,
            },
        ]
    )

    kept, dropped = prune_exact_duplicate_features(candidates)

    assert kept["feature"].tolist() == ["CORR20"]
    assert dropped.iloc[0]["feature"] == "TEMP_corr_cv_20"
    assert dropped.iloc[0]["dropped_by"] == "CORR20"


def test_build_robust_feature_summary_is_conservative_and_direction_aware() -> None:
    raw = pd.DataFrame(
        [
            {
                "feature": "stable_pos",
                "feature_group": "g",
                "coverage_pct": 0.99,
                "rank_ic_mean": 0.06,
                "rank_ic_ir": 0.30,
                "monthly_rank_ic_positive_rate": 0.70,
                "monthly_rank_ic_directional_hit_rate": 0.70,
                "monotonicity_mean": 0.20,
                "segment_monthly_directional_hit_mean": 0.65,
                "segment_rank_ic_mean_range": 0.08,
            },
            {
                "feature": "flip_sign",
                "feature_group": "g",
                "coverage_pct": 0.98,
                "rank_ic_mean": 0.05,
                "rank_ic_ir": 0.25,
                "monthly_rank_ic_positive_rate": 0.60,
                "monthly_rank_ic_directional_hit_rate": 0.60,
                "monotonicity_mean": 0.10,
                "segment_monthly_directional_hit_mean": 0.60,
                "segment_rank_ic_mean_range": 0.06,
            },
        ]
    )
    neutral = pd.DataFrame(
        [
            {
                "feature": "stable_pos",
                "feature_group": "g",
                "coverage_pct": 0.95,
                "rank_ic_mean": 0.03,
                "rank_ic_ir": 0.18,
                "monthly_rank_ic_positive_rate": 0.62,
                "monthly_rank_ic_directional_hit_rate": 0.62,
                "monotonicity_mean": 0.09,
                "segment_monthly_directional_hit_mean": 0.58,
                "segment_rank_ic_mean_range": 0.11,
            },
            {
                "feature": "flip_sign",
                "feature_group": "g",
                "coverage_pct": 0.96,
                "rank_ic_mean": -0.02,
                "rank_ic_ir": -0.10,
                "monthly_rank_ic_positive_rate": 0.40,
                "monthly_rank_ic_directional_hit_rate": 0.60,
                "monotonicity_mean": -0.03,
                "segment_monthly_directional_hit_mean": 0.55,
                "segment_rank_ic_mean_range": 0.07,
            },
        ]
    )

    robust = build_robust_feature_summary(raw, neutral).set_index("feature")

    assert robust.at["stable_pos", "coverage_pct"] == 0.95
    assert robust.at["stable_pos", "rank_ic_mean"] == 0.03
    assert robust.at["stable_pos", "rank_ic_ir"] == 0.18
    assert robust.at["stable_pos", "monthly_rank_ic_directional_hit_rate"] == 0.62
    assert robust.at["stable_pos", "segment_rank_ic_mean_range"] == 0.11
    assert bool(robust.at["stable_pos", "direction_consistent"]) is True
    assert abs(float(robust.at["stable_pos", "neutral_retention_rank_ic_abs"]) - 0.5) < 1e-12

    assert robust.at["flip_sign", "rank_ic_mean"] == 0.0
    assert robust.at["flip_sign", "rank_ic_ir"] == 0.0
    assert bool(robust.at["flip_sign", "direction_flip"]) is True
    assert bool(robust.at["flip_sign", "direction_consistent"]) is False
