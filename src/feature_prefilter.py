"""Heuristic feature prefiltering and redundancy pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.gen_feature import get_known_exact_duplicate_feature_groups
from src.feature_selection import apply_cross_sectional_rank


DEFAULT_MIN_COVERAGE_PCT = 0.95
DEFAULT_MIN_ABS_RANK_IC = 0.02
DEFAULT_MIN_ABS_RANK_IC_IR = 0.10
DEFAULT_MIN_MONTHLY_POSITIVE_RATE = 0.45
DEFAULT_MAX_ABS_CORR = 0.97
DEFAULT_MIN_SEGMENT_DIRECTIONAL_HIT_MEAN = 0.55
DEFAULT_MAX_SEGMENT_RANK_IC_MEAN_RANGE = 0.14


def load_diagnostics_summary(
    path: str | Path,
    *,
    segment_comparison_path: str | Path | None = None,
) -> pd.DataFrame:
    summary = pd.read_csv(path)
    if "feature" not in summary.columns:
        raise ValueError("Diagnostics summary must contain a 'feature' column.")
    if segment_comparison_path is not None:
        segment_comparison = pd.read_csv(segment_comparison_path)
        if "feature" not in segment_comparison.columns:
            raise ValueError("Segment comparison must contain a 'feature' column.")
        summary = summary.merge(segment_comparison, on="feature", how="left")
    return summary


def _feature_prefix_priority(feature_name: str) -> int:
    if feature_name.startswith("TS_"):
        return 0
    if feature_name.startswith("LGBM_"):
        return 1
    if feature_name.startswith("TECH_"):
        return 2
    if feature_name.startswith("TEMP_"):
        return 4
    return 3


def _scored_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["rank_ic_abs_mean"] = out["rank_ic_mean"].abs()
    out["rank_ic_ir_abs"] = out["rank_ic_ir"].abs()
    out["monotonicity_abs_mean"] = out["monotonicity_mean"].abs()
    out["prefix_priority"] = out["feature"].map(_feature_prefix_priority)
    out["direction_flip_sort"] = out.get("direction_flip", pd.Series(False, index=out.index)).fillna(False).astype(int)
    out["segment_monthly_directional_hit_mean"] = out.get(
        "segment_monthly_directional_hit_mean",
        pd.Series(np.nan, index=out.index, dtype=float),
    )
    out["segment_rank_ic_abs_max"] = out.get(
        "segment_rank_ic_abs_max",
        pd.Series(np.nan, index=out.index, dtype=float),
    )
    out["segment_rank_ic_mean_range"] = out.get(
        "segment_rank_ic_mean_range",
        pd.Series(np.nan, index=out.index, dtype=float),
    )
    return out.sort_values(
        [
            "direction_flip_sort",
            "segment_monthly_directional_hit_mean",
            "segment_rank_ic_mean_range",
            "segment_rank_ic_abs_max",
            "rank_ic_ir_abs",
            "rank_ic_abs_mean",
            "monotonicity_abs_mean",
            "monthly_rank_ic_positive_rate",
            "coverage_pct",
            "prefix_priority",
            "feature",
        ],
        ascending=[True, False, True, False, False, False, False, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def prefilter_feature_summary(
    summary: pd.DataFrame,
    *,
    min_coverage_pct: float = DEFAULT_MIN_COVERAGE_PCT,
    min_abs_rank_ic: float = DEFAULT_MIN_ABS_RANK_IC,
    min_abs_rank_ic_ir: float = DEFAULT_MIN_ABS_RANK_IC_IR,
    min_monthly_positive_rate: float = DEFAULT_MIN_MONTHLY_POSITIVE_RATE,
    min_segment_directional_hit_mean: float | None = None,
    max_segment_rank_ic_mean_range: float | None = None,
    exclude_direction_flip: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter factors by basic quality thresholds."""
    scored = _scored_summary(summary)
    if "monthly_rank_ic_directional_hit_rate" in scored.columns:
        monthly_hit_rate = scored["monthly_rank_ic_directional_hit_rate"]
    else:
        monthly_hit_rate = pd.concat(
            [
                scored.get("monthly_rank_ic_positive_rate", pd.Series(np.nan, index=scored.index)),
                1.0 - scored.get("monthly_rank_ic_positive_rate", pd.Series(np.nan, index=scored.index)),
            ],
            axis=1,
        ).max(axis=1)
    scored["monthly_rank_ic_directional_hit_rate"] = monthly_hit_rate
    required_mask = (
        (scored["coverage_pct"] >= float(min_coverage_pct))
        & (
            (scored["rank_ic_abs_mean"] >= float(min_abs_rank_ic))
            | (scored["rank_ic_ir_abs"] >= float(min_abs_rank_ic_ir))
        )
        & (scored["monthly_rank_ic_directional_hit_rate"] >= float(min_monthly_positive_rate))
    )
    if min_segment_directional_hit_mean is not None and "segment_monthly_directional_hit_mean" in scored.columns:
        required_mask &= scored["segment_monthly_directional_hit_mean"] >= float(min_segment_directional_hit_mean)
    if max_segment_rank_ic_mean_range is not None and "segment_rank_ic_mean_range" in scored.columns:
        required_mask &= scored["segment_rank_ic_mean_range"] <= float(max_segment_rank_ic_mean_range)
    if exclude_direction_flip and "direction_flip" in scored.columns:
        required_mask &= ~scored["direction_flip"].fillna(False)
    kept = scored.loc[required_mask].copy().reset_index(drop=True)
    dropped = scored.loc[~required_mask].copy().reset_index(drop=True)
    return kept, dropped


def prune_exact_duplicate_features(
    candidates: pd.DataFrame,
    *,
    duplicate_groups: list[tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the top-ranked representative from each known exact-duplicate group."""
    if candidates.empty:
        return candidates.copy(), pd.DataFrame(columns=["feature", "dropped_by", "duplicate_group"])

    ordered = _scored_summary(candidates)
    groups = duplicate_groups or get_known_exact_duplicate_feature_groups()
    present_features = set(ordered["feature"].tolist())
    normalized_groups = [tuple(name for name in group if name in present_features) for group in groups]
    normalized_groups = [group for group in normalized_groups if len(group) >= 2]
    if not normalized_groups:
        return ordered.reset_index(drop=True), pd.DataFrame(columns=["feature", "dropped_by", "duplicate_group"])

    feature_to_group: dict[str, tuple[str, ...]] = {}
    for group in normalized_groups:
        for feature in group:
            feature_to_group[feature] = group

    kept_rows: list[pd.Series] = []
    dropped_rows: list[dict[str, Any]] = []
    kept_by_group: dict[tuple[str, ...], str] = {}

    for _, row in ordered.iterrows():
        feature = str(row["feature"])
        group = feature_to_group.get(feature)
        if group is None:
            kept_rows.append(row)
            continue
        if group not in kept_by_group:
            kept_rows.append(row)
            kept_by_group[group] = feature
            continue
        dropped_rows.append(
            {
                **row.to_dict(),
                "dropped_by": kept_by_group[group],
                "duplicate_group": "|".join(group),
            }
        )

    kept = pd.DataFrame(kept_rows).reset_index(drop=True)
    dropped = pd.DataFrame(dropped_rows).reset_index(drop=True)
    return kept, dropped


def _build_redundancy_corr_frame(
    factor_frame: pd.DataFrame,
    *,
    feature_names: list[str],
    use_cross_sectional_rank: bool,
) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame()
    feature_data = factor_frame[feature_names].apply(pd.to_numeric, errors="coerce")
    if use_cross_sectional_rank:
        feature_data = apply_cross_sectional_rank(feature_data, factor_frame["date"])
    return feature_data.corr(method="pearson").abs()


def prune_correlated_features(
    factor_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    corr_threshold: float = DEFAULT_MAX_ABS_CORR,
    use_cross_sectional_rank: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily keep strong candidates while dropping highly correlated duplicates."""
    if candidates.empty:
        return candidates.copy(), pd.DataFrame(columns=["feature", "dropped_by", "abs_corr"])

    ordered = _scored_summary(candidates)
    corr_frame = _build_redundancy_corr_frame(
        factor_frame,
        feature_names=ordered["feature"].tolist(),
        use_cross_sectional_rank=use_cross_sectional_rank,
    )

    kept_rows: list[pd.Series] = []
    dropped_rows: list[dict[str, Any]] = []
    kept_features: list[str] = []
    threshold = float(corr_threshold)

    for _, row in ordered.iterrows():
        feature = str(row["feature"])
        if not kept_features:
            kept_rows.append(row)
            kept_features.append(feature)
            continue
        corr_series = corr_frame.loc[feature, kept_features].dropna()
        if corr_series.empty:
            kept_rows.append(row)
            kept_features.append(feature)
            continue
        best_match = str(corr_series.idxmax())
        best_corr = float(corr_series.loc[best_match])
        if best_corr >= threshold:
            dropped_rows.append(
                {
                    **row.to_dict(),
                    "dropped_by": best_match,
                    "abs_corr": best_corr,
                }
            )
            continue
        kept_rows.append(row)
        kept_features.append(feature)

    kept = pd.DataFrame(kept_rows).reset_index(drop=True)
    dropped = pd.DataFrame(dropped_rows).reset_index(drop=True)
    return kept, dropped


def build_profile_yaml(
    selected_columns: list[str],
    *,
    factor_store_name: str = "full_factor_space",
) -> dict[str, Any]:
    return {
        "alpha": "all_factors",
        "generation_space": "full_factor_space",
        "factor_store_name": factor_store_name,
        "selected_columns": list(selected_columns),
    }


def save_profile_yaml(
    selected_columns: list[str],
    *,
    output_path: str | Path,
    factor_store_name: str = "full_factor_space",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_profile_yaml(selected_columns, factor_store_name=factor_store_name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return path
