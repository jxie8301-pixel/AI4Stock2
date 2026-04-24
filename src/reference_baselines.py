"""Shared reference-baseline names and metric field helpers."""

from __future__ import annotations


REFERENCE_BASELINE_SPECS = (
    ("avg_factor_baseline", "Avg Unique Factor Baseline"),
    ("sign_aligned_factor_baseline", "Sign-Aligned Factor Baseline"),
    ("rank_avg_factor_baseline", "Rank-ZScore Average Factor Baseline"),
    ("rank_ic_weighted_factor_baseline", "RankIC-Weighted Factor Baseline"),
)

REFERENCE_BASELINE_PREFIXES = tuple(prefix for prefix, _ in REFERENCE_BASELINE_SPECS)

CORE_REFERENCE_BASELINE_PREFIXES = (
    "rank_avg_factor_baseline",
    "rank_ic_weighted_factor_baseline",
)


def reference_baseline_summary_fields(prefix: str) -> list[str]:
    return [
        f"{prefix}_excess_annualized_return",
        f"{prefix}_excess_information_ratio",
        f"months_beating_{prefix}_pct",
        f"months_beating_{prefix}_summary",
        f"rebalances_beating_{prefix}_pct",
        f"rebalances_beating_{prefix}_summary",
    ]
