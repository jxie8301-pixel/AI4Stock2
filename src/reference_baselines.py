"""Shared reference-baseline names and metric field helpers."""

from __future__ import annotations


REFERENCE_BASELINE_SPECS = (
    ("avg_factor_baseline", "Avg Unique Factor Baseline"),
    ("sign_aligned_factor_baseline", "Sign-Aligned Factor Baseline"),
    ("rank_avg_factor_baseline", "Rank-ZScore Average Factor Baseline"),
    ("rank_ic_weighted_factor_baseline", "RankIC-Weighted Factor Baseline"),
)

FIXED_RISK_REFERENCE_BASELINE_SPECS = tuple(
    (f"fixed_risk_{prefix}", f"Fixed-Risk {display_name}")
    for prefix, display_name in REFERENCE_BASELINE_SPECS
)

ALL_REFERENCE_BASELINE_SPECS = REFERENCE_BASELINE_SPECS + FIXED_RISK_REFERENCE_BASELINE_SPECS

REFERENCE_BASELINE_PREFIXES = tuple(prefix for prefix, _ in ALL_REFERENCE_BASELINE_SPECS)

CORE_REFERENCE_BASELINE_PREFIXES = (
    "rank_avg_factor_baseline",
    "rank_ic_weighted_factor_baseline",
)

FIXED_RISK_CORE_REFERENCE_BASELINE_PREFIXES = tuple(
    f"fixed_risk_{prefix}"
    for prefix in CORE_REFERENCE_BASELINE_PREFIXES
)

CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES = FIXED_RISK_CORE_REFERENCE_BASELINE_PREFIXES


def reference_baseline_summary_fields(prefix: str) -> list[str]:
    return [
        f"{prefix}_annualized_return",
        f"{prefix}_annualized_volatility",
        f"{prefix}_sharpe_ratio",
        f"{prefix}_information_ratio",
        f"{prefix}_max_drawdown",
        f"{prefix}_monthly_win_rate",
        f"{prefix}_profitable_month_summary",
        f"{prefix}_rebalance_win_rate",
        f"{prefix}_profitable_rebalance_summary",
        f"{prefix}_excess_annualized_return",
        f"{prefix}_excess_information_ratio",
        f"months_beating_{prefix}_pct",
        f"months_beating_{prefix}_summary",
        f"rebalances_beating_{prefix}_pct",
        f"rebalances_beating_{prefix}_summary",
    ]
