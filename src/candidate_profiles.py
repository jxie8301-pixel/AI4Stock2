"""Candidate-bank profile builders for rolling native backtest artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.reference_baselines import (
    CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES,
    CORE_REFERENCE_BASELINE_PREFIXES,
    REFERENCE_BASELINE_PREFIXES,
)


PORTFOLIO_PROFILE_KEYS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "excess_annualized_return",
    "excess_information_ratio",
    "monthly_win_rate",
    "rebalance_win_rate",
    "turnover_mean",
]

REFERENCE_BASELINE_PROFILE_FIELDS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "information_ratio",
    "max_drawdown",
    "monthly_win_rate",
    "profitable_month_summary",
    "rebalance_win_rate",
    "profitable_rebalance_summary",
    "excess_annualized_return",
    "excess_information_ratio",
    "months_beating_pct",
    "months_beating_summary",
    "rebalances_beating_pct",
    "rebalances_beating_summary",
]

VALIDATION_SIGNAL_METRICS = [
    "valid_topk_excess_mean",
    "valid_topk_positive_rate",
    "valid_topk_label_mean",
    "valid_topk_min_label_mean",
    "best_valid_daily_rank_ic",
]


@dataclass(frozen=True)
class CandidateProfileThresholds:
    min_annualized_return: float = 0.30
    min_sharpe_ratio: float = 1.50
    max_drawdown_floor: float = -0.15
    max_top5_positive_share: float = 0.50
    min_bucket_top_minus_bottom: float = 0.0
    min_yearly_bucket_top_minus_bottom: float = -0.005
    max_calibrated_best_bucket: int = 2
    max_calibrated_top_bucket_rank: int = 2
    min_validation_high_low_return_spread: float = 0.01
    min_validation_high_bin_win_rate: float = 0.60
    min_excess_annualized_vs_rank_avg_baseline: float = 0.0
    min_rebalance_win_vs_rank_avg_baseline: float = 0.55
    min_excess_annualized_vs_rank_ic_baseline: float = 0.0
    min_rebalance_win_vs_rank_ic_baseline: float = 0.55
    min_excess_annualized_vs_fixed_risk_rank_avg_baseline: float = 0.0
    min_rebalance_win_vs_fixed_risk_rank_avg_baseline: float = 0.55
    min_excess_annualized_vs_fixed_risk_rank_ic_baseline: float = 0.0
    min_rebalance_win_vs_fixed_risk_rank_ic_baseline: float = 0.55


def read_csv_artifact(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def read_json_artifact(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict) and "risk" in value:
        value = value["risk"]
    if value is None or pd.isna(value):
        return None
    return float(value)


def metric_text(metrics: dict[str, Any], key: str) -> str | None:
    value = metrics.get(key)
    if value is None:
        return None
    return str(value)


def summarize_reference_baselines(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for prefix in REFERENCE_BASELINE_PREFIXES:
        baseline = {
            "annualized_return": metric_value(metrics, f"{prefix}_annualized_return"),
            "annualized_volatility": metric_value(metrics, f"{prefix}_annualized_volatility"),
            "sharpe_ratio": metric_value(metrics, f"{prefix}_sharpe_ratio"),
            "information_ratio": metric_value(metrics, f"{prefix}_information_ratio"),
            "max_drawdown": metric_value(metrics, f"{prefix}_max_drawdown"),
            "monthly_win_rate": metric_value(metrics, f"{prefix}_monthly_win_rate"),
            "profitable_month_summary": metric_text(metrics, f"{prefix}_profitable_month_summary"),
            "rebalance_win_rate": metric_value(metrics, f"{prefix}_rebalance_win_rate"),
            "profitable_rebalance_summary": metric_text(metrics, f"{prefix}_profitable_rebalance_summary"),
            "excess_annualized_return": metric_value(metrics, f"{prefix}_excess_annualized_return"),
            "excess_information_ratio": metric_value(metrics, f"{prefix}_excess_information_ratio"),
            "months_beating_pct": metric_value(metrics, f"months_beating_{prefix}_pct"),
            "months_beating_summary": metric_text(metrics, f"months_beating_{prefix}_summary"),
            "rebalances_beating_pct": metric_value(metrics, f"rebalances_beating_{prefix}_pct"),
            "rebalances_beating_summary": metric_text(metrics, f"rebalances_beating_{prefix}_summary"),
        }
        if any(value is not None for value in baseline.values()):
            out[prefix] = baseline
    return out


def feature_family(feature: str) -> str:
    if feature.startswith("TS_stock_vs_industry_"):
        return "ts_stock_vs_industry"
    if feature.startswith("TS_industry_"):
        return "ts_industry_state"
    if feature.endswith("_minus_industry") or "_minus_industry_" in feature:
        return "ts_relative_value_quality"
    if feature.startswith(("TS_dividend", "TS_bp", "TS_sp", "TS_ep")):
        return "ts_valuation"
    if "amihud" in feature or "turnover" in feature:
        return "liquidity"
    if feature.startswith("LGBM_"):
        return "lgbm"
    if feature.startswith("TEMP_"):
        return "temporal"
    if feature.startswith("TECH_"):
        return "technical"
    return "other"


def bucket_shape(frame: pd.DataFrame, *, top_bucket: int = 1, middle_buckets: set[int] | None = None) -> dict[str, Any]:
    middle_buckets = middle_buckets or {3, 4, 5, 6, 7}
    numeric = frame.copy()
    numeric["bucket"] = pd.to_numeric(numeric["bucket"], errors="coerce")
    numeric["label_mean"] = pd.to_numeric(numeric["label_mean"], errors="coerce")
    numeric = numeric.dropna(subset=["bucket", "label_mean"]).sort_values("bucket")
    numeric["bucket"] = numeric["bucket"].astype(int)
    if numeric.empty:
        raise ValueError("Bucket report is empty after numeric conversion.")
    if top_bucket not in set(numeric["bucket"]):
        raise ValueError(f"Top bucket {top_bucket} not found.")

    bottom_bucket = int(numeric["bucket"].max())
    by_bucket = numeric.set_index("bucket")
    top_label = float(by_bucket.at[top_bucket, "label_mean"])
    bottom_label = float(by_bucket.at[bottom_bucket, "label_mean"])
    middle = numeric[numeric["bucket"].isin(middle_buckets)]
    middle_mean = float(middle["label_mean"].mean()) if not middle.empty else None
    middle_best = float(middle["label_mean"].max()) if not middle.empty else None
    best_row = numeric.loc[numeric["label_mean"].idxmax()]
    worst_row = numeric.loc[numeric["label_mean"].idxmin()]
    ranked = numeric.sort_values("label_mean", ascending=False).reset_index(drop=True)
    top_rank = int(ranked.index[ranked["bucket"] == top_bucket][0]) + 1
    spearman = numeric["bucket"].corr(numeric["label_mean"], method="spearman")
    return {
        "top_bucket": int(top_bucket),
        "bottom_bucket": bottom_bucket,
        "top_label_mean": top_label,
        "bottom_label_mean": bottom_label,
        "top_minus_bottom": top_label - bottom_label,
        "middle_label_mean": middle_mean,
        "middle_best_label_mean": middle_best,
        "top_minus_middle_mean": top_label - middle_mean if middle_mean is not None else None,
        "top_minus_middle_best": top_label - middle_best if middle_best is not None else None,
        "best_bucket": int(best_row["bucket"]),
        "best_bucket_label_mean": float(best_row["label_mean"]),
        "worst_bucket": int(worst_row["bucket"]),
        "top_bucket_label_rank": top_rank,
        "bucket_label_spearman": float(spearman) if pd.notna(spearman) else None,
    }


def summarize_yearly_path(monthly: pd.DataFrame) -> list[dict[str, Any]]:
    frame = monthly.copy()
    frame["year"] = frame["period"].astype(str).str[:4]
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        returns = pd.to_numeric(group["return"], errors="coerce")
        benchmark = pd.to_numeric(group["bench_return"], errors="coerce")
        excess = pd.to_numeric(group["excess_vs_benchmark"], errors="coerce")
        rows.append(
            {
                "year": str(year),
                "month_count": int(len(group)),
                "compound_return": float((1.0 + returns).prod() - 1.0),
                "benchmark_compound_return": float((1.0 + benchmark).prod() - 1.0),
                "compound_excess_vs_benchmark": float((1.0 + returns).prod() - (1.0 + benchmark).prod()),
                "negative_months": int((returns < 0).sum()),
                "months_beating_benchmark": int((excess > 0).sum()),
                "worst_month_return": float(returns.min()),
                "best_month_return": float(returns.max()),
                "mean_monthly_turnover": float(pd.to_numeric(group["avg_turnover"], errors="coerce").mean()),
            }
        )
    return rows


def summarize_concentration(monthly: pd.DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(monthly["return"], errors="coerce")
    positive = returns[returns > 0].sort_values(ascending=False)
    positive_sum = float(positive.sum())
    raw_sum = float(returns.sum())
    out: dict[str, Any] = {
        "positive_month_count": int((returns > 0).sum()),
        "negative_month_count": int((returns < 0).sum()),
        "raw_monthly_return_sum": raw_sum,
        "positive_month_return_sum": positive_sum,
        "top3_positive_share": float(positive.head(3).sum() / positive_sum) if positive_sum else None,
        "top5_positive_share": float(positive.head(5).sum() / positive_sum) if positive_sum else None,
        "top3_raw_sum_share": float(positive.head(3).sum() / raw_sum) if raw_sum else None,
        "top5_raw_sum_share": float(positive.head(5).sum() / raw_sum) if raw_sum else None,
        "best_months": [],
        "worst_months": [],
    }
    for idx in positive.head(5).index:
        out["best_months"].append(
            {
                "period": str(monthly.loc[idx, "period"]),
                "return": float(returns.loc[idx]),
            }
        )
    for idx in returns.sort_values().head(5).index:
        out["worst_months"].append(
            {
                "period": str(monthly.loc[idx, "period"]),
                "return": float(returns.loc[idx]),
                "bench_return": float(pd.to_numeric(monthly.loc[idx, "bench_return"], errors="coerce")),
                "excess_vs_benchmark": float(pd.to_numeric(monthly.loc[idx, "excess_vs_benchmark"], errors="coerce")),
            }
        )
    return out


def summarize_feature_families(run_dir: Path) -> list[dict[str, Any]]:
    try:
        frame = read_csv_artifact(run_dir, "feature_importance_gain_mean.csv")
    except FileNotFoundError:
        return []
    frame["feature_family"] = frame["feature"].astype(str).map(feature_family)
    frame["importance_gain"] = pd.to_numeric(frame["importance_gain"], errors="coerce").fillna(0.0)
    total = float(frame["importance_gain"].sum())
    rows: list[dict[str, Any]] = []
    for family, importance in frame.groupby("feature_family", sort=True)["importance_gain"].sum().items():
        rows.append(
            {
                "feature_family": str(family),
                "importance_gain": float(importance),
                "importance_share": float(importance / total) if total else None,
            }
        )
    return rows


def summarize_risk_profile(run_dir: Path) -> dict[str, Any]:
    daily = read_csv_artifact(run_dir, "native_daily_report.csv")
    out: dict[str, Any] = {"avg_risk_by_year": []}
    daily["year"] = daily["datetime"].astype(str).str[:4]
    for year, group in daily.groupby("year", sort=True):
        risk = pd.to_numeric(group["risk_degree"], errors="coerce")
        returns = pd.to_numeric(group["return"], errors="coerce")
        out["avg_risk_by_year"].append(
            {
                "year": str(year),
                "avg_risk": float(risk.mean()),
                "min_risk": float(risk.min()),
                "max_risk": float(risk.max()),
                "negative_days": int((returns < 0).sum()),
                "total_days": int(len(group)),
            }
        )

    values = pd.to_numeric(daily["account_value"], errors="coerce")
    dates = daily["datetime"].astype(str)
    peak = float(values.iloc[0])
    peak_date = str(dates.iloc[0])
    max_drawdown = 0.0
    drawdown_start = peak_date
    drawdown_trough = peak_date
    for date, value in zip(dates, values, strict=False):
        value = float(value)
        if value > peak:
            peak = value
            peak_date = str(date)
        drawdown = value / peak - 1.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            drawdown_start = peak_date
            drawdown_trough = str(date)
    out["max_drawdown_period"] = {
        "drawdown": max_drawdown,
        "peak_date": drawdown_start,
        "trough_date": drawdown_trough,
    }
    return out


def summarize_validation_bins(run_dir: Path) -> list[dict[str, Any]]:
    training = read_csv_artifact(run_dir, "training_summary.csv")
    rebalances = read_csv_artifact(run_dir, "native_rebalance_summary.csv")
    if "window_start" not in training or "period_start" not in rebalances:
        return []
    rebalances = rebalances.set_index("period_start")
    rows: list[dict[str, Any]] = []
    for signal_metric in VALIDATION_SIGNAL_METRICS:
        if signal_metric not in training:
            continue
        aligned_rows: list[dict[str, float]] = []
        for _, train_row in training.iterrows():
            period_start = str(train_row["window_start"])
            if period_start not in rebalances.index:
                continue
            rebalance_row = rebalances.loc[period_start]
            aligned_rows.append(
                {
                    "signal": float(pd.to_numeric(train_row[signal_metric], errors="coerce")),
                    "return": float(pd.to_numeric(rebalance_row["return"], errors="coerce")),
                    "excess_vs_benchmark": float(pd.to_numeric(rebalance_row["excess_vs_benchmark"], errors="coerce")),
                }
            )
        aligned = pd.DataFrame(aligned_rows).dropna()
        if len(aligned) < 9:
            continue
        ranked = aligned.sort_values("signal").reset_index(drop=True)
        thirds = len(ranked) // 3
        groups = {
            "low": ranked.iloc[:thirds],
            "mid": ranked.iloc[thirds : 2 * thirds],
            "high": ranked.iloc[2 * thirds :],
        }
        for bin_name, group in groups.items():
            rows.append(
                {
                    "signal_metric": signal_metric,
                    "bin": bin_name,
                    "window_count": int(len(group)),
                    "signal_mean": float(group["signal"].mean()),
                    "rebalance_return_mean": float(group["return"].mean()),
                    "rebalance_excess_mean": float(group["excess_vs_benchmark"].mean()),
                    "positive_rebalance_count": int((group["return"] > 0).sum()),
                    "positive_rebalance_rate": float((group["return"] > 0).mean()),
                }
            )
    return rows


def summarize_validation_edges(validation_bins: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = pd.DataFrame(validation_bins)
    if frame.empty:
        return out
    for signal_metric, group in frame.groupby("signal_metric", sort=True):
        by_bin = group.set_index("bin")
        if "high" not in by_bin.index or "low" not in by_bin.index:
            continue
        high = by_bin.loc["high"]
        low = by_bin.loc["low"]
        out[str(signal_metric)] = {
            "high_low_return_spread": float(high["rebalance_return_mean"] - low["rebalance_return_mean"]),
            "high_low_excess_spread": float(high["rebalance_excess_mean"] - low["rebalance_excess_mean"]),
            "high_positive_rebalance_rate": float(high["positive_rebalance_rate"]),
            "low_positive_rebalance_rate": float(low["positive_rebalance_rate"]),
        }
    return out


def _candidate_gate_baseline_prefixes(reference_baselines: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    pure_prefixes = tuple(
        prefix
        for prefix in CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES
        if prefix in reference_baselines
    )
    if pure_prefixes:
        return pure_prefixes
    return tuple(prefix for prefix in CORE_REFERENCE_BASELINE_PREFIXES if prefix in reference_baselines)


def _candidate_gate_baseline_thresholds(thresholds: CandidateProfileThresholds) -> dict[str, dict[str, float]]:
    return {
        "rank_avg_factor_baseline": {
            "min_excess_annualized_return": thresholds.min_excess_annualized_vs_rank_avg_baseline,
            "min_rebalances_beating_pct": thresholds.min_rebalance_win_vs_rank_avg_baseline,
        },
        "rank_ic_weighted_factor_baseline": {
            "min_excess_annualized_return": thresholds.min_excess_annualized_vs_rank_ic_baseline,
            "min_rebalances_beating_pct": thresholds.min_rebalance_win_vs_rank_ic_baseline,
        },
        "fixed_risk_rank_avg_factor_baseline": {
            "min_excess_annualized_return": thresholds.min_excess_annualized_vs_fixed_risk_rank_avg_baseline,
            "min_rebalances_beating_pct": thresholds.min_rebalance_win_vs_fixed_risk_rank_avg_baseline,
        },
        "fixed_risk_rank_ic_weighted_factor_baseline": {
            "min_excess_annualized_return": thresholds.min_excess_annualized_vs_fixed_risk_rank_ic_baseline,
            "min_rebalances_beating_pct": thresholds.min_rebalance_win_vs_fixed_risk_rank_ic_baseline,
        },
    }


def build_promotion_gates(
    *,
    portfolio: dict[str, Any],
    reference_baselines: dict[str, dict[str, Any]],
    concentration: dict[str, Any],
    overall_bucket: dict[str, Any],
    yearly_buckets: list[dict[str, Any]],
    validation_edges: dict[str, Any],
    thresholds: CandidateProfileThresholds,
) -> dict[str, dict[str, Any]]:
    valid_edge = validation_edges.get("valid_topk_excess_mean", {})
    min_yearly_top_minus_bottom = min(
        (float(row["top_minus_bottom"]) for row in yearly_buckets if row.get("top_minus_bottom") is not None),
        default=None,
    )
    gates = {
        "return_quality": {
            "passed": bool(
                (portfolio.get("annualized_return") or 0.0) >= thresholds.min_annualized_return
                and (portfolio.get("sharpe_ratio") or 0.0) >= thresholds.min_sharpe_ratio
            ),
            "annualized_return": portfolio.get("annualized_return"),
            "sharpe_ratio": portfolio.get("sharpe_ratio"),
        },
        "drawdown_control": {
            "passed": bool((portfolio.get("max_drawdown") or -1.0) >= thresholds.max_drawdown_floor),
            "max_drawdown": portfolio.get("max_drawdown"),
            "floor": thresholds.max_drawdown_floor,
        },
        "concentration_control": {
            "passed": bool((concentration.get("top5_positive_share") or 1.0) <= thresholds.max_top5_positive_share),
            "top5_positive_share": concentration.get("top5_positive_share"),
            "max_allowed": thresholds.max_top5_positive_share,
        },
        "bucket_separates_bad_tail": {
            "passed": bool((overall_bucket.get("top_minus_bottom") or 0.0) > thresholds.min_bucket_top_minus_bottom),
            "top_minus_bottom": overall_bucket.get("top_minus_bottom"),
            "minimum": thresholds.min_bucket_top_minus_bottom,
        },
        "bucket_calibrated_for_sizing": {
            "passed": bool(
                int(overall_bucket.get("best_bucket") or 999) <= thresholds.max_calibrated_best_bucket
                and int(overall_bucket.get("top_bucket_label_rank") or 999) <= thresholds.max_calibrated_top_bucket_rank
                and (overall_bucket.get("top_minus_middle_best") or -1.0) >= 0.0
            ),
            "best_bucket": overall_bucket.get("best_bucket"),
            "top_bucket_label_rank": overall_bucket.get("top_bucket_label_rank"),
            "top_minus_middle_best": overall_bucket.get("top_minus_middle_best"),
        },
        "yearly_bucket_not_inverted": {
            "passed": bool(
                min_yearly_top_minus_bottom is not None
                and min_yearly_top_minus_bottom >= thresholds.min_yearly_bucket_top_minus_bottom
            ),
            "min_yearly_top_minus_bottom": min_yearly_top_minus_bottom,
            "minimum": thresholds.min_yearly_bucket_top_minus_bottom,
        },
        "validation_metric_supports_risk_gate": {
            "passed": bool(
                (valid_edge.get("high_low_return_spread") or 0.0)
                >= thresholds.min_validation_high_low_return_spread
                and (valid_edge.get("high_positive_rebalance_rate") or 0.0)
                >= thresholds.min_validation_high_bin_win_rate
            ),
            "high_low_return_spread": valid_edge.get("high_low_return_spread"),
            "high_positive_rebalance_rate": valid_edge.get("high_positive_rebalance_rate"),
        },
    }
    core_baseline_thresholds = _candidate_gate_baseline_thresholds(thresholds)
    for prefix in _candidate_gate_baseline_prefixes(reference_baselines):
        baseline = reference_baselines.get(prefix)
        if not baseline:
            continue
        threshold = core_baseline_thresholds[prefix]
        excess_annualized_return = baseline.get("excess_annualized_return")
        rebalances_beating_pct = baseline.get("rebalances_beating_pct")
        gates[f"beats_{prefix}"] = {
            "passed": bool(
                excess_annualized_return is not None
                and rebalances_beating_pct is not None
                and float(excess_annualized_return) >= threshold["min_excess_annualized_return"]
                and float(rebalances_beating_pct) >= threshold["min_rebalances_beating_pct"]
            ),
            "excess_annualized_return": excess_annualized_return,
            "minimum_excess_annualized_return": threshold["min_excess_annualized_return"],
            "rebalances_beating_pct": rebalances_beating_pct,
            "minimum_rebalances_beating_pct": threshold["min_rebalances_beating_pct"],
            "rebalances_beating_summary": baseline.get("rebalances_beating_summary"),
        }
    return gates


def infer_candidate_role(gates: dict[str, dict[str, Any]]) -> str:
    if not gates["return_quality"]["passed"]:
        return "research_archive"
    baseline_gates = [
        gate
        for name, gate in gates.items()
        if name.startswith("beats_") and name.endswith("_factor_baseline")
    ]
    if baseline_gates and not all(gate["passed"] for gate in baseline_gates):
        return "portfolio_candidate_requires_router"
    if gates["bucket_calibrated_for_sizing"]["passed"] and gates["validation_metric_supports_risk_gate"]["passed"]:
        return "ranker_and_sizer"
    if gates["bucket_separates_bad_tail"]["passed"] and gates["validation_metric_supports_risk_gate"]["passed"]:
        return "pool_selector_with_validation_gate"
    if gates["bucket_separates_bad_tail"]["passed"]:
        return "pool_selector_only"
    return "portfolio_candidate_requires_router"


def build_candidate_profile(
    run_name: str,
    run_dir: Path,
    *,
    top_bucket: int = 1,
    middle_buckets: set[int] | None = None,
    thresholds: CandidateProfileThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or CandidateProfileThresholds()
    middle_buckets = middle_buckets or {3, 4, 5, 6, 7}
    metrics = read_json_artifact(run_dir, "native_portfolio_metrics.json")
    monthly = read_csv_artifact(run_dir, "native_monthly_summary.csv")
    bucket = read_csv_artifact(run_dir, "native_score_bucket_report.csv")
    yearly_bucket = read_csv_artifact(run_dir, "native_score_bucket_yearly_report.csv")

    portfolio = {key: metric_value(metrics, key) for key in PORTFOLIO_PROFILE_KEYS}
    reference_baselines = summarize_reference_baselines(metrics)
    yearly_path = summarize_yearly_path(monthly)
    concentration = summarize_concentration(monthly)
    overall_bucket = bucket_shape(bucket, top_bucket=top_bucket, middle_buckets=middle_buckets)
    yearly_buckets = [
        {"year": str(year), **bucket_shape(group, top_bucket=top_bucket, middle_buckets=middle_buckets)}
        for year, group in yearly_bucket.groupby("year", sort=True)
    ]
    validation_bins = summarize_validation_bins(run_dir)
    validation_edges = summarize_validation_edges(validation_bins)
    promotion_gates = build_promotion_gates(
        portfolio=portfolio,
        reference_baselines=reference_baselines,
        concentration=concentration,
        overall_bucket=overall_bucket,
        yearly_buckets=yearly_buckets,
        validation_edges=validation_edges,
        thresholds=thresholds,
    )
    failed_gates = [name for name, gate in promotion_gates.items() if not gate["passed"]]

    strong_years = [
        row["year"]
        for row in yearly_path
        if float(row["compound_return"]) > 0.25 and int(row["months_beating_benchmark"]) >= max(7, int(row["month_count"]) // 2)
    ]
    weak_years = [
        row["year"]
        for row in yearly_path
        if float(row["compound_return"]) < 0.15 or int(row["negative_months"]) >= int(row["month_count"]) // 2
    ]
    return {
        "name": run_name,
        "run_dir": str(run_dir),
        "candidate_role": infer_candidate_role(promotion_gates),
        "portfolio": portfolio,
        "reference_baselines": reference_baselines,
        "regime_profile": {
            "strong_years": strong_years,
            "weak_years": weak_years,
            "yearly_path": yearly_path,
        },
        "concentration_profile": concentration,
        "calibration_profile": {
            "overall_bucket": overall_bucket,
            "yearly_buckets": yearly_buckets,
            "weak_yearly_buckets": [
                row
                for row in yearly_buckets
                if (row.get("top_minus_bottom") or 0.0) < thresholds.min_yearly_bucket_top_minus_bottom
                or int(row.get("top_bucket_label_rank") or 999) > thresholds.max_calibrated_top_bucket_rank
            ],
        },
        "validation_profile": {
            "bins": validation_bins,
            "edges": validation_edges,
            "preferred_primary_metric": "valid_topk_excess_mean"
            if "valid_topk_excess_mean" in validation_edges
            else None,
        },
        "feature_profile": {
            "family_importance": summarize_feature_families(run_dir),
        },
        "risk_profile": summarize_risk_profile(run_dir),
        "promotion_gates": promotion_gates,
        "gate_summary": {
            "passed_count": len(promotion_gates) - len(failed_gates),
            "total_count": len(promotion_gates),
            "failed_gates": failed_gates,
        },
    }


def _flatten_reference_baselines(reference_baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for prefix in REFERENCE_BASELINE_PREFIXES:
        baseline = reference_baselines.get(prefix, {})
        for field in REFERENCE_BASELINE_PROFILE_FIELDS:
            flattened[f"{prefix}_{field}"] = baseline.get(field)
    return flattened


def flatten_candidate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    portfolio = profile["portfolio"]
    reference_baselines = profile.get("reference_baselines", {})
    overall_bucket = profile["calibration_profile"]["overall_bucket"]
    concentration = profile["concentration_profile"]
    validation_edge = profile["validation_profile"]["edges"].get("valid_topk_excess_mean", {})
    drawdown_period = profile["risk_profile"].get("max_drawdown_period", {})
    return {
        "name": profile["name"],
        "candidate_role": profile["candidate_role"],
        "annualized_return": portfolio.get("annualized_return"),
        "sharpe_ratio": portfolio.get("sharpe_ratio"),
        "max_drawdown": portfolio.get("max_drawdown"),
        "excess_information_ratio": portfolio.get("excess_information_ratio"),
        "monthly_win_rate": portfolio.get("monthly_win_rate"),
        "rebalance_win_rate": portfolio.get("rebalance_win_rate"),
        **_flatten_reference_baselines(reference_baselines),
        "top5_positive_share": concentration.get("top5_positive_share"),
        "best_bucket": overall_bucket.get("best_bucket"),
        "top_bucket_label_rank": overall_bucket.get("top_bucket_label_rank"),
        "top_minus_bottom": overall_bucket.get("top_minus_bottom"),
        "top_minus_middle_best": overall_bucket.get("top_minus_middle_best"),
        "validation_high_low_return_spread": validation_edge.get("high_low_return_spread"),
        "validation_high_positive_rebalance_rate": validation_edge.get("high_positive_rebalance_rate"),
        "strong_years": "|".join(profile["regime_profile"]["strong_years"]),
        "weak_years": "|".join(profile["regime_profile"]["weak_years"]),
        "max_drawdown_peak_date": drawdown_period.get("peak_date"),
        "max_drawdown_trough_date": drawdown_period.get("trough_date"),
        "passed_gate_count": profile["gate_summary"]["passed_count"],
        "total_gate_count": profile["gate_summary"]["total_count"],
        "failed_gates": "|".join(profile["gate_summary"]["failed_gates"]),
        "run_dir": profile["run_dir"],
    }


def build_candidate_profiles(
    runs: dict[str, Path],
    *,
    top_bucket: int = 1,
    middle_buckets: set[int] | None = None,
    thresholds: CandidateProfileThresholds | None = None,
) -> list[dict[str, Any]]:
    return [
        build_candidate_profile(
            run_name,
            run_dir,
            top_bucket=top_bucket,
            middle_buckets=middle_buckets,
            thresholds=thresholds,
        )
        for run_name, run_dir in runs.items()
    ]


def flatten_candidate_profiles(profiles: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([flatten_candidate_profile(profile) for profile in profiles])


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if pd.isna(value):
        return None
    return value
