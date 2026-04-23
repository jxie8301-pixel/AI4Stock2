"""Compare two rolling strategy runs and diagnose top-k ranking failure modes."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


METRIC_KEYS = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "daily_win_rate",
    "monthly_win_rate",
    "profit_factor",
    "top_1_positive_month_share",
    "top_3_positive_month_share",
    "top_5_positive_month_share",
    "excess_annualized_return",
    "excess_information_ratio",
    "turnover_mean",
    "avg_factor_baseline_annualized_return",
    "avg_factor_baseline_information_ratio",
]

TRAINING_COLUMNS = [
    "best_valid_daily_ic",
    "best_valid_daily_rank_ic",
    "valid_top1_positive_rate",
    "valid_topk_positive_rate",
    "valid_topk_label_mean",
    "valid_topk_excess_mean",
    "best_iteration",
    "num_iterations",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose differences between two rolling strategy run directories.")
    parser.add_argument("--candidate-run", required=True, help="Candidate run directory.")
    parser.add_argument("--baseline-run", required=True, help="Baseline run directory.")
    parser.add_argument("--candidate-name", default="candidate", help="Candidate label for reports.")
    parser.add_argument("--baseline-name", default="baseline", help="Baseline label for reports.")
    parser.add_argument("--output-dir", help="Optional output directory.")
    parser.add_argument("--top-bucket", type=int, default=1, help="Top score bucket id. Default: 1.")
    parser.add_argument(
        "--middle-buckets",
        default="3,4,5,6,7",
        help="Comma-separated bucket ids used as the middle-stability zone. Default: 3,4,5,6,7.",
    )
    return parser


def _read_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _read_json(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if isinstance(value, dict) and "risk" in value:
        return value["risk"]
    return value


def summarize_metrics(run_dirs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in run_dirs.items():
        metrics = _read_json(run_dir, "native_portfolio_metrics.json")
        row = {"run": run_name, "run_dir": str(run_dir)}
        for key in METRIC_KEYS:
            row[key] = _metric_value(metrics, key)
        row["profitable_month_summary"] = metrics.get("profitable_month_summary")
        row["profitable_rebalance_summary"] = metrics.get("profitable_rebalance_summary")
        row["months_beating_avg_factor_baseline_summary"] = metrics.get(
            "months_beating_avg_factor_baseline_summary"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_training(run_dirs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in run_dirs.items():
        frame = _read_csv(run_dir, "training_summary.csv")
        row: dict[str, Any] = {"run": run_name, "window_count": len(frame)}
        if "feature_count" in frame:
            row["feature_count_median"] = float(pd.to_numeric(frame["feature_count"], errors="coerce").median())
        for column in TRAINING_COLUMNS:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_median"] = values.median()
            row[f"{column}_min"] = values.min()
            row[f"{column}_max"] = values.max()
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_bucket_ids(raw: str) -> set[int]:
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values:
        raise ValueError("--middle-buckets must contain at least one integer bucket id.")
    return values


def _bucket_shape(frame: pd.DataFrame, *, top_bucket: int, middle_buckets: set[int]) -> dict[str, Any]:
    if "bucket" not in frame or "label_mean" not in frame:
        raise ValueError("Bucket report must contain 'bucket' and 'label_mean'.")
    by_bucket = frame.set_index("bucket")
    if top_bucket not in by_bucket.index:
        raise ValueError(f"Top bucket {top_bucket} not found in bucket report.")
    bottom_bucket = int(frame["bucket"].max())
    top_label = float(by_bucket.at[top_bucket, "label_mean"])
    bottom_label = float(by_bucket.at[bottom_bucket, "label_mean"])
    best_idx = int(frame.loc[frame["label_mean"].idxmax(), "bucket"])
    worst_idx = int(frame.loc[frame["label_mean"].idxmin(), "bucket"])
    middle_frame = frame[frame["bucket"].isin(middle_buckets)]
    middle_mean = float(middle_frame["label_mean"].mean()) if not middle_frame.empty else float("nan")
    middle_best = float(middle_frame["label_mean"].max()) if not middle_frame.empty else float("nan")
    sorted_labels = frame.sort_values("label_mean", ascending=False).reset_index(drop=True)
    top_label_rank = int(sorted_labels.index[sorted_labels["bucket"] == top_bucket][0]) + 1
    return {
        "top_bucket": top_bucket,
        "bottom_bucket": bottom_bucket,
        "top_label_mean": top_label,
        "bottom_label_mean": bottom_label,
        "top_minus_bottom": top_label - bottom_label,
        "middle_label_mean": middle_mean,
        "middle_best_label_mean": middle_best,
        "top_minus_middle_mean": top_label - middle_mean,
        "top_minus_middle_best": top_label - middle_best,
        "best_bucket": best_idx,
        "worst_bucket": worst_idx,
        "top_bucket_label_rank": top_label_rank,
    }


def compare_buckets(
    run_dirs: dict[str, Path],
    *,
    top_bucket: int,
    middle_buckets: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bucket_rows: list[pd.DataFrame] = []
    shape_rows: list[dict[str, Any]] = []
    yearly_shape_rows: list[dict[str, Any]] = []
    for run_name, run_dir in run_dirs.items():
        bucket = _read_csv(run_dir, "native_score_bucket_report.csv")
        bucket["run"] = run_name
        bucket_rows.append(bucket)
        shape_rows.append({"run": run_name, **_bucket_shape(bucket, top_bucket=top_bucket, middle_buckets=middle_buckets)})

        yearly = _read_csv(run_dir, "native_score_bucket_yearly_report.csv")
        for year, group in yearly.groupby("year"):
            yearly_shape_rows.append(
                {
                    "run": run_name,
                    "year": year,
                    **_bucket_shape(group, top_bucket=top_bucket, middle_buckets=middle_buckets),
                }
            )
    return pd.concat(bucket_rows, ignore_index=True), pd.DataFrame(shape_rows), pd.DataFrame(yearly_shape_rows)


def compare_monthly(candidate_name: str, candidate_dir: Path, baseline_name: str, baseline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _read_csv(candidate_dir, "native_monthly_summary.csv")
    baseline = _read_csv(baseline_dir, "native_monthly_summary.csv")
    keep_columns = [
        "period",
        "return",
        "excess_vs_benchmark",
        "excess_vs_avg_factor_baseline",
        "max_drawdown",
        "win_rate",
        "profit_factor",
    ]
    candidate = candidate[keep_columns].rename(columns={col: f"{candidate_name}_{col}" for col in keep_columns if col != "period"})
    baseline = baseline[keep_columns].rename(columns={col: f"{baseline_name}_{col}" for col in keep_columns if col != "period"})
    merged = candidate.merge(baseline, on="period", how="inner")
    merged["year"] = merged["period"].astype(str).str[:4]
    merged["return_diff"] = merged[f"{candidate_name}_return"] - merged[f"{baseline_name}_return"]
    merged["excess_vs_benchmark_diff"] = (
        merged[f"{candidate_name}_excess_vs_benchmark"] - merged[f"{baseline_name}_excess_vs_benchmark"]
    )
    merged["excess_vs_avg_factor_diff"] = (
        merged[f"{candidate_name}_excess_vs_avg_factor_baseline"]
        - merged[f"{baseline_name}_excess_vs_avg_factor_baseline"]
    )
    yearly = (
        merged.groupby("year")
        .agg(
            month_count=("period", "count"),
            candidate_compound=(f"{candidate_name}_return", lambda x: (1.0 + x).prod() - 1.0),
            baseline_compound=(f"{baseline_name}_return", lambda x: (1.0 + x).prod() - 1.0),
            return_diff_sum=("return_diff", "sum"),
            return_diff_mean=("return_diff", "mean"),
            return_diff_median=("return_diff", "median"),
            candidate_win_months=("return_diff", lambda x: int((x > 0).sum())),
            worst_diff=("return_diff", "min"),
            best_diff=("return_diff", "max"),
        )
        .reset_index()
    )
    yearly["compound_diff"] = yearly["candidate_compound"] - yearly["baseline_compound"]
    return merged, yearly


def compare_daily(candidate_name: str, candidate_dir: Path, baseline_name: str, baseline_dir: Path) -> pd.DataFrame:
    candidate = _read_csv(candidate_dir, "native_daily_report.csv")
    baseline = _read_csv(baseline_dir, "native_daily_report.csv")
    candidate = candidate[["datetime", "return", "bench", "risk_degree", "holdings", "turnover"]].rename(
        columns={col: f"{candidate_name}_{col}" for col in ["return", "bench", "risk_degree", "holdings", "turnover"]}
    )
    baseline = baseline[["datetime", "return", "bench", "risk_degree", "holdings", "turnover"]].rename(
        columns={col: f"{baseline_name}_{col}" for col in ["return", "bench", "risk_degree", "holdings", "turnover"]}
    )
    merged = candidate.merge(baseline, on="datetime", how="inner")
    merged["return_diff"] = merged[f"{candidate_name}_return"] - merged[f"{baseline_name}_return"]
    return merged


def _feature_family(feature: str) -> str:
    if feature.startswith("TS_stock_vs_industry_"):
        return "ts_stock_vs_industry"
    if feature.startswith("TS_industry_"):
        return "ts_industry_state"
    if feature.endswith("_minus_industry") or "_minus_industry_" in feature:
        return "ts_relative_value_quality"
    if feature.startswith("TS_dividend") or feature.startswith("TS_bp") or feature.startswith("TS_sp"):
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


def compare_feature_importance(candidate_name: str, candidate_dir: Path, baseline_name: str, baseline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = _read_csv(candidate_dir, "feature_importance_gain_mean.csv").rename(
        columns={"importance_gain": f"{candidate_name}_importance_gain"}
    )
    baseline = _read_csv(baseline_dir, "feature_importance_gain_mean.csv").rename(
        columns={"importance_gain": f"{baseline_name}_importance_gain"}
    )
    candidate[f"{candidate_name}_rank"] = candidate[f"{candidate_name}_importance_gain"].rank(
        ascending=False,
        method="min",
    )
    baseline[f"{baseline_name}_rank"] = baseline[f"{baseline_name}_importance_gain"].rank(
        ascending=False,
        method="min",
    )
    merged = candidate.merge(baseline, on="feature", how="outer").fillna(0.0)
    merged["importance_diff"] = merged[f"{candidate_name}_importance_gain"] - merged[f"{baseline_name}_importance_gain"]
    merged["feature_family"] = merged["feature"].map(_feature_family)
    family = (
        merged.groupby("feature_family")[
            [f"{candidate_name}_importance_gain", f"{baseline_name}_importance_gain", "importance_diff"]
        ]
        .sum()
        .reset_index()
    )
    candidate_total = family[f"{candidate_name}_importance_gain"].sum()
    baseline_total = family[f"{baseline_name}_importance_gain"].sum()
    family[f"{candidate_name}_share"] = family[f"{candidate_name}_importance_gain"] / candidate_total
    family[f"{baseline_name}_share"] = family[f"{baseline_name}_importance_gain"] / baseline_total
    family["share_diff"] = family[f"{candidate_name}_share"] - family[f"{baseline_name}_share"]
    return merged.sort_values("importance_diff", ascending=False).reset_index(drop=True), family.sort_values("share_diff", ascending=False)


def _parse_mapping(raw: Any) -> dict[str, float]:
    if pd.isna(raw) or raw == "":
        return {}
    value = ast.literal_eval(str(raw))
    if not isinstance(value, dict):
        return {}
    return {str(key): float(val) for key, val in value.items()}


def compare_trace_holdings(candidate_name: str, candidate_dir: Path, baseline_name: str, baseline_dir: Path) -> pd.DataFrame:
    candidate = _read_csv(candidate_dir, "native_backtest_trace.csv")
    baseline = _read_csv(baseline_dir, "native_backtest_trace.csv")
    candidate = candidate[["datetime", "holdings_after", "net_return", "risk_degree", "buy_count", "sell_count"]].rename(
        columns={
            "holdings_after": f"{candidate_name}_holdings_after",
            "net_return": f"{candidate_name}_net_return",
            "risk_degree": f"{candidate_name}_risk_degree",
            "buy_count": f"{candidate_name}_buy_count",
            "sell_count": f"{candidate_name}_sell_count",
        }
    )
    baseline = baseline[["datetime", "holdings_after", "net_return", "risk_degree", "buy_count", "sell_count"]].rename(
        columns={
            "holdings_after": f"{baseline_name}_holdings_after",
            "net_return": f"{baseline_name}_net_return",
            "risk_degree": f"{baseline_name}_risk_degree",
            "buy_count": f"{baseline_name}_buy_count",
            "sell_count": f"{baseline_name}_sell_count",
        }
    )
    merged = candidate.merge(baseline, on="datetime", how="inner")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        candidate_holdings = _parse_mapping(row[f"{candidate_name}_holdings_after"])
        baseline_holdings = _parse_mapping(row[f"{baseline_name}_holdings_after"])
        candidate_symbols = set(candidate_holdings)
        baseline_symbols = set(baseline_holdings)
        union = candidate_symbols | baseline_symbols
        overlap = candidate_symbols & baseline_symbols
        candidate_only = candidate_symbols - baseline_symbols
        baseline_only = baseline_symbols - candidate_symbols
        rows.append(
            {
                "datetime": row["datetime"],
                f"{candidate_name}_net_return": row[f"{candidate_name}_net_return"],
                f"{baseline_name}_net_return": row[f"{baseline_name}_net_return"],
                "net_return_diff": row[f"{candidate_name}_net_return"] - row[f"{baseline_name}_net_return"],
                f"{candidate_name}_risk_degree": row[f"{candidate_name}_risk_degree"],
                f"{baseline_name}_risk_degree": row[f"{baseline_name}_risk_degree"],
                f"{candidate_name}_buy_count": row[f"{candidate_name}_buy_count"],
                f"{baseline_name}_buy_count": row[f"{baseline_name}_buy_count"],
                f"{candidate_name}_sell_count": row[f"{candidate_name}_sell_count"],
                f"{baseline_name}_sell_count": row[f"{baseline_name}_sell_count"],
                "candidate_holding_count": len(candidate_symbols),
                "baseline_holding_count": len(baseline_symbols),
                "overlap_count": len(overlap),
                "candidate_only_count": len(candidate_only),
                "baseline_only_count": len(baseline_only),
                "holding_jaccard": len(overlap) / len(union) if union else 1.0,
                "candidate_only_symbols": "|".join(sorted(candidate_only)),
                "baseline_only_symbols": "|".join(sorted(baseline_only)),
            }
        )
    return pd.DataFrame(rows)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "diagnostics" / "strategy_pair" / f"{stamp}__{args.candidate_name}_vs_{args.baseline_name}"


def _write_readme(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    bucket_shape: pd.DataFrame,
    monthly_diff: pd.DataFrame,
    yearly_monthly_diff: pd.DataFrame,
    family_importance: pd.DataFrame,
    trace_overlap: pd.DataFrame,
) -> None:
    candidate_row = metrics.loc[metrics["run"] == args.candidate_name].iloc[0]
    baseline_row = metrics.loc[metrics["run"] == args.baseline_name].iloc[0]
    candidate_bucket = bucket_shape.loc[bucket_shape["run"] == args.candidate_name].iloc[0]
    baseline_bucket = bucket_shape.loc[bucket_shape["run"] == args.baseline_name].iloc[0]
    candidate_beats = int((monthly_diff["return_diff"] > 0).sum())
    total_months = int(len(monthly_diff))
    worst = monthly_diff.nsmallest(5, "return_diff")[["period", "return_diff"]]
    best = monthly_diff.nlargest(5, "return_diff")[["period", "return_diff"]]
    readme = [
        f"# {args.candidate_name} vs {args.baseline_name}",
        "",
        "## Inputs",
        "",
        f"- candidate_run: `{args.candidate_run}`",
        f"- baseline_run: `{args.baseline_run}`",
        "",
        "## Portfolio Summary",
        "",
        f"- candidate annualized_return: `{candidate_row['annualized_return']:.6f}`",
        f"- baseline annualized_return: `{baseline_row['annualized_return']:.6f}`",
        f"- candidate sharpe: `{candidate_row['sharpe_ratio']:.6f}`",
        f"- baseline sharpe: `{baseline_row['sharpe_ratio']:.6f}`",
        f"- candidate max_drawdown: `{candidate_row['max_drawdown']:.6f}`",
        f"- baseline max_drawdown: `{baseline_row['max_drawdown']:.6f}`",
        "",
        "## Bucket Shape",
        "",
        f"- candidate top_minus_bottom: `{candidate_bucket['top_minus_bottom']:.6f}`",
        f"- baseline top_minus_bottom: `{baseline_bucket['top_minus_bottom']:.6f}`",
        f"- candidate top_minus_middle_best: `{candidate_bucket['top_minus_middle_best']:.6f}`",
        f"- baseline top_minus_middle_best: `{baseline_bucket['top_minus_middle_best']:.6f}`",
        f"- candidate best_bucket: `{int(candidate_bucket['best_bucket'])}`",
        f"- baseline best_bucket: `{int(baseline_bucket['best_bucket'])}`",
        "",
        "## Monthly Difference",
        "",
        f"- candidate beats baseline months: `{candidate_beats} / {total_months}`",
        f"- mean monthly return_diff: `{monthly_diff['return_diff'].mean():.6f}`",
        f"- median monthly return_diff: `{monthly_diff['return_diff'].median():.6f}`",
        "",
        "## Worst Candidate Relative Months",
        "",
        *[f"- `{row.period}`: `{row.return_diff:.6f}`" for row in worst.itertuples(index=False)],
        "",
        "## Best Candidate Relative Months",
        "",
        *[f"- `{row.period}`: `{row.return_diff:.6f}`" for row in best.itertuples(index=False)],
        "",
        "## Yearly Monthly Difference",
        "",
        yearly_monthly_diff.to_markdown(index=False),
        "",
        "## Feature Family Importance",
        "",
        family_importance.to_markdown(index=False),
        "",
        "## Holding Overlap",
        "",
        f"- mean holding_jaccard: `{trace_overlap['holding_jaccard'].mean():.6f}`",
        f"- median holding_jaccard: `{trace_overlap['holding_jaccard'].median():.6f}`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme).strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    candidate_dir = Path(args.candidate_run)
    baseline_dir = Path(args.baseline_run)
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate run directory not found: {candidate_dir}")
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline run directory not found: {baseline_dir}")
    middle_buckets = _parse_bucket_ids(args.middle_buckets)
    run_dirs = {args.candidate_name: candidate_dir, args.baseline_name: baseline_dir}

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = summarize_metrics(run_dirs)
    training = summarize_training(run_dirs)
    buckets, bucket_shape, yearly_bucket_shape = compare_buckets(
        run_dirs,
        top_bucket=int(args.top_bucket),
        middle_buckets=middle_buckets,
    )
    monthly_diff, yearly_monthly_diff = compare_monthly(
        args.candidate_name,
        candidate_dir,
        args.baseline_name,
        baseline_dir,
    )
    daily_diff = compare_daily(args.candidate_name, candidate_dir, args.baseline_name, baseline_dir)
    feature_importance, family_importance = compare_feature_importance(
        args.candidate_name,
        candidate_dir,
        args.baseline_name,
        baseline_dir,
    )
    trace_overlap = compare_trace_holdings(args.candidate_name, candidate_dir, args.baseline_name, baseline_dir)

    metrics.to_csv(output_dir / "portfolio_metrics_compare.csv", index=False)
    training.to_csv(output_dir / "training_metrics_compare.csv", index=False)
    buckets.to_csv(output_dir / "bucket_compare_long.csv", index=False)
    bucket_shape.to_csv(output_dir / "bucket_shape_summary.csv", index=False)
    yearly_bucket_shape.to_csv(output_dir / "yearly_bucket_shape_summary.csv", index=False)
    monthly_diff.to_csv(output_dir / "monthly_return_diff.csv", index=False)
    yearly_monthly_diff.to_csv(output_dir / "yearly_monthly_return_diff.csv", index=False)
    daily_diff.to_csv(output_dir / "daily_return_diff.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance_diff.csv", index=False)
    family_importance.to_csv(output_dir / "feature_family_importance_diff.csv", index=False)
    trace_overlap.to_csv(output_dir / "trace_holding_overlap.csv", index=False)
    trace_overlap.nsmallest(20, "net_return_diff").to_csv(output_dir / "worst_trace_holding_diff.csv", index=False)
    trace_overlap.nlargest(20, "net_return_diff").to_csv(output_dir / "best_trace_holding_diff.csv", index=False)

    _write_readme(
        output_dir,
        args=args,
        metrics=metrics,
        bucket_shape=bucket_shape,
        monthly_diff=monthly_diff,
        yearly_monthly_diff=yearly_monthly_diff,
        family_importance=family_importance,
        trace_overlap=trace_overlap,
    )

    print(f"[+] Strategy-pair diagnostics saved to: {output_dir}")
    print(f"    portfolio metrics: {output_dir / 'portfolio_metrics_compare.csv'}")
    print(f"    bucket shape: {output_dir / 'bucket_shape_summary.csv'}")
    print(f"    monthly diff: {output_dir / 'monthly_return_diff.csv'}")
    print(f"    holding overlap: {output_dir / 'trace_holding_overlap.csv'}")
    print(f"    README: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
