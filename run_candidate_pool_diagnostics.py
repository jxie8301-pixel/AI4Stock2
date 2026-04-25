"""Diagnose whether candidate runs are finding a broad, stable winner pool."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidate_diagnostics import (
    bucket_shape,
    metric_value,
    parse_bucket_ids,
    read_required_csv,
    read_required_json,
)
from src.candidate_profiles import build_candidate_profiles, feature_family, flatten_candidate_profiles, to_jsonable
from src.reference_baselines import (
    CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES,
    CORE_REFERENCE_BASELINE_PREFIXES,
    reference_baseline_summary_fields,
)


CORE_AND_FIXED_RISK_BASELINE_PREFIXES = (
    *CORE_REFERENCE_BASELINE_PREFIXES,
    *CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES,
)


PORTFOLIO_KEYS = [
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "excess_annualized_return",
    "excess_information_ratio",
    "monthly_win_rate",
    "rebalance_win_rate",
    "top_1_positive_month_share",
    "top_3_positive_month_share",
    "top_5_positive_month_share",
    *[
        field
        for prefix in CORE_AND_FIXED_RISK_BASELINE_PREFIXES
        for field in reference_baseline_summary_fields(prefix)
    ],
]

TRAINING_SIGNAL_COLUMNS = [
    "valid_topk_positive_rate",
    "valid_topk_excess_mean",
    "valid_topk_label_mean",
    "valid_topk_min_label_mean",
    "best_valid_daily_rank_ic",
]

LATEST_SLIM_B_TOPK15_RUNS = {
    "old_stable_t10": "results/experiments/native/rolling/lgbm/20260411_204402__native__rolling__lgbm__top10_drop2_wscore_softmax_strank_pct_keepna_minsna_reb10__excess-flow-value-backtest-stable-posrate-industry-excess-flow-value-slim-b-v1",
    "old_offensive_t8": "results/experiments/native/rolling/lgbm/20260411_204756__native__rolling__lgbm__top8_drop2_wscore_softmax_strank_pct_keepna_minsna_reb10__excess-flow-value-backtest-offensive-posrate-industry-excess-flow-value-slim-b-v1",
    "new_stable_t10": "results/experiments/native/rolling/lgbm/20260424_000657__native__rolling__lgbm__top10_drop2_wscore_softmax_strank_pct_keepna_minsna_reb10__slim-b-topk15-candidate-recheck-strategy-n-drop-2-strategy-score-transform-rank-pct-strategy-topk-10-strategy-weighting-score-softmax",
    "new_offensive_t8": "results/experiments/native/rolling/lgbm/20260424_001123__native__rolling__lgbm__top8_drop2_wscore_softmax_strank_pct_keepna_minsna_reb10__slim-b-topk15-candidate-recheck-strategy-n-drop-2-strategy-score-transform-rank-pct-strategy-topk-8-strategy-weighting-score-softmax",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build candidate-pool diagnostics from rolling native backtest artifacts.",
    )
    parser.add_argument(
        "--preset",
        choices=["latest_slim_b_topk15"],
        help="Optional preset run set. Can be combined with explicit --run entries.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Run label and directory. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--candidate-root",
        help=(
            "Optional candidate shortlist root. Accepts either a shortlist directory "
            "containing candidates/ or a candidates/ directory directly."
        ),
    )
    parser.add_argument(
        "--no-sync-candidate-root",
        action="store_true",
        help="Do not copy generated profile outputs back into --candidate-root.",
    )
    parser.add_argument("--output-dir", help="Optional explicit output directory.")
    parser.add_argument("--tag", default="", help="Optional suffix for the default output directory.")
    parser.add_argument("--top-bucket", type=int, default=1, help="Top score bucket id. Default: 1.")
    parser.add_argument(
        "--middle-buckets",
        default="3,4,5,6,7",
        help="Comma-separated bucket ids used as the broad middle pool. Default: 3,4,5,6,7.",
    )
    return parser


def _parse_runs(args: argparse.Namespace) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    if args.preset == "latest_slim_b_topk15":
        runs.update({name: Path(path) for name, path in LATEST_SLIM_B_TOPK15_RUNS.items()})
    if args.candidate_root:
        root = Path(args.candidate_root)
        scan_root = root / "candidates" if (root / "candidates").is_dir() else root
        if not scan_root.is_dir():
            raise FileNotFoundError(f"Candidate root not found: {root}")
        for child in sorted(scan_root.iterdir()):
            snapshot = child / "snapshot"
            if child.is_dir() and snapshot.is_dir():
                runs[child.name] = snapshot
    for raw in args.run:
        if "=" not in raw:
            raise ValueError(f"--run must use NAME=DIR format, got: {raw}")
        name, path = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--run name cannot be empty: {raw}")
        runs[name] = Path(path.strip())
    if not runs:
        raise ValueError("Provide at least one --run or --preset.")
    missing = {name: path for name, path in runs.items() if not path.exists()}
    if missing:
        formatted = ", ".join(f"{name}={path}" for name, path in missing.items())
        raise FileNotFoundError(f"Run directories not found: {formatted}")
    return runs


def _resolve_candidate_dirs(args: argparse.Namespace) -> tuple[Path | None, dict[str, Path]]:
    if not args.candidate_root:
        return None, {}
    root = Path(args.candidate_root)
    candidate_root = root / "candidates" if (root / "candidates").is_dir() else root
    shortlist_root = candidate_root.parent if candidate_root.name == "candidates" else candidate_root
    candidate_dirs: dict[str, Path] = {}
    for child in sorted(candidate_root.iterdir()):
        if child.is_dir() and (child / "snapshot").is_dir():
            candidate_dirs[child.name] = child
    return shortlist_root, candidate_dirs


def _parse_bucket_ids(raw: str) -> set[int]:
    return parse_bucket_ids(raw, option_name="--middle-buckets")


def _read_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    return read_required_csv(run_dir, filename)


def _read_json(run_dir: Path, filename: str) -> dict[str, Any]:
    return read_required_json(run_dir, filename)


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    return metric_value(metrics, key)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"__{args.tag.strip()}" if args.tag.strip() else ""
    return Path("results") / "diagnostics" / "candidate_pool" / f"{stamp}{suffix}"


def summarize_portfolio(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        metrics = _read_json(run_dir, "native_portfolio_metrics.json")
        row: dict[str, Any] = {"run": run_name, "run_dir": str(run_dir)}
        for key in PORTFOLIO_KEYS:
            row[key] = _metric_value(metrics, key)
        row["profitable_month_summary"] = metrics.get("profitable_month_summary")
        row["profitable_rebalance_summary"] = metrics.get("profitable_rebalance_summary")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_yearly(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        monthly = _read_csv(run_dir, "native_monthly_summary.csv")
        monthly["year"] = monthly["period"].astype(str).str[:4]
        for year, group in monthly.groupby("year", sort=True):
            returns = pd.to_numeric(group["return"], errors="coerce")
            bench = pd.to_numeric(group["bench_return"], errors="coerce")
            excess = pd.to_numeric(group["excess_vs_benchmark"], errors="coerce")
            rows.append(
                {
                    "run": run_name,
                    "year": year,
                    "month_count": int(len(group)),
                    "compound_return": float((1.0 + returns).prod() - 1.0),
                    "benchmark_compound_return": float((1.0 + bench).prod() - 1.0),
                    "compound_excess_vs_benchmark": float((1.0 + returns).prod() - (1.0 + bench).prod()),
                    "negative_months": int((returns < 0).sum()),
                    "months_beating_benchmark": int((excess > 0).sum()),
                    "worst_month_return": float(returns.min()),
                    "best_month_return": float(returns.max()),
                    "mean_monthly_turnover": float(pd.to_numeric(group["avg_turnover"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_concentration(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        monthly = _read_csv(run_dir, "native_monthly_summary.csv")
        returns = pd.to_numeric(monthly["return"], errors="coerce")
        positive = returns[returns > 0].sort_values(ascending=False)
        positive_sum = float(positive.sum())
        raw_sum = float(returns.sum())
        row: dict[str, Any] = {
            "run": run_name,
            "positive_month_count": int((returns > 0).sum()),
            "negative_month_count": int((returns < 0).sum()),
            "raw_monthly_return_sum": raw_sum,
            "positive_month_return_sum": positive_sum,
            "top3_positive_share": float(positive.head(3).sum() / positive_sum) if positive_sum else float("nan"),
            "top5_positive_share": float(positive.head(5).sum() / positive_sum) if positive_sum else float("nan"),
            "top3_raw_sum_share": float(positive.head(3).sum() / raw_sum) if raw_sum else float("nan"),
            "top5_raw_sum_share": float(positive.head(5).sum() / raw_sum) if raw_sum else float("nan"),
        }
        for rank, idx in enumerate(positive.head(5).index, start=1):
            row[f"best_month_{rank}"] = str(monthly.loc[idx, "period"])
            row[f"best_month_{rank}_return"] = float(returns.loc[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_shape(frame: pd.DataFrame, *, top_bucket: int, middle_buckets: set[int]) -> dict[str, Any]:
    return bucket_shape(frame, top_bucket=top_bucket, middle_buckets=middle_buckets)


def summarize_buckets(runs: dict[str, Path], *, top_bucket: int, middle_buckets: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    shape_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        bucket = _read_csv(run_dir, "native_score_bucket_report.csv")
        shape_rows.append({"run": run_name, **_bucket_shape(bucket, top_bucket=top_bucket, middle_buckets=middle_buckets)})
        yearly = _read_csv(run_dir, "native_score_bucket_yearly_report.csv")
        for year, group in yearly.groupby("year", sort=True):
            yearly_rows.append(
                {
                    "run": run_name,
                    "year": str(year),
                    **_bucket_shape(group, top_bucket=top_bucket, middle_buckets=middle_buckets),
                }
            )
    return pd.DataFrame(shape_rows), pd.DataFrame(yearly_rows)


def _feature_family(feature: str) -> str:
    return feature_family(feature)


def summarize_feature_families(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        try:
            frame = _read_csv(run_dir, "feature_importance_gain_mean.csv")
        except FileNotFoundError:
            continue
        frame["feature_family"] = frame["feature"].astype(str).map(_feature_family)
        frame["importance_gain"] = pd.to_numeric(frame["importance_gain"], errors="coerce").fillna(0.0)
        total = float(frame["importance_gain"].sum())
        family = frame.groupby("feature_family", sort=True)["importance_gain"].sum()
        for family_name, importance in family.items():
            rows.append(
                {
                    "run": run_name,
                    "feature_family": family_name,
                    "importance_gain": float(importance),
                    "importance_share": float(importance / total) if total else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def summarize_validation_signal_bins(runs: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run_dir in runs.items():
        training = _read_csv(run_dir, "training_summary.csv")
        rebalances = _read_csv(run_dir, "native_rebalance_summary.csv")
        if "window_start" not in training or "period_start" not in rebalances:
            continue
        rebalances = rebalances.set_index("period_start")
        for signal in TRAINING_SIGNAL_COLUMNS:
            if signal not in training:
                continue
            aligned_rows: list[dict[str, float]] = []
            for _, train_row in training.iterrows():
                period_start = str(train_row["window_start"])
                if period_start not in rebalances.index:
                    continue
                rebalance_row = rebalances.loc[period_start]
                aligned_rows.append(
                    {
                        "signal": float(pd.to_numeric(train_row[signal], errors="coerce")),
                        "return": float(pd.to_numeric(rebalance_row["return"], errors="coerce")),
                        "excess_vs_benchmark": float(
                            pd.to_numeric(rebalance_row["excess_vs_benchmark"], errors="coerce")
                        ),
                    }
                )
            aligned = pd.DataFrame(aligned_rows).dropna()
            if len(aligned) < 9:
                continue
            ranked = aligned.sort_values("signal").reset_index(drop=True)
            thirds = len(ranked) // 3
            slices = {
                "low": ranked.iloc[:thirds],
                "mid": ranked.iloc[thirds : 2 * thirds],
                "high": ranked.iloc[2 * thirds :],
            }
            for bin_name, group in slices.items():
                rows.append(
                    {
                        "run": run_name,
                        "signal_metric": signal,
                        "bin": bin_name,
                        "window_count": int(len(group)),
                        "signal_mean": float(group["signal"].mean()),
                        "rebalance_return_mean": float(group["return"].mean()),
                        "rebalance_excess_mean": float(group["excess_vs_benchmark"].mean()),
                        "positive_rebalance_count": int((group["return"] > 0).sum()),
                        "positive_rebalance_rate": float((group["return"] > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int | None = None) -> list[str]:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return ["_No rows._"]
    rows = [[" " + str(col) + " " for col in columns]]
    rows.extend([" " + _fmt(row[col]) + " " for col in columns] for _, row in frame[columns].iterrows())
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(columns))]
    out = []
    header = "|" + "|".join(rows[0][idx].ljust(widths[idx]) for idx in range(len(columns))) + "|"
    sep = "|" + "|".join("-" * widths[idx] for idx in range(len(columns))) + "|"
    out.extend([header, sep])
    for row in rows[1:]:
        out.append("|" + "|".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + "|")
    return out


def write_readme(
    output_dir: Path,
    *,
    runs: dict[str, Path],
    portfolio: pd.DataFrame,
    yearly: pd.DataFrame,
    concentration: pd.DataFrame,
    bucket_shape: pd.DataFrame,
    yearly_bucket_shape: pd.DataFrame,
    validation_bins: pd.DataFrame,
    candidate_profiles: pd.DataFrame,
) -> None:
    validation_display = validation_bins[
        validation_bins["signal_metric"].isin(["valid_topk_positive_rate", "valid_topk_excess_mean"])
    ].copy()
    if not validation_display.empty:
        validation_display["bin_order"] = validation_display["bin"].map({"low": 0, "mid": 1, "high": 2})
        validation_display = validation_display.sort_values(["run", "signal_metric", "bin_order"])
    lines: list[str] = [
        "# Candidate Pool Diagnostics",
        "",
        "## Inputs",
        "",
        *[f"- `{name}`: `{path}`" for name, path in runs.items()],
        "",
        "## Portfolio Summary",
        "",
        *_markdown_table(
            portfolio,
            [
                "run",
                "annualized_return",
                "sharpe_ratio",
                "max_drawdown",
                "excess_annualized_return",
                "excess_information_ratio",
                "monthly_win_rate",
            ],
        ),
        "",
        "## Same-Gate Reference Baseline Edges",
        "",
        *_markdown_table(
            portfolio,
            [
                "run",
                "rank_avg_factor_baseline_excess_annualized_return",
                "rebalances_beating_rank_avg_factor_baseline_pct",
                "rank_ic_weighted_factor_baseline_excess_annualized_return",
                "rebalances_beating_rank_ic_weighted_factor_baseline_pct",
            ],
        ),
        "",
        "## Fixed-Risk Pure Reference Baseline Edges",
        "",
        *_markdown_table(
            portfolio,
            [
                "run",
                "fixed_risk_rank_avg_factor_baseline_excess_annualized_return",
                "rebalances_beating_fixed_risk_rank_avg_factor_baseline_pct",
                "fixed_risk_rank_ic_weighted_factor_baseline_excess_annualized_return",
                "rebalances_beating_fixed_risk_rank_ic_weighted_factor_baseline_pct",
            ],
        ),
        "",
        "## Yearly Path",
        "",
        *_markdown_table(
            yearly,
            [
                "run",
                "year",
                "compound_return",
                "benchmark_compound_return",
                "negative_months",
                "months_beating_benchmark",
                "worst_month_return",
            ],
        ),
        "",
        "## Concentration",
        "",
        *_markdown_table(
            concentration,
            [
                "run",
                "positive_month_count",
                "top3_positive_share",
                "top5_positive_share",
                "top3_raw_sum_share",
                "top5_raw_sum_share",
            ],
        ),
        "",
        "## Bucket Shape",
        "",
        *_markdown_table(
            bucket_shape,
            [
                "run",
                "top_minus_bottom",
                "top_minus_middle_best",
                "best_bucket",
                "top_bucket_label_rank",
                "bucket_label_spearman",
            ],
        ),
        "",
        "## Yearly Bucket Weak Spots",
        "",
        *_markdown_table(
            yearly_bucket_shape.sort_values(["top_minus_bottom", "bucket_label_spearman"], ascending=[True, True]),
            [
                "run",
                "year",
                "top_minus_bottom",
                "top_minus_middle_best",
                "best_bucket",
                "top_bucket_label_rank",
                "bucket_label_spearman",
            ],
            max_rows=12,
        ),
        "",
        "## Validation Signal Bins",
        "",
        *_markdown_table(
            validation_display,
            [
                "run",
                "signal_metric",
                "bin",
                "window_count",
                "signal_mean",
                "rebalance_return_mean",
                "rebalance_excess_mean",
                "positive_rebalance_rate",
            ],
        ),
        "",
        "## Candidate Roles",
        "",
        *_markdown_table(
            candidate_profiles,
            [
                "name",
                "candidate_role",
                "passed_gate_count",
                "total_gate_count",
                "failed_gates",
            ],
        ),
        "",
        "## Reading",
        "",
        "- Treat portfolio gains as real only when yearly path, concentration, bucket shape, and validation-bin behavior agree.",
        "- Negative top-minus-middle-best or weak yearly bucket shape means the score is useful as a selector but not yet calibrated enough for aggressive sizing.",
        "- Strong high-bin validation returns support using validation metrics as risk gates rather than fixed exposure.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def sync_outputs_to_candidate_root(
    *,
    output_dir: Path,
    shortlist_root: Path,
    candidate_dirs: dict[str, Path],
    candidate_profiles: list[dict[str, Any]],
    candidate_profile_frame: pd.DataFrame,
) -> None:
    copied_files = [
        "README.md",
        "portfolio_summary.csv",
        "yearly_path_summary.csv",
        "positive_month_concentration.csv",
        "bucket_shape_summary.csv",
        "yearly_bucket_shape_summary.csv",
        "feature_family_importance.csv",
        "validation_signal_bins.csv",
        "candidate_profiles.csv",
        "candidate_profiles.json",
    ]
    for filename in copied_files:
        source = output_dir / filename
        if source.exists():
            shutil.copy2(source, shortlist_root / filename)

    profiles_by_name = {str(profile["name"]): profile for profile in candidate_profiles}
    for candidate_name, candidate_dir in candidate_dirs.items():
        profile = profiles_by_name.get(candidate_name)
        if profile is None:
            continue
        profile_path = candidate_dir / "candidate_profile.json"
        profile_path.write_text(
            json.dumps(to_jsonable(profile), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        candidate_profile_frame[candidate_profile_frame["name"] == candidate_name].to_csv(
            candidate_dir / "candidate_profile.csv",
            index=False,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runs = _parse_runs(args)
    shortlist_root, candidate_dirs = _resolve_candidate_dirs(args)
    middle_buckets = _parse_bucket_ids(args.middle_buckets)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio = summarize_portfolio(runs)
    yearly = summarize_yearly(runs)
    concentration = summarize_concentration(runs)
    bucket_shape, yearly_bucket_shape = summarize_buckets(
        runs,
        top_bucket=int(args.top_bucket),
        middle_buckets=middle_buckets,
    )
    feature_families = summarize_feature_families(runs)
    validation_bins = summarize_validation_signal_bins(runs)
    candidate_profiles = build_candidate_profiles(
        runs,
        top_bucket=int(args.top_bucket),
        middle_buckets=middle_buckets,
    )
    candidate_profile_frame = flatten_candidate_profiles(candidate_profiles)

    portfolio.to_csv(output_dir / "portfolio_summary.csv", index=False)
    yearly.to_csv(output_dir / "yearly_path_summary.csv", index=False)
    concentration.to_csv(output_dir / "positive_month_concentration.csv", index=False)
    bucket_shape.to_csv(output_dir / "bucket_shape_summary.csv", index=False)
    yearly_bucket_shape.to_csv(output_dir / "yearly_bucket_shape_summary.csv", index=False)
    feature_families.to_csv(output_dir / "feature_family_importance.csv", index=False)
    validation_bins.to_csv(output_dir / "validation_signal_bins.csv", index=False)
    candidate_profile_frame.to_csv(output_dir / "candidate_profiles.csv", index=False)
    (output_dir / "candidate_profiles.json").write_text(
        json.dumps(to_jsonable(candidate_profiles), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(
        output_dir,
        runs=runs,
        portfolio=portfolio,
        yearly=yearly,
        concentration=concentration,
        bucket_shape=bucket_shape,
        yearly_bucket_shape=yearly_bucket_shape,
        validation_bins=validation_bins,
        candidate_profiles=candidate_profile_frame,
    )
    if shortlist_root is not None and not args.no_sync_candidate_root:
        sync_outputs_to_candidate_root(
            output_dir=output_dir,
            shortlist_root=shortlist_root,
            candidate_dirs=candidate_dirs,
            candidate_profiles=candidate_profiles,
            candidate_profile_frame=candidate_profile_frame,
        )

    print(f"[+] Candidate-pool diagnostics saved to: {output_dir}")
    print(f"    README: {output_dir / 'README.md'}")
    print(f"    portfolio: {output_dir / 'portfolio_summary.csv'}")
    print(f"    yearly: {output_dir / 'yearly_path_summary.csv'}")
    print(f"    bucket shape: {output_dir / 'bucket_shape_summary.csv'}")
    print(f"    validation bins: {output_dir / 'validation_signal_bins.csv'}")
    print(f"    candidate profiles: {output_dir / 'candidate_profiles.csv'}")
    if shortlist_root is not None and not args.no_sync_candidate_root:
        print(f"    synced candidate root: {shortlist_root}")


if __name__ == "__main__":
    main()
