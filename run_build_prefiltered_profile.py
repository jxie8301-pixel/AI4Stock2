"""Build a filtered feature profile from single-factor diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.factor_store import load_factor_frame
from src.feature_prefilter import (
    DEFAULT_MAX_ABS_CORR,
    DEFAULT_MIN_ABS_RANK_IC,
    DEFAULT_MIN_ABS_RANK_IC_IR,
    DEFAULT_MIN_COVERAGE_PCT,
    DEFAULT_MIN_MONTHLY_POSITIVE_RATE,
    DEFAULT_MIN_SEGMENT_DIRECTIONAL_HIT_MEAN,
    DEFAULT_MAX_SEGMENT_RANK_IC_MEAN_RANGE,
    load_diagnostics_summary,
    prefilter_feature_summary,
    prune_exact_duplicate_features,
    prune_correlated_features,
    save_profile_yaml,
)
from src.feature_profiles import get_native_factor_store_dir
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a prefiltered feature profile from diagnostics results.")
    add_common_runtime_args(parser, include_model_arg=False)
    parser.add_argument("--diagnostics-summary", required=True, help="Path to single_factor_summary.csv")
    parser.add_argument(
        "--segment-comparison",
        help="Optional path to single_factor_segment_comparison.csv for regime-aware prefiltering.",
    )
    parser.add_argument(
        "--period",
        choices=["train", "valid", "test", "all"],
        default="train",
        help="Date range used for correlation pruning. Default: train.",
    )
    parser.add_argument("--date-start", help="Optional explicit start date (overrides --period)")
    parser.add_argument("--date-end", help="Optional explicit end date (overrides --period)")
    parser.add_argument("--min-coverage-pct", type=float, default=DEFAULT_MIN_COVERAGE_PCT)
    parser.add_argument("--min-abs-rank-ic", type=float, default=DEFAULT_MIN_ABS_RANK_IC)
    parser.add_argument("--min-abs-rank-ic-ir", type=float, default=DEFAULT_MIN_ABS_RANK_IC_IR)
    parser.add_argument("--min-monthly-positive-rate", type=float, default=DEFAULT_MIN_MONTHLY_POSITIVE_RATE)
    parser.add_argument(
        "--min-segment-directional-hit-mean",
        type=float,
        default=None,
        help=(
            "Optional regime-stability threshold when segment diagnostics are supplied. "
            f"Example baseline: {DEFAULT_MIN_SEGMENT_DIRECTIONAL_HIT_MEAN}"
        ),
    )
    parser.add_argument(
        "--max-segment-rank-ic-mean-range",
        type=float,
        default=None,
        help=(
            "Optional cap on rank-IC drift across supplied segments. "
            f"Example baseline: {DEFAULT_MAX_SEGMENT_RANK_IC_MEAN_RANGE}"
        ),
    )
    parser.add_argument(
        "--exclude-direction-flip",
        action="store_true",
        help="Drop factors whose suggested direction flips across supplied segments.",
    )
    parser.add_argument("--max-abs-corr", type=float, default=DEFAULT_MAX_ABS_CORR)
    parser.add_argument(
        "--no-cross-sectional-rank-corr",
        action="store_true",
        help="Prune redundancy on raw factor values instead of cross-sectionally ranked values.",
    )
    parser.add_argument("--max-features", type=int, help="Optional cap applied after redundancy pruning.")
    parser.add_argument("--profile-name", required=True, help="Output profile name, for example core_v5_diag_prefilter_v1")
    parser.add_argument(
        "--write-config-profile",
        action="store_true",
        help="Write the generated profile into configs/features/<profile-name>.yaml",
    )
    parser.add_argument("--output-dir", help="Optional diagnostics artifact directory for this filtering run.")
    return parser


def _resolve_period_dates(cfg: dict, args: argparse.Namespace) -> tuple[str, str]:
    if args.date_start or args.date_end:
        if not args.date_start or not args.date_end:
            raise ValueError("Provide both --date-start and --date-end when overriding the filter range.")
        return str(args.date_start), str(args.date_end)
    if args.period == "all":
        start = min(cfg["time"]["train"][0], cfg["time"]["valid"][0], cfg["time"]["test"][0])
        end = max(cfg["time"]["train"][1], cfg["time"]["valid"][1], cfg["time"]["test"][1])
        return str(start), str(end)
    split = cfg["time"][args.period]
    return str(split[0]), str(split[1])


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / "diagnostics" / "prefilter_profiles" / f"{stamp}__{args.profile_name}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_validated_config_from_args(args, parser)
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)
    factor_store_dir = get_native_factor_store_dir(cfg)

    summary = load_diagnostics_summary(
        args.diagnostics_summary,
        segment_comparison_path=args.segment_comparison,
    )
    kept, dropped = prefilter_feature_summary(
        summary,
        min_coverage_pct=float(args.min_coverage_pct),
        min_abs_rank_ic=float(args.min_abs_rank_ic),
        min_abs_rank_ic_ir=float(args.min_abs_rank_ic_ir),
        min_monthly_positive_rate=float(args.min_monthly_positive_rate),
        min_segment_directional_hit_mean=args.min_segment_directional_hit_mean,
        max_segment_rank_ic_mean_range=args.max_segment_rank_ic_mean_range,
        exclude_direction_flip=bool(args.exclude_direction_flip),
    )
    if kept.empty:
        raise ValueError("No factors survived the prefilter thresholds.")
    kept_exact, dropped_exact = prune_exact_duplicate_features(kept)
    if kept_exact.empty:
        raise ValueError("No factors remained after exact-duplicate pruning.")

    date_start, date_end = _resolve_period_dates(cfg, args)
    factor_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=kept_exact["feature"].tolist(),
        label_column=label_column,
        date_start=date_start,
        date_end=date_end,
        universe_name=str(cfg.get("universe", "all")),
        universe_dir=cfg.get("native", {}).get("universe_dir", "data/universes"),
        sort_by=("date", "symbol"),
        progress_desc="loading factor frame for redundancy pruning",
    )
    if factor_frame.empty:
        raise ValueError("Factor frame is empty for the requested pruning period.")

    kept_pruned, dropped_corr = prune_correlated_features(
        factor_frame,
        kept_exact,
        corr_threshold=float(args.max_abs_corr),
        use_cross_sectional_rank=not args.no_cross_sectional_rank_corr,
    )
    selected_frame = kept_pruned.copy()
    if args.max_features is not None:
        selected_frame = selected_frame.head(int(args.max_features)).reset_index(drop=True)
    selected_columns = selected_frame["feature"].tolist()
    if not selected_columns:
        raise ValueError("No factors remained after correlation pruning.")

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefilter_path = output_dir / "prefilter_kept.csv"
    dropped_path = output_dir / "prefilter_dropped.csv"
    kept_exact_path = output_dir / "exact_duplicate_kept.csv"
    dropped_exact_path = output_dir / "exact_duplicate_pruned.csv"
    dropped_corr_path = output_dir / "correlation_pruned.csv"
    kept.to_csv(prefilter_path, index=False)
    dropped.to_csv(dropped_path, index=False)
    kept_exact.to_csv(kept_exact_path, index=False)
    dropped_exact.to_csv(dropped_exact_path, index=False)
    dropped_corr.to_csv(dropped_corr_path, index=False)

    profile_output_path = output_dir / f"{args.profile_name}.yaml"
    save_profile_yaml(selected_columns, output_path=profile_output_path)

    if args.write_config_profile:
        config_profile_path = Path("configs") / "features" / f"{args.profile_name}.yaml"
        save_profile_yaml(selected_columns, output_path=config_profile_path)
        print(f"[*] Config profile written: {config_profile_path}")

    readme_path = output_dir / "README.md"
    readme_lines = [
        f"# {args.profile_name}",
        "",
        "## Filter Settings",
        "",
        f"- diagnostics_summary: `{args.diagnostics_summary}`",
        f"- segment_comparison: `{args.segment_comparison}`",
        f"- data_source: `{cfg.get('data', {}).get('source', '')}`",
        f"- universe: `{cfg.get('universe', '')}`",
        f"- period: `{args.period}`",
        f"- date_start: `{date_start}`",
        f"- date_end: `{date_end}`",
        f"- min_coverage_pct: `{args.min_coverage_pct}`",
        f"- min_abs_rank_ic: `{args.min_abs_rank_ic}`",
        f"- min_abs_rank_ic_ir: `{args.min_abs_rank_ic_ir}`",
        f"- min_monthly_positive_rate: `{args.min_monthly_positive_rate}`",
        f"- min_segment_directional_hit_mean: `{args.min_segment_directional_hit_mean}`",
        f"- max_segment_rank_ic_mean_range: `{args.max_segment_rank_ic_mean_range}`",
        f"- exclude_direction_flip: `{args.exclude_direction_flip}`",
        f"- max_abs_corr: `{args.max_abs_corr}`",
        f"- max_features: `{args.max_features}`",
        f"- correlation_space: `{'cross_sectional_rank' if not args.no_cross_sectional_rank_corr else 'raw'}`",
        "",
        "## Counts",
        "",
        f"- original_features: `{len(summary)}`",
        f"- after_prefilter: `{len(kept)}`",
        f"- after_exact_duplicate_prune: `{len(kept_exact)}`",
        f"- after_corr_prune: `{len(selected_columns)}`",
        "",
        "## Selected Features",
        "",
        *[f"- `{name}`" for name in selected_columns],
    ]
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines).strip() + "\n")

    print(f"[+] Prefilter artifacts saved to: {output_dir}")
    print(f"    kept summary: {prefilter_path}")
    print(f"    exact-duplicate kept: {kept_exact_path}")
    print(f"    exact-duplicate drops: {dropped_exact_path}")
    print(f"    corr-pruned drops: {dropped_corr_path}")
    print(f"    profile yaml: {profile_output_path}")
    print(f"[*] Selected features ({len(selected_columns)}):")
    for name in selected_columns:
        print(f"    - {name}")


if __name__ == "__main__":
    main()
