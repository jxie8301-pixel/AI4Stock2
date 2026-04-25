"""Run single-factor diagnostics on the native factor store."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.factor_store import load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import resolve_selected_feature_columns
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args
from src.single_factor_runtime import (
    apply_industry_neutralization,
    derive_diagnostic_label_series,
    resolve_period_dates,
    resolve_segments,
)
from src.single_factor_diagnostics import (
    build_single_factor_diagnostics_bundle,
    save_single_factor_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-factor diagnostics for the current factor universe.")
    add_common_runtime_args(parser, include_model_arg=False)
    parser.add_argument(
        "--period",
        choices=["train", "valid", "test", "all"],
        default="test",
        help="Configured time split to evaluate. Default: test.",
    )
    parser.add_argument("--date-start", help="Optional explicit start date (overrides --period)")
    parser.add_argument("--date-end", help="Optional explicit end date (overrides --period)")
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Diagnose all cached features instead of the current feature profile subset.",
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=5,
        help="Cross-sectional quantile bins used for monotonicity and top-bottom spread checks.",
    )
    parser.add_argument("--top-n", type=int, default=50, help="How many factors to keep in top-factor exports.")
    parser.add_argument("--output-dir", help="Optional explicit output directory.")
    parser.add_argument(
        "--no-detail-artifacts",
        action="store_true",
        help="Skip daily bucket/spread/monthly/missing CSV artifacts when only summary diagnostics are needed.",
    )
    parser.add_argument(
        "--segment-scheme",
        choices=["none", "config_split", "yearly"],
        default="none",
        help=(
            "Optional segmented diagnostics scheme. "
            "`config_split` compares train/valid/test over the currently loaded range. "
            "`yearly` creates one segment per calendar year in range."
        ),
    )
    parser.add_argument(
        "--segments",
        help=(
            "Optional custom segments in 'name:start:end;name2:start:end' form. "
            "This is evaluated after the main date filter and can be combined with --segment-scheme=config_split."
        ),
    )
    parser.add_argument(
        "--diagnostic-label-space",
        choices=["raw_return", "industry_excess", "benchmark_excess"],
        default="raw_return",
        help="Which realized label space to diagnose factors against. Default: raw_return.",
    )
    parser.add_argument(
        "--diagnostic-threshold",
        type=float,
        default=0.0,
        help="Optional excess hurdle for relative diagnostics. Default: 0.0.",
    )
    parser.add_argument(
        "--industry-neutral",
        action="store_true",
        help="Demean each factor within date x industry before diagnostics when industry groups are available.",
    )
    return parser


def _default_output_dir(cfg: dict, args: argparse.Namespace, *, signal_horizon: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    run_tag = str(getattr(args, "run_tag", "") or "").strip()
    tag_suffix = f"__{run_tag}" if run_tag else ""
    feature_profile = str(cfg.get("features", {}).get("profile") or "all")
    data_source = str(cfg.get("data", {}).get("source") or "default")
    universe = str(cfg.get("universe") or "all")
    label_space = str(getattr(args, "diagnostic_label_space", "raw_return") or "raw_return")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("results")
        / "diagnostics"
        / "single_factor"
        / f"{stamp}__{data_source}__{universe}__{feature_profile}__h{signal_horizon}__{label_space}__{args.period}{tag_suffix}"
    )

def main() -> None:
    start_time = perf_counter()
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_validated_config_from_args(args, parser)

    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)
    factor_store_dir = get_native_factor_store_dir(cfg)
    factor_store_meta = load_factor_store_metadata(factor_store_dir)
    if args.all_features:
        feature_names = list(factor_store_meta.get("feature_names", []))
    else:
        _, source_columns = resolve_selected_feature_columns(factor_store_meta, cfg)
        feature_names = list(dict.fromkeys(source_columns))
    if not feature_names:
        raise ValueError("No features resolved for diagnostics.")

    date_start, date_end = resolve_period_dates(cfg, args)
    print(
        f"[*] Single-factor diagnostics: period={args.period}, "
        f"date_start={date_start}, date_end={date_end}, features={len(feature_names)}"
    )
    load_started = perf_counter()
    factor_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=feature_names,
        label_column=label_column,
        date_start=date_start,
        date_end=date_end,
        universe_name=str(cfg.get("universe", "all")),
        universe_dir=cfg.get("native", {}).get("universe_dir", "data/universes"),
        sort_by=("date", "symbol"),
        progress_desc="loading diagnostics factor store",
    )
    if factor_frame.empty:
        raise ValueError("Factor store returned no rows for the requested diagnostics period.")
    load_elapsed = perf_counter() - load_started

    diagnostic_label_space = str(args.diagnostic_label_space or "raw_return").strip().lower()
    label_started = perf_counter()
    diagnostic_labels = derive_diagnostic_label_series(
        factor_frame,
        cfg=cfg,
        signal_horizon=signal_horizon,
        diagnostic_label_space=diagnostic_label_space,
        diagnostic_threshold=float(args.diagnostic_threshold),
    )
    factor_frame = factor_frame.copy()
    factor_frame["label"] = diagnostic_labels.to_numpy(dtype=float, copy=False)
    factor_frame = factor_frame.loc[pd.to_numeric(factor_frame["label"], errors="coerce").notna()].reset_index(drop=True)
    if factor_frame.empty:
        raise ValueError("All rows were dropped after applying the requested diagnostic label space.")
    label_elapsed = perf_counter() - label_started

    neutralize_elapsed = 0.0
    neutralized_feature_count = 0
    if bool(getattr(args, "industry_neutral", False)):
        neutralize_started = perf_counter()
        factor_frame, neutralized_feature_count = apply_industry_neutralization(
            factor_frame,
            cfg=cfg,
            feature_names=feature_names,
        )
        neutralize_elapsed = perf_counter() - neutralize_started

    diagnostics_started = perf_counter()
    segments = resolve_segments(cfg, args, main_start=date_start, main_end=date_end)
    bundle = build_single_factor_diagnostics_bundle(
        factor_frame,
        feature_names=feature_names,
        label_column="label",
        quantile_bins=max(int(args.quantile_bins), 2),
        segments=segments,
        include_details=not bool(getattr(args, "no_detail_artifacts", False)),
    )
    diagnostics_elapsed = perf_counter() - diagnostics_started
    summary = bundle.summary
    detail_frames = bundle.detail_frames
    segment_comparison = bundle.segment_comparison
    segment_summaries = bundle.segment_summaries
    output_dir = _default_output_dir(cfg, args, signal_horizon=signal_horizon)
    metadata = {
        "data_source": cfg.get("data", {}).get("source", ""),
        "universe": cfg.get("universe", ""),
        "feature_profile": cfg.get("features", {}).get("profile", ""),
        "factor_store_dir": factor_store_dir,
        "signal_horizon": signal_horizon,
        "period": args.period,
        "date_start": date_start,
        "date_end": date_end,
        "diagnostic_label_space": diagnostic_label_space,
        "diagnostic_threshold": float(args.diagnostic_threshold),
        "industry_neutral": bool(getattr(args, "industry_neutral", False)),
        "neutralized_feature_count": neutralized_feature_count,
        "feature_count": len(feature_names),
        "row_count": len(factor_frame),
        "quantile_bins": max(int(args.quantile_bins), 2),
        "detail_artifacts": detail_frames is not None,
        "segment_scheme": args.segment_scheme,
        "segment_count": len(segment_summaries),
        "load_elapsed_sec": round(load_elapsed, 6),
        "label_elapsed_sec": round(label_elapsed, 6),
        "neutralize_elapsed_sec": round(neutralize_elapsed, 6),
        "diagnostics_elapsed_sec": round(diagnostics_elapsed, 6),
    }
    artifacts = save_single_factor_diagnostics(
        summary,
        output_dir=output_dir,
        config_snapshot=deepcopy(cfg),
        metadata=metadata,
        top_n=max(int(args.top_n), 1),
        segment_comparison=segment_comparison,
        segment_summaries=segment_summaries,
        detail_frames=detail_frames,
    )

    print(f"[+] Single-factor diagnostics saved to: {output_dir}")
    print(
        "    timings:"
        f" load={load_elapsed:.2f}s"
        f" label={label_elapsed:.2f}s"
        f" neutralize={neutralize_elapsed:.2f}s"
        f" diagnostics={diagnostics_elapsed:.2f}s"
        f" total={perf_counter() - start_time:.2f}s"
    )
    print(f"    summary: {artifacts['summary_csv']}")
    print(f"    top abs RankIC: {artifacts['top_abs_rankic_csv']}")
    print(f"    readme: {artifacts['readme_path']}")

    preview = summary.head(min(10, len(summary)))
    if not preview.empty:
        display = preview[
            [
                "feature",
                "rank_ic_mean",
                "rank_ic_ir",
                "coverage_pct",
                "monotonicity_mean",
                "monthly_rank_ic_directional_hit_rate",
            ]
        ].copy()
        print("\nTop factors by absolute RankIC:")
        print(display.to_string(index=False))
    if segment_comparison is not None and not segment_comparison.empty and "direction_flip" in segment_comparison.columns:
        drift_preview = segment_comparison[segment_comparison["direction_flip"].fillna(False)].head(
            min(10, len(segment_comparison))
        )
        if not drift_preview.empty:
            print("\nDirection-flip factors across segments:")
            cols = [col for col in ["feature", "best_segment_by_abs_rank_ic", "worst_segment_by_abs_rank_ic", "segment_rank_ic_mean_range"] if col in drift_preview.columns]
            print(drift_preview[cols].to_string(index=False))


if __name__ == "__main__":
    main()
