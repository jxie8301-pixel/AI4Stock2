"""Run single-factor diagnostics on the native factor store."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.factor_store import load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import resolve_selected_feature_columns
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args
from src.single_factor_diagnostics import (
    build_segmented_single_factor_diagnostics,
    build_single_factor_diagnostics,
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
        "--segment-scheme",
        choices=["none", "config_split"],
        default="none",
        help=(
            "Optional segmented diagnostics scheme. "
            "`config_split` compares train/valid/test over the currently loaded range."
        ),
    )
    parser.add_argument(
        "--segments",
        help=(
            "Optional custom segments in 'name:start:end;name2:start:end' form. "
            "This is evaluated after the main date filter and can be combined with --segment-scheme=config_split."
        ),
    )
    return parser


def _resolve_period_dates(cfg: dict, args: argparse.Namespace) -> tuple[str, str]:
    if args.date_start or args.date_end:
        if not args.date_start or not args.date_end:
            raise ValueError("Provide both --date-start and --date-end when overriding the diagnostics range.")
        return str(args.date_start), str(args.date_end)
    if args.period == "all":
        start = min(cfg["time"]["train"][0], cfg["time"]["valid"][0], cfg["time"]["test"][0])
        end = max(cfg["time"]["train"][1], cfg["time"]["valid"][1], cfg["time"]["test"][1])
        return str(start), str(end)
    split = cfg["time"][args.period]
    return str(split[0]), str(split[1])


def _default_output_dir(cfg: dict, args: argparse.Namespace, *, signal_horizon: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    run_tag = str(getattr(args, "run_tag", "") or "").strip()
    tag_suffix = f"__{run_tag}" if run_tag else ""
    feature_profile = str(cfg.get("features", {}).get("profile") or "all")
    data_source = str(cfg.get("data", {}).get("source") or "default")
    universe = str(cfg.get("universe") or "all")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("results")
        / "diagnostics"
        / "single_factor"
        / f"{stamp}__{data_source}__{universe}__{feature_profile}__h{signal_horizon}__{args.period}{tag_suffix}"
    )


def _parse_custom_segments(raw: str | None) -> list[tuple[str, str, str]]:
    if raw is None or not str(raw).strip():
        return []
    segments: list[tuple[str, str, str]] = []
    for item in str(raw).split(";"):
        text = item.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "Custom segments must use 'name:start:end' format separated by ';'. "
                f"Got: {text}"
            )
        segments.append((parts[0], parts[1], parts[2]))
    return segments


def _resolve_segments(cfg: dict, args: argparse.Namespace, *, main_start: str, main_end: str) -> list[tuple[str, str, str]]:
    segments: list[tuple[str, str, str]] = []
    if args.segment_scheme == "config_split":
        main_start_ts = pd.Timestamp(main_start)
        main_end_ts = pd.Timestamp(main_end)
        for name in ("train", "valid", "test"):
            split = cfg["time"].get(name)
            if not split:
                continue
            start = pd.Timestamp(split[0])
            end = pd.Timestamp(split[1])
            clipped_start = max(start, main_start_ts)
            clipped_end = min(end, main_end_ts)
            if clipped_start <= clipped_end:
                segments.append((name, str(clipped_start.date()), str(clipped_end.date())))
    segments.extend(_parse_custom_segments(args.segments))
    deduped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for name, start, end in segments:
        if name in seen:
            continue
        deduped.append((name, start, end))
        seen.add(name)
    return deduped


def main() -> None:
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

    date_start, date_end = _resolve_period_dates(cfg, args)
    print(
        f"[*] Single-factor diagnostics: period={args.period}, "
        f"date_start={date_start}, date_end={date_end}, features={len(feature_names)}"
    )
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

    summary = build_single_factor_diagnostics(
        factor_frame,
        feature_names=feature_names,
        label_column="label",
        quantile_bins=max(int(args.quantile_bins), 2),
    )
    segments = _resolve_segments(cfg, args, main_start=date_start, main_end=date_end)
    segment_comparison, segment_summaries = build_segmented_single_factor_diagnostics(
        factor_frame,
        feature_names=feature_names,
        segments=segments,
        label_column="label",
        quantile_bins=max(int(args.quantile_bins), 2),
    )
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
        "feature_count": len(feature_names),
        "row_count": len(factor_frame),
        "quantile_bins": max(int(args.quantile_bins), 2),
        "segment_scheme": args.segment_scheme,
        "segment_count": len(segment_summaries),
    }
    artifacts = save_single_factor_diagnostics(
        summary,
        output_dir=output_dir,
        config_snapshot=deepcopy(cfg),
        metadata=metadata,
        top_n=max(int(args.top_n), 1),
        segment_comparison=segment_comparison,
        segment_summaries=segment_summaries,
    )

    print(f"[+] Single-factor diagnostics saved to: {output_dir}")
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
