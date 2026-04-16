"""Run single-factor diagnostics on the native factor store."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.data_source import resolve_data_source_name
from src.evaluate import build_benchmark_series
from src.factor_store import load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import resolve_selected_feature_columns
from src.label_utils import build_opportunity_edge_series, get_label_column_name, resolve_signal_horizon
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args
from src.single_factor_diagnostics import (
    build_single_factor_detail_frames,
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
    label_space = str(getattr(args, "diagnostic_label_space", "raw_return") or "raw_return")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path("results")
        / "diagnostics"
        / "single_factor"
        / f"{stamp}__{data_source}__{universe}__{feature_profile}__h{signal_horizon}__{label_space}__{args.period}{tag_suffix}"
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
    if args.segment_scheme == "yearly":
        start_ts = pd.Timestamp(main_start)
        end_ts = pd.Timestamp(main_end)
        for year in range(int(start_ts.year), int(end_ts.year) + 1):
            seg_start = max(start_ts, pd.Timestamp(f"{year}-01-01"))
            seg_end = min(end_ts, pd.Timestamp(f"{year}-12-31"))
            if seg_start <= seg_end:
                segments.append((f"y{year}", str(seg_start.date()), str(seg_end.date())))
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


def _load_instrument_industry_groups(
    cfg: dict,
    *,
    instruments: pd.Index,
) -> pd.Series | None:
    data_source = resolve_data_source_name(cfg)
    raw_meta_dir = Path("data") / data_source / "raw" / "meta"
    symbol_cache_path = raw_meta_dir / "symbol_cache.parquet"
    if not symbol_cache_path.exists():
        return None
    try:
        frame = pd.read_parquet(symbol_cache_path, columns=["local_symbol", "industry"])
    except Exception:
        return None
    if frame.empty or "local_symbol" not in frame.columns or "industry" not in frame.columns:
        return None
    frame["local_symbol"] = frame["local_symbol"].astype(str).str.zfill(6)
    frame["industry"] = frame["industry"].fillna("").replace("", pd.NA)
    frame = frame.drop_duplicates("local_symbol", keep="last").set_index("local_symbol")
    instrument_index = pd.Index(instruments.astype(str), dtype=object)
    groups = frame["industry"].reindex(instrument_index)
    return groups if groups.notna().any() else None


def _apply_industry_neutralization(
    frame: pd.DataFrame,
    *,
    cfg: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, int]:
    groups = _load_instrument_industry_groups(
        cfg,
        instruments=frame["symbol"].drop_duplicates(),
    )
    if groups is None:
        raise ValueError("Industry-neutral diagnostics requested, but no industry groups were loaded.")

    out = frame.copy()
    industry_series = pd.Series(out["symbol"].astype(str)).map(groups.to_dict())
    out["_industry_group"] = industry_series.to_numpy(copy=False)
    valid_mask = out["_industry_group"].notna()
    if not bool(valid_mask.any()):
        raise ValueError("Industry-neutral diagnostics requested, but no rows mapped to a valid industry group.")
    neutralized_feature_count = 0
    grouped = out.loc[valid_mask].groupby(["date", "_industry_group"], observed=True)
    for feature_name in feature_names:
        out.loc[valid_mask, feature_name] = (
            out.loc[valid_mask, feature_name]
            - grouped[feature_name].transform("mean")
        )
        neutralized_feature_count += 1
    out = out.drop(columns=["_industry_group"])
    return out, neutralized_feature_count


def _build_forward_compound_return_series(
    daily_returns: pd.Series,
    *,
    horizon: int,
) -> pd.Series:
    clean = pd.Series(daily_returns, copy=True).sort_index().astype(float)
    horizon = max(int(horizon), 1)
    if clean.empty:
        return pd.Series(dtype=float)
    values = clean.to_numpy(dtype=np.float64, copy=False)
    if len(values) <= horizon:
        return pd.Series(np.full(len(clean), np.nan, dtype=np.float64), index=clean.index, dtype=float)
    future_windows = np.lib.stride_tricks.sliding_window_view(values[1:], horizon)
    valid_mask = np.isfinite(future_windows).all(axis=1)
    out = np.full(len(clean), np.nan, dtype=np.float64)
    if bool(valid_mask.any()):
        out[: len(future_windows)][valid_mask] = np.prod(1.0 + future_windows[valid_mask], axis=1) - 1.0
    return pd.Series(out, index=clean.index, dtype=float)


def _derive_diagnostic_label_series(
    frame: pd.DataFrame,
    *,
    cfg: dict,
    signal_horizon: int,
    diagnostic_label_space: str,
    diagnostic_threshold: float,
) -> pd.Series:
    label_space = str(diagnostic_label_space or "raw_return").strip().lower()
    base = frame[["date", "symbol", "label"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base["symbol"] = base["symbol"].astype(str)
    multi_index = pd.MultiIndex.from_arrays(
        [base["date"], base["symbol"]],
        names=["datetime", "instrument"],
    )
    labels = pd.Series(pd.to_numeric(base["label"], errors="coerce").to_numpy(), index=multi_index, dtype=float)

    if label_space == "raw_return":
        return labels

    opportunity_cfg = {
        "mode": label_space,
        "threshold": float(diagnostic_threshold),
        "neutral_band": 0.0,
    }
    instrument_groups = None
    benchmark_forward_returns = None

    if label_space == "industry_excess":
        instrument_groups = _load_instrument_industry_groups(
            cfg,
            instruments=base["symbol"].drop_duplicates(),
        )
        if instrument_groups is None:
            raise ValueError("Industry-excess diagnostics require instrument industry groups, but none were loaded.")
    elif label_space == "benchmark_excess":
        benchmark_series, _ = build_benchmark_series(labels, cfg.get("backtest", {}).get("benchmark"))
        benchmark_forward_returns = _build_forward_compound_return_series(
            benchmark_series,
            horizon=signal_horizon,
        )
        if benchmark_forward_returns.empty:
            raise ValueError("Benchmark-excess diagnostics produced an empty benchmark forward-return series.")

    return build_opportunity_edge_series(
        labels,
        opportunity_cfg=opportunity_cfg,
        instrument_groups=instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
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

    date_start, date_end = _resolve_period_dates(cfg, args)
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
    diagnostic_labels = _derive_diagnostic_label_series(
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
        factor_frame, neutralized_feature_count = _apply_industry_neutralization(
            factor_frame,
            cfg=cfg,
            feature_names=feature_names,
        )
        neutralize_elapsed = perf_counter() - neutralize_started

    summary_started = perf_counter()
    summary = build_single_factor_diagnostics(
        factor_frame,
        feature_names=feature_names,
        label_column="label",
        quantile_bins=max(int(args.quantile_bins), 2),
    )
    summary_elapsed = perf_counter() - summary_started
    detail_started = perf_counter()
    detail_frames = build_single_factor_detail_frames(
        factor_frame,
        feature_names=feature_names,
        label_column="label",
        quantile_bins=max(int(args.quantile_bins), 2),
    )
    detail_elapsed = perf_counter() - detail_started
    segments = _resolve_segments(cfg, args, main_start=date_start, main_end=date_end)
    segment_started = perf_counter()
    segment_comparison, segment_summaries = build_segmented_single_factor_diagnostics(
        factor_frame,
        feature_names=feature_names,
        segments=segments,
        label_column="label",
        quantile_bins=max(int(args.quantile_bins), 2),
    )
    segment_elapsed = perf_counter() - segment_started
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
        "segment_scheme": args.segment_scheme,
        "segment_count": len(segment_summaries),
        "load_elapsed_sec": round(load_elapsed, 6),
        "label_elapsed_sec": round(label_elapsed, 6),
        "neutralize_elapsed_sec": round(neutralize_elapsed, 6),
        "summary_elapsed_sec": round(summary_elapsed, 6),
        "detail_elapsed_sec": round(detail_elapsed, 6),
        "segment_elapsed_sec": round(segment_elapsed, 6),
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
        f" summary={summary_elapsed:.2f}s"
        f" detail={detail_elapsed:.2f}s"
        f" segments={segment_elapsed:.2f}s"
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
