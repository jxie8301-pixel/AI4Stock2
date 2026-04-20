"""Run multiple single-factor diagnostics cases with one shared factor-store load."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from run_single_factor_diagnostics import (
    _apply_industry_neutralization,
    _derive_diagnostic_label_series,
    _resolve_period_dates,
    _resolve_segments,
)
from src.factor_store import load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import resolve_selected_feature_columns
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.override_utils import parse_override_arg
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args
from src.single_factor_diagnostics import (
    build_single_factor_detail_frames,
    build_segmented_single_factor_diagnostics,
    build_single_factor_diagnostics,
    save_single_factor_diagnostics,
)


@dataclass(frozen=True)
class DiagnosticsBatchCase:
    name: str
    feature_profile: str
    diagnostic_label_space: str
    diagnostic_threshold: float
    output_dir: str | None
    run_tag: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple single-factor diagnostics cases while reusing one in-memory factor frame."
    )
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
        help="Diagnose all cached features instead of each case's feature-profile subset.",
    )
    parser.add_argument(
        "--quantile-bins",
        type=int,
        default=5,
        help="Cross-sectional quantile bins used for monotonicity and top-bottom spread checks.",
    )
    parser.add_argument("--top-n", type=int, default=50, help="How many factors to keep in top-factor exports.")
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
        "--base-output-dir",
        help="Optional root directory for per-case outputs. Defaults to a timestamped diagnostics batch directory.",
    )
    parser.add_argument(
        "--summary-path",
        help="Optional TSV summary path. Defaults to <base-output-dir>/batch_summary.tsv",
    )
    parser.add_argument(
        "--manifest-path",
        help="Optional TSV manifest path. Defaults to <base-output-dir>/batch_manifest.tsv",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs="+",
        required=True,
        help=(
            "Case overrides in key=value form. "
            "Required keys: name, feature_profile. "
            "Optional keys: diagnostic_label_space, diagnostic_threshold, output_dir, run_tag."
        ),
    )
    parser.add_argument(
        "--industry-neutral",
        action="store_true",
        help="Demean each factor within date x industry before diagnostics when industry groups are available.",
    )
    return parser


def _slugify(value: str) -> str:
    lowered = str(value).strip().lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in lowered)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "case"


def _parse_cases(raw_cases: list[list[str]] | None) -> list[DiagnosticsBatchCase]:
    cases: list[DiagnosticsBatchCase] = []
    for raw_group in raw_cases or []:
        payload: dict[str, Any] = {}
        for raw_item in raw_group:
            key, value = parse_override_arg(raw_item)
            payload[str(key)] = value
        feature_profile = str(payload.get("feature_profile") or "").strip()
        if not feature_profile:
            raise ValueError("Each --case must define feature_profile=...")
        name = str(payload.get("name") or feature_profile).strip()
        if not name:
            raise ValueError("Each --case must resolve to a non-empty name")
        diagnostic_label_space = str(payload.get("diagnostic_label_space") or "raw_return").strip().lower()
        cases.append(
            DiagnosticsBatchCase(
                name=name,
                feature_profile=feature_profile,
                diagnostic_label_space=diagnostic_label_space,
                diagnostic_threshold=float(payload.get("diagnostic_threshold", 0.0)),
                output_dir=(str(payload["output_dir"]).strip() if payload.get("output_dir") else None),
                run_tag=(str(payload["run_tag"]).strip() if payload.get("run_tag") else None),
            )
        )
    if not cases:
        raise ValueError("At least one --case is required")
    return cases


def _resolve_case_output_dir(
    base_output_dir: Path,
    case: DiagnosticsBatchCase,
) -> Path:
    if case.output_dir:
        return Path(case.output_dir)
    return base_output_dir / f"{_slugify(case.name)}__{_slugify(case.feature_profile)}__{_slugify(case.diagnostic_label_space)}"


def _init_tsv(path: Path, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(headers)


def _append_tsv_row(path: Path, row: list[Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(row)


def main() -> None:
    overall_started = perf_counter()
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_validated_config_from_args(args, parser)
    cases = _parse_cases(args.case)
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)

    date_start, date_end = _resolve_period_dates(cfg, args)
    base_output_dir = (
        Path(args.base_output_dir)
        if args.base_output_dir
        else Path("results") / "diagnostics" / "single_factor_batch" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    summary_path = Path(args.summary_path) if args.summary_path else base_output_dir / "batch_summary.tsv"
    manifest_path = Path(args.manifest_path) if args.manifest_path else base_output_dir / "batch_manifest.tsv"
    _init_tsv(
        summary_path,
        [
            "case_name",
            "feature_profile",
            "diagnostic_label_space",
            "diagnostic_threshold",
            "feature_count",
            "row_count",
            "top_feature",
            "top_rank_ic_mean",
            "top_rank_ic_ir",
            "top_monotonicity_mean",
            "top_monthly_directional_hit",
            "output_dir",
        ],
    )
    _init_tsv(
        manifest_path,
        [
            "case_name",
            "feature_profile",
            "diagnostic_label_space",
            "diagnostic_threshold",
            "output_dir",
            "readme_path",
            "summary_csv",
            "segment_comparison_csv",
        ],
    )

    factor_store_dir = get_native_factor_store_dir(cfg)
    factor_store_meta = load_factor_store_metadata(factor_store_dir)
    case_feature_map: dict[str, list[str]] = {}
    union_features: list[str] = []
    seen_union: set[str] = set()
    for case in cases:
        if bool(getattr(args, "all_features", False)):
            feature_names = list(factor_store_meta.get("feature_names", []))
        else:
            case_cfg = deepcopy(cfg)
            case_cfg.setdefault("features", {})
            case_cfg["features"]["profile"] = case.feature_profile
            _, source_columns = resolve_selected_feature_columns(factor_store_meta, case_cfg)
            feature_names = list(dict.fromkeys(source_columns))
        if not feature_names:
            raise ValueError(f"Case '{case.name}' resolved no features")
        case_feature_map[case.name] = feature_names
        for feature_name in feature_names:
            if feature_name not in seen_union:
                union_features.append(feature_name)
                seen_union.add(feature_name)

    print(
        f"[*] Single-factor diagnostics batch: cases={len(cases)}, "
        f"date_start={date_start}, date_end={date_end}, union_features={len(union_features)}"
    )
    load_started = perf_counter()
    base_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=union_features,
        label_column=label_column,
        date_start=date_start,
        date_end=date_end,
        universe_name=str(cfg.get("universe", "all")),
        universe_dir=cfg.get("native", {}).get("universe_dir", "data/universes"),
        sort_by=("date", "symbol"),
        progress_desc="loading diagnostics factor store",
    )
    if base_frame.empty:
        raise ValueError("Factor store returned no rows for the requested diagnostics period.")
    load_elapsed = perf_counter() - load_started

    segments = _resolve_segments(cfg, args, main_start=date_start, main_end=date_end)
    label_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for case in cases:
        label_key = (case.diagnostic_label_space, float(case.diagnostic_threshold))
        if label_key in label_cache:
            continue
        case_labels = _derive_diagnostic_label_series(
            base_frame,
            cfg=cfg,
            signal_horizon=signal_horizon,
            diagnostic_label_space=case.diagnostic_label_space,
            diagnostic_threshold=case.diagnostic_threshold,
        )
        label_values = case_labels.to_numpy(dtype=float, copy=False)
        valid_mask = np.isfinite(label_values)
        label_cache[label_key] = (label_values, valid_mask)

    print(f"[*] Shared load done in {load_elapsed:.2f}s; running cases in-memory...")
    for idx, case in enumerate(cases, start=1):
        case_started = perf_counter()
        feature_names = case_feature_map[case.name]
        label_values, valid_mask = label_cache[(case.diagnostic_label_space, float(case.diagnostic_threshold))]
        selected_columns = ["date", "symbol", *feature_names]
        case_frame = base_frame.loc[valid_mask, selected_columns].copy()
        case_frame["label"] = label_values[valid_mask]
        if case_frame.empty:
            raise ValueError(f"Case '{case.name}' dropped all rows after applying diagnostic labels.")

        neutralized_feature_count = 0
        neutralize_elapsed = 0.0
        if bool(getattr(args, "industry_neutral", False)):
            neutralize_started = perf_counter()
            case_frame, neutralized_feature_count = _apply_industry_neutralization(
                case_frame,
                cfg=cfg,
                feature_names=feature_names,
            )
            neutralize_elapsed = perf_counter() - neutralize_started
        summary = build_single_factor_diagnostics(
            case_frame,
            feature_names=feature_names,
            label_column="label",
            quantile_bins=max(int(args.quantile_bins), 2),
        )
        detail_frames = build_single_factor_detail_frames(
            case_frame,
            feature_names=feature_names,
            label_column="label",
            quantile_bins=max(int(args.quantile_bins), 2),
        )
        segment_comparison, segment_summaries = build_segmented_single_factor_diagnostics(
            case_frame,
            feature_names=feature_names,
            segments=segments,
            label_column="label",
            quantile_bins=max(int(args.quantile_bins), 2),
        )

        case_args = deepcopy(args)
        case_args.run_tag = case.run_tag or case.name
        case_args.diagnostic_label_space = case.diagnostic_label_space
        output_dir = _resolve_case_output_dir(base_output_dir, case)
        metadata = {
            "data_source": cfg.get("data", {}).get("source", ""),
            "universe": cfg.get("universe", ""),
            "feature_profile": case.feature_profile,
            "factor_store_dir": factor_store_dir,
            "signal_horizon": signal_horizon,
            "period": args.period,
            "date_start": date_start,
            "date_end": date_end,
            "diagnostic_label_space": case.diagnostic_label_space,
            "diagnostic_threshold": float(case.diagnostic_threshold),
            "industry_neutral": bool(getattr(args, "industry_neutral", False)),
            "neutralized_feature_count": neutralized_feature_count,
            "feature_count": len(feature_names),
            "row_count": len(case_frame),
            "quantile_bins": max(int(args.quantile_bins), 2),
            "segment_scheme": args.segment_scheme,
            "segment_count": len(segment_summaries),
            "shared_load_elapsed_sec": round(load_elapsed, 6),
            "neutralize_elapsed_sec": round(neutralize_elapsed, 6),
            "case_elapsed_sec": round(perf_counter() - case_started, 6),
        }
        case_cfg_snapshot = deepcopy(cfg)
        case_cfg_snapshot.setdefault("features", {})
        case_cfg_snapshot["features"]["profile"] = case.feature_profile
        artifacts = save_single_factor_diagnostics(
            summary,
            output_dir=output_dir,
            config_snapshot=case_cfg_snapshot,
            metadata=metadata,
            top_n=max(int(args.top_n), 1),
            segment_comparison=segment_comparison,
            segment_summaries=segment_summaries,
            detail_frames=detail_frames,
        )

        top = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
        _append_tsv_row(
            summary_path,
            [
                case.name,
                case.feature_profile,
                case.diagnostic_label_space,
                case.diagnostic_threshold,
                len(feature_names),
                len(case_frame),
                top.get("feature", ""),
                top.get("rank_ic_mean", ""),
                top.get("rank_ic_ir", ""),
                top.get("monotonicity_mean", ""),
                top.get("monthly_rank_ic_directional_hit_rate", ""),
                str(output_dir),
            ],
        )
        _append_tsv_row(
            manifest_path,
            [
                case.name,
                case.feature_profile,
                case.diagnostic_label_space,
                case.diagnostic_threshold,
                str(output_dir),
                artifacts.get("readme_path", ""),
                artifacts.get("summary_csv", ""),
                artifacts.get("segment_comparison_csv", ""),
            ],
        )
        print(
            f"[{idx}/{len(cases)}] {case.name}: "
            f"rows={len(case_frame)}, features={len(feature_names)}, "
            f"label_space={case.diagnostic_label_space}, "
            f"elapsed={perf_counter() - case_started:.2f}s"
        )

    print(f"[+] Batch outputs: {base_output_dir}")
    print(f"    summary: {summary_path}")
    print(f"    manifest: {manifest_path}")
    print(f"    total elapsed: {perf_counter() - overall_started:.2f}s")


if __name__ == "__main__":
    main()
