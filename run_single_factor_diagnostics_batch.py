"""Run multiple single-factor diagnostics cases through the Rust engine."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

import pandas as pd

from run_single_factor_diagnostics import _build_rust_single_factor_command
from src.factor_store import load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import resolve_selected_feature_columns
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.override_utils import parse_override_arg
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args
from src.single_factor_runtime import resolve_period_dates, resolve_segments


@dataclass(frozen=True)
class DiagnosticsBatchCase:
    name: str
    feature_profile: str
    baseline_feature_profile: str | None
    diagnostic_label_space: str
    diagnostic_threshold: float
    output_dir: str | None
    run_tag: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multiple single-factor diagnostics cases through the Rust diagnostics engine."
    )
    add_common_runtime_args(parser, include_model_arg=False)
    parser.add_argument(
        "--period",
        choices=["train", "valid", "test", "all"],
        default="train",
        help="Configured time split to evaluate. Default: train. Use test/all only for research-only diagnostics.",
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
            "Optional keys: baseline_feature_profile, diagnostic_label_space, "
            "diagnostic_threshold, output_dir, run_tag."
        ),
    )
    parser.add_argument(
        "--industry-neutral",
        action="store_true",
        help="Demean each factor within date x industry before diagnostics when industry groups are available.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "rust"],
        default="auto",
        help="Compatibility option. The active runtime is Rust.",
    )
    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=64,
        help="Rust engine feature chunk size per factor-store scan. Default: 64.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Rust Arrow record-batch size. Default: 65536.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print delegated Rust commands without running them.")
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
        baseline_feature_profile = (
            payload.get("baseline_feature_profile")
            or payload.get("compare_to_feature_profile")
            or payload.get("baseline_profile")
        )
        diagnostic_label_space = str(payload.get("diagnostic_label_space") or "raw_return").strip().lower()
        cases.append(
            DiagnosticsBatchCase(
                name=name,
                feature_profile=feature_profile,
                baseline_feature_profile=(
                    str(baseline_feature_profile).strip()
                    if baseline_feature_profile and str(baseline_feature_profile).strip()
                    else None
                ),
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


def _resolve_feature_profile_source_columns(
    factor_store_meta: dict[str, Any],
    cfg: dict[str, Any],
    feature_profile: str,
) -> list[str]:
    case_cfg = deepcopy(cfg)
    case_cfg.setdefault("features", {})
    case_cfg["features"]["profile"] = feature_profile
    _, source_columns = resolve_selected_feature_columns(factor_store_meta, case_cfg)
    return list(dict.fromkeys(source_columns))


def _resolve_incremental_feature_names(
    feature_names: list[str],
    baseline_feature_names: list[str] | None,
) -> list[str]:
    baseline_feature_set = set(baseline_feature_names or [])
    return [feature_name for feature_name in feature_names if feature_name not in baseline_feature_set]


def _filter_summary_for_features(summary: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    if not feature_names:
        return summary.iloc[0:0].copy()
    feature_set = set(feature_names)
    return summary.loc[summary["feature"].astype(str).isin(feature_set)].copy().reset_index(drop=True)


def _count_abs_metric_ge(summary: pd.DataFrame, column: str, threshold: float) -> int:
    if summary.empty or column not in summary.columns:
        return 0
    values = pd.to_numeric(summary[column], errors="coerce").abs()
    return int((values >= threshold).sum())


def _sort_top_factors(summary: pd.DataFrame, columns: list[str], ascending: list[bool], top_n: int) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    missing_columns = [column for column in columns if column not in summary.columns]
    if missing_columns:
        return summary.head(top_n).copy()
    return summary.sort_values(columns, ascending=ascending, na_position="last").head(top_n)


def _write_single_factor_subset_artifacts(
    summary: pd.DataFrame,
    *,
    output_dir: str | Path,
    prefix: str,
    top_n: int,
    segment_comparison: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / f"single_factor_{prefix}_summary.csv"
    top_abs_rankic_path = output_path / f"single_factor_{prefix}_top_abs_rankic.csv"
    top_rankic_path = output_path / f"single_factor_{prefix}_top_rankic.csv"
    top_icir_path = output_path / f"single_factor_{prefix}_top_rankic_ir.csv"

    summary.to_csv(summary_path, index=False)
    _sort_top_factors(summary, ["rank_ic_abs_mean", "coverage_pct"], [False, False], top_n).to_csv(
        top_abs_rankic_path,
        index=False,
    )
    _sort_top_factors(summary, ["rank_ic_mean", "coverage_pct"], [False, False], top_n).to_csv(
        top_rankic_path,
        index=False,
    )
    _sort_top_factors(summary, ["rank_ic_ir", "coverage_pct"], [False, False], top_n).to_csv(
        top_icir_path,
        index=False,
    )

    artifacts = {
        f"{prefix}_summary_csv": str(summary_path),
        f"{prefix}_top_abs_rankic_csv": str(top_abs_rankic_path),
        f"{prefix}_top_rankic_csv": str(top_rankic_path),
        f"{prefix}_top_rankic_ir_csv": str(top_icir_path),
    }

    if segment_comparison is not None and "feature" in segment_comparison.columns:
        subset_segment_comparison = _filter_summary_for_features(segment_comparison, feature_names or [])
        segment_comparison_path = output_path / f"single_factor_{prefix}_segment_comparison.csv"
        subset_segment_comparison.to_csv(segment_comparison_path, index=False)
        artifacts[f"{prefix}_segment_comparison_csv"] = str(segment_comparison_path)

    return artifacts


def _merge_artifacts_into_manifest(output_dir: str | Path, artifacts: dict[str, str]) -> None:
    if not artifacts:
        return
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest.setdefault("artifacts", {}).update(artifacts)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)


def _case_namespace(args: argparse.Namespace, case: DiagnosticsBatchCase) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["diagnostic_label_space"] = case.diagnostic_label_space
    payload["diagnostic_threshold"] = case.diagnostic_threshold
    return argparse.Namespace(**payload)


def _read_manifest_metadata(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _run_rust_batch(
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    cases: list[DiagnosticsBatchCase],
    case_feature_map: dict[str, list[str]],
    case_incremental_feature_map: dict[str, list[str]],
    factor_store_dir: Path,
    label_column: str,
    signal_horizon: int,
    date_start: str,
    date_end: str,
    segments: list[tuple[str, str, str]],
    base_output_dir: Path,
    summary_path: Path,
    manifest_path: Path,
    overall_started: float,
) -> None:
    print(
        f"[*] Single-factor diagnostics batch: engine=rust, cases={len(cases)}, "
        f"date_start={date_start}, date_end={date_end}"
    )
    for idx, case in enumerate(cases, start=1):
        case_started = perf_counter()
        feature_names = case_feature_map[case.name]
        incremental_feature_names = case_incremental_feature_map[case.name]
        output_dir = _resolve_case_output_dir(base_output_dir, case)
        case_cfg_snapshot = deepcopy(cfg)
        case_cfg_snapshot.setdefault("features", {})
        case_cfg_snapshot["features"]["profile"] = case.feature_profile
        case_args = _case_namespace(args, case)
        metadata = {
            "data_source": cfg.get("data", {}).get("source", ""),
            "universe": cfg.get("universe", ""),
            "feature_profile": case.feature_profile,
            "baseline_feature_profile": case.baseline_feature_profile or "",
            "factor_store_dir": factor_store_dir,
            "signal_horizon": signal_horizon,
            "period": args.period,
            "date_start": date_start,
            "date_end": date_end,
            "diagnostic_label_space": case.diagnostic_label_space,
            "diagnostic_threshold": float(case.diagnostic_threshold),
            "industry_neutral": bool(getattr(args, "industry_neutral", False)),
            "neutralized_feature_count": len(feature_names) if bool(getattr(args, "industry_neutral", False)) else 0,
            "feature_count": len(feature_names),
            "incremental_feature_count": len(incremental_feature_names),
            "incremental_features": incremental_feature_names,
            "quantile_bins": max(int(args.quantile_bins), 2),
            "detail_artifacts": not bool(getattr(args, "no_detail_artifacts", False)),
            "segment_scheme": args.segment_scheme,
            "segment_count": len(segments),
            "engine": "rust",
        }
        command = _build_rust_single_factor_command(
            cfg=case_cfg_snapshot,
            args=case_args,
            feature_names=feature_names,
            label_column=label_column,
            factor_store_dir=factor_store_dir,
            date_start=date_start,
            date_end=date_end,
            output_dir=output_dir,
            metadata=metadata,
            segments=segments,
        )
        rendered = " ".join(command)
        if bool(getattr(args, "dry_run", False)):
            print(f"[dry-run] {rendered}")
            continue
        print(f"[*] [{idx}/{len(cases)}] Rust diagnostics: {case.name}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(int(completed.returncode))

        summary = pd.read_csv(output_dir / "single_factor_summary.csv")
        incremental_summary = _filter_summary_for_features(summary, incremental_feature_names)
        segment_comparison_path = output_dir / "single_factor_segment_comparison.csv"
        segment_comparison = pd.read_csv(segment_comparison_path) if segment_comparison_path.exists() else None
        incremental_artifacts = _write_single_factor_subset_artifacts(
            incremental_summary,
            output_dir=output_dir,
            prefix="incremental",
            top_n=max(int(args.top_n), 1),
            segment_comparison=segment_comparison,
            feature_names=incremental_feature_names,
        )
        _merge_artifacts_into_manifest(output_dir, incremental_artifacts)

        top = summary.iloc[0] if not summary.empty else pd.Series(dtype=object)
        incremental_top = incremental_summary.iloc[0] if not incremental_summary.empty else pd.Series(dtype=object)
        manifest_metadata = _read_manifest_metadata(output_dir)
        row_count = manifest_metadata.get("row_count", "")
        artifacts = {
            "readme_path": str(output_dir / "README.md"),
            "summary_csv": str(output_dir / "single_factor_summary.csv"),
            "segment_comparison_csv": str(segment_comparison_path) if segment_comparison_path.exists() else "",
            **incremental_artifacts,
        }
        _append_tsv_row(
            summary_path,
            [
                case.name,
                case.feature_profile,
                case.baseline_feature_profile or "",
                case.diagnostic_label_space,
                case.diagnostic_threshold,
                len(feature_names),
                len(incremental_feature_names),
                row_count,
                top.get("feature", ""),
                top.get("rank_ic_mean", ""),
                top.get("rank_ic_abs_mean", ""),
                top.get("rank_ic_ir", ""),
                top.get("monotonicity_mean", ""),
                top.get("monthly_rank_ic_directional_hit_rate", ""),
                incremental_top.get("feature", ""),
                incremental_top.get("rank_ic_mean", ""),
                incremental_top.get("rank_ic_abs_mean", ""),
                incremental_top.get("rank_ic_ir", ""),
                incremental_top.get("monotonicity_mean", ""),
                incremental_top.get("monthly_rank_ic_directional_hit_rate", ""),
                _count_abs_metric_ge(incremental_summary, "rank_ic_mean", 0.03),
                _count_abs_metric_ge(incremental_summary, "rank_ic_mean", 0.05),
                str(output_dir),
            ],
        )
        _append_tsv_row(
            manifest_path,
            [
                case.name,
                case.feature_profile,
                case.baseline_feature_profile or "",
                case.diagnostic_label_space,
                case.diagnostic_threshold,
                str(output_dir),
                artifacts.get("readme_path", ""),
                artifacts.get("summary_csv", ""),
                artifacts.get("incremental_summary_csv", ""),
                artifacts.get("segment_comparison_csv", ""),
                artifacts.get("incremental_segment_comparison_csv", ""),
            ],
        )
        print(
            f"[{idx}/{len(cases)}] {case.name}: rows={row_count}, "
            f"features={len(feature_names)}, incremental_features={len(incremental_feature_names)}, "
            f"elapsed={perf_counter() - case_started:.2f}s"
        )

    print(f"[+] Batch outputs: {base_output_dir}")
    print(f"    summary: {summary_path}")
    print(f"    manifest: {manifest_path}")
    print(f"    total elapsed: {perf_counter() - overall_started:.2f}s")


def main() -> None:
    overall_started = perf_counter()
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_validated_config_from_args(args, parser)
    cases = _parse_cases(args.case)
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)

    date_start, date_end = resolve_period_dates(cfg, args)
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
            "baseline_feature_profile",
            "diagnostic_label_space",
            "diagnostic_threshold",
            "feature_count",
            "incremental_feature_count",
            "row_count",
            "top_feature",
            "top_rank_ic_mean",
            "top_rank_ic_abs_mean",
            "top_rank_ic_ir",
            "top_monotonicity_mean",
            "top_monthly_directional_hit",
            "incremental_top_feature",
            "incremental_top_rank_ic_mean",
            "incremental_top_rank_ic_abs_mean",
            "incremental_top_rank_ic_ir",
            "incremental_top_monotonicity_mean",
            "incremental_top_monthly_directional_hit",
            "incremental_n_abs_rankic_ge_0p03",
            "incremental_n_abs_rankic_ge_0p05",
            "output_dir",
        ],
    )
    _init_tsv(
        manifest_path,
        [
            "case_name",
            "feature_profile",
            "baseline_feature_profile",
            "diagnostic_label_space",
            "diagnostic_threshold",
            "output_dir",
            "readme_path",
            "summary_csv",
            "incremental_summary_csv",
            "segment_comparison_csv",
            "incremental_segment_comparison_csv",
        ],
    )

    factor_store_dir = get_native_factor_store_dir(cfg)
    factor_store_meta = load_factor_store_metadata(factor_store_dir)
    case_feature_map: dict[str, list[str]] = {}
    case_incremental_feature_map: dict[str, list[str]] = {}
    for case in cases:
        if bool(getattr(args, "all_features", False)):
            feature_names = list(factor_store_meta.get("feature_names", []))
        else:
            feature_names = _resolve_feature_profile_source_columns(factor_store_meta, cfg, case.feature_profile)
        if not feature_names:
            raise ValueError(f"Case '{case.name}' resolved no features")
        baseline_feature_names = (
            _resolve_feature_profile_source_columns(factor_store_meta, cfg, case.baseline_feature_profile)
            if case.baseline_feature_profile
            else []
        )
        incremental_feature_names = _resolve_incremental_feature_names(feature_names, baseline_feature_names)
        case_feature_map[case.name] = feature_names
        case_incremental_feature_map[case.name] = incremental_feature_names

    segments = resolve_segments(cfg, args, main_start=date_start, main_end=date_end)
    _run_rust_batch(
        cfg=cfg,
        args=args,
        cases=cases,
        case_feature_map=case_feature_map,
        case_incremental_feature_map=case_incremental_feature_map,
        factor_store_dir=Path(factor_store_dir),
        label_column=label_column,
        signal_horizon=signal_horizon,
        date_start=date_start,
        date_end=date_end,
        segments=segments,
        base_output_dir=base_output_dir,
        summary_path=summary_path,
        manifest_path=manifest_path,
        overall_started=overall_started,
    )


if __name__ == "__main__":
    main()
