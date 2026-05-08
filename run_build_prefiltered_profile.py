"""Compatibility wrapper for Rust diagnostics-driven feature profile building."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess

from src.feature_profiles import resolve_feature_profile
from src.research_safety import check_config_profile_write_safety
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args

DEFAULT_MAX_ABS_CORR = 0.97
DEFAULT_MIN_ABS_RANK_IC = 0.02
DEFAULT_MIN_ABS_RANK_IC_IR = 0.10
DEFAULT_MIN_COVERAGE_PCT = 0.95
DEFAULT_MIN_MONTHLY_POSITIVE_RATE = 0.45
DEFAULT_MIN_SEGMENT_DIRECTIONAL_HIT_MEAN = 0.55
DEFAULT_MAX_SEGMENT_RANK_IC_MEAN_RANGE = 0.14


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
    parser.add_argument(
        "--summary-engine",
        choices=["auto", "rust"],
        default="auto",
        help="Compatibility option. The active runtime is Rust.",
    )
    parser.add_argument(
        "--correlation-engine",
        choices=["auto", "rust"],
        default="auto",
        help="Compatibility option. The active runtime is Rust.",
    )
    parser.add_argument("--profile-name", required=True, help="Output profile name, for example core_v5_diag_prefilter_v1")
    parser.add_argument(
        "--write-config-profile",
        action="store_true",
        help="Write the generated profile into configs/features/<profile-name>.yaml",
    )
    parser.add_argument(
        "--allow-unsafe-profile-write",
        action="store_true",
        help=(
            "Allow --write-config-profile outside the training range. "
            "This records a research-selection-leakage warning in the output README."
        ),
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


def _rust_diagnostics_command() -> list[str]:
    env_value = os.environ.get("AI4STOCK_DIAGNOSTICS_BIN")
    if env_value:
        return shlex.split(env_value)
    return ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]


def _rust_env() -> dict[str, str]:
    env = os.environ.copy()
    pixi_lib = Path(".pixi/envs/default/lib")
    if pixi_lib.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([str(pixi_lib), *([current] if current else [])])
    return env


def _append_optional_float(command: list[str], flag: str, value: float | None) -> None:
    if value is not None:
        command.extend([flag, str(float(value))])


def _factor_store_name(feature_resolution: dict) -> str:
    raw = feature_resolution.get("raw")
    if isinstance(raw, dict):
        return str(raw.get("factor_store_name") or "full_factor_space")
    return "full_factor_space"


def _run_rust_builder(
    args: argparse.Namespace,
    *,
    cfg: dict,
    feature_resolution: dict,
    output_dir: Path,
    date_start: str,
    date_end: str,
    profile_write_safe: bool,
    profile_write_warning: str | None,
) -> dict:
    command = _rust_diagnostics_command()
    command.extend(
        [
            "build-prefilter-profile",
            "--diagnostics-summary",
            str(args.diagnostics_summary),
            "--factor-store",
            str(feature_resolution["factor_store_dir"]),
            "--output-dir",
            str(output_dir),
            "--profile-name",
            str(args.profile_name),
            "--date-start",
            date_start,
            "--date-end",
            date_end,
            "--universe-name",
            str(cfg.get("universe", "all")),
            "--universe-dir",
            str(cfg.get("native", {}).get("universe_dir", "data/universes")),
            "--corr-threshold",
            str(float(args.max_abs_corr)),
            "--factor-store-name",
            _factor_store_name(feature_resolution),
            "--min-coverage-pct",
            str(float(args.min_coverage_pct)),
            "--min-abs-rank-ic",
            str(float(args.min_abs_rank_ic)),
            "--min-abs-rank-ic-ir",
            str(float(args.min_abs_rank_ic_ir)),
            "--min-monthly-positive-rate",
            str(float(args.min_monthly_positive_rate)),
            "--setting",
            f"diagnostics_summary={args.diagnostics_summary}",
            "--setting",
            f"segment_comparison={args.segment_comparison}",
            "--setting",
            f"data_source={cfg.get('data', {}).get('source', '')}",
            "--setting",
            f"universe={cfg.get('universe', '')}",
            "--setting",
            f"period={args.period}",
            "--setting",
            f"date_start={date_start}",
            "--setting",
            f"date_end={date_end}",
            "--setting",
            f"write_config_profile={args.write_config_profile}",
            "--setting",
            f"profile_write_safety={'training_only' if profile_write_safe else 'unsafe_override'}",
            "--setting",
            f"min_coverage_pct={args.min_coverage_pct}",
            "--setting",
            f"min_abs_rank_ic={args.min_abs_rank_ic}",
            "--setting",
            f"min_abs_rank_ic_ir={args.min_abs_rank_ic_ir}",
            "--setting",
            f"min_monthly_positive_rate={args.min_monthly_positive_rate}",
            "--setting",
            f"min_segment_directional_hit_mean={args.min_segment_directional_hit_mean}",
            "--setting",
            f"max_segment_rank_ic_mean_range={args.max_segment_rank_ic_mean_range}",
            "--setting",
            f"exclude_direction_flip={args.exclude_direction_flip}",
            "--setting",
            f"max_abs_corr={args.max_abs_corr}",
            "--setting",
            f"max_features={args.max_features}",
            "--setting",
            f"correlation_space={'cross_sectional_rank' if not args.no_cross_sectional_rank_corr else 'raw'}",
            "--json",
        ]
    )
    if args.segment_comparison:
        command.extend(["--segment-comparison", str(args.segment_comparison)])
    _append_optional_float(command, "--min-segment-directional-hit-mean", args.min_segment_directional_hit_mean)
    _append_optional_float(command, "--max-segment-rank-ic-mean-range", args.max_segment_rank_ic_mean_range)
    if args.exclude_direction_flip:
        command.append("--exclude-direction-flip")
    if args.no_cross_sectional_rank_corr:
        command.append("--raw-values")
    else:
        command.append("--cross-sectional-rank")
    if args.max_features is not None:
        command.extend(["--max-features", str(int(args.max_features))])
    if args.write_config_profile:
        command.append("--write-config-profile")
        command.extend(["--config-profile-path", str(Path("configs") / "features" / f"{args.profile_name}.yaml")])
    if profile_write_warning:
        command.extend(["--safety-warning", profile_write_warning])
    print(f"[*] Rust prefiltered profile build: {shlex.join(command)}", flush=True)
    completed = subprocess.run(command, check=False, env=_rust_env(), text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
        raise SystemExit(int(completed.returncode))
    if completed.stderr:
        print(completed.stderr, end="")
    return json.loads(completed.stdout)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_validated_config_from_args(args, parser)
    feature_resolution = resolve_feature_profile(cfg)
    date_start, date_end = _resolve_period_dates(cfg, args)
    profile_write_safe = True
    profile_write_warning = None
    if args.write_config_profile:
        profile_write_safe, profile_write_warning = check_config_profile_write_safety(
            cfg,
            date_start=date_start,
            date_end=date_end,
            allow_unsafe=bool(args.allow_unsafe_profile_write),
            tool_name="run_build_prefiltered_profile.py",
            diagnostics_paths=[args.diagnostics_summary, args.segment_comparison],
        )
    summary = _run_rust_builder(
        args,
        cfg=cfg,
        feature_resolution=feature_resolution,
        output_dir=_resolve_output_dir(args),
        date_start=date_start,
        date_end=date_end,
        profile_write_safe=profile_write_safe,
        profile_write_warning=profile_write_warning,
    )
    profile_artifacts = summary["profile_artifacts"]
    prefilter_summary = summary["prefilter_summary"]
    corr_summary = summary["corr_prune_summary"]
    if args.write_config_profile and profile_artifacts.get("config_profile_path"):
        print(f"[*] Config profile written: {profile_artifacts['config_profile_path']}")
    print(f"[+] Prefilter artifacts saved to: {summary['output_dir']}")
    print(f"    kept summary: {prefilter_summary['kept_path']}")
    print(f"    exact-duplicate kept: {prefilter_summary['exact_kept_path']}")
    print(f"    exact-duplicate drops: {prefilter_summary['exact_dropped_path']}")
    print(f"    corr-pruned drops: {corr_summary['dropped_path']}")
    print(f"    profile yaml: {profile_artifacts['profile_path']}")
    print(f"[*] Selected features ({profile_artifacts['selected_feature_count']}):")
    for name in profile_artifacts["selected_features"]:
        print(f"    - {name}")


if __name__ == "__main__":
    main()
