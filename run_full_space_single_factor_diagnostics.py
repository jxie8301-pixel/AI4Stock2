"""Preset runner for full-space single-factor diagnostics.

Runs two passes by default:
1) raw full-space diagnostics
2) industry-neutral full-space diagnostics
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CASE_FEATURE_PROFILE = "all_features"
DEFAULT_LABEL_SPACE = "raw_return"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run raw and industry-neutral single-factor diagnostics over the full factor space."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument("--experiment-profile", required=True, help="Named experiment profile.")
    parser.add_argument("--model-profile", help="Optional model profile passthrough.")
    parser.add_argument("--data-source", help="Optional data source override.")
    parser.add_argument(
        "--period",
        choices=["train", "valid", "test", "all"],
        default="test",
        help="Diagnostics period. Default: test.",
    )
    parser.add_argument("--date-start", help="Optional explicit start date.")
    parser.add_argument("--date-end", help="Optional explicit end date.")
    parser.add_argument("--quantile-bins", type=int, default=5, help="Quantile bins for diagnostics.")
    parser.add_argument("--top-n", type=int, default=100, help="Top-N factors to export.")
    parser.add_argument(
        "--segment-scheme",
        choices=["none", "config_split", "yearly"],
        default="yearly",
        help="Segmented diagnostics scheme. Default: yearly.",
    )
    parser.add_argument("--segments", help="Optional custom segments passthrough.")
    parser.add_argument("--base-output-dir", help="Optional output root. Per-pass subdirectories will be created.")
    parser.add_argument("--run-tag", help="Optional run tag prefix.")
    parser.add_argument(
        "--diagnostic-label-space",
        choices=["raw_return", "industry_excess", "benchmark_excess"],
        default=DEFAULT_LABEL_SPACE,
        help="Diagnostic label space. Default: raw_return.",
    )
    parser.add_argument(
        "--diagnostic-threshold",
        type=float,
        default=0.0,
        help="Diagnostic threshold passed through to batch runner. Default: 0.0.",
    )
    parser.add_argument(
        "--skip-industry-neutral",
        action="store_true",
        help="Run only the raw full-space pass.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="set_overrides",
        help="Generic dotted override in key=value form, forwarded to the batch runner.",
    )
    return parser


def _append_optional(cmd: list[str], flag: str, value: str | None) -> None:
    if value:
        cmd.extend([flag, value])


def _build_batch_base_cmd(args: argparse.Namespace) -> list[str]:
    batch_script = Path(__file__).resolve().parent / "run_single_factor_diagnostics_batch.py"
    if not batch_script.exists():
        raise FileNotFoundError(f"Batch diagnostics script not found: {batch_script}")

    cmd = [
        sys.executable,
        str(batch_script),
        "--config",
        args.config,
        "--experiment-profile",
        args.experiment_profile,
        "--period",
        args.period,
        "--quantile-bins",
        str(max(int(args.quantile_bins), 2)),
        "--top-n",
        str(max(int(args.top_n), 1)),
        "--segment-scheme",
        args.segment_scheme,
        "--all-features",
    ]
    _append_optional(cmd, "--model-profile", args.model_profile)
    _append_optional(cmd, "--data-source", args.data_source)
    _append_optional(cmd, "--date-start", args.date_start)
    _append_optional(cmd, "--date-end", args.date_end)
    _append_optional(cmd, "--segments", args.segments)
    for raw_override in args.set_overrides or []:
        cmd.extend(["--set", raw_override])
    return cmd


def _build_case_args(
    *,
    name: str,
    diagnostic_label_space: str,
    diagnostic_threshold: float,
) -> list[str]:
    return [
        "--case",
        f"name={name}",
        f"feature_profile={DEFAULT_CASE_FEATURE_PROFILE}",
        f"diagnostic_label_space={diagnostic_label_space}",
        f"diagnostic_threshold={diagnostic_threshold}",
    ]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_tag_prefix = str(args.run_tag or "").strip()
    output_root = Path(args.base_output_dir) if args.base_output_dir else None
    passes: list[tuple[str, bool]] = [("raw", False)]
    if not args.skip_industry_neutral:
        passes.append(("industry_neutral", True))

    for pass_name, use_industry_neutral in passes:
        cmd = _build_batch_base_cmd(args)
        if use_industry_neutral:
            cmd.append("--industry-neutral")
        if output_root is not None:
            cmd.extend(["--base-output-dir", str(output_root / pass_name)])
        if run_tag_prefix:
            cmd.extend(["--run-tag", f"{run_tag_prefix}-{pass_name}"])
        case_name = f"full-space-{args.diagnostic_label_space}-{pass_name}"
        cmd.extend(
            _build_case_args(
                name=case_name,
                diagnostic_label_space=args.diagnostic_label_space,
                diagnostic_threshold=float(args.diagnostic_threshold),
            )
        )
        print("[*] Executing diagnostics batch:")
        print("    " + " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
