"""Preset batch runner for the first-batch Tushare quality/event/flow factors."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_FEATURE_PROFILE = "core_v4_techlite_tushare_plus_quality_event_flow_v1"
DEFAULT_CASES = (
    ("raw_return", 0.0),
    ("industry_excess", 0.0),
    ("benchmark_excess", 0.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-factor diagnostics for the first-batch Tushare quality/event/flow factors "
            "using preset diagnostic label spaces."
        )
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path.")
    parser.add_argument("--experiment-profile", required=True, help="Named experiment profile.")
    parser.add_argument(
        "--feature-profile",
        default=DEFAULT_FEATURE_PROFILE,
        help=f"Feature profile to diagnose. Default: {DEFAULT_FEATURE_PROFILE}",
    )
    parser.add_argument("--model-profile", help="Optional model profile passthrough.")
    parser.add_argument("--data-source", help="Optional data source override.")
    parser.add_argument(
        "--period",
        choices=["train", "valid", "test", "all"],
        default="train",
        help="Diagnostics period. Default: train. Use all only for research-only split comparison.",
    )
    parser.add_argument("--date-start", help="Optional explicit start date.")
    parser.add_argument("--date-end", help="Optional explicit end date.")
    parser.add_argument("--quantile-bins", type=int, default=5, help="Quantile bins for diagnostics.")
    parser.add_argument("--top-n", type=int, default=50, help="Top-N factors to export.")
    parser.add_argument(
        "--segment-scheme",
        choices=["none", "config_split"],
        default="config_split",
        help="Segmented diagnostics scheme. Default: config_split.",
    )
    parser.add_argument("--segments", help="Optional custom segments passthrough.")
    parser.add_argument("--base-output-dir", help="Optional output root.")
    parser.add_argument("--run-tag", help="Optional run tag passthrough.")
    parser.add_argument(
        "--include-benchmark-excess",
        action="store_true",
        help="Include benchmark-excess diagnostics in addition to raw_return and industry_excess.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="set_overrides",
        help="Generic dotted override in key=value form, forwarded to the batch runner.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    batch_script = Path(__file__).resolve().parent / "run_single_factor_diagnostics_batch.py"
    if not batch_script.exists():
        raise FileNotFoundError(f"Batch diagnostics script not found: {batch_script}")

    selected_cases = list(DEFAULT_CASES[:2])
    if args.include_benchmark_excess:
        selected_cases.append(DEFAULT_CASES[2])

    cmd = [
        sys.executable,
        str(batch_script),
        "--config",
        args.config,
        "--experiment-profile",
        args.experiment_profile,
        "--feature-profile",
        args.feature_profile,
        "--period",
        args.period,
        "--quantile-bins",
        str(max(int(args.quantile_bins), 2)),
        "--top-n",
        str(max(int(args.top_n), 1)),
        "--segment-scheme",
        args.segment_scheme,
    ]
    if args.model_profile:
        cmd.extend(["--model-profile", args.model_profile])
    if args.data_source:
        cmd.extend(["--data-source", args.data_source])
    if args.date_start:
        cmd.extend(["--date-start", args.date_start])
    if args.date_end:
        cmd.extend(["--date-end", args.date_end])
    if args.segments:
        cmd.extend(["--segments", args.segments])
    if args.base_output_dir:
        cmd.extend(["--base-output-dir", args.base_output_dir])
    if args.run_tag:
        cmd.extend(["--run-tag", args.run_tag])
    for raw_override in args.set_overrides or []:
        cmd.extend(["--set", raw_override])

    for label_space, threshold in selected_cases:
        case_name = f"{args.feature_profile}__{label_space}"
        cmd.extend(
            [
                "--case",
                f"name={case_name}",
                f"feature_profile={args.feature_profile}",
                f"diagnostic_label_space={label_space}",
                f"diagnostic_threshold={threshold}",
            ]
        )

    print("[*] Executing diagnostics batch:")
    print("    " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
