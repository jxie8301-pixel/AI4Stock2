"""Compatibility wrapper for Rust experiment sweep batches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native experiments in batch from sweep definitions.")
    parser.add_argument("--config", default="configs/config.yaml", help="Runtime config path")
    parser.add_argument("--pipeline", choices=["rolling"], default="rolling", help="Target pipeline")
    parser.add_argument("--experiment-profile", required=True, help="Base experiment profile")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--feature-profile", help="Override feature profile")
    parser.add_argument("--data-source", choices=["akshare", "tushare"], help="Override data source")
    parser.add_argument(
        "--set",
        action="append",
        dest="set_overrides",
        help="Fixed dotted override in key=value form, applied to every run",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        nargs="+",
        dest="sweep_overrides",
        help="Sweep override in key=[a,b,c] form. Quote it in zsh to avoid shell glob expansion.",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs="+",
        dest="case_overrides",
        help=(
            "Explicit grouped overrides for one run. "
            "Example: --case strategy.topk=5 strategy.n_drop=1 "
            "--case strategy.topk=10 strategy.n_drop=2"
        ),
    )
    parser.add_argument("--run-tag-prefix", help="Optional run-tag prefix added before per-run sweep suffix")
    parser.add_argument("--store-dir", help="Override local experiment store root")
    parser.add_argument(
        "--dedupe-predictions",
        action="store_true",
        help=(
            "Train once for runs with identical prediction-producing config, "
            "then replay compatible later runs from the saved prediction bundle."
        ),
    )
    parser.add_argument(
        "--skip-reference-baselines",
        action="store_true",
        help="Forward --skip-reference-baselines to child rolling runs for training-speed batches.",
    )
    parser.add_argument(
        "--python-runner",
        default=sys.executable,
        help="Python runner used by the Rust batch runner for child rolling commands.",
    )
    parser.add_argument(
        "--rust-runner",
        default=os.environ.get("AI4STOCK_EXPERIMENT_BIN", "cargo run --bin ai4stock-experiment --"),
        help="Rust experiment batch entrypoint.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expanded commands without executing them")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed child run")
    return parser


def _build_rust_batch_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    command = [
        *shlex.split(args.rust_runner),
        "batch",
        "--config",
        args.config,
        "--pipeline",
        args.pipeline,
        "--experiment-profile",
        args.experiment_profile,
        "--python-runner",
        args.python_runner,
        "--repo-root",
        str(repo_root),
    ]
    if args.model_profile:
        command += ["--model-profile", args.model_profile]
    if args.feature_profile:
        command += ["--feature-profile", args.feature_profile]
    if args.data_source:
        command += ["--data-source", args.data_source]
    if args.store_dir:
        command += ["--store-dir", args.store_dir]
    if args.dedupe_predictions:
        command.append("--dedupe-predictions")
    if args.skip_reference_baselines:
        command.append("--skip-reference-baselines")
    for item in args.set_overrides or []:
        command += ["--set", item]
    for group in args.sweep_overrides or []:
        for item in group:
            command += ["--sweep", item]
    for group in args.case_overrides or []:
        command.append("--case")
        command.extend(group)
    if args.run_tag_prefix:
        command += ["--run-tag-prefix", args.run_tag_prefix]
    if args.dry_run:
        command.append("--dry-run")
    if args.fail_fast:
        command.append("--fail-fast")
    return command


def _run_rust_batch(args: argparse.Namespace) -> int:
    command = _build_rust_batch_command(args)
    print("[rust_cmd] " + shlex.join(command), flush=True)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main() -> None:
    args = _build_parser().parse_args()
    raise SystemExit(_run_rust_batch(args))


if __name__ == "__main__":
    main()
