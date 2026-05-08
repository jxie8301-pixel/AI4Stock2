"""Compatibility wrapper for the Rust rolling pipeline.

This Python entrypoint intentionally no longer owns factor-store loading,
rolling-window construction, training, or backtest evaluation.  It preserves the
legacy CLI shape and delegates the actual work to the standalone Rust binaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Iterable

import yaml

from src.rolling_types import PREDICTION_ARTIFACT_DIRNAME
from src.config_loader import load_runtime_config
from src.runtime_cli import add_common_runtime_args, apply_common_runtime_overrides, load_validated_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI4Stock2 Rust Rolling Pipeline Wrapper")
    add_common_runtime_args(parser, include_model_arg=True)
    parser.add_argument(
        "--retrain-step",
        type=int,
        help="Rolling retrain step in trading days. If omitted, use config value.",
    )
    parser.add_argument("--horizon", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--train-days", type=int, help="Training window length in trading days. If omitted, use config value.")
    parser.add_argument("--valid-days", type=int, help="Validation window length in trading days. If omitted, use config value.")
    parser.add_argument("--test-start", help="Override test start date for Rust training.")
    parser.add_argument("--test-end", help="Override test end date for Rust training.")
    parser.add_argument("--factor-store", help="Override Rust factor-store root for training.")
    parser.add_argument("--output-dir", "--results-dir", dest="output_dir", help="Explicit run output directory.")
    parser.add_argument(
        "--torch-gpu",
        dest="torch_gpu",
        type=int,
        default=0,
        help="Accepted for legacy CLI compatibility; Rust LightGBM ignores it.",
    )
    parser.add_argument("--save-models", action="store_true", help="Save models for each rolling step")
    parser.add_argument("--load-models", action="store_true", help="Load existing models for each rolling step")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Keep/copy final rolling predictions and labels for later backtest reuse.",
    )
    parser.add_argument(
        "--load-predictions-dir",
        help=(
            "Reuse an existing rolling prediction bundle directory and skip training/inference. "
            f"Accepts either a run directory or a direct {PREDICTION_ARTIFACT_DIRNAME}/ path."
        ),
    )
    parser.add_argument(
        "--skip-reference-baselines",
        action="store_true",
        help="Skip factor-baseline reconstruction in Rust backtest.",
    )
    parser.add_argument(
        "--backtest-artifact-level",
        choices=("full", "reports", "metrics"),
        default="full",
        help="Control Rust backtest artifact volume.",
    )
    parser.add_argument("--skip-opportunity-diagnostics", action="store_true", help="Accepted for CLI compatibility.")
    parser.add_argument("--skip-backtest-plots", action="store_true", help="Skip Rust SVG plot artifacts.")
    parser.add_argument("--skip-backtest-trace", action="store_true", help="Accepted for CLI compatibility.")
    parser.add_argument("--baseline-jobs", type=int, help="Parallel baseline workers for Rust backtest.")
    parser.add_argument("--dry-run", action="store_true", help="Print delegated Rust commands without running them.")
    return parser


def run_rolling_pipeline_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    if parser is None:
        parser = build_parser()
    model_name = str(getattr(args, "model", None) or "lgbm")
    if model_name != "lgbm":
        parser.error("run_native_rolling.py is now a Rust wrapper and only supports model=lgbm.")

    output_dir = resolve_output_dir(args)
    if args.load_predictions_dir:
        config_snapshot = write_resolved_config_snapshot(args, parser, output_dir)
        backtest_cmd = build_backtest_command(
            args,
            bundle_dir=Path(args.load_predictions_dir),
            config_path=config_snapshot,
            output_dir=output_dir,
        )
        run_or_print([backtest_cmd], dry_run=bool(args.dry_run))
        return

    train_cmd = build_train_command(args, output_dir=output_dir)
    bundle_dir = output_dir / PREDICTION_ARTIFACT_DIRNAME
    config_snapshot = output_dir / "config_snapshot.yaml"
    backtest_cmd = build_backtest_command(
        args,
        bundle_dir=bundle_dir,
        config_path=config_snapshot,
        output_dir=output_dir,
    )
    run_or_print([train_cmd, backtest_cmd], dry_run=bool(args.dry_run))


def build_train_command(args: argparse.Namespace, *, output_dir: Path) -> list[str]:
    command = rust_binary_command("ai4stock-train", env_var="AI4STOCK_TRAIN_BIN")
    command.extend(["make-bundle-lgbm", "--output-dir", str(output_dir)])
    append_common_training_args(command, args)
    append_option(command, "--factor-store", getattr(args, "factor_store", None))
    append_option(command, "--test-start", getattr(args, "test_start", None))
    append_option(command, "--test-end", getattr(args, "test_end", None))
    append_flag(command, "--config-is-snapshot", bool(getattr(args, "config_is_snapshot", False)))
    append_flag(command, "--save-models", bool(getattr(args, "save_models", False)))
    append_flag(command, "--load-models", bool(getattr(args, "load_models", False)))
    return command


def build_backtest_command(
    args: argparse.Namespace,
    *,
    bundle_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> list[str]:
    command = rust_binary_command("ai4stock-backtest", env_var="AI4STOCK_BACKTEST_BIN")
    command.extend(
        [
            "run-bundle",
            "--bundle",
            str(bundle_dir),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    reduced_artifacts = str(getattr(args, "backtest_artifact_level", "full")) in {"reports", "metrics"}
    append_flag(
        command,
        "--skip-reference-baselines",
        bool(getattr(args, "skip_reference_baselines", False)),
    )
    append_flag(
        command,
        "--skip-backtest-plots",
        bool(getattr(args, "skip_backtest_plots", False)) or reduced_artifacts,
    )
    append_option(command, "--baseline-jobs", getattr(args, "baseline_jobs", None))
    return command


def append_common_training_args(command: list[str], args: argparse.Namespace) -> None:
    append_option(command, "--config", getattr(args, "config", None))
    append_option(command, "--experiment-profile", getattr(args, "experiment_profile", None))
    append_option(command, "--feature-profile", getattr(args, "feature_profile", None) or getattr(args, "profile", None))
    append_option(command, "--model-profile", getattr(args, "model_profile", None))
    append_option(command, "--data-source", getattr(args, "data_source", None))
    append_option(command, "--topk", getattr(args, "topk", None))
    append_option(command, "--n-drop", getattr(args, "n_drop", None))
    append_option(command, "--rebalance-freq", getattr(args, "rebalance_freq", None))
    append_option(command, "--signal-horizon", getattr(args, "signal_horizon", None) or getattr(args, "label_horizon", None))
    append_option(command, "--retrain-step", getattr(args, "retrain_step", None) or getattr(args, "horizon", None))
    append_option(command, "--train-days", getattr(args, "train_days", None))
    append_option(command, "--valid-days", getattr(args, "valid_days", None))
    for override in getattr(args, "set_overrides", None) or []:
        command.extend(["--set", str(override)])


def write_resolved_config_snapshot(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    output_dir: Path,
) -> Path:
    if getattr(args, "config_is_snapshot", False):
        cfg = load_runtime_config(args.config)
        apply_common_runtime_overrides(
            cfg,
            args,
            parser,
            allow_rolling_overrides=True,
        )
    else:
        cfg = load_validated_config_from_args(
            args,
            parser,
            allow_rolling_overrides=True,
            check_paths=False,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_snapshot.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    return path


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    root = Path(getattr(args, "store_dir", None) or "results/experiments")
    model_name = str(getattr(args, "model", None) or "lgbm")
    tag = slugify(getattr(args, "run_tag", None)) or "rust-wrapper"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "native" / "rolling" / model_name / f"{stamp}__{tag}"


def rust_binary_command(binary_name: str, *, env_var: str) -> list[str]:
    env_value = os.environ.get(env_var)
    if env_value:
        return shlex.split(env_value)
    return ["cargo", "run", "--bin", binary_name, "--"]


def run_or_print(commands: Iterable[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        rendered = shlex.join(command)
        if dry_run:
            print(f"[dry-run] {rendered}")
            continue
        print(f"[run] {rendered}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(int(completed.returncode))


def append_option(command: list[str], flag: str, value: object | None) -> None:
    if value is not None and str(value) != "":
        command.extend([flag, str(value)])


def append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def slugify(value: object | None) -> str:
    if value is None:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return slug or "run"


def run_rolling_pipeline(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_rolling_pipeline_from_args(args, parser=parser)


if __name__ == "__main__":
    try:
        run_rolling_pipeline()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
