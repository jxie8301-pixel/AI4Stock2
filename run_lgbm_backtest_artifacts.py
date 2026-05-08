"""Compatibility wrapper for Rust LGBM backtest artifact rebuilds.

The artifact rebuild scheduler is owned by ``ai4stock-backtest artifact-batch``.
This Python entrypoint is intentionally thin: it preserves the historical
command name while delegating row selection, marker handling, preflight checks,
parallel execution, and summaries to Rust.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild baseline diagnostics, plots, and full backtest artifacts "
            "from selected LGBM matrix rows without retraining. Runtime execution "
            "is delegated to ai4stock-backtest artifact-batch."
        )
    )
    parser.add_argument(
        "--selected-tsv",
        required=True,
        help="selected_lgbm_backtest_matrix_*.tsv produced by batch_backtest_past_lgbm_no_leak.sh.",
    )
    parser.add_argument("--matrix-id", action="append", default=[], help="Matrix id allowlist. Repeat or comma-separate.")
    parser.add_argument("--train-id", action="append", default=[], help="Train id allowlist. Repeat or comma-separate.")
    parser.add_argument("--backtest-id", action="append", default=[], help="Backtest profile id allowlist. Repeat or comma-separate.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of selected rows to run.")
    parser.add_argument("--start-after", default="", help="Skip rows through this matrix_id or backtest_profile_id.")
    parser.add_argument("--output-root", help="Where rebuilt artifact runs are written.")
    parser.add_argument("--log-dir", help="Where per-job logs and the summary TSV/JSON are written.")
    parser.add_argument(
        "--execution-mode",
        choices=("rust",),
        default="rust",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--python-runner",
        default=os.environ.get("PYTHON_RUNNER", "pixi run python"),
        help="Python runner used by the Rust post-bundle entrypoint.",
    )
    parser.add_argument(
        "--rust-runner",
        default=os.environ.get("RUST_BACKTEST_RUNNER", "target/release/ai4stock-backtest"),
        help="Rust artifact-batch entrypoint.",
    )
    parser.add_argument("--repo-root", default=str(Path.cwd()), help="Repository root passed to Rust artifact-batch.")
    parser.add_argument("--model", default="lgbm", help="Model argument passed through to the post-bundle plan.")
    parser.add_argument("--run-tag-prefix", default="artifact-rebuild-lgbm")
    parser.add_argument(
        "--backtest-artifact-level",
        choices=("full", "reports", "metrics"),
        default="full",
        help="Default full rebuilds baselines, diagnostics, plots, reports, and trace.",
    )
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Parallel artifact-batch workers.")
    parser.add_argument("--baseline-jobs", type=int, default=1, help="Per-bundle reference-baseline worker count.")
    parser.add_argument("--save-predictions", action="store_true", help="Copy prediction artifacts into rebuilt run dirs.")
    parser.add_argument("--skip-reference-baselines", action="store_true", help="Omit reference-baseline reconstruction.")
    parser.add_argument("--skip-opportunity-diagnostics", action="store_true", help="Omit buyability diagnostics.")
    parser.add_argument("--skip-backtest-plots", action="store_true", help="Omit cumulative/drawdown/monthly plots.")
    parser.add_argument("--skip-backtest-trace", action="store_true", help="Omit detailed backtest trace artifacts.")
    parser.add_argument("--disable-rust-backtest", action="store_true", help="Force the Python post-bundle backtest path.")
    parser.add_argument("--marker-dir", help="Optional batch marker directory. Existing <matrix_id>.done files are skipped.")
    parser.add_argument("--failed-tsv", help="Optional TSV append target for failed jobs in the batch script format.")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated Rust command without executing it.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed artifact rebuild.")
    return parser


def _append_value(command: list[str], option: str, value: str | int | None) -> None:
    if value is None:
        return
    text = str(value)
    if text:
        command.extend([option, text])


def _append_repeated(command: list[str], option: str, values: list[str]) -> None:
    for value in values:
        _append_value(command, option, value)


def build_artifact_batch_command(args: argparse.Namespace) -> list[str]:
    command = [
        *shlex.split(args.rust_runner),
        "artifact-batch",
        "--selected-tsv",
        str(args.selected_tsv),
        "--python-runner",
        str(args.python_runner),
        "--repo-root",
        str(args.repo_root),
        "--model",
        str(args.model),
        "--run-tag-prefix",
        str(args.run_tag_prefix),
        "--backtest-artifact-level",
        str(args.backtest_artifact_level),
        "--jobs",
        str(max(int(args.jobs), 1)),
        "--baseline-jobs",
        str(max(int(args.baseline_jobs), 1)),
    ]
    _append_repeated(command, "--matrix-id", list(args.matrix_id))
    _append_repeated(command, "--train-id", list(args.train_id))
    _append_repeated(command, "--backtest-id", list(args.backtest_id))
    if int(args.limit) > 0:
        command.extend(["--limit", str(int(args.limit))])
    _append_value(command, "--start-after", args.start_after)
    _append_value(command, "--output-root", args.output_root)
    _append_value(command, "--log-dir", args.log_dir)
    _append_value(command, "--marker-dir", args.marker_dir)
    _append_value(command, "--failed-tsv", args.failed_tsv)
    if args.save_predictions:
        command.append("--save-predictions")
    if args.skip_reference_baselines:
        command.append("--skip-reference-baselines")
    if args.skip_opportunity_diagnostics:
        command.append("--skip-opportunity-diagnostics")
    if args.skip_backtest_plots:
        command.append("--skip-backtest-plots")
    if args.skip_backtest_trace:
        command.append("--skip-backtest-trace")
    if args.disable_rust_backtest:
        command.append("--disable-rust-backtest")
    if args.dry_run:
        command.append("--dry-run")
    if args.fail_fast:
        command.append("--fail-fast")
    return command


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    mpl_config_dir = Path(env.get("MPLCONFIGDIR") or Path.cwd() / ".mpl-cache")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_config_dir)
    return env


def run_jobs(args: argparse.Namespace) -> int:
    command = build_artifact_batch_command(args)
    print("[rust_cmd] " + shlex.join(command))
    if args.dry_run:
        return 0
    completed = subprocess.run(command, env=_runtime_env(), check=False)
    return int(completed.returncode)


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_jobs(args))


if __name__ == "__main__":
    main()
