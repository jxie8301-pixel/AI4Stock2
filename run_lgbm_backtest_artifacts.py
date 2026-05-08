"""Rebuild full LGBM backtest artifacts for selected matrix rows.

This is the companion to FAST_BACKTEST=1 batch scans. Use the batch script to
select candidate train/backtest pairs, then run this script against the
generated selected_lgbm_backtest_matrix_*.tsv to rebuild full baselines,
diagnostics, plots, CSV reports, and optional trace artifacts for only the
rows that deserve inspection.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import shlex
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


NONE_MARKER = "__NONE__"


@dataclass(frozen=True)
class ArtifactJob:
    row: dict[str, str]
    argv: list[str]
    command: list[str]
    rust_command: list[str]
    log_path: Path

    @property
    def matrix_id(self) -> str:
        return self.row["matrix_id"]

    @property
    def train_id(self) -> str:
        return self.row.get("train_id", "")

    @property
    def backtest_profile_id(self) -> str:
        return self.row.get("backtest_profile_id", "")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild baseline diagnostics, plots, and full backtest artifacts "
            "from selected LGBM matrix rows without retraining."
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
    parser.add_argument("--log-dir", help="Where per-job logs and the summary TSV are written.")
    parser.add_argument(
        "--execution-mode",
        choices=("direct", "subprocess", "rust"),
        default="rust",
        help=(
            "direct calls run_native_rolling in this Python process; subprocess keeps the old per-job CLI execution; "
            "rust delegates each post-bundle job through ai4stock-backtest bundle."
        ),
    )
    parser.add_argument(
        "--python-runner",
        default=os.environ.get("PYTHON_RUNNER", "pixi run python"),
        help="Python runner used by subprocess mode and by the Rust post-bundle entrypoint.",
    )
    parser.add_argument(
        "--rust-runner",
        default=os.environ.get("RUST_BACKTEST_RUNNER", "target/release/ai4stock-backtest"),
        help="Rust post-bundle entrypoint used only with --execution-mode rust.",
    )
    parser.add_argument("--model", default="lgbm", help="Model argument passed through to run_native_rolling.py.")
    parser.add_argument("--run-tag-prefix", default="artifact-rebuild-lgbm")
    parser.add_argument(
        "--backtest-artifact-level",
        choices=("full", "reports", "metrics"),
        default="full",
        help="Default full rebuilds baselines, diagnostics, plots, reports, and trace.",
    )
    parser.add_argument("--save-predictions", action="store_true", help="Copy prediction artifacts into rebuilt run dirs.")
    parser.add_argument("--skip-reference-baselines", action="store_true", help="Omit reference-baseline reconstruction.")
    parser.add_argument("--skip-opportunity-diagnostics", action="store_true", help="Omit buyability diagnostics.")
    parser.add_argument("--skip-backtest-plots", action="store_true", help="Omit cumulative/drawdown/monthly plots.")
    parser.add_argument("--skip-backtest-trace", action="store_true", help="Omit detailed backtest trace artifacts.")
    parser.add_argument("--disable-rust-backtest", action="store_true", help="Force the Python post-bundle backtest path.")
    parser.add_argument("--marker-dir", help="Optional batch marker directory. Existing <matrix_id>.done files are skipped.")
    parser.add_argument("--failed-tsv", help="Optional TSV append target for failed jobs in the batch script format.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed artifact rebuild.")
    return parser


def split_allowlist(raw_values: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for raw in raw_values:
        values.update(part.strip() for part in str(raw).split(",") if part.strip())
    return values


def read_selected_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _has_value(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    return bool(value and value != NONE_MARKER)


def _append_optional(command: list[str], option: str, row: dict[str, str], key: str) -> None:
    if _has_value(row, key):
        command.extend([option, row[key]])


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in text)


def row_matches(
    row: dict[str, str],
    *,
    matrix_ids: set[str],
    train_ids: set[str],
    backtest_ids: set[str],
) -> bool:
    return (
        (not matrix_ids or row.get("matrix_id") in matrix_ids)
        and (not train_ids or row.get("train_id") in train_ids)
        and (not backtest_ids or row.get("backtest_profile_id") in backtest_ids)
    )


def select_rows(
    rows: list[dict[str, str]],
    *,
    matrix_ids: set[str],
    train_ids: set[str],
    backtest_ids: set[str],
    start_after: str,
    limit: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_start = not start_after
    for row in rows:
        if not seen_start:
            if row.get("matrix_id") == start_after or row.get("backtest_profile_id") == start_after:
                seen_start = True
            continue
        if not row_matches(row, matrix_ids=matrix_ids, train_ids=train_ids, backtest_ids=backtest_ids):
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def build_rolling_argv(
    row: dict[str, str],
    *,
    output_root: Path,
    model: str,
    run_tag_prefix: str,
    artifact_level: str,
    save_predictions: bool,
    skip_reference_baselines: bool,
    skip_opportunity_diagnostics: bool,
    skip_backtest_plots: bool,
    skip_backtest_trace: bool,
) -> list[str]:
    argv = [
        "--config",
        row["config_snapshot"],
        "--config-is-snapshot",
        "--run-tag",
        f"{run_tag_prefix}-{_safe_tag(row['matrix_id'])}",
        "--store-dir",
        str(output_root),
        "--load-predictions-dir",
        row["primary_predictions_dir"],
        "--model",
        model,
        "--signal-horizon",
        row.get("train_signal_horizon") or "20",
        "--backtest-artifact-level",
        artifact_level,
    ]
    _append_optional(argv, "--retrain-step", row, "train_retrain_step")
    _append_optional(argv, "--train-days", row, "train_train_days")
    _append_optional(argv, "--valid-days", row, "train_valid_days")
    if _has_value(row, "train_label_embargo_days"):
        argv.extend(["--set", f"rolling.label_embargo_days={row['train_label_embargo_days']}"])
    if _has_value(row, "train_model_profile"):
        argv.extend(["--set", f"model.profile={row['train_model_profile']}"])
    if _has_value(row, "train_feature_profile"):
        argv.extend(["--set", f"features.profile={row['train_feature_profile']}"])
    if _truthy(row.get("score_fusion_enabled", "")):
        argv.extend(["--set", f"strategy.score_fusion.secondary_predictions_dir={row['secondary_predictions_dir']}"])
    if save_predictions:
        argv.append("--save-predictions")
    if skip_reference_baselines:
        argv.append("--skip-reference-baselines")
    if skip_opportunity_diagnostics:
        argv.append("--skip-opportunity-diagnostics")
    if skip_backtest_plots:
        argv.append("--skip-backtest-plots")
    if skip_backtest_trace:
        argv.append("--skip-backtest-trace")
    return argv


def build_command(
    row: dict[str, str],
    *,
    python_runner: list[str],
    output_root: Path,
    model: str,
    run_tag_prefix: str,
    artifact_level: str,
    save_predictions: bool,
    skip_reference_baselines: bool,
    skip_opportunity_diagnostics: bool,
    skip_backtest_plots: bool,
    skip_backtest_trace: bool,
) -> list[str]:
    return [
        *python_runner,
        "run_native_rolling.py",
        *build_rolling_argv(
            row,
            output_root=output_root,
            model=model,
            run_tag_prefix=run_tag_prefix,
            artifact_level=artifact_level,
            save_predictions=save_predictions,
            skip_reference_baselines=skip_reference_baselines,
            skip_opportunity_diagnostics=skip_opportunity_diagnostics,
            skip_backtest_plots=skip_backtest_plots,
            skip_backtest_trace=skip_backtest_trace,
        ),
    ]


def build_rust_bundle_command(
    row: dict[str, str],
    *,
    rust_runner: list[str],
    python_runner: str,
    repo_root: Path,
    output_root: Path,
    model: str,
    run_tag_prefix: str,
    artifact_level: str,
    save_predictions: bool,
    skip_reference_baselines: bool,
    skip_opportunity_diagnostics: bool,
    skip_backtest_plots: bool,
    skip_backtest_trace: bool,
    disable_rust_backtest: bool,
) -> list[str]:
    command = [
        *rust_runner,
        "bundle",
        "--python-runner",
        python_runner,
        "--repo-root",
        str(repo_root),
    ]
    if disable_rust_backtest:
        command.append("--disable-rust-backtest")
    command.append("--")
    command.extend(
        build_rolling_argv(
            row,
            output_root=output_root,
            model=model,
            run_tag_prefix=run_tag_prefix,
            artifact_level=artifact_level,
            save_predictions=save_predictions,
            skip_reference_baselines=skip_reference_baselines,
            skip_opportunity_diagnostics=skip_opportunity_diagnostics,
            skip_backtest_plots=skip_backtest_plots,
            skip_backtest_trace=skip_backtest_trace,
        )
    )
    return command


def build_jobs(args: argparse.Namespace) -> tuple[list[ArtifactJob], Path, Path]:
    selected_tsv = Path(args.selected_tsv)
    output_root = Path(args.output_root) if args.output_root else selected_tsv.parent / "lgbm_backtest_artifact_runs"
    log_dir = Path(args.log_dir) if args.log_dir else selected_tsv.parent / "lgbm_backtest_artifact_logs"
    python_runner = shlex.split(args.python_runner)
    rust_runner = shlex.split(args.rust_runner)
    repo_root = Path.cwd()

    rows = select_rows(
        read_selected_rows(selected_tsv),
        matrix_ids=split_allowlist(args.matrix_id),
        train_ids=split_allowlist(args.train_id),
        backtest_ids=split_allowlist(args.backtest_id),
        start_after=args.start_after,
        limit=max(int(args.limit), 0),
    )
    jobs: list[ArtifactJob] = []
    for row in rows:
        argv = build_rolling_argv(
            row,
            output_root=output_root,
            model=args.model,
            run_tag_prefix=args.run_tag_prefix,
            artifact_level=args.backtest_artifact_level,
            save_predictions=bool(args.save_predictions),
            skip_reference_baselines=bool(args.skip_reference_baselines),
            skip_opportunity_diagnostics=bool(args.skip_opportunity_diagnostics),
            skip_backtest_plots=bool(args.skip_backtest_plots),
            skip_backtest_trace=bool(args.skip_backtest_trace),
        )
        jobs.append(
            ArtifactJob(
                row=row,
                argv=argv,
                command=[
                    *python_runner,
                    "run_native_rolling.py",
                    *argv,
                ],
                rust_command=build_rust_bundle_command(
                    row,
                    rust_runner=rust_runner,
                    python_runner=args.python_runner,
                    repo_root=repo_root,
                    output_root=output_root,
                    model=args.model,
                    run_tag_prefix=args.run_tag_prefix,
                    artifact_level=args.backtest_artifact_level,
                    save_predictions=bool(args.save_predictions),
                    skip_reference_baselines=bool(args.skip_reference_baselines),
                    skip_opportunity_diagnostics=bool(args.skip_opportunity_diagnostics),
                    skip_backtest_plots=bool(args.skip_backtest_plots),
                    skip_backtest_trace=bool(args.skip_backtest_trace),
                    disable_rust_backtest=bool(args.disable_rust_backtest),
                ),
                log_path=log_dir / f"{row['matrix_id']}.log",
            )
        )
    return jobs, output_root, log_dir


def _runtime_env_updates(args: argparse.Namespace) -> dict[str, str | None]:
    mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR") or Path.cwd() / ".mpl-cache")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    return {"MPLCONFIGDIR": str(mpl_config_dir)}


def _job_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in _runtime_env_updates(args).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


@contextlib.contextmanager
def _patched_environ(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_job_header(log_file, job: ArtifactJob, *, execution_mode: str) -> None:
    log_file.write(f"[matrix_id] {job.matrix_id}\n")
    log_file.write(f"[train_id] {job.train_id}\n")
    log_file.write(f"[backtest_profile_id] {job.backtest_profile_id}\n")
    log_file.write(f"[execution_mode] {execution_mode}\n")
    log_file.write(f"[start] {datetime.now().isoformat(timespec='seconds')}\n")
    log_file.write("[argv] run_native_rolling.py " + shlex.join(job.argv) + "\n")
    if execution_mode == "subprocess":
        log_file.write("[cmd] " + shlex.join(job.command) + "\n")
    if execution_mode == "rust":
        log_file.write("[rust_cmd] " + shlex.join(job.rust_command) + "\n")
    log_file.flush()


def _run_job_direct(job: ArtifactJob, *, env_updates: dict[str, str | None], log_file) -> int:
    with _patched_environ(env_updates), contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        try:
            from run_native_rolling import build_parser as build_rolling_parser
            from run_native_rolling import run_rolling_pipeline_from_args

            parser = build_rolling_parser()
            rolling_args = parser.parse_args(job.argv)
            run_rolling_pipeline_from_args(rolling_args, parser=parser)
            return 0
        except SystemExit as exc:
            if exc.code is None:
                return 0
            if isinstance(exc.code, int):
                return exc.code
            print(exc.code)
            return 1
        except Exception:
            traceback.print_exc()
            return 1


def _run_job_subprocess(job: ArtifactJob, *, env: dict[str, str], log_file) -> int:
    completed = subprocess.run(job.command, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
    return int(completed.returncode)


def _run_job_rust(job: ArtifactJob, *, env: dict[str, str], log_file) -> int:
    completed = subprocess.run(job.rust_command, stdout=log_file, stderr=subprocess.STDOUT, env=env, check=False)
    return int(completed.returncode)


def _job_markers(marker_dir: Path | None, job: ArtifactJob) -> tuple[Path | None, Path | None]:
    if marker_dir is None:
        return None, None
    return marker_dir / f"{job.matrix_id}.done", marker_dir / f"{job.matrix_id}.failed"


def _append_failed_record(path: Path | None, job: ArtifactJob, reason: str, detail: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{job.matrix_id}\t{job.train_id}\t{reason}\t{detail}\n")


def _preflight_job(job: ArtifactJob) -> tuple[bool, str, str]:
    primary_dir = job.row.get("primary_predictions_dir", "")
    if not primary_dir or not Path(primary_dir).is_dir():
        return False, "missing_primary_predictions", primary_dir
    if _truthy(job.row.get("score_fusion_enabled", "")):
        secondary_dir = job.row.get("secondary_predictions_dir", "")
        if not secondary_dir or secondary_dir == NONE_MARKER or not Path(secondary_dir).is_dir():
            return False, "missing_secondary_predictions", secondary_dir
    return True, "", ""


def run_jobs(args: argparse.Namespace) -> int:
    jobs, output_root, log_dir = build_jobs(args)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    marker_dir = Path(args.marker_dir) if args.marker_dir else None
    if marker_dir is not None:
        marker_dir.mkdir(parents=True, exist_ok=True)
    failed_tsv = Path(args.failed_tsv) if args.failed_tsv else None
    if failed_tsv is not None:
        failed_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / f"artifact_rebuild_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
    env = _job_env(args)
    env_updates = _runtime_env_updates(args)
    execution_mode = str(args.execution_mode)

    print(f"[info] selected_jobs={len(jobs)}")
    print(f"[info] output_root={output_root}")
    print(f"[info] log_dir={log_dir}")
    print(f"[info] artifact_level={args.backtest_artifact_level}")
    print(f"[info] execution_mode={execution_mode}")
    print(f"[info] rust_backtest={'disabled' if args.disable_rust_backtest else 'enabled'}")
    if marker_dir is not None:
        print(f"[info] marker_dir={marker_dir}")
    if failed_tsv is not None:
        print(f"[info] failed_tsv={failed_tsv}")
    if args.dry_run:
        for job in jobs:
            done_marker, _ = _job_markers(marker_dir, job)
            if done_marker is not None and done_marker.is_file():
                print(f"[dry-run skip] {job.matrix_id} already marked done")
                continue
            print(f"[dry-run] {job.matrix_id}")
            print("[argv] run_native_rolling.py " + shlex.join(job.argv))
            if execution_mode == "subprocess":
                print("[cmd] " + shlex.join(job.command))
            if execution_mode == "rust":
                print("[rust_cmd] " + shlex.join(job.rust_command))
        return 0

    failures = 0
    skipped = 0
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "matrix_id",
                "train_id",
                "backtest_profile_id",
                "status",
                "exit_code",
                "elapsed_seconds",
                "log_path",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for idx, job in enumerate(jobs, start=1):
            done_marker, failed_marker = _job_markers(marker_dir, job)
            if done_marker is not None and done_marker.is_file():
                skipped += 1
                writer.writerow(
                    {
                        "matrix_id": job.matrix_id,
                        "train_id": job.train_id,
                        "backtest_profile_id": job.backtest_profile_id,
                        "status": "skipped_done",
                        "exit_code": 0,
                        "elapsed_seconds": "0.000000",
                        "log_path": str(job.log_path),
                    }
                )
                summary_file.flush()
                print(f"[skip] {job.matrix_id} already marked done")
                continue
            print(f"[run] {idx}/{len(jobs)} {job.matrix_id}")
            start = time.perf_counter()
            with job.log_path.open("w", encoding="utf-8") as log_file:
                _write_job_header(log_file, job, execution_mode=execution_mode)
                ok, preflight_reason, preflight_detail = _preflight_job(job)
                if not ok:
                    print(f"[preflight] {preflight_reason}: {preflight_detail}")
                    exit_code = 1
                elif execution_mode == "direct":
                    if failed_marker is not None:
                        failed_marker.unlink(missing_ok=True)
                    exit_code = _run_job_direct(job, env_updates=env_updates, log_file=log_file)
                elif execution_mode == "rust":
                    if failed_marker is not None:
                        failed_marker.unlink(missing_ok=True)
                    exit_code = _run_job_rust(job, env=env, log_file=log_file)
                else:
                    if failed_marker is not None:
                        failed_marker.unlink(missing_ok=True)
                    exit_code = _run_job_subprocess(job, env=env, log_file=log_file)
                elapsed_seconds = time.perf_counter() - start
                log_file.write(f"[end] {datetime.now().isoformat(timespec='seconds')}\n")
                log_file.write(f"[elapsed_seconds] {elapsed_seconds:.6f}\n")
            status = "success" if exit_code == 0 else "failed"
            if status == "success" and done_marker is not None:
                done_marker.touch()
            if exit_code != 0:
                failures += 1
                if failed_marker is not None:
                    failed_marker.write_text(f"{exit_code}\n", encoding="utf-8")
                if preflight_reason:
                    _append_failed_record(failed_tsv, job, preflight_reason, preflight_detail)
                else:
                    _append_failed_record(failed_tsv, job, f"exit_{exit_code}", str(job.log_path))
            writer.writerow(
                {
                    "matrix_id": job.matrix_id,
                    "train_id": job.train_id,
                    "backtest_profile_id": job.backtest_profile_id,
                    "status": status,
                    "exit_code": exit_code,
                    "elapsed_seconds": f"{elapsed_seconds:.6f}",
                    "log_path": str(job.log_path),
                }
            )
            summary_file.flush()
            print(f"[{status}] {job.matrix_id} exit={exit_code} elapsed={elapsed_seconds:.2f}s log={job.log_path}")
            if exit_code != 0 and args.fail_fast:
                break

    print(f"[summary] skipped={skipped} failures={failures} summary={summary_path}")
    return 1 if failures else 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_jobs(args))


if __name__ == "__main__":
    main()
