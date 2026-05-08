from __future__ import annotations

import pytest

from run_lgbm_backtest_artifacts import build_artifact_batch_command, build_parser, run_jobs


def _parse_args(*extra: str):
    return build_parser().parse_args(["--selected-tsv", "selected.tsv", *extra])


def test_build_artifact_batch_command_delegates_to_rust() -> None:
    command = build_artifact_batch_command(
        _parse_args(
            "--rust-runner",
            "target/release/ai4stock-backtest",
            "--python-runner",
            "pixi run python",
            "--repo-root",
            "/repo",
            "--output-root",
            "artifact_runs",
            "--log-dir",
            "artifact_logs",
            "--jobs",
            "8",
            "--baseline-jobs",
            "2",
            "--backtest-artifact-level",
            "reports",
        )
    )

    assert command[:4] == [
        "target/release/ai4stock-backtest",
        "artifact-batch",
        "--selected-tsv",
        "selected.tsv",
    ]
    assert ["--python-runner", "pixi run python"] == command[command.index("--python-runner") : command.index("--python-runner") + 2]
    assert ["--repo-root", "/repo"] == command[command.index("--repo-root") : command.index("--repo-root") + 2]
    assert ["--output-root", "artifact_runs"] == command[command.index("--output-root") : command.index("--output-root") + 2]
    assert ["--log-dir", "artifact_logs"] == command[command.index("--log-dir") : command.index("--log-dir") + 2]
    assert ["--jobs", "8"] == command[command.index("--jobs") : command.index("--jobs") + 2]
    assert ["--baseline-jobs", "2"] == command[command.index("--baseline-jobs") : command.index("--baseline-jobs") + 2]
    assert command[command.index("--backtest-artifact-level") + 1] == "reports"


def test_build_artifact_batch_command_forwards_filters_and_artifact_flags() -> None:
    command = build_artifact_batch_command(
        _parse_args(
            "--matrix-id",
            "m1,m2",
            "--train-id",
            "train_a",
            "--backtest-id",
            "bt_x",
            "--limit",
            "10",
            "--start-after",
            "m0",
            "--marker-dir",
            "markers",
            "--failed-tsv",
            "failed.tsv",
            "--save-predictions",
            "--skip-reference-baselines",
            "--skip-opportunity-diagnostics",
            "--skip-backtest-plots",
            "--skip-backtest-trace",
            "--disable-rust-backtest",
            "--fail-fast",
        )
    )

    assert ["--matrix-id", "m1,m2"] == command[command.index("--matrix-id") : command.index("--matrix-id") + 2]
    assert ["--train-id", "train_a"] == command[command.index("--train-id") : command.index("--train-id") + 2]
    assert ["--backtest-id", "bt_x"] == command[command.index("--backtest-id") : command.index("--backtest-id") + 2]
    assert ["--limit", "10"] == command[command.index("--limit") : command.index("--limit") + 2]
    assert ["--start-after", "m0"] == command[command.index("--start-after") : command.index("--start-after") + 2]
    assert ["--marker-dir", "markers"] == command[command.index("--marker-dir") : command.index("--marker-dir") + 2]
    assert ["--failed-tsv", "failed.tsv"] == command[command.index("--failed-tsv") : command.index("--failed-tsv") + 2]
    assert "--save-predictions" in command
    assert "--skip-reference-baselines" in command
    assert "--skip-opportunity-diagnostics" in command
    assert "--skip-backtest-plots" in command
    assert "--skip-backtest-trace" in command
    assert "--disable-rust-backtest" in command
    assert "--fail-fast" in command


def test_execution_mode_no_longer_accepts_python_fallbacks() -> None:
    with pytest.raises(SystemExit):
        _parse_args("--execution-mode", "direct")
    with pytest.raises(SystemExit):
        _parse_args("--execution-mode", "subprocess")


def test_run_jobs_dry_run_prints_delegated_command(capsys) -> None:
    args = _parse_args("--rust-runner", "target/release/ai4stock-backtest", "--dry-run")

    assert run_jobs(args) == 0

    captured = capsys.readouterr()
    assert "target/release/ai4stock-backtest artifact-batch" in captured.out
    assert "--dry-run" in captured.out
