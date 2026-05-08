from __future__ import annotations

from run_lgbm_backtest_artifacts import build_delegated_command


def test_lgbm_artifact_wrapper_delegates_to_rust_batch(monkeypatch) -> None:
    monkeypatch.delenv("AI4STOCK_BACKTEST_BIN", raising=False)

    command = build_delegated_command(
        [
            "--selected-tsv",
            "selected.tsv",
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
        ]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-backtest", "--"]
    assert command[5] == "artifact-batch"
    assert ["--selected-tsv", "selected.tsv"] == command[
        command.index("--selected-tsv") : command.index("--selected-tsv") + 2
    ]
    assert ["--output-root", "artifact_runs"] == command[
        command.index("--output-root") : command.index("--output-root") + 2
    ]
    assert ["--jobs", "8"] == command[command.index("--jobs") : command.index("--jobs") + 2]
    assert command[command.index("--backtest-artifact-level") + 1] == "reports"


def test_lgbm_artifact_wrapper_honors_backtest_binary_override(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_BACKTEST_BIN", "target/release/ai4stock-backtest --flag")

    command = build_delegated_command(["--selected-tsv", "selected.tsv", "--dry-run"])

    assert command[:3] == ["target/release/ai4stock-backtest", "--flag", "artifact-batch"]
    assert "--dry-run" in command
