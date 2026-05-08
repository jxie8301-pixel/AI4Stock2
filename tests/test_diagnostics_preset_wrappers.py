from __future__ import annotations

from run_full_space_single_factor_diagnostics import build_delegated_command as build_full_command
from run_quality_event_flow_single_factor_batch import build_delegated_command as build_quality_command


def test_full_space_wrapper_delegates_to_rust_preset(monkeypatch) -> None:
    monkeypatch.delenv("AI4STOCK_DIAGNOSTICS_BIN", raising=False)

    command = build_full_command(
        [
            "--experiment-profile",
            "exp",
            "--base-output-dir",
            "out",
            "--run-tag",
            "tag",
            "--skip-industry-neutral",
            "--dry-run",
        ]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]
    assert command[5] == "full-space-single-factor"
    assert ["--base-output-dir", "out"] == command[
        command.index("--base-output-dir") : command.index("--base-output-dir") + 2
    ]
    assert "--skip-industry-neutral" in command
    assert "--dry-run" in command


def test_quality_event_wrapper_delegates_to_rust_preset(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_DIAGNOSTICS_BIN", "target/release/ai4stock-diagnostics --flag")

    command = build_quality_command(
        [
            "--experiment-profile",
            "exp",
            "--include-benchmark-excess",
            "--set",
            "data.source=tushare",
            "--dry-run",
        ]
    )

    assert command[:3] == ["target/release/ai4stock-diagnostics", "--flag", "quality-event-flow-single-factor"]
    assert ["--set", "data.source=tushare"] == command[command.index("--set") : command.index("--set") + 2]
    assert "--include-benchmark-excess" in command
    assert "--dry-run" in command
