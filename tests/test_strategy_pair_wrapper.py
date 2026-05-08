from __future__ import annotations

from run_strategy_pair_diagnostics import build_delegated_command


def test_strategy_pair_wrapper_delegates_to_rust() -> None:
    command = build_delegated_command(
        ["--candidate-run", "cand", "--baseline-run", "base", "--candidate-name", "c"]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]
    assert command[5:] == [
        "strategy-pair",
        "--candidate-run",
        "cand",
        "--baseline-run",
        "base",
        "--candidate-name",
        "c",
    ]


def test_strategy_pair_wrapper_honors_explicit_rust_binary(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_DIAGNOSTICS_BIN", "target/release/ai4stock-diagnostics --flag")

    command = build_delegated_command(["--candidate-run", "cand", "--baseline-run", "base"])

    assert command == [
        "target/release/ai4stock-diagnostics",
        "--flag",
        "strategy-pair",
        "--candidate-run",
        "cand",
        "--baseline-run",
        "base",
    ]
