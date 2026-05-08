from __future__ import annotations

from run_single_factor_diagnostics import build_delegated_command


def test_single_factor_wrapper_delegates_to_rust_profile_subcommand(monkeypatch) -> None:
    monkeypatch.delenv("AI4STOCK_DIAGNOSTICS_BIN", raising=False)

    command = build_delegated_command(
        [
            "--experiment-profile",
            "core",
            "--feature-profile",
            "core_v4_techlite",
            "--dry-run",
        ]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]
    assert command[5] == "single-factor-profile"
    assert "--experiment-profile" in command
    assert "--feature-profile" in command
    assert "--dry-run" in command


def test_single_factor_wrapper_honors_explicit_rust_binary(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_DIAGNOSTICS_BIN", "/tmp/ai4stock-diagnostics --flag")

    command = build_delegated_command(["--dry-run"])

    assert command == ["/tmp/ai4stock-diagnostics", "--flag", "single-factor-profile", "--dry-run"]
