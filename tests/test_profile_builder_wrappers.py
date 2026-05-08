from __future__ import annotations

from run_build_prefiltered_profile import build_delegated_command as build_prefilter_command
from run_build_robust_factor_profile import build_delegated_command as build_robust_command


def test_prefilter_profile_wrapper_delegates_to_rust_runtime(monkeypatch) -> None:
    monkeypatch.delenv("AI4STOCK_DIAGNOSTICS_BIN", raising=False)

    command = build_prefilter_command(
        [
            "--experiment-profile",
            "core",
            "--diagnostics-summary",
            "single_factor_summary.csv",
            "--profile-name",
            "profile_x",
        ]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]
    assert command[5] == "build-prefilter-profile-runtime"
    assert "--diagnostics-summary" in command
    assert "--profile-name" in command


def test_robust_profile_wrapper_delegates_to_rust_runtime(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_DIAGNOSTICS_BIN", "/tmp/ai4stock-diagnostics --flag")

    command = build_robust_command(
        [
            "--raw-summary",
            "raw.csv",
            "--neutral-summary",
            "neutral.csv",
            "--profile-name",
            "profile_x",
        ]
    )

    assert command[:3] == ["/tmp/ai4stock-diagnostics", "--flag", "build-robust-profile-runtime"]
    assert "--raw-summary" in command
    assert "--neutral-summary" in command
