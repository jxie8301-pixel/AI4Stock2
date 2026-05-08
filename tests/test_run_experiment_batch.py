from __future__ import annotations

from run_experiment_batch import build_delegated_command


def test_experiment_batch_wrapper_delegates_to_rust(monkeypatch) -> None:
    monkeypatch.delenv("AI4STOCK_EXPERIMENT_BIN", raising=False)

    command = build_delegated_command(
        [
            "--experiment-profile",
            "base",
            "--set",
            "strategy.n_drop=2",
            "--sweep",
            "rolling.retrain_step=[5,10]",
            "--case",
            "strategy.topk=5",
            "strategy.n_drop=1",
            "--run-tag-prefix",
            "sweep",
            "--dedupe-predictions",
            "--skip-reference-baselines",
            "--dry-run",
        ]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-experiment", "--"]
    assert command[5] == "batch"
    assert ["--set", "strategy.n_drop=2"] == command[command.index("--set") : command.index("--set") + 2]
    assert ["--sweep", "rolling.retrain_step=[5,10]"] == command[
        command.index("--sweep") : command.index("--sweep") + 2
    ]
    assert "--case" in command
    assert "strategy.topk=5" in command
    assert ["--run-tag-prefix", "sweep"] == command[
        command.index("--run-tag-prefix") : command.index("--run-tag-prefix") + 2
    ]
    assert "--dedupe-predictions" in command
    assert "--skip-reference-baselines" in command
    assert "--dry-run" in command


def test_experiment_batch_wrapper_honors_binary_override(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_EXPERIMENT_BIN", "target/release/ai4stock-experiment --flag")

    command = build_delegated_command(["--experiment-profile", "base", "--dry-run"])

    assert command[:3] == ["target/release/ai4stock-experiment", "--flag", "batch"]
    assert "--dry-run" in command
