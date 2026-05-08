import run_experiment_batch


def test_build_rust_batch_command_forwards_sweeps_cases_and_dedupe():
    args = run_experiment_batch._build_parser().parse_args(
        [
            "--experiment-profile",
            "base",
            "--rust-runner",
            "target/release/ai4stock-experiment",
            "--python-runner",
            "pixi run python",
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

    command = run_experiment_batch._build_rust_batch_command(args)

    assert command[:2] == ["target/release/ai4stock-experiment", "batch"]
    assert command[command.index("--python-runner") + 1] == "pixi run python"
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
