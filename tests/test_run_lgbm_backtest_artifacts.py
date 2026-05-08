from __future__ import annotations

import csv
from pathlib import Path

from run_lgbm_backtest_artifacts import build_command, build_parser, build_rust_bundle_command, run_jobs, select_rows


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "matrix_id": "train_a__bt_x",
        "train_id": "train_a",
        "backtest_profile_id": "bt_x",
        "config_snapshot": "configs/bt_x.yaml",
        "primary_predictions_dir": "runs/train_a/prediction_artifacts",
        "secondary_predictions_dir": "__NONE__",
        "train_signal_horizon": "20",
        "train_retrain_step": "10",
        "train_train_days": "242",
        "train_valid_days": "10",
        "train_label_embargo_days": "21",
        "train_model_profile": "lgbm_default_rankic",
        "train_feature_profile": "core_v4",
        "score_fusion_enabled": "False",
    }
    row.update(overrides)
    return row


def _write_selected_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(_row().keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def test_select_rows_filters_matrix_train_and_backtest_ids() -> None:
    rows = [
        _row(matrix_id="train_a__bt_x", train_id="train_a", backtest_profile_id="bt_x"),
        _row(matrix_id="train_a__bt_y", train_id="train_a", backtest_profile_id="bt_y"),
        _row(matrix_id="train_b__bt_x", train_id="train_b", backtest_profile_id="bt_x"),
    ]

    selected = select_rows(
        rows,
        matrix_ids=set(),
        train_ids={"train_a"},
        backtest_ids={"bt_y"},
        start_after="",
        limit=0,
    )

    assert [row["matrix_id"] for row in selected] == ["train_a__bt_y"]


def test_build_command_defaults_to_full_artifact_rebuild_without_skip_flags() -> None:
    command = build_command(
        _row(),
        python_runner=["python"],
        output_root=Path("artifact_runs"),
        model="lgbm",
        run_tag_prefix="artifact-rebuild-lgbm",
        artifact_level="full",
        save_predictions=False,
        skip_reference_baselines=False,
        skip_opportunity_diagnostics=False,
        skip_backtest_plots=False,
        skip_backtest_trace=False,
    )

    assert command[:2] == ["python", "run_native_rolling.py"]
    assert "--backtest-artifact-level" in command
    assert command[command.index("--backtest-artifact-level") + 1] == "full"
    assert "--skip-reference-baselines" not in command
    assert "--skip-opportunity-diagnostics" not in command
    assert "--skip-backtest-plots" not in command
    assert "--skip-backtest-trace" not in command
    assert ["--set", "rolling.label_embargo_days=21"] == command[command.index("--set") : command.index("--set") + 2]


def test_build_command_passes_secondary_predictions_for_score_fusion() -> None:
    command = build_command(
        _row(score_fusion_enabled="True", secondary_predictions_dir="runs/secondary/prediction_artifacts"),
        python_runner=["python"],
        output_root=Path("artifact_runs"),
        model="lgbm",
        run_tag_prefix="artifact-rebuild-lgbm",
        artifact_level="full",
        save_predictions=True,
        skip_reference_baselines=False,
        skip_opportunity_diagnostics=False,
        skip_backtest_plots=False,
        skip_backtest_trace=True,
    )

    assert "--save-predictions" in command
    assert "--skip-backtest-trace" in command
    assert "--set" in command
    assert "strategy.score_fusion.secondary_predictions_dir=runs/secondary/prediction_artifacts" in command


def test_build_rust_bundle_command_wraps_existing_post_bundle_argv() -> None:
    command = build_rust_bundle_command(
        _row(),
        rust_runner=["target/release/ai4stock-backtest"],
        python_runner="pixi run python",
        repo_root=Path("/repo"),
        output_root=Path("artifact_runs"),
        model="lgbm",
        run_tag_prefix="artifact-rebuild-lgbm",
        artifact_level="reports",
        save_predictions=False,
        skip_reference_baselines=True,
        skip_opportunity_diagnostics=True,
        skip_backtest_plots=True,
        skip_backtest_trace=True,
        disable_rust_backtest=False,
    )

    assert command[:7] == [
        "target/release/ai4stock-backtest",
        "bundle",
        "--python-runner",
        "pixi run python",
        "--repo-root",
        "/repo",
        "--",
    ]
    assert "--load-predictions-dir" in command
    assert command[command.index("--backtest-artifact-level") + 1] == "reports"
    assert "--skip-reference-baselines" in command
    assert "--skip-opportunity-diagnostics" in command
    assert "--skip-backtest-plots" in command
    assert "--skip-backtest-trace" in command


def test_run_jobs_skips_existing_done_marker(tmp_path: Path) -> None:
    selected_tsv = tmp_path / "selected.tsv"
    row = _row()
    _write_selected_tsv(selected_tsv, [row])
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    (marker_dir / f"{row['matrix_id']}.done").touch()
    log_dir = tmp_path / "logs"

    args = build_parser().parse_args(
        [
            "--selected-tsv",
            str(selected_tsv),
            "--output-root",
            str(tmp_path / "runs"),
            "--log-dir",
            str(log_dir),
            "--marker-dir",
            str(marker_dir),
        ]
    )

    assert run_jobs(args) == 0
    summary_paths = list(log_dir.glob("artifact_rebuild_summary_*.tsv"))
    assert len(summary_paths) == 1
    rows = list(csv.DictReader(summary_paths[0].open(newline=""), delimiter="\t"))
    assert rows[0]["status"] == "skipped_done"


def test_run_jobs_preflight_writes_batch_failure_markers(tmp_path: Path) -> None:
    selected_tsv = tmp_path / "selected.tsv"
    row = _row(primary_predictions_dir=str(tmp_path / "missing_predictions"))
    _write_selected_tsv(selected_tsv, [row])
    marker_dir = tmp_path / "markers"
    failed_tsv = tmp_path / "failed.tsv"
    log_dir = tmp_path / "logs"

    args = build_parser().parse_args(
        [
            "--selected-tsv",
            str(selected_tsv),
            "--output-root",
            str(tmp_path / "runs"),
            "--log-dir",
            str(log_dir),
            "--marker-dir",
            str(marker_dir),
            "--failed-tsv",
            str(failed_tsv),
        ]
    )

    assert run_jobs(args) == 1
    assert (marker_dir / f"{row['matrix_id']}.failed").read_text().strip() == "1"
    failed_rows = failed_tsv.read_text().strip().splitlines()
    assert failed_rows == [
        f"{row['matrix_id']}\t{row['train_id']}\tmissing_primary_predictions\t{row['primary_predictions_dir']}"
    ]
