from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd

from src.rolling_evaluate import (
    _append_run_warning,
    _bundle_reference_baseline_predictions,
    _fixed_risk_baselines_match_same_gate,
    _reuse_same_gate_reports_as_fixed_risk,
    _resolve_backtest_artifact_options,
    _write_run_warnings,
)
from src.rolling_types import PredictionBundle


def test_append_run_warning_records_context() -> None:
    warnings: list[dict[str, object]] = []

    _append_run_warning(
        warnings,
        code="baseline_reconstruction_failed",
        message="baseline failed",
        baseline="rank_ic_weighted_factor",
        error_type="RuntimeError",
        error="boom",
    )

    assert warnings == [
        {
            "severity": "warning",
            "code": "baseline_reconstruction_failed",
            "message": "baseline failed",
            "baseline": "rank_ic_weighted_factor",
            "error_type": "RuntimeError",
            "error": "boom",
        }
    ]


def test_write_run_warnings_skips_empty_and_writes_json(tmp_path: Path) -> None:
    assert _write_run_warnings(tmp_path, []) is None

    path = _write_run_warnings(
        tmp_path,
        [
            {
                "severity": "warning",
                "code": "opportunity_label_derivation_failed",
                "message": "failed",
            }
        ],
    )

    assert path == tmp_path / "run_warnings.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["code"] == "opportunity_label_derivation_failed"


def test_fixed_risk_baselines_match_same_gate_only_for_equivalent_fixed_risk() -> None:
    assert _fixed_risk_baselines_match_same_gate(None, fallback_risk_degree=0.95)
    assert _fixed_risk_baselines_match_same_gate({"mode": "fixed"}, fallback_risk_degree=0.95)
    assert _fixed_risk_baselines_match_same_gate(
        {"mode": "fixed", "risk_degree": 0.95},
        fallback_risk_degree=0.95,
    )
    assert not _fixed_risk_baselines_match_same_gate(
        {"mode": "fixed", "risk_degree": 0.80},
        fallback_risk_degree=0.95,
    )
    assert not _fixed_risk_baselines_match_same_gate(
        {"mode": "benchmark_ma"},
        fallback_risk_degree=0.95,
    )


def test_reuse_same_gate_reports_as_fixed_risk_preserves_report_data() -> None:
    report = pd.DataFrame({"net_return": [0.01, -0.02]})

    reused = _reuse_same_gate_reports_as_fixed_risk(
        {"avg_factor_baseline": ("Avg Unique Factor Baseline", report)}
    )

    assert list(reused) == ["fixed_risk_avg_factor_baseline"]
    display_name, reused_report = reused["fixed_risk_avg_factor_baseline"]
    assert display_name == "Fixed-Risk Avg Unique Factor Baseline"
    assert reused_report is report


def test_skip_reference_baselines_ignores_bundle_baseline_predictions() -> None:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["datetime", "instrument"],
    )
    baseline_predictions = [
        pd.Series([float(value)], index=index, name="prediction")
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    bundle = PredictionBundle(
        final_predictions=pd.Series([0.1], index=index, name="prediction"),
        label_series=pd.Series([0.01], index=index, name="label"),
        backtest_label_series=pd.Series([0.001], index=index, name="label"),
        selected_feature_names=[],
        metadata={},
        feature_importance_frames=[],
        training_summary_records=[],
        avg_factor_baseline_predictions=baseline_predictions[0],
        sign_aligned_factor_baseline_predictions=baseline_predictions[1],
        rank_avg_factor_baseline_predictions=baseline_predictions[2],
        rank_ic_weighted_factor_baseline_predictions=baseline_predictions[3],
    )

    assert _bundle_reference_baseline_predictions(
        bundle,
        skip_reference_baselines=True,
    ) == [None, None, None, None]
    assert _bundle_reference_baseline_predictions(
        bundle,
        skip_reference_baselines=False,
    ) == baseline_predictions


def test_resolve_backtest_artifact_options_keeps_full_default_and_reduces_reports() -> None:
    full = _resolve_backtest_artifact_options(Namespace())
    assert full == {
        "level": "full",
        "skip_opportunity_diagnostics": False,
        "skip_backtest_plots": False,
        "skip_backtest_trace": False,
        "write_report_artifacts": True,
    }

    reports = _resolve_backtest_artifact_options(
        Namespace(
            backtest_artifact_level="reports",
            skip_opportunity_diagnostics=False,
            skip_backtest_plots=False,
            skip_backtest_trace=False,
        )
    )
    assert reports == {
        "level": "reports",
        "skip_opportunity_diagnostics": True,
        "skip_backtest_plots": True,
        "skip_backtest_trace": True,
        "write_report_artifacts": True,
    }


def test_resolve_backtest_artifact_options_allows_explicit_skips_on_full() -> None:
    options = _resolve_backtest_artifact_options(
        Namespace(
            backtest_artifact_level="full",
            skip_opportunity_diagnostics=True,
            skip_backtest_plots=False,
            skip_backtest_trace=True,
        )
    )

    assert options["level"] == "full"
    assert options["skip_opportunity_diagnostics"] is True
    assert options["skip_backtest_plots"] is False
    assert options["skip_backtest_trace"] is True
    assert options["write_report_artifacts"] is True
