from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from run_candidate_pool_diagnostics import _parse_runs, sync_outputs_to_candidate_root
from src.candidate_diagnostics import bucket_shape, metric_value, parse_bucket_ids
from src.candidate_profiles import build_candidate_profile, flatten_candidate_profile, to_jsonable


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "native_portfolio_metrics.json",
        {
            "annualized_return": {"risk": 0.42},
            "annualized_volatility": {"risk": 0.16},
            "sharpe_ratio": {"risk": 2.4},
            "max_drawdown": {"risk": -0.08},
            "profit_factor": {"risk": 1.7},
            "excess_annualized_return": {"risk": 0.41},
            "excess_information_ratio": {"risk": 1.8},
            "monthly_win_rate": {"risk": 0.75},
            "rebalance_win_rate": {"risk": 0.67},
            "turnover_mean": {"risk": 0.04},
            "rank_avg_factor_baseline_excess_annualized_return": {"risk": 0.12},
            "rank_avg_factor_baseline_excess_information_ratio": {"risk": 1.1},
            "months_beating_rank_avg_factor_baseline_pct": {"risk": 0.62},
            "months_beating_rank_avg_factor_baseline_summary": "8 / 12 (66.67%)",
            "rebalances_beating_rank_avg_factor_baseline_pct": {"risk": 0.58},
            "rebalances_beating_rank_avg_factor_baseline_summary": "7 / 12 (58.33%)",
            "rank_ic_weighted_factor_baseline_excess_annualized_return": {"risk": 0.20},
            "rank_ic_weighted_factor_baseline_excess_information_ratio": {"risk": 1.4},
            "months_beating_rank_ic_weighted_factor_baseline_pct": {"risk": 0.75},
            "months_beating_rank_ic_weighted_factor_baseline_summary": "9 / 12 (75.00%)",
            "rebalances_beating_rank_ic_weighted_factor_baseline_pct": {"risk": 0.67},
            "rebalances_beating_rank_ic_weighted_factor_baseline_summary": "8 / 12 (66.67%)",
            "fixed_risk_rank_avg_factor_baseline_excess_annualized_return": {"risk": 0.18},
            "fixed_risk_rank_avg_factor_baseline_excess_information_ratio": {"risk": 1.5},
            "months_beating_fixed_risk_rank_avg_factor_baseline_pct": {"risk": 0.70},
            "months_beating_fixed_risk_rank_avg_factor_baseline_summary": "8 / 12 (66.67%)",
            "rebalances_beating_fixed_risk_rank_avg_factor_baseline_pct": {"risk": 0.60},
            "rebalances_beating_fixed_risk_rank_avg_factor_baseline_summary": "7 / 12 (58.33%)",
            "fixed_risk_rank_ic_weighted_factor_baseline_excess_annualized_return": {"risk": 0.22},
            "fixed_risk_rank_ic_weighted_factor_baseline_excess_information_ratio": {"risk": 1.6},
            "months_beating_fixed_risk_rank_ic_weighted_factor_baseline_pct": {"risk": 0.78},
            "months_beating_fixed_risk_rank_ic_weighted_factor_baseline_summary": "9 / 12 (75.00%)",
            "rebalances_beating_fixed_risk_rank_ic_weighted_factor_baseline_pct": {"risk": 0.68},
            "rebalances_beating_fixed_risk_rank_ic_weighted_factor_baseline_summary": "8 / 12 (66.67%)",
        },
    )
    monthly_rows = []
    for month in range(1, 13):
        monthly_rows.append(
            {
                "period": f"2024-{month:02d}",
                "return": 0.02,
                "bench_return": 0.0,
                "excess_vs_benchmark": 0.02,
                "avg_turnover": 0.04,
            }
        )
    _write_csv(run_dir / "native_monthly_summary.csv", monthly_rows)
    bucket_rows = [
        {"year": 2024, "bucket": 1, "label_mean": 0.04},
        {"year": 2024, "bucket": 2, "label_mean": 0.03},
        {"year": 2024, "bucket": 3, "label_mean": 0.02},
        {"year": 2024, "bucket": 4, "label_mean": 0.01},
        {"year": 2024, "bucket": 5, "label_mean": -0.01},
    ]
    _write_csv(run_dir / "native_score_bucket_report.csv", [{k: v for k, v in row.items() if k != "year"} for row in bucket_rows])
    _write_csv(run_dir / "native_score_bucket_yearly_report.csv", bucket_rows)
    _write_csv(
        run_dir / "feature_importance_gain_mean.csv",
        [
            {"feature": "TS_industry_std_60", "importance_gain": 5.0},
            {"feature": "LGBM_ep_ttm", "importance_gain": 2.0},
            {"feature": "TS_dividend_yield_ttm", "importance_gain": 1.0},
        ],
    )
    training_rows = []
    rebalance_rows = []
    for idx in range(9):
        date = f"2024-01-{idx + 1:02d}"
        signal = idx / 10
        training_rows.append(
            {
                "window_start": date,
                "valid_topk_excess_mean": signal,
                "valid_topk_positive_rate": signal,
                "valid_topk_label_mean": signal,
                "valid_topk_min_label_mean": signal,
                "best_valid_daily_rank_ic": signal,
            }
        )
        rebalance_rows.append(
            {
                "period_start": date,
                "return": 0.0 if idx < 3 else 0.02 if idx < 6 else 0.06,
                "excess_vs_benchmark": 0.0 if idx < 3 else 0.02 if idx < 6 else 0.06,
            }
        )
    _write_csv(run_dir / "training_summary.csv", training_rows)
    _write_csv(run_dir / "native_rebalance_summary.csv", rebalance_rows)
    _write_csv(
        run_dir / "native_daily_report.csv",
        [
            {"datetime": "2024-01-01", "return": 0.02, "risk_degree": 0.5, "account_value": 1.0},
            {"datetime": "2024-01-02", "return": -0.01, "risk_degree": 0.5, "account_value": 0.98},
            {"datetime": "2024-01-03", "return": 0.03, "risk_degree": 0.7, "account_value": 1.03},
        ],
    )
    return run_dir


def test_build_candidate_profile_marks_calibrated_ranker(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path)

    profile = build_candidate_profile("example", run_dir)
    flat = flatten_candidate_profile(profile)

    assert profile["candidate_role"] == "ranker_and_sizer"
    assert profile["gate_summary"]["failed_gates"] == []
    assert flat["best_bucket"] == 1
    assert flat["top_bucket_label_rank"] == 1
    assert profile["promotion_gates"]["beats_fixed_risk_rank_avg_factor_baseline"]["passed"]
    assert profile["promotion_gates"]["beats_fixed_risk_rank_ic_weighted_factor_baseline"]["passed"]
    assert flat["rank_avg_factor_baseline_rebalances_beating_pct"] == 0.58
    assert flat["rank_ic_weighted_factor_baseline_excess_annualized_return"] == 0.20
    assert flat["fixed_risk_rank_avg_factor_baseline_rebalances_beating_pct"] == 0.60
    assert flat["fixed_risk_rank_ic_weighted_factor_baseline_excess_annualized_return"] == 0.22
    assert flat["validation_high_low_return_spread"] > 0.05
    assert flat["strong_years"] == "2024"
    json.dumps(to_jsonable(profile))


def test_parse_runs_scans_candidate_root(tmp_path: Path) -> None:
    snapshot = tmp_path / "shortlist" / "candidates" / "candidate_a" / "snapshot"
    snapshot.mkdir(parents=True)
    args = Namespace(
        preset=None,
        candidate_root=str(tmp_path / "shortlist"),
        run=[],
    )

    runs = _parse_runs(args)

    assert runs == {"candidate_a": snapshot}


def test_sync_outputs_to_candidate_root_copies_aggregate_and_per_candidate_profiles(tmp_path: Path) -> None:
    output_dir = tmp_path / "diagnostics"
    shortlist_root = tmp_path / "shortlist"
    candidate_dir = shortlist_root / "candidates" / "candidate_a"
    candidate_dir.mkdir(parents=True)
    output_dir.mkdir()
    (output_dir / "README.md").write_text("# Diagnostics\n", encoding="utf-8")
    (output_dir / "portfolio_summary.csv").write_text("run,annualized_return\ncandidate_a,0.42\n", encoding="utf-8")
    profiles = [
        {
            "name": "candidate_a",
            "candidate_role": "ranker_and_sizer",
            "gate_summary": {"passed_count": 7, "total_count": 7, "failed_gates": []},
        }
    ]
    profile_frame = pd.DataFrame(
        [
            {
                "name": "candidate_a",
                "candidate_role": "ranker_and_sizer",
                "passed_gate_count": 7,
                "total_gate_count": 7,
                "failed_gates": "",
            }
        ]
    )

    sync_outputs_to_candidate_root(
        output_dir=output_dir,
        shortlist_root=shortlist_root,
        candidate_dirs={"candidate_a": candidate_dir},
        candidate_profiles=profiles,
        candidate_profile_frame=profile_frame,
    )

    assert (shortlist_root / "README.md").read_text(encoding="utf-8") == "# Diagnostics\n"
    assert (shortlist_root / "portfolio_summary.csv").exists()
    assert (candidate_dir / "candidate_profile.json").exists()
    assert (candidate_dir / "candidate_profile.csv").exists()


def test_candidate_diagnostics_shared_helpers_handle_metrics_and_buckets() -> None:
    frame = pd.DataFrame(
        [
            {"bucket": "1", "label_mean": "0.03"},
            {"bucket": "2", "label_mean": "0.01"},
            {"bucket": "3", "label_mean": "-0.02"},
        ]
    )

    shape = bucket_shape(frame, top_bucket=1, middle_buckets={2})

    assert metric_value({"sharpe_ratio": {"risk": 1.2}}, "sharpe_ratio") == 1.2
    assert parse_bucket_ids("2, 3") == {2, 3}
    assert shape["top_minus_bottom"] == 0.05
    assert shape["best_bucket"] == 1
    assert shape["top_bucket_label_rank"] == 1
