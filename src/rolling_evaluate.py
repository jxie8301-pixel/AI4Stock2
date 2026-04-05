"""Evaluation helpers for the native rolling pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiment_store import finalize_run_store, resolve_rebalance_freq, resolve_retrain_step
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.rolling_baselines import (
    build_average_factor_baseline_predictions,
    build_sign_aligned_factor_baseline_predictions,
)
from src.rolling_runtime import load_rolling_runtime_data
from src.rolling_types import PredictionBundle, RollingPaths


def _sanitize_dict_keys(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    return {str(key): _sanitize_dict_keys(value) for key, value in data.items()}


def evaluate_prediction_bundle(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    paths: RollingPaths,
    run_store,
    bundle: PredictionBundle,
    *,
    model_name: str,
) -> None:
    from src.evaluate import (
        align_benchmark_to_report_index,
        align_prediction_label_pairs,
        build_benchmark_series,
        build_period_summary,
        build_rebalance_period_summary,
        compute_portfolio_metrics,
        compute_signal_metrics,
        plot_cumulative_return,
        plot_drawdown,
        plot_monthly_heatmap,
        print_metrics,
        save_monthly_report,
        save_period_summary,
    )
    from src.native_backtest import run_native_backtest

    signal_horizon = int(bundle.metadata.get("signal_horizon", resolve_signal_horizon(cfg)))
    rebalance_freq = int(resolve_rebalance_freq(cfg, args))
    avg_factor_baseline_predictions = bundle.avg_factor_baseline_predictions
    sign_aligned_factor_baseline_predictions = bundle.sign_aligned_factor_baseline_predictions
    if avg_factor_baseline_predictions is None:
        try:
            runtime_data = load_rolling_runtime_data(
                cfg,
                train_days=int(cfg.get("rolling", {}).get("train_days", 242)),
                valid_days=int(cfg.get("rolling", {}).get("valid_days", 10)),
                label_column=get_label_column_name(signal_horizon),
                backtest_label_column=get_label_column_name(1),
            )
            avg_factor_baseline_predictions = build_average_factor_baseline_predictions(runtime_data)
        except Exception as exc:
            print(f"[!] Skipping avg-factor baseline reconstruction: {exc}")
    if sign_aligned_factor_baseline_predictions is None:
        try:
            runtime_data = load_rolling_runtime_data(
                cfg,
                train_days=int(cfg.get("rolling", {}).get("train_days", 242)),
                valid_days=int(cfg.get("rolling", {}).get("valid_days", 10)),
                label_column=get_label_column_name(signal_horizon),
                backtest_label_column=get_label_column_name(1),
            )
            sign_aligned_factor_baseline_predictions = build_sign_aligned_factor_baseline_predictions(runtime_data)
        except Exception as exc:
            print(f"[!] Skipping sign-aligned baseline reconstruction: {exc}")

    common_idx = bundle.final_predictions.index.intersection(bundle.label_series.index)
    aligned_preds = bundle.final_predictions.loc[common_idx]
    aligned_labels = bundle.label_series.loc[common_idx]
    aligned_preds, aligned_labels = align_prediction_label_pairs(aligned_preds, aligned_labels)
    signal_metrics, _ = compute_signal_metrics(aligned_preds, aligned_labels)

    print(
        "\n[Backtest] "
        f"topk={cfg['strategy']['topk']}, "
        f"n_drop={cfg['strategy']['n_drop']}, "
        f"weighting={cfg['strategy'].get('weighting', 'equal')}, "
        f"score_transform={cfg['strategy'].get('score_transform', 'none')}, "
        f"score_zscore_clip={cfg['strategy'].get('score_zscore_clip', 3.0)}, "
        f"max_weight={cfg['strategy'].get('max_weight', 'none')}, "
        f"keep_top_n={cfg['strategy'].get('keep_top_n', 'none')}, "
        f"min_score={cfg['strategy'].get('min_score', 'none')}, "
        f"rebalance={rebalance_freq}d, "
        f"signal_label={signal_horizon}d, "
        "backtest_label=1d"
    )
    bench_series, benchmark_name = build_benchmark_series(
        bundle.backtest_label_series,
        cfg.get("backtest", {}).get("benchmark"),
    )
    backtest_kwargs = {
        "labels": bundle.backtest_label_series,
        "topk": cfg["strategy"]["topk"],
        "n_drop": cfg["strategy"]["n_drop"],
        "cost_buy": cfg["backtest"]["cost"]["buy"],
        "cost_sell": cfg["backtest"]["cost"]["sell"],
        "min_cost": cfg["backtest"].get("min_cost", 5.0),
        "account": cfg["backtest"].get("account", 100_000_000),
        "risk_degree": cfg["backtest"].get("risk_degree", 0.95),
        "slippage": cfg["backtest"].get("slippage", 0.0),
        "rebalance_freq": rebalance_freq,
        "weighting": cfg["strategy"].get("weighting", "equal"),
        "score_transform": cfg["strategy"].get("score_transform", "none"),
        "score_zscore_clip": cfg["strategy"].get("score_zscore_clip", 3.0),
        "max_weight": cfg["strategy"].get("max_weight"),
        "keep_top_n": cfg["strategy"].get("keep_top_n"),
        "min_score": cfg["strategy"].get("min_score"),
        "benchmark_returns": bench_series,
        "risk_control": cfg["backtest"].get("risk_control"),
        "intraperiod_exit": cfg["backtest"].get("intraperiod_exit"),
    }
    backtest_report = run_native_backtest(
        preds=bundle.final_predictions,
        **backtest_kwargs,
    )
    avg_factor_baseline_report = None
    if avg_factor_baseline_predictions is not None and not avg_factor_baseline_predictions.empty:
        avg_factor_baseline_report = run_native_backtest(
            preds=avg_factor_baseline_predictions,
            **backtest_kwargs,
        )
    sign_aligned_factor_baseline_report = None
    if sign_aligned_factor_baseline_predictions is not None and not sign_aligned_factor_baseline_predictions.empty:
        sign_aligned_factor_baseline_report = run_native_backtest(
            preds=sign_aligned_factor_baseline_predictions,
            **backtest_kwargs,
        )
    plot_report = backtest_report.rename(columns={"net_return": "return"})
    plot_report["bench"] = align_benchmark_to_report_index(
        bench_series,
        plot_report.index,
        benchmark_name=benchmark_name,
    ).to_numpy()
    plot_report.attrs["benchmark_name"] = benchmark_name
    plot_report.attrs["rebalance_freq"] = rebalance_freq
    if avg_factor_baseline_report is not None:
        plot_report["avg_factor_baseline_return"] = (
            avg_factor_baseline_report["net_return"].reindex(plot_report.index).fillna(0.0).to_numpy()
        )
        plot_report.attrs["avg_factor_baseline_name"] = "Avg Unique Factor Baseline"
    if sign_aligned_factor_baseline_report is not None:
        plot_report["sign_aligned_factor_baseline_return"] = (
            sign_aligned_factor_baseline_report["net_return"].reindex(plot_report.index).fillna(0.0).to_numpy()
        )
        plot_report.attrs["sign_aligned_factor_baseline_name"] = "Sign-Aligned Factor Baseline"

    portfolio_results, metric_report = compute_portfolio_metrics((plot_report, None))
    monthly_summary = build_period_summary(metric_report, freq="ME")
    rebalance_summary = build_rebalance_period_summary(metric_report, rebalance_freq)

    plot_cumulative_return(metric_report, save_path=str(paths.results_dir / "native_cumulative_return.png"))
    plot_drawdown(metric_report, save_path=str(paths.results_dir / "native_drawdown.png"))
    plot_monthly_heatmap(metric_report, save_path=str(paths.results_dir / "native_monthly_heatmap.png"))
    save_monthly_report(metric_report, save_path=str(paths.results_dir / "native_monthly_report.csv"))
    metric_report.to_csv(paths.results_dir / "native_daily_report.csv", index=True)
    save_period_summary(monthly_summary, paths.results_dir / "native_monthly_summary.csv")
    save_period_summary(rebalance_summary, paths.results_dir / "native_rebalance_summary.csv")
    print(f"Artifacts saved under: {paths.results_dir}")

    aggregated_importance_path: Path | None = None
    if bundle.feature_importance_frames:
        feature_importance_all = pd.concat(bundle.feature_importance_frames, ignore_index=True)
        aggregated_importance = (
            feature_importance_all.groupby("feature", as_index=False)["importance_gain"]
            .mean()
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
        aggregated_importance_path = paths.results_dir / "feature_importance_gain_mean.csv"
        aggregated_importance.to_csv(aggregated_importance_path, index=False)
        print(f"Feature importance saved: {aggregated_importance_path}")

    training_summary_path: Path | None = None
    if bundle.training_summary_records:
        training_summary_df = pd.DataFrame(bundle.training_summary_records)
        training_summary_path = paths.results_dir / "training_summary.csv"
        training_summary_df.to_csv(training_summary_path, index=False)
        print(f"Training summary saved: {training_summary_path}")

    print_metrics(signal_metrics, portfolio_results, period_summary=monthly_summary, period_label="Monthly")
    if not rebalance_summary.empty:
        print_metrics({}, {}, period_summary=rebalance_summary, period_label=f"Rebalance ({rebalance_freq}d)")

    safe_portfolio_results = _sanitize_dict_keys(portfolio_results)
    with open(paths.results_dir / "native_portfolio_metrics.json", "w") as f:
        json.dump(safe_portfolio_results, f, indent=2, default=str)

    manifest_path = finalize_run_store(
        run_store,
        cfg=cfg,
        args=args,
        backend="native",
        pipeline="rolling",
        model_name=model_name,
        results_dir=paths.results_dir,
        signal_metrics=signal_metrics,
        portfolio_metrics=safe_portfolio_results,
        models_dir=paths.models_dir if (args.save_models or args.load_models) else None,
        extra_context={
            "retrain_step": bundle.metadata.get("retrain_step", resolve_retrain_step(cfg, args)),
            "signal_horizon": bundle.metadata.get("signal_horizon", resolve_signal_horizon(cfg)),
            "train_days": bundle.metadata.get("train_days", ""),
            "valid_days": bundle.metadata.get("valid_days", ""),
            "test_start": bundle.metadata.get("test_start", cfg.get("time", {}).get("test", ["", ""])[0]),
            "test_end": bundle.metadata.get("test_end", cfg.get("time", {}).get("test", ["", ""])[1]),
            "selected_features": bundle.selected_feature_names,
            "feature_importance_path": str(aggregated_importance_path) if aggregated_importance_path else "",
            "training_summary_path": str(training_summary_path) if training_summary_path else "",
            "prediction_artifact_dir": str(paths.prediction_artifact_dir) if paths.prediction_artifact_dir.exists() else "",
        },
    )
    if manifest_path:
        print(f"Local experiment manifest saved: {manifest_path}")
    print(f"\nNative rolling results saved to {paths.results_dir}")
