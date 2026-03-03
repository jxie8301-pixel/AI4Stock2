"""Modular Rolling Retrain Pipeline for AI4Stock2."""

import argparse
import json
import yaml
import pandas as pd
from pathlib import Path
import datetime
from qlib.data import D
from src.data_setup import init_qlib
from src.features import build_alpha158_handler
from src.dataset import build_ts_dataset
from src.evaluate import compute_signal_metrics, compute_portfolio_metrics, print_metrics
from src.backtest import run_backtest

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_rolling_pipeline():
    parser = argparse.ArgumentParser(description="AI4Stock2 Rolling Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", default="lstm", help="Model name")
    parser.add_argument("--horizon", type=int, default=120, help="Rolling horizon in trading days (default: ~6 months)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["name"] = args.model
    
    init_qlib(provider_uri=cfg["qlib"]["provider_uri"], region=cfg["qlib"]["region"])
    
    # 1. Setup Rolling Windows
    # We want to test from 2022 to 2025
    test_start = "2022-01-01"
    test_end = "2025-12-31"
    
    calendar = D.calendar(start_time=test_start, end_time=test_end)
    # Split calendar into chunks of 'horizon' days
    rolling_steps = range(0, len(calendar), args.horizon)
    
    all_predictions = []
    
    print(f"
[Rolling Start] Testing from {test_start} to {test_end} with {args.horizon}-day steps.")

    for i, start_idx in enumerate(rolling_steps):
        current_test_start = calendar[start_idx]
        end_idx = min(start_idx + args.horizon - 1, len(calendar) - 1)
        current_test_end = calendar[end_idx]
        
        # Training window: Use previous 6 years up to test_start
        # Validation window: Use 1 year before test_start
        train_start = (current_test_start - pd.Timedelta(days=365*6)).strftime("%Y-%m-%d")
        train_end = (current_test_start - pd.Timedelta(days=260)).strftime("%Y-%m-%d")
        valid_start = (current_test_start - pd.Timedelta(days=259)).strftime("%Y-%m-%d")
        valid_end = (current_test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"
>>> [Step {i+1}] Window: {current_test_start.date()} to {current_test_end.date()}")
        print(f"    Train: {train_start} ~ {train_end} | Valid: {valid_start} ~ {valid_end}")

        # 2. Build Handler & Dataset for this window
        handler = build_alpha158_handler(
            instruments=cfg["universe"],
            start_time=train_start,
            end_time=current_test_end.strftime("%Y-%m-%d"),
            fit_start_time=train_start,
            fit_end_time=train_end,
            use_valuation=cfg["features"].get("use_valuation", True)
        )
        
        segments = {
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (current_test_start.strftime("%Y-%m-%d"), current_test_end.strftime("%Y-%m-%d")),
        }
        
        dataset = build_ts_dataset(
            handler=handler,
            segments=segments,
            step_len=cfg["features"]["lookback"],
        )

        # 3. Train Model
        from main import _build_model
        model = _build_model(cfg, args.gpu)
        model.fit(dataset)
        
        # 4. Predict
        preds = model.predict(dataset)
        # Only keep predictions for the current test segment
        test_preds = preds.loc[current_test_start:current_test_end]
        all_predictions.append(test_preds)
        
    # 5. Aggregate Results
    final_predictions = pd.concat(all_predictions).sort_index()
    
    # 6. Global Evaluation
    print("
" + "="*50)
    print("GLOBAL ROLLING EVALUATION")
    print("="*50)
    
    # Fetch global labels for the whole test period
    # To avoid re-calculating features for the whole period, we can reuse the last handler
    # but technically we need to be careful with normalization fit. 
    # For IC calculation, we just need the raw labels.
    handler_global = build_alpha158_handler(
        instruments=cfg["universe"],
        start_time=test_start,
        end_time=test_end,
    )
    labels = handler_global.fetch(col_set="label")
    if hasattr(labels, "iloc"):
        labels = labels.iloc[:, 0]
        
    common_idx = final_predictions.index.intersection(labels.index)
    aligned_preds = final_predictions.loc[common_idx]
    aligned_labels = labels.loc[common_idx]
    
    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)
    print_metrics(signal_metrics)
    
    # 7. Global Backtest
    print("
[Global Backtest]")
    portfolio_metric = run_backtest(
        predictions=final_predictions,
        topk=cfg["strategy"]["topk"],
        n_drop=cfg["strategy"]["n_drop"],
        cost_buy=cfg["backtest"]["cost"]["buy"],
        cost_sell=cfg["backtest"]["cost"]["sell"],
    )
    
    portfolio_results, report = compute_portfolio_metrics(portfolio_metric)
    
    results_dir = Path("results") / f"rolling_{args.model}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    from src.evaluate import plot_cumulative_return, plot_drawdown, plot_monthly_heatmap
    plot_cumulative_return(report, save_path=str(results_dir / "cumulative_return.png"))
    plot_drawdown(report, save_path=str(results_dir / "drawdown.png"))
    plot_monthly_heatmap(report, save_path=str(results_dir / "monthly_heatmap.png"))
    
    print_metrics(signal_metrics, portfolio_results)
    print(f"
Rolling results saved to {results_dir}")

if __name__ == "__main__":
    run_rolling_pipeline()
