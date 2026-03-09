"""Native Modular Rolling Retrain Pipeline for AI4Stock2."""

import argparse
import json
import yaml
import pandas as pd
from pathlib import Path
import datetime
import torch
import numpy as np

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_rolling_pipeline():
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Rolling Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", default="lstm", help="Model name")
    parser.add_argument("--horizon", type=int, default=120, help="Rolling horizon in trading days")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--save-models", action="store_true", help="Save models for each rolling step")
    parser.add_argument("--load-models", action="store_true", help="Load existing models for each rolling step")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="Backtest rebalance frequency in days (default: 5 for weekly)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["name"] = args.model
    
    results_dir = Path("results") / f"native_rolling_{args.model}"
    models_dir = results_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models or args.load_models:
        models_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"\n>>> Running Native Rolling Pipeline (Backend: NATIVE) <<<")
    
    # ── 1. Load Global Native Data ────────────────────────
    alpha = cfg.get("alpha_version", 158)
    cache_dir = f"data/cache/alpha{alpha}_panel"
    lookback = cfg["features"]["lookback"]
    batch_size = cfg["model"]["batch_size"]
    
    print("\n[Step 1] Loading Global Native Memmap Data")
    meta_path = Path(cache_dir) / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    shape = tuple(meta["shape"])
    num_rows = meta["num_rows"]
    
    X = np.lib.format.open_memmap(Path(cache_dir) / "X.npy", mode="r", dtype=np.float32, shape=shape)
    y = np.lib.format.open_memmap(Path(cache_dir) / "y.npy", mode="r", dtype=np.float32, shape=(num_rows,))
    dates = np.lib.format.open_memmap(Path(cache_dir) / "date.npy", mode="r", dtype=np.int64, shape=(num_rows,))
    symbols = np.lib.format.open_memmap(Path(cache_dir) / "symbol.npy", mode="r", dtype=np.int32, shape=(num_rows,))
    
    id_to_symbol = {v: k for k, v in meta["symbol_to_id"].items()}
    dt_index = pd.to_datetime(dates)
    
    # Extract unique trading dates from the dataset
    all_trading_dates = sorted(dt_index.unique())
    calendar = pd.Series(all_trading_dates)
    
    # Filter global calendar to our test period
    test_start = pd.Timestamp(cfg["time"]["test"][0])
    test_end = pd.Timestamp(cfg["time"]["test"][1])
    
    test_calendar = calendar[(calendar >= test_start) & (calendar <= test_end)].reset_index(drop=True)
    
    # Universe Filtering
    universe_name = cfg.get("universe", "all")
    valid_symbol_ids = set()
    if universe_name != "all":
        try:
            uni_path = Path(cfg["qlib"]["provider_uri"]) / "instruments" / f"{universe_name}.txt"
            if uni_path.exists():
                with open(uni_path, "r") as f:
                    uni_symbols = set([line.split('\t')[0].strip() for line in f])
                for sym_key, sym_id in meta["symbol_to_id"].items():
                    raw_digit = ''.join(filter(str.isdigit, sym_key))
                    if raw_digit in uni_symbols:
                        valid_symbol_ids.add(sym_id)
        except Exception as e:
            pass
            
    if valid_symbol_ids:
        uni_mask = np.isin(symbols, list(valid_symbol_ids))
    else:
        uni_mask = np.ones(num_rows, dtype=bool)

    # ── 2. Setup Rolling Windows ────────────────────────
    rolling_steps = range(0, len(test_calendar), args.horizon)
    all_predictions = []
    
    print(f"\n[Rolling Setup] Testing from {test_start.date()} to {test_end.date()} with {args.horizon}-day steps.")

    from torch.utils.data import DataLoader
    from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer
    
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    for i, start_idx in enumerate(rolling_steps):
        current_test_start = test_calendar[start_idx]
        end_idx = min(start_idx + args.horizon - 1, len(test_calendar) - 1)
        current_test_end = test_calendar[end_idx]
        
        # Training window: Use previous 6 years up to test_start
        # Validation window: Use 1 year before test_start
        train_start = current_test_start - pd.Timedelta(days=365*6)
        train_end = current_test_start - pd.Timedelta(days=260)
        valid_start = current_test_start - pd.Timedelta(days=259)
        valid_end = current_test_start - pd.Timedelta(days=1)
        
        print(f"\n>>> [Step {i+1}/{len(rolling_steps)}] Window: {current_test_start.date()} to {current_test_end.date()}")
        print(f"    Train: {train_start.date()} ~ {train_end.date()} | Valid: {valid_start.date()} ~ {valid_end.date()}")

        # Data Slicing (Vectorized boolean masks)
        train_mask = (dt_index >= train_start) & (dt_index <= train_end) & uni_mask
        valid_mask = (dt_index >= valid_start) & (dt_index <= valid_end) & uni_mask
        test_mask  = (dt_index >= current_test_start)  & (dt_index <= current_test_end) & uni_mask
        
        train_dataset = NativeStockDataset(X, y, symbols, train_mask, lookback=lookback)
        valid_dataset = NativeStockDataset(X, y, symbols, valid_mask, lookback=lookback)
        test_dataset = NativeStockDataset(X, y, symbols, test_mask, lookback=lookback)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        # Train or Load Model
        model_path = models_dir / f"model_{current_test_start.strftime('%Y-%m-%d')}.pkl"
        
        trainer = NativeLSTMTrainer(
            d_feat=meta["num_features"],
            hidden_size=cfg["model"]["hidden_size"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            lr=cfg["model"]["lr"],
            loss_type=cfg["model"].get("loss", "pearson"),
            device=device
        )
        
        if args.load_models and model_path.exists():
            print(f"    Loading pre-trained model from {model_path}...")
            trainer.model.load_state_dict(torch.load(model_path, weights_only=True))
        else:
            print("    Training...")
            trainer.fit(train_loader, valid_loader, epochs=cfg["model"]["epochs"], early_stop=cfg["model"]["early_stop"])
            if args.save_models:
                torch.save(trainer.model.state_dict(), model_path)
                
        # Predict for current step
        trainer.model.eval()
        step_preds = []
        with torch.no_grad():
            for x_batch, _ in test_loader:
                p = trainer.model(x_batch.to(device))
                step_preds.append(p.cpu().numpy())
                
        if step_preds:
            preds_arr = np.concatenate(step_preds)
            end_indices = test_dataset.valid_end_indices
            aligned_dates = dt_index[end_indices]
            aligned_symbols = [id_to_symbol[sym] for sym in symbols[end_indices]]
            
            pred_series = pd.Series(
                preds_arr, 
                index=pd.MultiIndex.from_arrays([aligned_dates, aligned_symbols], names=['datetime', 'instrument'])
            ).sort_index()
            all_predictions.append(pred_series)
        
    # ── 5. Aggregate Results ────────────────────────
    if not all_predictions:
        print("No predictions generated.")
        return
        
    final_predictions = pd.concat(all_predictions).sort_index()
    
    # ── 6. Global Evaluation ────────────────────────
    print("\n" + "="*50)
    print("GLOBAL ROLLING EVALUATION")
    print("="*50)
    
    # Get all global labels for test period based on the universe
    global_test_mask = (dt_index >= test_start) & (dt_index <= test_end) & uni_mask
    global_dates = dt_index[global_test_mask]
    global_symbols = [id_to_symbol[sym] for sym in symbols[global_test_mask]]
    
    label_series = pd.Series(
        y[global_test_mask],
        index=pd.MultiIndex.from_arrays([global_dates, global_symbols], names=['datetime', 'instrument'])
    ).sort_index()
    
    # Align final predictions with global labels
    common_idx = final_predictions.index.intersection(label_series.index)
    aligned_preds = final_predictions.loc[common_idx]
    aligned_labels = label_series.loc[common_idx]
    
    from src.evaluate import compute_signal_metrics, print_metrics
    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)
    print_metrics(signal_metrics)
    
    # ── 7. Global Backtest ────────────────────────
    print(f"\n[Global Backtest] Rebalance Freq: {args.rebalance_freq} days")
    from src.native_backtest import run_native_backtest
    backtest_report = run_native_backtest(
        preds=aligned_preds,
        labels=aligned_labels,
        topk=cfg["strategy"]["topk"],
        cost_buy=cfg["backtest"]["cost"]["buy"],
        cost_sell=cfg["backtest"]["cost"]["sell"],
        rebalance_freq=args.rebalance_freq
    )
    
    plot_report = backtest_report.rename(columns={'net_return': 'return'})
    
    from src.evaluate import compute_portfolio_metrics, plot_cumulative_return, plot_drawdown, plot_monthly_heatmap, save_monthly_report
    portfolio_results, _ = compute_portfolio_metrics((plot_report, None))
    
    plot_cumulative_return(plot_report, save_path=str(results_dir / "native_cumulative_return.png"))
    plot_drawdown(plot_report, save_path=str(results_dir / "native_drawdown.png"))
    plot_monthly_heatmap(plot_report, save_path=str(results_dir / "native_monthly_heatmap.png"))
    save_monthly_report(plot_report, save_path=str(results_dir / "native_monthly_report.csv"))
    
    print_metrics(signal_metrics, portfolio_results)
    
    def sanitize_dict_keys(d):
        if not isinstance(d, dict): return d
        return {str(k): sanitize_dict_keys(v) for k, v in d.items()}
    
    with open(results_dir / "native_portfolio_metrics.json", "w") as f:
        json.dump(sanitize_dict_keys(portfolio_results), f, indent=2, default=str)

    print(f"\nNative rolling results saved to {results_dir}")

if __name__ == "__main__":
    run_rolling_pipeline()