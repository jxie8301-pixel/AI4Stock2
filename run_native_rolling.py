"""Native Modular Rolling Retrain Pipeline for AI4Stock2."""

import argparse
import json
import yaml
import pandas as pd
from pathlib import Path
import datetime
import torch
import numpy as np

from src.experiment_store import finalize_run_store, prepare_run_store
from src.feature_profiles import get_native_cache_dir
from src.feature_selection import compute_finite_feature_mask, resolve_selected_features
from src.label_utils import sanitize_label_array
from src.model_config import get_lgbm_config

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def run_rolling_pipeline():
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Rolling Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", default="lgbm", help="Model name")
    parser.add_argument("--horizon", type=int, default=10, help="Rolling horizon in trading days")
    parser.add_argument("--train-days", type=int, default=242, help="Training window length in trading days")
    parser.add_argument("--valid-days", type=int, default=10, help="Validation window length in trading days")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--save-models", action="store_true", help="Save models for each rolling step")
    parser.add_argument("--load-models", action="store_true", help="Load existing models for each rolling step")
    parser.add_argument("--rebalance-freq", type=int, default=5, help="Backtest rebalance frequency in days (default: 5 for weekly)")
    parser.add_argument("--topk", type=int, help="Override strategy top-k holdings")
    parser.add_argument("--n-drop", dest="n_drop", type=int, help="Override strategy daily replacement count")
    parser.add_argument("--run-tag", help="Short label for local experiment storage/comparison")
    parser.add_argument("--store-dir", help="Override local experiment store root")
    parser.add_argument("--disable-local-store", action="store_true", help="Disable automatic local experiment/model storage")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["model"]["name"] = args.model
    if args.topk is not None:
        cfg["strategy"]["topk"] = args.topk
    if args.n_drop is not None:
        cfg["strategy"]["n_drop"] = args.n_drop
    
    results_dir = Path("results") / f"native_rolling_{args.model}"
    models_dir = results_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models or args.load_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    run_store = prepare_run_store(
        cfg,
        args,
        backend="native",
        pipeline="rolling",
        model_name=args.model,
        model_ext=".pt" if args.model != "lgbm" else ".pkl",
    )
        
    print(f"\n>>> Running Native Rolling Pipeline (Backend: NATIVE) <<<")
    
    # ── 1. Load Global Native Data ────────────────────────
    cache_dir = get_native_cache_dir(cfg)
    lookback = cfg["features"]["lookback"]
    batch_size = cfg["model"]["batch_size"]
    
    print("\n[Step 1] Loading Global Native Memmap Data")
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Native cache missing: {cache_dir}. "
            "Please run `uv run python src/gen_feature.py --config configs/config.yaml` first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    selected_feature_idx, selected_feature_names = resolve_selected_features(meta, cfg)
    print(f"Selected features: {len(selected_feature_names)} / {len(meta['feature_names'])}")
        
    shape = tuple(meta["shape"])
    num_rows = meta["num_rows"]
    
    X = np.lib.format.open_memmap(Path(cache_dir) / "X.npy", mode="r", dtype=np.float32, shape=shape)
    y_memmap = np.lib.format.open_memmap(Path(cache_dir) / "y.npy", mode="r", dtype=np.float32, shape=(num_rows,))
    y = sanitize_label_array(y_memmap)
    dates = np.lib.format.open_memmap(Path(cache_dir) / "date.npy", mode="r", dtype=np.int64, shape=(num_rows,))
    symbols = np.lib.format.open_memmap(Path(cache_dir) / "symbol.npy", mode="r", dtype=np.int32, shape=(num_rows,))
    
    id_to_symbol = {v: k for k, v in meta["symbol_to_id"].items()}
    dt_index = pd.to_datetime(dates)
    
    # Extract unique trading dates from the dataset
    all_trading_dates = sorted(dt_index.unique())
    full_calendar = pd.Series(all_trading_dates)
    
    # Filter global calendar to our test period
    test_start = pd.Timestamp(cfg["time"]["test"][0])
    test_end = pd.Timestamp(cfg["time"]["test"][1])
    
    test_calendar = full_calendar[(full_calendar >= test_start) & (full_calendar <= test_end)].reset_index(drop=True)
    
    # Universe Filtering
    universe_name = cfg.get("universe", "all")
    if universe_name == "all":
        uni_mask = np.ones(num_rows, dtype=bool)
    else:
        from src.native_universe import build_universe_mask, resolve_universe_path

        universe_dir = cfg.get("native", {}).get("universe_dir", "data/universes")
        universe_path = resolve_universe_path(universe_name, universe_dir=universe_dir)
        uni_mask = build_universe_mask(
            dates_ns=dates,
            symbol_ids=symbols,
            symbol_to_id=meta["symbol_to_id"],
            universe_name=universe_name,
            universe_dir=universe_dir,
        )
        print(f"[*] Native universe '{universe_name}' loaded from {universe_path}. Matched rows: {int(uni_mask.sum())}.")

    # ── 2. Setup Rolling Windows ────────────────────────
    rolling_steps = range(0, len(test_calendar), args.horizon)
    all_predictions = []
    finite_feature_mask = compute_finite_feature_mask(X, selected_feature_idx, num_rows)
    
    print(f"\n[Rolling Setup] Testing from {test_start.date()} to {test_end.date()} with {args.horizon}-day steps.")

    from torch.utils.data import DataLoader
    from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer
    from src.evaluate import align_prediction_label_pairs
    
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    for i, start_idx in enumerate(rolling_steps):
        current_test_start = test_calendar[start_idx]
        end_idx = min(start_idx + args.horizon - 1, len(test_calendar) - 1)
        current_test_end = test_calendar[end_idx]
        full_start_idx = int(full_calendar.searchsorted(current_test_start))

        valid_end_idx = full_start_idx - 1
        valid_start_idx = valid_end_idx - args.valid_days + 1
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - args.train_days + 1

        if valid_end_idx < 0 or valid_start_idx < 0 or train_end_idx < 0 or train_start_idx < 0:
            print("    Skipping window: insufficient trading-day history for requested train/valid lengths.")
            continue

        train_start = full_calendar.iloc[train_start_idx]
        train_end = full_calendar.iloc[train_end_idx]
        valid_start = full_calendar.iloc[valid_start_idx]
        valid_end = full_calendar.iloc[valid_end_idx]
        
        print(f"\n>>> [Step {i+1}/{len(rolling_steps)}] Window: {current_test_start.date()} to {current_test_end.date()}")
        print(f"    Train: {train_start.date()} ~ {train_end.date()} | Valid: {valid_start.date()} ~ {valid_end.date()}")

        # Data Slicing (Vectorized boolean masks)
        train_mask = (dt_index >= train_start) & (dt_index <= train_end) & uni_mask
        valid_mask = (dt_index >= valid_start) & (dt_index <= valid_end) & uni_mask
        test_mask  = (dt_index >= current_test_start)  & (dt_index <= current_test_end) & uni_mask
        
        if args.model == "lgbm":
            from src.models.pure_lightgbm import NativeLGBM
            import pickle
            
            # Mask out NaNs in labels for training
            valid_train_mask = train_mask & finite_feature_mask & np.isfinite(y)
            valid_valid_mask = valid_mask & finite_feature_mask & np.isfinite(y)

            if not np.any(valid_train_mask):
                print("    Skipping window: no valid LightGBM training rows.")
                continue
            if not np.any(valid_valid_mask):
                print("    Skipping window: no valid LightGBM validation rows.")
                continue
            
            X_train_df = pd.DataFrame(X[valid_train_mask][:, selected_feature_idx], columns=selected_feature_names)
            y_train_series = pd.Series(y[valid_train_mask])
            X_valid_df = pd.DataFrame(X[valid_valid_mask][:, selected_feature_idx], columns=selected_feature_names)
            y_valid_series = pd.Series(y[valid_valid_mask])
            valid_dates = pd.to_datetime(dates[valid_valid_mask])
            
            model_path = models_dir / f"model_{current_test_start.strftime('%Y-%m-%d')}.pkl"
            model = NativeLGBM(**get_lgbm_config(cfg))
            
            if args.load_models and model_path.exists():
                print(f"    Loading pre-trained model from {model_path}...")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                print("    Training LightGBM...")
                model.fit(X_train_df, y_train_series, X_valid_df, y_valid_series, valid_dates=valid_dates)
                if args.save_models:
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
            
            test_valid_mask = test_mask & finite_feature_mask
            if not np.any(test_valid_mask):
                print("    Skipping window: no valid LightGBM test rows.")
                continue
            X_test_df = pd.DataFrame(X[test_valid_mask][:, selected_feature_idx], columns=selected_feature_names)
            
            preds_arr = model.predict(X_test_df)
            end_indices = np.where(test_valid_mask)[0]
            
        else:
            feature_indices = np.asarray(selected_feature_idx)
            train_dataset = NativeStockDataset(
                X,
                y,
                symbols,
                train_mask,
                lookback=lookback,
                full_dates=dates,
                feature_indices=feature_indices,
            )
            valid_dataset = NativeStockDataset(
                X,
                y,
                symbols,
                valid_mask,
                lookback=lookback,
                full_dates=dates,
                feature_indices=feature_indices,
            )
            test_dataset = NativeStockDataset(
                X,
                y,
                symbols,
                test_mask,
                lookback=lookback,
                full_dates=dates,
                feature_indices=feature_indices,
            )

            if len(train_dataset) == 0:
                print("    Skipping window: native LSTM training dataset is empty.")
                continue
            if len(valid_dataset) == 0:
                print("    Skipping window: native LSTM validation dataset is empty.")
                continue
            if len(test_dataset) == 0:
                print("    Skipping window: native LSTM test dataset is empty.")
                continue
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=len(train_dataset) >= batch_size,
            )
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
            # Train or Load Model
            model_path = models_dir / f"model_{current_test_start.strftime('%Y-%m-%d')}.pkl"
            
            trainer = NativeLSTMTrainer(
                d_feat=len(selected_feature_names),
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
                print("    Training LSTM...")
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
            else:
                preds_arr = np.array([])
            end_indices = test_dataset.valid_end_indices
            
        if len(preds_arr) > 0:
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
    aligned_preds, aligned_labels = align_prediction_label_pairs(aligned_preds, aligned_labels)
    
    from src.evaluate import compute_signal_metrics, print_metrics
    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)
    print_metrics(signal_metrics)
    
    # ── 7. Global Backtest ────────────────────────
    print(f"\n[Global Backtest] Rebalance Freq: {args.rebalance_freq} days")
    from src.native_backtest import run_native_backtest
    backtest_report = run_native_backtest(
        preds=final_predictions,
        labels=label_series,
        topk=cfg["strategy"]["topk"],
        n_drop=cfg["strategy"]["n_drop"],
        cost_buy=cfg["backtest"]["cost"]["buy"],
        cost_sell=cfg["backtest"]["cost"]["sell"],
        min_cost=cfg["backtest"].get("min_cost", 5.0),
        account=cfg["backtest"].get("account", 100_000_000),
        risk_degree=cfg["backtest"].get("risk_degree", 0.95),
        slippage=cfg["backtest"].get("slippage", 0.0005),
        rebalance_freq=args.rebalance_freq
    )
    
    plot_report = backtest_report.rename(columns={'net_return': 'return'})
    
    from src.evaluate import compute_portfolio_metrics, plot_cumulative_return, plot_drawdown, plot_monthly_heatmap, save_monthly_report
    portfolio_results, metric_report = compute_portfolio_metrics((plot_report, None))
    
    plot_cumulative_return(metric_report, save_path=str(results_dir / "native_cumulative_return.png"))
    plot_drawdown(metric_report, save_path=str(results_dir / "native_drawdown.png"))
    plot_monthly_heatmap(metric_report, save_path=str(results_dir / "native_monthly_heatmap.png"))
    save_monthly_report(metric_report, save_path=str(results_dir / "native_monthly_report.csv"))
    
    print_metrics(signal_metrics, portfolio_results)
    
    def sanitize_dict_keys(d):
        if not isinstance(d, dict): return d
        return {str(k): sanitize_dict_keys(v) for k, v in d.items()}
    
    with open(results_dir / "native_portfolio_metrics.json", "w") as f:
        json.dump(sanitize_dict_keys(portfolio_results), f, indent=2, default=str)

    manifest_path = finalize_run_store(
        run_store,
        cfg=cfg,
        args=args,
        backend="native",
        pipeline="rolling",
        model_name=args.model,
        results_dir=results_dir,
        signal_metrics=signal_metrics,
        portfolio_metrics=sanitize_dict_keys(portfolio_results),
        models_dir=models_dir if (args.save_models or args.load_models) else None,
        extra_context={
            "horizon": args.horizon,
            "train_days": args.train_days,
            "valid_days": args.valid_days,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "selected_features": selected_feature_names,
        },
    )
    if manifest_path:
        print(f"Local experiment manifest saved: {manifest_path}")

    print(f"\nNative rolling results saved to {results_dir}")

if __name__ == "__main__":
    run_rolling_pipeline()
