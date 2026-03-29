"""Native Modular Rolling Retrain Pipeline for AI4Stock2."""

import argparse
import json
import pandas as pd
from pathlib import Path
import datetime
import torch
import numpy as np

from src.config_loader import load_config
from src.experiment_store import finalize_run_store, prepare_run_store
from src.factor_store import load_available_dates, load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import apply_feature_transforms, compute_finite_feature_mask_frame, resolve_selected_features
from src.model_config import get_lgbm_config

def run_rolling_pipeline():
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Rolling Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", default="lgbm", help="Model name")
    parser.add_argument("--profile", help="Override features.profile and use the corresponding factor-store/profile")
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
    if args.profile:
        cfg.setdefault("features", {})
        cfg["features"]["profile"] = args.profile
    if args.topk is not None:
        cfg["strategy"]["topk"] = args.topk
    if args.n_drop is not None:
        cfg["strategy"]["n_drop"] = args.n_drop
    
    run_store = prepare_run_store(
        cfg,
        args,
        backend="native",
        pipeline="rolling",
        model_name=args.model,
        model_ext=".pt" if args.model != "lgbm" else ".pkl",
    )
    if run_store.enabled and run_store.run_dir:
        results_dir = run_store.run_dir
        models_dir = run_store.models_dir or (results_dir / "models")
    else:
        results_dir = Path("results") / f"native_rolling_{args.model}"
        models_dir = results_dir / "models"
    importance_dir = results_dir / "feature_importance"
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models or args.load_models:
        models_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "lgbm":
        importance_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"\n>>> Running Native Rolling Pipeline (Backend: NATIVE) <<<")
    
    # ── 1. Load Global Native Data ────────────────────────
    factor_store_dir = get_native_factor_store_dir(cfg)
    lookback = cfg["features"]["lookback"]
    batch_size = cfg["model"]["batch_size"]
    
    print("\n[Step 1] Loading Parquet Factor Store Metadata")
    meta = load_factor_store_metadata(factor_store_dir)

    selected_feature_idx, selected_feature_names = resolve_selected_features(meta, cfg)
    print(f"Selected features: {len(selected_feature_names)} / {len(meta['feature_names'])}")
    universe_name = cfg.get("universe", "all")
    universe_dir = cfg.get("native", {}).get("universe_dir", "data/universes")
    all_trading_dates = load_available_dates(
        store_dir=factor_store_dir,
        universe_name=universe_name,
        universe_dir=universe_dir,
        progress_desc="scanning trading dates",
    )
    full_calendar = pd.Series(all_trading_dates)
    
    # Filter global calendar to our test period
    test_start = pd.Timestamp(cfg["time"]["test"][0])
    test_end = pd.Timestamp(cfg["time"]["test"][1])
    
    test_calendar = full_calendar[(full_calendar >= test_start) & (full_calendar <= test_end)].reset_index(drop=True)
    if test_calendar.empty:
        raise ValueError("No trading dates available for the configured test range.")

    first_test_start = test_calendar.iloc[0]
    first_test_idx = int(full_calendar.searchsorted(first_test_start))
    earliest_idx = max(0, first_test_idx - args.train_days - args.valid_days)
    load_start = full_calendar.iloc[earliest_idx]
    factor_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=selected_feature_names,
        date_start=load_start,
        date_end=test_end,
        universe_name=universe_name,
        universe_dir=universe_dir,
        sort_by=("date", "symbol"),
        progress_desc="loading factor store",
    )
    if factor_frame.empty:
        raise ValueError("Parquet factor store returned no rows for the configured rolling date range.")
    y = factor_frame["label"].to_numpy(dtype=np.float32, copy=True)
    dt_index = pd.to_datetime(factor_frame["date"])

    # ── 2. Setup Rolling Windows ────────────────────────
    rolling_steps = range(0, len(test_calendar), args.horizon)
    all_predictions = []
    feature_importance_frames = []
    finite_feature_mask = compute_finite_feature_mask_frame(factor_frame, selected_feature_names)
    
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
        train_mask = (dt_index >= train_start) & (dt_index <= train_end)
        valid_mask = (dt_index >= valid_start) & (dt_index <= valid_end)
        test_mask  = (dt_index >= current_test_start)  & (dt_index <= current_test_end)
        
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
            
            X_train_df = factor_frame.loc[valid_train_mask, selected_feature_names].reset_index(drop=True)
            y_train_series = pd.Series(y[valid_train_mask])
            X_valid_df = factor_frame.loc[valid_valid_mask, selected_feature_names].reset_index(drop=True)
            y_valid_series = pd.Series(y[valid_valid_mask])
            train_dates = pd.to_datetime(dt_index[valid_train_mask]).reset_index(drop=True)
            valid_dates = pd.to_datetime(dt_index[valid_valid_mask]).reset_index(drop=True)
            X_train_df = apply_feature_transforms(X_train_df, train_dates, cfg)
            X_valid_df = apply_feature_transforms(X_valid_df, valid_dates, cfg)
            
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

            importance_path = importance_dir / f"feature_importance_{current_test_start.strftime('%Y-%m-%d')}.csv"
            model.save_feature_importance(importance_path)
            importance_df = model.get_feature_importance_frame("gain").rename(columns={"gain": "importance_gain"})
            importance_df["window_start"] = current_test_start.strftime("%Y-%m-%d")
            feature_importance_frames.append(importance_df)
            
            test_valid_mask = test_mask & finite_feature_mask
            if not np.any(test_valid_mask):
                print("    Skipping window: no valid LightGBM test rows.")
                continue
            X_test_df = factor_frame.loc[test_valid_mask, selected_feature_names].reset_index(drop=True)
            test_dates = pd.to_datetime(dt_index[test_valid_mask]).reset_index(drop=True)
            X_test_df = apply_feature_transforms(X_test_df, test_dates, cfg)
            
            preds_arr = model.predict(X_test_df)
            pred_dates = pd.to_datetime(dt_index[test_valid_mask]).reset_index(drop=True)
            pred_symbols = factor_frame.loc[test_valid_mask, "symbol"].reset_index(drop=True)
            
        else:
            seq_frame = factor_frame.sort_values(["symbol", "date"]).reset_index(drop=True)
            seq_dt_index = pd.to_datetime(seq_frame["date"])
            seq_symbols_str = seq_frame["symbol"].astype(str).to_numpy()
            seq_symbol_ids, unique_symbols = pd.factorize(seq_symbols_str, sort=True)
            id_to_symbol = {idx: symbol for idx, symbol in enumerate(unique_symbols)}
            X = seq_frame[selected_feature_names].to_numpy(dtype=np.float32, copy=True)
            y_seq = seq_frame["label"].to_numpy(dtype=np.float32, copy=True)
            feature_indices = np.arange(len(selected_feature_names))
            train_mask = (seq_dt_index >= train_start) & (seq_dt_index <= train_end)
            valid_mask = (seq_dt_index >= valid_start) & (seq_dt_index <= valid_end)
            test_mask = (seq_dt_index >= current_test_start) & (seq_dt_index <= current_test_end)
            train_dataset = NativeStockDataset(
                X,
                y_seq,
                seq_symbol_ids,
                train_mask,
                lookback=lookback,
                full_dates=seq_dt_index.to_numpy(),
                feature_indices=feature_indices,
            )
            valid_dataset = NativeStockDataset(
                X,
                y_seq,
                seq_symbol_ids,
                valid_mask,
                lookback=lookback,
                full_dates=seq_dt_index.to_numpy(),
                feature_indices=feature_indices,
            )
            test_dataset = NativeStockDataset(
                X,
                y_seq,
                seq_symbol_ids,
                test_mask,
                lookback=lookback,
                full_dates=seq_dt_index.to_numpy(),
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
            pred_dates = pd.to_datetime(seq_dt_index.iloc[test_dataset.valid_end_indices]).reset_index(drop=True)
            pred_symbols = pd.Series([id_to_symbol[sym] for sym in seq_symbol_ids[test_dataset.valid_end_indices]])
            
        if len(preds_arr) > 0:
            pred_series = pd.Series(
                preds_arr, 
                index=pd.MultiIndex.from_arrays([pred_dates, pred_symbols], names=['datetime', 'instrument'])
            ).sort_index()
            all_predictions.append(pred_series)
        
    # ── 5. Aggregate Results ────────────────────────
    if not all_predictions:
        print("No predictions generated.")
        return
        
    final_predictions = pd.concat(all_predictions).sort_index()
    
    # ── 6. Global Evaluation ────────────────────────
    # Get all global labels for test period based on the universe
    global_test_mask = (dt_index >= test_start) & (dt_index <= test_end)
    global_dates = dt_index[global_test_mask]
    global_symbols = factor_frame.loc[global_test_mask, "symbol"]
    
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
    
    # ── 7. Global Backtest ────────────────────────
    print(
        "\n[Backtest] "
        f"topk={cfg['strategy']['topk']}, "
        f"n_drop={cfg['strategy']['n_drop']}, "
        f"rebalance={args.rebalance_freq}d"
    )
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
    
    from src.evaluate import (
        build_period_summary,
        compute_portfolio_metrics,
        plot_cumulative_return,
        plot_drawdown,
        plot_monthly_heatmap,
        save_monthly_report,
        save_period_summary,
    )
    portfolio_results, metric_report = compute_portfolio_metrics((plot_report, None))
    monthly_summary = build_period_summary(metric_report, freq="ME")
    biweekly_summary = build_period_summary(metric_report, freq="2W-FRI")
    
    plot_cumulative_return(metric_report, save_path=str(results_dir / "native_cumulative_return.png"))
    plot_drawdown(metric_report, save_path=str(results_dir / "native_drawdown.png"))
    plot_monthly_heatmap(metric_report, save_path=str(results_dir / "native_monthly_heatmap.png"))
    save_monthly_report(metric_report, save_path=str(results_dir / "native_monthly_report.csv"))
    metric_report.to_csv(results_dir / "native_daily_report.csv", index=True)
    save_period_summary(monthly_summary, results_dir / "native_monthly_summary.csv")
    save_period_summary(biweekly_summary, results_dir / "native_biweekly_summary.csv")
    print(f"Artifacts saved under: {results_dir}")

    aggregated_importance_path = None
    if feature_importance_frames:
        feature_importance_all = pd.concat(feature_importance_frames, ignore_index=True)
        aggregated_importance = (
            feature_importance_all.groupby("feature", as_index=False)["importance_gain"]
            .mean()
            .sort_values("importance_gain", ascending=False)
            .reset_index(drop=True)
        )
        aggregated_importance_path = results_dir / "feature_importance_gain_mean.csv"
        aggregated_importance.to_csv(aggregated_importance_path, index=False)
        print(f"Feature importance saved: {aggregated_importance_path}")
    
    print_metrics(signal_metrics, portfolio_results, period_summary=monthly_summary, period_label="Monthly")
    
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
            "feature_importance_path": str(aggregated_importance_path) if aggregated_importance_path else "",
        },
    )
    if manifest_path:
        print(f"Local experiment manifest saved: {manifest_path}")

    print(f"\nNative rolling results saved to {results_dir}")

if __name__ == "__main__":
    run_rolling_pipeline()
