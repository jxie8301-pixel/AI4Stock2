"""AI4Stock2 - Quantitative investment pipeline with dual backend routing."""

import argparse
from pathlib import Path
import json
import pickle
import yaml

from src.label_utils import sanitize_label_array

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _ensure_parent_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_native_model(model_name: str, model_obj, save_path: str) -> None:
    path = _ensure_parent_dir(save_path)
    if model_name == "lgbm":
        with open(path, "wb") as f:
            pickle.dump(model_obj, f)
    else:
        import torch

        torch.save(model_obj.state_dict(), path)
    print(f"Model saved: {path}")


def _load_native_model(model_name: str, load_path: str, model_obj, device: str | None = None):
    path = Path(load_path)
    if not path.exists():
        raise FileNotFoundError(f"Native model file not found: {path}")

    if model_name == "lgbm":
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        print(f"Model loaded: {path}")
        return loaded

    import torch

    state_dict = torch.load(path, map_location=device or "cpu", weights_only=True)
    model_obj.load_state_dict(state_dict)
    print(f"Model loaded: {path}")
    return model_obj

def run_qlib_pipeline(cfg, args, results_dir, model_name):
    """Original Qlib-dependent pipeline."""
    # ── Step 1: Data setup ────────────────────────────────────────────
    print("\n[Step 1/6] Data Setup")
    from src.data_setup import download_data, init_qlib

    if args.download_only:
        download_data(
            target_dir=cfg["qlib"]["provider_uri"],
            region=cfg["qlib"]["region"],
        )
        print("Data download complete. Exiting.")
        return

    init_qlib(
        provider_uri=cfg["qlib"]["provider_uri"],
        region=cfg["qlib"]["region"],
    )

    # ── Step 2: Feature engineering ───────────────────────────────────
    print("\n[Step 2/6] Feature Engineering")
    from src.features import build_alpha158_handler

    handler = build_alpha158_handler(
        instruments=cfg["universe"],
        start_time=cfg["time"]["train"][0],
        end_time=cfg["time"]["test"][1],
        fit_start_time=cfg["time"]["train"][0],
        fit_end_time=cfg["time"]["train"][1],
        use_valuation=cfg["features"].get("use_valuation", True),
    )

    # ── Step 3: Dataset construction ──────────────────────────────────
    print("\n[Step 3/6] Dataset Construction")
    from src.dataset import build_ts_dataset, build_tabular_dataset

    segments = {
        "train": tuple(cfg["time"]["train"]),
        "valid": tuple(cfg["time"]["valid"]),
        "test": tuple(cfg["time"]["test"]),
    }
    
    if model_name == "lgbm":
        dataset = build_tabular_dataset(
            handler=handler,
            segments=segments,
        )
    else:
        dataset = build_ts_dataset(
            handler=handler,
            segments=segments,
            step_len=cfg["features"]["lookback"],
        )

    # ── Step 4: Model training ────────────────────────────────────────
    if args.load_model:
        print(f"\n[Step 4/6] Loading Pre-trained Model from {args.load_model}")
        import pickle
        with open(args.load_model, "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully. Skipping training.")
    else:
        print(f"\n[Step 4/6] Model Training ({model_name})")
        model = _build_model(cfg, args.gpu)
        model.fit(dataset)
        print("Training complete.")
        
        if args.save_model:
            print(f"Saving model to {args.save_model}...")
            model.to_pickle(args.save_model)
            print("Model saved.")

    # ── Step 5: Prediction & signal evaluation ────────────────────────
    print("\n[Step 5/6] Prediction & Signal Evaluation")
    from src.evaluate import (
        build_cross_section_benchmark,
        compute_signal_metrics,
        plot_ic_series,
        print_metrics,
    )

    predictions = model.predict(dataset)

    # Get test labels for IC calculation
    test_label = dataset.handler.fetch(col_set="label")
    test_start = cfg["time"]["test"][0]
    test_end = cfg["time"]["test"][1]
    test_label = test_label.loc[(slice(test_start, test_end), slice(None)), :]
    
    if hasattr(test_label, "iloc"):
        test_label = test_label.iloc[:, 0]

    test_preds = predictions.loc[predictions.index.get_level_values(0) >= test_start]

    common_idx = test_preds.index.intersection(test_label.index)
    aligned_preds = test_preds.loc[common_idx]
    aligned_labels = test_label.loc[common_idx]

    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)

    plot_ic_series(daily_ic, save_path=str(results_dir / "ic_series.png"))
    
    if args.skip_backtest:
        print_metrics(signal_metrics)
        print(f"\n[Step 6/6] Backtest skipped.\n\nAll results saved to: {results_dir}/")
        return

    # ── Step 6: Backtest ──────────────────────────────────────────────
    print("\n[Step 6/6] Backtest")
    from src.backtest import run_backtest
    from src.evaluate import (
        compute_portfolio_metrics,
        plot_cumulative_return,
        plot_drawdown,
        plot_monthly_heatmap,
        save_monthly_report,
    )

    portfolio_metric = run_backtest(
        predictions=test_preds,
        topk=cfg["strategy"]["topk"],
        n_drop=cfg["strategy"]["n_drop"],
        cost_buy=cfg["backtest"]["cost"]["buy"],
        cost_sell=cfg["backtest"]["cost"]["sell"],
    )

    portfolio_results, report = compute_portfolio_metrics(portfolio_metric)
    bench_series = build_cross_section_benchmark(aligned_labels)
    report["bench"] = bench_series.reindex(pd.to_datetime(report.index)).fillna(0.0).to_numpy()

    plot_cumulative_return(report, save_path=str(results_dir / "cumulative_return.png"))
    plot_drawdown(report, save_path=str(results_dir / "drawdown.png"))
    plot_monthly_heatmap(report, save_path=str(results_dir / "monthly_heatmap.png"))
    save_monthly_report(report, save_path=str(results_dir / "monthly_report.csv"))

    print_metrics(signal_metrics, portfolio_results)

    with open(results_dir / "portfolio_metrics.json", "w") as f:
        json.dump(portfolio_results, f, indent=2, default=str)

    print(f"\nAll results saved to: {results_dir}/")
    print("Done!")

def run_native_pipeline(cfg, args, results_dir, model_name):
    """Pure PyTorch pipeline independent of Qlib."""
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer
    from src.evaluate import (
        align_prediction_label_pairs,
        build_cross_section_benchmark,
        compute_signal_metrics,
        print_metrics,
        plot_ic_series,
    )
    
    alpha = cfg.get("alpha_version", 158)
    cache_dir = f"data/cache/alpha{alpha}_panel"
    lookback = cfg["features"]["lookback"]
    batch_size = cfg["model"]["batch_size"]
    
    print("\n[Step 1/6] Loading Native Memmap Data")
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Native cache missing. Please run `python src/gen_feature.py --alpha {alpha}` first.")
        
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    shape = tuple(meta["shape"])
    num_rows = meta["num_rows"]
    
    X = np.lib.format.open_memmap(Path(cache_dir) / "X.npy", mode="r", dtype=np.float32, shape=shape)
    y_memmap = np.lib.format.open_memmap(Path(cache_dir) / "y.npy", mode="r", dtype=np.float32, shape=(num_rows,))
    y = sanitize_label_array(y_memmap)
    dates = np.lib.format.open_memmap(Path(cache_dir) / "date.npy", mode="r", dtype=np.int64, shape=(num_rows,))
    symbols = np.lib.format.open_memmap(Path(cache_dir) / "symbol.npy", mode="r", dtype=np.int32, shape=(num_rows,))
    
    id_to_symbol = {v: k for k, v in meta["symbol_to_id"].items()}
    
    print("\n[Step 2/6] Vectorized Time Splitting & Universe Filtering")
    dt_index = pd.to_datetime(dates)
    
    # --- Universe Filtering ---
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

    # --- Time Splitting (Intersection with Universe) ---
    train_mask = (dt_index >= pd.Timestamp(cfg["time"]["train"][0])) & (dt_index <= pd.Timestamp(cfg["time"]["train"][1])) & uni_mask
    valid_mask = (dt_index >= pd.Timestamp(cfg["time"]["valid"][0])) & (dt_index <= pd.Timestamp(cfg["time"]["valid"][1])) & uni_mask
    test_mask  = (dt_index >= pd.Timestamp(cfg["time"]["test"][0]))  & (dt_index <= pd.Timestamp(cfg["time"]["test"][1])) & uni_mask
    
    print("\n[Step 3/6] Initializing Native Datasets / Models")
    finite_feature_mask = ~np.isinf(X).any(axis=1)
    
    if model_name == "lgbm":
        from src.models.pure_lightgbm import NativeLGBM
        print(f"\n[Step 4/6] Native Model Training ({model_name})")
        # Extract 2D tabular data directly using masks
        valid_train_mask = train_mask & finite_feature_mask & np.isfinite(y)
        valid_valid_mask = valid_mask & finite_feature_mask & np.isfinite(y)

        if not np.any(valid_train_mask):
            raise ValueError("No valid native LightGBM training rows after filtering labels.")
        if not np.any(valid_valid_mask):
            raise ValueError("No valid native LightGBM validation rows after filtering labels.")
        
        X_train_df = pd.DataFrame(X[valid_train_mask])
        y_train_series = pd.Series(y[valid_train_mask])
        X_valid_df = pd.DataFrame(X[valid_valid_mask])
        y_valid_series = pd.Series(y[valid_valid_mask])
        valid_dates = pd.to_datetime(dates[valid_valid_mask])
        
        model = NativeLGBM(**cfg["model"])
        if args.load_model:
            model = _load_native_model(model_name, args.load_model, model)
            print("Skipping training.")
        else:
            model.fit(X_train_df, y_train_series, X_valid_df, y_valid_series, valid_dates=valid_dates)
            if args.save_model:
                _save_native_model(model_name, model, args.save_model)
        
        print("\n[Step 5/6] Native Prediction & Signal Evaluation")
        test_valid_mask = test_mask & finite_feature_mask
        if not np.any(test_valid_mask):
            raise ValueError("No valid native LightGBM test rows after filtering invalid features.")
        X_test_df = pd.DataFrame(X[test_valid_mask])
        preds_arr = model.predict(X_test_df)
        
        # We need the indices to align dates and symbols
        end_indices = np.where(test_valid_mask)[0]
    else:
        from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer
        from torch.utils.data import DataLoader
        train_dataset = NativeStockDataset(X, y, symbols, train_mask, lookback=lookback, full_dates=dates)
        valid_dataset = NativeStockDataset(X, y, symbols, valid_mask, lookback=lookback, full_dates=dates)
        test_dataset = NativeStockDataset(X, y, symbols, test_mask, lookback=lookback, full_dates=dates)

        if len(train_dataset) == 0:
            raise ValueError("Native LSTM training dataset is empty for the configured train split.")
        if len(valid_dataset) == 0:
            raise ValueError("Native LSTM validation dataset is empty for the configured valid split.")
        if len(test_dataset) == 0:
            raise ValueError("Native LSTM test dataset is empty for the configured test split.")
        
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
        
        print(f"\n[Step 4/6] Native Model Training ({model_name})")
        device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        trainer = NativeLSTMTrainer(
            d_feat=meta["num_features"],
            hidden_size=cfg["model"]["hidden_size"],
            num_layers=cfg["model"]["num_layers"],
            dropout=cfg["model"]["dropout"],
            lr=cfg["model"]["lr"],
            loss_type=cfg["model"].get("loss", "pearson"),
            device=device
        )
        if args.load_model:
            trainer.model = _load_native_model(model_name, args.load_model, trainer.model, device=device)
            print("Skipping training.")
        else:
            trainer.fit(train_loader, valid_loader, epochs=cfg["model"]["epochs"], early_stop=cfg["model"]["early_stop"])
            if args.save_model:
                _save_native_model(model_name, trainer.model, args.save_model)
        
        print("\n[Step 5/6] Native Prediction & Signal Evaluation")
        trainer.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                p = trainer.model(x_batch)
                all_preds.append(p.cpu().numpy())
                
        if not all_preds:
            raise ValueError("No valid test windows were generated for native LSTM inference.")
        preds_arr = np.concatenate(all_preds)
        end_indices = test_dataset.valid_end_indices
    
    aligned_dates = dt_index[end_indices]
    aligned_symbols = [id_to_symbol[sym] for sym in symbols[end_indices]]
    
    pred_series = pd.Series(
        preds_arr, 
        index=pd.MultiIndex.from_arrays([aligned_dates, aligned_symbols], names=['datetime', 'instrument'])
    ).sort_index()
    
    label_series = pd.Series(
        y[end_indices],
        index=pred_series.index
    )
    aligned_preds, aligned_labels = align_prediction_label_pairs(pred_series, label_series)
    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)
    plot_ic_series(daily_ic, save_path=str(results_dir / "native_ic_series.png"))
    
    if args.skip_backtest:
        print_metrics(signal_metrics)
        print(f"\n[Step 6/6] Backtest skipped.\n\nAll native results saved to: {results_dir}/")
        return
        
    # --- Step 6: Native Backtest ---
    print("\n[Step 6/6] Native Vectorized Backtest")
    from src.native_backtest import run_native_backtest
    from src.evaluate import (
        compute_portfolio_metrics,
        plot_cumulative_return,
        plot_drawdown,
        plot_monthly_heatmap,
        save_monthly_report,
    )
    
    rebalance_freq = getattr(args, 'rebalance_freq', None) or cfg.get("backtest", {}).get("rebalance_freq", 1)

    backtest_report = run_native_backtest(
        preds=pred_series,
        labels=label_series,
        topk=cfg["strategy"]["topk"],
        n_drop=cfg["strategy"]["n_drop"],
        cost_buy=cfg["backtest"]["cost"]["buy"],
        cost_sell=cfg["backtest"]["cost"]["sell"],
        min_cost=cfg["backtest"].get("min_cost", 5.0),
        account=cfg["backtest"].get("account", 100_000_000),
        risk_degree=cfg["backtest"].get("risk_degree", 0.95),
        slippage=cfg["backtest"].get("slippage", 0.0005),
        rebalance_freq=rebalance_freq
    )
    
    # Rename for compatibility with plot functions
    plot_report = backtest_report.rename(columns={'net_return': 'return'})
    
    # Pass a tuple (report, indicator) to match Qlib's expected format in evaluate.py
    portfolio_results, metric_report = compute_portfolio_metrics((plot_report, None))
    bench_series = build_cross_section_benchmark(label_series)
    metric_report["bench"] = bench_series.reindex(metric_report.index).fillna(0.0).to_numpy()
    
    # Generate native-specific plots/reports
    plot_cumulative_return(metric_report, save_path=str(results_dir / "native_cumulative_return.png"))
    plot_drawdown(metric_report, save_path=str(results_dir / "native_drawdown.png"))
    plot_monthly_heatmap(metric_report, save_path=str(results_dir / "native_monthly_heatmap.png"))
    save_monthly_report(metric_report, save_path=str(results_dir / "native_monthly_report.csv"))
    
    print_metrics(signal_metrics, portfolio_results)
    
    # JSON keys must be strings, convert any Timestamp keys (e.g. from monthly returns) to strings
    def sanitize_dict_keys(d):
        if not isinstance(d, dict):
            return d
        return {str(k): sanitize_dict_keys(v) for k, v in d.items()}
        
    safe_portfolio_results = sanitize_dict_keys(portfolio_results)
    
    with open(results_dir / "native_portfolio_metrics.json", "w") as f:
        json.dump(safe_portfolio_results, f, indent=2, default=str)
        
    print(f"\nAll native results saved to: {results_dir}/")
    print("Done!")

def _build_model(cfg: dict, gpu: int):
    """Build model based on config."""
    model_name = cfg["model"]["name"]
    model_cfg = cfg["model"]
    
    d_feat = 158
    if cfg["features"].get("use_valuation", True):
        d_feat += 8

    if model_name == "lstm":
        from src.models.lstm_model import build_lstm_model
        return build_lstm_model(
            d_feat=d_feat,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            n_epochs=model_cfg["epochs"],
            lr=model_cfg["lr"],
            early_stop=model_cfg["early_stop"],
            batch_size=model_cfg["batch_size"],
            loss=model_cfg.get("loss", "mse"),
            GPU=gpu,
            n_jobs=model_cfg.get("n_jobs", 12),
        )
    elif model_name == "transformer":
        from src.models.transformer_model import build_transformer_model
        return build_transformer_model(
            d_feat=d_feat,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            n_epochs=model_cfg["epochs"],
            lr=model_cfg["lr"],
            early_stop=model_cfg["early_stop"],
            batch_size=model_cfg["batch_size"],
            loss=model_cfg.get("loss", "mse"),
            GPU=gpu,
        )
    elif model_name == "lgbm":
        from src.models.lgbm_model import build_lgbm_model
        return build_lgbm_model()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: lstm, transformer, lgbm")

def main():
    parser = argparse.ArgumentParser(description="AI4Stock2 Quantitative Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", default=None, help="Model name: lstm / transformer / lgbm")
    parser.add_argument("--backend", default=None, help="Force backend: qlib or native (overrides config)")
    parser.add_argument("--download-only", action="store_true", help="Only download data")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest, only train and evaluate signal")
    parser.add_argument("--load-model", help="Path to a saved model to load (skip training)")
    parser.add_argument("--save-model", help="Path to save the trained model (e.g. results/lstm/model.pkl)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id (-1 for CPU)")
    parser.add_argument("--rebalance-freq", type=int, default=None, help="Backtest rebalance frequency in days (default: from config or 1)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.backend:
        cfg["backend"] = args.backend

    backend = cfg.get("backend", "qlib")
    model_name = cfg["model"]["name"]
    results_dir = Path("results") / backend / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Running Pipeline with Backend: {backend.upper()} <<<")
    
    if backend == "native":
        run_native_pipeline(cfg, args, results_dir, model_name)
    else:
        run_qlib_pipeline(cfg, args, results_dir, model_name)

if __name__ == "__main__":
    main()
