"""AI4Stock2 native quantitative investment pipeline."""

import argparse
import inspect
import json
import pickle
from pathlib import Path

from src.experiment_store import finalize_run_store, prepare_run_store, resolve_rebalance_freq
from src.factor_store import load_factor_frame, load_factor_store_metadata
from src.feature_selection import (
    apply_feature_transforms,
    compute_finite_feature_mask_frame,
    resolve_selected_features,
)
from src.backtest_trace import parse_trace_dates_arg, save_trace_artifacts, select_trace_dates
from src.data_source import resolve_data_source_name
from src.feature_profiles import get_native_factor_store_dir
from src.label_utils import get_label_column_name, resolve_signal_horizon, sanitize_label_series
from src.model_config import get_lgbm_config
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args


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


def _maybe_export_backtest_trace(
    *,
    args,
    results_dir: Path,
    prefix: str,
    report_for_selection,
    rerun_fn,
):
    manual_dates = parse_trace_dates_arg(getattr(args, "trace_dates", None))
    auto_dates = set()
    if getattr(args, "trace_backtest", False):
        auto_dates = set(select_trace_dates(report_for_selection, top_n=getattr(args, "trace_top_days", 5)))

    selected_dates = sorted(manual_dates | auto_dates)
    if not selected_dates:
        return

    trace_df = rerun_fn(set(selected_dates))
    trace_path, dates_path = save_trace_artifacts(
        trace_df=trace_df,
        trace_dates=selected_dates,
        results_dir=results_dir,
        prefix=prefix,
    )
    print(f"Backtest trace saved: {trace_path}")
    print(f"Trace date manifest saved: {dates_path}")

def run_native_pipeline(cfg, args, results_dir, model_name):
    """Native training, evaluation, and backtest pipeline."""
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer
    from src.evaluate import (
        align_prediction_label_pairs,
        build_period_summary,
        build_cross_section_benchmark,
        compute_signal_metrics,
        print_metrics,
        plot_ic_series,
        save_period_summary,
    )
    
    factor_store_dir = get_native_factor_store_dir(cfg)
    data_source = resolve_data_source_name(cfg)
    lookback = cfg["features"]["lookback"]
    batch_size = cfg["model"]["batch_size"]
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)
    backtest_label_column = get_label_column_name(1)
    
    print(f"\n[Step 1/6] Loading Parquet Factor Store (data_source={data_source})")
    meta = load_factor_store_metadata(factor_store_dir)

    selected_feature_idx, selected_feature_names = resolve_selected_features(meta, cfg)
    print(f"Selected features: {len(selected_feature_names)} / {len(meta['feature_names'])}")
    universe_name = cfg.get("universe", "all")
    universe_dir = cfg.get("native", {}).get("universe_dir", "data/universes")
    load_start = min(cfg["time"]["train"][0], cfg["time"]["valid"][0], cfg["time"]["test"][0])
    load_end = max(cfg["time"]["train"][1], cfg["time"]["valid"][1], cfg["time"]["test"][1])
    factor_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=selected_feature_names + ([backtest_label_column] if backtest_label_column != label_column else []),
        label_column=label_column,
        date_start=load_start,
        date_end=load_end,
        universe_name=universe_name,
        universe_dir=universe_dir,
        sort_by=("date", "symbol"),
        progress_desc="loading factor store",
    )
    if factor_frame.empty:
        raise ValueError("Parquet factor store returned no rows for the configured date range and universe.")

    print("\n[Step 2/6] Time Splitting & Universe Filtering")
    dt_index = pd.to_datetime(factor_frame["date"])
    y = factor_frame["label"].to_numpy(dtype=np.float32, copy=True)
    if backtest_label_column in factor_frame.columns:
        backtest_y = sanitize_label_series(factor_frame[backtest_label_column]).to_numpy(dtype=np.float32, copy=True)
    else:
        backtest_y = y.copy()

    train_mask = (dt_index >= pd.Timestamp(cfg["time"]["train"][0])) & (dt_index <= pd.Timestamp(cfg["time"]["train"][1]))
    valid_mask = (dt_index >= pd.Timestamp(cfg["time"]["valid"][0])) & (dt_index <= pd.Timestamp(cfg["time"]["valid"][1]))
    test_mask  = (dt_index >= pd.Timestamp(cfg["time"]["test"][0]))  & (dt_index <= pd.Timestamp(cfg["time"]["test"][1]))
    
    print("\n[Step 3/6] Initializing Native Datasets / Models")
    finite_feature_mask = compute_finite_feature_mask_frame(factor_frame, selected_feature_names)
    
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
        
        X_train_df = factor_frame.loc[valid_train_mask, selected_feature_names].reset_index(drop=True)
        y_train_series = pd.Series(y[valid_train_mask])
        X_valid_df = factor_frame.loc[valid_valid_mask, selected_feature_names].reset_index(drop=True)
        y_valid_series = pd.Series(y[valid_valid_mask])
        train_dates = pd.to_datetime(dt_index[valid_train_mask]).reset_index(drop=True)
        valid_dates = pd.to_datetime(dt_index[valid_valid_mask]).reset_index(drop=True)
        X_train_df = apply_feature_transforms(X_train_df, train_dates, cfg)
        X_valid_df = apply_feature_transforms(X_valid_df, valid_dates, cfg)
        
        model = NativeLGBM(**get_lgbm_config(cfg))
        if args.load_model:
            model = _load_native_model(model_name, args.load_model, model)
            print("Skipping training.")
        else:
            fit_signature = inspect.signature(model.fit)
            fit_kwargs = {}
            if "train_dates" in fit_signature.parameters:
                fit_kwargs["train_dates"] = train_dates
            if "valid_dates" in fit_signature.parameters:
                fit_kwargs["valid_dates"] = valid_dates
            model.fit(
                X_train_df,
                y_train_series,
                X_valid_df,
                y_valid_series,
                **fit_kwargs,
            )
            if args.save_model:
                _save_native_model(model_name, model, args.save_model)
        feature_importance_path = results_dir / "feature_importance_gain.csv"
        model.save_feature_importance(feature_importance_path)
        print(f"Feature importance saved: {feature_importance_path}")
        
        print("\n[Step 5/6] Native Prediction & Signal Evaluation")
        test_valid_mask = test_mask & finite_feature_mask
        if not np.any(test_valid_mask):
            raise ValueError("No valid native LightGBM test rows after filtering invalid features.")
        X_test_df = factor_frame.loc[test_valid_mask, selected_feature_names].reset_index(drop=True)
        test_dates = pd.to_datetime(dt_index[test_valid_mask]).reset_index(drop=True)
        X_test_df = apply_feature_transforms(X_test_df, test_dates, cfg)
        preds_arr = model.predict(X_test_df)
        pred_dates = pd.to_datetime(dt_index[test_valid_mask]).reset_index(drop=True)
        pred_symbols = factor_frame.loc[test_valid_mask, "symbol"].reset_index(drop=True)
        pred_labels = y[test_valid_mask]
        pred_backtest_labels = backtest_y[test_valid_mask]
    else:
        from src.models.pure_pytorch_lstm import NativeStockDataset, NativeLSTMTrainer
        from torch.utils.data import DataLoader
        seq_frame = factor_frame.sort_values(["symbol", "date"]).reset_index(drop=True)
        seq_dates = pd.to_datetime(seq_frame["date"]).to_numpy()
        seq_symbols_str = seq_frame["symbol"].astype(str).to_numpy()
        seq_symbol_ids, unique_symbols = pd.factorize(seq_symbols_str, sort=True)
        id_to_symbol = {idx: symbol for idx, symbol in enumerate(unique_symbols)}
        X = seq_frame[selected_feature_names].to_numpy(dtype=np.float32, copy=True)
        y = seq_frame["label"].to_numpy(dtype=np.float32, copy=True)
        if backtest_label_column in seq_frame.columns:
            seq_backtest_y = sanitize_label_series(seq_frame[backtest_label_column]).to_numpy(dtype=np.float32, copy=True)
        else:
            seq_backtest_y = y.copy()
        dt_index = pd.to_datetime(seq_frame["date"])
        train_mask = (dt_index >= pd.Timestamp(cfg["time"]["train"][0])) & (dt_index <= pd.Timestamp(cfg["time"]["train"][1]))
        valid_mask = (dt_index >= pd.Timestamp(cfg["time"]["valid"][0])) & (dt_index <= pd.Timestamp(cfg["time"]["valid"][1]))
        test_mask  = (dt_index >= pd.Timestamp(cfg["time"]["test"][0]))  & (dt_index <= pd.Timestamp(cfg["time"]["test"][1]))
        train_dataset = NativeStockDataset(
            X,
            y,
            seq_symbol_ids,
            train_mask,
            lookback=lookback,
            full_dates=seq_dates,
            feature_indices=np.arange(len(selected_feature_names)),
        )
        valid_dataset = NativeStockDataset(
            X,
            y,
            seq_symbol_ids,
            valid_mask,
            lookback=lookback,
            full_dates=seq_dates,
            feature_indices=np.arange(len(selected_feature_names)),
        )
        test_dataset = NativeStockDataset(
            X,
            y,
            seq_symbol_ids,
            test_mask,
            lookback=lookback,
            full_dates=seq_dates,
            feature_indices=np.arange(len(selected_feature_names)),
        )

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
            d_feat=len(selected_feature_names),
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
        pred_dates = pd.to_datetime(seq_dates[test_dataset.valid_end_indices])
        pred_symbols = pd.Series([id_to_symbol[sym] for sym in seq_symbol_ids[test_dataset.valid_end_indices]])
        pred_labels = y[test_dataset.valid_end_indices]
        pred_backtest_labels = seq_backtest_y[test_dataset.valid_end_indices]

    pred_series = pd.Series(
        preds_arr,
        index=pd.MultiIndex.from_arrays([pred_dates, pred_symbols], names=['datetime', 'instrument'])
    ).sort_index()
    
    label_series = pd.Series(pred_labels, index=pred_series.index)
    backtest_label_series = pd.Series(pred_backtest_labels, index=pred_series.index)
    aligned_preds, aligned_labels = align_prediction_label_pairs(pred_series, label_series)
    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)
    plot_ic_series(daily_ic, save_path=str(results_dir / "native_ic_series.png"))
    
    if args.skip_backtest:
        print_metrics(signal_metrics)
        print(f"\n[Step 6/6] Backtest skipped.\n\nAll native results saved to: {results_dir}/")
        return {
            "signal_metrics": signal_metrics,
            "portfolio_metrics": None,
            "selected_features": selected_feature_names,
            "feature_importance_path": str(feature_importance_path) if model_name == "lgbm" else None,
        }
        
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
    
    rebalance_freq = resolve_rebalance_freq(cfg, args)

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
    backtest_report = run_native_backtest(
        preds=pred_series,
        labels=backtest_label_series,
        topk=cfg["strategy"]["topk"],
        n_drop=cfg["strategy"]["n_drop"],
        cost_buy=cfg["backtest"]["cost"]["buy"],
        cost_sell=cfg["backtest"]["cost"]["sell"],
        min_cost=cfg["backtest"].get("min_cost", 5.0),
        account=cfg["backtest"].get("account", 100_000_000),
        risk_degree=cfg["backtest"].get("risk_degree", 0.95),
        slippage=cfg["backtest"].get("slippage", 0.0),
        rebalance_freq=rebalance_freq,
        weighting=cfg["strategy"].get("weighting", "equal"),
        score_transform=cfg["strategy"].get("score_transform", "none"),
        score_zscore_clip=cfg["strategy"].get("score_zscore_clip", 3.0),
        max_weight=cfg["strategy"].get("max_weight"),
        keep_top_n=cfg["strategy"].get("keep_top_n"),
        min_score=cfg["strategy"].get("min_score"),
    )
    
    # Rename for compatibility with plot functions
    plot_report = backtest_report.rename(columns={'net_return': 'return'})
    
    portfolio_results, metric_report = compute_portfolio_metrics((plot_report, None))
    bench_series = build_cross_section_benchmark(backtest_label_series)
    metric_report["bench"] = bench_series.reindex(metric_report.index).fillna(0.0).to_numpy()
    monthly_summary = build_period_summary(metric_report, freq="ME")
    biweekly_summary = build_period_summary(metric_report, freq="2W-FRI")
    
    # Generate native-specific plots/reports
    plot_cumulative_return(metric_report, save_path=str(results_dir / "native_cumulative_return.png"))
    plot_drawdown(metric_report, save_path=str(results_dir / "native_drawdown.png"))
    plot_monthly_heatmap(metric_report, save_path=str(results_dir / "native_monthly_heatmap.png"))
    save_monthly_report(metric_report, save_path=str(results_dir / "native_monthly_report.csv"))
    metric_report.to_csv(results_dir / "native_daily_report.csv", index=True)
    save_period_summary(monthly_summary, results_dir / "native_monthly_summary.csv")
    save_period_summary(biweekly_summary, results_dir / "native_biweekly_summary.csv")
    print(f"Artifacts saved under: {results_dir}")

    _maybe_export_backtest_trace(
        args=args,
        results_dir=results_dir,
        prefix="native",
        report_for_selection=plot_report,
        rerun_fn=lambda trace_dates: run_native_backtest(
            preds=pred_series,
            labels=backtest_label_series,
            topk=cfg["strategy"]["topk"],
            n_drop=cfg["strategy"]["n_drop"],
            cost_buy=cfg["backtest"]["cost"]["buy"],
            cost_sell=cfg["backtest"]["cost"]["sell"],
            min_cost=cfg["backtest"].get("min_cost", 5.0),
            account=cfg["backtest"].get("account", 100_000_000),
            risk_degree=cfg["backtest"].get("risk_degree", 0.95),
            slippage=cfg["backtest"].get("slippage", 0.0),
            rebalance_freq=rebalance_freq,
            weighting=cfg["strategy"].get("weighting", "equal"),
            score_transform=cfg["strategy"].get("score_transform", "none"),
            score_zscore_clip=cfg["strategy"].get("score_zscore_clip", 3.0),
            max_weight=cfg["strategy"].get("max_weight"),
            keep_top_n=cfg["strategy"].get("keep_top_n"),
            min_score=cfg["strategy"].get("min_score"),
            return_trace=True,
            trace_dates=trace_dates,
        )[1],
    )
    
    print_metrics(signal_metrics, portfolio_results, period_summary=monthly_summary, period_label="Monthly")
    
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
    return {
        "signal_metrics": signal_metrics,
        "portfolio_metrics": safe_portfolio_results,
        "selected_features": selected_feature_names,
        "feature_importance_path": str(feature_importance_path) if model_name == "lgbm" else None,
        "signal_horizon": signal_horizon,
    }

def main():
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Quantitative Pipeline")
    add_common_runtime_args(parser, include_model_arg=True)
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest, only train and evaluate signal")
    parser.add_argument("--load-model", help="Path to a saved model to load (skip training)")
    parser.add_argument("--save-model", help="Path to save the trained model (e.g. results/lstm/model.pkl)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id (-1 for CPU)")
    parser.add_argument("--trace-backtest", action="store_true", help="Export detailed trace for selected backtest dates")
    parser.add_argument("--trace-top-days", type=int, default=5, help="Auto-select this many high-return/turnover/cost dates for trace export")
    parser.add_argument("--trace-dates", help="Comma-separated YYYY-MM-DD dates to include in backtest trace export")
    args = parser.parse_args()

    cfg = load_validated_config_from_args(args, parser)

    backend = "native"
    model_name = cfg["model"]["name"]
    model_ext = ".pt" if backend == "native" and model_name != "lgbm" else ".pkl"
    run_store = prepare_run_store(
        cfg,
        args,
        backend=backend,
        pipeline="single",
        model_name=model_name,
        model_ext=model_ext,
    )
    if run_store.enabled and run_store.run_dir:
        results_dir = run_store.run_dir
    else:
        results_dir = Path("results") / backend / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    if run_store.enabled and not args.save_model and not args.load_model:
        args.save_model = str(run_store.default_model_path)
        print(f"Local model store path: {args.save_model}")
    print(f"\n>>> Running Pipeline with Backend: {backend.upper()} <<<")

    run_summary = run_native_pipeline(cfg, args, results_dir, model_name)

    manifest_path = finalize_run_store(
        run_store,
        cfg=cfg,
        args=args,
        backend=backend,
        pipeline="single",
        model_name=model_name,
        results_dir=results_dir,
        signal_metrics=(run_summary or {}).get("signal_metrics"),
        portfolio_metrics=(run_summary or {}).get("portfolio_metrics"),
        model_path=args.save_model,
        load_model_path=args.load_model,
        extra_context={
            "selected_features": (run_summary or {}).get("selected_features", []),
            "feature_importance_path": (run_summary or {}).get("feature_importance_path"),
            "signal_horizon": (run_summary or {}).get("signal_horizon"),
        },
    )
    if manifest_path:
        print(f"Local experiment manifest saved: {manifest_path}")

if __name__ == "__main__":
    main()
