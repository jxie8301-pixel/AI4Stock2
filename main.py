"""AI4Stock2 - Qlib-based quantitative investment pipeline.

Usage:
    python main.py                          # Default: LSTM model
    python main.py --model transformer      # Use Transformer
    python main.py --download-only          # Only download data
"""

import argparse
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="AI4Stock2 Quantitative Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--model", default=None, help="Model name: lstm / transformer / lgbm")
    parser.add_argument("--download-only", action="store_true", help="Only download data")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest, only train and evaluate signal")
    parser.add_argument("--load-model", help="Path to a saved model to load (skip training)")
    parser.add_argument("--save-model", help="Path to save the trained model (e.g. results/lstm/model.pkl)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id (-1 for CPU)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model

    model_name = cfg["model"]["name"]
    results_dir = Path("results") / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data setup ────────────────────────────────────────────
    print("\n[Step 1/6] Data Setup")
    from src.data_setup import download_data, init_qlib

    download_data(
        target_dir=cfg["qlib"]["provider_uri"],
        region=cfg["qlib"]["region"],
    )

    if args.download_only:
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
        compute_signal_metrics,
        plot_ic_series,
        print_metrics,
    )

    predictions = model.predict(dataset)

    # Get test labels for IC calculation
    test_label = dataset.handler.fetch(col_set="label")
    # Filter label to test period only
    test_start = cfg["time"]["test"][0]
    test_end = cfg["time"]["test"][1]
    test_label = test_label.loc[(slice(test_start, test_end), slice(None)), :]
    
    if hasattr(test_label, "iloc"):
        test_label = test_label.iloc[:, 0]

    # Filter predictions to test period only
    test_preds = predictions.loc[predictions.index.get_level_values(0) >= test_start]

    # Align predictions and labels
    common_idx = test_preds.index.intersection(test_label.index)
    aligned_preds = test_preds.loc[common_idx]
    aligned_labels = test_label.loc[common_idx]

    signal_metrics, daily_ic = compute_signal_metrics(aligned_preds, aligned_labels)

    plot_ic_series(daily_ic, save_path=str(results_dir / "ic_series.png"))
    print_metrics(signal_metrics)

    # Save signal metrics
    import json
    with open(results_dir / "signal_metrics.json", "w") as f:
        json.dump(signal_metrics, f, indent=2)

    # ── Step 6: Backtest ──────────────────────────────────────────────
    if not args.skip_backtest:
        print("\n[Step 6/6] Backtest")
        from src.backtest import run_backtest
        from src.evaluate import (
            compute_portfolio_metrics,
            plot_cumulative_return,
            plot_drawdown,
        )

        portfolio_metric = run_backtest(
            predictions=test_preds,
            topk=cfg["strategy"]["topk"],
            n_drop=cfg["strategy"]["n_drop"],
            cost_buy=cfg["backtest"]["cost"]["buy"],
            cost_sell=cfg["backtest"]["cost"]["sell"],
        )

        portfolio_results, report = compute_portfolio_metrics(portfolio_metric)

        plot_cumulative_return(report, save_path=str(results_dir / "cumulative_return.png"))
        plot_drawdown(report, save_path=str(results_dir / "drawdown.png"))

        print_metrics(signal_metrics, portfolio_results)

        with open(results_dir / "portfolio_metrics.json", "w") as f:
            json.dump(portfolio_results, f, indent=2, default=str)
    else:
        print("\n[Step 6/6] Backtest skipped.")

    print(f"\nAll results saved to: {results_dir}/")
    print("Done!")


def _build_model(cfg: dict, gpu: int):
    """Build model based on config."""
    model_name = cfg["model"]["name"]
    model_cfg = cfg["model"]

    if model_name == "lstm":
        from src.models.lstm_model import build_lstm_model
        return build_lstm_model(
            d_feat=158,
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
    elif model_name == "transformer":
        from src.models.transformer_model import build_transformer_model
        return build_transformer_model(
            d_feat=158,
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


if __name__ == "__main__":
    main()
