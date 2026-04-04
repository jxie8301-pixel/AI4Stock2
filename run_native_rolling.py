"""Native Modular Rolling Retrain Pipeline for AI4Stock2."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config_loader import load_config
from src.config_validation import validate_training_config
from src.data_source import resolve_data_source_name
from src.experiment_store import (
    finalize_run_store,
    prepare_run_store,
    resolve_rebalance_freq,
    resolve_retrain_step,
)
from src.factor_store import load_available_dates, load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import apply_feature_transforms, compute_finite_feature_mask_frame, resolve_selected_features
from src.label_utils import get_label_column_name, resolve_signal_horizon, sanitize_label_series
from src.model_config import get_lgbm_config
from src.runtime_cli import add_common_runtime_args, apply_common_runtime_overrides, load_validated_config_from_args


PREDICTION_ARTIFACT_DIRNAME = "prediction_artifacts"
PREDICTIONS_FILENAME = "final_predictions.parquet"
SIGNAL_LABELS_FILENAME = "signal_labels.parquet"
BACKTEST_LABELS_FILENAME = "backtest_labels.parquet"
PREDICTION_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class RollingPaths:
    results_dir: Path
    models_dir: Path
    importance_dir: Path
    training_history_dir: Path
    prediction_artifact_dir: Path


@dataclass(frozen=True)
class RollingRuntimeData:
    factor_frame: pd.DataFrame
    dt_index: pd.Series
    y: np.ndarray
    backtest_y: np.ndarray
    full_calendar: pd.Series
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_calendar: pd.Series
    selected_feature_names: list[str]
    finite_feature_mask: np.ndarray
    lookback: int
    batch_size: int


@dataclass(frozen=True)
class PredictionBundle:
    final_predictions: pd.Series
    label_series: pd.Series
    backtest_label_series: pd.Series
    selected_feature_names: list[str]
    metadata: dict[str, Any]
    feature_importance_frames: list[pd.DataFrame]
    training_summary_records: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Rolling Pipeline")
    add_common_runtime_args(parser, include_model_arg=True)
    parser.add_argument("--retrain-step", type=int, help="Rolling retrain step in trading days. If omitted, use config value.")
    parser.add_argument("--horizon", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--train-days", type=int, help="Training window length in trading days. If omitted, use config value.")
    parser.add_argument("--valid-days", type=int, help="Validation window length in trading days. If omitted, use config value.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--save-models", action="store_true", help="Save models for each rolling step")
    parser.add_argument("--load-models", action="store_true", help="Load existing models for each rolling step")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist final rolling predictions and labels for later backtest reuse.",
    )
    parser.add_argument(
        "--load-predictions-dir",
        help=(
            "Reuse an existing rolling prediction bundle directory and skip training/inference. "
            f"Accepts either a run directory or a direct {PREDICTION_ARTIFACT_DIRNAME}/ path."
        ),
    )
    return parser


def _build_paths(run_store, model_name: str) -> RollingPaths:
    if run_store.enabled and run_store.run_dir:
        results_dir = run_store.run_dir
        models_dir = run_store.models_dir or (results_dir / "models")
    else:
        results_dir = Path("results") / f"native_rolling_{model_name}"
        models_dir = results_dir / "models"
    return RollingPaths(
        results_dir=results_dir,
        models_dir=models_dir,
        importance_dir=results_dir / "feature_importance",
        training_history_dir=results_dir / "training_history",
        prediction_artifact_dir=results_dir / PREDICTION_ARTIFACT_DIRNAME,
    )


def _ensure_output_dirs(paths: RollingPaths, *, save_models: bool, load_models: bool, model_name: str) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    if save_models or load_models:
        paths.models_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "lgbm":
        paths.importance_dir.mkdir(parents=True, exist_ok=True)
        paths.training_history_dir.mkdir(parents=True, exist_ok=True)


def _series_to_frame(series: pd.Series, value_column: str) -> pd.DataFrame:
    frame = series.rename(value_column).rename_axis(index=["datetime", "instrument"]).reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def _frame_to_series(frame: pd.DataFrame, value_column: str) -> pd.Series:
    required = {"datetime", "instrument", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {', '.join(sorted(missing))}")
    return pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(frame["datetime"]),
                frame["instrument"].astype(str),
            ],
            names=["datetime", "instrument"],
        ),
        name=value_column,
    ).sort_index()


def _write_prediction_bundle(bundle: PredictionBundle, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _series_to_frame(bundle.final_predictions, "prediction").to_parquet(artifact_dir / PREDICTIONS_FILENAME, index=False)
    _series_to_frame(bundle.label_series, "label").to_parquet(artifact_dir / SIGNAL_LABELS_FILENAME, index=False)
    _series_to_frame(bundle.backtest_label_series, "label").to_parquet(
        artifact_dir / BACKTEST_LABELS_FILENAME,
        index=False,
    )
    metadata = {
        **bundle.metadata,
        "selected_features": list(bundle.selected_feature_names),
    }
    with open(artifact_dir / PREDICTION_METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def _resolve_prediction_artifact_dir(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_dir() and (path / PREDICTION_METADATA_FILENAME).exists():
        return path
    nested = path / PREDICTION_ARTIFACT_DIRNAME
    if nested.is_dir() and (nested / PREDICTION_METADATA_FILENAME).exists():
        return nested
    raise FileNotFoundError(
        f"Prediction artifact directory not found under {path}. "
        f"Expected {PREDICTION_METADATA_FILENAME} inside the directory or its {PREDICTION_ARTIFACT_DIRNAME}/ child."
    )


def load_prediction_bundle(raw_path: str | Path) -> PredictionBundle:
    artifact_dir = _resolve_prediction_artifact_dir(raw_path)
    with open(artifact_dir / PREDICTION_METADATA_FILENAME, encoding="utf-8") as f:
        metadata = json.load(f)
    final_predictions = _frame_to_series(pd.read_parquet(artifact_dir / PREDICTIONS_FILENAME), "prediction")
    label_series = _frame_to_series(pd.read_parquet(artifact_dir / SIGNAL_LABELS_FILENAME), "label")
    backtest_label_series = _frame_to_series(pd.read_parquet(artifact_dir / BACKTEST_LABELS_FILENAME), "label")
    return PredictionBundle(
        final_predictions=final_predictions,
        label_series=label_series,
        backtest_label_series=backtest_label_series,
        selected_feature_names=list(metadata.get("selected_features", [])),
        metadata=metadata,
        feature_importance_frames=[],
        training_summary_records=[],
    )


def load_rolling_runtime_data(
    cfg: dict[str, Any],
    *,
    train_days: int,
    valid_days: int,
    label_column: str,
    backtest_label_column: str,
) -> RollingRuntimeData:
    factor_store_dir = get_native_factor_store_dir(cfg)
    data_source = resolve_data_source_name(cfg)
    lookback = int(cfg["features"]["lookback"])
    batch_size = int(cfg["model"]["batch_size"])

    print(f"\n[Step 1] Loading Parquet Factor Store Metadata (data_source={data_source})")
    meta = load_factor_store_metadata(factor_store_dir)
    _, selected_feature_names = resolve_selected_features(meta, cfg)
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

    test_start = pd.Timestamp(cfg["time"]["test"][0])
    test_end = pd.Timestamp(cfg["time"]["test"][1])
    test_calendar = full_calendar[(full_calendar >= test_start) & (full_calendar <= test_end)].reset_index(drop=True)
    if test_calendar.empty:
        raise ValueError("No trading dates available for the configured test range.")

    first_test_start = test_calendar.iloc[0]
    first_test_idx = int(full_calendar.searchsorted(first_test_start))
    earliest_idx = max(0, first_test_idx - train_days - valid_days)
    load_start = full_calendar.iloc[earliest_idx]
    factor_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=selected_feature_names + ([backtest_label_column] if backtest_label_column != label_column else []),
        label_column=label_column,
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
    if backtest_label_column in factor_frame.columns:
        backtest_y = sanitize_label_series(factor_frame[backtest_label_column]).to_numpy(dtype=np.float32, copy=True)
    else:
        backtest_y = y.copy()
    dt_index = pd.to_datetime(factor_frame["date"])
    finite_feature_mask = compute_finite_feature_mask_frame(factor_frame, selected_feature_names)
    return RollingRuntimeData(
        factor_frame=factor_frame,
        dt_index=dt_index,
        y=y,
        backtest_y=backtest_y,
        full_calendar=full_calendar,
        test_start=test_start,
        test_end=test_end,
        test_calendar=test_calendar,
        selected_feature_names=selected_feature_names,
        finite_feature_mask=finite_feature_mask,
        lookback=lookback,
        batch_size=batch_size,
    )


def _build_label_series(
    runtime_data: RollingRuntimeData,
) -> tuple[pd.Series, pd.Series]:
    global_test_mask = (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)
    global_dates = runtime_data.dt_index[global_test_mask]
    global_symbols = runtime_data.factor_frame.loc[global_test_mask, "symbol"].astype(str)
    label_series = pd.Series(
        runtime_data.y[global_test_mask],
        index=pd.MultiIndex.from_arrays([global_dates, global_symbols], names=["datetime", "instrument"]),
        name="label",
    ).sort_index()
    backtest_label_series = pd.Series(
        runtime_data.backtest_y[global_test_mask],
        index=pd.MultiIndex.from_arrays([global_dates, global_symbols], names=["datetime", "instrument"]),
        name="label",
    ).sort_index()
    return label_series, backtest_label_series


def _build_prediction_metadata(
    cfg: dict[str, Any],
    *,
    signal_horizon: int,
    retrain_step: int,
    train_days: int,
    valid_days: int,
    runtime_data: RollingRuntimeData,
    model_name: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "data_source": resolve_data_source_name(cfg),
        "universe": str(cfg.get("universe", "")),
        "signal_horizon": int(signal_horizon),
        "retrain_step": int(retrain_step),
        "train_days": int(train_days),
        "valid_days": int(valid_days),
        "test_start": str(runtime_data.test_start.date()),
        "test_end": str(runtime_data.test_end.date()),
        "selected_feature_count": int(len(runtime_data.selected_feature_names)),
    }


def _prediction_series_from_arrays(preds_arr: np.ndarray, pred_dates: pd.Series, pred_symbols: pd.Series) -> pd.Series | None:
    if len(preds_arr) == 0:
        return None
    return pd.Series(
        preds_arr,
        index=pd.MultiIndex.from_arrays(
            [pd.to_datetime(pred_dates).reset_index(drop=True), pred_symbols.astype(str).reset_index(drop=True)],
            names=["datetime", "instrument"],
        ),
        name="prediction",
    ).sort_index()


def _run_lgbm_window(
    cfg: dict[str, Any],
    runtime_data: RollingRuntimeData,
    *,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    test_mask: np.ndarray,
    current_test_start: pd.Timestamp,
    current_test_end: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    signal_horizon: int,
    paths: RollingPaths,
    load_models: bool,
    save_models: bool,
) -> tuple[pd.Series | None, pd.DataFrame | None, dict[str, Any] | None]:
    from src.models.pure_lightgbm import NativeLGBM

    valid_train_mask = train_mask & runtime_data.finite_feature_mask & np.isfinite(runtime_data.y)
    valid_valid_mask = valid_mask & runtime_data.finite_feature_mask & np.isfinite(runtime_data.y)
    if not np.any(valid_train_mask):
        print("    Skipping window: no valid LightGBM training rows.")
        return None, None, None
    if not np.any(valid_valid_mask):
        print("    Skipping window: no valid LightGBM validation rows.")
        return None, None, None

    feature_names = runtime_data.selected_feature_names
    X_train_df = runtime_data.factor_frame.loc[valid_train_mask, feature_names].reset_index(drop=True)
    y_train_series = pd.Series(runtime_data.y[valid_train_mask])
    X_valid_df = runtime_data.factor_frame.loc[valid_valid_mask, feature_names].reset_index(drop=True)
    y_valid_series = pd.Series(runtime_data.y[valid_valid_mask])
    train_dates = pd.to_datetime(runtime_data.dt_index[valid_train_mask]).reset_index(drop=True)
    valid_dates = pd.to_datetime(runtime_data.dt_index[valid_valid_mask]).reset_index(drop=True)
    X_train_df = apply_feature_transforms(X_train_df, train_dates, cfg)
    X_valid_df = apply_feature_transforms(X_valid_df, valid_dates, cfg)

    model_path = paths.models_dir / f"model_{current_test_start.strftime('%Y-%m-%d')}.pkl"
    model = NativeLGBM(**get_lgbm_config(cfg))
    loaded_model = False
    if load_models and model_path.exists():
        print(f"    Loading pre-trained model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        loaded_model = True
    else:
        print("    Training LightGBM...")
        model.fit(
            X_train_df,
            y_train_series,
            X_valid_df,
            y_valid_series,
            train_dates=train_dates,
            valid_dates=valid_dates,
        )
        if save_models:
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    importance_path = paths.importance_dir / f"feature_importance_{current_test_start.strftime('%Y-%m-%d')}.csv"
    model.save_feature_importance(importance_path)
    importance_df = model.get_feature_importance_frame("gain").rename(columns={"gain": "importance_gain"})
    importance_df["window_start"] = current_test_start.strftime("%Y-%m-%d")

    history_path = paths.training_history_dir / f"training_history_{current_test_start.strftime('%Y-%m-%d')}.csv"
    saved_history_path = model.save_training_history(history_path)
    training_summary = {
        "window_start": current_test_start.strftime("%Y-%m-%d"),
        "window_end": current_test_end.strftime("%Y-%m-%d"),
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "valid_start": valid_start.strftime("%Y-%m-%d"),
        "valid_end": valid_end.strftime("%Y-%m-%d"),
        "signal_horizon": int(signal_horizon),
        "train_rows": int(len(X_train_df)),
        "valid_rows": int(len(X_valid_df)),
        "feature_count": int(len(feature_names)),
        "loaded_model": bool(loaded_model),
        "training_history_path": str(saved_history_path) if saved_history_path else "",
        **model.get_training_summary(),
    }

    test_valid_mask = test_mask & runtime_data.finite_feature_mask
    if not np.any(test_valid_mask):
        print("    Skipping window: no valid LightGBM test rows.")
        return None, importance_df, training_summary
    X_test_df = runtime_data.factor_frame.loc[test_valid_mask, feature_names].reset_index(drop=True)
    test_dates = pd.to_datetime(runtime_data.dt_index[test_valid_mask]).reset_index(drop=True)
    X_test_df = apply_feature_transforms(X_test_df, test_dates, cfg)
    preds_arr = model.predict(X_test_df)
    pred_dates = pd.to_datetime(runtime_data.dt_index[test_valid_mask]).reset_index(drop=True)
    pred_symbols = runtime_data.factor_frame.loc[test_valid_mask, "symbol"].reset_index(drop=True)
    return _prediction_series_from_arrays(preds_arr, pred_dates, pred_symbols), importance_df, training_summary


def _run_lstm_window(
    cfg: dict[str, Any],
    runtime_data: RollingRuntimeData,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    current_test_end: pd.Timestamp,
    current_test_start: pd.Timestamp,
    paths: RollingPaths,
    load_models: bool,
    save_models: bool,
    device: str,
) -> pd.Series | None:
    from torch.utils.data import DataLoader
    from src.models.pure_pytorch_lstm import NativeLSTMTrainer, NativeStockDataset

    seq_frame = runtime_data.factor_frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    seq_dt_index = pd.to_datetime(seq_frame["date"])
    seq_symbols_str = seq_frame["symbol"].astype(str).to_numpy()
    seq_symbol_ids, unique_symbols = pd.factorize(seq_symbols_str, sort=True)
    id_to_symbol = {idx: symbol for idx, symbol in enumerate(unique_symbols)}
    X = seq_frame[runtime_data.selected_feature_names].to_numpy(dtype=np.float32, copy=True)
    y_seq = seq_frame["label"].to_numpy(dtype=np.float32, copy=True)
    feature_indices = np.arange(len(runtime_data.selected_feature_names))

    train_mask_seq = (seq_dt_index >= train_start) & (seq_dt_index <= train_end)
    valid_mask_seq = (seq_dt_index >= valid_start) & (seq_dt_index <= valid_end)
    test_mask_seq = (seq_dt_index >= current_test_start) & (seq_dt_index <= current_test_end)

    train_dataset = NativeStockDataset(
        X,
        y_seq,
        seq_symbol_ids,
        train_mask_seq,
        lookback=runtime_data.lookback,
        full_dates=seq_dt_index.to_numpy(),
        feature_indices=feature_indices,
    )
    valid_dataset = NativeStockDataset(
        X,
        y_seq,
        seq_symbol_ids,
        valid_mask_seq,
        lookback=runtime_data.lookback,
        full_dates=seq_dt_index.to_numpy(),
        feature_indices=feature_indices,
    )
    test_dataset = NativeStockDataset(
        X,
        y_seq,
        seq_symbol_ids,
        test_mask_seq,
        lookback=runtime_data.lookback,
        full_dates=seq_dt_index.to_numpy(),
        feature_indices=feature_indices,
    )
    if len(train_dataset) == 0:
        print("    Skipping window: native LSTM training dataset is empty.")
        return None
    if len(valid_dataset) == 0:
        print("    Skipping window: native LSTM validation dataset is empty.")
        return None
    if len(test_dataset) == 0:
        print("    Skipping window: native LSTM test dataset is empty.")
        return None

    train_loader = DataLoader(
        train_dataset,
        batch_size=runtime_data.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=len(train_dataset) >= runtime_data.batch_size,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=runtime_data.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=runtime_data.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model_path = paths.models_dir / f"model_{current_test_start.strftime('%Y-%m-%d')}.pkl"
    trainer = NativeLSTMTrainer(
        d_feat=len(runtime_data.selected_feature_names),
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        lr=cfg["model"]["lr"],
        loss_type=cfg["model"].get("loss", "pearson"),
        device=device,
    )
    if load_models and model_path.exists():
        print(f"    Loading pre-trained model from {model_path}...")
        trainer.model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print("    Training LSTM...")
        trainer.fit(train_loader, valid_loader, epochs=cfg["model"]["epochs"], early_stop=cfg["model"]["early_stop"])
        if save_models:
            torch.save(trainer.model.state_dict(), model_path)

    trainer.model.eval()
    step_preds: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            pred_batch = trainer.model(x_batch.to(device))
            step_preds.append(pred_batch.cpu().numpy())
    preds_arr = np.concatenate(step_preds) if step_preds else np.array([])
    pred_dates = pd.to_datetime(seq_dt_index.iloc[test_dataset.valid_end_indices]).reset_index(drop=True)
    pred_symbols = pd.Series([id_to_symbol[sym] for sym in seq_symbol_ids[test_dataset.valid_end_indices]])
    return _prediction_series_from_arrays(preds_arr, pred_dates, pred_symbols)


def generate_prediction_bundle(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    runtime_data: RollingRuntimeData,
    paths: RollingPaths,
    *,
    retrain_step: int,
    train_days: int,
    valid_days: int,
    signal_horizon: int,
    model_name: str,
) -> PredictionBundle:
    rolling_steps = range(0, len(runtime_data.test_calendar), retrain_step)
    all_predictions: list[pd.Series] = []
    feature_importance_frames: list[pd.DataFrame] = []
    training_summary_records: list[dict[str, Any]] = []

    print(
        f"\n[Rolling Setup] Testing from {runtime_data.test_start.date()} to {runtime_data.test_end.date()} "
        f"| retrain_step={retrain_step}d | signal_horizon={signal_horizon}d | "
        f"rebalance={resolve_rebalance_freq(cfg, args)}d"
    )
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

    for i, start_idx in enumerate(rolling_steps):
        current_test_start = runtime_data.test_calendar.iloc[start_idx]
        end_idx = min(start_idx + retrain_step - 1, len(runtime_data.test_calendar) - 1)
        current_test_end = runtime_data.test_calendar.iloc[end_idx]
        full_start_idx = int(runtime_data.full_calendar.searchsorted(current_test_start))

        valid_end_idx = full_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days + 1
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - train_days + 1
        if valid_end_idx < 0 or valid_start_idx < 0 or train_end_idx < 0 or train_start_idx < 0:
            print("    Skipping window: insufficient trading-day history for requested train/valid lengths.")
            continue

        train_start = runtime_data.full_calendar.iloc[train_start_idx]
        train_end = runtime_data.full_calendar.iloc[train_end_idx]
        valid_start = runtime_data.full_calendar.iloc[valid_start_idx]
        valid_end = runtime_data.full_calendar.iloc[valid_end_idx]
        print(f"\n>>> [Step {i + 1}/{len(runtime_data.test_calendar[::retrain_step])}] Window: {current_test_start.date()} to {current_test_end.date()}")
        print(f"    Train: {train_start.date()} ~ {train_end.date()} | Valid: {valid_start.date()} ~ {valid_end.date()}")

        train_mask = (runtime_data.dt_index >= train_start) & (runtime_data.dt_index <= train_end)
        valid_mask = (runtime_data.dt_index >= valid_start) & (runtime_data.dt_index <= valid_end)
        test_mask = (runtime_data.dt_index >= current_test_start) & (runtime_data.dt_index <= current_test_end)

        if model_name == "lgbm":
            pred_series, importance_df, training_summary = _run_lgbm_window(
                cfg,
                runtime_data,
                train_mask=train_mask,
                valid_mask=valid_mask,
                test_mask=test_mask,
                current_test_start=current_test_start,
                current_test_end=current_test_end,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                signal_horizon=signal_horizon,
                paths=paths,
                load_models=args.load_models,
                save_models=args.save_models,
            )
            if importance_df is not None:
                feature_importance_frames.append(importance_df)
            if training_summary is not None:
                training_summary_records.append(training_summary)
        else:
            pred_series = _run_lstm_window(
                cfg,
                runtime_data,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                current_test_end=current_test_end,
                current_test_start=current_test_start,
                paths=paths,
                load_models=args.load_models,
                save_models=args.save_models,
                device=device,
            )

        if pred_series is not None and not pred_series.empty:
            all_predictions.append(pred_series)

    if not all_predictions:
        raise ValueError("No predictions generated.")

    final_predictions = pd.concat(all_predictions).sort_index()
    label_series, backtest_label_series = _build_label_series(runtime_data)
    return PredictionBundle(
        final_predictions=final_predictions,
        label_series=label_series,
        backtest_label_series=backtest_label_series,
        selected_feature_names=list(runtime_data.selected_feature_names),
        metadata=_build_prediction_metadata(
            cfg,
            signal_horizon=signal_horizon,
            retrain_step=retrain_step,
            train_days=train_days,
            valid_days=valid_days,
            runtime_data=runtime_data,
            model_name=model_name,
        ),
        feature_importance_frames=feature_importance_frames,
        training_summary_records=training_summary_records,
    )


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
    backtest_report = run_native_backtest(
        preds=bundle.final_predictions,
        labels=bundle.backtest_label_series,
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
        benchmark_returns=bench_series,
        dynamic_risk=cfg["backtest"].get("dynamic_risk"),
    )
    plot_report = backtest_report.rename(columns={"net_return": "return"})
    plot_report["bench"] = align_benchmark_to_report_index(
        bench_series,
        plot_report.index,
        benchmark_name=benchmark_name,
    ).to_numpy()
    plot_report.attrs["benchmark_name"] = benchmark_name
    plot_report.attrs["rebalance_freq"] = rebalance_freq

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

    aggregated_importance_path = None
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

    training_summary_path = None
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


def run_rolling_pipeline() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.load_predictions_dir:
        try:
            cfg = load_config(
                args.config,
                experiment_profile_name=getattr(args, "experiment_profile", None),
                model_profile_name=getattr(args, "model_profile", None),
            )
            apply_common_runtime_overrides(cfg, args, parser, allow_rolling_overrides=True)
            validate_training_config(cfg, check_paths=False)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        cfg = load_validated_config_from_args(args, parser, allow_rolling_overrides=True)

    retrain_step = int(resolve_retrain_step(cfg, args))
    train_days = int(cfg.get("rolling", {}).get("train_days", args.train_days or 242))
    valid_days = int(cfg.get("rolling", {}).get("valid_days", args.valid_days or 10))
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)
    backtest_label_column = get_label_column_name(1)
    model_name = cfg["model"]["name"]

    run_store = prepare_run_store(
        cfg,
        args,
        backend="native",
        pipeline="rolling",
        model_name=model_name,
        model_ext=".pt" if model_name != "lgbm" else ".pkl",
    )
    paths = _build_paths(run_store, model_name)
    _ensure_output_dirs(paths, save_models=args.save_models, load_models=args.load_models, model_name=model_name)

    print(f"\n>>> Running Native Rolling Pipeline (Backend: NATIVE) <<<")
    if args.load_predictions_dir:
        bundle = load_prediction_bundle(args.load_predictions_dir)
        print(f"[*] Loaded prediction bundle from {_resolve_prediction_artifact_dir(args.load_predictions_dir)}")
    else:
        runtime_data = load_rolling_runtime_data(
            cfg,
            train_days=train_days,
            valid_days=valid_days,
            label_column=label_column,
            backtest_label_column=backtest_label_column,
        )
        bundle = generate_prediction_bundle(
            cfg,
            args,
            runtime_data,
            paths,
            retrain_step=retrain_step,
            train_days=train_days,
            valid_days=valid_days,
            signal_horizon=signal_horizon,
            model_name=model_name,
        )
        if args.save_predictions:
            _write_prediction_bundle(bundle, paths.prediction_artifact_dir)
            print(f"Prediction bundle saved: {paths.prediction_artifact_dir}")

    if args.load_predictions_dir and args.save_predictions:
        _write_prediction_bundle(bundle, paths.prediction_artifact_dir)
        print(f"Prediction bundle copied to current run: {paths.prediction_artifact_dir}")

    evaluate_prediction_bundle(
        cfg,
        args,
        paths,
        run_store,
        bundle,
        model_name=model_name,
    )


if __name__ == "__main__":
    run_rolling_pipeline()
