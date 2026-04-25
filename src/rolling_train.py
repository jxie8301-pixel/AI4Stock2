"""Training and prediction generation for the native rolling pipeline."""

from __future__ import annotations

import argparse
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.experiment_store import resolve_rebalance_freq
from src.feature_selection import apply_feature_transforms
from src.industry_groups import load_instrument_industry_groups
from src.label_utils import (
    build_opportunity_target_series,
    compute_opportunity_sample_weights,
    resolve_opportunity_label_cfg,
    resolve_train_label_transform_cfg,
    transform_training_label_series,
)
from src.model_config import get_lgbm_config
from src.evaluate import build_benchmark_series
from src.rolling_baselines import (
    build_average_factor_baseline_predictions,
    build_rank_average_factor_baseline_predictions,
    build_rank_ic_weighted_factor_baseline_predictions,
    build_sign_aligned_factor_baseline_predictions,
)
from src.return_horizon import build_forward_compound_return_series
from src.rolling_runtime import build_label_series, build_prediction_metadata
from src.rolling_types import PredictionBundle, RollingPaths, RollingRuntimeData


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


def _compute_validation_topk_summary(
    predictions: np.ndarray | pd.Series,
    labels: pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    topk: int,
    opportunity_labels: pd.Series | None = None,
) -> dict[str, float | int]:
    topk = max(1, int(topk))
    frame = pd.DataFrame(
        {
            "prediction": np.asarray(predictions, dtype=np.float32),
            "label": pd.to_numeric(labels, errors="coerce").to_numpy(dtype=np.float32, copy=False),
            "date": pd.to_datetime(pd.Series(dates)).to_numpy(),
        }
    ).dropna()
    if opportunity_labels is not None:
        frame["opportunity_label"] = pd.to_numeric(opportunity_labels, errors="coerce").to_numpy(dtype=np.float32, copy=False)
    if frame.empty:
        return {
            "valid_topk_days": 0,
            "valid_top1_label_mean": float("nan"),
            "valid_top1_positive_rate": float("nan"),
            "valid_topk_label_mean": float("nan"),
            "valid_topk_label_median": float("nan"),
            "valid_topk_min_label_mean": float("nan"),
            "valid_topk_positive_rate": float("nan"),
            "valid_topk_excess_mean": float("nan"),
            "valid_top1_opportunity_rate": float("nan"),
            "valid_topk_opportunity_rate": float("nan"),
        }

    daily_rows: list[dict[str, float]] = []
    for _, group in frame.groupby("date", sort=True):
        ranked = group.sort_values("prediction", ascending=False, kind="stable")
        selected = ranked.head(topk)
        if selected.empty:
            continue
        selected_labels = selected["label"].to_numpy(dtype=np.float64, copy=False)
        daily_rows.append(
            {
                "top1_label": float(selected_labels[0]),
                "top1_positive": float(selected_labels[0] > 0.0),
                "topk_label_mean": float(selected_labels.mean()),
                "topk_label_median": float(np.median(selected_labels)),
                "topk_min_label": float(selected_labels.min()),
                "topk_positive_rate": float((selected_labels > 0.0).mean()),
                "topk_excess_mean": float(selected_labels.mean() - group["label"].mean()),
                "top1_opportunity": float(selected["opportunity_label"].iloc[0]) if "opportunity_label" in selected.columns else np.nan,
                "topk_opportunity_rate": float(selected["opportunity_label"].mean()) if "opportunity_label" in selected.columns else np.nan,
            }
        )

    if not daily_rows:
        return {
            "valid_topk_days": 0,
            "valid_top1_label_mean": float("nan"),
            "valid_top1_positive_rate": float("nan"),
            "valid_topk_label_mean": float("nan"),
            "valid_topk_label_median": float("nan"),
            "valid_topk_min_label_mean": float("nan"),
            "valid_topk_positive_rate": float("nan"),
            "valid_topk_excess_mean": float("nan"),
            "valid_top1_opportunity_rate": float("nan"),
            "valid_topk_opportunity_rate": float("nan"),
        }

    daily = pd.DataFrame(daily_rows)
    return {
        "valid_topk_days": int(len(daily)),
        "valid_top1_label_mean": float(daily["top1_label"].mean()),
        "valid_top1_positive_rate": float(daily["top1_positive"].mean()),
        "valid_topk_label_mean": float(daily["topk_label_mean"].mean()),
        "valid_topk_label_median": float(daily["topk_label_median"].mean()),
        "valid_topk_min_label_mean": float(daily["topk_min_label"].mean()),
        "valid_topk_positive_rate": float(daily["topk_positive_rate"].mean()),
        "valid_topk_excess_mean": float(daily["topk_excess_mean"].mean()),
        "valid_top1_opportunity_rate": float(daily["top1_opportunity"].mean()) if "top1_opportunity" in daily.columns else float("nan"),
        "valid_topk_opportunity_rate": float(daily["topk_opportunity_rate"].mean()) if "topk_opportunity_rate" in daily.columns else float("nan"),
    }


def _build_buyability_sample_weight_series(
    labels: pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    lgbm_config: dict[str, Any],
    opportunity_cfg: dict[str, Any],
    opportunity_instrument_groups: pd.Series | None = None,
    benchmark_forward_returns: pd.Series | None = None,
) -> pd.Series | None:
    sample_weight_mode = str(lgbm_config.get("sample_weight_mode", "none") or "none").strip().lower()
    if sample_weight_mode == "none":
        return None
    return compute_opportunity_sample_weights(
        labels,
        dates,
        opportunity_cfg=opportunity_cfg,
        instrument_groups=opportunity_instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
        sample_weight_mode=sample_weight_mode,
        sample_weight_power=float(lgbm_config.get("sample_weight_power", 1.0)),
        sample_weight_scale=lgbm_config.get("sample_weight_scale"),
        sample_weight_min=float(lgbm_config.get("sample_weight_min", 0.0)),
        sample_weight_date_normalize=bool(lgbm_config.get("sample_weight_date_normalize", False)),
    )


def _build_full_runtime_label_series(runtime_data: RollingRuntimeData) -> pd.Series:
    dates = pd.to_datetime(runtime_data.dt_index).reset_index(drop=True)
    symbols = runtime_data.factor_frame["symbol"].astype(str).reset_index(drop=True)
    index = pd.MultiIndex.from_arrays([dates, symbols], names=["datetime", "instrument"])
    return pd.Series(runtime_data.y, index=index, name="label", dtype=float).sort_index()


def _build_window_label_series(
    values: np.ndarray,
    dates: pd.Series,
    symbols: pd.Series,
) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates).reset_index(drop=True), symbols.astype(str).reset_index(drop=True)],
        names=["datetime", "instrument"],
    )
    return pd.Series(values, index=index, name="label", dtype=float)


def _prepare_opportunity_training_context(
    cfg: dict[str, Any],
    runtime_data: RollingRuntimeData,
    *,
    signal_horizon: int,
) -> dict[str, Any]:
    opportunity_cfg = resolve_opportunity_label_cfg(cfg)
    context: dict[str, Any] = {
        "instrument_groups": None,
        "benchmark_forward_returns": None,
    }
    mode = str(opportunity_cfg["mode"])
    if mode == "industry_excess":
        context["instrument_groups"] = load_instrument_industry_groups(
            cfg,
            instruments=runtime_data.factor_frame["symbol"].astype(str).drop_duplicates(),
        )
    elif mode == "benchmark_excess":
        runtime_labels = _build_full_runtime_label_series(runtime_data)
        benchmark_series, _ = build_benchmark_series(runtime_labels, cfg.get("backtest", {}).get("benchmark"))
        context["benchmark_forward_returns"] = build_forward_compound_return_series(
            benchmark_series,
            horizon=signal_horizon,
        )
    return context


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
    opportunity_instrument_groups: pd.Series | None = None,
    benchmark_forward_returns: pd.Series | None = None,
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
    train_dates = pd.to_datetime(runtime_data.dt_index[valid_train_mask]).reset_index(drop=True)
    train_symbols = runtime_data.factor_frame.loc[valid_train_mask, "symbol"].astype(str).reset_index(drop=True)
    raw_y_train_series = _build_window_label_series(runtime_data.y[valid_train_mask], train_dates, train_symbols)
    y_train_series = raw_y_train_series.copy()
    X_valid_df = runtime_data.factor_frame.loc[valid_valid_mask, feature_names].reset_index(drop=True)
    valid_dates = pd.to_datetime(runtime_data.dt_index[valid_valid_mask]).reset_index(drop=True)
    valid_symbols = runtime_data.factor_frame.loc[valid_valid_mask, "symbol"].astype(str).reset_index(drop=True)
    raw_y_valid_series = _build_window_label_series(runtime_data.y[valid_valid_mask], valid_dates, valid_symbols)
    y_valid_series = raw_y_valid_series.copy()
    y_train_series = transform_training_label_series(
        y_train_series,
        train_dates,
        cfg,
        instrument_groups=opportunity_instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
    y_valid_series = transform_training_label_series(
        y_valid_series,
        valid_dates,
        cfg,
        instrument_groups=opportunity_instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
    train_label_transform = resolve_train_label_transform_cfg(cfg)
    opportunity_cfg = resolve_opportunity_label_cfg(cfg)
    raw_train_rows = int(len(X_train_df))
    raw_valid_rows = int(len(X_valid_df))

    train_keep_mask = np.isfinite(y_train_series.to_numpy(dtype=np.float64, copy=False))
    valid_keep_mask = np.isfinite(y_valid_series.to_numpy(dtype=np.float64, copy=False))
    if not train_keep_mask.any():
        print("    Skipping window: no effective LightGBM training rows after label transform.")
        return None, None, None
    if not valid_keep_mask.any():
        print("    Skipping window: no effective LightGBM validation rows after label transform.")
        return None, None, None

    if not train_keep_mask.all():
        X_train_df = X_train_df.loc[train_keep_mask].reset_index(drop=True)
        raw_y_train_series = raw_y_train_series.iloc[train_keep_mask]
        y_train_series = y_train_series.iloc[train_keep_mask]
        train_dates = train_dates.loc[train_keep_mask].reset_index(drop=True)
    if not valid_keep_mask.all():
        X_valid_df = X_valid_df.loc[valid_keep_mask].reset_index(drop=True)
        raw_y_valid_series = raw_y_valid_series.iloc[valid_keep_mask]
        y_valid_series = y_valid_series.iloc[valid_keep_mask]
        valid_dates = valid_dates.loc[valid_keep_mask].reset_index(drop=True)

    X_train_df = apply_feature_transforms(X_train_df, train_dates, cfg)
    X_valid_df = apply_feature_transforms(X_valid_df, valid_dates, cfg)

    lgbm_config = get_lgbm_config(cfg)
    train_sample_weight = _build_buyability_sample_weight_series(
        raw_y_train_series,
        train_dates,
        lgbm_config=lgbm_config,
        opportunity_cfg=opportunity_cfg,
        opportunity_instrument_groups=opportunity_instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
    valid_sample_weight = _build_buyability_sample_weight_series(
        raw_y_valid_series,
        valid_dates,
        lgbm_config=lgbm_config,
        opportunity_cfg=opportunity_cfg,
        opportunity_instrument_groups=opportunity_instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
    model_path = paths.models_dir / f"model_{current_test_start.strftime('%Y-%m-%d')}.pkl"
    model = NativeLGBM(**lgbm_config)
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
            valid_eval_labels=raw_y_valid_series,
            train_sample_weight=train_sample_weight,
            valid_sample_weight=valid_sample_weight,
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
    valid_opportunity_labels: pd.Series | None = None
    try:
        valid_opportunity_labels = build_opportunity_target_series(
            raw_y_valid_series,
            opportunity_cfg=opportunity_cfg,
            instrument_groups=opportunity_instrument_groups,
            benchmark_forward_returns=benchmark_forward_returns,
        )
    except Exception:
        valid_opportunity_labels = None
    valid_topk_summary = _compute_validation_topk_summary(
        model.predict(X_valid_df),
        raw_y_valid_series,
        valid_dates,
        topk=int(cfg["strategy"]["topk"]),
        opportunity_labels=valid_opportunity_labels,
    )
    training_summary = {
        "window_start": current_test_start.strftime("%Y-%m-%d"),
        "window_end": current_test_end.strftime("%Y-%m-%d"),
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": train_end.strftime("%Y-%m-%d"),
        "valid_start": valid_start.strftime("%Y-%m-%d"),
        "valid_end": valid_end.strftime("%Y-%m-%d"),
        "signal_horizon": int(signal_horizon),
        "train_label_transform_mode": str(train_label_transform["mode"]),
        "train_label_space": "binary_target" if str(train_label_transform["mode"]).startswith("buyability") else "return_target",
        "valid_custom_metric_label_space": "raw_return",
        "opportunity_label_mode": str(opportunity_cfg["mode"]),
        "opportunity_label_threshold": float(opportunity_cfg["threshold"]),
        "opportunity_label_neutral_band": float(opportunity_cfg["neutral_band"]),
        "train_label_transform_neutral_band": float(train_label_transform["neutral_band"]),
        "train_label_transform_tail_band": float(train_label_transform["tail_band"]),
        "train_label_transform_scale_multiplier": float(train_label_transform["scale_multiplier"]),
        "raw_train_rows": raw_train_rows,
        "raw_valid_rows": raw_valid_rows,
        "train_rows_dropped_after_label_transform": int(raw_train_rows - len(X_train_df)),
        "valid_rows_dropped_after_label_transform": int(raw_valid_rows - len(X_valid_df)),
        "train_rows": int(len(X_train_df)),
        "valid_rows": int(len(X_valid_df)),
        "feature_count": int(len(feature_names)),
        "loaded_model": bool(loaded_model),
        "training_history_path": str(saved_history_path) if saved_history_path else "",
        "train_sample_weight_mode": str(lgbm_config.get("sample_weight_mode", "none") or "none"),
        **valid_topk_summary,
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
    seq_train_labels = transform_training_label_series(seq_frame["label"], seq_dt_index, cfg)
    y_seq = seq_train_labels.to_numpy(dtype=np.float32, copy=True)
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
    opportunity_context = _prepare_opportunity_training_context(
        cfg,
        runtime_data,
        signal_horizon=signal_horizon,
    )

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
                opportunity_instrument_groups=opportunity_context.get("instrument_groups"),
                benchmark_forward_returns=opportunity_context.get("benchmark_forward_returns"),
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
    label_series, backtest_label_series = build_label_series(runtime_data)
    avg_factor_baseline_predictions = build_average_factor_baseline_predictions(runtime_data)
    sign_aligned_factor_baseline_predictions = build_sign_aligned_factor_baseline_predictions(runtime_data)
    rank_avg_factor_baseline_predictions = build_rank_average_factor_baseline_predictions(runtime_data)
    rank_ic_weighted_factor_baseline_predictions = build_rank_ic_weighted_factor_baseline_predictions(runtime_data)
    return PredictionBundle(
        final_predictions=final_predictions,
        label_series=label_series,
        backtest_label_series=backtest_label_series,
        avg_factor_baseline_predictions=avg_factor_baseline_predictions,
        sign_aligned_factor_baseline_predictions=sign_aligned_factor_baseline_predictions,
        selected_feature_names=list(runtime_data.selected_feature_names),
        metadata=build_prediction_metadata(
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
        rank_avg_factor_baseline_predictions=rank_avg_factor_baseline_predictions,
        rank_ic_weighted_factor_baseline_predictions=rank_ic_weighted_factor_baseline_predictions,
    )
