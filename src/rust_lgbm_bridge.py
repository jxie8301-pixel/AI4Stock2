"""Thin PyO3-facing LightGBM training bridge.

Rust owns runtime/profile resolution, factor-store reads, feature transforms,
rolling-window slicing, and prediction-bundle assembly.  Python remains here
only to call the existing NativeLGBM wrapper.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def train_lgbm_window_from_prepared_parquet(
    *,
    train_path: str,
    valid_path: str,
    test_path: str,
    prediction_path: str,
    model_path: str,
    feature_importance_path: str,
    training_history_path: str,
    lgbm_config_json: str,
    window_metadata_json: str,
    feature_names: list[str],
    training_config_json: str = "{}",
    save_model: bool = False,
    load_model: bool = False,
) -> dict[str, Any]:
    """Train/predict one pre-materialized LightGBM window.

    Rust owns factor-store reads, rolling-window slicing, label/backtest-label
    separation, feature transforms, training-label semantics, and final artifact
    assembly. Python only receives prepared window frames and calls LightGBM.
    """

    from src.models.pure_lightgbm import NativeLGBM

    lgbm_config = json.loads(lgbm_config_json)
    cfg = json.loads(training_config_json or "{}")
    window_metadata = json.loads(window_metadata_json)
    feature_names = list(feature_names)
    train_frame = pd.read_parquet(train_path)
    valid_frame = pd.read_parquet(valid_path)
    test_frame = pd.read_parquet(test_path)

    train_keep = _finite_training_mask(train_frame, feature_names)
    valid_keep = _finite_training_mask(valid_frame, feature_names)
    test_keep = _finite_feature_mask(test_frame, feature_names)
    if not train_keep.any():
        raise ValueError(f"no valid LightGBM training rows in prepared window: {train_path}")
    if not valid_keep.any():
        raise ValueError(f"no valid LightGBM validation rows in prepared window: {valid_path}")
    if not test_keep.any():
        raise ValueError(f"no valid LightGBM test rows in prepared window: {test_path}")

    train_use = train_frame.loc[train_keep].reset_index(drop=True)
    valid_use = valid_frame.loc[valid_keep].reset_index(drop=True)
    test_use = test_frame.loc[test_keep].reset_index(drop=True)

    train_dates = pd.to_datetime(train_use["datetime"]).reset_index(drop=True)
    valid_dates = pd.to_datetime(valid_use["datetime"]).reset_index(drop=True)
    train_symbols = train_use["instrument"].astype(str).reset_index(drop=True)
    valid_symbols = valid_use["instrument"].astype(str).reset_index(drop=True)
    raw_train_column = "raw_label" if "raw_label" in train_use.columns else "label"
    raw_valid_column = "raw_label" if "raw_label" in valid_use.columns else "label"
    raw_y_train_series = _build_label_series(train_use[raw_train_column], train_dates, train_symbols)
    raw_y_valid_series = _build_label_series(valid_use[raw_valid_column], valid_dates, valid_symbols)
    y_train_series = _build_label_series(train_use["label"], train_dates, train_symbols)
    y_valid_series = _build_label_series(valid_use["label"], valid_dates, valid_symbols)

    train_label_keep = np.isfinite(y_train_series.to_numpy(dtype=np.float64, copy=False))
    valid_label_keep = np.isfinite(y_valid_series.to_numpy(dtype=np.float64, copy=False))
    if not train_label_keep.any():
        raise ValueError(f"no effective LightGBM training rows after label transform: {train_path}")
    if not valid_label_keep.any():
        raise ValueError(f"no effective LightGBM validation rows after label transform: {valid_path}")

    train_use = train_use.loc[train_label_keep].reset_index(drop=True)
    valid_use = valid_use.loc[valid_label_keep].reset_index(drop=True)
    train_dates = train_dates.loc[train_label_keep].reset_index(drop=True)
    valid_dates = valid_dates.loc[valid_label_keep].reset_index(drop=True)
    raw_y_train_series = raw_y_train_series.iloc[train_label_keep]
    raw_y_valid_series = raw_y_valid_series.iloc[valid_label_keep]
    y_train_series = y_train_series.iloc[train_label_keep]
    y_valid_series = y_valid_series.iloc[valid_label_keep]

    X_train = train_use.loc[:, feature_names].reset_index(drop=True)
    X_valid = valid_use.loc[:, feature_names].reset_index(drop=True)
    X_test = test_use.loc[:, feature_names].reset_index(drop=True)

    train_sample_weight = _prepared_sample_weight(train_use)
    valid_sample_weight = _prepared_sample_weight(valid_use)

    model = NativeLGBM(**lgbm_config)
    model_path_obj = Path(model_path)
    loaded_model = False
    if load_model and model_path_obj.exists():
        with open(model_path_obj, "rb") as model_file:
            model = pickle.load(model_file)
        loaded_model = True
    else:
        model.fit(
            X_train,
            y_train_series,
            X_valid,
            y_valid_series,
            train_dates=train_dates,
            valid_dates=valid_dates,
            valid_eval_labels=raw_y_valid_series,
            train_sample_weight=train_sample_weight,
            valid_sample_weight=valid_sample_weight,
        )
        if save_model:
            model_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path_obj, "wb") as model_file:
                pickle.dump(model, model_file)

    predictions = model.predict(X_test)
    prediction_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(test_use["datetime"]).reset_index(drop=True),
            "instrument": test_use["instrument"].astype(str).reset_index(drop=True),
            "prediction": predictions,
        }
    )
    prediction_path_obj = Path(prediction_path)
    prediction_path_obj.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_parquet(prediction_path_obj, index=False)

    feature_importance_path_obj = Path(feature_importance_path)
    feature_importance_path_obj.parent.mkdir(parents=True, exist_ok=True)
    model.save_feature_importance(feature_importance_path_obj)
    training_history_path_obj = Path(training_history_path)
    training_history_path_obj.parent.mkdir(parents=True, exist_ok=True)
    saved_history_path = model.save_training_history(training_history_path_obj)

    importance_df = model.get_feature_importance_frame("gain")
    valid_opportunity_labels = _prepared_opportunity_labels(valid_use, valid_dates, valid_symbols)
    validation_topk = int(lgbm_config.get("validation_topk") or cfg.get("strategy", {}).get("topk") or 10)
    valid_topk_summary = _compute_validation_topk_summary(
        model.predict(X_valid),
        raw_y_valid_series,
        valid_dates,
        topk=validation_topk,
        opportunity_labels=valid_opportunity_labels,
    )
    train_label_transform_mode = str(
        window_metadata.get("train_label_transform_mode")
        or cfg.get("label", {}).get("train_transform", {}).get("mode")
        or "raw"
    )
    opportunity_mode = str(
        window_metadata.get("opportunity_label_mode")
        or cfg.get("label", {}).get("opportunity", {}).get("mode")
        or "positive"
    )
    opportunity_threshold = float(
        window_metadata.get(
            "opportunity_label_threshold",
            cfg.get("label", {}).get("opportunity", {}).get("threshold", 0.0),
        )
    )
    opportunity_neutral_band = float(
        window_metadata.get(
            "opportunity_label_neutral_band",
            cfg.get("label", {}).get("opportunity", {}).get("neutral_band", 0.0),
        )
    )
    sample_weight_mode = str(window_metadata.get("train_sample_weight_mode") or lgbm_config.get("sample_weight_mode", "none") or "none")
    summary = {
        **window_metadata,
        "train_rows": int(len(train_use)),
        "valid_rows": int(len(valid_use)),
        "test_rows": int(len(test_use)),
        "raw_train_rows": int(len(train_frame)),
        "raw_valid_rows": int(len(valid_frame)),
        "raw_test_rows": int(len(test_frame)),
        "train_rows_dropped_after_filter": int(len(train_frame) - len(train_use)),
        "valid_rows_dropped_after_filter": int(len(valid_frame) - len(valid_use)),
        "test_rows_dropped_after_filter": int(len(test_frame) - len(test_use)),
        "train_rows_dropped_after_label_transform": int(len(train_frame.loc[train_keep]) - len(train_use)),
        "valid_rows_dropped_after_label_transform": int(len(valid_frame.loc[valid_keep]) - len(valid_use)),
        "feature_count": int(len(feature_names)),
        "loaded_model": bool(loaded_model),
        "model_path": str(model_path_obj if save_model or loaded_model else ""),
        "prediction_path": str(prediction_path_obj),
        "feature_importance_path": str(feature_importance_path_obj),
        "training_history_path": str(saved_history_path or ""),
        "importance_gain_sum": float(pd.to_numeric(importance_df["gain"], errors="coerce").sum()),
        "train_label_transform_mode": train_label_transform_mode,
        "train_label_space": "binary_target"
        if train_label_transform_mode.startswith("buyability")
        else "return_target",
        "valid_custom_metric_label_space": "raw_return",
        "opportunity_label_mode": opportunity_mode,
        "opportunity_label_threshold": opportunity_threshold,
        "opportunity_label_neutral_band": opportunity_neutral_band,
        "train_sample_weight_mode": sample_weight_mode,
        **valid_topk_summary,
        **model.get_training_summary(),
    }
    return _json_safe(summary)


def _finite_feature_mask(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = frame.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
    return ~np.isinf(values).any(axis=1)


def _finite_training_mask(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return _finite_feature_mask(frame, feature_names) & np.isfinite(labels)


def _build_label_series(values: pd.Series, dates: pd.Series, symbols: pd.Series) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates).reset_index(drop=True), symbols.astype(str).reset_index(drop=True)],
        names=["datetime", "instrument"],
    )
    return pd.Series(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float), index=index, name="label")


def _prepared_sample_weight(frame: pd.DataFrame) -> pd.Series | None:
    if "sample_weight" not in frame.columns:
        return None
    values = pd.to_numeric(frame["sample_weight"], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=np.float64, copy=False))
    if not bool(finite.any()):
        return None
    return pd.Series(values.to_numpy(dtype=np.float64, copy=False), index=frame.index, name="sample_weight")


def _prepared_opportunity_labels(frame: pd.DataFrame, dates: pd.Series, symbols: pd.Series) -> pd.Series | None:
    if "opportunity_label" not in frame.columns:
        return None
    return _build_label_series(frame["opportunity_label"], dates, symbols).rename("opportunity_label")


def _compute_validation_topk_summary(
    predictions: np.ndarray,
    labels: pd.Series,
    dates: pd.Series,
    *,
    topk: int,
    opportunity_labels: pd.Series | None = None,
) -> dict[str, float | int]:
    pred_arr = np.asarray(predictions, dtype=np.float64)
    label_arr = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    date_arr = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]", copy=False)
    valid_mask = np.isfinite(pred_arr) & np.isfinite(label_arr) & ~np.isnat(date_arr)
    if not valid_mask.any():
        return _empty_validation_topk_summary()

    opportunity_arr = None
    if opportunity_labels is not None:
        opportunity_arr = pd.to_numeric(opportunity_labels, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if len(opportunity_arr) != len(pred_arr):
            raise ValueError("opportunity_labels must have the same length as predictions")
        opportunity_arr = opportunity_arr[valid_mask]

    pred_arr = pred_arr[valid_mask]
    label_arr = label_arr[valid_mask]
    date_arr = date_arr[valid_mask]
    order = np.argsort(date_arr, kind="stable")
    pred_sorted = pred_arr[order]
    label_sorted = label_arr[order]
    date_sorted = date_arr[order]
    opportunity_sorted = opportunity_arr[order] if opportunity_arr is not None else None
    topk = max(1, int(topk))
    daily_rows: list[dict[str, float]] = []
    boundaries = np.r_[0, np.flatnonzero(date_sorted[1:] != date_sorted[:-1]) + 1, len(date_sorted)]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        group_pred = pred_sorted[start:end]
        group_label = label_sorted[start:end]
        if group_pred.size == 0:
            continue
        selected_count = min(topk, group_pred.size)
        selected_pos = np.argsort(group_pred, kind="stable")[-selected_count:]
        top1_pos = selected_pos[np.argmax(group_pred[selected_pos])]
        row = {
            "top1_label": float(group_label[top1_pos]),
            "top1_positive": float(group_label[top1_pos] > 0.0),
            "topk_label_mean": float(group_label[selected_pos].mean()),
            "topk_label_median": float(np.median(group_label[selected_pos])),
            "topk_min_label": float(group_label[selected_pos].min()),
            "topk_positive_rate": float((group_label[selected_pos] > 0.0).mean()),
            "topk_excess_mean": float(group_label[selected_pos].mean() - group_label.mean()),
            "top1_opportunity": np.nan,
            "topk_opportunity_rate": np.nan,
        }
        if opportunity_sorted is not None:
            group_opportunity = opportunity_sorted[start:end]
            selected_opportunity = group_opportunity[selected_pos]
            row["top1_opportunity"] = float(group_opportunity[top1_pos])
            row["topk_opportunity_rate"] = (
                float(np.nanmean(selected_opportunity))
                if not np.isnan(selected_opportunity).all()
                else float("nan")
            )
        daily_rows.append(row)
    if not daily_rows:
        return _empty_validation_topk_summary()
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
        "valid_top1_opportunity_rate": float(daily["top1_opportunity"].mean()),
        "valid_topk_opportunity_rate": float(daily["topk_opportunity_rate"].mean()),
    }


def _empty_validation_topk_summary() -> dict[str, float | int]:
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
