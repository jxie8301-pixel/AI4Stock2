"""Thin PyO3-facing LightGBM training bridge.

Rust owns runtime/profile resolution, factor-store reads, feature transforms,
rolling-window slicing, prepared-window filtering, and prediction-bundle
assembly. Python remains here only to materialize in-memory arrays into the
existing `NativeLGBM` wrapper and call LightGBM.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


def train_lgbm_window_from_prepared_arrays(
    *,
    train_feature_bytes: bytes,
    valid_feature_bytes: bytes,
    test_feature_bytes: bytes,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    train_label_bytes: bytes,
    valid_label_bytes: bytes,
    raw_valid_label_bytes: bytes,
    train_date_ns_bytes: bytes,
    valid_date_ns_bytes: bytes,
    feature_names: list[str],
    model_path: str,
    feature_importance_path: str,
    training_history_path: str,
    lgbm_config_json: str,
    train_sample_weight_bytes: bytes | None = None,
    valid_sample_weight_bytes: bytes | None = None,
    save_model: bool = False,
    load_model: bool = False,
) -> dict[str, Any]:
    """Train/predict one pre-materialized LightGBM window.

    Rust already applies window slicing, label semantics, finite-row filtering,
    and artifact-oriented output assembly. Python only converts raw bytes into
    numpy arrays and calls the existing LightGBM wrapper.
    """

    bridge_started = perf_counter()
    import_started = perf_counter()
    from src.models.pure_lightgbm import NativeLGBM

    import_native_lgbm_seconds = perf_counter() - import_started
    lgbm_config = json.loads(lgbm_config_json)
    feature_names = list(feature_names)

    materialize_started = perf_counter()
    X_train = _decode_feature_matrix(
        train_feature_bytes,
        num_rows=int(train_rows),
        feature_names=feature_names,
        name="train_feature_bytes",
    )
    X_valid = _decode_feature_matrix(
        valid_feature_bytes,
        num_rows=int(valid_rows),
        feature_names=feature_names,
        name="valid_feature_bytes",
    )
    X_test = _decode_feature_matrix(
        test_feature_bytes,
        num_rows=int(test_rows),
        feature_names=feature_names,
        name="test_feature_bytes",
    )
    y_train_values = _decode_f64_vector(
        train_label_bytes,
        expected_len=int(train_rows),
        name="train_label_bytes",
    )
    y_valid_values = _decode_f64_vector(
        valid_label_bytes,
        expected_len=int(valid_rows),
        name="valid_label_bytes",
    )
    raw_y_valid_values = _decode_f64_vector(
        raw_valid_label_bytes,
        expected_len=int(valid_rows),
        name="raw_valid_label_bytes",
    )
    train_dates = _decode_i64_vector(
        train_date_ns_bytes,
        expected_len=int(train_rows),
        name="train_date_ns_bytes",
    ).astype("datetime64[ns]")
    valid_dates = _decode_i64_vector(
        valid_date_ns_bytes,
        expected_len=int(valid_rows),
        name="valid_date_ns_bytes",
    ).astype("datetime64[ns]")
    train_sample_weight = _decode_optional_f64_vector(
        train_sample_weight_bytes,
        expected_len=int(train_rows),
        name="train_sample_weight_bytes",
    )
    valid_sample_weight = _decode_optional_f64_vector(
        valid_sample_weight_bytes,
        expected_len=int(valid_rows),
        name="valid_sample_weight_bytes",
    )
    materialize_inputs_seconds = perf_counter() - materialize_started

    if len(X_train) == 0:
        raise ValueError("no valid LightGBM training rows in prepared Rust window")
    if len(X_valid) == 0:
        raise ValueError("no valid LightGBM validation rows in prepared Rust window")
    if len(X_test) == 0:
        raise ValueError("no valid LightGBM test rows in prepared Rust window")

    model = NativeLGBM(**lgbm_config)
    model_path_obj = Path(model_path)
    loaded_model = False
    if load_model and model_path_obj.exists():
        load_started = perf_counter()
        with open(model_path_obj, "rb") as model_file:
            model = pickle.load(model_file)
        loaded_model = True
        model_load_seconds = perf_counter() - load_started
        fit_seconds = 0.0
        model_save_seconds = 0.0
    else:
        fit_started = perf_counter()
        model.fit(
            X_train,
            y_train_values,
            X_valid,
            y_valid_values,
            train_dates=train_dates,
            valid_dates=valid_dates,
            valid_eval_labels=raw_y_valid_values,
            train_sample_weight=train_sample_weight,
            valid_sample_weight=valid_sample_weight,
            feature_names=feature_names,
        )
        fit_seconds = perf_counter() - fit_started
        model_load_seconds = 0.0
        if save_model:
            save_model_started = perf_counter()
            model_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path_obj, "wb") as model_file:
                pickle.dump(model, model_file)
            model_save_seconds = perf_counter() - save_model_started
        else:
            model_save_seconds = 0.0

    importance_started = perf_counter()
    feature_importance_path_obj = Path(feature_importance_path)
    feature_importance_path_obj.parent.mkdir(parents=True, exist_ok=True)
    model.save_feature_importance(feature_importance_path_obj)
    feature_importance_seconds = perf_counter() - importance_started

    history_started = perf_counter()
    training_history_path_obj = Path(training_history_path)
    training_history_path_obj.parent.mkdir(parents=True, exist_ok=True)
    saved_history_path = model.save_training_history(training_history_path_obj)
    training_history_seconds = perf_counter() - history_started

    summary_started = perf_counter()
    importance_gain_sum = float(
        np.nansum(np.asarray(model.get_feature_importance(importance_type="gain"), dtype=np.float64))
    )
    predict_valid_started = perf_counter()
    valid_predictions = model.predict(X_valid)
    predict_valid_seconds = perf_counter() - predict_valid_started
    predict_test_started = perf_counter()
    test_predictions = model.predict(X_test)
    predict_test_seconds = perf_counter() - predict_test_started
    summary = {
        "loaded_model": bool(loaded_model),
        "model_path": str(model_path_obj if save_model or loaded_model else ""),
        "feature_importance_path": str(feature_importance_path_obj),
        "training_history_path": str(saved_history_path or ""),
        "importance_gain_sum": importance_gain_sum,
        "python_import_native_lgbm_seconds": import_native_lgbm_seconds,
        "python_materialize_inputs_seconds": materialize_inputs_seconds,
        "python_model_load_seconds": model_load_seconds,
        "python_fit_seconds": fit_seconds,
        "python_model_save_seconds": model_save_seconds,
        "python_feature_importance_seconds": feature_importance_seconds,
        "python_training_history_seconds": training_history_seconds,
        "python_predict_valid_seconds": predict_valid_seconds,
        "python_predict_test_seconds": predict_test_seconds,
        **model.get_training_summary(),
    }
    summary["python_summary_build_seconds"] = perf_counter() - summary_started
    summary["python_bridge_total_seconds"] = perf_counter() - bridge_started
    return _json_safe(
        {
            "valid_prediction_bytes": np.asarray(valid_predictions, dtype=np.float64).tobytes(),
            "test_prediction_bytes": np.asarray(test_predictions, dtype=np.float64).tobytes(),
            "summary": summary,
        }
    )


def _decode_feature_matrix(
    payload: bytes,
    *,
    num_rows: int,
    feature_names: list[str],
    name: str,
) -> np.ndarray:
    feature_count = len(feature_names)
    expected_len = int(num_rows) * int(feature_count)
    values = _decode_f64_vector(payload, expected_len=expected_len, name=name)
    return values.reshape((int(num_rows), int(feature_count)))


def _decode_f64_vector(payload: bytes, *, expected_len: int, name: str) -> np.ndarray:
    values = np.frombuffer(payload, dtype="<f8")
    if len(values) != int(expected_len):
        raise ValueError(
            f"{name} length mismatch: expected {expected_len} float64 values, got {len(values)}"
        )
    return values


def _decode_i64_vector(payload: bytes, *, expected_len: int, name: str) -> np.ndarray:
    values = np.frombuffer(payload, dtype="<i8")
    if len(values) != int(expected_len):
        raise ValueError(
            f"{name} length mismatch: expected {expected_len} int64 values, got {len(values)}"
        )
    return values


def _decode_optional_f64_vector(
    payload: bytes | None,
    *,
    expected_len: int,
    name: str,
) -> np.ndarray | None:
    if payload is None:
        return None
    values = _decode_f64_vector(payload, expected_len=expected_len, name=name)
    if not np.isfinite(values).any():
        return None
    return values


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
