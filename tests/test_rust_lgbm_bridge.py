import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.rust_lgbm_bridge import _json_safe, train_lgbm_window_from_prepared_arrays


def _f64_bytes(values) -> bytes:
    return np.asarray(values, dtype=np.float64).tobytes()


def _i64_bytes(values) -> bytes:
    return np.asarray(values, dtype=np.int64).tobytes()


def test_train_window_helper_only_calls_native_lgbm_on_prepared_arrays(tmp_path):
    importance_path = tmp_path / "feature_importance.csv"
    history_path = tmp_path / "training_history.csv"
    model_path = tmp_path / "model.pkl"
    captured = {}

    class FakeNativeLGBM:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

        def fit(
            self,
            X_train,
            y_train,
            X_valid=None,
            y_valid=None,
            *,
            train_dates=None,
            valid_dates=None,
            valid_eval_labels=None,
            train_sample_weight=None,
            valid_sample_weight=None,
            feature_names=None,
        ):
            captured["X_train"] = np.array(X_train, copy=True)
            captured["y_train"] = np.array(y_train, copy=True)
            captured["X_valid"] = np.array(X_valid, copy=True)
            captured["y_valid"] = np.array(y_valid, copy=True)
            captured["train_dates"] = np.array(train_dates, copy=True)
            captured["valid_dates"] = np.array(valid_dates, copy=True)
            captured["valid_eval_labels"] = np.array(valid_eval_labels, copy=True)
            captured["train_sample_weight"] = train_sample_weight
            captured["valid_sample_weight"] = valid_sample_weight
            captured["feature_names"] = list(feature_names or [])
            return self

        def predict(self, X):
            captured.setdefault("predict_X", []).append(np.array(X, copy=True))
            return np.linspace(0.0, 1.0, len(X), dtype=np.float32)

        def save_feature_importance(self, save_path):
            path = Path(save_path)
            path.write_text("feature,gain\nf1,1.0\nf2,2.0\n", encoding="utf-8")
            return path

        def get_feature_importance(self, importance_type="gain"):
            assert importance_type == "gain"
            return np.array([1.0, 2.0], dtype=np.float64)

        def save_training_history(self, save_path):
            path = Path(save_path)
            path.write_text("iteration,valid\n1,0.1\n", encoding="utf-8")
            return path

        def get_training_summary(self):
            return {"num_iterations": 1}

    with patch("src.models.pure_lightgbm.NativeLGBM", FakeNativeLGBM):
        summary = train_lgbm_window_from_prepared_arrays(
            train_feature_bytes=_f64_bytes([[1.0, 3.0], [2.0, 4.0]]),
            valid_feature_bytes=_f64_bytes([[1.5, 3.5], [2.5, 4.5]]),
            test_feature_bytes=_f64_bytes([[np.nan, 7.0], [5.0, 8.0]]),
            train_rows=2,
            valid_rows=2,
            test_rows=2,
            train_label_bytes=_f64_bytes([-0.5, 0.5]),
            valid_label_bytes=_f64_bytes([-0.25, 0.25]),
            raw_valid_label_bytes=_f64_bytes([0.01, 0.03]),
            train_date_ns_bytes=_i64_bytes(
                [
                    np.datetime64("2024-01-02", "ns").astype(np.int64),
                    np.datetime64("2024-01-02", "ns").astype(np.int64),
                ]
            ),
            valid_date_ns_bytes=_i64_bytes(
                [
                    np.datetime64("2024-01-03", "ns").astype(np.int64),
                    np.datetime64("2024-01-03", "ns").astype(np.int64),
                ]
            ),
            train_sample_weight_bytes=None,
            valid_sample_weight_bytes=None,
            model_path=str(model_path),
            feature_importance_path=str(importance_path),
            training_history_path=str(history_path),
            lgbm_config_json=json.dumps({"loss": "huber", "validation_topk": 1}),
            feature_names=["f1", "f2"],
        )

    assert captured["config"]["loss"] == "huber"
    assert captured["feature_names"] == ["f1", "f2"]
    assert captured["X_train"].tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert captured["y_train"].tolist() == [-0.5, 0.5]
    assert captured["valid_eval_labels"].tolist() == [0.01, 0.03]
    assert captured["train_sample_weight"] is None
    assert captured["valid_sample_weight"] is None
    assert str(captured["train_dates"][0]) == "2024-01-02T00:00:00.000000000"
    assert str(captured["valid_dates"][0]) == "2024-01-03T00:00:00.000000000"
    assert not np.isnan(captured["predict_X"][0][:, 0]).any()
    assert np.isnan(captured["predict_X"][1][0, 0])
    assert np.frombuffer(summary["valid_prediction_bytes"], dtype=np.float64).tolist() == [0.0, 1.0]
    assert np.frombuffer(summary["test_prediction_bytes"], dtype=np.float64).tolist() == [0.0, 1.0]
    assert summary["summary"]["feature_importance_path"] == str(importance_path)
    assert summary["summary"]["training_history_path"] == str(history_path)
    assert summary["summary"]["importance_gain_sum"] == 3.0
    assert summary["summary"]["python_import_native_lgbm_seconds"] >= 0.0
    assert summary["summary"]["python_materialize_inputs_seconds"] >= 0.0
    assert summary["summary"]["python_fit_seconds"] >= 0.0
    assert (
        summary["summary"]["python_bridge_total_seconds"]
        >= summary["summary"]["python_fit_seconds"]
    )


def test_json_safe_replaces_nan_with_none():
    payload = {"a": float("nan"), "b": [1.0, np.float64("nan")], "c": {"d": float("inf")}}

    out = _json_safe(payload)

    assert out == {"a": None, "b": [1.0, None], "c": {"d": None}}
