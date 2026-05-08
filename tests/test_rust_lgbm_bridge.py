import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.rust_lgbm_bridge import _finite_feature_mask, train_lgbm_window_from_prepared_parquet


def test_finite_feature_mask_allows_nan_and_rejects_inf():
    frame = pd.DataFrame(
        {
            "f1": [1.0, np.nan, np.inf],
            "f2": [2.0, 3.0, 4.0],
        }
    )

    mask = _finite_feature_mask(frame, ["f1", "f2"])

    assert mask.tolist() == [True, True, False]


def test_train_window_helper_only_calls_native_lgbm_on_prepared_frames(tmp_path):
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    test_path = tmp_path / "test.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    importance_path = tmp_path / "feature_importance.csv"
    history_path = tmp_path / "training_history.csv"
    model_path = tmp_path / "model.pkl"
    base_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "instrument": ["000001.SZ", "000002.SZ"],
            "label": [0.01, 0.03],
            "backtest_label": [0.001, 0.002],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
        }
    )
    base_frame.to_parquet(train_path, index=False)
    base_frame.to_parquet(valid_path, index=False)
    base_frame.assign(f1=[np.nan, 5.0]).to_parquet(test_path, index=False)
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
        ):
            captured["X_train"] = X_train.copy()
            captured["y_train"] = pd.Series(y_train).reset_index(drop=True)
            captured["X_valid"] = X_valid.copy()
            captured["y_valid"] = pd.Series(y_valid).reset_index(drop=True)
            captured["valid_eval_labels"] = pd.Series(valid_eval_labels).reset_index(drop=True)
            captured["train_sample_weight"] = train_sample_weight
            captured["valid_sample_weight"] = valid_sample_weight
            return self

        def predict(self, X):
            captured.setdefault("predict_X", []).append(X.copy())
            return np.linspace(0.0, 1.0, len(X), dtype=np.float32)

        def save_feature_importance(self, save_path):
            path = Path(save_path)
            pd.DataFrame({"feature": ["f1", "f2"], "gain": [1.0, 2.0]}).to_csv(path, index=False)
            return path

        def get_feature_importance_frame(self, importance_type="gain"):
            return pd.DataFrame({"feature": ["f1", "f2"], importance_type: [1.0, 2.0]})

        def save_training_history(self, save_path):
            path = Path(save_path)
            pd.DataFrame({"iteration": [1], "valid": [0.1]}).to_csv(path, index=False)
            return path

        def get_training_summary(self):
            return {"num_iterations": 1}

    with patch("src.models.pure_lightgbm.NativeLGBM", FakeNativeLGBM):
        summary = train_lgbm_window_from_prepared_parquet(
            train_path=str(train_path),
            valid_path=str(valid_path),
            test_path=str(test_path),
            prediction_path=str(prediction_path),
            model_path=str(model_path),
            feature_importance_path=str(importance_path),
            training_history_path=str(history_path),
            lgbm_config_json=json.dumps({"loss": "huber", "validation_topk": 1}),
            window_metadata_json=json.dumps({"window_start": "2024-01-03"}),
            training_config_json=json.dumps(
                {
                    "label": {"train_transform": {"mode": "cross_section_rank"}},
                    "strategy": {"topk": 1},
                }
            ),
            feature_names=["f1", "f2"],
        )

    assert captured["config"]["loss"] == "huber"
    assert captured["y_train"].tolist() == [-0.5, 0.5]
    assert captured["valid_eval_labels"].tolist() == [0.01, 0.03]
    assert captured["train_sample_weight"] is None
    assert captured["predict_X"][0]["f1"].isna().iloc[0]
    assert summary["train_label_transform_mode"] == "cross_section_rank"
    assert summary["test_rows"] == 2
    assert pd.read_parquet(prediction_path).shape[0] == 2
