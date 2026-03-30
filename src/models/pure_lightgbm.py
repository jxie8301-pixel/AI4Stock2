"""Pure LightGBM Native Implementation."""

import inspect
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate import safe_cross_sectional_corr


def _daily_ic_metric(preds: np.ndarray, dataset: lgb.Dataset, dates: np.ndarray):
    labels = dataset.get_label()
    frame = pd.DataFrame(
        {
            "pred": np.asarray(preds, dtype=np.float32),
            "label": np.asarray(labels, dtype=np.float32),
            "date": pd.to_datetime(np.asarray(dates)),
        }
    ).dropna()
    if frame.empty:
        return "daily_ic", 0.0, True

    daily_ic = frame.groupby("date", sort=True).apply(
        lambda x: safe_cross_sectional_corr(x["pred"], x["label"], method="pearson"),
        include_groups=False,
    ).dropna()
    if daily_ic.empty:
        return "daily_ic", 0.0, True
    return "daily_ic", float(daily_ic.mean()), True


class NativeLGBM:
    """Native LightGBM wrapper mimicking the interface expected by our pipeline."""

    def __init__(
        self,
        loss: str = "huber",
        colsample_bytree: float = 0.8879,
        learning_rate: float = 0.05,
        subsample: float = 0.8789,
        subsample_freq: int = 1,
        lambda_l1: float = 1.0,
        lambda_l2: float = 1.0,
        max_depth: int = 8,
        num_leaves: int = 63,
        min_data_in_leaf: int = 100,
        num_threads: int = 20,
        early_stop: int = 50,
        num_boost_round: int = 1000,
        early_stopping_min_delta: float = 0.0,
        log_evaluation_period: int = 50,
        alpha: float = 0.9,
        seed: int = 42,
        **kwargs
    ):
        # Map generic loss names to lightgbm specific objectives
        objective = "regression"
        eval_metric = kwargs.get("eval_metric")
        if loss in {"mse", "regression", "l2"}:
            objective = "regression"
            default_metric = "l2"
        elif loss in {"mae", "regression_l1", "l1"}:
            objective = "regression_l1"
            default_metric = "l1"
        elif loss == "huber":
            objective = "huber"
            default_metric = "rmse"
        else:
            default_metric = "l2"

        self.eval_metric = eval_metric or default_metric
        self.params = {
            "objective": objective,
            # Use a stable regression metric for early stopping. Daily IC remains
            # as an auxiliary validation metric, but it is too noisy for 10-day
            # windows and can be undefined when predictions collapse.
            "metric": self.eval_metric,
            "colsample_bytree": colsample_bytree,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "num_threads": num_threads,
            "verbosity": -1,
            "seed": seed,
        }
        if objective == "huber":
            self.params["alpha"] = alpha
        self.early_stop = early_stop
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        self.log_evaluation_period = int(log_evaluation_period)
        self.model = None
        self.evals_result_: dict[str, dict[str, list[float]]] = {}
        self.best_iteration_: int | None = None
        self.best_score_: dict[str, dict[str, float]] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
        valid_dates: np.ndarray | None = None,
    ):
        """Fit the LightGBM model."""
        feature_names = X_train.columns.tolist()
        dtrain = lgb.Dataset(X_train.values, label=y_train.values, feature_name=feature_names)
        
        valid_sets = [dtrain]
        valid_names = ["train"]
        feval = None
        
        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(X_valid.values, label=y_valid.values, reference=dtrain, feature_name=feature_names)
            valid_sets = [dvalid]
            valid_names = ["valid"]
            if valid_dates is not None:
                feval = lambda preds, data: _daily_ic_metric(preds, data, valid_dates)
            
        evals_result: dict[str, dict[str, list[float]]] = {}
        callbacks = []
        callbacks.append(lgb.record_evaluation(evals_result))
        if self.early_stop > 0 and X_valid is not None:
            early_stopping_kwargs: dict[str, Any] = {
                "stopping_rounds": self.early_stop,
                "first_metric_only": True,
                "verbose": True,
            }
            if "min_delta" in inspect.signature(lgb.early_stopping).parameters:
                early_stopping_kwargs["min_delta"] = self.early_stopping_min_delta
            callbacks.append(
                lgb.early_stopping(**early_stopping_kwargs)
            )

        callbacks.append(lgb.log_evaluation(period=max(1, self.log_evaluation_period)))

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=max(1, self.num_boost_round),
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval,
            callbacks=callbacks
        )
        self.evals_result_ = evals_result
        self.best_score_ = {
            dataset_name: {metric_name: float(value) for metric_name, value in metrics.items()}
            for dataset_name, metrics in (getattr(self.model, "best_score", {}) or {}).items()
        }
        best_iteration = int(getattr(self.model, "best_iteration", 0) or 0)
        if best_iteration > 0:
            self.best_iteration_ = best_iteration
        else:
            self.best_iteration_ = int(getattr(self.model, "current_iteration", lambda: 0)() or 0) or None
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict using the trained model."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X_test.values)
        
    def get_feature_importance(self, importance_type="split"):
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.feature_importance(importance_type=importance_type)

    def get_feature_importance_frame(self, importance_type: str = "gain") -> pd.DataFrame:
        """Return feature importance as a sorted dataframe."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        feature_names = self.model.feature_name()
        importance = self.model.feature_importance(importance_type=importance_type)
        return (
            pd.DataFrame({"feature": feature_names, importance_type: importance})
            .sort_values(importance_type, ascending=False)
            .reset_index(drop=True)
        )

    def save_feature_importance(self, save_path: str | Path, importance_type: str = "gain") -> Path:
        """Persist feature importance to CSV."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.get_feature_importance_frame(importance_type=importance_type).to_csv(path, index=False)
        return path

    def get_training_history_frame(self) -> pd.DataFrame:
        """Return recorded per-iteration evaluation metrics as a dataframe."""
        evals_result = getattr(self, "evals_result_", {}) or {}
        max_len = 0
        for metric_map in evals_result.values():
            for values in metric_map.values():
                max_len = max(max_len, len(values))
        if max_len == 0:
            return pd.DataFrame(columns=["iteration"])

        history: dict[str, Any] = {"iteration": np.arange(1, max_len + 1, dtype=int)}
        for dataset_name, metric_map in evals_result.items():
            for metric_name, values in metric_map.items():
                arr = np.full(max_len, np.nan, dtype=np.float64)
                arr[: len(values)] = np.asarray(values, dtype=np.float64)
                history[f"{dataset_name}_{metric_name}"] = arr

        frame = pd.DataFrame(history)
        best_iteration = getattr(self, "best_iteration_", None)
        if best_iteration:
            frame["is_best_iteration"] = frame["iteration"] == int(best_iteration)
        return frame

    def save_training_history(self, save_path: str | Path) -> Path | None:
        """Persist per-iteration evaluation metrics to CSV."""
        frame = self.get_training_history_frame()
        if frame.empty:
            return None
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return path

    def get_training_summary(self) -> dict[str, Any]:
        """Return a compact summary of the recorded training history."""
        summary: dict[str, Any] = {}
        history = self.get_training_history_frame()
        if not history.empty:
            summary["num_iterations"] = int(history["iteration"].iloc[-1])
        best_iteration = getattr(self, "best_iteration_", None)
        summary["best_iteration"] = int(best_iteration) if best_iteration else None

        metric_columns = [col for col in history.columns if col not in {"iteration", "is_best_iteration"}]
        for column in metric_columns:
            series = history[column].dropna()
            if series.empty:
                continue
            summary[f"last_{column}"] = float(series.iloc[-1])
            if best_iteration:
                best_values = history.loc[history["iteration"] == int(best_iteration), column].dropna()
                if not best_values.empty:
                    summary[f"best_{column}"] = float(best_values.iloc[0])

        for dataset_name, metrics in (getattr(self, "best_score_", {}) or {}).items():
            for metric_name, value in metrics.items():
                summary[f"model_best_{dataset_name}_{metric_name}"] = float(value)
        return summary
