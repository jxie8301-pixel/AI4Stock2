"""Pure LightGBM Native Implementation."""

import inspect
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate import safe_cross_sectional_corr


def _daily_ic_metric_from_labels(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray):
    return _daily_corr_metric_from_labels(preds, labels, dates, method="pearson", metric_name="daily_ic")


def _daily_rank_ic_metric_from_labels(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray):
    return _daily_corr_metric_from_labels(preds, labels, dates, method="spearman", metric_name="daily_rank_ic")


def _daily_corr_metric_from_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    method: str,
    metric_name: str,
):
    frame = pd.DataFrame(
        {
            "pred": np.asarray(preds, dtype=np.float32),
            "label": np.asarray(labels, dtype=np.float32),
            "date": pd.to_datetime(np.asarray(dates)),
        }
    ).dropna()
    if frame.empty:
        return metric_name, 0.0, True

    daily_ic = frame.groupby("date", sort=True).apply(
        lambda x: safe_cross_sectional_corr(x["pred"], x["label"], method=method),
        include_groups=False,
    ).dropna()
    if daily_ic.empty:
        return metric_name, 0.0, True
    return metric_name, float(daily_ic.mean()), True


def _daily_ic_metric(preds: np.ndarray, dataset: lgb.Dataset, dates: np.ndarray):
    return _daily_ic_metric_from_labels(preds, dataset.get_label(), dates)


def _compute_time_decay_weights(
    dates: np.ndarray | pd.Series,
    half_life: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Return exp-with-floor time-decay weights normalized to mean 1.

    ``floor=0`` reduces to the legacy pure exponential schedule.
    """
    half_life = float(half_life)
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    floor = float(floor)
    if floor < 0.0 or floor >= 1.0:
        raise ValueError("floor must be in [0, 1)")

    dt = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if dt.empty:
        return np.array([], dtype=np.float32)

    latest = dt.max()
    age_days = (latest - dt).dt.days.clip(lower=0).to_numpy(dtype=np.float64, copy=False)
    exp_weights = np.power(0.5, age_days / half_life)
    weights = floor + (1.0 - floor) * exp_weights
    mean_weight = float(weights.mean()) if weights.size > 0 else 1.0
    if not np.isfinite(mean_weight) or np.isclose(mean_weight, 0.0):
        return np.ones(len(dt), dtype=np.float32)
    return (weights / mean_weight).astype(np.float32, copy=False)


def _sort_frame_by_dates(
    X: pd.DataFrame,
    y: pd.Series,
    dates: np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(X) != len(y) or len(X) != len(date_series):
        raise ValueError("X, y, and dates must have the same length")
    order = np.argsort(date_series.to_numpy(dtype="datetime64[ns]"), kind="stable")
    return (
        X.iloc[order].reset_index(drop=True),
        y.iloc[order].reset_index(drop=True),
        date_series.iloc[order].reset_index(drop=True),
    )


def _compute_ranking_groups(dates: np.ndarray | pd.Series) -> np.ndarray:
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if date_series.empty:
        return np.array([], dtype=np.int32)
    return date_series.groupby(date_series, sort=False).size().to_numpy(dtype=np.int32, copy=False)


def _build_ranking_relevance_labels(
    labels: np.ndarray | pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    num_bins: int,
) -> np.ndarray:
    """Convert raw returns into per-date integer relevance labels for LTR."""
    num_bins = max(2, int(num_bins))
    label_series = pd.Series(np.asarray(labels, dtype=np.float32)).reset_index(drop=True)
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(label_series) != len(date_series):
        raise ValueError("labels and dates must have the same length")

    relevance = np.zeros(len(label_series), dtype=np.int32)
    if label_series.empty:
        return relevance

    for _, idx in date_series.groupby(date_series, sort=False).groups.items():
        group_idx = np.asarray(idx, dtype=np.int64)
        values = label_series.iloc[group_idx].to_numpy(dtype=np.float32, copy=False)
        finite_mask = np.isfinite(values)
        if finite_mask.sum() <= 1:
            continue
        finite_values = values[finite_mask]
        if np.isclose(float(finite_values.max()), float(finite_values.min())):
            continue
        ranks = pd.Series(finite_values).rank(method="average", ascending=True).to_numpy(dtype=np.float32, copy=False)
        group_rel = np.floor((ranks - 1.0) * num_bins / len(finite_values) + 1e-8).astype(np.int32)
        group_rel = np.maximum(group_rel, 0)
        group_rel = np.minimum(group_rel, num_bins - 1)
        relevance[group_idx[finite_mask]] = group_rel
    return relevance


def _build_direct_ranking_relevance_labels(labels: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float32)
    relevance = np.zeros(len(values), dtype=np.int32)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return relevance

    finite_values = values[finite_mask]
    rounded = np.rint(finite_values)
    if not np.allclose(finite_values, rounded, atol=1e-6):
        raise ValueError("direct ranking relevance labels must be integer-like")
    min_value = int(rounded.min())
    shifted = rounded.astype(np.int32, copy=False) - min_value
    relevance[finite_mask] = shifted
    return relevance


def _should_use_direct_ranking_relevance_labels(
    labels: np.ndarray | pd.Series,
    *,
    max_unique_values: int,
) -> bool:
    values = np.asarray(labels, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return False
    rounded = np.rint(finite_values)
    if not np.allclose(finite_values, rounded, atol=1e-6):
        return False
    unique_values = np.unique(rounded.astype(np.int32, copy=False))
    return 2 <= unique_values.size <= int(max_unique_values)


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
        train_weight_half_life: float | None = None,
        train_weight_floor: float = 0.0,
        ranking_num_bins: int = 5,
        early_stopping_metric: str = "default",
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
        elif loss in {"lambdarank", "rank_xendcg"}:
            objective = loss
            default_metric = "ndcg"
        else:
            default_metric = "l2"

        self.eval_metric = eval_metric or default_metric
        self.is_ranking_objective = objective in {"lambdarank", "rank_xendcg"}
        self.ranking_num_bins = max(2, int(ranking_num_bins))
        self.early_stopping_metric = str(early_stopping_metric or "default").strip().lower()
        if self.early_stopping_metric not in {"default", "daily_ic", "daily_rank_ic"}:
            raise ValueError("early_stopping_metric must be one of: default, daily_ic, daily_rank_ic")
        metric_name = self.eval_metric if self.early_stopping_metric == "default" else "None"
        self.params = {
            "objective": objective,
            "metric": metric_name,
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
        self.train_weight_half_life = (
            None if train_weight_half_life is None else float(train_weight_half_life)
        )
        self.train_weight_floor = float(train_weight_floor)
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
        train_dates: np.ndarray | pd.Series | None = None,
        valid_dates: np.ndarray | None = None,
    ):
        """Fit the LightGBM model."""
        feature_names = X_train.columns.tolist()
        train_weight = None
        train_group = None
        train_label_values = y_train.to_numpy(dtype=np.float32, copy=False)
        X_train_use = X_train
        y_train_use = y_train
        train_dates_use = None if train_dates is None else pd.to_datetime(pd.Series(train_dates)).reset_index(drop=True)
        if self.train_weight_half_life is not None:
            if train_dates_use is None:
                raise ValueError("train_dates is required when train_weight_half_life is configured")
            if len(train_dates_use) != len(X_train):
                raise ValueError("train_dates length must match X_train rows")

        if self.is_ranking_objective:
            if train_dates is None:
                raise ValueError("train_dates is required when using a ranking objective")
            X_train_use, y_train_use, train_dates_use = _sort_frame_by_dates(X_train, y_train, train_dates)
            train_label_array = y_train_use.to_numpy(dtype=np.float32, copy=False)
            if _should_use_direct_ranking_relevance_labels(
                train_label_array,
                max_unique_values=self.ranking_num_bins,
            ):
                train_label_values = _build_direct_ranking_relevance_labels(train_label_array)
            else:
                train_label_values = _build_ranking_relevance_labels(
                    train_label_array,
                    train_dates_use,
                    num_bins=self.ranking_num_bins,
                )
            train_group = _compute_ranking_groups(train_dates_use)

        if self.train_weight_half_life is not None:
            train_weight = _compute_time_decay_weights(
                train_dates_use,
                self.train_weight_half_life,
                floor=self.train_weight_floor,
            )

        dtrain = lgb.Dataset(
            X_train_use.values,
            label=train_label_values,
            weight=train_weight,
            group=train_group,
            feature_name=feature_names,
        )
        
        valid_sets = [dtrain]
        valid_names = ["train"]
        feval = None
        
        valid_label_values = None
        if X_valid is not None and y_valid is not None:
            X_valid_use = X_valid
            y_valid_use = y_valid
            valid_dates_use = None if valid_dates is None else pd.to_datetime(pd.Series(valid_dates)).reset_index(drop=True)
            valid_group = None
            valid_label_values = y_valid.to_numpy(dtype=np.float32, copy=False)
            if self.is_ranking_objective:
                if valid_dates is None:
                    raise ValueError("valid_dates is required when using a ranking objective with validation")
                X_valid_use, y_valid_use, valid_dates_use = _sort_frame_by_dates(X_valid, y_valid, valid_dates)
                valid_label_array = y_valid_use.to_numpy(dtype=np.float32, copy=False)
                if _should_use_direct_ranking_relevance_labels(
                    valid_label_array,
                    max_unique_values=self.ranking_num_bins,
                ):
                    valid_label_values = _build_direct_ranking_relevance_labels(valid_label_array)
                else:
                    valid_label_values = _build_ranking_relevance_labels(
                        valid_label_array,
                        valid_dates_use,
                        num_bins=self.ranking_num_bins,
                    )
                valid_group = _compute_ranking_groups(valid_dates_use)
            dvalid = lgb.Dataset(
                X_valid_use.values,
                label=valid_label_values,
                group=valid_group,
                reference=dtrain,
                feature_name=feature_names,
            )
            valid_sets = [dvalid]
            valid_names = ["valid"]
            if valid_dates is not None:
                raw_valid_labels = y_valid_use.to_numpy(dtype=np.float32, copy=False) if self.is_ranking_objective else y_valid.to_numpy(dtype=np.float32, copy=False)
                raw_valid_dates = valid_dates_use if self.is_ranking_objective else valid_dates
                def _feval(preds, data, *, labels=raw_valid_labels, dates=raw_valid_dates, primary=self.early_stopping_metric):
                    if primary == "daily_ic":
                        return [
                            _daily_ic_metric_from_labels(preds, labels, dates),
                            _daily_rank_ic_metric_from_labels(preds, labels, dates),
                        ]
                    if primary == "daily_rank_ic":
                        return [
                            _daily_rank_ic_metric_from_labels(preds, labels, dates),
                            _daily_ic_metric_from_labels(preds, labels, dates),
                        ]
                    return [
                        _daily_ic_metric_from_labels(preds, labels, dates),
                        _daily_rank_ic_metric_from_labels(preds, labels, dates),
                    ]

                feval = _feval
            elif self.early_stopping_metric != "default" and self.early_stop > 0:
                raise ValueError("valid_dates is required when early_stopping_metric is daily_ic or daily_rank_ic")
            
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
