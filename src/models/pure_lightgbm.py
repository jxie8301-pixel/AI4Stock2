"""Pure LightGBM Native Implementation."""

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

import lightgbm as lgb
import lightgbm.callback as lgb_callback
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _MetricContext:
    labels: np.ndarray
    label_ranks: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    input_length: int


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    group_start_mask = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    starts = np.flatnonzero(group_start_mask)
    ends = np.r_[starts[1:], values.size]
    average_ranks = 0.5 * (starts.astype(np.float64) + ends.astype(np.float64) - 1.0) + 1.0
    ranks_sorted = np.repeat(average_ranks, ends - starts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _prepare_metric_context(
    labels: np.ndarray | pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    precompute_label_ranks: bool = True,
) -> _MetricContext:
    labels_arr = np.asarray(labels, dtype=np.float32)
    date_arr = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]", copy=False)
    if len(labels_arr) != len(date_arr):
        raise ValueError("labels and dates must have the same length")
    valid_date_positions = np.flatnonzero(~np.isnat(date_arr))
    if len(valid_date_positions) == 0:
        empty_idx = np.asarray([], dtype=np.int64)
        return _MetricContext(
            labels=np.asarray([], dtype=np.float32),
            label_ranks=np.asarray([], dtype=np.float64),
            order=empty_idx,
            starts=empty_idx,
            ends=empty_idx,
            input_length=int(len(labels_arr)),
        )
    order = valid_date_positions[np.argsort(date_arr[valid_date_positions], kind="stable")]
    sorted_dates = date_arr[order]
    boundaries = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
    starts = np.r_[0, boundaries].astype(np.int64, copy=False)
    ends = np.r_[boundaries, len(sorted_dates)].astype(np.int64, copy=False)
    sorted_labels = labels_arr[order]
    label_ranks = np.asarray([], dtype=np.float64)
    if precompute_label_ranks:
        label_ranks = np.full(len(sorted_labels), np.nan, dtype=np.float64)
        for start, end in zip(starts, ends, strict=False):
            label_slice = sorted_labels[start:end]
            finite_mask = np.isfinite(label_slice)
            if int(finite_mask.sum()) >= 2:
                label_ranks[start:end][finite_mask] = _rank_average(label_slice[finite_mask])
    return _MetricContext(
        labels=sorted_labels,
        label_ranks=label_ranks,
        order=order.astype(np.int64, copy=False),
        starts=starts,
        ends=ends,
        input_length=int(len(labels_arr)),
    )


def _ordered_predictions(preds: np.ndarray, context: _MetricContext) -> np.ndarray:
    preds_arr = np.asarray(preds, dtype=np.float32)
    if len(preds_arr) != context.input_length:
        raise ValueError("predictions length must match metric context")
    return preds_arr[context.order]


def _pearson_corr_from_arrays(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or ys.size < 2:
        return float("nan")
    xs = xs.astype(np.float64, copy=False)
    ys = ys.astype(np.float64, copy=False)
    xs_centered = xs - xs.mean()
    ys_centered = ys - ys.mean()
    xs_ss = float(np.dot(xs_centered, xs_centered))
    ys_ss = float(np.dot(ys_centered, ys_centered))
    if not np.isfinite(xs_ss) or not np.isfinite(ys_ss) or xs_ss <= 0.0 or ys_ss <= 0.0:
        return float("nan")
    return float(np.dot(xs_centered, ys_centered) / np.sqrt(xs_ss * ys_ss))


def _daily_corr_metric_from_context(
    preds: np.ndarray,
    context: _MetricContext,
    *,
    method: str,
    metric_name: str,
):
    ordered_preds = _ordered_predictions(preds, context)
    values: list[float] = []
    for start, end in zip(context.starts, context.ends, strict=False):
        pred_slice = ordered_preds[start:end]
        label_slice = context.labels[start:end]
        valid_mask = np.isfinite(pred_slice) & np.isfinite(label_slice)
        if int(valid_mask.sum()) < 2:
            continue
        xs = pred_slice[valid_mask]
        ys = label_slice[valid_mask]
        if method == "spearman":
            xs = _rank_average(xs)
            label_finite_mask = np.isfinite(label_slice)
            if len(context.label_ranks) == len(context.labels) and np.array_equal(valid_mask, label_finite_mask):
                ys = context.label_ranks[start:end][valid_mask]
            else:
                ys = _rank_average(ys)
        corr = _pearson_corr_from_arrays(xs, ys)
        if np.isfinite(corr):
            values.append(float(corr))
    if not values:
        return metric_name, 0.0, True
    return metric_name, float(np.mean(values)), True


def _topk_label_metric_from_context(
    preds: np.ndarray,
    context: _MetricContext,
    *,
    topk: int,
    metric_name: str,
    excess: bool,
):
    topk = max(1, int(topk))
    ordered_preds = _ordered_predictions(preds, context)
    values: list[float] = []
    for start, end in zip(context.starts, context.ends, strict=False):
        pred_slice = ordered_preds[start:end]
        label_slice = context.labels[start:end]
        valid_mask = np.isfinite(pred_slice) & np.isfinite(label_slice)
        if not valid_mask.any():
            continue
        pred_valid = pred_slice[valid_mask]
        label_valid = label_slice[valid_mask]
        if pred_valid.size <= topk:
            selected_labels = label_valid
        else:
            threshold = float(np.partition(pred_valid, pred_valid.size - topk)[pred_valid.size - topk])
            selected_mask = pred_valid > threshold
            needed = topk - int(selected_mask.sum())
            if needed > 0:
                tie_idx = np.flatnonzero(pred_valid == threshold)[:needed]
                selected_mask[tie_idx] = True
            selected_labels = label_valid[selected_mask]
        if selected_labels.size == 0:
            continue
        selected_mean = float(selected_labels.mean())
        if excess:
            selected_mean -= float(label_valid.mean())
        values.append(selected_mean)
    if not values:
        return metric_name, 0.0, True
    return metric_name, float(np.mean(values)), True


def _daily_ic_metric_from_labels(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray):
    context = _prepare_metric_context(labels, dates, precompute_label_ranks=False)
    return _daily_corr_metric_from_context(preds, context, method="pearson", metric_name="daily_ic")


def _daily_rank_ic_metric_from_labels(preds: np.ndarray, labels: np.ndarray, dates: np.ndarray):
    context = _prepare_metric_context(labels, dates)
    return _daily_corr_metric_from_context(preds, context, method="spearman", metric_name="daily_rank_ic")


def _topk_label_metric_from_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    topk: int,
    metric_name: str,
    excess: bool,
):
    context = _prepare_metric_context(labels, dates, precompute_label_ranks=False)
    return _topk_label_metric_from_context(
        preds,
        context,
        topk=topk,
        metric_name=metric_name,
        excess=excess,
    )


def _valid_topk_label_mean_metric_from_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    topk: int,
):
    return _topk_label_metric_from_labels(
        preds,
        labels,
        dates,
        topk=topk,
        metric_name="valid_topk_label_mean",
        excess=False,
    )


def _valid_topk_excess_mean_metric_from_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    topk: int,
):
    return _topk_label_metric_from_labels(
        preds,
        labels,
        dates,
        topk=topk,
        metric_name="valid_topk_excess_mean",
        excess=True,
    )


def _daily_corr_metric_from_labels(
    preds: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    method: str,
    metric_name: str,
):
    context = _prepare_metric_context(labels, dates)
    return _daily_corr_metric_from_context(preds, context, method=method, metric_name=metric_name)


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


def _combine_sample_weights(
    primary: np.ndarray | pd.Series | None,
    secondary: np.ndarray | pd.Series | None,
) -> np.ndarray | None:
    if primary is None and secondary is None:
        return None
    if primary is None:
        return np.asarray(secondary, dtype=np.float32)
    if secondary is None:
        return np.asarray(primary, dtype=np.float32)
    primary_arr = np.asarray(primary, dtype=np.float32)
    secondary_arr = np.asarray(secondary, dtype=np.float32)
    if len(primary_arr) != len(secondary_arr):
        raise ValueError("sample weight arrays must have the same length")
    return (primary_arr * secondary_arr).astype(np.float32, copy=False)


class _MinBoostEarlyStoppingCallback:
    """Early stopping callback with a minimum-iteration guard."""

    order = 30
    before_iteration = False

    def __init__(
        self,
        *,
        stopping_rounds: int,
        min_boost_round: int,
        first_metric_only: bool,
        verbose: bool,
        min_delta: float,
    ) -> None:
        self.stopping_rounds = int(stopping_rounds)
        self.min_boost_round = max(0, int(min_boost_round))
        self.first_metric_only = bool(first_metric_only)
        self.verbose = bool(verbose)
        self.min_delta = float(min_delta)
        self.enabled = self.stopping_rounds > 0
        self._reset()

    def _reset(self) -> None:
        self.best_score: list[float] = []
        self.best_iter: list[int] = []
        self.best_score_list: list[list[tuple[Any, ...]]] = []
        self.cmp_op: list[Any] = []
        self.first_metric_name = ""

    def _init(self, env: lgb_callback.CallbackEnv) -> None:
        if not env.evaluation_result_list:
            raise ValueError("For early stopping, at least one dataset and eval metric is required for evaluation")
        self._reset()
        if self.verbose:
            print(
                "Training until validation scores don't improve for "
                f"{self.stopping_rounds} rounds after round {self.min_boost_round}"
            )
        self.first_metric_name = str(env.evaluation_result_list[0][1])
        for eval_ret in env.evaluation_result_list:
            is_higher_better = bool(eval_ret[3])
            self.best_score.append(float("-inf") if is_higher_better else float("inf"))
            self.best_iter.append(0)
            self.best_score_list.append([])
            if is_higher_better:
                self.cmp_op.append(lambda curr, best, delta=self.min_delta: curr > (best + delta))
            else:
                self.cmp_op.append(lambda curr, best, delta=self.min_delta: curr < (best - delta))

    def _is_train_set(self, dataset_name: str, env: lgb_callback.CallbackEnv) -> bool:
        if isinstance(env.model, lgb.Booster) and dataset_name == getattr(env.model, "_train_data_name", "train"):
            return True
        return False

    def __call__(self, env: lgb_callback.CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError("early stopping callback enabled but no evaluation results found")

        first_time = not any(self.best_score_list)
        for i, eval_ret in enumerate(env.evaluation_result_list):
            dataset_name, metric_name, metric_value, *_ = eval_ret
            metric_value = float(metric_value)
            if first_time or self.cmp_op[i](metric_value, self.best_score[i]):
                self.best_score[i] = metric_value
                self.best_iter[i] = env.iteration
                self.best_score_list[i] = list(env.evaluation_result_list)
            if self.first_metric_only and metric_name != self.first_metric_name:
                continue
            if self._is_train_set(str(dataset_name), env):
                continue
            if (env.iteration + 1) <= self.min_boost_round:
                continue
            if env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    print(
                        "Early stopping, best iteration is:\n"
                        f"[{self.best_iter[i] + 1}]\t"
                        + "\t".join(
                            f"{item[0]}'s {item[1]}: {item[2]:g}" for item in self.best_score_list[i]
                        )
                    )
                    if self.first_metric_only:
                        print(f"Evaluated only: {metric_name}")
                raise lgb_callback.EarlyStopException(self.best_iter[i], self.best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                raise lgb_callback.EarlyStopException(self.best_iter[i], self.best_score_list[i])


def _normalize_feature_matrix(
    X: np.ndarray | pd.DataFrame,
    *,
    name: str,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=np.float64, copy=False)
        observed_feature_names = [str(column) for column in X.columns.tolist()]
    else:
        values = np.asarray(X, dtype=np.float64)
        observed_feature_names = None
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix")
    if feature_names is None:
        resolved_feature_names = observed_feature_names or [f"f{i}" for i in range(values.shape[1])]
    else:
        resolved_feature_names = [str(column) for column in feature_names]
    if len(resolved_feature_names) != values.shape[1]:
        raise ValueError(
            f"{name} feature count mismatch: got {values.shape[1]} columns, "
            f"but feature_names has {len(resolved_feature_names)} entries"
        )
    if observed_feature_names is not None and feature_names is not None:
        if observed_feature_names != resolved_feature_names:
            raise ValueError(
                f"{name} DataFrame columns do not match the provided feature_names"
            )
    return values, resolved_feature_names


def _normalize_vector(
    values: np.ndarray | pd.Series,
    *,
    expected_len: int | None = None,
    dtype: np.dtype | str = np.float64,
    name: str,
) -> np.ndarray:
    vector = np.asarray(values, dtype=dtype)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    elif vector.ndim != 1:
        vector = vector.reshape(-1)
    if expected_len is not None and len(vector) != int(expected_len):
        raise ValueError(f"{name} length must match expected_len={expected_len}")
    return vector


def _normalize_optional_vector(
    values: np.ndarray | pd.Series | None,
    *,
    expected_len: int,
    dtype: np.dtype | str = np.float64,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    return _normalize_vector(values, expected_len=expected_len, dtype=dtype, name=name)


def _normalize_dates(
    dates: np.ndarray | pd.Series,
    *,
    expected_len: int | None = None,
    name: str,
) -> np.ndarray:
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if expected_len is not None and len(date_series) != int(expected_len):
        raise ValueError(f"{name} length must match expected_len={expected_len}")
    return date_series.to_numpy(dtype="datetime64[ns]", copy=False)


def _sort_inputs_by_dates(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | pd.Series,
    *extra_vectors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray | None, ...]]:
    date_values = _normalize_dates(dates, expected_len=len(X), name="dates")
    if len(X) != len(y) or len(X) != len(date_values):
        raise ValueError("X, y, and dates must have the same length")
    raw_dates = date_values
    if len(raw_dates) <= 1 or bool(np.all(raw_dates[1:] >= raw_dates[:-1])):
        return X, y, raw_dates, extra_vectors

    order = np.argsort(raw_dates, kind="stable")
    sorted_extras = tuple(None if values is None else np.asarray(values)[order] for values in extra_vectors)
    return (
        X[order],
        y[order],
        raw_dates[order],
        sorted_extras,
    )


def _compute_ranking_groups(dates: np.ndarray | pd.Series) -> np.ndarray:
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if date_series.empty:
        return np.array([], dtype=np.int32)
    codes, _ = pd.factorize(date_series, sort=False)
    valid_codes = codes[codes >= 0]
    if valid_codes.size == 0:
        return np.array([], dtype=np.int32)
    return np.bincount(valid_codes).astype(np.int32, copy=False)


def _build_ranking_relevance_labels(
    labels: np.ndarray | pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    num_bins: int,
) -> np.ndarray:
    """Convert raw returns into per-date integer relevance labels for LTR."""
    num_bins = max(2, int(num_bins))
    label_values = np.asarray(labels, dtype=np.float32)
    date_values = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]", copy=False)
    if len(label_values) != len(date_values):
        raise ValueError("labels and dates must have the same length")

    relevance = np.zeros(len(label_values), dtype=np.int32)
    valid_date_positions = np.flatnonzero(~np.isnat(date_values))
    if valid_date_positions.size == 0:
        return relevance

    order = valid_date_positions[np.argsort(date_values[valid_date_positions], kind="stable")]
    sorted_dates = date_values[order]
    boundaries = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sorted_dates)]
    for start, end in zip(starts, ends, strict=False):
        group_idx = order[start:end]
        values = label_values[group_idx]
        finite_mask = np.isfinite(values)
        finite_count = int(finite_mask.sum())
        if finite_count <= 1:
            continue
        finite_values = values[finite_mask]
        if np.isclose(float(finite_values.max()), float(finite_values.min())):
            continue
        ranks = _rank_average(finite_values)
        group_rel = np.floor((ranks - 1.0) * num_bins / finite_count + 1e-8).astype(np.int32)
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


_LGBM_PASSTHROUGH_PARAMS = {
    "device_type",
    "max_bin",
    "gpu_platform_id",
    "gpu_device_id",
    "gpu_use_dp",
    "num_gpu",
    "is_enable_sparse",
}


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
        min_boost_round: int = 0,
        num_boost_round: int = 1000,
        early_stopping_min_delta: float = 0.0,
        log_evaluation_period: int = 50,
        alpha: float = 0.9,
        seed: int = 42,
        train_weight_half_life: float | None = None,
        train_weight_floor: float = 0.0,
        ranking_num_bins: int = 5,
        validation_topk: int = 10,
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
        elif loss in {"binary", "binary_logloss", "cross_entropy", "logloss"}:
            objective = "binary"
            default_metric = "binary_logloss"
        elif loss in {"lambdarank", "rank_xendcg"}:
            objective = loss
            default_metric = "ndcg"
        else:
            default_metric = "l2"

        self.eval_metric = eval_metric or default_metric
        self.is_ranking_objective = objective in {"lambdarank", "rank_xendcg"}
        self.ranking_num_bins = max(2, int(ranking_num_bins))
        self.validation_topk = max(1, int(validation_topk))
        self.early_stopping_metric = str(early_stopping_metric or "default").strip().lower()
        if self.early_stopping_metric not in {
            "default",
            "daily_ic",
            "daily_rank_ic",
            "valid_topk_label_mean",
            "valid_topk_excess_mean",
        }:
            raise ValueError(
                "early_stopping_metric must be one of: "
                "default, daily_ic, daily_rank_ic, valid_topk_label_mean, valid_topk_excess_mean"
            )
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
        for param_name in _LGBM_PASSTHROUGH_PARAMS:
            if kwargs.get(param_name) is not None:
                self.params[param_name] = kwargs[param_name]
        if objective == "huber":
            self.params["alpha"] = alpha
        self.early_stop = early_stop
        self.min_boost_round = max(0, int(min_boost_round))
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
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_valid: np.ndarray | pd.DataFrame = None,
        y_valid: np.ndarray | pd.Series = None,
        train_dates: np.ndarray | pd.Series | None = None,
        valid_dates: np.ndarray | None = None,
        valid_eval_labels: np.ndarray | pd.Series | None = None,
        train_sample_weight: np.ndarray | pd.Series | None = None,
        valid_sample_weight: np.ndarray | pd.Series | None = None,
        feature_names: list[str] | None = None,
    ):
        """Fit the LightGBM model."""
        X_train_values, feature_names = _normalize_feature_matrix(
            X_train,
            name="X_train",
            feature_names=feature_names,
        )
        y_train_values = _normalize_vector(
            y_train,
            expected_len=len(X_train_values),
            dtype=np.float64,
            name="y_train",
        )
        train_weight = None
        train_group = None
        train_label_values = y_train_values.astype(np.float32, copy=False)
        X_train_use = X_train_values
        y_train_use = y_train_values
        train_sample_weight_use = _normalize_optional_vector(
            train_sample_weight,
            expected_len=len(X_train_values),
            dtype=np.float64,
            name="train_sample_weight",
        )
        train_dates_use = (
            None
            if train_dates is None
            else _normalize_dates(train_dates, expected_len=len(X_train_values), name="train_dates")
        )
        if self.train_weight_half_life is not None:
            if train_dates_use is None:
                raise ValueError("train_dates is required when train_weight_half_life is configured")
            if len(train_dates_use) != len(X_train_values):
                raise ValueError("train_dates length must match X_train rows")
        if train_sample_weight_use is not None and len(train_sample_weight_use) != len(X_train_values):
            raise ValueError("train_sample_weight length must match X_train rows")

        if self.is_ranking_objective:
            if train_dates is None:
                raise ValueError("train_dates is required when using a ranking objective")
            X_train_use, y_train_use, train_dates_use, extra_vectors = _sort_inputs_by_dates(
                X_train_values,
                y_train_values,
                train_dates_use,
                train_sample_weight_use,
            )
            train_sample_weight_use = extra_vectors[0]
            train_label_array = y_train_use.astype(np.float32, copy=False)
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
        train_weight = _combine_sample_weights(train_weight, train_sample_weight_use)

        dtrain = lgb.Dataset(
            X_train_use,
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
            X_valid_values, _ = _normalize_feature_matrix(
                X_valid,
                name="X_valid",
                feature_names=feature_names,
            )
            y_valid_values = _normalize_vector(
                y_valid,
                expected_len=len(X_valid_values),
                dtype=np.float64,
                name="y_valid",
            )
            X_valid_use = X_valid_values
            y_valid_use = y_valid_values
            valid_sample_weight_use = _normalize_optional_vector(
                valid_sample_weight,
                expected_len=len(X_valid_values),
                dtype=np.float64,
                name="valid_sample_weight",
            )
            valid_dates_use = (
                None
                if valid_dates is None
                else _normalize_dates(valid_dates, expected_len=len(X_valid_values), name="valid_dates")
            )
            valid_group = None
            valid_label_values = y_valid_values.astype(np.float32, copy=False)
            if valid_sample_weight_use is not None and len(valid_sample_weight_use) != len(X_valid_values):
                raise ValueError("valid_sample_weight length must match X_valid rows")
            if self.is_ranking_objective:
                if valid_dates is None:
                    raise ValueError("valid_dates is required when using a ranking objective with validation")
                raw_valid_eval_values = _normalize_vector(
                    valid_eval_labels if valid_eval_labels is not None else y_valid_values,
                    expected_len=len(X_valid_values),
                    dtype=np.float64,
                    name="valid_eval_labels",
                )
                X_valid_use, y_valid_use, valid_dates_use, extra_vectors = _sort_inputs_by_dates(
                    X_valid_values,
                    y_valid_values,
                    valid_dates_use,
                    valid_sample_weight_use,
                    raw_valid_eval_values,
                )
                valid_sample_weight_use = extra_vectors[0]
                raw_valid_eval_values = extra_vectors[1]
                valid_label_array = y_valid_use.astype(np.float32, copy=False)
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
                X_valid_use,
                label=valid_label_values,
                weight=None if valid_sample_weight_use is None else valid_sample_weight_use.astype(np.float32, copy=False),
                group=valid_group,
                reference=dtrain,
                feature_name=feature_names,
            )
            valid_sets = [dvalid]
            valid_names = ["valid"]
            if valid_dates is not None:
                if not self.is_ranking_objective:
                    raw_valid_eval_values = _normalize_vector(
                        valid_eval_labels if valid_eval_labels is not None else y_valid_values,
                        expected_len=len(X_valid_values),
                        dtype=np.float64,
                        name="valid_eval_labels",
                    )
                if self.is_ranking_objective:
                    raw_valid_labels = raw_valid_eval_values.astype(np.float32, copy=False)
                    raw_valid_dates = valid_dates_use
                else:
                    raw_valid_labels = raw_valid_eval_values.astype(np.float32, copy=False)
                    raw_valid_dates = valid_dates_use
                metric_context = _prepare_metric_context(raw_valid_labels, raw_valid_dates)
                def _feval(
                    preds,
                    data,
                    *,
                    context=metric_context,
                    primary=self.early_stopping_metric,
                    topk=self.validation_topk,
                ):
                    if primary == "daily_ic":
                        return [
                            _daily_corr_metric_from_context(preds, context, method="pearson", metric_name="daily_ic"),
                            _daily_corr_metric_from_context(preds, context, method="spearman", metric_name="daily_rank_ic"),
                        ]
                    if primary == "daily_rank_ic":
                        return [
                            _daily_corr_metric_from_context(preds, context, method="spearman", metric_name="daily_rank_ic"),
                            _daily_corr_metric_from_context(preds, context, method="pearson", metric_name="daily_ic"),
                        ]
                    if primary == "valid_topk_label_mean":
                        return [
                            _topk_label_metric_from_context(
                                preds,
                                context,
                                topk=topk,
                                metric_name="valid_topk_label_mean",
                                excess=False,
                            ),
                            _daily_corr_metric_from_context(preds, context, method="spearman", metric_name="daily_rank_ic"),
                            _daily_corr_metric_from_context(preds, context, method="pearson", metric_name="daily_ic"),
                        ]
                    if primary == "valid_topk_excess_mean":
                        return [
                            _topk_label_metric_from_context(
                                preds,
                                context,
                                topk=topk,
                                metric_name="valid_topk_excess_mean",
                                excess=True,
                            ),
                            _daily_corr_metric_from_context(preds, context, method="spearman", metric_name="daily_rank_ic"),
                            _daily_corr_metric_from_context(preds, context, method="pearson", metric_name="daily_ic"),
                        ]
                    return [
                        _daily_corr_metric_from_context(preds, context, method="pearson", metric_name="daily_ic"),
                        _daily_corr_metric_from_context(preds, context, method="spearman", metric_name="daily_rank_ic"),
                    ]

                feval = _feval
            elif self.early_stopping_metric != "default" and self.early_stop > 0:
                raise ValueError("valid_dates is required when early_stopping_metric is daily_ic or daily_rank_ic")
            
        evals_result: dict[str, dict[str, list[float]]] = {}
        callbacks = []
        callbacks.append(lgb.record_evaluation(evals_result))
        if self.early_stop > 0 and X_valid is not None:
            if self.min_boost_round > 0:
                callbacks.append(
                    _MinBoostEarlyStoppingCallback(
                        stopping_rounds=self.early_stop,
                        min_boost_round=self.min_boost_round,
                        first_metric_only=True,
                        verbose=True,
                        min_delta=self.early_stopping_min_delta,
                    )
                )
            else:
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

    def predict(
        self,
        X_test: np.ndarray | pd.DataFrame,
        *,
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """Predict using the trained model."""
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        X_test_values, _ = _normalize_feature_matrix(
            X_test,
            name="X_test",
            feature_names=feature_names or list(self.model.feature_name()),
        )
        return self.model.predict(X_test_values)
        
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
        for param_name in (
            "device_type",
            "max_bin",
            "gpu_platform_id",
            "gpu_device_id",
            "gpu_use_dp",
            "num_gpu",
            "is_enable_sparse",
            "num_threads",
        ):
            if param_name in self.params:
                summary[f"lgbm_{param_name}"] = self.params[param_name]
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
