"""Baseline builders for the native rolling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rolling_types import RollingRuntimeData

FORMULA_SCORE_MODES = {"rank_avg", "sign_aligned_rank_avg", "rank_ic_weighted"}
FORMULA_SCORE_MODE_ALIASES = {
    "rank_average": "rank_avg",
    "rank_avg_factor_baseline": "rank_avg",
    "sign_aligned": "sign_aligned_rank_avg",
    "sign_aligned_factor_baseline": "sign_aligned_rank_avg",
    "rank_ic": "rank_ic_weighted",
    "rank_ic_weighted_factor_baseline": "rank_ic_weighted",
}


def _unique_source_columns(runtime_data: RollingRuntimeData) -> list[str]:
    return list(dict.fromkeys(runtime_data.selected_feature_sources))


def _global_test_mask(runtime_data: RollingRuntimeData) -> np.ndarray:
    return (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)


def _label_embargo_train_mask(runtime_data: RollingRuntimeData, *, label_embargo_days: int = 0) -> np.ndarray:
    label_embargo_days = int(label_embargo_days)
    if label_embargo_days < 0:
        raise ValueError("label_embargo_days must be >= 0")
    test_start_idx = int(runtime_data.full_calendar.searchsorted(runtime_data.test_start))
    train_end_idx = test_start_idx - 1 - label_embargo_days
    if train_end_idx < 0:
        return np.zeros(len(runtime_data.dt_index), dtype=bool)
    train_end = runtime_data.full_calendar.iloc[train_end_idx]
    return runtime_data.dt_index <= train_end


def _build_prediction_series(
    scores: pd.Series,
    runtime_data: RollingRuntimeData,
    mask: np.ndarray,
) -> pd.Series:
    dates = pd.to_datetime(runtime_data.dt_index[mask])
    symbols = runtime_data.factor_frame.loc[mask, "symbol"].astype(str)
    return pd.Series(
        scores.to_numpy(dtype=float, copy=False),
        index=pd.MultiIndex.from_arrays([dates, symbols], names=["datetime", "instrument"]),
        name="prediction",
    ).sort_index()


def _ensure_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
        return frame
    return frame.apply(pd.to_numeric, errors="coerce")


def _cross_sectional_rank_zscore(feature_frame: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    date_key = pd.Series(pd.to_datetime(dates).to_numpy(), index=numeric_frame.index)
    ranked = numeric_frame.groupby(date_key, sort=False).rank(method="average", pct=True)
    centered = ranked - ranked.groupby(date_key, sort=False).transform("mean")
    scale = centered.groupby(date_key, sort=False).transform("std").replace(0.0, np.nan)
    return centered.divide(scale).replace([np.inf, -np.inf], np.nan)


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


def _accumulate_rank_ic_fallback(
    feature_values: np.ndarray,
    labels: np.ndarray,
    corr_sum: np.ndarray,
    corr_count: np.ndarray,
) -> None:
    label_finite = np.isfinite(labels)
    for col_idx in range(feature_values.shape[1]):
        values = feature_values[:, col_idx]
        valid = label_finite & np.isfinite(values)
        if int(valid.sum()) < 2:
            continue
        xs = values[valid].astype(np.float64, copy=False)
        ys = labels[valid].astype(np.float64, copy=False)
        if np.unique(xs).size < 2 or np.unique(ys).size < 2:
            continue
        x_ranks = _rank_average(xs)
        y_ranks = _rank_average(ys)
        x_centered = x_ranks - x_ranks.mean()
        y_centered = y_ranks - y_ranks.mean()
        x_ss = float(np.dot(x_centered, x_centered))
        y_ss = float(np.dot(y_centered, y_centered))
        if x_ss <= 0.0 or y_ss <= 0.0:
            continue
        corr = float(np.dot(x_centered, y_centered) / np.sqrt(x_ss * y_ss))
        if np.isfinite(corr):
            corr_sum[col_idx] += corr
            corr_count[col_idx] += 1


def _accumulate_rank_ic_dense(
    feature_values: np.ndarray,
    labels: np.ndarray,
    corr_sum: np.ndarray,
    corr_count: np.ndarray,
) -> None:
    label_finite = np.isfinite(labels)
    if int(label_finite.sum()) < 2:
        return
    y = labels[label_finite].astype(np.float64, copy=False)
    X = feature_values[label_finite]
    if not np.isfinite(X).all():
        _accumulate_rank_ic_fallback(feature_values, labels, corr_sum, corr_count)
        return
    if np.unique(y).size < 2:
        return

    y_ranks = _rank_average(y)
    x_ranks = pd.DataFrame(X).rank(method="average").to_numpy(dtype=np.float64, copy=False)
    x_centered = x_ranks - x_ranks.mean(axis=0)
    y_centered = y_ranks - y_ranks.mean()
    x_ss = np.einsum("ij,ij->j", x_centered, x_centered)
    y_ss = float(np.dot(y_centered, y_centered))
    denom = np.sqrt(x_ss * y_ss)
    valid = np.isfinite(denom) & (denom > 0.0)
    if not valid.any():
        return
    numer = x_centered.T @ y_centered
    corrs = np.divide(numer, denom, out=np.full_like(numer, np.nan, dtype=np.float64), where=valid)
    finite = np.isfinite(corrs)
    corr_sum[finite] += corrs[finite]
    corr_count[finite] += 1


def _compute_train_rank_ic_weights(
    runtime_data: RollingRuntimeData,
    feature_names: list[str],
    train_mask: np.ndarray | None = None,
) -> pd.Series:
    if train_mask is None:
        train_mask = runtime_data.dt_index < runtime_data.test_start
    if not np.any(train_mask):
        return pd.Series(dtype=float)

    train_dates = pd.to_datetime(runtime_data.dt_index[train_mask]).to_numpy(dtype="datetime64[ns]", copy=False)
    train_labels = np.asarray(runtime_data.y[train_mask], dtype=np.float64)
    train_feature_values = runtime_data.factor_frame.loc[train_mask, feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64, copy=False)

    valid_date_positions = np.flatnonzero(~np.isnat(train_dates))
    if valid_date_positions.size == 0:
        return pd.Series(0.0, index=feature_names, dtype=float)

    order = valid_date_positions[np.argsort(train_dates[valid_date_positions], kind="stable")]
    sorted_dates = train_dates[order]
    sorted_labels = train_labels[order]
    sorted_features = train_feature_values[order]
    boundaries = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sorted_dates)]

    corr_sum = np.zeros(len(feature_names), dtype=np.float64)
    corr_count = np.zeros(len(feature_names), dtype=np.int32)
    for start, end in zip(starts, ends, strict=False):
        _accumulate_rank_ic_dense(
            sorted_features[start:end],
            sorted_labels[start:end],
            corr_sum,
            corr_count,
        )

    weights = np.divide(
        corr_sum,
        corr_count,
        out=np.zeros(len(feature_names), dtype=np.float64),
        where=corr_count > 0,
    )
    weights[~np.isfinite(weights)] = 0.0
    return pd.Series(weights, index=feature_names, dtype=float)


def normalize_formula_score_mode(mode: str | None) -> str:
    normalized = str(mode or "rank_avg").strip().lower()
    normalized = FORMULA_SCORE_MODE_ALIASES.get(normalized, normalized)
    if normalized not in FORMULA_SCORE_MODES:
        allowed = ", ".join(sorted(FORMULA_SCORE_MODES | set(FORMULA_SCORE_MODE_ALIASES)))
        raise ValueError(f"formula_score.mode must be one of: {allowed}")
    return normalized


def _formula_score_weights(
    runtime_data: RollingRuntimeData,
    feature_names: list[str],
    train_mask: np.ndarray,
    *,
    mode: str,
    min_abs_rank_ic: float = 0.0,
) -> pd.Series:
    mode = normalize_formula_score_mode(mode)
    if mode == "rank_avg":
        return pd.Series(1.0, index=feature_names, dtype=float)

    rank_ic_weights = _compute_train_rank_ic_weights(runtime_data, feature_names, train_mask=train_mask)
    rank_ic_weights = rank_ic_weights.reindex(feature_names).fillna(0.0)
    if min_abs_rank_ic > 0.0:
        rank_ic_weights = rank_ic_weights.where(rank_ic_weights.abs() >= min_abs_rank_ic, 0.0)

    if mode == "sign_aligned_rank_avg":
        return rank_ic_weights.apply(lambda value: 1.0 if value >= 0.0 else -1.0).where(
            rank_ic_weights != 0.0,
            0.0,
        )
    return rank_ic_weights


def build_formula_score_predictions(
    runtime_data: RollingRuntimeData,
    *,
    train_mask: np.ndarray,
    score_mask: np.ndarray,
    mode: str = "rank_avg",
    min_abs_rank_ic: float = 0.0,
) -> pd.Series | None:
    if not np.any(score_mask):
        return None
    unique_source_columns = _unique_source_columns(runtime_data)
    if not unique_source_columns:
        return None

    weights = _formula_score_weights(
        runtime_data,
        unique_source_columns,
        train_mask,
        mode=mode,
        min_abs_rank_ic=min_abs_rank_ic,
    )
    weights = weights.reindex(unique_source_columns).fillna(0.0)
    if weights.abs().sum() <= 0.0:
        return None

    feature_frame = runtime_data.factor_frame.loc[score_mask, unique_source_columns]
    score_dates = runtime_data.dt_index[score_mask]
    ranked_frame = _cross_sectional_rank_zscore(feature_frame, score_dates)
    weighted_sum = ranked_frame.mul(weights, axis=1).sum(axis=1, skipna=True)
    available_weight_sum = ranked_frame.notna().mul(weights.abs(), axis=1).sum(axis=1)
    scores = weighted_sum.divide(available_weight_sum.replace(0.0, np.nan))
    return _build_prediction_series(scores, runtime_data, score_mask)


def build_average_factor_baseline_predictions(
    runtime_data: RollingRuntimeData,
) -> pd.Series | None:
    global_test_mask = _global_test_mask(runtime_data)
    if not np.any(global_test_mask):
        return None
    unique_source_columns = _unique_source_columns(runtime_data)
    feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns]
    baseline_scores = _ensure_numeric_frame(feature_frame).mean(axis=1, skipna=True)
    return _build_prediction_series(baseline_scores, runtime_data, global_test_mask)


def build_sign_aligned_factor_baseline_predictions(
    runtime_data: RollingRuntimeData,
    *,
    label_embargo_days: int = 0,
) -> pd.Series | None:
    unique_source_columns = _unique_source_columns(runtime_data)
    if not unique_source_columns:
        return None

    train_mask = _label_embargo_train_mask(runtime_data, label_embargo_days=label_embargo_days)
    if not np.any(train_mask):
        return None
    global_test_mask = _global_test_mask(runtime_data)
    if not np.any(global_test_mask):
        return None

    rank_ic_weights = _compute_train_rank_ic_weights(
        runtime_data,
        unique_source_columns,
        train_mask=train_mask,
    )
    rank_ic_weights = rank_ic_weights.reindex(unique_source_columns).fillna(0.0)
    sign_values = pd.Series(
        np.where(rank_ic_weights.to_numpy(dtype=float, copy=False) < 0.0, -1.0, 1.0),
        index=unique_source_columns,
        dtype=float,
    )

    test_feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    aligned_feature_frame = test_feature_frame.mul(sign_values, axis=1)
    baseline_scores = aligned_feature_frame.mean(axis=1, skipna=True)
    return _build_prediction_series(baseline_scores, runtime_data, global_test_mask)


def build_rank_average_factor_baseline_predictions(
    runtime_data: RollingRuntimeData,
) -> pd.Series | None:
    global_test_mask = _global_test_mask(runtime_data)
    if not np.any(global_test_mask):
        return None
    unique_source_columns = _unique_source_columns(runtime_data)
    if not unique_source_columns:
        return None

    feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns]
    test_dates = runtime_data.dt_index[global_test_mask]
    ranked_frame = _cross_sectional_rank_zscore(feature_frame, test_dates)
    baseline_scores = ranked_frame.mean(axis=1, skipna=True)
    return _build_prediction_series(baseline_scores, runtime_data, global_test_mask)


def build_rank_ic_weighted_factor_baseline_predictions(
    runtime_data: RollingRuntimeData,
    *,
    label_embargo_days: int = 0,
) -> pd.Series | None:
    global_test_mask = _global_test_mask(runtime_data)
    if not np.any(global_test_mask):
        return None
    unique_source_columns = _unique_source_columns(runtime_data)
    if not unique_source_columns:
        return None

    train_mask = _label_embargo_train_mask(runtime_data, label_embargo_days=label_embargo_days)
    weights = _compute_train_rank_ic_weights(runtime_data, unique_source_columns, train_mask=train_mask)
    weights = weights.reindex(unique_source_columns).fillna(0.0)
    if weights.abs().sum() <= 0.0:
        return None

    feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns]
    test_dates = runtime_data.dt_index[global_test_mask]
    ranked_frame = _cross_sectional_rank_zscore(feature_frame, test_dates)
    weighted_sum = ranked_frame.mul(weights, axis=1).sum(axis=1, skipna=True)
    available_weight_sum = ranked_frame.notna().mul(weights.abs(), axis=1).sum(axis=1)
    baseline_scores = weighted_sum.divide(available_weight_sum.replace(0.0, np.nan))
    return _build_prediction_series(baseline_scores, runtime_data, global_test_mask)
