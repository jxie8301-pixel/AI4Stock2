"""Baseline builders for the native rolling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import safe_cross_sectional_corr
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


def _cross_sectional_rank_zscore(feature_frame: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    numeric_frame = feature_frame.apply(pd.to_numeric, errors="coerce")
    date_key = pd.Series(pd.to_datetime(dates).to_numpy(), index=numeric_frame.index)
    ranked = numeric_frame.groupby(date_key, sort=False).rank(method="average", pct=True)
    centered = ranked - ranked.groupby(date_key, sort=False).transform("mean")
    scale = centered.groupby(date_key, sort=False).transform("std").replace(0.0, np.nan)
    return centered.divide(scale).replace([np.inf, -np.inf], np.nan)


def _compute_train_rank_ic_weights(
    runtime_data: RollingRuntimeData,
    feature_names: list[str],
    train_mask: np.ndarray | None = None,
) -> pd.Series:
    if train_mask is None:
        train_mask = runtime_data.dt_index < runtime_data.test_start
    if not np.any(train_mask):
        return pd.Series(dtype=float)

    train_dates = pd.to_datetime(runtime_data.dt_index[train_mask])
    train_labels = pd.Series(runtime_data.y[train_mask], dtype=float)
    train_feature_frame = runtime_data.factor_frame.loc[train_mask, feature_names].apply(
        pd.to_numeric,
        errors="coerce",
    )
    weights: dict[str, float] = {}
    for feature_name in feature_names:
        frame = pd.DataFrame(
            {
                "date": train_dates,
                "feature": train_feature_frame[feature_name].to_numpy(dtype=float, copy=False),
                "label": train_labels.to_numpy(dtype=float, copy=False),
            }
        ).dropna()
        if frame.empty:
            weights[feature_name] = 0.0
            continue
        daily_rank_ic = frame.groupby("date", sort=True).apply(
            lambda x: safe_cross_sectional_corr(x["feature"], x["label"], method="spearman"),
            include_groups=False,
        ).dropna()
        mean_rank_ic = float(daily_rank_ic.mean()) if not daily_rank_ic.empty else 0.0
        weights[feature_name] = mean_rank_ic if np.isfinite(mean_rank_ic) else 0.0
    return pd.Series(weights, dtype=float)


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
    baseline_scores = feature_frame.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
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

    train_feature_frame = runtime_data.factor_frame.loc[train_mask, unique_source_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    train_dates = pd.to_datetime(runtime_data.dt_index[train_mask])
    train_labels = pd.Series(runtime_data.y[train_mask], dtype=float)

    sign_map: dict[str, float] = {}
    for feature_name in unique_source_columns:
        feature_values = pd.Series(
            train_feature_frame[feature_name].to_numpy(dtype=float, copy=False),
            dtype=float,
        )
        frame = pd.DataFrame({"date": train_dates, "feature": feature_values, "label": train_labels}).dropna()
        if frame.empty:
            sign_map[feature_name] = 1.0
            continue
        daily_rank_ic = frame.groupby("date", sort=True).apply(
            lambda x: safe_cross_sectional_corr(x["feature"], x["label"], method="spearman"),
            include_groups=False,
        ).dropna()
        if daily_rank_ic.empty:
            sign_map[feature_name] = 1.0
            continue
        mean_rank_ic = float(daily_rank_ic.mean())
        sign_map[feature_name] = 1.0 if not np.isfinite(mean_rank_ic) or mean_rank_ic >= 0 else -1.0

    test_feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    aligned_feature_frame = test_feature_frame.copy()
    for feature_name, sign_value in sign_map.items():
        aligned_feature_frame[feature_name] = aligned_feature_frame[feature_name] * float(sign_value)
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
