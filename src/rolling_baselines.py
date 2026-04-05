"""Baseline builders for the native rolling pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import safe_cross_sectional_corr
from src.rolling_types import RollingRuntimeData


def build_average_factor_baseline_predictions(
    runtime_data: RollingRuntimeData,
) -> pd.Series | None:
    global_test_mask = (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)
    if not np.any(global_test_mask):
        return None
    unique_source_columns = list(dict.fromkeys(runtime_data.selected_feature_sources))
    feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns]
    baseline_scores = feature_frame.apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
    dates = pd.to_datetime(runtime_data.dt_index[global_test_mask])
    symbols = runtime_data.factor_frame.loc[global_test_mask, "symbol"].astype(str)
    out = pd.Series(
        baseline_scores.to_numpy(dtype=float, copy=False),
        index=pd.MultiIndex.from_arrays([dates, symbols], names=["datetime", "instrument"]),
        name="prediction",
    ).sort_index()
    return out


def build_sign_aligned_factor_baseline_predictions(
    runtime_data: RollingRuntimeData,
) -> pd.Series | None:
    unique_source_columns = list(dict.fromkeys(runtime_data.selected_feature_sources))
    if not unique_source_columns:
        return None

    train_end = runtime_data.test_start - pd.Timedelta(days=1)
    train_mask = runtime_data.dt_index <= train_end
    if not np.any(train_mask):
        return None
    global_test_mask = (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)
    if not np.any(global_test_mask):
        return None

    train_feature_frame = runtime_data.factor_frame.loc[train_mask, unique_source_columns].apply(pd.to_numeric, errors="coerce")
    train_dates = pd.to_datetime(runtime_data.dt_index[train_mask])
    train_labels = pd.Series(runtime_data.y[train_mask], dtype=float)

    sign_map: dict[str, float] = {}
    for feature_name in unique_source_columns:
        feature_values = pd.Series(train_feature_frame[feature_name].to_numpy(dtype=float, copy=False), dtype=float)
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

    test_feature_frame = runtime_data.factor_frame.loc[global_test_mask, unique_source_columns].apply(pd.to_numeric, errors="coerce")
    aligned_feature_frame = test_feature_frame.copy()
    for feature_name, sign_value in sign_map.items():
        aligned_feature_frame[feature_name] = aligned_feature_frame[feature_name] * float(sign_value)
    baseline_scores = aligned_feature_frame.mean(axis=1, skipna=True)
    dates = pd.to_datetime(runtime_data.dt_index[global_test_mask])
    symbols = runtime_data.factor_frame.loc[global_test_mask, "symbol"].astype(str)
    return pd.Series(
        baseline_scores.to_numpy(dtype=float, copy=False),
        index=pd.MultiIndex.from_arrays([dates, symbols], names=["datetime", "instrument"]),
        name="prediction",
    ).sort_index()
