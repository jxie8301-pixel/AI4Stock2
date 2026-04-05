"""Runtime data loading helpers for the native rolling pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_source import resolve_data_source_name
from src.factor_store import load_available_dates, load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import (
    compute_finite_feature_mask_frame,
    materialize_selected_feature_frame,
    resolve_selected_feature_columns,
)
from src.label_utils import sanitize_label_series
from src.rolling_types import RollingRuntimeData


def load_rolling_runtime_data(
    cfg: dict[str, Any],
    *,
    train_days: int,
    valid_days: int,
    label_column: str,
    backtest_label_column: str,
) -> RollingRuntimeData:
    factor_store_dir = get_native_factor_store_dir(cfg)
    data_source = resolve_data_source_name(cfg)
    lookback = int(cfg["features"]["lookback"])
    batch_size = int(cfg["model"]["batch_size"])

    print(f"\n[Step 1] Loading Parquet Factor Store Metadata (data_source={data_source})")
    meta = load_factor_store_metadata(factor_store_dir)
    selected_feature_names, selected_feature_sources = resolve_selected_feature_columns(meta, cfg)
    print(f"Selected features: {len(selected_feature_names)} / {len(meta['feature_names'])}")

    universe_name = cfg.get("universe", "all")
    universe_dir = cfg.get("native", {}).get("universe_dir", "data/universes")
    all_trading_dates = load_available_dates(
        store_dir=factor_store_dir,
        universe_name=universe_name,
        universe_dir=universe_dir,
        progress_desc="scanning trading dates",
    )
    full_calendar = pd.Series(all_trading_dates)

    test_start = pd.Timestamp(cfg["time"]["test"][0])
    test_end = pd.Timestamp(cfg["time"]["test"][1])
    test_calendar = full_calendar[(full_calendar >= test_start) & (full_calendar <= test_end)].reset_index(drop=True)
    if test_calendar.empty:
        raise ValueError("No trading dates available for the configured test range.")

    first_test_start = test_calendar.iloc[0]
    first_test_idx = int(full_calendar.searchsorted(first_test_start))
    earliest_idx = max(0, first_test_idx - train_days - valid_days)
    load_start = full_calendar.iloc[earliest_idx]
    load_columns = list(dict.fromkeys(selected_feature_sources + ([backtest_label_column] if backtest_label_column != label_column else [])))
    factor_frame = load_factor_frame(
        store_dir=factor_store_dir,
        columns=load_columns,
        label_column=label_column,
        date_start=load_start,
        date_end=test_end,
        universe_name=universe_name,
        universe_dir=universe_dir,
        sort_by=("date", "symbol"),
        progress_desc="loading factor store",
    )
    if factor_frame.empty:
        raise ValueError("Parquet factor store returned no rows for the configured rolling date range.")
    factor_frame = materialize_selected_feature_frame(
        factor_frame,
        selected_columns=selected_feature_names,
        source_columns=selected_feature_sources,
    )

    y = factor_frame["label"].to_numpy(dtype=np.float32, copy=True)
    if backtest_label_column in factor_frame.columns:
        backtest_y = sanitize_label_series(factor_frame[backtest_label_column]).to_numpy(dtype=np.float32, copy=True)
    else:
        backtest_y = y.copy()
    dt_index = pd.to_datetime(factor_frame["date"])
    finite_feature_mask = compute_finite_feature_mask_frame(factor_frame, selected_feature_names)
    return RollingRuntimeData(
        factor_frame=factor_frame,
        dt_index=dt_index,
        y=y,
        backtest_y=backtest_y,
        full_calendar=full_calendar,
        test_start=test_start,
        test_end=test_end,
        test_calendar=test_calendar,
        selected_feature_names=selected_feature_names,
        selected_feature_sources=selected_feature_sources,
        finite_feature_mask=finite_feature_mask,
        lookback=lookback,
        batch_size=batch_size,
    )


def build_label_series(
    runtime_data: RollingRuntimeData,
) -> tuple[pd.Series, pd.Series]:
    global_test_mask = (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)
    global_dates = runtime_data.dt_index[global_test_mask]
    global_symbols = runtime_data.factor_frame.loc[global_test_mask, "symbol"].astype(str)
    label_series = pd.Series(
        runtime_data.y[global_test_mask],
        index=pd.MultiIndex.from_arrays([global_dates, global_symbols], names=["datetime", "instrument"]),
        name="label",
    ).sort_index()
    backtest_label_series = pd.Series(
        runtime_data.backtest_y[global_test_mask],
        index=pd.MultiIndex.from_arrays([global_dates, global_symbols], names=["datetime", "instrument"]),
        name="label",
    ).sort_index()
    return label_series, backtest_label_series


def build_prediction_metadata(
    cfg: dict[str, Any],
    *,
    signal_horizon: int,
    retrain_step: int,
    train_days: int,
    valid_days: int,
    runtime_data: RollingRuntimeData,
    model_name: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "data_source": resolve_data_source_name(cfg),
        "universe": str(cfg.get("universe", "")),
        "signal_horizon": int(signal_horizon),
        "retrain_step": int(retrain_step),
        "train_days": int(train_days),
        "valid_days": int(valid_days),
        "test_start": str(runtime_data.test_start.date()),
        "test_end": str(runtime_data.test_end.date()),
        "selected_feature_count": int(len(runtime_data.selected_feature_names)),
    }
