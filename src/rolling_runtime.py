"""Runtime data loading helpers for the native rolling pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_source import resolve_data_source_name, resolve_source_parquet_dir
from src.factor_store import load_available_dates, load_factor_frame, load_factor_store_metadata
from src.feature_profiles import get_native_factor_store_dir
from src.feature_selection import (
    _resolve_cross_sectional_rank_exclude_columns,
    apply_cross_sectional_rank,
    compute_finite_feature_mask_frame,
    materialize_selected_feature_frame,
    resolve_selected_feature_columns,
)
from src.label_utils import (
    get_label_column_name,
    resolve_label_embargo_days,
    resolve_opportunity_label_cfg,
    resolve_train_label_transform_cfg,
    sanitize_label_series,
)
from src.rolling_types import RollingRuntimeData
from src.source_store import load_source_frame


def load_rolling_runtime_data(
    cfg: dict[str, Any],
    *,
    train_days: int,
    valid_days: int,
    label_column: str,
    backtest_label_column: str,
    label_embargo_days: int | None = None,
    extra_columns: list[str] | None = None,
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
    if label_embargo_days is None:
        label_embargo_days = resolve_label_embargo_days(cfg)
    label_embargo_days = int(label_embargo_days)
    if label_embargo_days < 0:
        raise ValueError("label_embargo_days must be >= 0")
    earliest_idx = max(0, first_test_idx - label_embargo_days - train_days - valid_days)
    load_start = full_calendar.iloc[earliest_idx]
    extra_columns = list(extra_columns or [])
    load_columns = list(
        dict.fromkeys(
            selected_feature_sources
            + extra_columns
            + ([backtest_label_column] if backtest_label_column != label_column else [])
        )
    )
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
    ranked_training_feature_frame = _precompute_ranked_training_feature_frame(
        cfg,
        factor_frame=factor_frame,
        dt_index=dt_index,
        y=y,
        selected_feature_names=selected_feature_names,
        finite_feature_mask=finite_feature_mask,
    )
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
        ranked_training_feature_frame=ranked_training_feature_frame,
    )


def _precompute_ranked_training_feature_frame(
    cfg: dict[str, Any],
    *,
    factor_frame: pd.DataFrame,
    dt_index: pd.Series,
    y: np.ndarray,
    selected_feature_names: list[str],
    finite_feature_mask: np.ndarray,
) -> pd.DataFrame | None:
    transforms_cfg = cfg.get("features", {}).get("transforms", {}) or {}
    if str(cfg.get("model", {}).get("name", "")).strip().lower() != "lgbm":
        return None
    enabled_transforms = {
        str(name)
        for name, value in transforms_cfg.items()
        if name != "cross_sectional_rank_exclude_columns" and bool(value)
    }
    if enabled_transforms != {"cross_sectional_rank"}:
        return None

    train_label_transform = resolve_train_label_transform_cfg(cfg)
    if str(train_label_transform["mode"]) == "buyability_margin_binary":
        return None

    ranked_frame = factor_frame.loc[:, selected_feature_names].astype(np.float32, copy=True)
    eligible_mask = finite_feature_mask & np.isfinite(y)
    if not np.any(eligible_mask):
        return ranked_frame

    rank_exclude_columns = _resolve_cross_sectional_rank_exclude_columns(cfg)
    ranked_subset = apply_cross_sectional_rank(
        ranked_frame.loc[eligible_mask, selected_feature_names],
        dt_index[eligible_mask],
        exclude_columns=rank_exclude_columns,
    ).astype(np.float32, copy=False)
    ranked_frame.loc[eligible_mask, selected_feature_names] = ranked_subset
    return ranked_frame


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
    label_embargo_days: int | None = None,
    runtime_data: RollingRuntimeData,
    model_name: str,
) -> dict[str, Any]:
    opportunity_cfg = resolve_opportunity_label_cfg(cfg)
    rank_exclude_columns = sorted(_resolve_cross_sectional_rank_exclude_columns(cfg))
    if label_embargo_days is None:
        label_embargo_days = resolve_label_embargo_days(cfg, signal_horizon=signal_horizon)
    metadata = {
        "model_name": model_name,
        "data_source": resolve_data_source_name(cfg),
        "universe": str(cfg.get("universe", "")),
        "label_column": get_label_column_name(signal_horizon),
        "signal_label_column": get_label_column_name(signal_horizon),
        "signal_horizon": int(signal_horizon),
        "signal_label_horizon": int(signal_horizon),
        "backtest_label_column": get_label_column_name(1),
        "portfolio_return_label_column": get_label_column_name(1),
        "backtest_label_horizon": 1,
        "backtest_label_semantics": "daily_realized_return",
        "retrain_step": int(retrain_step),
        "train_days": int(train_days),
        "valid_days": int(valid_days),
        "label_embargo_days": int(label_embargo_days),
        "test_start": str(runtime_data.test_start.date()),
        "test_end": str(runtime_data.test_end.date()),
        "selected_feature_count": int(len(runtime_data.selected_feature_names)),
        "cross_sectional_rank_enabled": bool(cfg.get("features", {}).get("transforms", {}).get("cross_sectional_rank", False)),
        "cross_sectional_rank_exclude_columns": rank_exclude_columns,
        "opportunity_mode": str(opportunity_cfg["mode"]),
        "opportunity_threshold": float(opportunity_cfg["threshold"]),
        "opportunity_neutral_band": float(opportunity_cfg["neutral_band"]),
    }
    if model_name == "formula_score":
        formula_cfg = cfg.get("formula_score") or {}
        metadata["formula_score_mode"] = str(formula_cfg.get("mode", "rank_avg"))
        metadata["formula_score_min_abs_rank_ic"] = float(formula_cfg.get("min_abs_rank_ic", 0.0) or 0.0)
    return metadata


def build_market_data_frame(
    runtime_data: RollingRuntimeData,
    *,
    columns: list[str],
) -> pd.DataFrame:
    required_columns = ["date", "symbol", *columns]
    missing = [col for col in columns if col not in runtime_data.factor_frame.columns]
    if missing:
        raise ValueError(f"Missing market data columns in runtime_data.factor_frame: {missing}")
    global_test_mask = (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)
    frame = runtime_data.factor_frame.loc[global_test_mask, required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.set_index(["date", "symbol"]).sort_index()


def load_source_market_data_frame(
    cfg: dict[str, Any],
    runtime_data: RollingRuntimeData,
    *,
    columns: list[str],
) -> pd.DataFrame:
    global_test_mask = (runtime_data.dt_index >= runtime_data.test_start) & (runtime_data.dt_index <= runtime_data.test_end)
    symbols = (
        runtime_data.factor_frame.loc[global_test_mask, "symbol"]
        .astype(str)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not symbols:
        return pd.DataFrame(columns=columns)

    out = load_source_frame(
        store_dir=resolve_source_parquet_dir(cfg),
        columns=columns,
        date_start=runtime_data.test_start,
        date_end=runtime_data.test_end,
        symbols=symbols,
    )
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out.set_index(["date", "symbol"]).sort_index()
