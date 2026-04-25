"""Unified feature pipeline without qlib.

This module remains the public entrypoint for:
1) Feature family definitions.
2) Feature value computation.
3) Tushare sidecar augmentation.
4) Unified Parquet factor-store generation.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml

from src.data_source import (
    SUPPORTED_DATA_SOURCES,
    get_default_factor_store_dir,
    resolve_data_source_name,
    resolve_source_parquet_dir,
)
from src.label_utils import (
    get_label_column_name,
    get_label_definition,
    get_legacy_label_column_name,
    resolve_label_horizons,
)
from src.source_store import (
    detect_source_storage_layout,
    extract_bucket_id_from_path,
    list_bucket_paths,
    load_source_store_metadata,
)
from src.valuation_utils import positive_inverse

from src.feature_name_registry import (
    ALL_FACTORS_ALPHA360_PREFIX,
    ALL_FACTORS_LGBM_PREFIX,
    DEFAULT_ALPHA158_CONFIG,
    DEFAULT_FULL_FACTOR_STORE_DIR,
    DEFAULT_LGBM_PURIFIED_CONFIG,
    DEFAULT_TECHNICAL_FACTOR_CONFIG,
    DEFAULT_TEMPORAL_FACTOR_CONFIG,
    DEFAULT_TUSHARE_FACTOR_CONFIG,
    EPS,
    FULL_FACTOR_SPACE_NAME,
    TECHNICAL_FACTOR_PREFIX,
    TEMPORAL_FACTOR_PREFIX,
    TUSHARE_FACTOR_PREFIX,
    get_all_factor_feature_names,
    get_alpha158_feature_config,
    get_alpha360_feature_config,
    get_exact_duplicate_feature_source_map,
    get_full_factor_space_feature_names,
    get_known_exact_duplicate_feature_groups,
    get_lgbm_purified_feature_names,
    get_technical_factor_feature_names,
    get_temporal_factor_feature_names,
    get_tushare_factor_feature_names,
    deduplicate_exact_feature_names,
    validate_default_dimensions,
)
from src.feature_value_core import (
    _build_open_to_open_label_from_base,
    _index_to_epoch_ns,
    _prepare_ohlcv,
    _rolling_corr,
    _rolling_rank_pct,
    _rolling_regression_stats,
    _rolling_resi,
    _rolling_rsquare,
    _rolling_slope,
    _to_panel_arrays,
    build_open_to_open_label,
    build_open_to_open_labels,
    compute_all_factor_features,
    compute_alpha158,
    compute_alpha360,
    compute_lgbm_purified_features,
    compute_technical_factor_features,
    compute_temporal_factor_features,
    compute_tushare_factor_features,
)


TUSHARE_RAW_FINA_INDICATOR_DIR = Path("data/tushare/raw/fina_indicator")
TUSHARE_RAW_DIVIDEND_DIR = Path("data/tushare/raw/dividend")
TUSHARE_RAW_FORECAST_DIR = Path("data/tushare/raw/forecast")
TUSHARE_RAW_EXPRESS_DIR = Path("data/tushare/raw/express")
TUSHARE_RAW_META_DIR = Path("data/tushare/raw/meta")
TUSHARE_SYMBOL_CACHE_PATH = TUSHARE_RAW_META_DIR / "symbol_cache.parquet"
TUSHARE_INDUSTRY_CONTEXT_PATH = TUSHARE_RAW_META_DIR / "industry_context.parquet"
SHARD_DIRNAME = "_shards"
BUCKET_DIRNAME = "buckets"
BUCKET_MANIFEST_FILENAME = "manifest.parquet"
DEFAULT_BUCKET_COUNT = 512

TUSHARE_FINA_INDICATOR_FEATURE_PAIRS = [
    ("eps", "fi_eps"),
    ("dt_eps", "fi_dt_eps"),
    ("bps", "fi_bps"),
    ("ocfps", "fi_ocfps"),
    ("roe", "fi_roe"),
    ("roe_dt", "fi_roe_dt"),
    ("roa", "fi_roa"),
    ("grossprofit_margin", "fi_grossprofit_margin"),
    ("netprofit_margin", "fi_netprofit_margin"),
    ("debt_to_assets", "fi_debt_to_assets"),
    ("q_eps", "fi_q_eps"),
    ("q_dtprofit", "fi_q_dtprofit"),
    ("q_roe", "fi_q_roe"),
    ("q_dt_roe", "fi_q_dt_roe"),
    ("tr_yoy", "fi_tr_yoy"),
    ("or_yoy", "fi_or_yoy"),
    ("op_yoy", "fi_op_yoy"),
    ("netprofit_yoy", "fi_netprofit_yoy"),
    ("ocf_yoy", "fi_ocf_yoy"),
]
TUSHARE_FINA_INDICATOR_FEATURE_COLS = [target for _, target in TUSHARE_FINA_INDICATOR_FEATURE_PAIRS]

TUSHARE_DIVIDEND_FEATURE_PAIRS = [
    ("cash_div", "div_cash_div"),
    ("cash_div_tax", "div_cash_div_tax"),
    ("stk_div", "div_stk_div"),
    ("stk_bo_rate", "div_stk_bo_rate"),
    ("stk_co_rate", "div_stk_co_rate"),
    ("base_share", "div_base_share"),
]
TUSHARE_DIVIDEND_FEATURE_COLS = [target for _, target in TUSHARE_DIVIDEND_FEATURE_PAIRS]

TUSHARE_FORECAST_FEATURE_PAIRS = [
    ("p_change_min", "fc_p_change_min"),
    ("p_change_max", "fc_p_change_max"),
    ("net_profit_min", "fc_net_profit_min"),
    ("net_profit_max", "fc_net_profit_max"),
    ("last_parent_net", "fc_last_parent_net"),
]
TUSHARE_FORECAST_FEATURE_COLS = [target for _, target in TUSHARE_FORECAST_FEATURE_PAIRS]

TUSHARE_EXPRESS_FEATURE_PAIRS = [
    ("revenue", "exp_revenue"),
    ("operate_profit", "exp_operate_profit"),
    ("total_profit", "exp_total_profit"),
    ("n_income", "exp_n_income"),
    ("total_assets", "exp_total_assets"),
    ("diluted_eps", "exp_diluted_eps"),
    ("diluted_roe", "exp_diluted_roe"),
    ("yoy_sales", "exp_yoy_sales"),
    ("yoy_op", "exp_yoy_op"),
    ("yoy_tp", "exp_yoy_tp"),
    ("yoy_dedu_np", "exp_yoy_dedu_np"),
    ("yoy_eps", "exp_yoy_eps"),
    ("yoy_roe", "exp_yoy_roe"),
    ("growth_assets", "exp_growth_assets"),
    ("yoy_assets", "exp_yoy_assets"),
]
TUSHARE_EXPRESS_FEATURE_COLS = [target for _, target in TUSHARE_EXPRESS_FEATURE_PAIRS]


def _get_tushare_industry_context_feature_cols() -> list[str]:
    cols = [
        "ind_member_count",
        "ind_daily_ret",
        "ind_excess_daily_ret",
    ]
    for raw_window in DEFAULT_TUSHARE_FACTOR_CONFIG["industry_windows"]:
        window = int(raw_window)
        cols += [
            f"ind_ret_{window}",
            f"ind_std_{window}",
            f"ind_excess_ret_{window}",
            f"ind_pos_rate_{window}",
            f"ind_dispersion_{window}",
        ]
    for raw_window in DEFAULT_TUSHARE_FACTOR_CONFIG.get("relative_industry_windows", [20, 60]):
        window = int(raw_window)
        cols += [
            f"ind_turnover_mean_{window}",
            f"ind_free_turnover_mean_{window}",
            f"ind_volume_ratio_mean_{window}",
            f"ind_amihud_mean_{window}",
            f"ind_downside_amihud_mean_{window}",
            f"ind_amplitude_mean_{window}",
            f"ind_hit_up_limit_rate_{window}",
            f"ind_hit_down_limit_rate_{window}",
        ]
    cols += [
        "ind_ep_mean",
        "ind_sp_mean",
        "ind_sp_ttm_mean",
        "ind_bp_mean",
        "ind_ep_clean_mean",
        "ind_sp_clean_mean",
        "ind_sp_ttm_clean_mean",
        "ind_bp_clean_mean",
        "ind_dividend_yield_mean",
        "ind_dividend_yield_ttm_mean",
        "ind_dividend_cash_to_eps_mean",
        "ind_dividend_cash_to_ocfps_mean",
        "ind_dividend_cash_yield_proxy_mean",
        "ind_fi_ocf_to_eps_mean",
        "ind_fi_ocfps_minus_eps_mean",
        "ind_fi_roe_quality_gap_mean",
        "ind_fi_margin_quality_mean",
    ]
    return cols


def _get_tushare_bucket_source_required_columns() -> list[str]:
    required = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "turnover_free",
        "volume_ratio",
        "total_mv",
        "circ_mv",
        "total_share",
        "circ_share",
        "free_share",
        "pb",
        "pe",
        "pe_ttm",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "amplitude",
        "pct_chg",
        "limit_pre_close",
        "up_limit",
        "down_limit",
    ]
    required += TUSHARE_FINA_INDICATOR_FEATURE_COLS
    required += TUSHARE_DIVIDEND_FEATURE_COLS
    required += TUSHARE_FORECAST_FEATURE_COLS
    required += TUSHARE_EXPRESS_FEATURE_COLS
    required += _get_tushare_industry_context_feature_cols()
    return list(dict.fromkeys(required))


def _validate_tushare_bucket_source_schema(source_bucket_paths: list[Path]) -> dict[str, Any]:
    required_columns = _get_tushare_bucket_source_required_columns()
    required_set = set(required_columns)
    missing_by_path: list[tuple[str, list[str]]] = []
    for path in source_bucket_paths:
        schema_columns = set(pq.read_schema(path).names)
        missing = sorted(required_set - schema_columns)
        if missing:
            missing_by_path.append((str(path), missing))

    if missing_by_path:
        preview = "; ".join(
            f"{path}: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}"
            for path, missing in missing_by_path[:3]
        )
        raise ValueError(
            "Tushare bucket source is missing required sidecar/context columns. "
            "Regenerate the source store with Tushare sidecar and industry context columns before factor generation. "
            "For the default Tushare packed source, rerun gen_feature with --full-rebuild to auto-refresh stale source buckets. "
            f"Missing columns by bucket: {preview}"
        )

    return {
        "validated": True,
        "required_columns": required_columns,
        "validated_bucket_count": len(source_bucket_paths),
    }


def _rebuild_default_tushare_packed_source_if_stale(
    source_dir: Path,
    *,
    workers: int,
) -> dict[str, Any] | None:
    from src import collector_tushare

    if source_dir.resolve() != collector_tushare.PACKED_SOURCE_DIR.resolve():
        return None

    symbols = sorted(path.stem for path in collector_tushare.PROCESSED_DIR.glob("*.parquet"))
    if not symbols:
        raise FileNotFoundError(f"No processed Tushare parquet files found in {collector_tushare.PROCESSED_DIR}")

    print("[0/3] Tushare packed source schema is stale; rebuilding packed source buckets...")
    return collector_tushare.rebuild_packed_source_from_local(
        symbols,
        workers=max(1, int(workers)),
        incremental=True,
    )


def _rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
    safe = pd.to_numeric(series, errors="coerce").clip(lower=-0.999999)
    return np.expm1(np.log1p(safe).rolling(int(window), min_periods=1).sum())


def _clear_tushare_context_caches() -> None:
    _load_tushare_symbol_industry_map.cache_clear()
    _load_tushare_industry_context_frame.cache_clear()
    _ensure_tushare_industry_context_cache.cache_clear()


@lru_cache(maxsize=1)
def _load_tushare_symbol_industry_map() -> dict[str, str]:
    if not TUSHARE_SYMBOL_CACHE_PATH.exists():
        return {}
    frame = pd.read_parquet(TUSHARE_SYMBOL_CACHE_PATH, columns=["local_symbol", "industry"])
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["local_symbol"] = frame["local_symbol"].astype(str).str.zfill(6)
    frame["industry"] = frame["industry"].fillna("").replace("", "UNKNOWN")
    frame = frame.drop_duplicates("local_symbol", keep="last")
    return dict(zip(frame["local_symbol"], frame["industry"], strict=False))


@lru_cache(maxsize=1)
def _load_tushare_industry_context_frame() -> pd.DataFrame:
    columns = _get_tushare_industry_context_feature_cols()
    empty_index = pd.MultiIndex.from_arrays([[], []], names=["industry", "date"])
    if not TUSHARE_INDUSTRY_CONTEXT_PATH.exists():
        return pd.DataFrame(columns=columns, index=empty_index)
    frame = pd.read_parquet(TUSHARE_INDUSTRY_CONTEXT_PATH)
    if frame.empty:
        return pd.DataFrame(columns=columns, index=empty_index)
    frame = frame.copy()
    frame["industry"] = frame["industry"].fillna("").replace("", "UNKNOWN")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["industry", "date"]).set_index(["industry", "date"])
    for col in columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[columns]


def _build_tushare_industry_context_cache(parquet_dir: Path) -> Path:
    symbol_to_industry = _load_tushare_symbol_industry_map()
    rows: list[pd.DataFrame] = []
    for file_path in sorted(parquet_dir.glob("*.parquet")):
        symbol = file_path.stem
        required_columns = [
            "date",
            "close",
            "amount",
            "turnover",
            "turnover_free",
            "volume_ratio",
            "amplitude",
            "up_limit",
            "down_limit",
            "pe",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
        ]
        try:
            schema_names = set(pq.read_schema(file_path).names)
            read_columns = [col for col in required_columns if col in schema_names]
            frame = pd.read_parquet(file_path, columns=read_columns)
        except Exception:
            frame = pd.read_parquet(file_path)
        if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        if frame.empty:
            continue
        for col in required_columns:
            if col not in frame.columns:
                frame[col] = np.nan
            elif col != "date":
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        ret = frame["close"].pct_change(fill_method=None)
        amount_abs = frame["amount"].abs()
        amihud = ret.abs() / (amount_abs + EPS)
        downside_amihud = ret.abs().where(ret < 0) / (amount_abs + EPS)
        hit_up_limit = np.where(
            np.isfinite(frame["close"]) & np.isfinite(frame["up_limit"]),
            (frame["close"] >= frame["up_limit"] * (1.0 - 1e-6)).astype(float),
            np.nan,
        )
        hit_down_limit = np.where(
            np.isfinite(frame["close"]) & np.isfinite(frame["down_limit"]),
            (frame["close"] <= frame["down_limit"] * (1.0 + 1e-6)).astype(float),
            np.nan,
        )
        ep_clean = positive_inverse(frame["pe"])
        sp_clean = positive_inverse(frame["ps"])
        sp_ttm_clean = positive_inverse(frame["ps_ttm"])
        bp_clean = positive_inverse(frame["pb"])
        ep = ep_clean.fillna(-1.0)
        sp = sp_clean.fillna(-1.0)
        sp_ttm = sp_ttm_clean.fillna(-1.0)
        bp = bp_clean.fillna(-1.0)
        fina_indicator = _load_tushare_fina_indicator_features(symbol, pd.DatetimeIndex(frame["date"]))
        if fina_indicator is None or fina_indicator.empty:
            fi_frame = pd.DataFrame(index=frame.index)
        else:
            fi_frame = fina_indicator.reindex(pd.DatetimeIndex(frame["date"])).reset_index(drop=True)
        for col in TUSHARE_FINA_INDICATOR_FEATURE_COLS:
            if col not in fi_frame.columns:
                fi_frame[col] = np.nan
        dividend = _load_tushare_dividend_features(symbol, pd.DatetimeIndex(frame["date"]))
        if dividend is None or dividend.empty:
            div_frame = pd.DataFrame(index=frame.index)
        else:
            div_frame = dividend.reindex(pd.DatetimeIndex(frame["date"])).reset_index(drop=True)
        for col in TUSHARE_DIVIDEND_FEATURE_COLS:
            if col not in div_frame.columns:
                div_frame[col] = np.nan
        fi_ocf_to_eps = fi_frame["fi_ocfps"] / (fi_frame["fi_eps"].abs() + EPS)
        fi_ocfps_minus_eps = fi_frame["fi_ocfps"] - fi_frame["fi_eps"]
        fi_roe_quality_gap = fi_frame["fi_roe_dt"] - fi_frame["fi_roe"]
        fi_margin_quality = fi_frame["fi_grossprofit_margin"] - fi_frame["fi_netprofit_margin"]
        dividend_cash = div_frame["div_cash_div"]
        dividend_cash_to_eps = dividend_cash / fi_frame["fi_eps"].where(fi_frame["fi_eps"] > EPS)
        dividend_cash_to_ocfps = dividend_cash / fi_frame["fi_ocfps"].where(fi_frame["fi_ocfps"] > EPS)
        dividend_cash_yield_proxy = dividend_cash / frame["close"].where(frame["close"] > EPS)
        rows.append(
            pd.DataFrame(
                {
                    "date": frame["date"].to_numpy(copy=False),
                    "industry": symbol_to_industry.get(symbol, "UNKNOWN"),
                    "ret": ret.to_numpy(copy=False),
                    "turnover": frame["turnover"].to_numpy(copy=False),
                    "turnover_free": frame["turnover_free"].to_numpy(copy=False),
                    "volume_ratio": frame["volume_ratio"].to_numpy(copy=False),
                    "amplitude": frame["amplitude"].to_numpy(copy=False),
                    "amihud": amihud.to_numpy(copy=False),
                    "downside_amihud": downside_amihud.to_numpy(copy=False),
                    "hit_up_limit": hit_up_limit,
                    "hit_down_limit": hit_down_limit,
                    "ep": ep,
                    "sp": sp,
                    "sp_ttm": sp_ttm,
                    "bp": bp,
                    "ep_clean": ep_clean.to_numpy(copy=False),
                    "sp_clean": sp_clean.to_numpy(copy=False),
                    "sp_ttm_clean": sp_ttm_clean.to_numpy(copy=False),
                    "bp_clean": bp_clean.to_numpy(copy=False),
                    "dividend_yield": frame["dv_ratio"].to_numpy(copy=False),
                    "dividend_yield_ttm": frame["dv_ttm"].to_numpy(copy=False),
                    "dividend_cash_to_eps": dividend_cash_to_eps.to_numpy(copy=False),
                    "dividend_cash_to_ocfps": dividend_cash_to_ocfps.to_numpy(copy=False),
                    "dividend_cash_yield_proxy": dividend_cash_yield_proxy.to_numpy(copy=False),
                    "fi_ocf_to_eps": fi_ocf_to_eps.to_numpy(copy=False),
                    "fi_ocfps_minus_eps": fi_ocfps_minus_eps.to_numpy(copy=False),
                    "fi_roe_quality_gap": fi_roe_quality_gap.to_numpy(copy=False),
                    "fi_margin_quality": fi_margin_quality.to_numpy(copy=False),
                }
            )
        )

    if rows:
        combined = pd.concat(rows, ignore_index=True).dropna(subset=["date", "ret"])
        combined["industry"] = combined["industry"].fillna("").replace("", "UNKNOWN")
        industry_daily = (
            combined.groupby(["date", "industry"], observed=True)
            .agg(
                ind_member_count=("ret", "count"),
                ind_daily_ret=("ret", "mean"),
                ind_daily_pos_rate=("ret", lambda values: (pd.to_numeric(values, errors="coerce") > 0).mean()),
                ind_daily_dispersion=("ret", "std"),
                ind_daily_turnover=("turnover", "mean"),
                ind_daily_free_turnover=("turnover_free", "mean"),
                ind_daily_volume_ratio=("volume_ratio", "mean"),
                ind_daily_amihud=("amihud", "mean"),
                ind_daily_downside_amihud=("downside_amihud", "mean"),
                ind_daily_amplitude=("amplitude", "mean"),
                ind_daily_hit_up_limit_rate=("hit_up_limit", "mean"),
                ind_daily_hit_down_limit_rate=("hit_down_limit", "mean"),
                ind_ep_mean=("ep", "mean"),
                ind_sp_mean=("sp", "mean"),
                ind_sp_ttm_mean=("sp_ttm", "mean"),
                ind_bp_mean=("bp", "mean"),
                ind_ep_clean_mean=("ep_clean", "mean"),
                ind_sp_clean_mean=("sp_clean", "mean"),
                ind_sp_ttm_clean_mean=("sp_ttm_clean", "mean"),
                ind_bp_clean_mean=("bp_clean", "mean"),
                ind_dividend_yield_mean=("dividend_yield", "mean"),
                ind_dividend_yield_ttm_mean=("dividend_yield_ttm", "mean"),
                ind_dividend_cash_to_eps_mean=("dividend_cash_to_eps", "mean"),
                ind_dividend_cash_to_ocfps_mean=("dividend_cash_to_ocfps", "mean"),
                ind_dividend_cash_yield_proxy_mean=("dividend_cash_yield_proxy", "mean"),
                ind_fi_ocf_to_eps_mean=("fi_ocf_to_eps", "mean"),
                ind_fi_ocfps_minus_eps_mean=("fi_ocfps_minus_eps", "mean"),
                ind_fi_roe_quality_gap_mean=("fi_roe_quality_gap", "mean"),
                ind_fi_margin_quality_mean=("fi_margin_quality", "mean"),
            )
            .reset_index()
        )
        market_daily = (
            combined.groupby("date", observed=True)
            .agg(market_daily_ret=("ret", "mean"))
            .reset_index()
            .sort_values("date")
        )
        for raw_window in DEFAULT_TUSHARE_FACTOR_CONFIG["industry_windows"]:
            window = int(raw_window)
            market_daily[f"market_ret_{window}"] = _rolling_compound_return(market_daily["market_daily_ret"], window)
        industry_daily = industry_daily.merge(market_daily, on="date", how="left").sort_values(["industry", "date"])
        industry_daily["ind_excess_daily_ret"] = industry_daily["ind_daily_ret"] - industry_daily["market_daily_ret"]
        grouped_daily_ret = industry_daily.groupby("industry", observed=True, sort=False)["ind_daily_ret"]
        grouped_daily_pos_rate = industry_daily.groupby("industry", observed=True, sort=False)["ind_daily_pos_rate"]
        grouped_daily_dispersion = industry_daily.groupby("industry", observed=True, sort=False)["ind_daily_dispersion"]
        relative_context_sources = {
            "turnover_mean": "ind_daily_turnover",
            "free_turnover_mean": "ind_daily_free_turnover",
            "volume_ratio_mean": "ind_daily_volume_ratio",
            "amihud_mean": "ind_daily_amihud",
            "downside_amihud_mean": "ind_daily_downside_amihud",
            "amplitude_mean": "ind_daily_amplitude",
            "hit_up_limit_rate": "ind_daily_hit_up_limit_rate",
            "hit_down_limit_rate": "ind_daily_hit_down_limit_rate",
        }
        for raw_window in DEFAULT_TUSHARE_FACTOR_CONFIG["industry_windows"]:
            window = int(raw_window)
            industry_daily[f"ind_ret_{window}"] = grouped_daily_ret.transform(
                lambda series, w=window: _rolling_compound_return(series, w)
            )
            industry_daily[f"ind_std_{window}"] = grouped_daily_ret.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).std()
            )
            industry_daily[f"ind_excess_ret_{window}"] = (
                industry_daily[f"ind_ret_{window}"] - industry_daily[f"market_ret_{window}"]
            )
            industry_daily[f"ind_pos_rate_{window}"] = grouped_daily_pos_rate.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).mean()
            )
            industry_daily[f"ind_dispersion_{window}"] = grouped_daily_dispersion.transform(
                lambda series, w=window: series.rolling(w, min_periods=1).mean()
            )
        for raw_window in DEFAULT_TUSHARE_FACTOR_CONFIG.get("relative_industry_windows", [20, 60]):
            window = int(raw_window)
            for output_suffix, source_col in relative_context_sources.items():
                grouped_source = industry_daily.groupby("industry", observed=True, sort=False)[source_col]
                industry_daily[f"ind_{output_suffix}_{window}"] = grouped_source.transform(
                    lambda series, w=window: series.rolling(w, min_periods=1).mean()
                )
        output = industry_daily[["date", "industry", *_get_tushare_industry_context_feature_cols()]].copy()
    else:
        output = pd.DataFrame(columns=["date", "industry", *_get_tushare_industry_context_feature_cols()])

    output["industry"] = output.get("industry", pd.Series(dtype=object)).fillna("").replace("", "UNKNOWN")
    TUSHARE_INDUSTRY_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(TUSHARE_INDUSTRY_CONTEXT_PATH, index=False)
    _clear_tushare_context_caches()
    return TUSHARE_INDUSTRY_CONTEXT_PATH


@lru_cache(maxsize=8)
def _ensure_tushare_industry_context_cache(parquet_dir: Path) -> Path:
    if TUSHARE_INDUSTRY_CONTEXT_PATH.exists():
        schema_columns = set(pq.read_schema(TUSHARE_INDUSTRY_CONTEXT_PATH).names)
        required_columns = {"date", "industry", *_get_tushare_industry_context_feature_cols()}
        if required_columns.issubset(schema_columns):
            return TUSHARE_INDUSTRY_CONTEXT_PATH
    return _build_tushare_industry_context_cache(parquet_dir)


def _load_tushare_sidecar_features(
    symbol: str,
    date_index: pd.DatetimeIndex,
    *,
    raw_dir: Path,
    column_pairs: list[tuple[str, str]],
    days_since_ann_column: str | None = None,
) -> pd.DataFrame | None:
    path = raw_dir / f"{symbol}.parquet"
    if not path.exists():
        return None

    output_cols = [target for _, target in column_pairs]
    if days_since_ann_column is not None:
        output_cols.append(days_since_ann_column)
    available_pairs: list[tuple[str, str]]

    try:
        schema_names = set(pq.read_schema(path).names)
        if "ann_date" not in schema_names:
            return None
        available_pairs = [(source, target) for source, target in column_pairs if source in schema_names]
        if not available_pairs:
            return None
        frame = pd.read_parquet(path, columns=["ann_date", *(source for source, _ in available_pairs)])
    except Exception:
        frame = pd.read_parquet(path)
        if frame.empty or "ann_date" not in frame.columns:
            return None
        available_pairs = [(source, target) for source, target in column_pairs if source in frame.columns]
        if not available_pairs:
            return None

    if frame.empty:
        return None

    frame = frame.copy()
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="coerce")
    frame = frame.dropna(subset=["ann_date"]).sort_values("ann_date")
    if frame.empty:
        return None

    source_cols = [source for source, _ in available_pairs]
    right = frame[["ann_date", *source_cols]].copy()
    for col in source_cols:
        right[col] = pd.to_numeric(right[col], errors="coerce")
    right = right.rename(columns=dict(available_pairs))

    left = pd.DataFrame({"date": pd.to_datetime(date_index)}).sort_values("date")
    merged = pd.merge_asof(
        left,
        right.sort_values("ann_date"),
        left_on="date",
        right_on="ann_date",
        direction="backward",
    )
    if days_since_ann_column is not None:
        merged[days_since_ann_column] = (merged["date"] - merged["ann_date"]).dt.days
    merged = merged.drop(columns=["ann_date"], errors="ignore").set_index("date")
    for col in output_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged.reindex(columns=output_cols)


def _load_tushare_fina_indicator_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_FINA_INDICATOR_DIR,
        column_pairs=TUSHARE_FINA_INDICATOR_FEATURE_PAIRS,
    )


def _load_tushare_dividend_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_DIVIDEND_DIR,
        column_pairs=TUSHARE_DIVIDEND_FEATURE_PAIRS,
    )


def _load_tushare_forecast_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_FORECAST_DIR,
        column_pairs=TUSHARE_FORECAST_FEATURE_PAIRS,
        days_since_ann_column="fc_days_since_ann",
    )


def _load_tushare_express_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_EXPRESS_DIR,
        column_pairs=TUSHARE_EXPRESS_FEATURE_PAIRS,
        days_since_ann_column="exp_days_since_ann",
    )


def _load_tushare_industry_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    industry = _load_tushare_symbol_industry_map().get(str(symbol).zfill(6), "UNKNOWN")
    context = _load_tushare_industry_context_frame()
    if context.empty:
        return None
    try:
        right = context.loc[industry].reset_index().sort_values("date")
    except KeyError:
        return None
    if right.empty:
        return None
    left = pd.DataFrame({"date": pd.to_datetime(date_index)}).sort_values("date")
    merged = pd.merge_asof(left, right, on="date", direction="backward")
    merged = merged.set_index("date")
    columns = _get_tushare_industry_context_feature_cols()
    for col in columns:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged.reindex(columns=columns)


def _augment_tushare_symbol_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        return out
    date_index = pd.DatetimeIndex(pd.to_datetime(out["date"]))
    sidecar_frames: list[pd.DataFrame] = []
    for loader in (
        _load_tushare_fina_indicator_features,
        _load_tushare_dividend_features,
        _load_tushare_forecast_features,
        _load_tushare_express_features,
        _load_tushare_industry_features,
    ):
        frame = loader(symbol, date_index)
        if frame is None or frame.empty:
            continue
        aligned = frame.reindex(date_index).reset_index(drop=True)
        aligned.index = out.index
        sidecar_frames.append(aligned)
    if not sidecar_frames:
        return out
    replacement_columns = {column for frame in sidecar_frames for column in frame.columns}
    base = out.drop(columns=[column for column in replacement_columns if column in out.columns], errors="ignore")
    return pd.concat([base, *sidecar_frames], axis=1).copy()


def _compute_symbol_feat_label(
    file_path: str,
    *,
    data_source: str | None = None,
) -> tuple[str, pd.DataFrame, pd.Series]:
    df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    if data_source == "tushare":
        _ensure_tushare_industry_context_cache(Path(file_path).resolve().parent)
        df = _augment_tushare_symbol_frame(df, symbol=symbol)
    base = _prepare_ohlcv(df)
    feat = compute_all_factor_features(df, data_source=data_source, _base=base)
    label = _build_open_to_open_label_from_base(base, horizon_days=1)
    return symbol, feat, label


def _compute_symbol_feat_labels(
    file_path: str,
    *,
    label_horizons: list[int],
    data_source: str | None = None,
) -> tuple[str, pd.DataFrame, dict[str, pd.Series]]:
    df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    if data_source == "tushare":
        _ensure_tushare_industry_context_cache(Path(file_path).resolve().parent)
        df = _augment_tushare_symbol_frame(df, symbol=symbol)
    base = _prepare_ohlcv(df)
    feat = compute_all_factor_features(df, data_source=data_source, _base=base)
    labels = {
        get_label_column_name(horizon): _build_open_to_open_label_from_base(base, horizon_days=horizon)
        for horizon in label_horizons
    }
    labels[get_legacy_label_column_name()] = labels[get_label_column_name(1)]
    return symbol, feat, labels


def _count_file_worker(
    file_path: str,
    data_source: str | None = None,
) -> tuple[str, int, int]:
    symbol = Path(file_path).stem
    try:
        meta = pq.read_metadata(file_path)
        n_rows = int(meta.num_rows)
    except Exception:
        n_rows = int(len(pd.read_parquet(file_path, columns=["date"])))
    return file_path, symbol, len(get_full_factor_space_feature_names(data_source=data_source)), n_rows


def _build_file_payload_worker(
    file_path: str,
    data_source: str | None = None,
) -> tuple[str, int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    symbol, feat, label = _compute_symbol_feat_label(file_path, data_source=data_source)
    payload = _to_panel_arrays(feat, label)
    return symbol, feat.shape[1], payload


def _write_panel_file_slice_process(
    file_path: str,
    start: int,
    count: int,
    symbol_id: int,
    x_path: str,
    y_path: str,
    date_path: str,
    symbol_path: str,
    total_rows: int,
    n_feat: int,
    data_source: str | None = None,
) -> int:
    symbol, feat, label = _compute_symbol_feat_label(file_path, data_source=data_source)
    x_arr, y_arr, d_arr = _to_panel_arrays(feat, label)
    if x_arr.shape[0] != count:
        raise RuntimeError(
            f"Row count mismatch for {file_path}: counted={count}, computed={x_arr.shape[0]} (symbol={symbol})"
        )

    x_store = np.lib.format.open_memmap(x_path, mode="r+", dtype=np.float32, shape=(total_rows, n_feat))
    y_store = np.lib.format.open_memmap(y_path, mode="r+", dtype=np.float32, shape=(total_rows,))
    date_store = np.lib.format.open_memmap(date_path, mode="r+", dtype=np.int64, shape=(total_rows,))
    symbol_store = np.lib.format.open_memmap(symbol_path, mode="r+", dtype=np.int32, shape=(total_rows,))

    end = start + count
    x_store[start:end] = x_arr
    y_store[start:end] = y_arr
    date_store[start:end] = d_arr
    symbol_store[start:end] = symbol_id
    return count


def _json_dumps_canonical(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _source_file_signature(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _shard_base_name(file_path: str | Path) -> str:
    return Path(file_path).stem


def _shard_paths(shard_root: Path, shard_meta_root: Path, file_path: str | Path) -> tuple[Path, Path]:
    base = _shard_base_name(file_path)
    return shard_root / f"{base}.parquet", shard_meta_root / f"{base}.json"


def _stable_bucket_id(symbol: str, bucket_count: int) -> int:
    return zlib.crc32(symbol.encode("utf-8")) % max(1, int(bucket_count))


def _bucket_path(bucket_root: Path, bucket_id: int) -> Path:
    return bucket_root / f"part-{int(bucket_id):04d}.parquet"


def _bucket_manifest_path(store_root: Path) -> Path:
    return store_root / BUCKET_MANIFEST_FILENAME


def _load_bucket_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_reusable_shard_meta(
    *,
    shard_root: Path,
    shard_meta_root: Path,
    file_path: str | Path,
    feature_names: list[str],
    label_columns: list[str],
) -> dict[str, Any] | None:
    shard_path, meta_path = _shard_paths(shard_root, shard_meta_root, file_path)
    shard_meta = _load_json(meta_path)
    if shard_meta is None or not shard_path.exists():
        return None
    source_sig = _source_file_signature(file_path)
    if shard_meta.get("source") != source_sig:
        return None
    if shard_meta.get("factor_space") != FULL_FACTOR_SPACE_NAME:
        return None
    if shard_meta.get("feature_names") != feature_names:
        return None
    if shard_meta.get("label_columns") != label_columns:
        return None
    row_count = shard_meta.get("row_count")
    if not isinstance(row_count, int) or row_count < 0:
        return None
    return shard_meta


def _save_shard(
    *,
    shard_root: Path,
    shard_meta_root: Path,
    file_path: str | Path,
    symbol: str,
    feature_names: list[str],
    label_columns: list[str],
    shard_frame: pd.DataFrame,
) -> dict[str, Any]:
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_meta_root.mkdir(parents=True, exist_ok=True)
    shard_path, meta_path = _shard_paths(shard_root, shard_meta_root, file_path)
    shard_frame.to_parquet(shard_path, index=False, engine="pyarrow", compression="zstd")
    shard_meta = {
        "symbol": symbol,
        "row_count": int(len(shard_frame)),
        "num_features": len(feature_names),
        "factor_space": FULL_FACTOR_SPACE_NAME,
        "source": _source_file_signature(file_path),
        "feature_names": feature_names,
        "label_columns": label_columns,
        "min_date": str(pd.to_datetime(shard_frame["date"]).min().date()) if not shard_frame.empty else "",
        "max_date": str(pd.to_datetime(shard_frame["date"]).max().date()) if not shard_frame.empty else "",
        "shard_path": str(shard_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(shard_meta, f, ensure_ascii=False, indent=2)
    return shard_meta


def _build_shard_frame_from_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    label_horizons: list[int],
    data_source: str | None = None,
) -> pd.DataFrame:
    base = _prepare_ohlcv(df)
    feat = compute_all_factor_features(df, data_source=data_source, _base=base)
    labels = {
        get_label_column_name(horizon): _build_open_to_open_label_from_base(base, horizon_days=horizon)
        for horizon in label_horizons
    }
    labels[get_legacy_label_column_name()] = labels[get_label_column_name(1)]

    frame = feat.copy().astype(np.float32)
    frame.insert(0, "date", pd.to_datetime(frame.index))
    frame.insert(1, "symbol", str(symbol))
    insert_at = 2
    for label_column in [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]:
        if label_column in frame.columns:
            continue
        frame.insert(insert_at, label_column, labels[label_column].reindex(frame.index).astype(np.float32))
        insert_at += 1
    return frame.reset_index(drop=True)


def _build_shard_frame(
    file_path: str | Path,
    *,
    label_horizons: list[int],
    data_source: str | None = None,
) -> tuple[str, pd.DataFrame]:
    df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    if data_source == "tushare" and detect_source_storage_layout(Path(file_path).resolve().parent) != "bucket_shards":
        _ensure_tushare_industry_context_cache(Path(file_path).resolve().parent)
        df = _augment_tushare_symbol_frame(df, symbol=symbol)
    return symbol, _build_shard_frame_from_frame(
        df,
        symbol=symbol,
        label_horizons=label_horizons,
        data_source=data_source,
    )


def _write_factor_shard_worker(
    file_path: str,
    shard_path: str,
    meta_path: str,
    feature_names: list[str],
    label_horizons: list[int],
    data_source: str | None = None,
) -> dict[str, Any]:
    symbol, shard_frame = _build_shard_frame(
        file_path,
        label_horizons=label_horizons,
        data_source=data_source,
    )
    shard_root = Path(shard_path).parent
    shard_meta_root = Path(meta_path).parent
    return _save_shard(
        shard_root=shard_root,
        shard_meta_root=shard_meta_root,
        file_path=file_path,
        symbol=symbol,
        feature_names=feature_names,
        label_columns=[get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)],
        shard_frame=shard_frame,
    )


def _build_bucket_payload(
    file_paths: list[str],
    *,
    label_horizons: list[int],
    feature_names: list[str],
    bucket_id: int,
    data_source: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    for file_path in file_paths:
        symbol, shard_frame = _build_shard_frame(
            file_path,
            label_horizons=label_horizons,
            data_source=data_source,
        )
        frames.append(shard_frame)
        source_sig = _source_file_signature(file_path)
        manifest_rows.append(
            {
                "symbol": str(symbol),
                "bucket_id": int(bucket_id),
                "source_path": source_sig["path"],
                "source_size": int(source_sig["size"]),
                "source_mtime_ns": int(source_sig["mtime_ns"]),
                "row_count": int(len(shard_frame)),
                "min_date": str(pd.to_datetime(shard_frame["date"]).min().date()) if not shard_frame.empty else "",
                "max_date": str(pd.to_datetime(shard_frame["date"]).max().date()) if not shard_frame.empty else "",
                "feature_count": int(len(feature_names)),
                "label_columns": ",".join(label_columns),
            }
        )
    if frames:
        bucket_frame = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    else:
        bucket_frame = pd.DataFrame(columns=["date", "symbol", *label_columns, *feature_names])
    manifest_frame = pd.DataFrame(manifest_rows)
    return bucket_frame, manifest_frame


def _write_factor_bucket_worker(
    file_paths: list[str],
    *,
    bucket_id: int,
    bucket_path: str,
    label_horizons: list[int],
    feature_names: list[str],
    data_source: str | None = None,
) -> list[dict[str, Any]]:
    bucket_frame, manifest_frame = _build_bucket_payload(
        file_paths,
        label_horizons=label_horizons,
        feature_names=feature_names,
        bucket_id=bucket_id,
        data_source=data_source,
    )
    out_path = Path(bucket_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_frame.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    return manifest_frame.to_dict(orient="records")


def _build_factor_bucket_from_source_bucket(
    source_bucket_path: str | Path,
    *,
    label_horizons: list[int],
    feature_names: list[str],
    data_source: str | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    source_path = Path(source_bucket_path)
    bucket_id = extract_bucket_id_from_path(source_path)
    source_frame = pd.read_parquet(source_path)
    if source_frame.empty:
        empty_columns = ["date", "symbol", get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons), *feature_names]
        return bucket_id, pd.DataFrame(columns=empty_columns), pd.DataFrame(
            columns=["symbol", "bucket_id", "source_path", "source_size", "source_mtime_ns", "row_count", "min_date", "max_date", "feature_count", "label_columns"]
        )

    source_frame = source_frame.copy()
    source_frame["date"] = pd.to_datetime(source_frame["date"], errors="coerce")
    source_frame["symbol"] = source_frame["symbol"].astype(str)
    source_frame = source_frame.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    stat = source_path.stat()
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    for symbol, symbol_frame in source_frame.groupby("symbol", sort=True):
        built = _build_shard_frame_from_frame(
            symbol_frame.reset_index(drop=True),
            symbol=str(symbol),
            label_horizons=label_horizons,
            data_source=data_source,
        )
        frames.append(built)
        manifest_rows.append(
            {
                "symbol": str(symbol),
                "bucket_id": int(bucket_id),
                "source_path": str(source_path.resolve()),
                "source_size": int(stat.st_size),
                "source_mtime_ns": int(stat.st_mtime_ns),
                "row_count": int(len(built)),
                "min_date": str(pd.to_datetime(built["date"]).min().date()) if not built.empty else "",
                "max_date": str(pd.to_datetime(built["date"]).max().date()) if not built.empty else "",
                "feature_count": int(len(feature_names)),
                "label_columns": ",".join(label_columns),
            }
        )

    bucket_frame = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    manifest_frame = pd.DataFrame(manifest_rows)
    return bucket_id, bucket_frame, manifest_frame


def _write_factor_bucket_from_source_bucket_worker(
    source_bucket_path: str,
    *,
    output_bucket_root: str,
    label_horizons: list[int],
    feature_names: list[str],
    data_source: str | None = None,
) -> list[dict[str, Any]]:
    bucket_id, bucket_frame, manifest_frame = _build_factor_bucket_from_source_bucket(
        source_bucket_path,
        label_horizons=label_horizons,
        feature_names=feature_names,
        data_source=data_source,
    )
    out_path = _bucket_path(Path(output_bucket_root), bucket_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_frame.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    return manifest_frame.to_dict(orient="records")


def _remove_orphan_shards(
    *,
    shard_root: Path,
    shard_meta_root: Path,
    source_files: list[Path],
) -> None:
    valid_names = {_shard_base_name(path) for path in source_files}
    for meta_path in shard_meta_root.glob("*.json"):
        if meta_path.stem in valid_names:
            continue
        shard_path = shard_root / f"{meta_path.stem}.parquet"
        if shard_path.exists():
            shard_path.unlink()
        meta_path.unlink()


def _collect_shard_metas(shard_meta_root: Path) -> list[dict[str, Any]]:
    shard_metas: list[dict[str, Any]] = []
    for meta_path in sorted(shard_meta_root.glob("*.json")):
        shard_meta = _load_json(meta_path)
        if shard_meta is not None:
            shard_metas.append(shard_meta)
    return shard_metas


def _compute_available_dates_from_shards(shard_root: Path) -> list[str]:
    dataset = ds.dataset(shard_root, format="parquet")
    unique_dates: set[str] = set()
    scanner = dataset.scanner(columns=["date"], batch_size=65536)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        dates = pd.to_datetime(batch.column("date").to_pandas(), errors="coerce")
        for value in dates.dropna().unique():
            unique_dates.add(str(pd.Timestamp(value).date()))
    return sorted(unique_dates)


def _generate_factor_store_from_bucket_source(
    *,
    parquet_dir: str,
    output_dir: str,
    workers: int,
    label_horizons: list[int],
    data_source: str | None,
    auto_rebuild_stale_tushare_source: bool,
) -> dict[str, Any]:
    pdir = Path(parquet_dir)
    out_root = Path(output_dir)
    bucket_root = out_root / BUCKET_DIRNAME
    manifest_path = _bucket_manifest_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bucket_root.mkdir(parents=True, exist_ok=True)

    source_bucket_paths = list_bucket_paths(pdir)
    if not source_bucket_paths:
        raise FileNotFoundError(f"No source bucket shards found in {pdir}")

    feature_names = get_full_factor_space_feature_names(data_source=data_source)
    source_meta = load_source_store_metadata(pdir) or {}
    source_schema_validation: dict[str, Any] = {"validated": False, "reason": "not_required"}
    if data_source == "tushare":
        try:
            source_schema_validation = _validate_tushare_bucket_source_schema(source_bucket_paths)
        except ValueError:
            if not auto_rebuild_stale_tushare_source:
                raise
            rebuild_meta = _rebuild_default_tushare_packed_source_if_stale(pdir, workers=workers)
            if rebuild_meta is None:
                raise
            source_bucket_paths = list_bucket_paths(pdir)
            source_meta = load_source_store_metadata(pdir) or {}
            source_schema_validation = _validate_tushare_bucket_source_schema(source_bucket_paths)
            source_schema_validation["auto_rebuilt_stale_source"] = True
            source_schema_validation["source_rebuild_incremental"] = rebuild_meta.get("incremental")
    workers = max(1, int(workers))

    print(
        f"[1/3] Building factor store from {len(source_bucket_paths)} source bucket shards "
        f"(workers={workers})..."
    )
    written_manifest_rows: list[dict[str, Any]] = []
    if workers == 1:
        pbar = tqdm(source_bucket_paths, desc="buckets", total=len(source_bucket_paths), unit="bucket")
        for source_bucket_path in pbar:
            written_manifest_rows.extend(
                _write_factor_bucket_from_source_bucket_worker(
                    str(source_bucket_path),
                    output_bucket_root=str(bucket_root),
                    label_horizons=label_horizons,
                    feature_names=feature_names,
                    data_source=data_source,
                )
            )
        pbar.close()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _write_factor_bucket_from_source_bucket_worker,
                    str(source_bucket_path),
                    output_bucket_root=str(bucket_root),
                    label_horizons=label_horizons,
                    feature_names=feature_names,
                    data_source=data_source,
                )
                for source_bucket_path in source_bucket_paths
            ]
            pbar = tqdm(total=len(futures), desc="buckets", unit="bucket")
            for future in as_completed(futures):
                written_manifest_rows.extend(future.result())
                pbar.update(1)
            pbar.close()

    written_manifest = pd.DataFrame(written_manifest_rows)
    if written_manifest.empty:
        raise RuntimeError("Factor-store rebuild from bucket source produced no rows.")
    written_manifest = written_manifest.sort_values(["bucket_id", "symbol"]).reset_index(drop=True)
    written_manifest.to_parquet(manifest_path, index=False, engine="pyarrow", compression="zstd")

    active_bucket_ids = sorted(int(value) for value in written_manifest["bucket_id"].drop_duplicates().tolist())
    for path in bucket_root.glob("part-*.parquet"):
        if extract_bucket_id_from_path(path) not in active_bucket_ids:
            path.unlink(missing_ok=True)

    total_rows = int(written_manifest["row_count"].sum())
    metadata = {
        "storage_format": "parquet",
        "storage_layout": "bucket_shards",
        "factor_space": FULL_FACTOR_SPACE_NAME,
        "data_source": data_source or "",
        "source_parquet_dir": str(pdir),
        "source_storage_layout": "bucket_shards",
        "source_schema_validation": source_schema_validation,
        "source_layout_assumptions": {
            "bucket_source_contains_prejoined_rows": True,
            "tushare_bucket_source_requires_sidecar_context_columns": data_source == "tushare",
        },
        "source_bucket_count": int(source_meta.get("bucket_count") or len(active_bucket_ids)),
        "num_features": len(feature_names),
        "num_rows": total_rows,
        "shape": [total_rows, len(feature_names)],
        "feature_names": feature_names,
        "label": get_label_definition(1),
        "default_label_column": get_legacy_label_column_name(),
        "label_columns": [
            {
                "column": get_legacy_label_column_name(),
                "horizon": 1,
                "definition": get_label_definition(1),
                "legacy_alias": True,
            },
            *[
                {
                    "column": get_label_column_name(horizon),
                    "horizon": int(horizon),
                    "definition": get_label_definition(horizon),
                    "legacy_alias": False,
                }
                for horizon in label_horizons
            ],
        ],
        "factor_store_dir": str(out_root),
        "buckets_dir": str(bucket_root),
        "bucket_count": int(source_meta.get("bucket_count") or len(active_bucket_ids)),
        "bucket_ids": active_bucket_ids,
        "manifest_path": str(manifest_path),
        "available_dates": _compute_available_dates_from_shards(bucket_root),
        "incremental": {
            "enabled": False,
            "reason": "source_bucket_layout_full_rebuild",
        },
        "source_files": (
            written_manifest[
                ["source_path", "symbol", "row_count", "min_date", "max_date", "bucket_id"]
            ].rename(columns={"source_path": "file_path"}).to_dict(orient="records")
        ),
        "source_symbols": [
            {
                "symbol": row["symbol"],
                "bucket_id": int(row["bucket_id"]),
                "row_count": int(row["row_count"]),
                "file_path": row["source_path"],
            }
            for row in written_manifest.to_dict(orient="records")
        ],
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[3/3] Done. Parquet factor store saved to: {out_root}")
    return metadata


def generate_factor_store(
    parquet_dir: str = "data/processed/combined",
    output_dir: str = DEFAULT_FULL_FACTOR_STORE_DIR,
    workers: int = 1,
    incremental: bool = False,
    label_horizons: list[int] | None = None,
    data_source: str | None = None,
    bucket_count: int = DEFAULT_BUCKET_COUNT,
) -> dict[str, Any]:
    pdir = Path(parquet_dir)
    source_storage_layout = detect_source_storage_layout(pdir)
    label_horizons = resolve_label_horizons({"label": {"horizons": label_horizons}} if label_horizons is not None else {})
    if source_storage_layout == "bucket_shards":
        return _generate_factor_store_from_bucket_source(
            parquet_dir=parquet_dir,
            output_dir=output_dir,
            workers=workers,
            label_horizons=label_horizons,
            data_source=data_source,
            auto_rebuild_stale_tushare_source=not incremental,
        )

    out_root = Path(output_dir)
    bucket_root = out_root / BUCKET_DIRNAME
    manifest_path = _bucket_manifest_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bucket_root.mkdir(parents=True, exist_ok=True)
    bucket_count = max(1, int(bucket_count))

    files = sorted(pdir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pdir}")

    if data_source == "tushare":
        print("[0/3] Ensuring Tushare industry context cache...")
        _ensure_tushare_industry_context_cache(pdir)

    feature_names = get_full_factor_space_feature_names(data_source=data_source)
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    workers = max(1, int(workers))
    existing_meta = _load_json(out_root / "meta.json") or {}
    existing_manifest = pd.DataFrame()
    existing_manifest_lookup: dict[str, dict[str, Any]] = {}
    existing_label_columns = [
        str(item.get("column"))
        for item in existing_meta.get("label_columns", [])
        if isinstance(item, dict) and item.get("column")
    ]
    can_reuse_manifest = (
        incremental
        and manifest_path.exists()
        and existing_meta.get("storage_layout") == "bucket_shards"
        and existing_meta.get("feature_names") == feature_names
        and existing_label_columns == label_columns
        and int(existing_meta.get("bucket_count") or 0) == bucket_count
    )
    if can_reuse_manifest:
        existing_manifest = _load_bucket_manifest(manifest_path)
        if not existing_manifest.empty:
            existing_manifest_lookup = {
                str(row["source_path"]): row
                for row in existing_manifest.to_dict(orient="records")
                if row.get("source_path")
            }

    print(f"[1/3] Planning factor-store build from {len(files)} parquet files (workers={workers})...")
    t0 = time.perf_counter()
    bucket_to_files: dict[int, list[Path]] = {}
    dirty_buckets: set[int] = set()
    current_source_paths: set[str] = set()
    pbar = tqdm(files, desc="planning", total=len(files), unit="file")
    for idx, fp in enumerate(pbar, start=1):
        symbol = fp.stem
        bucket_id = _stable_bucket_id(symbol, bucket_count)
        bucket_to_files.setdefault(bucket_id, []).append(fp)
        source_sig = _source_file_signature(fp)
        source_path = str(source_sig["path"])
        current_source_paths.add(source_path)
        reusable = False
        existing = existing_manifest_lookup.get(source_path)
        if existing is not None and _bucket_path(bucket_root, bucket_id).exists():
            reusable = (
                int(existing.get("source_size") or -1) == int(source_sig["size"])
                and int(existing.get("source_mtime_ns") or -1) == int(source_sig["mtime_ns"])
                and int(existing.get("bucket_id") or -1) == bucket_id
            )
        if not reusable:
            dirty_buckets.add(bucket_id)
        reused_files = sum(len(files_in_bucket) for bucket, files_in_bucket in bucket_to_files.items() if bucket not in dirty_buckets)
        elapsed = time.perf_counter() - t0
        speed = idx / elapsed if elapsed > 0 else 0.0
        eta = (len(files) - idx) / speed if speed > 0 else float("inf")
        rebuild_files = sum(len(files_in_bucket) for bucket, files_in_bucket in bucket_to_files.items() if bucket in dirty_buckets)
        pbar.set_postfix(reused=reused_files, rebuild=rebuild_files, eta_m=f"{eta/60:.1f}")
    pbar.close()

    if not existing_manifest.empty:
        removed_rows = existing_manifest.loc[~existing_manifest["source_path"].isin(current_source_paths)]
        if not removed_rows.empty:
            dirty_buckets.update(int(value) for value in removed_rows["bucket_id"].tolist())

    active_bucket_ids = sorted(bucket_to_files)
    reused_buckets = [bucket_id for bucket_id in active_bucket_ids if bucket_id not in dirty_buckets and _bucket_path(bucket_root, bucket_id).exists()]
    buckets_to_recompute = sorted(bucket_id for bucket_id in active_bucket_ids if bucket_id in dirty_buckets or not _bucket_path(bucket_root, bucket_id).exists())
    reused_files = sum(len(bucket_to_files[bucket_id]) for bucket_id in reused_buckets)
    recomputed_files = sum(len(bucket_to_files[bucket_id]) for bucket_id in buckets_to_recompute)

    print(
        f"Bucket plan: {len(reused_buckets)} buckets reused, {len(buckets_to_recompute)} buckets recomputed "
        f"({reused_files} files reused, {recomputed_files} files recomputed)"
    )

    written_manifest_rows: list[dict[str, Any]] = []
    if buckets_to_recompute:
        print(f"[2/3] Writing Parquet bucket shards (workers={workers}, bucket_count={bucket_count})...")
        t1 = time.perf_counter()
        if workers == 1:
            pbar = tqdm(buckets_to_recompute, desc="buckets", total=len(buckets_to_recompute), unit="bucket")
            for idx, bucket_id in enumerate(pbar, start=1):
                written_manifest_rows.extend(
                    _write_factor_bucket_worker(
                        [str(path) for path in bucket_to_files.get(bucket_id, [])],
                        bucket_id=bucket_id,
                        bucket_path=str(_bucket_path(bucket_root, bucket_id)),
                        label_horizons=label_horizons,
                        feature_names=feature_names,
                        data_source=data_source,
                    )
                )
                elapsed = time.perf_counter() - t1
                speed = idx / elapsed if elapsed > 0 else 0.0
                eta = (len(buckets_to_recompute) - idx) / speed if speed > 0 else float("inf")
                pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
            pbar.close()
        else:
            futures = []
            executor_cls = ThreadPoolExecutor if data_source == "tushare" else ProcessPoolExecutor
            executor_kind = "threads" if executor_cls is ThreadPoolExecutor else "processes"
            print(f"[2/3] Parallel bucket writer using {executor_kind}.")
            with executor_cls(max_workers=workers) as executor:
                for bucket_id in buckets_to_recompute:
                    futures.append(
                        executor.submit(
                            _write_factor_bucket_worker,
                            [str(path) for path in bucket_to_files.get(bucket_id, [])],
                            bucket_id=bucket_id,
                            bucket_path=str(_bucket_path(bucket_root, bucket_id)),
                            label_horizons=label_horizons,
                            feature_names=feature_names,
                            data_source=data_source,
                        )
                    )
                pbar = tqdm(total=len(buckets_to_recompute), desc="buckets", unit="bucket")
                for idx, fut in enumerate(as_completed(futures), start=1):
                    written_manifest_rows.extend(fut.result())
                    pbar.update(1)
                    elapsed = time.perf_counter() - t1
                    speed = idx / elapsed if elapsed > 0 else 0.0
                    eta = (len(buckets_to_recompute) - idx) / speed if speed > 0 else float("inf")
                    pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
                pbar.close()
    else:
        print("[2/3] Writing Parquet buckets skipped: all source files reused.")

    print("[3/3] Finalizing factor-store metadata...")
    if not existing_manifest.empty:
        reused_manifest = existing_manifest.loc[
            existing_manifest["bucket_id"].astype(int).isin(reused_buckets)
            & existing_manifest["source_path"].isin(current_source_paths)
        ].copy()
    else:
        reused_manifest = pd.DataFrame()
    written_manifest = pd.DataFrame(written_manifest_rows)
    manifest_frame = pd.concat([reused_manifest, written_manifest], ignore_index=True) if not reused_manifest.empty or not written_manifest.empty else pd.DataFrame()
    if not manifest_frame.empty:
        manifest_frame = manifest_frame.sort_values(["bucket_id", "symbol"]).reset_index(drop=True)
        manifest_frame.to_parquet(manifest_path, index=False, engine="pyarrow", compression="zstd")
    total_rows = int(manifest_frame["row_count"].sum()) if not manifest_frame.empty else 0
    active_bucket_ids = sorted(int(value) for value in manifest_frame["bucket_id"].drop_duplicates().tolist()) if not manifest_frame.empty else []

    metadata = {
        "storage_format": "parquet",
        "storage_layout": "bucket_shards",
        "factor_space": FULL_FACTOR_SPACE_NAME,
        "data_source": data_source or "",
        "source_parquet_dir": str(pdir),
        "num_features": len(feature_names),
        "num_rows": total_rows,
        "shape": [total_rows, len(feature_names)],
        "feature_names": feature_names,
        "label": get_label_definition(1),
        "default_label_column": get_legacy_label_column_name(),
        "label_columns": [
            {
                "column": get_legacy_label_column_name(),
                "horizon": 1,
                "definition": get_label_definition(1),
                "legacy_alias": True,
            },
            *[
                {
                    "column": get_label_column_name(horizon),
                    "horizon": int(horizon),
                    "definition": get_label_definition(horizon),
                    "legacy_alias": False,
                }
                for horizon in label_horizons
            ],
        ],
        "factor_store_dir": str(out_root),
        "buckets_dir": str(bucket_root),
        "bucket_count": bucket_count,
        "bucket_ids": active_bucket_ids,
        "manifest_path": str(manifest_path),
        "available_dates": _compute_available_dates_from_shards(bucket_root),
        "incremental": {
            "enabled": incremental,
            "bucket_dir": str(bucket_root),
            "reused_files": reused_files,
            "recomputed_files": recomputed_files,
            "reused_buckets": len(reused_buckets),
            "recomputed_buckets": len(buckets_to_recompute),
        },
        "source_files": (
            manifest_frame[
                ["source_path", "symbol", "row_count", "min_date", "max_date", "bucket_id"]
            ].rename(columns={"source_path": "file_path"}).to_dict(orient="records")
            if not manifest_frame.empty
            else []
        ),
        "source_symbols": [
            {
                "symbol": row["symbol"],
                "bucket_id": int(row["bucket_id"]),
                "row_count": int(row["row_count"]),
                "file_path": row["source_path"],
            }
            for row in manifest_frame.to_dict(orient="records")
        ],
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[3/3] Done. Parquet factor store saved to: {out_root}")
    return metadata


def generate_panel_cache(
    parquet_dir: str = "data/processed/combined",
    output_dir: str = DEFAULT_FULL_FACTOR_STORE_DIR,
    workers: int = 1,
    incremental: bool = False,
    label_horizons: list[int] | None = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    return generate_factor_store(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        workers=workers,
        incremental=incremental,
        label_horizons=label_horizons,
        data_source=data_source,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the unified full-factor Parquet store from parquet.")
    parser.add_argument("--config", default="configs/config.yaml", help="Experiment config path for factor-store output settings.")
    parser.add_argument(
        "--data-source",
        choices=SUPPORTED_DATA_SOURCES,
        help="Named data source for default parquet/factor-store resolution.",
    )
    parser.add_argument("--parquet-dir", default=None, help="Input parquet directory.")
    parser.add_argument("--output-dir", default=None, help="Output factor-store directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for counting/writing.")
    parser.add_argument(
        "--label-horizons",
        help="Comma-separated label horizons to materialize, for example '1,5,10,20'. If omitted, use config/defaults.",
    )
    parser.set_defaults(incremental=True)
    parser.add_argument(
        "--incremental",
        dest="incremental",
        action="store_true",
        help="Reuse unchanged per-symbol feature shards and only recompute changed parquet files (default).",
    )
    parser.add_argument(
        "--full-rebuild",
        dest="incremental",
        action="store_false",
        help="Ignore reusable shards and rebuild all per-symbol feature shards.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_default_dimensions()
    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if args.data_source:
        cfg.setdefault("data", {})
        cfg["data"]["source"] = args.data_source
    label_horizons = resolve_label_horizons(cfg)
    if args.label_horizons:
        label_horizons = resolve_label_horizons(
            {"label": {"horizons": [int(item.strip()) for item in args.label_horizons.split(",") if item.strip()]}}
        )
    data_source = resolve_data_source_name(cfg)
    parquet_dir = args.parquet_dir or resolve_source_parquet_dir(cfg)
    out_dir = (
        args.output_dir
        or cfg.get("features", {}).get("factor_store_dir")
        or cfg.get("features", {}).get("cache_dir")
        or get_default_factor_store_dir(data_source, FULL_FACTOR_SPACE_NAME)
    )
    feature_names = get_full_factor_space_feature_names(data_source=data_source)
    alpha158_count = len(get_alpha158_feature_config()[1])
    lgbm_count = len(get_lgbm_purified_feature_names())
    temporal_count = len(get_temporal_factor_feature_names())
    technical_count = len(get_technical_factor_feature_names())
    tushare_count = len(get_tushare_factor_feature_names()) if data_source == "tushare" else 0

    print(
        "storage_format=parquet, "
        f"data_source={data_source}, "
        f"parquet_dir={parquet_dir}, "
        f"factor_space={FULL_FACTOR_SPACE_NAME}, "
        f"output={out_dir}, "
        f"incremental={args.incremental}, "
        f"label_horizons={label_horizons}"
    )
    print(
        "factor_groups="
        f"legacy158:{alpha158_count}, "
        f"lgbm_purified:{lgbm_count}, "
        f"temporal:{temporal_count}, "
        f"technical:{technical_count}, "
        f"tushare:{tushare_count}, "
        f"total:{len(feature_names)}"
    )
    generate_factor_store(
        parquet_dir=parquet_dir,
        output_dir=out_dir,
        workers=max(1, int(args.workers)),
        incremental=args.incremental,
        label_horizons=label_horizons,
        data_source=data_source,
    )


if __name__ == "__main__":
    main()
