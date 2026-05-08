"""Unified feature pipeline without qlib.

This module remains the public entrypoint for:
1) Feature family definitions.
2) Feature value computation.
3) Tushare sidecar augmentation.
4) Unified Parquet factor-store generation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from src.config_loader import load_runtime_config
from src.data_source import (
    SUPPORTED_DATA_SOURCES,
    resolve_data_source_name,
    resolve_source_parquet_dir,
)
from src.feature_profiles import resolve_feature_profile
from src.label_utils import (
    get_label_column_name,
    get_label_definition,
    get_legacy_label_column_name,
    resolve_label_horizons,
)
from src.override_utils import apply_override_args
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
    get_factor_family_counts,
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
    _prepare_ohlcv,
    _rolling_corr,
    _rolling_rank_pct,
    _rolling_regression_stats,
    _rolling_resi,
    _rolling_rsquare,
    _rolling_slope,
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
TUSHARE_EVENT_AVAILABILITY_POLICY = "strict_next_trading_day_after_ann_date"
TUSHARE_INDUSTRY_MAPPING_POLICY = "static_symbol_cache_current_classification"
BUCKET_DIRNAME = "buckets"
BUCKET_MANIFEST_FILENAME = "manifest.parquet"
DEFAULT_BUCKET_COUNT = 512
DEFAULT_FACTOR_GENERATION_TIMING_FILENAME = "factor_generation_timing.json"


def _get_process_max_rss_mb() -> float | None:
    try:
        import resource
    except Exception:
        return None
    try:
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if rss <= 0:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


@dataclass
class FactorGenerationTimer:
    """Small JSON-serializable phase timer for factor-store generation."""

    phases: dict[str, dict[str, float | int]] = field(default_factory=dict)
    started_perf_counter: float = field(default_factory=time.perf_counter)
    max_worker_rss_mb: float = 0.0

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        rows: int = 0,
        symbols: int = 0,
        buckets: int = 0,
        files: int = 0,
    ) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(
                name,
                time.perf_counter() - started,
                rows=rows,
                symbols=symbols,
                buckets=buckets,
                files=files,
            )

    def add(
        self,
        name: str,
        seconds: float,
        *,
        count: int = 1,
        rows: int = 0,
        symbols: int = 0,
        buckets: int = 0,
        files: int = 0,
    ) -> None:
        phase = self.phases.setdefault(
            str(name),
            {
                "seconds": 0.0,
                "count": 0,
                "rows": 0,
                "symbols": 0,
                "buckets": 0,
                "files": 0,
            },
        )
        phase["seconds"] = float(phase["seconds"]) + float(seconds)
        phase["count"] = int(phase["count"]) + int(count)
        phase["rows"] = int(phase["rows"]) + int(rows)
        phase["symbols"] = int(phase["symbols"]) + int(symbols)
        phase["buckets"] = int(phase["buckets"]) + int(buckets)
        phase["files"] = int(phase["files"]) + int(files)

    def merge(self, summary: dict[str, Any] | None) -> None:
        if not summary:
            return
        worker_rss = summary.get("max_process_rss_mb")
        if worker_rss is not None:
            self.max_worker_rss_mb = max(self.max_worker_rss_mb, float(worker_rss))
        for name, phase in dict(summary.get("phases") or {}).items():
            if not isinstance(phase, dict):
                continue
            self.add(
                str(name),
                float(phase.get("seconds") or 0.0),
                count=int(phase.get("count") or 0),
                rows=int(phase.get("rows") or 0),
                symbols=int(phase.get("symbols") or 0),
                buckets=int(phase.get("buckets") or 0),
                files=int(phase.get("files") or 0),
            )

    def as_dict(self) -> dict[str, Any]:
        phases = {
            name: {
                **phase,
                "seconds": round(float(phase["seconds"]), 6),
            }
            for name, phase in sorted(self.phases.items())
        }
        result = {
            "wall_seconds": round(time.perf_counter() - self.started_perf_counter, 6),
            "semantics": "Worker phase seconds are summed across workers and can exceed wall_seconds.",
            "phases": phases,
        }
        current_rss = _get_process_max_rss_mb()
        if current_rss is not None:
            result["max_process_rss_mb"] = round(float(current_rss), 3)
        if self.max_worker_rss_mb > 0:
            result["max_worker_rss_mb"] = round(float(self.max_worker_rss_mb), 3)
        return result


@dataclass(frozen=True)
class FactorBucketWriteResult:
    manifest_rows: list[dict[str, Any]]
    timing: dict[str, Any]

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


def _get_tushare_source_layout_assumptions() -> dict[str, str]:
    return {
        "tushare_event_availability_policy": TUSHARE_EVENT_AVAILABILITY_POLICY,
        "tushare_industry_mapping": TUSHARE_INDUSTRY_MAPPING_POLICY,
    }


def _extract_tushare_event_availability_policy(metadata: dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    assumptions = metadata.get("source_layout_assumptions")
    if isinstance(assumptions, dict):
        value = assumptions.get("tushare_event_availability_policy")
        if value is not None:
            return str(value).strip()
    value = metadata.get("tushare_event_availability_policy")
    return str(value or "").strip()


def _validate_tushare_bucket_source_schema(
    source_bucket_paths: list[Path],
    *,
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required_columns = _get_tushare_bucket_source_required_columns()
    required_set = set(required_columns)
    validation_engine = "pyarrow"
    missing_by_path: list[tuple[str, list[str]]] = []
    for path in source_bucket_paths:
        schema_columns = set(pq.read_schema(path).names)
        missing = sorted(required_set - schema_columns)
        if missing:
            missing_by_path.append((str(path), missing))
    validated_bucket_count = len(source_bucket_paths)

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

    event_policy = _extract_tushare_event_availability_policy(source_meta)
    if event_policy != TUSHARE_EVENT_AVAILABILITY_POLICY:
        found = event_policy or "<missing>"
        raise ValueError(
            "Tushare bucket source event availability policy mismatch. "
            f"Expected {TUSHARE_EVENT_AVAILABILITY_POLICY}, found {found}. "
            "Regenerate the packed source so announcement sidecars are lagged before factor generation."
        )

    return {
        "validated": True,
        "validation_engine": validation_engine,
        "required_columns": required_columns,
        "validated_bucket_count": validated_bucket_count,
        "tushare_event_availability_policy": event_policy,
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

    print("[0/3] Tushare packed source schema/policy is stale; rebuilding packed source buckets...")
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
        schema_names = set(pq.read_schema(file_path).names)
        missing_required = [col for col in ("date", "close") if col not in schema_names]
        if missing_required:
            raise ValueError(f"Tushare processed parquet missing required columns {missing_required}: {file_path}")
        read_columns = [col for col in required_columns if col in schema_names]
        frame = pd.read_parquet(file_path, columns=read_columns)
        if frame.empty:
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


def _as_datetime64ns_index(values: object) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).astype("datetime64[ns]")


def _as_datetime64ns_series(values: object, *, name: str = "date") -> pd.Series:
    series = pd.Series(pd.to_datetime(values, errors="coerce"), name=name)
    return series.astype("datetime64[ns]")


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

    schema_names = set(pq.read_schema(path).names)
    if "ann_date" not in schema_names:
        return None
    available_pairs = [(source, target) for source, target in column_pairs if source in schema_names]
    if not available_pairs:
        return None
    frame = pd.read_parquet(path, columns=["ann_date", *(source for source, _ in available_pairs)])

    if frame.empty:
        return None

    frame = frame.copy()
    frame["ann_date"] = _as_datetime64ns_series(frame["ann_date"], name="ann_date")
    frame = frame.dropna(subset=["ann_date"]).sort_values("ann_date")
    if frame.empty:
        return None

    trading_dates = _as_datetime64ns_index(date_index).dropna().unique().sort_values()
    if trading_dates.empty:
        return None
    available_positions = trading_dates.searchsorted(frame["ann_date"], side="right")
    has_available_date = available_positions < len(trading_dates)
    frame = frame.loc[has_available_date].copy()
    if frame.empty:
        return None
    frame["available_date"] = trading_dates.take(available_positions[has_available_date])

    source_cols = [source for source, _ in available_pairs]
    right = frame[["available_date", "ann_date", *source_cols]].copy()
    for col in source_cols:
        right[col] = pd.to_numeric(right[col], errors="coerce")
    right = right.rename(columns=dict(available_pairs))

    left = pd.DataFrame({"date": _as_datetime64ns_series(date_index)}).sort_values("date")
    merged = pd.merge_asof(
        left,
        right.sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    )
    if days_since_ann_column is not None:
        merged[days_since_ann_column] = (merged["date"] - merged["ann_date"]).dt.days
    merged = merged.drop(columns=["ann_date", "available_date"], errors="ignore").set_index("date")
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
    right = right.copy()
    right["date"] = _as_datetime64ns_series(right["date"])
    right = right.dropna(subset=["date"]).sort_values("date")
    if right.empty:
        return None
    left = pd.DataFrame({"date": _as_datetime64ns_series(date_index)}).sort_values("date")
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
    date_index = _as_datetime64ns_index(out["date"])
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


def _build_shard_frame_from_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
    label_horizons: list[int],
    data_source: str | None = None,
    timing: FactorGenerationTimer | None = None,
) -> pd.DataFrame:
    row_count = int(len(df))
    timer = timing or FactorGenerationTimer()
    with timer.phase("prepare_ohlcv", rows=row_count, symbols=1):
        base = _prepare_ohlcv(df)
    with timer.phase("factor_compute", rows=row_count, symbols=1):
        feat = compute_all_factor_features(df, data_source=data_source, _base=base)
    with timer.phase("label_build", rows=row_count, symbols=1):
        labels = {
            get_label_column_name(horizon): _build_open_to_open_label_from_base(base, horizon_days=horizon)
            for horizon in label_horizons
        }
        labels[get_legacy_label_column_name()] = labels[get_label_column_name(1)]

    with timer.phase("assemble_symbol_frame", rows=row_count, symbols=1):
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
    timing: FactorGenerationTimer | None = None,
) -> tuple[str, pd.DataFrame]:
    timer = timing or FactorGenerationTimer()
    with timer.phase("read_symbol_parquet", files=1):
        df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    if data_source == "tushare" and detect_source_storage_layout(Path(file_path).resolve().parent) != "bucket_shards":
        with timer.phase("tushare_context_cache", files=1):
            _ensure_tushare_industry_context_cache(Path(file_path).resolve().parent)
        with timer.phase("tushare_sidecar_augment", rows=len(df), symbols=1):
            df = _augment_tushare_symbol_frame(df, symbol=symbol)
    return symbol, _build_shard_frame_from_frame(
        df,
        symbol=symbol,
        label_horizons=label_horizons,
        data_source=data_source,
        timing=timer,
    )


def _build_bucket_payload(
    file_paths: list[str],
    *,
    label_horizons: list[int],
    feature_names: list[str],
    bucket_id: int,
    data_source: str | None = None,
    timing: FactorGenerationTimer | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timer = timing or FactorGenerationTimer()
    frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    for file_path in file_paths:
        symbol, shard_frame = _build_shard_frame(
            file_path,
            label_horizons=label_horizons,
            data_source=data_source,
            timing=timer,
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
    with timer.phase("concat_sort_bucket", rows=sum(len(frame) for frame in frames), buckets=1):
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
) -> FactorBucketWriteResult:
    timer = FactorGenerationTimer()
    with timer.phase("bucket_payload_total", buckets=1, files=len(file_paths)):
        bucket_frame, manifest_frame = _build_bucket_payload(
            file_paths,
            label_horizons=label_horizons,
            feature_names=feature_names,
            bucket_id=bucket_id,
            data_source=data_source,
            timing=timer,
        )
    out_path = Path(bucket_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with timer.phase("write_bucket_parquet", rows=len(bucket_frame), buckets=1, files=1):
        bucket_frame.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    return FactorBucketWriteResult(
        manifest_rows=manifest_frame.to_dict(orient="records"),
        timing=timer.as_dict(),
    )


def _build_factor_bucket_from_source_bucket(
    source_bucket_path: str | Path,
    *,
    label_horizons: list[int],
    feature_names: list[str],
    data_source: str | None = None,
    timing: FactorGenerationTimer | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    timer = timing or FactorGenerationTimer()
    source_path = Path(source_bucket_path)
    bucket_id = extract_bucket_id_from_path(source_path)
    with timer.phase("read_source_bucket", buckets=1, files=1):
        source_frame = pd.read_parquet(source_path)
    if source_frame.empty:
        empty_columns = ["date", "symbol", get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons), *feature_names]
        return bucket_id, pd.DataFrame(columns=empty_columns), pd.DataFrame(
            columns=["symbol", "bucket_id", "source_path", "source_size", "source_mtime_ns", "row_count", "min_date", "max_date", "feature_count", "label_columns"]
        )

    with timer.phase("normalize_source_bucket", rows=len(source_frame), buckets=1):
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
            timing=timer,
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

    with timer.phase("concat_sort_bucket", rows=sum(len(frame) for frame in frames), buckets=1):
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
) -> FactorBucketWriteResult:
    timer = FactorGenerationTimer()
    source_path = Path(source_bucket_path)
    bucket_id = extract_bucket_id_from_path(source_path)
    out_path = _bucket_path(Path(output_bucket_root), bucket_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        with timer.phase("bucket_payload_total", buckets=1, files=1):
            with timer.phase("read_source_bucket", buckets=1, files=1):
                source_frame = pd.read_parquet(source_path)
            if source_frame.empty:
                empty_columns = ["date", "symbol", *label_columns, *feature_names]
                empty_frame = pd.DataFrame(columns=empty_columns)
                with timer.phase("write_bucket_parquet", rows=0, buckets=1, files=1):
                    empty_frame.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
                return FactorBucketWriteResult(manifest_rows=[], timing=timer.as_dict())

            with timer.phase("normalize_source_bucket", rows=len(source_frame), buckets=1):
                source_frame = source_frame.copy()
                source_frame["date"] = pd.to_datetime(source_frame["date"], errors="coerce")
                source_frame["symbol"] = source_frame["symbol"].astype(str)
                source_frame = source_frame.dropna(subset=["date", "symbol"]).sort_values(["symbol", "date"]).reset_index(drop=True)

            stat = source_path.stat()
            for symbol, symbol_frame in source_frame.groupby("symbol", sort=True):
                built = _build_shard_frame_from_frame(
                    symbol_frame.reset_index(drop=True),
                    symbol=str(symbol),
                    label_horizons=label_horizons,
                    data_source=data_source,
                    timing=timer,
                )
                table = pa.Table.from_pandas(built, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                with timer.phase("write_symbol_row_group", rows=len(built), symbols=1, buckets=1):
                    writer.write_table(table)
                total_rows += int(len(built))
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
                del table
                del built
                gc.collect()
            if writer is None:
                empty_columns = ["date", "symbol", *label_columns, *feature_names]
                empty_frame = pd.DataFrame(columns=empty_columns)
                with timer.phase("write_bucket_parquet", rows=0, buckets=1, files=1):
                    empty_frame.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
        with timer.phase("write_bucket_parquet_close", rows=total_rows, buckets=1, files=1):
            if writer is not None:
                writer.close()
                writer = None
    finally:
        if writer is not None:
            writer.close()
    return FactorBucketWriteResult(
        manifest_rows=manifest_rows,
        timing=timer.as_dict(),
    )


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


def _write_factor_generation_timing(
    *,
    out_root: Path,
    timer: FactorGenerationTimer,
    timing_output_path: str | Path | None = None,
) -> Path:
    path = Path(timing_output_path) if timing_output_path else out_root / DEFAULT_FACTOR_GENERATION_TIMING_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = timer.as_dict()
    summary["artifact"] = {
        "path": str(path),
        "default_filename": DEFAULT_FACTOR_GENERATION_TIMING_FILENAME,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return path


def _generate_factor_store_from_bucket_source(
    *,
    parquet_dir: str,
    output_dir: str,
    workers: int,
    label_horizons: list[int],
    data_source: str | None,
    auto_rebuild_stale_tushare_source: bool,
    timing_output_path: str | Path | None = None,
    timer: FactorGenerationTimer | None = None,
) -> dict[str, Any]:
    timer = timer or FactorGenerationTimer()
    pdir = Path(parquet_dir)
    out_root = Path(output_dir)
    bucket_root = out_root / BUCKET_DIRNAME
    manifest_path = _bucket_manifest_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bucket_root.mkdir(parents=True, exist_ok=True)

    with timer.phase("list_source_buckets"):
        source_bucket_paths = list_bucket_paths(pdir)
    if not source_bucket_paths:
        raise FileNotFoundError(f"No source bucket shards found in {pdir}")

    with timer.phase("resolve_feature_names"):
        feature_names = get_full_factor_space_feature_names(data_source=data_source)
    with timer.phase("load_source_metadata", files=1):
        source_meta = load_source_store_metadata(pdir) or {}
    source_schema_validation: dict[str, Any] = {"validated": False, "reason": "not_required"}
    if data_source == "tushare":
        try:
            with timer.phase("source_schema_validation", buckets=len(source_bucket_paths)):
                source_schema_validation = _validate_tushare_bucket_source_schema(
                    source_bucket_paths,
                    source_meta=source_meta,
                )
        except ValueError:
            if not auto_rebuild_stale_tushare_source:
                raise
            with timer.phase("stale_tushare_source_rebuild"):
                rebuild_meta = _rebuild_default_tushare_packed_source_if_stale(pdir, workers=workers)
            if rebuild_meta is None:
                raise
            with timer.phase("list_source_buckets"):
                source_bucket_paths = list_bucket_paths(pdir)
            with timer.phase("load_source_metadata", files=1):
                source_meta = load_source_store_metadata(pdir) or {}
            with timer.phase("source_schema_validation", buckets=len(source_bucket_paths)):
                source_schema_validation = _validate_tushare_bucket_source_schema(
                    source_bucket_paths,
                    source_meta=source_meta,
                )
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
            result = _write_factor_bucket_from_source_bucket_worker(
                str(source_bucket_path),
                output_bucket_root=str(bucket_root),
                label_horizons=label_horizons,
                feature_names=feature_names,
                data_source=data_source,
            )
            written_manifest_rows.extend(result.manifest_rows)
            timer.merge(result.timing)
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
                result = future.result()
                written_manifest_rows.extend(result.manifest_rows)
                timer.merge(result.timing)
                pbar.update(1)
            pbar.close()

    with timer.phase("manifest_build_write", files=1):
        written_manifest = pd.DataFrame(written_manifest_rows)
        if written_manifest.empty:
            raise RuntimeError("Factor-store rebuild from bucket source produced no rows.")
        written_manifest = written_manifest.sort_values(["bucket_id", "symbol"]).reset_index(drop=True)
        written_manifest.to_parquet(manifest_path, index=False, engine="pyarrow", compression="zstd")

    active_bucket_ids = sorted(int(value) for value in written_manifest["bucket_id"].drop_duplicates().tolist())
    with timer.phase("stale_bucket_cleanup"):
        for path in bucket_root.glob("part-*.parquet"):
            if extract_bucket_id_from_path(path) not in active_bucket_ids:
                path.unlink(missing_ok=True)

    total_rows = int(written_manifest["row_count"].sum())
    with timer.phase("available_dates_scan"):
        available_dates = _compute_available_dates_from_shards(bucket_root)
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
            **(_get_tushare_source_layout_assumptions() if data_source == "tushare" else {}),
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
        "available_dates": available_dates,
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
    timing_path = out_root / DEFAULT_FACTOR_GENERATION_TIMING_FILENAME if timing_output_path is None else Path(timing_output_path)
    metadata["timing"] = {
        **timer.as_dict(),
        "timing_path": str(timing_path),
    }
    with timer.phase("metadata_write", files=1):
        with open(out_root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    timing_path = _write_factor_generation_timing(
        out_root=out_root,
        timer=timer,
        timing_output_path=timing_output_path,
    )
    metadata["timing"] = {
        **timer.as_dict(),
        "timing_path": str(timing_path),
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
    timing_output_path: str | Path | None = None,
) -> dict[str, Any]:
    timer = FactorGenerationTimer()
    pdir = Path(parquet_dir)
    with timer.phase("detect_source_layout"):
        source_storage_layout = detect_source_storage_layout(pdir)
    with timer.phase("resolve_label_horizons"):
        label_horizons = resolve_label_horizons({"label": {"horizons": label_horizons}} if label_horizons is not None else {})
    if source_storage_layout == "bucket_shards":
        return _generate_factor_store_from_bucket_source(
            parquet_dir=parquet_dir,
            output_dir=output_dir,
            workers=workers,
            label_horizons=label_horizons,
            data_source=data_source,
            auto_rebuild_stale_tushare_source=not incremental,
            timing_output_path=timing_output_path,
            timer=timer,
        )

    out_root = Path(output_dir)
    bucket_root = out_root / BUCKET_DIRNAME
    manifest_path = _bucket_manifest_path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    bucket_root.mkdir(parents=True, exist_ok=True)
    bucket_count = max(1, int(bucket_count))

    with timer.phase("list_symbol_parquets"):
        files = sorted(pdir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pdir}")

    if data_source == "tushare":
        print("[0/3] Ensuring Tushare industry context cache...")
        with timer.phase("tushare_context_cache"):
            _ensure_tushare_industry_context_cache(pdir)

    with timer.phase("resolve_feature_names"):
        feature_names = get_full_factor_space_feature_names(data_source=data_source)
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    workers = max(1, int(workers))
    with timer.phase("load_existing_metadata", files=1):
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
        and (
            data_source != "tushare"
            or _extract_tushare_event_availability_policy(existing_meta) == TUSHARE_EVENT_AVAILABILITY_POLICY
        )
    )
    if can_reuse_manifest:
        with timer.phase("load_existing_manifest", files=1):
            existing_manifest = _load_bucket_manifest(manifest_path)
        if not existing_manifest.empty:
            existing_manifest_lookup = {
                str(row["source_path"]): row
                for row in existing_manifest.to_dict(orient="records")
                if row.get("source_path")
            }

    print(f"[1/3] Planning factor-store build from {len(files)} parquet files (workers={workers})...")
    bucket_to_files: dict[int, list[Path]] = {}
    dirty_buckets: set[int] = set()
    current_source_paths: set[str] = set()
    with timer.phase("planning", files=len(files)):
        t0 = time.perf_counter()
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
                result = _write_factor_bucket_worker(
                    [str(path) for path in bucket_to_files.get(bucket_id, [])],
                    bucket_id=bucket_id,
                    bucket_path=str(_bucket_path(bucket_root, bucket_id)),
                    label_horizons=label_horizons,
                    feature_names=feature_names,
                    data_source=data_source,
                )
                written_manifest_rows.extend(result.manifest_rows)
                timer.merge(result.timing)
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
                    result = fut.result()
                    written_manifest_rows.extend(result.manifest_rows)
                    timer.merge(result.timing)
                    pbar.update(1)
                    elapsed = time.perf_counter() - t1
                    speed = idx / elapsed if elapsed > 0 else 0.0
                    eta = (len(buckets_to_recompute) - idx) / speed if speed > 0 else float("inf")
                    pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
                pbar.close()
    else:
        print("[2/3] Writing Parquet buckets skipped: all source files reused.")

    print("[3/3] Finalizing factor-store metadata...")
    with timer.phase("manifest_build_write", files=1):
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
    with timer.phase("available_dates_scan"):
        available_dates = _compute_available_dates_from_shards(bucket_root)

    metadata = {
        "storage_format": "parquet",
        "storage_layout": "bucket_shards",
        "factor_space": FULL_FACTOR_SPACE_NAME,
        "data_source": data_source or "",
        "source_parquet_dir": str(pdir),
        "source_layout_assumptions": {
            **(_get_tushare_source_layout_assumptions() if data_source == "tushare" else {}),
        },
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
        "available_dates": available_dates,
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
    timing_path = out_root / DEFAULT_FACTOR_GENERATION_TIMING_FILENAME if timing_output_path is None else Path(timing_output_path)
    metadata["timing"] = {
        **timer.as_dict(),
        "timing_path": str(timing_path),
    }
    with timer.phase("metadata_write", files=1):
        with open(out_root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    timing_path = _write_factor_generation_timing(
        out_root=out_root,
        timer=timer,
        timing_output_path=timing_output_path,
    )
    metadata["timing"] = {
        **timer.as_dict(),
        "timing_path": str(timing_path),
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[3/3] Done. Parquet factor store saved to: {out_root}")
    return metadata


@dataclass(frozen=True)
class FactorGenerationRuntime:
    cfg: dict[str, Any]
    data_source: str
    parquet_dir: str
    output_dir: str
    label_horizons: list[int]


def _parse_label_horizons_arg(raw: str) -> list[int]:
    horizons: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        horizons.append(int(text))
    return resolve_label_horizons({"label": {"horizons": horizons}})


def _resolve_factor_generation_runtime(args: argparse.Namespace) -> FactorGenerationRuntime:
    cfg = load_runtime_config(args.config) if args.config else {}
    cfg.setdefault("runtime", {})
    if args.config:
        cfg["runtime"]["config_path"] = args.config
    if args.data_source:
        cfg.setdefault("data", {})
        cfg["data"]["source"] = args.data_source
    if args.feature_profile:
        cfg.setdefault("features", {})
        cfg["features"]["profile"] = args.feature_profile
    apply_override_args(cfg, getattr(args, "set_overrides", None))

    label_horizons = resolve_label_horizons(cfg)
    if args.label_horizons:
        label_horizons = _parse_label_horizons_arg(args.label_horizons)

    data_source = resolve_data_source_name(cfg)
    feature_profile = resolve_feature_profile(cfg)
    return FactorGenerationRuntime(
        cfg=cfg,
        data_source=data_source,
        parquet_dir=args.parquet_dir or resolve_source_parquet_dir(cfg),
        output_dir=args.output_dir or str(feature_profile["factor_store_dir"]),
        label_horizons=label_horizons,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the unified full-factor Parquet store from parquet.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Experiment config path for factor-store output settings.",
    )
    parser.add_argument(
        "--data-source",
        choices=SUPPORTED_DATA_SOURCES,
        help="Named data source for default parquet/factor-store resolution.",
    )
    parser.add_argument(
        "--feature-profile",
        help="Resolve factor-store output path through a named feature profile.",
    )
    parser.add_argument("--parquet-dir", default=None, help="Input parquet directory.")
    parser.add_argument("--output-dir", default=None, help="Output factor-store directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for counting/writing.")
    parser.add_argument(
        "--set",
        action="append",
        dest="set_overrides",
        help="Generic dotted override in key=value form, for example features.factor_store_dir=data/factor_store/custom.",
    )
    parser.add_argument(
        "--label-horizons",
        help="Comma-separated label horizons to materialize, for example '1,5,10,20'. If omitted, use config/defaults.",
    )
    parser.add_argument(
        "--timing-output",
        default=None,
        help="Optional JSON timing summary path. Defaults to <output-dir>/factor_generation_timing.json.",
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
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    validate_default_dimensions()
    try:
        runtime = _resolve_factor_generation_runtime(args)
    except ValueError as exc:
        parser.error(str(exc))
    family_counts = get_factor_family_counts(data_source=runtime.data_source)

    print(
        "storage_format=parquet, "
        f"data_source={runtime.data_source}, "
        f"parquet_dir={runtime.parquet_dir}, "
        f"factor_space={FULL_FACTOR_SPACE_NAME}, "
        f"output={runtime.output_dir}, "
        f"incremental={args.incremental}, "
        f"label_horizons={runtime.label_horizons}"
    )
    print(
        "factor_groups="
        + ", ".join(
            f"{name}:{count}"
            for name, count in family_counts.items()
        )
    )
    generate_factor_store(
        parquet_dir=runtime.parquet_dir,
        output_dir=runtime.output_dir,
        workers=max(1, int(args.workers)),
        incremental=args.incremental,
        label_horizons=runtime.label_horizons,
        data_source=runtime.data_source,
        timing_output_path=args.timing_output,
    )


if __name__ == "__main__":
    main()
