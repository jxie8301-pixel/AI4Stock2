from __future__ import annotations

import argparse
from collections import deque
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tushare as ts
from tqdm import tqdm

TS_ROOT = Path("data/tushare")
RAW_ROOT = TS_ROOT / "raw"
RAW_META_DIR = RAW_ROOT / "meta"
RAW_DAILY_DIR = RAW_ROOT / "daily"
RAW_DAILY_BASIC_DIR = RAW_ROOT / "daily_basic"
RAW_ADJ_FACTOR_DIR = RAW_ROOT / "adj_factor"
RAW_STK_LIMIT_DIR = RAW_ROOT / "stk_limit"
RAW_FINA_INDICATOR_DIR = RAW_ROOT / "fina_indicator"
PROCESSED_DIR = TS_ROOT / "processed" / "combined"

SYMBOL_CACHE_PATH = RAW_META_DIR / "symbol_cache.parquet"
TRADE_CALENDAR_PATH = RAW_META_DIR / "trade_calendar.parquet"
SYMBOL_LIFECYCLE_PATH = RAW_META_DIR / "symbol_lifecycle.parquet"
BACKFILL_EXHAUSTION_PATH = RAW_META_DIR / "backfill_exhaustion.parquet"

DEFAULT_TOKEN_ENV = "TUSHARE_TOKEN"
DEFAULT_START_DATE = "19900101"
MAX_CONSECUTIVE_FAILURES = 10
PRECHECK_WORKERS = 16
MARKET_CHUNK_YEARS = 12
STK_LIMIT_COVERAGE_START = pd.Timestamp("2007-01-04")
RATE_LIMIT_COOLDOWN_SECONDS = 60.0

STOCK_BASIC_API_FIELDS = [
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "fullname",
    "enname",
    "cnspell",
    "market",
    "exchange",
    "curr_type",
    "list_status",
    "list_date",
    "delist_date",
    "is_hs",
    "act_name",
    "act_ent",
]

TRADE_CAL_API_FIELDS = ["exchange", "cal_date", "is_open", "pretrade_date"]

DAILY_API_FIELDS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]

DAILY_RAW_COLS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "volume",
    "amount",
]

DAILY_BASIC_API_FIELDS = [
    "ts_code",
    "trade_date",
    "close",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "total_share",
    "float_share",
    "free_share",
    "total_mv",
    "circ_mv",
]

ADJ_FACTOR_API_FIELDS = ["ts_code", "trade_date", "adj_factor"]
STK_LIMIT_API_FIELDS = ["ts_code", "trade_date", "pre_close", "up_limit", "down_limit"]
FINA_INDICATOR_API_FIELDS = [
    "ts_code",
    "ann_date",
    "end_date",
    "eps",
    "dt_eps",
    "bps",
    "ocfps",
    "roe",
    "roe_dt",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "debt_to_assets",
    "q_eps",
    "q_dtprofit",
    "q_roe",
    "q_dt_roe",
    "tr_yoy",
    "or_yoy",
    "op_yoy",
    "netprofit_yoy",
    "ocf_yoy",
]

SYMBOL_CACHE_COLS = [
    "local_symbol",
    "ts_code",
    "symbol",
    "name",
    "area",
    "industry",
    "fullname",
    "enname",
    "cnspell",
    "market",
    "exchange",
    "curr_type",
    "list_date",
    "delist_date",
    "list_status",
    "is_hs",
    "act_name",
    "act_ent",
    "fetched_at",
]

SYMBOL_LIFECYCLE_COLS = [
    "local_symbol",
    "ts_code",
    "list_status",
    "list_date",
    "delist_date",
    "lifecycle_status",
    "effective_end_date",
    "earliest_daily_date",
    "earliest_daily_basic_date",
    "earliest_adj_factor_date",
    "latest_daily_date",
    "latest_daily_basic_date",
    "latest_adj_factor_date",
    "latest_stk_limit_date",
    "latest_processed_date",
    "last_checked_at",
]

BACKFILL_EXHAUSTION_COLS = [
    "stage_name",
    "local_symbol",
    "expected_start_date",
    "observed_earliest_date",
    "recorded_at",
]

TS_PROCESSED_CANONICAL_COLS = [
    "date",
    "symbol",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
    "turnover_free",
    "volume_ratio",
    "total_mv",
    "circ_mv",
    "total_share",
    "circ_share",
    "free_share",
    "pe",
    "pe_ttm",
    "pb",
    "ps",
    "ps_ttm",
    "dv_ratio",
    "dv_ttm",
    "limit_pre_close",
    "up_limit",
    "down_limit",
    "adj_factor",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "raw_pre_close",
]

UINT32_MAX = np.iinfo(np.uint32).max
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

for directory in [
    RAW_META_DIR,
    RAW_DAILY_DIR,
    RAW_DAILY_BASIC_DIR,
    RAW_ADJ_FACTOR_DIR,
    RAW_STK_LIMIT_DIR,
    RAW_FINA_INDICATOR_DIR,
    PROCESSED_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TaskResult:
    symbol: str
    ok: bool
    detail: str


@dataclass(slots=True)
class StageTaskResult:
    symbol: str
    ok: bool
    detail: str
    payload: Any | None = None


@dataclass(slots=True)
class UpdateResult:
    status: str
    changed: bool
    latest: pd.Timestamp | None
    earliest: pd.Timestamp | None = None
    backfill_exhausted: bool = False


@dataclass(slots=True)
class SymbolState:
    symbol: str
    daily_earliest: pd.Timestamp | None
    daily_basic_earliest: pd.Timestamp | None
    adj_factor_earliest: pd.Timestamp | None
    daily_latest: pd.Timestamp | None
    daily_basic_latest: pd.Timestamp | None
    adj_factor_latest: pd.Timestamp | None
    stk_limit_latest: pd.Timestamp | None
    processed_latest: pd.Timestamp | None


@dataclass(slots=True)
class StageUpdatePlan:
    stage_name: str
    latest_by_symbol: dict[str, pd.Timestamp | None]
    earliest_by_symbol: dict[str, pd.Timestamp | None]
    pending_symbols: list[str]
    ready_outputs: dict[str, UpdateResult]


_THREAD_STATE = threading.local()


class EndpointRateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0.0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed - now
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval_seconds


def ts_symbol_to_local(symbol: str) -> str:
    return str(symbol).split(".", 1)[0].strip()


def local_symbol_to_ts(symbol: str) -> str:
    code = str(symbol).strip()
    if "." in code:
        return code
    if code.startswith(("600", "601", "603", "605", "688")):
        return f"{code}.SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{code}.SZ"
    raise ValueError(f"unsupported A-share code: {symbol}")


def _normalize_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.tz_localize(None).dt.normalize()
    return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce").dt.normalize()


def _normalize_symbol_date_frame(
    df: pd.DataFrame,
    date_column: str = "trade_date",
    symbol_column: str = "ts_code",
) -> pd.DataFrame:
    out = df.copy()
    out[date_column] = _normalize_date_series(out[date_column])
    out = out.dropna(subset=[date_column])
    dedupe_cols = [symbol_column, date_column] if symbol_column in out.columns else [date_column]
    out = out.drop_duplicates(subset=dedupe_cols, keep="last")
    sort_cols = [date_column]
    if symbol_column in out.columns:
        sort_cols.append(symbol_column)
    return out.sort_values(sort_cols).reset_index(drop=True)


def _optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_float_dtype(series.dtype):
            out[col] = series.astype("float32")
            continue
        if not pd.api.types.is_integer_dtype(series.dtype):
            continue
        if getattr(series, "isna", lambda: pd.Series([], dtype=bool))().any():
            continue
        col_min = int(series.min())
        col_max = int(series.max())
        if col_min >= 0 and col_max <= UINT32_MAX:
            out[col] = series.astype("uint32")
        elif INT32_MIN <= col_min <= INT32_MAX and INT32_MIN <= col_max <= INT32_MAX:
            out[col] = series.astype("int32")
    return out


def save_optimized_parquet(df: pd.DataFrame, path: Path) -> None:
    optimized = _optimize_numeric_dtypes(df)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        optimized.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.stat().st_size == 0:
            return None
    except OSError:
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return None


def _parquet_missing_columns(path: Path, required_columns: list[str]) -> list[str]:
    if not path.exists():
        return list(required_columns)
    try:
        meta = pq.read_metadata(str(path))
        present = set(meta.schema.names)
    except Exception:
        frame = _read_parquet_safe(path)
        if frame is None:
            return list(required_columns)
        present = set(frame.columns)
    return [col for col in required_columns if col not in present]


def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return _read_parquet_safe(path)


def infer_latest_date(path: Path, date_column: str = "trade_date") -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        meta = pq.read_metadata(str(path))
        if date_column not in meta.schema.names:
            return None
        index = meta.schema.names.index(date_column)
        row_group = meta.row_group(meta.num_row_groups - 1)
        stats = row_group.column(index).statistics
        if stats and stats.max is not None:
            return pd.Timestamp(stats.max).tz_localize(None).normalize()
    except Exception:
        pass

    frame = _read_parquet_safe(path, columns=[date_column])
    if frame is None or frame.empty or date_column not in frame.columns:
        return None
    return _normalize_date_series(frame[date_column]).max()


def infer_earliest_date(path: Path, date_column: str = "trade_date") -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        meta = pq.read_metadata(str(path))
        if date_column not in meta.schema.names:
            return None
        index = meta.schema.names.index(date_column)
        row_group = meta.row_group(0)
        stats = row_group.column(index).statistics
        if stats and stats.min is not None:
            return pd.Timestamp(stats.min).tz_localize(None).normalize()
    except Exception:
        pass

    frame = _read_parquet_safe(path, columns=[date_column])
    if frame is None or frame.empty or date_column not in frame.columns:
        return None
    return _normalize_date_series(frame[date_column]).min()


def _coerce_numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    exclude = exclude or set()
    for col in out.columns:
        if col in exclude:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_symbol_cache_frame(
    df: pd.DataFrame,
    list_status: str,
    fetched_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    required = {"ts_code", "symbol", "name", "market", "list_date", "list_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tushare symbol cache missing required columns: {sorted(missing)}")

    out = df.copy()
    out["ts_code"] = out["ts_code"].astype(str).str.strip()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["local_symbol"] = out["symbol"]
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    for field in [
        "area",
        "industry",
        "fullname",
        "enname",
        "cnspell",
        "market",
        "exchange",
        "curr_type",
        "is_hs",
        "act_name",
        "act_ent",
    ]:
        out[field] = (
            out[field].fillna("").astype(str).str.strip()
            if field in out.columns
            else pd.Series("", index=out.index, dtype=object)
        )
    out["list_date"] = pd.to_datetime(out["list_date"], format="%Y%m%d", errors="coerce")
    if "delist_date" in out.columns:
        out["delist_date"] = pd.to_datetime(out["delist_date"], format="%Y%m%d", errors="coerce")
    else:
        out["delist_date"] = pd.NaT
    out["list_status"] = list_status
    out["fetched_at"] = fetched_at or pd.Timestamp.now().normalize()
    out = out[
        out["local_symbol"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")
    ]
    out = (
        out.drop_duplicates(subset=["local_symbol"], keep="last")
        .sort_values("local_symbol")
        .reset_index(drop=True)
    )
    return out.reindex(columns=SYMBOL_CACHE_COLS)


def save_symbol_cache(df: pd.DataFrame) -> None:
    save_optimized_parquet(df.reindex(columns=SYMBOL_CACHE_COLS), SYMBOL_CACHE_PATH)


def load_symbol_cache() -> pd.DataFrame | None:
    frame = _read_parquet_safe(SYMBOL_CACHE_PATH)
    if frame is None or frame.empty:
        return None
    if _parquet_missing_columns(SYMBOL_CACHE_PATH, SYMBOL_CACHE_COLS):
        return None
    out = frame.copy()
    for col in ["list_date", "delist_date", "fetched_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.reindex(columns=SYMBOL_CACHE_COLS)


def save_symbol_lifecycle(df: pd.DataFrame) -> None:
    save_optimized_parquet(df.reindex(columns=SYMBOL_LIFECYCLE_COLS), SYMBOL_LIFECYCLE_PATH)


def load_symbol_lifecycle() -> pd.DataFrame | None:
    frame = _read_parquet_safe(SYMBOL_LIFECYCLE_PATH)
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    for col in [
        "list_date",
        "delist_date",
        "effective_end_date",
        "earliest_daily_date",
        "earliest_daily_basic_date",
        "earliest_adj_factor_date",
        "latest_daily_date",
        "latest_daily_basic_date",
        "latest_adj_factor_date",
        "latest_stk_limit_date",
        "latest_processed_date",
        "last_checked_at",
    ]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.reindex(columns=SYMBOL_LIFECYCLE_COLS)


def configure_tushare(token_env: str = DEFAULT_TOKEN_ENV) -> None:
    token = os.environ.get(token_env, "").strip()
    if token:
        ts.set_token(token)


def get_tushare_client():
    client = getattr(_THREAD_STATE, "tushare_pro", None)
    if client is None:
        client = ts.pro_api()
        _THREAD_STATE.tushare_pro = client
    return client


def _ts_call(endpoint_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return func(*args, **kwargs)


def _is_tushare_rate_limit_detail(detail: str) -> bool:
    return "每分钟最多访问该接口" in detail


def fetch_stock_basic_by_status(list_status: str) -> pd.DataFrame:
    pro = get_tushare_client()
    frame = pro.stock_basic(
        exchange="",
        list_status=list_status,
        fields=",".join(STOCK_BASIC_API_FIELDS),
    )
    if frame is None:
        return pd.DataFrame(columns=STOCK_BASIC_API_FIELDS)
    return frame.reindex(columns=STOCK_BASIC_API_FIELDS)


def refresh_symbol_cache() -> pd.DataFrame:
    fetched_at = pd.Timestamp.now().normalize()
    active = normalize_symbol_cache_frame(fetch_stock_basic_by_status("L"), "L", fetched_at)
    delisted = normalize_symbol_cache_frame(fetch_stock_basic_by_status("D"), "D", fetched_at)
    merged = (
        pd.concat([active, delisted], ignore_index=True)
        .drop_duplicates(subset=["local_symbol"], keep="first")
        .sort_values("local_symbol")
        .reset_index(drop=True)
    )
    save_symbol_cache(merged)
    print(
        f"[*] Tushare symbol cache refreshed: total={len(merged)}, "
        f"active={(merged['list_status'] == 'L').sum()}, delisted={(merged['list_status'] == 'D').sum()}"
    )
    return merged


def save_trade_calendar(df: pd.DataFrame) -> None:
    save_optimized_parquet(df, TRADE_CALENDAR_PATH)


def refresh_trade_calendar(target_end_date: pd.Timestamp) -> pd.DataFrame:
    pro = get_tushare_client()
    start_date = (target_end_date - pd.Timedelta(days=370)).strftime("%Y%m%d")
    end_date = target_end_date.strftime("%Y%m%d")
    frame = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        fields=",".join(TRADE_CAL_API_FIELDS),
    )
    if frame is None or frame.empty:
        raise RuntimeError("Tushare trade_cal returned empty data")
    out = frame.copy()
    out = out.reindex(columns=TRADE_CAL_API_FIELDS)
    out["cal_date"] = pd.to_datetime(out["cal_date"], format="%Y%m%d", errors="coerce")
    out["pretrade_date"] = pd.to_datetime(out["pretrade_date"], format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["cal_date"]).sort_values("cal_date").reset_index(drop=True)
    save_trade_calendar(out)
    return out


def resolve_latest_trading_date(target_end_date: pd.Timestamp) -> pd.Timestamp:
    cached = _read_parquet_safe(TRADE_CALENDAR_PATH)
    if cached is not None and not cached.empty and "cal_date" in cached.columns:
        cached = cached.copy()
        cached["cal_date"] = pd.to_datetime(cached["cal_date"], errors="coerce")
        usable = cached[
            (cached["cal_date"] <= target_end_date)
            & (pd.to_numeric(cached["is_open"], errors="coerce") == 1)
        ]
        if not usable.empty and usable["cal_date"].max() >= target_end_date - pd.Timedelta(days=40):
            return usable["cal_date"].max().normalize()
    frame = refresh_trade_calendar(target_end_date)
    usable = frame[(frame["cal_date"] <= target_end_date) & (frame["is_open"] == 1)]
    if usable.empty:
        raise RuntimeError(f"no trading date found on or before {target_end_date.date()}")
    return usable["cal_date"].max().normalize()


def list_local_symbols() -> list[str]:
    roots = [
        RAW_DAILY_DIR,
        RAW_DAILY_BASIC_DIR,
        RAW_ADJ_FACTOR_DIR,
        RAW_STK_LIMIT_DIR,
        RAW_FINA_INDICATOR_DIR,
        PROCESSED_DIR,
    ]
    symbols = {path.stem for root in roots for path in root.glob("*.parquet")}
    return sorted(symbols)


def resolve_all_symbols(refresh_live: bool = False) -> list[str]:
    cached = None if refresh_live else load_symbol_cache()
    if cached is None:
        cached = refresh_symbol_cache()
    else:
        print(f"[*] Using cached Tushare symbol list ({len(cached)} symbols).")
    local_symbols = set(list_local_symbols())
    cached_symbols = set(cached["local_symbol"].astype(str).tolist())
    resolved = sorted(local_symbols | cached_symbols)
    if local_symbols:
        print(
            f"[*] Tushare all-symbol set: {len(resolved)} "
            f"(local={len(local_symbols)}, cached={len(cached_symbols)})"
        )
    return resolved


def resolve_incremental_symbols(refresh_live: bool = False) -> list[str]:
    local_symbols = set(list_local_symbols())
    cached = None if refresh_live else load_symbol_cache()
    if cached is None:
        cached = refresh_symbol_cache()
    cached_symbols = set(cached["local_symbol"].astype(str).tolist())
    resolved = sorted(local_symbols | cached_symbols)
    print(
        f"[*] Tushare incremental symbol set: {len(resolved)} "
        f"(local={len(local_symbols)}, cached={len(cached_symbols)})"
    )
    return resolved


def _normalize_optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    ts_value = pd.Timestamp(value)
    if pd.isna(ts_value):
        return None
    if ts_value.tzinfo is not None:
        ts_value = ts_value.tz_localize(None)
    return ts_value.normalize()


def _latest_trade_date(state: SymbolState) -> pd.Timestamp | None:
    latest_dates = [
        item for item in [state.daily_latest, state.daily_basic_latest] if item is not None
    ]
    if not latest_dates:
        return None
    return min(latest_dates)


def load_backfill_exhaustion() -> pd.DataFrame:
    frame = _read_parquet_safe(BACKFILL_EXHAUSTION_PATH)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=BACKFILL_EXHAUSTION_COLS)
    out = frame.copy()
    for col in ["expected_start_date", "observed_earliest_date", "recorded_at"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out.reindex(columns=BACKFILL_EXHAUSTION_COLS)


def save_backfill_exhaustion(df: pd.DataFrame) -> None:
    save_optimized_parquet(df.reindex(columns=BACKFILL_EXHAUSTION_COLS), BACKFILL_EXHAUSTION_PATH)


def build_backfill_exhaustion_lookup(
    frame: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, pd.Timestamp | None]]:
    lookup: dict[tuple[str, str], dict[str, pd.Timestamp | None]] = {}
    if frame.empty:
        return lookup
    for _, row in frame.iterrows():
        stage_name = str(row.get("stage_name") or "").strip()
        symbol = str(row.get("local_symbol") or "").strip()
        if not stage_name or not symbol:
            continue
        lookup[(stage_name, symbol)] = {
            "expected_start_date": _normalize_optional_timestamp(row.get("expected_start_date")),
            "observed_earliest_date": _normalize_optional_timestamp(row.get("observed_earliest_date")),
            "recorded_at": _normalize_optional_timestamp(row.get("recorded_at")),
        }
    return lookup


def merge_backfill_exhaustion_records(
    existing: pd.DataFrame,
    updates: list[dict[str, Any]],
    removals: set[tuple[str, str]],
) -> pd.DataFrame:
    out = existing.copy()
    if removals and not out.empty:
        removal_keys = {(stage, symbol) for stage, symbol in removals}
        keep_mask = [
            (str(row.get("stage_name") or "").strip(), str(row.get("local_symbol") or "").strip())
            not in removal_keys
            for _, row in out.iterrows()
        ]
        out = out.loc[keep_mask].copy()
    if updates:
        updates_df = pd.DataFrame(updates).reindex(columns=BACKFILL_EXHAUSTION_COLS)
        if out.empty:
            out = updates_df.copy()
        else:
            out = pd.concat([out, updates_df], ignore_index=True)
        out = (
            out.drop_duplicates(subset=["stage_name", "local_symbol"], keep="last")
            .sort_values(["stage_name", "local_symbol"])
            .reset_index(drop=True)
        )
    elif not out.empty:
        out = out.sort_values(["stage_name", "local_symbol"]).reset_index(drop=True)
    return out.reindex(columns=BACKFILL_EXHAUSTION_COLS)


def is_backfill_satisfied(
    stage_name: str,
    symbol: str,
    expected_start_date: pd.Timestamp | None,
    earliest: pd.Timestamp | None,
    exhaustion_lookup: dict[tuple[str, str], dict[str, pd.Timestamp | None]] | None = None,
) -> bool:
    if expected_start_date is None:
        return True
    if earliest is not None and earliest <= expected_start_date:
        return True
    if exhaustion_lookup is None:
        return False
    record = exhaustion_lookup.get((stage_name, symbol))
    if not record:
        return False
    recorded_expected = _normalize_optional_timestamp(record.get("expected_start_date"))
    return recorded_expected == expected_start_date


def resolve_symbol_lifecycle_status(
    target_end_date: pd.Timestamp,
    list_status: str | None = None,
    list_date: Any = None,
    delist_date: Any = None,
    last_trade_date: Any = None,
    latest_adj_factor_date: Any = None,
    latest_stk_limit_date: Any = None,
) -> str:
    list_ts = _normalize_optional_timestamp(list_date)
    delist_ts = _normalize_optional_timestamp(delist_date)
    last_trade_ts = _normalize_optional_timestamp(last_trade_date)
    latest_adj_ts = _normalize_optional_timestamp(latest_adj_factor_date)
    latest_stk_limit_ts = _normalize_optional_timestamp(latest_stk_limit_date)
    if list_ts is not None and list_ts > target_end_date:
        return "not_yet_listed"
    if list_status == "D" and delist_ts is not None and delist_ts <= target_end_date:
        return "delisted"
    if (
        last_trade_ts is not None
        and last_trade_ts < target_end_date
        and latest_adj_ts is not None
        and latest_adj_ts >= target_end_date
        and (
            target_end_date < STK_LIMIT_COVERAGE_START
            or (latest_stk_limit_ts is not None and latest_stk_limit_ts >= target_end_date)
        )
    ):
        return "suspended"
    return "active"


def resolve_effective_end_date(
    target_end_date: pd.Timestamp,
    list_status: str | None = None,
    list_date: Any = None,
    delist_date: Any = None,
    last_trade_date: Any = None,
    latest_adj_factor_date: Any = None,
    latest_stk_limit_date: Any = None,
) -> pd.Timestamp:
    lifecycle_status = resolve_symbol_lifecycle_status(
        target_end_date,
        list_status=list_status,
        list_date=list_date,
        delist_date=delist_date,
        last_trade_date=last_trade_date,
        latest_adj_factor_date=latest_adj_factor_date,
        latest_stk_limit_date=latest_stk_limit_date,
    )
    if lifecycle_status == "delisted":
        delist_ts = _normalize_optional_timestamp(delist_date) or target_end_date
        trade_ts = _normalize_optional_timestamp(last_trade_date)
        if trade_ts is not None and trade_ts < delist_ts:
            return trade_ts
        return delist_ts
    if lifecycle_status == "suspended":
        return _normalize_optional_timestamp(last_trade_date) or target_end_date
    return target_end_date


def load_symbol_state(symbol: str) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        daily_earliest=infer_earliest_date(RAW_DAILY_DIR / f"{symbol}.parquet", "trade_date"),
        daily_basic_earliest=infer_earliest_date(RAW_DAILY_BASIC_DIR / f"{symbol}.parquet", "trade_date"),
        adj_factor_earliest=infer_earliest_date(RAW_ADJ_FACTOR_DIR / f"{symbol}.parquet", "trade_date"),
        daily_latest=infer_latest_date(RAW_DAILY_DIR / f"{symbol}.parquet", "trade_date"),
        daily_basic_latest=infer_latest_date(RAW_DAILY_BASIC_DIR / f"{symbol}.parquet", "trade_date"),
        adj_factor_latest=infer_latest_date(RAW_ADJ_FACTOR_DIR / f"{symbol}.parquet", "trade_date"),
        stk_limit_latest=infer_latest_date(RAW_STK_LIMIT_DIR / f"{symbol}.parquet", "trade_date"),
        processed_latest=infer_latest_date(PROCESSED_DIR / f"{symbol}.parquet", "date"),
    )


def _is_symbol_schema_complete(symbol: str, target_end_date: pd.Timestamp) -> bool:
    required_raw = [
        (RAW_DAILY_DIR / f"{symbol}.parquet", DAILY_RAW_COLS),
        (RAW_DAILY_BASIC_DIR / f"{symbol}.parquet", DAILY_BASIC_API_FIELDS),
        (RAW_ADJ_FACTOR_DIR / f"{symbol}.parquet", ADJ_FACTOR_API_FIELDS),
    ]
    if any(_parquet_missing_columns(path, columns) for path, columns in required_raw):
        return False
    if target_end_date >= STK_LIMIT_COVERAGE_START and _parquet_missing_columns(
        RAW_STK_LIMIT_DIR / f"{symbol}.parquet",
        STK_LIMIT_API_FIELDS,
    ):
        return False
    if _parquet_missing_columns(PROCESSED_DIR / f"{symbol}.parquet", TS_PROCESSED_CANONICAL_COLS):
        return False
    return True


def is_symbol_complete(
    state: SymbolState,
    target_end_date: pd.Timestamp,
    expected_start_date: pd.Timestamp | None = None,
    backfill_ready: dict[str, bool] | None = None,
) -> bool:
    if expected_start_date is not None:
        required_starts = {
            "daily": state.daily_earliest,
            "daily_basic": state.daily_basic_earliest,
            "adj_factor": state.adj_factor_earliest,
        }
        for stage_name, earliest in required_starts.items():
            if earliest is None and not (backfill_ready or {}).get(stage_name, False):
                return False
            if (
                earliest is not None
                and earliest > expected_start_date
                and not (backfill_ready or {}).get(stage_name, False)
            ):
                return False
    required = [
        state.daily_latest,
        state.daily_basic_latest,
        state.adj_factor_latest,
    ]
    if any(item is None or item < target_end_date for item in required):
        return False
    if target_end_date >= STK_LIMIT_COVERAGE_START and (
        state.stk_limit_latest is None or state.stk_limit_latest < target_end_date
    ):
        return False
    if state.processed_latest is None:
        return False
    return state.processed_latest >= state.daily_latest


def build_symbol_lifecycle_registry(
    symbols: list[str],
    target_end_date: pd.Timestamp,
    max_workers: int = PRECHECK_WORKERS,
) -> pd.DataFrame:
    if not symbols:
        empty = pd.DataFrame(columns=SYMBOL_LIFECYCLE_COLS)
        save_symbol_lifecycle(empty)
        return empty

    cached = load_symbol_cache()
    cache_by_symbol: dict[str, dict[str, Any]] = {}
    if cached is not None and not cached.empty:
        cache_by_symbol = cached.set_index("local_symbol", drop=False).to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    checked_at = pd.Timestamp.now().normalize()
    print(f"[*] Building Tushare symbol lifecycle registry for {len(symbols)} symbols...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(load_symbol_state, symbol): symbol for symbol in symbols}
        pbar = tqdm(as_completed(future_map), total=len(symbols), desc="lifecycle", unit="symbol")
        for future in pbar:
            symbol = future_map[future]
            state = future.result()
            cache_row = cache_by_symbol.get(symbol, {})
            list_status = str(cache_row.get("list_status") or "L")
            list_date = _normalize_optional_timestamp(cache_row.get("list_date"))
            delist_date = _normalize_optional_timestamp(cache_row.get("delist_date"))
            lifecycle_status = resolve_symbol_lifecycle_status(
                target_end_date,
                list_status=list_status,
                list_date=list_date,
                delist_date=delist_date,
                last_trade_date=_latest_trade_date(state),
                latest_adj_factor_date=state.adj_factor_latest,
                latest_stk_limit_date=state.stk_limit_latest,
            )
            effective_end_date = resolve_effective_end_date(
                target_end_date,
                list_status=list_status,
                list_date=list_date,
                delist_date=delist_date,
                last_trade_date=_latest_trade_date(state),
                latest_adj_factor_date=state.adj_factor_latest,
                latest_stk_limit_date=state.stk_limit_latest,
            )
            rows.append(
                {
                    "local_symbol": symbol,
                    "ts_code": cache_row.get("ts_code", local_symbol_to_ts(symbol)),
                    "list_status": list_status,
                    "list_date": list_date,
                    "delist_date": delist_date,
                    "lifecycle_status": lifecycle_status,
                    "effective_end_date": effective_end_date,
                    "earliest_daily_date": state.daily_earliest,
                    "earliest_daily_basic_date": state.daily_basic_earliest,
                    "earliest_adj_factor_date": state.adj_factor_earliest,
                    "latest_daily_date": state.daily_latest,
                    "latest_daily_basic_date": state.daily_basic_latest,
                    "latest_adj_factor_date": state.adj_factor_latest,
                    "latest_stk_limit_date": state.stk_limit_latest,
                    "latest_processed_date": state.processed_latest,
                    "last_checked_at": checked_at,
                }
            )

    registry = (
        pd.DataFrame(rows)
        .sort_values("local_symbol")
        .reset_index(drop=True)
        .reindex(columns=SYMBOL_LIFECYCLE_COLS)
    )
    save_symbol_lifecycle(registry)
    return registry


def split_symbols_by_completion(
    symbols: list[str],
    lifecycle_registry: pd.DataFrame,
    exhaustion_lookup: dict[tuple[str, str], dict[str, pd.Timestamp | None]] | None = None,
) -> tuple[list[str], list[str], dict[str, pd.Timestamp]]:
    if lifecycle_registry.empty:
        return list(symbols), [], {}

    registry_by_symbol = lifecycle_registry.set_index("local_symbol", drop=False)
    pending: list[str] = []
    completed: list[str] = []
    effective_end_dates: dict[str, pd.Timestamp] = {}

    for symbol in symbols:
        if symbol not in registry_by_symbol.index:
            pending.append(symbol)
            continue
        row = registry_by_symbol.loc[symbol]
        if str(row.get("lifecycle_status") or "") == "not_yet_listed":
            completed.append(symbol)
            continue
        effective_end = _normalize_optional_timestamp(row["effective_end_date"])
        if effective_end is None:
            pending.append(symbol)
            continue
        effective_end_dates[symbol] = effective_end
        expected_start_date = _normalize_optional_timestamp(row["list_date"])
        backfill_ready = {
            stage_name: is_backfill_satisfied(
                stage_name,
                symbol,
                expected_start_date,
                _normalize_optional_timestamp(row.get(column_name)),
                exhaustion_lookup,
            )
            for stage_name, column_name in {
                "daily": "earliest_daily_date",
                "daily_basic": "earliest_daily_basic_date",
                "adj_factor": "earliest_adj_factor_date",
            }.items()
        }
        state = SymbolState(
            symbol=symbol,
            daily_earliest=_normalize_optional_timestamp(row.get("earliest_daily_date")),
            daily_basic_earliest=_normalize_optional_timestamp(row.get("earliest_daily_basic_date")),
            adj_factor_earliest=_normalize_optional_timestamp(row.get("earliest_adj_factor_date")),
            daily_latest=_normalize_optional_timestamp(row["latest_daily_date"]),
            daily_basic_latest=_normalize_optional_timestamp(row["latest_daily_basic_date"]),
            adj_factor_latest=_normalize_optional_timestamp(row["latest_adj_factor_date"]),
            stk_limit_latest=_normalize_optional_timestamp(row["latest_stk_limit_date"]),
            processed_latest=_normalize_optional_timestamp(row["latest_processed_date"]),
        )
        if is_symbol_complete(
            state,
            effective_end,
            expected_start_date=expected_start_date,
            backfill_ready=backfill_ready,
        ) and _is_symbol_schema_complete(symbol, effective_end):
            completed.append(symbol)
        else:
            pending.append(symbol)
    return pending, completed, effective_end_dates


def precheck_pending_symbols(
    symbols: list[str],
    target_end_date: pd.Timestamp,
    max_workers: int = PRECHECK_WORKERS,
) -> tuple[list[str], list[str], dict[str, pd.Timestamp], pd.DataFrame]:
    if not symbols:
        empty = pd.DataFrame(columns=SYMBOL_LIFECYCLE_COLS)
        return [], [], {}, empty
    print(
        f"[*] Scanning {len(symbols)} Tushare symbols for lifecycle/completed state with {max_workers} workers..."
    )
    lifecycle_registry = build_symbol_lifecycle_registry(
        symbols,
        target_end_date=target_end_date,
        max_workers=max_workers,
    )
    exhaustion_lookup = build_backfill_exhaustion_lookup(load_backfill_exhaustion())
    pending, completed, effective_end_dates = split_symbols_by_completion(
        symbols, lifecycle_registry, exhaustion_lookup=exhaustion_lookup
    )
    print(f"[*] Tushare precheck done. completed={len(completed)}, pending={len(pending)}")
    return pending, completed, effective_end_dates, lifecycle_registry


def precheck_stage_updates(
    symbols: list[str],
    stage_name: str,
    stage_dir: Path,
    target_end_date: pd.Timestamp,
    effective_end_dates: dict[str, pd.Timestamp] | None = None,
    expected_start_dates: dict[str, pd.Timestamp] | None = None,
    coverage_start_date: pd.Timestamp | None = None,
    exhaustion_lookup: dict[tuple[str, str], dict[str, pd.Timestamp | None]] | None = None,
    required_columns: list[str] | None = None,
    max_workers: int = PRECHECK_WORKERS,
) -> StageUpdatePlan:
    if not symbols:
        return StageUpdatePlan(stage_name, {}, {}, [], {})
    print(f"[*] Prechecking local {stage_name} shards with {max_workers} workers...")
    latest_by_symbol: dict[str, pd.Timestamp | None] = {}
    earliest_by_symbol: dict[str, pd.Timestamp | None] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for symbol in symbols:
            path = stage_dir / f"{symbol}.parquet"
            future_map[executor.submit(infer_latest_date, path, "trade_date")] = (symbol, "latest")
            if expected_start_dates is not None:
                future_map[executor.submit(infer_earliest_date, path, "trade_date")] = (symbol, "earliest")
        pbar = tqdm(as_completed(future_map), total=len(future_map), desc=f"precheck {stage_name}", unit="task")
        for future in pbar:
            symbol, kind = future_map[future]
            value = future.result()
            if kind == "latest":
                latest_by_symbol[symbol] = value
            else:
                earliest_by_symbol[symbol] = value
    ready_outputs: dict[str, UpdateResult] = {}
    pending_set: set[str] = set()
    for symbol in symbols:
        latest = latest_by_symbol.get(symbol)
        required_end_date = (
            effective_end_dates.get(symbol, target_end_date)
            if effective_end_dates is not None
            else target_end_date
        )
        if coverage_start_date is not None and required_end_date < coverage_start_date:
            ready_outputs[symbol] = UpdateResult(
                status=f"{stage_name} not required before {coverage_start_date.date()}",
                changed=False,
                latest=latest,
            )
            continue
        expected_start_date = (
            expected_start_dates.get(symbol)
            if expected_start_dates is not None
            else None
        )
        earliest = earliest_by_symbol.get(symbol)
        is_backfilled = is_backfill_satisfied(
            stage_name,
            symbol,
            expected_start_date,
            earliest,
            exhaustion_lookup,
        )
        schema_complete = not _parquet_missing_columns(stage_dir / f"{symbol}.parquet", required_columns or [])
        if latest is not None and latest >= required_end_date and is_backfilled and schema_complete:
            ready_outputs[symbol] = UpdateResult(
                status=f"{stage_name} up-to-date at {latest.date()}",
                changed=False,
                latest=latest,
                earliest=earliest,
            )
        else:
            pending_set.add(symbol)

    pending_symbols = [symbol for symbol in symbols if symbol in pending_set]
    print(f"[*] {stage_name}: ready={len(ready_outputs)}, pending={len(pending_symbols)}")
    return StageUpdatePlan(stage_name, latest_by_symbol, earliest_by_symbol, pending_symbols, ready_outputs)


def _iter_market_date_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts > end_ts:
        return []
    windows: list[tuple[str, str]] = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.DateOffset(years=MARKET_CHUNK_YEARS) - pd.Timedelta(days=1), end_ts)
        windows.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return windows


def _fetch_market_table_in_chunks(
    symbol: str,
    start_date: str,
    end_date: str,
    per_chunk_fetcher: Callable[[str, str, str], pd.DataFrame],
    empty_columns: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _iter_market_date_chunks(start_date, end_date):
        frame = per_chunk_fetcher(symbol, chunk_start, chunk_end)
        if frame is None or frame.empty:
            continue
        frame = frame.reindex(columns=empty_columns)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=empty_columns)
    merged = pd.concat(frames, ignore_index=True)
    return _normalize_symbol_date_frame(merged, "trade_date", "ts_code")


def _fetch_daily_chunk(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = get_tushare_client()
    frame = _ts_call(
        "daily",
        pro.daily,
        ts_code=local_symbol_to_ts(symbol),
        start_date=start_date,
        end_date=end_date,
        fields=",".join(DAILY_API_FIELDS),
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DAILY_RAW_COLS)
    out = frame.copy().rename(columns={"vol": "volume"})
    out = out.reindex(columns=DAILY_RAW_COLS)
    return _normalize_symbol_date_frame(out, "trade_date", "ts_code")


def _fetch_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_market_table_in_chunks(
        symbol,
        start_date,
        end_date,
        _fetch_daily_chunk,
        DAILY_RAW_COLS,
    )


def _fetch_daily_basic_chunk(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = get_tushare_client()
    frame = _ts_call(
        "daily_basic",
        pro.daily_basic,
        ts_code=local_symbol_to_ts(symbol),
        start_date=start_date,
        end_date=end_date,
        fields=",".join(DAILY_BASIC_API_FIELDS),
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=DAILY_BASIC_API_FIELDS)
    out = frame.reindex(columns=DAILY_BASIC_API_FIELDS)
    return _normalize_symbol_date_frame(out, "trade_date", "ts_code")


def _fetch_daily_basic(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_market_table_in_chunks(
        symbol,
        start_date,
        end_date,
        _fetch_daily_basic_chunk,
        DAILY_BASIC_API_FIELDS,
    )


def _fetch_adj_factor_chunk(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = get_tushare_client()
    frame = _ts_call(
        "adj_factor",
        pro.adj_factor,
        ts_code=local_symbol_to_ts(symbol),
        start_date=start_date,
        end_date=end_date,
        fields=",".join(ADJ_FACTOR_API_FIELDS),
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=ADJ_FACTOR_API_FIELDS)
    out = frame.reindex(columns=ADJ_FACTOR_API_FIELDS)
    return _normalize_symbol_date_frame(out, "trade_date", "ts_code")


def _fetch_adj_factor(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_market_table_in_chunks(
        symbol,
        start_date,
        end_date,
        _fetch_adj_factor_chunk,
        ADJ_FACTOR_API_FIELDS,
    )


def _fetch_stk_limit_chunk(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = get_tushare_client()
    frame = _ts_call(
        "stk_limit",
        pro.stk_limit,
        ts_code=local_symbol_to_ts(symbol),
        start_date=start_date,
        end_date=end_date,
        fields=",".join(STK_LIMIT_API_FIELDS),
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=STK_LIMIT_API_FIELDS)
    out = frame.reindex(columns=STK_LIMIT_API_FIELDS)
    return _normalize_symbol_date_frame(out, "trade_date", "ts_code")


def _fetch_stk_limit(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_market_table_in_chunks(
        symbol,
        start_date,
        end_date,
        _fetch_stk_limit_chunk,
        STK_LIMIT_API_FIELDS,
    )


def _fetch_fina_indicator_chunk(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    pro = get_tushare_client()
    out = _ts_call(
        "fina_indicator",
        pro.fina_indicator,
        ts_code=local_symbol_to_ts(symbol),
        start_date=start_date,
        end_date=end_date,
        fields=",".join(FINA_INDICATOR_API_FIELDS),
    )
    if out is None:
        return pd.DataFrame(columns=FINA_INDICATOR_API_FIELDS)
    out = out.reindex(columns=FINA_INDICATOR_API_FIELDS)
    return _normalize_symbol_date_frame(out, "ann_date", "ts_code")


def _fetch_fina_indicator(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return _fetch_market_table_in_chunks(
        symbol,
        start_date,
        end_date,
        _fetch_fina_indicator_chunk,
        FINA_INDICATOR_API_FIELDS,
    )


def _concat_schema_preserving_frames(
    frames: list[pd.DataFrame | None],
    *,
    ordered_columns: list[str] | None = None,
) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=ordered_columns or [])

    prepared: list[pd.DataFrame] = []
    key_cols = {"ts_code", "trade_date"}
    for frame in valid_frames:
        current = frame.copy()
        if ordered_columns is not None:
            current = current.reindex(columns=ordered_columns)
        keep_cols = [
            col
            for col in current.columns
            if col in key_cols or not current[col].isna().all()
        ]
        prepared.append(current.loc[:, keep_cols])

    combined = pd.concat(prepared, ignore_index=True)
    if ordered_columns is not None:
        combined = combined.reindex(columns=ordered_columns)
    return combined


def update_raw_table(
    symbol: str,
    path: Path,
    fetcher: Callable[[str, str, str], pd.DataFrame],
    target_end_date: pd.Timestamp,
    latest: pd.Timestamp | None | object = ...,
    expected_start_date: pd.Timestamp | None = None,
    required_columns: list[str] | None = None,
    refetch_on_schema_mismatch: bool = False,
    date_column: str = "trade_date",
    symbol_column: str = "ts_code",
) -> UpdateResult:
    if latest is ...:
        latest = infer_latest_date(path, date_column)
    earliest = infer_earliest_date(path, date_column)
    schema_missing = _parquet_missing_columns(path, required_columns or [])
    needs_future = latest is None or latest < target_end_date
    needs_backfill = expected_start_date is not None and (
        earliest is None or earliest > expected_start_date
    )
    if not needs_future and not needs_backfill and not schema_missing:
        return UpdateResult(
            status=f"{path.parent.name} up-to-date at {latest.date()}",
            changed=False,
            latest=latest,
            earliest=earliest,
        )

    existing = load_parquet_if_exists(path)
    if existing is not None and required_columns:
        existing = existing.reindex(columns=required_columns)
    if existing is not None and not existing.empty and schema_missing and not refetch_on_schema_mismatch:
        if not needs_future and not needs_backfill:
            save_optimized_parquet(existing, path)
            return UpdateResult(
                status=f"{path.parent.name} schema normalized at {latest.date()}",
                changed=True,
                latest=latest,
                earliest=earliest,
            )

    fetch_start = expected_start_date or pd.Timestamp(DEFAULT_START_DATE)
    fetch_end = target_end_date
    if existing is not None and not existing.empty:
        if schema_missing and refetch_on_schema_mismatch:
            fetch_start = earliest or fetch_start
            existing = None
        elif needs_backfill and not needs_future and earliest is not None:
            fetch_end = earliest - pd.Timedelta(days=1)
        elif needs_future and not needs_backfill and latest is not None:
            fetch_start = latest + pd.Timedelta(days=1)

    if fetch_start > fetch_end:
        return UpdateResult(
            status=f"{path.parent.name} unchanged at {latest.date() if latest is not None else 'n/a'}",
            changed=False,
            latest=latest,
            earliest=earliest,
        )

    fetched = fetcher(symbol, fetch_start.strftime("%Y%m%d"), fetch_end.strftime("%Y%m%d"))

    if fetched.empty:
        if needs_backfill and not needs_future and existing is not None and not existing.empty and earliest is not None:
            return UpdateResult(
                status=f"{path.parent.name} backfill exhausted at {earliest.date()}",
                changed=False,
                latest=latest,
                earliest=earliest,
                backfill_exhausted=True,
            )
        if existing is not None and not existing.empty and latest is not None:
            return UpdateResult(
                status=f"{path.parent.name} unchanged at {latest.date()}",
                changed=False,
                latest=latest,
                earliest=earliest,
            )
        raise RuntimeError(f"{path.parent.name} fetch returned empty data")

    if required_columns:
        fetched = fetched.reindex(columns=required_columns)
    combined = _concat_schema_preserving_frames(
        [existing, fetched],
        ordered_columns=required_columns,
    )
    combined = _normalize_symbol_date_frame(combined, date_column, symbol_column)
    save_optimized_parquet(combined, path)
    latest = combined[date_column].max()
    earliest = combined[date_column].min()
    return UpdateResult(
        status=f"{path.parent.name} saved to {latest.date()}",
        changed=True,
        latest=latest,
        earliest=earliest,
    )


def _merge_left(left: pd.DataFrame, right: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    use_cols = [col for col in columns if col in right.columns]
    if not use_cols:
        return left
    subset = right[["ts_code", "trade_date", *use_cols]].copy()
    return left.merge(subset, on=["ts_code", "trade_date"], how="left")


def _prepare_optional_market_frame(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    out = _normalize_symbol_date_frame(frame, "trade_date", "ts_code")
    return _coerce_numeric_columns(out, exclude={"ts_code", "trade_date"})


def build_processed_symbol_frame_from_raw(
    symbol: str,
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame | None,
    adj_factor: pd.DataFrame | None,
    stk_limit: pd.DataFrame | None,
) -> pd.DataFrame:
    if daily is None or daily.empty:
        raise FileNotFoundError(f"missing Tushare daily parquet for {symbol}")

    out = _normalize_symbol_date_frame(daily.rename(columns={"vol": "volume"}), "trade_date", "ts_code")
    out = _coerce_numeric_columns(out, exclude={"ts_code", "trade_date"})
    daily_basic = _prepare_optional_market_frame(daily_basic)
    adj_factor = _prepare_optional_market_frame(adj_factor)
    stk_limit = _prepare_optional_market_frame(stk_limit)
    if stk_limit is not None and not stk_limit.empty and "pre_close" in stk_limit.columns:
        stk_limit = stk_limit.rename(columns={"pre_close": "limit_pre_close"})
    out = _merge_left(
        out,
        daily_basic,
        [
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ],
    )
    out = _merge_left(
        out,
        adj_factor,
        ["adj_factor"],
    )
    out = _merge_left(
        out,
        stk_limit,
        ["limit_pre_close", "up_limit", "down_limit"],
    )

    out["adj_factor"] = pd.to_numeric(out.get("adj_factor"), errors="coerce").fillna(1.0)

    raw_open = pd.to_numeric(out["open"], errors="coerce")
    raw_high = pd.to_numeric(out["high"], errors="coerce")
    raw_low = pd.to_numeric(out["low"], errors="coerce")
    raw_close = pd.to_numeric(out["close"], errors="coerce")
    raw_pre_close = pd.to_numeric(out["pre_close"], errors="coerce")
    factor = out["adj_factor"].astype(float)

    open_adj = raw_open * factor
    high_adj = raw_high * factor
    low_adj = raw_low * factor
    close_adj = raw_close * factor
    prev_close_adj = close_adj.shift(1)
    fallback_pre_close_adj = raw_pre_close * factor
    pre_close_adj = prev_close_adj.combine_first(fallback_pre_close_adj)

    out["date"] = out["trade_date"]
    out["symbol"] = symbol
    out["open"] = open_adj
    out["high"] = high_adj
    out["low"] = low_adj
    out["close"] = close_adj
    out["pre_close"] = pre_close_adj
    out["raw_open"] = raw_open
    out["raw_high"] = raw_high
    out["raw_low"] = raw_low
    out["raw_close"] = raw_close
    out["raw_pre_close"] = raw_pre_close
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["amount"] = pd.to_numeric(out.get("amount"), errors="coerce") * 1000.0
    out["turnover"] = pd.to_numeric(out.get("turnover_rate"), errors="coerce")
    out["turnover_free"] = pd.to_numeric(out.get("turnover_rate_f"), errors="coerce")
    out["volume_ratio"] = pd.to_numeric(out.get("volume_ratio"), errors="coerce")
    out["pe"] = pd.to_numeric(out.get("pe"), errors="coerce")
    out["pe_ttm"] = pd.to_numeric(out.get("pe_ttm"), errors="coerce")
    out["pb"] = pd.to_numeric(out.get("pb"), errors="coerce")
    out["ps"] = pd.to_numeric(out.get("ps"), errors="coerce")
    out["ps_ttm"] = pd.to_numeric(out.get("ps_ttm"), errors="coerce")
    out["dv_ratio"] = pd.to_numeric(out.get("dv_ratio"), errors="coerce")
    out["dv_ttm"] = pd.to_numeric(out.get("dv_ttm"), errors="coerce")
    out["limit_pre_close"] = pd.to_numeric(out.get("limit_pre_close"), errors="coerce") * factor
    out["total_share"] = pd.to_numeric(out.get("total_share"), errors="coerce") * 10000.0
    out["circ_share"] = pd.to_numeric(out.get("float_share"), errors="coerce") * 10000.0
    out["free_share"] = pd.to_numeric(out.get("free_share"), errors="coerce") * 10000.0
    out["total_mv"] = pd.to_numeric(out.get("total_mv"), errors="coerce") * 10000.0
    out["circ_mv"] = pd.to_numeric(out.get("circ_mv"), errors="coerce") * 10000.0
    out["up_limit"] = pd.to_numeric(out.get("up_limit"), errors="coerce") * factor
    out["down_limit"] = pd.to_numeric(out.get("down_limit"), errors="coerce") * factor

    out["change"] = out["close"] - out["pre_close"]
    out["pct_chg"] = np.where(out["pre_close"] > 0, out["change"] / out["pre_close"] * 100.0, np.nan)
    out["amplitude"] = np.where(
        out["pre_close"] > 0,
        (out["high"] - out["low"]) / out["pre_close"] * 100.0,
        np.nan,
    )

    for col in TS_PROCESSED_CANONICAL_COLS:
        if col not in out.columns:
            out[col] = np.nan
    out = out.reindex(columns=TS_PROCESSED_CANONICAL_COLS)
    return out.sort_values("date").reset_index(drop=True)


def build_processed_symbol_frame(symbol: str) -> pd.DataFrame:
    return build_processed_symbol_frame_from_raw(
        symbol,
        load_parquet_if_exists(RAW_DAILY_DIR / f"{symbol}.parquet"),
        load_parquet_if_exists(RAW_DAILY_BASIC_DIR / f"{symbol}.parquet"),
        load_parquet_if_exists(RAW_ADJ_FACTOR_DIR / f"{symbol}.parquet"),
        load_parquet_if_exists(RAW_STK_LIMIT_DIR / f"{symbol}.parquet"),
    )


def rebuild_processed_from_local(symbol: str) -> str:
    processed = build_processed_symbol_frame(symbol)
    path = PROCESSED_DIR / f"{symbol}.parquet"
    save_optimized_parquet(processed, path)
    return f"processed rows={len(processed)} last_date={processed['date'].max().date()}"


def should_rebuild_processed(symbol: str, updates: list[UpdateResult]) -> bool:
    if any(item.changed for item in updates):
        return True
    path = PROCESSED_DIR / f"{symbol}.parquet"
    if _parquet_missing_columns(path, TS_PROCESSED_CANONICAL_COLS):
        return True
    processed_latest = infer_latest_date(path, "date")
    daily_latest = infer_latest_date(RAW_DAILY_DIR / f"{symbol}.parquet", "trade_date")
    if processed_latest is None or daily_latest is None:
        return True
    return processed_latest < daily_latest


def _run_stage_worker(symbol: str, worker: Callable[[str], Any]) -> StageTaskResult:
    try:
        payload = worker(symbol)
        detail = payload.status if isinstance(payload, UpdateResult) else str(payload)
        return StageTaskResult(symbol=symbol, ok=True, detail=detail, payload=payload)
    except Exception as exc:
        return StageTaskResult(symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}")


def collect_stage_batch(
    symbols: list[str],
    stage_name: str,
    worker: Callable[[str], Any],
    max_workers: int,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    skip_remaining_on_threshold: bool = True,
) -> tuple[list[StageTaskResult], list[str], bool]:
    results: list[StageTaskResult] = []
    consecutive_failures = 0
    rate_limit_hit = False
    batch_symbols = list(symbols)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batch_symbols) or 1)) as executor:
        future_map = {executor.submit(_run_stage_worker, symbol, worker): symbol for symbol in symbols}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            if not result.ok and _is_tushare_rate_limit_detail(result.detail):
                rate_limit_hit = True
                print(
                    f"[!] Stage '{stage_name}' hit Tushare rate limit. "
                    f"Cooldown {RATE_LIMIT_COOLDOWN_SECONDS:.0f}s and switch to other stages."
                )
                for pending_future in future_map:
                    if not pending_future.done():
                        pending_future.cancel()
                break
            if result.ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                if skip_remaining_on_threshold:
                    print(
                        f"[!] Stage '{stage_name}' hit {consecutive_failures} consecutive failures. "
                        "Skipping remaining symbols in this stage and continuing."
                    )
                    for pending_future in future_map:
                        if not pending_future.done():
                            pending_future.cancel()
                    break
                raise SystemExit(
                    f"[!] Aborting stage '{stage_name}' after {consecutive_failures} consecutive failures."
                )
    seen_symbols = {item.symbol for item in results}
    deferred_symbols: list[str] = [symbol for symbol in batch_symbols if symbol not in seen_symbols]
    return results, deferred_symbols, rate_limit_hit


def collect_stage(
    symbols: list[str],
    stage_name: str,
    worker: Callable[[str], Any],
    max_workers: int,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    skip_remaining_on_threshold: bool = True,
) -> list[TaskResult]:
    results, deferred_symbols, rate_limit_hit = collect_stage_batch(
        symbols,
        stage_name,
        worker,
        max_workers,
        max_consecutive_failures=max_consecutive_failures,
        skip_remaining_on_threshold=skip_remaining_on_threshold,
    )
    if deferred_symbols:
        detail = (
            f"{stage_name} cooling down for {RATE_LIMIT_COOLDOWN_SECONDS:.0f}s after rate limit"
            if rate_limit_hit
            else f"{stage_name} skipped after {max_consecutive_failures} consecutive failures"
        )
        results.extend(
            StageTaskResult(symbol=symbol, ok=False, detail=detail) for symbol in deferred_symbols
        )
    return [TaskResult(symbol=item.symbol, ok=item.ok, detail=item.detail) for item in results]


def run_tushare_update_pipeline(
    symbols: list[str],
    target_end_date: pd.Timestamp,
    max_workers: int,
    effective_end_dates: dict[str, pd.Timestamp] | None = None,
) -> list[TaskResult]:
    exhaustion_frame = load_backfill_exhaustion()
    exhaustion_lookup = build_backfill_exhaustion_lookup(exhaustion_frame)
    cached = load_symbol_cache()
    expected_start_dates: dict[str, pd.Timestamp] = {}
    if cached is not None and not cached.empty:
        for _, row in cached.iterrows():
            list_date = _normalize_optional_timestamp(row.get("list_date"))
            if list_date is not None:
                expected_start_dates[str(row["local_symbol"])] = list_date

    stage_specs: list[
        tuple[str, Path, Callable[[str, pd.Timestamp | None, pd.Timestamp, pd.Timestamp | None], UpdateResult]]
    ] = [
        (
            "daily",
            RAW_DAILY_DIR,
            lambda symbol, latest, symbol_target_end_date, expected_start_date: update_raw_table(
                symbol,
                RAW_DAILY_DIR / f"{symbol}.parquet",
                _fetch_daily,
                symbol_target_end_date,
                latest=latest,
                expected_start_date=expected_start_date,
                required_columns=DAILY_RAW_COLS,
            ),
        ),
        (
            "daily_basic",
            RAW_DAILY_BASIC_DIR,
            lambda symbol, latest, symbol_target_end_date, expected_start_date: update_raw_table(
                symbol,
                RAW_DAILY_BASIC_DIR / f"{symbol}.parquet",
                _fetch_daily_basic,
                symbol_target_end_date,
                latest=latest,
                expected_start_date=expected_start_date,
                required_columns=DAILY_BASIC_API_FIELDS,
            ),
        ),
        (
            "adj_factor",
            RAW_ADJ_FACTOR_DIR,
            lambda symbol, latest, symbol_target_end_date, expected_start_date: update_raw_table(
                symbol,
                RAW_ADJ_FACTOR_DIR / f"{symbol}.parquet",
                _fetch_adj_factor,
                symbol_target_end_date,
                latest=latest,
                expected_start_date=expected_start_date,
                required_columns=ADJ_FACTOR_API_FIELDS,
            ),
        ),
        (
            "stk_limit",
            RAW_STK_LIMIT_DIR,
            lambda symbol, latest, symbol_target_end_date, expected_start_date: update_raw_table(
                symbol,
                RAW_STK_LIMIT_DIR / f"{symbol}.parquet",
                _fetch_stk_limit,
                symbol_target_end_date,
                latest=latest,
                required_columns=STK_LIMIT_API_FIELDS,
                refetch_on_schema_mismatch=True,
            ),
        ),
    ]

    print(f"[*] Prechecking local Tushare raw shards with {PRECHECK_WORKERS} workers...")
    stage_plans = {
        stage_name: precheck_stage_updates(
            symbols,
            stage_name,
            stage_dir,
            target_end_date,
            effective_end_dates=effective_end_dates,
            expected_start_dates=expected_start_dates if stage_name in {"daily", "daily_basic", "adj_factor"} else None,
            coverage_start_date=STK_LIMIT_COVERAGE_START if stage_name == "stk_limit" else None,
            exhaustion_lookup=exhaustion_lookup,
            required_columns={
                "daily": DAILY_RAW_COLS,
                "daily_basic": DAILY_BASIC_API_FIELDS,
                "adj_factor": ADJ_FACTOR_API_FIELDS,
                "stk_limit": STK_LIMIT_API_FIELDS,
            }[stage_name],
            max_workers=PRECHECK_WORKERS,
        )
        for stage_name, stage_dir, _ in stage_specs
    }

    stage_details: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    stage_outputs: dict[str, dict[str, UpdateResult]] = {name: {} for name, _, _ in stage_specs}
    failed_symbols: set[str] = set()
    stage_pending: dict[str, deque[str]] = {}
    stage_cooldown_until: dict[str, float] = {}
    stage_success: dict[str, int] = {}
    stage_failure: dict[str, int] = {}
    stage_done: dict[str, int] = {}
    stage_total: dict[str, int] = {}
    backfill_updates: list[dict[str, Any]] = []
    backfill_removals: set[tuple[str, str]] = set()

    for stage_name, _, _ in stage_specs:
        plan = stage_plans[stage_name]
        stage_outputs[stage_name].update(plan.ready_outputs)
        for symbol, output in plan.ready_outputs.items():
            stage_details[symbol].append(f"{stage_name}: {output.status}")
        stage_pending[stage_name] = deque(plan.pending_symbols)
        stage_cooldown_until[stage_name] = 0.0
        stage_success[stage_name] = len(plan.ready_outputs)
        stage_failure[stage_name] = 0
        stage_done[stage_name] = len(plan.ready_outputs)
        stage_total[stage_name] = len(plan.pending_symbols) + len(plan.ready_outputs)

    while any(stage_pending[name] for name, _, _ in stage_specs):
        progressed = False
        now = time.monotonic()
        for stage_name, stage_dir, worker in stage_specs:
            queue = stage_pending[stage_name]
            if not queue:
                continue
            cooldown_until = stage_cooldown_until[stage_name]
            if cooldown_until > now:
                continue

            batch_symbols: list[str] = []
            while queue and len(batch_symbols) < max_workers:
                batch_symbols.append(queue.popleft())
            if not batch_symbols:
                continue

            latest_by_symbol = stage_plans[stage_name].latest_by_symbol
            results, deferred_symbols, rate_limit_hit = collect_stage_batch(
                batch_symbols,
                stage_name,
                lambda symbol, worker=worker, latest_by_symbol=latest_by_symbol: worker(
                    symbol,
                    latest_by_symbol.get(symbol),
                    (effective_end_dates or {}).get(symbol, target_end_date),
                    expected_start_dates.get(symbol)
                    if stage_name in {"daily", "daily_basic", "adj_factor"}
                    else None,
                ),
                max_workers=max_workers,
            )
            progressed = True

            retry_symbols: list[str] = []
            for item in results:
                if item.ok:
                    stage_details[item.symbol].append(f"{stage_name}: {item.detail}")
                    if isinstance(item.payload, UpdateResult):
                        stage_outputs[stage_name][item.symbol] = item.payload
                        if item.payload.changed:
                            backfill_removals.add((stage_name, item.symbol))
                        elif item.payload.backfill_exhausted:
                            backfill_updates.append(
                                {
                                    "stage_name": stage_name,
                                    "local_symbol": item.symbol,
                                    "expected_start_date": expected_start_dates.get(item.symbol)
                                    if stage_name in {"daily", "daily_basic", "adj_factor"}
                                    else pd.NaT,
                                    "observed_earliest_date": item.payload.earliest,
                                    "recorded_at": pd.Timestamp.now().normalize(),
                                }
                            )
                    else:
                        stage_outputs[stage_name][item.symbol] = UpdateResult(
                            status=item.detail,
                            changed=" saved to " in item.detail,
                            latest=infer_latest_date(stage_dir / f"{item.symbol}.parquet", "trade_date"),
                        )
                    stage_success[stage_name] += 1
                    stage_done[stage_name] += 1
                    continue
                if _is_tushare_rate_limit_detail(item.detail):
                    retry_symbols.append(item.symbol)
                    continue
                stage_details[item.symbol].append(f"{stage_name}: {item.detail}")
                failed_symbols.add(item.symbol)
                stage_failure[stage_name] += 1
                stage_done[stage_name] += 1

            if rate_limit_hit:
                retry_symbols.extend(symbol for symbol in deferred_symbols if symbol not in retry_symbols)
                for symbol in reversed(retry_symbols):
                    queue.appendleft(symbol)
                stage_cooldown_until[stage_name] = time.monotonic() + RATE_LIMIT_COOLDOWN_SECONDS
                remaining = len(queue)
                print(
                    f"[*] Stage {stage_name} cooling down for {RATE_LIMIT_COOLDOWN_SECONDS:.0f}s "
                    f"(done={stage_done[stage_name]}/{stage_total[stage_name]}, pending={remaining})"
                )
                continue

            for symbol in deferred_symbols:
                stage_details[symbol].append(
                    f"{stage_name}: {stage_name} skipped after {MAX_CONSECUTIVE_FAILURES} consecutive failures"
                )
                failed_symbols.add(symbol)
                stage_failure[stage_name] += 1
                stage_done[stage_name] += 1

            print(
                f"[*] Stage {stage_name}: done={stage_done[stage_name]}/{stage_total[stage_name]} "
                f"ok={stage_success[stage_name]} fail={stage_failure[stage_name]} pending={len(queue)}"
            )

        if progressed:
            continue

        waiting_stages = [
            (name, stage_cooldown_until[name])
            for name, _, _ in stage_specs
            if stage_pending[name] and stage_cooldown_until[name] > now
        ]
        if not waiting_stages:
            break
        next_stage, next_ready_at = min(waiting_stages, key=lambda item: item[1])
        wait_for = max(next_ready_at - now, 0.0)
        print(
            f"[*] All pending raw stages are cooling down. "
            f"Waiting {wait_for:.1f}s for {next_stage} to resume..."
        )
        time.sleep(wait_for)

    if backfill_updates or backfill_removals:
        exhaustion_frame = merge_backfill_exhaustion_records(
            exhaustion_frame,
            updates=backfill_updates,
            removals=backfill_removals,
        )
        save_backfill_exhaustion(exhaustion_frame)

    build_symbols = [
        symbol
        for symbol in symbols
        if symbol not in failed_symbols
        and all(symbol in stage_outputs[name] for name, _, _ in stage_specs)
    ]
    if build_symbols:
        print("[*] Raw Tushare downloads finished. Rebuilding processed parquet...")

        def rebuild_if_needed(symbol: str) -> str:
            updates = [stage_outputs[name][symbol] for name, _, _ in stage_specs]
            if should_rebuild_processed(symbol, updates):
                return rebuild_processed_from_local(symbol)
            processed_latest = infer_latest_date(PROCESSED_DIR / f"{symbol}.parquet", "date")
            return f"processed up-to-date at {processed_latest.date()}"

        processed_results = collect_stage(
            build_symbols,
            "processed",
            rebuild_if_needed,
            max_workers=max_workers,
        )
        for item in processed_results:
            stage_details[item.symbol].append(f"processed: {item.detail}")
            if not item.ok:
                failed_symbols.add(item.symbol)

    final_results: list[TaskResult] = []
    for symbol in symbols:
        ok = symbol not in failed_symbols and any(
            detail.startswith("processed: ") for detail in stage_details[symbol]
        )
        final_results.append(TaskResult(symbol=symbol, ok=ok, detail="; ".join(stage_details[symbol])))
    return final_results


def rebuild_symbol(symbol: str) -> TaskResult:
    try:
        detail = rebuild_processed_from_local(symbol)
        return TaskResult(symbol=symbol, ok=True, detail=detail)
    except Exception as exc:
        return TaskResult(symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}")


def collect_symbols(
    symbols: list[str],
    worker: Callable[[str], TaskResult],
    max_workers: int,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    consecutive_failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, symbol): symbol for symbol in symbols}
        pbar = tqdm(as_completed(future_map), total=len(symbols))
        for future in pbar:
            result = future.result()
            results.append(result)
            if result.ok:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            ok_count = sum(1 for item in results if item.ok)
            pbar.set_postfix({"success": ok_count, "failed": len(results) - ok_count})
            if consecutive_failures >= max_consecutive_failures:
                raise SystemExit(
                    f"[!] Aborting after {consecutive_failures} consecutive Tushare failures."
                )
    return results


def resolve_target_end_date(end_date: str | None = None) -> pd.Timestamp:
    if end_date:
        return pd.Timestamp(end_date).normalize()
    return pd.Timestamp.today().normalize()


def parse_symbols_arg(symbols: str | None) -> list[str]:
    if not symbols:
        return []
    return sorted({ts_symbol_to_local(item.strip()) for item in symbols.split(",") if item.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Tushare market data into raw parquet and processed hfq-adjusted combined parquet."
    )
    parser.add_argument("--all", action="store_true", help="Process all symbols from Tushare symbol cache.")
    parser.add_argument("--symbols", help="Comma-separated symbols, supports bare code or ts_code.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch/update all locally known or cached Tushare symbols.",
    )
    parser.add_argument(
        "--rebuild-processed",
        action="store_true",
        help="Rebuild Tushare processed parquet from local raw files.",
    )
    parser.add_argument(
        "--fina-indicator",
        action="store_true",
        help="Fetch/update raw Tushare fina_indicator parquet only for the resolved symbols.",
    )
    parser.add_argument(
        "--refresh-symbols",
        action="store_true",
        help="Refresh Tushare symbol cache before resolving symbols.",
    )
    parser.add_argument(
        "--refresh-symbols-only",
        action="store_true",
        help="Refresh Tushare symbol cache and exit.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--end-date",
        default=None,
        help="Target end date in YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help=f"Optional environment variable containing Tushare token. Default: {DEFAULT_TOKEN_ENV}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_tushare(args.token_env)

    if args.refresh_symbols_only:
        refresh_symbol_cache()
        print("[+] Tushare symbol cache refresh completed.")
        return

    target_end_date = resolve_target_end_date(args.end_date)
    latest_trading_date = resolve_latest_trading_date(target_end_date)
    print(f"[*] Latest trading date on or before {target_end_date.date()}: {latest_trading_date.date()}")

    symbols = parse_symbols_arg(args.symbols)
    if args.rebuild_processed:
        if not symbols:
            symbols = list_local_symbols()
        if not symbols:
            raise SystemExit("[!] No local Tushare symbols found to rebuild.")
        print(
            f"[*] Rebuilding Tushare processed parquet for {len(symbols)} symbols from local raw files..."
        )
        results = collect_symbols(symbols, rebuild_symbol, max_workers=args.workers)
    elif args.fina_indicator:
        if args.all:
            symbols = resolve_all_symbols(refresh_live=args.refresh_symbols)
        elif args.update and not symbols:
            symbols = resolve_incremental_symbols(refresh_live=args.refresh_symbols)
        elif not symbols:
            parser.error("Provide one of --all, --update, or --symbols together with --fina-indicator.")

        if not symbols:
            raise SystemExit("[!] No Tushare symbols to process.")

        print(
            f"[*] Updating raw fina_indicator for {len(symbols)} symbols with {args.workers} workers "
            f"(end_date={latest_trading_date.date()})..."
        )

        def _fina_indicator_worker(symbol: str) -> TaskResult:
            try:
                latest = infer_latest_date(RAW_FINA_INDICATOR_DIR / f"{symbol}.parquet", "ann_date")
                result = update_raw_table(
                    symbol,
                    RAW_FINA_INDICATOR_DIR / f"{symbol}.parquet",
                    _fetch_fina_indicator,
                    latest_trading_date,
                    latest=latest,
                    expected_start_date=pd.Timestamp(DEFAULT_START_DATE),
                    required_columns=FINA_INDICATOR_API_FIELDS,
                    date_column="ann_date",
                )
                return TaskResult(symbol=symbol, ok=True, detail=result.status)
            except Exception as exc:
                return TaskResult(symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}")

        results = collect_symbols(symbols, _fina_indicator_worker, max_workers=args.workers)
    else:
        if args.all:
            symbols = resolve_all_symbols(refresh_live=args.refresh_symbols)
        elif args.update and not symbols:
            symbols = resolve_incremental_symbols(refresh_live=args.refresh_symbols)
        elif not symbols:
            parser.error("Provide one of --all, --update, --rebuild-processed, or --symbols.")

        if not symbols:
            raise SystemExit("[!] No Tushare symbols to process.")

        symbols, completed_symbols, effective_end_dates, lifecycle_registry = precheck_pending_symbols(
            symbols, target_end_date=latest_trading_date
        )
        if not lifecycle_registry.empty:
            lifecycle_counts = lifecycle_registry["lifecycle_status"].value_counts().to_dict()
            counts_text = ", ".join(f"{key}={value}" for key, value in sorted(lifecycle_counts.items()))
            print(f"[*] Tushare lifecycle snapshot: {counts_text}")
        if completed_symbols:
            print(f"[*] Skipping {len(completed_symbols)} already-complete Tushare symbols.")
        if not symbols:
            print("[+] All Tushare symbols are already complete. Nothing to update.")
            return

        print(
            f"[*] Updating {len(symbols)} pending Tushare symbols with {args.workers} workers "
            f"(end_date={latest_trading_date.date()})..."
        )
        results = run_tushare_update_pipeline(
            symbols,
            target_end_date=latest_trading_date,
            max_workers=args.workers,
            effective_end_dates=effective_end_dates,
        )

    success = sum(1 for item in results if item.ok)
    failed = [item for item in results if not item.ok]
    print(f"[+] Tushare collection done. Success: {success} / {len(results)}")
    if failed:
        print("[!] Failed Tushare symbols:")
        for item in failed[:20]:
            print(f"    {item.symbol}: {item.detail}")
        if len(failed) > 20:
            print(f"    ... {len(failed) - 20} more")


if __name__ == "__main__":
    main()
