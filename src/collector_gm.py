from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from gm import api as gmapi
from tqdm import tqdm

GM_ROOT = Path("data/gm")
RAW_ROOT = GM_ROOT / "raw"
RAW_META_DIR = RAW_ROOT / "meta"
RAW_BARS_DIR = RAW_ROOT / "bars_raw"
RAW_SYMBOL_DAY_DIR = RAW_ROOT / "symbol_day"
RAW_BASIC_DIR = RAW_ROOT / "daily_basic"
RAW_MKTVALUE_DIR = RAW_ROOT / "daily_mktvalue"
RAW_VALUATION_DIR = RAW_ROOT / "daily_valuation"
PROCESSED_DIR = GM_ROOT / "processed" / "combined"
SYMBOL_CACHE_PATH = RAW_META_DIR / "symbol_cache.parquet"
SYMBOL_LIFECYCLE_PATH = RAW_META_DIR / "symbol_lifecycle.parquet"

GM_STOCK_SEC_TYPE1 = 1010
GM_A_SHARE_SEC_TYPE2 = 101001
DEFAULT_TOKEN_ENV = "GM_TOKEN"
DEFAULT_START_DATE = "1990-01-01"
MAX_FIELDS_PER_REQUEST = 20
MAX_CONSECUTIVE_FAILURES = 10
PRECHECK_WORKERS = 16
ENDPOINT_MIN_INTERVAL_SECONDS = {
    "symbol_cache": 0,
    "history": 0,
    "history_symbol": 0,
    "daily_basic": 0,
    "daily_mktvalue": 0,
    "daily_valuation": 0,
}

SYMBOL_CACHE_COLS = [
    "local_symbol",
    "symbol",
    "sec_id",
    "sec_name",
    "exchange",
    "listed_date",
    "delisted_date",
    "board",
    "trade_n",
    "fetched_at",
]

SYMBOL_LIFECYCLE_COLS = [
    "local_symbol",
    "gm_symbol",
    "listed_date",
    "delisted_date",
    "lifecycle_status",
    "latest_is_suspended",
    "effective_end_date",
    "latest_bars_date",
    "latest_symbol_day_date",
    "latest_basic_date",
    "latest_mktvalue_date",
    "latest_valuation_date",
    "latest_processed_date",
    "last_checked_at",
]

GM_BAR_FIELDS = [
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "bob",
    "eob",
]

GM_BASIC_FIELDS = [
    "tclose",
    "turnrate",
    "ttl_shr",
    "circ_shr",
    "ttl_shr_unl",
    "ttl_shr_ltd",
    "a_shr_unl",
    "h_shr_unl",
]

GM_MKTVALUE_FIELDS = [
    "tot_mv",
    "tot_mv_csrc",
    "a_mv",
    "a_mv_ex_ltd",
    "b_mv",
    "b_mv_ex_ltd",
    "ev",
    "ev_ex_curr",
    "ev_ebitda",
    "equity_value",
]

GM_VALUATION_FIELDS = [
    "pe_ttm",
    "pe_lyr",
    "pe_mrq",
    "pe_1q",
    "pe_2q",
    "pe_3q",
    "pe_ttm_cut",
    "pe_lyr_cut",
    "pe_mrq_cut",
    "pe_1q_cut",
    "pe_2q_cut",
    "pe_3q_cut",
    "pb_lyr",
    "pb_mrq",
    "pb_lyr_1",
    "pb_mrq_1",
    "pcf_ttm_oper",
    "pcf_ttm_ncf",
    "pcf_lyr_oper",
    "pcf_lyr_ncf",
    "ps_ttm",
    "ps_lyr",
    "ps_mrq",
    "ps_1q",
    "ps_2q",
    "ps_3q",
    "peg_lyr",
    "peg_mrq",
    "peg_1q",
    "peg_2q",
    "peg_3q",
    "peg_np_cgr",
    "peg_npp_cgr",
    "dy_ttm",
    "dy_lfy",
]

GM_PROCESSED_CANONICAL_COLS = [
    "date",
    "symbol",
    "gm_symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
    "total_mv",
    "circ_mv",
    "total_share",
    "circ_share",
    "pe_ttm",
    "pe_static",
    "pb",
    "peg",
    "pcf",
    "ps",
    "upper_limit",
    "lower_limit",
    "adj_factor",
    "is_suspended",
    "is_st",
    "adj_factor_bwd",
    "adj_factor_bwd_acc",
    "adj_factor_fwd",
    "adj_factor_fwd_acc",
    "tot_mv",
    "tot_mv_csrc",
    "a_mv",
    "a_mv_ex_ltd",
    "ttl_shr",
    "circ_shr",
    "ttl_shr_unl",
    "ttl_shr_ltd",
    "a_shr_unl",
    "h_shr_unl",
    "turnrate",
    "tclose",
    "pe_lyr",
    "pe_mrq",
    "pb_lyr",
    "pb_mrq",
    "pcf_ttm_oper",
    "pcf_ttm_ncf",
    "ps_ttm",
    "ps_mrq",
    "peg_lyr",
    "peg_np_cgr",
    "dy_ttm",
]

UINT32_MAX = np.iinfo(np.uint32).max
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


for directory in [
    RAW_META_DIR,
    RAW_BARS_DIR,
    RAW_SYMBOL_DAY_DIR,
    RAW_BASIC_DIR,
    RAW_MKTVALUE_DIR,
    RAW_VALUATION_DIR,
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


@dataclass(slots=True)
class SymbolState:
    symbol: str
    bars_latest: pd.Timestamp | None
    meta_latest: pd.Timestamp | None
    basic_latest: pd.Timestamp | None
    mktvalue_latest: pd.Timestamp | None
    valuation_latest: pd.Timestamp | None
    processed_latest: pd.Timestamp | None


@dataclass(slots=True)
class StageUpdatePlan:
    stage_name: str
    latest_by_symbol: dict[str, pd.Timestamp | None]
    pending_symbols: list[str]
    ready_outputs: dict[str, UpdateResult]


class StageStatusPanel:
    def __init__(
        self,
        stage_names: list[str],
        raw_total: int,
        started_at: float | None = None,
    ) -> None:
        self.stage_names = stage_names
        self._isatty = sys.stdout.isatty()
        self._printed_lines = 0
        self._started_at = (
            time.monotonic() if started_at is None else float(started_at)
        )
        self._stats = {
            name: {
                "total": raw_total if name != "processed" else 0,
                "done": 0,
                "running": 0,
                "success": 0,
                "failed": 0,
                "state": "pending",
            }
            for name in stage_names
        }

    def _clear_rendered_block(self) -> None:
        if not self._isatty or self._printed_lines <= 0:
            return
        sys.stdout.write(f"\x1b[{self._printed_lines}F")
        for _ in range(self._printed_lines):
            sys.stdout.write("\x1b[2K")
            sys.stdout.write("\x1b[1E")
        sys.stdout.write(f"\x1b[{self._printed_lines}F")

    def _summary_line(self) -> str:
        done = sum(int(stats["done"]) for stats in self._stats.values())
        ok = sum(int(stats["success"]) for stats in self._stats.values())
        failed = sum(int(stats["failed"]) for stats in self._stats.values())
        total = sum(
            int(stats["total"])
            for stats in self._stats.values()
            if int(stats["total"]) > 0
        )
        elapsed = max(time.monotonic() - self._started_at, 1e-9)
        total_text = f"{done}/{total}" if total > 0 else f"{done}/-"
        return (
            f"Elapsed {elapsed:7.1f}s | Tasks {total_text:<11} | "
            f"OK {ok:<5} | Fail {failed:<5} | Avg {ok / elapsed:6.2f} parquet/s"
        )

    def log(self, message: str) -> None:
        if not self._isatty:
            print(message)
            return
        self._clear_rendered_block()
        self._printed_lines = 0
        print(message)

    def set_total(self, stage_name: str, total: int) -> None:
        self._stats[stage_name]["total"] = int(total)
        if total == 0 and self._stats[stage_name]["done"] == 0:
            self._stats[stage_name]["state"] = "skipped"

    def mark_pending(self, stage_name: str) -> None:
        if self._stats[stage_name]["done"] == 0:
            self._stats[stage_name]["state"] = "pending"

    def mark_running(self, stage_name: str, running: int = 0) -> None:
        stats = self._stats[stage_name]
        if stats["state"] == "done":
            stats["running"] = 0
            return
        stats["state"] = "running"
        stats["running"] = max(0, int(running))

    def set_running(self, stage_name: str, running: int) -> None:
        stats = self._stats[stage_name]
        if stats["state"] == "done":
            stats["running"] = 0
            return
        stats["running"] = max(0, int(running))
        if stats["running"] > 0:
            stats["state"] = "running"

    def handle_event(
        self, event_type: str, stage_name: str, ok: bool | None = None
    ) -> None:
        stats = self._stats[stage_name]
        if event_type == "start":
            if stats["state"] in {"pending", "skipped"}:
                stats["state"] = "running"
            stats["running"] += 1
            return
        if event_type != "finish":
            return
        stats["running"] = max(0, stats["running"] - 1)
        stats["done"] += 1
        if ok:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        total = stats["total"]
        if total > 0 and stats["done"] >= total:
            stats["state"] = "done"
        elif stats["running"] > 0:
            stats["state"] = "running"
        else:
            stats["state"] = "running"

    def _build_lines(self) -> list[str]:
        lines = [
            "GM Download Status",
            self._summary_line(),
            f"{'Stage':<16} {'State':<8} {'Done/Pend':>11} {'Run':>5} {'OK':>5} {'Fail':>5}",
            "-" * 58,
        ]
        for name in self.stage_names:
            stats = self._stats[name]
            total = stats["total"]
            done_text = (
                f"{stats['done']:>5}/{total:<5}"
                if total > 0
                else f"{stats['done']:>5}/{'-':<5}"
            )
            lines.append(
                f"{name:<16} {stats['state']:<8} {done_text:>11} {stats['running']:>5} "
                f"{stats['success']:>5} {stats['failed']:>5}"
            )
        return lines

    def render(self, final: bool = False) -> None:
        lines = self._build_lines()
        if not self._isatty:
            if final:
                print("\n".join(lines))
            return
        self._clear_rendered_block()
        for line in lines:
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(line)
            sys.stdout.write("\n")
        sys.stdout.flush()
        self._printed_lines = len(lines)
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._printed_lines = 0


def _chunked(
    items: list[str], chunk_size: int = MAX_FIELDS_PER_REQUEST
) -> list[list[str]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def gm_symbol_to_local(symbol: str) -> str:
    if "." not in symbol:
        return symbol.strip()
    return symbol.split(".", 1)[1].strip()


def local_symbol_to_gm(symbol: str) -> str:
    code = str(symbol).strip()
    if "." in code:
        return code
    if code.startswith(("600", "601", "603", "605", "688")):
        return f"SHSE.{code}"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"SZSE.{code}"
    raise ValueError(f"unsupported A-share code: {symbol}")


def _normalize_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return (
            pd.to_datetime(series, errors="coerce").dt.tz_localize(None).dt.normalize()
        )
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _normalize_raw_frame_dates(
    df: pd.DataFrame, date_column: str = "trade_date"
) -> pd.DataFrame:
    out = df.copy()
    out[date_column] = _normalize_date_series(out[date_column])
    out = out.dropna(subset=[date_column]).drop_duplicates(
        subset=["symbol", date_column], keep="last"
    )
    out = out.sort_values([date_column, "symbol"]).reset_index(drop=True)
    return out


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
    tmp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        optimized.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _read_parquet_safe(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
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


def infer_latest_date(
    path: Path, date_column: str = "trade_date"
) -> pd.Timestamp | None:
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


def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return _read_parquet_safe(path)


def infer_latest_flag(
    path: Path,
    value_column: str,
    date_column: str = "trade_date",
) -> bool | None:
    frame = _read_parquet_safe(path, columns=[date_column, value_column])
    if frame is None:
        return None
    if frame.empty or value_column not in frame.columns or date_column not in frame.columns:
        return None
    frame = frame.copy()
    frame[date_column] = _normalize_date_series(frame[date_column])
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)
    if frame.empty:
        return None
    value = frame.iloc[-1][value_column]
    if pd.isna(value):
        return None
    return bool(value)


def _series_or_nan(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index, dtype=float)


def combine_first_many(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        raise ValueError("at least one series is required")
    result = series_list[0].copy()
    for series in series_list[1:]:
        result = result.combine_first(series)
    return result


def normalize_symbol_cache_frame(
    df: pd.DataFrame, fetched_at: pd.Timestamp | None = None
) -> pd.DataFrame:
    required = {"symbol", "sec_id", "sec_name", "exchange"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"symbol cache missing required columns: {sorted(missing)}")

    out = df.copy()
    if "sec_type1" in out.columns:
        out = out[out["sec_type1"] == GM_STOCK_SEC_TYPE1]
    if "sec_type2" in out.columns:
        out = out[out["sec_type2"] == GM_A_SHARE_SEC_TYPE2]
    out["local_symbol"] = out["sec_id"].astype(str).str.strip()
    out = out[
        out["local_symbol"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")
    ]
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["sec_name"] = out["sec_name"].astype(str).str.strip()
    out["fetched_at"] = fetched_at or pd.Timestamp.now().normalize()
    for col in ["listed_date", "delisted_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    out = (
        out.drop_duplicates(subset=["local_symbol"], keep="last")
        .sort_values("local_symbol")
        .reset_index(drop=True)
    )
    return out.reindex(columns=SYMBOL_CACHE_COLS)


def save_symbol_cache(df: pd.DataFrame) -> None:
    save_optimized_parquet(df.reindex(columns=SYMBOL_CACHE_COLS), SYMBOL_CACHE_PATH)


def load_symbol_cache() -> pd.DataFrame | None:
    if not SYMBOL_CACHE_PATH.exists():
        return None
    df = pd.read_parquet(SYMBOL_CACHE_PATH)
    return normalize_symbol_cache_frame(df)


def save_symbol_lifecycle(df: pd.DataFrame) -> None:
    save_optimized_parquet(df.reindex(columns=SYMBOL_LIFECYCLE_COLS), SYMBOL_LIFECYCLE_PATH)


def load_symbol_lifecycle() -> pd.DataFrame | None:
    if not SYMBOL_LIFECYCLE_PATH.exists():
        return None
    df = pd.read_parquet(SYMBOL_LIFECYCLE_PATH)
    for col in [
        "listed_date",
        "delisted_date",
        "effective_end_date",
        "latest_bars_date",
        "latest_symbol_day_date",
        "latest_basic_date",
        "latest_mktvalue_date",
        "latest_valuation_date",
        "latest_processed_date",
        "last_checked_at",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.reindex(columns=SYMBOL_LIFECYCLE_COLS)


def configure_gm_token(token_env: str = DEFAULT_TOKEN_ENV) -> str:
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise SystemExit(
            f"[!] Missing GM token. Export {token_env}=<your_token> before running collector_gm.py"
        )
    gmapi.set_token(token)
    return token


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


ENDPOINT_LIMITERS = {
    name: EndpointRateLimiter(interval)
    for name, interval in ENDPOINT_MIN_INTERVAL_SECONDS.items()
}


def _gm_call(
    endpoint_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    ENDPOINT_LIMITERS[endpoint_name].wait()
    return func(*args, **kwargs)


def refresh_symbol_cache() -> pd.DataFrame:
    existing = load_symbol_cache()
    frame = _gm_call(
        "symbol_cache",
        gmapi.get_symbol_infos,
        sec_type1=GM_STOCK_SEC_TYPE1,
        sec_type2=GM_A_SHARE_SEC_TYPE2,
        exchanges=["SHSE", "SZSE"],
        df=True,
    )
    if frame is None or frame.empty:
        raise RuntimeError("GM get_symbol_infos returned empty stock universe")
    normalized = normalize_symbol_cache_frame(frame)
    if existing is not None and not existing.empty:
        normalized = (
            pd.concat([normalized, existing], ignore_index=True)
            .drop_duplicates(subset=["local_symbol"], keep="first")
            .sort_values("local_symbol")
            .reset_index(drop=True)
        )
    save_symbol_cache(normalized)
    print(f"[*] GM symbol cache refreshed: {len(normalized)} A-share symbols.")
    return normalized


def resolve_all_symbols(refresh_live: bool = False) -> list[str]:
    cached = None if refresh_live else load_symbol_cache()
    if cached is None:
        cached = refresh_symbol_cache()
    else:
        print(f"[*] Using cached GM symbol list ({len(cached)} symbols).")
    local_symbols = set(list_local_symbols())
    cached_symbols = set(cached["local_symbol"].astype(str).tolist())
    resolved = sorted(local_symbols | cached_symbols)
    if local_symbols:
        print(
            f"[*] GM all-symbol set: {len(resolved)} "
            f"(local={len(local_symbols)}, cached={len(cached_symbols)})"
        )
    return resolved


def list_local_symbols() -> list[str]:
    roots = [
        RAW_BARS_DIR,
        RAW_SYMBOL_DAY_DIR,
        RAW_BASIC_DIR,
        RAW_MKTVALUE_DIR,
        RAW_VALUATION_DIR,
        PROCESSED_DIR,
    ]
    symbols = {path.stem for root in roots for path in root.glob("*.parquet")}
    return sorted(symbols)


def resolve_incremental_symbols(refresh_live: bool = False) -> list[str]:
    local_symbols = set(list_local_symbols())
    cached = None if refresh_live else load_symbol_cache()
    if cached is None:
        cached = refresh_symbol_cache()
    cached_symbols = set(cached["local_symbol"].astype(str).tolist())
    resolved = sorted(local_symbols | cached_symbols)
    print(
        f"[*] GM incremental symbol set: {len(resolved)} "
        f"(local={len(local_symbols)}, cached={len(cached_symbols)})"
    )
    return resolved


def fetch_bars_raw(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    gm_symbol = local_symbol_to_gm(symbol)
    frame = _gm_call(
        "history",
        gmapi.history,
        symbol=gm_symbol,
        frequency="1d",
        start_time=start_date,
        end_time=end_date,
        fields=",".join(GM_BAR_FIELDS),
        adjust=gmapi.ADJUST_NONE,
        df=True,
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *GM_BAR_FIELDS[1:]])
    out = frame.copy()
    if "eob" in out.columns:
        out["trade_date"] = _normalize_date_series(out["eob"])
    elif "bob" in out.columns:
        out["trade_date"] = _normalize_date_series(out["bob"])
    else:
        raise ValueError("GM history bars missing both eob and bob columns")
    return _normalize_raw_frame_dates(out, "trade_date")


def fetch_symbol_day_raw(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    gm_symbol = local_symbol_to_gm(symbol)
    frame = _gm_call(
        "history_symbol",
        gmapi.get_history_symbol,
        gm_symbol,
        start_date=start_date,
        end_date=end_date,
        df=True,
    )
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date"])
    return _normalize_raw_frame_dates(frame, "trade_date")


def fetch_chunked_daily_endpoint(
    symbol: str,
    endpoint_name: str,
    fields: list[str],
    fetcher: Callable[..., pd.DataFrame | list[dict[str, Any]]],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    gm_symbol = local_symbol_to_gm(symbol)
    merged: pd.DataFrame | None = None
    for chunk in _chunked(fields):
        frame = _gm_call(
            endpoint_name,
            fetcher,
            gm_symbol,
            ",".join(chunk),
            start_date=start_date,
            end_date=end_date,
            df=True,
        )
        if frame is None or len(frame) == 0:
            continue
        current = pd.DataFrame(frame)
        if current.empty:
            continue
        current = _normalize_raw_frame_dates(current, "trade_date")
        merged = (
            current
            if merged is None
            else merged.merge(current, on=["symbol", "trade_date"], how="outer")
        )

    if merged is None:
        return pd.DataFrame(columns=["symbol", "trade_date", *fields])
    return _normalize_raw_frame_dates(merged, "trade_date")


def fetch_basic_raw(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_chunked_daily_endpoint(
        symbol,
        "daily_basic",
        GM_BASIC_FIELDS,
        gmapi.stk_get_daily_basic,
        start_date,
        end_date,
    )


def fetch_mktvalue_raw(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_chunked_daily_endpoint(
        symbol,
        "daily_mktvalue",
        GM_MKTVALUE_FIELDS,
        gmapi.stk_get_daily_mktvalue,
        start_date,
        end_date,
    )


def fetch_valuation_raw(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    return fetch_chunked_daily_endpoint(
        symbol,
        "daily_valuation",
        GM_VALUATION_FIELDS,
        gmapi.stk_get_daily_valuation,
        start_date,
        end_date,
    )


def update_raw_table(
    symbol: str,
    path: Path,
    fetcher: Callable[[str, str, str], pd.DataFrame],
    target_end_date: pd.Timestamp,
    latest: pd.Timestamp | None | object = ...,
) -> UpdateResult:
    if latest is ...:
        latest = infer_latest_date(path, "trade_date")
    if latest is not None and latest >= target_end_date:
        return UpdateResult(
            status=f"{path.parent.name} up-to-date at {latest.date()}",
            changed=False,
            latest=latest,
        )

    existing = load_parquet_if_exists(path)
    start_date = (
        DEFAULT_START_DATE
        if latest is None
        else (latest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )
    fetched = fetcher(symbol, start_date, target_end_date.strftime("%Y-%m-%d"))

    if fetched.empty:
        if existing is not None and not existing.empty:
            return UpdateResult(
                status=f"{path.parent.name} unchanged at {latest.date()}",
                changed=False,
                latest=latest,
            )
        raise RuntimeError(f"{path.parent.name} fetch returned empty data")

    combined = (
        fetched
        if existing is None
        else pd.concat([existing, fetched], ignore_index=True)
    )
    combined = _normalize_raw_frame_dates(combined, "trade_date")
    save_optimized_parquet(combined, path)
    latest = combined["trade_date"].max()
    return UpdateResult(
        status=f"{path.parent.name} saved to {latest.date()}",
        changed=True,
        latest=latest,
    )


def _merge_left(
    left: pd.DataFrame,
    right: pd.DataFrame | None,
    columns: list[str],
) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    use_cols = [col for col in columns if col in right.columns]
    if not use_cols:
        return left
    subset = right[["symbol", "trade_date", *use_cols]].copy()
    return left.merge(subset, on=["symbol", "trade_date"], how="left")


def build_processed_symbol_frame(symbol: str) -> pd.DataFrame:
    bars = load_parquet_if_exists(RAW_BARS_DIR / f"{symbol}.parquet")
    if bars is None or bars.empty:
        raise FileNotFoundError(f"missing GM bars parquet for {symbol}")

    bars = _normalize_raw_frame_dates(bars, "trade_date")
    out = bars.copy()

    symbol_day = load_parquet_if_exists(RAW_SYMBOL_DAY_DIR / f"{symbol}.parquet")
    basic = load_parquet_if_exists(RAW_BASIC_DIR / f"{symbol}.parquet")
    mktvalue = load_parquet_if_exists(RAW_MKTVALUE_DIR / f"{symbol}.parquet")
    valuation = load_parquet_if_exists(RAW_VALUATION_DIR / f"{symbol}.parquet")

    out = _merge_left(
        out,
        symbol_day,
        [
            "turn_rate",
            "upper_limit",
            "lower_limit",
            "adj_factor",
            "is_suspended",
            "is_st",
            "sec_name",
            "exchange",
        ],
    )
    out = _merge_left(out, basic, GM_BASIC_FIELDS)
    out = _merge_left(out, mktvalue, GM_MKTVALUE_FIELDS)
    out = _merge_left(out, valuation, GM_VALUATION_FIELDS)

    out["gm_symbol"] = out["symbol"].astype(str)
    out["symbol"] = out["gm_symbol"].map(gm_symbol_to_local)
    out["date"] = _normalize_date_series(out["trade_date"])

    pre_close = _series_or_nan(out, "pre_close")
    high = _series_or_nan(out, "high")
    low = _series_or_nan(out, "low")
    close = _series_or_nan(out, "close")

    out["turnover"] = combine_first_many(
        _series_or_nan(out, "turn_rate"), _series_or_nan(out, "turnrate")
    )
    out["pct_chg"] = np.where(pre_close > 0, (close / pre_close - 1.0) * 100.0, np.nan)
    out["change"] = np.where(pre_close.notna(), close - pre_close, np.nan)
    out["amplitude"] = np.where(pre_close > 0, (high - low) / pre_close * 100.0, np.nan)

    out["total_mv"] = _series_or_nan(out, "tot_mv")
    out["circ_mv"] = combine_first_many(
        _series_or_nan(out, "a_mv_ex_ltd"), _series_or_nan(out, "a_mv")
    )
    out["total_share"] = _series_or_nan(out, "ttl_shr")
    out["circ_share"] = combine_first_many(
        _series_or_nan(out, "ttl_shr_unl"), _series_or_nan(out, "circ_shr")
    )

    out["pe_ttm"] = _series_or_nan(out, "pe_ttm")
    out["pe_static"] = combine_first_many(
        _series_or_nan(out, "pe_lyr"), _series_or_nan(out, "pe_mrq")
    )
    out["pb"] = combine_first_many(
        _series_or_nan(out, "pb_mrq"), _series_or_nan(out, "pb_lyr")
    )
    out["peg"] = combine_first_many(
        _series_or_nan(out, "peg_lyr"), _series_or_nan(out, "peg_np_cgr")
    )
    out["pcf"] = combine_first_many(
        _series_or_nan(out, "pcf_ttm_oper"), _series_or_nan(out, "pcf_ttm_ncf")
    )
    out["ps"] = combine_first_many(
        _series_or_nan(out, "ps_ttm"), _series_or_nan(out, "ps_mrq")
    )

    for bool_col in ["is_suspended", "is_st"]:
        if bool_col in out.columns:
            out[bool_col] = out[bool_col].astype("boolean")

    for col in GM_PROCESSED_CANONICAL_COLS:
        if col not in out.columns:
            out[col] = np.nan

    out = out.reindex(columns=GM_PROCESSED_CANONICAL_COLS)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def rebuild_processed_from_local(symbol: str) -> str:
    processed = build_processed_symbol_frame(symbol)
    path = PROCESSED_DIR / f"{symbol}.parquet"
    save_optimized_parquet(processed, path)
    return f"processed rows={len(processed)} last_date={processed['date'].max().date()}"


def should_rebuild_processed(symbol: str, updates: list[UpdateResult]) -> bool:
    if any(item.changed for item in updates):
        return True
    path = PROCESSED_DIR / f"{symbol}.parquet"
    processed_latest = infer_latest_date(path, "date")
    bars_latest = infer_latest_date(RAW_BARS_DIR / f"{symbol}.parquet", "trade_date")
    if processed_latest is None or bars_latest is None:
        return True
    return processed_latest < bars_latest


def load_symbol_state(symbol: str) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        bars_latest=infer_latest_date(RAW_BARS_DIR / f"{symbol}.parquet", "trade_date"),
        meta_latest=infer_latest_date(
            RAW_SYMBOL_DAY_DIR / f"{symbol}.parquet", "trade_date"
        ),
        basic_latest=infer_latest_date(
            RAW_BASIC_DIR / f"{symbol}.parquet", "trade_date"
        ),
        mktvalue_latest=infer_latest_date(
            RAW_MKTVALUE_DIR / f"{symbol}.parquet", "trade_date"
        ),
        valuation_latest=infer_latest_date(
            RAW_VALUATION_DIR / f"{symbol}.parquet", "trade_date"
        ),
        processed_latest=infer_latest_date(PROCESSED_DIR / f"{symbol}.parquet", "date"),
    )


def _normalize_optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def resolve_symbol_lifecycle_status(
    target_end_date: pd.Timestamp,
    listed_date: Any = None,
    delisted_date: Any = None,
    latest_is_suspended: bool | None = None,
) -> str:
    listed_ts = _normalize_optional_timestamp(listed_date)
    delisted_ts = _normalize_optional_timestamp(delisted_date)
    if listed_ts is not None and listed_ts > target_end_date:
        return "not_yet_listed"
    if delisted_ts is not None and delisted_ts <= target_end_date:
        return "delisted"
    if latest_is_suspended is True:
        return "suspended"
    return "active"


def resolve_effective_end_date(
    target_end_date: pd.Timestamp,
    listed_date: Any = None,
    delisted_date: Any = None,
    latest_is_suspended: bool | None = None,
) -> pd.Timestamp:
    status = resolve_symbol_lifecycle_status(
        target_end_date,
        listed_date,
        delisted_date,
        latest_is_suspended=latest_is_suspended,
    )
    if status == "delisted":
        return _normalize_optional_timestamp(delisted_date) or target_end_date
    return target_end_date


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
        cache_by_symbol = (
            cached.set_index("local_symbol", drop=False).to_dict(orient="index")
        )

    print(f"[*] Building GM symbol lifecycle registry for {len(symbols)} symbols...")
    rows: list[dict[str, Any]] = []
    checked_at = pd.Timestamp.now().normalize()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(load_symbol_state, symbol): symbol for symbol in symbols
        }
        pbar = tqdm(
            as_completed(future_map), total=len(symbols), desc="lifecycle", unit="symbol"
        )
        for future in pbar:
            symbol = future_map[future]
            state = future.result()
            cache_row = cache_by_symbol.get(symbol, {})
            listed_date = _normalize_optional_timestamp(cache_row.get("listed_date"))
            delisted_date = _normalize_optional_timestamp(cache_row.get("delisted_date"))
            latest_is_suspended = infer_latest_flag(
                RAW_SYMBOL_DAY_DIR / f"{symbol}.parquet",
                "is_suspended",
                date_column="trade_date",
            )
            lifecycle_status = resolve_symbol_lifecycle_status(
                target_end_date,
                listed_date=listed_date,
                delisted_date=delisted_date,
                latest_is_suspended=latest_is_suspended,
            )
            effective_end_date = resolve_effective_end_date(
                target_end_date,
                listed_date=listed_date,
                delisted_date=delisted_date,
                latest_is_suspended=latest_is_suspended,
            )
            rows.append(
                {
                    "local_symbol": symbol,
                    "gm_symbol": cache_row.get("symbol", local_symbol_to_gm(symbol)),
                    "listed_date": listed_date,
                    "delisted_date": delisted_date,
                    "lifecycle_status": lifecycle_status,
                    "latest_is_suspended": latest_is_suspended,
                    "effective_end_date": effective_end_date,
                    "latest_bars_date": state.bars_latest,
                    "latest_symbol_day_date": state.meta_latest,
                    "latest_basic_date": state.basic_latest,
                    "latest_mktvalue_date": state.mktvalue_latest,
                    "latest_valuation_date": state.valuation_latest,
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
        effective_end = _normalize_optional_timestamp(row["effective_end_date"])
        if effective_end is None:
            pending.append(symbol)
            continue
        effective_end_dates[symbol] = effective_end
        state = SymbolState(
            symbol=symbol,
            bars_latest=_normalize_optional_timestamp(row["latest_bars_date"]),
            meta_latest=_normalize_optional_timestamp(row["latest_symbol_day_date"]),
            basic_latest=_normalize_optional_timestamp(row["latest_basic_date"]),
            mktvalue_latest=_normalize_optional_timestamp(row["latest_mktvalue_date"]),
            valuation_latest=_normalize_optional_timestamp(row["latest_valuation_date"]),
            processed_latest=_normalize_optional_timestamp(row["latest_processed_date"]),
        )
        if is_symbol_complete(state, effective_end):
            completed.append(symbol)
        else:
            pending.append(symbol)

    return pending, completed, effective_end_dates


def is_symbol_complete(state: SymbolState, target_end_date: pd.Timestamp) -> bool:
    required = [
        state.bars_latest,
        state.meta_latest,
        state.basic_latest,
        state.mktvalue_latest,
        state.valuation_latest,
    ]
    if any(item is None or item < target_end_date for item in required):
        return False
    if state.processed_latest is None:
        return False
    return state.processed_latest >= state.bars_latest


def precheck_pending_symbols(
    symbols: list[str],
    target_end_date: pd.Timestamp,
    max_workers: int = PRECHECK_WORKERS,
) -> tuple[list[str], list[str], dict[str, pd.Timestamp], pd.DataFrame]:
    if not symbols:
        empty = pd.DataFrame(columns=SYMBOL_LIFECYCLE_COLS)
        return [], [], {}, empty

    print(
        f"[*] Scanning {len(symbols)} GM symbols for lifecycle/completed state with {max_workers} workers..."
    )
    lifecycle_registry = build_symbol_lifecycle_registry(
        symbols,
        target_end_date=target_end_date,
        max_workers=max_workers,
    )
    pending, completed, effective_end_dates = split_symbols_by_completion(
        symbols, lifecycle_registry
    )
    print(f"[*] GM precheck done. completed={len(completed)}, pending={len(pending)}")
    return pending, completed, effective_end_dates, lifecycle_registry


def precheck_stage_updates(
    symbols: list[str],
    stage_name: str,
    stage_dir: Path,
    date_column: str,
    target_end_date: pd.Timestamp,
    effective_end_dates: dict[str, pd.Timestamp] | None = None,
    max_workers: int = PRECHECK_WORKERS,
) -> StageUpdatePlan:
    if not symbols:
        return StageUpdatePlan(
            stage_name=stage_name,
            latest_by_symbol={},
            pending_symbols=[],
            ready_outputs={},
        )

    print(f"[*] Prechecking local {stage_name} shards with {max_workers} workers...")
    latest_by_symbol: dict[str, pd.Timestamp | None] = {}
    ready_outputs: dict[str, UpdateResult] = {}
    pending_set: set[str] = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                infer_latest_date, stage_dir / f"{symbol}.parquet", date_column
            ): symbol
            for symbol in symbols
        }
        ready_count = 0
        pbar = tqdm(
            as_completed(future_map),
            total=len(symbols),
            desc=f"precheck {stage_name}",
            unit="symbol",
        )
        for future in pbar:
            symbol = future_map[future]
            latest = future.result()
            latest_by_symbol[symbol] = latest
            required_end_date = (
                effective_end_dates.get(symbol, target_end_date)
                if effective_end_dates is not None
                else target_end_date
            )
            if latest is not None and latest >= required_end_date:
                ready_outputs[symbol] = UpdateResult(
                    status=f"{stage_name} up-to-date at {latest.date()}",
                    changed=False,
                    latest=latest,
                )
                ready_count += 1
            else:
                pending_set.add(symbol)
            pbar.set_postfix({"ready": ready_count, "pending": len(pending_set)})

    pending_symbols = [symbol for symbol in symbols if symbol in pending_set]
    print(
        f"[*] {stage_name}: ready={len(ready_outputs)}, pending={len(pending_symbols)}"
    )
    return StageUpdatePlan(
        stage_name=stage_name,
        latest_by_symbol=latest_by_symbol,
        pending_symbols=pending_symbols,
        ready_outputs=ready_outputs,
    )


def update_and_rebuild_symbol(symbol: str, target_end_date: pd.Timestamp) -> TaskResult:
    try:
        stage_specs: list[tuple[str, Callable[[], UpdateResult]]] = [
            (
                "bars_raw",
                lambda: update_raw_table(
                    symbol,
                    RAW_BARS_DIR / f"{symbol}.parquet",
                    fetch_bars_raw,
                    target_end_date,
                ),
            ),
            (
                "symbol_day",
                lambda: update_raw_table(
                    symbol,
                    RAW_SYMBOL_DAY_DIR / f"{symbol}.parquet",
                    fetch_symbol_day_raw,
                    target_end_date,
                ),
            ),
            (
                "daily_basic",
                lambda: update_raw_table(
                    symbol,
                    RAW_BASIC_DIR / f"{symbol}.parquet",
                    fetch_basic_raw,
                    target_end_date,
                ),
            ),
            (
                "daily_mktvalue",
                lambda: update_raw_table(
                    symbol,
                    RAW_MKTVALUE_DIR / f"{symbol}.parquet",
                    fetch_mktvalue_raw,
                    target_end_date,
                ),
            ),
            (
                "daily_valuation",
                lambda: update_raw_table(
                    symbol,
                    RAW_VALUATION_DIR / f"{symbol}.parquet",
                    fetch_valuation_raw,
                    target_end_date,
                ),
            ),
        ]
        updates_by_stage: dict[str, UpdateResult] = {}
        with ThreadPoolExecutor(max_workers=len(stage_specs)) as executor:
            future_map = {
                executor.submit(worker): stage_name
                for stage_name, worker in stage_specs
            }
            for future in as_completed(future_map):
                stage_name = future_map[future]
                updates_by_stage[stage_name] = future.result()
        updates = [updates_by_stage[name] for name, _ in stage_specs]
        if should_rebuild_processed(symbol, updates):
            processed_status = rebuild_processed_from_local(symbol)
        else:
            processed_latest = infer_latest_date(
                PROCESSED_DIR / f"{symbol}.parquet", "date"
            )
            processed_status = f"processed up-to-date at {processed_latest.date()}"
        detail = "; ".join(item.status for item in updates) + f"; {processed_status}"
        return TaskResult(symbol=symbol, ok=True, detail=detail)
    except Exception as exc:
        return TaskResult(
            symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )


def rebuild_symbol(symbol: str) -> TaskResult:
    try:
        detail = rebuild_processed_from_local(symbol)
        return TaskResult(symbol=symbol, ok=True, detail=detail)
    except Exception as exc:
        return TaskResult(
            symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )


def _run_stage_worker(
    stage_name: str,
    symbol: str,
    worker: Callable[[str], Any],
) -> StageTaskResult:
    try:
        payload = worker(symbol)
        if isinstance(payload, UpdateResult):
            detail = payload.status
        else:
            detail = str(payload)
        result = StageTaskResult(symbol=symbol, ok=True, detail=detail, payload=payload)
    except Exception as exc:
        result = StageTaskResult(
            symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}"
        )
    return result


def collect_stage(
    symbols: list[str],
    stage_name: str,
    worker: Callable[[str], Any],
    max_workers: int,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    panel: StageStatusPanel | None = None,
) -> list[StageTaskResult]:
    results: list[StageTaskResult] = []
    ok_count = 0
    consecutive_failures = 0
    abort_message: str | None = None
    use_inline_panel = panel is not None
    total = len(symbols)
    last_render = 0.0

    if use_inline_panel and total > 0:
        panel.mark_running(stage_name, min(max_workers, total))
        panel.render()
        last_render = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_run_stage_worker, stage_name, symbol, worker): symbol
            for symbol in symbols
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            if result.ok:
                ok_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if use_inline_panel:
                panel.handle_event("finish", stage_name, result.ok)
                remaining = total - len(results)
                panel.set_running(stage_name, min(max_workers, remaining))
                now = time.monotonic()
                if now - last_render >= 0.2:
                    panel.render()
                    last_render = now
            if consecutive_failures >= max_consecutive_failures:
                abort_message = (
                    f"[!] Aborting stage '{stage_name}' after {consecutive_failures} consecutive GM failures. "
                    "Inspect token validity or the failing symbols before resuming."
                )
                if use_inline_panel:
                    panel.log(abort_message)
                else:
                    tqdm.write(abort_message)
                for pending_future in future_map:
                    if not pending_future.done():
                        pending_future.cancel()
                break
    if abort_message is not None:
        raise SystemExit(abort_message)
    if use_inline_panel:
        panel.render()
    return results


def collect_symbols(
    symbols: list[str],
    worker: Callable[[str], TaskResult],
    max_workers: int,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    ok_count = 0
    consecutive_failures = 0
    abort_message: str | None = None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, symbol): symbol for symbol in symbols}
        pbar = tqdm(as_completed(future_map), total=len(symbols))
        for future in pbar:
            result = future.result()
            results.append(result)
            if result.ok:
                ok_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            pbar.set_postfix({"success": ok_count, "failed": len(results) - ok_count})
            if consecutive_failures >= max_consecutive_failures:
                abort_message = (
                    f"[!] Aborting after {consecutive_failures} consecutive GM failures. "
                    "Inspect token validity or the failing symbol before resuming."
                )
                print(abort_message)
                for pending_future in future_map:
                    if not pending_future.done():
                        pending_future.cancel()
                break
    if abort_message is not None:
        raise SystemExit(abort_message)
    return results


def parse_symbols_arg(symbols: str | None) -> list[str]:
    if not symbols:
        return []
    return sorted(
        {
            gm_symbol_to_local(item.strip())
            for item in symbols.split(",")
            if item.strip()
        }
    )


def resolve_target_end_date(end_date: str | None = None) -> pd.Timestamp:
    if end_date:
        return pd.Timestamp(end_date).normalize()
    return pd.Timestamp.today().normalize()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch GM daily data into raw full-field parquet and normalized processed parquet."
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all symbols from GM symbol cache."
    )
    parser.add_argument(
        "--symbols", help="Comma-separated symbols, supports bare code or GM symbol."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch/update all locally known or cached GM symbols.",
    )
    parser.add_argument(
        "--rebuild-processed",
        action="store_true",
        help="Rebuild GM processed parquet from local raw files.",
    )
    parser.add_argument(
        "--refresh-symbols",
        action="store_true",
        help="Refresh live GM symbol cache before resolving symbols.",
    )
    parser.add_argument(
        "--refresh-symbols-only",
        action="store_true",
        help="Refresh GM symbol cache and exit.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--end-date",
        default=None,
        help="Target end date in YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help=f"Environment variable containing GM token. Default: {DEFAULT_TOKEN_ENV}",
    )
    return parser


def run_gm_update_pipeline(
    symbols: list[str],
    target_end_date: pd.Timestamp,
    max_workers: int,
    effective_end_dates: dict[str, pd.Timestamp] | None = None,
    started_at: float | None = None,
) -> list[TaskResult]:
    stage_specs: list[
        tuple[
            str,
            Path,
            str,
            Callable[[str, pd.Timestamp | None, pd.Timestamp], UpdateResult],
        ]
    ] = [
        (
            "bars_raw",
            RAW_BARS_DIR,
            "trade_date",
            lambda symbol, latest, symbol_target_end_date: update_raw_table(
                symbol,
                RAW_BARS_DIR / f"{symbol}.parquet",
                fetch_bars_raw,
                symbol_target_end_date,
                latest=latest,
            ),
        ),
        (
            "symbol_day",
            RAW_SYMBOL_DAY_DIR,
            "trade_date",
            lambda symbol, latest, symbol_target_end_date: update_raw_table(
                symbol,
                RAW_SYMBOL_DAY_DIR / f"{symbol}.parquet",
                fetch_symbol_day_raw,
                symbol_target_end_date,
                latest=latest,
            ),
        ),
        (
            "daily_basic",
            RAW_BASIC_DIR,
            "trade_date",
            lambda symbol, latest, symbol_target_end_date: update_raw_table(
                symbol,
                RAW_BASIC_DIR / f"{symbol}.parquet",
                fetch_basic_raw,
                symbol_target_end_date,
                latest=latest,
            ),
        ),
        (
            "daily_mktvalue",
            RAW_MKTVALUE_DIR,
            "trade_date",
            lambda symbol, latest, symbol_target_end_date: update_raw_table(
                symbol,
                RAW_MKTVALUE_DIR / f"{symbol}.parquet",
                fetch_mktvalue_raw,
                symbol_target_end_date,
                latest=latest,
            ),
        ),
        (
            "daily_valuation",
            RAW_VALUATION_DIR,
            "trade_date",
            lambda symbol, latest, symbol_target_end_date: update_raw_table(
                symbol,
                RAW_VALUATION_DIR / f"{symbol}.parquet",
                fetch_valuation_raw,
                symbol_target_end_date,
                latest=latest,
            ),
        ),
    ]

    panel = StageStatusPanel(
        [name for name, _, _, _ in stage_specs] + ["processed"],
        raw_total=0,
        started_at=started_at,
    )

    print(
        f"[*] Prechecking local GM raw shards sequentially by endpoint with {PRECHECK_WORKERS} workers..."
    )
    stage_plans = {
        stage_name: precheck_stage_updates(
            symbols,
            stage_name,
            stage_dir,
            date_column,
            target_end_date,
            effective_end_dates=effective_end_dates,
            max_workers=PRECHECK_WORKERS,
        )
        for stage_name, stage_dir, date_column, _ in stage_specs
    }

    stage_details: dict[str, list[str]] = {symbol: [] for symbol in symbols}
    stage_outputs: dict[str, dict[str, UpdateResult]] = {
        name: {} for name, _, _, _ in stage_specs
    }
    failed_symbols: set[str] = set()

    runnable_stage_specs: list[tuple[str, Callable[[str], UpdateResult], list[str]]] = (
        []
    )
    for stage_name, _, _, worker in stage_specs:
        plan = stage_plans[stage_name]
        stage_outputs[stage_name].update(plan.ready_outputs)
        for symbol, output in plan.ready_outputs.items():
            stage_details[symbol].append(f"{stage_name}: {output.status}")
        panel.set_total(stage_name, len(plan.pending_symbols))
        if not plan.pending_symbols:
            continue
        latest_by_symbol = plan.latest_by_symbol
        runnable_stage_specs.append(
            (
                stage_name,
                lambda symbol, worker=worker, latest_by_symbol=latest_by_symbol: worker(
                    symbol,
                    latest_by_symbol.get(symbol),
                    (effective_end_dates or {}).get(symbol, target_end_date),
                ),
                plan.pending_symbols,
            )
        )

    if runnable_stage_specs:
        print(
            "[*] Downloading raw GM tables stage-by-stage with single-layer symbol parallelism..."
        )
        panel.render()
        for stage_name, worker, stage_symbols in runnable_stage_specs:
            results = collect_stage(
                stage_symbols,
                stage_name,
                worker,
                max_workers,
                MAX_CONSECUTIVE_FAILURES,
                panel=panel,
            )
            for item in results:
                stage_details[item.symbol].append(f"{stage_name}: {item.detail}")
                if item.ok and isinstance(item.payload, UpdateResult):
                    stage_outputs[stage_name][item.symbol] = item.payload
                else:
                    failed_symbols.add(item.symbol)
    else:
        print("[*] All raw GM tables are already current. Skipping raw downloads.")

    build_symbols = [
        symbol
        for symbol in symbols
        if symbol not in failed_symbols
        and all(symbol in stage_outputs[name] for name, _, _, _ in stage_specs)
    ]
    if build_symbols:
        panel.log("[*] Raw downloads finished. Rebuilding processed parquet...")
        panel.set_total("processed", len(build_symbols))
        panel.mark_pending("processed")
        panel.render()

        def rebuild_if_needed(symbol: str) -> str:
            updates = [stage_outputs[name][symbol] for name, _, _, _ in stage_specs]
            if should_rebuild_processed(symbol, updates):
                return rebuild_processed_from_local(symbol)
            processed_latest = infer_latest_date(
                PROCESSED_DIR / f"{symbol}.parquet", "date"
            )
            return f"processed up-to-date at {processed_latest.date()}"

        processed_results = collect_stage(
            build_symbols,
            "processed",
            rebuild_if_needed,
            max_workers,
            MAX_CONSECUTIVE_FAILURES,
            panel=panel,
        )
        for item in processed_results:
            stage_details[item.symbol].append(f"processed: {item.detail}")
            if not item.ok:
                failed_symbols.add(item.symbol)
    else:
        panel.set_total("processed", 0)

    panel.render(final=True)

    final_results: list[TaskResult] = []
    for symbol in symbols:
        ok = symbol not in failed_symbols and any(
            detail.startswith("processed: ") for detail in stage_details[symbol]
        )
        final_results.append(
            TaskResult(symbol=symbol, ok=ok, detail="; ".join(stage_details[symbol]))
        )
    return final_results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    execution_started_at = time.monotonic()

    configure_gm_token(args.token_env)

    if args.refresh_symbols_only:
        refresh_symbol_cache()
        print("[+] GM symbol cache refresh completed.")
        return

    target_end_date = resolve_target_end_date(args.end_date)
    symbols = parse_symbols_arg(args.symbols)

    if args.rebuild_processed:
        if not symbols:
            symbols = list_local_symbols()
        if not symbols:
            raise SystemExit("[!] No local GM symbols found to rebuild.")
        print(
            f"[*] Rebuilding GM processed parquet for {len(symbols)} symbols from local raw files..."
        )
        results = collect_symbols(symbols, rebuild_symbol, max_workers=args.workers)
    else:
        if args.all:
            symbols = resolve_all_symbols(refresh_live=args.refresh_symbols)
        elif args.update and not symbols:
            symbols = resolve_incremental_symbols(refresh_live=args.refresh_symbols)
        elif not symbols:
            parser.error(
                "Provide one of --all, --update, --rebuild-processed, or --symbols."
            )

        if not symbols:
            raise SystemExit("[!] No GM symbols to process.")

        symbols, completed_symbols, effective_end_dates, lifecycle_registry = (
            precheck_pending_symbols(symbols, target_end_date=target_end_date)
        )
        if not lifecycle_registry.empty:
            lifecycle_counts = lifecycle_registry["lifecycle_status"].value_counts().to_dict()
            counts_text = ", ".join(
                f"{key}={value}" for key, value in sorted(lifecycle_counts.items())
            )
            print(f"[*] GM lifecycle snapshot: {counts_text}")
        if completed_symbols:
            print(f"[*] Skipping {len(completed_symbols)} already-complete GM symbols.")
        if not symbols:
            print("[+] All GM symbols are already complete. Nothing to update.")
            return

        limiter_desc = ", ".join(
            f"{name}={interval:.2f}s"
            for name, interval in ENDPOINT_MIN_INTERVAL_SECONDS.items()
        )
        print(
            f"[*] Updating {len(symbols)} pending GM symbols with {args.workers} workers (end_date={target_end_date.date()})..."
        )
        print(f"[*] Endpoint-local throttling: {limiter_desc}")
        results = run_gm_update_pipeline(
            symbols,
            target_end_date=target_end_date,
            max_workers=args.workers,
            effective_end_dates=effective_end_dates,
            started_at=execution_started_at,
        )

    success = sum(1 for item in results if item.ok)
    failed = [item for item in results if not item.ok]
    print(f"[+] GM collection done. Success: {success} / {len(results)}")
    if failed:
        print("[!] Failed GM symbols:")
        for item in failed[:20]:
            print(f"    {item.symbol}: {item.detail}")
        if len(failed) > 20:
            print(f"    ... {len(failed) - 20} more")


if __name__ == "__main__":
    main()
