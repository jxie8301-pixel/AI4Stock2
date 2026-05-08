from __future__ import annotations

import argparse
import math
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import akshare as ak
import akshare_proxy_patch as proxy_patch
from curl_cffi import requests as curl_requests
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from src.collector_io import (
    optimize_numeric_dtypes,
    read_parquet_safe,
    write_optimized_parquet_atomic,
)


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/center/gridlist.html",
    "Connection": "keep-alive",
}

RAW_DAILY_DIR = Path("data/raw/daily")
RAW_VAL_DIR = Path("data/raw/valuation")
RAW_META_DIR = Path("data/raw/meta")
PROCESSED_DIR = Path("data/processed/combined")
SYMBOL_CACHE_PATH = RAW_META_DIR / "stock_list.parquet"
STOCK_LIST_PAGE_DIR = RAW_META_DIR / "stock_list_pages"
STOCK_LIST_MANIFEST_PATH = RAW_META_DIR / "stock_list_manifest.json"
STOCK_LIST_URL = "https://82.push2.eastmoney.com/api/qt/clist/get"
STOCK_LIST_PAGE_SIZE = 100

DAILY_RENAME_MAP = {
    "日期": "date",
    "股票代码": "symbol",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover",
}

DAILY_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
]

VAL_RENAME_MAP = {
    "数据日期": "date",
    "当日收盘价": "v_close",
    "当日涨跌幅": "val_pct_chg",
    "总市值": "total_mv",
    "流通市值": "circ_mv",
    "总股本": "total_share",
    "流通股本": "circ_share",
    "PE(TTM)": "pe_ttm",
    "PE(静)": "pe_static",
    "市净率": "pb",
    "PEG值": "peg",
    "市现率": "pcf",
    "市销率": "ps",
}

VAL_COLS = [
    "date",
    "v_close",
    "val_pct_chg",
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
]

PROCESSED_COLS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "change",
    "turnover",
    "val_pct_chg",
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
]

SYMBOL_CACHE_COLS = ["symbol", "name", "fetched_at"]

NON_NEGATIVE_COLS = {"volume", "amount", "turnover", "total_mv", "circ_mv", "total_share", "circ_share"}
REQUEST_SLEEP_SECONDS = 0.5
MAX_CONSECUTIVE_FAILURES = 10
PRECHECK_WORKERS = 16
DEFAULT_PROXY_AUTH_IP = "101.201.173.125"
PROXY_HOOK_DOMAINS = [
    "push2.eastmoney.com",
    "push2his.eastmoney.com",
    "datacenter-web.eastmoney.com",
    "82.push2.eastmoney.com",
]


for directory in [RAW_DAILY_DIR, RAW_VAL_DIR, RAW_META_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
STOCK_LIST_PAGE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TaskResult:
    symbol: str
    ok: bool
    detail: str


@dataclass(slots=True)
class UpdateResult:
    status: str
    changed: bool
    latest: pd.Timestamp | None


@dataclass(slots=True)
class SymbolState:
    symbol: str
    daily_latest: pd.Timestamp | None
    valuation_latest: pd.Timestamp | None
    processed_latest: pd.Timestamp | None


class RequestPatcher:
    def __init__(self, cookies_file: str = "data/cookies.json") -> None:
        self.session = curl_requests.Session(impersonate="chrome120")
        self.session.headers.update(DEFAULT_HEADERS)
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        self.cookies_file = cookies_file

    def load_cookies(self) -> bool:
        path = Path(self.cookies_file)
        if not path.exists():
            print(f"[!] Warning: {self.cookies_file} not found. Running without custom cookies.")
            return False

        with path.open("r", encoding="utf-8") as fh:
            cookie_list = json.load(fh)
        cookies = {item["name"]: item["value"] for item in cookie_list}
        self.session.cookies.update(cookies)
        print(f"[*] Successfully loaded {len(cookie_list)} cookies.")
        return True

    def patch(self) -> None:
        def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            kwargs.pop("session", None)
            kwargs.pop("verify", None)
            kwargs.pop("stream", None)
            kwargs.pop("proxies", None)
            kwargs.setdefault("timeout", 30)
            return kwargs

        def patched_request(method: str, url: str, **kwargs: Any):
            _maybe_sleep_request_interval()
            return self.session.request(method=method, url=url, **normalize_kwargs(kwargs))

        def patched_session_request(_session_self, method: str, url: str, **kwargs: Any):
            _maybe_sleep_request_interval()
            return self.session.request(method=method, url=url, **normalize_kwargs(kwargs))

        def patched_get(url: str, **kwargs: Any):
            return patched_request("GET", url, **kwargs)

        def patched_post(url: str, **kwargs: Any):
            return patched_request("POST", url, **kwargs)

        requests.request = patched_request  # type: ignore[assignment]
        requests.get = patched_get  # type: ignore[assignment]
        requests.post = patched_post  # type: ignore[assignment]
        requests.sessions.Session.request = patched_session_request  # type: ignore[assignment]
        print("[*] Global requests patched with curl_cffi session.")


def _maybe_sleep_request_interval() -> None:
    if REQUEST_SLEEP_SECONDS > 0.0:
        time.sleep(REQUEST_SLEEP_SECONDS)


def install_proxy_patch(auth_token: str = "") -> None:
    proxy_patch.install_patch(
        auth_ip=DEFAULT_PROXY_AUTH_IP,
        auth_token=auth_token,
        hook_domains=PROXY_HOOK_DOMAINS,
    )
    print(
        "[*] Global requests patched with akshare-proxy-patch "
        f"(auth_ip={DEFAULT_PROXY_AUTH_IP}, hook_domains={','.join(PROXY_HOOK_DOMAINS)})."
    )


def resolve_target_end_date(end_date: str | None = None) -> pd.Timestamp:
    if end_date:
        return pd.Timestamp(end_date).normalize()
    return pd.Timestamp.today().normalize()


def normalize_stock_list_frame(df: pd.DataFrame, fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    if "代码" not in df.columns:
        raise ValueError("stock list missing required column: 代码")

    out = pd.DataFrame(
        {
            "symbol": df["代码"].astype(str).str.strip(),
            "name": df["名称"].astype(str).str.strip() if "名称" in df.columns else pd.Series("", index=df.index, dtype=object),
        }
    )
    out = out[out["symbol"].str.match(r"^(000|001|002|003|300|301|600|601|603|605|688)")]
    out = out[out["symbol"].ne("")]
    out["fetched_at"] = (fetched_at or pd.Timestamp.today().normalize())
    out = out.drop_duplicates(subset=["symbol"], keep="last").sort_values("symbol").reset_index(drop=True)
    return out.reindex(columns=SYMBOL_CACHE_COLS)


def stock_list_page_path(page_number: int) -> Path:
    return STOCK_LIST_PAGE_DIR / f"page_{page_number:04d}.parquet"


def load_stock_list_manifest() -> dict[str, Any] | None:
    if not STOCK_LIST_MANIFEST_PATH.exists():
        return None
    with STOCK_LIST_MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_stock_list_manifest(manifest: dict[str, Any]) -> None:
    with STOCK_LIST_MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def list_cached_stock_list_pages() -> list[int]:
    pages: list[int] = []
    for path in STOCK_LIST_PAGE_DIR.glob("page_*.parquet"):
        try:
            pages.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return sorted(set(pages))


def build_stock_list_page_params(page_number: int, page_size: int = STOCK_LIST_PAGE_SIZE) -> dict[str, str]:
    return {
        "pn": str(page_number),
        "pz": str(page_size),
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f12",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": (
            "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
            "f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152"
        ),
    }


def fetch_stock_list_page(page_number: int, page_size: int = STOCK_LIST_PAGE_SIZE) -> tuple[pd.DataFrame, int]:
    response = requests.get(STOCK_LIST_URL, params=build_stock_list_page_params(page_number, page_size))
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    diff = data.get("diff") or []
    total = int(data.get("total") or 0)
    if total <= 0:
        raise RuntimeError(f"stock list page {page_number} returned empty total")
    total_pages = math.ceil(total / page_size)
    frame = pd.DataFrame(diff)
    if frame.empty:
        raise RuntimeError(f"stock list page {page_number} returned empty diff")
    frame = frame.rename(columns={"f12": "代码", "f14": "名称"})
    normalized = normalize_stock_list_frame(frame)
    return normalized, total_pages


def save_stock_list_page(page_number: int, frame: pd.DataFrame) -> None:
    save_optimized_parquet(frame, stock_list_page_path(page_number))


def build_stock_list_cache_from_pages(total_pages: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for page_number in range(1, total_pages + 1):
        path = stock_list_page_path(page_number)
        if not path.exists():
            raise FileNotFoundError(f"missing stock list page cache: {path}")
        frames.append(pd.read_parquet(path))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol"], keep="last").sort_values("symbol").reset_index(drop=True)
    return merged.reindex(columns=SYMBOL_CACHE_COLS)


def refresh_stock_list_cache() -> pd.DataFrame:
    manifest = load_stock_list_manifest() or {}
    cached_pages = set(list_cached_stock_list_pages())
    total_pages = int(manifest.get("total_pages") or 0)

    if total_pages <= 0:
        print("[*] Fetching stock list page 1 to initialize paged cache...")
        page_frame, total_pages = fetch_stock_list_page(1)
        save_stock_list_page(1, page_frame)
        cached_pages.add(1)
        manifest = {
            "page_size": STOCK_LIST_PAGE_SIZE,
            "total_pages": total_pages,
            "completed_pages": sorted(cached_pages),
            "complete": False,
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        save_stock_list_manifest(manifest)
        print(f"[*] Stock list cache initialized: total_pages={total_pages}, cached_pages=1")

    missing_pages = [page for page in range(1, total_pages + 1) if page not in cached_pages]
    if missing_pages:
        print(
            f"[*] Refreshing stock list pages incrementally: "
            f"cached={len(cached_pages)}/{total_pages}, next_page={missing_pages[0]}"
        )

    for page_number in missing_pages:
        page_frame, discovered_total_pages = fetch_stock_list_page(page_number)
        if discovered_total_pages != total_pages:
            total_pages = discovered_total_pages
        save_stock_list_page(page_number, page_frame)
        cached_pages.add(page_number)
        manifest = {
            "page_size": STOCK_LIST_PAGE_SIZE,
            "total_pages": total_pages,
            "completed_pages": sorted(cached_pages),
            "complete": len(cached_pages) == total_pages,
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        save_stock_list_manifest(manifest)
        print(f"[*] Stock list page saved: {page_number}/{total_pages}")

    if len(cached_pages) != total_pages:
        raise RuntimeError(
            f"stock list refresh incomplete: cached {len(cached_pages)} / {total_pages} pages. "
            "Switch cookie and rerun with --refresh-stock-list or --refresh-stock-list-only."
        )

    merged = build_stock_list_cache_from_pages(total_pages)
    save_symbol_cache(merged)
    manifest["complete"] = True
    manifest["updated_at"] = pd.Timestamp.now().isoformat()
    save_stock_list_manifest(manifest)
    print(f"[*] Stock list cache finalized: {len(merged)} symbols")
    return merged


def load_symbol_cache() -> pd.DataFrame | None:
    if not SYMBOL_CACHE_PATH.exists():
        return None
    df = pd.read_parquet(SYMBOL_CACHE_PATH)
    expected = set(SYMBOL_CACHE_COLS)
    if not expected.issubset(df.columns):
        raise ValueError(f"symbol cache missing required columns: {sorted(expected - set(df.columns))}")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    if "name" in out.columns:
        out["name"] = out["name"].fillna("").astype(str)
    out["fetched_at"] = pd.to_datetime(out["fetched_at"], errors="coerce")
    out = out[out["symbol"].ne("")].drop_duplicates(subset=["symbol"], keep="last").sort_values("symbol").reset_index(drop=True)
    return out.reindex(columns=SYMBOL_CACHE_COLS)


def save_symbol_cache(df: pd.DataFrame) -> None:
    save_optimized_parquet(df.reindex(columns=SYMBOL_CACHE_COLS), SYMBOL_CACHE_PATH)


def fetch_stock_list_frame(save_cache: bool = True) -> pd.DataFrame:
    out = refresh_stock_list_cache()
    if save_cache:
        save_symbol_cache(out)
    return out


def fetch_stock_list() -> list[str]:
    return fetch_stock_list_frame(save_cache=True)["symbol"].tolist()


def resolve_all_symbols(refresh_live: bool = False) -> list[str]:
    cached = load_symbol_cache()
    if cached is not None and not refresh_live:
        symbols = cached["symbol"].tolist()
        print(f"[*] Using cached stock list ({len(symbols)} symbols).")
        return symbols

    if refresh_live:
        print("[*] Refreshing paged stock list cache because --refresh-stock-list was set.")
    else:
        print("[*] Stock list cache not found. Building paged stock list cache...")
    return fetch_stock_list()


def merge_incremental_symbol_sets(
    local_symbols: set[str],
    cached_symbols: set[str],
    live_symbols: set[str],
) -> list[str]:
    return sorted(local_symbols | cached_symbols | live_symbols)


def resolve_incremental_symbols(refresh_live: bool = False) -> list[str]:
    local_symbols = set(list_local_symbols())

    cached_symbols: set[str] = set()
    cached = load_symbol_cache()
    if cached is not None:
        cached_symbols = set(cached["symbol"].tolist())

    live_symbols: set[str] = set()
    if refresh_live:
        try:
            live = fetch_stock_list_frame(save_cache=True)
            live_symbols = set(live["symbol"].tolist())
            new_symbols = sorted(live_symbols - local_symbols)
            print(
                f"[*] Incremental symbol resolution: local={len(local_symbols)}, "
                f"live={len(live_symbols)}, newly_listed={len(new_symbols)}"
            )
        except Exception as exc:
            print(f"[!] Live stock-list refresh failed: {type(exc).__name__}: {exc}")
            if cached_symbols:
                print(f"[*] Falling back to cached stock list ({len(cached_symbols)} symbols).")
            else:
                print("[*] No cached stock list available. Falling back to local symbols only.")
    else:
        print("[*] Using local + cached stock list only. Skip live stock-list refresh.")

    resolved = merge_incremental_symbol_sets(local_symbols, cached_symbols, live_symbols)
    print(
        f"[*] Final incremental symbol set: {len(resolved)} "
        f"(local={len(local_symbols)}, cached={len(cached_symbols)}, live={len(live_symbols)})"
    )
    return resolved


def list_local_symbols() -> list[str]:
    symbols = {
        path.stem
        for root in [RAW_DAILY_DIR, RAW_VAL_DIR, PROCESSED_DIR]
        for path in root.glob("*.parquet")
    }
    return sorted(symbols)


def save_optimized_parquet(df: pd.DataFrame, path: Path) -> None:
    write_optimized_parquet_atomic(df, path)


def _optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return optimize_numeric_dtypes(df)


def _read_parquet_safe(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    return read_parquet_safe(path, columns=columns)


def _normalize_date_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    out = out.dropna(subset=[column]).drop_duplicates(subset=[column], keep="last").sort_values(column)
    return out


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


def _fill_missing_daily_fields(out: pd.DataFrame) -> pd.DataFrame:
    prev_close = out["close"].shift(1).replace(0, np.nan)
    pct_chg = out["close"].pct_change(fill_method=None) * 100.0
    change = out["close"].diff()
    amplitude = (out["high"] - out["low"]) / prev_close * 100.0

    out["pct_chg"] = out["pct_chg"].where(out["pct_chg"].notna(), pct_chg)
    out["change"] = out["change"].where(out["change"].notna(), change)
    out["amplitude"] = out["amplitude"].where(out["amplitude"].notna(), amplitude)
    return out


def sanitize_daily_history_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy().rename(columns=DAILY_RENAME_MAP)
    if "symbol" not in out.columns:
        out["symbol"] = symbol
    out["symbol"] = out["symbol"].fillna(symbol).astype(str)

    missing = set(DAILY_COLS) - set(out.columns)
    if missing:
        raise ValueError(f"daily data missing required columns: {sorted(missing)}")

    out = _normalize_date_column(out, "date")
    out = _coerce_numeric_columns(out, exclude={"date", "symbol"})

    for col in ["open", "high", "low"]:
        out[col] = out[col].where(out[col].notna(), out["close"])
    for col in ["volume", "amount", "turnover"]:
        out[col] = out[col].fillna(0.0)

    out = _fill_missing_daily_fields(out)

    for col in NON_NEGATIVE_COLS & set(out.columns):
        out.loc[out[col] < 0, col] = np.nan

    out = out.reindex(columns=DAILY_COLS)
    return out


def sanitize_valuation_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().rename(columns=VAL_RENAME_MAP)
    missing = set(VAL_COLS) - set(out.columns)
    if missing:
        raise ValueError(f"valuation data missing required columns: {sorted(missing)}")

    out = _normalize_date_column(out, "date")
    out = _coerce_numeric_columns(out, exclude={"date"})

    invalid_share_pair = (out["total_share"] < 0) | (out["circ_share"] < 0)
    out.loc[invalid_share_pair, ["total_share", "circ_share"]] = np.nan

    invalid_mv_pair = (out["total_mv"] < 0) | (out["circ_mv"] < 0)
    out.loc[invalid_mv_pair, ["total_mv", "circ_mv"]] = np.nan

    invalid_float = (
        out["total_share"].notna()
        & out["circ_share"].notna()
        & (out["circ_share"] > out["total_share"])
    )
    out.loc[invalid_float, ["total_share", "circ_share"]] = np.nan

    invalid_mv = (
        out["total_mv"].notna()
        & out["circ_mv"].notna()
        & (out["circ_mv"] > out["total_mv"])
    )
    out.loc[invalid_mv, ["total_mv", "circ_mv"]] = np.nan

    out = out.reindex(columns=VAL_COLS)
    return out


def merge_daily_and_valuation(daily_df: pd.DataFrame, valuation_df: pd.DataFrame | None, symbol: str) -> pd.DataFrame:
    daily = sanitize_daily_history_frame(daily_df, symbol=symbol)
    if valuation_df is None or valuation_df.empty:
        merged = daily.copy()
        for col in VAL_COLS:
            if col != "date":
                merged[col] = np.nan
    else:
        valuation = sanitize_valuation_frame(valuation_df)
        merged = daily.merge(valuation, on="date", how="left", sort=True)

    if "v_close" in merged.columns:
        merged["close"] = merged["close"].where(merged["close"].notna(), merged["v_close"])
        merged = merged.drop(columns=["v_close"])

    for col in ["open", "high", "low"]:
        merged[col] = merged[col].where(merged[col].notna(), merged["close"])
    for col in ["volume", "amount", "turnover"]:
        merged[col] = merged[col].fillna(0.0)

    merged["symbol"] = symbol
    merged = merged.reindex(columns=PROCESSED_COLS)
    return merged


def load_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return _read_parquet_safe(path)


def infer_latest_date(
    path: Path,
    date_column: str,
    required_columns: list[str] | None = None,
) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        meta = pq.read_metadata(str(path))
        schema_names = set(meta.schema.names)
        if required_columns is not None and not set(required_columns).issubset(schema_names):
            return None
        index = meta.schema.names.index(date_column)
        row_group = meta.row_group(meta.num_row_groups - 1)
        stats = row_group.column(index).statistics
        if stats and stats.max is not None:
            return pd.Timestamp(stats.max).normalize()
    except Exception:
        pass

    frame = _read_parquet_safe(path, columns=[date_column])
    if frame is None or frame.empty or date_column not in frame.columns:
        return None
    return pd.to_datetime(frame[date_column], errors="coerce").max().normalize()


def update_daily_history(symbol: str, target_end_date: pd.Timestamp, adjust: str = "hfq") -> UpdateResult:
    file_path = RAW_DAILY_DIR / f"{symbol}.parquet"
    latest = infer_latest_date(file_path, "date", required_columns=DAILY_COLS)

    if latest is not None and latest >= target_end_date:
        return UpdateResult(status=f"daily up-to-date at {latest.date()}", changed=False, latest=latest)

    existing = load_parquet_if_exists(file_path)
    existing_daily = sanitize_daily_history_frame(existing, symbol) if existing is not None else None
    if latest is None:
        latest = None if existing_daily is None or existing_daily.empty else existing_daily["date"].max()

    start_date = "19900101" if latest is None else (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
    fetched_raw = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=target_end_date.strftime("%Y%m%d"),
        adjust=adjust,
    )
    if fetched_raw is None or fetched_raw.empty:
        if existing_daily is not None:
            return UpdateResult(status=f"daily unchanged at {latest.date()}", changed=False, latest=latest)
        raise RuntimeError("daily fetch returned empty data")

    fetched_daily = sanitize_daily_history_frame(fetched_raw, symbol=symbol)
    combined = fetched_daily if existing_daily is None else pd.concat([existing_daily, fetched_daily], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    save_optimized_parquet(combined.reindex(columns=DAILY_COLS), file_path)
    latest = combined["date"].max()
    return UpdateResult(status=f"daily saved to {latest.date()}", changed=True, latest=latest)


def update_valuation_history(symbol: str, target_end_date: pd.Timestamp) -> UpdateResult:
    file_path = RAW_VAL_DIR / f"{symbol}.parquet"
    latest = infer_latest_date(file_path, "date", required_columns=VAL_COLS)
    if latest is not None and latest >= target_end_date:
        return UpdateResult(status=f"valuation up-to-date at {latest.date()}", changed=False, latest=latest)

    existing = load_parquet_if_exists(file_path)
    existing_val = sanitize_valuation_frame(existing) if existing is not None else None
    if latest is None:
        latest = None if existing_val is None or existing_val.empty else existing_val["date"].max()

    fetched_raw = ak.stock_value_em(symbol=symbol)
    if fetched_raw is None or fetched_raw.empty:
        if existing_val is not None and not existing_val.empty:
            return UpdateResult(
                status=f"valuation unchanged at {latest.date() if latest is not None else 'unknown'}",
                changed=False,
                latest=latest,
            )
        raise RuntimeError("valuation fetch returned empty data")

    fetched_val = sanitize_valuation_frame(fetched_raw)
    combined = fetched_val if existing_val is None else pd.concat([existing_val, fetched_val], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    save_optimized_parquet(combined.reindex(columns=VAL_COLS), file_path)
    latest = combined["date"].max()
    return UpdateResult(status=f"valuation saved to {latest.date()}", changed=True, latest=latest)


def should_rebuild_processed(symbol: str, daily_result: UpdateResult, valuation_result: UpdateResult) -> bool:
    if daily_result.changed or valuation_result.changed:
        return True
    if daily_result.latest is None:
        return True

    processed_path = PROCESSED_DIR / f"{symbol}.parquet"
    processed_latest = infer_latest_date(processed_path, "date", required_columns=PROCESSED_COLS)
    if processed_latest is None:
        return True
    return processed_latest < daily_result.latest


def load_symbol_state(symbol: str) -> SymbolState:
    return SymbolState(
        symbol=symbol,
        daily_latest=infer_latest_date(RAW_DAILY_DIR / f"{symbol}.parquet", "date", required_columns=DAILY_COLS),
        valuation_latest=infer_latest_date(RAW_VAL_DIR / f"{symbol}.parquet", "date", required_columns=VAL_COLS),
        processed_latest=infer_latest_date(PROCESSED_DIR / f"{symbol}.parquet", "date", required_columns=PROCESSED_COLS),
    )


def is_symbol_complete(state: SymbolState, target_end_date: pd.Timestamp) -> bool:
    if state.daily_latest is None or state.daily_latest < target_end_date:
        return False
    if state.valuation_latest is None or state.valuation_latest < target_end_date:
        return False
    if state.processed_latest is None:
        return False
    return state.processed_latest >= state.daily_latest


def precheck_pending_symbols(
    symbols: list[str],
    target_end_date: pd.Timestamp,
    max_workers: int = PRECHECK_WORKERS,
) -> tuple[list[str], list[str]]:
    if not symbols:
        return [], []

    print(f"[*] Scanning {len(symbols)} symbols for completed state with {max_workers} workers...")
    completed_set: set[str] = set()
    pending_set: set[str] = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(load_symbol_state, symbol): symbol for symbol in symbols}
        pbar = tqdm(as_completed(future_map), total=len(symbols), desc="precheck", unit="symbol")
        for future in pbar:
            symbol = future_map[future]
            state = future.result()
            if is_symbol_complete(state, target_end_date):
                completed_set.add(symbol)
            else:
                pending_set.add(symbol)
            pbar.set_postfix({"completed": len(completed_set), "pending": len(pending_set)})

    completed = [symbol for symbol in symbols if symbol in completed_set]
    pending = [symbol for symbol in symbols if symbol in pending_set]
    print(f"[*] Precheck done. completed={len(completed)}, pending={len(pending)}")
    return pending, completed


def rebuild_processed_from_local(symbol: str) -> str:
    daily_path = RAW_DAILY_DIR / f"{symbol}.parquet"
    valuation_path = RAW_VAL_DIR / f"{symbol}.parquet"
    daily = load_parquet_if_exists(daily_path)
    if daily is None or daily.empty:
        raise FileNotFoundError(f"missing daily parquet: {daily_path}")
    valuation = load_parquet_if_exists(valuation_path)
    processed = merge_daily_and_valuation(daily, valuation, symbol=symbol)
    processed_path = PROCESSED_DIR / f"{symbol}.parquet"
    save_optimized_parquet(processed, processed_path)
    return f"processed rows={len(processed)} last_date={processed['date'].max().date()}"


def update_and_rebuild_symbol(symbol: str, target_end_date: pd.Timestamp, adjust: str = "hfq") -> TaskResult:
    try:
        daily_status = update_daily_history(symbol, target_end_date=target_end_date, adjust=adjust)
        valuation_status = update_valuation_history(symbol, target_end_date=target_end_date)
        if should_rebuild_processed(symbol, daily_status, valuation_status):
            processed_status = rebuild_processed_from_local(symbol)
        else:
            processed_status = f"processed up-to-date at {daily_status.latest.date()}"
        return TaskResult(
            symbol=symbol,
            ok=True,
            detail=f"{daily_status.status}; {valuation_status.status}; {processed_status}",
        )
    except Exception as exc:
        return TaskResult(symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}")


def rebuild_symbol(symbol: str) -> TaskResult:
    try:
        detail = rebuild_processed_from_local(symbol)
        return TaskResult(symbol=symbol, ok=True, detail=detail)
    except Exception as exc:
        return TaskResult(symbol=symbol, ok=False, detail=f"{type(exc).__name__}: {exc}")


def collect_symbols(
    symbols: list[str],
    worker,
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
                    f"[!] Aborting after {consecutive_failures} consecutive failures. "
                    "Switch cookie or inspect the failing request path before resuming."
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
    return sorted({item.strip() for item in symbols.split(",") if item.strip()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and fuse A-share daily + valuation parquet data.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all symbols from stock-list cache; fetch live stock list only if cache is missing or --refresh-stock-list is set.",
    )
    parser.add_argument("--symbols", help="Comma-separated symbols. With --rebuild-processed, rebuild locally only.")
    parser.add_argument("--update", action="store_true", help="Fetch/update all locally known symbols from network.")
    parser.add_argument("--rebuild-processed", action="store_true", help="Rebuild processed parquet from local raw files only.")
    parser.add_argument(
        "--refresh-stock-list-only",
        action="store_true",
        help="Refresh or resume the paged stock-list cache only, then exit.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--end-date", default=None, help="Target end date in YYYY-MM-DD. Default: today.")
    parser.add_argument("--adjust", default="hfq", choices=["hfq", "qfq", ""], help="Eastmoney adjustment mode.")
    parser.add_argument(
        "--network-backend",
        default="cookie",
        choices=["cookie", "proxy_patch"],
        help="Network backend for Eastmoney requests.",
    )
    parser.add_argument("--proxy-auth-token", default="", help="Proxy auth token for akshare-proxy-patch backend.")
    parser.add_argument(
        "--refresh-stock-list",
        action="store_true",
        help="With --update, refresh the full live stock list before resolving symbols.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable Rust collector summary.")
    return parser


def _rust_collector_command() -> list[str]:
    env_value = os.environ.get("AI4STOCK_COLLECT_BIN")
    if env_value:
        return shlex.split(env_value)
    return ["cargo", "run", "--bin", "ai4stock-collect", "--"]


def _rust_collector_env() -> dict[str, str]:
    env = os.environ.copy()
    pixi_lib = Path(".pixi/envs/default/lib")
    if pixi_lib.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [str(pixi_lib), *([current] if current else [])]
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def main() -> None:
    if os.environ.get("AI4STOCK_PY_AKSHARE_COLLECTOR_RUNTIME", "").strip() != "1":
        command = _rust_collector_command()
        command.extend(["akshare", *sys.argv[1:]])
        print(f"[run] {shlex.join(command)}", flush=True)
        completed = subprocess.run(command, check=False, env=_rust_collector_env())
        raise SystemExit(int(completed.returncode))

    parser = build_parser()
    args = parser.parse_args()

    symbols = parse_symbols_arg(args.symbols)
    target_end_date = resolve_target_end_date(args.end_date)

    if args.rebuild_processed:
        if not symbols:
            symbols = list_local_symbols()
        if not symbols:
            raise SystemExit("[!] No local symbols found to rebuild.")
        print(f"[*] Rebuilding processed parquet for {len(symbols)} symbols from local raw files...")
        results = collect_symbols(symbols, rebuild_symbol, max_workers=args.workers)
    else:
        if args.network_backend == "cookie":
            patcher = RequestPatcher()
            patcher.load_cookies()
            patcher.patch()
        else:
            install_proxy_patch(auth_token=args.proxy_auth_token)

        if args.refresh_stock_list_only:
            refresh_stock_list_cache()
            print("[+] Stock-list cache refresh completed.")
            return

        if args.all:
            symbols = resolve_all_symbols(refresh_live=args.refresh_stock_list)
        elif args.update and not symbols:
            symbols = resolve_incremental_symbols(refresh_live=args.refresh_stock_list)
        elif not symbols:
            parser.error("Provide one of --all, --update, --rebuild-processed, or --symbols.")

        if not symbols:
            raise SystemExit("[!] No symbols to process.")

        completed_symbols: list[str] = []
        if not args.symbols:
            symbols, completed_symbols = precheck_pending_symbols(symbols, target_end_date=target_end_date)
        if completed_symbols:
            print(f"[*] Skipping {len(completed_symbols)} already-complete symbols.")
        if not symbols:
            print("[+] All symbols are already complete. Nothing to update.")
            return

        print(
            f"[*] Updating {len(symbols)} pending symbols with {args.workers} workers "
            f"(adjust={args.adjust or 'raw'}, end_date={target_end_date.date()})..."
        )
        worker = lambda symbol: update_and_rebuild_symbol(symbol, target_end_date=target_end_date, adjust=args.adjust)
        results = collect_symbols(symbols, worker, max_workers=args.workers)

    success = sum(1 for item in results if item.ok)
    failed = [item for item in results if not item.ok]
    print(f"[+] Done. Success: {success} / {len(results)}")
    if failed:
        print("[!] Failed symbols:")
        for item in failed[:20]:
            print(f"    {item.symbol}: {item.detail}")
        if len(failed) > 20:
            print(f"    ... {len(failed) - 20} more")


if __name__ == "__main__":
    main()
