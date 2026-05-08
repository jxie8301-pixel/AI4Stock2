"""Thin PyO3-facing data collection bridge.

Rust owns collection entrypoints and scheduling. Python remains here as the
provider adapter for Tushare-specific API calls and the existing per-symbol
collector primitives while the local parquet builders are migrated.
"""

from __future__ import annotations

import json
from typing import Any

_AKSHARE_RUNTIME_KEY: tuple[str, str] | None = None


def refresh_tushare_symbol_cache(*, token_env: str = "TUSHARE_TOKEN") -> str:
    from src import collector_tushare

    collector_tushare.configure_tushare(token_env)
    frame = collector_tushare.refresh_symbol_cache()
    return _json_dumps({"ok": True, "rows": int(len(frame))})


def resolve_tushare_latest_trading_date(
    *,
    end_date: str,
    token_env: str = "TUSHARE_TOKEN",
) -> str:
    from src import collector_tushare

    collector_tushare.configure_tushare(token_env)
    target_end_date = collector_tushare.resolve_target_end_date(end_date)
    latest = collector_tushare.resolve_latest_trading_date(target_end_date)
    return _json_dumps({"ok": True, "latest_trading_date": str(latest.date())})


def refresh_tushare_benchmarks(
    *,
    end_date: str,
    token_env: str = "TUSHARE_TOKEN",
) -> str:
    from src import collector_tushare

    collector_tushare.configure_tushare(token_env)
    latest = collector_tushare.resolve_target_end_date(end_date)
    paths = collector_tushare.refresh_tushare_benchmarks(latest)
    return _json_dumps({"ok": True, "paths": [str(path) for path in paths]})


def resolve_tushare_symbols(
    *,
    symbols_csv: str = "",
    update: bool = False,
    refresh_symbols: bool = False,
    token_env: str = "TUSHARE_TOKEN",
) -> str:
    from src import collector_tushare

    collector_tushare.configure_tushare(token_env)
    symbols = collector_tushare.parse_symbols_arg(symbols_csv)
    if update and not symbols:
        symbols = collector_tushare.resolve_incremental_symbols(refresh_live=refresh_symbols)
    return _json_dumps({"ok": True, "symbols": symbols})


def list_tushare_local_symbols() -> str:
    from src import collector_tushare

    return _json_dumps({"ok": True, "symbols": collector_tushare.list_local_symbols()})


def run_tushare_stage(
    *,
    symbol: str,
    stage_name: str,
    end_date: str,
    token_env: str = "TUSHARE_TOKEN",
) -> str:
    from src import collector_tushare

    collector_tushare.configure_tushare(token_env)
    target_end_date = collector_tushare.resolve_target_end_date(end_date)
    try:
        result = _run_tushare_stage_impl(
            collector_tushare,
            symbol=str(symbol),
            stage_name=str(stage_name),
            target_end_date=target_end_date,
        )
        _persist_tushare_stage_side_effects(
            collector_tushare,
            symbol=str(symbol),
            stage_name=str(stage_name),
            target_end_date=target_end_date,
            result=result,
        )
        ok = bool(getattr(result, "ok", True))
        detail = str(
            getattr(
                result,
                "detail",
                result.status if hasattr(result, "status") else result,
            )
        )
        return _json_dumps(
            {
                "symbol": str(symbol),
                "stage_name": str(stage_name),
                "ok": ok,
                "detail": detail,
                "changed": bool(getattr(result, "changed", False)),
            }
        )
    except Exception as exc:
        return _json_dumps(
            {
                "symbol": str(symbol),
                "stage_name": str(stage_name),
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
                "changed": False,
            }
        )


def rebuild_tushare_packed_source(
    *,
    symbols: list[str] | None = None,
    workers: int = 8,
    incremental: bool = True,
) -> str:
    from src import collector_tushare

    metadata = collector_tushare.rebuild_packed_source_from_local(
        symbols,
        workers=int(workers),
        incremental=bool(incremental),
    )
    return _json_dumps({"ok": True, "metadata": metadata})


def prepare_akshare_runtime(
    *,
    network_backend: str = "cookie",
    proxy_auth_token: str = "",
) -> str:
    from src import collector_akshare

    _prepare_akshare_runtime_impl(
        collector_akshare,
        network_backend=str(network_backend),
        proxy_auth_token=str(proxy_auth_token),
    )
    return _json_dumps({"ok": True, "network_backend": str(network_backend)})


def refresh_akshare_stock_list(
    *,
    network_backend: str = "cookie",
    proxy_auth_token: str = "",
) -> str:
    from src import collector_akshare

    _prepare_akshare_runtime_impl(
        collector_akshare,
        network_backend=str(network_backend),
        proxy_auth_token=str(proxy_auth_token),
    )
    symbols = collector_akshare.fetch_stock_list()
    return _json_dumps({"ok": True, "symbols": symbols, "rows": len(symbols)})


def resolve_akshare_symbols(
    *,
    symbols_csv: str = "",
    all: bool = False,
    update: bool = False,
    refresh_stock_list: bool = False,
    network_backend: str = "cookie",
    proxy_auth_token: str = "",
) -> str:
    from src import collector_akshare

    symbols = collector_akshare.parse_symbols_arg(symbols_csv)
    if all:
        _prepare_akshare_runtime_impl(
            collector_akshare,
            network_backend=str(network_backend),
            proxy_auth_token=str(proxy_auth_token),
        )
        symbols = collector_akshare.resolve_all_symbols(refresh_live=bool(refresh_stock_list))
    elif update and not symbols:
        _prepare_akshare_runtime_impl(
            collector_akshare,
            network_backend=str(network_backend),
            proxy_auth_token=str(proxy_auth_token),
        )
        symbols = collector_akshare.resolve_incremental_symbols(refresh_live=bool(refresh_stock_list))
    return _json_dumps({"ok": True, "symbols": symbols})


def list_akshare_local_symbols() -> str:
    from src import collector_akshare

    return _json_dumps({"ok": True, "symbols": collector_akshare.list_local_symbols()})


def run_akshare_raw_update(
    *,
    symbol: str,
    end_date: str,
    adjust: str = "hfq",
    network_backend: str = "cookie",
    proxy_auth_token: str = "",
) -> str:
    from src import collector_akshare

    _prepare_akshare_runtime_impl(
        collector_akshare,
        network_backend=str(network_backend),
        proxy_auth_token=str(proxy_auth_token),
    )
    target_end_date = collector_akshare.resolve_target_end_date(end_date)
    try:
        daily = collector_akshare.update_daily_history(
            str(symbol),
            target_end_date=target_end_date,
            adjust=str(adjust),
        )
        valuation = collector_akshare.update_valuation_history(
            str(symbol),
            target_end_date=target_end_date,
        )
        return _json_dumps(
            {
                "symbol": str(symbol),
                "stage_name": "raw",
                "ok": True,
                "detail": f"{daily.status}; {valuation.status}",
                "changed": bool(daily.changed or valuation.changed),
            }
        )
    except Exception as exc:
        return _json_dumps(
            {
                "symbol": str(symbol),
                "stage_name": "raw",
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
                "changed": False,
            }
        )


def fetch_akshare_index_constituents(
    *,
    index_code: str,
    network_backend: str = "cookie",
    proxy_auth_token: str = "",
) -> str:
    from src import collector_akshare
    import akshare as ak

    _prepare_akshare_runtime_impl(
        collector_akshare,
        network_backend=str(network_backend),
        proxy_auth_token=str(proxy_auth_token),
    )
    df = ak.index_stock_cons_csindex(symbol=str(index_code))
    if df is None or df.empty:
        raise ValueError(f"No constituents returned for index {index_code}")
    records = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): _jsonable_scalar(value) for key, value in row.items()})
    return _json_dumps({"ok": True, "index_code": str(index_code), "records": records})


def _run_tushare_stage_impl(
    collector_tushare: Any,
    *,
    symbol: str,
    stage_name: str,
    target_end_date: Any,
) -> Any:
    if stage_name == "daily":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_DAILY_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_daily,
            target_end_date,
            required_columns=collector_tushare.DAILY_RAW_COLS,
            stage_name="daily",
        )
    if stage_name == "daily_basic":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_DAILY_BASIC_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_daily_basic,
            target_end_date,
            required_columns=collector_tushare.DAILY_BASIC_API_FIELDS,
            stage_name="daily_basic",
        )
    if stage_name == "adj_factor":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_ADJ_FACTOR_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_adj_factor,
            target_end_date,
            required_columns=collector_tushare.ADJ_FACTOR_API_FIELDS,
            stage_name="adj_factor",
        )
    if stage_name == "stk_limit":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_STK_LIMIT_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_stk_limit,
            target_end_date,
            required_columns=collector_tushare.STK_LIMIT_API_FIELDS,
            refetch_on_schema_mismatch=True,
            stage_name="stk_limit",
        )
    if stage_name == "fina_indicator":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_FINA_INDICATOR_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_fina_indicator,
            target_end_date,
            expected_start_date=collector_tushare.pd.Timestamp(collector_tushare.DEFAULT_START_DATE),
            required_columns=collector_tushare.FINA_INDICATOR_API_FIELDS,
            date_column="ann_date",
            allow_empty_success=True,
            stage_name="fina_indicator",
        )
    if stage_name == "dividend":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_DIVIDEND_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_dividend,
            target_end_date,
            expected_start_date=collector_tushare.pd.Timestamp(collector_tushare.DEFAULT_START_DATE),
            required_columns=collector_tushare.DIVIDEND_API_FIELDS,
            date_column="ann_date",
            allow_empty_success=True,
            stage_name="dividend",
        )
    if stage_name == "forecast":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_FORECAST_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_forecast,
            target_end_date,
            expected_start_date=collector_tushare.pd.Timestamp(collector_tushare.DEFAULT_START_DATE),
            required_columns=collector_tushare.FORECAST_API_FIELDS,
            date_column="ann_date",
            allow_empty_success=True,
            stage_name="forecast",
        )
    if stage_name == "express":
        return collector_tushare.update_raw_table(
            symbol,
            collector_tushare.RAW_EXPRESS_DIR / f"{symbol}.parquet",
            collector_tushare._fetch_express,
            target_end_date,
            expected_start_date=collector_tushare.pd.Timestamp(collector_tushare.DEFAULT_START_DATE),
            required_columns=collector_tushare.EXPRESS_API_FIELDS,
            date_column="ann_date",
            allow_empty_success=True,
            stage_name="express",
        )
    if stage_name == "processed":
        return collector_tushare.rebuild_symbol(symbol)
    raise ValueError(f"unsupported Tushare stage: {stage_name}")


def _persist_tushare_stage_side_effects(
    collector_tushare: Any,
    *,
    symbol: str,
    stage_name: str,
    target_end_date: Any,
    result: Any,
) -> None:
    manifest_row = getattr(result, "stage_manifest_row", None)
    if manifest_row is not None:
        collector_tushare.merge_stage_manifest_rows(
            stage_name,
            [manifest_row],
            inspected_symbols=[symbol],
        )
    if stage_name in collector_tushare.EVENT_STAGE_NAMES and hasattr(result, "status"):
        aux_frame = collector_tushare.load_aux_empty_results()
        aux_frame = collector_tushare.merge_aux_empty_records(
            aux_frame,
            updates=[
                {
                    "stage_name": stage_name,
                    "local_symbol": symbol,
                    "checked_end_date": target_end_date,
                    "recorded_at": collector_tushare.pd.Timestamp.now().normalize(),
                }
            ],
            removals=set(),
        )
        collector_tushare.save_aux_empty_results(aux_frame)
    if stage_name in collector_tushare.PROCESSED_DEPENDENCY_STAGE_NAMES and hasattr(result, "status"):
        exhaustion_frame = collector_tushare.load_backfill_exhaustion()
        if bool(getattr(result, "changed", False)):
            exhaustion_frame = collector_tushare.merge_backfill_exhaustion_records(
                exhaustion_frame,
                updates=[],
                removals={(stage_name, symbol)},
            )
            collector_tushare.save_backfill_exhaustion(exhaustion_frame)
        elif bool(getattr(result, "backfill_exhausted", False)):
            exhaustion_frame = collector_tushare.merge_backfill_exhaustion_records(
                exhaustion_frame,
                updates=[
                    {
                        "stage_name": stage_name,
                        "local_symbol": symbol,
                        "expected_start_date": collector_tushare.pd.NaT,
                        "observed_earliest_date": getattr(result, "earliest", collector_tushare.pd.NaT),
                        "recorded_at": collector_tushare.pd.Timestamp.now().normalize(),
                    }
                ],
                removals=set(),
            )
            collector_tushare.save_backfill_exhaustion(exhaustion_frame)


def _prepare_akshare_runtime_impl(
    collector_akshare: Any,
    *,
    network_backend: str = "cookie",
    proxy_auth_token: str = "",
) -> None:
    global _AKSHARE_RUNTIME_KEY
    runtime_key = (str(network_backend), str(proxy_auth_token))
    if _AKSHARE_RUNTIME_KEY == runtime_key:
        return
    if str(network_backend) == "cookie":
        patcher = collector_akshare.RequestPatcher()
        patcher.load_cookies()
        patcher.patch()
    elif str(network_backend) == "proxy_patch":
        collector_akshare.install_proxy_patch(auth_token=str(proxy_auth_token))
    else:
        raise ValueError(f"unsupported AkShare network backend: {network_backend}")
    _AKSHARE_RUNTIME_KEY = runtime_key


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _jsonable_scalar(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return str(value)
    return value
