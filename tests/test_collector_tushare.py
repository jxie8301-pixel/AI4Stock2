from __future__ import annotations

from pathlib import Path
import time

import pandas as pd
import pytest

from src.collector_tushare import (
    EndpointRateLimiter,
    PRECHECK_WORKERS,
    TS_PROCESSED_CANONICAL_COLS,
    SymbolState,
    build_processed_symbol_frame_from_raw,
    collect_stage,
    is_symbol_complete,
    local_symbol_to_ts,
    normalize_symbol_cache_frame,
    precheck_stage_updates,
    resolve_all_symbols,
    resolve_effective_end_date,
    resolve_symbol_lifecycle_status,
    ts_symbol_to_local,
)


def test_symbol_conversion_round_trip_for_a_share_codes() -> None:
    assert local_symbol_to_ts("600000") == "600000.SH"
    assert local_symbol_to_ts("000001") == "000001.SZ"
    assert ts_symbol_to_local("600000.SH") == "600000"
    assert ts_symbol_to_local("000001.SZ") == "000001"


def test_endpoint_rate_limiter_zero_interval_bypasses_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = EndpointRateLimiter(0.0)

    def fail_sleep(seconds: float) -> None:
        raise AssertionError(f"time.sleep should not be called, got {seconds}")

    def fail_monotonic() -> float:
        raise AssertionError("time.monotonic should not be called for zero interval")

    monkeypatch.setattr("src.collector_tushare.time.sleep", fail_sleep)
    monkeypatch.setattr("src.collector_tushare.time.monotonic", fail_monotonic)

    limiter.wait()


def test_normalize_symbol_cache_frame_handles_missing_optional_columns() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "symbol": ["000001", "600000"],
            "name": ["平安银行", "浦发银行"],
            "market": ["主板", "主板"],
            "list_date": ["19910403", "19991110"],
            "delist_date": [None, None],
            "list_status": ["L", "L"],
        }
    )

    out = normalize_symbol_cache_frame(raw, list_status="L", fetched_at=pd.Timestamp("2026-04-01"))

    assert out["local_symbol"].tolist() == ["000001", "600000"]
    assert out["area"].tolist() == ["", ""]
    assert out["industry"].tolist() == ["", ""]
    assert out["fetched_at"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-01", "2026-04-01"]


def test_resolve_symbol_lifecycle_status_and_effective_end_date_for_delisted() -> None:
    target_end_date = pd.Timestamp("2026-03-31")
    delisted_date = pd.Timestamp("2025-12-15")

    assert (
        resolve_symbol_lifecycle_status(
            target_end_date,
            list_status="D",
            list_date="2000-01-01",
            delist_date=delisted_date,
        )
        == "delisted"
    )
    assert resolve_effective_end_date(
        target_end_date,
        list_status="D",
        list_date="2000-01-01",
        delist_date=delisted_date,
    ) == delisted_date


def test_resolve_all_symbols_preserves_local_history_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.collector_tushare.list_local_symbols", lambda: ["000001", "600000"])
    monkeypatch.setattr(
        "src.collector_tushare.load_symbol_cache",
        lambda: pd.DataFrame(
            {
                "local_symbol": ["000001", "300001"],
                "ts_code": ["000001.SZ", "300001.SZ"],
                "symbol": ["000001", "300001"],
                "name": ["平安银行", "特锐德"],
                "area": ["深圳", "青岛"],
                "industry": ["银行", "电气设备"],
                "market": ["主板", "创业板"],
                "list_date": [pd.Timestamp("1991-01-01"), pd.Timestamp("2009-10-30")],
                "delist_date": [pd.NaT, pd.NaT],
                "list_status": ["L", "L"],
                "fetched_at": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-01")],
            }
        ),
    )

    symbols = resolve_all_symbols()

    assert symbols == ["000001", "300001", "600000"]


def test_is_symbol_complete_requires_all_main_raw_layers_and_processed() -> None:
    complete = SymbolState(
        symbol="000001",
        daily_earliest=pd.Timestamp("2000-01-01"),
        daily_basic_earliest=pd.Timestamp("2000-01-01"),
        adj_factor_earliest=pd.Timestamp("2000-01-01"),
        daily_latest=pd.Timestamp("2026-03-31"),
        daily_basic_latest=pd.Timestamp("2026-03-31"),
        adj_factor_latest=pd.Timestamp("2026-03-31"),
        stk_limit_latest=pd.Timestamp("2026-03-31"),
        processed_latest=pd.Timestamp("2026-03-31"),
    )
    missing = SymbolState(
        symbol="000002",
        daily_earliest=pd.Timestamp("2000-01-01"),
        daily_basic_earliest=pd.Timestamp("2000-01-01"),
        adj_factor_earliest=pd.Timestamp("2000-01-01"),
        daily_latest=pd.Timestamp("2026-03-31"),
        daily_basic_latest=pd.Timestamp("2026-03-31"),
        adj_factor_latest=None,
        stk_limit_latest=pd.Timestamp("2026-03-31"),
        processed_latest=pd.Timestamp("2026-03-31"),
    )

    assert is_symbol_complete(complete, pd.Timestamp("2026-03-31")) is True
    assert is_symbol_complete(missing, pd.Timestamp("2026-03-31")) is False


def test_is_symbol_complete_skips_stk_limit_before_coverage_start() -> None:
    state = SymbolState(
        symbol="000003",
        daily_earliest=pd.Timestamp("2000-01-01"),
        daily_basic_earliest=pd.Timestamp("2000-01-01"),
        adj_factor_earliest=pd.Timestamp("2000-01-01"),
        daily_latest=pd.Timestamp("2002-06-14"),
        daily_basic_latest=pd.Timestamp("2002-06-14"),
        adj_factor_latest=pd.Timestamp("2002-06-14"),
        stk_limit_latest=None,
        processed_latest=pd.Timestamp("2002-06-14"),
    )

    assert is_symbol_complete(state, pd.Timestamp("2002-06-14")) is True


def test_precheck_stage_updates_respects_effective_end_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    latest_map = {
        "000001": pd.Timestamp("2026-03-31"),
        "600001": pd.Timestamp("2025-12-15"),
    }

    def fake_infer_latest_date(path, date_column="trade_date"):
        assert date_column == "trade_date"
        return latest_map[path.stem]

    monkeypatch.setattr("src.collector_tushare.infer_latest_date", fake_infer_latest_date)
    monkeypatch.setattr("src.collector_tushare.infer_earliest_date", lambda path, date_column="trade_date": pd.Timestamp("2000-01-01"))

    plan = precheck_stage_updates(
        ["600001", "000001"],
        stage_name="daily",
        stage_dir=Path("/tmp/unused"),
        target_end_date=pd.Timestamp("2026-03-31"),
        effective_end_dates={
            "600001": pd.Timestamp("2025-12-15"),
            "000001": pd.Timestamp("2026-03-31"),
        },
        max_workers=PRECHECK_WORKERS,
    )

    assert plan.pending_symbols == []
    assert set(plan.ready_outputs) == {"600001", "000001"}


def test_build_processed_symbol_frame_from_raw_adjusts_prices_and_units() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260330", "20260331"],
            "open": [10.0, 12.0],
            "high": [10.5, 12.5],
            "low": [9.8, 11.8],
            "close": [10.0, 12.0],
            "pre_close": [9.5, 10.0],
            "change": [0.5, 2.0],
            "pct_chg": [5.2632, 20.0],
            "volume": [100.0, 120.0],
            "amount": [1.5, 1.8],
        }
    )
    daily_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260330", "20260331"],
            "turnover_rate": [1.0, 1.2],
            "turnover_rate_f": [2.0, 2.2],
            "volume_ratio": [1.1, 1.3],
            "pe": [10.0, 12.0],
            "pe_ttm": [9.0, 11.0],
            "pb": [1.0, 1.2],
            "ps": [2.0, 2.2],
            "ps_ttm": [2.1, 2.3],
            "dv_ratio": [1.0, 1.1],
            "dv_ttm": [1.2, 1.3],
            "total_share": [100.0, 100.0],
            "float_share": [80.0, 80.0],
            "free_share": [60.0, 60.0],
            "total_mv": [1000.0, 1200.0],
            "circ_mv": [800.0, 900.0],
        }
    )
    adj_factor = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260330", "20260331"],
            "adj_factor": [1.0, 2.0],
        }
    )
    stk_limit = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20260330", "20260331"],
            "up_limit": [11.0, 13.2],
            "down_limit": [9.0, 10.8],
        }
    )

    out = build_processed_symbol_frame_from_raw("000001", daily, daily_basic, adj_factor, stk_limit)

    assert list(out.columns) == TS_PROCESSED_CANONICAL_COLS
    assert out["symbol"].tolist() == ["000001", "000001"]
    assert out["ts_code"].tolist() == ["000001.SZ", "000001.SZ"]
    assert out.loc[0, "close"] == pytest.approx(10.0)
    assert out.loc[1, "close"] == pytest.approx(24.0)
    assert out.loc[1, "pre_close"] == pytest.approx(10.0)
    assert out.loc[1, "pct_chg"] == pytest.approx(140.0)
    assert out.loc[0, "amount"] == pytest.approx(1500.0)
    assert out.loc[1, "total_mv"] == pytest.approx(12_000_000.0)
    assert out.loc[1, "circ_share"] == pytest.approx(800_000.0)
    assert out.loc[1, "up_limit"] == pytest.approx(26.4)


def test_collect_stage_skips_remaining_symbols_after_failure_threshold() -> None:
    def flaky_worker(symbol: str) -> str:
        if symbol in {"a", "b"}:
            raise RuntimeError("rate limited")
        time.sleep(0.05)
        return f"ok:{symbol}"

    results = collect_stage(
        ["a", "b", "c", "d"],
        stage_name="daily",
        worker=flaky_worker,
        max_workers=1,
        max_consecutive_failures=2,
        skip_remaining_on_threshold=True,
    )

    detail_by_symbol = {item.symbol: item.detail for item in results}

    assert detail_by_symbol["a"].startswith("RuntimeError:")
    assert detail_by_symbol["b"].startswith("RuntimeError:")
    assert detail_by_symbol["c"] == "daily skipped after 2 consecutive failures"
    assert detail_by_symbol["d"] == "daily skipped after 2 consecutive failures"
