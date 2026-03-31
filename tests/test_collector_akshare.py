from __future__ import annotations

import json
import pandas as pd
import pytest

from src.collector_akshare import (
    MAX_CONSECUTIVE_FAILURES,
    PRECHECK_WORKERS,
    DAILY_COLS,
    DEFAULT_PROXY_AUTH_IP,
    PROXY_HOOK_DOMAINS,
    PROCESSED_COLS,
    SymbolState,
    UpdateResult,
    VAL_COLS,
    _optimize_numeric_dtypes,
    TaskResult,
    collect_symbols,
    is_symbol_complete,
    install_proxy_patch,
    precheck_pending_symbols,
    merge_incremental_symbol_sets,
    merge_daily_and_valuation,
    load_symbol_state,
    normalize_stock_list_frame,
    resolve_all_symbols,
    resolve_incremental_symbols,
    refresh_stock_list_cache,
    sanitize_daily_history_frame,
    sanitize_valuation_frame,
    should_rebuild_processed,
)


def test_optimize_numeric_dtypes_keeps_large_int64() -> None:
    frame = pd.DataFrame(
        {
            "small_int": pd.Series([1, 2], dtype="int64"),
            "big_int": pd.Series([3_000_000_000, 5_000_000_000], dtype="int64"),
            "float_col": pd.Series([1.25, 2.5], dtype="float64"),
        }
    )

    out = _optimize_numeric_dtypes(frame)

    assert str(out["small_int"].dtype) in {"uint32", "int32"}
    assert str(out["big_int"].dtype) == "int64"
    assert str(out["float_col"].dtype) == "float32"


def test_sanitize_daily_history_frame_backfills_new_fields() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "股票代码": ["000001", "000001"],
            "开盘": [10.0, 11.0],
            "收盘": [10.0, 12.0],
            "最高": [10.5, 12.5],
            "最低": [9.5, 10.8],
            "成交量": [100, 120],
            "成交额": [1000.0, 1440.0],
            "振幅": [None, None],
            "涨跌幅": [None, None],
            "涨跌额": [None, None],
            "换手率": [1.0, 1.2],
        }
    )

    out = sanitize_daily_history_frame(raw, symbol="000001")

    assert list(out.columns) == DAILY_COLS
    assert out.loc[1, "pct_chg"] == pytest.approx(20.0)
    assert out.loc[1, "change"] == pytest.approx(2.0)
    assert out.loc[1, "amplitude"] == pytest.approx(17.0)


def test_sanitize_daily_history_frame_requires_new_schema() -> None:
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02"],
            "股票代码": ["000001"],
            "开盘": [10.0],
            "收盘": [10.0],
            "最高": [10.5],
            "最低": [9.5],
            "成交量": [100],
            "成交额": [1000.0],
            "换手率": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        sanitize_daily_history_frame(raw, symbol="000001")


def test_sanitize_valuation_frame_nulls_invalid_negative_shares() -> None:
    raw = pd.DataFrame(
        {
            "数据日期": ["2024-01-02"],
            "当日收盘价": [10.0],
            "当日涨跌幅": [1.0],
            "总市值": [1_000_000_000.0],
            "流通市值": [800_000_000.0],
            "总股本": [-100],
            "流通股本": [80_000_000],
            "PE(TTM)": [12.0],
            "PE(静)": [11.0],
            "市净率": [1.5],
            "PEG值": [1.2],
            "市现率": [5.0],
            "市销率": [2.0],
        }
    )

    out = sanitize_valuation_frame(raw)

    assert list(out.columns) == VAL_COLS
    assert pd.isna(out.loc[0, "total_share"])
    assert pd.isna(out.loc[0, "circ_share"])


def test_merge_daily_and_valuation_preserves_daily_calendar_only() -> None:
    daily = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "symbol": ["000001", "000001"],
            "open": [10.0, 11.0],
            "high": [10.5, 12.5],
            "low": [9.5, 10.8],
            "close": [10.0, 12.0],
            "volume": [100, 120],
            "amount": [1000.0, 1440.0],
            "amplitude": [10.0, 17.0],
            "pct_chg": [0.0, 20.0],
            "change": [0.0, 2.0],
            "turnover": [1.0, 1.2],
        }
    )
    valuation = pd.DataFrame(
        {
            "数据日期": ["2024-01-01", "2024-01-03"],
            "当日收盘价": [9.0, 12.0],
            "当日涨跌幅": [0.5, 20.0],
            "总市值": [900_000_000.0, 1_200_000_000.0],
            "流通市值": [700_000_000.0, 900_000_000.0],
            "总股本": [90_000_000, 100_000_000],
            "流通股本": [70_000_000, 80_000_000],
            "PE(TTM)": [10.0, 12.0],
            "PE(静)": [9.0, 11.0],
            "市净率": [1.0, 1.2],
            "PEG值": [1.1, 1.3],
            "市现率": [4.0, 4.5],
            "市销率": [2.0, 2.2],
        }
    )

    out = merge_daily_and_valuation(daily, valuation, symbol="000001")

    assert list(out.columns) == PROCESSED_COLS
    assert out["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]
    assert pd.isna(out.loc[0, "total_mv"])
    assert out.loc[1, "total_mv"] == 1_200_000_000.0


def test_normalize_stock_list_frame_filters_and_deduplicates() -> None:
    raw = pd.DataFrame(
        {
            "代码": ["000001", "000001", "688001", "430001", ""],
            "名称": ["平安银行", "平安银行", "华兴源创", "北交所样例", ""],
        }
    )

    out = normalize_stock_list_frame(raw, fetched_at=pd.Timestamp("2026-03-31"))

    assert out["symbol"].tolist() == ["000001", "688001"]
    assert out["name"].tolist() == ["平安银行", "华兴源创"]
    assert out["fetched_at"].dt.strftime("%Y-%m-%d").tolist() == ["2026-03-31", "2026-03-31"]


def test_merge_incremental_symbol_sets_keeps_new_and_existing_symbols() -> None:
    merged = merge_incremental_symbol_sets(
        local_symbols={"000001", "600000"},
        cached_symbols={"000001", "300001"},
        live_symbols={"300001", "688001"},
    )

    assert merged == ["000001", "300001", "600000", "688001"]


def test_resolve_incremental_symbols_skips_live_refresh_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.collector_akshare.list_local_symbols", lambda: ["000001"])
    monkeypatch.setattr(
        "src.collector_akshare.load_symbol_cache",
        lambda: pd.DataFrame(
            {
                "symbol": ["300001"],
                "name": ["样例"],
                "fetched_at": [pd.Timestamp("2026-03-31")],
            }
        ),
    )

    def fail_fetch(*args, **kwargs):
        raise AssertionError("live stock list should not be fetched")

    monkeypatch.setattr("src.collector_akshare.fetch_stock_list_frame", fail_fetch)

    merged = resolve_incremental_symbols()

    assert merged == ["000001", "300001"]


def test_resolve_all_symbols_prefers_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.collector_akshare.load_symbol_cache",
        lambda: pd.DataFrame(
            {
                "symbol": ["000001", "300001"],
                "name": ["平安银行", "特锐德"],
                "fetched_at": [pd.Timestamp("2026-03-31"), pd.Timestamp("2026-03-31")],
            }
        ),
    )

    def fail_fetch(*args, **kwargs):
        raise AssertionError("live stock list should not be fetched when cache exists")

    monkeypatch.setattr("src.collector_akshare.fetch_stock_list", fail_fetch)

    merged = resolve_all_symbols()

    assert merged == ["000001", "300001"]


def test_refresh_stock_list_cache_resumes_missing_pages(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    page_dir = tmp_path / "stock_list_pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "stock_list_manifest.json"
    cache_path = tmp_path / "stock_list.parquet"

    monkeypatch.setattr("src.collector_akshare.STOCK_LIST_PAGE_DIR", page_dir)
    monkeypatch.setattr("src.collector_akshare.STOCK_LIST_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr("src.collector_akshare.SYMBOL_CACHE_PATH", cache_path)

    pd.DataFrame(
        {
            "symbol": ["000001"],
            "name": ["平安银行"],
            "fetched_at": [pd.Timestamp("2026-03-31")],
        }
    ).to_parquet(page_dir / "page_0001.parquet", index=False)
    pd.DataFrame(
        {
            "symbol": ["300001"],
            "name": ["特锐德"],
            "fetched_at": [pd.Timestamp("2026-03-31")],
        }
    ).to_parquet(page_dir / "page_0002.parquet", index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "page_size": 100,
                "total_pages": 3,
                "completed_pages": [1, 2],
                "complete": False,
                "updated_at": "2026-03-31T00:00:00",
            }
        ),
        encoding="utf-8",
    )

    def fake_fetch_stock_list_page(page_number: int, page_size: int = 100):
        assert page_number == 3
        return (
            pd.DataFrame(
                {
                    "symbol": ["688001"],
                    "name": ["华兴源创"],
                    "fetched_at": [pd.Timestamp("2026-03-31")],
                }
            ),
            3,
        )

    monkeypatch.setattr("src.collector_akshare.fetch_stock_list_page", fake_fetch_stock_list_page)

    merged = refresh_stock_list_cache()

    assert merged["symbol"].tolist() == ["000001", "300001", "688001"]
    assert cache_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["completed_pages"] == [1, 2, 3]


def test_collect_symbols_aborts_after_consecutive_failures() -> None:
    def always_fail(symbol: str) -> TaskResult:
        return TaskResult(symbol=symbol, ok=False, detail="network error")

    symbols = [f"{i:06d}" for i in range(MAX_CONSECUTIVE_FAILURES + 2)]

    with pytest.raises(SystemExit, match="Aborting after"):
        collect_symbols(symbols, always_fail, max_workers=1)


def test_should_rebuild_processed_skips_when_everything_is_current(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.collector_akshare.PROCESSED_DIR", tmp_path)

    frame = pd.DataFrame([{col: pd.NA for col in PROCESSED_COLS}])
    frame["date"] = pd.to_datetime(["2026-03-31"])
    frame["symbol"] = ["000001"]
    frame.to_parquet(tmp_path / "000001.parquet", index=False)

    daily = UpdateResult(status="daily up-to-date", changed=False, latest=pd.Timestamp("2026-03-31"))
    valuation = UpdateResult(status="valuation up-to-date", changed=False, latest=pd.Timestamp("2026-03-31"))

    assert should_rebuild_processed("000001", daily, valuation) is False


def test_is_symbol_complete_requires_daily_valuation_and_processed() -> None:
    complete_state = SymbolState(
        symbol="000001",
        daily_latest=pd.Timestamp("2026-03-31"),
        valuation_latest=pd.Timestamp("2026-03-31"),
        processed_latest=pd.Timestamp("2026-03-31"),
    )
    missing_processed = SymbolState(
        symbol="000001",
        daily_latest=pd.Timestamp("2026-03-31"),
        valuation_latest=pd.Timestamp("2026-03-31"),
        processed_latest=None,
    )

    assert is_symbol_complete(complete_state, pd.Timestamp("2026-03-31")) is True
    assert is_symbol_complete(missing_processed, pd.Timestamp("2026-03-31")) is False


def test_precheck_pending_symbols_filters_completed_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    state_map = {
        "000001": SymbolState("000001", pd.Timestamp("2026-03-31"), pd.Timestamp("2026-03-31"), pd.Timestamp("2026-03-31")),
        "000002": SymbolState("000002", pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31"), pd.Timestamp("2026-03-30")),
        "000003": SymbolState("000003", pd.Timestamp("2026-03-31"), pd.Timestamp("2026-03-31"), pd.Timestamp("2026-03-31")),
        "000004": SymbolState("000004", None, None, None),
    }

    monkeypatch.setattr("src.collector_akshare.load_symbol_state", lambda symbol: state_map[symbol])

    pending, completed = precheck_pending_symbols(
        ["000004", "000001", "000002", "000003"],
        target_end_date=pd.Timestamp("2026-03-31"),
        max_workers=PRECHECK_WORKERS,
    )

    assert pending == ["000004", "000002"]
    assert completed == ["000001", "000003"]


def test_proxy_hook_domains_cover_valuation_endpoint() -> None:
    assert "datacenter-web.eastmoney.com" in PROXY_HOOK_DOMAINS


def test_install_proxy_patch_passes_expected_hook_domains(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_install_patch(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("src.collector_akshare.proxy_patch.install_patch", fake_install_patch)

    install_proxy_patch(auth_token="demo")

    assert captured["auth_ip"] == DEFAULT_PROXY_AUTH_IP
    assert captured["auth_token"] == "demo"
    assert captured["hook_domains"] == PROXY_HOOK_DOMAINS
