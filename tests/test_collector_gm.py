from __future__ import annotations

import pandas as pd
import pytest

from src.collector_gm import (
    GM_PROCESSED_CANONICAL_COLS,
    PRECHECK_WORKERS,
    RAW_BARS_DIR,
    RAW_BASIC_DIR,
    RAW_MKTVALUE_DIR,
    RAW_SYMBOL_DAY_DIR,
    RAW_VALUATION_DIR,
    PROCESSED_DIR,
    SymbolState,
    _chunked,
    build_processed_symbol_frame,
    gm_symbol_to_local,
    is_symbol_complete,
    local_symbol_to_gm,
    normalize_symbol_cache_frame,
    precheck_pending_symbols,
    precheck_stage_updates,
)


def test_symbol_conversion_round_trip_for_a_share_codes() -> None:
    assert local_symbol_to_gm("600000") == "SHSE.600000"
    assert local_symbol_to_gm("000001") == "SZSE.000001"
    assert gm_symbol_to_local("SHSE.600000") == "600000"
    assert gm_symbol_to_local("SZSE.000001") == "000001"


def test_chunked_caps_requests_to_20_fields() -> None:
    fields = [f"f{i}" for i in range(45)]
    chunks = _chunked(fields, chunk_size=20)
    assert [len(chunk) for chunk in chunks] == [20, 20, 5]


def test_normalize_symbol_cache_frame_keeps_a_share_universe_only() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["SHSE.600000", "SZSE.000001", "SHSE.900901"],
            "sec_id": ["600000", "000001", "900901"],
            "sec_name": ["浦发银行", "平安银行", "B股样本"],
            "exchange": ["SHSE", "SZSE", "SHSE"],
            "sec_type1": [1010, 1010, 1010],
            "sec_type2": [101001, 101001, 101002],
        }
    )

    out = normalize_symbol_cache_frame(raw, fetched_at=pd.Timestamp("2026-03-31"))

    assert out["local_symbol"].tolist() == ["000001", "600000"]
    assert out["symbol"].tolist() == ["SZSE.000001", "SHSE.600000"]


def test_is_symbol_complete_requires_all_raw_layers_and_processed() -> None:
    complete = SymbolState(
        symbol="000001",
        bars_latest=pd.Timestamp("2026-03-31"),
        meta_latest=pd.Timestamp("2026-03-31"),
        basic_latest=pd.Timestamp("2026-03-31"),
        mktvalue_latest=pd.Timestamp("2026-03-31"),
        valuation_latest=pd.Timestamp("2026-03-31"),
        processed_latest=pd.Timestamp("2026-03-31"),
    )
    missing = SymbolState(
        symbol="000002",
        bars_latest=pd.Timestamp("2026-03-31"),
        meta_latest=pd.Timestamp("2026-03-31"),
        basic_latest=None,
        mktvalue_latest=pd.Timestamp("2026-03-31"),
        valuation_latest=pd.Timestamp("2026-03-31"),
        processed_latest=pd.Timestamp("2026-03-31"),
    )

    assert is_symbol_complete(complete, pd.Timestamp("2026-03-31")) is True
    assert is_symbol_complete(missing, pd.Timestamp("2026-03-31")) is False


def test_precheck_pending_symbols_filters_completed_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    state_map = {
        "000001": SymbolState("000001", *(pd.Timestamp("2026-03-31"),) * 6),
        "000002": SymbolState(
            "000002",
            pd.Timestamp("2026-03-31"),
            pd.Timestamp("2026-03-31"),
            pd.Timestamp("2026-03-30"),
            pd.Timestamp("2026-03-31"),
            pd.Timestamp("2026-03-31"),
            pd.Timestamp("2026-03-31"),
        ),
    }

    monkeypatch.setattr("src.collector_gm.load_symbol_state", lambda symbol: state_map[symbol])

    pending, completed = precheck_pending_symbols(
        ["000002", "000001"],
        target_end_date=pd.Timestamp("2026-03-31"),
        max_workers=PRECHECK_WORKERS,
    )

    assert pending == ["000002"]
    assert completed == ["000001"]


def test_precheck_stage_updates_marks_ready_and_pending_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    latest_map = {
        "000001": pd.Timestamp("2026-03-31"),
        "000002": pd.Timestamp("2026-03-30"),
        "000003": None,
    }

    def fake_infer_latest_date(path, date_column="trade_date"):
        assert date_column == "trade_date"
        return latest_map[path.stem]

    monkeypatch.setattr("src.collector_gm.infer_latest_date", fake_infer_latest_date)

    plan = precheck_stage_updates(
        ["000003", "000001", "000002"],
        stage_name="bars_raw",
        stage_dir=tmp_path,
        date_column="trade_date",
        target_end_date=pd.Timestamp("2026-03-31"),
        max_workers=1,
    )

    assert plan.pending_symbols == ["000003", "000002"]
    assert list(plan.ready_outputs) == ["000001"]
    assert plan.ready_outputs["000001"].status == "bars_raw up-to-date at 2026-03-31"


def test_build_processed_symbol_frame_maps_gm_raw_layers(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("src.collector_gm.RAW_BARS_DIR", tmp_path / "bars")
    monkeypatch.setattr("src.collector_gm.RAW_SYMBOL_DAY_DIR", tmp_path / "symbol_day")
    monkeypatch.setattr("src.collector_gm.RAW_BASIC_DIR", tmp_path / "basic")
    monkeypatch.setattr("src.collector_gm.RAW_MKTVALUE_DIR", tmp_path / "mkt")
    monkeypatch.setattr("src.collector_gm.RAW_VALUATION_DIR", tmp_path / "val")
    monkeypatch.setattr("src.collector_gm.PROCESSED_DIR", tmp_path / "processed")

    for directory in [
        tmp_path / "bars",
        tmp_path / "symbol_day",
        tmp_path / "basic",
        tmp_path / "mkt",
        tmp_path / "val",
        tmp_path / "processed",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    symbol = "600000"
    gm_symbol = "SHSE.600000"
    pd.DataFrame(
        {
            "symbol": [gm_symbol],
            "trade_date": [pd.Timestamp("2026-03-31")],
            "open": [10.0],
            "high": [10.8],
            "low": [9.8],
            "close": [10.5],
            "volume": [1000],
            "amount": [10500.0],
            "pre_close": [10.0],
            "bob": [pd.Timestamp("2026-03-31 09:30:00")],
            "eob": [pd.Timestamp("2026-03-31 15:00:00")],
        }
    ).to_parquet((tmp_path / "bars" / f"{symbol}.parquet"), index=False)
    pd.DataFrame(
        {
            "symbol": [gm_symbol],
            "trade_date": [pd.Timestamp("2026-03-31")],
            "turn_rate": [1.23],
            "upper_limit": [11.0],
            "lower_limit": [9.0],
            "adj_factor": [6.5],
            "is_suspended": [False],
            "is_st": [False],
        }
    ).to_parquet((tmp_path / "symbol_day" / f"{symbol}.parquet"), index=False)
    pd.DataFrame(
        {
            "symbol": [gm_symbol],
            "trade_date": [pd.Timestamp("2026-03-31")],
            "tclose": [10.5],
            "turnrate": [1.2],
            "ttl_shr": [100_000_000],
            "circ_shr": [80_000_000],
            "ttl_shr_unl": [60_000_000],
            "ttl_shr_ltd": [20_000_000],
            "a_shr_unl": [60_000_000],
            "h_shr_unl": [0],
        }
    ).to_parquet((tmp_path / "basic" / f"{symbol}.parquet"), index=False)
    pd.DataFrame(
        {
            "symbol": [gm_symbol],
            "trade_date": [pd.Timestamp("2026-03-31")],
            "tot_mv": [1_050_000_000.0],
            "tot_mv_csrc": [1_040_000_000.0],
            "a_mv": [850_000_000.0],
            "a_mv_ex_ltd": [630_000_000.0],
        }
    ).to_parquet((tmp_path / "mkt" / f"{symbol}.parquet"), index=False)
    pd.DataFrame(
        {
            "symbol": [gm_symbol],
            "trade_date": [pd.Timestamp("2026-03-31")],
            "pe_ttm": [8.0],
            "pe_lyr": [7.5],
            "pb_mrq": [0.9],
            "pb_lyr": [0.8],
            "peg_lyr": [1.1],
            "pcf_ttm_oper": [5.0],
            "ps_ttm": [1.8],
            "dy_ttm": [3.2],
        }
    ).to_parquet((tmp_path / "val" / f"{symbol}.parquet"), index=False)

    out = build_processed_symbol_frame(symbol)

    assert list(out.columns) == GM_PROCESSED_CANONICAL_COLS
    assert out.loc[0, "symbol"] == symbol
    assert out.loc[0, "gm_symbol"] == gm_symbol
    assert out.loc[0, "turnover"] == pytest.approx(1.23)
    assert out.loc[0, "pct_chg"] == pytest.approx(5.0)
    assert out.loc[0, "change"] == pytest.approx(0.5)
    assert out.loc[0, "amplitude"] == pytest.approx(10.0)
    assert out.loc[0, "circ_mv"] == pytest.approx(630_000_000.0)
    assert out.loc[0, "circ_share"] == pytest.approx(60_000_000.0)
    assert out.loc[0, "pe_static"] == pytest.approx(7.5)
    assert out.loc[0, "pb"] == pytest.approx(0.9)
    assert out.loc[0, "pcf"] == pytest.approx(5.0)
    assert out.loc[0, "ps"] == pytest.approx(1.8)
    assert pd.isna(out.loc[0, "adj_factor_bwd"])
