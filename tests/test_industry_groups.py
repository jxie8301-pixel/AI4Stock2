from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.industry_groups import (
    load_instrument_industry_group_result,
    load_instrument_industry_groups,
)


def test_load_instrument_industry_groups_maps_normalized_symbols(tmp_path: Path) -> None:
    symbol_cache_path = tmp_path / "symbol_cache.parquet"
    pd.DataFrame(
        {
            "local_symbol": ["1", "000002", "ABC", "000003"],
            "industry": ["bank", "tech", "other", ""],
        }
    ).to_parquet(symbol_cache_path, index=False)

    groups = load_instrument_industry_groups(
        {},
        instruments=pd.Index(["000001", "000002", "ABC", "000003"]),
        symbol_cache_path=symbol_cache_path,
    )

    assert groups is not None
    assert groups.loc["000001"] == "bank"
    assert groups.loc["000002"] == "tech"
    assert groups.loc["ABC"] == "other"
    assert pd.isna(groups.loc["000003"])


def test_load_instrument_industry_groups_reports_missing_file(tmp_path: Path) -> None:
    symbol_cache_path = tmp_path / "missing.parquet"

    result = load_instrument_industry_group_result(
        {},
        instruments=pd.Index(["000001"]),
        symbol_cache_path=symbol_cache_path,
    )

    assert result.status == "missing_file"
    assert result.groups is None
    with pytest.raises(ValueError, match="missing_file"):
        load_instrument_industry_groups(
            {},
            instruments=pd.Index(["000001"]),
            required=True,
            symbol_cache_path=symbol_cache_path,
        )


def test_load_instrument_industry_groups_reports_missing_columns(tmp_path: Path) -> None:
    symbol_cache_path = tmp_path / "symbol_cache.parquet"
    pd.DataFrame({"local_symbol": ["000001"], "name": ["A"]}).to_parquet(symbol_cache_path, index=False)

    result = load_instrument_industry_group_result(
        {},
        instruments=pd.Index(["000001"]),
        symbol_cache_path=symbol_cache_path,
    )

    assert result.status == "missing_columns"
    assert result.groups is None
    assert "industry" in str(result.error)
