from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.collector_io import read_parquet_safe


def test_read_parquet_safe_returns_none_for_missing_or_empty_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.parquet"
    empty_path = tmp_path / "empty.parquet"
    empty_path.touch()

    assert read_parquet_safe(missing_path) is None
    assert read_parquet_safe(empty_path) is None


def test_read_parquet_safe_propagates_invalid_parquet_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    path.write_text("not a parquet file", encoding="utf-8")

    with pytest.raises(Exception):
        read_parquet_safe(path)


def test_read_parquet_safe_propagates_missing_column_errors(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2024-01-02"])}).to_parquet(path, index=False)

    with pytest.raises(Exception):
        read_parquet_safe(path, columns=["missing"])
