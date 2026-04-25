from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.parquet_io import load_dataset_frame


def test_load_dataset_frame_returns_empty_for_empty_fragment_list(tmp_path: Path) -> None:
    frame = load_dataset_frame(
        dataset_root=tmp_path,
        selected_columns=["date", "symbol"],
        scan_filter=None,
        fragment_paths=[],
    )

    assert list(frame.columns) == ["date", "symbol"]
    assert frame.empty


def test_load_dataset_frame_reads_selected_symbol_shards(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["A"],
            "value": [1.0],
            "ignored": [2.0],
        }
    ).to_parquet(tmp_path / "A.parquet", index=False)

    frame = load_dataset_frame(
        dataset_root=tmp_path,
        selected_columns=["date", "symbol", "value"],
        scan_filter=None,
        allowed_symbols={"A", "B"},
    )

    assert list(frame.columns) == ["date", "symbol", "value"]
    assert frame["symbol"].tolist() == ["A"]
    assert frame["value"].tolist() == [1.0]
