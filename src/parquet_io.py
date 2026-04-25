"""Shared Parquet dataset scan helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


def load_dataset_frame(
    *,
    dataset_root: Path,
    selected_columns: list[str],
    scan_filter: Any,
    progress_desc: str | None = None,
    fragment_paths: list[str] | None = None,
    allowed_symbols: set[str] | None = None,
) -> pd.DataFrame:
    if fragment_paths is not None:
        if not fragment_paths:
            return pd.DataFrame(columns=selected_columns)
        scan_dataset = ds.dataset(fragment_paths, format="parquet")
        fragment_count = len(fragment_paths)
    elif allowed_symbols is not None:
        symbol_fragment_paths = [
            str(dataset_root / f"{symbol}.parquet")
            for symbol in sorted(allowed_symbols)
            if (dataset_root / f"{symbol}.parquet").exists()
        ]
        if not symbol_fragment_paths:
            return pd.DataFrame(columns=selected_columns)
        scan_dataset = ds.dataset(symbol_fragment_paths, format="parquet")
        fragment_count = len(symbol_fragment_paths)
    else:
        dataset = ds.dataset(dataset_root, format="parquet")
        fragments = list(dataset.get_fragments(filter=scan_filter))
        if not fragments:
            return pd.DataFrame(columns=selected_columns)
        scan_dataset = dataset
        fragment_count = len(fragments)

    if progress_desc:
        print(f"{progress_desc}: reading {fragment_count} shard(s) with pyarrow dataset scan...")

    table = scan_dataset.to_table(
        columns=selected_columns,
        filter=scan_filter,
        use_threads=True,
        batch_readahead=32,
        fragment_readahead=16,
    )
    if table.num_rows == 0:
        return pd.DataFrame(columns=selected_columns)
    return table.to_pandas()
