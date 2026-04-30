from __future__ import annotations

import os
from pathlib import Path
import threading

import numpy as np
import pandas as pd


UINT32_MAX = np.iinfo(np.uint32).max
INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max


def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
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


def write_optimized_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    optimized = optimize_numeric_dtypes(df)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        optimized.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def read_parquet_safe(path: Path, columns: list[str] | None = None) -> pd.DataFrame | None:
    """Read an optional parquet file, returning None only when the file is absent or empty."""
    if not path.exists():
        return None
    try:
        if path.stat().st_size == 0:
            return None
    except OSError:
        return None
    return pd.read_parquet(path, columns=columns)
