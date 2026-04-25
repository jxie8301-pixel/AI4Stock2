from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_float_series(series: pd.Series | np.ndarray, *, index: pd.Index | None = None) -> pd.Series:
    if isinstance(series, pd.Series):
        values = pd.to_numeric(series, errors="coerce").astype(float)
        if index is None or values.index.equals(index):
            return values
        if len(values) == len(index):
            return pd.Series(values.to_numpy(copy=False), index=index, dtype=float)
        return values.reindex(index)
    return pd.Series(series, index=index, dtype=float)


def positive_inverse(series: pd.Series | np.ndarray, *, index: pd.Index | None = None) -> pd.Series:
    values = _coerce_float_series(series, index=index)
    return (1.0 / values.where(values > 0)).replace([np.inf, -np.inf], np.nan)


def nonpositive_invalid_flag(series: pd.Series | np.ndarray, *, index: pd.Index | None = None) -> pd.Series:
    values = _coerce_float_series(series, index=index)
    valid = values.notna()
    flags = pd.Series(np.nan, index=values.index, dtype=float)
    flags.loc[valid] = (values.loc[valid] <= 0).astype(float)
    return flags
