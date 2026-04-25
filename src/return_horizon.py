"""Helpers for forward realized-return horizons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_forward_compound_return_series(
    daily_returns: pd.Series,
    *,
    horizon: int,
) -> pd.Series:
    """Return the forward compound return over ``t+1 ... t+horizon`` for each date."""
    clean = pd.Series(daily_returns, copy=True).sort_index().astype(float)
    horizon = max(int(horizon), 1)
    if clean.empty:
        return pd.Series(dtype=float)

    values = clean.to_numpy(dtype=np.float64, copy=False)
    out = np.full(len(clean), np.nan, dtype=np.float64)
    if len(values) <= horizon:
        return pd.Series(out, index=clean.index, dtype=float)

    future_windows = np.lib.stride_tricks.sliding_window_view(values[1:], horizon)
    valid_mask = np.isfinite(future_windows).all(axis=1)
    if bool(valid_mask.any()):
        valid_positions = np.flatnonzero(valid_mask)
        out[valid_positions] = np.prod(1.0 + future_windows[valid_mask], axis=1) - 1.0
    return pd.Series(out, index=clean.index, dtype=float)
