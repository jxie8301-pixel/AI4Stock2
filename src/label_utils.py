"""Helpers for sanitizing open-to-open return labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LABEL_ABS_CAP = 0.35


def sanitize_label_array(
    labels: np.ndarray,
    abs_cap: float = DEFAULT_LABEL_ABS_CAP,
) -> np.ndarray:
    """Return a float32 copy with non-finite / unrealistic labels replaced by NaN."""
    out = np.array(labels, dtype=np.float32, copy=True)
    invalid = ~np.isfinite(out)
    if abs_cap > 0:
        invalid |= np.abs(out) > float(abs_cap)
    out[invalid] = np.nan
    return out


def sanitize_label_series(
    labels: pd.Series,
    abs_cap: float = DEFAULT_LABEL_ABS_CAP,
) -> pd.Series:
    """Return a copy of labels with unrealistic values masked out."""
    out = labels.astype(float).copy()
    invalid = ~np.isfinite(out.to_numpy(dtype=np.float64, copy=False))
    if abs_cap > 0:
        invalid |= np.abs(out.to_numpy(dtype=np.float64, copy=False)) > float(abs_cap)
    out.iloc[np.where(invalid)[0]] = np.nan
    return out
