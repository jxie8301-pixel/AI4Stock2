"""Helpers for sanitizing and resolving realized-return label semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LABEL_ABS_CAP = 0.35
DEFAULT_SIGNAL_HORIZON = 10
DEFAULT_LABEL_HORIZONS = (1, 5, 10, 20)


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


def normalize_signal_horizon(value: int | str | None, *, default: int = DEFAULT_SIGNAL_HORIZON) -> int:
    """Return a validated positive signal horizon in trading days."""
    if value is None:
        horizon = int(default)
    else:
        horizon = int(value)
    if horizon <= 0:
        raise ValueError(f"signal horizon must be positive, got {horizon}")
    return horizon


def normalize_label_horizons(
    values: list[int] | tuple[int, ...] | None,
    *,
    default: tuple[int, ...] = DEFAULT_LABEL_HORIZONS,
) -> list[int]:
    """Return validated, sorted, de-duplicated label horizons."""
    raw_values = list(default if values is None else values)
    normalized = sorted({normalize_signal_horizon(value) for value in raw_values})
    if not normalized:
        raise ValueError("at least one label horizon must be configured")
    return normalized


def get_label_column_name(horizon: int) -> str:
    """Return the canonical factor-store column name for a realized-return horizon."""
    return f"label_{normalize_signal_horizon(horizon)}d"


def get_legacy_label_column_name() -> str:
    """Return the legacy label column kept for backward compatibility."""
    return "label"


def get_label_definition(horizon: int) -> str:
    """Return a human-readable definition for an open-to-open label horizon."""
    horizon = normalize_signal_horizon(horizon)
    entry_offset = 1
    exit_offset = horizon + 1
    return f"open_t+{exit_offset} / open_t+{entry_offset} - 1"


def resolve_signal_horizon(cfg: dict | None = None) -> int:
    """Resolve the primary signal horizon from config."""
    cfg = cfg or {}
    label_cfg = cfg.get("label", {})
    value = label_cfg.get("signal_horizon", label_cfg.get("horizon"))
    return normalize_signal_horizon(value, default=DEFAULT_SIGNAL_HORIZON)


def resolve_label_horizons(cfg: dict | None = None) -> list[int]:
    """Resolve all realized-return horizons that should be materialized in factor storage."""
    cfg = cfg or {}
    label_cfg = cfg.get("label", {})
    primary = resolve_signal_horizon(cfg)
    horizons = normalize_label_horizons(label_cfg.get("horizons"), default=DEFAULT_LABEL_HORIZONS)
    horizons = sorted({1, primary, *horizons})
    return horizons


# Backward-compatible aliases for internal call sites that may still use the old name.
DEFAULT_LABEL_HORIZON = DEFAULT_SIGNAL_HORIZON


def normalize_label_horizon(value: int | str | None, *, default: int = DEFAULT_LABEL_HORIZON) -> int:
    return normalize_signal_horizon(value, default=default)


def resolve_label_horizon(cfg: dict | None = None) -> int:
    return resolve_signal_horizon(cfg)
