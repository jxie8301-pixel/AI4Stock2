"""Helpers for sanitizing and resolving realized-return label semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LABEL_ABS_CAP = 0.35
DEFAULT_SIGNAL_HORIZON = 10
DEFAULT_LABEL_HORIZONS = (1, 5, 10, 20)
DEFAULT_TRAIN_LABEL_TRANSFORM_MODE = "raw"
SUPPORTED_TRAIN_LABEL_TRANSFORM_MODES = (
    "raw",
    "profit_tanh",
    "profit_bucket",
    "cross_section_rank",
)


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


def normalize_train_label_transform_mode(value: str | None) -> str:
    mode = str(value or DEFAULT_TRAIN_LABEL_TRANSFORM_MODE).strip().lower()
    if mode not in SUPPORTED_TRAIN_LABEL_TRANSFORM_MODES:
        supported = ", ".join(SUPPORTED_TRAIN_LABEL_TRANSFORM_MODES)
        raise ValueError(f"unsupported training label transform mode: {mode}. Supported: {supported}")
    return mode


def resolve_train_label_transform_cfg(cfg: dict | None = None) -> dict[str, float | str]:
    cfg = cfg or {}
    label_cfg = cfg.get("label", {}) if isinstance(cfg, dict) else {}
    transform_cfg = label_cfg.get("train_transform", {}) if isinstance(label_cfg, dict) else {}
    if transform_cfg is None:
        transform_cfg = {}
    if not isinstance(transform_cfg, dict):
        raise ValueError("label.train_transform must be a mapping when provided")

    mode = normalize_train_label_transform_mode(transform_cfg.get("mode"))
    neutral_band = float(transform_cfg.get("neutral_band", 0.0))
    if neutral_band < 0:
        raise ValueError("label.train_transform.neutral_band must be >= 0")
    default_tail_band = (neutral_band * 3.0) if neutral_band > 0 else 0.03
    tail_band = float(transform_cfg.get("tail_band", default_tail_band))
    if tail_band < neutral_band:
        raise ValueError("label.train_transform.tail_band must be >= label.train_transform.neutral_band")
    scale_multiplier = float(transform_cfg.get("scale_multiplier", 1.0))
    if scale_multiplier <= 0:
        raise ValueError("label.train_transform.scale_multiplier must be > 0")
    min_scale = float(transform_cfg.get("min_scale", 1e-4))
    if min_scale <= 0:
        raise ValueError("label.train_transform.min_scale must be > 0")
    return {
        "mode": mode,
        "neutral_band": neutral_band,
        "tail_band": tail_band,
        "scale_multiplier": scale_multiplier,
        "min_scale": min_scale,
    }


def _compute_group_robust_scale(values: np.ndarray, *, min_scale: float) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float(min_scale)

    median = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - median)) * 1.4826)
    if np.isfinite(mad) and mad >= min_scale:
        return mad

    std = float(np.std(finite_values))
    if np.isfinite(std) and std >= min_scale:
        return std

    abs_median = float(np.median(np.abs(finite_values)))
    if np.isfinite(abs_median) and abs_median >= min_scale:
        return abs_median
    return float(min_scale)


def _bucketize_profit_labels(
    values: np.ndarray,
    *,
    neutral_band: float,
    tail_band: float,
) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.float64)
    out[values <= -tail_band] = -2.0
    out[(values > -tail_band) & (values < -neutral_band)] = -1.0
    out[(values >= neutral_band) & (values < tail_band)] = 1.0
    out[values >= tail_band] = 2.0
    return out


def transform_training_label_series(
    labels: pd.Series,
    dates: np.ndarray | pd.Series,
    cfg: dict | None = None,
) -> pd.Series:
    """Apply an optional training-only label transform while preserving NaNs."""
    transform_cfg = resolve_train_label_transform_cfg(cfg)
    mode = str(transform_cfg["mode"])
    out = labels.astype(float).copy()
    if out.empty or mode == "raw":
        return out

    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(date_series) != len(out):
        raise ValueError("labels and dates must have the same length")

    values = out.to_numpy(dtype=np.float64, copy=True)
    neutral_band = float(transform_cfg["neutral_band"])
    tail_band = float(transform_cfg["tail_band"])
    scale_multiplier = float(transform_cfg["scale_multiplier"])
    min_scale = float(transform_cfg["min_scale"])

    for _, idx in date_series.groupby(date_series, sort=False).groups.items():
        group_idx = np.asarray(idx, dtype=np.int64)
        group_values = values[group_idx]
        finite_mask = np.isfinite(group_values)
        if not finite_mask.any():
            continue
        finite_values = group_values[finite_mask]

        if mode == "cross_section_rank":
            if finite_values.size == 1:
                group_out = np.zeros(1, dtype=np.float64)
            else:
                ranks = (
                    pd.Series(finite_values)
                    .rank(method="average", ascending=True)
                    .to_numpy(dtype=np.float64, copy=False)
                )
                group_out = ((ranks - 1.0) / (finite_values.size - 1.0)) - 0.5
        elif mode == "profit_tanh":
            scale = _compute_group_robust_scale(finite_values, min_scale=min_scale) * scale_multiplier
            adjusted = np.sign(finite_values) * np.maximum(np.abs(finite_values) - neutral_band, 0.0)
            group_out = np.tanh(adjusted / scale)
        elif mode == "profit_bucket":
            group_out = _bucketize_profit_labels(
                finite_values,
                neutral_band=neutral_band,
                tail_band=tail_band,
            )
        else:
            raise ValueError(f"Unsupported training label transform mode: {mode}")

        group_values[finite_mask] = group_out
        values[group_idx] = group_values

    return pd.Series(values, index=out.index, name=out.name, dtype=float)


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
