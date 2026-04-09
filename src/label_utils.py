"""Helpers for sanitizing and resolving realized-return label semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LABEL_ABS_CAP = 0.35
DEFAULT_SIGNAL_HORIZON = 10
DEFAULT_LABEL_HORIZONS = (1, 5, 10, 20)
DEFAULT_TRAIN_LABEL_TRANSFORM_MODE = "raw"
DEFAULT_OPPORTUNITY_MODE = "positive"
DEFAULT_OPPORTUNITY_NEUTRAL_BAND = 0.0
SUPPORTED_TRAIN_LABEL_TRANSFORM_MODES = (
    "raw",
    "profit_tanh",
    "profit_bucket",
    "cross_section_rank",
    "buyability_binary",
    "buyability_margin_binary",
)
SUPPORTED_OPPORTUNITY_MODES = (
    "positive",
    "threshold",
    "industry_excess",
    "benchmark_excess",
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


def normalize_opportunity_mode(value: str | None) -> str:
    mode = str(value or DEFAULT_OPPORTUNITY_MODE).strip().lower()
    if mode not in SUPPORTED_OPPORTUNITY_MODES:
        supported = ", ".join(SUPPORTED_OPPORTUNITY_MODES)
        raise ValueError(f"unsupported opportunity mode: {mode}. Supported: {supported}")
    return mode


def resolve_opportunity_label_cfg(cfg: dict | None = None) -> dict[str, float | str]:
    cfg = cfg or {}
    label_cfg = cfg.get("label", {}) if isinstance(cfg, dict) else {}
    opportunity_cfg = label_cfg.get("opportunity", {}) if isinstance(label_cfg, dict) else {}
    if opportunity_cfg is None:
        opportunity_cfg = {}
    if not isinstance(opportunity_cfg, dict):
        raise ValueError("label.opportunity must be a mapping when provided")

    mode = normalize_opportunity_mode(opportunity_cfg.get("mode"))
    threshold = float(opportunity_cfg.get("threshold", 0.0))
    neutral_band = float(opportunity_cfg.get("neutral_band", DEFAULT_OPPORTUNITY_NEUTRAL_BAND))
    if neutral_band < 0:
        raise ValueError("label.opportunity.neutral_band must be >= 0")
    return {
        "mode": mode,
        "threshold": threshold,
        "neutral_band": neutral_band,
    }


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


def build_opportunity_edge_series(
    labels: pd.Series,
    *,
    cfg: dict | None = None,
    opportunity_cfg: dict[str, float | str] | None = None,
    instrument_groups: pd.Series | None = None,
    benchmark_forward_returns: pd.Series | None = None,
) -> pd.Series:
    """Return signed edge over the configured opportunity hurdle."""
    out = pd.Series(np.nan, index=labels.index, name="opportunity_edge", dtype=float)
    if labels.empty:
        return out

    opportunity_cfg = opportunity_cfg or resolve_opportunity_label_cfg(cfg)
    mode = str(opportunity_cfg["mode"])
    threshold = float(opportunity_cfg["threshold"])
    clean = pd.to_numeric(labels, errors="coerce").astype(float)

    if mode == "positive":
        return pd.Series(clean.to_numpy(dtype=np.float64, copy=False), index=clean.index, name="opportunity_edge", dtype=float)

    if mode == "threshold":
        values = clean.to_numpy(dtype=np.float64, copy=False)
        result = np.full(len(clean), np.nan, dtype=np.float64)
        finite_mask = np.isfinite(values)
        result[finite_mask] = values[finite_mask] - threshold
        return pd.Series(result, index=clean.index, name="opportunity_edge", dtype=float)

    if mode == "industry_excess":
        if not isinstance(clean.index, pd.MultiIndex) or clean.index.nlevels < 2:
            raise ValueError("industry_excess opportunity mode requires a MultiIndex (datetime, instrument) label series")
        if instrument_groups is None:
            raise ValueError("industry_excess opportunity mode requires instrument_groups")
        frame = clean.rename("label").reset_index()
        frame.columns = ["datetime", "instrument", "label"]
        frame["instrument"] = frame["instrument"].astype(str)
        frame["industry"] = instrument_groups.reindex(frame["instrument"]).to_numpy()
        frame = frame.dropna(subset=["label", "industry"])
        if frame.empty:
            return out
        frame["industry_mean_label"] = frame.groupby(["datetime", "industry"], sort=False)["label"].transform("mean")
        frame["opportunity_edge"] = frame["label"] - frame["industry_mean_label"] - threshold
        rebuilt_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(frame["datetime"]), frame["instrument"].astype(str)],
            names=labels.index.names,
        )
        out.loc[rebuilt_index] = frame["opportunity_edge"].to_numpy(dtype=float, copy=False)
        return out

    if mode == "benchmark_excess":
        if benchmark_forward_returns is None:
            raise ValueError("benchmark_excess opportunity mode requires benchmark_forward_returns")
        dates = pd.to_datetime(clean.index.get_level_values(0) if isinstance(clean.index, pd.MultiIndex) else clean.index)
        benchmark_aligned = benchmark_forward_returns.reindex(pd.Index(dates)).to_numpy(dtype=np.float64, copy=False)
        values = clean.to_numpy(dtype=np.float64, copy=False)
        result = np.full(len(clean), np.nan, dtype=np.float64)
        finite_mask = np.isfinite(values) & np.isfinite(benchmark_aligned)
        result[finite_mask] = values[finite_mask] - benchmark_aligned[finite_mask] - threshold
        return pd.Series(result, index=clean.index, name="opportunity_edge", dtype=float)

    raise ValueError(f"Unsupported opportunity mode: {mode}")


def transform_training_label_series(
    labels: pd.Series,
    dates: np.ndarray | pd.Series,
    cfg: dict | None = None,
    *,
    instrument_groups: pd.Series | None = None,
    benchmark_forward_returns: pd.Series | None = None,
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

    if mode in {"buyability_binary", "buyability_margin_binary"}:
        opportunity_cfg = resolve_opportunity_label_cfg(cfg)
        edge = build_opportunity_edge_series(
            out,
            opportunity_cfg=opportunity_cfg,
            instrument_groups=instrument_groups,
            benchmark_forward_returns=benchmark_forward_returns,
        )
        edge_values = edge.to_numpy(dtype=np.float64, copy=False)
        finite_mask = np.isfinite(edge_values)
        if mode == "buyability_binary":
            values[finite_mask] = (edge_values[finite_mask] > 0.0).astype(np.float64)
            return pd.Series(values, index=out.index, name=out.name, dtype=float)
        margin_neutral_band = float(opportunity_cfg["neutral_band"])
        positive_mask = finite_mask & (edge_values > margin_neutral_band)
        negative_mask = finite_mask & (edge_values < -margin_neutral_band)
        neutral_mask = finite_mask & ~(positive_mask | negative_mask)
        values[positive_mask] = 1.0
        values[negative_mask] = 0.0
        values[neutral_mask] = np.nan
        return pd.Series(values, index=out.index, name=out.name, dtype=float)

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


def build_opportunity_target_series(
    labels: pd.Series,
    *,
    cfg: dict | None = None,
    opportunity_cfg: dict[str, float | str] | None = None,
    instrument_groups: pd.Series | None = None,
    benchmark_forward_returns: pd.Series | None = None,
) -> pd.Series:
    """Return a 0/1 opportunity target series aligned to realized labels."""
    edge = build_opportunity_edge_series(
        labels,
        cfg=cfg,
        opportunity_cfg=opportunity_cfg,
        instrument_groups=instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
    values = edge.to_numpy(dtype=np.float64, copy=False)
    out = pd.Series(np.nan, index=edge.index, name="opportunity_label", dtype=float)
    finite_mask = np.isfinite(values)
    out.loc[finite_mask] = (values[finite_mask] > 0.0).astype(float)
    return out


def compute_opportunity_sample_weights(
    labels: pd.Series,
    dates: np.ndarray | pd.Series,
    *,
    opportunity_cfg: dict[str, float | str] | None = None,
    instrument_groups: pd.Series | None = None,
    benchmark_forward_returns: pd.Series | None = None,
    sample_weight_mode: str = "none",
    sample_weight_power: float = 1.0,
    sample_weight_scale: float | None = None,
    sample_weight_min: float = 0.0,
    sample_weight_date_normalize: bool = False,
) -> pd.Series:
    """Return optional per-row weights emphasizing distance from the buyability threshold."""
    mode = str(sample_weight_mode or "none").strip().lower()
    clean = pd.to_numeric(labels, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=clean.index, name="sample_weight", dtype=float)
    finite_mask = np.isfinite(clean.to_numpy(dtype=np.float64, copy=False))
    if not finite_mask.any():
        return out
    if mode == "none":
        out.loc[finite_mask] = 1.0
        return out
    if mode != "opportunity_distance":
        raise ValueError(f"Unsupported sample weight mode: {mode}")

    opportunity_cfg = opportunity_cfg or {"mode": "positive", "threshold": 0.0, "neutral_band": 0.0}
    edge = build_opportunity_edge_series(
        clean,
        opportunity_cfg=opportunity_cfg,
        instrument_groups=instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
    edge_values = edge.to_numpy(dtype=np.float64, copy=False)
    default_scale = float(opportunity_cfg.get("neutral_band", 0.0))
    if sample_weight_scale is None:
        sample_weight_scale = default_scale if default_scale > 0 else 0.01
    sample_weight_scale = max(float(sample_weight_scale), 1e-6)
    sample_weight_power = max(float(sample_weight_power), 1e-6)
    sample_weight_min = max(float(sample_weight_min), 0.0)

    distance = np.abs(edge_values) / sample_weight_scale
    weights = np.full(len(clean), np.nan, dtype=np.float64)
    edge_finite_mask = finite_mask & np.isfinite(distance)
    weights[edge_finite_mask] = 1.0 + np.power(distance[edge_finite_mask], sample_weight_power)
    if sample_weight_min > 0:
        weights[edge_finite_mask] = np.maximum(weights[edge_finite_mask], sample_weight_min)

    if sample_weight_date_normalize:
        date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        if len(date_series) != len(clean):
            raise ValueError("labels and dates must have the same length")
        weight_series = pd.Series(weights, index=clean.index, dtype=float)
        for _, idx in date_series.groupby(date_series, sort=False).groups.items():
            group_idx = np.asarray(idx, dtype=np.int64)
            group_values = weight_series.iloc[group_idx].to_numpy(dtype=np.float64, copy=False)
            group_finite = np.isfinite(group_values)
            if not group_finite.any():
                continue
            group_mean = float(np.mean(group_values[group_finite]))
            if np.isfinite(group_mean) and group_mean > 0:
                group_values[group_finite] = group_values[group_finite] / group_mean
                weight_series.iloc[group_idx] = group_values
        weights = weight_series.to_numpy(dtype=np.float64, copy=False)

    finite_weight_mask = np.isfinite(weights)
    if finite_weight_mask.any():
        mean_weight = float(np.mean(weights[finite_weight_mask]))
        if np.isfinite(mean_weight) and mean_weight > 0:
            weights[finite_weight_mask] = weights[finite_weight_mask] / mean_weight
    return pd.Series(weights, index=clean.index, name="sample_weight", dtype=float)


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
