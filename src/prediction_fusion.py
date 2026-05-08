"""Helpers for fusing primary ranking scores with secondary buyability scores."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.rolling_types import PredictionBundle


def resolve_score_fusion_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    strategy_cfg = cfg.get("strategy", {}) if isinstance(cfg, dict) else {}
    fusion_cfg = strategy_cfg.get("score_fusion", {}) if isinstance(strategy_cfg, dict) else {}
    if fusion_cfg is None:
        fusion_cfg = {}
    if not isinstance(fusion_cfg, dict):
        raise ValueError("strategy.score_fusion must be a mapping when provided")
    return {
        "enabled": bool(fusion_cfg.get("enabled", False)),
        "secondary_predictions_dir": str(fusion_cfg.get("secondary_predictions_dir", "") or "").strip(),
        "mode": str(fusion_cfg.get("mode", "multiply") or "multiply").strip().lower(),
        "primary_transform": str(fusion_cfg.get("primary_transform", "raw") or "raw").strip().lower(),
        "secondary_transform": str(fusion_cfg.get("secondary_transform", "raw") or "raw").strip().lower(),
        "primary_power": float(fusion_cfg.get("primary_power", 1.0)),
        "secondary_power": float(fusion_cfg.get("secondary_power", 1.0)),
        "blend_weight": float(fusion_cfg.get("blend_weight", 0.5)),
        "filter_threshold": float(fusion_cfg.get("filter_threshold", 0.5)),
        "filter_value": float(fusion_cfg.get("filter_value", -1.0)),
    }


def _rank_pct(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    if isinstance(series.index, pd.MultiIndex) and series.index.nlevels >= 2:
        dates = pd.to_datetime(series.index.get_level_values(0))
        return series.groupby(dates, sort=False).rank(pct=True, method="average").astype(float)
    return series.rank(pct=True, method="average").astype(float)


def _transform_scores(series: pd.Series, transform: str) -> pd.Series:
    transform = str(transform or "raw").strip().lower()
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if transform == "raw":
        return numeric
    if transform == "rank_pct":
        return _rank_pct(numeric)
    raise ValueError(f"Unsupported score transform for fusion: {transform}")


def _fusion_summary(
    frame: pd.DataFrame,
    *,
    fusion_cfg: dict[str, Any],
    primary_t: pd.Series,
    secondary_t: pd.Series,
    fused: pd.Series,
) -> dict[str, Any]:
    return {
        "fusion_mode": str(fusion_cfg["mode"]),
        "fusion_primary_transform": str(fusion_cfg["primary_transform"]),
        "fusion_secondary_transform": str(fusion_cfg["secondary_transform"]),
        "fusion_primary_power": float(fusion_cfg["primary_power"]),
        "fusion_secondary_power": float(fusion_cfg["secondary_power"]),
        "fusion_blend_weight": float(fusion_cfg["blend_weight"]),
        "fusion_filter_threshold": float(fusion_cfg["filter_threshold"]),
        "fusion_overlap_rows": int(len(frame)),
        "fusion_overlap_dates": int(pd.Index(frame.index.get_level_values(0)).nunique()) if isinstance(frame.index, pd.MultiIndex) else int(pd.Index(frame.index).nunique()),
        "fusion_primary_mean": float(primary_t.mean()),
        "fusion_secondary_mean": float(secondary_t.mean()),
        "fusion_output_mean": float(fused.mean()),
    }


def fuse_prediction_series(
    primary_predictions: pd.Series,
    secondary_predictions: pd.Series,
    *,
    fusion_cfg: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    common_idx = primary_predictions.index.intersection(secondary_predictions.index)
    if common_idx.empty:
        raise ValueError("No overlapping prediction index between primary and secondary bundles")

    primary = pd.to_numeric(primary_predictions.loc[common_idx], errors="coerce").astype(float)
    secondary = pd.to_numeric(secondary_predictions.loc[common_idx], errors="coerce").astype(float)
    frame = pd.DataFrame({"primary": primary, "secondary": secondary}).dropna()
    if frame.empty:
        raise ValueError("No overlapping finite prediction pairs between primary and secondary bundles")

    mode = str(fusion_cfg["mode"])
    primary_t = _transform_scores(frame["primary"], str(fusion_cfg["primary_transform"]))
    secondary_t = _transform_scores(frame["secondary"], str(fusion_cfg["secondary_transform"]))
    primary_p = primary_t.pow(float(fusion_cfg["primary_power"]))
    secondary_p = secondary_t.pow(float(fusion_cfg["secondary_power"]))

    if mode == "multiply":
        fused = primary_p * secondary_p
    elif mode == "blend":
        weight = float(fusion_cfg["blend_weight"])
        fused = weight * primary_p + (1.0 - weight) * secondary_p
    elif mode == "filter":
        threshold = float(fusion_cfg["filter_threshold"])
        filter_value = float(fusion_cfg["filter_value"])
        fused = primary_p.copy()
        fused.loc[secondary_t < threshold] = filter_value
    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")

    summary = _fusion_summary(
        frame,
        fusion_cfg=fusion_cfg,
        primary_t=primary_t,
        secondary_t=secondary_t,
        fused=fused,
    )
    return fused.rename(primary_predictions.name), summary


def fuse_prediction_bundle(
    primary_bundle: PredictionBundle,
    secondary_bundle: PredictionBundle,
    *,
    fusion_cfg: dict[str, Any],
) -> PredictionBundle:
    fused_predictions, fusion_summary = fuse_prediction_series(
        primary_bundle.final_predictions,
        secondary_bundle.final_predictions,
        fusion_cfg=fusion_cfg,
    )
    metadata = dict(primary_bundle.metadata)
    metadata.update(fusion_summary)
    metadata["fusion_enabled"] = True
    metadata["fusion_secondary_prediction_dir"] = str(fusion_cfg.get("secondary_predictions_dir", ""))

    return PredictionBundle(
        final_predictions=fused_predictions.sort_index(),
        label_series=primary_bundle.label_series,
        backtest_label_series=primary_bundle.backtest_label_series,
        avg_factor_baseline_predictions=primary_bundle.avg_factor_baseline_predictions,
        sign_aligned_factor_baseline_predictions=primary_bundle.sign_aligned_factor_baseline_predictions,
        selected_feature_names=primary_bundle.selected_feature_names,
        metadata=metadata,
        feature_importance_frames=primary_bundle.feature_importance_frames,
        training_summary_records=primary_bundle.training_summary_records,
        rank_avg_factor_baseline_predictions=primary_bundle.rank_avg_factor_baseline_predictions,
        rank_ic_weighted_factor_baseline_predictions=primary_bundle.rank_ic_weighted_factor_baseline_predictions,
    )
