"""Training-time feature subset resolution and transforms for native pipelines."""

from __future__ import annotations

from typing import Any


def resolve_selected_features(meta: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[int], list[str]]:
    """Resolve selected feature indices from cache metadata and config.

    If ``features.selected_columns`` is omitted or empty, profile-defined columns are
    used when available; otherwise all cached features are used.
    """
    try:
        from src.feature_profiles import resolve_feature_profile
    except ModuleNotFoundError:
        from feature_profiles import resolve_feature_profile  # type: ignore

    feature_names = meta.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Cache metadata missing non-empty feature_names.")

    selected_columns = cfg.get("features", {}).get("selected_columns")
    if not selected_columns:
        selected_columns = resolve_feature_profile(cfg).get("selected_columns")
    if not selected_columns:
        return list(range(len(feature_names))), list(feature_names)

    if not isinstance(selected_columns, list) or not all(isinstance(col, str) for col in selected_columns):
        raise ValueError("features.selected_columns must be a list of feature-name strings.")

    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in selected_columns if name not in index_by_name]
    if missing:
        preview = ", ".join(feature_names[:12])
        raise ValueError(
            "Selected feature columns not found in cache metadata: "
            f"{missing}. Cache starts with: {preview}"
        )

    # Preserve user order but drop duplicates.
    deduped_names = list(dict.fromkeys(selected_columns))
    selected_idx = [index_by_name[name] for name in deduped_names]
    return selected_idx, deduped_names


def compute_finite_feature_mask(X, selected_idx: list[int], num_rows: int) -> Any:
    """Return a row mask that excludes rows with +/-inf on any selected column."""
    import numpy as np

    finite_mask = np.ones(num_rows, dtype=bool)
    for idx in selected_idx:
        finite_mask &= ~np.isinf(X[:, idx])
    return finite_mask


def apply_cross_sectional_rank(frame, dates):
    """Apply per-date percentile rank to each feature column."""
    import pandas as pd

    if frame.empty:
        return frame.copy()

    ranked = frame.copy()
    date_index = pd.to_datetime(pd.Series(dates, index=ranked.index))
    for col in ranked.columns:
        ranked[col] = ranked[col].groupby(date_index, sort=False).rank(pct=True, method="average")
    return ranked


def apply_feature_transforms(frame, dates, cfg: dict[str, Any]):
    """Apply configured training-time transforms to a tabular feature frame."""
    transforms_cfg = cfg.get("features", {}).get("transforms", {})
    if not transforms_cfg:
        return frame

    transformed = frame
    if transforms_cfg.get("cross_sectional_rank", False):
        transformed = apply_cross_sectional_rank(transformed, dates)
    return transformed
