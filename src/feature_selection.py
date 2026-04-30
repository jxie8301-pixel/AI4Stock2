"""Training-time feature subset resolution and transforms for native pipelines."""

from __future__ import annotations

from typing import Any


def _resolve_cross_sectional_rank_exclude_columns(cfg: dict[str, Any]) -> set[str]:
    transforms_cfg = cfg.get("features", {}).get("transforms", {}) or {}
    excluded = transforms_cfg.get("cross_sectional_rank_exclude_columns")
    if excluded is None:
        from src.feature_profiles import resolve_feature_profile
        profile = resolve_feature_profile(cfg)
        excluded = profile.get("cross_sectional_rank_exclude_columns")
    if excluded is None:
        return set()
    if not isinstance(excluded, list):
        raise ValueError("cross_sectional_rank_exclude_columns must resolve to a list of strings")
    return {str(item) for item in excluded if str(item).strip()}


def resolve_selected_feature_columns(meta: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (display_columns, source_columns) after applying profile expansion."""
    from src.feature_profiles import resolve_feature_profile
    from src.gen_feature import get_exact_duplicate_feature_source_map

    feature_names = meta.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Cache metadata missing non-empty feature_names.")

    selected_columns_override = cfg.get("features", {}).get("selected_columns")
    if selected_columns_override is not None:
        if not isinstance(selected_columns_override, list) or not all(isinstance(col, str) for col in selected_columns_override):
            raise ValueError("features.selected_columns must be a list of feature-name strings.")
        display_columns = list(selected_columns_override)
        source_columns = list(selected_columns_override)
    else:
        profile = resolve_feature_profile(cfg)
        display_columns = list(profile.get("selected_columns") or [])
        source_columns = list(profile.get("source_columns") or display_columns)

    if not display_columns:
        return list(feature_names), list(feature_names)
    if len(display_columns) != len(source_columns):
        raise ValueError("Resolved feature profile has mismatched display/source column lengths.")

    duplicate_source_map = get_exact_duplicate_feature_source_map()
    source_columns = [duplicate_source_map.get(name, name) for name in source_columns]

    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in source_columns if name not in index_by_name]
    if missing:
        preview = ", ".join(feature_names[:12])
        raise ValueError(
            "Selected feature columns not found in cache metadata: "
            f"{missing}. Cache starts with: {preview}"
        )
    return display_columns, source_columns


def resolve_selected_features(meta: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[int], list[str]]:
    """Resolve selected feature indices from cache metadata and config.

    If ``features.selected_columns`` is omitted or empty, profile-defined columns are
    used when available; otherwise all cached features are used.
    """
    feature_names = meta.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("Cache metadata missing non-empty feature_names.")

    selected_columns, source_columns = resolve_selected_feature_columns(meta, cfg)
    if not selected_columns:
        return list(range(len(feature_names))), list(feature_names)

    index_by_name = {name: idx for idx, name in enumerate(feature_names)}
    selected_idx = [index_by_name[name] for name in source_columns]
    return selected_idx, list(selected_columns)


def materialize_selected_feature_frame(frame, selected_columns: list[str], source_columns: list[str]):
    """Add alias/repeated feature columns required by the resolved profile."""
    import pandas as pd

    if len(selected_columns) != len(source_columns):
        raise ValueError("selected_columns and source_columns must have the same length")
    if frame.empty or not selected_columns:
        return frame.copy()

    expanded = frame.copy()
    for selected_name, source_name in zip(selected_columns, source_columns):
        if source_name not in expanded.columns:
            raise ValueError(f"Missing source feature column: {source_name}")
        if selected_name == source_name:
            continue
        expanded[selected_name] = expanded[source_name].to_numpy(copy=False)
    return expanded


def compute_finite_feature_mask(X, selected_idx: list[int], num_rows: int) -> Any:
    """Return a row mask that excludes rows with +/-inf on any selected column."""
    import numpy as np

    finite_mask = np.ones(num_rows, dtype=bool)
    for idx in selected_idx:
        finite_mask &= ~np.isinf(X[:, idx])
    return finite_mask


def compute_finite_feature_mask_frame(frame, selected_columns: list[str]):
    """Return a row mask that excludes +/-inf on any selected dataframe column."""
    import numpy as np

    if frame.empty or not selected_columns:
        return np.ones(len(frame), dtype=bool)
    values = frame[selected_columns].to_numpy(dtype=np.float32, copy=False)
    return ~np.isinf(values).any(axis=1)


def apply_cross_sectional_rank(frame, dates, *, exclude_columns: set[str] | None = None):
    """Apply per-date percentile rank to each feature column."""
    import pandas as pd

    if frame.empty:
        return frame.copy()

    exclude_columns = set(exclude_columns or set())
    ranked = frame.copy()
    date_index = pd.to_datetime(pd.Series(dates, index=ranked.index))
    rank_columns = [col for col in ranked.columns if col not in exclude_columns]
    if rank_columns:
        ranked[rank_columns] = ranked[rank_columns].groupby(date_index, sort=False).rank(
            pct=True,
            method="average",
        )
    return ranked


def apply_feature_transforms(frame, dates, cfg: dict[str, Any]):
    """Apply configured training-time transforms to a tabular feature frame."""
    transforms_cfg = cfg.get("features", {}).get("transforms", {})
    if not transforms_cfg:
        return frame

    transformed = frame
    if transforms_cfg.get("cross_sectional_rank", False):
        exclude_columns = _resolve_cross_sectional_rank_exclude_columns(cfg)
        transformed = apply_cross_sectional_rank(
            transformed,
            dates,
            exclude_columns=exclude_columns,
        )
    return transformed
