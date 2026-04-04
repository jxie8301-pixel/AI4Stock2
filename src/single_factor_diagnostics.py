"""Single-factor diagnostics for native factor research."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluate import safe_cross_sectional_corr


DEFAULT_QUANTILE_BINS = 5


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator_value = float(denominator)
    if denominator_value <= 0:
        return 0.0
    return float(numerator) / denominator_value


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(pd.Series(values, dtype=float).mean())


def _safe_std(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.std())


def _safe_icir(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    std = float(values.std())
    if not np.isfinite(std) or np.isclose(std, 0.0):
        return float("nan")
    return float(values.mean() / std)


def _build_date_slices(dates: pd.Series | pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, list[slice]]:
    date_series = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if date_series.empty:
        return pd.DatetimeIndex([]), []

    raw = date_series.to_numpy(dtype="datetime64[ns]")
    boundaries = np.flatnonzero(raw[1:] != raw[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(raw)]
    unique_dates = pd.DatetimeIndex(date_series.iloc[starts])
    slices = [slice(int(start), int(end)) for start, end in zip(starts, ends)]
    return unique_dates, slices


def _rank_to_quantile_bins(values: pd.Series, quantile_bins: int) -> pd.Series:
    rank_pct = values.rank(method="average", pct=True)
    bucket = np.floor(rank_pct.to_numpy(dtype=float, copy=False) * float(quantile_bins) - 1e-12).astype(int)
    bucket = np.clip(bucket, 0, max(int(quantile_bins) - 1, 0))
    return pd.Series(bucket, index=values.index, dtype=int)


def normalize_segments(
    segments: list[tuple[str, str | pd.Timestamp, str | pd.Timestamp]] | None,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if not segments:
        return []
    normalized: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    seen_names: set[str] = set()
    for raw_name, raw_start, raw_end in segments:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Segment name must be non-empty.")
        if name in seen_names:
            raise ValueError(f"Duplicate segment name: {name}")
        start = pd.Timestamp("".join(str(raw_start).split()))
        end = pd.Timestamp("".join(str(raw_end).split()))
        if start > end:
            raise ValueError(f"Segment '{name}' start must be <= end.")
        normalized.append((name, start, end))
        seen_names.add(name)
    return normalized


def compute_feature_diagnostics(
    feature_values: pd.Series,
    labels: pd.Series,
    *,
    quantile_bins: int = DEFAULT_QUANTILE_BINS,
) -> dict[str, Any]:
    """Summarize one factor's cross-sectional predictive quality over time."""
    if not feature_values.index.equals(labels.index):
        raise ValueError("feature_values and labels must share the same index")
    if feature_values.index.nlevels != 2:
        raise ValueError("feature_values must be indexed by [datetime, instrument]")

    quantile_bins = max(int(quantile_bins), 2)
    paired = pd.DataFrame(
        {
            "feature": pd.to_numeric(feature_values, errors="coerce"),
            "label": pd.to_numeric(labels, errors="coerce"),
        },
        index=feature_values.index,
    ).sort_index()

    dates = pd.DatetimeIndex(paired.index.get_level_values(0))
    date_index, date_slices = _build_date_slices(dates)
    total_obs = int(len(paired))
    feature_arr = paired["feature"].to_numpy(dtype=float, copy=False)
    label_arr = paired["label"].to_numpy(dtype=float, copy=False)

    daily_coverage_values: list[float] = []
    daily_ic_values: list[float] = []
    daily_rank_ic_values: list[float] = []
    monotonicity_values: list[float] = []
    spread_values: list[float] = []
    effective_dates: list[pd.Timestamp] = []
    monotonic_dates: list[pd.Timestamp] = []
    valid_observation_count = 0

    for current_date, row_slice in zip(date_index, date_slices):
        feature_slice = feature_arr[row_slice]
        label_slice = label_arr[row_slice]
        valid_mask = np.isfinite(feature_slice) & np.isfinite(label_slice)
        valid_count = int(valid_mask.sum())
        row_count = int(len(feature_slice))
        valid_observation_count += valid_count
        daily_coverage_values.append(_safe_ratio(valid_count, row_count))
        if valid_count < 2:
            continue

        xs = pd.Series(feature_slice[valid_mask], dtype=float)
        ys = pd.Series(label_slice[valid_mask], dtype=float)
        ic = safe_cross_sectional_corr(xs, ys, method="pearson")
        rank_ic = safe_cross_sectional_corr(xs, ys, method="spearman")
        if np.isfinite(ic) and np.isfinite(rank_ic):
            effective_dates.append(pd.Timestamp(current_date))
            daily_ic_values.append(float(ic))
            daily_rank_ic_values.append(float(rank_ic))

        if xs.nunique(dropna=True) < 2 or ys.nunique(dropna=True) < 2:
            continue
        if valid_count < quantile_bins:
            continue

        bucket = _rank_to_quantile_bins(xs, quantile_bins)
        grouped = ys.groupby(bucket).mean().sort_index()
        if len(grouped) < 2:
            continue

        spread = float(grouped.iloc[-1] - grouped.iloc[0])
        monotonicity = safe_cross_sectional_corr(
            pd.Series(grouped.index.to_numpy(dtype=float, copy=False), dtype=float),
            pd.Series(grouped.to_numpy(dtype=float, copy=False), dtype=float),
            method="spearman",
        )
        if np.isfinite(monotonicity):
            monotonic_dates.append(pd.Timestamp(current_date))
            monotonicity_values.append(float(monotonicity))
            spread_values.append(spread)

    daily_ic = pd.Series(daily_ic_values, index=pd.DatetimeIndex(effective_dates), dtype=float).sort_index()
    daily_rank_ic = pd.Series(daily_rank_ic_values, index=pd.DatetimeIndex(effective_dates), dtype=float).sort_index()
    daily_monotonicity = pd.Series(monotonicity_values, index=pd.DatetimeIndex(monotonic_dates), dtype=float).sort_index()
    daily_spread = pd.Series(spread_values, index=pd.DatetimeIndex(monotonic_dates), dtype=float).sort_index()
    monthly_rank_ic = daily_rank_ic.resample("ME").mean().dropna() if not daily_rank_ic.empty else pd.Series(dtype=float)

    rank_ic_mean = float(daily_rank_ic.mean()) if not daily_rank_ic.empty else float("nan")
    direction = 0
    if np.isfinite(rank_ic_mean) and not np.isclose(rank_ic_mean, 0.0):
        direction = 1 if rank_ic_mean > 0 else -1
    directional_hit_rate = float("nan")
    monthly_directional_hit_rate = float("nan")
    if direction != 0 and not daily_rank_ic.empty:
        directional_hit_rate = float((daily_rank_ic * direction > 0).mean())
    if direction != 0 and not monthly_rank_ic.empty:
        monthly_directional_hit_rate = float((monthly_rank_ic * direction > 0).mean())

    return {
        "observation_count": total_obs,
        "valid_observation_count": int(valid_observation_count),
        "coverage_pct": _safe_ratio(valid_observation_count, total_obs),
        "avg_daily_coverage_pct": _safe_mean(daily_coverage_values),
        "date_count": int(len(date_index)),
        "effective_date_count": int(len(daily_ic)),
        "monotonic_date_count": int(len(daily_monotonicity)),
        "ic_mean": float(daily_ic.mean()) if not daily_ic.empty else float("nan"),
        "ic_std": _safe_std(daily_ic),
        "ic_ir": _safe_icir(daily_ic),
        "ic_positive_rate": float((daily_ic > 0).mean()) if not daily_ic.empty else float("nan"),
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": _safe_std(daily_rank_ic),
        "rank_ic_ir": _safe_icir(daily_rank_ic),
        "rank_ic_positive_rate": float((daily_rank_ic > 0).mean()) if not daily_rank_ic.empty else float("nan"),
        "rank_ic_directional_hit_rate": directional_hit_rate,
        "rank_ic_abs_mean": abs(rank_ic_mean) if np.isfinite(rank_ic_mean) else float("nan"),
        "monthly_rank_ic_mean": float(monthly_rank_ic.mean()) if not monthly_rank_ic.empty else float("nan"),
        "monthly_rank_ic_positive_rate": float((monthly_rank_ic > 0).mean()) if not monthly_rank_ic.empty else float("nan"),
        "monthly_rank_ic_directional_hit_rate": monthly_directional_hit_rate,
        "monthly_rank_ic_months": int(len(monthly_rank_ic)),
        "monotonicity_mean": float(daily_monotonicity.mean()) if not daily_monotonicity.empty else float("nan"),
        "monotonicity_positive_rate": (
            float((daily_monotonicity > 0).mean()) if not daily_monotonicity.empty else float("nan")
        ),
        "top_bottom_spread_mean": float(daily_spread.mean()) if not daily_spread.empty else float("nan"),
        "top_bottom_spread_positive_rate": float((daily_spread > 0).mean()) if not daily_spread.empty else float("nan"),
        "suggested_direction": int(direction),
    }


def build_single_factor_diagnostics(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    label_column: str = "label",
    quantile_bins: int = DEFAULT_QUANTILE_BINS,
) -> pd.DataFrame:
    """Compute summary diagnostics for a list of features."""
    if "date" not in frame.columns:
        raise ValueError("frame must contain a 'date' column")
    if label_column not in frame.columns:
        raise ValueError(f"frame must contain label column: {label_column}")

    required_columns = ["date", "symbol", label_column, *feature_names]
    subset = frame[required_columns].copy()
    subset["date"] = pd.to_datetime(subset["date"])
    subset["symbol"] = subset["symbol"].astype(str)
    subset = subset.sort_values(["date", "symbol"]).reset_index(drop=True)
    multi_index = pd.MultiIndex.from_arrays(
        [subset["date"], subset["symbol"]],
        names=["datetime", "instrument"],
    )
    labels = pd.Series(pd.to_numeric(subset[label_column], errors="coerce").to_numpy(), index=multi_index, dtype=float)

    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        feature_series = pd.Series(
            pd.to_numeric(subset[feature_name], errors="coerce").to_numpy(),
            index=multi_index,
            dtype=float,
        )
        metrics = compute_feature_diagnostics(
            feature_series,
            labels,
            quantile_bins=quantile_bins,
        )
        rows.append({"feature": feature_name, **metrics})

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["rank_ic_ir_abs"] = summary["rank_ic_ir"].abs()
    summary["ic_ir_abs"] = summary["ic_ir"].abs()
    summary = summary.sort_values(
        ["rank_ic_abs_mean", "rank_ic_ir_abs", "coverage_pct", "feature"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return summary


def build_segmented_single_factor_diagnostics(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
    segments: list[tuple[str, str | pd.Timestamp, str | pd.Timestamp]] | None,
    label_column: str = "label",
    quantile_bins: int = DEFAULT_QUANTILE_BINS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    normalized_segments = normalize_segments(segments)
    if not normalized_segments:
        return pd.DataFrame(), {}
    if "date" not in frame.columns:
        raise ValueError("frame must contain a 'date' column")

    base = frame.copy()
    base["date"] = pd.to_datetime(base["date"])
    segment_frames: dict[str, pd.DataFrame] = {}
    comparison_rows: list[pd.DataFrame] = []

    for name, start, end in normalized_segments:
        segment_frame = base[(base["date"] >= start) & (base["date"] <= end)].copy()
        if segment_frame.empty:
            continue
        summary = build_single_factor_diagnostics(
            segment_frame,
            feature_names=feature_names,
            label_column=label_column,
            quantile_bins=quantile_bins,
        )
        segment_frames[name] = summary
        rename_map = {
            col: f"{col}__{name}"
            for col in summary.columns
            if col != "feature"
        }
        comparison_rows.append(summary.rename(columns=rename_map))

    if not comparison_rows:
        return pd.DataFrame(), {}

    merged = comparison_rows[0]
    for extra in comparison_rows[1:]:
        merged = merged.merge(extra, on="feature", how="outer")

    rank_ic_cols = [col for col in merged.columns if col.startswith("rank_ic_mean__")]
    direction_cols = [col for col in merged.columns if col.startswith("suggested_direction__")]
    monthly_hit_cols = [col for col in merged.columns if col.startswith("monthly_rank_ic_directional_hit_rate__")]

    if rank_ic_cols:
        rank_ic_frame = merged[rank_ic_cols]
        merged["segment_rank_ic_abs_max"] = rank_ic_frame.abs().max(axis=1, skipna=True)
        merged["segment_rank_ic_abs_min"] = rank_ic_frame.abs().min(axis=1, skipna=True)
        merged["segment_rank_ic_mean_range"] = rank_ic_frame.max(axis=1, skipna=True) - rank_ic_frame.min(axis=1, skipna=True)
        best_idx = rank_ic_frame.abs().idxmax(axis=1, skipna=True)
        worst_idx = rank_ic_frame.abs().idxmin(axis=1, skipna=True)
        merged["best_segment_by_abs_rank_ic"] = best_idx.str.replace("rank_ic_mean__", "", regex=False)
        merged["worst_segment_by_abs_rank_ic"] = worst_idx.str.replace("rank_ic_mean__", "", regex=False)

    if direction_cols:
        direction_frame = merged[direction_cols].fillna(0.0)
        positive_count = (direction_frame > 0).sum(axis=1)
        negative_count = (direction_frame < 0).sum(axis=1)
        nonzero_count = (direction_frame != 0).sum(axis=1)
        merged["positive_direction_segments"] = positive_count.astype(int)
        merged["negative_direction_segments"] = negative_count.astype(int)
        merged["nonzero_direction_segments"] = nonzero_count.astype(int)
        merged["direction_flip"] = (positive_count > 0) & (negative_count > 0)

    if monthly_hit_cols:
        monthly_hit_frame = merged[monthly_hit_cols]
        merged["segment_monthly_directional_hit_mean"] = monthly_hit_frame.mean(axis=1, skipna=True)
        merged["segment_monthly_directional_hit_min"] = monthly_hit_frame.min(axis=1, skipna=True)

    merged = merged.sort_values(
        ["segment_rank_ic_abs_max", "segment_monthly_directional_hit_mean", "feature"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return merged, segment_frames


def _render_markdown_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No factors available._\n"
    preview = frame.head(max_rows).copy()
    display_cols = [
        "feature",
        "rank_ic_mean",
        "rank_ic_ir",
        "coverage_pct",
        "monotonicity_mean",
        "monthly_rank_ic_directional_hit_rate",
    ]
    preview = preview[[col for col in display_cols if col in preview.columns]]
    formatted = preview.copy()
    for col in [c for c in formatted.columns if c != "feature"]:
        formatted[col] = formatted[col].map(
            lambda value: f"{float(value):.4f}" if pd.notna(value) and np.isfinite(float(value)) else "nan"
        )
    return formatted.to_markdown(index=False) + "\n"


def _render_segment_comparison_table(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No segment comparison available._\n"
    preview = frame.head(max_rows).copy()
    display_cols = [
        "feature",
        "direction_flip",
        "best_segment_by_abs_rank_ic",
        "worst_segment_by_abs_rank_ic",
        "segment_rank_ic_abs_max",
        "segment_rank_ic_mean_range",
        "segment_monthly_directional_hit_mean",
    ]
    preview = preview[[col for col in display_cols if col in preview.columns]]
    formatted = preview.copy()
    for col in formatted.columns:
        if col == "feature":
            continue
        if col == "direction_flip":
            formatted[col] = formatted[col].map(lambda value: "yes" if bool(value) else "no")
            continue
        formatted[col] = formatted[col].map(
            lambda value: (
                f"{float(value):.4f}"
                if pd.notna(value)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(float(value))
                else str(value)
            )
        )
    return formatted.to_markdown(index=False) + "\n"


def save_single_factor_diagnostics(
    summary: pd.DataFrame,
    *,
    output_dir: str | Path,
    config_snapshot: dict[str, Any],
    metadata: dict[str, Any],
    top_n: int = 50,
    segment_comparison: pd.DataFrame | None = None,
    segment_summaries: dict[str, pd.DataFrame] | None = None,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_path = output_path / "single_factor_summary.csv"
    top_abs_rankic_path = output_path / "single_factor_top_abs_rankic.csv"
    top_rankic_path = output_path / "single_factor_top_rankic.csv"
    top_icir_path = output_path / "single_factor_top_rankic_ir.csv"
    summary_md_path = output_path / "README.md"
    manifest_path = output_path / "manifest.json"
    config_path = output_path / "config_snapshot.yaml"
    segment_comparison_path = output_path / "single_factor_segment_comparison.csv"
    segment_dir = output_path / "segments"

    summary.to_csv(summary_path, index=False)
    summary.sort_values(["rank_ic_abs_mean", "coverage_pct"], ascending=[False, False]).head(top_n).to_csv(
        top_abs_rankic_path,
        index=False,
    )
    summary.sort_values(["rank_ic_mean", "coverage_pct"], ascending=[False, False]).head(top_n).to_csv(
        top_rankic_path,
        index=False,
    )
    summary.sort_values(["rank_ic_ir", "coverage_pct"], ascending=[False, False]).head(top_n).to_csv(
        top_icir_path,
        index=False,
    )

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_snapshot, f, allow_unicode=True, sort_keys=False)

    artifacts: dict[str, str] = {
        "summary_csv": str(summary_path),
        "top_abs_rankic_csv": str(top_abs_rankic_path),
        "top_rankic_csv": str(top_rankic_path),
        "top_rankic_ir_csv": str(top_icir_path),
        "readme_path": str(summary_md_path),
        "manifest_path": str(manifest_path),
        "config_snapshot_path": str(config_path),
    }

    segment_section: list[str] = []
    if segment_comparison is not None and not segment_comparison.empty:
        segment_comparison.to_csv(segment_comparison_path, index=False)
        artifacts["segment_comparison_csv"] = str(segment_comparison_path)
        segment_section.extend(
            [
                "## Segment Comparison",
                "",
                _render_segment_comparison_table(
                    segment_comparison.sort_values(
                        ["segment_rank_ic_abs_max", "segment_monthly_directional_hit_mean"],
                        ascending=[False, False],
                        na_position="last",
                    ),
                    max_rows=min(int(top_n), 20),
                ),
            ]
        )
    if segment_summaries:
        segment_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in segment_summaries.items():
            segment_path = segment_dir / f"{name}.csv"
            frame.to_csv(segment_path, index=False)
            artifacts[f"segment_{name}_csv"] = str(segment_path)

    summary_lines = [
        "# Single-Factor Diagnostics",
        "",
        "## Metadata",
        "",
        *[f"- {key}: `{value}`" for key, value in metadata.items()],
        "",
        "## Top Factors By Absolute RankIC",
        "",
        _render_markdown_table(
            summary.sort_values(["rank_ic_abs_mean", "coverage_pct"], ascending=[False, False]),
            max_rows=min(int(top_n), 20),
        ),
        "## Top Factors By RankICIR",
        "",
        _render_markdown_table(
            summary.sort_values(["rank_ic_ir", "coverage_pct"], ascending=[False, False]),
            max_rows=min(int(top_n), 20),
        ),
        *segment_section,
    ]
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).strip() + "\n")

    manifest = {
        "metadata": metadata,
        "artifacts": artifacts,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return artifacts
