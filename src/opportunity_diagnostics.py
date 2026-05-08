"""Diagnostics for whether model scores align with buyability / opportunity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _align_frame(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    opportunity_labels: pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "prediction": predictions,
            "label": labels,
        }
    )
    if opportunity_labels is not None:
        frame["opportunity_label"] = opportunity_labels
    frame = frame.dropna(subset=["prediction", "label"])
    if "opportunity_label" in frame.columns:
        frame["opportunity_label"] = pd.to_numeric(frame["opportunity_label"], errors="coerce")
    frame = frame.sort_index()
    if isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels >= 2:
        frame["datetime"] = pd.to_datetime(frame.index.get_level_values(0))
        frame["instrument"] = frame.index.get_level_values(1).astype(str)
    else:
        frame["datetime"] = pd.to_datetime(frame.index)
        frame["instrument"] = ""
    frame["positive_label"] = (frame["label"] > 0.0).astype(float)
    return frame


def _has_opportunity_labels(frame: pd.DataFrame) -> bool:
    return "opportunity_label" in frame.columns and frame["opportunity_label"].notna().any()


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    data = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(data) < 2:
        return float("nan")
    left_rank = data["left"].rank(method="average").to_numpy(dtype=float)
    right_rank = data["right"].rank(method="average").to_numpy(dtype=float)
    if np.nanstd(left_rank) == 0.0 or np.nanstd(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _build_topk_opportunity_daily_report_from_frame(
    frame: pd.DataFrame,
    *,
    topk: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    rank_desc = frame.groupby(frame["datetime"], sort=False)["prediction"].rank(method="first", ascending=False)
    selected = frame.loc[rank_desc <= max(int(topk), 1)]
    universe = frame.groupby(frame["datetime"], sort=True).agg(
        universe_count=("label", "size"),
        universe_label_mean=("label", "mean"),
        universe_positive_rate=("positive_label", "mean"),
    )
    selected_agg = selected.groupby(selected["datetime"], sort=True).agg(
        selected_count=("label", "size"),
        selected_prediction_mean=("prediction", "mean"),
        selected_label_mean=("label", "mean"),
        selected_positive_rate=("positive_label", "mean"),
    )

    report = pd.DataFrame(index=universe.index)
    report["universe_count"] = universe["universe_count"].astype(int)
    report["selected_count"] = selected_agg["selected_count"].reindex(report.index).fillna(0).astype(int)
    report["selected_prediction_mean"] = selected_agg["selected_prediction_mean"].reindex(report.index)
    report["selected_label_mean"] = selected_agg["selected_label_mean"].reindex(report.index)
    report["selected_positive_rate"] = selected_agg["selected_positive_rate"].reindex(report.index)
    report["universe_label_mean"] = universe["universe_label_mean"]
    report["universe_positive_rate"] = universe["universe_positive_rate"]
    report["selected_minus_universe_label_mean"] = report["selected_label_mean"] - report["universe_label_mean"]
    report["selected_minus_universe_positive_rate"] = (
        report["selected_positive_rate"] - report["universe_positive_rate"]
    )

    if _has_opportunity_labels(frame):
        selected_opportunity_rate = selected.groupby(selected["datetime"], sort=True)["opportunity_label"].mean()
        universe_opportunity_rate = frame.groupby(frame["datetime"], sort=True)["opportunity_label"].mean()
        report["selected_opportunity_rate"] = selected_opportunity_rate.reindex(report.index)
        report["universe_opportunity_rate"] = universe_opportunity_rate.reindex(report.index)
        report["selected_minus_universe_opportunity_rate"] = (
            report["selected_opportunity_rate"] - report["universe_opportunity_rate"]
        )

    return report.reset_index().sort_values("datetime").reset_index(drop=True)


def build_topk_opportunity_daily_report(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    topk: int,
    opportunity_labels: pd.Series | None = None,
) -> pd.DataFrame:
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    return _build_topk_opportunity_daily_report_from_frame(frame, topk=max(int(topk), 1))


def _build_score_bucket_report_from_frame(
    frame: pd.DataFrame,
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    work = frame.copy()
    bucket_count = max(1, min(int(n_buckets), len(work)))
    rank_desc = work["prediction"].rank(method="first", ascending=False)
    if bucket_count == 1:
        work["bucket"] = np.ones(len(work), dtype=int)
    else:
        work["bucket"] = pd.qcut(rank_desc, q=bucket_count, labels=False, duplicates="drop").astype(int) + 1

    agg_spec: dict[str, tuple[str, str]] = {
        "count": ("label", "size"),
        "prediction_mean": ("prediction", "mean"),
        "label_mean": ("label", "mean"),
        "positive_rate": ("positive_label", "mean"),
    }
    if _has_opportunity_labels(work):
        agg_spec["opportunity_rate"] = ("opportunity_label", "mean")
    return work.groupby("bucket", sort=True).agg(**agg_spec).reset_index()


def build_score_bucket_report(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    opportunity_labels: pd.Series | None = None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    return _build_score_bucket_report_from_frame(frame, n_buckets=n_buckets)


def _build_yearly_score_bucket_report_from_frame(
    frame: pd.DataFrame,
    *,
    n_buckets: int = 10,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["year"] = work["datetime"].dt.year.astype(int)

    rows: list[pd.DataFrame] = []
    for year, group in work.groupby("year", sort=True):
        if group.empty:
            continue
        group = group.copy()
        bucket_count = max(1, min(int(n_buckets), len(group)))
        rank_desc = group["prediction"].rank(method="first", ascending=False)
        if bucket_count == 1:
            group["bucket"] = np.ones(len(group), dtype=int)
        else:
            group["bucket"] = pd.qcut(rank_desc, q=bucket_count, labels=False, duplicates="drop").astype(int) + 1
        agg_spec: dict[str, tuple[str, str]] = {
            "count": ("label", "size"),
            "prediction_mean": ("prediction", "mean"),
            "label_mean": ("label", "mean"),
            "positive_rate": ("positive_label", "mean"),
        }
        if _has_opportunity_labels(group):
            agg_spec["opportunity_rate"] = ("opportunity_label", "mean")
        out = group.groupby("bucket", sort=True).agg(**agg_spec).reset_index()
        out.insert(0, "year", int(year))
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_yearly_score_bucket_report(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    opportunity_labels: pd.Series | None = None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    return _build_yearly_score_bucket_report_from_frame(frame, n_buckets=n_buckets)


def summarize_score_bucket_report(report: pd.DataFrame) -> dict[str, Any]:
    if report.empty:
        return {
            "bucket_count": 0,
            "top_minus_bottom_label_mean": None,
            "top_minus_bottom_positive_rate": None,
            "top_minus_bottom_opportunity_rate": None,
            "return_monotonicity_spearman": None,
            "positive_rate_monotonicity_spearman": None,
            "opportunity_rate_monotonicity_spearman": None,
        }

    sorted_report = report.sort_values("bucket")
    has_opportunity = "opportunity_rate" in sorted_report.columns and sorted_report["opportunity_rate"].notna().any()
    top = sorted_report.iloc[0]
    bottom = sorted_report.iloc[-1]
    bucket_score = -pd.Series(report["bucket"].astype(float))
    summary = {
        "bucket_count": int(len(report)),
        "top_minus_bottom_label_mean": float(top["label_mean"] - bottom["label_mean"]),
        "top_minus_bottom_positive_rate": float(top["positive_rate"] - bottom["positive_rate"]),
        "top_minus_bottom_opportunity_rate": None,
        "return_monotonicity_spearman": _spearman_corr(bucket_score, report["label_mean"]),
        "positive_rate_monotonicity_spearman": _spearman_corr(bucket_score, report["positive_rate"]),
        "opportunity_rate_monotonicity_spearman": None,
    }
    if has_opportunity:
        summary["top_minus_bottom_opportunity_rate"] = float(top["opportunity_rate"] - bottom["opportunity_rate"])
        summary["opportunity_rate_monotonicity_spearman"] = _spearman_corr(bucket_score, report["opportunity_rate"])
    return summary


def summarize_topk_opportunity_daily_report(report: pd.DataFrame) -> dict[str, Any]:
    if report.empty:
        return {
            "date_count": 0,
            "selected_label_mean": None,
            "selected_positive_rate": None,
            "selected_minus_universe_label_mean": None,
            "selected_minus_universe_positive_rate": None,
            "selected_opportunity_rate": None,
            "selected_minus_universe_opportunity_rate": None,
        }
    has_opportunity = "selected_opportunity_rate" in report.columns
    summary = {
        "date_count": int(len(report)),
        "selected_label_mean": float(report["selected_label_mean"].mean()),
        "selected_positive_rate": float(report["selected_positive_rate"].mean()),
        "selected_minus_universe_label_mean": float(report["selected_minus_universe_label_mean"].mean()),
        "selected_minus_universe_positive_rate": float(report["selected_minus_universe_positive_rate"].mean()),
        "selected_opportunity_rate": None,
        "selected_minus_universe_opportunity_rate": None,
    }
    if has_opportunity:
        summary["selected_opportunity_rate"] = float(report["selected_opportunity_rate"].mean())
        summary["selected_minus_universe_opportunity_rate"] = float(
            report["selected_minus_universe_opportunity_rate"].mean()
        )
    return summary


def save_opportunity_diagnostics(
    results_dir: Path,
    *,
    predictions: pd.Series,
    labels: pd.Series,
    topk: int,
    opportunity_labels: pd.Series | None = None,
    opportunity_mode: str,
    opportunity_threshold: float,
    n_buckets: int = 10,
) -> dict[str, str]:
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    daily_report = _build_topk_opportunity_daily_report_from_frame(frame, topk=topk)
    bucket_report = _build_score_bucket_report_from_frame(frame, n_buckets=n_buckets)
    yearly_bucket_report = _build_yearly_score_bucket_report_from_frame(frame, n_buckets=n_buckets)
    summary = {
        "opportunity_mode": opportunity_mode,
        "opportunity_threshold": float(opportunity_threshold),
        "topk_summary": summarize_topk_opportunity_daily_report(daily_report),
        "bucket_summary": summarize_score_bucket_report(bucket_report),
    }

    paths = {
        "buyability_daily_report_path": str(results_dir / "native_buyability_daily_report.csv"),
        "score_bucket_report_path": str(results_dir / "native_score_bucket_report.csv"),
        "score_bucket_yearly_report_path": str(results_dir / "native_score_bucket_yearly_report.csv"),
        "buyability_summary_path": str(results_dir / "native_buyability_summary.json"),
    }
    daily_report.to_csv(paths["buyability_daily_report_path"], index=False)
    bucket_report.to_csv(paths["score_bucket_report_path"], index=False)
    yearly_bucket_report.to_csv(paths["score_bucket_yearly_report_path"], index=False)
    with open(paths["buyability_summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return paths
