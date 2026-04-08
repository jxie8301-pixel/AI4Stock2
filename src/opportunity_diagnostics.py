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


def build_topk_opportunity_daily_report(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    topk: int,
    opportunity_labels: pd.Series | None = None,
) -> pd.DataFrame:
    topk = max(int(topk), 1)
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby(frame["datetime"], sort=True):
        ranked = group.sort_values("prediction", ascending=False, kind="stable")
        selected = ranked.head(topk)
        row = {
            "datetime": pd.Timestamp(date),
            "universe_count": int(len(group)),
            "selected_count": int(len(selected)),
            "selected_prediction_mean": float(selected["prediction"].mean()),
            "selected_label_mean": float(selected["label"].mean()),
            "selected_positive_rate": float(selected["positive_label"].mean()),
            "universe_label_mean": float(group["label"].mean()),
            "universe_positive_rate": float(group["positive_label"].mean()),
            "selected_minus_universe_label_mean": float(selected["label"].mean() - group["label"].mean()),
            "selected_minus_universe_positive_rate": float(selected["positive_label"].mean() - group["positive_label"].mean()),
        }
        if "opportunity_label" in group.columns and group["opportunity_label"].notna().any():
            row["selected_opportunity_rate"] = float(selected["opportunity_label"].mean())
            row["universe_opportunity_rate"] = float(group["opportunity_label"].mean())
            row["selected_minus_universe_opportunity_rate"] = float(
                selected["opportunity_label"].mean() - group["opportunity_label"].mean()
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def build_score_bucket_report(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    opportunity_labels: pd.Series | None = None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    if frame.empty:
        return pd.DataFrame()

    bucket_count = max(1, min(int(n_buckets), len(frame)))
    rank_desc = frame["prediction"].rank(method="first", ascending=False)
    if bucket_count == 1:
        bucket_ids = pd.Series(np.ones(len(frame), dtype=int), index=frame.index)
    else:
        bucket_ids = pd.qcut(rank_desc, q=bucket_count, labels=False, duplicates="drop") + 1
        bucket_ids = pd.Series(bucket_ids, index=frame.index, dtype=int)
    frame["bucket"] = bucket_ids

    rows: list[dict[str, Any]] = []
    for bucket, group in frame.groupby("bucket", sort=True):
        row = {
            "bucket": int(bucket),
            "count": int(len(group)),
            "prediction_mean": float(group["prediction"].mean()),
            "label_mean": float(group["label"].mean()),
            "positive_rate": float(group["positive_label"].mean()),
        }
        if "opportunity_label" in group.columns and group["opportunity_label"].notna().any():
            row["opportunity_rate"] = float(group["opportunity_label"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def build_yearly_score_bucket_report(
    predictions: pd.Series,
    labels: pd.Series,
    *,
    opportunity_labels: pd.Series | None = None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    frame = _align_frame(predictions, labels, opportunity_labels=opportunity_labels)
    if frame.empty:
        return pd.DataFrame()
    frame["year"] = frame["datetime"].dt.year.astype(int)

    rows: list[pd.DataFrame] = []
    for year, group in frame.groupby(frame["year"], sort=True):
        if group.empty:
            continue
        bucket_count = max(1, min(int(n_buckets), len(group)))
        rank_desc = group["prediction"].rank(method="first", ascending=False)
        if bucket_count == 1:
            bucket_ids = pd.Series(np.ones(len(group), dtype=int), index=group.index)
        else:
            bucket_ids = pd.qcut(rank_desc, q=bucket_count, labels=False, duplicates="drop") + 1
            bucket_ids = pd.Series(bucket_ids, index=group.index, dtype=int)
        group = group.copy()
        group["bucket"] = bucket_ids
        out = (
            group.groupby("bucket", sort=True)
            .agg(
                count=("label", "size"),
                prediction_mean=("prediction", "mean"),
                label_mean=("label", "mean"),
                positive_rate=("positive_label", "mean"),
            )
            .reset_index()
        )
        if "opportunity_label" in group.columns and group["opportunity_label"].notna().any():
            opp = group.groupby("bucket", sort=True)["opportunity_label"].mean().reset_index(name="opportunity_rate")
            out = out.merge(opp, on="bucket", how="left")
        out.insert(0, "year", int(year))
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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

    top = report.sort_values("bucket").iloc[0]
    bottom = report.sort_values("bucket").iloc[-1]
    bucket_score = -pd.Series(report["bucket"].astype(float))
    summary = {
        "bucket_count": int(len(report)),
        "top_minus_bottom_label_mean": float(top["label_mean"] - bottom["label_mean"]),
        "top_minus_bottom_positive_rate": float(top["positive_rate"] - bottom["positive_rate"]),
        "top_minus_bottom_opportunity_rate": None,
        "return_monotonicity_spearman": float(bucket_score.corr(report["label_mean"], method="spearman")),
        "positive_rate_monotonicity_spearman": float(bucket_score.corr(report["positive_rate"], method="spearman")),
        "opportunity_rate_monotonicity_spearman": None,
    }
    if "opportunity_rate" in report.columns and report["opportunity_rate"].notna().any():
        summary["top_minus_bottom_opportunity_rate"] = float(top["opportunity_rate"] - bottom["opportunity_rate"])
        summary["opportunity_rate_monotonicity_spearman"] = float(
            bucket_score.corr(report["opportunity_rate"], method="spearman")
        )
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
    summary = {
        "date_count": int(len(report)),
        "selected_label_mean": float(report["selected_label_mean"].mean()),
        "selected_positive_rate": float(report["selected_positive_rate"].mean()),
        "selected_minus_universe_label_mean": float(report["selected_minus_universe_label_mean"].mean()),
        "selected_minus_universe_positive_rate": float(report["selected_minus_universe_positive_rate"].mean()),
        "selected_opportunity_rate": None,
        "selected_minus_universe_opportunity_rate": None,
    }
    if "selected_opportunity_rate" in report.columns:
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
    daily_report = build_topk_opportunity_daily_report(
        predictions,
        labels,
        topk=topk,
        opportunity_labels=opportunity_labels,
    )
    bucket_report = build_score_bucket_report(
        predictions,
        labels,
        opportunity_labels=opportunity_labels,
        n_buckets=n_buckets,
    )
    yearly_bucket_report = build_yearly_score_bucket_report(
        predictions,
        labels,
        opportunity_labels=opportunity_labels,
        n_buckets=n_buckets,
    )
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
