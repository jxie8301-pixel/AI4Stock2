"""Helpers for selecting and saving backtest trace diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.backtest_report import get_backtest_return_series


def parse_trace_dates_arg(trace_dates_arg: str | None) -> set[pd.Timestamp]:
    if not trace_dates_arg:
        return set()
    return {pd.Timestamp(token.strip()) for token in trace_dates_arg.split(",") if token.strip()}


def select_trace_dates(report: pd.DataFrame, top_n: int = 5) -> list[pd.Timestamp]:
    """Pick a compact set of dates worth manual inspection."""
    if report.empty:
        return []

    top_n = max(0, int(top_n))
    report = report.copy()
    report.index = pd.to_datetime(report.index)

    return_series = get_backtest_return_series(report)
    candidates: list[pd.Timestamp] = []
    if top_n > 0 and return_series is not None:
        candidates.extend(pd.to_datetime(return_series.abs().nlargest(top_n).index).tolist())
    if top_n > 0 and "turnover" in report.columns:
        candidates.extend(pd.to_datetime(report["turnover"].nlargest(top_n).index).tolist())
    if top_n > 0 and "cost" in report.columns:
        candidates.extend(pd.to_datetime(report["cost"].nlargest(top_n).index).tolist())

    if return_series is not None:
        cum_returns = (1.0 + return_series).cumprod()
        drawdown = cum_returns / cum_returns.cummax() - 1.0
        if not drawdown.empty:
            candidates.append(pd.Timestamp(drawdown.idxmin()))

    unique_dates = sorted({pd.Timestamp(date) for date in candidates})
    return unique_dates


def save_trace_artifacts(
    trace_df: pd.DataFrame,
    trace_dates: list[pd.Timestamp],
    results_dir: str | Path,
    prefix: str = "",
) -> tuple[Path, Path]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""

    trace_path = out_dir / f"{prefix}backtest_trace.csv"
    dates_path = out_dir / f"{prefix}backtest_trace_dates.json"

    trace_to_save = trace_df.copy()
    if not trace_to_save.empty:
        trace_to_save.index = pd.to_datetime(trace_to_save.index)
        trace_to_save.index.name = "datetime"
        trace_to_save.to_csv(trace_path)
    else:
        pd.DataFrame().to_csv(trace_path, index=False)

    with open(dates_path, "w", encoding="utf-8") as f:
        json.dump([pd.Timestamp(date).strftime("%Y-%m-%d") for date in trace_dates], f, indent=2)

    return trace_path, dates_path
