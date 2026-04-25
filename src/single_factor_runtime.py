"""Runtime helpers shared by single-factor diagnostics entrypoints."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluate import build_benchmark_series
from src.industry_groups import load_instrument_industry_groups
from src.label_utils import build_opportunity_edge_series
from src.return_horizon import build_forward_compound_return_series


def resolve_period_dates(cfg: dict[str, Any], args: Any) -> tuple[str, str]:
    if args.date_start or args.date_end:
        if not args.date_start or not args.date_end:
            raise ValueError("Provide both --date-start and --date-end when overriding the diagnostics range.")
        return str(args.date_start), str(args.date_end)
    if args.period == "all":
        start = min(cfg["time"]["train"][0], cfg["time"]["valid"][0], cfg["time"]["test"][0])
        end = max(cfg["time"]["train"][1], cfg["time"]["valid"][1], cfg["time"]["test"][1])
        return str(start), str(end)
    split = cfg["time"][args.period]
    return str(split[0]), str(split[1])


def parse_custom_segments(raw: str | None) -> list[tuple[str, str, str]]:
    if raw is None or not str(raw).strip():
        return []
    segments: list[tuple[str, str, str]] = []
    for item in str(raw).split(";"):
        text = item.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "Custom segments must use 'name:start:end' format separated by ';'. "
                f"Got: {text}"
            )
        segments.append((parts[0], parts[1], parts[2]))
    return segments


def resolve_segments(cfg: dict[str, Any], args: Any, *, main_start: str, main_end: str) -> list[tuple[str, str, str]]:
    segments: list[tuple[str, str, str]] = []
    if args.segment_scheme == "yearly":
        start_ts = pd.Timestamp(main_start)
        end_ts = pd.Timestamp(main_end)
        for year in range(int(start_ts.year), int(end_ts.year) + 1):
            seg_start = max(start_ts, pd.Timestamp(f"{year}-01-01"))
            seg_end = min(end_ts, pd.Timestamp(f"{year}-12-31"))
            if seg_start <= seg_end:
                segments.append((f"y{year}", str(seg_start.date()), str(seg_end.date())))
    if args.segment_scheme == "config_split":
        main_start_ts = pd.Timestamp(main_start)
        main_end_ts = pd.Timestamp(main_end)
        for name in ("train", "valid", "test"):
            split = cfg["time"].get(name)
            if not split:
                continue
            start = pd.Timestamp(split[0])
            end = pd.Timestamp(split[1])
            clipped_start = max(start, main_start_ts)
            clipped_end = min(end, main_end_ts)
            if clipped_start <= clipped_end:
                segments.append((name, str(clipped_start.date()), str(clipped_end.date())))
    segments.extend(parse_custom_segments(args.segments))
    deduped: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for name, start, end in segments:
        if name in seen:
            continue
        deduped.append((name, start, end))
        seen.add(name)
    return deduped


def apply_industry_neutralization(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    feature_names: list[str],
) -> tuple[pd.DataFrame, int]:
    groups = load_instrument_industry_groups(
        cfg,
        instruments=frame["symbol"].drop_duplicates(),
        required=True,
    )

    out = frame.copy()
    industry_series = pd.Series(out["symbol"].astype(str)).map(groups.to_dict())
    out["_industry_group"] = industry_series.to_numpy(copy=False)
    valid_mask = out["_industry_group"].notna()
    if not bool(valid_mask.any()):
        raise ValueError("Industry-neutral diagnostics requested, but no rows mapped to a valid industry group.")
    neutralized_feature_count = 0
    grouped = out.loc[valid_mask].groupby(["date", "_industry_group"], observed=True)
    for feature_name in feature_names:
        out.loc[valid_mask, feature_name] = (
            out.loc[valid_mask, feature_name]
            - grouped[feature_name].transform("mean")
        )
        neutralized_feature_count += 1
    out = out.drop(columns=["_industry_group"])
    return out, neutralized_feature_count


def derive_diagnostic_label_series(
    frame: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    signal_horizon: int,
    diagnostic_label_space: str,
    diagnostic_threshold: float,
) -> pd.Series:
    label_space = str(diagnostic_label_space or "raw_return").strip().lower()
    base = frame[["date", "symbol", "label"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    base["symbol"] = base["symbol"].astype(str)
    multi_index = pd.MultiIndex.from_arrays(
        [base["date"], base["symbol"]],
        names=["datetime", "instrument"],
    )
    labels = pd.Series(pd.to_numeric(base["label"], errors="coerce").to_numpy(), index=multi_index, dtype=float)

    if label_space == "raw_return":
        return labels

    opportunity_cfg = {
        "mode": label_space,
        "threshold": float(diagnostic_threshold),
        "neutral_band": 0.0,
    }
    instrument_groups = None
    benchmark_forward_returns = None

    if label_space == "industry_excess":
        instrument_groups = load_instrument_industry_groups(
            cfg,
            instruments=base["symbol"].drop_duplicates(),
            required=True,
        )
    elif label_space == "benchmark_excess":
        benchmark_series, _ = build_benchmark_series(labels, cfg.get("backtest", {}).get("benchmark"))
        benchmark_forward_returns = build_forward_compound_return_series(
            benchmark_series,
            horizon=signal_horizon,
        )
        if benchmark_forward_returns.empty:
            raise ValueError("Benchmark-excess diagnostics produced an empty benchmark forward-return series.")

    return build_opportunity_edge_series(
        labels,
        opportunity_cfg=opportunity_cfg,
        instrument_groups=instrument_groups,
        benchmark_forward_returns=benchmark_forward_returns,
    )
