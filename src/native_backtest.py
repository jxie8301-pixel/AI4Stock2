"""Native backtest aligned with Qlib TopkDropoutStrategy semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.label_utils import sanitize_label_series


DEFAULT_ACCOUNT = 100_000_000.0
DEFAULT_MIN_COST = 5.0
DEFAULT_RISK_DEGREE = 0.95
DEFAULT_SLIPPAGE = 0.0
DEFAULT_TRANSACTION_COST = 0.001
DEFAULT_WEIGHTING = "equal"
SUPPORTED_WEIGHTING_MODES = ("equal", "rank", "score_softmax")
DEFAULT_SCORE_TRANSFORM = "none"
SUPPORTED_SCORE_TRANSFORMS = ("none", "rank_pct", "zscore_clip")
SUPPORTED_RISK_CONTROL_MODES = ("fixed", "benchmark_ma", "signal_strength", "benchmark_ma_signal_strength")
SUPPORTED_SIGNAL_STRENGTH_METRICS = ("top1", "topk_mean", "topk_sum")
SUPPORTED_INTRAPERIOD_EXIT_MODES = ("none", "score_threshold")
SUPPORTED_EXIT_SCORE_SOURCES = ("raw", "transformed", "rank_pct", "zscore")


def _snapshot_holdings(holdings: dict[str, float]) -> dict[str, float]:
    return {stock: float(value) for stock, value in sorted(holdings.items())}


def _normalize_weighting_mode(weighting: str | None) -> str:
    mode = str(weighting or DEFAULT_WEIGHTING).strip().lower()
    return mode or DEFAULT_WEIGHTING


def _normalize_score_transform(score_transform: str | None) -> str:
    mode = str(score_transform or DEFAULT_SCORE_TRANSFORM).strip().lower()
    return mode or DEFAULT_SCORE_TRANSFORM


def _normalize_keep_top_n(keep_top_n: int | None, topk: int) -> int | None:
    if keep_top_n is None:
        return None
    keep_top_n = max(int(keep_top_n), int(topk))
    return keep_top_n


def _normalize_min_score(min_score: float | None) -> float | None:
    if min_score is None:
        return None
    return float(min_score)


def _normalize_intraperiod_exit_config(
    intraperiod_exit: dict[str, object] | None,
) -> dict[str, float | str] | None:
    if intraperiod_exit is None:
        return None
    if not isinstance(intraperiod_exit, dict):
        raise ValueError("intraperiod_exit must be a mapping when provided")
    mode = str(intraperiod_exit.get("mode", "none") or "none").strip().lower()
    if mode == "none":
        return None
    if mode not in SUPPORTED_INTRAPERIOD_EXIT_MODES:
        raise ValueError(
            f"Unsupported intraperiod exit mode: {mode}. Supported: {', '.join(SUPPORTED_INTRAPERIOD_EXIT_MODES)}"
        )
    score_source = str(intraperiod_exit.get("score_source", "raw") or "raw").strip().lower()
    if score_source not in SUPPORTED_EXIT_SCORE_SOURCES:
        raise ValueError(
            f"Unsupported intraperiod exit score_source: {score_source}. "
            f"Supported: {', '.join(SUPPORTED_EXIT_SCORE_SOURCES)}"
        )
    threshold = float(intraperiod_exit.get("threshold", 0.0))
    return {
        "mode": mode,
        "score_source": score_source,
        "threshold": threshold,
    }


def _resolve_intraperiod_exit_scores(
    scores: pd.Series,
    *,
    score_source: str,
    score_transform: str,
    zscore_clip: float,
) -> pd.Series:
    mode = str(score_source or "raw").strip().lower()
    if mode == "raw":
        return pd.to_numeric(scores, errors="coerce").astype(float)
    if mode == "transformed":
        return _transform_scores(scores, score_transform=score_transform, zscore_clip=zscore_clip)
    if mode == "rank_pct":
        return _transform_scores(scores, score_transform="rank_pct", zscore_clip=zscore_clip)
    if mode == "zscore":
        return _transform_scores(scores, score_transform="zscore_clip", zscore_clip=zscore_clip)
    raise ValueError(
        f"Unsupported intraperiod exit score_source: {score_source}. "
        f"Supported: {', '.join(SUPPORTED_EXIT_SCORE_SOURCES)}"
    )


def _validate_risk_degree(value: float, field: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return out


def _normalize_risk_control_config(
    risk_control: dict[str, object] | None,
    *,
    fallback_risk_degree: float,
) -> dict[str, float | int | str]:
    if risk_control is None:
        return {
            "mode": "fixed",
            "risk_degree": _validate_risk_degree(float(fallback_risk_degree), "risk_control.risk_degree"),
        }
    if not isinstance(risk_control, dict):
        raise ValueError("risk_control must be a mapping when provided")
    mode = str(risk_control.get("mode", "fixed") or "fixed").strip().lower()
    if mode not in SUPPORTED_RISK_CONTROL_MODES:
        raise ValueError(
            f"Unsupported risk control mode: {mode}. Supported: {', '.join(SUPPORTED_RISK_CONTROL_MODES)}"
        )
    if mode == "fixed":
        return {
            "mode": "fixed",
            "risk_degree": _validate_risk_degree(
                float(risk_control.get("risk_degree", fallback_risk_degree)),
                "risk_control.risk_degree",
            ),
        }
    if mode in {"signal_strength", "benchmark_ma_signal_strength"}:
        signal_metric = str(risk_control.get("signal_metric", "topk_mean") or "topk_mean").strip().lower()
        if signal_metric not in SUPPORTED_SIGNAL_STRENGTH_METRICS:
            raise ValueError(
                "risk_control.signal_metric must be one of: "
                + ", ".join(SUPPORTED_SIGNAL_STRENGTH_METRICS)
            )
        min_signal = float(risk_control.get("min_signal", 0.0))
        max_signal = float(risk_control.get("max_signal", 2.0))
        if max_signal <= min_signal:
            raise ValueError("risk_control.max_signal must be greater than risk_control.min_signal")
        min_signal_quantile = risk_control.get("min_signal_quantile")
        max_signal_quantile = risk_control.get("max_signal_quantile")
        if min_signal_quantile is not None:
            min_signal_quantile = float(min_signal_quantile)
            if min_signal_quantile < 0.0 or min_signal_quantile > 1.0:
                raise ValueError("risk_control.min_signal_quantile must be in [0, 1]")
        if max_signal_quantile is not None:
            max_signal_quantile = float(max_signal_quantile)
            if max_signal_quantile < 0.0 or max_signal_quantile > 1.0:
                raise ValueError("risk_control.max_signal_quantile must be in [0, 1]")
        if min_signal_quantile is not None and max_signal_quantile is not None and max_signal_quantile <= min_signal_quantile:
            raise ValueError("risk_control.max_signal_quantile must be greater than risk_control.min_signal_quantile")
        min_risk = _validate_risk_degree(
            float(risk_control.get("min_risk", min(fallback_risk_degree, 0.3))),
            "risk_control.min_risk",
        )
        max_risk = _validate_risk_degree(
            float(risk_control.get("max_risk", fallback_risk_degree)),
            "risk_control.max_risk",
        )
        if max_risk < min_risk:
            raise ValueError("risk_control.max_risk must be >= risk_control.min_risk")
        out: dict[str, float | int | str] = {
            "mode": mode,
            "risk_degree": _validate_risk_degree(float(fallback_risk_degree), "risk_control.risk_degree"),
            "signal_metric": signal_metric,
            "min_signal": min_signal,
            "max_signal": max_signal,
            "min_risk": min_risk,
            "max_risk": max_risk,
        }
        if min_signal_quantile is not None:
            out["min_signal_quantile"] = min_signal_quantile
        if max_signal_quantile is not None:
            out["max_signal_quantile"] = max_signal_quantile
        if mode == "signal_strength":
            return out
        fast_window = max(int(risk_control.get("fast_window", 120) or 120), 1)
        slow_window = max(int(risk_control.get("slow_window", 250) or 250), 1)
        if fast_window >= slow_window:
            raise ValueError("risk_control.fast_window must be smaller than risk_control.slow_window")
        bull_risk = _validate_risk_degree(
            float(risk_control.get("bull_risk", fallback_risk_degree)),
            "risk_control.bull_risk",
        )
        neutral_risk = _validate_risk_degree(
            float(risk_control.get("neutral_risk", min(fallback_risk_degree, 0.5))),
            "risk_control.neutral_risk",
        )
        bear_risk = _validate_risk_degree(float(risk_control.get("bear_risk", 0.15)), "risk_control.bear_risk")
        out.update(
            {
                "fast_window": fast_window,
                "slow_window": slow_window,
                "bull_risk": bull_risk,
                "neutral_risk": neutral_risk,
                "bear_risk": bear_risk,
            }
        )
        return out
    fast_window = max(int(risk_control.get("fast_window", 120) or 120), 1)
    slow_window = max(int(risk_control.get("slow_window", 250) or 250), 1)
    if fast_window >= slow_window:
        raise ValueError("risk_control.fast_window must be smaller than risk_control.slow_window")
    bull_risk = _validate_risk_degree(float(risk_control.get("bull_risk", fallback_risk_degree)), "risk_control.bull_risk")
    neutral_risk = _validate_risk_degree(
        float(risk_control.get("neutral_risk", min(fallback_risk_degree, 0.5))),
        "risk_control.neutral_risk",
    )
    bear_risk = _validate_risk_degree(float(risk_control.get("bear_risk", 0.15)), "risk_control.bear_risk")
    return {
        "mode": mode,
        "risk_degree": _validate_risk_degree(float(fallback_risk_degree), "risk_control.risk_degree"),
        "fast_window": fast_window,
        "slow_window": slow_window,
        "bull_risk": bull_risk,
        "neutral_risk": neutral_risk,
        "bear_risk": bear_risk,
    }


def _compute_signal_strength_value(
    scores: pd.Series,
    *,
    topk: int,
    min_score: float | None,
    score_transform: str,
    zscore_clip: float,
    signal_metric: str,
) -> float:
    transformed = _transform_scores(scores, score_transform=score_transform, zscore_clip=zscore_clip)
    ranked = transformed.dropna().sort_values(ascending=False)
    min_score_value = _normalize_min_score(min_score)
    if min_score_value is not None:
        ranked = ranked[ranked > min_score_value]
    if ranked.empty:
        return float("nan")
    top_values = ranked.iloc[: max(int(topk), 1)]
    if signal_metric == "top1":
        return float(top_values.iloc[0])
    if signal_metric == "topk_sum":
        return float(top_values.sum())
    return float(top_values.mean())


def _build_benchmark_ma_schedule(
    benchmark_returns: pd.Series | None,
    *,
    risk_control: dict[str, float | int | str],
    fallback_risk_degree: float,
) -> pd.Series:
    if benchmark_returns is None:
        raise ValueError("benchmark_returns is required when risk_control.mode needs benchmark context")
    benchmark_returns = pd.Series(benchmark_returns).astype(float).sort_index()
    if benchmark_returns.empty:
        raise ValueError("benchmark_returns is empty while risk_control.mode needs benchmark context")

    nav = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    fast_window = int(risk_control["fast_window"])
    slow_window = int(risk_control["slow_window"])
    fast_ma = nav.rolling(fast_window, min_periods=1).mean()
    slow_ma = nav.rolling(slow_window, min_periods=1).mean()

    raw_schedule = pd.Series(float(risk_control["bear_risk"]), index=nav.index, dtype=float)
    raw_schedule.loc[nav >= slow_ma] = float(risk_control["neutral_risk"])
    raw_schedule.loc[nav >= fast_ma] = float(risk_control["bull_risk"])
    return raw_schedule.shift(1).fillna(float(fallback_risk_degree)).astype(float)


def _build_signal_strength_schedule(
    pred_matrix: pd.DataFrame,
    *,
    risk_control: dict[str, float | int | str],
    topk: int,
    min_score: float | None,
    score_transform: str,
    zscore_clip: float,
) -> tuple[pd.Series, pd.Series]:
    signal_metric = str(risk_control["signal_metric"])
    signal_values = pd.Series(
        {
            date: _compute_signal_strength_value(
                pred_matrix.loc[date],
                topk=topk,
                min_score=min_score,
                score_transform=score_transform,
                zscore_clip=zscore_clip,
                signal_metric=signal_metric,
            )
            for date in pred_matrix.index
        },
        dtype=float,
    ).sort_index()
    min_threshold = pd.Series(float(risk_control["min_signal"]), index=signal_values.index, dtype=float)
    max_threshold = pd.Series(float(risk_control["max_signal"]), index=signal_values.index, dtype=float)
    min_q = risk_control.get("min_signal_quantile")
    max_q = risk_control.get("max_signal_quantile")
    shifted = signal_values.shift(1)
    if min_q is not None:
        min_threshold = shifted.expanding(min_periods=1).quantile(float(min_q)).reindex(signal_values.index)
        min_threshold = min_threshold.fillna(float(risk_control["min_signal"])).astype(float)
    if max_q is not None:
        max_threshold = shifted.expanding(min_periods=1).quantile(float(max_q)).reindex(signal_values.index)
        max_threshold = max_threshold.fillna(float(risk_control["max_signal"])).astype(float)
    invalid_thresholds = max_threshold <= min_threshold
    if invalid_thresholds.any():
        min_threshold.loc[invalid_thresholds] = float(risk_control["min_signal"])
        max_threshold.loc[invalid_thresholds] = float(risk_control["max_signal"])
    width = (max_threshold - min_threshold).where(lambda s: s > 0, np.nan)
    scale = ((signal_values - min_threshold) / width).clip(lower=0.0, upper=1.0).fillna(0.0)
    min_risk = float(risk_control["min_risk"])
    max_risk = float(risk_control["max_risk"])
    schedule = (min_risk + scale * (max_risk - min_risk)).astype(float)
    return schedule, signal_values


def _build_risk_control_schedule(
    benchmark_returns: pd.Series | None,
    pred_matrix: pd.DataFrame,
    *,
    risk_control: dict[str, float | int | str],
    fallback_risk_degree: float,
    topk: int,
    min_score: float | None,
    score_transform: str,
    zscore_clip: float,
) -> tuple[pd.Series | None, pd.Series | None]:
    mode = str(risk_control.get("mode") or "fixed")
    if mode == "fixed":
        return None, None
    if mode == "signal_strength":
        schedule, signal_values = _build_signal_strength_schedule(
            pred_matrix,
            risk_control=risk_control,
            topk=topk,
            min_score=min_score,
            score_transform=score_transform,
            zscore_clip=zscore_clip,
        )
        return schedule, signal_values
    if mode == "benchmark_ma":
        return (
            _build_benchmark_ma_schedule(
                benchmark_returns,
                risk_control=risk_control,
                fallback_risk_degree=fallback_risk_degree,
            ),
            None,
        )
    if mode == "benchmark_ma_signal_strength":
        bench_schedule = _build_benchmark_ma_schedule(
            benchmark_returns,
            risk_control=risk_control,
            fallback_risk_degree=fallback_risk_degree,
        )
        signal_schedule, signal_values = _build_signal_strength_schedule(
            pred_matrix,
            risk_control=risk_control,
            topk=topk,
            min_score=min_score,
            score_transform=score_transform,
            zscore_clip=zscore_clip,
        )
        combined = pd.concat([bench_schedule.rename("bench"), signal_schedule.rename("signal")], axis=1)
        final_schedule = combined.min(axis=1).astype(float)
        return final_schedule, signal_values
    raise ValueError(
        f"Unsupported risk control mode: {mode}. Supported: {', '.join(SUPPORTED_RISK_CONTROL_MODES)}"
    )


def _transform_scores(
    scores: pd.Series,
    *,
    score_transform: str,
    zscore_clip: float,
) -> pd.Series:
    transformed = pd.to_numeric(scores, errors="coerce").astype(float)
    mode = _normalize_score_transform(score_transform)

    if transformed.empty or mode == "none":
        return transformed
    if mode == "rank_pct":
        out = transformed.rank(method="average", pct=True)
        out[transformed.isna()] = np.nan
        return out.astype(float)
    if mode == "zscore_clip":
        finite = transformed.dropna()
        if finite.empty:
            return transformed
        std = float(finite.std(ddof=0))
        if np.isfinite(std) and not np.isclose(std, 0.0):
            out = (transformed - float(finite.mean())) / std
        else:
            out = pd.Series(0.0, index=transformed.index, dtype=float)
            out[transformed.isna()] = np.nan
            return out
        clip_value = max(float(zscore_clip), 0.0)
        if clip_value > 0:
            out = out.clip(lower=-clip_value, upper=clip_value)
        return out.astype(float)
    raise ValueError(
        f"Unsupported score transform: {score_transform}. Supported: {', '.join(SUPPORTED_SCORE_TRANSFORMS)}"
    )


def _select_topk_dropout_trades(
    scores: pd.Series,
    current_holdings: list[str],
    topk: int,
    n_drop: int,
    locked_holdings: set[str] | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
    score_transform: str = DEFAULT_SCORE_TRANSFORM,
    zscore_clip: float = 3.0,
) -> tuple[list[str], list[str]]:
    """Mirror Qlib TopkDropoutStrategy's sell/buy selection."""
    raw_scores = _transform_scores(scores, score_transform=score_transform, zscore_clip=zscore_clip)
    ranked_scores = raw_scores.dropna().sort_values(ascending=False)
    min_score_value = _normalize_min_score(min_score)
    eligible_scores = ranked_scores if min_score_value is None else ranked_scores[ranked_scores > min_score_value]
    locked_set = set() if locked_holdings is None else set(locked_holdings)

    if eligible_scores.empty:
        if min_score_value is None:
            return [], []
        return [stock for stock in current_holdings if stock not in locked_set], []

    current_index = pd.Index(current_holdings, dtype=object)
    ranked_current = ranked_scores.reindex(current_index).sort_values(
        ascending=False,
        na_position="last",
    ).index
    keep_top_n_value = _normalize_keep_top_n(keep_top_n, topk)
    eligible_ranks = {stock: rank for rank, stock in enumerate(eligible_scores.index.tolist(), start=1)}
    forced_sell = [
        stock
        for stock in current_holdings
        if stock not in locked_set and (stock not in eligible_ranks)
    ]
    buffer_protected = set()
    if keep_top_n_value is not None:
        buffer_protected = {
            stock
            for stock in current_holdings
            if stock not in locked_set
            and stock not in forced_sell
            and eligible_ranks.get(stock, keep_top_n_value + 1) <= keep_top_n_value
        }

    sellable_current = pd.Index(
        [stock for stock in ranked_current if stock not in locked_set and stock not in forced_sell and stock not in buffer_protected],
        dtype=object,
    )

    n_drop = max(0, int(n_drop))
    topk = max(0, int(topk))
    candidate_count = len(forced_sell) + n_drop + max(topk - len(ranked_current), 0)
    today = eligible_scores[~eligible_scores.index.isin(ranked_current)].index[:candidate_count]

    comb = eligible_scores.reindex(ranked_current.union(today)).sort_values(
        ascending=False,
        na_position="last",
    ).index
    sellable_comb = pd.Index(
        [stock for stock in comb if stock not in locked_set and stock not in forced_sell and stock not in buffer_protected],
        dtype=object,
    )
    effective_drop = min(n_drop, len(sellable_current))
    drop_set = set(sellable_comb[-effective_drop:]) if effective_drop > 0 else set()

    sell = forced_sell + [stock for stock in sellable_current if stock in drop_set]
    buy_count = len(sell) + max(topk - len(ranked_current), 0)
    buy = list(today[:buy_count])
    return sell, buy


def _trade_cost(trade_value: float, rate: float, min_cost: float) -> float:
    if trade_value <= 0:
        return 0.0
    return max(trade_value * rate, min_cost)


def _max_affordable_trade_value(cash: float, rate: float, min_cost: float) -> float:
    if cash <= 0:
        return 0.0
    if rate <= 0:
        return max(cash - min_cost, 0.0) if min_cost > 0 else cash

    proportional_limit = cash / (1.0 + rate)
    threshold = min_cost / rate
    if proportional_limit >= threshold:
        return max(proportional_limit, 0.0)
    return max(cash - min_cost, 0.0)


def _ordered_unique_symbols(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(str(symbol) for symbol in symbols))


def _cap_target_weights(weights: pd.Series, max_weight: float | None) -> pd.Series:
    if weights.empty:
        return weights.astype(float)

    total = float(weights.sum())
    if total <= 0:
        return pd.Series(0.0, index=weights.index, dtype=float)

    normalized = weights.astype(float) / total
    if max_weight is None:
        return normalized

    cap = float(max_weight)
    final = pd.Series(0.0, index=normalized.index, dtype=float)
    remaining_idx = normalized.index
    remaining_budget = 1.0

    while len(remaining_idx) > 0 and remaining_budget > 1e-12:
        active = normalized.loc[remaining_idx]
        active_total = float(active.sum())
        if active_total <= 0:
            break
        proposal = active / active_total * remaining_budget
        over_idx = proposal[proposal > cap + 1e-12].index
        if len(over_idx) == 0:
            final.loc[remaining_idx] = proposal
            remaining_budget = 0.0
            break
        final.loc[over_idx] = cap
        remaining_budget = max(remaining_budget - cap * len(over_idx), 0.0)
        remaining_idx = remaining_idx.difference(over_idx, sort=False)

    return final


def _compute_target_weights(
    scores: pd.Series,
    target_holdings: list[str],
    *,
    weighting: str,
    max_weight: float | None,
    score_transform: str,
    zscore_clip: float,
) -> pd.Series:
    target_index = pd.Index(_ordered_unique_symbols(target_holdings), dtype=object)
    if target_index.empty:
        return pd.Series(dtype=float)

    mode = _normalize_weighting_mode(weighting)
    transformed_scores = _transform_scores(scores, score_transform=score_transform, zscore_clip=zscore_clip)
    target_scores = transformed_scores.reindex(target_index).astype(float)
    if target_scores.notna().any():
        fill_value = float(target_scores.min(skipna=True)) - 1.0
        target_scores = target_scores.fillna(fill_value)
    else:
        target_scores = pd.Series(0.0, index=target_index, dtype=float)

    if mode == "equal":
        raw = pd.Series(1.0, index=target_index, dtype=float)
    elif mode == "rank":
        ranks = target_scores.rank(ascending=False, method="average")
        raw = float(len(target_scores)) - ranks + 1.0
    elif mode == "score_softmax":
        std = float(target_scores.std(ddof=0))
        if np.isfinite(std) and not np.isclose(std, 0.0):
            scaled = (target_scores - target_scores.mean()) / std
        else:
            scaled = pd.Series(0.0, index=target_index, dtype=float)
        scaled = scaled.clip(lower=-20.0, upper=20.0)
        raw = pd.Series(np.exp((scaled - scaled.max()).to_numpy()), index=target_index, dtype=float)
    else:
        raise ValueError(
            f"Unsupported weighting mode: {weighting}. Supported: {', '.join(SUPPORTED_WEIGHTING_MODES)}"
        )

    return _cap_target_weights(raw, max_weight)


def run_native_backtest(
    preds: pd.Series,
    labels: pd.Series,
    topk: int = 30,
    n_drop: int = 5,
    cost_buy: float = DEFAULT_TRANSACTION_COST,
    cost_sell: float = DEFAULT_TRANSACTION_COST,
    min_cost: float = DEFAULT_MIN_COST,
    account: float = DEFAULT_ACCOUNT,
    risk_degree: float = DEFAULT_RISK_DEGREE,
    slippage: float = DEFAULT_SLIPPAGE,
    rebalance_freq: int = 1,
    weighting: str = DEFAULT_WEIGHTING,
    score_transform: str = DEFAULT_SCORE_TRANSFORM,
    score_zscore_clip: float = 3.0,
    max_weight: float | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
    benchmark_returns: pd.Series | None = None,
    risk_control: dict[str, object] | None = None,
    intraperiod_exit: dict[str, object] | None = None,
    dynamic_risk: dict[str, object] | None = None,
    return_trace: bool = False,
    trace_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a Top-K long-only portfolio with Qlib-like dropout and costs.

    The execution model is still simplified versus Qlib because limit-up/down,
    suspension, round lot sizing, and benchmark exchange rules are not modeled.
    """
    rebalance_freq = max(1, int(rebalance_freq))
    topk = max(1, int(topk))
    n_drop = max(0, int(n_drop))
    weighting = _normalize_weighting_mode(weighting)
    score_transform = _normalize_score_transform(score_transform)
    keep_top_n = _normalize_keep_top_n(keep_top_n, topk)
    min_score = _normalize_min_score(min_score)
    risk_degree = _validate_risk_degree(float(risk_degree), "risk_degree")
    score_zscore_clip = max(float(score_zscore_clip), 0.0)
    intraperiod_exit_cfg = _normalize_intraperiod_exit_config(intraperiod_exit)
    if risk_control is not None and dynamic_risk is not None:
        raise ValueError("Provide only one of risk_control or dynamic_risk")
    raw_risk_control = risk_control if risk_control is not None else dynamic_risk
    risk_control_cfg = _normalize_risk_control_config(raw_risk_control, fallback_risk_degree=risk_degree)
    open_rate = float(cost_buy) + float(slippage)
    close_rate = float(cost_sell) + float(slippage)

    common_idx = preds.index.intersection(labels.index)
    preds = preds.loc[common_idx].sort_index()
    labels = sanitize_label_series(labels.loc[common_idx].sort_index())
    if preds.empty:
        raise ValueError("Native backtest received no overlapping prediction/label index.")

    pred_matrix = preds.unstack(level="instrument").sort_index()
    label_matrix = labels.unstack(level="instrument").reindex(pred_matrix.index).sort_index()
    valid_dates = label_matrix.notna().any(axis=1)
    pred_matrix = pred_matrix.loc[valid_dates]
    label_matrix = label_matrix.loc[valid_dates]
    if pred_matrix.empty:
        raise ValueError("Native backtest received no dates with any realized returns.")
    risk_schedule, risk_signal_schedule = _build_risk_control_schedule(
        benchmark_returns,
        pred_matrix,
        risk_control=risk_control_cfg,
        fallback_risk_degree=risk_degree,
        topk=topk,
        min_score=min_score,
        score_transform=score_transform,
        zscore_clip=score_zscore_clip,
    )
    rebalance_dates = set(pred_matrix.index[::rebalance_freq])
    trace_dates_norm = None
    if trace_dates is not None:
        trace_dates_norm = {pd.Timestamp(date) for date in trace_dates}

    cash = float(account)
    holdings: dict[str, float] = {}
    records: list[dict[str, float | int | pd.Timestamp]] = []
    trace_records: list[dict[str, object]] = []

    for date in pred_matrix.index:
        cash_before = float(cash)
        holdings_before = _snapshot_holdings(holdings)
        start_value = cash + sum(holdings.values())
        current_risk_degree = float(risk_schedule.get(date, risk_control_cfg["risk_degree"])) if risk_schedule is not None else float(risk_control_cfg["risk_degree"])
        current_risk_signal = (
            float(risk_signal_schedule.get(date))
            if risk_signal_schedule is not None and pd.notna(risk_signal_schedule.get(date))
            else np.nan
        )
        trade_cost_value = 0.0
        buy_value = 0.0
        sell_value = 0.0
        buy_count = 0
        sell_count = 0
        intraperiod_exit_count = 0
        frozen_holdings = 0
        date_returns = label_matrix.loc[date]
        locked_holdings = {
            stock
            for stock in holdings
            if pd.isna(date_returns.get(stock, np.nan))
        }

        if date in rebalance_dates:
            sell_list, buy_list = _select_topk_dropout_trades(
                scores=pred_matrix.loc[date],
                current_holdings=list(holdings.keys()),
                topk=topk,
                n_drop=n_drop,
                locked_holdings=locked_holdings,
                keep_top_n=keep_top_n,
                min_score=min_score,
                score_transform=score_transform,
                zscore_clip=score_zscore_clip,
            )
            trade_sell_list: list[str] = []
            trade_buy_list: list[str] = []

            for stock in sell_list:
                position_value = holdings.pop(stock, 0.0)
                if position_value <= 0:
                    continue
                cost_value = _trade_cost(position_value, close_rate, min_cost)
                cash += position_value - cost_value
                trade_cost_value += cost_value
                sell_value += position_value
                sell_count += 1
                trade_sell_list.append(stock)

            tradable_holdings = [stock for stock in holdings if stock not in locked_holdings]
            if min_score is None:
                eligible_current_holdings = tradable_holdings
            else:
                day_scores = _transform_scores(
                    pred_matrix.loc[date],
                    score_transform=score_transform,
                    zscore_clip=score_zscore_clip,
                )
                eligible_current_holdings = [
                    stock
                    for stock in tradable_holdings
                    if float(day_scores.get(stock, np.nan)) > min_score
                ]
            target_holdings = _ordered_unique_symbols(eligible_current_holdings + list(buy_list))
            target_weights = _compute_target_weights(
                pred_matrix.loc[date],
                target_holdings,
                weighting=weighting,
                max_weight=max_weight,
                score_transform=score_transform,
                zscore_clip=score_zscore_clip,
            )
            locked_value = float(sum(holdings.get(stock, 0.0) for stock in locked_holdings))
            tradable_budget = max(start_value * current_risk_degree - locked_value, 0.0)
            target_values = target_weights * tradable_budget

            for stock in list(tradable_holdings):
                current_value = float(holdings.get(stock, 0.0))
                target_value = float(target_values.get(stock, 0.0))
                trade_value = current_value - target_value
                if trade_value <= 1e-12:
                    continue
                cost_value = _trade_cost(trade_value, close_rate, min_cost)
                cash += trade_value - cost_value
                holdings[stock] = current_value - trade_value
                if holdings[stock] <= 1e-12:
                    holdings.pop(stock, None)
                trade_cost_value += cost_value
                sell_value += trade_value
                sell_count += 1
                if stock not in trade_sell_list:
                    trade_sell_list.append(stock)

            for stock in target_values.sort_values(ascending=False).index:
                target_value = float(target_values.get(stock, 0.0))
                current_value = float(holdings.get(stock, 0.0))
                deficit_value = target_value - current_value
                if deficit_value <= 1e-12:
                    continue
                max_trade_value = _max_affordable_trade_value(cash, open_rate, min_cost)
                trade_value = min(deficit_value, max_trade_value)
                if trade_value <= 0:
                    continue
                cost_value = _trade_cost(trade_value, open_rate, min_cost)
                if trade_value + cost_value > cash:
                    trade_value = _max_affordable_trade_value(cash, open_rate, min_cost)
                    cost_value = _trade_cost(trade_value, open_rate, min_cost) if trade_value > 0 else 0.0
                if trade_value <= 0 or trade_value + cost_value > cash:
                    continue
                holdings[stock] = holdings.get(stock, 0.0) + trade_value
                cash -= trade_value + cost_value
                trade_cost_value += cost_value
                buy_value += trade_value
                buy_count += 1
                if stock not in trade_buy_list:
                    trade_buy_list.append(stock)
        else:
            sell_list = []
            buy_list = []
            trade_sell_list = []
            trade_buy_list = []
            target_weights = pd.Series(dtype=float)
            target_values = pd.Series(dtype=float)
            if intraperiod_exit_cfg is not None:
                day_scores = _resolve_intraperiod_exit_scores(
                    pred_matrix.loc[date],
                    score_source=str(intraperiod_exit_cfg["score_source"]),
                    score_transform=score_transform,
                    zscore_clip=score_zscore_clip,
                )
                threshold = float(intraperiod_exit_cfg["threshold"])
                for stock in list(holdings.keys()):
                    if stock in locked_holdings:
                        continue
                    score_value = day_scores.get(stock, np.nan)
                    if pd.isna(score_value) or float(score_value) > threshold:
                        continue
                    position_value = holdings.pop(stock, 0.0)
                    if position_value <= 0:
                        continue
                    cost_value = _trade_cost(position_value, close_rate, min_cost)
                    cash += position_value - cost_value
                    trade_cost_value += cost_value
                    sell_value += position_value
                    sell_count += 1
                    intraperiod_exit_count += 1
                    sell_list.append(stock)
                    trade_sell_list.append(stock)

        gross_pnl = 0.0
        for stock, position_value in list(holdings.items()):
            stock_ret = date_returns.get(stock, np.nan)
            if pd.isna(stock_ret):
                # Treat missing realized return as a frozen position: mark flat
                # for this step, keep the capital tied up, and block rebalancing.
                frozen_holdings += 1
                stock_ret = 0.0
            new_value = position_value * (1.0 + float(stock_ret))
            gross_pnl += new_value - position_value
            holdings[stock] = new_value

        end_value = cash + sum(holdings.values())
        denom = start_value if start_value > 0 else 1.0
        gross_return = gross_pnl / denom
        net_return = (end_value - start_value) / denom
        turnover = (buy_value + sell_value) / (2.0 * denom)
        cost_return = trade_cost_value / denom

        records.append(
            {
                "datetime": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": cost_return,
                "bench": float(date_returns.mean(skipna=True))
                if not date_returns.empty and not pd.isna(date_returns.mean(skipna=True))
                else 0.0,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "intraperiod_exit_count": intraperiod_exit_count,
                "holdings": len(holdings),
                "frozen_holdings": frozen_holdings,
                "account_value": end_value,
                "risk_degree": current_risk_degree,
                "risk_control_signal": current_risk_signal,
            }
        )
        if return_trace and (trace_dates_norm is None or pd.Timestamp(date) in trace_dates_norm):
            trace_records.append(
                {
                    "datetime": date,
                    "start_value": float(start_value),
                    "end_value": float(end_value),
                    "cash_before": cash_before,
                    "cash_after": float(cash),
                    "holdings_before": holdings_before,
                    "holdings_after": _snapshot_holdings(holdings),
                    "locked_holdings": sorted(locked_holdings),
                    "sell_list": list(sell_list),
                    "buy_list": list(buy_list),
                    "trade_sell_list": list(trade_sell_list),
                    "trade_buy_list": list(trade_buy_list),
                    "weighting": weighting,
                    "risk_degree": current_risk_degree,
                    "risk_control_mode": risk_control_cfg["mode"],
                    "risk_control_signal": current_risk_signal,
                    "intraperiod_exit_mode": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg["mode"],
                    "intraperiod_exit_score_source": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg["score_source"],
                    "intraperiod_exit_threshold": None if intraperiod_exit_cfg is None else float(intraperiod_exit_cfg["threshold"]),
                    "intraperiod_exit_count": int(intraperiod_exit_count),
                    "score_transform": score_transform,
                    "score_zscore_clip": score_zscore_clip,
                    "keep_top_n": keep_top_n,
                    "min_score": min_score,
                    "target_weights": {stock: float(value) for stock, value in target_weights.items()},
                    "target_values": {stock: float(value) for stock, value in target_values.items()},
                    "buy_count": int(buy_count),
                    "sell_count": int(sell_count),
                    "buy_value": float(buy_value),
                    "sell_value": float(sell_value),
                    "trade_cost_value": float(trade_cost_value),
                    "gross_return": float(gross_return),
                    "net_return": float(net_return),
                    "frozen_holdings": int(frozen_holdings),
                }
            )

    report = pd.DataFrame.from_records(records).set_index("datetime")
    report["cum_gross_return"] = (1.0 + report["gross_return"]).cumprod()
    report["cum_net_return"] = (1.0 + report["net_return"]).cumprod()

    if not return_trace:
        return report

    trace_df = pd.DataFrame.from_records(trace_records)
    if not trace_df.empty:
        trace_df = trace_df.set_index("datetime")
    return report, trace_df
