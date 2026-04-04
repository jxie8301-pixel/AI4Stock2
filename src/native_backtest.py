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
SUPPORTED_RISK_CONTROL_MODES = ("fixed", "benchmark_ma")


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


def _build_risk_control_schedule(
    benchmark_returns: pd.Series | None,
    *,
    risk_control: dict[str, float | int | str],
    fallback_risk_degree: float,
) -> pd.Series | None:
    mode = str(risk_control.get("mode") or "fixed")
    if mode == "fixed":
        return None
    if benchmark_returns is None:
        raise ValueError("benchmark_returns is required when risk_control.mode needs benchmark context")
    benchmark_returns = pd.Series(benchmark_returns).astype(float).sort_index()
    if benchmark_returns.empty:
        raise ValueError("benchmark_returns is empty while risk_control.mode needs benchmark context")

    if mode != "benchmark_ma":
        raise ValueError(
            f"Unsupported risk control mode: {mode}. Supported: {', '.join(SUPPORTED_RISK_CONTROL_MODES)}"
        )

    nav = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    fast_window = int(risk_control["fast_window"])
    slow_window = int(risk_control["slow_window"])
    fast_ma = nav.rolling(fast_window, min_periods=1).mean()
    slow_ma = nav.rolling(slow_window, min_periods=1).mean()

    raw_schedule = pd.Series(float(risk_control["bear_risk"]), index=nav.index, dtype=float)
    raw_schedule.loc[nav >= slow_ma] = float(risk_control["neutral_risk"])
    raw_schedule.loc[nav >= fast_ma] = float(risk_control["bull_risk"])

    # Use only information available before the current trading day.
    return raw_schedule.shift(1).fillna(float(fallback_risk_degree)).astype(float)


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
    if risk_control is not None and dynamic_risk is not None:
        raise ValueError("Provide only one of risk_control or dynamic_risk")
    raw_risk_control = risk_control if risk_control is not None else dynamic_risk
    risk_control_cfg = _normalize_risk_control_config(raw_risk_control, fallback_risk_degree=risk_degree)
    risk_schedule = _build_risk_control_schedule(
        benchmark_returns,
        risk_control=risk_control_cfg,
        fallback_risk_degree=risk_degree,
    )
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
        trade_cost_value = 0.0
        buy_value = 0.0
        sell_value = 0.0
        buy_count = 0
        sell_count = 0
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
                "holdings": len(holdings),
                "frozen_holdings": frozen_holdings,
                "account_value": end_value,
                "risk_degree": current_risk_degree,
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
