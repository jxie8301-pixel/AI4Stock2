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


def _snapshot_holdings(holdings: dict[str, float]) -> dict[str, float]:
    return {stock: float(value) for stock, value in sorted(holdings.items())}


def _normalize_weighting_mode(weighting: str | None) -> str:
    mode = str(weighting or DEFAULT_WEIGHTING).strip().lower()
    return mode or DEFAULT_WEIGHTING


def _select_topk_dropout_trades(
    scores: pd.Series,
    current_holdings: list[str],
    topk: int,
    n_drop: int,
    locked_holdings: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Mirror Qlib TopkDropoutStrategy's sell/buy selection."""
    ranked_scores = scores.dropna().sort_values(ascending=False)
    if ranked_scores.empty:
        return [], []

    current_index = pd.Index(current_holdings, dtype=object)
    ranked_current = ranked_scores.reindex(current_index).sort_values(
        ascending=False,
        na_position="last",
    ).index
    locked_set = set() if locked_holdings is None else set(locked_holdings)
    sellable_current = pd.Index([stock for stock in ranked_current if stock not in locked_set], dtype=object)

    n_drop = max(0, int(n_drop))
    topk = max(0, int(topk))
    candidate_count = n_drop + max(topk - len(ranked_current), 0)
    today = ranked_scores[~ranked_scores.index.isin(ranked_current)].index[:candidate_count]

    comb = ranked_scores.reindex(ranked_current.union(today)).sort_values(
        ascending=False,
        na_position="last",
    ).index
    sellable_comb = pd.Index([stock for stock in comb if stock not in locked_set], dtype=object)
    effective_drop = min(n_drop, len(sellable_current))
    drop_set = set(sellable_comb[-effective_drop:]) if effective_drop > 0 else set()

    sell = [stock for stock in sellable_current if stock in drop_set]
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
) -> pd.Series:
    target_index = pd.Index(_ordered_unique_symbols(target_holdings), dtype=object)
    if target_index.empty:
        return pd.Series(dtype=float)

    mode = _normalize_weighting_mode(weighting)
    target_scores = pd.to_numeric(scores.reindex(target_index), errors="coerce").astype(float)
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
    max_weight: float | None = None,
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
            target_holdings = _ordered_unique_symbols(tradable_holdings + list(buy_list))
            target_weights = _compute_target_weights(
                pred_matrix.loc[date],
                target_holdings,
                weighting=weighting,
                max_weight=max_weight,
            )
            locked_value = float(sum(holdings.get(stock, 0.0) for stock in locked_holdings))
            tradable_budget = max(start_value * float(risk_degree) - locked_value, 0.0)
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
