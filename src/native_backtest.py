"""Native backtest aligned with Qlib TopkDropoutStrategy semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.label_utils import sanitize_label_series


DEFAULT_ACCOUNT = 100_000_000.0
DEFAULT_MIN_COST = 5.0
DEFAULT_RISK_DEGREE = 0.95
DEFAULT_SLIPPAGE = 0.0005


def _select_topk_dropout_trades(
    scores: pd.Series,
    current_holdings: list[str],
    topk: int,
    n_drop: int,
) -> tuple[list[str], list[str]]:
    """Mirror Qlib TopkDropoutStrategy's sell/buy selection."""
    ranked_scores = scores.dropna().sort_values(ascending=False)
    if ranked_scores.empty:
        return [], []

    current_index = pd.Index(current_holdings, dtype=object)
    ranked_current = ranked_scores.reindex(current_index).sort_values(ascending=False).index

    n_drop = max(0, int(n_drop))
    topk = max(0, int(topk))
    candidate_count = n_drop + max(topk - len(ranked_current), 0)
    today = ranked_scores[~ranked_scores.index.isin(ranked_current)].index[:candidate_count]

    comb = ranked_scores.reindex(ranked_current.union(today)).sort_values(ascending=False).index
    effective_drop = min(n_drop, len(ranked_current))
    drop_set = set(comb[-effective_drop:]) if effective_drop > 0 else set()

    sell = [stock for stock in ranked_current if stock in drop_set]
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


def run_native_backtest(
    preds: pd.Series,
    labels: pd.Series,
    topk: int = 30,
    n_drop: int = 5,
    cost_buy: float = 0.0003,
    cost_sell: float = 0.0013,
    min_cost: float = DEFAULT_MIN_COST,
    account: float = DEFAULT_ACCOUNT,
    risk_degree: float = DEFAULT_RISK_DEGREE,
    slippage: float = DEFAULT_SLIPPAGE,
    rebalance_freq: int = 1,
) -> pd.DataFrame:
    """
    Simulate a Top-K long-only portfolio with Qlib-like dropout and costs.

    The execution model is still simplified versus Qlib because limit-up/down,
    suspension, round lot sizing, and benchmark exchange rules are not modeled.
    """
    rebalance_freq = max(1, int(rebalance_freq))
    topk = max(1, int(topk))
    n_drop = max(0, int(n_drop))
    open_rate = float(cost_buy) + float(slippage)
    close_rate = float(cost_sell) + float(slippage)

    print(
        "[*] Starting Native Backtest "
        f"(Top-{topk}, n_drop={n_drop}, Rebalance: {rebalance_freq} days)..."
    )

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

    cash = float(account)
    holdings: dict[str, float] = {}
    records: list[dict[str, float | int | pd.Timestamp]] = []

    for date in pred_matrix.index:
        start_value = cash + sum(holdings.values())
        trade_cost_value = 0.0
        buy_value = 0.0
        sell_value = 0.0
        buy_count = 0
        sell_count = 0

        if date in rebalance_dates:
            sell_list, buy_list = _select_topk_dropout_trades(
                scores=pred_matrix.loc[date],
                current_holdings=list(holdings.keys()),
                topk=topk,
                n_drop=n_drop,
            )

            for stock in sell_list:
                position_value = holdings.pop(stock, 0.0)
                if position_value <= 0:
                    continue
                cost_value = _trade_cost(position_value, close_rate, min_cost)
                cash += position_value - cost_value
                trade_cost_value += cost_value
                sell_value += position_value
                sell_count += 1

            if buy_list:
                budget_per_stock = cash * float(risk_degree) / len(buy_list)
                for stock in buy_list:
                    max_trade_value = _max_affordable_trade_value(cash, open_rate, min_cost)
                    trade_value = min(budget_per_stock, max_trade_value)
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

        date_returns = label_matrix.loc[date]
        gross_pnl = 0.0
        for stock, position_value in list(holdings.items()):
            stock_ret = date_returns.get(stock, np.nan)
            if pd.isna(stock_ret):
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
                "account_value": end_value,
            }
        )

    report = pd.DataFrame.from_records(records).set_index("datetime")
    report["cum_gross_return"] = (1.0 + report["gross_return"]).cumprod()
    report["cum_net_return"] = (1.0 + report["net_return"]).cumprod()

    ann_factor = 242
    ann_ret = report["net_return"].mean() * ann_factor
    ann_vol = report["net_return"].std() * np.sqrt(ann_factor)
    if pd.isna(ann_vol):
        ann_vol = 0.0
    sharpe = ann_ret / (ann_vol + 1e-8) if ann_vol > 0 else 0.0
    max_drawdown = (report["cum_net_return"] / report["cum_net_return"].cummax() - 1.0).min()

    print("\n[Backtest Results]")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Vol  : {ann_vol:.2%}")
    print(f"Sharpe Ratio    : {sharpe:.4f}")
    print(f"Max Drawdown    : {max_drawdown:.2%}")
    print(f"Avg Daily Turnover: {report['turnover'].mean():.2%}")

    return report
