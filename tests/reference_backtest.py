from __future__ import annotations

import numpy as np
import pandas as pd

from src.label_utils import sanitize_label_series

DEFAULT_WEIGHTING = "equal"
DEFAULT_SCORE_TRANSFORM = "none"


def _select_topk_dropout_trades_reference(
    scores: pd.Series,
    current_holdings: list[str],
    topk: int,
    n_drop: int,
    locked_holdings: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    ranked_scores = scores.dropna().sort_values(ascending=False)
    if ranked_scores.empty:
        return [], []

    current_ranked = (
        ranked_scores.reindex(pd.Index(current_holdings, dtype=object))
        .sort_values(ascending=False, na_position="last")
        .index.tolist()
    )
    locked_set = set() if locked_holdings is None else set(locked_holdings)
    sellable_current = [stock for stock in current_ranked if stock not in locked_set]

    n_drop = max(0, int(n_drop))
    topk = max(0, int(topk))
    candidate_count = n_drop + max(topk - len(current_ranked), 0)
    today = [stock for stock in ranked_scores.index.tolist() if stock not in current_ranked][:candidate_count]

    comb = (
        ranked_scores.reindex(pd.Index(current_ranked).union(pd.Index(today)))
        .sort_values(ascending=False, na_position="last")
        .index.tolist()
    )
    sellable_comb = [stock for stock in comb if stock not in locked_set]
    effective_drop = min(n_drop, len(sellable_current))
    drop_set = set(sellable_comb[-effective_drop:]) if effective_drop > 0 else set()

    sell = [stock for stock in sellable_current if stock in drop_set]
    buy_count = len(sell) + max(topk - len(current_ranked), 0)
    buy = today[:buy_count]
    return sell, buy


def _trade_cost_reference(trade_value: float, rate: float, min_cost: float) -> float:
    if trade_value <= 0:
        return 0.0
    return max(trade_value * rate, min_cost)


def _max_affordable_trade_value_reference(cash: float, rate: float, min_cost: float) -> float:
    if cash <= 0:
        return 0.0
    if rate <= 0:
        return max(cash - min_cost, 0.0) if min_cost > 0 else cash

    proportional_limit = cash / (1.0 + rate)
    threshold = min_cost / rate
    if proportional_limit >= threshold:
        return max(proportional_limit, 0.0)
    return max(cash - min_cost, 0.0)


def _normalize_keep_top_n_reference(keep_top_n: int | None, topk: int) -> int | None:
    if keep_top_n is None:
        return None
    return max(int(keep_top_n), int(topk))


def _normalize_min_score_reference(min_score: float | None) -> float | None:
    if min_score is None:
        return None
    return float(min_score)


def _normalize_score_transform_reference(score_transform: str | None) -> str:
    mode = str(score_transform or DEFAULT_SCORE_TRANSFORM).strip().lower()
    return mode or DEFAULT_SCORE_TRANSFORM


def _transform_scores_reference(
    scores: pd.Series,
    *,
    score_transform: str,
    zscore_clip: float,
) -> pd.Series:
    transformed = pd.to_numeric(scores, errors="coerce").astype(float)
    mode = _normalize_score_transform_reference(score_transform)

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
    raise ValueError(f"Unsupported score transform: {score_transform}")


def _ordered_unique_symbols_reference(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(str(symbol) for symbol in symbols))


def _cap_target_weights_reference(weights: pd.Series, max_weight: float | None) -> pd.Series:
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


def _compute_target_weights_reference(
    scores: pd.Series,
    target_holdings: list[str],
    *,
    weighting: str,
    max_weight: float | None,
    score_transform: str,
    zscore_clip: float,
) -> pd.Series:
    target_index = pd.Index(_ordered_unique_symbols_reference(target_holdings), dtype=object)
    if target_index.empty:
        return pd.Series(dtype=float)

    transformed_scores = _transform_scores_reference(scores, score_transform=score_transform, zscore_clip=zscore_clip)
    target_scores = transformed_scores.reindex(target_index).astype(float)
    if target_scores.notna().any():
        fill_value = float(target_scores.min(skipna=True)) - 1.0
        target_scores = target_scores.fillna(fill_value)
    else:
        target_scores = pd.Series(0.0, index=target_index, dtype=float)

    mode = str(weighting or DEFAULT_WEIGHTING).strip().lower() or DEFAULT_WEIGHTING
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
        raise ValueError(f"Unsupported weighting mode: {weighting}")

    return _cap_target_weights_reference(raw, max_weight)


def _select_trades_with_overrides_reference(
    scores: pd.Series,
    current_holdings: list[str],
    *,
    topk: int,
    n_drop: int,
    locked_holdings: set[str] | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
    score_transform: str = DEFAULT_SCORE_TRANSFORM,
    zscore_clip: float = 3.0,
) -> tuple[list[str], list[str]]:
    raw_scores = _transform_scores_reference(scores, score_transform=score_transform, zscore_clip=zscore_clip)
    ranked_scores = raw_scores.dropna().sort_values(ascending=False)
    min_score_value = _normalize_min_score_reference(min_score)
    eligible_scores = ranked_scores if min_score_value is None else ranked_scores[ranked_scores > min_score_value]
    locked_set = set() if locked_holdings is None else set(locked_holdings)

    if eligible_scores.empty:
        if min_score_value is None:
            return [], []
        return [stock for stock in current_holdings if stock not in locked_set], []

    current_index = pd.Index(current_holdings, dtype=object)
    ranked_current = ranked_scores.reindex(current_index).sort_values(ascending=False, na_position="last").index.tolist()
    keep_top_n_value = _normalize_keep_top_n_reference(keep_top_n, topk)
    eligible_ranks = {stock: rank for rank, stock in enumerate(eligible_scores.index.tolist(), start=1)}
    forced_sell = [
        stock
        for stock in current_holdings
        if stock not in locked_set and stock not in eligible_ranks
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

    sellable_current = [stock for stock in ranked_current if stock not in locked_set and stock not in forced_sell and stock not in buffer_protected]
    candidate_count = len(forced_sell) + max(0, int(n_drop)) + max(max(0, int(topk)) - len(ranked_current), 0)
    today = [stock for stock in eligible_scores.index.tolist() if stock not in ranked_current][:candidate_count]
    comb = (
        eligible_scores.reindex(pd.Index(ranked_current).union(pd.Index(today)))
        .sort_values(ascending=False, na_position="last")
        .index.tolist()
    )
    sellable_comb = [
        stock for stock in comb if stock not in locked_set and stock not in forced_sell and stock not in buffer_protected
    ]
    effective_drop = min(max(0, int(n_drop)), len(sellable_current))
    drop_set = set(sellable_comb[-effective_drop:]) if effective_drop > 0 else set()
    sell = forced_sell + [stock for stock in sellable_current if stock in drop_set]
    buy_count = len(sell) + max(max(0, int(topk)) - len(ranked_current), 0)
    buy = today[:buy_count]
    return sell, buy


def run_reference_backtest(
    preds: pd.Series,
    labels: pd.Series,
    topk: int = 30,
    n_drop: int = 5,
    cost_buy: float = 0.0003,
    cost_sell: float = 0.0013,
    min_cost: float = 5.0,
    account: float = 100_000_000.0,
    risk_degree: float = 0.95,
    slippage: float = 0.0005,
    rebalance_freq: int = 1,
    weighting: str = DEFAULT_WEIGHTING,
    score_transform: str = DEFAULT_SCORE_TRANSFORM,
    score_zscore_clip: float = 3.0,
    max_weight: float | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
) -> pd.DataFrame:
    common_idx = preds.index.intersection(labels.index)
    preds = preds.loc[common_idx].sort_index()
    labels = sanitize_label_series(labels.loc[common_idx].sort_index())
    if preds.empty:
        raise ValueError("Reference backtest received no overlapping prediction/label index.")

    pred_matrix = preds.unstack(level="instrument").sort_index()
    label_matrix = labels.unstack(level="instrument").reindex(pred_matrix.index).sort_index()
    valid_dates = label_matrix.notna().any(axis=1)
    pred_matrix = pred_matrix.loc[valid_dates]
    label_matrix = label_matrix.loc[valid_dates]
    if pred_matrix.empty:
        raise ValueError("Reference backtest received no dates with any realized returns.")

    open_rate = float(cost_buy) + float(slippage)
    close_rate = float(cost_sell) + float(slippage)
    rebalance_dates = set(pred_matrix.index[:: max(1, int(rebalance_freq))])
    keep_top_n = _normalize_keep_top_n_reference(keep_top_n, topk)
    min_score = _normalize_min_score_reference(min_score)

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

        date_returns = label_matrix.loc[date]
        locked_holdings = {
            stock for stock in list(holdings.keys()) if pd.isna(date_returns.get(stock, np.nan))
        }

        if date in rebalance_dates:
            sell_list, buy_list = _select_trades_with_overrides_reference(
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

            for stock in sell_list:
                position_value = holdings.pop(stock, 0.0)
                if position_value <= 0:
                    continue
                cost_value = _trade_cost_reference(position_value, close_rate, min_cost)
                cash += position_value - cost_value
                trade_cost_value += cost_value
                sell_value += position_value
                sell_count += 1

            tradable_holdings = [stock for stock in holdings if stock not in locked_holdings]
            if min_score is None:
                eligible_current_holdings = tradable_holdings
            else:
                day_scores = _transform_scores_reference(
                    pred_matrix.loc[date],
                    score_transform=score_transform,
                    zscore_clip=score_zscore_clip,
                )
                eligible_current_holdings = [
                    stock
                    for stock in tradable_holdings
                    if float(day_scores.get(stock, np.nan)) > min_score
                ]
            target_holdings = _ordered_unique_symbols_reference(eligible_current_holdings + list(buy_list))
            target_weights = _compute_target_weights_reference(
                pred_matrix.loc[date],
                target_holdings,
                weighting=weighting,
                max_weight=max_weight,
                score_transform=score_transform,
                zscore_clip=score_zscore_clip,
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
                cost_value = _trade_cost_reference(trade_value, close_rate, min_cost)
                cash += trade_value - cost_value
                holdings[stock] = current_value - trade_value
                if holdings[stock] <= 1e-12:
                    holdings.pop(stock, None)
                trade_cost_value += cost_value
                sell_value += trade_value
                sell_count += 1

            for stock in target_values.sort_values(ascending=False).index:
                target_value = float(target_values.get(stock, 0.0))
                current_value = float(holdings.get(stock, 0.0))
                deficit_value = target_value - current_value
                if deficit_value <= 1e-12:
                    continue
                max_trade_value = _max_affordable_trade_value_reference(cash, open_rate, min_cost)
                trade_value = min(deficit_value, max_trade_value)
                if trade_value <= 0:
                    continue
                cost_value = _trade_cost_reference(trade_value, open_rate, min_cost)
                if trade_value + cost_value > cash:
                    trade_value = _max_affordable_trade_value_reference(cash, open_rate, min_cost)
                    cost_value = _trade_cost_reference(trade_value, open_rate, min_cost) if trade_value > 0 else 0.0
                if trade_value <= 0 or trade_value + cost_value > cash:
                    continue
                holdings[stock] = holdings.get(stock, 0.0) + trade_value
                cash -= trade_value + cost_value
                trade_cost_value += cost_value
                buy_value += trade_value
                buy_count += 1

        gross_pnl = 0.0
        frozen_holdings = 0
        for stock, position_value in list(holdings.items()):
            stock_ret = date_returns.get(stock, np.nan)
            if pd.isna(stock_ret):
                frozen_holdings += 1
                stock_ret = 0.0
            new_value = position_value * (1.0 + float(stock_ret))
            gross_pnl += new_value - position_value
            holdings[stock] = new_value

        end_value = cash + sum(holdings.values())
        denom = start_value if start_value > 0 else 1.0
        gross_return = gross_pnl / denom
        net_return = (end_value - start_value) / denom

        records.append(
            {
                "datetime": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": (buy_value + sell_value) / (2.0 * denom),
                "cost": trade_cost_value / denom,
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

    report = pd.DataFrame.from_records(records).set_index("datetime")
    report["cum_gross_return"] = (1.0 + report["gross_return"]).cumprod()
    report["cum_net_return"] = (1.0 + report["net_return"]).cumprod()
    return report
