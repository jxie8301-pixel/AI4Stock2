"""Execution engine for native backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_controls import (
    SUPPORTED_EXIT_SCORE_SOURCES,
    append_intraperiod_history,
    build_intraperiod_remaining_steps,
    build_intraperiod_price_confirm_matrix,
    build_intraperiod_residual_return_matrix,
    build_risk_control_schedule,
    estimate_intraperiod_expected_returns,
    normalize_intraperiod_exit_config,
    normalize_risk_control_config,
    validate_risk_degree,
)
from src.backtest_scoring import (
    DEFAULT_SCORE_TRANSFORM,
    DEFAULT_WEIGHTING,
    compute_target_weights,
    max_affordable_trade_value,
    normalize_keep_top_n,
    normalize_min_score,
    normalize_score_transform,
    normalize_weighting_mode,
    select_topk_dropout_trades,
    snapshot_holdings,
    trade_cost,
    transform_score_matrix,
)
from src.label_utils import sanitize_label_series


DEFAULT_ACCOUNT = 100_000_000.0
DEFAULT_MIN_COST = 5.0
DEFAULT_RISK_DEGREE = 0.95
DEFAULT_SLIPPAGE = 0.0
DEFAULT_TRANSACTION_COST = 0.001


def _build_intraperiod_price_confirm_matrix_for_backtest(
    market_data: pd.DataFrame | None,
    pred_matrix: pd.DataFrame,
    intraperiod_exit_cfg: dict[str, float | int | str] | None,
) -> pd.DataFrame | None:
    if intraperiod_exit_cfg is None or "price_confirm_mode" not in intraperiod_exit_cfg:
        return None
    if market_data is None or "close" not in market_data.columns:
        raise ValueError("market_data with a 'close' column is required when intraperiod_exit.price_confirm is enabled")
    close_series = market_data["close"]
    if not isinstance(close_series.index, pd.MultiIndex):
        raise ValueError("market_data must use a MultiIndex of (datetime, instrument)")
    close_series = close_series.copy()
    close_series.index = close_series.index.set_names(["datetime", "instrument"])
    close_matrix = (
        close_series.astype(float)
        .unstack(level="instrument")
        .reindex(index=pred_matrix.index, columns=pred_matrix.columns)
    )
    return build_intraperiod_price_confirm_matrix(
        close_matrix,
        confirm_mode=str(intraperiod_exit_cfg["price_confirm_mode"]),
        ma_window=int(intraperiod_exit_cfg["price_confirm_ma_window"]),
    )


def _risk_control_requires_score_matrix(risk_control_cfg: dict[str, float | int | str]) -> bool:
    mode = str(risk_control_cfg.get("mode") or "fixed")
    signal_source = str(risk_control_cfg.get("signal_source") or "score_strength")
    return mode in {"signal_strength", "benchmark_ma_signal_strength"} and signal_source != "validation_metric"


def _build_instrument_group_series(
    instrument_groups: pd.Series | dict[str, str] | None,
    instruments: pd.Index,
) -> pd.Series | None:
    if instrument_groups is None:
        return None
    if isinstance(instrument_groups, pd.Series):
        instrument_group_series = instrument_groups.copy()
    else:
        instrument_group_series = pd.Series(instrument_groups, dtype=object)
    instrument_group_series.index = instrument_group_series.index.map(str)
    instrument_group_series = instrument_group_series.reindex(instruments.map(str))
    instrument_group_series.index = instruments
    fallback_groups = pd.Series(
        [f"__ungrouped__{symbol}" for symbol in instrument_group_series.index.astype(str)],
        index=instrument_group_series.index,
        dtype=object,
    )
    return instrument_group_series.fillna(fallback_groups).astype(str)


def _build_exit_score_matrix(
    pred_matrix: pd.DataFrame,
    *,
    intraperiod_exit_cfg: dict[str, float | int | str] | None,
    strategy_score_matrix: pd.DataFrame,
    zscore_clip: float,
) -> pd.DataFrame | None:
    if intraperiod_exit_cfg is None:
        return None
    source = str(intraperiod_exit_cfg["score_source"])
    if source == "raw":
        return pred_matrix
    if source == "transformed":
        return strategy_score_matrix
    if source == "rank_pct":
        return transform_score_matrix(pred_matrix, score_transform="rank_pct", zscore_clip=zscore_clip)
    if source == "zscore":
        return transform_score_matrix(pred_matrix, score_transform="zscore_clip", zscore_clip=zscore_clip)
    raise ValueError(
        f"Unsupported intraperiod exit score_source: {source}. Supported: {', '.join(SUPPORTED_EXIT_SCORE_SOURCES)}"
    )


def _loop_native_backtests(preds_list: list[pd.Series], **kwargs) -> list[pd.DataFrame]:
    return [run_native_backtest(preds=preds, **kwargs) for preds in preds_list]


def _matrix_to_dense_series(matrix: pd.DataFrame) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [matrix.index, matrix.columns],
        names=["datetime", "instrument"],
    )
    return pd.Series(matrix.to_numpy(dtype=float, copy=False).reshape(-1), index=index)


def _loop_native_backtests_with_batch_alignment(
    preds_list: list[pd.Series],
    labels: pd.Series,
    fallback_kwargs: dict[str, object],
) -> list[pd.DataFrame]:
    aligned = _align_prediction_matrices(preds_list, labels)
    if aligned is None:
        return _loop_native_backtests(preds_list, **fallback_kwargs)
    pred_matrices, label_matrix = aligned
    aligned_kwargs = dict(fallback_kwargs)
    aligned_kwargs["labels"] = _matrix_to_dense_series(label_matrix)
    aligned_preds = [_matrix_to_dense_series(matrix) for matrix in pred_matrices]
    return _loop_native_backtests(aligned_preds, **aligned_kwargs)


def _align_prediction_matrices(
    preds_list: list[pd.Series],
    labels: pd.Series,
) -> tuple[list[pd.DataFrame], pd.DataFrame] | None:
    if not preds_list:
        return [], pd.DataFrame()
    if not isinstance(labels.index, pd.MultiIndex) or "instrument" not in labels.index.names:
        return None

    common_idx = labels.index
    for preds in preds_list:
        if not isinstance(preds.index, pd.MultiIndex) or "instrument" not in preds.index.names:
            return None
        common_idx = common_idx.intersection(preds.index)
    if len(common_idx) == 0:
        raise ValueError("Native backtest received no overlapping prediction/label index.")

    aligned_labels = sanitize_label_series(labels.loc[common_idx]).sort_index()
    pred_matrices = [
        pd.to_numeric(preds.loc[common_idx], errors="coerce")
        .astype(float)
        .sort_index()
        .unstack(level="instrument")
        .sort_index()
        for preds in preds_list
    ]
    label_matrix = (
        aligned_labels.unstack(level="instrument")
        .reindex(index=pred_matrices[0].index, columns=pred_matrices[0].columns)
        .sort_index()
    )
    valid_dates = label_matrix.notna().any(axis=1)
    if not valid_dates.any():
        raise ValueError("Native backtest received no dates with any realized returns.")
    label_matrix = label_matrix.loc[valid_dates]
    pred_matrices = [
        matrix.reindex(index=label_matrix.index, columns=label_matrix.columns)
        for matrix in pred_matrices
    ]
    return pred_matrices, label_matrix


def run_native_backtest_batch(
    preds_list: list[pd.Series],
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
    max_industry_weight: float | None = None,
    desticky_signal_threshold: float | None = None,
    desticky_n_drop: int | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
    instrument_groups: pd.Series | dict[str, str] | None = None,
    benchmark_returns: pd.Series | None = None,
    market_data: pd.DataFrame | None = None,
    risk_control: dict[str, object] | None = None,
    risk_control_signal_values: pd.Series | dict[str, pd.Series] | None = None,
    intraperiod_exit: dict[str, object] | None = None,
    dynamic_risk: dict[str, object] | None = None,
) -> list[pd.DataFrame]:
    preds_items = [preds for preds in preds_list if preds is not None]
    if not preds_items:
        return []

    fallback_kwargs = {
        "labels": labels,
        "topk": topk,
        "n_drop": n_drop,
        "cost_buy": cost_buy,
        "cost_sell": cost_sell,
        "min_cost": min_cost,
        "account": account,
        "risk_degree": risk_degree,
        "slippage": slippage,
        "rebalance_freq": rebalance_freq,
        "weighting": weighting,
        "score_transform": score_transform,
        "score_zscore_clip": score_zscore_clip,
        "max_weight": max_weight,
        "max_industry_weight": max_industry_weight,
        "desticky_signal_threshold": desticky_signal_threshold,
        "desticky_n_drop": desticky_n_drop,
        "keep_top_n": keep_top_n,
        "min_score": min_score,
        "instrument_groups": instrument_groups,
        "benchmark_returns": benchmark_returns,
        "market_data": market_data,
        "risk_control": risk_control,
        "risk_control_signal_values": risk_control_signal_values,
        "intraperiod_exit": intraperiod_exit,
        "dynamic_risk": dynamic_risk,
    }
    return _loop_native_backtests_with_batch_alignment(preds_items, labels, fallback_kwargs)


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
    max_industry_weight: float | None = None,
    desticky_signal_threshold: float | None = None,
    desticky_n_drop: int | None = None,
    keep_top_n: int | None = None,
    min_score: float | None = None,
    instrument_groups: pd.Series | dict[str, str] | None = None,
    benchmark_returns: pd.Series | None = None,
    market_data: pd.DataFrame | None = None,
    risk_control: dict[str, object] | None = None,
    risk_control_signal_values: pd.Series | None = None,
    intraperiod_exit: dict[str, object] | None = None,
    dynamic_risk: dict[str, object] | None = None,
    return_trace: bool = False,
    trace_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_freq = max(1, int(rebalance_freq))
    topk = max(1, int(topk))
    n_drop = max(0, int(n_drop))
    weighting = normalize_weighting_mode(weighting)
    score_transform = normalize_score_transform(score_transform)
    keep_top_n = normalize_keep_top_n(keep_top_n, topk)
    min_score = normalize_min_score(min_score)
    risk_degree = validate_risk_degree(float(risk_degree), "risk_degree")
    score_zscore_clip = max(float(score_zscore_clip), 0.0)
    if max_industry_weight is not None:
        max_industry_weight = validate_risk_degree(float(max_industry_weight), "max_industry_weight")
    if desticky_n_drop is not None:
        desticky_n_drop = max(int(desticky_n_drop), int(n_drop))
        if desticky_n_drop >= topk:
            raise ValueError("desticky_n_drop must be smaller than topk")
    if desticky_signal_threshold is not None:
        desticky_signal_threshold = float(desticky_signal_threshold)
    intraperiod_exit_cfg = normalize_intraperiod_exit_config(intraperiod_exit)
    if risk_control is not None and dynamic_risk is not None:
        raise ValueError("Provide only one of risk_control or dynamic_risk")
    raw_risk_control = risk_control if risk_control is not None else dynamic_risk
    risk_control_cfg = normalize_risk_control_config(raw_risk_control, fallback_risk_degree=risk_degree)
    open_rate = float(cost_buy) + float(slippage)
    close_rate = float(cost_sell) + float(slippage)

    if preds.index.equals(labels.index) and preds.index.is_monotonic_increasing:
        aligned_preds = preds
        aligned_labels = labels
    else:
        common_idx = preds.index.intersection(labels.index)
        aligned_preds = preds.loc[common_idx].sort_index()
        aligned_labels = labels.loc[common_idx].sort_index()
    preds = pd.to_numeric(aligned_preds, errors="coerce").astype(float)
    labels = sanitize_label_series(aligned_labels)
    if preds.empty:
        raise ValueError("Native backtest received no overlapping prediction/label index.")

    pred_matrix = preds.unstack(level="instrument").sort_index()
    label_matrix = labels.unstack(level="instrument").reindex(pred_matrix.index).sort_index()
    valid_dates = label_matrix.notna().any(axis=1)
    pred_matrix = pred_matrix.loc[valid_dates]
    label_matrix = label_matrix.loc[valid_dates]
    if pred_matrix.empty:
        raise ValueError("Native backtest received no dates with any realized returns.")

    strategy_score_matrix = transform_score_matrix(
        pred_matrix,
        score_transform=score_transform,
        zscore_clip=score_zscore_clip,
    )
    risk_needs_strategy_scores = _risk_control_requires_score_matrix(risk_control_cfg)
    risk_score_matrix = (
        strategy_score_matrix
        if risk_needs_strategy_scores
        else pred_matrix.iloc[:, :0]
    )
    risk_schedule, risk_signal_schedule = build_risk_control_schedule(
        benchmark_returns,
        risk_score_matrix,
        risk_control=risk_control_cfg,
        fallback_risk_degree=risk_degree,
        topk=topk,
        min_score=min_score,
        external_signal_values=risk_control_signal_values,
    )
    intraperiod_price_confirm_matrix = _build_intraperiod_price_confirm_matrix_for_backtest(
        market_data,
        pred_matrix,
        intraperiod_exit_cfg,
    )

    trace_dates_norm = {pd.Timestamp(date) for date in trace_dates} if trace_dates is not None else None
    instruments = pred_matrix.columns
    instrument_to_col = {str(symbol): idx for idx, symbol in enumerate(instruments)}
    instrument_group_series = _build_instrument_group_series(instrument_groups, instruments)
    label_values = label_matrix.to_numpy(dtype=float, copy=False)
    bench_values = label_matrix.mean(axis=1, skipna=True).fillna(0.0).to_numpy(dtype=float, copy=False)

    intraperiod_score_matrix = _build_exit_score_matrix(
        pred_matrix,
        intraperiod_exit_cfg=intraperiod_exit_cfg,
        strategy_score_matrix=strategy_score_matrix,
        zscore_clip=score_zscore_clip,
    )
    intraperiod_remaining_steps = build_intraperiod_remaining_steps(pred_matrix.index, rebalance_freq=rebalance_freq)
    intraperiod_residual_matrix = None
    intraperiod_history: dict[int, dict[str, object]] | None = None
    if intraperiod_exit_cfg is not None:
        intraperiod_residual_matrix = build_intraperiod_residual_return_matrix(
            label_matrix,
            remaining_steps=intraperiod_remaining_steps,
        )
    if intraperiod_exit_cfg is not None and str(intraperiod_exit_cfg["mode"]) == "expected_return_threshold":
        intraperiod_history = {}
    strategy_score_values = strategy_score_matrix.to_numpy(dtype=float, copy=False)
    intraperiod_score_values = (
        intraperiod_score_matrix.to_numpy(dtype=float, copy=False)
        if intraperiod_score_matrix is not None
        else None
    )
    intraperiod_price_confirm_values = (
        intraperiod_price_confirm_matrix.to_numpy(dtype=bool, copy=False)
        if intraperiod_price_confirm_matrix is not None
        else None
    )
    intraperiod_residual_values = (
        intraperiod_residual_matrix.to_numpy(dtype=float, copy=False)
        if intraperiod_residual_matrix is not None
        else None
    )

    cash = float(account)
    holdings: dict[str, float] = {}
    records: list[dict[str, float | int | pd.Timestamp]] = []
    trace_records: list[dict[str, object]] = []

    for pos, date in enumerate(pred_matrix.index):
        cash_before = float(cash)
        holdings_before = snapshot_holdings(holdings)
        start_value = cash + sum(holdings.values())
        current_risk_degree = (
            float(risk_schedule.get(date, risk_control_cfg["risk_degree"]))
            if risk_schedule is not None
            else float(risk_control_cfg["risk_degree"])
        )
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
        intraperiod_exit_score_candidate_count = 0
        intraperiod_exit_price_confirm_required_count = 0
        intraperiod_exit_price_confirm_blocked_count = 0
        intraperiod_exit_price_confirm_bypassed_remaining_steps_count = 0
        intraperiod_exit_price_confirm_bypassed_force_exit_count = 0
        intraperiod_exit_remaining_steps = 0
        intraperiod_exit_signal_mean = np.nan
        intraperiod_exit_signal_min = np.nan
        intraperiod_exit_signal_values: dict[str, float] = {}
        intraperiod_exit_residual_mean = np.nan
        intraperiod_exit_residual_min = np.nan
        intraperiod_exit_residual_max = np.nan
        intraperiod_exit_residual_values: dict[str, float] = {}
        intraperiod_exit_saved_return = 0.0
        intraperiod_exit_missed_return = 0.0
        intraperiod_exit_beneficial_count = 0
        intraperiod_exit_harmful_count = 0
        intraperiod_exit_events: list[dict[str, float | int | str | None]] = []
        frozen_holdings = 0

        date_return_values = label_values[pos]
        locked_holdings = {
            stock
            for stock in holdings
            if (col_idx := instrument_to_col.get(str(stock))) is None or pd.isna(date_return_values[col_idx])
        }
        is_rebalance = (pos % rebalance_freq) == 0
        effective_n_drop = int(n_drop)
        desticky_active = False
        if (
            is_rebalance
            and desticky_n_drop is not None
            and desticky_signal_threshold is not None
            and pd.notna(current_risk_signal)
            and float(current_risk_signal) <= float(desticky_signal_threshold)
        ):
            effective_n_drop = int(desticky_n_drop)
            desticky_active = effective_n_drop > int(n_drop)

        if is_rebalance:
            strategy_scores_row = pd.Series(strategy_score_values[pos], index=instruments, dtype=float, copy=False)
            sell_list, buy_list = select_topk_dropout_trades(
                transformed_scores=strategy_scores_row,
                current_holdings=list(holdings.keys()),
                topk=topk,
                n_drop=effective_n_drop,
                locked_holdings=locked_holdings,
                keep_top_n=keep_top_n,
                min_score=min_score,
            )
            trade_sell_list: list[str] = []
            trade_buy_list: list[str] = []

            for stock in sell_list:
                position_value = holdings.pop(stock, 0.0)
                if position_value <= 0:
                    continue
                cost_value = trade_cost(position_value, close_rate, min_cost)
                cash += position_value - cost_value
                trade_cost_value += cost_value
                sell_value += position_value
                sell_count += 1
                trade_sell_list.append(stock)

            tradable_holdings = [stock for stock in holdings if stock not in locked_holdings]
            if min_score is None:
                eligible_current_holdings = tradable_holdings
            else:
                eligible_current_holdings = [
                    stock for stock in tradable_holdings if float(strategy_scores_row.get(stock, np.nan)) > min_score
                ]
            target_holdings = list(dict.fromkeys(eligible_current_holdings + list(buy_list)))
            target_weights = compute_target_weights(
                strategy_scores_row,
                target_holdings,
                weighting=weighting,
                max_weight=max_weight,
                group_labels=instrument_group_series,
                max_group_weight=max_industry_weight,
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
                cost_value = trade_cost(trade_value, close_rate, min_cost)
                cash += trade_value - cost_value
                holdings[stock] = target_value
                if holdings[stock] <= 1e-12:
                    holdings.pop(stock, None)
                trade_cost_value += cost_value
                sell_value += trade_value
                sell_count += 1
                if stock not in trade_sell_list:
                    trade_sell_list.append(stock)

            for stock in target_values.sort_values(ascending=False, kind="stable").index:
                target_value = float(target_values.get(stock, 0.0))
                current_value = float(holdings.get(stock, 0.0))
                deficit_value = target_value - current_value
                if deficit_value <= 1e-12:
                    continue
                max_trade_value = max_affordable_trade_value(cash, open_rate, min_cost)
                trade_value = min(deficit_value, max_trade_value)
                if trade_value <= 0:
                    continue
                cost_value = trade_cost(trade_value, open_rate, min_cost)
                if trade_value + cost_value > cash:
                    trade_value = max_affordable_trade_value(cash, open_rate, min_cost)
                    cost_value = trade_cost(trade_value, open_rate, min_cost) if trade_value > 0 else 0.0
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
                intraperiod_exit_remaining_steps = int(intraperiod_remaining_steps.iloc[pos])
                signal_values = (
                    pd.Series(intraperiod_score_values[pos], index=instruments, dtype=float, copy=False)
                    if intraperiod_score_values is not None
                    else pd.Series(dtype=float)
                )
                threshold = float(intraperiod_exit_cfg["threshold"])
                exit_mode = str(intraperiod_exit_cfg["mode"])
                if exit_mode == "expected_return_threshold":
                    signal_values = estimate_intraperiod_expected_returns(
                        signal_values,
                        history_entry=None
                        if intraperiod_history is None
                        else intraperiod_history.get(intraperiod_exit_remaining_steps),
                        n_bins=int(intraperiod_exit_cfg.get("n_bins", 20)),
                        min_history=int(intraperiod_exit_cfg.get("min_history", 200)),
                    )
                held_signal_values = signal_values.reindex(list(holdings.keys())).dropna()
                if not held_signal_values.empty:
                    intraperiod_exit_signal_mean = float(held_signal_values.mean())
                    intraperiod_exit_signal_min = float(held_signal_values.min())
                    intraperiod_exit_signal_values = {stock: float(value) for stock, value in held_signal_values.items()}
                exit_denom = start_value if start_value > 0 else 1.0
                intraperiod_exit_residual_samples: list[float] = []
                for stock in list(holdings.keys()):
                    if stock in locked_holdings:
                        continue
                    score_value = signal_values.get(stock, np.nan)
                    if pd.isna(score_value) or float(score_value) > threshold:
                        continue
                    intraperiod_exit_score_candidate_count += 1
                    price_confirm_passed = True
                    price_confirm_required = False
                    price_confirm_bypass_reason: str | None = None
                    if intraperiod_price_confirm_values is not None:
                        force_exit_threshold = intraperiod_exit_cfg.get("price_confirm_force_exit_threshold")
                        if force_exit_threshold is not None and float(score_value) <= float(force_exit_threshold):
                            price_confirm_bypass_reason = "force_exit_threshold"
                            intraperiod_exit_price_confirm_bypassed_force_exit_count += 1
                        elif intraperiod_exit_remaining_steps < int(
                            intraperiod_exit_cfg.get("price_confirm_min_remaining_steps", 0)
                        ):
                            price_confirm_bypass_reason = "remaining_steps"
                            intraperiod_exit_price_confirm_bypassed_remaining_steps_count += 1
                        else:
                            price_confirm_required = True
                            intraperiod_exit_price_confirm_required_count += 1
                            col_idx = instrument_to_col.get(str(stock))
                            price_confirm_passed = bool(
                                False
                                if col_idx is None
                                else intraperiod_price_confirm_values[pos, col_idx]
                            )
                            if not price_confirm_passed:
                                intraperiod_exit_price_confirm_blocked_count += 1
                                continue
                    position_value = holdings.pop(stock, 0.0)
                    if position_value <= 0:
                        continue
                    col_idx = instrument_to_col.get(str(stock))
                    residual_value = (
                        np.nan
                        if intraperiod_residual_values is None or col_idx is None
                        else intraperiod_residual_values[pos, col_idx]
                    )
                    if pd.notna(residual_value):
                        residual_value = float(residual_value)
                        intraperiod_exit_residual_samples.append(residual_value)
                        intraperiod_exit_residual_values[stock] = residual_value
                        saved_return = max(-residual_value, 0.0) * float(position_value) / exit_denom
                        missed_return = max(residual_value, 0.0) * float(position_value) / exit_denom
                        intraperiod_exit_saved_return += saved_return
                        intraperiod_exit_missed_return += missed_return
                        intraperiod_exit_beneficial_count += int(residual_value < 0.0)
                        intraperiod_exit_harmful_count += int(residual_value > 0.0)
                    else:
                        saved_return = 0.0
                        missed_return = 0.0
                    cost_value = trade_cost(position_value, close_rate, min_cost)
                    cash += position_value - cost_value
                    trade_cost_value += cost_value
                    sell_value += position_value
                    sell_count += 1
                    intraperiod_exit_count += 1
                    sell_list.append(stock)
                    trade_sell_list.append(stock)
                    intraperiod_exit_events.append(
                        {
                            "stock": str(stock),
                            "score_value": None if pd.isna(score_value) else float(score_value),
                            "residual_return_if_held": None if pd.isna(residual_value) else float(residual_value),
                            "position_value": float(position_value),
                            "saved_return_contribution": float(saved_return),
                            "missed_return_contribution": float(missed_return),
                            "remaining_steps": int(intraperiod_exit_remaining_steps),
                            "price_confirm_required": bool(price_confirm_required),
                            "price_confirm_passed": bool(price_confirm_passed),
                            "price_confirm_bypass_reason": price_confirm_bypass_reason,
                            "price_confirm_signal_timing": None
                            if intraperiod_exit_cfg is None
                            else intraperiod_exit_cfg.get("price_confirm_signal_timing"),
                            "price_confirm_execution_timing": None
                            if intraperiod_exit_cfg is None
                            else intraperiod_exit_cfg.get("price_confirm_execution_timing"),
                        }
                    )
                if intraperiod_exit_residual_samples:
                    residual_array = np.asarray(intraperiod_exit_residual_samples, dtype=float)
                    intraperiod_exit_residual_mean = float(residual_array.mean())
                    intraperiod_exit_residual_min = float(residual_array.min())
                    intraperiod_exit_residual_max = float(residual_array.max())

        gross_pnl = 0.0
        for stock, position_value in list(holdings.items()):
            col_idx = instrument_to_col.get(str(stock))
            stock_ret = np.nan if col_idx is None else date_return_values[col_idx]
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
        turnover = (buy_value + sell_value) / (2.0 * denom)
        cost_return = trade_cost_value / denom

        records.append(
            {
                "datetime": date,
                "gross_return": gross_return,
                "net_return": net_return,
                "turnover": turnover,
                "cost": cost_return,
                "bench": float(bench_values[pos]),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "intraperiod_exit_count": intraperiod_exit_count,
                "intraperiod_exit_score_candidate_count": intraperiod_exit_score_candidate_count,
                "intraperiod_exit_price_confirm_required_count": intraperiod_exit_price_confirm_required_count,
                "intraperiod_exit_price_confirm_blocked_count": intraperiod_exit_price_confirm_blocked_count,
                "intraperiod_exit_price_confirm_bypassed_remaining_steps_count": intraperiod_exit_price_confirm_bypassed_remaining_steps_count,
                "intraperiod_exit_price_confirm_bypassed_force_exit_count": intraperiod_exit_price_confirm_bypassed_force_exit_count,
                "intraperiod_exit_price_confirm_signal_timing": None
                if intraperiod_exit_cfg is None
                else intraperiod_exit_cfg.get("price_confirm_signal_timing"),
                "intraperiod_exit_price_confirm_execution_timing": None
                if intraperiod_exit_cfg is None
                else intraperiod_exit_cfg.get("price_confirm_execution_timing"),
                "intraperiod_exit_remaining_steps": intraperiod_exit_remaining_steps,
                "intraperiod_exit_signal_mean": intraperiod_exit_signal_mean,
                "intraperiod_exit_signal_min": intraperiod_exit_signal_min,
                "intraperiod_exit_residual_mean": intraperiod_exit_residual_mean,
                "intraperiod_exit_residual_min": intraperiod_exit_residual_min,
                "intraperiod_exit_residual_max": intraperiod_exit_residual_max,
                "intraperiod_exit_saved_return": intraperiod_exit_saved_return,
                "intraperiod_exit_missed_return": intraperiod_exit_missed_return,
                "intraperiod_exit_beneficial_count": intraperiod_exit_beneficial_count,
                "intraperiod_exit_harmful_count": intraperiod_exit_harmful_count,
                "holdings": len(holdings),
                "frozen_holdings": frozen_holdings,
                "account_value": end_value,
                "risk_degree": current_risk_degree,
                "risk_control_signal": current_risk_signal,
                "effective_n_drop": int(effective_n_drop),
                "desticky_active": bool(desticky_active),
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
                    "holdings_after": snapshot_holdings(holdings),
                    "locked_holdings": sorted(locked_holdings),
                    "sell_list": list(sell_list),
                    "buy_list": list(buy_list),
                    "trade_sell_list": list(trade_sell_list),
                    "trade_buy_list": list(trade_buy_list),
                    "weighting": weighting,
                    "risk_degree": current_risk_degree,
                    "risk_control_mode": risk_control_cfg["mode"],
                    "risk_control_signal": current_risk_signal,
                    "effective_n_drop": int(effective_n_drop),
                    "desticky_active": bool(desticky_active),
                    "intraperiod_exit_mode": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg["mode"],
                    "intraperiod_exit_score_source": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg["score_source"],
                    "intraperiod_exit_threshold": None if intraperiod_exit_cfg is None else float(intraperiod_exit_cfg["threshold"]),
                    "intraperiod_exit_price_confirm_mode": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg.get("price_confirm_mode"),
                    "intraperiod_exit_price_confirm_ma_window": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg.get("price_confirm_ma_window"),
                    "intraperiod_exit_price_confirm_min_remaining_steps": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg.get("price_confirm_min_remaining_steps"),
                    "intraperiod_exit_price_confirm_force_exit_threshold": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg.get("price_confirm_force_exit_threshold"),
                    "intraperiod_exit_price_confirm_signal_timing": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg.get("price_confirm_signal_timing"),
                    "intraperiod_exit_price_confirm_execution_timing": None if intraperiod_exit_cfg is None else intraperiod_exit_cfg.get("price_confirm_execution_timing"),
                    "intraperiod_exit_count": int(intraperiod_exit_count),
                    "intraperiod_exit_score_candidate_count": int(intraperiod_exit_score_candidate_count),
                    "intraperiod_exit_price_confirm_required_count": int(intraperiod_exit_price_confirm_required_count),
                    "intraperiod_exit_price_confirm_blocked_count": int(intraperiod_exit_price_confirm_blocked_count),
                    "intraperiod_exit_price_confirm_bypassed_remaining_steps_count": int(
                        intraperiod_exit_price_confirm_bypassed_remaining_steps_count
                    ),
                    "intraperiod_exit_price_confirm_bypassed_force_exit_count": int(
                        intraperiod_exit_price_confirm_bypassed_force_exit_count
                    ),
                    "intraperiod_exit_remaining_steps": int(intraperiod_exit_remaining_steps),
                    "intraperiod_exit_signal_mean": float(intraperiod_exit_signal_mean) if pd.notna(intraperiod_exit_signal_mean) else np.nan,
                    "intraperiod_exit_signal_min": float(intraperiod_exit_signal_min) if pd.notna(intraperiod_exit_signal_min) else np.nan,
                    "intraperiod_exit_residual_mean": float(intraperiod_exit_residual_mean) if pd.notna(intraperiod_exit_residual_mean) else np.nan,
                    "intraperiod_exit_residual_min": float(intraperiod_exit_residual_min) if pd.notna(intraperiod_exit_residual_min) else np.nan,
                    "intraperiod_exit_residual_max": float(intraperiod_exit_residual_max) if pd.notna(intraperiod_exit_residual_max) else np.nan,
                    "intraperiod_exit_saved_return": float(intraperiod_exit_saved_return),
                    "intraperiod_exit_missed_return": float(intraperiod_exit_missed_return),
                    "intraperiod_exit_beneficial_count": int(intraperiod_exit_beneficial_count),
                    "intraperiod_exit_harmful_count": int(intraperiod_exit_harmful_count),
                    "intraperiod_exit_signal_values": dict(intraperiod_exit_signal_values),
                    "intraperiod_exit_residual_values": dict(intraperiod_exit_residual_values),
                    "intraperiod_exit_events": list(intraperiod_exit_events),
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

        if intraperiod_history is not None and intraperiod_score_matrix is not None and intraperiod_residual_matrix is not None:
            remaining_steps = int(intraperiod_remaining_steps.iloc[pos])
            append_intraperiod_history(
                intraperiod_history,
                remaining_steps=remaining_steps,
                score_row=pd.Series(intraperiod_score_values[pos], index=instruments, dtype=float, copy=False),
                residual_row=pd.Series(intraperiod_residual_values[pos], index=instruments, dtype=float, copy=False),
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
