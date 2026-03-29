"""Project-owned backtest wrapper with a legacy-compatible return shape."""

import pandas as pd

from src.native_backtest import (
    DEFAULT_ACCOUNT,
    DEFAULT_MIN_COST,
    DEFAULT_RISK_DEGREE,
    DEFAULT_SLIPPAGE,
    DEFAULT_TRANSACTION_COST,
    run_native_backtest,
)


def run_backtest(
    predictions: pd.Series,
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
    return_trace: bool = False,
    trace_dates: set[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, None] | tuple[tuple[pd.DataFrame, None], pd.DataFrame]:
    """Run the project-owned daily backtest on model predictions.

    Parameters
    ----------
    predictions : pd.Series
        Model prediction scores with MultiIndex (datetime, instrument).
    labels : pd.Series
        Realized open-to-open returns with the same MultiIndex.
    topk : int
        Number of stocks to hold in portfolio.
    n_drop : int
        Max stocks to replace per rebalancing.
    cost_buy : float
        Buy transaction cost rate (commission).
    cost_sell : float
        Sell transaction cost rate (commission + stamp duty).
    Returns
    -------
    tuple[pd.DataFrame, None]
        Legacy-compatible ``(report, indicator)`` tuple expected by downstream code.
    """
    native_result = run_native_backtest(
        preds=predictions,
        labels=labels,
        topk=topk,
        n_drop=n_drop,
        cost_buy=cost_buy,
        cost_sell=cost_sell,
        min_cost=min_cost,
        account=account,
        risk_degree=risk_degree,
        slippage=slippage,
        rebalance_freq=rebalance_freq,
        return_trace=return_trace,
        trace_dates=trace_dates,
    )
    if return_trace:
        report, trace_df = native_result
    else:
        report = native_result
    report = report.rename(columns={"net_return": "return"})
    print(
        "Backtest complete: "
        f"{report.index.min()} ~ {report.index.max()} (native engine)"
    )
    if not return_trace:
        return report, None
    return (report, None), trace_df
