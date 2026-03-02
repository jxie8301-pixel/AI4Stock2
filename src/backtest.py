"""Backtest wrapper using Qlib's built-in backtest engine."""

import pandas as pd
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


def run_backtest(
    predictions: pd.Series,
    topk: int = 30,
    n_drop: int = 5,
    cost_buy: float = 0.0003,
    cost_sell: float = 0.0013,
    benchmark: str = "SH000300",
) -> tuple[pd.DataFrame, dict]:
    """Run daily backtest on model predictions.

    Parameters
    ----------
    predictions : pd.Series
        Model prediction scores with MultiIndex (datetime, instrument).
    topk : int
        Number of stocks to hold in portfolio.
    n_drop : int
        Max stocks to replace per rebalancing.
    cost_buy : float
        Buy transaction cost rate (commission).
    cost_sell : float
        Sell transaction cost rate (commission + stamp duty).
    benchmark : str
        Benchmark index code for comparison.

    Returns
    -------
    portfolio_metric : pd.DataFrame
        Daily portfolio metrics (return, turnover, etc.).
    indicator : dict
        Summary indicators.
    """
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "signal": predictions,
            "topk": topk,
            "n_drop": n_drop,
        },
    }

    backtest_config = {
        "start_time": predictions.index.get_level_values(0).min(),
        "end_time": predictions.index.get_level_values(0).max(),
        "account": 100_000_000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": cost_buy,
            "close_cost": cost_sell,
            "min_cost": 5,
        },
    }

    portfolio_metric = backtest_daily(
        strategy=strategy_config,
        **backtest_config,
    )

    print(f"Backtest complete: {backtest_config['start_time']} ~ {backtest_config['end_time']}")
    return portfolio_metric
