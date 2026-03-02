"""TopK dropout strategy for portfolio construction."""

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


def build_topk_strategy(
    topk: int = 30,
    n_drop: int = 5,
) -> dict:
    """Return config dict for TopkDropoutStrategy.

    The strategy selects top-k stocks by predicted score each rebalancing day,
    and limits turnover by replacing at most n_drop stocks per period.

    Parameters
    ----------
    topk : int
        Number of stocks to hold.
    n_drop : int
        Maximum number of stocks to replace each rebalancing day.

    Returns
    -------
    dict
        Strategy config dict usable by qlib.contrib.evaluate.backtest_daily.
    """
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "topk": topk,
            "n_drop": n_drop,
            "signal": None,  # will be set by backtest
        },
    }
    print(f"Strategy config: TopkDropout(topk={topk}, n_drop={n_drop})")
    return strategy_config
