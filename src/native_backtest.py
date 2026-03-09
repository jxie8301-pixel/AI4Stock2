"""Fast Vectorized Backtester for Top-K Long-only Strategies."""

import pandas as pd
import numpy as np
from typing import Tuple

def run_native_backtest(
    preds: pd.Series, 
    labels: pd.Series, 
    topk: int = 30, 
    cost_buy: float = 0.0003, 
    cost_sell: float = 0.0013,
    rebalance_freq: int = 1
) -> pd.DataFrame:
    """
    Perform a fast vectorized backtest on a Top-K long-only strategy.
    
    Parameters
    ----------
    preds : pd.Series
        Model predictions with MultiIndex (datetime, instrument).
    labels : pd.Series
        True returns (T+1 open to T+2 open) with same MultiIndex.
    topk : int
        Number of stocks to hold each day.
    cost_buy : float
        Transaction cost for buying (e.g., 0.03%).
    cost_sell : float
        Transaction cost for selling (e.g., 0.13% including stamp duty).
    rebalance_freq : int
        How often to rebalance in trading days. 1 = daily, 5 = weekly.
        
    Returns
    -------
    pd.DataFrame
        Daily report containing returns, turnover, and cumulative metrics.
    """
    print(f"[*] Starting Native Vectorized Backtest (Top-{topk}, Rebalance: {rebalance_freq} days)...")
    
    # Ensure indices are aligned
    common_idx = preds.index.intersection(labels.index)
    preds = preds.loc[common_idx]
    labels = labels.loc[common_idx]

    # 1. Rank predictions
    # We unstack to get a (date x instrument) matrix for easier time-based filtering
    preds_matrix = preds.unstack(level='instrument')
    
    # Apply rebalance frequency: Only keep predictions on rebalance days
    # We take every Nth row. Other days get NaN predictions.
    rebalance_dates = preds_matrix.index[::rebalance_freq]
    
    # Create a mask for rebalance days
    is_rebalance_day = preds_matrix.index.isin(rebalance_dates)
    
    # Rank only on rebalance days, then forward fill the ranks to keep the same portfolio
    ranks_matrix = preds_matrix.where(is_rebalance_day).rank(axis=1, ascending=False, method='first').ffill()
    
    # Re-stack to Series
    ranks = ranks_matrix.stack()
    
    # 2. Identify positions (Long-only Top-K)
    # Binary mask: 1 if in top-k, 0 otherwise
    pos_mask = (ranks <= topk).astype(float)
    
    # Equal weight: each stock gets 1/K of the capital
    weights = pos_mask / topk
    
    # 3. Calculate Daily Gross Returns
    # Matrix multiplication logic: sum(weight * return) per day
    daily_gross_returns = (weights * labels).groupby(level='datetime').sum()
    
    # 4. Calculate Daily Turnover
    # We pivot to (date x instrument) matrix to calculate position diffs efficiently
    # Fillna(0) ensures we account for stocks entering/leaving the universe
    pos_matrix = weights.unstack(level='instrument').fillna(0)
    
    # Turnover is the sum of absolute changes in weights across all assets
    # For long-only with constant total weight 1.0, 
    # Turnover = 0.5 * sum(|w_t - w_{t-1}|) reflects the fraction of portfolio traded
    daily_pos_diff = pos_matrix.diff().abs().sum(axis=1)
    
    # Handle the first day (initial purchase)
    daily_pos_diff.iloc[0] = pos_matrix.iloc[0].sum()
    
    turnover = daily_pos_diff / 2.0
    
    # 5. Calculate Transaction Costs
    transaction_costs = turnover * (cost_buy + cost_sell)
    
    # 6. Calculate Net Returns
    daily_net_returns = daily_gross_returns - transaction_costs
    
    # Calculate a simple equal-weight benchmark return (average of all stocks that day)
    daily_bench_returns = labels.groupby(level='datetime').mean()
    
    # 7. Aggregate Results
    report = pd.DataFrame({
        'gross_return': daily_gross_returns,
        'net_return': daily_net_returns,
        'turnover': turnover,
        'cost': transaction_costs,
        'bench': daily_bench_returns  # Added benchmark for compatibility with Qlib evaluate functions
    })
    
    # Cumulative stats
    report['cum_gross_return'] = (1 + report['gross_return']).cumprod()
    report['cum_net_return'] = (1 + report['net_return']).cumprod()
    
    # Calculate key metrics
    n_days = len(report)
    ann_factor = 242 # Average trading days in China
    
    ann_ret = (report['net_return'].mean() * ann_factor)
    ann_vol = (report['net_return'].std() * np.sqrt(ann_factor))
    sharpe = ann_ret / (ann_vol + 1e-8)
    
    # Max Drawdown
    cum_net = report['cum_net_return']
    max_drawdown = (cum_net / cum_net.cummax() - 1).min()
    
    print(f"\n[Backtest Results]")
    print(f"Annualized Return: {ann_ret:.2%}")
    print(f"Annualized Vol  : {ann_vol:.2%}")
    print(f"Sharpe Ratio    : {sharpe:.4f}")
    print(f"Max Drawdown    : {max_drawdown:.2%}")
    print(f"Avg Daily Turnover: {turnover.mean():.2%}")
    
    return report
