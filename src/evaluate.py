"""Evaluation metrics and visualization for quantitative strategies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qlib.contrib.evaluate import risk_analysis


def compute_signal_metrics(predictions: pd.Series, labels: pd.Series) -> dict:
    """Compute IC and ICIR metrics between predictions and actual returns.

    Parameters
    ----------
    predictions : pd.Series
        Model predicted scores, MultiIndex (datetime, instrument).
    labels : pd.Series
        Actual future returns, same index as predictions.

    Returns
    -------
    dict with keys: IC_mean, IC_std, ICIR, Rank_IC_mean, Rank_IC_std, Rank_ICIR.
    """
    df = pd.DataFrame({"pred": predictions, "label": labels}).dropna()

    # Group by date, compute daily IC (Pearson correlation of ranks)
    daily_ic = df.groupby(level=0).apply(
        lambda x: x["pred"].corr(x["label"]), include_groups=False
    )
    daily_rank_ic = df.groupby(level=0).apply(
        lambda x: x["pred"].corr(x["label"], method="spearman"), include_groups=False
    )

    metrics = {
        "IC_mean": daily_ic.mean(),
        "IC_std": daily_ic.std(),
        "ICIR": daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0,
        "Rank_IC_mean": daily_rank_ic.mean(),
        "Rank_IC_std": daily_rank_ic.std(),
        "Rank_ICIR": daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else 0,
    }
    return metrics, daily_ic


def compute_portfolio_metrics(portfolio_metric) -> dict:
    """Extract portfolio-level metrics from backtest results using Qlib's risk_analysis."""
    report, indicator = portfolio_metric
    analysis = risk_analysis(report["return"])

    result = {}
    for key in analysis.index:
        result[key] = analysis.loc[key].to_dict()
    return result, report


def plot_cumulative_return(report: pd.DataFrame, save_path: str = None):
    """Plot cumulative return curve: strategy vs benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    cum_return = (1 + report["return"]).cumprod()
    cum_bench = (1 + report["bench"]).cumprod()

    ax.plot(cum_return.index, cum_return.values, label="Strategy", linewidth=1.5)
    ax.plot(cum_bench.index, cum_bench.values, label="Benchmark (CSI300)", linewidth=1.5, alpha=0.7)
    ax.set_title("Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_ic_series(daily_ic: pd.Series, save_path: str = None):
    """Plot daily IC time series with rolling mean."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(daily_ic.index, daily_ic.values, alpha=0.3, width=1, label="Daily IC")
    rolling_ic = daily_ic.rolling(20).mean()
    ax.plot(rolling_ic.index, rolling_ic.values, color="red", linewidth=1.5, label="Rolling 20-day IC")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title(f"IC Series (mean={daily_ic.mean():.4f}, ICIR={daily_ic.mean()/daily_ic.std():.4f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_drawdown(report: pd.DataFrame, save_path: str = None):
    """Plot drawdown curve."""
    fig, ax = plt.subplots(figsize=(12, 3))

    cum_return = (1 + report["return"]).cumprod()
    running_max = cum_return.cummax()
    drawdown = (cum_return - running_max) / running_max

    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax.set_title(f"Drawdown (max={drawdown.min():.2%})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


def print_metrics(signal_metrics: dict, portfolio_metrics: dict = None):
    """Pretty-print all evaluation metrics."""
    print("\n" + "=" * 50)
    print("Signal Quality Metrics")
    print("=" * 50)
    for k, v in signal_metrics.items():
        print(f"  {k:15s}: {v:+.4f}")

    if portfolio_metrics:
        print("\n" + "=" * 50)
        print("Portfolio Metrics")
        print("=" * 50)
        for category, values in portfolio_metrics.items():
            print(f"\n  [{category}]")
            for k, v in values.items():
                if isinstance(v, float):
                    print(f"    {k:20s}: {v:+.4f}")
                else:
                    print(f"    {k:20s}: {v}")
    print("=" * 50)
