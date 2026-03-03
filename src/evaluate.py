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
    """Extract portfolio-level metrics and apply sanity checks."""
    report, indicator = portfolio_metric
    
    # Sanity Check: Clip extreme daily returns (e.g. > 50% or < -50% in one day) 
    # which are usually data errors in non-leveraged portfolios.
    report["return"] = report["return"].clip(-0.2, 0.5) 
    
    analysis = risk_analysis(report["return"])

    result = {}
    for key in analysis.index:
        result[key] = analysis.loc[key].to_dict()
    
    # Add monthly returns calculation
    report.index = pd.to_datetime(report.index)
    monthly_ret = report["return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    result["monthly_return"] = monthly_ret.to_dict()
    
    return result, report


def plot_monthly_heatmap(report: pd.DataFrame, save_path: str = None):
    """Plot a heatmap of monthly returns (Year vs Month)."""
    import seaborn as sns
    
    monthly_ret = report["return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df_monthly = monthly_ret.to_frame(name="ret")
    df_monthly["year"] = df_monthly.index.year
    df_monthly["month"] = df_monthly.index.month
    
    pivot_table = df_monthly.pivot(index="year", columns="month", values="ret")
    
    fig, ax = plt.subplots(figsize=(12, min(len(pivot_table) * 0.8 + 2, 8)))
    sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Monthly Returns Heatmap")
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)


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


def save_monthly_report(report: pd.DataFrame, save_path: str = None):
    """Save monthly returns to a CSV file."""
    monthly_ret = report["return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df_monthly = monthly_ret.to_frame(name="monthly_return")
    df_monthly.index.name = "date"
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_monthly.to_csv(save_path)
        print(f"Monthly report saved: {save_path}")
    return df_monthly


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
            if category == "monthly_return":
                continue # Print table separately
            print(f"\n  [{category}]")
            for k, v in values.items():
                if isinstance(v, float):
                    print(f"    {k:20s}: {v:+.4f}")
                else:
                    print(f"    {k:20s}: {v}")
        
        if "monthly_return" in portfolio_metrics:
            print("\n  [Monthly Returns Table]")
            m_ret_dict = portfolio_metrics["monthly_return"]
            # Convert dict back to series for better formatting
            m_rets = pd.Series(m_ret_dict)
            m_rets.index = pd.to_datetime(m_rets.index)
            
            print(f"    {'Month':<15} | {'Return':>10}")
            print(f"    {'-'*15}-|-{'-'*10}")
            for date, ret in m_rets.items():
                print(f"    {date.strftime('%Y-%m'):<15} | {ret:>+10.2%}")
            
            print(f"\n  [Monthly Returns Summary]")
            m_values = list(m_ret_dict.values())
            print(f"    Max Monthly Return   : {max(m_values):+.2%}")
            print(f"    Min Monthly Return   : {min(m_values):+.2%}")
            print(f"    Positive Months      : {sum(1 for r in m_values if r > 0)} / {len(m_values)}")
            
    print("=" * 50)
