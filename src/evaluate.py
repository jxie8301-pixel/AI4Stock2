"""Evaluation metrics and visualization for quantitative strategies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_cross_sectional_corr(
    pred: pd.Series,
    label: pd.Series,
    method: str = "pearson",
) -> float:
    """Return NaN instead of warning when a cross-section has no variance."""
    frame = pd.DataFrame({"pred": pred, "label": label}).dropna()
    if len(frame) < 2:
        return np.nan
    if frame["pred"].nunique(dropna=True) < 2:
        return np.nan
    if frame["label"].nunique(dropna=True) < 2:
        return np.nan
    return float(frame["pred"].corr(frame["label"], method=method))


def build_cross_section_benchmark(labels: pd.Series) -> pd.Series:
    """Compute daily cross-sectional mean return as a common reference series."""
    aligned_labels = labels.dropna()
    if aligned_labels.empty:
        return pd.Series(dtype=float)
    return aligned_labels.groupby(level=0).mean().sort_index()


def align_prediction_label_pairs(
    predictions: pd.Series,
    labels: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align prediction/label pairs on the same MultiIndex and drop invalid rows."""
    frame = pd.DataFrame({"pred": predictions, "label": labels}).dropna()
    if frame.empty:
        empty_index = predictions.index[:0]
        return (
            pd.Series(dtype=float, index=empty_index, name=getattr(predictions, "name", None)),
            pd.Series(dtype=float, index=empty_index, name=getattr(labels, "name", None)),
        )
    return frame["pred"].sort_index(), frame["label"].sort_index()


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
    aligned_preds, aligned_labels = align_prediction_label_pairs(predictions, labels)
    df = pd.DataFrame({"pred": aligned_preds, "label": aligned_labels})
    if df.empty:
        empty_daily_ic = pd.Series(dtype=float)
        return {
            "IC_mean": 0.0,
            "IC_std": 0.0,
            "ICIR": 0.0,
            "Rank_IC_mean": 0.0,
            "Rank_IC_std": 0.0,
            "Rank_ICIR": 0.0,
        }, empty_daily_ic

    # Group by date, compute daily IC (Pearson correlation of ranks)
    daily_ic = df.groupby(level=0).apply(
        lambda x: safe_cross_sectional_corr(x["pred"], x["label"], method="pearson"),
        include_groups=False,
    )
    daily_rank_ic = df.groupby(level=0).apply(
        lambda x: safe_cross_sectional_corr(x["pred"], x["label"], method="spearman"),
        include_groups=False,
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
    """Extract portfolio metrics with a consistent net-return interpretation.

    Qlib's daily report stores ``return`` before trading cost, while the native
    engine already reports net return. Normalize both to net return here so
    downstream metrics and plots operate on the same semantic series.
    """
    report, _ = portfolio_metric
    report = report.copy()

    # Qlib report columns: account/value/cash/... and ``return`` is gross of cost.
    is_qlib_style_report = {
        "account",
        "cash",
        "value",
        "cost",
        "turnover",
    }.issubset(report.columns) and "account_value" not in report.columns
    if is_qlib_style_report:
        report["gross_return"] = report["return"].astype(float)
        report["return"] = report["gross_return"] - report["cost"].astype(float)

    returns = report["return"].astype(float)
    ann_factor = 242  # A-share average trading days per year
    
    # Calculate native metrics
    mean_ret = returns.mean()
    std_ret = returns.std()
    ann_ret = mean_ret * ann_factor
    
    # Information Ratio (equivalent to Sharpe here since risk-free rate is often 0 in Qlib default)
    info_ratio = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret > 0 else 0.0
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    max_drawdown = ((cum_returns / cum_returns.cummax()) - 1.0).min()
    
    # Mimic Qlib's risk_analysis return structure
    result = {
        "mean": {"risk": mean_ret},
        "std": {"risk": std_ret},
        "annualized_return": {"risk": ann_ret},
        "information_ratio": {"risk": info_ratio},
        "max_drawdown": {"risk": max_drawdown}
    }
    
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
    ax.plot(cum_bench.index, cum_bench.values, label="Benchmark (Cross-Section Mean)", linewidth=1.5, alpha=0.7)
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
    daily_ic_std = daily_ic.std()
    icir = daily_ic.mean() / daily_ic_std if daily_ic_std and daily_ic_std > 0 else 0.0
    ax.set_title(f"IC Series (mean={daily_ic.mean():.4f}, ICIR={icir:.4f})")
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
