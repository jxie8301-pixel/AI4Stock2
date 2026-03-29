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
            "IC_win_rate": 0.0,
            "Rank_IC_mean": 0.0,
            "Rank_IC_std": 0.0,
            "Rank_ICIR": 0.0,
            "Rank_IC_win_rate": 0.0,
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

    daily_ic_valid = daily_ic.dropna()
    daily_rank_ic_valid = daily_rank_ic.dropna()
    metrics = {
        "IC_mean": daily_ic.mean(),
        "IC_std": daily_ic.std(),
        "ICIR": daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0,
        "IC_win_rate": float((daily_ic_valid > 0).mean()) if not daily_ic_valid.empty else 0.0,
        "Rank_IC_mean": daily_rank_ic.mean(),
        "Rank_IC_std": daily_rank_ic.std(),
        "Rank_ICIR": daily_rank_ic.mean() / daily_rank_ic.std() if daily_rank_ic.std() > 0 else 0,
        "Rank_IC_win_rate": float((daily_rank_ic_valid > 0).mean()) if not daily_rank_ic_valid.empty else 0.0,
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
    valid_returns = returns.dropna()
    ann_factor = 242  # A-share average trading days per year
    
    # Calculate native metrics
    mean_ret = valid_returns.mean() if not valid_returns.empty else 0.0
    std_ret = valid_returns.std() if not valid_returns.empty else 0.0
    if pd.isna(std_ret):
        std_ret = 0.0
    ann_ret = mean_ret * ann_factor
    ann_vol = std_ret * np.sqrt(ann_factor)
    
    # Information Ratio (equivalent to Sharpe here since risk-free rate is often 0 in Qlib default)
    info_ratio = (mean_ret / std_ret) * np.sqrt(ann_factor) if std_ret > 0 else 0.0
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    max_drawdown = ((cum_returns / cum_returns.cummax()) - 1.0).min() if not cum_returns.empty else 0.0
    
    # Mimic Qlib's risk_analysis return structure
    result = {
        "mean": {"risk": mean_ret},
        "std": {"risk": std_ret},
        "annualized_return": {"risk": ann_ret},
        "annualized_volatility": {"risk": ann_vol},
        "information_ratio": {"risk": info_ratio},
        "max_drawdown": {"risk": max_drawdown},
        "daily_win_rate": {"risk": float((valid_returns > 0).mean()) if not valid_returns.empty else 0.0},
    }
    
    # Add monthly returns calculation
    report.index = pd.to_datetime(report.index)
    monthly_ret = report["return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    result["monthly_return"] = monthly_ret.to_dict()
    result["monthly_win_rate"] = {"risk": float((monthly_ret > 0).mean()) if not monthly_ret.empty else 0.0}
    if "turnover" in report.columns:
        result["turnover_mean"] = {"risk": float(report["turnover"].astype(float).mean())}
    
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
    plt.close(fig)


def save_monthly_report(report: pd.DataFrame, save_path: str = None):
    """Save monthly returns to a CSV file."""
    monthly_ret = report["return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df_monthly = monthly_ret.to_frame(name="monthly_return")
    df_monthly.index.name = "date"
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_monthly.to_csv(save_path)
    return df_monthly


def _compute_max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cum_returns = (1.0 + returns.astype(float)).cumprod()
    drawdown = (cum_returns / cum_returns.cummax()) - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _format_period_label(index: pd.DatetimeIndex, freq: str) -> str:
    start = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max())
    freq_upper = str(freq).upper()
    if freq_upper in {"ME", "M", "MS"}:
        return start.strftime("%Y-%m")
    return f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}"


def build_period_summary(report: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Aggregate daily backtest report into period-level summary rows."""
    period_rows: list[dict[str, float | int | str]] = []
    period_report = report.copy()
    period_report.index = pd.to_datetime(period_report.index)

    for _, period_frame in period_report.groupby(pd.Grouper(freq=freq)):
        if period_frame.empty:
            continue
        returns = period_frame["return"].astype(float).dropna()
        if returns.empty:
            continue
        bench_returns = (
            period_frame["bench"].astype(float).dropna()
            if "bench" in period_frame.columns
            else pd.Series(dtype=float)
        )
        period_rows.append(
            {
                "period": _format_period_label(pd.DatetimeIndex(period_frame.index), freq),
                "period_start": str(pd.Timestamp(period_frame.index.min()).date()),
                "period_end": str(pd.Timestamp(period_frame.index.max()).date()),
                "days": int(len(returns)),
                "return": float((1.0 + returns).prod() - 1.0),
                "win_rate": float((returns > 0).mean()),
                "avg_daily_return": float(returns.mean()),
                "daily_volatility": float(returns.std()) if len(returns) > 1 else 0.0,
                "max_drawdown": _compute_max_drawdown(returns),
                "avg_turnover": float(period_frame["turnover"].astype(float).mean())
                if "turnover" in period_frame.columns
                else np.nan,
                "bench_return": float((1.0 + bench_returns).prod() - 1.0) if not bench_returns.empty else np.nan,
            }
        )

    return pd.DataFrame(period_rows)


def save_period_summary(summary: pd.DataFrame, save_path: str | Path | None = None) -> pd.DataFrame:
    """Save a precomputed period summary to CSV."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    return summary


def _extract_metric(metrics: dict | None, key: str) -> float | None:
    if not metrics or key not in metrics:
        return None
    value = metrics[key]
    if isinstance(value, dict):
        value = value.get("risk")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_number(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}%}"


def _print_metric_line(label: str, value_text: str) -> None:
    print(f"  {label:<18}: {value_text}")


def print_metrics(
    signal_metrics: dict,
    portfolio_metrics: dict = None,
    period_summary: pd.DataFrame | None = None,
    period_label: str = "Monthly",
):
    """Print evaluation metrics without duplicate sections."""
    if period_summary is not None and not period_summary.empty:
        print("\n" + "=" * 72)
        print(f"{period_label} Summary")
        print("=" * 72)
        print(f"  {'Period':<22} {'Return':>10} {'WinRate':>10} {'MaxDD':>10} {'Turnover':>10} {'Days':>6}")
        print(f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 6}")
        for _, row in period_summary.iterrows():
            avg_turnover = row.get("avg_turnover")
            turnover_text = _format_pct(float(avg_turnover)) if pd.notna(avg_turnover) else "n/a"
            print(
                f"  {str(row['period']):<22} "
                f"{_format_pct(float(row['return']), signed=True):>10} "
                f"{_format_pct(float(row['win_rate'])):>10} "
                f"{_format_pct(float(row['max_drawdown'])):>10} "
                f"{turnover_text:>10} "
                f"{int(row['days']):>6d}"
            )

    if signal_metrics:
        print("\n" + "=" * 72)
        print("Signal Metrics")
        print("=" * 72)
        _print_metric_line("IC_mean", _format_number(_extract_metric(signal_metrics, "IC_mean")))
        _print_metric_line("IC_std", _format_number(_extract_metric(signal_metrics, "IC_std")))
        _print_metric_line("ICIR", _format_number(_extract_metric(signal_metrics, "ICIR")))
        _print_metric_line("IC>0", _format_pct(_extract_metric(signal_metrics, "IC_win_rate")))
        print()
        _print_metric_line("RankIC_mean", _format_number(_extract_metric(signal_metrics, "Rank_IC_mean")))
        _print_metric_line("RankIC_std", _format_number(_extract_metric(signal_metrics, "Rank_IC_std")))
        _print_metric_line("RankICIR", _format_number(_extract_metric(signal_metrics, "Rank_ICIR")))
        _print_metric_line("RankIC>0", _format_pct(_extract_metric(signal_metrics, "Rank_IC_win_rate")))

    if portfolio_metrics:
        print("\n" + "=" * 72)
        print("Portfolio Metrics")
        print("=" * 72)
        _print_metric_line("AnnRet", _format_pct(_extract_metric(portfolio_metrics, "annualized_return")))
        _print_metric_line("AnnVol", _format_pct(_extract_metric(portfolio_metrics, "annualized_volatility")))
        _print_metric_line("Sharpe", _format_number(_extract_metric(portfolio_metrics, "information_ratio")))
        _print_metric_line("MaxDD", _format_pct(_extract_metric(portfolio_metrics, "max_drawdown")))
        _print_metric_line("Daily win", _format_pct(_extract_metric(portfolio_metrics, "daily_win_rate")))
        _print_metric_line("Monthly win", _format_pct(_extract_metric(portfolio_metrics, "monthly_win_rate")))
        turnover_mean = _extract_metric(portfolio_metrics, "turnover_mean")
        _print_metric_line("Avg turnover", _format_pct(turnover_mean) if turnover_mean is not None else "n/a")

    print("=" * 72)
