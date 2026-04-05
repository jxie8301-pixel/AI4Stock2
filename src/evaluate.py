"""Evaluation metrics and visualization for quantitative strategies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_BENCHMARK_MODE = "cross_section_mean"
SUPPORTED_BENCHMARK_MODES = ("cross_section_mean", "file")
SUPPORTED_BENCHMARK_VALUE_TYPES = ("return", "close")


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


def _compute_return_distribution_metrics(returns: pd.Series) -> dict[str, float | int | None]:
    """Summarize win/loss asymmetry for a return series."""
    clean_returns = returns.astype(float).dropna()
    wins = clean_returns[clean_returns > 0]
    losses = clean_returns[clean_returns < 0]
    flat_days = int((clean_returns == 0).sum())
    avg_win = float(wins.mean()) if not wins.empty else None
    avg_loss = float(losses.mean()) if not losses.empty else None
    payoff_ratio = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff_ratio = float(avg_win / abs(avg_loss))
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses.sum())) if not losses.empty else 0.0
    profit_factor = None
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    return {
        "win_days": int(len(wins)),
        "loss_days": int(len(losses)),
        "flat_days": flat_days,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
    }


def _compute_chunked_period_returns(returns: pd.Series, period_size: int) -> pd.Series:
    clean_returns = returns.astype(float).dropna()
    if clean_returns.empty:
        return pd.Series(dtype=float)
    period_size = max(int(period_size), 1)
    rows: list[dict[str, object]] = []
    for start in range(0, len(clean_returns), period_size):
        chunk = clean_returns.iloc[start : start + period_size]
        if chunk.empty:
            continue
        rows.append(
            {
                "period_end": pd.Timestamp(chunk.index[-1]),
                "return": float((1.0 + chunk).prod() - 1.0),
            }
        )
    if not rows:
        return pd.Series(dtype=float)
    out = pd.DataFrame(rows).set_index("period_end")["return"].astype(float).sort_index()
    out.index.name = "date"
    return out


def _format_count_ratio(count: int, total: int) -> str:
    total_value = max(int(total), 0)
    count_value = max(int(count), 0)
    pct = (count_value / total_value) if total_value > 0 else 0.0
    return f"{count_value} / {total_value} = {pct:.2%}"


def build_cross_section_benchmark(labels: pd.Series) -> pd.Series:
    """Compute daily cross-sectional mean return as a common reference series."""
    aligned_labels = labels.dropna()
    if aligned_labels.empty:
        return pd.Series(dtype=float)
    return aligned_labels.groupby(level=0).mean().sort_index()


def _load_benchmark_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported benchmark file format: {path}. "
        "Use .csv, .txt, .parquet, or .pq."
    )


def _coerce_benchmark_returns(
    frame: pd.DataFrame,
    *,
    date_column: str,
    value_column: str,
    value_type: str,
) -> pd.Series:
    if date_column not in frame.columns:
        raise ValueError(f"Benchmark file is missing date column: {date_column}")
    if value_column not in frame.columns:
        raise ValueError(f"Benchmark file is missing value column: {value_column}")

    data = frame[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna().sort_values(date_column).drop_duplicates(subset=[date_column], keep="last")
    if data.empty:
        return pd.Series(dtype=float)

    series = data.set_index(date_column)[value_column].astype(float).sort_index()
    if value_type == "close":
        series = series.pct_change(fill_method=None)
    elif value_type != "return":
        raise ValueError(
            f"Unsupported benchmark value_type: {value_type}. "
            f"Supported: {', '.join(SUPPORTED_BENCHMARK_VALUE_TYPES)}"
        )
    return series.astype(float).sort_index()


def build_benchmark_series(
    labels: pd.Series,
    benchmark_cfg: dict | None = None,
) -> tuple[pd.Series, str]:
    """Resolve benchmark series for plots and excess-return analytics."""
    benchmark_cfg = benchmark_cfg or {}
    mode = str(benchmark_cfg.get("mode", DEFAULT_BENCHMARK_MODE) or DEFAULT_BENCHMARK_MODE).strip().lower()
    if mode == "cross_section_mean":
        return build_cross_section_benchmark(labels), "Cross-Section Mean"
    if mode == "file":
        raw_path = str(benchmark_cfg.get("path") or "").strip()
        if not raw_path:
            raise ValueError("backtest.benchmark.path must be set when benchmark.mode == 'file'")
        path = Path(raw_path)
        frame = _load_benchmark_frame(path)
        date_column = str(benchmark_cfg.get("date_column") or "date").strip() or "date"
        value_column = str(benchmark_cfg.get("value_column") or "close").strip() or "close"
        value_type = str(benchmark_cfg.get("value_type") or "close").strip().lower() or "close"
        name = str(benchmark_cfg.get("name") or path.stem).strip() or path.stem
        benchmark_series = _coerce_benchmark_returns(
            frame,
            date_column=date_column,
            value_column=value_column,
            value_type=value_type,
        )
        if benchmark_series.empty:
            raise ValueError(
                f"Benchmark {name} returned no usable rows from {path}. "
                "Check the file contents and configured date/value columns."
            )
        return benchmark_series, name
    raise ValueError(
        f"Unsupported benchmark mode: {mode}. "
        f"Supported: {', '.join(SUPPORTED_BENCHMARK_MODES)}"
    )


def align_benchmark_to_report_index(
    benchmark_series: pd.Series,
    report_index: pd.Index,
    *,
    benchmark_name: str,
) -> pd.Series:
    """Align benchmark series to report dates and fail loudly on empty/no-overlap inputs."""
    target_index = pd.DatetimeIndex(pd.to_datetime(report_index))
    if benchmark_series.empty:
        raise ValueError(f"Benchmark {benchmark_name} returned no rows.")
    aligned = benchmark_series.reindex(target_index)
    if int(aligned.notna().sum()) == 0:
        raise ValueError(
            f"Benchmark {benchmark_name} has no overlap with backtest period "
            f"{target_index.min().date()} ~ {target_index.max().date()}."
        )
    return aligned.fillna(0.0).astype(float)


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
    report.attrs = dict(getattr(portfolio_metric[0], "attrs", {}) or {})

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
    
    benchmark_name = str(report.attrs.get("benchmark_name") or "").strip()

    # Mimic Qlib's risk_analysis return structure
    distribution_metrics = _compute_return_distribution_metrics(valid_returns)
    result = {
        "mean": {"risk": mean_ret},
        "std": {"risk": std_ret},
        "annualized_return": {"risk": ann_ret},
        "annualized_volatility": {"risk": ann_vol},
        "information_ratio": {"risk": info_ratio},
        "max_drawdown": {"risk": max_drawdown},
        "daily_win_rate": {"risk": float((valid_returns > 0).mean()) if not valid_returns.empty else 0.0},
        "avg_win": {"risk": distribution_metrics["avg_win"]},
        "avg_loss": {"risk": distribution_metrics["avg_loss"]},
        "payoff_ratio": {"risk": distribution_metrics["payoff_ratio"]},
        "profit_factor": {"risk": distribution_metrics["profit_factor"]},
    }
    if "bench" in report.columns:
        bench_returns = report["bench"].astype(float).dropna()
        if not bench_returns.empty:
            bench_mean = float(bench_returns.mean())
            bench_std = float(bench_returns.std()) if len(bench_returns) > 1 else 0.0
            if pd.isna(bench_std):
                bench_std = 0.0
            bench_ann_ret = bench_mean * ann_factor
            bench_ann_vol = bench_std * np.sqrt(ann_factor)
            bench_max_drawdown = _compute_max_drawdown(bench_returns)
            aligned_strategy, aligned_bench = valid_returns.align(bench_returns, join="inner")
            excess_returns = aligned_strategy - aligned_bench
            excess_mean = float(excess_returns.mean()) if not excess_returns.empty else 0.0
            excess_std = float(excess_returns.std()) if len(excess_returns) > 1 else 0.0
            if pd.isna(excess_std):
                excess_std = 0.0
            excess_info_ratio = (
                (excess_mean / excess_std) * np.sqrt(ann_factor) if excess_std > 0 else 0.0
            )
            result["benchmark_name"] = benchmark_name or "Benchmark"
            result["benchmark_annualized_return"] = {"risk": bench_ann_ret}
            result["benchmark_annualized_volatility"] = {"risk": bench_ann_vol}
            result["benchmark_max_drawdown"] = {"risk": bench_max_drawdown}
            result["excess_annualized_return"] = {"risk": excess_mean * ann_factor}
            result["excess_information_ratio"] = {"risk": excess_info_ratio}
    
    # Add monthly returns calculation
    report.index = pd.to_datetime(report.index)
    monthly_ret = report["return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    result["monthly_return"] = monthly_ret.to_dict()
    monthly_positive_count = int((monthly_ret > 0).sum()) if not monthly_ret.empty else 0
    monthly_total_count = int(len(monthly_ret))
    result["monthly_win_rate"] = {"risk": float((monthly_ret > 0).mean()) if not monthly_ret.empty else 0.0}
    result["profitable_month_count"] = monthly_positive_count
    result["total_month_count"] = monthly_total_count
    result["profitable_month_pct"] = {
        "risk": float((monthly_positive_count / monthly_total_count) if monthly_total_count > 0 else 0.0)
    }
    result["profitable_month_summary"] = _format_count_ratio(monthly_positive_count, monthly_total_count)
    rebalance_freq = int(report.attrs.get("rebalance_freq", 0) or 0)
    if rebalance_freq > 0:
        rebalance_ret = _compute_chunked_period_returns(report["return"], rebalance_freq)
        result["rebalance_return"] = rebalance_ret.to_dict()
        rebalance_positive_count = int((rebalance_ret > 0).sum()) if not rebalance_ret.empty else 0
        rebalance_total_count = int(len(rebalance_ret))
        result["rebalance_win_rate"] = {
            "risk": float((rebalance_ret > 0).mean()) if not rebalance_ret.empty else 0.0
        }
        result["profitable_rebalance_count"] = rebalance_positive_count
        result["total_rebalance_count"] = rebalance_total_count
        result["profitable_rebalance_pct"] = {
            "risk": float((rebalance_positive_count / rebalance_total_count) if rebalance_total_count > 0 else 0.0)
        }
        result["profitable_rebalance_summary"] = _format_count_ratio(
            rebalance_positive_count,
            rebalance_total_count,
        )
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
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn_r",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        ax=ax,
    )
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

    ax.plot(cum_return.index, cum_return.values, label="Strategy", linewidth=1.5)
    if "bench" in report.columns:
        cum_bench = (1 + report["bench"]).cumprod()
        bench_label = str(report.attrs.get("benchmark_name") or "Benchmark").strip() or "Benchmark"
        ax.plot(cum_bench.index, cum_bench.values, label=bench_label, linewidth=1.5, alpha=0.7)
    if "avg_factor_baseline_return" in report.columns:
        cum_avg_factor = (1 + report["avg_factor_baseline_return"]).cumprod()
        avg_factor_label = str(report.attrs.get("avg_factor_baseline_name") or "Avg Unique Factor Baseline").strip()
        ax.plot(
            cum_avg_factor.index,
            cum_avg_factor.values,
            label=avg_factor_label or "Avg Factor Baseline",
            linewidth=1.5,
            linestyle="--",
            alpha=0.9,
        )
    if "sign_aligned_factor_baseline_return" in report.columns:
        cum_sign_aligned = (1 + report["sign_aligned_factor_baseline_return"]).cumprod()
        sign_aligned_label = str(
            report.attrs.get("sign_aligned_factor_baseline_name") or "Sign-Aligned Factor Baseline"
        ).strip()
        ax.plot(
            cum_sign_aligned.index,
            cum_sign_aligned.values,
            label=sign_aligned_label or "Sign-Aligned Factor Baseline",
            linewidth=1.5,
            linestyle=":",
            alpha=0.9,
        )
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
        distribution_metrics = _compute_return_distribution_metrics(returns)
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
                "win_days": int(distribution_metrics["win_days"]),
                "loss_days": int(distribution_metrics["loss_days"]),
                "flat_days": int(distribution_metrics["flat_days"]),
                "avg_win": distribution_metrics["avg_win"],
                "avg_loss": distribution_metrics["avg_loss"],
                "payoff_ratio": distribution_metrics["payoff_ratio"],
                "profit_factor": distribution_metrics["profit_factor"],
            }
        )

    return pd.DataFrame(period_rows)


def build_rebalance_period_summary(report: pd.DataFrame, rebalance_freq: int) -> pd.DataFrame:
    """Aggregate daily report into fixed trading-day rebalance periods."""
    rebalance_freq = max(int(rebalance_freq), 1)
    period_report = report.copy()
    period_report.index = pd.to_datetime(period_report.index)
    period_rows: list[dict[str, float | int | str]] = []

    for idx, start in enumerate(range(0, len(period_report), rebalance_freq), start=1):
        period_frame = period_report.iloc[start : start + rebalance_freq]
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
        distribution_metrics = _compute_return_distribution_metrics(returns)
        period_rows.append(
            {
                "period": f"rebalance_{idx:03d}",
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
                "win_days": int(distribution_metrics["win_days"]),
                "loss_days": int(distribution_metrics["loss_days"]),
                "flat_days": int(distribution_metrics["flat_days"]),
                "avg_win": distribution_metrics["avg_win"],
                "avg_loss": distribution_metrics["avg_loss"],
                "payoff_ratio": distribution_metrics["payoff_ratio"],
                "profit_factor": distribution_metrics["profit_factor"],
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
        print(
            f"  {'Period':<22} {'Return':>10} {'WinRate':>10} {'Payoff':>8} "
            f"{'MaxDD':>10} {'Turnover':>10} {'Days':>6}"
        )
        print(
            f"  {'-' * 22} {'-' * 10} {'-' * 10} {'-' * 8} "
            f"{'-' * 10} {'-' * 10} {'-' * 6}"
        )
        for _, row in period_summary.iterrows():
            avg_turnover = row.get("avg_turnover")
            turnover_text = _format_pct(float(avg_turnover)) if pd.notna(avg_turnover) else "n/a"
            payoff_text = _format_number(row.get("payoff_ratio"))
            print(
                f"  {str(row['period']):<22} "
                f"{_format_pct(float(row['return']), signed=True):>10} "
                f"{_format_pct(float(row['win_rate'])):>10} "
                f"{payoff_text:>8} "
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
        _print_metric_line("Rebalance win", _format_pct(_extract_metric(portfolio_metrics, "rebalance_win_rate")))
        profitable_month_summary = portfolio_metrics.get("profitable_month_summary")
        if profitable_month_summary:
            _print_metric_line("Profitable months", str(profitable_month_summary))
        profitable_rebalance_summary = portfolio_metrics.get("profitable_rebalance_summary")
        if profitable_rebalance_summary:
            _print_metric_line("Profitable rebal", str(profitable_rebalance_summary))
        benchmark_name = portfolio_metrics.get("benchmark_name")
        if benchmark_name:
            _print_metric_line("Benchmark", str(benchmark_name))
            _print_metric_line(
                "Bench AnnRet",
                _format_pct(_extract_metric(portfolio_metrics, "benchmark_annualized_return")),
            )
            _print_metric_line(
                "Excess AnnRet",
                _format_pct(_extract_metric(portfolio_metrics, "excess_annualized_return")),
            )
            _print_metric_line(
                "Excess IR",
                _format_number(_extract_metric(portfolio_metrics, "excess_information_ratio")),
            )
        print()
        _print_metric_line("Avg win", _format_pct(_extract_metric(portfolio_metrics, "avg_win")))
        _print_metric_line("Avg loss", _format_pct(_extract_metric(portfolio_metrics, "avg_loss")))
        _print_metric_line("Payoff ratio", _format_number(_extract_metric(portfolio_metrics, "payoff_ratio")))
        _print_metric_line("Profit factor", _format_number(_extract_metric(portfolio_metrics, "profit_factor")))
        turnover_mean = _extract_metric(portfolio_metrics, "turnover_mean")
        _print_metric_line("Avg turnover", _format_pct(turnover_mean) if turnover_mean is not None else "n/a")

    print("=" * 72)
