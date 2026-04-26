"""Risk control and intraperiod exit helpers for native backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest_scoring import compute_signal_strength_value, normalize_min_score


SUPPORTED_RISK_CONTROL_MODES = ("fixed", "benchmark_ma", "signal_strength", "benchmark_ma_signal_strength")
SUPPORTED_SIGNAL_STRENGTH_METRICS = ("top1", "topk_mean", "topk_sum")
SUPPORTED_SIGNAL_SOURCES = ("score_strength", "validation_metric")
SUPPORTED_VALIDATION_SIGNAL_METRICS = (
    "best_valid_daily_rank_ic",
    "best_valid_daily_ic",
    "valid_top1_label_mean",
    "valid_top1_positive_rate",
    "valid_topk_label_mean",
    "valid_topk_label_median",
    "valid_topk_min_label_mean",
    "valid_topk_positive_rate",
    "valid_topk_excess_mean",
)
SUPPORTED_RISK_CURVES = ("linear", "convex", "concave", "sigmoid")
SUPPORTED_INTRAPERIOD_EXIT_MODES = ("none", "score_threshold", "expected_return_threshold")
SUPPORTED_EXIT_SCORE_SOURCES = ("raw", "transformed", "rank_pct", "zscore")
SUPPORTED_INTRAPERIOD_EXIT_CALIBRATIONS = ("quantile_bins",)
SUPPORTED_INTRAPERIOD_PRICE_CONFIRM_MODES = ("close_below_ma",)
INTRAPERIOD_PRICE_CONFIRM_SIGNAL_TIMING = "same_signal_date_close"
INTRAPERIOD_PRICE_CONFIRM_EXECUTION_TIMING = "next_open"


def validate_risk_degree(value: float, field: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return out


def normalize_intraperiod_exit_config(
    intraperiod_exit: dict[str, object] | None,
) -> dict[str, float | int | str] | None:
    if intraperiod_exit is None:
        return None
    if not isinstance(intraperiod_exit, dict):
        raise ValueError("intraperiod_exit must be a mapping when provided")
    mode = str(intraperiod_exit.get("mode", "none") or "none").strip().lower()
    if mode == "none":
        return None
    if mode not in SUPPORTED_INTRAPERIOD_EXIT_MODES:
        raise ValueError(
            f"Unsupported intraperiod exit mode: {mode}. Supported: {', '.join(SUPPORTED_INTRAPERIOD_EXIT_MODES)}"
        )
    score_source = str(intraperiod_exit.get("score_source", "raw") or "raw").strip().lower()
    if score_source not in SUPPORTED_EXIT_SCORE_SOURCES:
        raise ValueError(
            f"Unsupported intraperiod exit score_source: {score_source}. "
            f"Supported: {', '.join(SUPPORTED_EXIT_SCORE_SOURCES)}"
        )
    threshold = float(intraperiod_exit.get("threshold", 0.0))
    out: dict[str, float | int | str] = {"mode": mode, "score_source": score_source, "threshold": threshold}
    if mode == "expected_return_threshold":
        calibration = str(intraperiod_exit.get("calibration", "quantile_bins") or "quantile_bins").strip().lower()
        if calibration not in SUPPORTED_INTRAPERIOD_EXIT_CALIBRATIONS:
            raise ValueError(
                f"Unsupported intraperiod exit calibration: {calibration}. "
                f"Supported: {', '.join(SUPPORTED_INTRAPERIOD_EXIT_CALIBRATIONS)}"
            )
        out["calibration"] = calibration
        out["n_bins"] = max(int(intraperiod_exit.get("n_bins", 20) or 20), 2)
        out["min_history"] = max(int(intraperiod_exit.get("min_history", 200) or 200), 1)
    price_confirm = intraperiod_exit.get("price_confirm")
    if price_confirm is not None:
        if not isinstance(price_confirm, dict):
            raise ValueError("intraperiod_exit.price_confirm must be a mapping when provided")
        confirm_mode = str(price_confirm.get("mode", "close_below_ma") or "close_below_ma").strip().lower()
        if confirm_mode not in SUPPORTED_INTRAPERIOD_PRICE_CONFIRM_MODES:
            raise ValueError(
                "Unsupported intraperiod_exit.price_confirm.mode: "
                f"{confirm_mode}. Supported: {', '.join(SUPPORTED_INTRAPERIOD_PRICE_CONFIRM_MODES)}"
            )
        out["price_confirm_mode"] = confirm_mode
        out["price_confirm_ma_window"] = max(int(price_confirm.get("ma_window", 10) or 10), 1)
        out["price_confirm_min_remaining_steps"] = max(
            int(price_confirm.get("min_remaining_steps", 0) or 0),
            0,
        )
        signal_timing = str(
            price_confirm.get("signal_timing", INTRAPERIOD_PRICE_CONFIRM_SIGNAL_TIMING)
            or INTRAPERIOD_PRICE_CONFIRM_SIGNAL_TIMING
        ).strip().lower()
        execution_timing = str(
            price_confirm.get("execution_timing", INTRAPERIOD_PRICE_CONFIRM_EXECUTION_TIMING)
            or INTRAPERIOD_PRICE_CONFIRM_EXECUTION_TIMING
        ).strip().lower()
        if signal_timing != INTRAPERIOD_PRICE_CONFIRM_SIGNAL_TIMING:
            raise ValueError(
                "intraperiod_exit.price_confirm.signal_timing must be "
                f"{INTRAPERIOD_PRICE_CONFIRM_SIGNAL_TIMING}"
            )
        if execution_timing != INTRAPERIOD_PRICE_CONFIRM_EXECUTION_TIMING:
            raise ValueError(
                "intraperiod_exit.price_confirm.execution_timing must be "
                f"{INTRAPERIOD_PRICE_CONFIRM_EXECUTION_TIMING}"
            )
        out["price_confirm_signal_timing"] = signal_timing
        out["price_confirm_execution_timing"] = execution_timing
        force_exit_threshold = price_confirm.get("force_exit_threshold")
        if force_exit_threshold is not None:
            out["price_confirm_force_exit_threshold"] = float(force_exit_threshold)
    return out


def build_intraperiod_price_confirm_matrix(
    close_matrix: pd.DataFrame,
    *,
    confirm_mode: str,
    ma_window: int,
) -> pd.DataFrame:
    close_matrix = close_matrix.astype(float)
    confirm_mode = str(confirm_mode or "close_below_ma").strip().lower()
    if confirm_mode == "close_below_ma":
        rolling_ma = close_matrix.rolling(int(ma_window), min_periods=int(ma_window)).mean()
        out = close_matrix.lt(rolling_ma)
        return out.fillna(False)
    raise ValueError(
        "Unsupported intraperiod_exit.price_confirm.mode: "
        f"{confirm_mode}. Supported: {', '.join(SUPPORTED_INTRAPERIOD_PRICE_CONFIRM_MODES)}"
    )


def build_intraperiod_remaining_steps(index: pd.Index, *, rebalance_freq: int) -> pd.Series:
    positions = np.arange(len(index), dtype=int)
    next_rebalance = np.minimum(((positions // rebalance_freq) + 1) * rebalance_freq, len(index))
    steps = np.where(positions % rebalance_freq == 0, 0, np.maximum(next_rebalance - positions, 0))
    return pd.Series(steps, index=index, dtype=int)


def build_intraperiod_residual_return_matrix(
    label_matrix: pd.DataFrame,
    *,
    remaining_steps: pd.Series,
) -> pd.DataFrame:
    values = label_matrix.to_numpy(dtype=float, copy=False)
    out = np.full(values.shape, np.nan, dtype=float)
    steps_array = remaining_steps.reindex(label_matrix.index).to_numpy(dtype=int, copy=False)

    for pos, steps in enumerate(steps_array):
        if steps <= 0:
            continue
        window = values[pos : pos + int(steps)]
        if window.size == 0:
            continue
        valid = np.isfinite(window).all(axis=0)
        if not valid.any():
            continue
        out[pos, valid] = np.prod(1.0 + window[:, valid], axis=0) - 1.0

    return pd.DataFrame(out, index=label_matrix.index, columns=label_matrix.columns, dtype=float)


def get_intraperiod_history_arrays(
    history_entry: dict[str, object] | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if history_entry is None:
        return None, None
    scores_cache = history_entry.get("scores_cache")
    returns_cache = history_entry.get("returns_cache")
    if isinstance(scores_cache, np.ndarray) and isinstance(returns_cache, np.ndarray):
        return scores_cache, returns_cache

    score_chunks = history_entry.get("score_chunks")
    return_chunks = history_entry.get("return_chunks")
    if not isinstance(score_chunks, list) or not isinstance(return_chunks, list) or not score_chunks:
        return None, None

    scores = np.concatenate(score_chunks).astype(float, copy=False)
    returns = np.concatenate(return_chunks).astype(float, copy=False)
    history_entry["scores_cache"] = scores
    history_entry["returns_cache"] = returns
    return scores, returns


def append_intraperiod_history(
    history: dict[int, dict[str, object]],
    *,
    remaining_steps: int,
    score_row: pd.Series,
    residual_row: pd.Series,
) -> None:
    if remaining_steps <= 0:
        return
    valid_mask = score_row.notna() & residual_row.notna()
    if not valid_mask.any():
        return
    entry = history.setdefault(
        int(remaining_steps),
        {"score_chunks": [], "return_chunks": [], "scores_cache": None, "returns_cache": None},
    )
    score_values = score_row.loc[valid_mask].to_numpy(dtype=float, copy=False)
    residual_values = residual_row.loc[valid_mask].to_numpy(dtype=float, copy=False)
    if score_values.size == 0:
        return
    score_chunks = entry["score_chunks"]
    return_chunks = entry["return_chunks"]
    if isinstance(score_chunks, list) and isinstance(return_chunks, list):
        score_chunks.append(score_values)
        return_chunks.append(residual_values)
    entry["scores_cache"] = None
    entry["returns_cache"] = None


def estimate_intraperiod_expected_returns(
    score_row: pd.Series,
    *,
    history_entry: dict[str, object] | None,
    n_bins: int,
    min_history: int,
) -> pd.Series:
    out = pd.Series(np.nan, index=score_row.index, dtype=float)
    history_scores, history_returns = get_intraperiod_history_arrays(history_entry)
    if history_scores is None or history_returns is None:
        return out

    valid_history = np.isfinite(history_scores) & np.isfinite(history_returns)
    if int(valid_history.sum()) < int(min_history):
        return out

    history_scores = history_scores[valid_history]
    history_returns = history_returns[valid_history]
    if history_scores.size == 0:
        return out

    bin_count = max(1, min(int(n_bins), int(history_scores.size)))
    if bin_count == 1:
        out.loc[score_row.notna()] = float(history_returns.mean())
        return out

    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.quantile(history_scores, quantiles)
    if not np.isfinite(edges).all() or np.isclose(edges[0], edges[-1]):
        out.loc[score_row.notna()] = float(history_returns.mean())
        return out

    inner_edges = edges[1:-1]
    history_bucket = np.searchsorted(inner_edges, history_scores, side="right")
    bucket_counts = np.bincount(history_bucket, minlength=bin_count)
    bucket_sums = np.bincount(history_bucket, weights=history_returns, minlength=bin_count)
    global_mean = float(history_returns.mean())
    bucket_means = np.full(bin_count, global_mean, dtype=float)
    nonzero = bucket_counts > 0
    bucket_means[nonzero] = bucket_sums[nonzero] / bucket_counts[nonzero]

    current_values = score_row.to_numpy(dtype=float, copy=False)
    valid_current = np.isfinite(current_values)
    if not valid_current.any():
        return out
    current_bucket = np.searchsorted(inner_edges, current_values[valid_current], side="right")
    out.iloc[np.flatnonzero(valid_current)] = bucket_means[current_bucket]
    return out


def normalize_risk_control_config(
    risk_control: dict[str, object] | None,
    *,
    fallback_risk_degree: float,
) -> dict[str, float | int | str]:
    if risk_control is None:
        return {"mode": "fixed", "risk_degree": validate_risk_degree(float(fallback_risk_degree), "risk_control.risk_degree")}
    if not isinstance(risk_control, dict):
        raise ValueError("risk_control must be a mapping when provided")
    mode = str(risk_control.get("mode", "fixed") or "fixed").strip().lower()
    if mode not in SUPPORTED_RISK_CONTROL_MODES:
        raise ValueError(
            f"Unsupported risk control mode: {mode}. Supported: {', '.join(SUPPORTED_RISK_CONTROL_MODES)}"
        )
    if mode == "fixed":
        return {
            "mode": "fixed",
            "risk_degree": validate_risk_degree(float(risk_control.get("risk_degree", fallback_risk_degree)), "risk_control.risk_degree"),
        }
    if mode in {"signal_strength", "benchmark_ma_signal_strength"}:
        signal_metric = str(risk_control.get("signal_metric", "topk_mean") or "topk_mean").strip().lower()
        if signal_metric not in SUPPORTED_SIGNAL_STRENGTH_METRICS:
            raise ValueError("risk_control.signal_metric must be one of: " + ", ".join(SUPPORTED_SIGNAL_STRENGTH_METRICS))
        signal_source = str(risk_control.get("signal_source", "score_strength") or "score_strength").strip().lower()
        if signal_source not in SUPPORTED_SIGNAL_SOURCES:
            raise ValueError("risk_control.signal_source must be one of: " + ", ".join(SUPPORTED_SIGNAL_SOURCES))
        min_signal = float(risk_control.get("min_signal", 0.0))
        max_signal = float(risk_control.get("max_signal", 2.0))
        if max_signal <= min_signal:
            raise ValueError("risk_control.max_signal must be greater than risk_control.min_signal")
        min_signal_quantile = risk_control.get("min_signal_quantile")
        max_signal_quantile = risk_control.get("max_signal_quantile")
        if min_signal_quantile is not None:
            min_signal_quantile = float(min_signal_quantile)
            if min_signal_quantile < 0.0 or min_signal_quantile > 1.0:
                raise ValueError("risk_control.min_signal_quantile must be in [0, 1]")
        if max_signal_quantile is not None:
            max_signal_quantile = float(max_signal_quantile)
            if max_signal_quantile < 0.0 or max_signal_quantile > 1.0:
                raise ValueError("risk_control.max_signal_quantile must be in [0, 1]")
        if min_signal_quantile is not None and max_signal_quantile is not None and max_signal_quantile <= min_signal_quantile:
            raise ValueError("risk_control.max_signal_quantile must be greater than risk_control.min_signal_quantile")
        min_risk = validate_risk_degree(float(risk_control.get("min_risk", min(fallback_risk_degree, 0.3))), "risk_control.min_risk")
        max_risk = validate_risk_degree(float(risk_control.get("max_risk", fallback_risk_degree)), "risk_control.max_risk")
        if max_risk < min_risk:
            raise ValueError("risk_control.max_risk must be >= risk_control.min_risk")
        risk_curve = str(risk_control.get("risk_curve", "linear") or "linear").strip().lower()
        if risk_curve not in SUPPORTED_RISK_CURVES:
            raise ValueError("risk_control.risk_curve must be one of: " + ", ".join(SUPPORTED_RISK_CURVES))
        risk_curve_power = float(risk_control.get("risk_curve_power", 2.0))
        if risk_curve_power <= 0:
            raise ValueError("risk_control.risk_curve_power must be > 0")
        risk_curve_center = float(risk_control.get("risk_curve_center", 0.5))
        if risk_curve_center < 0.0 or risk_curve_center > 1.0:
            raise ValueError("risk_control.risk_curve_center must be in [0, 1]")
        risk_curve_steepness = float(risk_control.get("risk_curve_steepness", 8.0))
        if risk_curve_steepness <= 0:
            raise ValueError("risk_control.risk_curve_steepness must be > 0")
        out: dict[str, float | int | str] = {
            "mode": mode,
            "risk_degree": validate_risk_degree(float(fallback_risk_degree), "risk_control.risk_degree"),
            "signal_metric": signal_metric,
            "signal_source": signal_source,
            "min_signal": min_signal,
            "max_signal": max_signal,
            "min_risk": min_risk,
            "max_risk": max_risk,
            "risk_curve": risk_curve,
            "risk_curve_power": risk_curve_power,
            "risk_curve_center": risk_curve_center,
            "risk_curve_steepness": risk_curve_steepness,
        }
        if signal_source == "validation_metric":
            validation_metric = str(risk_control.get("validation_metric", "valid_topk_label_mean") or "valid_topk_label_mean").strip().lower()
            if validation_metric not in SUPPORTED_VALIDATION_SIGNAL_METRICS:
                raise ValueError(
                    "risk_control.validation_metric must be one of: "
                    + ", ".join(SUPPORTED_VALIDATION_SIGNAL_METRICS)
                )
            out["validation_metric"] = validation_metric
            secondary_validation_metric = risk_control.get("secondary_validation_metric")
            if secondary_validation_metric is not None:
                secondary_validation_metric = str(secondary_validation_metric).strip().lower()
                if secondary_validation_metric not in SUPPORTED_VALIDATION_SIGNAL_METRICS:
                    raise ValueError(
                        "risk_control.secondary_validation_metric must be one of: "
                        + ", ".join(SUPPORTED_VALIDATION_SIGNAL_METRICS)
                    )
                secondary_min_signal = float(risk_control.get("secondary_min_signal", min_signal))
                secondary_max_signal = float(risk_control.get("secondary_max_signal", max_signal))
                if secondary_max_signal <= secondary_min_signal:
                    raise ValueError(
                        "risk_control.secondary_max_signal must be greater than risk_control.secondary_min_signal"
                    )
                secondary_min_signal_quantile = risk_control.get("secondary_min_signal_quantile")
                secondary_max_signal_quantile = risk_control.get("secondary_max_signal_quantile")
                if secondary_min_signal_quantile is not None:
                    secondary_min_signal_quantile = float(secondary_min_signal_quantile)
                    if secondary_min_signal_quantile < 0.0 or secondary_min_signal_quantile > 1.0:
                        raise ValueError("risk_control.secondary_min_signal_quantile must be in [0, 1]")
                if secondary_max_signal_quantile is not None:
                    secondary_max_signal_quantile = float(secondary_max_signal_quantile)
                    if secondary_max_signal_quantile < 0.0 or secondary_max_signal_quantile > 1.0:
                        raise ValueError("risk_control.secondary_max_signal_quantile must be in [0, 1]")
                if (
                    secondary_min_signal_quantile is not None
                    and secondary_max_signal_quantile is not None
                    and secondary_max_signal_quantile <= secondary_min_signal_quantile
                ):
                    raise ValueError(
                        "risk_control.secondary_max_signal_quantile must be greater than "
                        "risk_control.secondary_min_signal_quantile"
                    )
                secondary_min_risk = validate_risk_degree(
                    float(risk_control.get("secondary_min_risk", min_risk)),
                    "risk_control.secondary_min_risk",
                )
                secondary_max_risk = validate_risk_degree(
                    float(risk_control.get("secondary_max_risk", max_risk)),
                    "risk_control.secondary_max_risk",
                )
                if secondary_max_risk < secondary_min_risk:
                    raise ValueError(
                        "risk_control.secondary_max_risk must be >= risk_control.secondary_min_risk"
                    )
                out["secondary_validation_metric"] = secondary_validation_metric
                out["secondary_min_signal"] = secondary_min_signal
                out["secondary_max_signal"] = secondary_max_signal
                out["secondary_min_risk"] = secondary_min_risk
                out["secondary_max_risk"] = secondary_max_risk
                if secondary_min_signal_quantile is not None:
                    out["secondary_min_signal_quantile"] = secondary_min_signal_quantile
                if secondary_max_signal_quantile is not None:
                    out["secondary_max_signal_quantile"] = secondary_max_signal_quantile
        if min_signal_quantile is not None:
            out["min_signal_quantile"] = min_signal_quantile
        if max_signal_quantile is not None:
            out["max_signal_quantile"] = max_signal_quantile
        if mode == "signal_strength":
            return out
        fast_window = max(int(risk_control.get("fast_window", 120) or 120), 1)
        slow_window = max(int(risk_control.get("slow_window", 250) or 250), 1)
        if fast_window >= slow_window:
            raise ValueError("risk_control.fast_window must be smaller than risk_control.slow_window")
        bull_risk = validate_risk_degree(float(risk_control.get("bull_risk", fallback_risk_degree)), "risk_control.bull_risk")
        neutral_risk = validate_risk_degree(float(risk_control.get("neutral_risk", min(fallback_risk_degree, 0.5))), "risk_control.neutral_risk")
        bear_risk = validate_risk_degree(float(risk_control.get("bear_risk", 0.15)), "risk_control.bear_risk")
        out.update(
            {
                "fast_window": fast_window,
                "slow_window": slow_window,
                "bull_risk": bull_risk,
                "neutral_risk": neutral_risk,
                "bear_risk": bear_risk,
            }
        )
        return out
    fast_window = max(int(risk_control.get("fast_window", 120) or 120), 1)
    slow_window = max(int(risk_control.get("slow_window", 250) or 250), 1)
    if fast_window >= slow_window:
        raise ValueError("risk_control.fast_window must be smaller than risk_control.slow_window")
    bull_risk = validate_risk_degree(float(risk_control.get("bull_risk", fallback_risk_degree)), "risk_control.bull_risk")
    neutral_risk = validate_risk_degree(float(risk_control.get("neutral_risk", min(fallback_risk_degree, 0.5))), "risk_control.neutral_risk")
    bear_risk = validate_risk_degree(float(risk_control.get("bear_risk", 0.15)), "risk_control.bear_risk")
    return {
        "mode": mode,
        "risk_degree": validate_risk_degree(float(fallback_risk_degree), "risk_control.risk_degree"),
        "fast_window": fast_window,
        "slow_window": slow_window,
        "bull_risk": bull_risk,
        "neutral_risk": neutral_risk,
        "bear_risk": bear_risk,
    }


def build_benchmark_ma_schedule(
    benchmark_returns: pd.Series | None,
    *,
    risk_control: dict[str, float | int | str],
    fallback_risk_degree: float,
) -> pd.Series:
    if benchmark_returns is None:
        raise ValueError("benchmark_returns is required when risk_control.mode needs benchmark context")
    benchmark_returns = pd.Series(benchmark_returns).astype(float).sort_index()
    if benchmark_returns.empty:
        raise ValueError("benchmark_returns is empty while risk_control.mode needs benchmark context")

    nav = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    fast_window = int(risk_control["fast_window"])
    slow_window = int(risk_control["slow_window"])
    fast_ma = nav.rolling(fast_window, min_periods=1).mean()
    slow_ma = nav.rolling(slow_window, min_periods=1).mean()

    raw_schedule = pd.Series(float(risk_control["bear_risk"]), index=nav.index, dtype=float)
    raw_schedule.loc[nav >= slow_ma] = float(risk_control["neutral_risk"])
    raw_schedule.loc[nav >= fast_ma] = float(risk_control["bull_risk"])
    return raw_schedule.shift(1).fillna(float(fallback_risk_degree)).astype(float)


def build_signal_strength_schedule(
    transformed_score_matrix: pd.DataFrame,
    *,
    risk_control: dict[str, float | int | str],
    topk: int,
    min_score: float | None,
) -> tuple[pd.Series, pd.Series]:
    signal_metric = str(risk_control["signal_metric"])
    signal_values = _compute_signal_strength_series(
        transformed_score_matrix,
        topk=topk,
        min_score=min_score,
        signal_metric=signal_metric,
    )
    return build_signal_schedule_from_series(signal_values, risk_control=risk_control)


def build_signal_schedule_from_series(
    signal_values: pd.Series,
    *,
    risk_control: dict[str, float | int | str],
) -> tuple[pd.Series, pd.Series]:
    signal_values = pd.Series(signal_values, copy=False).astype(float).sort_index()
    min_threshold = pd.Series(float(risk_control["min_signal"]), index=signal_values.index, dtype=float)
    max_threshold = pd.Series(float(risk_control["max_signal"]), index=signal_values.index, dtype=float)
    min_q = risk_control.get("min_signal_quantile")
    max_q = risk_control.get("max_signal_quantile")
    shifted = signal_values.shift(1)
    if min_q is not None:
        min_threshold = shifted.expanding(min_periods=1).quantile(float(min_q)).reindex(signal_values.index)
        min_threshold = min_threshold.fillna(float(risk_control["min_signal"])).astype(float)
    if max_q is not None:
        max_threshold = shifted.expanding(min_periods=1).quantile(float(max_q)).reindex(signal_values.index)
        max_threshold = max_threshold.fillna(float(risk_control["max_signal"])).astype(float)
    invalid_thresholds = max_threshold <= min_threshold
    if invalid_thresholds.any():
        min_threshold.loc[invalid_thresholds] = float(risk_control["min_signal"])
        max_threshold.loc[invalid_thresholds] = float(risk_control["max_signal"])
    width = (max_threshold - min_threshold).where(lambda s: s > 0, np.nan)
    scale = ((signal_values - min_threshold) / width).clip(lower=0.0, upper=1.0).fillna(0.0)
    scale = _apply_risk_curve(scale, risk_control=risk_control)
    min_risk = float(risk_control["min_risk"])
    max_risk = float(risk_control["max_risk"])
    schedule = (min_risk + scale * (max_risk - min_risk)).astype(float)
    return schedule, signal_values


def build_combined_validation_metric_schedule(
    signal_values_by_metric: dict[str, pd.Series],
    *,
    risk_control: dict[str, float | int | str],
) -> tuple[pd.Series, pd.Series]:
    primary_metric = str(risk_control["validation_metric"])
    if primary_metric not in signal_values_by_metric:
        raise ValueError(f"Missing primary validation metric series: {primary_metric}")
    primary_schedule, primary_signal_values = build_signal_schedule_from_series(
        signal_values_by_metric[primary_metric],
        risk_control=risk_control,
    )
    secondary_metric = risk_control.get("secondary_validation_metric")
    if secondary_metric is None:
        return primary_schedule, primary_signal_values
    secondary_metric = str(secondary_metric)
    if secondary_metric not in signal_values_by_metric:
        raise ValueError(f"Missing secondary validation metric series: {secondary_metric}")
    secondary_cfg: dict[str, float | int | str] = {
        "min_signal": float(risk_control["secondary_min_signal"]),
        "max_signal": float(risk_control["secondary_max_signal"]),
        "min_risk": float(risk_control["secondary_min_risk"]),
        "max_risk": float(risk_control["secondary_max_risk"]),
        "risk_curve": str(risk_control.get("risk_curve", "linear")),
        "risk_curve_power": float(risk_control.get("risk_curve_power", 2.0)),
        "risk_curve_center": float(risk_control.get("risk_curve_center", 0.5)),
        "risk_curve_steepness": float(risk_control.get("risk_curve_steepness", 8.0)),
    }
    if "secondary_min_signal_quantile" in risk_control:
        secondary_cfg["min_signal_quantile"] = float(risk_control["secondary_min_signal_quantile"])
    if "secondary_max_signal_quantile" in risk_control:
        secondary_cfg["max_signal_quantile"] = float(risk_control["secondary_max_signal_quantile"])
    secondary_schedule, _ = build_signal_schedule_from_series(
        signal_values_by_metric[secondary_metric],
        risk_control=secondary_cfg,
    )
    combined = pd.concat(
        [primary_schedule.rename("primary"), secondary_schedule.rename("secondary")],
        axis=1,
    )
    return combined.min(axis=1).astype(float), primary_signal_values


def _apply_risk_curve(
    scale: pd.Series,
    *,
    risk_control: dict[str, float | int | str],
) -> pd.Series:
    curve = str(risk_control.get("risk_curve", "linear") or "linear").strip().lower()
    scale = pd.Series(scale, copy=False).astype(float).clip(lower=0.0, upper=1.0)
    if curve == "linear":
        return scale

    power = float(risk_control.get("risk_curve_power", 2.0))
    if curve == "convex":
        return scale.pow(power).clip(lower=0.0, upper=1.0)
    if curve == "concave":
        return (1.0 - (1.0 - scale).pow(power)).clip(lower=0.0, upper=1.0)
    if curve == "sigmoid":
        center = float(risk_control.get("risk_curve_center", 0.5))
        steepness = float(risk_control.get("risk_curve_steepness", 8.0))
        centered = 1.0 / (1.0 + np.exp(-steepness * (scale - center)))
        lower = 1.0 / (1.0 + np.exp(-steepness * (0.0 - center)))
        upper = 1.0 / (1.0 + np.exp(-steepness * (1.0 - center)))
        denom = max(upper - lower, 1e-12)
        return pd.Series((centered - lower) / denom, index=scale.index, dtype=float).clip(lower=0.0, upper=1.0)
    raise ValueError(f"Unsupported risk curve: {curve}")


def _compute_signal_strength_series(
    transformed_score_matrix: pd.DataFrame,
    *,
    topk: int,
    min_score: float | None,
    signal_metric: str,
) -> pd.Series:
    if transformed_score_matrix.empty:
        return pd.Series(dtype=float)

    min_score_value = normalize_min_score(min_score)
    values = transformed_score_matrix.to_numpy(dtype=float, copy=False)
    valid = np.isfinite(values)
    if min_score_value is not None:
        valid &= values > min_score_value
    filled = np.where(valid, values, -np.inf)
    n_rows, n_cols = filled.shape
    k = max(1, min(int(topk), n_cols))

    if signal_metric == "top1":
        row_max = filled.max(axis=1)
        out = np.where(np.isfinite(row_max), row_max, np.nan)
        return pd.Series(out, index=transformed_score_matrix.index, dtype=float)

    top_values = np.partition(filled, n_cols - k, axis=1)[:, -k:]
    top_valid = np.isfinite(top_values)
    sums = np.where(top_valid, top_values, 0.0).sum(axis=1)

    if signal_metric == "topk_sum":
        out = np.where(top_valid.any(axis=1), sums, np.nan)
        return pd.Series(out, index=transformed_score_matrix.index, dtype=float)

    if signal_metric == "topk_mean":
        counts = top_valid.sum(axis=1)
        out = np.divide(sums, counts, out=np.full(n_rows, np.nan, dtype=float), where=counts > 0)
        return pd.Series(out, index=transformed_score_matrix.index, dtype=float)

    return pd.Series(
        {
            date: compute_signal_strength_value(
                transformed_score_matrix.loc[date],
                topk=topk,
                min_score=min_score,
                signal_metric=signal_metric,
            )
            for date in transformed_score_matrix.index
        },
        dtype=float,
    ).sort_index()


def build_risk_control_schedule(
    benchmark_returns: pd.Series | None,
    transformed_score_matrix: pd.DataFrame,
    *,
    risk_control: dict[str, float | int | str],
    fallback_risk_degree: float,
    topk: int,
    min_score: float | None,
    external_signal_values: pd.Series | dict[str, pd.Series] | None = None,
) -> tuple[pd.Series | None, pd.Series | None]:
    mode = str(risk_control.get("mode") or "fixed")
    signal_source = str(risk_control.get("signal_source") or "score_strength")
    if mode == "fixed":
        return None, None
    if mode == "signal_strength":
        if signal_source == "validation_metric":
            if external_signal_values is None:
                raise ValueError("external_signal_values is required when risk_control.signal_source == 'validation_metric'")
            if isinstance(external_signal_values, dict):
                return build_combined_validation_metric_schedule(external_signal_values, risk_control=risk_control)
            return build_signal_schedule_from_series(external_signal_values, risk_control=risk_control)
        return build_signal_strength_schedule(
            transformed_score_matrix,
            risk_control=risk_control,
            topk=topk,
            min_score=min_score,
        )
    if mode == "benchmark_ma":
        return build_benchmark_ma_schedule(
            benchmark_returns,
            risk_control=risk_control,
            fallback_risk_degree=fallback_risk_degree,
        ), None
    if mode == "benchmark_ma_signal_strength":
        bench_schedule = build_benchmark_ma_schedule(
            benchmark_returns,
            risk_control=risk_control,
            fallback_risk_degree=fallback_risk_degree,
        )
        if signal_source == "validation_metric":
            if external_signal_values is None:
                raise ValueError("external_signal_values is required when risk_control.signal_source == 'validation_metric'")
            if isinstance(external_signal_values, dict):
                signal_schedule, signal_values = build_combined_validation_metric_schedule(
                    external_signal_values,
                    risk_control=risk_control,
                )
            else:
                signal_schedule, signal_values = build_signal_schedule_from_series(
                    external_signal_values,
                    risk_control=risk_control,
                )
        else:
            signal_schedule, signal_values = build_signal_strength_schedule(
                transformed_score_matrix,
                risk_control=risk_control,
                topk=topk,
                min_score=min_score,
            )
        combined = pd.concat([bench_schedule.rename("bench"), signal_schedule.rename("signal")], axis=1)
        return combined.min(axis=1).astype(float), signal_values
    raise ValueError(
        f"Unsupported risk control mode: {mode}. Supported: {', '.join(SUPPORTED_RISK_CONTROL_MODES)}"
    )
