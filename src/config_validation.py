"""Strict config/profile validation for native training pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_config
from src.feature_profiles import get_native_factor_store_dir
from src.label_utils import normalize_train_label_transform_mode

try:
    from src.data_source import resolve_data_source_name
except ModuleNotFoundError:
    from data_source import resolve_data_source_name  # type: ignore


TOP_LEVEL_SCHEMA = {
    "data": {
        "source": None,
        "parquet_dir": None,
    },
    "runtime": {
        "config_path": None,
    },
    "native": {
        "universe_dir": None,
    },
    "artifacts": {
        "enable_local_store": None,
        "store_dir": None,
    },
    "experiment": {
        "profile": None,
        "profile_path": None,
    },
    "features": {
        "profile": None,
        "selected_columns": None,
        "transforms": {
            "cross_sectional_rank": None,
        },
        "handler": None,
        "lookback": None,
        "use_valuation": None,
        "factor_store_dir": None,
        "cache_dir": None,
    },
    "model": {
        "name": None,
        "profile": None,
        "profile_path": None,
        "batch_size": None,
        "n_jobs": None,
        "early_stop": None,
        "hidden_size": None,
        "num_layers": None,
        "dropout": None,
        "lr": None,
        "epochs": None,
        "loss": None,
    },
    "lgbm": {
        "loss": None,
        "learning_rate": None,
        "num_boost_round": None,
        "early_stop": None,
        "early_stopping_metric": None,
        "early_stopping_min_delta": None,
        "train_weight_half_life": None,
        "train_weight_floor": None,
        "ranking_num_bins": None,
        "validation_topk": None,
        "log_evaluation_period": None,
        "colsample_bytree": None,
        "subsample": None,
        "subsample_freq": None,
        "lambda_l1": None,
        "lambda_l2": None,
        "max_depth": None,
        "num_leaves": None,
        "min_data_in_leaf": None,
        "num_threads": None,
        "seed": None,
        "eval_metric": None,
        "alpha": None,
    },
    "label": {
        "signal_horizon": None,
        "horizons": None,
        "horizon": None,
        "train_transform": {
            "mode": None,
            "neutral_band": None,
            "tail_band": None,
            "scale_multiplier": None,
            "min_scale": None,
        },
    },
    "rolling": {
        "retrain_step": None,
        "train_days": None,
        "valid_days": None,
    },
    "strategy": {
        "topk": None,
        "n_drop": None,
        "weighting": None,
        "score_transform": None,
        "score_zscore_clip": None,
        "max_weight": None,
        "keep_top_n": None,
        "min_score": None,
    },
    "backtest": {
        "rebalance_freq": None,
        "cost": {
            "buy": None,
            "sell": None,
        },
        "benchmark": {
            "mode": None,
            "path": None,
            "date_column": None,
            "value_column": None,
            "value_type": None,
            "name": None,
        },
        "slippage": None,
        "min_cost": None,
        "account": None,
        "risk_degree": None,
        "risk_control": {
            "mode": None,
            "risk_degree": None,
            "fast_window": None,
            "slow_window": None,
            "bull_risk": None,
            "neutral_risk": None,
            "bear_risk": None,
            "signal_metric": None,
            "min_signal": None,
            "max_signal": None,
            "min_signal_quantile": None,
            "max_signal_quantile": None,
            "min_risk": None,
            "max_risk": None,
        },
        "intraperiod_exit": {
            "mode": None,
            "score_source": None,
            "threshold": None,
            "calibration": None,
            "n_bins": None,
            "min_history": None,
            "price_confirm": {
                "mode": None,
                "ma_window": None,
                "min_remaining_steps": None,
                "force_exit_threshold": None,
            },
        },
        "dynamic_risk": {
            "mode": None,
            "fast_window": None,
            "slow_window": None,
            "bull_risk": None,
            "neutral_risk": None,
            "bear_risk": None,
            "signal_metric": None,
            "min_signal": None,
            "max_signal": None,
            "min_signal_quantile": None,
            "max_signal_quantile": None,
            "min_risk": None,
            "max_risk": None,
        },
    },
    "time": {
        "train": None,
        "valid": None,
        "test": None,
    },
    "universe": None,
}


def _validate_known_keys(mapping: dict[str, Any], schema: dict[str, Any], path: str = "config") -> None:
    unknown = sorted(set(mapping) - set(schema))
    if unknown:
        formatted = ", ".join(f"{path}.{key}" for key in unknown)
        raise ValueError(f"Unknown config keys: {formatted}")

    for key, subschema in schema.items():
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(subschema, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path}.{key} must be a mapping")
            _validate_known_keys(value, subschema, f"{path}.{key}")


def _expect_positive_int(value: Any, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out <= 0:
        raise ValueError(f"{field} must be > 0")
    return out


def _expect_nonnegative_int(value: Any, field: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if out < 0:
        raise ValueError(f"{field} must be >= 0")
    return out


def _expect_nonnegative_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if out < 0:
        raise ValueError(f"{field} must be >= 0")
    return out


def _expect_positive_float(value: Any, field: str) -> float:
    out = _expect_nonnegative_float(value, field)
    if out <= 0:
        raise ValueError(f"{field} must be > 0")
    return out


def _expect_split_dates(value: Any, field: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item date list")
    start = pd.Timestamp(value[0])
    end = pd.Timestamp(value[1])
    if start > end:
        raise ValueError(f"{field} start must be <= end")
    return start, end


def validate_training_config(
    cfg: dict[str, Any],
    *,
    check_paths: bool = True,
) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    _validate_known_keys(cfg, TOP_LEVEL_SCHEMA)

    experiment_profile = str(cfg.get("experiment", {}).get("profile") or "").strip()
    if not experiment_profile:
        raise ValueError("experiment.profile must be set explicitly")

    universe = str(cfg.get("universe") or "").strip()
    if not universe:
        raise ValueError("universe must be non-empty")
    resolve_data_source_name(cfg)

    features_cfg = cfg.get("features", {})
    feature_profile = str(features_cfg.get("profile") or "").strip()
    if not feature_profile:
        raise ValueError("features.profile must be non-empty")
    lookback = _expect_positive_int(features_cfg.get("lookback"), "features.lookback")

    selected_columns = features_cfg.get("selected_columns")
    if selected_columns is not None:
        if not isinstance(selected_columns, list) or not all(isinstance(item, str) and item for item in selected_columns):
            raise ValueError("features.selected_columns must be a list of non-empty strings")

    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name") or "").strip()
    if not model_name:
        raise ValueError("model.name must be non-empty")
    _expect_positive_int(model_cfg.get("batch_size"), "model.batch_size")
    _expect_positive_int(model_cfg.get("n_jobs"), "model.n_jobs")

    label_cfg = cfg.get("label", {})
    signal_horizon = _expect_positive_int(
        label_cfg.get("signal_horizon", label_cfg.get("horizon")),
        "label.signal_horizon",
    )
    horizons = label_cfg.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("label.horizons must be a non-empty list")
    horizon_values = [_expect_positive_int(value, "label.horizons[]") for value in horizons]
    if 1 not in horizon_values:
        raise ValueError("label.horizons must include 1 for realized backtest returns")
    if signal_horizon not in horizon_values:
        raise ValueError("label.signal_horizon must be included in label.horizons")
    train_transform_cfg = label_cfg.get("train_transform")
    if train_transform_cfg is not None:
        if not isinstance(train_transform_cfg, dict):
            raise ValueError("label.train_transform must be a mapping")
        mode = normalize_train_label_transform_mode(train_transform_cfg.get("mode"))
        train_transform_cfg["mode"] = mode
        neutral_band = float(train_transform_cfg.get("neutral_band", 0.0))
        if neutral_band < 0:
            raise ValueError("label.train_transform.neutral_band must be >= 0")
        train_transform_cfg["neutral_band"] = neutral_band
        default_tail_band = (neutral_band * 3.0) if neutral_band > 0 else 0.03
        tail_band = float(train_transform_cfg.get("tail_band", default_tail_band))
        if tail_band < neutral_band:
            raise ValueError("label.train_transform.tail_band must be >= label.train_transform.neutral_band")
        train_transform_cfg["tail_band"] = tail_band
        scale_multiplier = float(train_transform_cfg.get("scale_multiplier", 1.0))
        if scale_multiplier <= 0:
            raise ValueError("label.train_transform.scale_multiplier must be > 0")
        train_transform_cfg["scale_multiplier"] = scale_multiplier
        min_scale = float(train_transform_cfg.get("min_scale", 1e-4))
        if min_scale <= 0:
            raise ValueError("label.train_transform.min_scale must be > 0")
        train_transform_cfg["min_scale"] = min_scale

    rolling_cfg = cfg.get("rolling", {})
    retrain_step = _expect_positive_int(rolling_cfg.get("retrain_step"), "rolling.retrain_step")
    _expect_positive_int(rolling_cfg.get("train_days"), "rolling.train_days")
    _expect_positive_int(rolling_cfg.get("valid_days"), "rolling.valid_days")

    strategy_cfg = cfg.get("strategy", {})
    topk = _expect_positive_int(strategy_cfg.get("topk"), "strategy.topk")
    n_drop = _expect_nonnegative_int(strategy_cfg.get("n_drop"), "strategy.n_drop")
    if n_drop >= topk:
        raise ValueError("strategy.n_drop must be smaller than strategy.topk")
    weighting = str(strategy_cfg.get("weighting", "equal") or "equal").strip().lower()
    if weighting not in {"equal", "rank", "score_softmax"}:
        raise ValueError("strategy.weighting must be one of: equal, rank, score_softmax")
    strategy_cfg["weighting"] = weighting
    score_transform = str(strategy_cfg.get("score_transform", "none") or "none").strip().lower()
    if score_transform not in {"none", "rank_pct", "zscore_clip"}:
        raise ValueError("strategy.score_transform must be one of: none, rank_pct, zscore_clip")
    strategy_cfg["score_transform"] = score_transform
    score_zscore_clip = strategy_cfg.get("score_zscore_clip", 3.0)
    score_zscore_clip = float(score_zscore_clip)
    if score_zscore_clip <= 0:
        raise ValueError("strategy.score_zscore_clip must be > 0")
    strategy_cfg["score_zscore_clip"] = score_zscore_clip
    max_weight = strategy_cfg.get("max_weight")
    if max_weight is not None:
        max_weight = float(max_weight)
        if max_weight <= 0 or max_weight > 1:
            raise ValueError("strategy.max_weight must be in (0, 1] when provided")
        strategy_cfg["max_weight"] = max_weight
    keep_top_n = strategy_cfg.get("keep_top_n")
    if keep_top_n is not None:
        keep_top_n = _expect_positive_int(keep_top_n, "strategy.keep_top_n")
        if keep_top_n < topk:
            raise ValueError("strategy.keep_top_n must be >= strategy.topk when provided")
        strategy_cfg["keep_top_n"] = keep_top_n
    min_score = strategy_cfg.get("min_score")
    if min_score is not None:
        strategy_cfg["min_score"] = float(min_score)

    backtest_cfg = cfg.get("backtest", {})
    _expect_positive_int(backtest_cfg.get("rebalance_freq"), "backtest.rebalance_freq")
    cost_cfg = backtest_cfg.get("cost", {})
    _expect_nonnegative_float(cost_cfg.get("buy"), "backtest.cost.buy")
    _expect_nonnegative_float(cost_cfg.get("sell"), "backtest.cost.sell")
    _expect_nonnegative_float(backtest_cfg.get("slippage", 0.0), "backtest.slippage")
    _expect_nonnegative_float(backtest_cfg.get("min_cost", 0.0), "backtest.min_cost")
    account = float(backtest_cfg.get("account", 0.0))
    if account <= 0:
        raise ValueError("backtest.account must be > 0")
    risk_degree = float(backtest_cfg.get("risk_degree", 0.0))
    if risk_degree <= 0 or risk_degree > 1:
        raise ValueError("backtest.risk_degree must be in (0, 1]")
    risk_control_cfg = backtest_cfg.get("risk_control")
    dynamic_risk_cfg = backtest_cfg.get("dynamic_risk")
    if risk_control_cfg is not None and dynamic_risk_cfg is not None:
        raise ValueError("Use either backtest.risk_control or backtest.dynamic_risk, not both")
    raw_risk_cfg = risk_control_cfg if risk_control_cfg is not None else dynamic_risk_cfg
    validated_risk_cfg = {"mode": "fixed", "risk_degree": risk_degree}
    if raw_risk_cfg is not None:
        if not isinstance(raw_risk_cfg, dict):
            cfg_name = "backtest.risk_control" if risk_control_cfg is not None else "backtest.dynamic_risk"
            raise ValueError(f"{cfg_name} must be a mapping when provided")
        risk_mode = str(raw_risk_cfg.get("mode", "fixed") or "fixed").strip().lower()
        if risk_control_cfg is None and risk_mode == "none":
            risk_mode = "fixed"
        if risk_mode not in {"fixed", "benchmark_ma", "signal_strength", "benchmark_ma_signal_strength"}:
            raise ValueError(
                "backtest.risk_control.mode must be one of: fixed, benchmark_ma, signal_strength, benchmark_ma_signal_strength"
            )
        validated_risk_cfg["mode"] = risk_mode
        if risk_mode == "fixed":
            fixed_risk = float(raw_risk_cfg.get("risk_degree", risk_degree))
            if fixed_risk < 0 or fixed_risk > 1:
                raise ValueError("backtest.risk_control.risk_degree must be in [0, 1]")
            validated_risk_cfg["risk_degree"] = fixed_risk
        elif risk_mode in {"benchmark_ma", "benchmark_ma_signal_strength"}:
            fast_window = _expect_positive_int(raw_risk_cfg.get("fast_window", 120), "backtest.risk_control.fast_window")
            slow_window = _expect_positive_int(raw_risk_cfg.get("slow_window", 250), "backtest.risk_control.slow_window")
            if fast_window >= slow_window:
                raise ValueError("backtest.risk_control.fast_window must be smaller than backtest.risk_control.slow_window")
            bull_risk = float(raw_risk_cfg.get("bull_risk", risk_degree))
            neutral_risk = float(raw_risk_cfg.get("neutral_risk", min(risk_degree, 0.5)))
            bear_risk = float(raw_risk_cfg.get("bear_risk", 0.15))
            for field_name, value in (
                ("backtest.risk_control.bull_risk", bull_risk),
                ("backtest.risk_control.neutral_risk", neutral_risk),
                ("backtest.risk_control.bear_risk", bear_risk),
            ):
                if value < 0 or value > 1:
                    raise ValueError(f"{field_name} must be in [0, 1]")
            validated_risk_cfg.update(
                {
                    "fast_window": fast_window,
                    "slow_window": slow_window,
                    "bull_risk": bull_risk,
                    "neutral_risk": neutral_risk,
                    "bear_risk": bear_risk,
                }
            )
        if risk_mode in {"signal_strength", "benchmark_ma_signal_strength"}:
            signal_metric = str(raw_risk_cfg.get("signal_metric", "topk_mean") or "topk_mean").strip().lower()
            if signal_metric not in {"top1", "topk_mean", "topk_sum"}:
                raise ValueError("backtest.risk_control.signal_metric must be one of: top1, topk_mean, topk_sum")
            min_signal = float(raw_risk_cfg.get("min_signal", 0.0))
            max_signal = float(raw_risk_cfg.get("max_signal", 2.0))
            if max_signal <= min_signal:
                raise ValueError("backtest.risk_control.max_signal must be greater than backtest.risk_control.min_signal")
            min_signal_quantile = raw_risk_cfg.get("min_signal_quantile")
            max_signal_quantile = raw_risk_cfg.get("max_signal_quantile")
            if min_signal_quantile is not None:
                min_signal_quantile = float(min_signal_quantile)
                if min_signal_quantile < 0 or min_signal_quantile > 1:
                    raise ValueError("backtest.risk_control.min_signal_quantile must be in [0, 1]")
            if max_signal_quantile is not None:
                max_signal_quantile = float(max_signal_quantile)
                if max_signal_quantile < 0 or max_signal_quantile > 1:
                    raise ValueError("backtest.risk_control.max_signal_quantile must be in [0, 1]")
            if (
                min_signal_quantile is not None
                and max_signal_quantile is not None
                and max_signal_quantile <= min_signal_quantile
            ):
                raise ValueError(
                    "backtest.risk_control.max_signal_quantile must be greater than backtest.risk_control.min_signal_quantile"
                )
            min_risk = float(raw_risk_cfg.get("min_risk", min(risk_degree, 0.3)))
            max_risk = float(raw_risk_cfg.get("max_risk", risk_degree))
            for field_name, value in (
                ("backtest.risk_control.min_risk", min_risk),
                ("backtest.risk_control.max_risk", max_risk),
            ):
                if value < 0 or value > 1:
                    raise ValueError(f"{field_name} must be in [0, 1]")
            if max_risk < min_risk:
                raise ValueError("backtest.risk_control.max_risk must be >= backtest.risk_control.min_risk")
            validated_risk_cfg.update(
                {
                    "signal_metric": signal_metric,
                    "min_signal": min_signal,
                    "max_signal": max_signal,
                    "min_risk": min_risk,
                    "max_risk": max_risk,
                }
            )
            if min_signal_quantile is not None:
                validated_risk_cfg["min_signal_quantile"] = min_signal_quantile
            if max_signal_quantile is not None:
                validated_risk_cfg["max_signal_quantile"] = max_signal_quantile
    backtest_cfg["risk_control"] = validated_risk_cfg
    intraperiod_exit_cfg = backtest_cfg.get("intraperiod_exit")
    if intraperiod_exit_cfg is not None:
        if not isinstance(intraperiod_exit_cfg, dict):
            raise ValueError("backtest.intraperiod_exit must be a mapping when provided")
        exit_mode = str(intraperiod_exit_cfg.get("mode", "none") or "none").strip().lower()
        if exit_mode not in {"none", "score_threshold", "expected_return_threshold"}:
            raise ValueError(
                "backtest.intraperiod_exit.mode must be one of: none, score_threshold, expected_return_threshold"
            )
        intraperiod_exit_cfg["mode"] = exit_mode
        if exit_mode in {"score_threshold", "expected_return_threshold"}:
            score_source = str(intraperiod_exit_cfg.get("score_source", "raw") or "raw").strip().lower()
            if score_source not in {"raw", "transformed", "rank_pct", "zscore"}:
                raise ValueError(
                    "backtest.intraperiod_exit.score_source must be one of: raw, transformed, rank_pct, zscore"
                )
            intraperiod_exit_cfg["score_source"] = score_source
            intraperiod_exit_cfg["threshold"] = float(intraperiod_exit_cfg.get("threshold", 0.0))
        if exit_mode == "expected_return_threshold":
            calibration = str(intraperiod_exit_cfg.get("calibration", "quantile_bins") or "quantile_bins").strip().lower()
            if calibration not in {"quantile_bins"}:
                raise ValueError("backtest.intraperiod_exit.calibration must be one of: quantile_bins")
            intraperiod_exit_cfg["calibration"] = calibration
            intraperiod_exit_cfg["n_bins"] = max(int(intraperiod_exit_cfg.get("n_bins", 20) or 20), 2)
            intraperiod_exit_cfg["min_history"] = max(int(intraperiod_exit_cfg.get("min_history", 200) or 200), 1)
        price_confirm_cfg = intraperiod_exit_cfg.get("price_confirm")
        if price_confirm_cfg is not None:
            if not isinstance(price_confirm_cfg, dict):
                raise ValueError("backtest.intraperiod_exit.price_confirm must be a mapping when provided")
            confirm_mode = str(price_confirm_cfg.get("mode", "close_below_ma") or "close_below_ma").strip().lower()
            if confirm_mode not in {"close_below_ma"}:
                raise ValueError("backtest.intraperiod_exit.price_confirm.mode must be one of: close_below_ma")
            price_confirm_cfg["mode"] = confirm_mode
            price_confirm_cfg["ma_window"] = max(int(price_confirm_cfg.get("ma_window", 10) or 10), 1)
            price_confirm_cfg["min_remaining_steps"] = max(
                int(price_confirm_cfg.get("min_remaining_steps", 0) or 0),
                0,
            )
            force_exit_threshold = price_confirm_cfg.get("force_exit_threshold")
            if force_exit_threshold is not None:
                force_exit_threshold = float(force_exit_threshold)
                if exit_mode in {"score_threshold", "expected_return_threshold"} and force_exit_threshold > float(
                    intraperiod_exit_cfg["threshold"]
                ):
                    raise ValueError(
                        "backtest.intraperiod_exit.price_confirm.force_exit_threshold must be <= "
                        "backtest.intraperiod_exit.threshold"
                    )
                price_confirm_cfg["force_exit_threshold"] = force_exit_threshold
    benchmark_cfg = backtest_cfg.get("benchmark")
    if benchmark_cfg is not None:
        if not isinstance(benchmark_cfg, dict):
            raise ValueError("backtest.benchmark must be a mapping when provided")
        benchmark_mode = str(benchmark_cfg.get("mode", "cross_section_mean") or "cross_section_mean").strip().lower()
        if benchmark_mode not in {"cross_section_mean", "file"}:
            raise ValueError("backtest.benchmark.mode must be one of: cross_section_mean, file")
        benchmark_cfg["mode"] = benchmark_mode
        if benchmark_mode == "file":
            benchmark_path = str(benchmark_cfg.get("path") or "").strip()
            if not benchmark_path:
                raise ValueError("backtest.benchmark.path must be non-empty when benchmark.mode == 'file'")
            date_column = str(benchmark_cfg.get("date_column") or "date").strip()
            value_column = str(benchmark_cfg.get("value_column") or "close").strip()
            value_type = str(benchmark_cfg.get("value_type") or "close").strip().lower()
            if not date_column:
                raise ValueError("backtest.benchmark.date_column must be non-empty")
            if not value_column:
                raise ValueError("backtest.benchmark.value_column must be non-empty")
            if value_type not in {"return", "close"}:
                raise ValueError("backtest.benchmark.value_type must be one of: return, close")
            benchmark_cfg["path"] = benchmark_path
            benchmark_cfg["date_column"] = date_column
            benchmark_cfg["value_column"] = value_column
            benchmark_cfg["value_type"] = value_type

    time_cfg = cfg.get("time", {})
    train_start, train_end = _expect_split_dates(time_cfg.get("train"), "time.train")
    valid_start, valid_end = _expect_split_dates(time_cfg.get("valid"), "time.valid")
    test_start, test_end = _expect_split_dates(time_cfg.get("test"), "time.test")
    if not (train_end < valid_start and valid_end < test_start):
        raise ValueError("time.train, time.valid, and time.test must be strictly ordered and non-overlapping")

    if model_name == "lgbm":
        lgbm_cfg = cfg.get("lgbm")
        if not isinstance(lgbm_cfg, dict):
            raise ValueError("lgbm config block is required when model.name == 'lgbm'")
        early_stopping_metric = str(lgbm_cfg.get("early_stopping_metric", "default") or "default").strip().lower()
        if early_stopping_metric not in {
            "default",
            "daily_ic",
            "daily_rank_ic",
            "valid_topk_label_mean",
            "valid_topk_excess_mean",
        }:
            raise ValueError(
                "lgbm.early_stopping_metric must be one of: "
                "default, daily_ic, daily_rank_ic, valid_topk_label_mean, valid_topk_excess_mean"
            )
        lgbm_cfg["early_stopping_metric"] = early_stopping_metric
        _expect_positive_int(lgbm_cfg.get("num_boost_round"), "lgbm.num_boost_round")
        _expect_nonnegative_int(lgbm_cfg.get("early_stop"), "lgbm.early_stop")
        _expect_nonnegative_float(lgbm_cfg.get("early_stopping_min_delta", 0.0), "lgbm.early_stopping_min_delta")
        train_weight_half_life = lgbm_cfg.get("train_weight_half_life")
        if train_weight_half_life is not None:
            _expect_positive_float(train_weight_half_life, "lgbm.train_weight_half_life")
        train_weight_floor = lgbm_cfg.get("train_weight_floor")
        if train_weight_floor is not None:
            train_weight_floor = float(train_weight_floor)
            if train_weight_half_life is None:
                raise ValueError("lgbm.train_weight_floor requires lgbm.train_weight_half_life")
            if train_weight_floor < 0.0 or train_weight_floor >= 1.0:
                raise ValueError("lgbm.train_weight_floor must be in [0, 1)")
            lgbm_cfg["train_weight_floor"] = train_weight_floor
        ranking_num_bins = lgbm_cfg.get("ranking_num_bins")
        if ranking_num_bins is not None:
            ranking_num_bins = _expect_positive_int(ranking_num_bins, "lgbm.ranking_num_bins")
            if ranking_num_bins > 31:
                raise ValueError("lgbm.ranking_num_bins must be <= 31")
        validation_topk = lgbm_cfg.get("validation_topk")
        if validation_topk is not None:
            lgbm_cfg["validation_topk"] = _expect_positive_int(validation_topk, "lgbm.validation_topk")
        _expect_positive_int(lgbm_cfg.get("log_evaluation_period"), "lgbm.log_evaluation_period")
        _expect_positive_int(lgbm_cfg.get("num_threads"), "lgbm.num_threads")
        _expect_positive_int(lgbm_cfg.get("num_leaves"), "lgbm.num_leaves")
        _expect_positive_int(lgbm_cfg.get("min_data_in_leaf"), "lgbm.min_data_in_leaf")
        _expect_positive_int(max(1, lookback), "features.lookback")
        if retrain_step <= 0 or signal_horizon <= 0:
            raise ValueError("rolling.retrain_step and label.signal_horizon must be positive")

    if check_paths:
        factor_store_dir = Path(get_native_factor_store_dir(cfg))
        if not factor_store_dir.exists():
            raise ValueError(
                f"Resolved factor store does not exist: {factor_store_dir}. "
                "Build it first with src/gen_feature.py."
            )
        universe_dir = Path(cfg.get("native", {}).get("universe_dir", "data/universes"))
        if universe != "all":
            universe_path = universe_dir / f"{universe}.txt"
            if not universe_path.exists():
                raise ValueError(f"Universe file not found: {universe_path}")
        benchmark_cfg = cfg.get("backtest", {}).get("benchmark")
        if isinstance(benchmark_cfg, dict) and benchmark_cfg.get("mode") == "file":
            benchmark_path = Path(str(benchmark_cfg.get("path")))
            if not benchmark_path.exists():
                raise ValueError(f"Benchmark file not found: {benchmark_path}")

    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AI4Stock2 config/profile composition")
    parser.add_argument("--config", default="configs/config.yaml", help="Runtime config path")
    parser.add_argument("--experiment-profile", required=True, help="Named experiment profile to validate")
    parser.add_argument("--model-profile", help="Override model profile during validation")
    parser.add_argument("--feature-profile", help="Override feature profile during validation")
    parser.add_argument("--signal-horizon", type=int, help="Override signal horizon during validation")
    parser.add_argument("--retrain-step", type=int, help="Override retrain step during validation")
    parser.add_argument("--rebalance-freq", type=int, help="Override rebalance frequency during validation")
    parser.add_argument("--topk", type=int, help="Override strategy topk during validation")
    parser.add_argument("--n-drop", dest="n_drop", type=int, help="Override strategy n_drop during validation")
    parser.add_argument("--skip-path-checks", action="store_true", help="Skip factor store and universe filesystem checks")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(
        args.config,
        experiment_profile_name=args.experiment_profile,
        model_profile_name=args.model_profile,
    )
    if args.feature_profile:
        cfg.setdefault("features", {})
        cfg["features"]["profile"] = args.feature_profile
    if args.signal_horizon is not None:
        cfg.setdefault("label", {})
        cfg["label"]["signal_horizon"] = int(args.signal_horizon)
    if args.retrain_step is not None:
        cfg.setdefault("rolling", {})
        cfg["rolling"]["retrain_step"] = int(args.retrain_step)
    if args.rebalance_freq is not None:
        cfg.setdefault("backtest", {})
        cfg["backtest"]["rebalance_freq"] = int(args.rebalance_freq)
    if args.topk is not None:
        cfg.setdefault("strategy", {})
        cfg["strategy"]["topk"] = int(args.topk)
    if args.n_drop is not None:
        cfg.setdefault("strategy", {})
        cfg["strategy"]["n_drop"] = int(args.n_drop)

    validate_training_config(cfg, check_paths=not args.skip_path_checks)
    print("Config validation passed.")


if __name__ == "__main__":
    main()
