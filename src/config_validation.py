"""Strict config/profile validation for native training pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_config
from src.feature_profiles import get_native_factor_store_dir

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
        "early_stopping_min_delta": None,
        "train_weight_half_life": None,
        "ranking_num_bins": None,
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
        "slippage": None,
        "min_cost": None,
        "account": None,
        "risk_degree": None,
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
        _expect_positive_int(lgbm_cfg.get("num_boost_round"), "lgbm.num_boost_round")
        _expect_nonnegative_int(lgbm_cfg.get("early_stop"), "lgbm.early_stop")
        _expect_nonnegative_float(lgbm_cfg.get("early_stopping_min_delta", 0.0), "lgbm.early_stopping_min_delta")
        train_weight_half_life = lgbm_cfg.get("train_weight_half_life")
        if train_weight_half_life is not None:
            _expect_positive_float(train_weight_half_life, "lgbm.train_weight_half_life")
        ranking_num_bins = lgbm_cfg.get("ranking_num_bins")
        if ranking_num_bins is not None:
            ranking_num_bins = _expect_positive_int(ranking_num_bins, "lgbm.ranking_num_bins")
            if ranking_num_bins > 31:
                raise ValueError("lgbm.ranking_num_bins must be <= 31")
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
