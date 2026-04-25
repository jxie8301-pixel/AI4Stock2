"""Shared CLI helpers for native single/rolling entrypoints."""

from __future__ import annotations

import argparse
from typing import Any

from src.config_loader import load_config, load_runtime_config
from src.config_validation import validate_training_config
from src.data_source import SUPPORTED_DATA_SOURCES
from src.override_utils import apply_override_args


def add_common_runtime_args(parser: argparse.ArgumentParser, *, include_model_arg: bool = True) -> None:
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument(
        "--config-is-snapshot",
        action="store_true",
        help="Treat --config as a fully resolved config snapshot and skip experiment/model profile composition.",
    )
    if include_model_arg:
        parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--experiment-profile", help="Named experiment profile from configs/experiment_profiles.yaml")
    parser.add_argument("--feature-profile", help="Override features.profile for this run")
    parser.add_argument("--model-profile", help="Override model.profile for this run")
    parser.add_argument("--data-source", choices=SUPPORTED_DATA_SOURCES, help="Override runtime data source")
    parser.add_argument(
        "--set",
        action="append",
        dest="set_overrides",
        help="Generic dotted override in key=value form, for example strategy.topk=20",
    )
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    parser.add_argument("--topk", type=int, help="Override strategy top-k holdings")
    parser.add_argument("--n-drop", dest="n_drop", type=int, help="Override strategy daily replacement count")
    parser.add_argument("--run-tag", help="Short label for local experiment storage/comparison")
    parser.add_argument("--store-dir", help="Override local experiment store root")
    parser.add_argument("--disable-local-store", action="store_true", help="Disable automatic local experiment/model storage")
    parser.add_argument("--rebalance-freq", type=int, help="Override backtest rebalance frequency in days. If omitted, use config value.")
    parser.add_argument("--signal-horizon", type=int, help="Prediction signal horizon in trading days. If omitted, use experiment profile.")
    parser.add_argument("--label-horizon", type=int, help=argparse.SUPPRESS)


def apply_common_runtime_overrides(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    allow_rolling_overrides: bool = False,
) -> dict[str, Any]:
    if getattr(args, "model", None):
        cfg.setdefault("model", {})
        cfg["model"]["name"] = args.model
    if getattr(args, "data_source", None):
        cfg.setdefault("data", {})
        cfg["data"]["source"] = args.data_source
    apply_override_args(cfg, getattr(args, "set_overrides", None))

    feature_profile_override = getattr(args, "feature_profile", None) or getattr(args, "profile", None)
    if feature_profile_override:
        cfg.setdefault("features", {})
        cfg["features"]["profile"] = feature_profile_override

    if getattr(args, "topk", None) is not None:
        cfg.setdefault("strategy", {})
        cfg["strategy"]["topk"] = int(args.topk)
    if getattr(args, "n_drop", None) is not None:
        cfg.setdefault("strategy", {})
        cfg["strategy"]["n_drop"] = int(args.n_drop)

    signal_horizon_override = getattr(args, "signal_horizon", None)
    if signal_horizon_override is None:
        signal_horizon_override = getattr(args, "label_horizon", None)
    if signal_horizon_override is not None:
        cfg.setdefault("label", {})
        cfg["label"]["signal_horizon"] = int(signal_horizon_override)

    if getattr(args, "rebalance_freq", None) is not None:
        cfg.setdefault("backtest", {})
        cfg["backtest"]["rebalance_freq"] = int(args.rebalance_freq)

    if allow_rolling_overrides:
        retrain_step = getattr(args, "retrain_step", None)
        horizon = getattr(args, "horizon", None)
        if retrain_step is not None and horizon is not None and int(retrain_step) != int(horizon):
            parser.error("--retrain-step and --horizon refer to the same parameter; use one value.")
        if retrain_step is None and horizon is not None:
            retrain_step = horizon
        if retrain_step is not None:
            cfg.setdefault("rolling", {})
            cfg["rolling"]["retrain_step"] = int(retrain_step)
        if getattr(args, "train_days", None) is not None:
            cfg.setdefault("rolling", {})
            cfg["rolling"]["train_days"] = int(args.train_days)
        if getattr(args, "valid_days", None) is not None:
            cfg.setdefault("rolling", {})
            cfg["rolling"]["valid_days"] = int(args.valid_days)

    return cfg


def load_validated_config_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    allow_rolling_overrides: bool = False,
    check_paths: bool = True,
) -> dict[str, Any]:
    try:
        if getattr(args, "config_is_snapshot", False):
            cfg = load_runtime_config(args.config)
            cfg.setdefault("runtime", {})
            cfg["runtime"]["config_path"] = args.config
        else:
            cfg = load_config(
                args.config,
                experiment_profile_name=getattr(args, "experiment_profile", None),
                model_profile_name=getattr(args, "model_profile", None),
            )
        apply_common_runtime_overrides(
            cfg,
            args,
            parser,
            allow_rolling_overrides=allow_rolling_overrides,
        )
        validate_training_config(cfg, check_paths=check_paths)
        return cfg
    except ValueError as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable")
