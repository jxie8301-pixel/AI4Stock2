"""Native Modular Rolling Retrain Pipeline for AI4Stock2."""

from __future__ import annotations

import argparse

from src.config_loader import load_config
from src.config_validation import validate_training_config
from src.experiment_store import (
    prepare_run_store,
    resolve_retrain_step,
)
from src.label_utils import get_label_column_name, resolve_signal_horizon
from src.rolling_artifacts import (
    build_paths as _build_paths,
    ensure_output_dirs as _ensure_output_dirs,
    load_prediction_bundle,
    resolve_prediction_artifact_dir as _resolve_prediction_artifact_dir,
    write_prediction_bundle as _write_prediction_bundle,
)
from src.rolling_baselines import (
    build_average_factor_baseline_predictions as _build_average_factor_baseline_predictions,
    build_sign_aligned_factor_baseline_predictions as _build_sign_aligned_factor_baseline_predictions,
)
from src.rolling_evaluate import evaluate_prediction_bundle
from src.rolling_runtime import load_rolling_runtime_data
from src.rolling_train import generate_prediction_bundle
from src.rolling_types import (
    PREDICTION_ARTIFACT_DIRNAME,
    PREDICTION_METADATA_FILENAME,
    PredictionBundle,
    RollingPaths,
    RollingRuntimeData,
)
from src.runtime_cli import add_common_runtime_args, apply_common_runtime_overrides, load_validated_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Rolling Pipeline")
    add_common_runtime_args(parser, include_model_arg=True)
    parser.add_argument("--retrain-step", type=int, help="Rolling retrain step in trading days. If omitted, use config value.")
    parser.add_argument("--horizon", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--train-days", type=int, help="Training window length in trading days. If omitted, use config value.")
    parser.add_argument("--valid-days", type=int, help="Validation window length in trading days. If omitted, use config value.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id")
    parser.add_argument("--save-models", action="store_true", help="Save models for each rolling step")
    parser.add_argument("--load-models", action="store_true", help="Load existing models for each rolling step")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Persist final rolling predictions and labels for later backtest reuse.",
    )
    parser.add_argument(
        "--load-predictions-dir",
        help=(
            "Reuse an existing rolling prediction bundle directory and skip training/inference. "
            f"Accepts either a run directory or a direct {PREDICTION_ARTIFACT_DIRNAME}/ path."
        ),
    )
    return parser


def run_rolling_pipeline() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.load_predictions_dir:
        try:
            cfg = load_config(
                args.config,
                experiment_profile_name=getattr(args, "experiment_profile", None),
                model_profile_name=getattr(args, "model_profile", None),
            )
            apply_common_runtime_overrides(cfg, args, parser, allow_rolling_overrides=True)
            validate_training_config(cfg, check_paths=False)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        cfg = load_validated_config_from_args(args, parser, allow_rolling_overrides=True)

    retrain_step = int(resolve_retrain_step(cfg, args))
    train_days = int(cfg.get("rolling", {}).get("train_days", args.train_days or 242))
    valid_days = int(cfg.get("rolling", {}).get("valid_days", args.valid_days or 10))
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_column = get_label_column_name(signal_horizon)
    backtest_label_column = get_label_column_name(1)
    model_name = cfg["model"]["name"]

    run_store = prepare_run_store(
        cfg,
        args,
        backend="native",
        pipeline="rolling",
        model_name=model_name,
        model_ext=".pt" if model_name != "lgbm" else ".pkl",
    )
    paths = _build_paths(run_store, model_name)
    _ensure_output_dirs(paths, save_models=args.save_models, load_models=args.load_models, model_name=model_name)

    print(f"\n>>> Running Native Rolling Pipeline (Backend: NATIVE) <<<")
    if args.load_predictions_dir:
        bundle = load_prediction_bundle(args.load_predictions_dir)
        print(f"[*] Loaded prediction bundle from {_resolve_prediction_artifact_dir(args.load_predictions_dir)}")
    else:
        runtime_data = load_rolling_runtime_data(
            cfg,
            train_days=train_days,
            valid_days=valid_days,
            label_column=label_column,
            backtest_label_column=backtest_label_column,
        )
        bundle = generate_prediction_bundle(
            cfg,
            args,
            runtime_data,
            paths,
            retrain_step=retrain_step,
            train_days=train_days,
            valid_days=valid_days,
            signal_horizon=signal_horizon,
            model_name=model_name,
        )
        if args.save_predictions:
            _write_prediction_bundle(bundle, paths.prediction_artifact_dir)
            print(f"Prediction bundle saved: {paths.prediction_artifact_dir}")

    if args.load_predictions_dir and args.save_predictions:
        _write_prediction_bundle(bundle, paths.prediction_artifact_dir)
        print(f"Prediction bundle copied to current run: {paths.prediction_artifact_dir}")

    evaluate_prediction_bundle(
        cfg,
        args,
        paths,
        run_store,
        bundle,
        model_name=model_name,
    )


if __name__ == "__main__":
    run_rolling_pipeline()
