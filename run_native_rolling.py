"""Native Modular Rolling Retrain Pipeline for AI4Stock2."""

from __future__ import annotations

import argparse

from src.experiment_store import (
    prepare_run_store,
    resolve_retrain_step,
)
from src.label_utils import get_label_column_name, resolve_label_embargo_days, resolve_signal_horizon
from src.rolling_artifacts import (
    build_paths,
    ensure_output_dirs,
    load_prediction_bundle,
    resolve_prediction_artifact_dir,
    write_prediction_bundle,
)
from src.rolling_evaluate import evaluate_prediction_bundle
from src.prediction_fusion import fuse_prediction_bundle, resolve_score_fusion_cfg
from src.rolling_runtime import load_rolling_runtime_data
from src.rolling_train import generate_prediction_bundle
from src.rolling_types import PREDICTION_ARTIFACT_DIRNAME
from src.runtime_cli import add_common_runtime_args, load_validated_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI4Stock2 Native Rolling Pipeline")
    add_common_runtime_args(parser, include_model_arg=True)
    parser.add_argument(
        "--retrain-step",
        type=int,
        help="Rolling retrain step in trading days. If omitted, use config value.",
    )
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
        cfg = load_validated_config_from_args(
            args,
            parser,
            allow_rolling_overrides=True,
            check_paths=False,
        )
    else:
        cfg = load_validated_config_from_args(args, parser, allow_rolling_overrides=True)

    retrain_step = int(resolve_retrain_step(cfg, args))
    train_days = int(cfg.get("rolling", {}).get("train_days", args.train_days or 242))
    valid_days = int(cfg.get("rolling", {}).get("valid_days", args.valid_days or 10))
    signal_horizon = int(resolve_signal_horizon(cfg))
    label_embargo_days = int(resolve_label_embargo_days(cfg, signal_horizon=signal_horizon))
    label_column = get_label_column_name(signal_horizon)
    backtest_label_column = get_label_column_name(1)
    model_name = cfg["model"]["name"]

    model_ext = ".pkl" if model_name == "lgbm" else ".json" if model_name == "formula_score" else ".pt"
    run_store = prepare_run_store(
        cfg,
        args,
        backend="native",
        pipeline="rolling",
        model_name=model_name,
        model_ext=model_ext,
    )
    paths = build_paths(run_store, model_name)
    ensure_output_dirs(paths, save_models=args.save_models, load_models=args.load_models, model_name=model_name)

    print(f"\n>>> Running Native Rolling Pipeline (Backend: NATIVE) <<<")
    if args.load_predictions_dir:
        bundle = load_prediction_bundle(args.load_predictions_dir)
        print(f"[*] Loaded prediction bundle from {resolve_prediction_artifact_dir(args.load_predictions_dir)}")
    else:
        runtime_data = load_rolling_runtime_data(
            cfg,
            train_days=train_days,
            valid_days=valid_days,
            label_column=label_column,
            backtest_label_column=backtest_label_column,
            label_embargo_days=label_embargo_days,
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
            label_embargo_days=label_embargo_days,
            model_name=model_name,
        )
        if args.save_predictions:
            write_prediction_bundle(bundle, paths.prediction_artifact_dir)
            print(f"Prediction bundle saved: {paths.prediction_artifact_dir}")

    if args.load_predictions_dir and args.save_predictions:
        write_prediction_bundle(bundle, paths.prediction_artifact_dir)
        print(f"Prediction bundle copied to current run: {paths.prediction_artifact_dir}")

    fusion_cfg = resolve_score_fusion_cfg(cfg)
    if fusion_cfg["enabled"]:
        secondary_dir = fusion_cfg["secondary_predictions_dir"]
        secondary_bundle = load_prediction_bundle(secondary_dir)
        bundle = fuse_prediction_bundle(
            bundle,
            secondary_bundle,
            fusion_cfg=fusion_cfg,
        )
        print(
            "[*] Applied score fusion: "
            f"mode={fusion_cfg['mode']} "
            f"primary_transform={fusion_cfg['primary_transform']} "
            f"secondary_transform={fusion_cfg['secondary_transform']} "
            f"secondary={secondary_dir}"
        )
        if args.save_predictions:
            write_prediction_bundle(bundle, paths.prediction_artifact_dir)
            print(f"Fused prediction bundle saved: {paths.prediction_artifact_dir}")

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
