"""Prediction artifact and output-path helpers for the native rolling pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rolling_types import (
    AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME,
    PREDICTION_ARTIFACT_DIRNAME,
    PREDICTION_METADATA_FILENAME,
    PREDICTIONS_FILENAME,
    BACKTEST_LABELS_FILENAME,
    RANK_AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME,
    RANK_IC_WEIGHTED_FACTOR_BASELINE_PREDICTIONS_FILENAME,
    SIGNAL_LABELS_FILENAME,
    SIGN_ALIGNED_FACTOR_BASELINE_PREDICTIONS_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    PredictionBundle,
    RollingPaths,
)


def build_paths(run_store, model_name: str) -> RollingPaths:
    if run_store.enabled and run_store.run_dir:
        results_dir = run_store.run_dir
        models_dir = run_store.models_dir or (results_dir / "models")
    else:
        results_dir = Path("results") / f"native_rolling_{model_name}"
        models_dir = results_dir / "models"
    return RollingPaths(
        results_dir=results_dir,
        models_dir=models_dir,
        importance_dir=results_dir / "feature_importance",
        training_history_dir=results_dir / "training_history",
        prediction_artifact_dir=results_dir / PREDICTION_ARTIFACT_DIRNAME,
    )


def ensure_output_dirs(paths: RollingPaths, *, save_models: bool, load_models: bool, model_name: str) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    if save_models or load_models:
        paths.models_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "lgbm":
        paths.importance_dir.mkdir(parents=True, exist_ok=True)
        paths.training_history_dir.mkdir(parents=True, exist_ok=True)


def _series_to_frame(series: pd.Series, value_column: str) -> pd.DataFrame:
    frame = series.rename(value_column).rename_axis(index=["datetime", "instrument"]).reset_index()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["instrument"] = frame["instrument"].astype(str)
    return frame


def _frame_to_series(frame: pd.DataFrame, value_column: str) -> pd.Series:
    required = {"datetime", "instrument", value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction artifact is missing columns: {', '.join(sorted(missing))}")
    return pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(frame["datetime"]),
                frame["instrument"].astype(str),
            ],
            names=["datetime", "instrument"],
        ),
        name=value_column,
    ).sort_index()


def write_prediction_bundle(bundle: PredictionBundle, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _series_to_frame(bundle.final_predictions, "prediction").to_parquet(artifact_dir / PREDICTIONS_FILENAME, index=False)
    _series_to_frame(bundle.label_series, "label").to_parquet(artifact_dir / SIGNAL_LABELS_FILENAME, index=False)
    _series_to_frame(bundle.backtest_label_series, "label").to_parquet(
        artifact_dir / BACKTEST_LABELS_FILENAME,
        index=False,
    )
    if bundle.avg_factor_baseline_predictions is not None:
        _series_to_frame(bundle.avg_factor_baseline_predictions, "prediction").to_parquet(
            artifact_dir / AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME,
            index=False,
        )
    if bundle.sign_aligned_factor_baseline_predictions is not None:
        _series_to_frame(bundle.sign_aligned_factor_baseline_predictions, "prediction").to_parquet(
            artifact_dir / SIGN_ALIGNED_FACTOR_BASELINE_PREDICTIONS_FILENAME,
            index=False,
        )
    if bundle.rank_avg_factor_baseline_predictions is not None:
        _series_to_frame(bundle.rank_avg_factor_baseline_predictions, "prediction").to_parquet(
            artifact_dir / RANK_AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME,
            index=False,
        )
    if bundle.rank_ic_weighted_factor_baseline_predictions is not None:
        _series_to_frame(bundle.rank_ic_weighted_factor_baseline_predictions, "prediction").to_parquet(
            artifact_dir / RANK_IC_WEIGHTED_FACTOR_BASELINE_PREDICTIONS_FILENAME,
            index=False,
        )
    if bundle.training_summary_records:
        pd.DataFrame(bundle.training_summary_records).to_csv(
            artifact_dir / TRAINING_SUMMARY_FILENAME,
            index=False,
        )
    metadata = {
        **bundle.metadata,
        "selected_features": list(bundle.selected_feature_names),
    }
    with open(artifact_dir / PREDICTION_METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def resolve_prediction_artifact_dir(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_dir() and (path / PREDICTION_METADATA_FILENAME).exists():
        return path
    nested = path / PREDICTION_ARTIFACT_DIRNAME
    if nested.is_dir() and (nested / PREDICTION_METADATA_FILENAME).exists():
        return nested
    raise FileNotFoundError(
        f"Prediction artifact directory not found under {path}. "
        f"Expected {PREDICTION_METADATA_FILENAME} inside the directory or its {PREDICTION_ARTIFACT_DIRNAME}/ child."
    )


def load_prediction_bundle(raw_path: str | Path) -> PredictionBundle:
    artifact_dir = resolve_prediction_artifact_dir(raw_path)
    with open(artifact_dir / PREDICTION_METADATA_FILENAME, encoding="utf-8") as f:
        metadata = json.load(f)
    final_predictions = _frame_to_series(pd.read_parquet(artifact_dir / PREDICTIONS_FILENAME), "prediction")
    label_series = _frame_to_series(pd.read_parquet(artifact_dir / SIGNAL_LABELS_FILENAME), "label")
    backtest_label_series = _frame_to_series(pd.read_parquet(artifact_dir / BACKTEST_LABELS_FILENAME), "label")
    avg_factor_path = artifact_dir / AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME
    sign_aligned_factor_path = artifact_dir / SIGN_ALIGNED_FACTOR_BASELINE_PREDICTIONS_FILENAME
    rank_avg_factor_path = artifact_dir / RANK_AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME
    rank_ic_weighted_factor_path = artifact_dir / RANK_IC_WEIGHTED_FACTOR_BASELINE_PREDICTIONS_FILENAME
    avg_factor_baseline_predictions = (
        _frame_to_series(pd.read_parquet(avg_factor_path), "prediction")
        if avg_factor_path.exists()
        else None
    )
    sign_aligned_factor_baseline_predictions = (
        _frame_to_series(pd.read_parquet(sign_aligned_factor_path), "prediction")
        if sign_aligned_factor_path.exists()
        else None
    )
    rank_avg_factor_baseline_predictions = (
        _frame_to_series(pd.read_parquet(rank_avg_factor_path), "prediction")
        if rank_avg_factor_path.exists()
        else None
    )
    rank_ic_weighted_factor_baseline_predictions = (
        _frame_to_series(pd.read_parquet(rank_ic_weighted_factor_path), "prediction")
        if rank_ic_weighted_factor_path.exists()
        else None
    )
    training_summary_path = artifact_dir / TRAINING_SUMMARY_FILENAME
    training_summary_records = (
        pd.read_csv(training_summary_path).to_dict(orient="records")
        if training_summary_path.exists()
        else []
    )
    return PredictionBundle(
        final_predictions=final_predictions,
        label_series=label_series,
        backtest_label_series=backtest_label_series,
        avg_factor_baseline_predictions=avg_factor_baseline_predictions,
        sign_aligned_factor_baseline_predictions=sign_aligned_factor_baseline_predictions,
        selected_feature_names=list(metadata.get("selected_features", [])),
        metadata=metadata,
        feature_importance_frames=[],
        training_summary_records=training_summary_records,
        rank_avg_factor_baseline_predictions=rank_avg_factor_baseline_predictions,
        rank_ic_weighted_factor_baseline_predictions=rank_ic_weighted_factor_baseline_predictions,
    )
