"""Shared types and constants for the native rolling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PREDICTION_ARTIFACT_DIRNAME = "prediction_artifacts"
PREDICTIONS_FILENAME = "final_predictions.parquet"
SIGNAL_LABELS_FILENAME = "signal_labels.parquet"
BACKTEST_LABELS_FILENAME = "backtest_labels.parquet"
AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME = "avg_factor_baseline_predictions.parquet"
SIGN_ALIGNED_FACTOR_BASELINE_PREDICTIONS_FILENAME = "sign_aligned_factor_baseline_predictions.parquet"
RANK_AVG_FACTOR_BASELINE_PREDICTIONS_FILENAME = "rank_avg_factor_baseline_predictions.parquet"
RANK_IC_WEIGHTED_FACTOR_BASELINE_PREDICTIONS_FILENAME = "rank_ic_weighted_factor_baseline_predictions.parquet"
PREDICTION_METADATA_FILENAME = "metadata.json"
TRAINING_SUMMARY_FILENAME = "training_summary.csv"


@dataclass(frozen=True)
class RollingPaths:
    results_dir: Path
    models_dir: Path
    importance_dir: Path
    training_history_dir: Path
    prediction_artifact_dir: Path


@dataclass(frozen=True)
class RollingRuntimeData:
    factor_frame: pd.DataFrame
    dt_index: pd.Series
    y: np.ndarray
    backtest_y: np.ndarray
    full_calendar: pd.Series
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_calendar: pd.Series
    selected_feature_names: list[str]
    selected_feature_sources: list[str]
    finite_feature_mask: np.ndarray
    lookback: int
    batch_size: int


@dataclass(frozen=True)
class PredictionBundle:
    final_predictions: pd.Series
    label_series: pd.Series
    backtest_label_series: pd.Series
    avg_factor_baseline_predictions: pd.Series | None
    sign_aligned_factor_baseline_predictions: pd.Series | None
    selected_feature_names: list[str]
    metadata: dict[str, Any]
    feature_importance_frames: list[pd.DataFrame]
    training_summary_records: list[dict[str, Any]]
    rank_avg_factor_baseline_predictions: pd.Series | None = None
    rank_ic_weighted_factor_baseline_predictions: pd.Series | None = None
