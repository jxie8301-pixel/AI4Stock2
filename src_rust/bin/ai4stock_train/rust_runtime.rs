use super::LgbmBundleOptions;
use ai4stock2_native::common::benchmark::{
    cross_section_mean_returns, load_file_benchmark_returns,
};
use ai4stock2_native::common::parquet::{
    date_value_ns, discover_bucket_parquet_paths, format_date_ns, numeric_value,
    open_projected_parquet_reader, parse_datetime_ns as parse_date_ns, required_column,
    required_string_value as string_value,
};
use ai4stock2_native::common::profiles::default_factor_store_dir;
use ai4stock2_native::common::python::prepare_python_path;
use ai4stock2_native::common::yaml::{
    deep_merge_yaml, parse_key_value_arg, read_yaml_file, set_yaml_dotted, yaml_path,
    yaml_path_bool, yaml_path_f64, yaml_path_string, yaml_path_usize, yaml_string_sequence,
    yaml_usize,
};
use arrow_array::{Array, Float64Array, RecordBatch, StringArray, TimestampNanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use parquet::arrow::ArrowWriter;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const PREDICTION_ARTIFACT_DIRNAME: &str = "prediction_artifacts";
const PREDICTION_METADATA_FILENAME: &str = "metadata.json";
const PREDICTIONS_FILENAME: &str = "final_predictions.parquet";
const SIGNAL_LABELS_FILENAME: &str = "signal_labels.parquet";
const BACKTEST_LABELS_FILENAME: &str = "backtest_labels.parquet";
const TRAINING_SUMMARY_FILENAME: &str = "training_summary.csv";
const DEFAULT_LGBM_BUNDLE_TIMING_FILENAME: &str = "lgbm_bundle_timing.json";
const DEFAULT_LABEL_ABS_CAP: f64 = 0.35;
type DateInterval = (Option<i64>, Option<i64>);
type IntervalsBySymbol = BTreeMap<String, Vec<DateInterval>>;

#[derive(Debug, Clone)]
struct ResolvedRuntimeConfig {
    cfg: YamlValue,
    factor_store: PathBuf,
    data_source: String,
    label_column: String,
    backtest_label_column: String,
    signal_horizon: usize,
    train_days: usize,
    valid_days: usize,
    retrain_step: usize,
    label_embargo_days: usize,
    test_start_ns: i64,
    test_end_ns: i64,
    selected_feature_names: Vec<String>,
    selected_feature_sources: Vec<String>,
    cross_sectional_rank: bool,
    cross_sectional_rank_exclude_columns: BTreeSet<String>,
    universe_name: Option<String>,
    universe_dir: PathBuf,
    lgbm_config: JsonValue,
    train_label_transform: TrainLabelTransformConfig,
    opportunity: OpportunityLabelConfig,
    sample_weight: SampleWeightConfig,
}

#[derive(Debug, Clone)]
struct FactorRow {
    date_ns: i64,
    symbol: String,
    label: f64,
    backtest_label: f64,
    features: Vec<f64>,
}

#[derive(Debug, Clone)]
struct LoadedFactorData {
    rows: Vec<FactorRow>,
    full_calendar: Vec<i64>,
    test_calendar: Vec<i64>,
    selected_feature_names: Vec<String>,
    selected_feature_sources: Vec<String>,
}

#[derive(Debug, Clone)]
struct UniverseFilter {
    intervals_by_symbol: IntervalsBySymbol,
}

#[derive(Debug, Clone)]
struct RollingWindow {
    step_index: usize,
    test_start_ns: i64,
    test_end_ns: i64,
    train_start_ns: i64,
    train_end_ns: i64,
    valid_start_ns: i64,
    valid_end_ns: i64,
}

#[derive(Debug, Clone)]
struct LongValueRow {
    date_ns: i64,
    instrument: String,
    value: f64,
}

#[derive(Debug, Clone)]
struct WindowDataPaths {
    train_path: PathBuf,
    valid_path: PathBuf,
    test_path: PathBuf,
    prediction_path: PathBuf,
    model_path: PathBuf,
    importance_path: PathBuf,
    history_path: PathBuf,
}

#[derive(Debug, Clone)]
struct PythonWindowTrainResult {
    summary: JsonValue,
    valid_predictions: Vec<f64>,
    test_predictions: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PreparedPythonLgbmWindow {
    train_rows: usize,
    valid_rows: usize,
    test_rows: usize,
    train_feature_bytes: Vec<u8>,
    valid_feature_bytes: Vec<u8>,
    test_feature_bytes: Vec<u8>,
    train_label_bytes: Vec<u8>,
    valid_label_bytes: Vec<u8>,
    raw_valid_label_bytes: Vec<u8>,
    train_date_ns_bytes: Vec<u8>,
    valid_date_ns_bytes: Vec<u8>,
    train_sample_weight_bytes: Option<Vec<u8>>,
    valid_sample_weight_bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct TrainLabelTransformConfig {
    mode: String,
    neutral_band: f64,
    tail_band: f64,
    scale_multiplier: f64,
    min_scale: f64,
}

#[derive(Debug, Clone)]
struct OpportunityLabelConfig {
    mode: String,
    threshold: f64,
    neutral_band: f64,
}

#[derive(Debug, Clone)]
struct SampleWeightConfig {
    mode: String,
    power: f64,
    scale: Option<f64>,
    min: f64,
    date_normalize: bool,
}

#[derive(Debug, Clone, Default)]
struct TrainingContext {
    industry_by_symbol: BTreeMap<String, String>,
    benchmark_forward_returns: BTreeMap<i64, f64>,
}

#[derive(Debug, Clone)]
struct PreparedWindowLabelColumns {
    label: Vec<f64>,
    raw_label: Vec<f64>,
    sample_weight: Vec<f64>,
    opportunity_label: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
struct TimingPhase {
    seconds: f64,
    count: usize,
    rows: usize,
    windows: usize,
    files: usize,
}

#[derive(Debug, Clone)]
struct RuntimePhaseTimer {
    started: Instant,
    phases: BTreeMap<String, TimingPhase>,
}

impl RuntimePhaseTimer {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            phases: BTreeMap::new(),
        }
    }

    fn add(
        &mut self,
        name: &str,
        seconds: f64,
        count: usize,
        rows: usize,
        windows: usize,
        files: usize,
    ) {
        let phase = self.phases.entry(name.to_owned()).or_default();
        phase.seconds += seconds;
        phase.count += count;
        phase.rows += rows;
        phase.windows += windows;
        phase.files += files;
    }

    fn as_json(&self, path: &Path) -> JsonValue {
        let phases = self
            .phases
            .iter()
            .map(|(name, phase)| {
                (
                    name.clone(),
                    serde_json::json!({
                        "seconds": round_six(phase.seconds),
                        "count": phase.count,
                        "rows": phase.rows,
                        "windows": phase.windows,
                        "files": phase.files,
                    }),
                )
            })
            .collect::<serde_json::Map<String, JsonValue>>();
        serde_json::json!({
            "wall_seconds": round_six(self.started.elapsed().as_secs_f64()),
            "semantics": "Window phase seconds are summed across windows and can exceed wall_seconds when phases overlap logically.",
            "artifact": {
                "path": path.display().to_string(),
                "default_filename": DEFAULT_LGBM_BUNDLE_TIMING_FILENAME,
            },
            "phases": phases,
        })
    }
}

pub(crate) fn make_bundle_lgbm_rust_runtime(
    options: &LgbmBundleOptions,
) -> Result<JsonValue, String> {
    if !options.skip_reference_baselines {
        return Err(
            "Rust runtime LightGBM currently keeps reference baselines disabled; run Rust backtest/evaluation separately for baselines."
                .to_owned(),
        );
    }
    let started = Instant::now();
    let mut phase_timer = RuntimePhaseTimer::new();
    let mut phase_started = Instant::now();
    let resolved = resolve_runtime_config(options)?;
    phase_timer.add(
        "resolve_runtime_config",
        phase_started.elapsed().as_secs_f64(),
        1,
        0,
        0,
        0,
    );
    if resolved.selected_feature_names.len() != resolved.selected_feature_sources.len() {
        return Err("selected feature names/sources length mismatch".to_owned());
    }
    phase_started = Instant::now();
    let mut data = load_factor_data(options, &resolved)?;
    phase_timer.add(
        "load_factor_data",
        phase_started.elapsed().as_secs_f64(),
        1,
        data.rows.len(),
        0,
        0,
    );
    if data.selected_feature_names.is_empty() {
        return Err("no selected features available for Rust runtime LightGBM training".to_owned());
    }
    if resolved.cross_sectional_rank {
        phase_started = Instant::now();
        apply_cross_sectional_rank_transform(
            &mut data.rows,
            &data.selected_feature_names,
            &data.selected_feature_sources,
            &resolved.cross_sectional_rank_exclude_columns,
        );
        phase_timer.add(
            "cross_sectional_rank_transform",
            phase_started.elapsed().as_secs_f64(),
            1,
            data.rows.len(),
            0,
            0,
        );
    }
    phase_started = Instant::now();
    let training_context = build_training_context(&resolved, &data)?;
    phase_timer.add(
        "build_training_context",
        phase_started.elapsed().as_secs_f64(),
        1,
        data.rows.len(),
        0,
        0,
    );
    phase_started = Instant::now();
    let row_indices_by_date = build_row_indices_by_date(&data.rows);
    let windows = build_rolling_windows(&resolved, &data.full_calendar, &data.test_calendar);
    phase_timer.add(
        "build_rolling_windows",
        phase_started.elapsed().as_secs_f64(),
        1,
        data.rows.len(),
        windows.len(),
        0,
    );
    if windows.is_empty() {
        return Err("no rolling windows were generated; check train/valid/test dates".to_owned());
    }

    let artifact_dir = resolve_output_artifact_dir(&options.output_dir);
    let results_dir = artifact_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| options.output_dir.clone());
    let models_dir = results_dir.join("models");
    let window_dir = results_dir.join("prepared_lgbm_windows");
    let importance_dir = results_dir.join("feature_importance");
    let history_dir = results_dir.join("training_history");
    let timing_path = results_dir.join(DEFAULT_LGBM_BUNDLE_TIMING_FILENAME);
    phase_started = Instant::now();
    fs::create_dir_all(&artifact_dir)
        .map_err(|err| format!("failed to create {}: {err}", artifact_dir.display()))?;
    fs::create_dir_all(&window_dir)
        .map_err(|err| format!("failed to create {}: {err}", window_dir.display()))?;
    fs::create_dir_all(&importance_dir)
        .map_err(|err| format!("failed to create {}: {err}", importance_dir.display()))?;
    fs::create_dir_all(&history_dir)
        .map_err(|err| format!("failed to create {}: {err}", history_dir.display()))?;
    if options.save_models || options.load_models {
        fs::create_dir_all(&models_dir)
            .map_err(|err| format!("failed to create {}: {err}", models_dir.display()))?;
    }
    phase_timer.add(
        "create_output_dirs",
        phase_started.elapsed().as_secs_f64(),
        1,
        0,
        0,
        0,
    );

    let mut prediction_rows = Vec::new();
    let mut training_records = Vec::<BTreeMap<String, String>>::new();
    let mut window_timing_records = Vec::<JsonValue>::new();
    for window in &windows {
        let mut window_timing = serde_json::Map::new();
        window_timing.insert(
            "window_start".to_owned(),
            JsonValue::String(format_date_ns(window.test_start_ns)),
        );
        window_timing.insert(
            "window_end".to_owned(),
            JsonValue::String(format_date_ns(window.test_end_ns)),
        );
        let window_started = Instant::now();
        phase_started = Instant::now();
        let train_indices = collect_indices_for_date_range(
            &row_indices_by_date,
            window.train_start_ns,
            window.train_end_ns,
        );
        let valid_indices = collect_indices_for_date_range(
            &row_indices_by_date,
            window.valid_start_ns,
            window.valid_end_ns,
        );
        let test_indices = collect_indices_for_date_range(
            &row_indices_by_date,
            window.test_start_ns,
            window.test_end_ns,
        );
        let collect_window_indices_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "collect_window_indices",
            collect_window_indices_seconds,
            1,
            train_indices.len() + valid_indices.len() + test_indices.len(),
            1,
            0,
        );
        window_timing.insert(
            "collect_window_indices_seconds".to_owned(),
            json_f64_or_null(collect_window_indices_seconds),
        );
        let paths = build_window_paths(
            &window_dir,
            &models_dir,
            &importance_dir,
            &history_dir,
            window,
        );
        phase_started = Instant::now();
        write_window_frame_parquet(
            &paths.train_path,
            &data.rows,
            &train_indices,
            &data.selected_feature_names,
            &resolved,
            &training_context,
        )?;
        write_window_frame_parquet(
            &paths.valid_path,
            &data.rows,
            &valid_indices,
            &data.selected_feature_names,
            &resolved,
            &training_context,
        )?;
        write_window_frame_parquet(
            &paths.test_path,
            &data.rows,
            &test_indices,
            &data.selected_feature_names,
            &resolved,
            &training_context,
        )?;
        let write_window_parquet_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "write_window_frame_parquet",
            write_window_parquet_seconds,
            3,
            train_indices.len() + valid_indices.len() + test_indices.len(),
            1,
            3,
        );
        window_timing.insert(
            "write_window_frame_parquet_seconds".to_owned(),
            json_f64_or_null(write_window_parquet_seconds),
        );
        phase_started = Instant::now();
        let train_prepared = build_prepared_window_label_columns(
            &data.rows,
            &train_indices,
            &resolved,
            &training_context,
        )?;
        let valid_prepared = build_prepared_window_label_columns(
            &data.rows,
            &valid_indices,
            &resolved,
            &training_context,
        )?;
        let train_keep_positions =
            keep_positions_for_training_window(&data.rows, &train_indices, &train_prepared);
        let valid_keep_positions =
            keep_positions_for_training_window(&data.rows, &valid_indices, &valid_prepared);
        let test_keep_positions = keep_positions_for_prediction_window(&data.rows, &test_indices);
        let python_window = build_python_lgbm_window(&PythonLgbmWindowBuildInput {
            rows: &data.rows,
            train_indices: &train_indices,
            train_keep_positions: &train_keep_positions,
            train_prepared: &train_prepared,
            valid_indices: &valid_indices,
            valid_keep_positions: &valid_keep_positions,
            valid_prepared: &valid_prepared,
            test_indices: &test_indices,
            test_keep_positions: &test_keep_positions,
        });
        let prepare_window_labels_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "prepare_window_labels",
            prepare_window_labels_seconds,
            2,
            train_indices.len() + valid_indices.len(),
            1,
            0,
        );
        window_timing.insert(
            "prepare_window_labels_seconds".to_owned(),
            json_f64_or_null(prepare_window_labels_seconds),
        );
        phase_started = Instant::now();
        let python_result = call_python_train_window(
            options,
            &resolved,
            &data.selected_feature_names,
            &paths,
            &python_window,
        )?;
        let python_train_window_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "python_train_window",
            python_train_window_seconds,
            1,
            train_keep_positions.len() + valid_keep_positions.len() + test_keep_positions.len(),
            1,
            0,
        );
        accumulate_python_phase_timings(&mut phase_timer, &python_result.summary);
        window_timing.insert(
            "python_train_window_seconds".to_owned(),
            json_f64_or_null(python_train_window_seconds),
        );
        window_timing.insert(
            "python_subphases".to_owned(),
            python_phase_timings_json(&python_result.summary),
        );
        phase_started = Instant::now();
        let mut window_predictions = build_window_prediction_rows(
            &data.rows,
            &test_indices,
            &test_keep_positions,
            &python_result.test_predictions,
        )?;
        write_long_value_parquet(&paths.prediction_path, "prediction", &window_predictions)?;
        let write_window_prediction_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "write_window_prediction",
            write_window_prediction_seconds,
            1,
            window_predictions.len(),
            1,
            1,
        );
        window_timing.insert(
            "write_window_prediction_seconds".to_owned(),
            json_f64_or_null(write_window_prediction_seconds),
        );
        phase_started = Instant::now();
        let valid_topk_summary = compute_validation_topk_summary(
            &data.rows,
            &valid_indices,
            &valid_keep_positions,
            &valid_prepared,
            &python_result.valid_predictions,
            resolved
                .lgbm_config
                .get("validation_topk")
                .and_then(JsonValue::as_u64)
                .unwrap_or(10) as usize,
        )?;
        let compute_validation_topk_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "compute_validation_topk_summary",
            compute_validation_topk_seconds,
            1,
            valid_keep_positions.len(),
            1,
            0,
        );
        window_timing.insert(
            "compute_validation_topk_summary_seconds".to_owned(),
            json_f64_or_null(compute_validation_topk_seconds),
        );
        phase_started = Instant::now();
        let summary = build_window_training_summary(&WindowTrainingSummaryInput {
            python_summary: &python_result.summary,
            paths: &paths,
            window,
            resolved: &resolved,
            raw_train_rows: train_indices.len(),
            raw_valid_rows: valid_indices.len(),
            raw_test_rows: test_indices.len(),
            train_rows: train_keep_positions.len(),
            valid_rows: valid_keep_positions.len(),
            test_rows: test_keep_positions.len(),
            feature_count: data.selected_feature_names.len(),
            valid_topk_summary,
        });
        training_records.push(json_object_to_string_map(&summary));
        prediction_rows.append(&mut window_predictions);
        let assemble_window_summary_seconds = phase_started.elapsed().as_secs_f64();
        phase_timer.add(
            "assemble_window_summary",
            assemble_window_summary_seconds,
            1,
            train_keep_positions.len() + valid_keep_positions.len() + test_keep_positions.len(),
            1,
            0,
        );
        window_timing.insert(
            "assemble_window_summary_seconds".to_owned(),
            json_f64_or_null(assemble_window_summary_seconds),
        );
        window_timing.insert(
            "window_total_seconds".to_owned(),
            json_f64_or_null(window_started.elapsed().as_secs_f64()),
        );
        window_timing.insert(
            "raw_train_rows".to_owned(),
            serde_json::json!(train_indices.len()),
        );
        window_timing.insert(
            "raw_valid_rows".to_owned(),
            serde_json::json!(valid_indices.len()),
        );
        window_timing.insert(
            "raw_test_rows".to_owned(),
            serde_json::json!(test_indices.len()),
        );
        window_timing.insert(
            "train_rows_kept".to_owned(),
            serde_json::json!(train_keep_positions.len()),
        );
        window_timing.insert(
            "valid_rows_kept".to_owned(),
            serde_json::json!(valid_keep_positions.len()),
        );
        window_timing.insert(
            "test_rows_kept".to_owned(),
            serde_json::json!(test_keep_positions.len()),
        );
        window_timing_records.push(JsonValue::Object(window_timing));
    }
    phase_started = Instant::now();
    prediction_rows.sort_by(|left, right| {
        left.date_ns
            .cmp(&right.date_ns)
            .then(left.instrument.cmp(&right.instrument))
    });
    let prediction_keys = prediction_rows
        .iter()
        .map(|row| (row.date_ns, row.instrument.clone()))
        .collect::<BTreeSet<_>>();
    phase_timer.add(
        "sort_predictions",
        phase_started.elapsed().as_secs_f64(),
        1,
        prediction_rows.len(),
        0,
        0,
    );

    phase_started = Instant::now();
    let test_indices = collect_indices_for_date_range(
        &row_indices_by_date,
        resolved.test_start_ns,
        resolved.test_end_ns,
    );
    let signal_label_rows = test_indices
        .iter()
        .filter(|row_index| {
            prediction_keys.contains(&(
                data.rows[**row_index].date_ns,
                data.rows[**row_index].symbol.clone(),
            ))
        })
        .map(|row_index| LongValueRow {
            date_ns: data.rows[*row_index].date_ns,
            instrument: data.rows[*row_index].symbol.clone(),
            value: data.rows[*row_index].label,
        })
        .collect::<Vec<_>>();
    let backtest_label_rows = test_indices
        .iter()
        .filter(|row_index| {
            prediction_keys.contains(&(
                data.rows[**row_index].date_ns,
                data.rows[**row_index].symbol.clone(),
            ))
        })
        .map(|row_index| LongValueRow {
            date_ns: data.rows[*row_index].date_ns,
            instrument: data.rows[*row_index].symbol.clone(),
            value: data.rows[*row_index].backtest_label,
        })
        .collect::<Vec<_>>();
    phase_timer.add(
        "collect_bundle_labels",
        phase_started.elapsed().as_secs_f64(),
        2,
        signal_label_rows.len() + backtest_label_rows.len(),
        0,
        0,
    );
    phase_started = Instant::now();
    write_long_value_parquet(
        &artifact_dir.join(PREDICTIONS_FILENAME),
        "prediction",
        &prediction_rows,
    )?;
    write_long_value_parquet(
        &artifact_dir.join(SIGNAL_LABELS_FILENAME),
        "label",
        &signal_label_rows,
    )?;
    write_long_value_parquet(
        &artifact_dir.join(BACKTEST_LABELS_FILENAME),
        "label",
        &backtest_label_rows,
    )?;
    phase_timer.add(
        "write_bundle_label_prediction_parquet",
        phase_started.elapsed().as_secs_f64(),
        3,
        prediction_rows.len() + signal_label_rows.len() + backtest_label_rows.len(),
        0,
        3,
    );
    phase_started = Instant::now();
    write_training_summary(
        &artifact_dir.join(TRAINING_SUMMARY_FILENAME),
        &training_records,
    )?;
    write_training_summary(
        &results_dir.join(TRAINING_SUMMARY_FILENAME),
        &training_records,
    )?;
    phase_timer.add(
        "write_training_summary",
        phase_started.elapsed().as_secs_f64(),
        2,
        training_records.len(),
        0,
        2,
    );
    phase_started = Instant::now();
    write_config_snapshot(&results_dir.join("config_snapshot.yaml"), &resolved.cfg)?;
    phase_timer.add(
        "write_config_snapshot",
        phase_started.elapsed().as_secs_f64(),
        1,
        0,
        0,
        1,
    );
    phase_started = Instant::now();
    write_metadata(
        &artifact_dir.join(PREDICTION_METADATA_FILENAME),
        &resolved,
        &data,
    )?;
    phase_timer.add(
        "write_prediction_metadata",
        phase_started.elapsed().as_secs_f64(),
        1,
        0,
        0,
        1,
    );

    let finite_predictions = prediction_rows
        .iter()
        .filter(|row| row.value.is_finite())
        .count();
    let timing_summary =
        build_lgbm_bundle_timing_summary(&phase_timer, &timing_path, &window_timing_records);
    phase_started = Instant::now();
    fs::write(
        &timing_path,
        serde_json::to_string_pretty(&timing_summary)
            .map_err(|err| format!("failed to serialize timing summary: {err}"))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", timing_path.display()))?;
    phase_timer.add(
        "write_timing_summary",
        phase_started.elapsed().as_secs_f64(),
        1,
        window_timing_records.len(),
        windows.len(),
        1,
    );
    let summary = serde_json::json!({
        "artifact_dir": artifact_dir.display().to_string(),
        "results_dir": results_dir.display().to_string(),
        "models_dir": models_dir.display().to_string(),
        "model_name": "lgbm",
        "runtime_builder": "rust",
        "python_role": "lightgbm_fit_predict_only",
        "factor_store": resolved.factor_store.display().to_string(),
        "data_source": resolved.data_source,
        "label_column": resolved.label_column,
        "backtest_label_column": resolved.backtest_label_column,
        "signal_horizon": resolved.signal_horizon,
        "backtest_label_horizon": 1,
        "label_embargo_days": resolved.label_embargo_days,
        "retrain_step": resolved.retrain_step,
        "train_days": resolved.train_days,
        "valid_days": resolved.valid_days,
        "test_start": format_date_ns(resolved.test_start_ns),
        "test_end": format_date_ns(resolved.test_end_ns),
        "selected_feature_count": data.selected_feature_names.len(),
        "loaded_rows": data.rows.len(),
        "test_rows": test_indices.len(),
        "prediction_rows": prediction_rows.len(),
        "finite_predictions": finite_predictions,
        "windows": windows.len(),
        "cross_sectional_rank_enabled": resolved.cross_sectional_rank,
        "cross_sectional_rank_exclude_columns": sorted_string_set(&resolved.cross_sectional_rank_exclude_columns),
        "universe": resolved.universe_name.clone().unwrap_or_else(|| "all".to_owned()),
        "reference_baselines_enabled": false,
        "train_label_transform_mode": resolved.train_label_transform.mode,
        "train_label_space": if resolved.train_label_transform.mode.starts_with("buyability") { "binary_target" } else { "return_target" },
        "opportunity_label_mode": resolved.opportunity.mode,
        "opportunity_label_threshold": resolved.opportunity.threshold,
        "opportunity_label_neutral_band": resolved.opportunity.neutral_band,
        "train_sample_weight_mode": resolved.sample_weight.mode,
        "training_summary_path": results_dir.join(TRAINING_SUMMARY_FILENAME).display().to_string(),
        "prediction_training_summary_path": artifact_dir.join(TRAINING_SUMMARY_FILENAME).display().to_string(),
        "config_snapshot_path": results_dir.join("config_snapshot.yaml").display().to_string(),
        "timing_path": timing_path.display().to_string(),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
    });
    fs::write(
        results_dir.join("lgbm_bundle_summary.json"),
        serde_json::to_string_pretty(&summary)
            .map_err(|err| format!("failed to serialize summary: {err}"))?,
    )
    .map_err(|err| format!("failed to write lgbm_bundle_summary.json: {err}"))?;
    Ok(summary)
}

pub(crate) fn write_resolved_config_snapshot(
    options: &LgbmBundleOptions,
    path: &Path,
) -> Result<(), String> {
    let resolved = resolve_runtime_config(options)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    write_config_snapshot(path, &resolved.cfg)
}

pub(crate) fn validate_lgbm_bundle_options(options: &LgbmBundleOptions) -> Result<(), String> {
    resolve_runtime_config(options).map(|_| ())
}

fn resolve_runtime_config(options: &LgbmBundleOptions) -> Result<ResolvedRuntimeConfig, String> {
    let mut cfg = load_resolved_config(options)?;
    if let Some(data_source) = &options.data_source {
        set_yaml_dotted(
            &mut cfg,
            "data.source",
            YamlValue::String(data_source.clone()),
        )?;
    }
    if let Some(feature_profile) = &options.feature_profile {
        set_yaml_dotted(
            &mut cfg,
            "features.profile",
            YamlValue::String(feature_profile.clone()),
        )?;
    }
    if let Some(topk) = options.topk {
        set_yaml_dotted(&mut cfg, "strategy.topk", yaml_usize(topk))?;
    }
    if let Some(n_drop) = options.n_drop {
        set_yaml_dotted(&mut cfg, "strategy.n_drop", yaml_usize(n_drop))?;
    }
    if let Some(rebalance_freq) = options.rebalance_freq {
        set_yaml_dotted(
            &mut cfg,
            "backtest.rebalance_freq",
            yaml_usize(rebalance_freq),
        )?;
    }
    if let Some(train_days) = options.train_days {
        set_yaml_dotted(&mut cfg, "rolling.train_days", yaml_usize(train_days))?;
    }
    if let Some(valid_days) = options.valid_days {
        set_yaml_dotted(&mut cfg, "rolling.valid_days", yaml_usize(valid_days))?;
    }
    if let Some(retrain_step) = options.retrain_step {
        set_yaml_dotted(&mut cfg, "rolling.retrain_step", yaml_usize(retrain_step))?;
    }
    if let Some(signal_horizon) = options.signal_horizon {
        set_yaml_dotted(&mut cfg, "label.signal_horizon", yaml_usize(signal_horizon))?;
    }
    if let Some(label_embargo_days) = options.label_embargo_days {
        set_yaml_dotted(
            &mut cfg,
            "rolling.label_embargo_days",
            yaml_usize(label_embargo_days),
        )?;
    }
    if options.test_start.is_some() != options.test_end.is_some() {
        return Err("--test-start and --test-end must be supplied together".to_owned());
    }
    if let (Some(test_start), Some(test_end)) = (&options.test_start, &options.test_end) {
        set_yaml_dotted(
            &mut cfg,
            "time.test",
            YamlValue::Sequence(vec![
                YamlValue::String(test_start.clone()),
                YamlValue::String(test_end.clone()),
            ]),
        )?;
    }
    for override_arg in &options.set_overrides {
        let (key, value) = parse_key_value_arg(override_arg, "Override")?;
        set_yaml_dotted(&mut cfg, &key, value)?;
    }
    let resolved_model_name = yaml_path_string(&cfg, &["model", "name"])
        .unwrap_or_else(|| "lgbm".to_owned())
        .trim()
        .to_ascii_lowercase();
    if resolved_model_name != "lgbm" {
        return Err(format!(
            "ai4stock-train make-bundle-lgbm only supports model.name == 'lgbm'; resolved model.name={resolved_model_name:?}"
        ));
    }
    set_yaml_dotted(&mut cfg, "model.name", YamlValue::String("lgbm".to_owned()))?;

    let data_source =
        yaml_path_string(&cfg, &["data", "source"]).unwrap_or_else(|| "akshare".to_owned());
    let signal_horizon = yaml_path_usize(&cfg, &["label", "signal_horizon"]).unwrap_or(20);
    let label_column = label_column_name(signal_horizon);
    let backtest_label_column = label_column_name(1);
    let train_days = required_yaml_usize(&cfg, &["rolling", "train_days"])?;
    let valid_days = required_yaml_usize(&cfg, &["rolling", "valid_days"])?;
    let retrain_step = required_yaml_usize(&cfg, &["rolling", "retrain_step"])?;
    let label_embargo_days =
        yaml_path_usize(&cfg, &["rolling", "label_embargo_days"]).unwrap_or(signal_horizon + 1);
    let (test_start_ns, test_end_ns) = read_test_range_ns(&cfg)?;
    if test_end_ns < test_start_ns {
        return Err("time.test end must be >= start".to_owned());
    }

    let feature_resolution = resolve_feature_selection(options, &cfg)?;
    let cross_sectional_rank = options.cross_sectional_rank.unwrap_or_else(|| {
        yaml_path_bool(&cfg, &["features", "transforms", "cross_sectional_rank"]).unwrap_or(false)
    });
    if options.cross_sectional_rank.is_some() {
        set_yaml_dotted(
            &mut cfg,
            "features.transforms.cross_sectional_rank",
            YamlValue::Bool(cross_sectional_rank),
        )?;
    }
    let cross_sectional_rank_exclude_columns =
        resolve_cross_sectional_rank_exclude_columns(&cfg, &feature_resolution)?;
    let universe_name = yaml_path_string(&cfg, &["universe"])
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let universe_dir = yaml_path_string(&cfg, &["native", "universe_dir"])
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/universes"));
    let factor_store = options.factor_store.clone().unwrap_or_else(|| {
        yaml_path_string(&cfg, &["features", "factor_store_dir"])
            .or_else(|| yaml_path_string(&cfg, &["features", "cache_dir"]))
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                default_factor_store_dir(&data_source, &feature_resolution.factor_store_name)
            })
    });
    let mut lgbm_config = resolve_lgbm_config(&cfg)?;
    if lgbm_config.get("validation_topk").is_none() {
        if let Some(topk) = yaml_path_usize(&cfg, &["strategy", "topk"]) {
            lgbm_config["validation_topk"] = serde_json::json!(topk);
        }
    }
    let train_label_transform = resolve_train_label_transform_config(&cfg)?;
    let opportunity = resolve_opportunity_label_config(&cfg)?;
    let sample_weight = resolve_sample_weight_config(&lgbm_config);

    Ok(ResolvedRuntimeConfig {
        cfg,
        factor_store,
        data_source,
        label_column,
        backtest_label_column,
        signal_horizon,
        train_days,
        valid_days,
        retrain_step,
        label_embargo_days,
        test_start_ns,
        test_end_ns,
        selected_feature_names: feature_resolution.selected_names,
        selected_feature_sources: feature_resolution.selected_sources,
        cross_sectional_rank,
        cross_sectional_rank_exclude_columns,
        universe_name,
        universe_dir,
        lgbm_config,
        train_label_transform,
        opportunity,
        sample_weight,
    })
}

#[derive(Debug, Clone)]
struct FeatureSelectionResolution {
    selected_names: Vec<String>,
    selected_sources: Vec<String>,
    factor_store_name: String,
    cross_sectional_rank_exclude_columns: BTreeSet<String>,
}

fn resolve_feature_selection(
    options: &LgbmBundleOptions,
    cfg: &YamlValue,
) -> Result<FeatureSelectionResolution, String> {
    if !options.explicit_features.is_empty() {
        return Ok(FeatureSelectionResolution {
            selected_names: options.explicit_features.clone(),
            selected_sources: options
                .explicit_features
                .iter()
                .map(|feature| canonical_feature_source(feature))
                .collect(),
            factor_store_name: "full_factor_space".to_owned(),
            cross_sectional_rank_exclude_columns: BTreeSet::new(),
        });
    }
    if let Some(path) = &options.features_json {
        let features = read_feature_list_json(path)?;
        return Ok(FeatureSelectionResolution {
            selected_names: features.clone(),
            selected_sources: features
                .iter()
                .map(|feature| canonical_feature_source(feature))
                .collect(),
            factor_store_name: "full_factor_space".to_owned(),
            cross_sectional_rank_exclude_columns: BTreeSet::new(),
        });
    }
    let feature_profile = yaml_path_string(cfg, &["features", "profile"]);
    if let Some(profile_name) = feature_profile {
        if let Some(resolution) = resolve_feature_profile(&profile_name)? {
            return Ok(resolution);
        }
    }
    Ok(FeatureSelectionResolution {
        selected_names: Vec::new(),
        selected_sources: Vec::new(),
        factor_store_name: "full_factor_space".to_owned(),
        cross_sectional_rank_exclude_columns: BTreeSet::new(),
    })
}

fn load_factor_data(
    options: &LgbmBundleOptions,
    resolved: &ResolvedRuntimeConfig,
) -> Result<LoadedFactorData, String> {
    let meta = read_factor_store_meta(&resolved.factor_store)?;
    let all_features = read_meta_string_list(&meta, "feature_names")?;
    let selected_sources = if resolved.selected_feature_sources.is_empty() {
        select_features_from_meta(&all_features, options.max_features)?
    } else {
        validate_selected_sources(&all_features, &resolved.selected_feature_sources)?
    };
    let selected_names = if resolved.selected_feature_names.is_empty() {
        selected_sources.clone()
    } else {
        resolved.selected_feature_names.clone()
    };
    let calendar = read_meta_dates(&meta, "available_dates")?;
    let test_calendar = calendar
        .iter()
        .copied()
        .filter(|date_ns| *date_ns >= resolved.test_start_ns && *date_ns <= resolved.test_end_ns)
        .collect::<Vec<_>>();
    let load_start_ns = if let Some(first_test_pos) = calendar
        .iter()
        .position(|date_ns| *date_ns >= resolved.test_start_ns)
    {
        let earliest_pos = first_test_pos.saturating_sub(
            resolved.label_embargo_days + resolved.train_days + resolved.valid_days,
        );
        Some(calendar[earliest_pos])
    } else {
        None
    };
    let columns = projected_columns(resolved, &selected_sources);
    let (_, paths) = discover_bucket_parquet_paths(&resolved.factor_store)?;
    if paths.is_empty() {
        return Err(format!(
            "no bucket shard parquet files found under {}",
            resolved.factor_store.display()
        ));
    }
    let universe_filter = load_universe_filter(resolved)?;
    let mut rows = Vec::new();
    for path in paths {
        append_factor_rows_from_bucket(
            &mut rows,
            &FactorBucketLoadOptions {
                path: &path,
                columns: &columns,
                resolved,
                selected_sources: &selected_sources,
                universe_filter: universe_filter.as_ref(),
                load_start_ns,
                batch_size: options.batch_size,
            },
        )?;
    }
    rows.sort_by(|left, right| {
        left.date_ns
            .cmp(&right.date_ns)
            .then(left.symbol.cmp(&right.symbol))
    });
    let full_calendar = if calendar.is_empty() {
        rows.iter()
            .map(|row| row.date_ns)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    } else {
        calendar
    };
    let test_calendar = if test_calendar.is_empty() {
        full_calendar
            .iter()
            .copied()
            .filter(|date_ns| {
                *date_ns >= resolved.test_start_ns && *date_ns <= resolved.test_end_ns
            })
            .collect::<Vec<_>>()
    } else {
        test_calendar
    };
    if test_calendar.is_empty() {
        return Err("no trading dates available in requested test range".to_owned());
    }
    if rows.is_empty() {
        return Err("factor-store projection returned no rows".to_owned());
    }
    if selected_names.len() != selected_sources.len() {
        return Err(
            "selected feature names/sources length mismatch after metadata validation".to_owned(),
        );
    }
    Ok(LoadedFactorData {
        rows,
        full_calendar,
        test_calendar,
        selected_feature_names: selected_names,
        selected_feature_sources: selected_sources,
    })
}

fn load_universe_filter(
    resolved: &ResolvedRuntimeConfig,
) -> Result<Option<UniverseFilter>, String> {
    let Some(universe_name) = resolved.universe_name.as_deref() else {
        return Ok(None);
    };
    if universe_name == "all" {
        return Ok(None);
    }
    let path = resolve_universe_path(universe_name, &resolved.universe_dir)?;
    let delimiter = if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
    {
        b','
    } else {
        b'\t'
    };
    let has_headers = delimiter == b',';
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_headers)
        .flexible(true)
        .from_path(&path)
        .map_err(|err| format!("failed to open universe file {}: {err}", path.display()))?;
    let mut intervals_by_symbol: IntervalsBySymbol = BTreeMap::new();
    for record in reader.records() {
        let record = record
            .map_err(|err| format!("failed to read universe file {}: {err}", path.display()))?;
        let symbol = normalize_symbol(record.get(0).unwrap_or_default());
        if symbol.is_empty() {
            continue;
        }
        let start_ns = record
            .get(1)
            .map(parse_optional_date_ns)
            .transpose()
            .map_err(|err| {
                format!(
                    "failed to parse universe start date in {}: {err}",
                    path.display()
                )
            })?
            .flatten();
        let end_ns = record
            .get(2)
            .map(parse_optional_date_ns)
            .transpose()
            .map_err(|err| {
                format!(
                    "failed to parse universe end date in {}: {err}",
                    path.display()
                )
            })?
            .flatten();
        intervals_by_symbol
            .entry(symbol)
            .or_default()
            .push((start_ns, end_ns));
    }
    if intervals_by_symbol.is_empty() {
        return Err(format!(
            "universe file {} did not contain any symbols",
            path.display()
        ));
    }
    Ok(Some(UniverseFilter {
        intervals_by_symbol,
    }))
}

fn load_industry_map(resolved: &ResolvedRuntimeConfig) -> Result<BTreeMap<String, String>, String> {
    let path = PathBuf::from("data")
        .join(&resolved.data_source)
        .join("raw")
        .join("meta")
        .join("symbol_cache.parquet");
    if !path.exists() {
        return Err(format!(
            "industry_excess opportunity mode requires symbol cache at {}",
            path.display()
        ));
    }
    let mut last_error = String::new();
    for symbol_column in ["local_symbol", "symbol"] {
        match read_industry_map_with_symbol_column(&path, symbol_column) {
            Ok(map) if !map.is_empty() => return Ok(map),
            Ok(_) => {
                last_error = format!("{} had no usable industry mappings", path.display());
            }
            Err(error) => last_error = error,
        }
    }
    Err(format!(
        "industry_excess opportunity mode could not load industry mapping: {last_error}"
    ))
}

fn read_industry_map_with_symbol_column(
    path: &Path,
    symbol_column: &str,
) -> Result<BTreeMap<String, String>, String> {
    let columns = vec![symbol_column.to_owned(), "industry".to_owned()];
    let reader = open_projected_parquet_reader(path, &columns, 65_536)?;
    let mut map = BTreeMap::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let symbol_array = required_column(&batch, symbol_column, path)?;
        let industry_array = required_column(&batch, "industry", path)?;
        for row_index in 0..batch.num_rows() {
            if symbol_array.is_null(row_index) || industry_array.is_null(row_index) {
                continue;
            }
            let symbol = normalize_local_symbol(&string_value(symbol_array, row_index, path)?);
            let industry = string_value(industry_array, row_index, path)?;
            if !symbol.is_empty() && !industry.trim().is_empty() {
                map.insert(symbol, industry);
            }
        }
    }
    Ok(map)
}

fn build_benchmark_forward_returns(
    resolved: &ResolvedRuntimeConfig,
    rows: &[FactorRow],
) -> Result<BTreeMap<i64, f64>, String> {
    let benchmark_returns = build_benchmark_returns(resolved, rows)?;
    if benchmark_returns.is_empty() {
        return Err(
            "benchmark_excess opportunity mode produced an empty benchmark series".to_owned(),
        );
    }
    Ok(build_forward_compound_return_map(
        &benchmark_returns,
        resolved.signal_horizon,
    ))
}

fn build_benchmark_returns(
    resolved: &ResolvedRuntimeConfig,
    rows: &[FactorRow],
) -> Result<Vec<(i64, f64)>, String> {
    let mode = yaml_path_string(&resolved.cfg, &["backtest", "benchmark", "mode"])
        .unwrap_or_else(|| "cross_section_mean".to_owned())
        .trim()
        .to_ascii_lowercase();
    if mode == "cross_section_mean" || mode.is_empty() {
        return build_cross_section_benchmark_returns(rows);
    }
    if mode != "file" {
        return Err(format!(
            "unsupported benchmark mode for Rust training runtime: {mode}; supported: cross_section_mean, file"
        ));
    }
    let path = yaml_path_string(&resolved.cfg, &["backtest", "benchmark", "path"])
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            "backtest.benchmark.path is required when benchmark mode is file".to_owned()
        })?;
    let date_column = yaml_path_string(&resolved.cfg, &["backtest", "benchmark", "date_column"])
        .unwrap_or_else(|| "date".to_owned());
    let value_column = yaml_path_string(&resolved.cfg, &["backtest", "benchmark", "value_column"])
        .unwrap_or_else(|| "close".to_owned());
    let value_type = yaml_path_string(&resolved.cfg, &["backtest", "benchmark", "value_type"])
        .unwrap_or_else(|| "close".to_owned())
        .trim()
        .to_ascii_lowercase();
    load_file_benchmark_returns(&path, &date_column, &value_column, &value_type)
}

fn build_cross_section_benchmark_returns(rows: &[FactorRow]) -> Result<Vec<(i64, f64)>, String> {
    cross_section_mean_returns(
        rows.iter().map(|row| (row.date_ns, row.label)),
        "cross_section_mean benchmark had no finite labels",
    )
}

fn build_forward_compound_return_map(
    daily_returns: &[(i64, f64)],
    horizon: usize,
) -> BTreeMap<i64, f64> {
    let horizon = horizon.max(1);
    let mut out = BTreeMap::new();
    if daily_returns.len() <= horizon {
        return out;
    }
    for position in 0..(daily_returns.len() - horizon) {
        let window = &daily_returns[(position + 1)..=(position + horizon)];
        if window.iter().all(|(_, value)| value.is_finite()) {
            let compounded = window
                .iter()
                .fold(1.0, |acc, (_, value)| acc * (1.0 + *value))
                - 1.0;
            out.insert(daily_returns[position].0, compounded);
        }
    }
    out
}

fn resolve_universe_path(universe_name: &str, universe_dir: &Path) -> Result<PathBuf, String> {
    let candidates = [
        universe_dir.join(universe_name),
        universe_dir.join(format!("{universe_name}.txt")),
        universe_dir.join(format!("{universe_name}.csv")),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "universe file not found for '{}' under {}",
        universe_name,
        universe_dir.display()
    ))
}

impl UniverseFilter {
    fn allows(&self, symbol: &str, date_ns: i64) -> bool {
        let symbol = normalize_symbol(symbol);
        let Some(intervals) = self.intervals_by_symbol.get(&symbol) else {
            return false;
        };
        intervals.iter().any(|(start_ns, end_ns)| {
            start_ns.is_none_or(|start| date_ns >= start) && end_ns.is_none_or(|end| date_ns <= end)
        })
    }
}

fn normalize_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect()
}

fn normalize_local_symbol(symbol: &str) -> String {
    let trimmed = symbol.trim();
    let digits = normalize_symbol(trimmed);
    if !digits.is_empty() && digits.len() <= 6 {
        format!("{digits:0>6}")
    } else {
        trimmed.to_owned()
    }
}

fn parse_optional_date_ns(raw: &str) -> Result<Option<i64>, String> {
    let value = raw.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let lower = value.to_ascii_lowercase();
    if matches!(lower.as_str(), "nan" | "nat" | "none" | "null") {
        return Ok(None);
    }
    parse_date_ns(value).map(Some)
}

struct FactorBucketLoadOptions<'a> {
    path: &'a Path,
    columns: &'a [String],
    resolved: &'a ResolvedRuntimeConfig,
    selected_sources: &'a [String],
    universe_filter: Option<&'a UniverseFilter>,
    load_start_ns: Option<i64>,
    batch_size: usize,
}

fn append_factor_rows_from_bucket(
    rows: &mut Vec<FactorRow>,
    options: &FactorBucketLoadOptions<'_>,
) -> Result<(), String> {
    let reader = open_projected_parquet_reader(options.path, options.columns, options.batch_size)?;
    for batch in reader {
        let batch =
            batch.map_err(|err| format!("failed to read {}: {err}", options.path.display()))?;
        append_factor_batch(
            rows,
            &batch,
            options.path,
            options.resolved,
            options.selected_sources,
            options.universe_filter,
            options.load_start_ns,
        )?;
    }
    Ok(())
}

fn append_factor_batch(
    rows: &mut Vec<FactorRow>,
    batch: &RecordBatch,
    path: &Path,
    resolved: &ResolvedRuntimeConfig,
    selected_sources: &[String],
    universe_filter: Option<&UniverseFilter>,
    load_start_ns: Option<i64>,
) -> Result<(), String> {
    let date_array = required_column(batch, "date", path)?;
    let symbol_array = required_column(batch, "symbol", path)?;
    let label_array = required_column(batch, &resolved.label_column, path)?;
    let backtest_label_array = required_column(batch, &resolved.backtest_label_column, path)?;
    let feature_arrays = selected_sources
        .iter()
        .map(|feature| required_column(batch, feature, path))
        .collect::<Result<Vec<_>, _>>()?;
    for row_index in 0..batch.num_rows() {
        let date_ns = date_value_ns(date_array, row_index, path)?;
        if load_start_ns.is_some_and(|start_ns| date_ns < start_ns)
            || date_ns > resolved.test_end_ns
        {
            continue;
        }
        let symbol = string_value(symbol_array, row_index, path)?;
        if universe_filter.is_some_and(|filter| !filter.allows(&symbol, date_ns)) {
            continue;
        }
        let label = sanitize_label(numeric_value(label_array, row_index, path)?);
        let backtest_label = sanitize_label(numeric_value(backtest_label_array, row_index, path)?);
        let features = feature_arrays
            .iter()
            .map(|array| numeric_value(*array, row_index, path))
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(FactorRow {
            date_ns,
            symbol,
            label,
            backtest_label,
            features,
        });
    }
    Ok(())
}

fn apply_cross_sectional_rank_transform(
    rows: &mut [FactorRow],
    selected_feature_names: &[String],
    selected_feature_sources: &[String],
    exclude_columns: &BTreeSet<String>,
) {
    let mut by_date: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row_index, row) in rows.iter().enumerate() {
        by_date.entry(row.date_ns).or_default().push(row_index);
    }
    let feature_count = selected_feature_names.len();
    for indices in by_date.values() {
        for feature_index in 0..feature_count {
            if exclude_columns.contains(&selected_feature_names[feature_index])
                || exclude_columns.contains(&selected_feature_sources[feature_index])
            {
                continue;
            }
            let mut valid_positions = Vec::new();
            let mut values = Vec::new();
            for row_index in indices {
                let value = rows[*row_index].features[feature_index];
                if value.is_finite() {
                    valid_positions.push(*row_index);
                    values.push(value);
                }
            }
            if values.is_empty() {
                continue;
            }
            let ranks = rank_average(&values);
            let denominator = values.len() as f64;
            for (position_index, row_index) in valid_positions.iter().enumerate() {
                rows[*row_index].features[feature_index] = ranks[position_index] / denominator;
            }
        }
    }
}

fn build_rolling_windows(
    resolved: &ResolvedRuntimeConfig,
    full_calendar: &[i64],
    test_calendar: &[i64],
) -> Vec<RollingWindow> {
    let position_by_date = full_calendar
        .iter()
        .enumerate()
        .map(|(position, date_ns)| (*date_ns, position))
        .collect::<BTreeMap<_, _>>();
    let mut windows = Vec::new();
    for (step_index, test_start_pos) in (0..test_calendar.len())
        .step_by(resolved.retrain_step.max(1))
        .enumerate()
    {
        let test_start_ns = test_calendar[test_start_pos];
        let test_end_pos =
            (test_start_pos + resolved.retrain_step - 1).min(test_calendar.len() - 1);
        let test_end_ns = test_calendar[test_end_pos];
        let Some(full_start_pos) = position_by_date.get(&test_start_ns).copied() else {
            continue;
        };
        let Some((train_start_pos, train_end_pos, valid_start_pos, valid_end_pos)) =
            compute_rolling_window_indices(
                full_start_pos,
                resolved.train_days,
                resolved.valid_days,
                resolved.label_embargo_days,
            )
        else {
            continue;
        };
        windows.push(RollingWindow {
            step_index,
            test_start_ns,
            test_end_ns,
            train_start_ns: full_calendar[train_start_pos],
            train_end_ns: full_calendar[train_end_pos],
            valid_start_ns: full_calendar[valid_start_pos],
            valid_end_ns: full_calendar[valid_end_pos],
        });
    }
    windows
}

fn compute_rolling_window_indices(
    full_start_pos: usize,
    train_days: usize,
    valid_days: usize,
    label_embargo_days: usize,
) -> Option<(usize, usize, usize, usize)> {
    let valid_end_pos = full_start_pos.checked_sub(1 + label_embargo_days)?;
    let valid_start_pos = valid_end_pos.checked_sub(valid_days.checked_sub(1)?)?;
    let train_end_pos = valid_start_pos.checked_sub(1)?;
    let train_start_pos = train_end_pos.checked_sub(train_days.checked_sub(1)?)?;
    Some((
        train_start_pos,
        train_end_pos,
        valid_start_pos,
        valid_end_pos,
    ))
}

fn call_python_train_window(
    options: &LgbmBundleOptions,
    resolved: &ResolvedRuntimeConfig,
    selected_feature_names: &[String],
    paths: &WindowDataPaths,
    prepared_window: &PreparedPythonLgbmWindow,
) -> Result<PythonWindowTrainResult, String> {
    let feature_names = selected_feature_names.to_vec();
    let lgbm_config_json = serde_json::to_string(&resolved.lgbm_config)
        .map_err(|err| format!("failed to encode lgbm config: {err}"))?;
    Python::attach(|python| -> PyResult<PythonWindowTrainResult> {
        prepare_python_path(python)?;
        let bridge_module = python.import("src.rust_lgbm_bridge")?;
        let kwargs = PyDict::new(python);
        kwargs.set_item(
            "train_feature_bytes",
            PyBytes::new(python, &prepared_window.train_feature_bytes),
        )?;
        kwargs.set_item(
            "valid_feature_bytes",
            PyBytes::new(python, &prepared_window.valid_feature_bytes),
        )?;
        kwargs.set_item(
            "test_feature_bytes",
            PyBytes::new(python, &prepared_window.test_feature_bytes),
        )?;
        kwargs.set_item("train_rows", prepared_window.train_rows)?;
        kwargs.set_item("valid_rows", prepared_window.valid_rows)?;
        kwargs.set_item("test_rows", prepared_window.test_rows)?;
        kwargs.set_item(
            "train_label_bytes",
            PyBytes::new(python, &prepared_window.train_label_bytes),
        )?;
        kwargs.set_item(
            "valid_label_bytes",
            PyBytes::new(python, &prepared_window.valid_label_bytes),
        )?;
        kwargs.set_item(
            "raw_valid_label_bytes",
            PyBytes::new(python, &prepared_window.raw_valid_label_bytes),
        )?;
        kwargs.set_item(
            "train_date_ns_bytes",
            PyBytes::new(python, &prepared_window.train_date_ns_bytes),
        )?;
        kwargs.set_item(
            "valid_date_ns_bytes",
            PyBytes::new(python, &prepared_window.valid_date_ns_bytes),
        )?;
        kwargs.set_item("model_path", paths.model_path.to_string_lossy().as_ref())?;
        kwargs.set_item(
            "feature_importance_path",
            paths.importance_path.to_string_lossy().as_ref(),
        )?;
        kwargs.set_item(
            "training_history_path",
            paths.history_path.to_string_lossy().as_ref(),
        )?;
        kwargs.set_item("lgbm_config_json", &lgbm_config_json)?;
        if let Some(train_sample_weight_bytes) = &prepared_window.train_sample_weight_bytes {
            kwargs.set_item(
                "train_sample_weight_bytes",
                PyBytes::new(python, train_sample_weight_bytes),
            )?;
        }
        if let Some(valid_sample_weight_bytes) = &prepared_window.valid_sample_weight_bytes {
            kwargs.set_item(
                "valid_sample_weight_bytes",
                PyBytes::new(python, valid_sample_weight_bytes),
            )?;
        }
        kwargs.set_item("save_model", options.save_models)?;
        kwargs.set_item("load_model", options.load_models)?;
        kwargs.set_item(
            "feature_names",
            PyList::new(python, feature_names.iter().map(String::as_str))?,
        )?;
        let result = bridge_module
            .getattr("train_lgbm_window_from_prepared_arrays")?
            .call((), Some(&kwargs))?;
        let result_dict = result.cast_into::<PyDict>()?;
        let valid_prediction_bytes = result_dict
            .get_item("valid_prediction_bytes")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing valid_prediction_bytes"))?
            .extract::<Vec<u8>>()?;
        let test_prediction_bytes = result_dict
            .get_item("test_prediction_bytes")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing test_prediction_bytes"))?
            .extract::<Vec<u8>>()?;
        let summary_value = result_dict
            .get_item("summary")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing summary"))?;
        let json_module = python.import("json")?;
        let summary_json = json_module
            .call_method1("dumps", (summary_value,))?
            .extract::<String>()?;
        let summary = serde_json::from_str(&summary_json).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Python training summary returned invalid JSON: {error}: {summary_json}"
            ))
        })?;
        Ok(PythonWindowTrainResult {
            summary,
            valid_predictions: decode_f64_le_bytes(&valid_prediction_bytes)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
            test_predictions: decode_f64_le_bytes(&test_prediction_bytes)
                .map_err(pyo3::exceptions::PyValueError::new_err)?,
        })
    })
    .map_err(|error| format!("Python LightGBM training failed: {error}"))
}

fn build_window_paths(
    window_dir: &Path,
    models_dir: &Path,
    importance_dir: &Path,
    history_dir: &Path,
    window: &RollingWindow,
) -> WindowDataPaths {
    let tag = format!(
        "{:04}_{}",
        window.step_index + 1,
        format_date_ns(window.test_start_ns)
    );
    WindowDataPaths {
        train_path: window_dir.join(format!("{tag}_train.parquet")),
        valid_path: window_dir.join(format!("{tag}_valid.parquet")),
        test_path: window_dir.join(format!("{tag}_test.parquet")),
        prediction_path: window_dir.join(format!("{tag}_predictions.parquet")),
        model_path: models_dir.join(format!(
            "model_{}.pkl",
            format_date_ns(window.test_start_ns)
        )),
        importance_path: importance_dir.join(format!(
            "feature_importance_{}.csv",
            format_date_ns(window.test_start_ns)
        )),
        history_path: history_dir.join(format!(
            "training_history_{}.csv",
            format_date_ns(window.test_start_ns)
        )),
    }
}

fn build_training_context(
    resolved: &ResolvedRuntimeConfig,
    data: &LoadedFactorData,
) -> Result<TrainingContext, String> {
    let mut context = TrainingContext::default();
    if resolved.opportunity.mode == "industry_excess" {
        context.industry_by_symbol = load_industry_map(resolved)?;
    }
    if resolved.opportunity.mode == "benchmark_excess" {
        context.benchmark_forward_returns = build_benchmark_forward_returns(resolved, &data.rows)?;
    }
    Ok(context)
}

fn build_prepared_window_label_columns(
    rows: &[FactorRow],
    indices: &[usize],
    resolved: &ResolvedRuntimeConfig,
    training_context: &TrainingContext,
) -> Result<PreparedWindowLabelColumns, String> {
    let raw_label = indices
        .iter()
        .map(|idx| rows[*idx].label)
        .collect::<Vec<_>>();
    let effective_mask = indices
        .iter()
        .map(|idx| {
            rows[*idx].label.is_finite()
                && rows[*idx].features.iter().all(|value| !value.is_infinite())
        })
        .collect::<Vec<_>>();
    let opportunity_edge =
        build_opportunity_edges(rows, indices, &effective_mask, resolved, training_context)?;
    let label = transform_window_labels(
        &raw_label,
        indices,
        rows,
        &effective_mask,
        &opportunity_edge,
        &resolved.train_label_transform,
        &resolved.opportunity,
    )?;
    let sample_weight = build_sample_weights(
        &raw_label,
        indices,
        rows,
        &effective_mask,
        &opportunity_edge,
        &resolved.sample_weight,
        &resolved.opportunity,
    )?;
    let opportunity_label = opportunity_edge
        .iter()
        .map(|edge| {
            if edge.is_finite() {
                if *edge > 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();
    Ok(PreparedWindowLabelColumns {
        label,
        raw_label,
        sample_weight,
        opportunity_label,
    })
}

fn build_opportunity_edges(
    rows: &[FactorRow],
    indices: &[usize],
    effective_mask: &[bool],
    resolved: &ResolvedRuntimeConfig,
    training_context: &TrainingContext,
) -> Result<Vec<f64>, String> {
    let mut out = vec![f64::NAN; indices.len()];
    match resolved.opportunity.mode.as_str() {
        "positive" => {
            for (position, idx) in indices.iter().enumerate() {
                if effective_mask[position] {
                    out[position] = rows[*idx].label;
                }
            }
        }
        "threshold" => {
            for (position, idx) in indices.iter().enumerate() {
                if effective_mask[position] {
                    out[position] = rows[*idx].label - resolved.opportunity.threshold;
                }
            }
        }
        "industry_excess" => {
            let mut sums: BTreeMap<(i64, String), (f64, usize)> = BTreeMap::new();
            let mut industries = vec![String::new(); indices.len()];
            for (position, idx) in indices.iter().enumerate() {
                if !effective_mask[position] {
                    continue;
                }
                let symbol = normalize_local_symbol(&rows[*idx].symbol);
                let Some(industry) = training_context.industry_by_symbol.get(&symbol) else {
                    continue;
                };
                if industry.is_empty() {
                    continue;
                }
                industries[position] = industry.clone();
                let entry = sums
                    .entry((rows[*idx].date_ns, industry.clone()))
                    .or_insert((0.0, 0));
                entry.0 += rows[*idx].label;
                entry.1 += 1;
            }
            for (position, idx) in indices.iter().enumerate() {
                if !effective_mask[position] || industries[position].is_empty() {
                    continue;
                }
                if let Some((sum, count)) =
                    sums.get(&(rows[*idx].date_ns, industries[position].clone()))
                {
                    if *count > 0 {
                        out[position] = rows[*idx].label
                            - (*sum / *count as f64)
                            - resolved.opportunity.threshold;
                    }
                }
            }
        }
        "benchmark_excess" => {
            for (position, idx) in indices.iter().enumerate() {
                if !effective_mask[position] {
                    continue;
                }
                if let Some(benchmark_return) = training_context
                    .benchmark_forward_returns
                    .get(&rows[*idx].date_ns)
                {
                    if benchmark_return.is_finite() {
                        out[position] =
                            rows[*idx].label - *benchmark_return - resolved.opportunity.threshold;
                    }
                }
            }
        }
        other => {
            return Err(format!(
                "unsupported opportunity mode in Rust training runtime: {other}"
            ))
        }
    }
    Ok(out)
}

fn transform_window_labels(
    raw_label: &[f64],
    indices: &[usize],
    rows: &[FactorRow],
    effective_mask: &[bool],
    opportunity_edge: &[f64],
    transform: &TrainLabelTransformConfig,
    opportunity: &OpportunityLabelConfig,
) -> Result<Vec<f64>, String> {
    let mut out = vec![f64::NAN; raw_label.len()];
    match transform.mode.as_str() {
        "raw" => {
            for (position, value) in raw_label.iter().enumerate() {
                if effective_mask[position] {
                    out[position] = *value;
                }
            }
        }
        "buyability_binary" => {
            for (position, edge) in opportunity_edge.iter().enumerate() {
                if edge.is_finite() {
                    out[position] = if *edge > 0.0 { 1.0 } else { 0.0 };
                }
            }
        }
        "buyability_margin_binary" => {
            for (position, edge) in opportunity_edge.iter().enumerate() {
                if !edge.is_finite() {
                    continue;
                }
                if *edge > opportunity.neutral_band {
                    out[position] = 1.0;
                } else if *edge < -opportunity.neutral_band {
                    out[position] = 0.0;
                }
            }
        }
        "cross_section_rank" | "profit_tanh" | "profit_bucket" => {
            let positions_by_date = positions_by_date(indices, rows, effective_mask);
            for positions in positions_by_date.values() {
                let finite_positions = positions
                    .iter()
                    .copied()
                    .filter(|position| raw_label[*position].is_finite())
                    .collect::<Vec<_>>();
                if finite_positions.is_empty() {
                    continue;
                }
                let finite_values = finite_positions
                    .iter()
                    .map(|position| raw_label[*position])
                    .collect::<Vec<_>>();
                let transformed = match transform.mode.as_str() {
                    "cross_section_rank" => cross_section_rank_labels(&finite_values),
                    "profit_tanh" => profit_tanh_labels(&finite_values, transform),
                    "profit_bucket" => profit_bucket_labels(&finite_values, transform),
                    _ => unreachable!(),
                };
                for (offset, position) in finite_positions.iter().enumerate() {
                    out[*position] = transformed[offset];
                }
            }
        }
        other => {
            return Err(format!(
                "unsupported train label transform in Rust training runtime: {other}"
            ))
        }
    }
    Ok(out)
}

fn positions_by_date(
    indices: &[usize],
    rows: &[FactorRow],
    effective_mask: &[bool],
) -> BTreeMap<i64, Vec<usize>> {
    let mut out: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (position, idx) in indices.iter().enumerate() {
        if effective_mask[position] {
            out.entry(rows[*idx].date_ns).or_default().push(position);
        }
    }
    out
}

fn cross_section_rank_labels(values: &[f64]) -> Vec<f64> {
    if values.len() == 1 {
        return vec![0.0];
    }
    let ranks = rank_average(values);
    let denominator = values.len() as f64 - 1.0;
    ranks
        .into_iter()
        .map(|rank| ((rank - 1.0) / denominator) - 0.5)
        .collect()
}

fn profit_tanh_labels(values: &[f64], transform: &TrainLabelTransformConfig) -> Vec<f64> {
    let scale = robust_scale(values, transform.min_scale) * transform.scale_multiplier;
    values
        .iter()
        .map(|value| {
            let adjusted = value.signum() * (value.abs() - transform.neutral_band).max(0.0);
            (adjusted / scale).tanh()
        })
        .collect()
}

fn profit_bucket_labels(values: &[f64], transform: &TrainLabelTransformConfig) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if *value <= -transform.tail_band {
                -2.0
            } else if *value > -transform.tail_band && *value < -transform.neutral_band {
                -1.0
            } else if *value >= transform.neutral_band && *value < transform.tail_band {
                1.0
            } else if *value >= transform.tail_band {
                2.0
            } else {
                0.0
            }
        })
        .collect()
}

fn build_sample_weights(
    raw_label: &[f64],
    indices: &[usize],
    rows: &[FactorRow],
    effective_mask: &[bool],
    opportunity_edge: &[f64],
    sample_weight: &SampleWeightConfig,
    opportunity: &OpportunityLabelConfig,
) -> Result<Vec<f64>, String> {
    let mut weights = vec![f64::NAN; raw_label.len()];
    match sample_weight.mode.as_str() {
        "none" => return Ok(weights),
        "opportunity_distance" => {}
        other => {
            return Err(format!(
                "unsupported sample weight mode in Rust training runtime: {other}"
            ))
        }
    }
    let default_scale = if opportunity.neutral_band > 0.0 {
        opportunity.neutral_band
    } else {
        0.01
    };
    let scale = sample_weight.scale.unwrap_or(default_scale).max(1e-6);
    let power = sample_weight.power.max(1e-6);
    let min_weight = sample_weight.min.max(0.0);
    for (position, edge) in opportunity_edge.iter().enumerate() {
        if !effective_mask[position] || !edge.is_finite() {
            continue;
        }
        let mut weight = 1.0 + (edge.abs() / scale).powf(power);
        if min_weight > 0.0 {
            weight = weight.max(min_weight);
        }
        weights[position] = weight;
    }
    if sample_weight.date_normalize {
        let mut sums: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
        for (position, idx) in indices.iter().enumerate() {
            if weights[position].is_finite() {
                let entry = sums.entry(rows[*idx].date_ns).or_insert((0.0, 0));
                entry.0 += weights[position];
                entry.1 += 1;
            }
        }
        for (position, idx) in indices.iter().enumerate() {
            if weights[position].is_finite() {
                if let Some((sum, count)) = sums.get(&rows[*idx].date_ns) {
                    let mean = *sum / *count as f64;
                    if mean.is_finite() && mean > 0.0 {
                        weights[position] /= mean;
                    }
                }
            }
        }
    }
    let finite = weights
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if !finite.is_empty() {
        let mean = finite.iter().sum::<f64>() / finite.len() as f64;
        if mean.is_finite() && mean > 0.0 {
            for weight in &mut weights {
                if weight.is_finite() {
                    *weight /= mean;
                }
            }
        }
    }
    Ok(weights)
}

fn keep_positions_for_training_window(
    rows: &[FactorRow],
    indices: &[usize],
    prepared: &PreparedWindowLabelColumns,
) -> Vec<usize> {
    indices
        .iter()
        .enumerate()
        .filter_map(|(position, idx)| {
            if rows[*idx].features.iter().all(|value| !value.is_infinite())
                && prepared.label[position].is_finite()
            {
                Some(position)
            } else {
                None
            }
        })
        .collect()
}

fn keep_positions_for_prediction_window(rows: &[FactorRow], indices: &[usize]) -> Vec<usize> {
    indices
        .iter()
        .enumerate()
        .filter_map(|(position, idx)| {
            if rows[*idx].features.iter().all(|value| !value.is_infinite()) {
                Some(position)
            } else {
                None
            }
        })
        .collect()
}

fn build_window_prediction_rows(
    rows: &[FactorRow],
    indices: &[usize],
    keep_positions: &[usize],
    predictions: &[f64],
) -> Result<Vec<LongValueRow>, String> {
    if keep_positions.len() != predictions.len() {
        return Err(format!(
            "Python test prediction length mismatch: expected {} rows after filter, got {}",
            keep_positions.len(),
            predictions.len()
        ));
    }
    Ok(keep_positions
        .iter()
        .zip(predictions.iter())
        .map(|(position, prediction)| {
            let row = &rows[indices[*position]];
            LongValueRow {
                date_ns: row.date_ns,
                instrument: row.symbol.clone(),
                value: *prediction,
            }
        })
        .collect())
}

struct PythonLgbmWindowBuildInput<'a> {
    rows: &'a [FactorRow],
    train_indices: &'a [usize],
    train_keep_positions: &'a [usize],
    train_prepared: &'a PreparedWindowLabelColumns,
    valid_indices: &'a [usize],
    valid_keep_positions: &'a [usize],
    valid_prepared: &'a PreparedWindowLabelColumns,
    test_indices: &'a [usize],
    test_keep_positions: &'a [usize],
}

fn build_python_lgbm_window(input: &PythonLgbmWindowBuildInput<'_>) -> PreparedPythonLgbmWindow {
    PreparedPythonLgbmWindow {
        train_rows: input.train_keep_positions.len(),
        valid_rows: input.valid_keep_positions.len(),
        test_rows: input.test_keep_positions.len(),
        train_feature_bytes: encode_feature_rows_le(
            input.rows,
            input.train_indices,
            input.train_keep_positions,
        ),
        valid_feature_bytes: encode_feature_rows_le(
            input.rows,
            input.valid_indices,
            input.valid_keep_positions,
        ),
        test_feature_bytes: encode_feature_rows_le(
            input.rows,
            input.test_indices,
            input.test_keep_positions,
        ),
        train_label_bytes: encode_f64_values_le(&collect_prepared_values_by_position(
            &input.train_prepared.label,
            input.train_keep_positions,
        )),
        valid_label_bytes: encode_f64_values_le(&collect_prepared_values_by_position(
            &input.valid_prepared.label,
            input.valid_keep_positions,
        )),
        raw_valid_label_bytes: encode_f64_values_le(&collect_prepared_values_by_position(
            &input.valid_prepared.raw_label,
            input.valid_keep_positions,
        )),
        train_date_ns_bytes: encode_i64_values_le(&collect_date_ns_by_position(
            input.rows,
            input.train_indices,
            input.train_keep_positions,
        )),
        valid_date_ns_bytes: encode_i64_values_le(&collect_date_ns_by_position(
            input.rows,
            input.valid_indices,
            input.valid_keep_positions,
        )),
        train_sample_weight_bytes: encode_optional_f64_values_le(
            &collect_prepared_values_by_position(
                &input.train_prepared.sample_weight,
                input.train_keep_positions,
            ),
        ),
        valid_sample_weight_bytes: encode_optional_f64_values_le(
            &collect_prepared_values_by_position(
                &input.valid_prepared.sample_weight,
                input.valid_keep_positions,
            ),
        ),
    }
}

fn encode_feature_rows_le(
    rows: &[FactorRow],
    indices: &[usize],
    keep_positions: &[usize],
) -> Vec<u8> {
    let mut values = Vec::new();
    for position in keep_positions {
        values.extend_from_slice(&rows[indices[*position]].features);
    }
    encode_f64_values_le(&values)
}

fn collect_prepared_values_by_position(values: &[f64], keep_positions: &[usize]) -> Vec<f64> {
    keep_positions
        .iter()
        .map(|position| values[*position])
        .collect()
}

fn collect_date_ns_by_position(
    rows: &[FactorRow],
    indices: &[usize],
    keep_positions: &[usize],
) -> Vec<i64> {
    keep_positions
        .iter()
        .map(|position| rows[indices[*position]].date_ns)
        .collect()
}

fn encode_optional_f64_values_le(values: &[f64]) -> Option<Vec<u8>> {
    if values.iter().any(|value| value.is_finite()) {
        Some(encode_f64_values_le(values))
    } else {
        None
    }
}

fn encode_f64_values_le(values: &[f64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn encode_i64_values_le(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn compute_validation_topk_summary(
    rows: &[FactorRow],
    indices: &[usize],
    keep_positions: &[usize],
    prepared: &PreparedWindowLabelColumns,
    predictions: &[f64],
    topk: usize,
) -> Result<JsonValue, String> {
    if keep_positions.len() != predictions.len() {
        return Err(format!(
            "Python validation prediction length mismatch: expected {} rows after filter, got {}",
            keep_positions.len(),
            predictions.len()
        ));
    }
    if keep_positions.is_empty() {
        return Ok(empty_validation_topk_summary());
    }

    let topk = topk.max(1);
    let mut group_start = 0usize;
    let mut daily_top1_label = Vec::new();
    let mut daily_top1_positive = Vec::new();
    let mut daily_topk_label_mean = Vec::new();
    let mut daily_topk_label_median = Vec::new();
    let mut daily_topk_min_label = Vec::new();
    let mut daily_topk_positive_rate = Vec::new();
    let mut daily_topk_excess_mean = Vec::new();
    let mut daily_top1_opportunity = Vec::new();
    let mut daily_topk_opportunity_rate = Vec::new();

    while group_start < keep_positions.len() {
        let first_position = keep_positions[group_start];
        let first_date = rows[indices[first_position]].date_ns;
        let mut group_end = group_start + 1;
        while group_end < keep_positions.len() {
            let position = keep_positions[group_end];
            if rows[indices[position]].date_ns != first_date {
                break;
            }
            group_end += 1;
        }

        let group_len = group_end - group_start;
        if group_len > 0 {
            let mut order = (group_start..group_end).collect::<Vec<_>>();
            order.sort_by(|left, right| predictions[*left].total_cmp(&predictions[*right]));
            let selected_count = topk.min(group_len);
            let selected = &order[(order.len() - selected_count)..];
            let mut top1_index = selected[0];
            for index in selected.iter().copied().skip(1) {
                if predictions[index] > predictions[top1_index] {
                    top1_index = index;
                }
            }

            let mut selected_labels = Vec::with_capacity(selected.len());
            let mut selected_opportunity = Vec::with_capacity(selected.len());
            let mut group_labels = Vec::with_capacity(group_len);
            for position in keep_positions.iter().take(group_end).skip(group_start) {
                group_labels.push(rows[indices[*position]].label);
            }
            for absolute_index in selected.iter().copied() {
                let position = keep_positions[absolute_index];
                selected_labels.push(rows[indices[position]].label);
                selected_opportunity.push(prepared.opportunity_label[position]);
            }

            let top1_position = keep_positions[top1_index];
            daily_top1_label.push(rows[indices[top1_position]].label);
            daily_top1_positive.push(if rows[indices[top1_position]].label > 0.0 {
                1.0
            } else {
                0.0
            });
            daily_topk_label_mean.push(mean(&selected_labels));
            daily_topk_label_median.push(median(&selected_labels));
            daily_topk_min_label.push(
                selected_labels
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
            );
            daily_topk_positive_rate.push(
                selected_labels.iter().filter(|value| **value > 0.0).count() as f64
                    / selected_labels.len() as f64,
            );
            daily_topk_excess_mean.push(mean(&selected_labels) - mean(&group_labels));
            daily_top1_opportunity.push(prepared.opportunity_label[top1_position]);
            daily_topk_opportunity_rate.push(mean_ignore_nan(&selected_opportunity));
        }

        group_start = group_end;
    }

    if daily_top1_label.is_empty() {
        return Ok(empty_validation_topk_summary());
    }

    Ok(serde_json::json!({
        "valid_topk_days": daily_top1_label.len(),
        "valid_top1_label_mean": json_f64_or_null(mean(&daily_top1_label)),
        "valid_top1_positive_rate": json_f64_or_null(mean(&daily_top1_positive)),
        "valid_topk_label_mean": json_f64_or_null(mean(&daily_topk_label_mean)),
        "valid_topk_label_median": json_f64_or_null(mean(&daily_topk_label_median)),
        "valid_topk_min_label_mean": json_f64_or_null(mean(&daily_topk_min_label)),
        "valid_topk_positive_rate": json_f64_or_null(mean(&daily_topk_positive_rate)),
        "valid_topk_excess_mean": json_f64_or_null(mean(&daily_topk_excess_mean)),
        "valid_top1_opportunity_rate": json_f64_or_null(mean_ignore_nan(&daily_top1_opportunity)),
        "valid_topk_opportunity_rate": json_f64_or_null(mean_ignore_nan(&daily_topk_opportunity_rate)),
    }))
}

fn empty_validation_topk_summary() -> JsonValue {
    serde_json::json!({
        "valid_topk_days": 0,
        "valid_top1_label_mean": JsonValue::Null,
        "valid_top1_positive_rate": JsonValue::Null,
        "valid_topk_label_mean": JsonValue::Null,
        "valid_topk_label_median": JsonValue::Null,
        "valid_topk_min_label_mean": JsonValue::Null,
        "valid_topk_positive_rate": JsonValue::Null,
        "valid_topk_excess_mean": JsonValue::Null,
        "valid_top1_opportunity_rate": JsonValue::Null,
        "valid_topk_opportunity_rate": JsonValue::Null,
    })
}

struct WindowTrainingSummaryInput<'a> {
    python_summary: &'a JsonValue,
    paths: &'a WindowDataPaths,
    window: &'a RollingWindow,
    resolved: &'a ResolvedRuntimeConfig,
    raw_train_rows: usize,
    raw_valid_rows: usize,
    raw_test_rows: usize,
    train_rows: usize,
    valid_rows: usize,
    test_rows: usize,
    feature_count: usize,
    valid_topk_summary: JsonValue,
}

fn build_window_training_summary(input: &WindowTrainingSummaryInput<'_>) -> JsonValue {
    let resolved = input.resolved;
    let window = input.window;
    let train_label_transform_mode = resolved.train_label_transform.mode.clone();
    let train_label_space = if resolved
        .train_label_transform
        .mode
        .starts_with("buyability")
    {
        "binary_target"
    } else {
        "return_target"
    };
    let mut summary = serde_json::json!({
        "window_start": format_date_ns(window.test_start_ns),
        "window_end": format_date_ns(window.test_end_ns),
        "train_start": format_date_ns(window.train_start_ns),
        "train_end": format_date_ns(window.train_end_ns),
        "valid_start": format_date_ns(window.valid_start_ns),
        "valid_end": format_date_ns(window.valid_end_ns),
        "signal_horizon": resolved.signal_horizon,
        "label_embargo_days": resolved.label_embargo_days,
        "train_rows": input.train_rows,
        "valid_rows": input.valid_rows,
        "test_rows": input.test_rows,
        "raw_train_rows": input.raw_train_rows,
        "raw_valid_rows": input.raw_valid_rows,
        "raw_test_rows": input.raw_test_rows,
        "train_rows_dropped_after_filter": input.raw_train_rows.saturating_sub(input.train_rows),
        "valid_rows_dropped_after_filter": input.raw_valid_rows.saturating_sub(input.valid_rows),
        "test_rows_dropped_after_filter": input.raw_test_rows.saturating_sub(input.test_rows),
        "train_rows_dropped_after_label_transform": 0,
        "valid_rows_dropped_after_label_transform": 0,
        "feature_count": input.feature_count,
        "prediction_path": input.paths.prediction_path.display().to_string(),
        "train_label_transform_mode": train_label_transform_mode,
        "train_label_space": train_label_space,
        "valid_custom_metric_label_space": "raw_return",
        "opportunity_label_mode": resolved.opportunity.mode,
        "opportunity_label_threshold": resolved.opportunity.threshold,
        "opportunity_label_neutral_band": resolved.opportunity.neutral_band,
        "train_sample_weight_mode": resolved.sample_weight.mode,
        "validation_topk": resolved
            .lgbm_config
            .get("validation_topk")
            .and_then(JsonValue::as_u64)
            .unwrap_or(10),
    });
    merge_json_object(&mut summary, input.python_summary);
    merge_json_object(&mut summary, &input.valid_topk_summary);
    summary
}

fn merge_json_object(target: &mut JsonValue, extra: &JsonValue) {
    let Some(target_obj) = target.as_object_mut() else {
        return;
    };
    let Some(extra_obj) = extra.as_object() else {
        return;
    };
    for (key, value) in extra_obj {
        target_obj.insert(key.clone(), value.clone());
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn mean_ignore_nan(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    mean(&finite)
}

fn build_lgbm_bundle_timing_summary(
    phase_timer: &RuntimePhaseTimer,
    timing_path: &Path,
    window_timing_records: &[JsonValue],
) -> JsonValue {
    let mut summary = phase_timer.as_json(timing_path);
    if let Some(object) = summary.as_object_mut() {
        object.insert(
            "windows".to_owned(),
            JsonValue::Array(window_timing_records.to_vec()),
        );
    }
    summary
}

fn python_phase_seconds(summary: &JsonValue, key: &str) -> f64 {
    summary.get(key).and_then(JsonValue::as_f64).unwrap_or(0.0)
}

fn python_phase_timings_json(summary: &JsonValue) -> JsonValue {
    serde_json::json!({
        "import_native_lgbm_seconds": json_f64_or_null(python_phase_seconds(summary, "python_import_native_lgbm_seconds")),
        "materialize_inputs_seconds": json_f64_or_null(python_phase_seconds(summary, "python_materialize_inputs_seconds")),
        "model_load_seconds": json_f64_or_null(python_phase_seconds(summary, "python_model_load_seconds")),
        "fit_seconds": json_f64_or_null(python_phase_seconds(summary, "python_fit_seconds")),
        "model_save_seconds": json_f64_or_null(python_phase_seconds(summary, "python_model_save_seconds")),
        "feature_importance_seconds": json_f64_or_null(python_phase_seconds(summary, "python_feature_importance_seconds")),
        "training_history_seconds": json_f64_or_null(python_phase_seconds(summary, "python_training_history_seconds")),
        "predict_valid_seconds": json_f64_or_null(python_phase_seconds(summary, "python_predict_valid_seconds")),
        "predict_test_seconds": json_f64_or_null(python_phase_seconds(summary, "python_predict_test_seconds")),
        "summary_build_seconds": json_f64_or_null(python_phase_seconds(summary, "python_summary_build_seconds")),
        "bridge_total_seconds": json_f64_or_null(python_phase_seconds(summary, "python_bridge_total_seconds")),
    })
}

fn accumulate_python_phase_timings(phase_timer: &mut RuntimePhaseTimer, summary: &JsonValue) {
    for (phase_name, key) in [
        (
            "python_import_native_lgbm",
            "python_import_native_lgbm_seconds",
        ),
        (
            "python_materialize_inputs",
            "python_materialize_inputs_seconds",
        ),
        ("python_model_load", "python_model_load_seconds"),
        ("python_fit", "python_fit_seconds"),
        ("python_model_save", "python_model_save_seconds"),
        (
            "python_feature_importance",
            "python_feature_importance_seconds",
        ),
        ("python_training_history", "python_training_history_seconds"),
        ("python_predict_valid", "python_predict_valid_seconds"),
        ("python_predict_test", "python_predict_test_seconds"),
        ("python_summary_build", "python_summary_build_seconds"),
        ("python_bridge_total", "python_bridge_total_seconds"),
    ] {
        let seconds = python_phase_seconds(summary, key);
        if seconds > 0.0 {
            phase_timer.add(phase_name, seconds, 1, 0, 1, 0);
        }
    }
}

fn round_six(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn json_f64_or_null(value: f64) -> JsonValue {
    if value.is_finite() {
        serde_json::Number::from_f64(value)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null)
    } else {
        JsonValue::Null
    }
}

fn decode_f64_le_bytes(bytes: &[u8]) -> Result<Vec<f64>, String> {
    if !bytes.len().is_multiple_of(mem::size_of::<f64>()) {
        return Err(format!(
            "prediction byte payload length {} is not divisible by {}",
            bytes.len(),
            mem::size_of::<f64>()
        ));
    }
    Ok(bytes
        .chunks_exact(mem::size_of::<f64>())
        .map(|chunk| {
            let mut raw = [0u8; std::mem::size_of::<f64>()];
            raw.copy_from_slice(chunk);
            f64::from_le_bytes(raw)
        })
        .collect())
}

fn write_window_frame_parquet(
    path: &Path,
    rows: &[FactorRow],
    indices: &[usize],
    feature_names: &[String],
    resolved: &ResolvedRuntimeConfig,
    training_context: &TrainingContext,
) -> Result<(), String> {
    let prepared_labels =
        build_prepared_window_label_columns(rows, indices, resolved, training_context)?;
    let mut fields = vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("instrument", DataType::Utf8, false),
        Field::new("label", DataType::Float64, true),
        Field::new("raw_label", DataType::Float64, true),
        Field::new("backtest_label", DataType::Float64, true),
        Field::new("sample_weight", DataType::Float64, true),
        Field::new("opportunity_label", DataType::Float64, true),
    ];
    for feature in feature_names {
        fields.push(Field::new(feature, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));
    let mut arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(TimestampNanosecondArray::from(
            indices
                .iter()
                .map(|idx| rows[*idx].date_ns)
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            indices
                .iter()
                .map(|idx| rows[*idx].symbol.clone())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(prepared_labels.label)),
        Arc::new(Float64Array::from(prepared_labels.raw_label)),
        Arc::new(Float64Array::from(
            indices
                .iter()
                .map(|idx| rows[*idx].backtest_label)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(prepared_labels.sample_weight)),
        Arc::new(Float64Array::from(prepared_labels.opportunity_label)),
    ];
    for feature_index in 0..feature_names.len() {
        arrays.push(Arc::new(Float64Array::from(
            indices
                .iter()
                .map(|idx| rows[*idx].features[feature_index])
                .collect::<Vec<_>>(),
        )));
    }
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|err| format!("failed to build window batch: {err}"))?;
    write_record_batch_parquet(path, schema, &batch)
}

fn write_long_value_parquet(
    path: &Path,
    value_column: &str,
    rows: &[LongValueRow],
) -> Result<(), String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("instrument", DataType::Utf8, false),
        Field::new(value_column, DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(TimestampNanosecondArray::from(
                rows.iter().map(|row| row.date_ns).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.instrument.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|row| row.value).collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|err| format!("failed to build long-value batch: {err}"))?;
    write_record_batch_parquet(path, schema, &batch)
}

fn write_record_batch_parquet(
    path: &Path,
    schema: Arc<Schema>,
    batch: &RecordBatch,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|err| format!("failed to create parquet writer {}: {err}", path.display()))?;
    writer
        .write(batch)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    writer
        .close()
        .map_err(|err| format!("failed to close {}: {err}", path.display()))?;
    Ok(())
}

fn write_training_summary(path: &Path, records: &[BTreeMap<String, String>]) -> Result<(), String> {
    let mut headers = BTreeSet::new();
    for record in records {
        headers.extend(record.keys().cloned());
    }
    let headers = headers.into_iter().collect::<Vec<_>>();
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record(&headers)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    for record in records {
        let row = headers
            .iter()
            .map(|key| record.get(key).cloned().unwrap_or_default())
            .collect::<Vec<_>>();
        writer
            .write_record(&row)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn write_metadata(
    path: &Path,
    resolved: &ResolvedRuntimeConfig,
    data: &LoadedFactorData,
) -> Result<(), String> {
    let metadata = serde_json::json!({
        "model_name": "lgbm",
        "generator": "rust",
        "runtime_builder": "rust",
        "python_role": "lightgbm_fit_predict_only",
        "data_source": resolved.data_source,
        "label_column": resolved.label_column,
        "signal_label_column": resolved.label_column,
        "signal_horizon": resolved.signal_horizon,
        "signal_label_horizon": resolved.signal_horizon,
        "backtest_label_column": resolved.backtest_label_column,
        "portfolio_return_label_column": resolved.backtest_label_column,
        "backtest_label_horizon": 1,
        "backtest_label_semantics": "daily_realized_return",
        "retrain_step": resolved.retrain_step,
        "train_days": resolved.train_days,
        "valid_days": resolved.valid_days,
        "label_embargo_days": resolved.label_embargo_days,
        "test_start": format_date_ns(resolved.test_start_ns),
        "test_end": format_date_ns(resolved.test_end_ns),
        "selected_feature_count": data.selected_feature_names.len(),
        "selected_features": data.selected_feature_names,
        "selected_feature_sources": data.selected_feature_sources,
        "cross_sectional_rank_enabled": resolved.cross_sectional_rank,
        "cross_sectional_rank_exclude_columns": sorted_string_set(&resolved.cross_sectional_rank_exclude_columns),
        "universe": resolved.universe_name.clone().unwrap_or_else(|| "all".to_owned()),
        "universe_dir": resolved.universe_dir.display().to_string(),
        "train_label_transform_mode": resolved.train_label_transform.mode,
        "train_label_space": if resolved.train_label_transform.mode.starts_with("buyability") { "binary_target" } else { "return_target" },
        "opportunity_label_mode": resolved.opportunity.mode,
        "opportunity_label_threshold": resolved.opportunity.threshold,
        "opportunity_label_neutral_band": resolved.opportunity.neutral_band,
        "train_sample_weight_mode": resolved.sample_weight.mode,
        "reference_baselines_enabled": false,
    });
    fs::write(
        path,
        serde_json::to_string_pretty(&metadata)
            .map_err(|err| format!("failed to serialize metadata: {err}"))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn write_config_snapshot(path: &Path, cfg: &YamlValue) -> Result<(), String> {
    fs::write(
        path,
        serde_yaml::to_string(cfg)
            .map_err(|err| format!("failed to serialize config snapshot: {err}"))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn build_row_indices_by_date(rows: &[FactorRow]) -> BTreeMap<i64, Vec<usize>> {
    let mut by_date: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (row_index, row) in rows.iter().enumerate() {
        by_date.entry(row.date_ns).or_default().push(row_index);
    }
    by_date
}

fn collect_indices_for_date_range(
    by_date: &BTreeMap<i64, Vec<usize>>,
    start_ns: i64,
    end_ns: i64,
) -> Vec<usize> {
    by_date
        .range(start_ns..=end_ns)
        .flat_map(|(_, indices)| indices.iter().copied())
        .collect()
}

fn read_factor_store_meta(factor_store: &Path) -> Result<JsonValue, String> {
    let path = factor_store.join("meta.json");
    let file =
        File::open(&path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    serde_json::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn read_meta_string_list(meta: &JsonValue, key: &str) -> Result<Vec<String>, String> {
    let Some(values) = meta.get(key).and_then(JsonValue::as_array) else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("metadata {key} contains a non-string value"))
        })
        .collect()
}

fn read_meta_dates(meta: &JsonValue, key: &str) -> Result<Vec<i64>, String> {
    let mut dates = read_meta_string_list(meta, key)?
        .iter()
        .map(|value| parse_date_ns(value))
        .collect::<Result<Vec<_>, _>>()?;
    dates.sort_unstable();
    dates.dedup();
    Ok(dates)
}

fn select_features_from_meta(
    all_features: &[String],
    max_features: usize,
) -> Result<Vec<String>, String> {
    if all_features.is_empty() {
        return Err("factor-store metadata has no feature_names list".to_owned());
    }
    if max_features == 0 {
        return Ok(all_features.to_vec());
    }
    Ok(all_features.iter().take(max_features).cloned().collect())
}

fn validate_selected_sources(
    all_features: &[String],
    selected_sources: &[String],
) -> Result<Vec<String>, String> {
    let available = all_features
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let missing = selected_sources
        .iter()
        .filter(|feature| !available.contains(feature.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "requested feature source(s) not found in factor-store metadata: {}",
            missing.join(",")
        ));
    }
    Ok(selected_sources.to_vec())
}

fn projected_columns(resolved: &ResolvedRuntimeConfig, selected_sources: &[String]) -> Vec<String> {
    let mut columns = vec![
        "date".to_owned(),
        "symbol".to_owned(),
        resolved.label_column.clone(),
        resolved.backtest_label_column.clone(),
    ];
    columns.extend(selected_sources.iter().cloned());
    sorted_unique_nonempty(columns)
}

fn sorted_unique_nonempty(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if !trimmed.is_empty() && seen.insert(trimmed.to_owned()) {
            out.push(trimmed.to_owned());
        }
    }
    out
}

fn sanitize_label(value: f64) -> f64 {
    if !value.is_finite() || value.abs() > DEFAULT_LABEL_ABS_CAP {
        f64::NAN
    } else {
        value
    }
}

fn robust_scale(values: &[f64], min_scale: f64) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return min_scale;
    }
    let median_value = median(&finite);
    let deviations = finite
        .iter()
        .map(|value| (value - median_value).abs())
        .collect::<Vec<_>>();
    let mad = median(&deviations) * 1.4826;
    if mad.is_finite() && mad >= min_scale {
        return mad;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    let std = variance.sqrt();
    if std.is_finite() && std >= min_scale {
        return std;
    }
    let abs_values = finite.iter().map(|value| value.abs()).collect::<Vec<_>>();
    let abs_median = median(&abs_values);
    if abs_median.is_finite() && abs_median >= min_scale {
        return abs_median;
    }
    min_scale
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn rank_average(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let average_rank = 0.5 * (start as f64 + end as f64 - 1.0) + 1.0;
        for order_index in start..end {
            ranks[order[order_index]] = average_rank;
        }
        start = end;
    }
    ranks
}

fn resolve_output_artifact_dir(output_dir: &Path) -> PathBuf {
    if output_dir
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name == PREDICTION_ARTIFACT_DIRNAME)
    {
        output_dir.to_path_buf()
    } else {
        output_dir.join(PREDICTION_ARTIFACT_DIRNAME)
    }
}

fn json_object_to_string_map(value: &JsonValue) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Some(object) = value.as_object() {
        for (key, value) in object {
            out.insert(
                key.clone(),
                match value {
                    JsonValue::Null => String::new(),
                    JsonValue::String(inner) => inner.clone(),
                    _ => value.to_string(),
                },
            );
        }
    }
    out
}

fn sorted_string_set(values: &BTreeSet<String>) -> Vec<String> {
    values.iter().cloned().collect()
}

fn load_resolved_config(options: &LgbmBundleOptions) -> Result<YamlValue, String> {
    let mut cfg = read_yaml_file(&options.config)?;
    if !options.config_is_snapshot {
        if let Some(experiment_profile) = &options.experiment_profile {
            let experiment_cfg = resolve_named_profile(
                "configs/experiment_profiles.yaml",
                experiment_profile,
                "experiment",
            )?;
            deep_merge_yaml(&mut cfg, experiment_cfg);
        }
        let model_profile = options
            .model_profile
            .clone()
            .or_else(|| yaml_path_string(&cfg, &["model", "profile"]));
        if let Some(model_profile) = model_profile {
            let model_cfg =
                resolve_named_profile("configs/model_profiles.yaml", &model_profile, "model")?;
            deep_merge_yaml(&mut cfg, model_cfg);
            set_yaml_dotted(&mut cfg, "model.profile", YamlValue::String(model_profile))?;
        }
    }
    Ok(cfg)
}

fn resolve_named_profile(
    profile_config_path: &str,
    profile_name: &str,
    profile_kind: &str,
) -> Result<YamlValue, String> {
    let profile_data = read_yaml_file(Path::new(profile_config_path))?;
    let profiles = yaml_mapping_get(yaml_as_mapping(&profile_data)?, "profiles")
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| format!("{profile_config_path} has no profiles mapping"))?;
    resolve_profile_from_mapping(profiles, profile_name, profile_kind, &mut Vec::new())
}

fn resolve_profile_from_mapping(
    profiles: &YamlMapping,
    profile_name: &str,
    profile_kind: &str,
    stack: &mut Vec<String>,
) -> Result<YamlValue, String> {
    if stack.iter().any(|item| item == profile_name) {
        return Err(format!(
            "cyclic {profile_kind} profile extends chain: {} -> {profile_name}",
            stack.join(" -> ")
        ));
    }
    let entry = profiles
        .get(YamlValue::String(profile_name.to_owned()))
        .ok_or_else(|| format!("unknown {profile_kind} profile: {profile_name}"))?
        .clone();
    let entry_mapping = yaml_as_mapping(&entry)?.clone();
    let mut merged =
        if let Some(path) = yaml_mapping_get(&entry_mapping, "path").and_then(YamlValue::as_str) {
            read_yaml_file(Path::new(path))?
        } else {
            YamlValue::Mapping(YamlMapping::new())
        };
    let extends_name = yaml_mapping_get(&entry_mapping, "extends")
        .and_then(YamlValue::as_str)
        .map(str::to_owned);
    if let Some(parent_name) = extends_name {
        stack.push(profile_name.to_owned());
        let mut parent = resolve_profile_from_mapping(profiles, &parent_name, profile_kind, stack)?;
        stack.pop();
        deep_merge_yaml(&mut parent, merged);
        merged = parent;
    }
    let mut inline = YamlValue::Mapping(entry_mapping);
    remove_yaml_mapping_key(&mut inline, "path");
    remove_yaml_mapping_key(&mut inline, "extends");
    deep_merge_yaml(&mut merged, inline);
    Ok(merged)
}

fn resolve_feature_profile(
    profile_name: &str,
) -> Result<Option<FeatureSelectionResolution>, String> {
    let profile = resolve_named_profile("configs/feature_profiles.yaml", profile_name, "feature")?;
    let Some(mapping) = profile.as_mapping() else {
        return Ok(None);
    };
    let Some(selected_columns_value) = yaml_mapping_get(mapping, "selected_columns") else {
        return Ok(None);
    };
    let selected_columns = yaml_string_sequence(selected_columns_value)?;
    let drop_columns = yaml_mapping_get(mapping, "drop_columns")
        .map(yaml_string_sequence)
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let add_columns = yaml_mapping_get(mapping, "add_columns")
        .map(yaml_string_sequence)
        .transpose()?
        .unwrap_or_default();
    let repeat_columns = yaml_mapping_get(mapping, "repeat_columns")
        .map(yaml_string_usize_mapping)
        .transpose()?
        .unwrap_or_default();
    let cross_sectional_rank_exclude_columns =
        yaml_mapping_get(mapping, "cross_sectional_rank_exclude_columns")
            .map(yaml_string_sequence)
            .transpose()?
            .unwrap_or_default()
            .into_iter()
            .collect::<BTreeSet<_>>();
    let mut base = selected_columns
        .into_iter()
        .filter(|item| !drop_columns.contains(item))
        .collect::<Vec<_>>();
    let mut seen = base.iter().cloned().collect::<BTreeSet<_>>();
    for item in add_columns {
        if seen.insert(item.clone()) {
            base.push(item);
        }
    }
    let mut selected_names = Vec::new();
    let mut selected_sources = Vec::new();
    for source in base {
        let repeat_count = repeat_columns.get(&source).copied().unwrap_or(1).max(1);
        for repeat_index in 0..repeat_count {
            let name = if repeat_index == 0 {
                source.clone()
            } else {
                format!("{source}__rep{}", repeat_index + 1)
            };
            selected_names.push(name);
            selected_sources.push(canonical_feature_source(&source));
        }
    }
    let factor_store_name = yaml_mapping_get(mapping, "factor_store_name")
        .and_then(YamlValue::as_str)
        .unwrap_or("full_factor_space")
        .to_owned();
    Ok(Some(FeatureSelectionResolution {
        selected_names,
        selected_sources,
        factor_store_name,
        cross_sectional_rank_exclude_columns,
    }))
}

fn resolve_cross_sectional_rank_exclude_columns(
    cfg: &YamlValue,
    feature_resolution: &FeatureSelectionResolution,
) -> Result<BTreeSet<String>, String> {
    if let Some(value) = yaml_path(
        cfg,
        &[
            "features",
            "transforms",
            "cross_sectional_rank_exclude_columns",
        ],
    ) {
        return Ok(yaml_string_sequence(value)?.into_iter().collect());
    }
    Ok(feature_resolution
        .cross_sectional_rank_exclude_columns
        .clone())
}

fn canonical_feature_source(feature_name: &str) -> String {
    match feature_name {
        "TEMP_rsv_5" => "RSV5",
        "TEMP_rsv_10" => "RSV10",
        "TEMP_rsv_20" => "RSV20",
        "TEMP_rsv_30" => "RSV30",
        "TEMP_rsv_60" => "RSV60",
        "TEMP_corr_cv_5" => "CORR5",
        "TEMP_corr_cv_10" => "CORR10",
        "TEMP_corr_cv_20" => "CORR20",
        "TEMP_corr_cv_30" => "CORR30",
        "TEMP_corr_cv_60" => "CORR60",
        "TEMP_ret_20" => "LGBM_ret_20",
        "TEMP_ret_60" => "LGBM_ret_60",
        "TEMP_ma_gap_20" => "LGBM_dist_ma20",
        "TEMP_ma_gap_60" => "LGBM_dist_ma60",
        "TEMP_ma_gap_120" => "LGBM_dist_ma120",
        "TEMP_std_60" => "LGBM_std_60",
        "TEMP_amihud_20" => "LGBM_amihud_20",
        "TEMP_turnover_mean_20" => "LGBM_turnover_20",
        "TEMP_high_gap_20" => "LGBM_dist_high_20",
        "TEMP_low_gap_20" => "LGBM_dist_low_20",
        other => other,
    }
    .to_owned()
}

fn resolve_lgbm_config(cfg: &YamlValue) -> Result<JsonValue, String> {
    let lgbm = yaml_path(cfg, &["lgbm"])
        .ok_or_else(|| "resolved config is missing lgbm block".to_owned())?;
    let json = serde_json::to_value(lgbm)
        .map_err(|err| format!("failed to convert lgbm config to JSON: {err}"))?;
    Ok(json)
}

fn resolve_train_label_transform_config(
    cfg: &YamlValue,
) -> Result<TrainLabelTransformConfig, String> {
    let mode = yaml_path_string(cfg, &["label", "train_transform", "mode"])
        .unwrap_or_else(|| "raw".to_owned())
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        mode.as_str(),
        "raw"
            | "profit_tanh"
            | "profit_bucket"
            | "cross_section_rank"
            | "buyability_binary"
            | "buyability_margin_binary"
    ) {
        return Err(format!("unsupported training label transform mode: {mode}"));
    }
    let neutral_band =
        yaml_path_f64(cfg, &["label", "train_transform", "neutral_band"]).unwrap_or(0.0);
    if neutral_band < 0.0 {
        return Err("label.train_transform.neutral_band must be >= 0".to_owned());
    }
    let default_tail_band = if neutral_band > 0.0 {
        neutral_band * 3.0
    } else {
        0.03
    };
    let tail_band =
        yaml_path_f64(cfg, &["label", "train_transform", "tail_band"]).unwrap_or(default_tail_band);
    if tail_band < neutral_band {
        return Err(
            "label.train_transform.tail_band must be >= label.train_transform.neutral_band"
                .to_owned(),
        );
    }
    let scale_multiplier =
        yaml_path_f64(cfg, &["label", "train_transform", "scale_multiplier"]).unwrap_or(1.0);
    if scale_multiplier <= 0.0 {
        return Err("label.train_transform.scale_multiplier must be > 0".to_owned());
    }
    let min_scale = yaml_path_f64(cfg, &["label", "train_transform", "min_scale"]).unwrap_or(1e-4);
    if min_scale <= 0.0 {
        return Err("label.train_transform.min_scale must be > 0".to_owned());
    }
    Ok(TrainLabelTransformConfig {
        mode,
        neutral_band,
        tail_band,
        scale_multiplier,
        min_scale,
    })
}

fn resolve_opportunity_label_config(cfg: &YamlValue) -> Result<OpportunityLabelConfig, String> {
    let mode = yaml_path_string(cfg, &["label", "opportunity", "mode"])
        .unwrap_or_else(|| "positive".to_owned())
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        mode.as_str(),
        "positive" | "threshold" | "industry_excess" | "benchmark_excess"
    ) {
        return Err(format!("unsupported opportunity mode: {mode}"));
    }
    let threshold = yaml_path_f64(cfg, &["label", "opportunity", "threshold"]).unwrap_or(0.0);
    let neutral_band = yaml_path_f64(cfg, &["label", "opportunity", "neutral_band"]).unwrap_or(0.0);
    if neutral_band < 0.0 {
        return Err("label.opportunity.neutral_band must be >= 0".to_owned());
    }
    Ok(OpportunityLabelConfig {
        mode,
        threshold,
        neutral_band,
    })
}

fn resolve_sample_weight_config(lgbm_config: &JsonValue) -> SampleWeightConfig {
    SampleWeightConfig {
        mode: json_key_string(lgbm_config, "sample_weight_mode")
            .unwrap_or_else(|| "none".to_owned())
            .trim()
            .to_ascii_lowercase(),
        power: json_key_f64(lgbm_config, "sample_weight_power").unwrap_or(1.0),
        scale: json_key_f64(lgbm_config, "sample_weight_scale"),
        min: json_key_f64(lgbm_config, "sample_weight_min").unwrap_or(0.0),
        date_normalize: json_key_bool(lgbm_config, "sample_weight_date_normalize").unwrap_or(false),
    }
}

fn json_key_string(value: &JsonValue, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .map(str::to_owned)
}

fn json_key_f64(value: &JsonValue, key: &str) -> Option<f64> {
    value.get(key).and_then(|item| {
        item.as_f64()
            .or_else(|| item.as_i64().map(|inner| inner as f64))
            .or_else(|| item.as_u64().map(|inner| inner as f64))
    })
}

fn json_key_bool(value: &JsonValue, key: &str) -> Option<bool> {
    value.get(key).and_then(JsonValue::as_bool)
}

fn required_yaml_usize(value: &YamlValue, path: &[&str]) -> Result<usize, String> {
    yaml_path_usize(value, path).ok_or_else(|| {
        format!(
            "resolved config missing positive integer {}",
            path.join(".")
        )
    })
}

fn read_test_range_ns(cfg: &YamlValue) -> Result<(i64, i64), String> {
    let test = yaml_path(cfg, &["time", "test"])
        .and_then(YamlValue::as_sequence)
        .ok_or_else(|| "resolved config missing time.test sequence".to_owned())?;
    if test.len() != 2 {
        return Err("time.test must contain exactly [start, end]".to_owned());
    }
    let start = test[0]
        .as_str()
        .ok_or_else(|| "time.test[0] must be a date string".to_owned())?;
    let end = test[1]
        .as_str()
        .ok_or_else(|| "time.test[1] must be a date string".to_owned())?;
    Ok((parse_date_ns(start)?, parse_date_ns(end)?))
}

fn yaml_as_mapping(value: &YamlValue) -> Result<&YamlMapping, String> {
    value
        .as_mapping()
        .ok_or_else(|| "expected YAML mapping".to_owned())
}

fn yaml_mapping_get<'a>(mapping: &'a YamlMapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_owned()))
}

fn remove_yaml_mapping_key(value: &mut YamlValue, key: &str) {
    if let Some(mapping) = value.as_mapping_mut() {
        mapping.remove(YamlValue::String(key.to_owned()));
    }
}

fn yaml_string_usize_mapping(value: &YamlValue) -> Result<BTreeMap<String, usize>, String> {
    let mapping = value
        .as_mapping()
        .ok_or_else(|| "expected YAML mapping".to_owned())?;
    let mut out = BTreeMap::new();
    for (key, value) in mapping {
        let key = key
            .as_str()
            .ok_or_else(|| "expected string key in repeat_columns".to_owned())?;
        let count = value
            .as_i64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| format!("repeat_columns.{key} must be a positive integer"))?;
        if count == 0 {
            return Err(format!("repeat_columns.{key} must be a positive integer"));
        }
        out.insert(key.to_owned(), count);
    }
    Ok(out)
}

fn label_column_name(horizon: usize) -> String {
    format!("label_{horizon}d")
}

fn read_feature_list_json(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: JsonValue = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let values = match value {
        JsonValue::Array(values) => values,
        JsonValue::Object(mut object) => object
            .remove("selected_features")
            .or_else(|| object.remove("features"))
            .and_then(|item| item.as_array().cloned())
            .ok_or_else(|| {
                format!(
                    "{} must be a JSON list or object with selected_features/features",
                    path.display()
                )
            })?,
        _ => {
            return Err(format!(
                "{} must be a JSON list or object with selected_features/features",
                path.display()
            ))
        }
    };
    values
        .into_iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{} contains a non-string feature name", path.display()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_python_compatible_window_indices() {
        assert_eq!(
            compute_rolling_window_indices(100, 60, 10, 21),
            Some((9, 68, 69, 78))
        );
    }

    #[test]
    fn ranks_cross_sectional_values_by_percentile() {
        let mut rows = vec![
            FactorRow {
                date_ns: 0,
                symbol: "a".into(),
                label: 0.0,
                backtest_label: 0.0,
                features: vec![2.0],
            },
            FactorRow {
                date_ns: 0,
                symbol: "b".into(),
                label: 0.0,
                backtest_label: 0.0,
                features: vec![4.0],
            },
            FactorRow {
                date_ns: 0,
                symbol: "c".into(),
                label: 0.0,
                backtest_label: 0.0,
                features: vec![4.0],
            },
        ];
        apply_cross_sectional_rank_transform(
            &mut rows,
            &["f1".to_owned()],
            &["f1".to_owned()],
            &BTreeSet::new(),
        );
        let values = rows
            .into_iter()
            .map(|row| row.features[0])
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0 / 3.0, 2.5 / 3.0, 2.5 / 3.0]);
    }

    #[test]
    fn cross_sectional_rank_respects_exclude_columns() {
        let mut rows = vec![
            FactorRow {
                date_ns: 0,
                symbol: "a".into(),
                label: 0.0,
                backtest_label: 0.0,
                features: vec![2.0, 10.0],
            },
            FactorRow {
                date_ns: 0,
                symbol: "b".into(),
                label: 0.0,
                backtest_label: 0.0,
                features: vec![4.0, 20.0],
            },
        ];
        apply_cross_sectional_rank_transform(
            &mut rows,
            &["ranked".to_owned(), "raw".to_owned()],
            &["ranked".to_owned(), "raw".to_owned()],
            &BTreeSet::from(["raw".to_owned()]),
        );
        assert_eq!(rows[0].features, vec![0.5, 10.0]);
        assert_eq!(rows[1].features, vec![1.0, 20.0]);
    }

    #[test]
    fn transforms_training_labels_in_rust_window_format() {
        let rows = vec![
            FactorRow {
                date_ns: 0,
                symbol: "a".into(),
                label: 0.10,
                backtest_label: 0.0,
                features: vec![1.0],
            },
            FactorRow {
                date_ns: 0,
                symbol: "b".into(),
                label: 0.00,
                backtest_label: 0.0,
                features: vec![2.0],
            },
            FactorRow {
                date_ns: 0,
                symbol: "c".into(),
                label: -0.10,
                backtest_label: 0.0,
                features: vec![3.0],
            },
        ];
        let indices = vec![0, 1, 2];
        let effective = vec![true, true, true];
        let transformed = transform_window_labels(
            &[0.10, 0.00, -0.10],
            &indices,
            &rows,
            &effective,
            &[f64::NAN; 3],
            &TrainLabelTransformConfig {
                mode: "cross_section_rank".to_owned(),
                neutral_band: 0.0,
                tail_band: 0.03,
                scale_multiplier: 1.0,
                min_scale: 1e-4,
            },
            &OpportunityLabelConfig {
                mode: "positive".to_owned(),
                threshold: 0.0,
                neutral_band: 0.0,
            },
        )
        .unwrap();
        assert_eq!(transformed, vec![0.5, 0.0, -0.5]);
    }

    #[test]
    fn sample_weights_match_date_normalized_opportunity_distance_semantics() {
        let rows = vec![
            FactorRow {
                date_ns: 1,
                symbol: "a".into(),
                label: 0.03,
                backtest_label: 0.0,
                features: vec![1.0],
            },
            FactorRow {
                date_ns: 1,
                symbol: "b".into(),
                label: -0.01,
                backtest_label: 0.0,
                features: vec![1.0],
            },
            FactorRow {
                date_ns: 2,
                symbol: "c".into(),
                label: 0.02,
                backtest_label: 0.0,
                features: vec![1.0],
            },
            FactorRow {
                date_ns: 2,
                symbol: "d".into(),
                label: -0.04,
                backtest_label: 0.0,
                features: vec![1.0],
            },
        ];
        let weights = build_sample_weights(
            &[0.03, -0.01, 0.02, -0.04],
            &[0, 1, 2, 3],
            &rows,
            &[true, true, true, true],
            &[0.03, -0.01, 0.02, -0.04],
            &SampleWeightConfig {
                mode: "opportunity_distance".to_owned(),
                power: 1.0,
                scale: Some(0.01),
                min: 0.0,
                date_normalize: true,
            },
            &OpportunityLabelConfig {
                mode: "positive".to_owned(),
                threshold: 0.0,
                neutral_band: 0.005,
            },
        )
        .unwrap();
        assert!(weights.iter().all(|value| value.is_finite()));
        assert!(((weights[0] + weights[1]) / 2.0 - 1.0).abs() < 1e-12);
        assert!(((weights[2] + weights[3]) / 2.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn validation_topk_summary_matches_expected_daily_aggregation() {
        let rows = vec![
            FactorRow {
                date_ns: 1,
                symbol: "a".into(),
                label: 0.01,
                backtest_label: 0.0,
                features: vec![1.0],
            },
            FactorRow {
                date_ns: 1,
                symbol: "b".into(),
                label: 0.03,
                backtest_label: 0.0,
                features: vec![2.0],
            },
            FactorRow {
                date_ns: 2,
                symbol: "c".into(),
                label: -0.02,
                backtest_label: 0.0,
                features: vec![3.0],
            },
            FactorRow {
                date_ns: 2,
                symbol: "d".into(),
                label: 0.04,
                backtest_label: 0.0,
                features: vec![4.0],
            },
        ];
        let prepared = PreparedWindowLabelColumns {
            label: vec![0.01, 0.03, -0.02, 0.04],
            raw_label: vec![0.01, 0.03, -0.02, 0.04],
            sample_weight: vec![f64::NAN; 4],
            opportunity_label: vec![1.0, 1.0, 0.0, 1.0],
        };
        let summary = compute_validation_topk_summary(
            &rows,
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &prepared,
            &[0.2, 0.9, 0.1, 0.8],
            1,
        )
        .unwrap();
        let object = summary.as_object().unwrap();
        assert_eq!(object["valid_topk_days"], serde_json::json!(2));
        assert_eq!(object["valid_top1_label_mean"], serde_json::json!(0.035));
        assert_eq!(object["valid_top1_positive_rate"], serde_json::json!(1.0));
        assert_eq!(object["valid_topk_label_mean"], serde_json::json!(0.035));
        assert_eq!(object["valid_topk_label_median"], serde_json::json!(0.035));
        assert_eq!(
            object["valid_topk_min_label_mean"],
            serde_json::json!(0.035)
        );
        assert_eq!(object["valid_topk_positive_rate"], serde_json::json!(1.0));
        assert!((object["valid_topk_excess_mean"].as_f64().unwrap() - 0.02).abs() < 1e-12);
        assert_eq!(
            object["valid_top1_opportunity_rate"],
            serde_json::json!(1.0)
        );
        assert_eq!(
            object["valid_topk_opportunity_rate"],
            serde_json::json!(1.0)
        );
    }

    #[test]
    fn maps_known_duplicate_feature_sources() {
        assert_eq!(
            canonical_feature_source("TEMP_turnover_mean_20"),
            "LGBM_turnover_20"
        );
        assert_eq!(canonical_feature_source("TEMP_rsv_20"), "RSV20");
        assert_eq!(canonical_feature_source("KMID"), "KMID");
    }
}
