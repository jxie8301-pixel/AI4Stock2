use super::LgbmBundleOptions;
use ai4stock2_native::gen_feature::discover_bucket_parquet_paths;
use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, NaiveDate, Utc};
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    ArrowWriter, ProjectionMask,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const PREDICTION_ARTIFACT_DIRNAME: &str = "prediction_artifacts";
const PREDICTION_METADATA_FILENAME: &str = "metadata.json";
const PREDICTIONS_FILENAME: &str = "final_predictions.parquet";
const SIGNAL_LABELS_FILENAME: &str = "signal_labels.parquet";
const BACKTEST_LABELS_FILENAME: &str = "backtest_labels.parquet";
const TRAINING_SUMMARY_FILENAME: &str = "training_summary.csv";
const DEFAULT_LABEL_ABS_CAP: f64 = 0.35;

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
    intervals_by_symbol: BTreeMap<String, Vec<(Option<i64>, Option<i64>)>>,
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
    let resolved = resolve_runtime_config(options)?;
    if resolved.selected_feature_names.len() != resolved.selected_feature_sources.len() {
        return Err("selected feature names/sources length mismatch".to_owned());
    }
    let mut data = load_factor_data(options, &resolved)?;
    if data.selected_feature_names.is_empty() {
        return Err("no selected features available for Rust runtime LightGBM training".to_owned());
    }
    if resolved.cross_sectional_rank {
        apply_cross_sectional_rank_transform(
            &mut data.rows,
            &data.selected_feature_names,
            &data.selected_feature_sources,
            &resolved.cross_sectional_rank_exclude_columns,
        );
    }
    let row_indices_by_date = build_row_indices_by_date(&data.rows);
    let windows = build_rolling_windows(&resolved, &data.full_calendar, &data.test_calendar);
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

    let mut prediction_rows = Vec::new();
    let mut training_records = Vec::<BTreeMap<String, String>>::new();
    for window in &windows {
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
        let paths = build_window_paths(
            &window_dir,
            &models_dir,
            &importance_dir,
            &history_dir,
            window,
        );
        write_window_frame_parquet(
            &paths.train_path,
            &data.rows,
            &train_indices,
            &data.selected_feature_names,
        )?;
        write_window_frame_parquet(
            &paths.valid_path,
            &data.rows,
            &valid_indices,
            &data.selected_feature_names,
        )?;
        write_window_frame_parquet(
            &paths.test_path,
            &data.rows,
            &test_indices,
            &data.selected_feature_names,
        )?;
        let summary = call_python_train_window(
            options,
            &resolved,
            &data.selected_feature_names,
            window,
            &paths,
        )?;
        training_records.push(json_object_to_string_map(&summary));
        let mut window_predictions = read_long_value_parquet(&paths.prediction_path, "prediction")?;
        prediction_rows.append(&mut window_predictions);
    }
    prediction_rows.sort_by(|left, right| {
        left.date_ns
            .cmp(&right.date_ns)
            .then(left.instrument.cmp(&right.instrument))
    });
    let prediction_keys = prediction_rows
        .iter()
        .map(|row| (row.date_ns, row.instrument.clone()))
        .collect::<BTreeSet<_>>();

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
    write_training_summary(
        &artifact_dir.join(TRAINING_SUMMARY_FILENAME),
        &training_records,
    )?;
    write_training_summary(
        &results_dir.join(TRAINING_SUMMARY_FILENAME),
        &training_records,
    )?;
    write_config_snapshot(&results_dir.join("config_snapshot.yaml"), &resolved.cfg)?;
    write_metadata(
        &artifact_dir.join(PREDICTION_METADATA_FILENAME),
        &resolved,
        &data,
    )?;

    let finite_predictions = prediction_rows
        .iter()
        .filter(|row| row.value.is_finite())
        .count();
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
        "training_summary_path": results_dir.join(TRAINING_SUMMARY_FILENAME).display().to_string(),
        "prediction_training_summary_path": artifact_dir.join(TRAINING_SUMMARY_FILENAME).display().to_string(),
        "config_snapshot_path": results_dir.join("config_snapshot.yaml").display().to_string(),
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
        let (key, value) = parse_set_override(override_arg)?;
        set_yaml_dotted(&mut cfg, &key, value)?;
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
            &path,
            &columns,
            resolved,
            &selected_sources,
            universe_filter.as_ref(),
            load_start_ns,
            options.batch_size,
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
    let mut intervals_by_symbol: BTreeMap<String, Vec<(Option<i64>, Option<i64>)>> =
        BTreeMap::new();
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

fn append_factor_rows_from_bucket(
    rows: &mut Vec<FactorRow>,
    path: &Path,
    columns: &[String],
    resolved: &ResolvedRuntimeConfig,
    selected_sources: &[String],
    universe_filter: Option<&UniverseFilter>,
    load_start_ns: Option<i64>,
    batch_size: usize,
) -> Result<(), String> {
    let reader = open_projected_parquet_reader(path, columns, batch_size)?;
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        append_factor_batch(
            rows,
            &batch,
            path,
            resolved,
            selected_sources,
            universe_filter,
            load_start_ns,
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
    window: &RollingWindow,
    paths: &WindowDataPaths,
) -> Result<JsonValue, String> {
    let repo_root = env::current_dir().map_err(|error| format!("failed to read cwd: {error}"))?;
    let conda_prefix = env::var("CONDA_PREFIX").ok();
    let site_packages_dir = conda_prefix
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.join(".pixi/envs/default"))
        .join("lib/python3.12/site-packages");
    let feature_names = selected_feature_names.to_vec();
    let lgbm_config_json = serde_json::to_string(&resolved.lgbm_config)
        .map_err(|err| format!("failed to encode lgbm config: {err}"))?;
    let training_config_json = serde_json::to_string(&resolved.cfg)
        .map_err(|err| format!("failed to encode training config: {err}"))?;
    let window_metadata_json = serde_json::to_string(&serde_json::json!({
        "window_start": format_date_ns(window.test_start_ns),
        "window_end": format_date_ns(window.test_end_ns),
        "train_start": format_date_ns(window.train_start_ns),
        "train_end": format_date_ns(window.train_end_ns),
        "valid_start": format_date_ns(window.valid_start_ns),
        "valid_end": format_date_ns(window.valid_end_ns),
        "signal_horizon": resolved.signal_horizon,
        "label_embargo_days": resolved.label_embargo_days,
    }))
    .map_err(|err| format!("failed to encode window metadata: {err}"))?;
    let summary_json = Python::attach(|python| -> PyResult<String> {
        let sys_module = python.import("sys")?;
        sys_module
            .getattr("path")?
            .call_method1("insert", (0, repo_root.as_os_str()))?;
        sys_module
            .getattr("path")?
            .call_method1("insert", (0, site_packages_dir.as_os_str()))?;
        let bridge_module = python.import("src.rust_lgbm_bridge")?;
        let kwargs = PyDict::new(python);
        kwargs.set_item("train_path", paths.train_path.to_string_lossy().as_ref())?;
        kwargs.set_item("valid_path", paths.valid_path.to_string_lossy().as_ref())?;
        kwargs.set_item("test_path", paths.test_path.to_string_lossy().as_ref())?;
        kwargs.set_item(
            "prediction_path",
            paths.prediction_path.to_string_lossy().as_ref(),
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
        kwargs.set_item("window_metadata_json", &window_metadata_json)?;
        kwargs.set_item("training_config_json", &training_config_json)?;
        kwargs.set_item("save_model", options.save_models)?;
        kwargs.set_item("load_model", options.load_models)?;
        kwargs.set_item(
            "feature_names",
            PyList::new(python, feature_names.iter().map(String::as_str))?,
        )?;
        let result = bridge_module
            .getattr("train_lgbm_window_from_prepared_parquet")?
            .call((), Some(&kwargs))?;
        let json_module = python.import("json")?;
        json_module
            .call_method1("dumps", (result,))?
            .extract::<String>()
    })
    .map_err(|error| format!("Python LightGBM training failed: {error}"))?;
    serde_json::from_str(&summary_json)
        .map_err(|error| format!("Python training returned invalid JSON: {error}: {summary_json}"))
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

fn write_window_frame_parquet(
    path: &Path,
    rows: &[FactorRow],
    indices: &[usize],
    feature_names: &[String],
) -> Result<(), String> {
    let mut fields = vec![
        Field::new(
            "datetime",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("instrument", DataType::Utf8, false),
        Field::new("label", DataType::Float64, true),
        Field::new("backtest_label", DataType::Float64, true),
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
        Arc::new(Float64Array::from(
            indices
                .iter()
                .map(|idx| rows[*idx].label)
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            indices
                .iter()
                .map(|idx| rows[*idx].backtest_label)
                .collect::<Vec<_>>(),
        )),
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

fn read_long_value_parquet(path: &Path, value_column: &str) -> Result<Vec<LongValueRow>, String> {
    let reader = open_projected_parquet_reader(
        path,
        &[
            "datetime".to_owned(),
            "instrument".to_owned(),
            value_column.to_owned(),
        ],
        65_536,
    )?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let date_array = required_column(&batch, "datetime", path)?;
        let instrument_array = required_column(&batch, "instrument", path)?;
        let value_array = required_column(&batch, value_column, path)?;
        for row_index in 0..batch.num_rows() {
            rows.push(LongValueRow {
                date_ns: date_value_ns(date_array, row_index, path)?,
                instrument: string_value(instrument_array, row_index, path)?,
                value: numeric_value(value_array, row_index, path)?,
            });
        }
    }
    Ok(rows)
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

fn open_projected_parquet_reader(
    path: &Path,
    column_names: &[String],
    batch_size: usize,
) -> Result<ParquetRecordBatchReader, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to open parquet {}: {err}", path.display()))?;
    let schema = builder.schema();
    let mut indices = BTreeSet::new();
    let mut missing = Vec::new();
    for column_name in column_names {
        match schema.index_of(column_name) {
            Ok(index) => {
                indices.insert(index);
            }
            Err(_) => missing.push(column_name.clone()),
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "{} is missing required column(s): {}",
            path.display(),
            missing.join(",")
        ));
    }
    let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
    builder
        .with_projection(projection)
        .with_batch_size(batch_size.max(1))
        .build()
        .map_err(|err| {
            format!(
                "failed to build parquet reader for {}: {err}",
                path.display()
            )
        })
}

fn required_column<'a>(
    batch: &'a RecordBatch,
    name: &str,
    path: &Path,
) -> Result<&'a dyn Array, String> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("{} is missing required column {name}", path.display()))?;
    Ok(batch.column(index).as_ref())
}

fn numeric_value(array: &dyn Array, row_index: usize, path: &Path) -> Result<f64, String> {
    if array.is_null(row_index) {
        return Ok(f64::NAN);
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row_index));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row_index) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(values.value(row_index) as f64);
    }
    Err(format!(
        "{} has unsupported numeric type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn string_value(array: &dyn Array, row_index: usize, path: &Path) -> Result<String, String> {
    if array.is_null(row_index) {
        return Err(format!(
            "{} has null string value at row {row_index}",
            path.display()
        ));
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row_index).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row_index).to_owned());
    }
    Err(format!(
        "{} has unsupported string type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn date_value_ns(array: &dyn Array, row_index: usize, path: &Path) -> Result<i64, String> {
    if array.is_null(row_index) {
        return Err(format!(
            "{} has null date value at row {row_index}",
            path.display()
        ));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(values.value(row_index));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(values.value(row_index) * 1_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(values.value(row_index) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return Ok(values.value(row_index) * 1_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date64Array>() {
        return Ok(values.value(row_index) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(values.value(row_index) as i64 * 86_400_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return parse_date_ns(values.value(row_index));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return parse_date_ns(values.value(row_index));
    }
    Err(format!(
        "{} has unsupported date type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn parse_date_ns(raw: &str) -> Result<i64, String> {
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .and_then(|datetime| datetime.and_utc().timestamp_nanos_opt())
            .ok_or_else(|| format!("invalid date timestamp: {raw}"));
    }
    DateTime::parse_from_rfc3339(raw)
        .ok()
        .and_then(|datetime| datetime.with_timezone(&Utc).timestamp_nanos_opt())
        .ok_or_else(|| format!("invalid date: {raw}"))
}

fn format_date_ns(value: i64) -> String {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .map(|datetime| datetime.date_naive().to_string())
        .unwrap_or_default()
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
    resolve_profile_from_mapping(
        profile_config_path,
        profiles,
        profile_name,
        profile_kind,
        &mut Vec::new(),
    )
}

fn resolve_profile_from_mapping(
    profile_config_path: &str,
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
        let mut parent = resolve_profile_from_mapping(
            profile_config_path,
            profiles,
            &parent_name,
            profile_kind,
            stack,
        )?;
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

fn read_yaml_file(path: impl AsRef<Path>) -> Result<YamlValue, String> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_yaml::from_str(&raw).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn deep_merge_yaml(base: &mut YamlValue, overlay: YamlValue) {
    match (base, overlay) {
        (YamlValue::Mapping(base_map), YamlValue::Mapping(overlay_map)) => {
            for (key, overlay_value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(base_value) => deep_merge_yaml(base_value, overlay_value),
                    None => {
                        base_map.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => {
            *base_slot = overlay_value;
        }
    }
}

fn set_yaml_dotted(cfg: &mut YamlValue, dotted_key: &str, value: YamlValue) -> Result<(), String> {
    let parts = dotted_key
        .split('.')
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(format!("invalid dotted override key: {dotted_key}"));
    }
    let mut cursor = cfg;
    for part in &parts[..parts.len() - 1] {
        if !cursor.is_mapping() {
            *cursor = YamlValue::Mapping(YamlMapping::new());
        }
        let mapping = cursor.as_mapping_mut().expect("mapping just initialized");
        cursor = mapping
            .entry(YamlValue::String((*part).to_owned()))
            .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    }
    if !cursor.is_mapping() {
        *cursor = YamlValue::Mapping(YamlMapping::new());
    }
    cursor
        .as_mapping_mut()
        .expect("mapping just initialized")
        .insert(YamlValue::String(parts[parts.len() - 1].to_owned()), value);
    Ok(())
}

fn parse_set_override(raw: &str) -> Result<(String, YamlValue), String> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| format!("override must be in key=value form, got: {raw}"))?;
    let key = key.trim().to_owned();
    if key.is_empty() {
        return Err(format!("override key must be non-empty, got: {raw}"));
    }
    let value = serde_yaml::from_str::<YamlValue>(value.trim())
        .unwrap_or_else(|_| YamlValue::String(value.trim().to_owned()));
    Ok((key, value))
}

fn yaml_path<'a>(value: &'a YamlValue, path: &[&str]) -> Option<&'a YamlValue> {
    let mut cursor = value;
    for key in path {
        let mapping = cursor.as_mapping()?;
        cursor = mapping.get(YamlValue::String((*key).to_owned()))?;
    }
    Some(cursor)
}

fn yaml_path_string(value: &YamlValue, path: &[&str]) -> Option<String> {
    yaml_path(value, path)
        .and_then(YamlValue::as_str)
        .map(str::to_owned)
}

fn yaml_path_usize(value: &YamlValue, path: &[&str]) -> Option<usize> {
    yaml_path(value, path)
        .and_then(YamlValue::as_i64)
        .and_then(|value| usize::try_from(value).ok())
}

fn yaml_path_bool(value: &YamlValue, path: &[&str]) -> Option<bool> {
    yaml_path(value, path).and_then(YamlValue::as_bool)
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

fn yaml_string_sequence(value: &YamlValue) -> Result<Vec<String>, String> {
    let sequence = value
        .as_sequence()
        .ok_or_else(|| "expected YAML string sequence".to_owned())?;
    sequence
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "expected YAML string sequence".to_owned())
        })
        .collect()
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

fn yaml_usize(value: usize) -> YamlValue {
    YamlValue::Number(serde_yaml::Number::from(value as u64))
}

fn label_column_name(horizon: usize) -> String {
    format!("label_{horizon}d")
}

fn default_factor_store_dir(data_source: &str, factor_store_name: &str) -> PathBuf {
    if data_source == "akshare" {
        PathBuf::from("data/factor_store").join(factor_store_name)
    } else {
        PathBuf::from("data/factor_store").join(format!("{data_source}_{factor_store_name}"))
    }
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
    fn maps_known_duplicate_feature_sources() {
        assert_eq!(
            canonical_feature_source("TEMP_turnover_mean_20"),
            "LGBM_turnover_20"
        );
        assert_eq!(canonical_feature_source("TEMP_rsv_20"), "RSV20");
        assert_eq!(canonical_feature_source("KMID"), "KMID");
    }
}
