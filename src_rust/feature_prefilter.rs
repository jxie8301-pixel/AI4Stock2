use crate::gen_feature::discover_bucket_parquet_paths;
use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use chrono::{DateTime, NaiveDate, Utc};
use csv::{ReaderBuilder, WriterBuilder};
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    ProjectionMask,
};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

type Row = BTreeMap<String, String>;

const DEFAULT_MIN_COVERAGE_PCT: f64 = 0.95;
const DEFAULT_MIN_ABS_RANK_IC: f64 = 0.02;
const DEFAULT_MIN_ABS_RANK_IC_IR: f64 = 0.10;
const DEFAULT_MIN_MONTHLY_POSITIVE_RATE: f64 = 0.45;

#[derive(Debug, Clone)]
pub struct PrefilterThresholds {
    pub min_coverage_pct: f64,
    pub min_abs_rank_ic: f64,
    pub min_abs_rank_ic_ir: f64,
    pub min_monthly_positive_rate: f64,
    pub min_segment_directional_hit_mean: Option<f64>,
    pub max_segment_rank_ic_mean_range: Option<f64>,
    pub exclude_direction_flip: bool,
}

impl Default for PrefilterThresholds {
    fn default() -> Self {
        Self {
            min_coverage_pct: DEFAULT_MIN_COVERAGE_PCT,
            min_abs_rank_ic: DEFAULT_MIN_ABS_RANK_IC,
            min_abs_rank_ic_ir: DEFAULT_MIN_ABS_RANK_IC_IR,
            min_monthly_positive_rate: DEFAULT_MIN_MONTHLY_POSITIVE_RATE,
            min_segment_directional_hit_mean: None,
            max_segment_rank_ic_mean_range: None,
            exclude_direction_flip: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefilterOptions {
    pub diagnostics_summary: PathBuf,
    pub segment_comparison: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub thresholds: PrefilterThresholds,
}

#[derive(Debug, Clone)]
pub struct RobustPrefilterOptions {
    pub raw_summary: PathBuf,
    pub neutral_summary: PathBuf,
    pub raw_segment_comparison: Option<PathBuf>,
    pub neutral_segment_comparison: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub thresholds: PrefilterThresholds,
}

#[derive(Debug, Clone)]
pub struct CorrPruneOptions {
    pub factor_store: PathBuf,
    pub candidates_csv: PathBuf,
    pub output_dir: PathBuf,
    pub date_start: Option<String>,
    pub date_end: Option<String>,
    pub universe_name: String,
    pub universe_dir: PathBuf,
    pub corr_threshold: f64,
    pub use_cross_sectional_rank: bool,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub enum ProfileReadmeMode {
    Prefilter,
    Robust,
}

#[derive(Debug, Clone)]
pub struct ProfileArtifactOptions {
    pub output_dir: PathBuf,
    pub selected_csv: PathBuf,
    pub profile_name: String,
    pub max_features: Option<usize>,
    pub write_config_profile: bool,
    pub config_profile_path: Option<PathBuf>,
    pub factor_store_name: String,
    pub readme_mode: ProfileReadmeMode,
    pub settings: Vec<(String, String)>,
    pub safety_warning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PrefilterProfileBuildOptions {
    pub diagnostics_summary: PathBuf,
    pub segment_comparison: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub thresholds: PrefilterThresholds,
    pub factor_store: PathBuf,
    pub date_start: Option<String>,
    pub date_end: Option<String>,
    pub universe_name: String,
    pub universe_dir: PathBuf,
    pub corr_threshold: f64,
    pub use_cross_sectional_rank: bool,
    pub batch_size: usize,
    pub profile_name: String,
    pub max_features: Option<usize>,
    pub write_config_profile: bool,
    pub config_profile_path: Option<PathBuf>,
    pub factor_store_name: String,
    pub settings: Vec<(String, String)>,
    pub safety_warning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RobustProfileBuildOptions {
    pub raw_summary: PathBuf,
    pub neutral_summary: PathBuf,
    pub raw_segment_comparison: Option<PathBuf>,
    pub neutral_segment_comparison: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub thresholds: PrefilterThresholds,
    pub factor_store: PathBuf,
    pub date_start: Option<String>,
    pub date_end: Option<String>,
    pub universe_name: String,
    pub universe_dir: PathBuf,
    pub corr_threshold: f64,
    pub use_cross_sectional_rank: bool,
    pub batch_size: usize,
    pub profile_name: String,
    pub max_features: Option<usize>,
    pub write_config_profile: bool,
    pub config_profile_path: Option<PathBuf>,
    pub factor_store_name: String,
    pub settings: Vec<(String, String)>,
    pub safety_warning: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FeaturePrefilterSummary {
    pub output_dir: String,
    pub original_features: usize,
    pub after_prefilter: usize,
    pub after_exact_duplicate_prune: usize,
    pub dropped_prefilter: usize,
    pub dropped_exact_duplicates: usize,
    pub robust_summary_path: Option<String>,
    pub kept_path: String,
    pub dropped_path: String,
    pub exact_kept_path: String,
    pub exact_dropped_path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorrPruneSummary {
    pub output_dir: String,
    pub input_features: usize,
    pub kept_features: usize,
    pub dropped_features: usize,
    pub row_count: usize,
    pub corr_threshold: f64,
    pub use_cross_sectional_rank: bool,
    pub kept_path: String,
    pub dropped_path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileArtifactSummary {
    pub output_dir: String,
    pub profile_name: String,
    pub selected_feature_count: usize,
    pub profile_path: String,
    pub config_profile_path: Option<String>,
    pub readme_path: String,
    pub selected_features: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileBuildSummary {
    pub output_dir: String,
    pub profile_name: String,
    pub original_features: usize,
    pub after_prefilter: usize,
    pub after_exact_duplicate_prune: usize,
    pub after_corr_prune: usize,
    pub selected_feature_count: usize,
    pub prefilter_summary: FeaturePrefilterSummary,
    pub corr_prune_summary: CorrPruneSummary,
    pub profile_artifacts: ProfileArtifactSummary,
}

pub fn run_prefilter_summary(
    options: &PrefilterOptions,
) -> Result<FeaturePrefilterSummary, String> {
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create prefilter output dir {}: {err}",
            options.output_dir.display()
        )
    })?;
    let summary = load_diagnostics_summary(
        &options.diagnostics_summary,
        options.segment_comparison.as_deref(),
    )?;
    let original_features = summary.len();
    let (kept, dropped) = prefilter_feature_summary(summary, &options.thresholds);
    if kept.is_empty() {
        return Err("No factors survived the prefilter thresholds.".to_owned());
    }
    let after_prefilter = kept.len();
    let (kept_exact, dropped_exact) = prune_exact_duplicate_features(kept.clone());
    if kept_exact.is_empty() {
        return Err("No factors remained after exact-duplicate pruning.".to_owned());
    }

    let kept_path = options.output_dir.join("prefilter_kept.csv");
    let dropped_path = options.output_dir.join("prefilter_dropped.csv");
    let exact_kept_path = options.output_dir.join("exact_duplicate_kept.csv");
    let exact_dropped_path = options.output_dir.join("exact_duplicate_pruned.csv");
    write_rows_csv(&kept_path, &kept)?;
    write_rows_csv(&dropped_path, &dropped)?;
    write_rows_csv(&exact_kept_path, &kept_exact)?;
    write_rows_csv(&exact_dropped_path, &dropped_exact)?;

    Ok(FeaturePrefilterSummary {
        output_dir: options.output_dir.display().to_string(),
        original_features,
        after_prefilter,
        after_exact_duplicate_prune: kept_exact.len(),
        dropped_prefilter: dropped.len(),
        dropped_exact_duplicates: dropped_exact.len(),
        robust_summary_path: None,
        kept_path: kept_path.display().to_string(),
        dropped_path: dropped_path.display().to_string(),
        exact_kept_path: exact_kept_path.display().to_string(),
        exact_dropped_path: exact_dropped_path.display().to_string(),
    })
}

pub fn run_robust_prefilter_summary(
    options: &RobustPrefilterOptions,
) -> Result<FeaturePrefilterSummary, String> {
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create robust prefilter output dir {}: {err}",
            options.output_dir.display()
        )
    })?;
    let raw_summary = load_diagnostics_summary(
        &options.raw_summary,
        options.raw_segment_comparison.as_deref(),
    )?;
    let neutral_summary = load_diagnostics_summary(
        &options.neutral_summary,
        options.neutral_segment_comparison.as_deref(),
    )?;
    let robust_summary = build_robust_feature_summary(&raw_summary, &neutral_summary)?;
    let original_features = robust_summary.len();
    let (kept, dropped) = prefilter_feature_summary(robust_summary.clone(), &options.thresholds);
    if kept.is_empty() {
        return Err("No factors survived the robust prefilter thresholds.".to_owned());
    }
    let after_prefilter = kept.len();
    let (kept_exact, dropped_exact) = prune_exact_duplicate_features(kept.clone());
    if kept_exact.is_empty() {
        return Err("No factors remained after exact-duplicate pruning.".to_owned());
    }

    let robust_summary_path = options.output_dir.join("robust_summary.csv");
    let kept_path = options.output_dir.join("robust_prefilter_kept.csv");
    let dropped_path = options.output_dir.join("robust_prefilter_dropped.csv");
    let exact_kept_path = options.output_dir.join("exact_duplicate_kept.csv");
    let exact_dropped_path = options.output_dir.join("exact_duplicate_pruned.csv");
    write_rows_csv(&robust_summary_path, &robust_summary)?;
    write_rows_csv(&kept_path, &kept)?;
    write_rows_csv(&dropped_path, &dropped)?;
    write_rows_csv(&exact_kept_path, &kept_exact)?;
    write_rows_csv(&exact_dropped_path, &dropped_exact)?;

    Ok(FeaturePrefilterSummary {
        output_dir: options.output_dir.display().to_string(),
        original_features,
        after_prefilter,
        after_exact_duplicate_prune: kept_exact.len(),
        dropped_prefilter: dropped.len(),
        dropped_exact_duplicates: dropped_exact.len(),
        robust_summary_path: Some(robust_summary_path.display().to_string()),
        kept_path: kept_path.display().to_string(),
        dropped_path: dropped_path.display().to_string(),
        exact_kept_path: exact_kept_path.display().to_string(),
        exact_dropped_path: exact_dropped_path.display().to_string(),
    })
}

pub fn run_corr_prune(options: &CorrPruneOptions) -> Result<CorrPruneSummary, String> {
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create correlation-prune output dir {}: {err}",
            options.output_dir.display()
        )
    })?;
    let candidates = read_rows_csv(&options.candidates_csv)?;
    if candidates.is_empty() {
        return Err("No candidate rows available for correlation pruning.".to_owned());
    }
    let mut ordered = candidates.into_iter().map(score_row).collect::<Vec<_>>();
    sort_scored_rows(&mut ordered);
    let feature_names = ordered
        .iter()
        .map(|row| row.get("feature").cloned().unwrap_or_default())
        .filter(|feature| !feature.trim().is_empty())
        .collect::<Vec<_>>();
    if feature_names.is_empty() {
        return Err("Candidate CSV contains no usable feature values.".to_owned());
    }
    let mut factor_rows = load_factor_rows_for_corr(options, &feature_names)?;
    if factor_rows.is_empty() {
        return Err("Factor store returned no rows for correlation pruning.".to_owned());
    }
    if options.use_cross_sectional_rank {
        apply_cross_sectional_rank_rows(&mut factor_rows);
    }
    let (kept, dropped) = prune_correlated_rows(
        ordered,
        &factor_rows,
        &feature_names,
        options.corr_threshold,
    )?;
    if kept.is_empty() {
        return Err("No factors remained after correlation pruning.".to_owned());
    }
    let kept_path = options.output_dir.join("correlation_kept.csv");
    let dropped_path = options.output_dir.join("correlation_pruned.csv");
    write_rows_csv(&kept_path, &kept)?;
    write_rows_csv(&dropped_path, &dropped)?;
    Ok(CorrPruneSummary {
        output_dir: options.output_dir.display().to_string(),
        input_features: feature_names.len(),
        kept_features: kept.len(),
        dropped_features: dropped.len(),
        row_count: factor_rows.len(),
        corr_threshold: options.corr_threshold,
        use_cross_sectional_rank: options.use_cross_sectional_rank,
        kept_path: kept_path.display().to_string(),
        dropped_path: dropped_path.display().to_string(),
    })
}

pub fn write_profile_artifacts(
    options: &ProfileArtifactOptions,
) -> Result<ProfileArtifactSummary, String> {
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create profile output dir {}: {err}",
            options.output_dir.display()
        )
    })?;
    let selected_rows = read_rows_csv(&options.selected_csv)?;
    let mut selected_features = selected_rows
        .iter()
        .filter_map(|row| row.get("feature").cloned())
        .filter(|feature| !feature.trim().is_empty())
        .collect::<Vec<_>>();
    if let Some(max_features) = options.max_features {
        selected_features.truncate(max_features);
    }
    if selected_features.is_empty() {
        return Err("No selected features available for profile artifact writing.".to_owned());
    }

    let profile_path = options
        .output_dir
        .join(format!("{}.yaml", options.profile_name));
    write_profile_yaml(
        &profile_path,
        &selected_features,
        &options.factor_store_name,
    )?;
    let config_profile_path = if options.write_config_profile {
        let path = options.config_profile_path.clone().unwrap_or_else(|| {
            PathBuf::from("configs")
                .join("features")
                .join(format!("{}.yaml", options.profile_name))
        });
        write_profile_yaml(&path, &selected_features, &options.factor_store_name)?;
        Some(path)
    } else {
        None
    };

    let readme_path = options.output_dir.join("README.md");
    write_profile_readme(&readme_path, options, &selected_features)?;
    Ok(ProfileArtifactSummary {
        output_dir: options.output_dir.display().to_string(),
        profile_name: options.profile_name.clone(),
        selected_feature_count: selected_features.len(),
        profile_path: profile_path.display().to_string(),
        config_profile_path: config_profile_path
            .as_ref()
            .map(|path| path.display().to_string()),
        readme_path: readme_path.display().to_string(),
        selected_features,
    })
}

pub fn run_build_prefilter_profile(
    options: &PrefilterProfileBuildOptions,
) -> Result<ProfileBuildSummary, String> {
    let prefilter_summary = run_prefilter_summary(&PrefilterOptions {
        diagnostics_summary: options.diagnostics_summary.clone(),
        segment_comparison: options.segment_comparison.clone(),
        output_dir: options.output_dir.clone(),
        thresholds: options.thresholds.clone(),
    })?;
    let corr_prune_summary = run_corr_prune(&CorrPruneOptions {
        factor_store: options.factor_store.clone(),
        candidates_csv: PathBuf::from(&prefilter_summary.exact_kept_path),
        output_dir: options.output_dir.clone(),
        date_start: options.date_start.clone(),
        date_end: options.date_end.clone(),
        universe_name: options.universe_name.clone(),
        universe_dir: options.universe_dir.clone(),
        corr_threshold: options.corr_threshold,
        use_cross_sectional_rank: options.use_cross_sectional_rank,
        batch_size: options.batch_size,
    })?;
    let mut settings = options.settings.clone();
    append_profile_count_settings(&mut settings, &prefilter_summary);
    let profile_artifacts = write_profile_artifacts(&ProfileArtifactOptions {
        output_dir: options.output_dir.clone(),
        selected_csv: PathBuf::from(&corr_prune_summary.kept_path),
        profile_name: options.profile_name.clone(),
        max_features: options.max_features,
        write_config_profile: options.write_config_profile,
        config_profile_path: options.config_profile_path.clone(),
        factor_store_name: options.factor_store_name.clone(),
        readme_mode: ProfileReadmeMode::Prefilter,
        settings,
        safety_warning: options.safety_warning.clone(),
    })?;
    Ok(ProfileBuildSummary {
        output_dir: options.output_dir.display().to_string(),
        profile_name: options.profile_name.clone(),
        original_features: prefilter_summary.original_features,
        after_prefilter: prefilter_summary.after_prefilter,
        after_exact_duplicate_prune: prefilter_summary.after_exact_duplicate_prune,
        after_corr_prune: corr_prune_summary.kept_features,
        selected_feature_count: profile_artifacts.selected_feature_count,
        prefilter_summary,
        corr_prune_summary,
        profile_artifacts,
    })
}

pub fn run_build_robust_profile(
    options: &RobustProfileBuildOptions,
) -> Result<ProfileBuildSummary, String> {
    let prefilter_summary = run_robust_prefilter_summary(&RobustPrefilterOptions {
        raw_summary: options.raw_summary.clone(),
        neutral_summary: options.neutral_summary.clone(),
        raw_segment_comparison: options.raw_segment_comparison.clone(),
        neutral_segment_comparison: options.neutral_segment_comparison.clone(),
        output_dir: options.output_dir.clone(),
        thresholds: options.thresholds.clone(),
    })?;
    let corr_prune_summary = run_corr_prune(&CorrPruneOptions {
        factor_store: options.factor_store.clone(),
        candidates_csv: PathBuf::from(&prefilter_summary.exact_kept_path),
        output_dir: options.output_dir.clone(),
        date_start: options.date_start.clone(),
        date_end: options.date_end.clone(),
        universe_name: options.universe_name.clone(),
        universe_dir: options.universe_dir.clone(),
        corr_threshold: options.corr_threshold,
        use_cross_sectional_rank: options.use_cross_sectional_rank,
        batch_size: options.batch_size,
    })?;
    let mut settings = options.settings.clone();
    append_profile_count_settings(&mut settings, &prefilter_summary);
    let profile_artifacts = write_profile_artifacts(&ProfileArtifactOptions {
        output_dir: options.output_dir.clone(),
        selected_csv: PathBuf::from(&corr_prune_summary.kept_path),
        profile_name: options.profile_name.clone(),
        max_features: options.max_features,
        write_config_profile: options.write_config_profile,
        config_profile_path: options.config_profile_path.clone(),
        factor_store_name: options.factor_store_name.clone(),
        readme_mode: ProfileReadmeMode::Robust,
        settings,
        safety_warning: options.safety_warning.clone(),
    })?;
    Ok(ProfileBuildSummary {
        output_dir: options.output_dir.display().to_string(),
        profile_name: options.profile_name.clone(),
        original_features: prefilter_summary.original_features,
        after_prefilter: prefilter_summary.after_prefilter,
        after_exact_duplicate_prune: prefilter_summary.after_exact_duplicate_prune,
        after_corr_prune: corr_prune_summary.kept_features,
        selected_feature_count: profile_artifacts.selected_feature_count,
        prefilter_summary,
        corr_prune_summary,
        profile_artifacts,
    })
}

fn append_profile_count_settings(
    settings: &mut Vec<(String, String)>,
    prefilter_summary: &FeaturePrefilterSummary,
) {
    settings.extend([
        (
            "original_features".to_owned(),
            prefilter_summary.original_features.to_string(),
        ),
        (
            "after_prefilter".to_owned(),
            prefilter_summary.after_prefilter.to_string(),
        ),
        (
            "after_exact_duplicate_prune".to_owned(),
            prefilter_summary.after_exact_duplicate_prune.to_string(),
        ),
    ]);
}

fn load_diagnostics_summary(path: &Path, segment_path: Option<&Path>) -> Result<Vec<Row>, String> {
    let mut summary = read_rows_csv(path)?;
    if summary.iter().any(|row| !row.contains_key("feature")) {
        return Err("Diagnostics summary must contain a 'feature' column.".to_owned());
    }
    if let Some(segment_path) = segment_path {
        let segment_rows = read_rows_csv(segment_path)?;
        if segment_rows.iter().any(|row| !row.contains_key("feature")) {
            return Err("Segment comparison must contain a 'feature' column.".to_owned());
        }
        let by_feature = segment_rows
            .into_iter()
            .filter_map(|row| row.get("feature").cloned().map(|feature| (feature, row)))
            .collect::<HashMap<_, _>>();
        for row in &mut summary {
            if let Some(feature) = row.get("feature") {
                if let Some(segment) = by_feature.get(feature) {
                    for (key, value) in segment {
                        if key != "feature" {
                            row.insert(key.clone(), value.clone());
                        }
                    }
                }
            }
        }
    }
    Ok(summary)
}

fn read_rows_csv(path: &Path) -> Result<Vec<Row>, String> {
    let mut reader = ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|err| format!("failed to open CSV {}: {err}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|err| format!("failed to read CSV headers {}: {err}", path.display()))?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record =
            record.map_err(|err| format!("failed to read CSV {}: {err}", path.display()))?;
        let mut row = Row::new();
        for (index, header) in headers.iter().enumerate() {
            row.insert(header.clone(), record.get(index).unwrap_or("").to_owned());
        }
        rows.push(row);
    }
    Ok(rows)
}

fn write_rows_csv(path: &Path, rows: &[Row]) -> Result<(), String> {
    let mut headers = Vec::new();
    let mut seen = BTreeSet::new();
    for preferred in [
        "feature",
        "feature_group",
        "direction_consistent",
        "direction_flip",
        "suggested_direction",
        "coverage_pct",
        "rank_ic_mean",
        "rank_ic_abs_mean",
        "rank_ic_ir",
        "rank_ic_ir_abs",
        "monthly_rank_ic_directional_hit_rate",
        "monotonicity_mean",
        "segment_monthly_directional_hit_mean",
        "segment_rank_ic_mean_range",
        "segment_rank_ic_abs_max",
        "dropped_by",
        "duplicate_group",
    ] {
        if rows.iter().any(|row| row.contains_key(preferred)) && seen.insert(preferred.to_owned()) {
            headers.push(preferred.to_owned());
        }
    }
    for row in rows {
        for key in row.keys() {
            if seen.insert(key.clone()) {
                headers.push(key.clone());
            }
        }
    }
    if headers.is_empty() {
        headers.extend(
            [
                "feature",
                "dropped_by",
                "abs_corr",
                "coverage_pct",
                "rank_ic_mean",
                "rank_ic_abs_mean",
                "rank_ic_ir",
                "rank_ic_ir_abs",
                "monthly_rank_ic_directional_hit_rate",
                "monotonicity_mean",
            ]
            .iter()
            .map(|value| (*value).to_owned()),
        );
    }
    let mut writer = WriterBuilder::new()
        .from_path(path)
        .map_err(|err| format!("failed to create CSV {}: {err}", path.display()))?;
    writer
        .write_record(&headers)
        .map_err(|err| format!("failed to write CSV headers {}: {err}", path.display()))?;
    for row in rows {
        let record = headers
            .iter()
            .map(|key| row.get(key).map(String::as_str).unwrap_or(""))
            .collect::<Vec<_>>();
        writer
            .write_record(record)
            .map_err(|err| format!("failed to write CSV row {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush CSV {}: {err}", path.display()))
}

fn build_robust_feature_summary(
    raw_summary: &[Row],
    neutral_summary: &[Row],
) -> Result<Vec<Row>, String> {
    if raw_summary.iter().any(|row| !row.contains_key("feature"))
        || neutral_summary
            .iter()
            .any(|row| !row.contains_key("feature"))
    {
        return Err("Both summaries must contain a 'feature' column.".to_owned());
    }
    let neutral_by_feature = neutral_summary
        .iter()
        .filter_map(|row| row.get("feature").map(|feature| (feature.clone(), row)))
        .collect::<HashMap<_, _>>();
    let mut out = Vec::new();
    for raw in raw_summary {
        let Some(feature) = raw.get("feature") else {
            continue;
        };
        let Some(neutral) = neutral_by_feature.get(feature) else {
            continue;
        };
        let mut row = Row::new();
        row.insert("feature".to_owned(), feature.clone());
        for (key, value) in raw {
            if key != "feature" {
                row.insert(format!("{key}_raw"), value.clone());
            }
        }
        for (key, value) in *neutral {
            if key != "feature" {
                row.insert(format!("{key}_neutral"), value.clone());
            }
        }
        let raw_rank_ic = numeric(raw, "rank_ic_mean");
        let neutral_rank_ic = numeric(neutral, "rank_ic_mean");
        let raw_rank_ic_ir = numeric(raw, "rank_ic_ir");
        let neutral_rank_ic_ir = numeric(neutral, "rank_ic_ir");
        let raw_direction = sign(raw_rank_ic);
        let neutral_direction = sign(neutral_rank_ic);
        let direction_consistent = raw_direction == neutral_direction && raw_direction != 0.0;
        let robust_direction = if direction_consistent {
            raw_direction
        } else {
            0.0
        };

        set_string(
            &mut row,
            "feature_group",
            raw.get("feature_group")
                .or_else(|| neutral.get("feature_group"))
                .cloned()
                .unwrap_or_default(),
        );
        set_bool(&mut row, "direction_consistent", direction_consistent);
        set_bool(
            &mut row,
            "direction_flip",
            bool_value(raw, "direction_flip")
                || bool_value(neutral, "direction_flip")
                || !direction_consistent,
        );
        set_float(&mut row, "raw_direction", raw_direction);
        set_float(&mut row, "neutral_direction", neutral_direction);
        set_float(&mut row, "suggested_direction", robust_direction);

        for column in [
            "observation_count",
            "valid_observation_count",
            "coverage_pct",
            "avg_daily_coverage_pct",
            "date_count",
            "effective_date_count",
            "monotonic_date_count",
            "rank_ic_positive_rate",
            "rank_ic_directional_hit_rate",
            "monthly_rank_ic_positive_rate",
            "monthly_rank_ic_directional_hit_rate",
            "monthly_rank_ic_months",
            "monotonicity_positive_rate",
            "top_bottom_spread_positive_rate",
            "segment_monthly_directional_hit_mean",
            "segment_monthly_directional_hit_min",
            "segment_rank_ic_abs_max",
            "segment_rank_ic_abs_min",
        ] {
            set_float(
                &mut row,
                column,
                fmin_ignore_nan(numeric(raw, column), numeric(neutral, column)),
            );
        }
        for column in ["ic_std", "rank_ic_std", "segment_rank_ic_mean_range"] {
            set_float(
                &mut row,
                column,
                fmax_ignore_nan(numeric(raw, column), numeric(neutral, column)),
            );
        }
        for column in [
            "ic_mean",
            "ic_ir",
            "rank_ic_mean",
            "rank_ic_ir",
            "monthly_rank_ic_mean",
            "monotonicity_mean",
            "top_bottom_spread_mean",
        ] {
            set_float(
                &mut row,
                column,
                robust_direction
                    * fmin_ignore_nan(numeric(raw, column).abs(), numeric(neutral, column).abs()),
            );
        }
        let rank_ic_mean = numeric(&row, "rank_ic_mean");
        let rank_ic_ir = numeric(&row, "rank_ic_ir");
        let ic_ir = numeric(&row, "ic_ir");
        set_float(&mut row, "rank_ic_abs_mean", rank_ic_mean.abs());
        set_float(&mut row, "rank_ic_ir_abs", rank_ic_ir.abs());
        set_float(&mut row, "ic_ir_abs", ic_ir.abs());
        set_float(
            &mut row,
            "neutral_retention_rank_ic_abs",
            ratio_or_nan(neutral_rank_ic.abs(), raw_rank_ic.abs()),
        );
        set_float(
            &mut row,
            "neutral_retention_rank_ic_ir_abs",
            ratio_or_nan(neutral_rank_ic_ir.abs(), raw_rank_ic_ir.abs()),
        );
        set_float(
            &mut row,
            "robust_rank_ic_abs_ratio",
            ratio_or_nan(
                fmin_ignore_nan(raw_rank_ic.abs(), neutral_rank_ic.abs()),
                fmax_ignore_nan(raw_rank_ic.abs(), neutral_rank_ic.abs()),
            ),
        );
        set_float(
            &mut row,
            "robust_rank_ic_ir_abs_ratio",
            ratio_or_nan(
                fmin_ignore_nan(raw_rank_ic_ir.abs(), neutral_rank_ic_ir.abs()),
                fmax_ignore_nan(raw_rank_ic_ir.abs(), neutral_rank_ic_ir.abs()),
            ),
        );
        out.push(row);
    }
    if out.is_empty() {
        return Err("Raw and neutral summaries have no overlapping features.".to_owned());
    }
    Ok(out)
}

fn prefilter_feature_summary(
    summary: Vec<Row>,
    thresholds: &PrefilterThresholds,
) -> (Vec<Row>, Vec<Row>) {
    let mut scored = summary.into_iter().map(score_row).collect::<Vec<_>>();
    sort_scored_rows(&mut scored);
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for row in scored {
        if passes_prefilter(&row, thresholds) {
            kept.push(row);
        } else {
            dropped.push(row);
        }
    }
    (kept, dropped)
}

fn score_row(mut row: Row) -> Row {
    let rank_ic_mean = numeric(&row, "rank_ic_mean");
    let rank_ic_ir = numeric(&row, "rank_ic_ir");
    let monotonicity_mean = numeric(&row, "monotonicity_mean");
    let monthly_directional = if row.contains_key("monthly_rank_ic_directional_hit_rate") {
        numeric(&row, "monthly_rank_ic_directional_hit_rate")
    } else {
        let positive_rate = numeric(&row, "monthly_rank_ic_positive_rate");
        if positive_rate.is_nan() {
            f64::NAN
        } else {
            positive_rate.max(1.0 - positive_rate)
        }
    };
    set_float(&mut row, "rank_ic_abs_mean", rank_ic_mean.abs());
    set_float(&mut row, "rank_ic_ir_abs", rank_ic_ir.abs());
    set_float(&mut row, "monotonicity_abs_mean", monotonicity_mean.abs());
    let prefix_priority =
        feature_prefix_priority(row.get("feature").map(String::as_str).unwrap_or("")) as f64;
    let direction_flip_sort = if bool_value(&row, "direction_flip") {
        1.0
    } else {
        0.0
    };
    set_float(&mut row, "prefix_priority", prefix_priority);
    set_float(&mut row, "direction_flip_sort", direction_flip_sort);
    if !row.contains_key("segment_monthly_directional_hit_mean") {
        set_float(&mut row, "segment_monthly_directional_hit_mean", f64::NAN);
    }
    if !row.contains_key("segment_rank_ic_abs_max") {
        set_float(&mut row, "segment_rank_ic_abs_max", f64::NAN);
    }
    if !row.contains_key("segment_rank_ic_mean_range") {
        set_float(&mut row, "segment_rank_ic_mean_range", f64::NAN);
    }
    set_float(
        &mut row,
        "monthly_rank_ic_directional_hit_rate",
        monthly_directional,
    );
    row
}

fn passes_prefilter(row: &Row, thresholds: &PrefilterThresholds) -> bool {
    let coverage = numeric(row, "coverage_pct");
    let rank_ic_abs = numeric(row, "rank_ic_abs_mean");
    let rank_ic_ir_abs = numeric(row, "rank_ic_ir_abs");
    let monthly_hit = numeric(row, "monthly_rank_ic_directional_hit_rate");
    if coverage < thresholds.min_coverage_pct {
        return false;
    }
    if rank_ic_abs < thresholds.min_abs_rank_ic && rank_ic_ir_abs < thresholds.min_abs_rank_ic_ir {
        return false;
    }
    if monthly_hit < thresholds.min_monthly_positive_rate {
        return false;
    }
    if let Some(min_segment_hit) = thresholds.min_segment_directional_hit_mean {
        if numeric(row, "segment_monthly_directional_hit_mean") < min_segment_hit {
            return false;
        }
    }
    if let Some(max_segment_range) = thresholds.max_segment_rank_ic_mean_range {
        if numeric(row, "segment_rank_ic_mean_range") > max_segment_range {
            return false;
        }
    }
    if thresholds.exclude_direction_flip && bool_value(row, "direction_flip") {
        return false;
    }
    true
}

fn prune_exact_duplicate_features(candidates: Vec<Row>) -> (Vec<Row>, Vec<Row>) {
    if candidates.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut ordered = candidates.into_iter().map(score_row).collect::<Vec<_>>();
    sort_scored_rows(&mut ordered);
    let present = ordered
        .iter()
        .filter_map(|row| row.get("feature").cloned())
        .collect::<BTreeSet<_>>();
    let groups = known_exact_duplicate_feature_groups()
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .filter(|feature| present.contains(feature))
                .collect::<Vec<_>>()
        })
        .filter(|group| group.len() >= 2)
        .collect::<Vec<_>>();
    if groups.is_empty() {
        return (ordered, Vec::new());
    }
    let mut feature_to_group = HashMap::new();
    for group in &groups {
        let key = group.join("|");
        for feature in group {
            feature_to_group.insert(feature.clone(), key.clone());
        }
    }
    let mut kept_by_group: HashMap<String, String> = HashMap::new();
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for mut row in ordered {
        let feature = row.get("feature").cloned().unwrap_or_default();
        let Some(group_key) = feature_to_group.get(&feature) else {
            kept.push(row);
            continue;
        };
        if let Some(keeper) = kept_by_group.get(group_key) {
            row.insert("dropped_by".to_owned(), keeper.clone());
            row.insert("duplicate_group".to_owned(), group_key.clone());
            dropped.push(row);
        } else {
            kept_by_group.insert(group_key.clone(), feature);
            kept.push(row);
        }
    }
    (kept, dropped)
}

#[derive(Debug, Clone)]
struct FactorCorrRow {
    date_ns: i64,
    symbol: String,
    values: Vec<f64>,
}

#[derive(Debug, Clone)]
struct UniverseFilter {
    intervals_by_symbol: BTreeMap<String, Vec<(Option<i64>, Option<i64>)>>,
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

fn load_factor_rows_for_corr(
    options: &CorrPruneOptions,
    feature_names: &[String],
) -> Result<Vec<FactorCorrRow>, String> {
    let (_, paths) = discover_bucket_parquet_paths(&options.factor_store)?;
    if paths.is_empty() {
        return Err(format!(
            "no bucket shard parquet files found under {}",
            options.factor_store.display()
        ));
    }
    let start_ns = options
        .date_start
        .as_deref()
        .map(parse_date_ns)
        .transpose()?;
    let end_ns = options.date_end.as_deref().map(parse_date_ns).transpose()?;
    let universe_filter = load_universe_filter(&options.universe_name, &options.universe_dir)?;
    let mut columns = vec!["date".to_owned(), "symbol".to_owned()];
    columns.extend(feature_names.iter().cloned());
    columns.sort();
    columns.dedup();
    let mut rows = Vec::new();
    for path in paths {
        let reader = open_projected_parquet_reader(&path, &columns, options.batch_size)?;
        for batch in reader {
            let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            append_factor_corr_batch(
                &mut rows,
                &batch,
                &path,
                feature_names,
                start_ns,
                end_ns,
                universe_filter.as_ref(),
            )?;
        }
    }
    rows.sort_by(|left, right| {
        left.date_ns
            .cmp(&right.date_ns)
            .then(left.symbol.cmp(&right.symbol))
    });
    Ok(rows)
}

fn append_factor_corr_batch(
    rows: &mut Vec<FactorCorrRow>,
    batch: &RecordBatch,
    path: &Path,
    feature_names: &[String],
    start_ns: Option<i64>,
    end_ns: Option<i64>,
    universe_filter: Option<&UniverseFilter>,
) -> Result<(), String> {
    let date_array = required_column(batch, "date", path)?;
    let symbol_array = required_column(batch, "symbol", path)?;
    let feature_arrays = feature_names
        .iter()
        .map(|feature| required_column(batch, feature, path))
        .collect::<Result<Vec<_>, _>>()?;
    for row_index in 0..batch.num_rows() {
        let date_ns = date_value_ns(date_array, row_index, path)?;
        if start_ns.is_some_and(|start| date_ns < start) || end_ns.is_some_and(|end| date_ns > end)
        {
            continue;
        }
        let symbol = string_value(symbol_array, row_index, path)?;
        if universe_filter.is_some_and(|filter| !filter.allows(&symbol, date_ns)) {
            continue;
        }
        let values = feature_arrays
            .iter()
            .map(|array| numeric_value(*array, row_index, path))
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(FactorCorrRow {
            date_ns,
            symbol,
            values,
        });
    }
    Ok(())
}

fn apply_cross_sectional_rank_rows(rows: &mut [FactorCorrRow]) {
    if rows.is_empty() {
        return;
    }
    let feature_count = rows[0].values.len();
    let mut start = 0usize;
    while start < rows.len() {
        let date_ns = rows[start].date_ns;
        let mut end = start + 1;
        while end < rows.len() && rows[end].date_ns == date_ns {
            end += 1;
        }
        for feature_index in 0..feature_count {
            apply_percentile_rank_for_slice(rows, start, end, feature_index);
        }
        start = end;
    }
}

fn apply_percentile_rank_for_slice(
    rows: &mut [FactorCorrRow],
    start: usize,
    end: usize,
    feature_index: usize,
) {
    let mut order = (start..end)
        .filter(|row_index| rows[*row_index].values[feature_index].is_finite())
        .collect::<Vec<_>>();
    if order.is_empty() {
        return;
    }
    order.sort_by(|left, right| {
        rows[*left].values[feature_index].total_cmp(&rows[*right].values[feature_index])
    });
    let count = order.len() as f64;
    let mut rank_start = 0usize;
    while rank_start < order.len() {
        let mut rank_end = rank_start + 1;
        while rank_end < order.len()
            && rows[order[rank_end]].values[feature_index]
                == rows[order[rank_start]].values[feature_index]
        {
            rank_end += 1;
        }
        let average_rank = 0.5 * (rank_start as f64 + rank_end as f64 - 1.0) + 1.0;
        let pct_rank = average_rank / count;
        for row_index in &order[rank_start..rank_end] {
            rows[*row_index].values[feature_index] = pct_rank;
        }
        rank_start = rank_end;
    }
}

fn prune_correlated_rows(
    candidates: Vec<Row>,
    factor_rows: &[FactorCorrRow],
    feature_names: &[String],
    corr_threshold: f64,
) -> Result<(Vec<Row>, Vec<Row>), String> {
    let index_by_feature = feature_names
        .iter()
        .enumerate()
        .map(|(index, feature)| (feature.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    let mut kept_features = Vec::<String>::new();
    let threshold = corr_threshold.abs();
    for mut row in candidates {
        let feature = row.get("feature").cloned().unwrap_or_default();
        let Some(&feature_index) = index_by_feature.get(&feature) else {
            continue;
        };
        if kept_features.is_empty() {
            kept.push(row);
            kept_features.push(feature);
            continue;
        }
        let mut best_match = String::new();
        let mut best_corr = f64::NAN;
        for kept_feature in &kept_features {
            let Some(&kept_index) = index_by_feature.get(kept_feature) else {
                continue;
            };
            let corr = abs_pearson_corr(factor_rows, feature_index, kept_index);
            if corr.is_finite() && (!best_corr.is_finite() || corr > best_corr) {
                best_corr = corr;
                best_match = kept_feature.clone();
            }
        }
        if !best_corr.is_finite() {
            kept.push(row);
            kept_features.push(feature);
            continue;
        }
        if best_corr >= threshold {
            row.insert("dropped_by".to_owned(), best_match);
            set_float(&mut row, "abs_corr", best_corr);
            dropped.push(row);
        } else {
            kept.push(row);
            kept_features.push(feature);
        }
    }
    Ok((kept, dropped))
}

fn abs_pearson_corr(rows: &[FactorCorrRow], left_index: usize, right_index: usize) -> f64 {
    let mut count = 0usize;
    let mut sum_left = 0.0;
    let mut sum_right = 0.0;
    let mut sum_left_right = 0.0;
    let mut sum_left_sq = 0.0;
    let mut sum_right_sq = 0.0;
    for row in rows {
        let left = row.values[left_index];
        let right = row.values[right_index];
        if !left.is_finite() || !right.is_finite() {
            continue;
        }
        count += 1;
        sum_left += left;
        sum_right += right;
        sum_left_right += left * right;
        sum_left_sq += left * left;
        sum_right_sq += right * right;
    }
    if count < 2 {
        return f64::NAN;
    }
    let count_f = count as f64;
    let numerator = sum_left_right - (sum_left * sum_right / count_f);
    let left_var = sum_left_sq - (sum_left * sum_left / count_f);
    let right_var = sum_right_sq - (sum_right * sum_right / count_f);
    if !left_var.is_finite() || !right_var.is_finite() || left_var <= 0.0 || right_var <= 0.0 {
        return f64::NAN;
    }
    (numerator / (left_var * right_var).sqrt()).abs()
}

fn sort_scored_rows(rows: &mut [Row]) {
    rows.sort_by(|left, right| compare_scored_rows(left, right));
}

fn compare_scored_rows(left: &Row, right: &Row) -> Ordering {
    for (column, descending) in [
        ("direction_flip_sort", false),
        ("segment_monthly_directional_hit_mean", true),
        ("segment_rank_ic_mean_range", false),
        ("segment_rank_ic_abs_max", true),
        ("rank_ic_ir_abs", true),
        ("rank_ic_abs_mean", true),
        ("monotonicity_abs_mean", true),
        ("monthly_rank_ic_positive_rate", true),
        ("coverage_pct", true),
        ("prefix_priority", false),
    ] {
        let ordering =
            compare_float_nan_last(numeric(left, column), numeric(right, column), descending);
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    left.get("feature")
        .map(String::as_str)
        .unwrap_or("")
        .cmp(right.get("feature").map(String::as_str).unwrap_or(""))
}

fn compare_float_nan_last(left: f64, right: f64, descending: bool) -> Ordering {
    let left_nan = left.is_nan();
    let right_nan = right.is_nan();
    match (left_nan, right_nan) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            let ordering = left.partial_cmp(&right).unwrap_or(Ordering::Equal);
            if descending {
                ordering.reverse()
            } else {
                ordering
            }
        }
    }
}

fn feature_prefix_priority(feature_name: &str) -> i32 {
    if feature_name.starts_with("TS_") {
        0
    } else if feature_name.starts_with("LGBM_") {
        1
    } else if feature_name.starts_with("TECH_") {
        2
    } else if feature_name.starts_with("TEMP_") {
        4
    } else {
        3
    }
}

fn known_exact_duplicate_feature_groups() -> Vec<Vec<String>> {
    let mut groups = Vec::new();
    for window in [5, 10, 20, 30, 60] {
        groups.push(vec![format!("RSV{window}"), format!("TEMP_rsv_{window}")]);
        groups.push(vec![
            format!("CORR{window}"),
            format!("TEMP_corr_cv_{window}"),
        ]);
    }
    for window in [20, 60] {
        groups.push(vec![
            format!("LGBM_ret_{window}"),
            format!("TEMP_ret_{window}"),
        ]);
    }
    for window in [20, 60, 120] {
        groups.push(vec![
            format!("LGBM_dist_ma{window}"),
            format!("TEMP_ma_gap_{window}"),
        ]);
    }
    groups.extend([
        vec!["LGBM_std_60".to_owned(), "TEMP_std_60".to_owned()],
        vec!["LGBM_amihud_20".to_owned(), "TEMP_amihud_20".to_owned()],
        vec![
            "LGBM_turnover_20".to_owned(),
            "TEMP_turnover_mean_20".to_owned(),
        ],
        vec![
            "LGBM_dist_high_20".to_owned(),
            "TEMP_high_gap_20".to_owned(),
        ],
        vec!["LGBM_dist_low_20".to_owned(), "TEMP_low_gap_20".to_owned()],
    ]);
    groups
}

fn numeric(row: &Row, column: &str) -> f64 {
    row.get(column)
        .map(String::as_str)
        .unwrap_or("")
        .trim()
        .parse::<f64>()
        .unwrap_or(f64::NAN)
}

fn bool_value(row: &Row, column: &str) -> bool {
    matches!(
        row.get(column)
            .map(|value| value.trim().to_ascii_lowercase())
            .unwrap_or_default()
            .as_str(),
        "true" | "1" | "yes"
    )
}

fn set_float(row: &mut Row, column: &str, value: f64) {
    row.insert(
        column.to_owned(),
        if value.is_nan() {
            String::new()
        } else {
            format!("{value:.15}")
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned()
        },
    );
}

fn set_bool(row: &mut Row, column: &str, value: bool) {
    row.insert(
        column.to_owned(),
        if value { "True" } else { "False" }.to_owned(),
    );
}

fn set_string(row: &mut Row, column: &str, value: String) {
    row.insert(column.to_owned(), value);
}

fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn fmin_ignore_nan(left: f64, right: f64) -> f64 {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => f64::NAN,
        (true, false) => right,
        (false, true) => left,
        (false, false) => left.min(right),
    }
}

fn fmax_ignore_nan(left: f64, right: f64) -> f64 {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => f64::NAN,
        (true, false) => right,
        (false, true) => left,
        (false, false) => left.max(right),
    }
}

fn ratio_or_nan(numerator: f64, denominator: f64) -> f64 {
    if denominator.is_nan() || denominator <= 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

fn write_profile_yaml(
    path: &Path,
    selected_features: &[String],
    factor_store_name: &str,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create profile dir {}: {err}", parent.display()))?;
    }
    let payload = serde_yaml::Mapping::from_iter([
        (
            serde_yaml::Value::String("alpha".to_owned()),
            serde_yaml::Value::String("all_factors".to_owned()),
        ),
        (
            serde_yaml::Value::String("generation_space".to_owned()),
            serde_yaml::Value::String("full_factor_space".to_owned()),
        ),
        (
            serde_yaml::Value::String("factor_store_name".to_owned()),
            serde_yaml::Value::String(factor_store_name.to_owned()),
        ),
        (
            serde_yaml::Value::String("selected_columns".to_owned()),
            serde_yaml::Value::Sequence(
                selected_features
                    .iter()
                    .cloned()
                    .map(serde_yaml::Value::String)
                    .collect(),
            ),
        ),
    ]);
    let rendered = serde_yaml::to_string(&payload)
        .map_err(|err| format!("failed to render profile YAML {}: {err}", path.display()))?;
    fs::write(path, rendered)
        .map_err(|err| format!("failed to write profile YAML {}: {err}", path.display()))
}

fn write_profile_readme(
    path: &Path,
    options: &ProfileArtifactOptions,
    selected_features: &[String],
) -> Result<(), String> {
    let mut lines = Vec::new();
    lines.push(format!("# {}", options.profile_name));
    lines.push(String::new());
    if matches!(options.readme_mode, ProfileReadmeMode::Robust) {
        lines.extend([
            "## Robust Summary Logic".to_owned(),
            String::new(),
            "- Robust `rank_ic_mean` / `rank_ic_ir` keep the shared direction and take the smaller absolute value across raw and industry-neutral passes.".to_owned(),
            "- Coverage and directional-hit metrics use the lower of the two passes.".to_owned(),
            "- Segment drift uses the worse of the two passes.".to_owned(),
            "- `direction_flip=true` also covers raw/neutral sign disagreement.".to_owned(),
            String::new(),
        ]);
    }
    lines.push("## Filter Settings".to_owned());
    lines.push(String::new());
    for (key, value) in &options.settings {
        lines.push(format!("- {key}: `{value}`"));
    }
    lines.push(String::new());
    lines.push("## Counts".to_owned());
    lines.push(String::new());
    lines.push(format!("- after_corr_prune: `{}`", selected_features.len()));
    lines.push(String::new());
    lines.push("## Selected Features".to_owned());
    lines.push(String::new());
    lines.extend(selected_features.iter().map(|name| format!("- `{name}`")));
    if let Some(warning) = &options.safety_warning {
        if !warning.trim().is_empty() {
            lines.extend([
                String::new(),
                "## Safety Warning".to_owned(),
                String::new(),
                format!("- {warning}"),
            ]);
        }
    }
    fs::write(path, format!("{}\n", lines.join("\n").trim()))
        .map_err(|err| format!("failed to write README {}: {err}", path.display()))
}

fn load_universe_filter(
    universe_name: &str,
    universe_dir: &Path,
) -> Result<Option<UniverseFilter>, String> {
    if universe_name == "all" || universe_name.trim().is_empty() {
        return Ok(None);
    }
    let path = resolve_universe_path(universe_name, universe_dir)?;
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
    let mut reader = ReaderBuilder::new()
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
            .transpose()?
            .flatten();
        let end_ns = record
            .get(2)
            .map(parse_optional_date_ns)
            .transpose()?
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
        "universe file not found for '{universe_name}' under {}",
        universe_dir.display()
    ))
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
        return Ok(String::new());
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

fn parse_date_ns(raw: &str) -> Result<i64, String> {
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .and_then(|datetime| datetime.and_utc().timestamp_nanos_opt())
            .ok_or_else(|| format!("invalid date timestamp: {raw}"));
    }
    if let Ok(date) = NaiveDate::parse_from_str(raw, "%Y%m%d") {
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

fn normalize_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[(&str, &str)]) -> Row {
        values
            .iter()
            .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn prefilter_applies_thresholds_and_sorting() {
        let rows = vec![
            row(&[
                ("feature", "good_a"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "0.03"),
                ("rank_ic_ir", "0.08"),
                ("monthly_rank_ic_positive_rate", "0.60"),
                ("monotonicity_mean", "0.10"),
            ]),
            row(&[
                ("feature", "good_b"),
                ("coverage_pct", "0.97"),
                ("rank_ic_mean", "0.01"),
                ("rank_ic_ir", "0.20"),
                ("monthly_rank_ic_positive_rate", "0.55"),
                ("monotonicity_mean", "0.03"),
            ]),
            row(&[
                ("feature", "bad_cov"),
                ("coverage_pct", "0.70"),
                ("rank_ic_mean", "0.10"),
                ("rank_ic_ir", "0.50"),
                ("monthly_rank_ic_positive_rate", "0.90"),
                ("monotonicity_mean", "0.20"),
            ]),
        ];
        let (kept, dropped) = prefilter_feature_summary(
            rows,
            &PrefilterThresholds {
                min_coverage_pct: 0.95,
                min_abs_rank_ic: 0.02,
                min_abs_rank_ic_ir: 0.10,
                min_monthly_positive_rate: 0.45,
                ..Default::default()
            },
        );
        assert_eq!(features(&kept), vec!["good_b", "good_a"]);
        assert_eq!(features(&dropped), vec!["bad_cov"]);
    }

    #[test]
    fn exact_duplicate_prune_keeps_priority_representative() {
        let rows = vec![
            row(&[
                ("feature", "CORR20"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "-0.05"),
                ("rank_ic_ir", "-0.30"),
                ("monthly_rank_ic_positive_rate", "0.30"),
                ("monthly_rank_ic_directional_hit_rate", "0.70"),
                ("monotonicity_mean", "-0.15"),
            ]),
            row(&[
                ("feature", "TEMP_corr_cv_20"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "-0.05"),
                ("rank_ic_ir", "-0.30"),
                ("monthly_rank_ic_positive_rate", "0.30"),
                ("monthly_rank_ic_directional_hit_rate", "0.70"),
                ("monotonicity_mean", "-0.15"),
            ]),
        ];
        let (kept, dropped) = prune_exact_duplicate_features(rows);
        assert_eq!(features(&kept), vec!["CORR20"]);
        assert_eq!(features(&dropped), vec!["TEMP_corr_cv_20"]);
        assert_eq!(dropped[0].get("dropped_by").unwrap(), "CORR20");
    }

    #[test]
    fn robust_summary_is_conservative_and_direction_aware() {
        let raw = vec![
            row(&[
                ("feature", "stable_pos"),
                ("feature_group", "g"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "0.06"),
                ("rank_ic_ir", "0.30"),
                ("monthly_rank_ic_positive_rate", "0.70"),
                ("monthly_rank_ic_directional_hit_rate", "0.70"),
                ("monotonicity_mean", "0.20"),
                ("segment_monthly_directional_hit_mean", "0.65"),
                ("segment_rank_ic_mean_range", "0.08"),
            ]),
            row(&[
                ("feature", "flip_sign"),
                ("feature_group", "g"),
                ("coverage_pct", "0.98"),
                ("rank_ic_mean", "0.05"),
                ("rank_ic_ir", "0.25"),
                ("monthly_rank_ic_positive_rate", "0.60"),
                ("monthly_rank_ic_directional_hit_rate", "0.60"),
                ("monotonicity_mean", "0.10"),
                ("segment_monthly_directional_hit_mean", "0.60"),
                ("segment_rank_ic_mean_range", "0.06"),
            ]),
        ];
        let neutral = vec![
            row(&[
                ("feature", "stable_pos"),
                ("feature_group", "g"),
                ("coverage_pct", "0.95"),
                ("rank_ic_mean", "0.03"),
                ("rank_ic_ir", "0.18"),
                ("monthly_rank_ic_positive_rate", "0.62"),
                ("monthly_rank_ic_directional_hit_rate", "0.62"),
                ("monotonicity_mean", "0.09"),
                ("segment_monthly_directional_hit_mean", "0.58"),
                ("segment_rank_ic_mean_range", "0.11"),
            ]),
            row(&[
                ("feature", "flip_sign"),
                ("feature_group", "g"),
                ("coverage_pct", "0.96"),
                ("rank_ic_mean", "-0.02"),
                ("rank_ic_ir", "-0.10"),
                ("monthly_rank_ic_positive_rate", "0.40"),
                ("monthly_rank_ic_directional_hit_rate", "0.60"),
                ("monotonicity_mean", "-0.03"),
                ("segment_monthly_directional_hit_mean", "0.55"),
                ("segment_rank_ic_mean_range", "0.07"),
            ]),
        ];
        let robust = build_robust_feature_summary(&raw, &neutral).unwrap();
        let by_feature = robust
            .iter()
            .map(|row| (row.get("feature").unwrap().clone(), row))
            .collect::<HashMap<_, _>>();
        let stable = by_feature.get("stable_pos").unwrap();
        assert_eq!(stable.get("coverage_pct").unwrap(), "0.95");
        assert_eq!(stable.get("rank_ic_mean").unwrap(), "0.03");
        assert_eq!(stable.get("rank_ic_ir").unwrap(), "0.18");
        assert_eq!(
            stable.get("monthly_rank_ic_directional_hit_rate").unwrap(),
            "0.62"
        );
        assert_eq!(stable.get("segment_rank_ic_mean_range").unwrap(), "0.11");
        assert_eq!(stable.get("direction_consistent").unwrap(), "True");
        let retention = stable
            .get("neutral_retention_rank_ic_abs")
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!((retention - 0.5).abs() < 1e-12);

        let flip = by_feature.get("flip_sign").unwrap();
        assert_eq!(flip.get("rank_ic_mean").unwrap(), "0");
        assert_eq!(flip.get("rank_ic_ir").unwrap(), "0");
        assert_eq!(flip.get("direction_flip").unwrap(), "True");
        assert_eq!(flip.get("direction_consistent").unwrap(), "False");
    }

    #[test]
    fn corr_prune_drops_highly_correlated_features() {
        let candidates = vec![
            row(&[
                ("feature", "LGBM_bp"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "0.06"),
                ("rank_ic_ir", "0.30"),
                ("monthly_rank_ic_positive_rate", "0.60"),
                ("monotonicity_mean", "0.10"),
            ]),
            row(&[
                ("feature", "TECH_other"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "0.03"),
                ("rank_ic_ir", "0.12"),
                ("monthly_rank_ic_positive_rate", "0.55"),
                ("monotonicity_mean", "0.02"),
            ]),
            row(&[
                ("feature", "LGBM_clone"),
                ("coverage_pct", "0.99"),
                ("rank_ic_mean", "0.02"),
                ("rank_ic_ir", "0.11"),
                ("monthly_rank_ic_positive_rate", "0.55"),
                ("monotonicity_mean", "0.02"),
            ]),
        ];
        let mut rows = vec![
            FactorCorrRow {
                date_ns: 1,
                symbol: "A".to_owned(),
                values: vec![1.0, 2.0, 1.0],
            },
            FactorCorrRow {
                date_ns: 1,
                symbol: "B".to_owned(),
                values: vec![2.0, 1.0, 2.0],
            },
            FactorCorrRow {
                date_ns: 2,
                symbol: "A".to_owned(),
                values: vec![1.2, 1.0, 1.2],
            },
            FactorCorrRow {
                date_ns: 2,
                symbol: "B".to_owned(),
                values: vec![2.2, 2.0, 2.2],
            },
        ];
        apply_cross_sectional_rank_rows(&mut rows);
        let feature_names = vec![
            "LGBM_bp".to_owned(),
            "TECH_other".to_owned(),
            "LGBM_clone".to_owned(),
        ];
        let (kept, dropped) =
            prune_correlated_rows(candidates, &rows, &feature_names, 0.99).unwrap();
        assert!(features(&kept).contains(&"LGBM_bp"));
        assert_eq!(features(&dropped), vec!["LGBM_clone"]);
        assert_eq!(dropped[0].get("dropped_by").unwrap(), "LGBM_bp");
    }

    fn features(rows: &[Row]) -> Vec<&str> {
        rows.iter()
            .map(|row| row.get("feature").map(String::as_str).unwrap_or(""))
            .collect()
    }
}
