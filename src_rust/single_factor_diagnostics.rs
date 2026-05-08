use crate::common::benchmark::cross_section_mean_returns;
use crate::common::parquet::{
    date_value_ns, numeric_value, open_projected_parquet_reader,
    parse_datetime_ns as parse_date_ns, required_column,
};
use crate::gen_feature::discover_bucket_parquet_paths;
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_LABEL_ABS_CAP: f64 = 0.35;
type DateInterval = (Option<i64>, Option<i64>);
type IntervalsBySymbol = BTreeMap<String, Vec<DateInterval>>;

#[derive(Debug, Clone)]
pub struct SingleFactorOptions {
    pub factor_store: PathBuf,
    pub output_dir: PathBuf,
    pub feature_names: Vec<String>,
    pub label_column: String,
    pub signal_horizon: usize,
    pub date_start: Option<String>,
    pub date_end: Option<String>,
    pub universe_name: String,
    pub universe_dir: PathBuf,
    pub quantile_bins: usize,
    pub top_n: usize,
    pub include_details: bool,
    pub segments: Vec<SegmentSpec>,
    pub diagnostic_label_space: DiagnosticLabelSpace,
    pub diagnostic_threshold: f64,
    pub industry_neutral: bool,
    pub industry_map_path: Option<PathBuf>,
    pub feature_chunk_size: usize,
    pub batch_size: usize,
    pub metadata_json_path: Option<PathBuf>,
    pub config_snapshot_path: Option<PathBuf>,
    pub benchmark: BenchmarkOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosticLabelSpace {
    RawReturn,
    IndustryExcess,
    BenchmarkExcess,
}

impl DiagnosticLabelSpace {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "raw_return" | "raw" => Ok(Self::RawReturn),
            "industry_excess" => Ok(Self::IndustryExcess),
            "benchmark_excess" => Ok(Self::BenchmarkExcess),
            other => Err(format!(
                "unsupported Rust single-factor diagnostic label space: {other}; supported: raw_return, industry_excess, benchmark_excess"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RawReturn => "raw_return",
            Self::IndustryExcess => "industry_excess",
            Self::BenchmarkExcess => "benchmark_excess",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkMode {
    CrossSectionMean,
    File,
}

impl BenchmarkMode {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "cross_section_mean" => Ok(Self::CrossSectionMean),
            "file" => Ok(Self::File),
            other => Err(format!(
                "unsupported benchmark mode: {other}; supported: cross_section_mean, file"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CrossSectionMean => "cross_section_mean",
            Self::File => "file",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkValueType {
    Close,
    Return,
}

impl BenchmarkValueType {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "close" => Ok(Self::Close),
            "return" => Ok(Self::Return),
            other => Err(format!(
                "unsupported benchmark value_type: {other}; supported: close, return"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Close => "close",
            Self::Return => "return",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub mode: BenchmarkMode,
    pub path: Option<PathBuf>,
    pub date_column: String,
    pub value_column: String,
    pub value_type: BenchmarkValueType,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self {
            mode: BenchmarkMode::CrossSectionMean,
            path: None,
            date_column: "date".to_owned(),
            value_column: "close".to_owned(),
            value_type: BenchmarkValueType::Close,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentSpec {
    pub name: String,
    pub start_ns: i64,
    pub end_ns: i64,
}

impl SegmentSpec {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let parts = raw.split(':').map(str::trim).collect::<Vec<_>>();
        if parts.len() != 3 || parts[0].is_empty() {
            return Err(format!("segment must use name:start:end format; got {raw}"));
        }
        let start_ns = parse_date_ns(&parts[1].split_whitespace().collect::<String>())?;
        let end_ns = parse_date_ns(&parts[2].split_whitespace().collect::<String>())?;
        if start_ns > end_ns {
            return Err(format!("segment '{}' start must be <= end", parts[0]));
        }
        Ok(Self {
            name: parts[0].to_owned(),
            start_ns,
            end_ns,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SingleFactorRunSummary {
    pub output_dir: String,
    pub factor_store: String,
    pub feature_count: usize,
    pub row_count: usize,
    pub quantile_bins: usize,
    pub diagnostic_label_space: String,
    pub diagnostic_threshold: f64,
    pub industry_neutral: bool,
    pub segment_count: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone)]
struct DiagnosticRow {
    date_ns: i64,
    symbol: String,
    label: f64,
    industry: Option<String>,
    features: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
struct DiagnosticContext {
    rows: Vec<DiagnosticRow>,
    date_slices: Vec<DateSlice>,
}

#[derive(Debug, Clone, Copy)]
struct DateSlice {
    date_ns: i64,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Default)]
struct SummaryRow {
    feature: String,
    feature_group: String,
    observation_count: usize,
    valid_observation_count: usize,
    coverage_pct: f64,
    avg_daily_coverage_pct: f64,
    date_count: usize,
    effective_date_count: usize,
    monotonic_date_count: usize,
    ic_mean: f64,
    ic_std: f64,
    ic_ir: f64,
    ic_positive_rate: f64,
    rank_ic_mean: f64,
    rank_ic_std: f64,
    rank_ic_ir: f64,
    rank_ic_positive_rate: f64,
    rank_ic_directional_hit_rate: f64,
    rank_ic_abs_mean: f64,
    monthly_rank_ic_mean: f64,
    monthly_rank_ic_positive_rate: f64,
    monthly_rank_ic_directional_hit_rate: f64,
    monthly_rank_ic_months: usize,
    monotonicity_mean: f64,
    monotonicity_positive_rate: f64,
    top_bottom_spread_mean: f64,
    top_bottom_spread_positive_rate: f64,
    suggested_direction: i32,
    rank_ic_ir_abs: f64,
    ic_ir_abs: f64,
}

#[derive(Debug, Clone)]
struct BucketReturnRow {
    feature: String,
    feature_group: String,
    date_ns: i64,
    bucket: usize,
    bucket_mean_label: f64,
    bucket_count: usize,
}

#[derive(Debug, Clone)]
struct SpreadRow {
    feature: String,
    feature_group: String,
    date_ns: i64,
    bottom_bucket: usize,
    top_bucket: usize,
    bottom_mean_label: f64,
    top_mean_label: f64,
    top_bottom_spread: f64,
    valid_count: usize,
}

#[derive(Debug, Clone)]
struct DailyRankIcRow {
    feature: String,
    feature_group: String,
    date_ns: i64,
    ic: f64,
    rank_ic: f64,
    valid_count: usize,
}

#[derive(Debug, Clone)]
struct MonthlyRankIcRow {
    feature: String,
    feature_group: String,
    month_ns: i64,
    ic_mean: f64,
    rank_ic_mean: f64,
    rank_ic_count: usize,
    avg_valid_count: f64,
}

#[derive(Debug, Clone)]
struct MissingByYearRow {
    feature: String,
    feature_group: String,
    year: i32,
    observation_count: usize,
    feature_valid_count: usize,
    paired_valid_count: usize,
    feature_coverage_pct: f64,
    paired_coverage_pct: f64,
    feature_missing_pct: f64,
}

#[derive(Debug, Clone, Default)]
struct DetailRows {
    bucket_rows: Vec<BucketReturnRow>,
    spread_rows: Vec<SpreadRow>,
    monthly_rank_ic_rows: Vec<MonthlyRankIcRow>,
    missing_rows: Vec<MissingByYearRow>,
}

#[derive(Debug, Clone)]
struct SegmentComparisonRow {
    feature: String,
    values: BTreeMap<String, String>,
    sort_rank_ic_abs_max: f64,
    sort_monthly_hit_mean: f64,
}

#[derive(Debug, Clone)]
struct FactorStoreMeta {
    feature_names: HashSet<String>,
    available_label_columns: HashSet<String>,
    default_label_column: Option<String>,
}

#[derive(Debug, Clone)]
struct UniverseFilter {
    intervals_by_symbol: IntervalsBySymbol,
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

pub fn run_single_factor_diagnostics(
    options: &SingleFactorOptions,
) -> Result<SingleFactorRunSummary, String> {
    let started = Instant::now();
    if options.feature_names.is_empty() {
        return Err("at least one feature is required".to_owned());
    }
    let quantile_bins = options.quantile_bins.max(2);
    let top_n = options.top_n.max(1);
    let chunk_size = options.feature_chunk_size.max(1);
    let batch_size = options.batch_size.max(1);
    let signal_horizon = options.signal_horizon.max(1);
    let start_ns = options
        .date_start
        .as_deref()
        .map(parse_date_ns)
        .transpose()?;
    let end_ns = options.date_end.as_deref().map(parse_date_ns).transpose()?;
    if let (Some(start), Some(end)) = (start_ns, end_ns) {
        if start > end {
            return Err("date_start must be <= date_end".to_owned());
        }
    }

    let meta = read_factor_store_meta(&options.factor_store)?;
    validate_features(&meta, &options.feature_names)?;
    let label_column = resolve_label_column(&meta, &options.label_column)?;
    let (_, bucket_paths) = discover_bucket_parquet_paths(&options.factor_store)?;
    if bucket_paths.is_empty() {
        return Err(format!(
            "no bucket parquet shards found under {}",
            options.factor_store.display()
        ));
    }
    let universe_filter = load_universe_filter(&options.universe_name, &options.universe_dir)?;
    let needs_industry = options.industry_neutral
        || options.diagnostic_label_space == DiagnosticLabelSpace::IndustryExcess;
    let industry_map = if needs_industry {
        let path = options.industry_map_path.as_ref().ok_or_else(|| {
            "industry diagnostics requested but no --industry-map path was provided".to_owned()
        })?;
        Some(load_industry_map(path)?)
    } else {
        None
    };

    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create output dir {}: {err}",
            options.output_dir.display()
        )
    })?;

    let mut all_summary_rows = Vec::new();
    let mut all_detail_rows = DetailRows::default();
    let mut segment_summary_rows: BTreeMap<String, Vec<SummaryRow>> = BTreeMap::new();
    let mut total_rows = 0usize;

    for feature_chunk in options.feature_names.chunks(chunk_size) {
        let mut context = load_factor_chunk(&FactorChunkLoadOptions {
            bucket_paths: &bucket_paths,
            feature_names: feature_chunk,
            label_column: &label_column,
            start_ns,
            end_ns,
            universe_filter: universe_filter.as_ref(),
            industry_map: industry_map.as_ref(),
            batch_size,
        })?;
        if context.rows.is_empty() {
            continue;
        }
        apply_diagnostic_label_space(
            &mut context.rows,
            &options.diagnostic_label_space,
            options.diagnostic_threshold,
            signal_horizon,
            &options.benchmark,
        )?;
        context.rows.retain(|row| row.label.is_finite());
        if context.rows.is_empty() {
            continue;
        }
        if options.industry_neutral {
            apply_industry_neutralization(&mut context.rows, feature_chunk.len())?;
        }
        rebuild_date_slices(&mut context);
        total_rows = total_rows.max(context.rows.len());
        let mut summary_rows = build_summary_rows(&context, feature_chunk, quantile_bins);
        all_summary_rows.append(&mut summary_rows);
        if options.include_details {
            let mut detail_rows = build_detail_rows(&context, feature_chunk, quantile_bins);
            all_detail_rows
                .bucket_rows
                .append(&mut detail_rows.bucket_rows);
            all_detail_rows
                .spread_rows
                .append(&mut detail_rows.spread_rows);
            all_detail_rows
                .monthly_rank_ic_rows
                .append(&mut detail_rows.monthly_rank_ic_rows);
            all_detail_rows
                .missing_rows
                .append(&mut detail_rows.missing_rows);
        }
        for segment in &options.segments {
            let rows = build_segment_summary_rows(&context, feature_chunk, segment, quantile_bins);
            if !rows.is_empty() {
                segment_summary_rows
                    .entry(segment.name.clone())
                    .or_default()
                    .extend(rows);
            }
        }
    }

    sort_summary_rows(&mut all_summary_rows);
    sort_detail_rows(&mut all_detail_rows);
    for rows in segment_summary_rows.values_mut() {
        sort_summary_rows(rows);
    }
    let segment_comparison = build_segment_comparison(&segment_summary_rows, &options.segments);

    write_summary_outputs(&options.output_dir, &all_summary_rows, top_n)?;
    if options.include_details {
        write_detail_outputs(&options.output_dir, &all_detail_rows)?;
    }
    write_segment_outputs(
        &options.output_dir,
        &segment_summary_rows,
        &segment_comparison,
    )?;
    copy_config_snapshot(&options.output_dir, options.config_snapshot_path.as_ref())?;
    write_readme(
        &options.output_dir,
        &all_summary_rows,
        &segment_comparison,
        top_n,
        options.include_details,
    )?;
    let mut metadata = read_metadata_json(options.metadata_json_path.as_ref())?;
    metadata.insert(
        "factor_store_dir".to_owned(),
        options.factor_store.display().to_string(),
    );
    metadata.insert(
        "feature_count".to_owned(),
        options.feature_names.len().to_string(),
    );
    metadata.insert("row_count".to_owned(), total_rows.to_string());
    metadata.insert("quantile_bins".to_owned(), quantile_bins.to_string());
    metadata.insert(
        "detail_artifacts".to_owned(),
        options.include_details.to_string(),
    );
    metadata.insert(
        "diagnostic_label_space".to_owned(),
        options.diagnostic_label_space.as_str().to_owned(),
    );
    metadata.insert("signal_horizon".to_owned(), signal_horizon.to_string());
    if options.diagnostic_label_space == DiagnosticLabelSpace::BenchmarkExcess {
        metadata.insert(
            "benchmark_mode".to_owned(),
            options.benchmark.mode.as_str().to_owned(),
        );
        metadata.insert(
            "benchmark_value_type".to_owned(),
            options.benchmark.value_type.as_str().to_owned(),
        );
        if let Some(path) = options.benchmark.path.as_ref() {
            metadata.insert("benchmark_path".to_owned(), path.display().to_string());
        }
    }
    metadata.insert(
        "industry_neutral".to_owned(),
        options.industry_neutral.to_string(),
    );
    metadata.insert(
        "segment_count".to_owned(),
        segment_summary_rows.len().to_string(),
    );
    metadata.insert("rust_engine".to_owned(), "true".to_owned());
    write_manifest(
        &options.output_dir,
        &metadata,
        options.include_details,
        !segment_comparison.is_empty(),
    )?;

    Ok(SingleFactorRunSummary {
        output_dir: options.output_dir.display().to_string(),
        factor_store: options.factor_store.display().to_string(),
        feature_count: options.feature_names.len(),
        row_count: total_rows,
        quantile_bins,
        diagnostic_label_space: options.diagnostic_label_space.as_str().to_owned(),
        diagnostic_threshold: options.diagnostic_threshold,
        industry_neutral: options.industry_neutral,
        segment_count: segment_summary_rows.len(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

fn read_factor_store_meta(factor_store: &Path) -> Result<FactorStoreMeta, String> {
    let path = factor_store.join("meta.json");
    let file =
        File::open(&path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let value: JsonValue = serde_json::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let feature_names = value
        .get("feature_names")
        .and_then(JsonValue::as_array)
        .into_iter()
        .flatten()
        .filter_map(JsonValue::as_str)
        .map(str::to_owned)
        .collect::<HashSet<_>>();
    let mut available_label_columns = HashSet::new();
    if let Some(items) = value.get("label_columns").and_then(JsonValue::as_array) {
        for item in items {
            if let Some(column) = item.get("column").and_then(JsonValue::as_str) {
                available_label_columns.insert(column.to_owned());
            }
        }
    }
    let default_label_column = value
        .get("default_label_column")
        .and_then(JsonValue::as_str)
        .map(str::to_owned);
    if let Some(column) = default_label_column.as_deref() {
        available_label_columns.insert(column.to_owned());
    }
    if available_label_columns.is_empty() {
        available_label_columns.insert("label".to_owned());
    }
    Ok(FactorStoreMeta {
        feature_names,
        available_label_columns,
        default_label_column,
    })
}

fn validate_features(meta: &FactorStoreMeta, feature_names: &[String]) -> Result<(), String> {
    if meta.feature_names.is_empty() {
        return Ok(());
    }
    let missing = feature_names
        .iter()
        .filter(|feature| !meta.feature_names.contains(feature.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "requested feature(s) not found in factor-store metadata: {}",
            missing.join(",")
        ));
    }
    Ok(())
}

fn resolve_label_column(meta: &FactorStoreMeta, requested: &str) -> Result<String, String> {
    let requested = if requested.trim().is_empty() {
        "label"
    } else {
        requested.trim()
    };
    if meta.available_label_columns.contains(requested) {
        return Ok(requested.to_owned());
    }
    if requested == "label_1d" && meta.available_label_columns.contains("label") {
        return Ok("label".to_owned());
    }
    if let Some(default) = meta.default_label_column.as_deref() {
        if requested == default {
            return Ok(default.to_owned());
        }
    }
    let mut available = meta
        .available_label_columns
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    available.sort();
    Err(format!(
        "requested label column '{requested}' is not available in factor store. Available label columns: {}",
        available.join(",")
    ))
}

struct FactorChunkLoadOptions<'a> {
    bucket_paths: &'a [PathBuf],
    feature_names: &'a [String],
    label_column: &'a str,
    start_ns: Option<i64>,
    end_ns: Option<i64>,
    universe_filter: Option<&'a UniverseFilter>,
    industry_map: Option<&'a HashMap<String, String>>,
    batch_size: usize,
}

fn load_factor_chunk(options: &FactorChunkLoadOptions<'_>) -> Result<DiagnosticContext, String> {
    let mut columns = vec![
        "date".to_owned(),
        "symbol".to_owned(),
        options.label_column.to_owned(),
    ];
    columns.extend(options.feature_names.iter().cloned());
    columns.sort();
    columns.dedup();
    let mut rows = Vec::new();
    for path in options.bucket_paths {
        let reader = open_projected_parquet_reader(path, &columns, options.batch_size)?;
        for batch in reader {
            let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            append_batch_rows(
                &mut rows,
                &batch,
                path,
                options.feature_names,
                options.label_column,
                options.start_ns,
                options.end_ns,
                options.universe_filter,
                options.industry_map,
            )?;
        }
    }
    rows.sort_by(|left, right| {
        left.date_ns
            .cmp(&right.date_ns)
            .then(left.symbol.cmp(&right.symbol))
    });
    let mut context = DiagnosticContext {
        rows,
        date_slices: Vec::new(),
    };
    rebuild_date_slices(&mut context);
    Ok(context)
}

#[allow(clippy::too_many_arguments)]
fn append_batch_rows(
    rows: &mut Vec<DiagnosticRow>,
    batch: &RecordBatch,
    path: &Path,
    feature_names: &[String],
    label_column: &str,
    start_ns: Option<i64>,
    end_ns: Option<i64>,
    universe_filter: Option<&UniverseFilter>,
    industry_map: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    let date_array = required_column(batch, "date", path)?;
    let symbol_array = required_column(batch, "symbol", path)?;
    let label_array = required_column(batch, label_column, path)?;
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
        let label = sanitize_label(numeric_value(label_array, row_index, path)?);
        let features = feature_arrays
            .iter()
            .map(|array| numeric_value(*array, row_index, path))
            .collect::<Result<Vec<_>, _>>()?;
        let industry =
            industry_map.and_then(|mapping| mapping.get(&normalize_local_symbol(&symbol)).cloned());
        rows.push(DiagnosticRow {
            date_ns,
            symbol,
            label,
            industry,
            features,
        });
    }
    Ok(())
}

fn rebuild_date_slices(context: &mut DiagnosticContext) {
    context.date_slices.clear();
    if context.rows.is_empty() {
        return;
    }
    let mut start = 0usize;
    while start < context.rows.len() {
        let date_ns = context.rows[start].date_ns;
        let mut end = start + 1;
        while end < context.rows.len() && context.rows[end].date_ns == date_ns {
            end += 1;
        }
        context.date_slices.push(DateSlice {
            date_ns,
            start,
            end,
        });
        start = end;
    }
}

fn apply_diagnostic_label_space(
    rows: &mut [DiagnosticRow],
    label_space: &DiagnosticLabelSpace,
    threshold: f64,
    signal_horizon: usize,
    benchmark: &BenchmarkOptions,
) -> Result<(), String> {
    match label_space {
        DiagnosticLabelSpace::RawReturn => Ok(()),
        DiagnosticLabelSpace::IndustryExcess => {
            let mut sums: BTreeMap<(i64, String), (f64, usize)> = BTreeMap::new();
            for row in rows.iter() {
                if let Some(industry) = row.industry.as_deref() {
                    if row.label.is_finite() {
                        let entry = sums
                            .entry((row.date_ns, industry.to_owned()))
                            .or_insert((0.0, 0));
                        entry.0 += row.label;
                        entry.1 += 1;
                    }
                }
            }
            if sums.is_empty() {
                return Err("industry_excess diagnostics requested but no date x industry label groups were available".to_owned());
            }
            let means = sums
                .into_iter()
                .filter_map(|(key, (sum, count))| (count > 0).then_some((key, sum / count as f64)))
                .collect::<BTreeMap<_, _>>();
            for row in rows.iter_mut() {
                let Some(industry) = row.industry.as_deref() else {
                    row.label = f64::NAN;
                    continue;
                };
                let Some(mean) = means.get(&(row.date_ns, industry.to_owned())) else {
                    row.label = f64::NAN;
                    continue;
                };
                if row.label.is_finite() {
                    row.label = row.label - *mean - threshold;
                }
            }
            Ok(())
        }
        DiagnosticLabelSpace::BenchmarkExcess => {
            let benchmark_returns = build_benchmark_returns(rows, benchmark)?;
            if benchmark_returns.is_empty() {
                return Err(
                    "benchmark_excess diagnostics produced an empty benchmark return series"
                        .to_owned(),
                );
            }
            let forward_returns =
                build_forward_compound_return_map(&benchmark_returns, signal_horizon);
            for row in rows.iter_mut() {
                let Some(benchmark_return) = forward_returns.get(&row.date_ns) else {
                    row.label = f64::NAN;
                    continue;
                };
                if row.label.is_finite() && benchmark_return.is_finite() {
                    row.label = row.label - *benchmark_return - threshold;
                } else {
                    row.label = f64::NAN;
                }
            }
            Ok(())
        }
    }
}

fn build_benchmark_returns(
    rows: &[DiagnosticRow],
    benchmark: &BenchmarkOptions,
) -> Result<Vec<(i64, f64)>, String> {
    match benchmark.mode {
        BenchmarkMode::CrossSectionMean => build_cross_section_benchmark_returns(rows),
        BenchmarkMode::File => load_file_benchmark_returns(benchmark),
    }
}

fn build_cross_section_benchmark_returns(
    rows: &[DiagnosticRow],
) -> Result<Vec<(i64, f64)>, String> {
    cross_section_mean_returns(
        rows.iter().map(|row| (row.date_ns, row.label)),
        "benchmark_excess cross_section_mean benchmark had no finite labels",
    )
}

fn load_file_benchmark_returns(benchmark: &BenchmarkOptions) -> Result<Vec<(i64, f64)>, String> {
    let path = benchmark
        .path
        .as_ref()
        .ok_or_else(|| "benchmark path is required when benchmark mode is file".to_owned())?;
    crate::common::benchmark::load_file_benchmark_returns(
        path,
        &benchmark.date_column,
        &benchmark.value_column,
        benchmark.value_type.as_str(),
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

fn apply_industry_neutralization(
    rows: &mut [DiagnosticRow],
    feature_count: usize,
) -> Result<(), String> {
    let mapped_rows = rows.iter().filter(|row| row.industry.is_some()).count();
    if mapped_rows == 0 {
        return Err(
            "industry-neutral diagnostics requested but no rows mapped to a valid industry group"
                .to_owned(),
        );
    }
    for feature_index in 0..feature_count {
        let mut sums: BTreeMap<(i64, String), (f64, usize)> = BTreeMap::new();
        for row in rows.iter() {
            let Some(industry) = row.industry.as_deref() else {
                continue;
            };
            let value = row.features[feature_index];
            if value.is_finite() {
                let entry = sums
                    .entry((row.date_ns, industry.to_owned()))
                    .or_insert((0.0, 0));
                entry.0 += value;
                entry.1 += 1;
            }
        }
        let means = sums
            .into_iter()
            .filter_map(|(key, (sum, count))| (count > 0).then_some((key, sum / count as f64)))
            .collect::<BTreeMap<_, _>>();
        for row in rows.iter_mut() {
            let Some(industry) = row.industry.as_deref() else {
                continue;
            };
            if let Some(mean) = means.get(&(row.date_ns, industry.to_owned())) {
                if row.features[feature_index].is_finite() {
                    row.features[feature_index] -= *mean;
                }
            }
        }
    }
    Ok(())
}

fn build_summary_rows(
    context: &DiagnosticContext,
    feature_names: &[String],
    quantile_bins: usize,
) -> Vec<SummaryRow> {
    feature_names
        .iter()
        .enumerate()
        .map(|(feature_index, feature_name)| {
            compute_summary_row(context, feature_index, feature_name, quantile_bins, None)
        })
        .collect()
}

fn build_segment_summary_rows(
    context: &DiagnosticContext,
    feature_names: &[String],
    segment: &SegmentSpec,
    quantile_bins: usize,
) -> Vec<SummaryRow> {
    let date_slices = context
        .date_slices
        .iter()
        .copied()
        .filter(|slice| slice.date_ns >= segment.start_ns && slice.date_ns <= segment.end_ns)
        .collect::<Vec<_>>();
    if date_slices.is_empty() {
        return Vec::new();
    }
    feature_names
        .iter()
        .enumerate()
        .map(|(feature_index, feature_name)| {
            compute_summary_row(
                context,
                feature_index,
                feature_name,
                quantile_bins,
                Some(&date_slices),
            )
        })
        .collect()
}

fn compute_summary_row(
    context: &DiagnosticContext,
    feature_index: usize,
    feature_name: &str,
    quantile_bins: usize,
    date_slices_override: Option<&[DateSlice]>,
) -> SummaryRow {
    let date_slices = date_slices_override.unwrap_or(&context.date_slices);
    let total_obs = date_slices
        .iter()
        .map(|slice| slice.end - slice.start)
        .sum::<usize>();
    let mut daily_coverage_values = Vec::new();
    let mut daily_ic_values = Vec::new();
    let mut daily_rank_ic_values = Vec::new();
    let mut daily_rank_ic_dates = Vec::new();
    let mut monotonicity_values = Vec::new();
    let mut spread_values = Vec::new();
    let mut valid_observation_count = 0usize;

    for date_slice in date_slices {
        let rows = &context.rows[date_slice.start..date_slice.end];
        let row_count = rows.len();
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for row in rows {
            let x = row.features[feature_index];
            let y = row.label;
            if x.is_finite() && y.is_finite() {
                xs.push(x);
                ys.push(y);
            }
        }
        let valid_count = xs.len();
        valid_observation_count += valid_count;
        daily_coverage_values.push(safe_ratio(valid_count, row_count));
        if valid_count < 2 {
            continue;
        }
        let ic = pearson_corr(&xs, &ys);
        let rank_ic = spearman_corr(&xs, &ys);
        if ic.is_finite() && rank_ic.is_finite() {
            daily_ic_values.push(ic);
            daily_rank_ic_values.push(rank_ic);
            daily_rank_ic_dates.push(date_slice.date_ns);
        }
        if unique_count(&xs) < 2 || unique_count(&ys) < 2 || valid_count < quantile_bins {
            continue;
        }
        let bucket = rank_to_quantile_bins(&xs, quantile_bins);
        let (bucket_indices, bucket_means, _) = bucket_means(&bucket, &ys, quantile_bins);
        if bucket_indices.len() < 2 {
            continue;
        }
        let spread = bucket_means[bucket_means.len() - 1] - bucket_means[0];
        let bucket_indices_f64 = bucket_indices
            .iter()
            .map(|value| *value as f64)
            .collect::<Vec<_>>();
        let monotonicity = spearman_corr(&bucket_indices_f64, &bucket_means);
        if monotonicity.is_finite() {
            monotonicity_values.push(monotonicity);
            spread_values.push(spread);
        }
    }

    let monthly_rank_ic = monthly_means(&daily_rank_ic_dates, &daily_rank_ic_values);
    let rank_ic_mean = mean(&daily_rank_ic_values);
    let direction = if rank_ic_mean.is_finite() && !is_close_zero(rank_ic_mean) {
        if rank_ic_mean > 0.0 {
            1
        } else {
            -1
        }
    } else {
        0
    };
    let rank_ic_directional_hit_rate = if direction != 0 && !daily_rank_ic_values.is_empty() {
        mean(
            &daily_rank_ic_values
                .iter()
                .map(|value| {
                    if value * direction as f64 > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        )
    } else {
        f64::NAN
    };
    let monthly_values = monthly_rank_ic.values().copied().collect::<Vec<_>>();
    let monthly_rank_ic_directional_hit_rate = if direction != 0 && !monthly_values.is_empty() {
        mean(
            &monthly_values
                .iter()
                .map(|value| {
                    if value * direction as f64 > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
        )
    } else {
        f64::NAN
    };
    let rank_ic_ir = safe_icir(&daily_rank_ic_values);
    let ic_ir = safe_icir(&daily_ic_values);
    SummaryRow {
        feature: feature_name.to_owned(),
        feature_group: infer_feature_group(feature_name).to_owned(),
        observation_count: total_obs,
        valid_observation_count,
        coverage_pct: safe_ratio(valid_observation_count, total_obs),
        avg_daily_coverage_pct: mean(&daily_coverage_values),
        date_count: date_slices.len(),
        effective_date_count: daily_ic_values.len(),
        monotonic_date_count: monotonicity_values.len(),
        ic_mean: mean(&daily_ic_values),
        ic_std: sample_std(&daily_ic_values),
        ic_ir,
        ic_positive_rate: positive_rate(&daily_ic_values),
        rank_ic_mean,
        rank_ic_std: sample_std(&daily_rank_ic_values),
        rank_ic_ir,
        rank_ic_positive_rate: positive_rate(&daily_rank_ic_values),
        rank_ic_directional_hit_rate,
        rank_ic_abs_mean: if rank_ic_mean.is_finite() {
            rank_ic_mean.abs()
        } else {
            f64::NAN
        },
        monthly_rank_ic_mean: mean(&monthly_values),
        monthly_rank_ic_positive_rate: positive_rate(&monthly_values),
        monthly_rank_ic_directional_hit_rate,
        monthly_rank_ic_months: monthly_values.len(),
        monotonicity_mean: mean(&monotonicity_values),
        monotonicity_positive_rate: positive_rate(&monotonicity_values),
        top_bottom_spread_mean: mean(&spread_values),
        top_bottom_spread_positive_rate: positive_rate(&spread_values),
        suggested_direction: direction,
        rank_ic_ir_abs: if rank_ic_ir.is_finite() {
            rank_ic_ir.abs()
        } else {
            f64::NAN
        },
        ic_ir_abs: if ic_ir.is_finite() {
            ic_ir.abs()
        } else {
            f64::NAN
        },
    }
}

fn build_detail_rows(
    context: &DiagnosticContext,
    feature_names: &[String],
    quantile_bins: usize,
) -> DetailRows {
    let mut detail = DetailRows::default();
    let years = context
        .rows
        .iter()
        .map(|row| year_from_ns(row.date_ns))
        .collect::<Vec<_>>();
    for (feature_index, feature_name) in feature_names.iter().enumerate() {
        let feature_group = infer_feature_group(feature_name).to_owned();
        let mut daily_rank_rows = Vec::new();
        for date_slice in &context.date_slices {
            let rows = &context.rows[date_slice.start..date_slice.end];
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for row in rows {
                let x = row.features[feature_index];
                let y = row.label;
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                }
            }
            let valid_count = xs.len();
            if valid_count < 2 {
                continue;
            }
            let ic = pearson_corr(&xs, &ys);
            let rank_ic = spearman_corr(&xs, &ys);
            daily_rank_rows.push(DailyRankIcRow {
                feature: feature_name.clone(),
                feature_group: feature_group.clone(),
                date_ns: date_slice.date_ns,
                ic,
                rank_ic,
                valid_count,
            });
            if unique_count(&xs) < 2 || valid_count < quantile_bins {
                continue;
            }
            let bucket = rank_to_quantile_bins(&xs, quantile_bins);
            let (bucket_indices, bucket_mean_values, bucket_counts) =
                bucket_means(&bucket, &ys, quantile_bins);
            if bucket_indices.len() < 2 {
                continue;
            }
            for (position, bucket_id) in bucket_indices.iter().enumerate() {
                detail.bucket_rows.push(BucketReturnRow {
                    feature: feature_name.clone(),
                    feature_group: feature_group.clone(),
                    date_ns: date_slice.date_ns,
                    bucket: *bucket_id,
                    bucket_mean_label: bucket_mean_values[position],
                    bucket_count: bucket_counts[position],
                });
            }
            let bottom_bucket = bucket_indices[0];
            let top_bucket = bucket_indices[bucket_indices.len() - 1];
            let bottom_mean = bucket_mean_values[0];
            let top_mean = bucket_mean_values[bucket_mean_values.len() - 1];
            detail.spread_rows.push(SpreadRow {
                feature: feature_name.clone(),
                feature_group: feature_group.clone(),
                date_ns: date_slice.date_ns,
                bottom_bucket,
                top_bucket,
                bottom_mean_label: bottom_mean,
                top_mean_label: top_mean,
                top_bottom_spread: top_mean - bottom_mean,
                valid_count,
            });
        }
        detail
            .monthly_rank_ic_rows
            .extend(build_monthly_rank_rows(&daily_rank_rows));
        detail.missing_rows.extend(build_missing_rows(
            context,
            &years,
            feature_index,
            feature_name,
            &feature_group,
        ));
    }
    detail
}

fn build_monthly_rank_rows(rows: &[DailyRankIcRow]) -> Vec<MonthlyRankIcRow> {
    let mut grouped: BTreeMap<i64, Vec<&DailyRankIcRow>> = BTreeMap::new();
    for row in rows {
        grouped
            .entry(month_end_ns(row.date_ns))
            .or_default()
            .push(row);
    }
    let mut out = Vec::new();
    for (month_ns, values) in grouped {
        if values.is_empty() {
            continue;
        }
        let ic_values = values
            .iter()
            .map(|row| row.ic)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let rank_ic_values = values
            .iter()
            .map(|row| row.rank_ic)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let valid_counts = values
            .iter()
            .map(|row| row.valid_count as f64)
            .collect::<Vec<_>>();
        out.push(MonthlyRankIcRow {
            feature: values[0].feature.clone(),
            feature_group: values[0].feature_group.clone(),
            month_ns,
            ic_mean: mean(&ic_values),
            rank_ic_mean: mean(&rank_ic_values),
            rank_ic_count: rank_ic_values.len(),
            avg_valid_count: mean(&valid_counts),
        });
    }
    out
}

fn build_missing_rows(
    context: &DiagnosticContext,
    years: &[i32],
    feature_index: usize,
    feature_name: &str,
    feature_group: &str,
) -> Vec<MissingByYearRow> {
    let mut grouped: BTreeMap<i32, (usize, usize, usize)> = BTreeMap::new();
    for (row_index, row) in context.rows.iter().enumerate() {
        let entry = grouped.entry(years[row_index]).or_insert((0, 0, 0));
        entry.0 += 1;
        let feature_valid = row.features[feature_index].is_finite();
        let label_valid = row.label.is_finite();
        if feature_valid {
            entry.1 += 1;
        }
        if feature_valid && label_valid {
            entry.2 += 1;
        }
    }
    grouped
        .into_iter()
        .map(
            |(year, (obs, feature_valid, paired_valid))| MissingByYearRow {
                feature: feature_name.to_owned(),
                feature_group: feature_group.to_owned(),
                year,
                observation_count: obs,
                feature_valid_count: feature_valid,
                paired_valid_count: paired_valid,
                feature_coverage_pct: safe_ratio(feature_valid, obs),
                paired_coverage_pct: safe_ratio(paired_valid, obs),
                feature_missing_pct: 1.0 - safe_ratio(feature_valid, obs),
            },
        )
        .collect()
}

fn build_segment_comparison(
    segment_summary_rows: &BTreeMap<String, Vec<SummaryRow>>,
    segments: &[SegmentSpec],
) -> Vec<SegmentComparisonRow> {
    if segment_summary_rows.is_empty() {
        return Vec::new();
    }
    let mut by_feature: BTreeMap<String, BTreeMap<String, SummaryRow>> = BTreeMap::new();
    for segment in segments {
        let Some(rows) = segment_summary_rows.get(&segment.name) else {
            continue;
        };
        for row in rows {
            by_feature
                .entry(row.feature.clone())
                .or_default()
                .insert(segment.name.clone(), row.clone());
        }
    }
    let mut out = Vec::new();
    for (feature, per_segment) in by_feature {
        let mut values = BTreeMap::new();
        let mut rank_ic_values = Vec::new();
        let mut direction_values = Vec::new();
        let mut monthly_hit_values = Vec::new();
        for segment in segments {
            let Some(row) = per_segment.get(&segment.name) else {
                continue;
            };
            for (key, value) in summary_metric_pairs(row) {
                values.insert(format!("{key}__{}", segment.name), value);
            }
            if row.rank_ic_mean.is_finite() {
                rank_ic_values.push((segment.name.clone(), row.rank_ic_mean));
            }
            direction_values.push(row.suggested_direction);
            if row.monthly_rank_ic_directional_hit_rate.is_finite() {
                monthly_hit_values.push(row.monthly_rank_ic_directional_hit_rate);
            }
        }
        let rank_abs_max = rank_ic_values
            .iter()
            .map(|(_, value)| value.abs())
            .fold(f64::NAN, nanmax);
        let rank_abs_min = rank_ic_values
            .iter()
            .map(|(_, value)| value.abs())
            .fold(f64::NAN, nanmin);
        let rank_range = if rank_ic_values.is_empty() {
            f64::NAN
        } else {
            let max_value = rank_ic_values
                .iter()
                .map(|(_, value)| *value)
                .fold(f64::NAN, nanmax);
            let min_value = rank_ic_values
                .iter()
                .map(|(_, value)| *value)
                .fold(f64::NAN, nanmin);
            max_value - min_value
        };
        let best_segment = rank_ic_values
            .iter()
            .max_by(|left, right| cmp_f64(left.1.abs(), right.1.abs()))
            .map(|(name, _)| name.clone())
            .unwrap_or_default();
        let worst_segment = rank_ic_values
            .iter()
            .min_by(|left, right| cmp_f64(left.1.abs(), right.1.abs()))
            .map(|(name, _)| name.clone())
            .unwrap_or_default();
        let positive = direction_values.iter().filter(|value| **value > 0).count();
        let negative = direction_values.iter().filter(|value| **value < 0).count();
        let nonzero = direction_values.iter().filter(|value| **value != 0).count();
        let monthly_hit_mean = mean(&monthly_hit_values);
        let monthly_hit_min = monthly_hit_values.iter().copied().fold(f64::NAN, nanmin);
        values.insert(
            "segment_rank_ic_abs_max".to_owned(),
            format_float(rank_abs_max),
        );
        values.insert(
            "segment_rank_ic_abs_min".to_owned(),
            format_float(rank_abs_min),
        );
        values.insert(
            "segment_rank_ic_mean_range".to_owned(),
            format_float(rank_range),
        );
        values.insert("best_segment_by_abs_rank_ic".to_owned(), best_segment);
        values.insert("worst_segment_by_abs_rank_ic".to_owned(), worst_segment);
        values.insert(
            "positive_direction_segments".to_owned(),
            positive.to_string(),
        );
        values.insert(
            "negative_direction_segments".to_owned(),
            negative.to_string(),
        );
        values.insert("nonzero_direction_segments".to_owned(), nonzero.to_string());
        values.insert(
            "direction_flip".to_owned(),
            ((positive > 0) && (negative > 0)).to_string(),
        );
        values.insert(
            "segment_monthly_directional_hit_mean".to_owned(),
            format_float(monthly_hit_mean),
        );
        values.insert(
            "segment_monthly_directional_hit_min".to_owned(),
            format_float(monthly_hit_min),
        );
        out.push(SegmentComparisonRow {
            feature,
            values,
            sort_rank_ic_abs_max: rank_abs_max,
            sort_monthly_hit_mean: monthly_hit_mean,
        });
    }
    out.sort_by(|left, right| {
        cmp_f64_desc(left.sort_rank_ic_abs_max, right.sort_rank_ic_abs_max)
            .then(cmp_f64_desc(
                left.sort_monthly_hit_mean,
                right.sort_monthly_hit_mean,
            ))
            .then(left.feature.cmp(&right.feature))
    });
    out
}

fn summary_metric_pairs(row: &SummaryRow) -> Vec<(&'static str, String)> {
    vec![
        ("feature_group", row.feature_group.clone()),
        ("observation_count", row.observation_count.to_string()),
        (
            "valid_observation_count",
            row.valid_observation_count.to_string(),
        ),
        ("coverage_pct", format_float(row.coverage_pct)),
        (
            "avg_daily_coverage_pct",
            format_float(row.avg_daily_coverage_pct),
        ),
        ("date_count", row.date_count.to_string()),
        ("effective_date_count", row.effective_date_count.to_string()),
        ("monotonic_date_count", row.monotonic_date_count.to_string()),
        ("ic_mean", format_float(row.ic_mean)),
        ("ic_std", format_float(row.ic_std)),
        ("ic_ir", format_float(row.ic_ir)),
        ("ic_positive_rate", format_float(row.ic_positive_rate)),
        ("rank_ic_mean", format_float(row.rank_ic_mean)),
        ("rank_ic_std", format_float(row.rank_ic_std)),
        ("rank_ic_ir", format_float(row.rank_ic_ir)),
        (
            "rank_ic_positive_rate",
            format_float(row.rank_ic_positive_rate),
        ),
        (
            "rank_ic_directional_hit_rate",
            format_float(row.rank_ic_directional_hit_rate),
        ),
        ("rank_ic_abs_mean", format_float(row.rank_ic_abs_mean)),
        (
            "monthly_rank_ic_mean",
            format_float(row.monthly_rank_ic_mean),
        ),
        (
            "monthly_rank_ic_positive_rate",
            format_float(row.monthly_rank_ic_positive_rate),
        ),
        (
            "monthly_rank_ic_directional_hit_rate",
            format_float(row.monthly_rank_ic_directional_hit_rate),
        ),
        (
            "monthly_rank_ic_months",
            row.monthly_rank_ic_months.to_string(),
        ),
        ("monotonicity_mean", format_float(row.monotonicity_mean)),
        (
            "monotonicity_positive_rate",
            format_float(row.monotonicity_positive_rate),
        ),
        (
            "top_bottom_spread_mean",
            format_float(row.top_bottom_spread_mean),
        ),
        (
            "top_bottom_spread_positive_rate",
            format_float(row.top_bottom_spread_positive_rate),
        ),
        ("suggested_direction", row.suggested_direction.to_string()),
        ("rank_ic_ir_abs", format_float(row.rank_ic_ir_abs)),
        ("ic_ir_abs", format_float(row.ic_ir_abs)),
    ]
}

fn sort_summary_rows(rows: &mut [SummaryRow]) {
    rows.sort_by(|left, right| {
        cmp_f64_desc(left.rank_ic_abs_mean, right.rank_ic_abs_mean)
            .then(cmp_f64_desc(left.rank_ic_ir_abs, right.rank_ic_ir_abs))
            .then(cmp_f64_desc(left.coverage_pct, right.coverage_pct))
            .then(left.feature.cmp(&right.feature))
    });
}

fn sort_detail_rows(detail: &mut DetailRows) {
    detail.bucket_rows.sort_by(|left, right| {
        left.feature
            .cmp(&right.feature)
            .then(left.date_ns.cmp(&right.date_ns))
            .then(left.bucket.cmp(&right.bucket))
    });
    detail.spread_rows.sort_by(|left, right| {
        left.feature
            .cmp(&right.feature)
            .then(left.date_ns.cmp(&right.date_ns))
    });
    detail.monthly_rank_ic_rows.sort_by(|left, right| {
        left.feature
            .cmp(&right.feature)
            .then(left.month_ns.cmp(&right.month_ns))
    });
    detail.missing_rows.sort_by(|left, right| {
        left.feature
            .cmp(&right.feature)
            .then(left.year.cmp(&right.year))
    });
}

fn write_summary_outputs(
    output_dir: &Path,
    rows: &[SummaryRow],
    top_n: usize,
) -> Result<(), String> {
    write_summary_csv(&output_dir.join("single_factor_summary.csv"), rows)?;
    let mut top_abs = rows.to_vec();
    top_abs.sort_by(|left, right| {
        cmp_f64_desc(left.rank_ic_abs_mean, right.rank_ic_abs_mean)
            .then(cmp_f64_desc(left.coverage_pct, right.coverage_pct))
    });
    write_summary_csv(
        &output_dir.join("single_factor_top_abs_rankic.csv"),
        take_top(&top_abs, top_n),
    )?;
    let mut top_rank = rows.to_vec();
    top_rank.sort_by(|left, right| {
        cmp_f64_desc(left.rank_ic_mean, right.rank_ic_mean)
            .then(cmp_f64_desc(left.coverage_pct, right.coverage_pct))
    });
    write_summary_csv(
        &output_dir.join("single_factor_top_rankic.csv"),
        take_top(&top_rank, top_n),
    )?;
    let mut top_ir = rows.to_vec();
    top_ir.sort_by(|left, right| {
        cmp_f64_desc(left.rank_ic_ir, right.rank_ic_ir)
            .then(cmp_f64_desc(left.coverage_pct, right.coverage_pct))
    });
    write_summary_csv(
        &output_dir.join("single_factor_top_rankic_ir.csv"),
        take_top(&top_ir, top_n),
    )?;
    Ok(())
}

fn take_top(rows: &[SummaryRow], top_n: usize) -> &[SummaryRow] {
    &rows[..rows.len().min(top_n)]
}

fn write_summary_csv(path: &Path, rows: &[SummaryRow]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record(summary_headers())
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    for row in rows {
        writer
            .write_record(summary_record(row))
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn summary_headers() -> &'static [&'static str] {
    &[
        "feature",
        "feature_group",
        "observation_count",
        "valid_observation_count",
        "coverage_pct",
        "avg_daily_coverage_pct",
        "date_count",
        "effective_date_count",
        "monotonic_date_count",
        "ic_mean",
        "ic_std",
        "ic_ir",
        "ic_positive_rate",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_ic_ir",
        "rank_ic_positive_rate",
        "rank_ic_directional_hit_rate",
        "rank_ic_abs_mean",
        "monthly_rank_ic_mean",
        "monthly_rank_ic_positive_rate",
        "monthly_rank_ic_directional_hit_rate",
        "monthly_rank_ic_months",
        "monotonicity_mean",
        "monotonicity_positive_rate",
        "top_bottom_spread_mean",
        "top_bottom_spread_positive_rate",
        "suggested_direction",
        "rank_ic_ir_abs",
        "ic_ir_abs",
    ]
}

fn summary_record(row: &SummaryRow) -> Vec<String> {
    vec![
        row.feature.clone(),
        row.feature_group.clone(),
        row.observation_count.to_string(),
        row.valid_observation_count.to_string(),
        format_float(row.coverage_pct),
        format_float(row.avg_daily_coverage_pct),
        row.date_count.to_string(),
        row.effective_date_count.to_string(),
        row.monotonic_date_count.to_string(),
        format_float(row.ic_mean),
        format_float(row.ic_std),
        format_float(row.ic_ir),
        format_float(row.ic_positive_rate),
        format_float(row.rank_ic_mean),
        format_float(row.rank_ic_std),
        format_float(row.rank_ic_ir),
        format_float(row.rank_ic_positive_rate),
        format_float(row.rank_ic_directional_hit_rate),
        format_float(row.rank_ic_abs_mean),
        format_float(row.monthly_rank_ic_mean),
        format_float(row.monthly_rank_ic_positive_rate),
        format_float(row.monthly_rank_ic_directional_hit_rate),
        row.monthly_rank_ic_months.to_string(),
        format_float(row.monotonicity_mean),
        format_float(row.monotonicity_positive_rate),
        format_float(row.top_bottom_spread_mean),
        format_float(row.top_bottom_spread_positive_rate),
        row.suggested_direction.to_string(),
        format_float(row.rank_ic_ir_abs),
        format_float(row.ic_ir_abs),
    ]
}

fn write_detail_outputs(output_dir: &Path, detail: &DetailRows) -> Result<(), String> {
    write_bucket_csv(
        &output_dir.join("single_factor_bucket_return_daily.csv"),
        &detail.bucket_rows,
    )?;
    write_spread_csv(
        &output_dir.join("single_factor_top_bottom_spread_daily.csv"),
        &detail.spread_rows,
    )?;
    write_monthly_csv(
        &output_dir.join("single_factor_rank_ic_monthly.csv"),
        &detail.monthly_rank_ic_rows,
    )?;
    write_missing_csv(
        &output_dir.join("single_factor_missing_by_year.csv"),
        &detail.missing_rows,
    )?;
    Ok(())
}

fn write_bucket_csv(path: &Path, rows: &[BucketReturnRow]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record([
            "feature",
            "feature_group",
            "date",
            "bucket",
            "bucket_mean_label",
            "bucket_count",
        ])
        .map_err(|err| err.to_string())?;
    for row in rows {
        writer
            .write_record([
                row.feature.clone(),
                row.feature_group.clone(),
                format_date_ns(row.date_ns),
                row.bucket.to_string(),
                format_float(row.bucket_mean_label),
                row.bucket_count.to_string(),
            ])
            .map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn write_spread_csv(path: &Path, rows: &[SpreadRow]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record([
            "feature",
            "feature_group",
            "date",
            "bottom_bucket",
            "top_bucket",
            "bottom_mean_label",
            "top_mean_label",
            "top_bottom_spread",
            "valid_count",
        ])
        .map_err(|err| err.to_string())?;
    for row in rows {
        writer
            .write_record([
                row.feature.clone(),
                row.feature_group.clone(),
                format_date_ns(row.date_ns),
                row.bottom_bucket.to_string(),
                row.top_bucket.to_string(),
                format_float(row.bottom_mean_label),
                format_float(row.top_mean_label),
                format_float(row.top_bottom_spread),
                row.valid_count.to_string(),
            ])
            .map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn write_monthly_csv(path: &Path, rows: &[MonthlyRankIcRow]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record([
            "feature",
            "feature_group",
            "month",
            "ic_mean",
            "rank_ic_mean",
            "rank_ic_count",
            "avg_valid_count",
        ])
        .map_err(|err| err.to_string())?;
    for row in rows {
        writer
            .write_record([
                row.feature.clone(),
                row.feature_group.clone(),
                format_date_ns(row.month_ns),
                format_float(row.ic_mean),
                format_float(row.rank_ic_mean),
                row.rank_ic_count.to_string(),
                format_float(row.avg_valid_count),
            ])
            .map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn write_missing_csv(path: &Path, rows: &[MissingByYearRow]) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record([
            "feature",
            "feature_group",
            "year",
            "observation_count",
            "feature_valid_count",
            "paired_valid_count",
            "feature_coverage_pct",
            "paired_coverage_pct",
            "feature_missing_pct",
        ])
        .map_err(|err| err.to_string())?;
    for row in rows {
        writer
            .write_record([
                row.feature.clone(),
                row.feature_group.clone(),
                row.year.to_string(),
                row.observation_count.to_string(),
                row.feature_valid_count.to_string(),
                row.paired_valid_count.to_string(),
                format_float(row.feature_coverage_pct),
                format_float(row.paired_coverage_pct),
                format_float(row.feature_missing_pct),
            ])
            .map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn write_segment_outputs(
    output_dir: &Path,
    segment_summary_rows: &BTreeMap<String, Vec<SummaryRow>>,
    comparison: &[SegmentComparisonRow],
) -> Result<(), String> {
    if !comparison.is_empty() {
        write_segment_comparison_csv(
            &output_dir.join("single_factor_segment_comparison.csv"),
            comparison,
        )?;
    }
    if !segment_summary_rows.is_empty() {
        let segment_dir = output_dir.join("segments");
        fs::create_dir_all(&segment_dir)
            .map_err(|err| format!("failed to create {}: {err}", segment_dir.display()))?;
        for (name, rows) in segment_summary_rows {
            write_summary_csv(&segment_dir.join(format!("{name}.csv")), rows)?;
        }
    }
    Ok(())
}

fn write_segment_comparison_csv(path: &Path, rows: &[SegmentComparisonRow]) -> Result<(), String> {
    let mut headers = vec!["feature".to_owned()];
    let mut seen = BTreeSet::new();
    for row in rows {
        for key in row.values.keys() {
            if seen.insert(key.clone()) {
                headers.push(key.clone());
            }
        }
    }
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record(headers.iter())
        .map_err(|err| err.to_string())?;
    for row in rows {
        let mut record = vec![row.feature.clone()];
        for header in headers.iter().skip(1) {
            record.push(row.values.get(header).cloned().unwrap_or_default());
        }
        writer.write_record(record).map_err(|err| err.to_string())?;
    }
    writer.flush().map_err(|err| err.to_string())
}

fn write_readme(
    output_dir: &Path,
    summary_rows: &[SummaryRow],
    segment_comparison: &[SegmentComparisonRow],
    top_n: usize,
    include_details: bool,
) -> Result<(), String> {
    let mut content = String::new();
    content.push_str("# Single-Factor Diagnostics\n\n");
    content.push_str("## Top Factors By Absolute RankIC\n\n");
    content.push_str(&markdown_summary_table(summary_rows, top_n.min(20)));
    content.push_str("\n## Top Factors By RankICIR\n\n");
    let mut by_ir = summary_rows.to_vec();
    by_ir.sort_by(|left, right| {
        cmp_f64_desc(left.rank_ic_ir, right.rank_ic_ir)
            .then(cmp_f64_desc(left.coverage_pct, right.coverage_pct))
    });
    content.push_str(&markdown_summary_table(&by_ir, top_n.min(20)));
    content.push_str("\n## Detailed Artifacts\n\n");
    if include_details {
        content.push_str("- bucket_return_daily_csv: `single_factor_bucket_return_daily.csv`\n");
        content.push_str(
            "- top_bottom_spread_daily_csv: `single_factor_top_bottom_spread_daily.csv`\n",
        );
        content.push_str("- rank_ic_monthly_csv: `single_factor_rank_ic_monthly.csv`\n");
        content.push_str("- missing_by_year_csv: `single_factor_missing_by_year.csv`\n");
    } else {
        content.push_str("_Detail artifacts disabled._\n");
    }
    if !segment_comparison.is_empty() {
        content.push_str("\n## Segment Comparison\n\n");
        content.push_str("| feature | direction_flip | best_segment_by_abs_rank_ic | worst_segment_by_abs_rank_ic | segment_rank_ic_abs_max | segment_rank_ic_mean_range | segment_monthly_directional_hit_mean |\n");
        content.push_str("|---|---:|---|---|---:|---:|---:|\n");
        for row in segment_comparison.iter().take(top_n.min(20)) {
            content.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} |\n",
                row.feature,
                row.values
                    .get("direction_flip")
                    .cloned()
                    .unwrap_or_default(),
                row.values
                    .get("best_segment_by_abs_rank_ic")
                    .cloned()
                    .unwrap_or_default(),
                row.values
                    .get("worst_segment_by_abs_rank_ic")
                    .cloned()
                    .unwrap_or_default(),
                row.values
                    .get("segment_rank_ic_abs_max")
                    .cloned()
                    .unwrap_or_default(),
                row.values
                    .get("segment_rank_ic_mean_range")
                    .cloned()
                    .unwrap_or_default(),
                row.values
                    .get("segment_monthly_directional_hit_mean")
                    .cloned()
                    .unwrap_or_default(),
            ));
        }
    }
    fs::write(output_dir.join("README.md"), content)
        .map_err(|err| format!("failed to write README.md: {err}"))
}

fn markdown_summary_table(rows: &[SummaryRow], top_n: usize) -> String {
    if rows.is_empty() {
        return "_No factors available._\n".to_owned();
    }
    let mut out = String::new();
    out.push_str("| feature | rank_ic_mean | rank_ic_ir | coverage_pct | monotonicity_mean | monthly_rank_ic_directional_hit_rate |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for row in rows.iter().take(top_n) {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            row.feature,
            format_float(row.rank_ic_mean),
            format_float(row.rank_ic_ir),
            format_float(row.coverage_pct),
            format_float(row.monotonicity_mean),
            format_float(row.monthly_rank_ic_directional_hit_rate),
        ));
    }
    out
}

fn write_manifest(
    output_dir: &Path,
    metadata: &BTreeMap<String, String>,
    include_details: bool,
    has_segment_comparison: bool,
) -> Result<(), String> {
    let mut artifacts = serde_json::Map::new();
    artifacts.insert(
        "summary_csv".to_owned(),
        JsonValue::String(
            output_dir
                .join("single_factor_summary.csv")
                .display()
                .to_string(),
        ),
    );
    artifacts.insert(
        "top_abs_rankic_csv".to_owned(),
        JsonValue::String(
            output_dir
                .join("single_factor_top_abs_rankic.csv")
                .display()
                .to_string(),
        ),
    );
    artifacts.insert(
        "top_rankic_csv".to_owned(),
        JsonValue::String(
            output_dir
                .join("single_factor_top_rankic.csv")
                .display()
                .to_string(),
        ),
    );
    artifacts.insert(
        "top_rankic_ir_csv".to_owned(),
        JsonValue::String(
            output_dir
                .join("single_factor_top_rankic_ir.csv")
                .display()
                .to_string(),
        ),
    );
    artifacts.insert(
        "readme_path".to_owned(),
        JsonValue::String(output_dir.join("README.md").display().to_string()),
    );
    artifacts.insert(
        "manifest_path".to_owned(),
        JsonValue::String(output_dir.join("manifest.json").display().to_string()),
    );
    artifacts.insert(
        "config_snapshot_path".to_owned(),
        JsonValue::String(
            output_dir
                .join("config_snapshot.yaml")
                .display()
                .to_string(),
        ),
    );
    if include_details {
        artifacts.insert(
            "bucket_return_daily_csv".to_owned(),
            JsonValue::String(
                output_dir
                    .join("single_factor_bucket_return_daily.csv")
                    .display()
                    .to_string(),
            ),
        );
        artifacts.insert(
            "top_bottom_spread_daily_csv".to_owned(),
            JsonValue::String(
                output_dir
                    .join("single_factor_top_bottom_spread_daily.csv")
                    .display()
                    .to_string(),
            ),
        );
        artifacts.insert(
            "rank_ic_monthly_csv".to_owned(),
            JsonValue::String(
                output_dir
                    .join("single_factor_rank_ic_monthly.csv")
                    .display()
                    .to_string(),
            ),
        );
        artifacts.insert(
            "missing_by_year_csv".to_owned(),
            JsonValue::String(
                output_dir
                    .join("single_factor_missing_by_year.csv")
                    .display()
                    .to_string(),
            ),
        );
    }
    if has_segment_comparison {
        artifacts.insert(
            "segment_comparison_csv".to_owned(),
            JsonValue::String(
                output_dir
                    .join("single_factor_segment_comparison.csv")
                    .display()
                    .to_string(),
            ),
        );
    }
    let metadata_value = metadata
        .iter()
        .map(|(key, value)| (key.clone(), JsonValue::String(value.clone())))
        .collect::<serde_json::Map<_, _>>();
    let manifest = serde_json::json!({
        "metadata": metadata_value,
        "artifacts": artifacts,
    });
    fs::write(
        output_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)
            .map_err(|err| format!("failed to serialize manifest: {err}"))?,
    )
    .map_err(|err| format!("failed to write manifest.json: {err}"))
}

fn copy_config_snapshot(output_dir: &Path, source: Option<&PathBuf>) -> Result<(), String> {
    let target = output_dir.join("config_snapshot.yaml");
    if let Some(source) = source {
        if source.exists() {
            if source != &target {
                fs::copy(source, &target).map_err(|err| {
                    format!(
                        "failed to copy config snapshot {} -> {}: {err}",
                        source.display(),
                        target.display()
                    )
                })?;
            }
            return Ok(());
        }
    }
    if !target.exists() {
        fs::write(&target, "{}\n")
            .map_err(|err| format!("failed to write {}: {err}", target.display()))?;
    }
    Ok(())
}

fn read_metadata_json(path: Option<&PathBuf>) -> Result<BTreeMap<String, String>, String> {
    let Some(path) = path else {
        return Ok(BTreeMap::new());
    };
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: JsonValue = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let mut out = BTreeMap::new();
    if let Some(object) = value.as_object() {
        for (key, value) in object {
            let rendered = match value {
                JsonValue::Null => String::new(),
                JsonValue::String(inner) => inner.clone(),
                _ => value.to_string(),
            };
            out.insert(key.clone(), rendered);
        }
    }
    Ok(out)
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

fn load_industry_map(path: &Path) -> Result<HashMap<String, String>, String> {
    let mut columns = vec![
        "industry".to_owned(),
        "local_symbol".to_owned(),
        "symbol".to_owned(),
    ];
    columns.sort();
    columns.dedup();
    let available = read_parquet_columns(path)?;
    let symbol_column = if available.contains("local_symbol") {
        "local_symbol"
    } else if available.contains("symbol") {
        "symbol"
    } else {
        return Err(format!(
            "{} missing local_symbol/symbol column",
            path.display()
        ));
    };
    if !available.contains("industry") {
        return Err(format!("{} missing industry column", path.display()));
    }
    columns = vec![symbol_column.to_owned(), "industry".to_owned()];
    let reader = open_projected_parquet_reader(path, &columns, 65_536)?;
    let mut map = HashMap::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let symbol_array = required_column(&batch, symbol_column, path)?;
        let industry_array = required_column(&batch, "industry", path)?;
        for row_index in 0..batch.num_rows() {
            if industry_array.is_null(row_index) {
                continue;
            }
            let symbol = normalize_local_symbol(&string_value(symbol_array, row_index, path)?);
            let industry = string_value(industry_array, row_index, path)?
                .trim()
                .to_owned();
            if !symbol.is_empty() && !industry.is_empty() {
                map.insert(symbol, industry);
            }
        }
    }
    if map.is_empty() {
        return Err(format!(
            "{} did not contain any usable industry mappings",
            path.display()
        ));
    }
    Ok(map)
}

fn read_parquet_columns(path: &Path) -> Result<HashSet<String>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to open parquet {}: {err}", path.display()))?;
    Ok(builder
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().to_owned())
        .collect())
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

fn safe_ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return f64::NAN;
    }
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn safe_icir(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let std = sample_std(values);
    if !std.is_finite() || is_close_zero(std) {
        return f64::NAN;
    }
    mean(values) / std
}

fn positive_rate(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().filter(|value| **value > 0.0).count() as f64 / values.len() as f64
}

fn pearson_corr(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 2 || ys.len() < 2 || xs.len() != ys.len() {
        return f64::NAN;
    }
    let x_mean = mean(xs);
    let y_mean = mean(ys);
    let mut xy = 0.0;
    let mut xx = 0.0;
    let mut yy = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let xc = *x - x_mean;
        let yc = *y - y_mean;
        xy += xc * yc;
        xx += xc * xc;
        yy += yc * yc;
    }
    if !xx.is_finite() || !yy.is_finite() || xx <= 0.0 || yy <= 0.0 {
        return f64::NAN;
    }
    xy / (xx * yy).sqrt()
}

fn spearman_corr(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() < 2 || ys.len() < 2 || xs.len() != ys.len() {
        return f64::NAN;
    }
    let x_ranks = rank_average(xs);
    let y_ranks = rank_average(ys);
    pearson_corr(&x_ranks, &y_ranks)
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

fn rank_to_quantile_bins(values: &[f64], quantile_bins: usize) -> Vec<usize> {
    let ranks = rank_average(values);
    let denominator = values.len() as f64;
    ranks
        .iter()
        .map(|rank| {
            let raw = ((rank / denominator) * quantile_bins as f64 - 1e-12).floor() as isize;
            raw.clamp(0, quantile_bins.saturating_sub(1) as isize) as usize
        })
        .collect()
}

fn bucket_means(
    bucket: &[usize],
    labels: &[f64],
    quantile_bins: usize,
) -> (Vec<usize>, Vec<f64>, Vec<usize>) {
    let mut sums = vec![0.0; quantile_bins];
    let mut counts = vec![0usize; quantile_bins];
    for (bucket_id, label) in bucket.iter().zip(labels.iter()) {
        sums[*bucket_id] += *label;
        counts[*bucket_id] += 1;
    }
    let mut indices = Vec::new();
    let mut means = Vec::new();
    let mut populated_counts = Vec::new();
    for bucket_id in 0..quantile_bins {
        if counts[bucket_id] > 0 {
            indices.push(bucket_id);
            means.push(sums[bucket_id] / counts[bucket_id] as f64);
            populated_counts.push(counts[bucket_id]);
        }
    }
    (indices, means, populated_counts)
}

fn unique_count(values: &[f64]) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let mut count = 0usize;
    let mut previous: Option<f64> = None;
    for value in sorted {
        if previous.is_none_or(|prev| value != prev) {
            count += 1;
            previous = Some(value);
        }
    }
    count
}

fn monthly_means(dates_ns: &[i64], values: &[f64]) -> BTreeMap<i64, f64> {
    let mut grouped: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (date_ns, value) in dates_ns.iter().zip(values.iter()) {
        grouped
            .entry(month_end_ns(*date_ns))
            .or_default()
            .push(*value);
    }
    grouped
        .into_iter()
        .map(|(month, values)| (month, mean(&values)))
        .collect()
}

fn sanitize_label(value: f64) -> f64 {
    if !value.is_finite() || value.abs() > DEFAULT_LABEL_ABS_CAP {
        f64::NAN
    } else {
        value
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

fn format_date_ns(value: i64) -> String {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .map(|datetime| datetime.date_naive().to_string())
        .unwrap_or_default()
}

fn year_from_ns(value: i64) -> i32 {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .map(|datetime| datetime.year())
        .unwrap_or_default()
}

fn month_end_ns(value: i64) -> i64 {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    let Some(datetime) = DateTime::<Utc>::from_timestamp(secs, nanos) else {
        return value;
    };
    let date = datetime.date_naive();
    let (next_year, next_month) = if date.month() == 12 {
        (date.year() + 1, 1)
    } else {
        (date.year(), date.month() + 1)
    };
    let Some(first_next_month) = NaiveDate::from_ymd_opt(next_year, next_month, 1) else {
        return value;
    };
    first_next_month
        .pred_opt()
        .and_then(|month_end| month_end.and_hms_opt(0, 0, 0))
        .and_then(|datetime| datetime.and_utc().timestamp_nanos_opt())
        .unwrap_or(value)
}

fn normalize_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect()
}

fn normalize_local_symbol(symbol: &str) -> String {
    let text = symbol.trim();
    if text.chars().all(|ch| ch.is_ascii_digit()) && text.len() <= 6 {
        format!("{text:0>6}")
    } else {
        text.to_owned()
    }
}

fn infer_feature_group(feature_name: &str) -> &'static str {
    if let Some(raw) = feature_name.strip_prefix("TS_") {
        if raw.starts_with("sem_") {
            return "tushare_semantic_alpha";
        }
        if raw.starts_with("industry_")
            || raw.starts_with("stock_vs_industry")
            || raw.starts_with("stock_industry")
            || raw.starts_with("stock_relative_strength")
        {
            return "tushare_industry";
        }
        if raw.starts_with("fi_")
            || raw.starts_with("latest_eps")
            || raw.starts_with("latest_dt_eps")
            || raw.starts_with("latest_bps")
            || raw.starts_with("latest_ocfps")
            || raw.starts_with("latest_roe")
            || raw.starts_with("latest_roa")
            || raw.starts_with("latest_grossprofit")
            || raw.starts_with("latest_netprofit")
            || raw.starts_with("latest_debt")
            || raw.starts_with("latest_q_")
            || raw.starts_with("latest_tr_")
            || raw.starts_with("latest_or_")
            || raw.starts_with("latest_op_")
            || raw.starts_with("latest_netprofit_yoy")
            || raw.starts_with("latest_ocf")
        {
            return "tushare_financial_quality";
        }
        if raw.starts_with("fc_")
            || raw.starts_with("exp_")
            || raw.starts_with("latest_fc")
            || raw.starts_with("latest_exp")
        {
            return "tushare_event_forecast_express";
        }
        if raw.contains("limit")
            || raw.starts_with("gap_")
            || raw.starts_with("hit_")
            || raw.starts_with("near_")
            || raw.starts_with("days_since_last")
            || raw.starts_with("up_limit")
            || raw.starts_with("down_limit")
        {
            return "tushare_limit_state";
        }
        if raw.contains("turnover")
            || raw.contains("amihud")
            || raw.starts_with("free_")
            || raw.starts_with("volume_ratio")
            || raw.starts_with("downside_")
        {
            return "tushare_flow_liquidity";
        }
        if raw.starts_with("ep")
            || raw.starts_with("sp")
            || raw.starts_with("bp")
            || raw.starts_with("dividend")
            || raw.starts_with("latest_div")
            || raw.starts_with("has_dividend")
            || raw.starts_with("has_stock_dividend")
        {
            return "tushare_valuation_dividend";
        }
        return "tushare_other";
    }
    if feature_name.starts_with("LGBM_") {
        return "lgbm_purified";
    }
    if feature_name.starts_with("TEMP_") {
        return "temporal";
    }
    if feature_name.starts_with("TECH_") {
        return "technical";
    }
    if feature_name.starts_with("A360_") {
        return "alpha360";
    }
    "alpha158"
}

fn is_close_zero(value: f64) -> bool {
    value.abs() <= 1e-8
}

fn nanmax(left: f64, right: f64) -> f64 {
    if !left.is_finite() {
        right
    } else if !right.is_finite() {
        left
    } else {
        left.max(right)
    }
}

fn nanmin(left: f64, right: f64) -> f64 {
    if !left.is_finite() {
        right
    } else if !right.is_finite() {
        left
    } else {
        left.min(right)
    }
}

fn cmp_f64(left: f64, right: f64) -> Ordering {
    match (left.is_finite(), right.is_finite()) {
        (true, true) => left.total_cmp(&right),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => Ordering::Equal,
    }
}

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    cmp_f64(left, right).reverse()
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> DiagnosticContext {
        let dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"];
        let symbols = ["A", "B", "C", "D"];
        let labels = [("A", 0.04), ("B", 0.02), ("C", -0.01), ("D", -0.03)]
            .into_iter()
            .collect::<HashMap<_, _>>();
        let signal = [("A", 4.0), ("B", 3.0), ("C", 2.0), ("D", 1.0)]
            .into_iter()
            .collect::<HashMap<_, _>>();
        let inverse = [("A", 1.0), ("B", 2.0), ("C", 3.0), ("D", 4.0)]
            .into_iter()
            .collect::<HashMap<_, _>>();
        let mut rows = Vec::new();
        for date in dates {
            for symbol in symbols {
                rows.push(DiagnosticRow {
                    date_ns: parse_date_ns(date).unwrap(),
                    symbol: symbol.to_owned(),
                    label: *labels.get(symbol).unwrap(),
                    industry: Some(
                        if symbol == "A" || symbol == "B" {
                            "I1"
                        } else {
                            "I2"
                        }
                        .to_owned(),
                    ),
                    features: vec![
                        *signal.get(symbol).unwrap(),
                        *inverse.get(symbol).unwrap(),
                        1.0,
                    ],
                });
            }
        }
        rows.sort_by(|left, right| {
            left.date_ns
                .cmp(&right.date_ns)
                .then(left.symbol.cmp(&right.symbol))
        });
        let mut context = DiagnosticContext {
            rows,
            date_slices: Vec::new(),
        };
        rebuild_date_slices(&mut context);
        context
    }

    #[test]
    fn detects_positive_and_negative_rank_ic_direction() {
        let context = test_context();
        let features = vec!["signal".to_owned(), "inverse".to_owned(), "flat".to_owned()];
        let rows = build_summary_rows(&context, &features, 2);
        let by_feature = rows
            .into_iter()
            .map(|row| (row.feature.clone(), row))
            .collect::<HashMap<_, _>>();
        assert!(by_feature["signal"].rank_ic_mean > 0.99);
        assert_eq!(by_feature["signal"].suggested_direction, 1);
        assert!(by_feature["inverse"].rank_ic_mean < -0.99);
        assert_eq!(by_feature["inverse"].suggested_direction, -1);
        assert_eq!(by_feature["flat"].effective_date_count, 0);
        assert!(!by_feature["flat"].rank_ic_mean.is_finite());
    }

    #[test]
    fn detects_segment_direction_flip() {
        let mut context = test_context();
        for row in context.rows.iter_mut() {
            if row.date_ns >= parse_date_ns("2024-01-04").unwrap() {
                row.features[0] = 5.0 - row.features[0];
            }
        }
        let features = vec!["signal".to_owned()];
        let segments = vec![
            SegmentSpec {
                name: "seg_a".to_owned(),
                start_ns: parse_date_ns("2024-01-01").unwrap(),
                end_ns: parse_date_ns("2024-01-03").unwrap(),
            },
            SegmentSpec {
                name: "seg_b".to_owned(),
                start_ns: parse_date_ns("2024-01-04").unwrap(),
                end_ns: parse_date_ns("2024-01-31").unwrap(),
            },
        ];
        let mut segment_rows = BTreeMap::new();
        for segment in &segments {
            segment_rows.insert(
                segment.name.clone(),
                build_segment_summary_rows(&context, &features, segment, 2),
            );
        }
        let comparison = build_segment_comparison(&segment_rows, &segments);
        assert_eq!(comparison.len(), 1);
        assert_eq!(
            comparison[0].values.get("direction_flip"),
            Some(&"true".to_owned())
        );
        assert_eq!(
            comparison[0].values.get("suggested_direction__seg_a"),
            Some(&"1".to_owned())
        );
        assert_eq!(
            comparison[0].values.get("suggested_direction__seg_b"),
            Some(&"-1".to_owned())
        );
    }

    #[test]
    fn industry_excess_labels_are_point_in_time_cross_sectional() {
        let mut context = test_context();
        apply_diagnostic_label_space(
            &mut context.rows,
            &DiagnosticLabelSpace::IndustryExcess,
            0.0,
            1,
            &BenchmarkOptions::default(),
        )
        .unwrap();
        let first_day = context
            .rows
            .iter()
            .filter(|row| row.date_ns == parse_date_ns("2024-01-02").unwrap())
            .collect::<Vec<_>>();
        assert!((first_day[0].label - 0.01).abs() < 1e-12);
        assert!((first_day[1].label + 0.01).abs() < 1e-12);
    }

    #[test]
    fn benchmark_excess_uses_forward_compound_returns() {
        let mut context = test_context();
        apply_diagnostic_label_space(
            &mut context.rows,
            &DiagnosticLabelSpace::BenchmarkExcess,
            0.001,
            2,
            &BenchmarkOptions::default(),
        )
        .unwrap();
        let first_date = parse_date_ns("2024-01-02").unwrap();
        let first_a = context
            .rows
            .iter()
            .find(|row| row.date_ns == first_date && row.symbol == "A")
            .unwrap();
        let expected_forward = (1.0_f64 + 0.005).powi(2) - 1.0;
        assert!((first_a.label - (0.04 - expected_forward - 0.001)).abs() < 1e-12);
        let late_date = parse_date_ns("2024-01-04").unwrap();
        assert!(context
            .rows
            .iter()
            .filter(|row| row.date_ns >= late_date)
            .all(|row| !row.label.is_finite()));
    }

    #[test]
    fn drops_non_finite_labels_before_summary_like_python_entrypoint() {
        let mut context = test_context();
        context.rows[0].label = f64::NAN;
        context.rows.retain(|row| row.label.is_finite());
        rebuild_date_slices(&mut context);
        let rows = build_summary_rows(&context, &["signal".to_owned()], 2);
        assert_eq!(rows[0].observation_count, 15);
        assert_eq!(rows[0].valid_observation_count, 15);
        assert_eq!(rows[0].coverage_pct, 1.0);
    }
}
