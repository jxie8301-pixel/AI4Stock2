use crate::factor_kernels;
use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, NaiveDate, Utc};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::file::reader::{FileReader, SerializedFileReader};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ParquetShardStats {
    pub path: String,
    pub rows: i64,
    pub columns: usize,
    pub row_groups: usize,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ParquetLayoutSummary {
    pub root: String,
    pub bucket_root: String,
    pub file_count: usize,
    pub total_rows: i64,
    pub total_row_groups: usize,
    pub total_size_bytes: u64,
    pub min_rows: i64,
    pub median_rows: f64,
    pub max_rows: i64,
    pub min_columns: usize,
    pub median_columns: f64,
    pub max_columns: usize,
    pub min_row_groups: usize,
    pub median_row_groups: f64,
    pub max_row_groups: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MissingColumnsByPath {
    pub path: String,
    pub missing_columns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RequiredColumnsValidation {
    pub root: String,
    pub bucket_root: String,
    pub validated: bool,
    pub file_count: usize,
    pub required_columns_count: usize,
    pub missing_file_count: usize,
    pub missing_by_path: Vec<MissingColumnsByPath>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SourceBucketScanSummary {
    pub path: String,
    pub selected_columns: Vec<String>,
    pub batch_size: usize,
    pub batch_count: usize,
    pub row_count: usize,
    pub symbol_count: usize,
}

pub fn discover_bucket_parquet_paths(root: &Path) -> Result<(PathBuf, Vec<PathBuf>), String> {
    let bucket_root = if root.join("buckets").is_dir() {
        root.join("buckets")
    } else {
        root.to_path_buf()
    };
    let mut paths = Vec::new();
    let entries = fs::read_dir(&bucket_root)
        .map_err(|err| format!("failed to read {}: {err}", bucket_root.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read directory entry: {err}"))?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if file_name.starts_with("part-") && file_name.ends_with(".parquet") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok((bucket_root, paths))
}

pub fn read_parquet_schema_columns(path: &Path) -> Result<BTreeSet<String>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|err| format!("failed to read parquet metadata {}: {err}", path.display()))?;
    Ok(reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .map(|column| column.name().to_owned())
        .collect())
}

pub fn read_parquet_shard_stats(path: &Path) -> Result<ParquetShardStats, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .map_err(|err| format!("failed to read parquet metadata {}: {err}", path.display()))?;
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();
    let size_bytes = fs::metadata(path)
        .map_err(|err| format!("failed to stat {}: {err}", path.display()))?
        .len();
    Ok(ParquetShardStats {
        path: path.display().to_string(),
        rows: file_metadata.num_rows(),
        columns: file_metadata.schema_descr().num_columns(),
        row_groups: metadata.num_row_groups(),
        size_bytes,
    })
}

pub fn validate_required_columns(
    root: &Path,
    required_columns: &[String],
) -> Result<RequiredColumnsValidation, String> {
    let (bucket_root, paths) = discover_bucket_parquet_paths(root)?;
    if paths.is_empty() {
        return Err(format!(
            "no part-*.parquet bucket shards found under {}",
            bucket_root.display()
        ));
    }
    validate_required_columns_for_paths(root, &bucket_root, &paths, required_columns)
}

pub fn validate_required_columns_for_paths(
    root: &Path,
    bucket_root: &Path,
    paths: &[PathBuf],
    required_columns: &[String],
) -> Result<RequiredColumnsValidation, String> {
    if paths.is_empty() {
        return Err("no parquet bucket shard paths were provided".to_owned());
    }
    let required: BTreeSet<String> = required_columns
        .iter()
        .map(|value| value.to_owned())
        .collect();
    let mut missing_by_path = Vec::new();
    for path in paths {
        let schema_columns = read_parquet_schema_columns(path)?;
        let missing_columns = required
            .difference(&schema_columns)
            .map(|value| value.to_owned())
            .collect::<Vec<_>>();
        if !missing_columns.is_empty() {
            missing_by_path.push(MissingColumnsByPath {
                path: path.display().to_string(),
                missing_columns,
            });
        }
    }
    Ok(RequiredColumnsValidation {
        root: root.display().to_string(),
        bucket_root: bucket_root.display().to_string(),
        validated: missing_by_path.is_empty(),
        file_count: paths.len(),
        required_columns_count: required.len(),
        missing_file_count: missing_by_path.len(),
        missing_by_path,
    })
}

pub fn inspect_parquet_layout(root: &Path) -> Result<ParquetLayoutSummary, String> {
    let (bucket_root, paths) = discover_bucket_parquet_paths(root)?;
    if paths.is_empty() {
        return Err(format!(
            "no part-*.parquet bucket shards found under {}",
            bucket_root.display()
        ));
    }
    let mut stats = Vec::with_capacity(paths.len());
    for path in &paths {
        stats.push(read_parquet_shard_stats(path)?);
    }
    Ok(summarize_parquet_shards(root, &bucket_root, &stats))
}

pub fn scan_source_bucket(
    path: &Path,
    columns: &[String],
    batch_size: usize,
) -> Result<SourceBucketScanSummary, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to create parquet reader {}: {err}", path.display()))?
        .with_batch_size(batch_size.max(1));
    let schema_descr = builder.metadata().file_metadata().schema_descr_ptr();
    let mut selected_columns = Vec::new();
    if !columns.is_empty() {
        builder = builder.with_projection(ProjectionMask::columns(
            &schema_descr,
            columns.iter().map(String::as_str),
        ));
        selected_columns.extend(columns.iter().cloned());
    } else {
        selected_columns.extend(
            builder
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().to_owned()),
        );
    }
    let reader = builder
        .build()
        .map_err(|err| format!("failed to build parquet reader {}: {err}", path.display()))?;
    let mut row_count = 0usize;
    let mut batch_count = 0usize;
    let mut symbols = BTreeSet::new();
    for batch in reader {
        let batch = batch
            .map_err(|err| format!("failed to read record batch {}: {err}", path.display()))?;
        row_count += batch.num_rows();
        batch_count += 1;
        collect_symbols(&batch, &mut symbols)?;
    }
    Ok(SourceBucketScanSummary {
        path: path.display().to_string(),
        selected_columns,
        batch_size: batch_size.max(1),
        batch_count,
        row_count,
        symbol_count: symbols.len(),
    })
}

fn collect_symbols(batch: &RecordBatch, symbols: &mut BTreeSet<String>) -> Result<(), String> {
    let Ok(index) = batch.schema().index_of("symbol") else {
        return Ok(());
    };
    let column = batch.column(index);
    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
        for idx in 0..array.len() {
            if array.is_valid(idx) {
                symbols.insert(array.value(idx).to_owned());
            }
        }
        return Ok(());
    }
    if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
        for idx in 0..array.len() {
            if array.is_valid(idx) {
                symbols.insert(array.value(idx).to_owned());
            }
        }
        return Ok(());
    }
    Err("symbol column is not a UTF-8 string array".to_owned())
}

const LABEL_ABS_CAP: f64 = 0.35;
const TUSHARE_EVENT_AVAILABILITY_POLICY: &str = "strict_next_trading_day_after_ann_date";

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub parquet_dir: PathBuf,
    pub output_dir: PathBuf,
    pub data_source: String,
    pub workers: usize,
    pub label_horizons: Vec<usize>,
    pub batch_size: usize,
    pub bucket_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GenerateSummary {
    pub storage_format: String,
    pub storage_layout: String,
    pub generator: String,
    pub data_source: String,
    pub source_parquet_dir: String,
    pub factor_store_dir: String,
    pub buckets_dir: String,
    pub manifest_path: String,
    pub bucket_count: usize,
    pub bucket_ids: Vec<i32>,
    pub num_rows: usize,
    pub num_features: usize,
    pub feature_names: Vec<String>,
    pub label_columns: Vec<String>,
    pub available_dates: Vec<String>,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ManifestRow {
    pub symbol: String,
    pub bucket_id: i32,
    pub source_path: String,
    pub source_size: i64,
    pub source_mtime_ns: i64,
    pub row_count: i64,
    pub min_date: String,
    pub max_date: String,
    pub feature_count: i32,
    pub label_columns: String,
}

#[derive(Debug)]
struct SourceBucketResult {
    bucket_id: i32,
    row_count: usize,
    manifest_rows: Vec<ManifestRow>,
    feature_names: Vec<String>,
    available_dates: BTreeSet<String>,
}

#[derive(Debug, Default)]
struct SymbolRows {
    dates_ns: Vec<i64>,
    columns: HashMap<String, Vec<f64>>,
    all_numeric_float32: bool,
}

#[derive(Debug)]
struct SymbolFactorFrame {
    symbol: String,
    dates_ns: Vec<i64>,
    label_columns: Vec<(String, Vec<f32>)>,
    feature_columns: Vec<(String, Vec<f32>)>,
}

pub fn generate_factor_store(options: &GenerateOptions) -> Result<GenerateSummary, String> {
    let started = Instant::now();
    let (_, mut source_paths) = discover_bucket_parquet_paths(&options.parquet_dir)?;
    if source_paths.is_empty() {
        return Err(format!(
            "no part-*.parquet source buckets found under {}",
            options.parquet_dir.display()
        ));
    }
    if let Some(limit) = options.bucket_limit {
        source_paths.truncate(limit);
    }
    validate_standalone_source_policy(options)?;

    let bucket_root = options.output_dir.join("buckets");
    fs::create_dir_all(&bucket_root)
        .map_err(|err| format!("failed to create {}: {err}", bucket_root.display()))?;

    let worker_count = options.workers.max(1);
    println!(
        "[1/3] Rust gen_feature generating {} bucket shard(s), workers={worker_count}",
        source_paths.len()
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|err| format!("failed to build rayon thread pool: {err}"))?;
    let mut results = pool.install(|| {
        source_paths
            .par_iter()
            .map(|path| write_factor_bucket_from_source_bucket(path, &bucket_root, options))
            .collect::<Vec<_>>()
    });
    let mut bucket_results = Vec::with_capacity(results.len());
    for result in results.drain(..) {
        bucket_results.push(result?);
    }
    bucket_results.sort_by_key(|result| result.bucket_id);

    let mut manifest_rows = Vec::new();
    let mut bucket_ids = Vec::new();
    let mut available_dates = BTreeSet::new();
    let mut num_rows = 0usize;
    let mut feature_names = Vec::new();
    for result in bucket_results {
        bucket_ids.push(result.bucket_id);
        num_rows += result.row_count;
        if feature_names.is_empty() {
            feature_names = result.feature_names;
        }
        available_dates.extend(result.available_dates);
        manifest_rows.extend(result.manifest_rows);
    }
    if manifest_rows.is_empty() {
        return Err("Rust gen_feature produced no rows".to_owned());
    }
    manifest_rows.sort_by(|left, right| {
        left.bucket_id
            .cmp(&right.bucket_id)
            .then(left.symbol.cmp(&right.symbol))
    });
    let label_columns = label_column_names(&options.label_horizons);
    let manifest_path = options.output_dir.join("manifest.parquet");
    println!("[2/3] Writing Rust manifest and metadata");
    write_manifest(&manifest_path, &manifest_rows)?;

    let summary = GenerateSummary {
        storage_format: "parquet".to_owned(),
        storage_layout: "bucket_shards".to_owned(),
        generator: "rust".to_owned(),
        data_source: options.data_source.clone(),
        source_parquet_dir: options.parquet_dir.display().to_string(),
        factor_store_dir: options.output_dir.display().to_string(),
        buckets_dir: bucket_root.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        bucket_count: bucket_ids.len(),
        bucket_ids,
        num_rows,
        num_features: feature_names.len(),
        feature_names,
        label_columns,
        available_dates: available_dates.into_iter().collect(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
    };
    write_meta(&options.output_dir.join("meta.json"), &summary)?;
    println!(
        "[3/3] Done. Rust factor store saved to: {}",
        options.output_dir.display()
    );
    Ok(summary)
}

fn validate_standalone_source_policy(options: &GenerateOptions) -> Result<(), String> {
    if options.data_source != "tushare" {
        return Ok(());
    }
    let meta_path = options.parquet_dir.join("meta.json");
    let raw = fs::read_to_string(&meta_path).map_err(|err| {
        format!(
            "failed to read Tushare source metadata {}: {err}",
            meta_path.display()
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse Tushare source metadata {}: {err}",
            meta_path.display()
        )
    })?;
    let policy = value
        .get("source_layout_assumptions")
        .and_then(|item| item.get("tushare_event_availability_policy"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if policy != TUSHARE_EVENT_AVAILABILITY_POLICY {
        return Err(format!(
            "Tushare source metadata event policy mismatch: expected {TUSHARE_EVENT_AVAILABILITY_POLICY}, got {policy:?}. Rebuild the source store first."
        ));
    }
    Ok(())
}

fn write_factor_bucket_from_source_bucket(
    source_path: &Path,
    bucket_root: &Path,
    options: &GenerateOptions,
) -> Result<SourceBucketResult, String> {
    let bucket_id = extract_bucket_id(source_path)?;
    let out_path = bucket_root.join(format!("part-{bucket_id:04}.parquet"));
    let source_stat = fs::metadata(source_path)
        .map_err(|err| format!("failed to stat {}: {err}", source_path.display()))?;
    let source_size = source_stat.len() as i64;
    let source_mtime_ns = source_mtime_ns(&source_stat);
    let mut symbols = read_source_bucket_symbols(source_path, options.batch_size)?;
    let mut writer: Option<ArrowWriter<File>> = None;
    let mut manifest_rows = Vec::new();
    let mut total_rows = 0usize;
    let mut feature_names = Vec::new();
    let mut available_dates = BTreeSet::new();
    let label_columns = label_column_names(&options.label_horizons);
    let label_columns_joined = label_columns.join(",");

    for (symbol, rows) in symbols.iter_mut() {
        sort_symbol_rows(rows);
        ensure_vwap(rows);
        let frame = build_symbol_factor_frame(symbol, rows, options)?;
        if feature_names.is_empty() {
            feature_names = frame
                .feature_columns
                .iter()
                .map(|(name, _)| name.clone())
                .collect();
        }
        let batch = symbol_frame_to_record_batch(&frame)?;
        if writer.is_none() {
            let file = File::create(&out_path)
                .map_err(|err| format!("failed to create {}: {err}", out_path.display()))?;
            writer = Some(
                ArrowWriter::try_new(file, batch.schema(), None).map_err(|err| {
                    format!(
                        "failed to create parquet writer {}: {err}",
                        out_path.display()
                    )
                })?,
            );
        }
        writer
            .as_mut()
            .expect("writer initialized")
            .write(&batch)
            .map_err(|err| format!("failed to write {}: {err}", out_path.display()))?;
        let min_date = frame
            .dates_ns
            .first()
            .map(|value| format_date_ns(*value))
            .unwrap_or_default();
        let max_date = frame
            .dates_ns
            .last()
            .map(|value| format_date_ns(*value))
            .unwrap_or_default();
        available_dates.extend(frame.dates_ns.iter().map(|value| format_date_ns(*value)));
        total_rows += frame.dates_ns.len();
        manifest_rows.push(ManifestRow {
            symbol: symbol.clone(),
            bucket_id,
            source_path: source_path
                .canonicalize()
                .unwrap_or_else(|_| source_path.to_path_buf())
                .display()
                .to_string(),
            source_size,
            source_mtime_ns,
            row_count: frame.dates_ns.len() as i64,
            min_date,
            max_date,
            feature_count: frame.feature_columns.len() as i32,
            label_columns: label_columns_joined.clone(),
        });
    }
    if let Some(writer) = writer {
        writer
            .close()
            .map_err(|err| format!("failed to close {}: {err}", out_path.display()))?;
    }
    Ok(SourceBucketResult {
        bucket_id,
        row_count: total_rows,
        manifest_rows,
        feature_names,
        available_dates,
    })
}

fn read_source_bucket_symbols(
    source_path: &Path,
    batch_size: usize,
) -> Result<BTreeMap<String, SymbolRows>, String> {
    let file = File::open(source_path)
        .map_err(|err| format!("failed to open {}: {err}", source_path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| {
            format!(
                "failed to create parquet reader {}: {err}",
                source_path.display()
            )
        })?
        .with_batch_size(batch_size.max(1))
        .build()
        .map_err(|err| {
            format!(
                "failed to build parquet reader {}: {err}",
                source_path.display()
            )
        })?;
    let mut symbols: BTreeMap<String, SymbolRows> = BTreeMap::new();
    for batch in reader {
        let batch = batch.map_err(|err| {
            format!(
                "failed to read record batch {}: {err}",
                source_path.display()
            )
        })?;
        append_source_batch(&mut symbols, &batch)?;
    }
    Ok(symbols)
}

fn append_source_batch(
    symbols: &mut BTreeMap<String, SymbolRows>,
    batch: &RecordBatch,
) -> Result<(), String> {
    let schema = batch.schema();
    let symbol_idx = schema
        .index_of("symbol")
        .map_err(|_| "source bucket is missing symbol column".to_owned())?;
    let date_idx = schema
        .index_of("date")
        .map_err(|_| "source bucket is missing date column".to_owned())?;
    let symbol_array = batch.column(symbol_idx);
    let date_array = batch.column(date_idx);
    let numeric_columns = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(idx, field)| {
            let name = field.name();
            if name == "date" || name == "symbol" || name == "ts_code" {
                return None;
            }
            numeric_values(batch.column(idx).as_ref()).map(|values| {
                (
                    name.to_owned(),
                    values,
                    matches!(field.data_type(), DataType::Float32),
                )
            })
        })
        .collect::<Vec<_>>();

    for row in 0..batch.num_rows() {
        let Some(symbol) = string_value(symbol_array.as_ref(), row) else {
            continue;
        };
        let Some(date_ns) = date_value_ns(date_array.as_ref(), row) else {
            continue;
        };
        let entry = symbols.entry(symbol).or_insert_with(|| SymbolRows {
            dates_ns: Vec::new(),
            columns: HashMap::new(),
            all_numeric_float32: true,
        });
        entry.dates_ns.push(date_ns);
        for (name, values, is_float32) in &numeric_columns {
            entry.all_numeric_float32 &= *is_float32;
            entry
                .columns
                .entry(name.clone())
                .or_default()
                .push(values[row]);
        }
    }
    Ok(())
}

fn build_symbol_factor_frame(
    symbol: &str,
    rows: &SymbolRows,
    options: &GenerateOptions,
) -> Result<SymbolFactorFrame, String> {
    let mut feature_columns = Vec::new();
    let open = column_or_nan(rows, "open");
    let high = column_or_nan(rows, "high");
    let low = column_or_nan(rows, "low");
    let close = column_or_nan(rows, "close");
    let volume = column_or_nan(rows, "volume");
    let amount = column_or_nan(rows, "amount");
    let vwap = column_or_nan(rows, "vwap");
    append_prefixed(
        &mut feature_columns,
        "",
        factor_kernels::alpha158_features(
            &open,
            &high,
            &low,
            &close,
            &vwap,
            &volume,
            true,
            &[
                "OPEN".to_owned(),
                "HIGH".to_owned(),
                "LOW".to_owned(),
                "VWAP".to_owned(),
            ],
            &[0],
            &[],
            &[5, 10, 20, 30, 60],
            &alpha158_rolling_ops(),
        )?,
    );
    append_prefixed(
        &mut feature_columns,
        "LGBM_",
        factor_kernels::lgbm_purified_features(
            &high,
            &low,
            &close,
            &volume,
            &amount,
            &vwap,
            &column_or_nan(rows, "circ_mv"),
            &column_or_nan(rows, "pe_ttm"),
            &column_or_nan(rows, "pb"),
            &column_or_nan(rows, "turnover"),
            rows.columns.contains_key("pe_ttm"),
            rows.columns.contains_key("pb"),
            &[20, 60],
            &[20, 60, 120],
            60,
            14,
            20,
            20,
            20,
            20,
            20,
        )?,
    );
    append_prefixed(
        &mut feature_columns,
        "TEMP_",
        factor_kernels::temporal_factor_features(
            &close,
            &high,
            &low,
            &volume,
            &amount,
            &column_or_nan(rows, "turnover"),
            &[1, 5, 10, 20, 30, 60, 120],
            &[
                "ret".to_owned(),
                "ma_gap".to_owned(),
                "std".to_owned(),
                "rsv".to_owned(),
                "price_rank".to_owned(),
                "volume_ratio".to_owned(),
                "turnover_mean".to_owned(),
                "amihud".to_owned(),
                "high_gap".to_owned(),
                "low_gap".to_owned(),
                "corr_cv".to_owned(),
            ],
        )?,
    );
    append_prefixed(
        &mut feature_columns,
        "TECH_",
        factor_kernels::technical_factor_features(
            &high,
            &low,
            &close,
            &volume,
            &[12],
            &[26],
            &[9],
            &[6, 14, 24],
            &[20, 60],
            2.0,
            &[14],
            &[14],
            &[20],
            &[14],
            &[25],
            &[15],
            9,
            &[20, 60],
        )?,
    );
    if options.data_source == "tushare" {
        append_prefixed(
            &mut feature_columns,
            "TS_",
            factor_kernels::tushare_factor_features(
                &rows.columns,
                &[5, 20],
                &[5, 20],
                &[5, 20],
                &[5, 20],
                &[20, 60],
                &[20, 60],
                &[5, 20, 60],
                &[20, 60],
                20,
                rows.all_numeric_float32,
            )?,
        );
    }
    let feature_columns = deduplicate_exact_feature_columns(feature_columns);
    let labels = build_labels(rows, &options.label_horizons);
    Ok(SymbolFactorFrame {
        symbol: symbol.to_owned(),
        dates_ns: rows.dates_ns.clone(),
        label_columns: labels,
        feature_columns: feature_columns
            .into_iter()
            .map(|(name, values)| {
                (
                    name,
                    values
                        .into_iter()
                        .map(|value| value as f32)
                        .collect::<Vec<_>>(),
                )
            })
            .collect(),
    })
}

fn append_prefixed(
    out: &mut Vec<(String, Vec<f64>)>,
    prefix: &str,
    values: Vec<(String, Vec<f64>)>,
) {
    out.extend(
        values
            .into_iter()
            .map(|(name, values)| (format!("{prefix}{name}"), values)),
    );
}

fn deduplicate_exact_feature_columns(
    feature_columns: Vec<(String, Vec<f64>)>,
) -> Vec<(String, Vec<f64>)> {
    let source_map = exact_duplicate_feature_source_map();
    let mut seen_sources = HashSet::new();
    let mut deduped = Vec::with_capacity(feature_columns.len());
    for (name, values) in feature_columns {
        let source_name = source_map
            .get(&name)
            .map(String::as_str)
            .unwrap_or(name.as_str());
        if seen_sources.insert(source_name.to_owned()) {
            deduped.push((name, values));
        }
    }
    deduped
}

fn exact_duplicate_feature_source_map() -> HashMap<String, String> {
    let mut source_map = HashMap::new();
    for group in known_exact_duplicate_feature_groups() {
        let canonical = group[0].clone();
        source_map.insert(canonical.clone(), canonical.clone());
        for feature_name in group.into_iter().skip(1) {
            source_map.insert(feature_name, canonical.clone());
        }
    }
    source_map
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

fn alpha158_rolling_ops() -> Vec<String> {
    [
        "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "LOW", "QTLU", "QTLD", "RANK", "RSV",
        "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
        "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
    ]
    .iter()
    .map(|value| (*value).to_owned())
    .collect()
}

fn build_labels(rows: &SymbolRows, horizons: &[usize]) -> Vec<(String, Vec<f32>)> {
    let label_1d = build_open_to_open_label(rows, 1);
    let mut labels = vec![("label".to_owned(), label_1d)];
    for &horizon in horizons {
        labels.push((
            label_column_name(horizon),
            build_open_to_open_label(rows, horizon),
        ));
    }
    labels
}

fn build_open_to_open_label(rows: &SymbolRows, horizon: usize) -> Vec<f32> {
    let len = rows.dates_ns.len();
    let open = column_or_nan(rows, "open");
    let volume = rows.columns.get("volume");
    let amount = rows.columns.get("amount");
    let mut out = vec![f32::NAN; len];
    for (idx, output) in out.iter_mut().enumerate().take(len) {
        let next_idx = idx + 1;
        let exit_idx = idx + 1 + horizon;
        if exit_idx >= len {
            continue;
        }
        let next_open = open[next_idx];
        let exit_open = open[exit_idx];
        if !next_open.is_finite() || !exit_open.is_finite() || next_open <= 0.0 || exit_open <= 0.0
        {
            continue;
        }
        if let Some(values) = volume {
            let next = values[next_idx];
            let exit = values[exit_idx];
            if !next.is_finite() || !exit.is_finite() || next <= 0.0 || exit <= 0.0 {
                continue;
            }
        }
        if let Some(values) = amount {
            let next = values[next_idx];
            let exit = values[exit_idx];
            if !next.is_finite() || !exit.is_finite() || next <= 0.0 || exit <= 0.0 {
                continue;
            }
        }
        let label = exit_open / next_open - 1.0;
        if label.is_finite() && label.abs() <= LABEL_ABS_CAP {
            *output = label as f32;
        }
    }
    out
}

fn symbol_frame_to_record_batch(frame: &SymbolFactorFrame) -> Result<RecordBatch, String> {
    let rows = frame.dates_ns.len();
    let mut fields = vec![
        Field::new(
            "date",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
    ];
    fields.extend(
        frame
            .label_columns
            .iter()
            .map(|(name, _)| Field::new(name, DataType::Float32, true)),
    );
    fields.extend(
        frame
            .feature_columns
            .iter()
            .map(|(name, _)| Field::new(name, DataType::Float32, true)),
    );
    let schema = Arc::new(Schema::new(fields));
    let mut arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(TimestampNanosecondArray::from(frame.dates_ns.clone())),
        Arc::new(StringArray::from(vec![frame.symbol.clone(); rows])),
    ];
    arrays.extend(
        frame
            .label_columns
            .iter()
            .map(|(_, values)| Arc::new(Float32Array::from(values.clone())) as Arc<dyn Array>),
    );
    arrays.extend(
        frame
            .feature_columns
            .iter()
            .map(|(_, values)| Arc::new(Float32Array::from(values.clone())) as Arc<dyn Array>),
    );
    RecordBatch::try_new(schema, arrays)
        .map_err(|err| format!("failed to build record batch: {err}"))
}

fn write_manifest(path: &Path, rows: &[ManifestRow]) -> Result<(), String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, false),
        Field::new("bucket_id", DataType::Int32, false),
        Field::new("source_path", DataType::Utf8, false),
        Field::new("source_size", DataType::Int64, false),
        Field::new("source_mtime_ns", DataType::Int64, false),
        Field::new("row_count", DataType::Int64, false),
        Field::new("min_date", DataType::Utf8, false),
        Field::new("max_date", DataType::Utf8, false),
        Field::new("feature_count", DataType::Int32, false),
        Field::new("label_columns", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.symbol.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|row| row.bucket_id).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.source_path.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.source_size).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| row.source_mtime_ns)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.row_count).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.min_date.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.max_date.clone())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|row| row.feature_count).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.label_columns.clone())
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|err| format!("failed to build manifest batch: {err}"))?;
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|err| format!("failed to create manifest writer {}: {err}", path.display()))?;
    writer
        .write(&batch)
        .map_err(|err| format!("failed to write manifest {}: {err}", path.display()))?;
    writer
        .close()
        .map_err(|err| format!("failed to close manifest {}: {err}", path.display()))?;
    Ok(())
}

fn write_meta(path: &Path, summary: &GenerateSummary) -> Result<(), String> {
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| format!("metadata path has no parent: {}", path.display()))?,
    )
    .map_err(|err| {
        format!(
            "failed to create metadata directory {}: {err}",
            path.display()
        )
    })?;
    let mut value =
        serde_json::to_value(summary).map_err(|err| format!("failed to encode metadata: {err}"))?;
    if let serde_json::Value::Object(ref mut object) = value {
        object.insert(
            "factor_space".to_owned(),
            serde_json::json!("full_factor_space"),
        );
        object.insert("num_rows".to_owned(), serde_json::json!(summary.num_rows));
        object.insert(
            "shape".to_owned(),
            serde_json::json!([summary.num_rows, summary.num_features]),
        );
        object.insert(
            "default_label_column".to_owned(),
            serde_json::json!("label"),
        );
        object.insert(
            "source_storage_layout".to_owned(),
            serde_json::json!("bucket_shards"),
        );
        object.insert(
            "incremental".to_owned(),
            serde_json::json!({"enabled": false, "reason": "rust_standalone_full_rebuild"}),
        );
    }
    let raw = serde_json::to_string_pretty(&value)
        .map_err(|err| format!("failed to serialize metadata: {err}"))?;
    fs::write(path, raw).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn sort_symbol_rows(rows: &mut SymbolRows) {
    let mut order = (0..rows.dates_ns.len()).collect::<Vec<_>>();
    order.sort_by_key(|idx| rows.dates_ns[*idx]);
    rows.dates_ns = order.iter().map(|idx| rows.dates_ns[*idx]).collect();
    for values in rows.columns.values_mut() {
        *values = order.iter().map(|idx| values[*idx]).collect();
    }
}

fn ensure_vwap(rows: &mut SymbolRows) {
    if rows.columns.contains_key("vwap") {
        return;
    }
    let close = column_or_nan(rows, "close");
    let amount = column_or_nan(rows, "amount");
    let volume = column_or_nan(rows, "volume");
    let vwap = amount
        .iter()
        .zip(volume)
        .zip(close)
        .map(|((amount, volume), close)| {
            if volume.is_finite() && volume != 0.0 {
                let value = amount / volume;
                if value.is_finite() {
                    value
                } else {
                    close
                }
            } else {
                close
            }
        })
        .collect::<Vec<_>>();
    rows.columns.insert("vwap".to_owned(), vwap);
}

fn column_or_nan(rows: &SymbolRows, name: &str) -> Vec<f64> {
    rows.columns
        .get(name)
        .cloned()
        .unwrap_or_else(|| vec![f64::NAN; rows.dates_ns.len()])
}

fn label_column_names(horizons: &[usize]) -> Vec<String> {
    let mut out = vec!["label".to_owned()];
    out.extend(horizons.iter().map(|horizon| label_column_name(*horizon)));
    out
}

fn label_column_name(horizon: usize) -> String {
    format!("label_{}d", horizon.max(1))
}

fn numeric_values(array: &dyn Array) -> Option<Vec<f64>> {
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(
            (0..values.len())
                .map(|idx| {
                    if values.is_valid(idx) {
                        values.value(idx)
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Some(
            (0..values.len())
                .map(|idx| {
                    if values.is_valid(idx) {
                        values.value(idx) as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(
            (0..values.len())
                .map(|idx| {
                    if values.is_valid(idx) {
                        values.value(idx) as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Some(
            (0..values.len())
                .map(|idx| {
                    if values.is_valid(idx) {
                        values.value(idx) as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(
            (0..values.len())
                .map(|idx| {
                    if values.is_valid(idx) {
                        values.value(idx) as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(
            (0..values.len())
                .map(|idx| {
                    if values.is_valid(idx) {
                        values.value(idx) as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        );
    }
    None
}

fn string_value(array: &dyn Array, idx: usize) -> Option<String> {
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return values.is_valid(idx).then(|| values.value(idx).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return values.is_valid(idx).then(|| values.value(idx).to_owned());
    }
    None
}

fn date_value_ns(array: &dyn Array, idx: usize) -> Option<i64> {
    if let Some(values) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return values.is_valid(idx).then(|| values.value(idx));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return values.is_valid(idx).then(|| values.value(idx) * 1_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return values.is_valid(idx).then(|| values.value(idx) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return values
            .is_valid(idx)
            .then(|| values.value(idx) * 1_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date64Array>() {
        return values.is_valid(idx).then(|| values.value(idx) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return values
            .is_valid(idx)
            .then(|| values.value(idx) as i64 * 86_400_000_000_000);
    }
    if let Some(value) = string_value(array, idx) {
        return parse_date_ns(&value);
    }
    None
}

fn parse_date_ns(value: &str) -> Option<i64> {
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .map(|dt| dt.and_utc().timestamp_nanos_opt().unwrap_or_default());
    }
    DateTime::parse_from_rfc3339(value)
        .ok()
        .and_then(|dt| dt.with_timezone(&Utc).timestamp_nanos_opt())
}

fn format_date_ns(value: i64) -> String {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .map(|dt| dt.date_naive().to_string())
        .unwrap_or_default()
}

fn extract_bucket_id(path: &Path) -> Result<i32, String> {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("invalid bucket path: {}", path.display()))?;
    let raw = stem.strip_prefix("part-").ok_or_else(|| {
        format!(
            "bucket file must be named part-XXXX.parquet: {}",
            path.display()
        )
    })?;
    raw.parse::<i32>()
        .map_err(|err| format!("invalid bucket id in {}: {err}", path.display()))
}

fn source_mtime_ns(metadata: &fs::Metadata) -> i64 {
    #[cfg(unix)]
    {
        metadata.mtime() * 1_000_000_000 + metadata.mtime_nsec()
    }
    #[cfg(not(unix))]
    {
        metadata
            .modified()
            .ok()
            .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|value| value.as_nanos() as i64)
            .unwrap_or_default()
    }
}

pub fn summarize_parquet_shards(
    root: &Path,
    bucket_root: &Path,
    stats: &[ParquetShardStats],
) -> ParquetLayoutSummary {
    let rows: Vec<i64> = stats.iter().map(|item| item.rows).collect();
    let columns: Vec<usize> = stats.iter().map(|item| item.columns).collect();
    let row_groups: Vec<usize> = stats.iter().map(|item| item.row_groups).collect();
    ParquetLayoutSummary {
        root: root.display().to_string(),
        bucket_root: bucket_root.display().to_string(),
        file_count: stats.len(),
        total_rows: rows.iter().sum(),
        total_row_groups: row_groups.iter().sum(),
        total_size_bytes: stats.iter().map(|item| item.size_bytes).sum(),
        min_rows: *rows.iter().min().unwrap_or(&0),
        median_rows: median_i64(&rows),
        max_rows: *rows.iter().max().unwrap_or(&0),
        min_columns: *columns.iter().min().unwrap_or(&0),
        median_columns: median_usize(&columns),
        max_columns: *columns.iter().max().unwrap_or(&0),
        min_row_groups: *row_groups.iter().min().unwrap_or(&0),
        median_row_groups: median_usize(&row_groups),
        max_row_groups: *row_groups.iter().max().unwrap_or(&0),
    }
}

fn median_i64(values: &[i64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) * 0.5
    } else {
        sorted[mid] as f64
    }
}

fn median_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) * 0.5
    } else {
        sorted[mid] as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_labels, build_open_to_open_label, build_symbol_factor_frame,
        deduplicate_exact_feature_columns, summarize_parquet_shards, validate_required_columns,
        GenerateOptions, ParquetShardStats, SymbolRows,
    };
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn summarizes_parquet_bucket_stats() {
        let stats = vec![
            ParquetShardStats {
                path: "part-0001.parquet".to_owned(),
                rows: 10,
                columns: 3,
                row_groups: 2,
                size_bytes: 100,
            },
            ParquetShardStats {
                path: "part-0002.parquet".to_owned(),
                rows: 20,
                columns: 5,
                row_groups: 4,
                size_bytes: 200,
            },
            ParquetShardStats {
                path: "part-0003.parquet".to_owned(),
                rows: 30,
                columns: 7,
                row_groups: 6,
                size_bytes: 300,
            },
        ];

        let summary =
            summarize_parquet_shards(Path::new("source"), Path::new("source/buckets"), &stats);

        assert_eq!(summary.file_count, 3);
        assert_eq!(summary.total_rows, 60);
        assert_eq!(summary.total_row_groups, 12);
        assert_eq!(summary.total_size_bytes, 600);
        assert_eq!(summary.min_rows, 10);
        assert_eq!(summary.median_rows, 20.0);
        assert_eq!(summary.max_rows, 30);
        assert_eq!(summary.median_columns, 5.0);
        assert_eq!(summary.median_row_groups, 4.0);
    }

    #[test]
    fn validates_required_columns_from_parquet_schema() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "ai4stock2_required_columns_test_{}_{}",
            std::process::id(),
            unique
        ));
        let bucket_dir = temp_dir.join("buckets");
        std::fs::create_dir_all(&bucket_dir).unwrap();
        let path = bucket_dir.join("part-0001.parquet");
        let message_type = parse_message_type(
            "
            message schema {
              REQUIRED INT64 date;
              REQUIRED BYTE_ARRAY symbol (STRING);
              OPTIONAL DOUBLE close;
            }
            ",
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .build();
        let file = File::create(&path).unwrap();
        let writer = SerializedFileWriter::new(file, message_type.into(), props.into()).unwrap();
        writer.close().unwrap();

        let summary = validate_required_columns(
            &temp_dir,
            &["date".to_owned(), "symbol".to_owned(), "close".to_owned()],
        )
        .unwrap();
        assert!(summary.validated);
        assert_eq!(summary.file_count, 1);
        assert_eq!(summary.required_columns_count, 3);

        let missing =
            validate_required_columns(&temp_dir, &["date".to_owned(), "missing_col".to_owned()])
                .unwrap();
        assert!(!missing.validated);
        assert_eq!(missing.missing_file_count, 1);
        assert_eq!(
            missing.missing_by_path[0].missing_columns,
            vec!["missing_col"]
        );
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }

    #[test]
    fn deduplicates_exact_feature_sources_with_first_seen_canonical_name() {
        let feature_columns = vec![
            ("RSV5".to_owned(), vec![1.0]),
            ("TEMP_rsv_5".to_owned(), vec![2.0]),
            ("LGBM_ret_20".to_owned(), vec![3.0]),
            ("TEMP_ret_20".to_owned(), vec![4.0]),
            ("TECH_MACD_12_26_9".to_owned(), vec![5.0]),
        ];

        let names = deduplicate_exact_feature_columns(feature_columns)
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();

        assert_eq!(names, vec!["RSV5", "LGBM_ret_20", "TECH_MACD_12_26_9"]);
    }

    #[test]
    fn build_open_to_open_labels_match_expected_horizons() {
        let mut columns = HashMap::new();
        columns.insert("open".to_owned(), vec![10.0, 10.5, 11.0, 10.8, 11.5, 12.0]);
        columns.insert(
            "volume".to_owned(),
            vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        );
        columns.insert(
            "amount".to_owned(),
            vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        );
        let rows = SymbolRows {
            dates_ns: (0..6)
                .map(|row_idx| row_idx as i64 * 86_400_000_000_000)
                .collect(),
            columns,
            all_numeric_float32: true,
        };

        let labels = build_labels(&rows, &[1, 3])
            .into_iter()
            .collect::<HashMap<_, _>>();

        assert!(labels.contains_key("label"));
        assert!(labels.contains_key("label_1d"));
        assert!(labels.contains_key("label_3d"));
        for (left, right) in labels["label"].iter().zip(labels["label_1d"].iter()) {
            assert!((*left == *right) || (left.is_nan() && right.is_nan()));
        }
        assert!((labels["label_1d"][0] - (11.0 / 10.5 - 1.0) as f32).abs() < 1e-7);
        assert!((labels["label_3d"][0] - (11.5 / 10.5 - 1.0) as f32).abs() < 1e-7);
        assert!(labels["label_3d"][2].is_nan());
    }

    #[test]
    fn open_to_open_label_rejects_non_tradable_or_capped_rows() {
        let mut columns = HashMap::new();
        columns.insert("open".to_owned(), vec![10.0, 10.0, 20.0, 22.0, 23.0]);
        columns.insert("volume".to_owned(), vec![100.0, 100.0, 100.0, 0.0, 100.0]);
        columns.insert(
            "amount".to_owned(),
            vec![1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        );
        let rows = SymbolRows {
            dates_ns: (0..5)
                .map(|row_idx| row_idx as i64 * 86_400_000_000_000)
                .collect(),
            columns,
            all_numeric_float32: true,
        };

        let label = build_open_to_open_label(&rows, 1);

        assert!(label[0].is_nan());
        assert!(label[1].is_nan());
        assert!(label[2].is_nan());
        assert!(label[3].is_nan());
    }

    #[test]
    fn standalone_tushare_full_factor_count_matches_native_registry() {
        let row_count = 130usize;
        let mut columns = HashMap::new();
        let open = (0..row_count)
            .map(|row_idx| 10.0 + row_idx as f64 * 0.01)
            .collect::<Vec<_>>();
        let close = open.iter().map(|value| value * 1.001).collect::<Vec<_>>();
        let high = close.iter().map(|value| value + 0.2).collect::<Vec<_>>();
        let low = open.iter().map(|value| value - 0.2).collect::<Vec<_>>();
        let volume = (0..row_count)
            .map(|row_idx| 1_000.0 + row_idx as f64)
            .collect::<Vec<_>>();
        let amount = volume
            .iter()
            .zip(close.iter())
            .map(|(volume_value, close_value)| volume_value * close_value)
            .collect::<Vec<_>>();
        columns.insert("open".to_owned(), open);
        columns.insert("high".to_owned(), high);
        columns.insert("low".to_owned(), low);
        columns.insert("close".to_owned(), close);
        columns.insert("volume".to_owned(), volume);
        columns.insert("amount".to_owned(), amount);
        columns.insert("turnover".to_owned(), vec![1.0; row_count]);
        columns.insert("circ_mv".to_owned(), vec![100_000.0; row_count]);
        columns.insert("pe_ttm".to_owned(), vec![12.0; row_count]);
        columns.insert("pb".to_owned(), vec![1.5; row_count]);
        let rows = SymbolRows {
            dates_ns: (0..row_count)
                .map(|row_idx| row_idx as i64 * 86_400_000_000_000)
                .collect(),
            columns,
            all_numeric_float32: true,
        };
        let options = GenerateOptions {
            parquet_dir: PathBuf::from("unused-source"),
            output_dir: PathBuf::from("unused-output"),
            data_source: "tushare".to_owned(),
            workers: 1,
            label_horizons: vec![1, 5, 10, 20],
            batch_size: 1024,
            bucket_limit: None,
        };

        let frame = build_symbol_factor_frame("000001", &rows, &options).unwrap();
        let names = frame
            .feature_columns
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(names.len(), 517);
        assert!(names.contains(&"RSV5"));
        assert!(!names.contains(&"TEMP_rsv_5"));
        assert!(names.contains(&"LGBM_dist_ma120"));
        assert!(!names.contains(&"TEMP_ma_gap_120"));
    }
}
