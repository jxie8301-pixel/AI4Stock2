use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    ProjectionMask,
};
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde::Serialize;
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

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

pub fn open_projected_parquet_reader(
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
    build_projected_reader(path, builder, indices, batch_size)
}

pub fn open_existing_projected_parquet_reader(
    path: &Path,
    column_candidates: &[&str],
    batch_size: usize,
) -> Result<ParquetRecordBatchReader, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to open parquet {}: {err}", path.display()))?;
    let indices = column_candidates
        .iter()
        .filter_map(|name| builder.schema().index_of(name).ok())
        .collect::<BTreeSet<_>>();
    build_projected_reader(path, builder, indices, batch_size)
}

fn build_projected_reader(
    path: &Path,
    builder: ParquetRecordBatchReaderBuilder<File>,
    indices: BTreeSet<usize>,
    batch_size: usize,
) -> Result<ParquetRecordBatchReader, String> {
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

pub fn optional_column<'a>(batch: &'a RecordBatch, candidates: &[&str]) -> Option<&'a dyn Array> {
    candidates.iter().find_map(|name| {
        batch
            .schema()
            .index_of(name)
            .ok()
            .map(|idx| batch.column(idx).as_ref())
    })
}

pub fn required_column<'a>(
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

pub fn required_column_any<'a>(
    batch: &'a RecordBatch,
    candidates: &[&str],
    path: &Path,
) -> Result<&'a dyn Array, String> {
    optional_column(batch, candidates).ok_or_else(|| {
        format!(
            "{} is missing required column: {}",
            path.display(),
            candidates.join(" or ")
        )
    })
}

pub fn string_value_or_empty(
    array: &dyn Array,
    row_index: usize,
    path: &Path,
) -> Result<String, String> {
    if array.is_null(row_index) {
        return Ok(String::new());
    }
    string_value_inner(array, row_index, path)
}

pub fn required_string_value(
    array: &dyn Array,
    row_index: usize,
    path: &Path,
) -> Result<String, String> {
    if array.is_null(row_index) {
        return Err(format!(
            "{} has null string value at row {row_index}",
            path.display()
        ));
    }
    string_value_inner(array, row_index, path)
}

fn string_value_inner(array: &dyn Array, row_index: usize, path: &Path) -> Result<String, String> {
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

pub fn numeric_value(array: &dyn Array, row_index: usize, path: &Path) -> Result<f64, String> {
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
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return values
            .value(row_index)
            .trim()
            .parse::<f64>()
            .map_err(|err| {
                format!(
                    "{} has non-numeric value at row {row_index}: {err}",
                    path.display()
                )
            });
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return values
            .value(row_index)
            .trim()
            .parse::<f64>()
            .map_err(|err| {
                format!(
                    "{} has non-numeric value at row {row_index}: {err}",
                    path.display()
                )
            });
    }
    Err(format!(
        "{} has unsupported numeric type {:?}",
        path.display(),
        array.data_type()
    ))
}

pub fn date_value_ns(array: &dyn Array, row_index: usize, path: &Path) -> Result<i64, String> {
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
        return parse_datetime_ns(values.value(row_index));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return parse_datetime_ns(values.value(row_index));
    }
    Err(format!(
        "{} has unsupported date type {:?}",
        path.display(),
        array.data_type()
    ))
}

pub fn parse_datetime_ns(raw: &str) -> Result<i64, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("empty datetime".to_owned());
    }
    if text.len() == 8 && text.chars().all(|ch| ch.is_ascii_digit()) {
        let date = NaiveDate::parse_from_str(text, "%Y%m%d")
            .map_err(|err| format!("failed to parse date {text}: {err}"))?;
        return naive_datetime_to_ns(
            date.and_hms_opt(0, 0, 0)
                .ok_or_else(|| format!("invalid date {text}"))?,
        );
    }
    if let Ok(date) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return naive_datetime_to_ns(
            date.and_hms_opt(0, 0, 0)
                .ok_or_else(|| format!("invalid date {text}"))?,
        );
    }
    for fmt in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(datetime) = NaiveDateTime::parse_from_str(text, fmt) {
            return naive_datetime_to_ns(datetime);
        }
    }
    if let Ok(datetime) = DateTime::parse_from_rfc3339(text) {
        return datetime
            .timestamp_nanos_opt()
            .ok_or_else(|| format!("datetime out of range: {text}"));
    }
    Err(format!("failed to parse datetime: {text}"))
}

fn naive_datetime_to_ns(datetime: NaiveDateTime) -> Result<i64, String> {
    datetime
        .and_utc()
        .timestamp_nanos_opt()
        .ok_or_else(|| format!("datetime out of range: {datetime}"))
}

pub fn format_date_ns(value: i64) -> String {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    DateTime::<Utc>::from_timestamp(secs, nanos)
        .map(|datetime| datetime.date_naive().to_string())
        .unwrap_or_default()
}
