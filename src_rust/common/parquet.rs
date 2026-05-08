use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    ProjectionMask,
};
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

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
