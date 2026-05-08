use crate::common::parquet::{
    date_value_ns, numeric_value, open_projected_parquet_reader, parse_datetime_ns, required_column,
};
use std::collections::BTreeMap;
use std::path::Path;

pub fn read_benchmark_csv(
    path: &Path,
    date_column: &str,
    value_column: &str,
) -> Result<Vec<(i64, f64)>, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|err| format!("failed to read benchmark CSV {}: {err}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|err| {
            format!(
                "failed to parse benchmark CSV {} headers: {err}",
                path.display()
            )
        })?
        .clone();
    let date_index = headers
        .iter()
        .position(|name| name == date_column)
        .ok_or_else(|| format!("benchmark file is missing date column: {date_column}"))?;
    let value_index = headers
        .iter()
        .position(|name| name == value_column)
        .ok_or_else(|| format!("benchmark file is missing value column: {value_column}"))?;
    let mut by_date = BTreeMap::new();
    for record in reader.records() {
        let record = record
            .map_err(|err| format!("failed to parse benchmark CSV {}: {err}", path.display()))?;
        let Some(raw_date) = record.get(date_index) else {
            continue;
        };
        let Some(raw_value) = record.get(value_index) else {
            continue;
        };
        let Ok(date_ns) = parse_datetime_ns(raw_date.trim()) else {
            continue;
        };
        let Ok(value) = raw_value.trim().parse::<f64>() else {
            continue;
        };
        if value.is_finite() {
            by_date.insert(date_ns, value);
        }
    }
    Ok(by_date.into_iter().collect())
}

pub fn read_benchmark_parquet(
    path: &Path,
    date_column: &str,
    value_column: &str,
) -> Result<Vec<(i64, f64)>, String> {
    let columns = vec![date_column.to_owned(), value_column.to_owned()];
    let reader = open_projected_parquet_reader(path, &columns, 65_536)?;
    let mut by_date = BTreeMap::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let date_array = required_column(&batch, date_column, path)?;
        let value_array = required_column(&batch, value_column, path)?;
        for row_index in 0..batch.num_rows() {
            if date_array.is_null(row_index) || value_array.is_null(row_index) {
                continue;
            }
            let date_ns = date_value_ns(date_array, row_index, path)?;
            let value = numeric_value(value_array, row_index, path)?;
            if value.is_finite() {
                by_date.insert(date_ns, value);
            }
        }
    }
    Ok(by_date.into_iter().collect())
}

pub fn coerce_benchmark_returns(
    values: Vec<(i64, f64)>,
    value_type: &str,
) -> Result<Vec<(i64, f64)>, String> {
    match value_type {
        "return" => Ok(values),
        "close" | "" => {
            let mut out = Vec::with_capacity(values.len());
            let mut previous: Option<f64> = None;
            for (date_ns, close) in values {
                let daily_return = if let Some(prev) = previous {
                    if prev != 0.0 {
                        close / prev - 1.0
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                out.push((date_ns, daily_return));
                previous = Some(close);
            }
            Ok(out)
        }
        other => Err(format!(
            "unsupported benchmark value_type: {other}; supported: close, return"
        )),
    }
}

pub fn load_file_benchmark_returns(
    path: &Path,
    date_column: &str,
    value_column: &str,
    value_type: &str,
) -> Result<Vec<(i64, f64)>, String> {
    let values = match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "csv" | "txt" => read_benchmark_csv(path, date_column, value_column)?,
        "parquet" | "pq" => read_benchmark_parquet(path, date_column, value_column)?,
        _ => {
            return Err(format!(
                "unsupported benchmark file format: {}. Use .csv, .txt, .parquet, or .pq.",
                path.display()
            ));
        }
    };
    if values.is_empty() {
        return Err(format!(
            "benchmark returned no usable rows from {}",
            path.display()
        ));
    }
    coerce_benchmark_returns(values, value_type)
}

pub fn cross_section_mean_returns<I>(
    values: I,
    empty_error: &str,
) -> Result<Vec<(i64, f64)>, String>
where
    I: IntoIterator<Item = (i64, f64)>,
{
    let mut sums: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
    for (date_ns, value) in values {
        if value.is_finite() {
            let entry = sums.entry(date_ns).or_insert((0.0, 0));
            entry.0 += value;
            entry.1 += 1;
        }
    }
    let out = sums
        .into_iter()
        .filter_map(|(date_ns, (sum, count))| (count > 0).then_some((date_ns, sum / count as f64)))
        .collect::<Vec<_>>();
    if out.is_empty() {
        return Err(empty_error.to_owned());
    }
    Ok(out)
}
