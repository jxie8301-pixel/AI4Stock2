use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub type CsvRow = BTreeMap<String, String>;
pub type JsonRow = BTreeMap<String, JsonValue>;

pub fn read_required_csv_rows(path: &Path) -> Result<Vec<CsvRow>, String> {
    if !path.exists() {
        return Err(format!("Missing required artifact: {}", path.display()));
    }
    let mut reader = csv::ReaderBuilder::new()
        .from_path(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|err| format!("failed to read headers {}: {err}", path.display()))?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
        rows.push(
            headers
                .iter()
                .zip(record.iter())
                .map(|(key, value)| (key.clone(), value.to_owned()))
                .collect(),
        );
    }
    Ok(rows)
}

pub fn read_required_json(path: &Path) -> Result<JsonValue, String> {
    if !path.exists() {
        return Err(format!("Missing required artifact: {}", path.display()));
    }
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

pub fn write_json_rows_csv(path: &Path, rows: &[JsonRow]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let headers = collect_json_headers(rows);
    let mut writer = csv::WriterBuilder::new()
        .from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record(&headers)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    for row in rows {
        writer
            .write_record(
                headers
                    .iter()
                    .map(|key| json_to_csv_string(row.get(key).unwrap_or(&JsonValue::Null))),
            )
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

pub fn write_json_pretty(
    path: &Path,
    value: &JsonValue,
    trailing_newline: bool,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut text = serde_json::to_string_pretty(value)
        .map_err(|err| format!("failed to encode {}: {err}", path.display()))?;
    if trailing_newline {
        text.push('\n');
    }
    fs::write(path, text).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

pub fn collect_json_headers(rows: &[JsonRow]) -> Vec<String> {
    let mut headers = Vec::new();
    let mut seen = BTreeSet::new();
    for row in rows {
        for key in row.keys() {
            if seen.insert(key.clone()) {
                headers.push(key.clone());
            }
        }
    }
    headers
}

pub fn json_to_csv_string(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => String::new(),
        JsonValue::String(text) => text.clone(),
        JsonValue::Number(number) => number.to_string(),
        JsonValue::Bool(value) => value.to_string(),
        other => other.to_string(),
    }
}
