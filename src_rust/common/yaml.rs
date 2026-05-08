use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};
use std::fs;
use std::path::Path;

pub fn read_yaml_file(path: impl AsRef<Path>) -> Result<YamlValue, String> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_yaml::from_str(&text).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

pub fn write_yaml_file(path: &Path, value: &YamlValue) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let text = serde_yaml::to_string(value)
        .map_err(|err| format!("failed to encode YAML {}: {err}", path.display()))?;
    fs::write(path, text).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

pub fn deep_merge_yaml(base: &mut YamlValue, overlay: YamlValue) {
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
        (base_slot, overlay_value) => *base_slot = overlay_value,
    }
}

pub fn ensure_mapping(value: &mut YamlValue) {
    if !value.is_mapping() {
        *value = YamlValue::Mapping(YamlMapping::new());
    }
}

pub fn set_yaml_dotted(
    cfg: &mut YamlValue,
    dotted_key: &str,
    value: YamlValue,
) -> Result<(), String> {
    let parts = dotted_key
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(format!("invalid dotted override key: {dotted_key}"));
    }
    let mut cursor = cfg;
    for part in &parts[..parts.len() - 1] {
        ensure_mapping(cursor);
        let mapping = cursor.as_mapping_mut().expect("mapping just initialized");
        cursor = mapping
            .entry(YamlValue::String((*part).to_owned()))
            .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    }
    ensure_mapping(cursor);
    cursor
        .as_mapping_mut()
        .expect("mapping just initialized")
        .insert(YamlValue::String(parts[parts.len() - 1].to_owned()), value);
    Ok(())
}

pub fn parse_key_value_arg(raw: &str, label: &str) -> Result<(String, YamlValue), String> {
    let Some((key, value)) = raw.trim().split_once('=') else {
        return Err(format!("{label} must be in key=value form, got: {raw}"));
    };
    let key = key.trim().to_owned();
    if key.is_empty() {
        return Err(format!("{label} key must be non-empty, got: {raw}"));
    }
    let parsed = serde_yaml::from_str::<YamlValue>(value.trim())
        .unwrap_or_else(|_| YamlValue::String(value.trim().to_owned()));
    Ok((key, parsed))
}

pub fn yaml_path<'a>(value: &'a YamlValue, path: &[&str]) -> Option<&'a YamlValue> {
    let mut cursor = value;
    for key in path {
        let mapping = cursor.as_mapping()?;
        cursor = mapping.get(YamlValue::String((*key).to_owned()))?;
    }
    Some(cursor)
}

pub fn yaml_path_string(value: &YamlValue, path: &[&str]) -> Option<String> {
    yaml_path(value, path).and_then(|value| yaml_string_scalar(value).ok())
}

pub fn yaml_path_usize(value: &YamlValue, path: &[&str]) -> Option<usize> {
    yaml_path(value, path).and_then(yaml_usize_value)
}

pub fn yaml_path_f64(value: &YamlValue, path: &[&str]) -> Option<f64> {
    yaml_path(value, path).and_then(yaml_f64_value)
}

pub fn yaml_path_bool(value: &YamlValue, path: &[&str]) -> Option<bool> {
    yaml_path(value, path).and_then(YamlValue::as_bool)
}

pub fn yaml_sequence_strings(value: Option<&YamlValue>) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    yaml_string_sequence(value).map(Some)
}

pub fn yaml_string_sequence(value: &YamlValue) -> Result<Vec<String>, String> {
    let sequence = value
        .as_sequence()
        .ok_or_else(|| "expected YAML string sequence".to_owned())?;
    sequence.iter().map(yaml_string_scalar).collect()
}

pub fn yaml_string_scalar(value: &YamlValue) -> Result<String, String> {
    match value {
        YamlValue::String(text) => Ok(text.clone()),
        YamlValue::Number(number) => Ok(number.to_string()),
        YamlValue::Bool(value) => Ok(value.to_string()),
        _ => Err("expected scalar value".to_owned()),
    }
}

pub fn yaml_usize_value(value: &YamlValue) -> Option<usize> {
    match value {
        YamlValue::Number(number) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok()),
        YamlValue::String(text) => text.parse::<usize>().ok(),
        _ => None,
    }
}

pub fn yaml_f64_value(value: &YamlValue) -> Option<f64> {
    match value {
        YamlValue::Number(number) => number.as_f64(),
        YamlValue::String(text) => text.parse::<f64>().ok(),
        _ => None,
    }
}

pub fn yaml_usize(value: usize) -> YamlValue {
    YamlValue::Number(serde_yaml::Number::from(value as u64))
}
