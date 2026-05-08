use serde_yaml::Value as YamlValue;
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
