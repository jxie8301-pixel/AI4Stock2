use std::path::{Path, PathBuf};

pub fn resolve_relative_to_repo(profile_config_path: &Path, raw_path: &str) -> PathBuf {
    let path = PathBuf::from(raw_path);
    if path.is_absolute() {
        return path;
    }
    profile_config_path
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

pub fn normalize_data_source(value: &str) -> Result<String, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" => Ok("akshare".to_owned()),
        "eastmoney" | "em" | "akshare" => Ok("akshare".to_owned()),
        "tushare" => Ok("tushare".to_owned()),
        other => Err(format!(
            "Unsupported data source: {other}. Available: akshare, tushare"
        )),
    }
}

pub fn default_factor_store_dir(data_source: &str, factor_store_name: &str) -> PathBuf {
    if data_source == "akshare" {
        PathBuf::from("data/factor_store").join(factor_store_name)
    } else {
        PathBuf::from("data/factor_store").join(format!("{data_source}_{factor_store_name}"))
    }
}
