use crate::prediction_bundle::PredictionBundle;
use crate::run_bundle::{self, PredictionBundleCache, RunBundleExecutionOptions};
use chrono::Local;
use serde_yaml::{Mapping, Number, Value};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PREDICTION_ARTIFACT_DIRNAME: &str = "prediction_artifacts";
const PREDICTION_METADATA_FILENAME: &str = "metadata.json";

#[derive(Debug)]
pub(crate) enum BundleDispatch {
    Ran(ExitCode),
    Fallback(String),
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedBundleRun {
    pub(crate) bundle_dir: PathBuf,
    pub(crate) config_path: PathBuf,
    pub(crate) output_dir: PathBuf,
    pub(crate) config: Value,
    pub(crate) execution: RunBundleExecutionOptions,
    pub(crate) save_predictions: bool,
}

pub(crate) enum BundlePlan {
    Planned(PreparedBundleRun),
    Fallback(String),
}

#[derive(Debug, Clone)]
struct NativeRollingBundleArgs {
    config_path: PathBuf,
    config_is_snapshot: bool,
    load_predictions_dir: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    store_dir: Option<PathBuf>,
    run_tag: Option<String>,
    model: Option<String>,
    data_source: Option<String>,
    experiment_profile: Option<String>,
    feature_profile: Option<String>,
    profile: Option<String>,
    topk: Option<i64>,
    n_drop: Option<i64>,
    rebalance_freq: Option<i64>,
    signal_horizon: Option<i64>,
    retrain_step: Option<i64>,
    train_days: Option<i64>,
    valid_days: Option<i64>,
    set_overrides: Vec<String>,
    backtest_artifact_level: String,
    disable_local_store: bool,
    save_predictions: bool,
    skip_reference_baselines: bool,
    skip_opportunity_diagnostics: bool,
    skip_backtest_plots: bool,
    skip_backtest_trace: bool,
}

impl Default for NativeRollingBundleArgs {
    fn default() -> Self {
        Self {
            config_path: PathBuf::from("configs/config.yaml"),
            config_is_snapshot: false,
            load_predictions_dir: None,
            output_dir: None,
            store_dir: None,
            run_tag: None,
            model: None,
            data_source: None,
            experiment_profile: None,
            feature_profile: None,
            profile: None,
            topk: None,
            n_drop: None,
            rebalance_freq: None,
            signal_horizon: None,
            retrain_step: None,
            train_days: None,
            valid_days: None,
            set_overrides: Vec::new(),
            backtest_artifact_level: "full".to_owned(),
            disable_local_store: false,
            save_predictions: false,
            skip_reference_baselines: false,
            skip_opportunity_diagnostics: false,
            skip_backtest_plots: false,
            skip_backtest_trace: false,
        }
    }
}

pub(crate) fn try_run(
    args: &[String],
    repo_root: &Path,
    baseline_jobs: usize,
) -> Result<BundleDispatch, String> {
    std::env::set_current_dir(repo_root)
        .map_err(|err| format!("failed to enter repo root {}: {err}", repo_root.display()))?;
    let plan = match prepare_run(args, repo_root, baseline_jobs)? {
        BundlePlan::Planned(plan) => plan,
        BundlePlan::Fallback(reason) => return Ok(BundleDispatch::Fallback(reason)),
    };
    let code = run_prepared(plan)?;
    Ok(BundleDispatch::Ran(code))
}

pub(crate) fn prepare_run(
    args: &[String],
    repo_root: &Path,
    baseline_jobs: usize,
) -> Result<BundlePlan, String> {
    let parsed = match parse_native_rolling_bundle_args(args) {
        Ok(parsed) => parsed,
        Err(reason) => return Ok(BundlePlan::Fallback(reason)),
    };
    if !parsed.config_is_snapshot {
        return Ok(BundlePlan::Fallback(
            "direct Rust bundle path requires --config-is-snapshot".to_owned(),
        ));
    }
    let Some(bundle_dir) = parsed.load_predictions_dir.clone() else {
        return Ok(BundlePlan::Fallback(
            "direct Rust bundle path requires --load-predictions-dir".to_owned(),
        ));
    };

    let config_path = resolve_repo_path(repo_root, &parsed.config_path);
    let mut config = read_yaml(&config_path)?;
    apply_runtime_overrides(&mut config, &parsed)?;

    if parsed.save_predictions && bool_at(&config, &["strategy", "score_fusion", "enabled"], false)
    {
        return Ok(BundlePlan::Fallback(
            "direct Rust bundle path cannot yet persist fused prediction bundles for --save-predictions".to_owned(),
        ));
    }

    let output_dir = parsed
        .output_dir
        .clone()
        .map(|path| resolve_repo_path(repo_root, &path))
        .unwrap_or_else(|| resolve_run_output_dir(repo_root, &config, &parsed));
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    write_config_snapshot(&output_dir, &config)?;

    let reduced_artifact_level = matches!(
        parsed.backtest_artifact_level.as_str(),
        "reports" | "metrics"
    );
    let skip_backtest_plots = parsed.skip_backtest_plots || reduced_artifact_level;

    let bundle_dir = resolve_repo_path(repo_root, &bundle_dir);
    Ok(BundlePlan::Planned(PreparedBundleRun {
        bundle_dir,
        config_path,
        output_dir,
        config,
        execution: RunBundleExecutionOptions {
            skip_reference_baselines: parsed.skip_reference_baselines,
            skip_backtest_plots,
            baseline_jobs: baseline_jobs.max(1),
            quiet: false,
        },
        save_predictions: parsed.save_predictions,
    }))
}

pub(crate) fn run_prepared(plan: PreparedBundleRun) -> Result<ExitCode, String> {
    let code = run_bundle::run_with_config(
        &plan.bundle_dir,
        &plan.config_path,
        &plan.output_dir,
        plan.config,
        plan.execution,
    )?;
    if plan.save_predictions {
        copy_prediction_artifacts(
            &plan.bundle_dir,
            &plan.output_dir.join(PREDICTION_ARTIFACT_DIRNAME),
        )?;
    }
    Ok(code)
}

pub(crate) fn run_prepared_with_bundle_and_cache(
    plan: PreparedBundleRun,
    bundle: &PredictionBundle,
    secondary_cache: &mut PredictionBundleCache,
) -> Result<ExitCode, String> {
    let code = run_bundle::run_with_loaded_bundle_and_cache(
        bundle,
        &plan.config_path,
        &plan.output_dir,
        plan.config,
        plan.execution,
        secondary_cache,
    )?;
    if plan.save_predictions {
        copy_prediction_artifacts(
            &plan.bundle_dir,
            &plan.output_dir.join(PREDICTION_ARTIFACT_DIRNAME),
        )?;
    }
    Ok(code)
}

fn parse_native_rolling_bundle_args(args: &[String]) -> Result<NativeRollingBundleArgs, String> {
    let mut parsed = NativeRollingBundleArgs::default();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        match split_option(arg) {
            ("--config", value) => {
                parsed.config_path =
                    PathBuf::from(value_or_next(value, args, &mut idx, "--config")?);
            }
            ("--config-is-snapshot", None) => parsed.config_is_snapshot = true,
            ("--load-predictions-dir", value) => {
                parsed.load_predictions_dir = Some(PathBuf::from(value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--load-predictions-dir",
                )?));
            }
            ("--output-dir" | "--results-dir", value) => {
                parsed.output_dir = Some(PathBuf::from(value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--output-dir",
                )?));
            }
            ("--store-dir", value) => {
                parsed.store_dir = Some(PathBuf::from(value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--store-dir",
                )?));
            }
            ("--run-tag", value) => {
                parsed.run_tag = Some(value_or_next(value, args, &mut idx, "--run-tag")?);
            }
            ("--model", value) => {
                parsed.model = Some(value_or_next(value, args, &mut idx, "--model")?);
            }
            ("--data-source", value) => {
                parsed.data_source = Some(value_or_next(value, args, &mut idx, "--data-source")?);
            }
            ("--experiment-profile", value) => {
                parsed.experiment_profile = Some(value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--experiment-profile",
                )?);
            }
            ("--feature-profile", value) => {
                parsed.feature_profile =
                    Some(value_or_next(value, args, &mut idx, "--feature-profile")?);
            }
            ("--profile", value) => {
                parsed.profile = Some(value_or_next(value, args, &mut idx, "--profile")?);
            }
            ("--topk", value) => {
                parsed.topk = Some(parse_i64(&value_or_next(value, args, &mut idx, "--topk")?)?);
            }
            ("--n-drop", value) => {
                parsed.n_drop = Some(parse_i64(&value_or_next(
                    value, args, &mut idx, "--n-drop",
                )?)?);
            }
            ("--rebalance-freq", value) => {
                parsed.rebalance_freq = Some(parse_i64(&value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--rebalance-freq",
                )?)?);
            }
            ("--signal-horizon" | "--label-horizon", value) => {
                parsed.signal_horizon = Some(parse_i64(&value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--signal-horizon",
                )?)?);
            }
            ("--retrain-step" | "--horizon", value) => {
                parsed.retrain_step = Some(parse_i64(&value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--retrain-step",
                )?)?);
            }
            ("--train-days", value) => {
                parsed.train_days = Some(parse_i64(&value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--train-days",
                )?)?);
            }
            ("--valid-days", value) => {
                parsed.valid_days = Some(parse_i64(&value_or_next(
                    value,
                    args,
                    &mut idx,
                    "--valid-days",
                )?)?);
            }
            ("--set", value) => {
                parsed
                    .set_overrides
                    .push(value_or_next(value, args, &mut idx, "--set")?);
            }
            ("--backtest-artifact-level", value) => {
                parsed.backtest_artifact_level =
                    value_or_next(value, args, &mut idx, "--backtest-artifact-level")?
                        .trim()
                        .to_ascii_lowercase();
                if !matches!(
                    parsed.backtest_artifact_level.as_str(),
                    "full" | "reports" | "metrics"
                ) {
                    return Err(format!(
                        "unsupported --backtest-artifact-level {}",
                        parsed.backtest_artifact_level
                    ));
                }
            }
            ("--torch-gpu", value) => {
                let _ = value_or_next(value, args, &mut idx, "--torch-gpu")?;
            }
            ("--disable-local-store", None) => parsed.disable_local_store = true,
            ("--save-models" | "--load-models", None) => {}
            ("--save-predictions", None) => parsed.save_predictions = true,
            ("--skip-reference-baselines", None) => parsed.skip_reference_baselines = true,
            ("--skip-opportunity-diagnostics", None) => parsed.skip_opportunity_diagnostics = true,
            ("--skip-backtest-plots", None) => parsed.skip_backtest_plots = true,
            ("--skip-backtest-trace", None) => parsed.skip_backtest_trace = true,
            ("-h" | "--help", None) => {
                return Err("help requested for Python-compatible bundle entrypoint".to_owned());
            }
            (other, _) => {
                return Err(format!(
                    "unsupported Python-compatible bundle option for direct Rust path: {other}"
                ));
            }
        }
        idx += 1;
    }
    Ok(parsed)
}

fn split_option(arg: &str) -> (&str, Option<String>) {
    if let Some((option, value)) = arg.split_once('=') {
        (option, Some(value.to_owned()))
    } else {
        (arg, None)
    }
}

fn value_or_next(
    inline: Option<String>,
    args: &[String],
    idx: &mut usize,
    option: &str,
) -> Result<String, String> {
    if let Some(value) = inline {
        return Ok(value);
    }
    *idx += 1;
    args.get(*idx)
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn parse_i64(raw: &str) -> Result<i64, String> {
    raw.trim()
        .parse::<i64>()
        .map_err(|err| format!("failed to parse integer {raw}: {err}"))
}

fn resolve_repo_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn read_yaml(path: &Path) -> Result<Value, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    serde_yaml::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn write_config_snapshot(output_dir: &Path, config: &Value) -> Result<(), String> {
    let path = output_dir.join("config_snapshot.yaml");
    let file =
        File::create(&path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    serde_yaml::to_writer(file, config)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn apply_runtime_overrides(
    config: &mut Value,
    args: &NativeRollingBundleArgs,
) -> Result<(), String> {
    if let Some(model) = &args.model {
        set_path(config, &["model", "name"], Value::String(model.clone()))?;
    }
    if let Some(data_source) = &args.data_source {
        set_path(
            config,
            &["data", "source"],
            Value::String(data_source.clone()),
        )?;
    }
    if let Some(profile) = args.feature_profile.as_ref().or(args.profile.as_ref()) {
        set_path(
            config,
            &["features", "profile"],
            Value::String(profile.clone()),
        )?;
    }
    if let Some(topk) = args.topk {
        set_path(
            config,
            &["strategy", "topk"],
            Value::Number(Number::from(topk)),
        )?;
    }
    if let Some(n_drop) = args.n_drop {
        set_path(
            config,
            &["strategy", "n_drop"],
            Value::Number(Number::from(n_drop)),
        )?;
    }
    if let Some(signal_horizon) = args.signal_horizon {
        set_path(
            config,
            &["label", "signal_horizon"],
            Value::Number(Number::from(signal_horizon)),
        )?;
    }
    if let Some(rebalance_freq) = args.rebalance_freq {
        set_path(
            config,
            &["backtest", "rebalance_freq"],
            Value::Number(Number::from(rebalance_freq)),
        )?;
    }
    if let Some(retrain_step) = args.retrain_step {
        set_path(
            config,
            &["rolling", "retrain_step"],
            Value::Number(Number::from(retrain_step)),
        )?;
    }
    if let Some(train_days) = args.train_days {
        set_path(
            config,
            &["rolling", "train_days"],
            Value::Number(Number::from(train_days)),
        )?;
    }
    if let Some(valid_days) = args.valid_days {
        set_path(
            config,
            &["rolling", "valid_days"],
            Value::Number(Number::from(valid_days)),
        )?;
    }
    for raw in &args.set_overrides {
        let (key, value) = parse_override_arg(raw)?;
        apply_dotted_override(config, &key, value)?;
    }
    Ok(())
}

fn parse_override_arg(raw: &str) -> Result<(String, Value), String> {
    let text = raw.trim();
    let Some((key, value)) = text.split_once('=') else {
        return Err(format!("Override must be in key=value form, got: {raw}"));
    };
    let key = key.trim();
    if key.is_empty() {
        return Err(format!("Override key must be non-empty, got: {raw}"));
    }
    let value = serde_yaml::from_str(value.trim())
        .unwrap_or_else(|_| Value::String(value.trim().to_owned()));
    Ok((key.to_owned(), value))
}

fn apply_dotted_override(root: &mut Value, dotted_key: &str, value: Value) -> Result<(), String> {
    let parts = dotted_key
        .split('.')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(format!("Invalid override key: {dotted_key}"));
    }
    set_path(root, &parts, value)
}

fn set_path(root: &mut Value, path: &[&str], value: Value) -> Result<(), String> {
    let Some((last, parents)) = path.split_last() else {
        return Err("empty YAML path".to_owned());
    };
    let mut cursor = root;
    for part in parents {
        let mapping = ensure_mapping(cursor)?;
        let key = Value::String((*part).to_owned());
        if !mapping.contains_key(&key) {
            mapping.insert(key.clone(), Value::Mapping(Mapping::new()));
        }
        cursor = mapping
            .get_mut(&key)
            .ok_or_else(|| format!("failed to descend into YAML key {part}"))?;
        if !cursor.is_mapping() {
            return Err(format!(
                "Cannot override nested key through non-mapping path: {}",
                path.join(".")
            ));
        }
    }
    ensure_mapping(cursor)?.insert(Value::String((*last).to_owned()), value);
    Ok(())
}

fn ensure_mapping(value: &mut Value) -> Result<&mut Mapping, String> {
    if value.is_null() {
        *value = Value::Mapping(Mapping::new());
    }
    match value {
        Value::Mapping(mapping) => Ok(mapping),
        _ => Err("expected YAML mapping".to_owned()),
    }
}

fn resolve_run_output_dir(
    repo_root: &Path,
    config: &Value,
    args: &NativeRollingBundleArgs,
) -> PathBuf {
    let model_name = string_at(config, &["model", "name"]).unwrap_or_else(|| "lgbm".to_owned());
    let local_store_enabled =
        bool_at(config, &["artifacts", "enable_local_store"], true) && !args.disable_local_store;
    if !local_store_enabled {
        return repo_root
            .join("results")
            .join(format!("native_rolling_{model_name}"));
    }
    let root = args.store_dir.clone().unwrap_or_else(|| {
        raw_string_at(config, &["artifacts", "store_dir"])
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("results/experiments"))
    });
    let root = resolve_repo_path(repo_root, &root);
    root.join("native")
        .join("rolling")
        .join(&model_name)
        .join(build_run_id(config, args, &model_name))
}

fn build_run_id(config: &Value, args: &NativeRollingBundleArgs, model_name: &str) -> String {
    let strategy_slug = format!(
        "top{}_drop{}_w{}_st{}_keep{}_mins{}_reb{}",
        display_value_at(config, &["strategy", "topk"], "na"),
        display_value_at(config, &["strategy", "n_drop"], "na"),
        display_value_at(config, &["strategy", "weighting"], "equal"),
        display_value_at(config, &["strategy", "score_transform"], "none"),
        display_value_at(config, &["strategy", "keep_top_n"], "na"),
        display_value_at(config, &["strategy", "min_score"], "na"),
        display_value_at(config, &["backtest", "rebalance_freq"], "5"),
    );
    let mut parts = vec![
        Local::now().format("%Y%m%d_%H%M%S").to_string(),
        "native".to_owned(),
        "rolling".to_owned(),
        model_name.to_owned(),
        strategy_slug,
    ];
    let config_experiment_profile = raw_string_at(config, &["experiment", "profile"]);
    let config_feature_profile = raw_string_at(config, &["features", "profile"]);
    let tag_slug = slugify(args.run_tag.as_deref());
    let experiment_slug = slugify(
        args.experiment_profile
            .as_deref()
            .or(config_experiment_profile.as_deref()),
    );
    let feature_slug = slugify(
        args.feature_profile
            .as_deref()
            .or(args.profile.as_deref())
            .or(config_feature_profile.as_deref()),
    );
    if tag_slug.is_empty() {
        if !experiment_slug.is_empty() {
            parts.push(experiment_slug);
        } else if !feature_slug.is_empty() {
            parts.push(feature_slug);
        }
    } else {
        parts.push(tag_slug);
    }
    parts.join("__")
}

fn display_value_at(root: &Value, path: &[&str], default: &str) -> String {
    let Some(value) = value_at(root, path) else {
        return default.to_owned();
    };
    match value {
        Value::Null => default.to_owned(),
        Value::String(text) => text.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        _ => default.to_owned(),
    }
}

fn raw_string_at(root: &Value, path: &[&str]) -> Option<String> {
    value_at(root, path).and_then(|value| match value {
        Value::String(text) => Some(text.clone()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn string_at(root: &Value, path: &[&str]) -> Option<String> {
    raw_string_at(root, path).map(|value| value.trim().to_ascii_lowercase())
}

fn bool_at(root: &Value, path: &[&str], default: bool) -> bool {
    value_at(root, path)
        .and_then(|value| match value {
            Value::Bool(value) => Some(*value),
            Value::Number(value) => value.as_i64().map(|number| number != 0),
            Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "y" | "on" => Some(true),
                "0" | "false" | "no" | "n" | "off" => Some(false),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(default)
}

fn value_at<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut cursor = root;
    for part in path {
        cursor = cursor
            .as_mapping()?
            .get(Value::String((*part).to_owned()))?;
    }
    Some(cursor)
}

fn slugify(value: Option<&str>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    let mut out = String::new();
    let mut last_dash = false;
    for ch in value.trim().chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_owned()
}

fn resolve_prediction_artifact_dir(raw_path: &Path) -> Result<PathBuf, String> {
    if raw_path.is_dir() && raw_path.join(PREDICTION_METADATA_FILENAME).is_file() {
        return Ok(raw_path.to_path_buf());
    }
    let nested = raw_path.join(PREDICTION_ARTIFACT_DIRNAME);
    if nested.is_dir() && nested.join(PREDICTION_METADATA_FILENAME).is_file() {
        return Ok(nested);
    }
    Err(format!(
        "prediction artifact directory not found under {}; expected {} directly or under {}/",
        raw_path.display(),
        PREDICTION_METADATA_FILENAME,
        PREDICTION_ARTIFACT_DIRNAME
    ))
}

fn copy_prediction_artifacts(raw_source: &Path, target: &Path) -> Result<(), String> {
    let source = resolve_prediction_artifact_dir(raw_source)?;
    if target.exists()
        && source
            .canonicalize()
            .ok()
            .zip(target.canonicalize().ok())
            .is_some_and(|(left, right)| left == right)
    {
        return Ok(());
    }
    copy_dir_recursive(&source, target)
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<(), String> {
    fs::create_dir_all(target)
        .map_err(|err| format!("failed to create {}: {err}", target.display()))?;
    for entry in
        fs::read_dir(source).map_err(|err| format!("failed to read {}: {err}", source.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read {}: {err}", source.display()))?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_recursive(&source_path, &target_path)?;
        } else {
            fs::copy(&source_path, &target_path).map_err(|err| {
                format!(
                    "failed to copy {} to {}: {err}",
                    source_path.display(),
                    target_path.display()
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_python_bundle_args_needed_by_artifact_batch() {
        let args = vec![
            "--config".to_owned(),
            "configs/bt.yaml".to_owned(),
            "--config-is-snapshot".to_owned(),
            "--load-predictions-dir".to_owned(),
            "runs/train/prediction_artifacts".to_owned(),
            "--store-dir".to_owned(),
            "artifact_runs".to_owned(),
            "--run-tag".to_owned(),
            "artifact-rebuild-lgbm-train_a__bt_x".to_owned(),
            "--model".to_owned(),
            "lgbm".to_owned(),
            "--signal-horizon".to_owned(),
            "20".to_owned(),
            "--retrain-step".to_owned(),
            "10".to_owned(),
            "--train-days".to_owned(),
            "242".to_owned(),
            "--valid-days".to_owned(),
            "10".to_owned(),
            "--set".to_owned(),
            "strategy.score_fusion.secondary_predictions_dir=runs/secondary".to_owned(),
            "--skip-reference-baselines".to_owned(),
        ];
        let parsed = parse_native_rolling_bundle_args(&args).unwrap();
        assert!(parsed.config_is_snapshot);
        assert_eq!(
            parsed.load_predictions_dir.unwrap(),
            PathBuf::from("runs/train/prediction_artifacts")
        );
        assert_eq!(parsed.retrain_step, Some(10));
        assert!(parsed.skip_reference_baselines);
        assert_eq!(parsed.set_overrides.len(), 1);
    }

    #[test]
    fn applies_dotted_overrides_with_yaml_scalars() {
        let mut config = serde_yaml::from_str::<Value>("strategy:\n  topk: 5\n").unwrap();
        apply_dotted_override(
            &mut config,
            "strategy.score_fusion.enabled",
            Value::Bool(true),
        )
        .unwrap();
        apply_dotted_override(
            &mut config,
            "strategy.score_fusion.secondary_predictions_dir",
            Value::String("runs/secondary".to_owned()),
        )
        .unwrap();
        assert!(bool_at(
            &config,
            &["strategy", "score_fusion", "enabled"],
            false
        ));
        assert_eq!(
            raw_string_at(
                &config,
                &["strategy", "score_fusion", "secondary_predictions_dir"]
            )
            .unwrap(),
            "runs/secondary"
        );
    }

    #[test]
    fn builds_python_compatible_run_id_slug() {
        let config = serde_yaml::from_str::<Value>(
            r#"
strategy:
  topk: 10
  n_drop: 2
  weighting: score_softmax
  score_transform: rank_pct
backtest:
  rebalance_freq: 10
model:
  name: lgbm
"#,
        )
        .unwrap();
        let args = NativeRollingBundleArgs {
            run_tag: Some("artifact-rebuild-lgbm-train_a__bt_x".to_owned()),
            ..NativeRollingBundleArgs::default()
        };
        let run_id = build_run_id(&config, &args, "lgbm");
        assert!(run_id.contains("__native__rolling__lgbm__"));
        assert!(run_id.contains("top10_drop2_wscore_softmax_strank_pct_keepna_minsna_reb10"));
        assert!(run_id.ends_with("__artifact-rebuild-lgbm-train-a-bt-x"));
    }
}
