use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{SystemTime, UNIX_EPOCH};

const PREDICTION_ARTIFACT_DIRNAME: &str = "prediction_artifacts";
const PREDICTION_METADATA_FILENAME: &str = "metadata.json";

fn usage() -> &'static str {
    "\
ai4stock-experiment: Rust batch runner for AI4Stock2 experiment sweeps

Usage:
  ai4stock-experiment batch --experiment-profile <NAME> [OPTIONS]

Options:
  --config <PATH>             Runtime config path. Default: configs/config.yaml.
  --pipeline <rolling>        Target pipeline. Default: rolling.
  --experiment-profile <NAME> Base experiment profile.
  --model-profile <NAME>      Override model profile.
  --feature-profile <NAME>    Override feature profile.
  --data-source <NAME>        Override data source.
  --set <KEY=VALUE>           Fixed dotted override for every run. Repeatable.
  --sweep <KEY=VALUE>         Sweep override. Repeatable. Values accept YAML lists or {a,b}.
  --case <KEY=VALUE>...       Explicit grouped overrides for one run. Repeatable.
  --run-tag-prefix <TEXT>     Prefix added before per-run override suffix.
  --store-dir <PATH>          Override local experiment store root.
  --dedupe-predictions        Train once for identical prediction-producing configs, then replay.
  --skip-reference-baselines  Forward to child rolling runs.
  --repo-root <PATH>          Repo root. Default: current directory.
  --dry-run                   Print expanded commands without executing.
  --fail-fast                 Stop on the first failed child run.
  -h, --help                  Show this help.
"
}

#[derive(Debug, Clone)]
struct Options {
    config: PathBuf,
    pipeline: String,
    experiment_profile: String,
    model_profile: Option<String>,
    feature_profile: Option<String>,
    data_source: Option<String>,
    set_overrides: Vec<String>,
    sweep_overrides: Vec<String>,
    case_overrides: Vec<Vec<String>>,
    run_tag_prefix: Option<String>,
    store_dir: Option<String>,
    dedupe_predictions: bool,
    skip_reference_baselines: bool,
    repo_root: PathBuf,
    dry_run: bool,
    fail_fast: bool,
}

#[derive(Debug, Clone)]
struct BatchRun {
    overrides: BTreeMap<String, YamlValue>,
    run_tag: Option<String>,
    command: Vec<String>,
}

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match run(&args) {
        Ok(code) => code,
        Err(message) if message == usage() => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(2)
        }
    }
}

fn run(args: &[String]) -> Result<ExitCode, String> {
    let Some(command) = args.first() else {
        return Err(usage().to_owned());
    };
    match command.as_str() {
        "-h" | "--help" => Err(usage().to_owned()),
        "batch" => run_batch(&args[1..]),
        other => Err(format!("unknown command: {other}\n\n{}", usage())),
    }
}

fn run_batch(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_options(args)?;
    if options.pipeline != "rolling" {
        return Err(format!(
            "unsupported pipeline: {}. Only rolling is supported.",
            options.pipeline
        ));
    }
    let explicit_cases = resolve_explicit_cases(&options.case_overrides)?;
    let (sweep_map, overrides) = if explicit_cases.is_empty() {
        let sweep_map = resolve_sweep_map(&options)?;
        let overrides = expand_sweep_grid(&sweep_map);
        (sweep_map, overrides)
    } else {
        (BTreeMap::new(), explicit_cases)
    };
    let runs = build_runs(&options, &overrides)?;

    println!("[*] Batch pipeline: {}", options.pipeline);
    println!(
        "[*] Base experiment profile: {}",
        options.experiment_profile
    );
    println!("[*] Expanded runs: {}", runs.len());
    if !options.case_overrides.is_empty() {
        println!("[*] Explicit cases:");
        for (index, run) in runs.iter().enumerate() {
            println!(
                "    case {}: {}",
                index + 1,
                render_override_items(&run.overrides)
            );
        }
    } else if !sweep_map.is_empty() {
        for (key, values) in &sweep_map {
            println!("    {key} -> {}", render_yaml_list(values));
        }
    }

    let mut failures: Vec<(usize, i32)> = Vec::new();
    let mut prediction_artifacts_by_fingerprint: BTreeMap<String, PathBuf> = BTreeMap::new();
    for (index, run) in runs.iter().enumerate() {
        let mut command = run.command.clone();
        let mut first_deduped_artifact = false;
        let mut resolved_cfg = None;
        let mut fingerprint = String::new();
        if options.dedupe_predictions {
            let cfg = resolve_run_config(&options, &run.overrides)?;
            fingerprint = prediction_fingerprint(&cfg)?;
            if let Some(artifact_dir) = prediction_artifacts_by_fingerprint.get(&fingerprint) {
                command.extend([
                    "--load-predictions-dir".to_owned(),
                    path_to_string(artifact_dir),
                ]);
            } else {
                command.push("--save-predictions".to_owned());
                first_deduped_artifact = true;
            }
            resolved_cfg = Some(cfg);
        }
        println!();
        println!(
            "[{}/{}] {}",
            index + 1,
            runs.len(),
            render_command(&command)
        );
        if options.dry_run {
            if first_deduped_artifact && !fingerprint.is_empty() {
                prediction_artifacts_by_fingerprint.insert(
                    fingerprint,
                    PathBuf::from(format!("<prediction_artifact_from_run_{}>", index + 1)),
                );
            }
            continue;
        }
        let started_at = now_epoch_seconds();
        let status = Command::new(&command[0])
            .args(&command[1..])
            .current_dir(&options.repo_root)
            .status()
            .map_err(|err| format!("failed to run {}: {err}", command[0]))?;
        let code = status.code().unwrap_or(1);
        if code != 0 {
            failures.push((index + 1, code));
            if options.fail_fast {
                return Ok(ExitCode::from(exit_code_u8(code)));
            }
        } else if first_deduped_artifact && !fingerprint.is_empty() {
            let cfg = resolved_cfg
                .as_ref()
                .ok_or_else(|| "internal error: missing resolved config for dedupe".to_owned())?;
            let artifact_dir =
                find_prediction_artifact_dir(cfg, &options, run.run_tag.as_deref(), started_at)?;
            prediction_artifacts_by_fingerprint.insert(fingerprint, artifact_dir);
        }
    }
    if options.dry_run {
        println!();
        println!("[+] Dry run completed.");
        return Ok(ExitCode::SUCCESS);
    }
    if !failures.is_empty() {
        println!();
        println!("[!] Batch completed with failures:");
        for (index, code) in failures {
            println!("    run {index}: exit_code={code}");
        }
        return Ok(ExitCode::FAILURE);
    }
    println!();
    println!("[+] Batch completed successfully.");
    Ok(ExitCode::SUCCESS)
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        config: PathBuf::from("configs/config.yaml"),
        pipeline: "rolling".to_owned(),
        experiment_profile: String::new(),
        model_profile: None,
        feature_profile: None,
        data_source: None,
        set_overrides: Vec::new(),
        sweep_overrides: Vec::new(),
        case_overrides: Vec::new(),
        run_tag_prefix: None,
        store_dir: None,
        dedupe_predictions: false,
        skip_reference_baselines: false,
        repo_root: env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        dry_run: false,
        fail_fast: false,
    };

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--config" => options.config = PathBuf::from(next_arg(args, &mut index, "--config")?),
            value if value.starts_with("--config=") => {
                options.config = PathBuf::from(split_value(value, "--config")?)
            }
            "--pipeline" => options.pipeline = next_arg(args, &mut index, "--pipeline")?,
            value if value.starts_with("--pipeline=") => {
                options.pipeline = split_value(value, "--pipeline")?
            }
            "--experiment-profile" => {
                options.experiment_profile = next_arg(args, &mut index, "--experiment-profile")?
            }
            value if value.starts_with("--experiment-profile=") => {
                options.experiment_profile = split_value(value, "--experiment-profile")?
            }
            "--model-profile" => {
                options.model_profile = Some(next_arg(args, &mut index, "--model-profile")?)
            }
            value if value.starts_with("--model-profile=") => {
                options.model_profile = Some(split_value(value, "--model-profile")?)
            }
            "--feature-profile" => {
                options.feature_profile = Some(next_arg(args, &mut index, "--feature-profile")?)
            }
            value if value.starts_with("--feature-profile=") => {
                options.feature_profile = Some(split_value(value, "--feature-profile")?)
            }
            "--data-source" => {
                options.data_source = Some(next_arg(args, &mut index, "--data-source")?)
            }
            value if value.starts_with("--data-source=") => {
                options.data_source = Some(split_value(value, "--data-source")?)
            }
            "--set" => options
                .set_overrides
                .push(next_arg(args, &mut index, "--set")?),
            value if value.starts_with("--set=") => {
                options.set_overrides.push(split_value(value, "--set")?)
            }
            "--sweep" => options
                .sweep_overrides
                .push(next_arg(args, &mut index, "--sweep")?),
            value if value.starts_with("--sweep=") => {
                options.sweep_overrides.push(split_value(value, "--sweep")?)
            }
            "--case" => {
                index += 1;
                let mut group = Vec::new();
                while index < args.len() && !args[index].starts_with("--") {
                    group.push(args[index].clone());
                    index += 1;
                }
                if group.is_empty() {
                    return Err("--case requires at least one KEY=VALUE item".to_owned());
                }
                options.case_overrides.push(group);
                continue;
            }
            value if value.starts_with("--case=") => {
                options
                    .case_overrides
                    .push(vec![split_value(value, "--case")?]);
            }
            "--run-tag-prefix" => {
                options.run_tag_prefix = Some(next_arg(args, &mut index, "--run-tag-prefix")?)
            }
            value if value.starts_with("--run-tag-prefix=") => {
                options.run_tag_prefix = Some(split_value(value, "--run-tag-prefix")?)
            }
            "--store-dir" => options.store_dir = Some(next_arg(args, &mut index, "--store-dir")?),
            value if value.starts_with("--store-dir=") => {
                options.store_dir = Some(split_value(value, "--store-dir")?)
            }
            "--dedupe-predictions" => options.dedupe_predictions = true,
            "--skip-reference-baselines" => options.skip_reference_baselines = true,
            "--repo-root" => {
                options.repo_root = PathBuf::from(next_arg(args, &mut index, "--repo-root")?)
            }
            value if value.starts_with("--repo-root=") => {
                options.repo_root = PathBuf::from(split_value(value, "--repo-root")?)
            }
            "--dry-run" => options.dry_run = true,
            "--fail-fast" => options.fail_fast = true,
            other => return Err(format!("unknown batch option: {other}\n\n{}", usage())),
        }
        index += 1;
    }
    if options.experiment_profile.trim().is_empty() {
        return Err("--experiment-profile is required".to_owned());
    }
    Ok(options)
}

fn next_arg(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn split_value(value: &str, option: &str) -> Result<String, String> {
    let raw = value
        .split_once('=')
        .map(|(_, right)| right)
        .unwrap_or_default();
    if raw.is_empty() {
        return Err(format!("missing value for {option}"));
    }
    Ok(raw.to_owned())
}

fn rust_binary_command(binary_name: &str, env_var: &str) -> Vec<String> {
    env::var(env_var)
        .ok()
        .map(|value| split_command(&value))
        .filter(|parts| !parts.is_empty())
        .unwrap_or_else(|| {
            vec![
                "cargo".to_owned(),
                "run".to_owned(),
                "--bin".to_owned(),
                binary_name.to_owned(),
                "--".to_owned(),
            ]
        })
}

fn split_command(raw: &str) -> Vec<String> {
    raw.split_whitespace()
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn resolve_sweep_map(options: &Options) -> Result<BTreeMap<String, Vec<YamlValue>>, String> {
    let profile = resolve_named_profile(
        &options.repo_root.join("configs/experiment_profiles.yaml"),
        &options.experiment_profile,
        "experiment",
    )?;
    let mut sweep_map = BTreeMap::new();
    if let Some(sweep) = yaml_mapping_get_value(&profile, "sweep") {
        flatten_sweep_mapping(sweep, "", &mut sweep_map)?;
    }
    for raw in &options.sweep_overrides {
        let (key, values) = parse_sweep_arg(raw)?;
        sweep_map.insert(key, values);
    }
    Ok(sweep_map)
}

fn resolve_explicit_cases(
    raw_cases: &[Vec<String>],
) -> Result<Vec<BTreeMap<String, YamlValue>>, String> {
    let mut cases = Vec::new();
    for raw_group in raw_cases {
        let mut case = BTreeMap::new();
        for raw in raw_group {
            let (key, value) = parse_key_value_arg(raw, "Override")?;
            case.insert(key, value);
        }
        if !case.is_empty() {
            cases.push(case);
        }
    }
    Ok(cases)
}

fn parse_sweep_arg(raw: &str) -> Result<(String, Vec<YamlValue>), String> {
    let (key, value) = parse_raw_key_value(raw, "Sweep")?;
    let values = parse_sweep_values(value)?;
    if values.is_empty() {
        return Err(format!("Sweep values for {key} must be non-empty"));
    }
    Ok((key, values))
}

fn parse_key_value_arg(raw: &str, label: &str) -> Result<(String, YamlValue), String> {
    let (key, value) = parse_raw_key_value(raw, label)?;
    Ok((key, parse_yaml_scalar(value)))
}

fn parse_raw_key_value<'a>(raw: &'a str, label: &str) -> Result<(String, &'a str), String> {
    let Some((key, value)) = raw.trim().split_once('=') else {
        return Err(format!("{label} must be in key=value form, got: {raw}"));
    };
    let key = key.trim().to_owned();
    if key.is_empty() {
        return Err(format!("{label} key must be non-empty, got: {raw}"));
    }
    Ok((key, value.trim()))
}

fn parse_yaml_scalar(raw: &str) -> YamlValue {
    serde_yaml::from_str::<YamlValue>(raw).unwrap_or_else(|_| YamlValue::String(raw.to_owned()))
}

fn parse_sweep_values(raw: &str) -> Result<Vec<YamlValue>, String> {
    let text = raw.trim();
    if text.starts_with('{') && text.ends_with('}') {
        let inner = text[1..text.len() - 1].trim();
        if inner.is_empty() {
            return Ok(Vec::new());
        }
        return Ok(inner
            .split(',')
            .map(|part| parse_yaml_scalar(part.trim()))
            .collect());
    }
    if text.starts_with('[') && text.ends_with(']') {
        let parsed = parse_yaml_scalar(text);
        if let YamlValue::Sequence(items) = parsed {
            return Ok(items);
        }
    }
    Ok(vec![parse_yaml_scalar(text)])
}

fn flatten_sweep_mapping(
    value: &YamlValue,
    prefix: &str,
    out: &mut BTreeMap<String, Vec<YamlValue>>,
) -> Result<(), String> {
    let mapping = value
        .as_mapping()
        .ok_or_else(|| "experiment profile sweep must be a mapping".to_owned())?;
    for (raw_key, raw_value) in mapping {
        let key = raw_key
            .as_str()
            .ok_or_else(|| "sweep keys must be strings".to_owned())?;
        let dotted_key = if prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{prefix}.{key}")
        };
        if raw_value.is_mapping() {
            flatten_sweep_mapping(raw_value, &dotted_key, out)?;
        } else {
            let values = yaml_sweep_values(raw_value)?;
            if values.is_empty() {
                return Err(format!("Sweep values for {dotted_key} must be non-empty"));
            }
            out.insert(dotted_key, values);
        }
    }
    Ok(())
}

fn yaml_sweep_values(value: &YamlValue) -> Result<Vec<YamlValue>, String> {
    match value {
        YamlValue::Sequence(items) => Ok(items.clone()),
        YamlValue::String(text) => parse_sweep_values(text),
        other => Ok(vec![other.clone()]),
    }
}

fn expand_sweep_grid(
    sweep_map: &BTreeMap<String, Vec<YamlValue>>,
) -> Vec<BTreeMap<String, YamlValue>> {
    if sweep_map.is_empty() {
        return vec![BTreeMap::new()];
    }
    let items = sweep_map.iter().collect::<Vec<_>>();
    let mut runs = Vec::new();
    let mut current = BTreeMap::new();
    expand_sweep_grid_recursive(&items, 0, &mut current, &mut runs);
    runs
}

fn expand_sweep_grid_recursive(
    items: &[(&String, &Vec<YamlValue>)],
    index: usize,
    current: &mut BTreeMap<String, YamlValue>,
    runs: &mut Vec<BTreeMap<String, YamlValue>>,
) {
    if index >= items.len() {
        runs.push(current.clone());
        return;
    }
    let (key, values) = items[index];
    for value in values.iter().cloned() {
        current.insert((*key).clone(), value);
        expand_sweep_grid_recursive(items, index + 1, current, runs);
    }
    current.remove(key);
}

fn build_runs(
    options: &Options,
    overrides: &[BTreeMap<String, YamlValue>],
) -> Result<Vec<BatchRun>, String> {
    overrides
        .iter()
        .map(|run_overrides| {
            let run_tag = build_run_tag(options.run_tag_prefix.as_deref(), run_overrides);
            let command = build_run_command(options, run_overrides, run_tag.as_deref())?;
            Ok(BatchRun {
                overrides: run_overrides.clone(),
                run_tag,
                command,
            })
        })
        .collect()
}

fn build_run_command(
    options: &Options,
    overrides: &BTreeMap<String, YamlValue>,
    run_tag: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut command = rust_binary_command("ai4stock-train", "AI4STOCK_TRAIN_BIN");
    command.push("rolling-lgbm".to_owned());
    command.extend([
        "--config".to_owned(),
        path_to_string(&options.config),
        "--experiment-profile".to_owned(),
        options.experiment_profile.clone(),
    ]);
    append_option(
        &mut command,
        "--model-profile",
        options.model_profile.as_deref(),
    );
    append_option(
        &mut command,
        "--feature-profile",
        options.feature_profile.as_deref(),
    );
    append_option(
        &mut command,
        "--data-source",
        options.data_source.as_deref(),
    );
    append_option(&mut command, "--store-dir", options.store_dir.as_deref());
    if options.skip_reference_baselines {
        command.push("--skip-reference-baselines".to_owned());
    }
    for item in &options.set_overrides {
        command.extend(["--set".to_owned(), item.clone()]);
    }
    for (key, value) in overrides {
        command.extend([
            "--set".to_owned(),
            format!("{key}={}", yaml_value_to_cli(value)?),
        ]);
    }
    if let Some(run_tag) = run_tag {
        command.extend(["--run-tag".to_owned(), run_tag.to_owned()]);
    }
    Ok(command)
}

fn resolve_run_config(
    options: &Options,
    overrides: &BTreeMap<String, YamlValue>,
) -> Result<YamlValue, String> {
    let mut cfg = read_yaml_file(resolve_repo_path(&options.repo_root, &options.config))?;
    ensure_mapping(&mut cfg);

    let experiment = resolve_profile_with_path(
        &options.repo_root.join("configs/experiment_profiles.yaml"),
        &options.experiment_profile,
        "experiment",
    )?;
    let mut experiment_cfg = experiment.config;
    remove_yaml_mapping_key(&mut experiment_cfg, "name");
    remove_yaml_mapping_key(&mut experiment_cfg, "path");
    remove_yaml_mapping_key(&mut experiment_cfg, "sweep");
    deep_merge_yaml(&mut cfg, experiment_cfg);
    set_yaml_dotted(
        &mut cfg,
        "experiment.profile",
        YamlValue::String(options.experiment_profile.clone()),
    )?;
    set_yaml_dotted(
        &mut cfg,
        "experiment.profile_path",
        YamlValue::String(experiment.path),
    )?;

    let model_profile_name = options
        .model_profile
        .clone()
        .or_else(|| yaml_path_string(&cfg, &["model", "profile"]))
        .or_else(|| yaml_path_string(&cfg, &["runtime", "default_model_profile"]))
        .or_else(|| default_model_profile(&options.repo_root).ok())
        .ok_or_else(|| "model profile could not be resolved".to_owned())?;
    let model = resolve_profile_with_path(
        &options.repo_root.join("configs/model_profiles.yaml"),
        &model_profile_name,
        "model",
    )?;
    let mut model_cfg = model.config;
    remove_yaml_mapping_key(&mut model_cfg, "name");
    remove_yaml_mapping_key(&mut model_cfg, "path");
    deep_merge_yaml(&mut cfg, model_cfg);
    set_yaml_dotted(
        &mut cfg,
        "model.profile",
        YamlValue::String(model_profile_name),
    )?;
    set_yaml_dotted(
        &mut cfg,
        "model.profile_path",
        YamlValue::String(model.path),
    )?;
    set_yaml_dotted(
        &mut cfg,
        "runtime.config_path",
        YamlValue::String(path_to_string(&options.config)),
    )?;

    if let Some(feature_profile) = &options.feature_profile {
        set_yaml_dotted(
            &mut cfg,
            "features.profile",
            YamlValue::String(feature_profile.clone()),
        )?;
    }
    if let Some(data_source) = &options.data_source {
        set_yaml_dotted(
            &mut cfg,
            "data.source",
            YamlValue::String(data_source.clone()),
        )?;
    }
    if let Some(store_dir) = &options.store_dir {
        set_yaml_dotted(
            &mut cfg,
            "artifacts.store_dir",
            YamlValue::String(store_dir.clone()),
        )?;
    }
    for raw in &options.set_overrides {
        let (key, value) = parse_key_value_arg(raw, "Override")?;
        set_yaml_dotted(&mut cfg, &key, value)?;
    }
    for (key, value) in overrides {
        set_yaml_dotted(&mut cfg, key, value.clone())?;
    }
    Ok(cfg)
}

fn prediction_fingerprint(cfg: &YamlValue) -> Result<String, String> {
    let mut relevant = BTreeMap::<String, serde_json::Value>::new();
    relevant.insert(
        "data".to_owned(),
        yaml_path_json(cfg, &["data"], json_object()),
    );
    relevant.insert(
        "native".to_owned(),
        yaml_path_json(cfg, &["native"], json_object()),
    );
    relevant.insert(
        "universe".to_owned(),
        yaml_path_json(cfg, &["universe"], serde_json::Value::String(String::new())),
    );
    relevant.insert(
        "time".to_owned(),
        yaml_path_json(cfg, &["time"], json_object()),
    );
    relevant.insert(
        "features".to_owned(),
        yaml_path_json(cfg, &["features"], json_object()),
    );
    relevant.insert(
        "model".to_owned(),
        yaml_path_json(cfg, &["model"], json_object()),
    );
    relevant.insert(
        "label".to_owned(),
        yaml_path_json(cfg, &["label"], json_object()),
    );
    relevant.insert(
        "rolling".to_owned(),
        yaml_path_json(cfg, &["rolling"], json_object()),
    );
    relevant.insert(
        "prediction_fusion".to_owned(),
        yaml_path_json(cfg, &["prediction_fusion"], json_object()),
    );
    relevant.insert(
        "score_fusion".to_owned(),
        yaml_path_json(cfg, &["score_fusion"], json_object()),
    );
    relevant.insert(
        "backtest_benchmark".to_owned(),
        yaml_path_json(cfg, &["backtest", "benchmark"], serde_json::Value::Null),
    );
    let model_name = yaml_path_string(cfg, &["model", "name"])
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if model_name == "lgbm" {
        relevant.insert("effective_lgbm".to_owned(), effective_lgbm_config(cfg)?);
    } else if model_name == "formula_score" {
        relevant.insert(
            "formula_score".to_owned(),
            yaml_path_json(cfg, &["formula_score"], json_object()),
        );
    }
    serde_json::to_string(&relevant)
        .map_err(|err| format!("failed to encode prediction fingerprint: {err}"))
}

fn effective_lgbm_config(cfg: &YamlValue) -> Result<serde_json::Value, String> {
    let mut lgbm = yaml_path_json(cfg, &["lgbm"], json_object());
    if !lgbm.is_object() {
        return Ok(lgbm);
    }
    let object = lgbm.as_object_mut().expect("checked object");
    if !object.contains_key("early_stop") {
        if let Some(value) = yaml_path(cfg, &["model", "early_stop"]) {
            object.insert("early_stop".to_owned(), yaml_to_json(value)?);
        }
    }
    if !object.contains_key("validation_topk") {
        if let Some(value) = yaml_path(cfg, &["strategy", "topk"]) {
            object.insert("validation_topk".to_owned(), yaml_to_json(value)?);
        }
    }
    Ok(lgbm)
}

fn find_prediction_artifact_dir(
    cfg: &YamlValue,
    options: &Options,
    run_tag: Option<&str>,
    started_at: f64,
) -> Result<PathBuf, String> {
    let model_name = yaml_path_string(cfg, &["model", "name"]).unwrap_or_else(|| "lgbm".to_owned());
    let store_dir = options
        .store_dir
        .clone()
        .or_else(|| yaml_path_string(cfg, &["artifacts", "store_dir"]))
        .unwrap_or_else(|| "results/experiments".to_owned());
    let run_root = resolve_repo_path(&options.repo_root, Path::new(&store_dir))
        .join("native")
        .join("rolling")
        .join(model_name);
    if !run_root.exists() {
        return Err(format!(
            "Run store directory not found after training: {}",
            run_root.display()
        ));
    }
    let suffix = run_tag
        .map(slugify)
        .filter(|slug| !slug.is_empty())
        .map(|slug| format!("__{slug}"));
    let mut candidates: Vec<(f64, PathBuf)> = Vec::new();
    for entry in fs::read_dir(&run_root)
        .map_err(|err| format!("failed to list {}: {err}", run_root.display()))?
    {
        let entry = entry.map_err(|err| format!("failed to read {}: {err}", run_root.display()))?;
        let run_dir = entry.path();
        if !run_dir.is_dir() {
            continue;
        }
        if let Some(suffix) = &suffix {
            let name = run_dir
                .file_name()
                .map(|name| name.to_string_lossy())
                .unwrap_or_default();
            if !name.ends_with(suffix) {
                continue;
            }
        }
        let artifact_dir = run_dir.join(PREDICTION_ARTIFACT_DIRNAME);
        let metadata_path = artifact_dir.join(PREDICTION_METADATA_FILENAME);
        let Ok(metadata) = fs::metadata(&metadata_path) else {
            continue;
        };
        let modified = metadata
            .modified()
            .map_err(|err| format!("failed to read mtime {}: {err}", metadata_path.display()))?;
        let modified_epoch = system_time_epoch_seconds(modified);
        if modified_epoch + 1.0 < started_at {
            continue;
        }
        candidates.push((modified_epoch, artifact_dir));
    }
    candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
    candidates
        .into_iter()
        .map(|(_, path)| path)
        .next()
        .ok_or_else(|| {
            format!(
                "Saved prediction artifact not found after training. run_root={}, run_tag={}",
                run_root.display(),
                run_tag.unwrap_or("")
            )
        })
}

fn append_option(command: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        if !value.is_empty() {
            command.extend([flag.to_owned(), value.to_owned()]);
        }
    }
}

fn build_run_tag(prefix: Option<&str>, overrides: &BTreeMap<String, YamlValue>) -> Option<String> {
    let suffix = build_override_tag(overrides);
    let prefix = prefix.filter(|item| !item.is_empty());
    if let Some(prefix) = prefix {
        if suffix.is_empty() {
            Some(prefix.to_owned())
        } else {
            Some(format!("{prefix}__{suffix}"))
        }
    } else if suffix.is_empty() {
        None
    } else {
        Some(suffix)
    }
}

fn build_override_tag(overrides: &BTreeMap<String, YamlValue>) -> String {
    overrides
        .iter()
        .map(|(key, value)| {
            format!(
                "{}-{}",
                slugify_override_key(key),
                slugify_override_value(&yaml_value_plain(value))
            )
        })
        .collect::<Vec<_>>()
        .join("__")
}

fn slugify_override_key(key: &str) -> String {
    key.trim().replace('.', "-").replace('_', "-")
}

fn slugify_override_value(value: &str) -> String {
    let mut safe = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    while safe.contains("--") {
        safe = safe.replace("--", "-");
    }
    let trimmed = safe.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "na".to_owned()
    } else {
        trimmed
    }
}

fn yaml_value_to_cli(value: &YamlValue) -> Result<String, String> {
    Ok(match value {
        YamlValue::String(text) => text.clone(),
        YamlValue::Bool(value) => {
            if *value {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        YamlValue::Number(number) => number.to_string(),
        YamlValue::Null => "null".to_owned(),
        _ => serde_yaml::to_string(value)
            .map_err(|err| format!("failed to render YAML value: {err}"))?
            .trim()
            .to_owned(),
    })
}

fn yaml_value_plain(value: &YamlValue) -> String {
    match value {
        YamlValue::String(text) => text.clone(),
        YamlValue::Bool(value) => value.to_string(),
        YamlValue::Number(number) => number.to_string(),
        YamlValue::Null => "null".to_owned(),
        _ => serde_yaml::to_string(value)
            .unwrap_or_else(|_| format!("{value:?}"))
            .trim()
            .to_owned(),
    }
}

fn render_override_items(overrides: &BTreeMap<String, YamlValue>) -> String {
    overrides
        .iter()
        .map(|(key, value)| format!("{key}={}", yaml_value_plain(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_yaml_list(values: &[YamlValue]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(yaml_value_plain)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_command(command: &[String]) -> String {
    command
        .iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '='))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn resolve_named_profile(
    profile_config_path: &Path,
    profile_name: &str,
    profile_kind: &str,
) -> Result<YamlValue, String> {
    Ok(resolve_profile_with_path(profile_config_path, profile_name, profile_kind)?.config)
}

#[derive(Debug, Clone)]
struct ResolvedProfile {
    config: YamlValue,
    path: String,
}

fn resolve_profile_with_path(
    profile_config_path: &Path,
    profile_name: &str,
    profile_kind: &str,
) -> Result<ResolvedProfile, String> {
    let profile_data = read_yaml_file(profile_config_path)?;
    let profiles = yaml_mapping_get_value(&profile_data, "profiles")
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| format!("{} has no profiles mapping", profile_config_path.display()))?;
    resolve_profile_from_mapping(
        profile_config_path,
        profiles,
        profile_name,
        profile_kind,
        &mut Vec::new(),
    )
}

fn resolve_profile_from_mapping(
    profile_config_path: &Path,
    profiles: &YamlMapping,
    profile_name: &str,
    profile_kind: &str,
    stack: &mut Vec<String>,
) -> Result<ResolvedProfile, String> {
    if stack.iter().any(|item| item == profile_name) {
        return Err(format!(
            "cyclic {profile_kind} profile extends chain: {} -> {profile_name}",
            stack.join(" -> ")
        ));
    }
    let entry = profiles
        .get(YamlValue::String(profile_name.to_owned()))
        .ok_or_else(|| format!("unknown {profile_kind} profile: {profile_name}"))?
        .clone();
    let entry_mapping = entry
        .as_mapping()
        .ok_or_else(|| format!("{profile_kind} profile {profile_name} must be a mapping"))?
        .clone();
    let profile_dir = profile_config_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let repo_root = profile_dir.parent().unwrap_or_else(|| Path::new("."));
    let inline_path = format!("{}::{profile_name}", profile_config_path.display());
    let (mut merged, source_path) =
        if let Some(path) = yaml_mapping_get(&entry_mapping, "path").and_then(YamlValue::as_str) {
            let resolved_path = resolve_repo_path(repo_root, Path::new(path));
            (
                read_yaml_file(&resolved_path)?,
                path_to_string(&resolved_path),
            )
        } else {
            (YamlValue::Mapping(YamlMapping::new()), inline_path)
        };
    let extends_name = yaml_mapping_get(&entry_mapping, "extends")
        .and_then(YamlValue::as_str)
        .map(str::to_owned);
    if let Some(parent_name) = extends_name {
        stack.push(profile_name.to_owned());
        let parent = resolve_profile_from_mapping(
            profile_config_path,
            profiles,
            &parent_name,
            profile_kind,
            stack,
        )?;
        stack.pop();
        let mut parent_config = parent.config;
        remove_yaml_mapping_key(&mut parent_config, "name");
        remove_yaml_mapping_key(&mut parent_config, "path");
        deep_merge_yaml(&mut parent_config, merged);
        merged = parent_config;
    }
    let mut inline = YamlValue::Mapping(entry_mapping);
    remove_yaml_mapping_key(&mut inline, "path");
    remove_yaml_mapping_key(&mut inline, "extends");
    deep_merge_yaml(&mut merged, inline);
    set_yaml_dotted(
        &mut merged,
        "name",
        YamlValue::String(profile_name.to_owned()),
    )?;
    set_yaml_dotted(&mut merged, "path", YamlValue::String(source_path.clone()))?;
    Ok(ResolvedProfile {
        config: merged,
        path: source_path,
    })
}

fn read_yaml_file(path: impl AsRef<Path>) -> Result<YamlValue, String> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_yaml::from_str(&raw).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn deep_merge_yaml(base: &mut YamlValue, overlay: YamlValue) {
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
        (base_slot, overlay_value) => {
            *base_slot = overlay_value;
        }
    }
}

fn ensure_mapping(value: &mut YamlValue) {
    if !value.is_mapping() {
        *value = YamlValue::Mapping(YamlMapping::new());
    }
}

fn set_yaml_dotted(cfg: &mut YamlValue, dotted_key: &str, value: YamlValue) -> Result<(), String> {
    let parts = dotted_key
        .split('.')
        .filter(|part| !part.trim().is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(format!("invalid dotted override key: {dotted_key}"));
    }
    let mut cursor = cfg;
    for part in &parts[..parts.len() - 1] {
        if !cursor.is_mapping() {
            *cursor = YamlValue::Mapping(YamlMapping::new());
        }
        let mapping = cursor.as_mapping_mut().expect("mapping just initialized");
        cursor = mapping
            .entry(YamlValue::String((*part).to_owned()))
            .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    }
    if !cursor.is_mapping() {
        *cursor = YamlValue::Mapping(YamlMapping::new());
    }
    cursor
        .as_mapping_mut()
        .expect("mapping just initialized")
        .insert(YamlValue::String(parts[parts.len() - 1].to_owned()), value);
    Ok(())
}

fn yaml_path<'a>(value: &'a YamlValue, path: &[&str]) -> Option<&'a YamlValue> {
    let mut cursor = value;
    for key in path {
        cursor = yaml_mapping_get_value(cursor, key)?;
    }
    Some(cursor)
}

fn yaml_path_string(value: &YamlValue, path: &[&str]) -> Option<String> {
    yaml_path(value, path)
        .and_then(YamlValue::as_str)
        .map(str::to_owned)
}

fn yaml_path_json(
    value: &YamlValue,
    path: &[&str],
    default_value: serde_json::Value,
) -> serde_json::Value {
    yaml_path(value, path)
        .and_then(|value| yaml_to_json(value).ok())
        .unwrap_or(default_value)
}

fn yaml_to_json(value: &YamlValue) -> Result<serde_json::Value, String> {
    serde_json::to_value(value)
        .map_err(|err| format!("failed to convert YAML value to JSON: {err}"))
}

fn json_object() -> serde_json::Value {
    serde_json::Value::Object(serde_json::Map::new())
}

fn default_model_profile(repo_root: &Path) -> Result<String, String> {
    let model_profiles = read_yaml_file(repo_root.join("configs/model_profiles.yaml"))?;
    yaml_path_string(&model_profiles, &["default_profile"])
        .ok_or_else(|| "configs/model_profiles.yaml is missing default_profile".to_owned())
}

fn yaml_mapping_get_value<'a>(value: &'a YamlValue, key: &str) -> Option<&'a YamlValue> {
    value
        .as_mapping()
        .and_then(|mapping| yaml_mapping_get(mapping, key))
}

fn yaml_mapping_get<'a>(mapping: &'a YamlMapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_owned()))
}

fn remove_yaml_mapping_key(value: &mut YamlValue, key: &str) {
    if let Some(mapping) = value.as_mapping_mut() {
        mapping.remove(YamlValue::String(key.to_owned()));
    }
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn resolve_repo_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn now_epoch_seconds() -> f64 {
    system_time_epoch_seconds(SystemTime::now())
}

fn system_time_epoch_seconds(value: SystemTime) -> f64 {
    value
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn slugify(value: &str) -> String {
    let mut slug = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    let slug = slug.trim_matches('-').to_owned();
    if slug.is_empty() {
        "run".to_owned()
    } else {
        slug
    }
}

fn exit_code_u8(code: i32) -> u8 {
    if (0..=255).contains(&code) {
        code as u8
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cases_as_grouped_overrides() {
        let args = vec![
            "--experiment-profile".to_owned(),
            "base".to_owned(),
            "--case".to_owned(),
            "strategy.topk=5".to_owned(),
            "strategy.n_drop=1".to_owned(),
            "--case".to_owned(),
            "strategy.topk=10".to_owned(),
        ];
        let options = parse_options(&args).unwrap();
        let cases = resolve_explicit_cases(&options.case_overrides).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(
            yaml_value_plain(cases[0].get("strategy.topk").unwrap()),
            "5"
        );
        assert_eq!(
            yaml_value_plain(cases[0].get("strategy.n_drop").unwrap()),
            "1"
        );
        assert_eq!(
            yaml_value_plain(cases[1].get("strategy.topk").unwrap()),
            "10"
        );
    }

    #[test]
    fn parses_equals_style_batch_options() {
        let options = parse_options(&[
            "--experiment-profile=base".to_owned(),
            "--config=configs/config.yaml".to_owned(),
            "--pipeline=rolling".to_owned(),
            "--model-profile=lgbm_fast".to_owned(),
            "--feature-profile=core_v4".to_owned(),
            "--data-source=tushare".to_owned(),
            "--set=strategy.n_drop=2".to_owned(),
            "--sweep=rolling.retrain_step=[5,10]".to_owned(),
            "--case=strategy.topk=5".to_owned(),
            "--run-tag-prefix=sweep".to_owned(),
            "--store-dir=results/demo".to_owned(),
            "--repo-root=/repo".to_owned(),
            "--dedupe-predictions".to_owned(),
            "--skip-reference-baselines".to_owned(),
            "--dry-run".to_owned(),
        ])
        .unwrap();

        assert_eq!(options.experiment_profile, "base");
        assert_eq!(options.model_profile.as_deref(), Some("lgbm_fast"));
        assert_eq!(options.feature_profile.as_deref(), Some("core_v4"));
        assert_eq!(options.data_source.as_deref(), Some("tushare"));
        assert_eq!(options.set_overrides, vec!["strategy.n_drop=2"]);
        assert_eq!(options.sweep_overrides, vec!["rolling.retrain_step=[5,10]"]);
        assert_eq!(options.case_overrides, vec![vec!["strategy.topk=5"]]);
        assert_eq!(options.run_tag_prefix.as_deref(), Some("sweep"));
        assert_eq!(options.store_dir.as_deref(), Some("results/demo"));
        assert_eq!(options.repo_root, PathBuf::from("/repo"));
        assert!(options.dedupe_predictions);
        assert!(options.skip_reference_baselines);
        assert!(options.dry_run);
    }

    #[test]
    fn expands_sweep_grid_in_key_order() {
        let mut sweep_map = BTreeMap::new();
        sweep_map.insert(
            "rolling.retrain_step".to_owned(),
            vec![YamlValue::Number(5.into()), YamlValue::Number(10.into())],
        );
        sweep_map.insert(
            "strategy.topk".to_owned(),
            vec![YamlValue::Number(20.into()), YamlValue::Number(30.into())],
        );
        let runs = expand_sweep_grid(&sweep_map);
        assert_eq!(runs.len(), 4);
        assert_eq!(
            yaml_value_plain(runs[0].get("rolling.retrain_step").unwrap()),
            "5"
        );
        assert_eq!(
            yaml_value_plain(runs[0].get("strategy.topk").unwrap()),
            "20"
        );
        assert_eq!(
            yaml_value_plain(runs[3].get("rolling.retrain_step").unwrap()),
            "10"
        );
        assert_eq!(
            yaml_value_plain(runs[3].get("strategy.topk").unwrap()),
            "30"
        );
    }

    #[test]
    fn builds_python_rolling_command() {
        let options = parse_options(&[
            "--experiment-profile".to_owned(),
            "base".to_owned(),
            "--repo-root".to_owned(),
            "/repo".to_owned(),
            "--run-tag-prefix".to_owned(),
            "sweep".to_owned(),
            "--skip-reference-baselines".to_owned(),
        ])
        .unwrap();
        let mut overrides = BTreeMap::new();
        overrides.insert("strategy.topk".to_owned(), YamlValue::Number(20.into()));
        let run_tag = build_run_tag(options.run_tag_prefix.as_deref(), &overrides);
        let command = build_run_command(&options, &overrides, run_tag.as_deref()).unwrap();
        assert_eq!(
            command[0..6],
            [
                "cargo",
                "run",
                "--bin",
                "ai4stock-train",
                "--",
                "rolling-lgbm"
            ]
        );
        assert!(command.contains(&"--skip-reference-baselines".to_owned()));
        assert!(command.contains(&"strategy.topk=20".to_owned()));
        assert!(command.contains(&"sweep__strategy-topk-20".to_owned()));
    }

    #[test]
    fn prediction_fingerprint_ignores_replay_only_n_drop() {
        let base = serde_yaml::from_str::<YamlValue>(
            r#"
data: {source: tushare}
features: {profile: core_v4_techlite}
model: {name: lgbm}
lgbm: {validation_topk: 8}
strategy: {topk: 30, n_drop: 5}
backtest: {rebalance_freq: 10}
label: {signal_horizon: 20}
rolling: {retrain_step: 20, train_days: 242, valid_days: 10}
time: {test: [2022-01-01, 2022-02-07]}
"#,
        )
        .unwrap();
        let replay = serde_yaml::from_str::<YamlValue>(
            r#"
data: {source: tushare}
features: {profile: core_v4_techlite}
model: {name: lgbm}
lgbm: {validation_topk: 8}
strategy: {topk: 30, n_drop: 4}
backtest: {rebalance_freq: 10}
label: {signal_horizon: 20}
rolling: {retrain_step: 20, train_days: 242, valid_days: 10}
time: {test: [2022-01-01, 2022-02-07]}
"#,
        )
        .unwrap();
        assert_eq!(
            prediction_fingerprint(&base).unwrap(),
            prediction_fingerprint(&replay).unwrap()
        );
    }

    #[test]
    fn prediction_fingerprint_includes_effective_validation_topk() {
        let base = serde_yaml::from_str::<YamlValue>(
            r#"
data: {source: tushare}
features: {profile: core_v4_techlite}
model: {name: lgbm}
lgbm: {}
strategy: {topk: 30, n_drop: 5}
backtest: {rebalance_freq: 10}
label: {signal_horizon: 20}
rolling: {retrain_step: 20, train_days: 242, valid_days: 10}
time: {test: [2022-01-01, 2022-02-07]}
"#,
        )
        .unwrap();
        let changed = serde_yaml::from_str::<YamlValue>(
            r#"
data: {source: tushare}
features: {profile: core_v4_techlite}
model: {name: lgbm}
lgbm: {}
strategy: {topk: 20, n_drop: 5}
backtest: {rebalance_freq: 10}
label: {signal_horizon: 20}
rolling: {retrain_step: 20, train_days: 242, valid_days: 10}
time: {test: [2022-01-01, 2022-02-07]}
"#,
        )
        .unwrap();
        assert_ne!(
            prediction_fingerprint(&base).unwrap(),
            prediction_fingerprint(&changed).unwrap()
        );
    }
}
