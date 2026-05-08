use super::{rust_runtime, LgbmBundleOptions};
use ai4stock2_native::common::cli::{display_command, next_arg, split_value};
use chrono::Local;
use serde_json::Value as JsonValue;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const PREDICTION_ARTIFACT_DIRNAME: &str = "prediction_artifacts";

#[derive(Debug, Clone)]
struct RollingLgbmOptions {
    bundle_options: LgbmBundleOptions,
    load_predictions_dir: Option<PathBuf>,
    output_dir: PathBuf,
    skip_reference_baselines: bool,
    skip_backtest_plots: bool,
    disable_local_store: bool,
    baseline_jobs: Option<usize>,
    dry_run: bool,
    json: bool,
}

pub(crate) fn run_rolling_lgbm(args: &[String]) -> Result<(), String> {
    let options = parse_rolling_lgbm_options(args)?;
    let config_snapshot = options.output_dir.join("config_snapshot.yaml");
    let bundle_dir = options
        .load_predictions_dir
        .clone()
        .unwrap_or_else(|| options.output_dir.join(PREDICTION_ARTIFACT_DIRNAME));

    if options.dry_run {
        rust_runtime::validate_lgbm_bundle_options(&options.bundle_options)?;
        print_dry_run(&options, &bundle_dir, &config_snapshot);
        return Ok(());
    }

    let train_summary = if options.load_predictions_dir.is_some() {
        rust_runtime::write_resolved_config_snapshot(&options.bundle_options, &config_snapshot)?;
        JsonValue::Null
    } else {
        rust_runtime::make_bundle_lgbm_rust_runtime(&options.bundle_options)?
    };
    run_backtest_command(&options, &bundle_dir, &config_snapshot)?;
    if options.json {
        let payload = serde_json::json!({
            "output_dir": options.output_dir.display().to_string(),
            "bundle_dir": bundle_dir.display().to_string(),
            "config_snapshot_path": config_snapshot.display().to_string(),
            "loaded_predictions": options.load_predictions_dir.is_some(),
            "training_summary": train_summary,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&payload)
                .map_err(|err| format!("failed to encode rolling summary JSON: {err}"))?
        );
    }
    Ok(())
}

fn parse_rolling_lgbm_options(args: &[String]) -> Result<RollingLgbmOptions, String> {
    let mut config = PathBuf::from("configs/config.yaml");
    let mut config_is_snapshot = false;
    let mut experiment_profile = None;
    let mut feature_profile = None;
    let mut model_profile = None;
    let mut data_source = None;
    let mut factor_store = None;
    let mut output_dir = None;
    let mut store_dir = None;
    let mut run_tag = None;
    let mut test_start = None;
    let mut test_end = None;
    let mut train_days = None;
    let mut valid_days = None;
    let mut retrain_step = None;
    let mut signal_horizon = None;
    let mut label_embargo_days = None;
    let mut topk = None;
    let mut n_drop = None;
    let mut rebalance_freq = None;
    let mut set_overrides = Vec::new();
    let mut explicit_features = Vec::new();
    let mut features_json = None;
    let mut max_features = 64usize;
    let mut batch_size = 65_536usize;
    let mut cross_sectional_rank = None;
    let mut save_models = false;
    let mut load_models = false;
    let mut load_predictions_dir = None;
    let mut model_name = "lgbm".to_owned();
    let mut skip_reference_baselines = false;
    let mut skip_backtest_plots = false;
    let mut disable_local_store = false;
    let mut backtest_artifact_level = "full".to_owned();
    let mut baseline_jobs = None;
    let mut dry_run = false;
    let mut json = false;

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(super::usage().to_owned()),
            "--config-is-snapshot" => config_is_snapshot = true,
            "--save-models" => save_models = true,
            "--load-models" => load_models = true,
            "--save-predictions" => {}
            "--disable-local-store" => disable_local_store = true,
            "--skip-opportunity-diagnostics" => {}
            "--skip-backtest-trace" => {}
            "--skip-reference-baselines" => skip_reference_baselines = true,
            "--skip-backtest-plots" => skip_backtest_plots = true,
            "--cross-sectional-rank" => cross_sectional_rank = Some(true),
            "--no-cross-sectional-rank" => cross_sectional_rank = Some(false),
            "--dry-run" => dry_run = true,
            "--json" => json = true,
            "--config" => config = PathBuf::from(next_arg(args, &mut index, "--config")?),
            value if value.starts_with("--config=") => {
                config = PathBuf::from(split_value(value, "--config")?)
            }
            "--experiment-profile" => {
                experiment_profile = Some(next_arg(args, &mut index, "--experiment-profile")?)
            }
            value if value.starts_with("--experiment-profile=") => {
                experiment_profile = Some(split_value(value, "--experiment-profile")?)
            }
            "--feature-profile" | "--profile" => {
                feature_profile = Some(next_arg(args, &mut index, "--feature-profile")?)
            }
            value if value.starts_with("--feature-profile=") || value.starts_with("--profile=") => {
                feature_profile = Some(value.split_once('=').unwrap().1.to_owned())
            }
            "--model" => model_name = next_arg(args, &mut index, "--model")?,
            value if value.starts_with("--model=") => {
                model_name = split_value(value, "--model")?;
            }
            "--model-profile" => {
                model_profile = Some(next_arg(args, &mut index, "--model-profile")?)
            }
            value if value.starts_with("--model-profile=") => {
                model_profile = Some(split_value(value, "--model-profile")?)
            }
            "--data-source" => data_source = Some(next_arg(args, &mut index, "--data-source")?),
            value if value.starts_with("--data-source=") => {
                data_source = Some(split_value(value, "--data-source")?)
            }
            "--factor-store" => {
                factor_store = Some(PathBuf::from(next_arg(args, &mut index, "--factor-store")?));
            }
            value if value.starts_with("--factor-store=") => {
                factor_store = Some(PathBuf::from(split_value(value, "--factor-store")?));
            }
            "--output-dir" | "--results-dir" => {
                output_dir = Some(PathBuf::from(next_arg(args, &mut index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") || value.starts_with("--results-dir=") => {
                output_dir = Some(PathBuf::from(value.split_once('=').unwrap().1));
            }
            "--store-dir" => {
                store_dir = Some(PathBuf::from(next_arg(args, &mut index, "--store-dir")?))
            }
            value if value.starts_with("--store-dir=") => {
                store_dir = Some(PathBuf::from(split_value(value, "--store-dir")?))
            }
            "--run-tag" => run_tag = Some(next_arg(args, &mut index, "--run-tag")?),
            value if value.starts_with("--run-tag=") => {
                run_tag = Some(split_value(value, "--run-tag")?)
            }
            "--test-start" => test_start = Some(next_arg(args, &mut index, "--test-start")?),
            value if value.starts_with("--test-start=") => {
                test_start = Some(split_value(value, "--test-start")?)
            }
            "--test-end" => test_end = Some(next_arg(args, &mut index, "--test-end")?),
            value if value.starts_with("--test-end=") => {
                test_end = Some(split_value(value, "--test-end")?)
            }
            "--train-days" => {
                train_days = Some(parse_usize(
                    next_arg(args, &mut index, "--train-days")?,
                    "--train-days",
                )?)
            }
            value if value.starts_with("--train-days=") => {
                train_days = Some(parse_usize(
                    split_value(value, "--train-days")?,
                    "--train-days",
                )?)
            }
            "--valid-days" => {
                valid_days = Some(parse_usize(
                    next_arg(args, &mut index, "--valid-days")?,
                    "--valid-days",
                )?)
            }
            value if value.starts_with("--valid-days=") => {
                valid_days = Some(parse_usize(
                    split_value(value, "--valid-days")?,
                    "--valid-days",
                )?)
            }
            "--retrain-step" | "--horizon" => {
                retrain_step = Some(parse_usize(
                    next_arg(args, &mut index, "--retrain-step")?,
                    "--retrain-step",
                )?)
            }
            value if value.starts_with("--retrain-step=") || value.starts_with("--horizon=") => {
                retrain_step = Some(parse_usize(
                    value.split_once('=').unwrap().1.to_owned(),
                    "--retrain-step",
                )?)
            }
            "--signal-horizon" | "--label-horizon" => {
                signal_horizon = Some(parse_usize(
                    next_arg(args, &mut index, "--signal-horizon")?,
                    "--signal-horizon",
                )?)
            }
            value
                if value.starts_with("--signal-horizon=")
                    || value.starts_with("--label-horizon=") =>
            {
                signal_horizon = Some(parse_usize(
                    value.split_once('=').unwrap().1.to_owned(),
                    "--signal-horizon",
                )?)
            }
            "--label-embargo-days" => {
                label_embargo_days = Some(parse_usize(
                    next_arg(args, &mut index, "--label-embargo-days")?,
                    "--label-embargo-days",
                )?)
            }
            value if value.starts_with("--label-embargo-days=") => {
                label_embargo_days = Some(parse_usize(
                    split_value(value, "--label-embargo-days")?,
                    "--label-embargo-days",
                )?)
            }
            "--topk" => {
                topk = Some(parse_usize(
                    next_arg(args, &mut index, "--topk")?,
                    "--topk",
                )?)
            }
            value if value.starts_with("--topk=") => {
                topk = Some(parse_usize(split_value(value, "--topk")?, "--topk")?)
            }
            "--n-drop" => {
                n_drop = Some(parse_usize(
                    next_arg(args, &mut index, "--n-drop")?,
                    "--n-drop",
                )?)
            }
            value if value.starts_with("--n-drop=") => {
                n_drop = Some(parse_usize(split_value(value, "--n-drop")?, "--n-drop")?)
            }
            "--rebalance-freq" => {
                rebalance_freq = Some(parse_usize(
                    next_arg(args, &mut index, "--rebalance-freq")?,
                    "--rebalance-freq",
                )?)
            }
            value if value.starts_with("--rebalance-freq=") => {
                rebalance_freq = Some(parse_usize(
                    split_value(value, "--rebalance-freq")?,
                    "--rebalance-freq",
                )?)
            }
            "--set" => set_overrides.push(next_arg(args, &mut index, "--set")?),
            value if value.starts_with("--set=") => {
                set_overrides.push(split_value(value, "--set")?)
            }
            "--feature" => explicit_features.push(next_arg(args, &mut index, "--feature")?),
            value if value.starts_with("--feature=") => {
                explicit_features.push(split_value(value, "--feature")?)
            }
            "--features-json" => {
                features_json = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--features-json",
                )?))
            }
            value if value.starts_with("--features-json=") => {
                features_json = Some(PathBuf::from(split_value(value, "--features-json")?))
            }
            "--max-features" => {
                max_features = parse_usize(
                    next_arg(args, &mut index, "--max-features")?,
                    "--max-features",
                )?
            }
            value if value.starts_with("--max-features=") => {
                max_features = parse_usize(split_value(value, "--max-features")?, "--max-features")?
            }
            "--batch-size" => {
                batch_size =
                    parse_usize(next_arg(args, &mut index, "--batch-size")?, "--batch-size")?
            }
            value if value.starts_with("--batch-size=") => {
                batch_size = parse_usize(split_value(value, "--batch-size")?, "--batch-size")?
            }
            "--load-predictions-dir" => {
                load_predictions_dir = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--load-predictions-dir",
                )?))
            }
            value if value.starts_with("--load-predictions-dir=") => {
                load_predictions_dir =
                    Some(PathBuf::from(split_value(value, "--load-predictions-dir")?))
            }
            "--backtest-artifact-level" => {
                backtest_artifact_level = next_arg(args, &mut index, "--backtest-artifact-level")?;
            }
            value if value.starts_with("--backtest-artifact-level=") => {
                backtest_artifact_level = split_value(value, "--backtest-artifact-level")?;
            }
            "--baseline-jobs" => {
                baseline_jobs = Some(parse_usize(
                    next_arg(args, &mut index, "--baseline-jobs")?,
                    "--baseline-jobs",
                )?)
            }
            value if value.starts_with("--baseline-jobs=") => {
                baseline_jobs = Some(parse_usize(
                    split_value(value, "--baseline-jobs")?,
                    "--baseline-jobs",
                )?)
            }
            other => return Err(format!("unknown option for rolling-lgbm: {other}")),
        }
        index += 1;
    }

    if !model_name.trim().eq_ignore_ascii_case("lgbm") {
        return Err("ai4stock-train rolling-lgbm only supports --model lgbm".to_owned());
    }
    if test_start.is_some() != test_end.is_some() {
        return Err("--test-start and --test-end must be supplied together".to_owned());
    }
    let artifact_level = backtest_artifact_level.trim().to_ascii_lowercase();
    if !matches!(artifact_level.as_str(), "full" | "reports" | "metrics") {
        return Err(format!(
            "--backtest-artifact-level must be full, reports, or metrics; got {backtest_artifact_level}"
        ));
    }
    if matches!(artifact_level.as_str(), "reports" | "metrics") {
        skip_backtest_plots = true;
    }

    let output_dir = output_dir
        .unwrap_or_else(|| default_output_dir(store_dir, run_tag.as_deref(), disable_local_store));
    let bundle_options = LgbmBundleOptions {
        config,
        config_is_snapshot,
        experiment_profile,
        feature_profile,
        model_profile,
        data_source,
        factor_store,
        output_dir: output_dir.clone(),
        test_start,
        test_end,
        train_days,
        valid_days,
        retrain_step,
        signal_horizon,
        label_embargo_days,
        topk,
        n_drop,
        rebalance_freq,
        set_overrides,
        explicit_features,
        features_json,
        max_features,
        batch_size,
        cross_sectional_rank,
        save_models,
        load_models,
        skip_reference_baselines: true,
        json: false,
    };
    Ok(RollingLgbmOptions {
        bundle_options,
        load_predictions_dir,
        output_dir,
        skip_reference_baselines,
        skip_backtest_plots,
        disable_local_store,
        baseline_jobs,
        dry_run,
        json,
    })
}

fn run_backtest_command(
    options: &RollingLgbmOptions,
    bundle_dir: &Path,
    config_snapshot: &Path,
) -> Result<(), String> {
    let command = build_backtest_command(options, bundle_dir, config_snapshot);
    println!("[run] {}", display_command(&command));
    let status = Command::new(&command[0])
        .args(&command[1..])
        .status()
        .map_err(|err| format!("failed to run {}: {err}", display_command(&command)))?;
    if !status.success() {
        return Err(format!(
            "backtest command failed with status {status}: {}",
            display_command(&command)
        ));
    }
    Ok(())
}

fn build_backtest_command(
    options: &RollingLgbmOptions,
    bundle_dir: &Path,
    config_snapshot: &Path,
) -> Vec<String> {
    let mut command = rust_binary_command("ai4stock-backtest", "AI4STOCK_BACKTEST_BIN");
    command.extend([
        "run-bundle".to_owned(),
        "--bundle".to_owned(),
        bundle_dir.display().to_string(),
        "--config".to_owned(),
        config_snapshot.display().to_string(),
        "--output-dir".to_owned(),
        options.output_dir.display().to_string(),
    ]);
    if options.skip_reference_baselines {
        command.push("--skip-reference-baselines".to_owned());
    }
    if options.skip_backtest_plots {
        command.push("--skip-backtest-plots".to_owned());
    }
    if options.disable_local_store {
        command.push("--disable-local-store".to_owned());
    }
    if let Some(baseline_jobs) = options.baseline_jobs {
        command.extend(["--baseline-jobs".to_owned(), baseline_jobs.to_string()]);
    }
    command
}

fn print_dry_run(options: &RollingLgbmOptions, bundle_dir: &Path, config_snapshot: &Path) {
    if options.load_predictions_dir.is_none() {
        println!(
            "[dry-run] {}",
            display_command(&build_train_command(&options.bundle_options))
        );
    } else {
        println!(
            "[dry-run] write resolved config snapshot: {}",
            config_snapshot.display()
        );
    }
    println!(
        "[dry-run] {}",
        display_command(&build_backtest_command(
            options,
            bundle_dir,
            config_snapshot
        ))
    );
}

fn build_train_command(options: &LgbmBundleOptions) -> Vec<String> {
    let mut command = rust_binary_command("ai4stock-train", "AI4STOCK_TRAIN_BIN");
    command.extend([
        "make-bundle-lgbm".to_owned(),
        "--output-dir".to_owned(),
        options.output_dir.display().to_string(),
    ]);
    push_path(&mut command, "--config", &options.config);
    push_flag(
        &mut command,
        "--config-is-snapshot",
        options.config_is_snapshot,
    );
    push_option(
        &mut command,
        "--experiment-profile",
        options.experiment_profile.as_deref(),
    );
    push_option(
        &mut command,
        "--feature-profile",
        options.feature_profile.as_deref(),
    );
    push_option(
        &mut command,
        "--model-profile",
        options.model_profile.as_deref(),
    );
    push_option(
        &mut command,
        "--data-source",
        options.data_source.as_deref(),
    );
    if let Some(path) = &options.factor_store {
        push_path(&mut command, "--factor-store", path);
    }
    push_option(&mut command, "--test-start", options.test_start.as_deref());
    push_option(&mut command, "--test-end", options.test_end.as_deref());
    push_usize(&mut command, "--train-days", options.train_days);
    push_usize(&mut command, "--valid-days", options.valid_days);
    push_usize(&mut command, "--retrain-step", options.retrain_step);
    push_usize(&mut command, "--signal-horizon", options.signal_horizon);
    push_usize(
        &mut command,
        "--label-embargo-days",
        options.label_embargo_days,
    );
    push_usize(&mut command, "--topk", options.topk);
    push_usize(&mut command, "--n-drop", options.n_drop);
    push_usize(&mut command, "--rebalance-freq", options.rebalance_freq);
    for override_arg in &options.set_overrides {
        command.extend(["--set".to_owned(), override_arg.clone()]);
    }
    for feature in &options.explicit_features {
        command.extend(["--feature".to_owned(), feature.clone()]);
    }
    if let Some(path) = &options.features_json {
        push_path(&mut command, "--features-json", path);
    }
    command.extend([
        "--max-features".to_owned(),
        options.max_features.to_string(),
    ]);
    command.extend(["--batch-size".to_owned(), options.batch_size.to_string()]);
    match options.cross_sectional_rank {
        Some(true) => command.push("--cross-sectional-rank".to_owned()),
        Some(false) => command.push("--no-cross-sectional-rank".to_owned()),
        None => {}
    }
    push_flag(&mut command, "--save-models", options.save_models);
    push_flag(&mut command, "--load-models", options.load_models);
    command
}

fn default_output_dir(
    store_dir: Option<PathBuf>,
    run_tag: Option<&str>,
    disable_local_store: bool,
) -> PathBuf {
    if disable_local_store {
        return PathBuf::from("results/native_rolling_lgbm");
    }
    let root = store_dir.unwrap_or_else(|| PathBuf::from("results/experiments"));
    let tag = slugify(run_tag.unwrap_or("rust-wrapper"));
    root.join("native")
        .join("rolling")
        .join("lgbm")
        .join(format!("{}__{}", Local::now().format("%Y%m%d_%H%M%S"), tag))
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

fn slugify(raw: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in raw.trim().to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    let trimmed = out.trim_matches('-').to_owned();
    if trimmed.is_empty() {
        "run".to_owned()
    } else {
        trimmed
    }
}

fn push_option(command: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        if !value.is_empty() {
            command.extend([flag.to_owned(), value.to_owned()]);
        }
    }
}

fn push_path(command: &mut Vec<String>, flag: &str, value: &Path) {
    command.extend([flag.to_owned(), value.display().to_string()]);
}

fn push_usize(command: &mut Vec<String>, flag: &str, value: Option<usize>) {
    if let Some(value) = value {
        command.extend([flag.to_owned(), value.to_string()]);
    }
}

fn push_flag(command: &mut Vec<String>, flag: &str, enabled: bool) {
    if enabled {
        command.push(flag.to_owned());
    }
}

fn parse_usize(value: String, option: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid {option} {value}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_rolling_lgbm_runtime_options() {
        let parsed = parse_rolling_lgbm_options(&args(&[
            "--config",
            "configs/config.yaml",
            "--experiment-profile",
            "core_v4_lgbm_default_10x20x10",
            "--model-profile",
            "lgbm_fast",
            "--feature-profile",
            "core_v4_techlite",
            "--run-tag",
            "demo",
            "--save-models",
            "--set",
            "lgbm.num_boost_round=2",
            "--backtest-artifact-level",
            "reports",
            "--baseline-jobs",
            "2",
        ]))
        .unwrap();

        assert_eq!(
            parsed.bundle_options.experiment_profile.as_deref(),
            Some("core_v4_lgbm_default_10x20x10")
        );
        assert_eq!(
            parsed.bundle_options.model_profile.as_deref(),
            Some("lgbm_fast")
        );
        assert_eq!(
            parsed.bundle_options.feature_profile.as_deref(),
            Some("core_v4_techlite")
        );
        assert_eq!(
            parsed.bundle_options.set_overrides,
            vec!["lgbm.num_boost_round=2"]
        );
        assert!(parsed.bundle_options.save_models);
        assert!(parsed.skip_backtest_plots);
        assert_eq!(parsed.baseline_jobs, Some(2));
        assert!(parsed
            .output_dir
            .display()
            .to_string()
            .contains("results/experiments/native/rolling/lgbm/"));
    }

    #[test]
    fn rejects_non_lgbm_model() {
        let error =
            parse_rolling_lgbm_options(&args(&["--output-dir", "/tmp/run", "--model", "linear"]))
                .unwrap_err();

        assert!(error.contains("only supports --model lgbm"));
    }

    #[test]
    fn disable_local_store_uses_legacy_output_and_forwards_to_backtest() {
        let parsed =
            parse_rolling_lgbm_options(&args(&["--disable-local-store", "--save-models"])).unwrap();
        assert_eq!(
            parsed.output_dir,
            PathBuf::from("results/native_rolling_lgbm")
        );
        assert!(parsed.disable_local_store);

        let command = build_backtest_command(
            &parsed,
            Path::new("results/native_rolling_lgbm/prediction_artifacts"),
            Path::new("results/native_rolling_lgbm/config_snapshot.yaml"),
        );
        assert!(command.contains(&"--disable-local-store".to_owned()));
    }

    #[test]
    fn builds_backtest_command() {
        let parsed = parse_rolling_lgbm_options(&args(&[
            "--output-dir",
            "/tmp/run",
            "--skip-reference-baselines",
            "--skip-backtest-plots",
            "--baseline-jobs",
            "3",
        ]))
        .unwrap();
        let command = build_backtest_command(
            &parsed,
            Path::new("/tmp/run/prediction_artifacts"),
            Path::new("/tmp/run/config_snapshot.yaml"),
        );

        assert!(command.contains(&"run-bundle".to_owned()));
        assert!(command.contains(&"--skip-reference-baselines".to_owned()));
        assert!(command.contains(&"--skip-backtest-plots".to_owned()));
        assert!(command.contains(&"3".to_owned()));
    }
}
