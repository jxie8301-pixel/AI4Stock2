use serde_json::Value;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

#[path = "ai4stock_train/rust_runtime.rs"]
mod rust_runtime;

fn usage() -> &'static str {
    "\
ai4stock-train: LightGBM training-artifact builder for AI4Stock2

Usage:
  ai4stock-train make-bundle-lgbm --output-dir <PATH> [options]

Options:
  --config <PATH>             Config file path. Default: configs/config.yaml.
  --config-is-snapshot        Treat --config as a fully resolved config snapshot.
  --experiment-profile <NAME> Named experiment profile.
  --feature-profile <NAME>    Override features.profile.
  --model-profile <NAME>      Override model.profile. The model name is still forced to lgbm.
  --data-source <NAME>        Override runtime data source.
  --factor-store <PATH>       Factor-store root. Defaults from resolved feature profile/data source.
  --output-dir <PATH>         Run output dir. Writes prediction_artifacts/ unless PATH is already that dir.
  --test-start <DATE>         Override time.test start date.
  --test-end <DATE>           Override time.test end date.
  --train-days <N>            Override rolling.train_days.
  --valid-days <N>            Override rolling.valid_days.
  --retrain-step <N>          Override rolling.retrain_step.
  --signal-horizon <N>        Override label.signal_horizon.
  --topk <N>                  Override strategy.topk.
  --n-drop <N>                Override strategy.n_drop.
  --rebalance-freq <N>        Override backtest.rebalance_freq.
  --set <KEY=VALUE>           Generic dotted config override. Can be repeated.
  --feature <NAME>            Selected feature source column. Can be repeated.
  --features-json <PATH>      JSON list, or object with selected_features/features.
  --max-features <N>          If no feature profile/list resolves, take first N metadata features. Default: 64; 0 means all.
  --batch-size <N>            Arrow record-batch size. Default: 65536.
  --label-embargo-days <N>    Override rolling.label_embargo_days.
  --cross-sectional-rank      Force date-local cross-sectional rank transform.
  --no-cross-sectional-rank   Disable Rust cross-sectional rank transform for selected features.
  --save-models               Save each rolling LightGBM model pickle.
  --load-models               Load existing rolling model pickles when present.
  --include-reference-baselines
                              Also write factor reference-baseline predictions. Default: skipped.
  --json                      Print machine-readable JSON summary.
  -h, --help                  Show this help.
"
}

#[derive(Debug, Clone)]
pub(crate) struct LgbmBundleOptions {
    pub(crate) config: PathBuf,
    pub(crate) config_is_snapshot: bool,
    pub(crate) experiment_profile: Option<String>,
    pub(crate) feature_profile: Option<String>,
    pub(crate) model_profile: Option<String>,
    pub(crate) data_source: Option<String>,
    pub(crate) factor_store: Option<PathBuf>,
    pub(crate) output_dir: PathBuf,
    pub(crate) test_start: Option<String>,
    pub(crate) test_end: Option<String>,
    pub(crate) train_days: Option<usize>,
    pub(crate) valid_days: Option<usize>,
    pub(crate) retrain_step: Option<usize>,
    pub(crate) signal_horizon: Option<usize>,
    pub(crate) label_embargo_days: Option<usize>,
    pub(crate) topk: Option<usize>,
    pub(crate) n_drop: Option<usize>,
    pub(crate) rebalance_freq: Option<usize>,
    pub(crate) set_overrides: Vec<String>,
    pub(crate) explicit_features: Vec<String>,
    pub(crate) features_json: Option<PathBuf>,
    pub(crate) max_features: usize,
    pub(crate) batch_size: usize,
    pub(crate) cross_sectional_rank: Option<bool>,
    pub(crate) save_models: bool,
    pub(crate) load_models: bool,
    pub(crate) skip_reference_baselines: bool,
    pub(crate) json: bool,
}

fn main() -> ExitCode {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
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

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        return Err(usage().to_owned());
    };
    match command.as_str() {
        "-h" | "--help" => Err(usage().to_owned()),
        "make-bundle-lgbm" => run_make_bundle_lgbm(&args[1..]),
        other => Err(format!("unknown command: {other}\n\n{}", usage())),
    }
}

fn run_make_bundle_lgbm(args: &[String]) -> Result<(), String> {
    let options = parse_lgbm_bundle_options(args)?;
    let summary = rust_runtime::make_bundle_lgbm_rust_runtime(&options)?;
    if options.json {
        let payload = serde_json::to_string_pretty(&summary)
            .map_err(|error| format!("failed to encode summary JSON: {error}"))?;
        println!("{payload}");
        return Ok(());
    }
    println!("artifact_dir={}", summary_display(&summary, "artifact_dir"));
    println!("model_name={}", summary_display(&summary, "model_name"));
    println!(
        "selected_feature_count={}",
        summary_display(&summary, "selected_feature_count")
    );
    println!(
        "prediction_rows={}",
        summary_display(&summary, "prediction_rows")
    );
    println!(
        "finite_predictions={}",
        summary_display(&summary, "finite_predictions")
    );
    println!("windows={}", summary_display(&summary, "windows"));
    println!(
        "elapsed_seconds={}",
        summary_display(&summary, "elapsed_seconds")
    );
    Ok(())
}

fn parse_lgbm_bundle_options(args: &[String]) -> Result<LgbmBundleOptions, String> {
    let mut config = PathBuf::from("configs/config.yaml");
    let mut config_is_snapshot = false;
    let mut experiment_profile = None;
    let mut feature_profile = None;
    let mut model_profile = None;
    let mut data_source = None;
    let mut factor_store = None;
    let mut output_dir = None;
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
    let mut skip_reference_baselines = true;
    let mut json = false;

    let mut arg_index = 0usize;
    while arg_index < args.len() {
        match args[arg_index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--config-is-snapshot" => config_is_snapshot = true,
            "--save-models" => save_models = true,
            "--load-models" => load_models = true,
            "--skip-reference-baselines" => skip_reference_baselines = true,
            "--include-reference-baselines" => skip_reference_baselines = false,
            "--cross-sectional-rank" => cross_sectional_rank = Some(true),
            "--no-cross-sectional-rank" => cross_sectional_rank = Some(false),
            "--json" => json = true,
            "--config" => {
                arg_index += 1;
                config = PathBuf::from(next_value(args, arg_index, "--config")?);
            }
            value if value.starts_with("--config=") => {
                config = PathBuf::from(split_value(value, "--config")?);
            }
            "--experiment-profile" => {
                arg_index += 1;
                experiment_profile = Some(next_value(args, arg_index, "--experiment-profile")?);
            }
            value if value.starts_with("--experiment-profile=") => {
                experiment_profile = Some(split_value(value, "--experiment-profile")?);
            }
            "--feature-profile" => {
                arg_index += 1;
                feature_profile = Some(next_value(args, arg_index, "--feature-profile")?);
            }
            value if value.starts_with("--feature-profile=") => {
                feature_profile = Some(split_value(value, "--feature-profile")?);
            }
            "--model-profile" => {
                arg_index += 1;
                model_profile = Some(next_value(args, arg_index, "--model-profile")?);
            }
            value if value.starts_with("--model-profile=") => {
                model_profile = Some(split_value(value, "--model-profile")?);
            }
            "--data-source" => {
                arg_index += 1;
                data_source = Some(next_value(args, arg_index, "--data-source")?);
            }
            value if value.starts_with("--data-source=") => {
                data_source = Some(split_value(value, "--data-source")?);
            }
            "--factor-store" => {
                arg_index += 1;
                factor_store = Some(PathBuf::from(next_value(
                    args,
                    arg_index,
                    "--factor-store",
                )?));
            }
            value if value.starts_with("--factor-store=") => {
                factor_store = Some(PathBuf::from(split_value(value, "--factor-store")?));
            }
            "--output-dir" => {
                arg_index += 1;
                output_dir = Some(PathBuf::from(next_value(args, arg_index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--test-start" => {
                arg_index += 1;
                test_start = Some(next_value(args, arg_index, "--test-start")?);
            }
            value if value.starts_with("--test-start=") => {
                test_start = Some(split_value(value, "--test-start")?);
            }
            "--test-end" => {
                arg_index += 1;
                test_end = Some(next_value(args, arg_index, "--test-end")?);
            }
            value if value.starts_with("--test-end=") => {
                test_end = Some(split_value(value, "--test-end")?);
            }
            "--train-days" => {
                arg_index += 1;
                train_days = Some(parse_usize(
                    next_value(args, arg_index, "--train-days")?,
                    "--train-days",
                )?);
            }
            value if value.starts_with("--train-days=") => {
                train_days = Some(parse_usize(
                    split_value(value, "--train-days")?,
                    "--train-days",
                )?);
            }
            "--valid-days" => {
                arg_index += 1;
                valid_days = Some(parse_usize(
                    next_value(args, arg_index, "--valid-days")?,
                    "--valid-days",
                )?);
            }
            value if value.starts_with("--valid-days=") => {
                valid_days = Some(parse_usize(
                    split_value(value, "--valid-days")?,
                    "--valid-days",
                )?);
            }
            "--retrain-step" => {
                arg_index += 1;
                retrain_step = Some(parse_usize(
                    next_value(args, arg_index, "--retrain-step")?,
                    "--retrain-step",
                )?);
            }
            value if value.starts_with("--retrain-step=") => {
                retrain_step = Some(parse_usize(
                    split_value(value, "--retrain-step")?,
                    "--retrain-step",
                )?);
            }
            "--signal-horizon" => {
                arg_index += 1;
                signal_horizon = Some(parse_usize(
                    next_value(args, arg_index, "--signal-horizon")?,
                    "--signal-horizon",
                )?);
            }
            value if value.starts_with("--signal-horizon=") => {
                signal_horizon = Some(parse_usize(
                    split_value(value, "--signal-horizon")?,
                    "--signal-horizon",
                )?);
            }
            "--label-embargo-days" => {
                arg_index += 1;
                label_embargo_days = Some(parse_usize(
                    next_value(args, arg_index, "--label-embargo-days")?,
                    "--label-embargo-days",
                )?);
            }
            value if value.starts_with("--label-embargo-days=") => {
                label_embargo_days = Some(parse_usize(
                    split_value(value, "--label-embargo-days")?,
                    "--label-embargo-days",
                )?);
            }
            "--topk" => {
                arg_index += 1;
                topk = Some(parse_usize(
                    next_value(args, arg_index, "--topk")?,
                    "--topk",
                )?);
            }
            value if value.starts_with("--topk=") => {
                topk = Some(parse_usize(split_value(value, "--topk")?, "--topk")?);
            }
            "--n-drop" => {
                arg_index += 1;
                n_drop = Some(parse_usize(
                    next_value(args, arg_index, "--n-drop")?,
                    "--n-drop",
                )?);
            }
            value if value.starts_with("--n-drop=") => {
                n_drop = Some(parse_usize(split_value(value, "--n-drop")?, "--n-drop")?);
            }
            "--rebalance-freq" => {
                arg_index += 1;
                rebalance_freq = Some(parse_usize(
                    next_value(args, arg_index, "--rebalance-freq")?,
                    "--rebalance-freq",
                )?);
            }
            value if value.starts_with("--rebalance-freq=") => {
                rebalance_freq = Some(parse_usize(
                    split_value(value, "--rebalance-freq")?,
                    "--rebalance-freq",
                )?);
            }
            "--set" => {
                arg_index += 1;
                set_overrides.push(next_value(args, arg_index, "--set")?);
            }
            value if value.starts_with("--set=") => {
                set_overrides.push(split_value(value, "--set")?);
            }
            "--feature" => {
                arg_index += 1;
                explicit_features.push(next_value(args, arg_index, "--feature")?);
            }
            value if value.starts_with("--feature=") => {
                explicit_features.push(split_value(value, "--feature")?);
            }
            "--features-json" => {
                arg_index += 1;
                features_json = Some(PathBuf::from(next_value(
                    args,
                    arg_index,
                    "--features-json",
                )?));
            }
            value if value.starts_with("--features-json=") => {
                features_json = Some(PathBuf::from(split_value(value, "--features-json")?));
            }
            "--max-features" => {
                arg_index += 1;
                max_features = parse_usize(
                    next_value(args, arg_index, "--max-features")?,
                    "--max-features",
                )?;
            }
            value if value.starts_with("--max-features=") => {
                max_features =
                    parse_usize(split_value(value, "--max-features")?, "--max-features")?;
            }
            "--batch-size" => {
                arg_index += 1;
                batch_size =
                    parse_usize(next_value(args, arg_index, "--batch-size")?, "--batch-size")?;
            }
            value if value.starts_with("--batch-size=") => {
                batch_size = parse_usize(split_value(value, "--batch-size")?, "--batch-size")?;
            }
            other => return Err(format!("unknown option for make-bundle-lgbm: {other}")),
        }
        arg_index += 1;
    }

    let output_dir =
        output_dir.ok_or_else(|| format!("--output-dir is required\n\n{}", usage()))?;
    if test_start.is_some() != test_end.is_some() {
        return Err("--test-start and --test-end must be supplied together".to_owned());
    }

    Ok(LgbmBundleOptions {
        config,
        config_is_snapshot,
        experiment_profile,
        feature_profile,
        model_profile,
        data_source,
        factor_store,
        output_dir,
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
        skip_reference_baselines,
        json,
    })
}

fn next_value(args: &[String], arg_index: usize, flag: &str) -> Result<String, String> {
    args.get(arg_index)
        .cloned()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn split_value(value: &str, flag: &str) -> Result<String, String> {
    value
        .strip_prefix(&format!("{flag}="))
        .map(str::to_owned)
        .ok_or_else(|| format!("invalid {flag}=... option: {value}"))
}

fn parse_usize(value: String, field: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("{field} must be a non-negative integer, got {value}: {error}"))
}

fn summary_display(summary: &Value, key: &str) -> String {
    match summary.get(key) {
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
        None => "null".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn parses_lgbm_bundle_options() {
        let options = parse_lgbm_bundle_options(&args(&[
            "--output-dir",
            "/tmp/run",
            "--experiment-profile",
            "core_v4_lgbm_default_10x20x10",
            "--model-profile",
            "lgbm_fast",
            "--test-start",
            "2025-01-02",
            "--test-end",
            "2025-01-10",
            "--train-days=60",
            "--valid-days",
            "10",
            "--retrain-step",
            "5",
            "--signal-horizon",
            "20",
            "--set",
            "lgbm.num_boost_round=20",
            "--save-models",
            "--json",
        ]))
        .unwrap();

        assert_eq!(options.output_dir, PathBuf::from("/tmp/run"));
        assert_eq!(
            options.experiment_profile.as_deref(),
            Some("core_v4_lgbm_default_10x20x10")
        );
        assert_eq!(options.model_profile.as_deref(), Some("lgbm_fast"));
        assert_eq!(options.train_days, Some(60));
        assert_eq!(options.valid_days, Some(10));
        assert_eq!(options.retrain_step, Some(5));
        assert_eq!(options.signal_horizon, Some(20));
        assert_eq!(options.set_overrides, vec!["lgbm.num_boost_round=20"]);
        assert!(options.save_models);
        assert!(options.skip_reference_baselines);
        assert!(options.json);
    }

    #[test]
    fn requires_output_dir() {
        let error = parse_lgbm_bundle_options(&args(&[
            "--experiment-profile",
            "core_v4_lgbm_default_10x20x10",
        ]))
        .unwrap_err();

        assert!(error.contains("--output-dir is required"));
    }

    #[test]
    fn requires_test_window_pair() {
        let error = parse_lgbm_bundle_options(&args(&[
            "--output-dir",
            "/tmp/run",
            "--test-start",
            "2025-01-02",
        ]))
        .unwrap_err();

        assert!(error.contains("--test-start and --test-end"));
    }
}
