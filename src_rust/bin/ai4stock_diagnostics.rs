use ai4stock2_native::common::cli::{display_command, path_to_string, split_value};
use ai4stock2_native::feature_prefilter::{
    run_build_prefilter_profile, run_build_robust_profile, run_corr_prune, run_prefilter_summary,
    run_robust_prefilter_summary, write_profile_artifacts, CorrPruneOptions, PrefilterOptions,
    PrefilterProfileBuildOptions, PrefilterThresholds, ProfileArtifactOptions, ProfileReadmeMode,
    RobustPrefilterOptions, RobustProfileBuildOptions,
};
use ai4stock2_native::single_factor_diagnostics::{
    run_single_factor_diagnostics, BenchmarkMode, BenchmarkOptions, BenchmarkValueType,
    DiagnosticLabelSpace, SegmentSpec, SingleFactorOptions,
};
use serde_json::Value as JsonValue;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "ai4stock_diagnostics/candidate_pool.rs"]
mod candidate_pool;
#[path = "ai4stock_diagnostics/single_factor_batch.rs"]
mod single_factor_batch;
#[path = "ai4stock_diagnostics/strategy_pair.rs"]
mod strategy_pair;

fn usage() -> &'static str {
    "\
ai4stock-diagnostics: Rust diagnostics entrypoint for AI4Stock2

Usage:
  ai4stock-diagnostics single-factor --factor-store <PATH> --output-dir <PATH> --label-column <COL> (--feature <NAME> | --features-json <PATH>)... [options]
  ai4stock-diagnostics single-factor-profile --experiment-profile <NAME> [options]
  ai4stock-diagnostics single-factor-batch --experiment-profile <NAME> --case KEY=VALUE... [options]
  ai4stock-diagnostics prefilter-summary --diagnostics-summary <CSV> --output-dir <PATH> [options]
  ai4stock-diagnostics robust-prefilter-summary --raw-summary <CSV> --neutral-summary <CSV> --output-dir <PATH> [options]
  ai4stock-diagnostics corr-prune --factor-store <PATH> --candidates <CSV> --output-dir <PATH> [options]
  ai4stock-diagnostics write-profile --selected-csv <CSV> --profile-name <NAME> --output-dir <PATH> [options]
  ai4stock-diagnostics build-prefilter-profile --diagnostics-summary <CSV> --factor-store <PATH> --profile-name <NAME> --output-dir <PATH> [options]
  ai4stock-diagnostics build-robust-profile --raw-summary <CSV> --neutral-summary <CSV> --factor-store <PATH> --profile-name <NAME> --output-dir <PATH> [options]
  ai4stock-diagnostics build-prefilter-profile-runtime --diagnostics-summary <CSV> --experiment-profile <NAME> --profile-name <NAME> [options]
  ai4stock-diagnostics build-robust-profile-runtime --raw-summary <CSV> --neutral-summary <CSV> --experiment-profile <NAME> --profile-name <NAME> [options]
  ai4stock-diagnostics full-space-single-factor --experiment-profile <NAME> [options]
  ai4stock-diagnostics quality-event-flow-single-factor --experiment-profile <NAME> [options]
  ai4stock-diagnostics candidate-pool (--run NAME=DIR | --candidate-root DIR) [options]
  ai4stock-diagnostics strategy-pair --candidate-run DIR --baseline-run DIR [options]

Options:
  --factor-store <PATH>       Factor-store root. Supports <root>/buckets/part-*.parquet.
  --output-dir <PATH>         Diagnostics output directory.
  --label-column <COL>        Realized-return label column, e.g. label_10d.
  --signal-horizon <N>        Forward-return horizon for benchmark_excess. Default: 10.
  --feature <NAME>            Feature source column. Can be repeated.
  --features-json <PATH>      JSON list, or object with selected_features/features.
  --date-start <DATE>         Inclusive diagnostics start date.
  --date-end <DATE>           Inclusive diagnostics end date.
  --universe-name <NAME>      Universe name. Default: all.
  --universe-dir <PATH>       Universe directory. Default: data/universes.
  --quantile-bins <N>         Cross-sectional quantile bins. Default: 5.
  --top-n <N>                 Top factor rows to export. Default: 50.
  --no-detail-artifacts       Skip daily bucket/spread/monthly/missing CSVs.
  --segment <NAME:START:END>  Segment window. Can be repeated.
  --diagnostic-label-space <NAME>
                              raw_return, industry_excess, or benchmark_excess. Default: raw_return.
  --diagnostic-threshold <X>  Hurdle for industry_excess. Default: 0.0.
  --benchmark-mode <NAME>     cross_section_mean or file. Default: cross_section_mean.
  --benchmark-path <PATH>     Benchmark file when benchmark mode is file.
  --benchmark-date-column <C> Benchmark date column. Default: date.
  --benchmark-value-column <C>
                              Benchmark close/return column. Default: close.
  --benchmark-value-type <T>  close or return. Default: close.
  --industry-neutral          Demean factors within date x industry before diagnostics.
  --industry-map <PATH>       Symbol cache parquet containing local_symbol/symbol and industry.
  --feature-chunk-size <N>    Features processed per factor-store scan. Default: 64.
  --batch-size <N>            Arrow record-batch size. Default: 65536.
  --metadata-json <PATH>      Optional metadata JSON object to include in manifest.
  --config-snapshot <PATH>    Optional resolved config snapshot to copy into output.
  --json                      Print machine-readable JSON summary.
  -h, --help                  Show this help.

Single-factor batch options:
  --config <PATH>             Runtime config path. Default: configs/config.yaml.
  --config-is-snapshot        Treat --config as already-resolved YAML.
  --experiment-profile <NAME> Experiment profile used to compose runtime config.
  --feature-profile <NAME>    Optional global feature profile override.
  --model-profile <NAME>      Optional model profile override used during config composition.
  --data-source <NAME>        Override data.source.
  --set <KEY=VALUE>           Dotted runtime config override. Can be repeated.
  --period <NAME>             train, valid, test, or all. Default: train.
  --date-start <DATE>         Explicit diagnostics start date.
  --date-end <DATE>           Explicit diagnostics end date.
  --all-features              Diagnose all cached factor-store features per case.
  --segment-scheme <NAME>     none, config_split, or yearly. Default: none.
  --segments <SPEC>           Custom 'name:start:end;...' segments.
  --base-output-dir <PATH>    Root directory for per-case outputs.
  --summary-path <PATH>       Batch TSV summary path.
  --manifest-path <PATH>      Batch TSV manifest path.
  --case KEY=VALUE ...        Case keys include name, feature_profile, baseline_feature_profile,
                              diagnostic_label_space, diagnostic_threshold, output_dir, run_tag.
  --dry-run                   Print per-case Rust single-factor commands without executing.

Prefilter options:
  --diagnostics-summary <CSV> Single-factor summary CSV.
  --segment-comparison <CSV>  Optional segment comparison CSV.
  --raw-summary <CSV>         Raw-return summary CSV for robust prefiltering.
  --neutral-summary <CSV>     Industry-neutral summary CSV for robust prefiltering.
  --raw-segment-comparison <CSV>
                              Optional raw segment comparison CSV.
  --neutral-segment-comparison <CSV>
                              Optional neutral segment comparison CSV.
  --min-coverage-pct <X>      Default: 0.95.
  --min-abs-rank-ic <X>       Default: 0.02.
  --min-abs-rank-ic-ir <X>    Default: 0.10.
  --min-monthly-positive-rate <X>
                              Default: 0.45.
  --min-segment-directional-hit-mean <X>
                              Optional segment stability floor.
  --max-segment-rank-ic-mean-range <X>
                              Optional segment drift cap.
  --exclude-direction-flip    Drop direction-flip features.

Correlation-prune options:
  --candidates <CSV>          Candidate feature CSV with a feature column.
  --corr-threshold <X>        Absolute correlation drop threshold. Default: 0.97.
  --cross-sectional-rank      Rank each feature within date before correlation. Default.
  --raw-values                Use raw factor values for correlation.

Profile artifact options:
  --selected-csv <CSV>        Selected feature CSV, usually correlation_kept.csv.
  --profile-name <NAME>       Output profile name.
  --max-features <N>          Optional selected feature cap.
  --write-config-profile      Also write configs/features/<profile-name>.yaml.
  --config-profile-path <PATH>
                              Override config profile output path.
  --factor-store-name <NAME>  YAML factor_store_name. Default: full_factor_space.
  --readme-mode <NAME>        prefilter or robust. Default: prefilter.
  --setting <KEY=VALUE>       README setting row. Can be repeated.
  --safety-warning <TEXT>     Optional README safety warning.
"
}

#[derive(Debug, Clone)]
struct CliOptions {
    factor_store: PathBuf,
    output_dir: PathBuf,
    feature_names: Vec<String>,
    label_column: String,
    signal_horizon: usize,
    date_start: Option<String>,
    date_end: Option<String>,
    universe_name: String,
    universe_dir: PathBuf,
    quantile_bins: usize,
    top_n: usize,
    include_details: bool,
    segments: Vec<SegmentSpec>,
    diagnostic_label_space: DiagnosticLabelSpace,
    diagnostic_threshold: f64,
    industry_neutral: bool,
    industry_map_path: Option<PathBuf>,
    feature_chunk_size: usize,
    batch_size: usize,
    metadata_json_path: Option<PathBuf>,
    config_snapshot_path: Option<PathBuf>,
    benchmark: BenchmarkOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct PrefilterCliOptions {
    options: PrefilterOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct RobustPrefilterCliOptions {
    options: RobustPrefilterOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct CorrPruneCliOptions {
    options: CorrPruneOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct ProfileArtifactCliOptions {
    options: ProfileArtifactOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct PrefilterProfileBuildCliOptions {
    options: PrefilterProfileBuildOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct RobustProfileBuildCliOptions {
    options: RobustProfileBuildOptions,
    json: bool,
}

#[derive(Debug, Clone)]
struct FullSpacePresetOptions {
    config: String,
    experiment_profile: String,
    model_profile: Option<String>,
    data_source: Option<String>,
    period: String,
    date_start: Option<String>,
    date_end: Option<String>,
    quantile_bins: usize,
    top_n: usize,
    segment_scheme: String,
    segments: Option<String>,
    base_output_dir: Option<PathBuf>,
    run_tag: Option<String>,
    diagnostic_label_space: String,
    diagnostic_threshold: f64,
    skip_industry_neutral: bool,
    set_overrides: Vec<String>,
    dry_run: bool,
}

#[derive(Debug, Clone)]
struct QualityEventFlowPresetOptions {
    config: String,
    experiment_profile: String,
    feature_profile: String,
    model_profile: Option<String>,
    data_source: Option<String>,
    period: String,
    date_start: Option<String>,
    date_end: Option<String>,
    quantile_bins: usize,
    top_n: usize,
    segment_scheme: String,
    segments: Option<String>,
    base_output_dir: Option<PathBuf>,
    run_tag: Option<String>,
    include_benchmark_excess: bool,
    set_overrides: Vec<String>,
    dry_run: bool,
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
        "single-factor" => run_single_factor(&args[1..]),
        "single-factor-profile" => {
            single_factor_batch::run_single_factor_profile_command(&args[1..])
        }
        "single-factor-batch" => single_factor_batch::run_single_factor_batch_command(&args[1..]),
        "prefilter-summary" => run_prefilter_summary_command(&args[1..]),
        "robust-prefilter-summary" => run_robust_prefilter_summary_command(&args[1..]),
        "corr-prune" => run_corr_prune_command(&args[1..]),
        "write-profile" => run_write_profile_command(&args[1..]),
        "build-prefilter-profile" => run_build_prefilter_profile_command(&args[1..]),
        "build-robust-profile" => run_build_robust_profile_command(&args[1..]),
        "build-prefilter-profile-runtime" => {
            single_factor_batch::run_prefilter_profile_runtime_command(&args[1..])
        }
        "build-robust-profile-runtime" => {
            single_factor_batch::run_robust_profile_runtime_command(&args[1..])
        }
        "full-space-single-factor" => run_full_space_single_factor_command(&args[1..]),
        "quality-event-flow-single-factor" => {
            run_quality_event_flow_single_factor_command(&args[1..])
        }
        "candidate-pool" => candidate_pool::run_candidate_pool_command(&args[1..]),
        "strategy-pair" => strategy_pair::run_strategy_pair_command(&args[1..]),
        other => Err(format!("unknown command: {other}\n\n{}", usage())),
    }
}

fn run_prefilter_summary_command(args: &[String]) -> Result<(), String> {
    let options = parse_prefilter_options(args)?;
    let json = options.json;
    let summary = run_prefilter_summary(&options.options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("original_features={}", summary.original_features);
        println!("after_prefilter={}", summary.after_prefilter);
        println!(
            "after_exact_duplicate_prune={}",
            summary.after_exact_duplicate_prune
        );
    }
    Ok(())
}

fn run_robust_prefilter_summary_command(args: &[String]) -> Result<(), String> {
    let options = parse_robust_prefilter_options(args)?;
    let json = options.json;
    let summary = run_robust_prefilter_summary(&options.options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("original_features={}", summary.original_features);
        println!("after_prefilter={}", summary.after_prefilter);
        println!(
            "after_exact_duplicate_prune={}",
            summary.after_exact_duplicate_prune
        );
    }
    Ok(())
}

fn run_corr_prune_command(args: &[String]) -> Result<(), String> {
    let options = parse_corr_prune_options(args)?;
    let json = options.json;
    let summary = run_corr_prune(&options.options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("input_features={}", summary.input_features);
        println!("kept_features={}", summary.kept_features);
        println!("dropped_features={}", summary.dropped_features);
        println!("row_count={}", summary.row_count);
    }
    Ok(())
}

fn run_write_profile_command(args: &[String]) -> Result<(), String> {
    let options = parse_profile_artifact_options(args)?;
    let json = options.json;
    let summary = write_profile_artifacts(&options.options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("profile_path={}", summary.profile_path);
        println!("selected_feature_count={}", summary.selected_feature_count);
        if let Some(path) = summary.config_profile_path {
            println!("config_profile_path={path}");
        }
    }
    Ok(())
}

fn run_build_prefilter_profile_command(args: &[String]) -> Result<(), String> {
    let options = parse_prefilter_profile_build_options(args)?;
    let json = options.json;
    let summary = run_build_prefilter_profile(&options.options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("profile_name={}", summary.profile_name);
        println!("original_features={}", summary.original_features);
        println!("after_prefilter={}", summary.after_prefilter);
        println!(
            "after_exact_duplicate_prune={}",
            summary.after_exact_duplicate_prune
        );
        println!("after_corr_prune={}", summary.after_corr_prune);
        println!("selected_feature_count={}", summary.selected_feature_count);
        println!("profile_path={}", summary.profile_artifacts.profile_path);
    }
    Ok(())
}

fn run_build_robust_profile_command(args: &[String]) -> Result<(), String> {
    let options = parse_robust_profile_build_options(args)?;
    let json = options.json;
    let summary = run_build_robust_profile(&options.options)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("profile_name={}", summary.profile_name);
        println!("original_features={}", summary.original_features);
        println!("after_prefilter={}", summary.after_prefilter);
        println!(
            "after_exact_duplicate_prune={}",
            summary.after_exact_duplicate_prune
        );
        println!("after_corr_prune={}", summary.after_corr_prune);
        println!("selected_feature_count={}", summary.selected_feature_count);
        println!("profile_path={}", summary.profile_artifacts.profile_path);
    }
    Ok(())
}

fn run_single_factor(args: &[String]) -> Result<(), String> {
    let cli = parse_single_factor_options(args)?;
    let options = SingleFactorOptions {
        factor_store: cli.factor_store,
        output_dir: cli.output_dir,
        feature_names: cli.feature_names,
        label_column: cli.label_column,
        signal_horizon: cli.signal_horizon,
        date_start: cli.date_start,
        date_end: cli.date_end,
        universe_name: cli.universe_name,
        universe_dir: cli.universe_dir,
        quantile_bins: cli.quantile_bins,
        top_n: cli.top_n,
        include_details: cli.include_details,
        segments: cli.segments,
        diagnostic_label_space: cli.diagnostic_label_space,
        diagnostic_threshold: cli.diagnostic_threshold,
        industry_neutral: cli.industry_neutral,
        industry_map_path: cli.industry_map_path,
        feature_chunk_size: cli.feature_chunk_size,
        batch_size: cli.batch_size,
        metadata_json_path: cli.metadata_json_path,
        config_snapshot_path: cli.config_snapshot_path,
        benchmark: cli.benchmark,
    };
    let summary = run_single_factor_diagnostics(&options)?;
    if cli.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
    } else {
        println!("output_dir={}", summary.output_dir);
        println!("feature_count={}", summary.feature_count);
        println!("row_count={}", summary.row_count);
        println!("diagnostic_label_space={}", summary.diagnostic_label_space);
        println!("industry_neutral={}", summary.industry_neutral);
        println!("segment_count={}", summary.segment_count);
        println!("elapsed_seconds={}", summary.elapsed_seconds);
    }
    Ok(())
}

fn run_full_space_single_factor_command(args: &[String]) -> Result<(), String> {
    let options = parse_full_space_preset_options(args)?;
    let mut passes = vec![("raw".to_owned(), false)];
    if !options.skip_industry_neutral {
        passes.push(("industry_neutral".to_owned(), true));
    }
    for (pass_name, industry_neutral) in passes {
        let command = build_full_space_preset_command(&options, &pass_name, industry_neutral);
        println!("[*] Executing diagnostics batch:");
        println!(
            "    {}",
            display_command(&prepend_batch_subcommand(&command))
        );
        if options.dry_run {
            continue;
        }
        single_factor_batch::run_single_factor_batch_command(&command)?;
    }
    Ok(())
}

fn run_quality_event_flow_single_factor_command(args: &[String]) -> Result<(), String> {
    let options = parse_quality_event_flow_preset_options(args)?;
    let command = build_quality_event_flow_preset_command(&options);
    println!("[*] Executing diagnostics batch:");
    println!(
        "    {}",
        display_command(&prepend_batch_subcommand(&command))
    );
    if options.dry_run {
        return Ok(());
    }
    single_factor_batch::run_single_factor_batch_command(&command)?;
    Ok(())
}

fn parse_single_factor_options(args: &[String]) -> Result<CliOptions, String> {
    let mut factor_store = None;
    let mut output_dir = None;
    let mut feature_names = Vec::new();
    let mut features_json = Vec::new();
    let mut label_column = None;
    let mut signal_horizon = 10usize;
    let mut date_start = None;
    let mut date_end = None;
    let mut universe_name = "all".to_owned();
    let mut universe_dir = PathBuf::from("data/universes");
    let mut quantile_bins = 5usize;
    let mut top_n = 50usize;
    let mut include_details = true;
    let mut segments = Vec::new();
    let mut diagnostic_label_space = DiagnosticLabelSpace::RawReturn;
    let mut diagnostic_threshold = 0.0f64;
    let mut industry_neutral = false;
    let mut industry_map_path = None;
    let mut feature_chunk_size = 64usize;
    let mut batch_size = 65_536usize;
    let mut metadata_json_path = None;
    let mut config_snapshot_path = None;
    let mut benchmark = BenchmarkOptions::default();
    let mut json = false;

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--no-detail-artifacts" => include_details = false,
            "--industry-neutral" => industry_neutral = true,
            "--json" => json = true,
            "--factor-store" => {
                index += 1;
                factor_store = Some(PathBuf::from(next_value(args, index, "--factor-store")?));
            }
            value if value.starts_with("--factor-store=") => {
                factor_store = Some(PathBuf::from(split_value(value, "--factor-store")?));
            }
            "--output-dir" => {
                index += 1;
                output_dir = Some(PathBuf::from(next_value(args, index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--label-column" => {
                index += 1;
                label_column = Some(next_value(args, index, "--label-column")?);
            }
            value if value.starts_with("--label-column=") => {
                label_column = Some(split_value(value, "--label-column")?);
            }
            "--signal-horizon" => {
                index += 1;
                signal_horizon = parse_usize(
                    next_value(args, index, "--signal-horizon")?,
                    "--signal-horizon",
                )?;
            }
            value if value.starts_with("--signal-horizon=") => {
                signal_horizon =
                    parse_usize(split_value(value, "--signal-horizon")?, "--signal-horizon")?
            }
            "--feature" => {
                index += 1;
                feature_names.push(next_value(args, index, "--feature")?);
            }
            value if value.starts_with("--feature=") => {
                feature_names.push(split_value(value, "--feature")?);
            }
            "--features-json" => {
                index += 1;
                features_json.push(PathBuf::from(next_value(args, index, "--features-json")?));
            }
            value if value.starts_with("--features-json=") => {
                features_json.push(PathBuf::from(split_value(value, "--features-json")?));
            }
            "--date-start" => {
                index += 1;
                date_start = Some(next_value(args, index, "--date-start")?);
            }
            value if value.starts_with("--date-start=") => {
                date_start = Some(split_value(value, "--date-start")?)
            }
            "--date-end" => {
                index += 1;
                date_end = Some(next_value(args, index, "--date-end")?);
            }
            value if value.starts_with("--date-end=") => {
                date_end = Some(split_value(value, "--date-end")?)
            }
            "--universe-name" => {
                index += 1;
                universe_name = next_value(args, index, "--universe-name")?;
            }
            value if value.starts_with("--universe-name=") => {
                universe_name = split_value(value, "--universe-name")?
            }
            "--universe-dir" => {
                index += 1;
                universe_dir = PathBuf::from(next_value(args, index, "--universe-dir")?);
            }
            value if value.starts_with("--universe-dir=") => {
                universe_dir = PathBuf::from(split_value(value, "--universe-dir")?)
            }
            "--quantile-bins" => {
                index += 1;
                quantile_bins = parse_usize(
                    next_value(args, index, "--quantile-bins")?,
                    "--quantile-bins",
                )?;
            }
            value if value.starts_with("--quantile-bins=") => {
                quantile_bins =
                    parse_usize(split_value(value, "--quantile-bins")?, "--quantile-bins")?
            }
            "--top-n" => {
                index += 1;
                top_n = parse_usize(next_value(args, index, "--top-n")?, "--top-n")?;
            }
            value if value.starts_with("--top-n=") => {
                top_n = parse_usize(split_value(value, "--top-n")?, "--top-n")?
            }
            "--segment" => {
                index += 1;
                segments.push(SegmentSpec::parse(&next_value(args, index, "--segment")?)?);
            }
            value if value.starts_with("--segment=") => {
                segments.push(SegmentSpec::parse(&split_value(value, "--segment")?)?)
            }
            "--diagnostic-label-space" => {
                index += 1;
                diagnostic_label_space = DiagnosticLabelSpace::parse(&next_value(
                    args,
                    index,
                    "--diagnostic-label-space",
                )?)?;
            }
            value if value.starts_with("--diagnostic-label-space=") => {
                diagnostic_label_space =
                    DiagnosticLabelSpace::parse(&split_value(value, "--diagnostic-label-space")?)?
            }
            "--diagnostic-threshold" => {
                index += 1;
                diagnostic_threshold = parse_f64(
                    next_value(args, index, "--diagnostic-threshold")?,
                    "--diagnostic-threshold",
                )?;
            }
            value if value.starts_with("--diagnostic-threshold=") => {
                diagnostic_threshold = parse_f64(
                    split_value(value, "--diagnostic-threshold")?,
                    "--diagnostic-threshold",
                )?
            }
            "--benchmark-mode" => {
                index += 1;
                benchmark.mode =
                    BenchmarkMode::parse(&next_value(args, index, "--benchmark-mode")?)?;
            }
            value if value.starts_with("--benchmark-mode=") => {
                benchmark.mode = BenchmarkMode::parse(&split_value(value, "--benchmark-mode")?)?
            }
            "--benchmark-path" => {
                index += 1;
                benchmark.path = Some(PathBuf::from(next_value(args, index, "--benchmark-path")?));
            }
            value if value.starts_with("--benchmark-path=") => {
                benchmark.path = Some(PathBuf::from(split_value(value, "--benchmark-path")?))
            }
            "--benchmark-date-column" => {
                index += 1;
                benchmark.date_column = next_value(args, index, "--benchmark-date-column")?;
            }
            value if value.starts_with("--benchmark-date-column=") => {
                benchmark.date_column = split_value(value, "--benchmark-date-column")?
            }
            "--benchmark-value-column" => {
                index += 1;
                benchmark.value_column = next_value(args, index, "--benchmark-value-column")?;
            }
            value if value.starts_with("--benchmark-value-column=") => {
                benchmark.value_column = split_value(value, "--benchmark-value-column")?
            }
            "--benchmark-value-type" => {
                index += 1;
                benchmark.value_type =
                    BenchmarkValueType::parse(&next_value(args, index, "--benchmark-value-type")?)?;
            }
            value if value.starts_with("--benchmark-value-type=") => {
                benchmark.value_type =
                    BenchmarkValueType::parse(&split_value(value, "--benchmark-value-type")?)?
            }
            "--industry-map" => {
                index += 1;
                industry_map_path = Some(PathBuf::from(next_value(args, index, "--industry-map")?));
            }
            value if value.starts_with("--industry-map=") => {
                industry_map_path = Some(PathBuf::from(split_value(value, "--industry-map")?))
            }
            "--feature-chunk-size" => {
                index += 1;
                feature_chunk_size = parse_usize(
                    next_value(args, index, "--feature-chunk-size")?,
                    "--feature-chunk-size",
                )?;
            }
            value if value.starts_with("--feature-chunk-size=") => {
                feature_chunk_size = parse_usize(
                    split_value(value, "--feature-chunk-size")?,
                    "--feature-chunk-size",
                )?
            }
            "--batch-size" => {
                index += 1;
                batch_size = parse_usize(next_value(args, index, "--batch-size")?, "--batch-size")?;
            }
            value if value.starts_with("--batch-size=") => {
                batch_size = parse_usize(split_value(value, "--batch-size")?, "--batch-size")?
            }
            "--metadata-json" => {
                index += 1;
                metadata_json_path =
                    Some(PathBuf::from(next_value(args, index, "--metadata-json")?));
            }
            value if value.starts_with("--metadata-json=") => {
                metadata_json_path = Some(PathBuf::from(split_value(value, "--metadata-json")?))
            }
            "--config-snapshot" => {
                index += 1;
                config_snapshot_path =
                    Some(PathBuf::from(next_value(args, index, "--config-snapshot")?));
            }
            value if value.starts_with("--config-snapshot=") => {
                config_snapshot_path = Some(PathBuf::from(split_value(value, "--config-snapshot")?))
            }
            other => return Err(format!("unknown option for single-factor: {other}")),
        }
        index += 1;
    }

    for path in features_json {
        feature_names.extend(read_feature_list_json(&path)?);
    }
    feature_names = sorted_unique_nonempty(feature_names);
    if feature_names.is_empty() {
        return Err("at least one --feature or --features-json feature is required".to_owned());
    }
    if benchmark.date_column.trim().is_empty() {
        return Err("--benchmark-date-column must be non-empty".to_owned());
    }
    if benchmark.value_column.trim().is_empty() {
        return Err("--benchmark-value-column must be non-empty".to_owned());
    }
    Ok(CliOptions {
        factor_store: factor_store.ok_or_else(|| "--factor-store is required".to_owned())?,
        output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
        feature_names,
        label_column: label_column.ok_or_else(|| "--label-column is required".to_owned())?,
        signal_horizon,
        date_start,
        date_end,
        universe_name,
        universe_dir,
        quantile_bins,
        top_n,
        include_details,
        segments,
        diagnostic_label_space,
        diagnostic_threshold,
        industry_neutral,
        industry_map_path,
        feature_chunk_size,
        batch_size,
        metadata_json_path,
        config_snapshot_path,
        benchmark,
        json,
    })
}

fn parse_prefilter_options(args: &[String]) -> Result<PrefilterCliOptions, String> {
    let mut diagnostics_summary = None;
    let mut segment_comparison = None;
    let mut output_dir = None;
    let mut thresholds = PrefilterThresholds::default();
    let mut json = false;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--diagnostics-summary" => {
                index += 1;
                diagnostics_summary = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--diagnostics-summary",
                )?));
            }
            value if value.starts_with("--diagnostics-summary=") => {
                diagnostics_summary =
                    Some(PathBuf::from(split_value(value, "--diagnostics-summary")?));
            }
            "--segment-comparison" => {
                index += 1;
                segment_comparison = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--segment-comparison",
                )?));
            }
            value if value.starts_with("--segment-comparison=") => {
                segment_comparison =
                    Some(PathBuf::from(split_value(value, "--segment-comparison")?));
            }
            "--output-dir" => {
                index += 1;
                output_dir = Some(PathBuf::from(next_value(args, index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--min-coverage-pct" => {
                index += 1;
                thresholds.min_coverage_pct = parse_f64(
                    next_value(args, index, "--min-coverage-pct")?,
                    "--min-coverage-pct",
                )?;
            }
            value if value.starts_with("--min-coverage-pct=") => {
                thresholds.min_coverage_pct = parse_f64(
                    split_value(value, "--min-coverage-pct")?,
                    "--min-coverage-pct",
                )?;
            }
            "--min-abs-rank-ic" => {
                index += 1;
                thresholds.min_abs_rank_ic = parse_f64(
                    next_value(args, index, "--min-abs-rank-ic")?,
                    "--min-abs-rank-ic",
                )?;
            }
            value if value.starts_with("--min-abs-rank-ic=") => {
                thresholds.min_abs_rank_ic = parse_f64(
                    split_value(value, "--min-abs-rank-ic")?,
                    "--min-abs-rank-ic",
                )?;
            }
            "--min-abs-rank-ic-ir" => {
                index += 1;
                thresholds.min_abs_rank_ic_ir = parse_f64(
                    next_value(args, index, "--min-abs-rank-ic-ir")?,
                    "--min-abs-rank-ic-ir",
                )?;
            }
            value if value.starts_with("--min-abs-rank-ic-ir=") => {
                thresholds.min_abs_rank_ic_ir = parse_f64(
                    split_value(value, "--min-abs-rank-ic-ir")?,
                    "--min-abs-rank-ic-ir",
                )?;
            }
            "--min-monthly-positive-rate" => {
                index += 1;
                thresholds.min_monthly_positive_rate = parse_f64(
                    next_value(args, index, "--min-monthly-positive-rate")?,
                    "--min-monthly-positive-rate",
                )?;
            }
            value if value.starts_with("--min-monthly-positive-rate=") => {
                thresholds.min_monthly_positive_rate = parse_f64(
                    split_value(value, "--min-monthly-positive-rate")?,
                    "--min-monthly-positive-rate",
                )?;
            }
            "--min-segment-directional-hit-mean" => {
                index += 1;
                thresholds.min_segment_directional_hit_mean = Some(parse_f64(
                    next_value(args, index, "--min-segment-directional-hit-mean")?,
                    "--min-segment-directional-hit-mean",
                )?);
            }
            value if value.starts_with("--min-segment-directional-hit-mean=") => {
                thresholds.min_segment_directional_hit_mean = Some(parse_f64(
                    split_value(value, "--min-segment-directional-hit-mean")?,
                    "--min-segment-directional-hit-mean",
                )?);
            }
            "--max-segment-rank-ic-mean-range" => {
                index += 1;
                thresholds.max_segment_rank_ic_mean_range = Some(parse_f64(
                    next_value(args, index, "--max-segment-rank-ic-mean-range")?,
                    "--max-segment-rank-ic-mean-range",
                )?);
            }
            value if value.starts_with("--max-segment-rank-ic-mean-range=") => {
                thresholds.max_segment_rank_ic_mean_range = Some(parse_f64(
                    split_value(value, "--max-segment-rank-ic-mean-range")?,
                    "--max-segment-rank-ic-mean-range",
                )?);
            }
            "--exclude-direction-flip" => thresholds.exclude_direction_flip = true,
            other => return Err(format!("unknown option for prefilter-summary: {other}")),
        }
        index += 1;
    }
    Ok(PrefilterCliOptions {
        options: PrefilterOptions {
            diagnostics_summary: diagnostics_summary
                .ok_or_else(|| "--diagnostics-summary is required".to_owned())?,
            segment_comparison,
            output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
            thresholds,
        },
        json,
    })
}

fn parse_robust_prefilter_options(args: &[String]) -> Result<RobustPrefilterCliOptions, String> {
    let mut raw_summary = None;
    let mut neutral_summary = None;
    let mut raw_segment_comparison = None;
    let mut neutral_segment_comparison = None;
    let mut output_dir = None;
    let mut thresholds = PrefilterThresholds {
        min_segment_directional_hit_mean: Some(0.55),
        max_segment_rank_ic_mean_range: Some(0.14),
        ..Default::default()
    };
    let mut json = false;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--raw-summary" => {
                index += 1;
                raw_summary = Some(PathBuf::from(next_value(args, index, "--raw-summary")?));
            }
            value if value.starts_with("--raw-summary=") => {
                raw_summary = Some(PathBuf::from(split_value(value, "--raw-summary")?));
            }
            "--neutral-summary" => {
                index += 1;
                neutral_summary =
                    Some(PathBuf::from(next_value(args, index, "--neutral-summary")?));
            }
            value if value.starts_with("--neutral-summary=") => {
                neutral_summary = Some(PathBuf::from(split_value(value, "--neutral-summary")?));
            }
            "--raw-segment-comparison" => {
                index += 1;
                raw_segment_comparison = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--raw-segment-comparison",
                )?));
            }
            value if value.starts_with("--raw-segment-comparison=") => {
                raw_segment_comparison = Some(PathBuf::from(split_value(
                    value,
                    "--raw-segment-comparison",
                )?));
            }
            "--neutral-segment-comparison" => {
                index += 1;
                neutral_segment_comparison = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--neutral-segment-comparison",
                )?));
            }
            value if value.starts_with("--neutral-segment-comparison=") => {
                neutral_segment_comparison = Some(PathBuf::from(split_value(
                    value,
                    "--neutral-segment-comparison",
                )?));
            }
            "--output-dir" => {
                index += 1;
                output_dir = Some(PathBuf::from(next_value(args, index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            other => {
                let parsed =
                    parse_prefilter_threshold_option(other, args, &mut index, &mut thresholds)?;
                if !parsed {
                    return Err(format!(
                        "unknown option for robust-prefilter-summary: {other}"
                    ));
                }
            }
        }
        index += 1;
    }
    Ok(RobustPrefilterCliOptions {
        options: RobustPrefilterOptions {
            raw_summary: raw_summary.ok_or_else(|| "--raw-summary is required".to_owned())?,
            neutral_summary: neutral_summary
                .ok_or_else(|| "--neutral-summary is required".to_owned())?,
            raw_segment_comparison,
            neutral_segment_comparison,
            output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
            thresholds,
        },
        json,
    })
}

fn parse_corr_prune_options(args: &[String]) -> Result<CorrPruneCliOptions, String> {
    let mut factor_store = None;
    let mut candidates_csv = None;
    let mut output_dir = None;
    let mut date_start = None;
    let mut date_end = None;
    let mut universe_name = "all".to_owned();
    let mut universe_dir = PathBuf::from("data/universes");
    let mut corr_threshold = 0.97f64;
    let mut use_cross_sectional_rank = true;
    let mut batch_size = 65_536usize;
    let mut json = false;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--factor-store" => {
                index += 1;
                factor_store = Some(PathBuf::from(next_value(args, index, "--factor-store")?));
            }
            value if value.starts_with("--factor-store=") => {
                factor_store = Some(PathBuf::from(split_value(value, "--factor-store")?));
            }
            "--candidates" => {
                index += 1;
                candidates_csv = Some(PathBuf::from(next_value(args, index, "--candidates")?));
            }
            value if value.starts_with("--candidates=") => {
                candidates_csv = Some(PathBuf::from(split_value(value, "--candidates")?));
            }
            "--output-dir" => {
                index += 1;
                output_dir = Some(PathBuf::from(next_value(args, index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--date-start" => {
                index += 1;
                date_start = Some(next_value(args, index, "--date-start")?);
            }
            value if value.starts_with("--date-start=") => {
                date_start = Some(split_value(value, "--date-start")?);
            }
            "--date-end" => {
                index += 1;
                date_end = Some(next_value(args, index, "--date-end")?);
            }
            value if value.starts_with("--date-end=") => {
                date_end = Some(split_value(value, "--date-end")?);
            }
            "--universe-name" => {
                index += 1;
                universe_name = next_value(args, index, "--universe-name")?;
            }
            value if value.starts_with("--universe-name=") => {
                universe_name = split_value(value, "--universe-name")?;
            }
            "--universe-dir" => {
                index += 1;
                universe_dir = PathBuf::from(next_value(args, index, "--universe-dir")?);
            }
            value if value.starts_with("--universe-dir=") => {
                universe_dir = PathBuf::from(split_value(value, "--universe-dir")?);
            }
            "--corr-threshold" => {
                index += 1;
                corr_threshold = parse_f64(
                    next_value(args, index, "--corr-threshold")?,
                    "--corr-threshold",
                )?;
            }
            value if value.starts_with("--corr-threshold=") => {
                corr_threshold =
                    parse_f64(split_value(value, "--corr-threshold")?, "--corr-threshold")?;
            }
            "--cross-sectional-rank" => use_cross_sectional_rank = true,
            "--raw-values" | "--no-cross-sectional-rank" => use_cross_sectional_rank = false,
            "--batch-size" => {
                index += 1;
                batch_size = parse_usize(next_value(args, index, "--batch-size")?, "--batch-size")?;
            }
            value if value.starts_with("--batch-size=") => {
                batch_size = parse_usize(split_value(value, "--batch-size")?, "--batch-size")?;
            }
            other => return Err(format!("unknown option for corr-prune: {other}")),
        }
        index += 1;
    }
    Ok(CorrPruneCliOptions {
        options: CorrPruneOptions {
            factor_store: factor_store.ok_or_else(|| "--factor-store is required".to_owned())?,
            candidates_csv: candidates_csv.ok_or_else(|| "--candidates is required".to_owned())?,
            output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
            date_start,
            date_end,
            universe_name,
            universe_dir,
            corr_threshold,
            use_cross_sectional_rank,
            batch_size,
        },
        json,
    })
}

fn parse_profile_artifact_options(args: &[String]) -> Result<ProfileArtifactCliOptions, String> {
    let mut selected_csv = None;
    let mut output_dir = None;
    let mut profile_name = None;
    let mut max_features = None;
    let mut write_config_profile = false;
    let mut config_profile_path = None;
    let mut factor_store_name = "full_factor_space".to_owned();
    let mut readme_mode = ProfileReadmeMode::Prefilter;
    let mut settings = Vec::new();
    let mut safety_warning = None;
    let mut json = false;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--write-config-profile" => write_config_profile = true,
            "--selected-csv" => {
                index += 1;
                selected_csv = Some(PathBuf::from(next_value(args, index, "--selected-csv")?));
            }
            value if value.starts_with("--selected-csv=") => {
                selected_csv = Some(PathBuf::from(split_value(value, "--selected-csv")?));
            }
            "--output-dir" => {
                index += 1;
                output_dir = Some(PathBuf::from(next_value(args, index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--profile-name" => {
                index += 1;
                profile_name = Some(next_value(args, index, "--profile-name")?);
            }
            value if value.starts_with("--profile-name=") => {
                profile_name = Some(split_value(value, "--profile-name")?);
            }
            "--max-features" => {
                index += 1;
                max_features = Some(parse_usize(
                    next_value(args, index, "--max-features")?,
                    "--max-features",
                )?);
            }
            value if value.starts_with("--max-features=") => {
                max_features = Some(parse_usize(
                    split_value(value, "--max-features")?,
                    "--max-features",
                )?);
            }
            "--config-profile-path" => {
                index += 1;
                config_profile_path = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--config-profile-path",
                )?));
            }
            value if value.starts_with("--config-profile-path=") => {
                config_profile_path =
                    Some(PathBuf::from(split_value(value, "--config-profile-path")?));
            }
            "--factor-store-name" => {
                index += 1;
                factor_store_name = next_value(args, index, "--factor-store-name")?;
            }
            value if value.starts_with("--factor-store-name=") => {
                factor_store_name = split_value(value, "--factor-store-name")?;
            }
            "--readme-mode" => {
                index += 1;
                readme_mode = parse_readme_mode(&next_value(args, index, "--readme-mode")?)?;
            }
            value if value.starts_with("--readme-mode=") => {
                readme_mode = parse_readme_mode(&split_value(value, "--readme-mode")?)?;
            }
            "--setting" => {
                index += 1;
                settings.push(parse_setting(&next_value(args, index, "--setting")?)?);
            }
            value if value.starts_with("--setting=") => {
                settings.push(parse_setting(&split_value(value, "--setting")?)?);
            }
            "--safety-warning" => {
                index += 1;
                safety_warning = Some(next_value(args, index, "--safety-warning")?);
            }
            value if value.starts_with("--safety-warning=") => {
                safety_warning = Some(split_value(value, "--safety-warning")?);
            }
            other => return Err(format!("unknown option for write-profile: {other}")),
        }
        index += 1;
    }
    Ok(ProfileArtifactCliOptions {
        options: ProfileArtifactOptions {
            output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
            selected_csv: selected_csv.ok_or_else(|| "--selected-csv is required".to_owned())?,
            profile_name: profile_name.ok_or_else(|| "--profile-name is required".to_owned())?,
            max_features,
            write_config_profile,
            config_profile_path,
            factor_store_name,
            readme_mode,
            settings,
            safety_warning,
        },
        json,
    })
}

#[derive(Debug, Clone)]
struct ProfileBuildCommonOptions {
    output_dir: Option<PathBuf>,
    factor_store: Option<PathBuf>,
    date_start: Option<String>,
    date_end: Option<String>,
    universe_name: String,
    universe_dir: PathBuf,
    corr_threshold: f64,
    use_cross_sectional_rank: bool,
    batch_size: usize,
    profile_name: Option<String>,
    max_features: Option<usize>,
    write_config_profile: bool,
    config_profile_path: Option<PathBuf>,
    factor_store_name: String,
    settings: Vec<(String, String)>,
    safety_warning: Option<String>,
    json: bool,
}

impl Default for ProfileBuildCommonOptions {
    fn default() -> Self {
        Self {
            output_dir: None,
            factor_store: None,
            date_start: None,
            date_end: None,
            universe_name: "all".to_owned(),
            universe_dir: PathBuf::from("data/universes"),
            corr_threshold: 0.97,
            use_cross_sectional_rank: true,
            batch_size: 65_536,
            profile_name: None,
            max_features: None,
            write_config_profile: false,
            config_profile_path: None,
            factor_store_name: "full_factor_space".to_owned(),
            settings: Vec::new(),
            safety_warning: None,
            json: false,
        }
    }
}

fn parse_prefilter_profile_build_options(
    args: &[String],
) -> Result<PrefilterProfileBuildCliOptions, String> {
    let mut diagnostics_summary = None;
    let mut segment_comparison = None;
    let mut thresholds = PrefilterThresholds::default();
    let mut common = ProfileBuildCommonOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--diagnostics-summary" => {
                index += 1;
                diagnostics_summary = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--diagnostics-summary",
                )?));
            }
            value if value.starts_with("--diagnostics-summary=") => {
                diagnostics_summary =
                    Some(PathBuf::from(split_value(value, "--diagnostics-summary")?));
            }
            "--segment-comparison" => {
                index += 1;
                segment_comparison = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--segment-comparison",
                )?));
            }
            value if value.starts_with("--segment-comparison=") => {
                segment_comparison =
                    Some(PathBuf::from(split_value(value, "--segment-comparison")?));
            }
            other => {
                if !parse_prefilter_threshold_option(other, args, &mut index, &mut thresholds)?
                    && !parse_profile_build_common_option(other, args, &mut index, &mut common)?
                {
                    return Err(format!(
                        "unknown option for build-prefilter-profile: {other}"
                    ));
                }
            }
        }
        index += 1;
    }
    Ok(PrefilterProfileBuildCliOptions {
        json: common.json,
        options: PrefilterProfileBuildOptions {
            diagnostics_summary: diagnostics_summary
                .ok_or_else(|| "--diagnostics-summary is required".to_owned())?,
            segment_comparison,
            output_dir: common
                .output_dir
                .ok_or_else(|| "--output-dir is required".to_owned())?,
            thresholds,
            factor_store: common
                .factor_store
                .ok_or_else(|| "--factor-store is required".to_owned())?,
            date_start: common.date_start,
            date_end: common.date_end,
            universe_name: common.universe_name,
            universe_dir: common.universe_dir,
            corr_threshold: common.corr_threshold,
            use_cross_sectional_rank: common.use_cross_sectional_rank,
            batch_size: common.batch_size,
            profile_name: common
                .profile_name
                .ok_or_else(|| "--profile-name is required".to_owned())?,
            max_features: common.max_features,
            write_config_profile: common.write_config_profile,
            config_profile_path: common.config_profile_path,
            factor_store_name: common.factor_store_name,
            settings: common.settings,
            safety_warning: common.safety_warning,
        },
    })
}

fn parse_robust_profile_build_options(
    args: &[String],
) -> Result<RobustProfileBuildCliOptions, String> {
    let mut raw_summary = None;
    let mut neutral_summary = None;
    let mut raw_segment_comparison = None;
    let mut neutral_segment_comparison = None;
    let mut thresholds = PrefilterThresholds {
        min_segment_directional_hit_mean: Some(0.55),
        max_segment_rank_ic_mean_range: Some(0.14),
        ..Default::default()
    };
    let mut common = ProfileBuildCommonOptions::default();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--raw-summary" => {
                index += 1;
                raw_summary = Some(PathBuf::from(next_value(args, index, "--raw-summary")?));
            }
            value if value.starts_with("--raw-summary=") => {
                raw_summary = Some(PathBuf::from(split_value(value, "--raw-summary")?));
            }
            "--neutral-summary" => {
                index += 1;
                neutral_summary =
                    Some(PathBuf::from(next_value(args, index, "--neutral-summary")?));
            }
            value if value.starts_with("--neutral-summary=") => {
                neutral_summary = Some(PathBuf::from(split_value(value, "--neutral-summary")?));
            }
            "--raw-segment-comparison" => {
                index += 1;
                raw_segment_comparison = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--raw-segment-comparison",
                )?));
            }
            value if value.starts_with("--raw-segment-comparison=") => {
                raw_segment_comparison = Some(PathBuf::from(split_value(
                    value,
                    "--raw-segment-comparison",
                )?));
            }
            "--neutral-segment-comparison" => {
                index += 1;
                neutral_segment_comparison = Some(PathBuf::from(next_value(
                    args,
                    index,
                    "--neutral-segment-comparison",
                )?));
            }
            value if value.starts_with("--neutral-segment-comparison=") => {
                neutral_segment_comparison = Some(PathBuf::from(split_value(
                    value,
                    "--neutral-segment-comparison",
                )?));
            }
            other => {
                if !parse_prefilter_threshold_option(other, args, &mut index, &mut thresholds)?
                    && !parse_profile_build_common_option(other, args, &mut index, &mut common)?
                {
                    return Err(format!("unknown option for build-robust-profile: {other}"));
                }
            }
        }
        index += 1;
    }
    Ok(RobustProfileBuildCliOptions {
        json: common.json,
        options: RobustProfileBuildOptions {
            raw_summary: raw_summary.ok_or_else(|| "--raw-summary is required".to_owned())?,
            neutral_summary: neutral_summary
                .ok_or_else(|| "--neutral-summary is required".to_owned())?,
            raw_segment_comparison,
            neutral_segment_comparison,
            output_dir: common
                .output_dir
                .ok_or_else(|| "--output-dir is required".to_owned())?,
            thresholds,
            factor_store: common
                .factor_store
                .ok_or_else(|| "--factor-store is required".to_owned())?,
            date_start: common.date_start,
            date_end: common.date_end,
            universe_name: common.universe_name,
            universe_dir: common.universe_dir,
            corr_threshold: common.corr_threshold,
            use_cross_sectional_rank: common.use_cross_sectional_rank,
            batch_size: common.batch_size,
            profile_name: common
                .profile_name
                .ok_or_else(|| "--profile-name is required".to_owned())?,
            max_features: common.max_features,
            write_config_profile: common.write_config_profile,
            config_profile_path: common.config_profile_path,
            factor_store_name: common.factor_store_name,
            settings: common.settings,
            safety_warning: common.safety_warning,
        },
    })
}

fn parse_profile_build_common_option(
    option: &str,
    args: &[String],
    index: &mut usize,
    common: &mut ProfileBuildCommonOptions,
) -> Result<bool, String> {
    match option {
        "--json" => common.json = true,
        "--write-config-profile" => common.write_config_profile = true,
        "--factor-store" => {
            *index += 1;
            common.factor_store = Some(PathBuf::from(next_value(args, *index, option)?));
        }
        value if value.starts_with("--factor-store=") => {
            common.factor_store = Some(PathBuf::from(split_value(value, "--factor-store")?));
        }
        "--output-dir" => {
            *index += 1;
            common.output_dir = Some(PathBuf::from(next_value(args, *index, option)?));
        }
        value if value.starts_with("--output-dir=") => {
            common.output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
        }
        "--date-start" => {
            *index += 1;
            common.date_start = Some(next_value(args, *index, option)?);
        }
        value if value.starts_with("--date-start=") => {
            common.date_start = Some(split_value(value, "--date-start")?);
        }
        "--date-end" => {
            *index += 1;
            common.date_end = Some(next_value(args, *index, option)?);
        }
        value if value.starts_with("--date-end=") => {
            common.date_end = Some(split_value(value, "--date-end")?);
        }
        "--universe-name" => {
            *index += 1;
            common.universe_name = next_value(args, *index, option)?;
        }
        value if value.starts_with("--universe-name=") => {
            common.universe_name = split_value(value, "--universe-name")?;
        }
        "--universe-dir" => {
            *index += 1;
            common.universe_dir = PathBuf::from(next_value(args, *index, option)?);
        }
        value if value.starts_with("--universe-dir=") => {
            common.universe_dir = PathBuf::from(split_value(value, "--universe-dir")?);
        }
        "--corr-threshold" | "--max-abs-corr" => {
            *index += 1;
            common.corr_threshold = parse_f64(next_value(args, *index, option)?, option)?;
        }
        value if value.starts_with("--corr-threshold=") => {
            common.corr_threshold =
                parse_f64(split_value(value, "--corr-threshold")?, "--corr-threshold")?;
        }
        value if value.starts_with("--max-abs-corr=") => {
            common.corr_threshold =
                parse_f64(split_value(value, "--max-abs-corr")?, "--max-abs-corr")?;
        }
        "--cross-sectional-rank" => common.use_cross_sectional_rank = true,
        "--raw-values" | "--no-cross-sectional-rank" | "--no-cross-sectional-rank-corr" => {
            common.use_cross_sectional_rank = false;
        }
        "--batch-size" => {
            *index += 1;
            common.batch_size = parse_usize(next_value(args, *index, option)?, option)?;
        }
        value if value.starts_with("--batch-size=") => {
            common.batch_size = parse_usize(split_value(value, "--batch-size")?, "--batch-size")?;
        }
        "--profile-name" => {
            *index += 1;
            common.profile_name = Some(next_value(args, *index, option)?);
        }
        value if value.starts_with("--profile-name=") => {
            common.profile_name = Some(split_value(value, "--profile-name")?);
        }
        "--max-features" => {
            *index += 1;
            common.max_features = Some(parse_usize(next_value(args, *index, option)?, option)?);
        }
        value if value.starts_with("--max-features=") => {
            common.max_features = Some(parse_usize(
                split_value(value, "--max-features")?,
                "--max-features",
            )?);
        }
        "--config-profile-path" => {
            *index += 1;
            common.config_profile_path = Some(PathBuf::from(next_value(args, *index, option)?));
        }
        value if value.starts_with("--config-profile-path=") => {
            common.config_profile_path =
                Some(PathBuf::from(split_value(value, "--config-profile-path")?));
        }
        "--factor-store-name" => {
            *index += 1;
            common.factor_store_name = next_value(args, *index, option)?;
        }
        value if value.starts_with("--factor-store-name=") => {
            common.factor_store_name = split_value(value, "--factor-store-name")?;
        }
        "--setting" => {
            *index += 1;
            common
                .settings
                .push(parse_setting(&next_value(args, *index, option)?)?);
        }
        value if value.starts_with("--setting=") => {
            common
                .settings
                .push(parse_setting(&split_value(value, "--setting")?)?);
        }
        "--safety-warning" => {
            *index += 1;
            common.safety_warning = Some(next_value(args, *index, option)?);
        }
        value if value.starts_with("--safety-warning=") => {
            common.safety_warning = Some(split_value(value, "--safety-warning")?);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn parse_readme_mode(raw: &str) -> Result<ProfileReadmeMode, String> {
    match raw.trim() {
        "prefilter" => Ok(ProfileReadmeMode::Prefilter),
        "robust" => Ok(ProfileReadmeMode::Robust),
        other => Err(format!(
            "invalid --readme-mode {other}; expected prefilter or robust"
        )),
    }
}

fn parse_setting(raw: &str) -> Result<(String, String), String> {
    let (key, value) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid --setting {raw}; expected KEY=VALUE"))?;
    let key = key.trim();
    if key.is_empty() {
        return Err("invalid --setting with empty key".to_owned());
    }
    Ok((key.to_owned(), value.to_owned()))
}

fn parse_prefilter_threshold_option(
    option: &str,
    args: &[String],
    index: &mut usize,
    thresholds: &mut PrefilterThresholds,
) -> Result<bool, String> {
    match option {
        "--min-coverage-pct" => {
            *index += 1;
            thresholds.min_coverage_pct = parse_f64(next_value(args, *index, option)?, option)?;
        }
        value if value.starts_with("--min-coverage-pct=") => {
            thresholds.min_coverage_pct = parse_f64(
                split_value(value, "--min-coverage-pct")?,
                "--min-coverage-pct",
            )?;
        }
        "--min-abs-rank-ic" => {
            *index += 1;
            thresholds.min_abs_rank_ic = parse_f64(next_value(args, *index, option)?, option)?;
        }
        value if value.starts_with("--min-abs-rank-ic=") => {
            thresholds.min_abs_rank_ic = parse_f64(
                split_value(value, "--min-abs-rank-ic")?,
                "--min-abs-rank-ic",
            )?;
        }
        "--min-abs-rank-ic-ir" => {
            *index += 1;
            thresholds.min_abs_rank_ic_ir = parse_f64(next_value(args, *index, option)?, option)?;
        }
        value if value.starts_with("--min-abs-rank-ic-ir=") => {
            thresholds.min_abs_rank_ic_ir = parse_f64(
                split_value(value, "--min-abs-rank-ic-ir")?,
                "--min-abs-rank-ic-ir",
            )?;
        }
        "--min-monthly-positive-rate" => {
            *index += 1;
            thresholds.min_monthly_positive_rate =
                parse_f64(next_value(args, *index, option)?, option)?;
        }
        value if value.starts_with("--min-monthly-positive-rate=") => {
            thresholds.min_monthly_positive_rate = parse_f64(
                split_value(value, "--min-monthly-positive-rate")?,
                "--min-monthly-positive-rate",
            )?;
        }
        "--min-segment-directional-hit-mean" => {
            *index += 1;
            thresholds.min_segment_directional_hit_mean =
                Some(parse_f64(next_value(args, *index, option)?, option)?);
        }
        value if value.starts_with("--min-segment-directional-hit-mean=") => {
            thresholds.min_segment_directional_hit_mean = Some(parse_f64(
                split_value(value, "--min-segment-directional-hit-mean")?,
                "--min-segment-directional-hit-mean",
            )?);
        }
        "--max-segment-rank-ic-mean-range" => {
            *index += 1;
            thresholds.max_segment_rank_ic_mean_range =
                Some(parse_f64(next_value(args, *index, option)?, option)?);
        }
        value if value.starts_with("--max-segment-rank-ic-mean-range=") => {
            thresholds.max_segment_rank_ic_mean_range = Some(parse_f64(
                split_value(value, "--max-segment-rank-ic-mean-range")?,
                "--max-segment-rank-ic-mean-range",
            )?);
        }
        "--exclude-direction-flip" => thresholds.exclude_direction_flip = true,
        _ => return Ok(false),
    }
    Ok(true)
}

fn read_feature_list_json(path: &Path) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: JsonValue = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
    let values = match value {
        JsonValue::Array(values) => values,
        JsonValue::Object(mut object) => object
            .remove("selected_features")
            .or_else(|| object.remove("features"))
            .and_then(|item| item.as_array().cloned())
            .ok_or_else(|| {
                format!(
                    "{} must be a JSON list or object with selected_features/features",
                    path.display()
                )
            })?,
        _ => {
            return Err(format!(
                "{} must be a JSON list or object with selected_features/features",
                path.display()
            ))
        }
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{} contains non-string feature value", path.display()))
        })
        .collect()
}

fn sorted_unique_nonempty(values: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if !trimmed.is_empty() && seen.insert(trimmed.to_owned()) {
            out.push(trimmed.to_owned());
        }
    }
    out
}

fn parse_full_space_preset_options(args: &[String]) -> Result<FullSpacePresetOptions, String> {
    let mut options = FullSpacePresetOptions {
        config: "configs/config.yaml".to_owned(),
        experiment_profile: String::new(),
        model_profile: None,
        data_source: None,
        period: "train".to_owned(),
        date_start: None,
        date_end: None,
        quantile_bins: 5,
        top_n: 100,
        segment_scheme: "yearly".to_owned(),
        segments: None,
        base_output_dir: None,
        run_tag: None,
        diagnostic_label_space: "raw_return".to_owned(),
        diagnostic_threshold: 0.0,
        skip_industry_neutral: false,
        set_overrides: Vec::new(),
        dry_run: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--config" => {
                index += 1;
                options.config = next_value(args, index, "--config")?;
            }
            "--experiment-profile" => {
                index += 1;
                options.experiment_profile = next_value(args, index, "--experiment-profile")?;
            }
            "--model-profile" => {
                index += 1;
                options.model_profile = Some(next_value(args, index, "--model-profile")?);
            }
            "--data-source" => {
                index += 1;
                options.data_source = Some(next_value(args, index, "--data-source")?);
            }
            "--period" => {
                index += 1;
                options.period = next_value(args, index, "--period")?;
            }
            "--date-start" => {
                index += 1;
                options.date_start = Some(next_value(args, index, "--date-start")?);
            }
            "--date-end" => {
                index += 1;
                options.date_end = Some(next_value(args, index, "--date-end")?);
            }
            "--quantile-bins" => {
                index += 1;
                options.quantile_bins = parse_usize(
                    next_value(args, index, "--quantile-bins")?,
                    "--quantile-bins",
                )?
                .max(2);
            }
            "--top-n" => {
                index += 1;
                options.top_n = parse_usize(next_value(args, index, "--top-n")?, "--top-n")?.max(1);
            }
            "--segment-scheme" => {
                index += 1;
                options.segment_scheme = next_value(args, index, "--segment-scheme")?;
            }
            "--segments" => {
                index += 1;
                options.segments = Some(next_value(args, index, "--segments")?);
            }
            "--base-output-dir" => {
                index += 1;
                options.base_output_dir =
                    Some(PathBuf::from(next_value(args, index, "--base-output-dir")?));
            }
            "--run-tag" => {
                index += 1;
                options.run_tag = Some(next_value(args, index, "--run-tag")?);
            }
            "--diagnostic-label-space" => {
                index += 1;
                options.diagnostic_label_space =
                    next_value(args, index, "--diagnostic-label-space")?;
            }
            "--diagnostic-threshold" => {
                index += 1;
                options.diagnostic_threshold = parse_f64(
                    next_value(args, index, "--diagnostic-threshold")?,
                    "--diagnostic-threshold",
                )?;
            }
            "--skip-industry-neutral" => options.skip_industry_neutral = true,
            "--set" => {
                index += 1;
                options
                    .set_overrides
                    .push(next_value(args, index, "--set")?);
            }
            "--dry-run" => options.dry_run = true,
            other => return Err(format!("unknown full-space preset option: {other}")),
        }
        index += 1;
    }
    if options.experiment_profile.trim().is_empty() {
        return Err("--experiment-profile is required".to_owned());
    }
    Ok(options)
}

fn parse_quality_event_flow_preset_options(
    args: &[String],
) -> Result<QualityEventFlowPresetOptions, String> {
    let mut options = QualityEventFlowPresetOptions {
        config: "configs/config.yaml".to_owned(),
        experiment_profile: String::new(),
        feature_profile: "core_v4_techlite_tushare_plus_quality_event_flow_v1".to_owned(),
        model_profile: None,
        data_source: None,
        period: "train".to_owned(),
        date_start: None,
        date_end: None,
        quantile_bins: 5,
        top_n: 50,
        segment_scheme: "config_split".to_owned(),
        segments: None,
        base_output_dir: None,
        run_tag: None,
        include_benchmark_excess: false,
        set_overrides: Vec::new(),
        dry_run: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--config" => {
                index += 1;
                options.config = next_value(args, index, "--config")?;
            }
            "--experiment-profile" => {
                index += 1;
                options.experiment_profile = next_value(args, index, "--experiment-profile")?;
            }
            "--feature-profile" => {
                index += 1;
                options.feature_profile = next_value(args, index, "--feature-profile")?;
            }
            "--model-profile" => {
                index += 1;
                options.model_profile = Some(next_value(args, index, "--model-profile")?);
            }
            "--data-source" => {
                index += 1;
                options.data_source = Some(next_value(args, index, "--data-source")?);
            }
            "--period" => {
                index += 1;
                options.period = next_value(args, index, "--period")?;
            }
            "--date-start" => {
                index += 1;
                options.date_start = Some(next_value(args, index, "--date-start")?);
            }
            "--date-end" => {
                index += 1;
                options.date_end = Some(next_value(args, index, "--date-end")?);
            }
            "--quantile-bins" => {
                index += 1;
                options.quantile_bins = parse_usize(
                    next_value(args, index, "--quantile-bins")?,
                    "--quantile-bins",
                )?
                .max(2);
            }
            "--top-n" => {
                index += 1;
                options.top_n = parse_usize(next_value(args, index, "--top-n")?, "--top-n")?.max(1);
            }
            "--segment-scheme" => {
                index += 1;
                options.segment_scheme = next_value(args, index, "--segment-scheme")?;
            }
            "--segments" => {
                index += 1;
                options.segments = Some(next_value(args, index, "--segments")?);
            }
            "--base-output-dir" => {
                index += 1;
                options.base_output_dir =
                    Some(PathBuf::from(next_value(args, index, "--base-output-dir")?));
            }
            "--run-tag" => {
                index += 1;
                options.run_tag = Some(next_value(args, index, "--run-tag")?);
            }
            "--include-benchmark-excess" => options.include_benchmark_excess = true,
            "--set" => {
                index += 1;
                options
                    .set_overrides
                    .push(next_value(args, index, "--set")?);
            }
            "--dry-run" => options.dry_run = true,
            other => return Err(format!("unknown quality/event preset option: {other}")),
        }
        index += 1;
    }
    if options.experiment_profile.trim().is_empty() {
        return Err("--experiment-profile is required".to_owned());
    }
    Ok(options)
}

fn build_single_factor_batch_base_command(
    config: &str,
    experiment_profile: &str,
    period: &str,
    quantile_bins: usize,
    top_n: usize,
    segment_scheme: &str,
) -> Vec<String> {
    vec![
        "--config".to_owned(),
        config.to_owned(),
        "--experiment-profile".to_owned(),
        experiment_profile.to_owned(),
        "--period".to_owned(),
        period.to_owned(),
        "--quantile-bins".to_owned(),
        quantile_bins.max(2).to_string(),
        "--top-n".to_owned(),
        top_n.max(1).to_string(),
        "--segment-scheme".to_owned(),
        segment_scheme.to_owned(),
    ]
}

fn build_full_space_preset_command(
    options: &FullSpacePresetOptions,
    pass_name: &str,
    industry_neutral: bool,
) -> Vec<String> {
    let mut command = build_single_factor_batch_base_command(
        &options.config,
        &options.experiment_profile,
        &options.period,
        options.quantile_bins,
        options.top_n,
        &options.segment_scheme,
    );
    command.push("--all-features".to_owned());
    append_string_option(
        &mut command,
        "--model-profile",
        options.model_profile.as_deref(),
    );
    append_string_option(
        &mut command,
        "--data-source",
        options.data_source.as_deref(),
    );
    append_string_option(&mut command, "--date-start", options.date_start.as_deref());
    append_string_option(&mut command, "--date-end", options.date_end.as_deref());
    append_string_option(&mut command, "--segments", options.segments.as_deref());
    for raw_override in &options.set_overrides {
        command.extend(["--set".to_owned(), raw_override.clone()]);
    }
    if industry_neutral {
        command.push("--industry-neutral".to_owned());
    }
    if let Some(base_output_dir) = &options.base_output_dir {
        command.extend([
            "--base-output-dir".to_owned(),
            path_to_string(&base_output_dir.join(pass_name)),
        ]);
    }
    if let Some(run_tag) = &options.run_tag {
        if !run_tag.trim().is_empty() {
            command.extend(["--run-tag".to_owned(), format!("{run_tag}-{pass_name}")]);
        }
    }
    let case_name = format!("full-space-{}-{pass_name}", options.diagnostic_label_space);
    command.extend([
        "--case".to_owned(),
        format!("name={case_name}"),
        "feature_profile=all_features".to_owned(),
        format!("diagnostic_label_space={}", options.diagnostic_label_space),
        format!("diagnostic_threshold={}", options.diagnostic_threshold),
    ]);
    if options.dry_run {
        command.push("--dry-run".to_owned());
    }
    command
}

fn build_quality_event_flow_preset_command(options: &QualityEventFlowPresetOptions) -> Vec<String> {
    let mut command = build_single_factor_batch_base_command(
        &options.config,
        &options.experiment_profile,
        &options.period,
        options.quantile_bins,
        options.top_n,
        &options.segment_scheme,
    );
    command.extend([
        "--feature-profile".to_owned(),
        options.feature_profile.clone(),
    ]);
    append_string_option(
        &mut command,
        "--model-profile",
        options.model_profile.as_deref(),
    );
    append_string_option(
        &mut command,
        "--data-source",
        options.data_source.as_deref(),
    );
    append_string_option(&mut command, "--date-start", options.date_start.as_deref());
    append_string_option(&mut command, "--date-end", options.date_end.as_deref());
    append_string_option(&mut command, "--segments", options.segments.as_deref());
    if let Some(base_output_dir) = &options.base_output_dir {
        command.extend([
            "--base-output-dir".to_owned(),
            path_to_string(base_output_dir),
        ]);
    }
    append_string_option(&mut command, "--run-tag", options.run_tag.as_deref());
    for raw_override in &options.set_overrides {
        command.extend(["--set".to_owned(), raw_override.clone()]);
    }
    let mut cases = vec![("raw_return", 0.0f64), ("industry_excess", 0.0f64)];
    if options.include_benchmark_excess {
        cases.push(("benchmark_excess", 0.0));
    }
    for (label_space, threshold) in cases {
        let case_name = format!("{}__{label_space}", options.feature_profile);
        command.extend([
            "--case".to_owned(),
            format!("name={case_name}"),
            format!("feature_profile={}", options.feature_profile),
            format!("diagnostic_label_space={label_space}"),
            format!("diagnostic_threshold={threshold}"),
        ]);
    }
    if options.dry_run {
        command.push("--dry-run".to_owned());
    }
    command
}

fn append_string_option(command: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        if !value.trim().is_empty() {
            command.extend([flag.to_owned(), value.to_owned()]);
        }
    }
}

fn prepend_batch_subcommand(args: &[String]) -> Vec<String> {
    let mut command = vec![
        "ai4stock-diagnostics".to_owned(),
        "single-factor-batch".to_owned(),
    ];
    command.extend(args.iter().cloned());
    command
}

fn next_value(args: &[String], index: usize, option: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn parse_usize(value: String, option: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid {option} {value}: {err}"))
}

fn parse_f64(value: String, option: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|err| format!("invalid {option} {value}: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_factor_options() {
        let args = vec![
            "--factor-store".to_owned(),
            "data/factor_store/x".to_owned(),
            "--output-dir".to_owned(),
            "results/x".to_owned(),
            "--label-column".to_owned(),
            "label_10d".to_owned(),
            "--signal-horizon".to_owned(),
            "10".to_owned(),
            "--feature".to_owned(),
            "signal".to_owned(),
            "--benchmark-mode".to_owned(),
            "file".to_owned(),
            "--benchmark-path".to_owned(),
            "data/benchmarks/tushare/csi300.parquet".to_owned(),
            "--segment".to_owned(),
            "train:2024-01-01:2024-12-31".to_owned(),
            "--industry-neutral".to_owned(),
            "--json".to_owned(),
        ];
        let parsed = parse_single_factor_options(&args).unwrap();
        assert_eq!(parsed.feature_names, vec!["signal"]);
        assert_eq!(parsed.signal_horizon, 10);
        assert_eq!(parsed.benchmark.mode, BenchmarkMode::File);
        assert_eq!(parsed.segments.len(), 1);
        assert!(parsed.industry_neutral);
        assert!(parsed.json);
    }

    #[test]
    fn builds_full_space_preset_commands() {
        let options = parse_full_space_preset_options(&[
            "--experiment-profile".to_owned(),
            "exp".to_owned(),
            "--base-output-dir".to_owned(),
            "out".to_owned(),
            "--run-tag".to_owned(),
            "tag".to_owned(),
            "--dry-run".to_owned(),
        ])
        .unwrap();
        let command = build_full_space_preset_command(&options, "raw", false);
        assert_eq!(command[0], "--config");
        assert!(!command.iter().any(|item| item.ends_with(".py")));
        assert!(command.contains(&"--all-features".to_owned()));
        assert!(command.contains(&"out/raw".to_owned()));
        assert!(command.contains(&"tag-raw".to_owned()));
        assert!(command.contains(&"feature_profile=all_features".to_owned()));
        assert!(command.contains(&"--dry-run".to_owned()));
    }

    #[test]
    fn builds_quality_event_preset_command_with_benchmark_case() {
        let options = parse_quality_event_flow_preset_options(&[
            "--experiment-profile".to_owned(),
            "exp".to_owned(),
            "--include-benchmark-excess".to_owned(),
            "--dry-run".to_owned(),
        ])
        .unwrap();
        let command = build_quality_event_flow_preset_command(&options);
        assert_eq!(command[0], "--config");
        assert!(!command.iter().any(|item| item.ends_with(".py")));
        assert!(command.contains(&"--feature-profile".to_owned()));
        assert!(command.contains(&"diagnostic_label_space=benchmark_excess".to_owned()));
        assert!(command.contains(&"--dry-run".to_owned()));
    }

    #[test]
    fn parses_prefilter_profile_build_options() {
        let args = vec![
            "--diagnostics-summary".to_owned(),
            "summary.csv".to_owned(),
            "--segment-comparison".to_owned(),
            "segments.csv".to_owned(),
            "--factor-store".to_owned(),
            "data/factor_store/x".to_owned(),
            "--output-dir".to_owned(),
            "results/profile".to_owned(),
            "--profile-name".to_owned(),
            "profile_x".to_owned(),
            "--date-start".to_owned(),
            "2024-01-01".to_owned(),
            "--date-end".to_owned(),
            "2024-01-31".to_owned(),
            "--universe-name".to_owned(),
            "csi300".to_owned(),
            "--max-abs-corr".to_owned(),
            "0.95".to_owned(),
            "--raw-values".to_owned(),
            "--max-features".to_owned(),
            "12".to_owned(),
            "--setting".to_owned(),
            "period=train".to_owned(),
            "--json".to_owned(),
        ];
        let parsed = parse_prefilter_profile_build_options(&args).unwrap();
        assert_eq!(
            parsed.options.diagnostics_summary.to_string_lossy(),
            "summary.csv"
        );
        assert_eq!(parsed.options.profile_name, "profile_x");
        assert_eq!(parsed.options.universe_name, "csi300");
        assert_eq!(parsed.options.corr_threshold, 0.95);
        assert!(!parsed.options.use_cross_sectional_rank);
        assert_eq!(parsed.options.max_features, Some(12));
        assert_eq!(
            parsed.options.settings,
            vec![("period".to_owned(), "train".to_owned())]
        );
        assert!(parsed.json);
    }

    #[test]
    fn parses_robust_profile_build_options() {
        let args = vec![
            "--raw-summary=raw.csv".to_owned(),
            "--neutral-summary=neutral.csv".to_owned(),
            "--factor-store=data/factor_store/x".to_owned(),
            "--output-dir=results/robust".to_owned(),
            "--profile-name=robust_x".to_owned(),
            "--raw-segment-comparison=raw_segments.csv".to_owned(),
            "--neutral-segment-comparison=neutral_segments.csv".to_owned(),
            "--write-config-profile".to_owned(),
            "--config-profile-path=configs/features/robust_x.yaml".to_owned(),
            "--factor-store-name=custom_store".to_owned(),
            "--safety-warning=unsafe".to_owned(),
        ];
        let parsed = parse_robust_profile_build_options(&args).unwrap();
        assert_eq!(parsed.options.raw_summary.to_string_lossy(), "raw.csv");
        assert_eq!(
            parsed.options.neutral_summary.to_string_lossy(),
            "neutral.csv"
        );
        assert_eq!(parsed.options.profile_name, "robust_x");
        assert_eq!(parsed.options.factor_store_name, "custom_store");
        assert!(parsed.options.write_config_profile);
        assert_eq!(parsed.options.safety_warning.as_deref(), Some("unsafe"));
        assert_eq!(
            parsed
                .options
                .raw_segment_comparison
                .unwrap()
                .to_string_lossy(),
            "raw_segments.csv"
        );
    }
}
