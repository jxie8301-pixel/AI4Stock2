use ai4stock2_native::single_factor_diagnostics::{
    run_single_factor_diagnostics, BenchmarkMode, BenchmarkOptions, BenchmarkValueType,
    DiagnosticLabelSpace, SegmentSpec, SingleFactorOptions,
};
use serde_json::Value as JsonValue;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn usage() -> &'static str {
    "\
ai4stock-diagnostics: Rust diagnostics entrypoint for AI4Stock2

Usage:
  ai4stock-diagnostics single-factor --factor-store <PATH> --output-dir <PATH> --label-column <COL> (--feature <NAME> | --features-json <PATH>)... [options]

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
        other => Err(format!("unknown command: {other}\n\n{}", usage())),
    }
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

fn next_value(args: &[String], index: usize, option: &str) -> Result<String, String> {
    args.get(index)
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
}
