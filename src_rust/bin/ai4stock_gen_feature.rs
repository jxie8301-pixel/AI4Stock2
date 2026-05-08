use ai4stock2_native::gen_feature::{
    generate_factor_store, inspect_parquet_layout, scan_source_bucket, validate_required_columns,
    GenerateOptions,
};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() -> &'static str {
    "\
ai4stock-gen-feature: Rust migration entrypoint for AI4Stock2 feature generation

Usage:
  ai4stock-gen-feature generate --parquet-dir <PATH> --output-dir <PATH> [--data-source <NAME>] [--workers <N>] [--label-horizons <CSV>] [--batch-size <N>] [--bucket-limit <N>] [--json]
  ai4stock-gen-feature inspect-source --parquet-dir <PATH> [--json]
  ai4stock-gen-feature inspect-store --store-dir <PATH> [--json]
  ai4stock-gen-feature scan-source-bucket --bucket-path <PATH> [--column <COL>...] [--batch-size <N>] [--json]
  ai4stock-gen-feature validate-required-columns --parquet-dir <PATH> (--required-column <COL> | --required-columns-json <PATH>)... [--json]

Options:
  --parquet-dir <PATH>        Source Parquet root. Supports <root>/buckets/part-*.parquet.
  --output-dir <PATH>         Output factor-store root for `generate`.
  --data-source <NAME>        Source name for `generate`. Use `tushare` to include TS_ factors.
  --workers <N>               Worker threads for `generate`. Default: available parallelism.
  --label-horizons <CSV>      Label horizons for `generate`. Default: 1,5,10,20.
  --bucket-limit <N>          Limit generated source buckets for smoke tests.
  --store-dir <PATH>          Factor-store root. Supports <root>/buckets/part-*.parquet.
  --bucket-path <PATH>        Single source bucket Parquet shard to scan.
  --column <COL>              Project a source column while scanning. Can be repeated.
  --batch-size <N>            Arrow record-batch size for payload scans. Default: 65536.
  --required-column <COL>     Required source column. Can be repeated.
  --required-columns-json <PATH>
                              JSON list, or JSON object with `required_columns`.
  --json                      Print machine-readable JSON.
  -h, --help                  Show this help.
"
}

#[derive(Debug)]
struct InspectOptions {
    root: PathBuf,
    json: bool,
}

#[derive(Debug)]
struct RequiredColumnsOptions {
    root: PathBuf,
    required_columns: Vec<String>,
    json: bool,
}

#[derive(Debug)]
struct ScanBucketOptions {
    path: PathBuf,
    columns: Vec<String>,
    batch_size: usize,
    json: bool,
}

#[derive(Debug)]
struct GenerateCliOptions {
    options: GenerateOptions,
    json: bool,
}

fn parse_inspect_options(args: &[String], root_option: &str) -> Result<InspectOptions, String> {
    let mut root: Option<PathBuf> = None;
    let mut json = false;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            value if value == root_option => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| format!("missing value for {root_option}"))?;
                root = Some(PathBuf::from(raw));
            }
            value if value.starts_with(&format!("{root_option}=")) => {
                let raw = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                if raw.is_empty() {
                    return Err(format!("missing value for {root_option}"));
                }
                root = Some(PathBuf::from(raw));
            }
            other => return Err(format!("unknown option for inspect command: {other}")),
        }
        idx += 1;
    }
    let root = root.ok_or_else(|| format!("{root_option} is required"))?;
    Ok(InspectOptions { root, json })
}

fn parse_required_columns_json(path: &PathBuf) -> Result<Vec<String>, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("failed to parse JSON {}: {err}", path.display()))?;
    let values = match value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(mut object) => match object.remove("required_columns") {
            Some(serde_json::Value::Array(items)) => items,
            _ => {
                return Err(format!(
                    "{} must be a JSON list or object with required_columns list",
                    path.display()
                ))
            }
        },
        _ => {
            return Err(format!(
                "{} must be a JSON list or object with required_columns list",
                path.display()
            ))
        }
    };
    let mut columns = Vec::with_capacity(values.len());
    for value in values {
        let serde_json::Value::String(column) = value else {
            return Err(format!(
                "{} contains a non-string column name",
                path.display()
            ));
        };
        if !column.trim().is_empty() {
            columns.push(column);
        }
    }
    Ok(columns)
}

fn parse_required_columns_options(args: &[String]) -> Result<RequiredColumnsOptions, String> {
    let mut root: Option<PathBuf> = None;
    let mut required_columns = Vec::new();
    let mut json = false;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--parquet-dir" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --parquet-dir".to_owned())?;
                root = Some(PathBuf::from(raw));
            }
            value if value.starts_with("--parquet-dir=") => {
                let raw = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                if raw.is_empty() {
                    return Err("missing value for --parquet-dir".to_owned());
                }
                root = Some(PathBuf::from(raw));
            }
            "--required-column" => {
                idx += 1;
                let column = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --required-column".to_owned())?;
                required_columns.push(column.to_owned());
            }
            value if value.starts_with("--required-column=") => {
                let column = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                if column.is_empty() {
                    return Err("missing value for --required-column".to_owned());
                }
                required_columns.push(column.to_owned());
            }
            "--required-columns-json" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --required-columns-json".to_owned())?;
                required_columns.extend(parse_required_columns_json(&PathBuf::from(raw))?);
            }
            value if value.starts_with("--required-columns-json=") => {
                let raw = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                if raw.is_empty() {
                    return Err("missing value for --required-columns-json".to_owned());
                }
                required_columns.extend(parse_required_columns_json(&PathBuf::from(raw))?);
            }
            other => {
                return Err(format!(
                    "unknown option for validate-required-columns command: {other}"
                ))
            }
        }
        idx += 1;
    }
    let root = root.ok_or_else(|| "--parquet-dir is required".to_owned())?;
    required_columns.sort();
    required_columns.dedup();
    if required_columns.is_empty() {
        return Err("at least one required column must be specified".to_owned());
    }
    Ok(RequiredColumnsOptions {
        root,
        required_columns,
        json,
    })
}

fn parse_scan_bucket_options(args: &[String]) -> Result<ScanBucketOptions, String> {
    let mut path: Option<PathBuf> = None;
    let mut columns = Vec::new();
    let mut batch_size = 65_536usize;
    let mut json = false;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--bucket-path" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --bucket-path".to_owned())?;
                path = Some(PathBuf::from(raw));
            }
            value if value.starts_with("--bucket-path=") => {
                let raw = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                if raw.is_empty() {
                    return Err("missing value for --bucket-path".to_owned());
                }
                path = Some(PathBuf::from(raw));
            }
            "--column" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --column".to_owned())?;
                columns.push(raw.to_owned());
            }
            value if value.starts_with("--column=") => {
                let raw = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                if raw.is_empty() {
                    return Err("missing value for --column".to_owned());
                }
                columns.push(raw.to_owned());
            }
            "--batch-size" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --batch-size".to_owned())?;
                batch_size = raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --batch-size {raw}: {err}"))?;
            }
            value if value.starts_with("--batch-size=") => {
                let raw = value
                    .split_once('=')
                    .map(|(_, right)| right)
                    .unwrap_or_default();
                batch_size = raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --batch-size {raw}: {err}"))?;
            }
            other => {
                return Err(format!(
                    "unknown option for scan-source-bucket command: {other}"
                ))
            }
        }
        idx += 1;
    }
    let path = path.ok_or_else(|| "--bucket-path is required".to_owned())?;
    columns.sort();
    columns.dedup();
    Ok(ScanBucketOptions {
        path,
        columns,
        batch_size,
        json,
    })
}

fn parse_label_horizons(raw: &str) -> Result<Vec<usize>, String> {
    let mut horizons = Vec::new();
    for value in raw.split(',') {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        let horizon = trimmed
            .parse::<usize>()
            .map_err(|err| format!("invalid label horizon {trimmed}: {err}"))?;
        if horizon == 0 {
            return Err("label horizons must be positive".to_owned());
        }
        horizons.push(horizon);
    }
    horizons.sort_unstable();
    horizons.dedup();
    if !horizons.contains(&1) {
        horizons.insert(0, 1);
    }
    if horizons.is_empty() {
        return Err("at least one label horizon is required".to_owned());
    }
    Ok(horizons)
}

fn parse_generate_options(args: &[String]) -> Result<GenerateCliOptions, String> {
    let mut parquet_dir: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut data_source = "akshare".to_owned();
    let mut workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut label_horizons = vec![1, 5, 10, 20];
    let mut batch_size = 65_536usize;
    let mut bucket_limit = None;
    let mut json = false;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--json" => json = true,
            "--parquet-dir" => {
                idx += 1;
                parquet_dir =
                    Some(PathBuf::from(args.get(idx).ok_or_else(|| {
                        "missing value for --parquet-dir".to_owned()
                    })?));
            }
            value if value.starts_with("--parquet-dir=") => {
                parquet_dir = Some(PathBuf::from(value.split_once('=').unwrap().1));
            }
            "--output-dir" => {
                idx += 1;
                output_dir =
                    Some(PathBuf::from(args.get(idx).ok_or_else(|| {
                        "missing value for --output-dir".to_owned()
                    })?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(value.split_once('=').unwrap().1));
            }
            "--data-source" => {
                idx += 1;
                data_source = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --data-source".to_owned())?
                    .trim()
                    .to_ascii_lowercase();
            }
            value if value.starts_with("--data-source=") => {
                data_source = value.split_once('=').unwrap().1.trim().to_ascii_lowercase();
            }
            "--workers" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --workers".to_owned())?;
                workers = raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --workers {raw}: {err}"))?;
            }
            value if value.starts_with("--workers=") => {
                let raw = value.split_once('=').unwrap().1;
                workers = raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --workers {raw}: {err}"))?;
            }
            "--label-horizons" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --label-horizons".to_owned())?;
                label_horizons = parse_label_horizons(raw)?;
            }
            value if value.starts_with("--label-horizons=") => {
                label_horizons = parse_label_horizons(value.split_once('=').unwrap().1)?;
            }
            "--batch-size" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --batch-size".to_owned())?;
                batch_size = raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --batch-size {raw}: {err}"))?;
            }
            value if value.starts_with("--batch-size=") => {
                let raw = value.split_once('=').unwrap().1;
                batch_size = raw
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --batch-size {raw}: {err}"))?;
            }
            "--bucket-limit" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --bucket-limit".to_owned())?;
                bucket_limit = Some(
                    raw.parse::<usize>()
                        .map_err(|err| format!("invalid --bucket-limit {raw}: {err}"))?,
                );
            }
            value if value.starts_with("--bucket-limit=") => {
                let raw = value.split_once('=').unwrap().1;
                bucket_limit = Some(
                    raw.parse::<usize>()
                        .map_err(|err| format!("invalid --bucket-limit {raw}: {err}"))?,
                );
            }
            other => return Err(format!("unknown option for generate command: {other}")),
        }
        idx += 1;
    }
    Ok(GenerateCliOptions {
        options: GenerateOptions {
            parquet_dir: parquet_dir.ok_or_else(|| "--parquet-dir is required".to_owned())?,
            output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_owned())?,
            data_source,
            workers: workers.max(1),
            label_horizons,
            batch_size: batch_size.max(1),
            bucket_limit,
        },
        json,
    })
}

fn run_inspect(args: &[String], root_option: &str) -> Result<(), String> {
    let options = parse_inspect_options(args, root_option)?;
    let summary = inspect_parquet_layout(&options.root)?;
    if options.json {
        let payload = serde_json::to_string_pretty(&summary)
            .map_err(|err| format!("failed to encode summary JSON: {err}"))?;
        println!("{payload}");
        return Ok(());
    }

    println!("root={}", summary.root);
    println!("bucket_root={}", summary.bucket_root);
    println!("file_count={}", summary.file_count);
    println!("total_rows={}", summary.total_rows);
    println!("total_row_groups={}", summary.total_row_groups);
    println!(
        "total_size_mb={:.3}",
        summary.total_size_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "rows_min_median_max={}/{:.1}/{}",
        summary.min_rows, summary.median_rows, summary.max_rows
    );
    println!(
        "columns_min_median_max={}/{:.1}/{}",
        summary.min_columns, summary.median_columns, summary.max_columns
    );
    println!(
        "row_groups_min_median_max={}/{:.1}/{}",
        summary.min_row_groups, summary.median_row_groups, summary.max_row_groups
    );
    Ok(())
}

fn run_validate_required_columns(args: &[String]) -> Result<(), String> {
    let options = parse_required_columns_options(args)?;
    let summary = validate_required_columns(&options.root, &options.required_columns)?;
    if options.json {
        let payload = serde_json::to_string_pretty(&summary)
            .map_err(|err| format!("failed to encode validation JSON: {err}"))?;
        println!("{payload}");
        return Ok(());
    }

    println!("root={}", summary.root);
    println!("bucket_root={}", summary.bucket_root);
    println!("validated={}", summary.validated);
    println!("file_count={}", summary.file_count);
    println!("required_columns_count={}", summary.required_columns_count);
    println!("missing_file_count={}", summary.missing_file_count);
    for item in summary.missing_by_path.iter().take(5) {
        println!("missing {}: {}", item.path, item.missing_columns.join(","));
    }
    if !summary.validated {
        return Err("required-column validation failed".to_owned());
    }
    Ok(())
}

fn run_scan_source_bucket(args: &[String]) -> Result<(), String> {
    let options = parse_scan_bucket_options(args)?;
    let summary = scan_source_bucket(&options.path, &options.columns, options.batch_size)?;
    if options.json {
        let payload = serde_json::to_string_pretty(&summary)
            .map_err(|err| format!("failed to encode scan JSON: {err}"))?;
        println!("{payload}");
        return Ok(());
    }
    println!("path={}", summary.path);
    println!("selected_columns={}", summary.selected_columns.join(","));
    println!("batch_size={}", summary.batch_size);
    println!("batch_count={}", summary.batch_count);
    println!("row_count={}", summary.row_count);
    println!("symbol_count={}", summary.symbol_count);
    Ok(())
}

fn run_generate(args: &[String]) -> Result<(), String> {
    let cli_options = parse_generate_options(args)?;
    let summary = generate_factor_store(&cli_options.options)?;
    if cli_options.json {
        let payload = serde_json::to_string_pretty(&summary)
            .map_err(|err| format!("failed to encode generate JSON: {err}"))?;
        println!("{payload}");
        return Ok(());
    }
    println!("generator={}", summary.generator);
    println!("factor_store_dir={}", summary.factor_store_dir);
    println!("bucket_count={}", summary.bucket_count);
    println!("num_rows={}", summary.num_rows);
    println!("num_features={}", summary.num_features);
    println!("elapsed_seconds={:.3}", summary.elapsed_seconds);
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    let Some(command) = args.first() else {
        return Err(usage().to_owned());
    };
    match command.as_str() {
        "-h" | "--help" => Err(usage().to_owned()),
        "generate" => run_generate(&args[1..]),
        "inspect-source" => run_inspect(&args[1..], "--parquet-dir"),
        "inspect-store" => run_inspect(&args[1..], "--store-dir"),
        "scan-source-bucket" => run_scan_source_bucket(&args[1..]),
        "validate-required-columns" => run_validate_required_columns(&args[1..]),
        other => Err(format!("unknown command: {other}\n\n{}", usage())),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
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

#[cfg(test)]
mod tests {
    use super::{
        parse_generate_options, parse_inspect_options, parse_required_columns_options,
        parse_scan_bucket_options,
    };

    #[test]
    fn parses_generate_options() {
        let args = vec![
            "--parquet-dir=data/source".to_owned(),
            "--output-dir".to_owned(),
            "data/factor_store/rust".to_owned(),
            "--data-source".to_owned(),
            "tushare".to_owned(),
            "--workers=16".to_owned(),
            "--label-horizons=1,5,20".to_owned(),
            "--bucket-limit".to_owned(),
            "2".to_owned(),
            "--json".to_owned(),
        ];

        let options = parse_generate_options(&args).unwrap();

        assert_eq!(options.options.parquet_dir.to_string_lossy(), "data/source");
        assert_eq!(
            options.options.output_dir.to_string_lossy(),
            "data/factor_store/rust"
        );
        assert_eq!(options.options.data_source, "tushare");
        assert_eq!(options.options.workers, 16);
        assert_eq!(options.options.label_horizons, vec![1, 5, 20]);
        assert_eq!(options.options.bucket_limit, Some(2));
        assert!(options.json);
    }

    #[test]
    fn parses_inspect_options() {
        let args = vec![
            "--parquet-dir".to_owned(),
            "data/source".to_owned(),
            "--json".to_owned(),
        ];

        let options = parse_inspect_options(&args, "--parquet-dir").unwrap();

        assert_eq!(options.root.to_string_lossy(), "data/source");
        assert!(options.json);
    }

    #[test]
    fn parses_required_columns_options() {
        let args = vec![
            "--parquet-dir".to_owned(),
            "data/source".to_owned(),
            "--required-column".to_owned(),
            "close".to_owned(),
            "--required-column=symbol".to_owned(),
            "--json".to_owned(),
        ];

        let options = parse_required_columns_options(&args).unwrap();

        assert_eq!(options.root.to_string_lossy(), "data/source");
        assert_eq!(options.required_columns, vec!["close", "symbol"]);
        assert!(options.json);
    }

    #[test]
    fn parses_scan_bucket_options() {
        let args = vec![
            "--bucket-path".to_owned(),
            "data/source/buckets/part-0001.parquet".to_owned(),
            "--column=symbol".to_owned(),
            "--column".to_owned(),
            "close".to_owned(),
            "--batch-size".to_owned(),
            "4096".to_owned(),
            "--json".to_owned(),
        ];

        let options = parse_scan_bucket_options(&args).unwrap();

        assert_eq!(
            options.path.to_string_lossy(),
            "data/source/buckets/part-0001.parquet"
        );
        assert_eq!(options.columns, vec!["close", "symbol"]);
        assert_eq!(options.batch_size, 4096);
        assert!(options.json);
    }
}
