use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

#[path = "ai4stock_backtest/artifact_batch.rs"]
mod artifact_batch;
#[path = "ai4stock_backtest/bundle_entry.rs"]
mod bundle_entry;
#[allow(dead_code)]
#[path = "../engine.rs"]
mod engine;
#[path = "ai4stock_backtest/prediction_bundle.rs"]
mod prediction_bundle;
#[path = "ai4stock_backtest/run_bundle.rs"]
mod run_bundle;

#[derive(Debug)]
struct RunnerOptions {
    repo_root: PathBuf,
    dry_run: bool,
    baseline_jobs: usize,
    passthrough: Vec<String>,
}

fn usage() -> &'static str {
    "\
ai4stock-backtest: Rust entrypoint for post-bundle AI4Stock2 backtests

Usage:
  ai4stock-backtest run-bundle --bundle <PATH> --config <config_snapshot.yaml> [--output-dir <PATH>]
  ai4stock-backtest bundle [OPTIONS] -- <rolling-lgbm-compatible args>
  ai4stock-backtest artifact-batch --selected-tsv <PATH> [OPTIONS]
  ai4stock-backtest inspect-bundle --bundle <PATH> [--json]

Options:
  --repo-root <PATH>          AI4Stock2 repo root, default current directory
  --baseline-jobs <N>         Parallel baseline workers for the direct Rust bundle path
  --dry-run                   Print the delegated command without running it
  -h, --help                  Show this help
"
}

fn default_repo_root() -> PathBuf {
    env::var("AI4STOCK_REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn parse_options(args: &[String]) -> Result<RunnerOptions, String> {
    let mut options = RunnerOptions {
        repo_root: default_repo_root(),
        dry_run: false,
        baseline_jobs: run_bundle::default_baseline_jobs(),
        passthrough: Vec::new(),
    };

    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--" => {
                options.passthrough.extend_from_slice(&args[idx + 1..]);
                break;
            }
            "--repo-root" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --repo-root".to_owned())?;
                options.repo_root = PathBuf::from(raw);
            }
            "--dry-run" => options.dry_run = true,
            "--baseline-jobs" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --baseline-jobs".to_owned())?;
                options.baseline_jobs = raw.trim().parse::<usize>().unwrap_or(1).max(1);
            }
            "-h" | "--help" => return Err(usage().to_owned()),
            value => options.passthrough.push(value.to_owned()),
        }
        idx += 1;
    }

    Ok(options)
}

fn main_result() -> Result<ExitCode, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{}", usage());
        return Ok(ExitCode::SUCCESS);
    }

    let subcommand = &args[0];
    match subcommand.as_str() {
        "run-bundle" => run_bundle::run(&args[1..]),
        "bundle" => {
            let options = parse_options(&args[1..])?;
            if options.dry_run {
                println!(
                    "[dry-run] ai4stock-backtest bundle --repo-root {} --baseline-jobs {} -- {}",
                    options.repo_root.display(),
                    options.baseline_jobs,
                    options.passthrough.join(" ")
                );
                return Ok(ExitCode::SUCCESS);
            }
            bundle_entry::try_run(
                &options.passthrough,
                &options.repo_root,
                options.baseline_jobs,
            )
        }
        "artifact-batch" | "artifacts" => artifact_batch::run(&args[1..]),
        "inspect-bundle" => prediction_bundle::inspect(&args[1..]),
        other => Err(format!("unknown subcommand: {other}\n\n{}", usage())),
    }
}

fn main() -> ExitCode {
    match main_result() {
        Ok(code) => code,
        Err(message) => {
            if message == usage() || message.starts_with("Usage:\n  ai4stock-backtest ") {
                println!("{message}");
                ExitCode::SUCCESS
            } else {
                eprintln!("{message}");
                ExitCode::from(2)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_options;

    #[test]
    fn parses_rust_options_and_preserves_python_args() {
        let args = vec![
            "--repo-root".to_owned(),
            "/repo".to_owned(),
            "--baseline-jobs".to_owned(),
            "3".to_owned(),
            "--".to_owned(),
            "--config".to_owned(),
            "cfg.yaml".to_owned(),
            "--load-predictions-dir".to_owned(),
            "bundle".to_owned(),
        ];
        let options = parse_options(&args).unwrap();
        assert_eq!(options.repo_root.to_string_lossy(), "/repo");
        assert_eq!(options.baseline_jobs, 3);
        assert_eq!(options.passthrough[0], "--config");
        assert!(options
            .passthrough
            .contains(&"--load-predictions-dir".to_owned()));
    }
}
