use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

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
    python_runner: Vec<String>,
    repo_root: PathBuf,
    dry_run: bool,
    disable_rust_backtest: bool,
    allow_non_bundle: bool,
    baseline_jobs: usize,
    passthrough: Vec<String>,
}

fn usage() -> &'static str {
    "\
ai4stock-backtest: Rust entrypoint for post-bundle AI4Stock2 backtests

Usage:
  ai4stock-backtest run-bundle --bundle <PATH> --config <config_snapshot.yaml> [--output-dir <PATH>]
  ai4stock-backtest bundle [OPTIONS] -- <run_native_rolling.py args>
  ai4stock-backtest artifact-batch --selected-tsv <PATH> [OPTIONS]
  ai4stock-backtest inspect-bundle --bundle <PATH> [--json]

Options:
  --python-runner <CMD>       Python runner, e.g. 'python' or 'pixi run python'
  --python <CMD>              Alias for --python-runner
  --repo-root <PATH>          AI4Stock2 repo root, default current directory
  --disable-rust-backtest     Force the Python post-bundle backtest path
  --baseline-jobs <N>         Parallel baseline workers for the direct Rust bundle path
  --allow-non-bundle          Allow bundle command without --load-predictions-dir
  --dry-run                   Print the delegated command without running it
  -h, --help                  Show this help
"
}

fn split_runner(raw: &str) -> Vec<String> {
    raw.split_whitespace()
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn default_python_runner() -> Vec<String> {
    if let Ok(value) = env::var("AI4STOCK_PYTHON_RUNNER") {
        let runner = split_runner(&value);
        if !runner.is_empty() {
            return runner;
        }
    }
    if let Ok(value) = env::var("PYTHON_RUNNER") {
        let runner = split_runner(&value);
        if !runner.is_empty() {
            return runner;
        }
    }
    vec!["python".to_owned()]
}

fn default_repo_root() -> PathBuf {
    env::var("AI4STOCK_REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn parse_options(args: &[String]) -> Result<RunnerOptions, String> {
    let mut options = RunnerOptions {
        python_runner: default_python_runner(),
        repo_root: default_repo_root(),
        dry_run: false,
        disable_rust_backtest: false,
        allow_non_bundle: false,
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
            "--python-runner" | "--python" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --python-runner".to_owned())?;
                options.python_runner = split_runner(raw);
                if options.python_runner.is_empty() {
                    return Err("--python-runner cannot be empty".to_owned());
                }
            }
            "--repo-root" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --repo-root".to_owned())?;
                options.repo_root = PathBuf::from(raw);
            }
            "--dry-run" => options.dry_run = true,
            "--disable-rust-backtest" => options.disable_rust_backtest = true,
            "--baseline-jobs" => {
                idx += 1;
                let raw = args
                    .get(idx)
                    .ok_or_else(|| "missing value for --baseline-jobs".to_owned())?;
                options.baseline_jobs = raw.trim().parse::<usize>().unwrap_or(1).max(1);
            }
            "--allow-non-bundle" => options.allow_non_bundle = true,
            "-h" | "--help" => return Err(usage().to_owned()),
            value => options.passthrough.push(value.to_owned()),
        }
        idx += 1;
    }

    Ok(options)
}

fn has_option(args: &[String], option: &str) -> bool {
    let prefix = format!("{option}=");
    args.iter()
        .any(|arg| arg == option || arg.starts_with(prefix.as_str()))
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

fn display_command(program: &str, args: &[String]) -> String {
    std::iter::once(program)
        .chain(args.iter().map(String::as_str))
        .map(shell_quote)
        .collect::<Vec<_>>()
        .join(" ")
}

fn run_python_script(
    options: RunnerOptions,
    script_name: &str,
    require_prediction_bundle: bool,
) -> Result<ExitCode, String> {
    if require_prediction_bundle
        && !options.allow_non_bundle
        && !has_option(&options.passthrough, "--load-predictions-dir")
    {
        return Err(
            "bundle command requires --load-predictions-dir; pass --allow-non-bundle to override"
                .to_owned(),
        );
    }
    if options.python_runner.is_empty() {
        return Err("python runner cannot be empty".to_owned());
    }

    let script_path = options.repo_root.join(script_name);
    if !script_path.is_file() {
        return Err(format!(
            "Python entrypoint not found: {}",
            script_path.display()
        ));
    }

    let program = options.python_runner[0].clone();
    let mut command_args = options.python_runner[1..].to_vec();
    command_args.push(path_to_string(&script_path));
    command_args.extend(options.passthrough);

    if options.dry_run {
        println!(
            "[dry-run] cwd={} {}",
            options.repo_root.display(),
            display_command(&program, &command_args)
        );
        println!(
            "[dry-run] post_bundle_path={}",
            if options.disable_rust_backtest {
                "python"
            } else {
                "rust"
            }
        );
        println!("[dry-run] baseline_jobs={}", options.baseline_jobs);
        return Ok(ExitCode::SUCCESS);
    }

    let mut command = Command::new(&program);
    command.args(&command_args).current_dir(&options.repo_root);
    if env::var_os("MPLCONFIGDIR").is_none() {
        let mpl_config_dir = options.repo_root.join(".mpl-cache");
        fs::create_dir_all(&mpl_config_dir).map_err(|err| {
            format!(
                "failed to create MPLCONFIGDIR {}: {err}",
                mpl_config_dir.display()
            )
        })?;
        command.env("MPLCONFIGDIR", path_to_string(&mpl_config_dir));
    }
    let status = command
        .status()
        .map_err(|err| format!("failed to run {program}: {err}"))?;
    Ok(match status.code() {
        Some(code) => ExitCode::from(code as u8),
        None => ExitCode::FAILURE,
    })
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn main_result() -> Result<ExitCode, String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        println!("{}", usage());
        return Ok(ExitCode::SUCCESS);
    }

    let subcommand = &args[0];
    match subcommand.as_str() {
        "run-bundle" | "rust-bundle" => run_bundle::run(&args[1..]),
        "bundle" | "native-rolling" | "run-native-rolling" => {
            let options = parse_options(&args[1..])?;
            if options.disable_rust_backtest {
                run_python_script(options, "run_native_rolling.py", true)
            } else {
                match bundle_entry::try_run(
                    &options.passthrough,
                    &options.repo_root,
                    options.baseline_jobs,
                )? {
                    bundle_entry::BundleDispatch::Ran(code) => Ok(code),
                    bundle_entry::BundleDispatch::Fallback(reason) => {
                        eprintln!("[bundle] falling back to Python: {reason}");
                        run_python_script(options, "run_native_rolling.py", true)
                    }
                }
            }
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
    use super::{has_option, parse_options};

    #[test]
    fn parses_rust_options_and_preserves_python_args() {
        let args = vec![
            "--python-runner".to_owned(),
            "pixi run python".to_owned(),
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
        assert_eq!(options.python_runner, ["pixi", "run", "python"]);
        assert_eq!(options.repo_root.to_string_lossy(), "/repo");
        assert_eq!(options.baseline_jobs, 3);
        assert!(has_option(&options.passthrough, "--load-predictions-dir"));
        assert_eq!(options.passthrough[0], "--config");
    }

    #[test]
    fn detects_equals_style_options() {
        let args = vec!["--load-predictions-dir=bundle".to_owned()];
        assert!(has_option(&args, "--load-predictions-dir"));
    }
}
