use crate::bundle_entry::{self, BundlePlan};
use crate::prediction_bundle::{read_prediction_bundle, PredictionBundle};
use ai4stock2_native::common::cli::{display_command, next_arg, path_to_string, shell_quote};
use csv::{ReaderBuilder, WriterBuilder};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const NONE_MARKER: &str = "__NONE__";

#[derive(Debug, Clone)]
struct BatchOptions {
    selected_tsv: PathBuf,
    output_root: PathBuf,
    log_dir: PathBuf,
    marker_dir: Option<PathBuf>,
    failed_tsv: Option<PathBuf>,
    matrix_ids: BTreeSet<String>,
    train_ids: BTreeSet<String>,
    backtest_ids: BTreeSet<String>,
    limit: usize,
    start_after: String,
    jobs: usize,
    baseline_jobs: usize,
    repo_root: PathBuf,
    model: String,
    run_tag_prefix: String,
    backtest_artifact_level: String,
    save_predictions: bool,
    skip_reference_baselines: bool,
    skip_opportunity_diagnostics: bool,
    skip_backtest_plots: bool,
    skip_backtest_trace: bool,
    dry_run: bool,
    fail_fast: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ArtifactRow {
    #[serde(default)]
    matrix_id: String,
    #[serde(default)]
    train_id: String,
    #[serde(default)]
    backtest_profile_id: String,
    #[serde(default)]
    config_snapshot: String,
    #[serde(default)]
    primary_predictions_dir: String,
    #[serde(default)]
    secondary_predictions_dir: String,
    #[serde(default)]
    train_signal_horizon: String,
    #[serde(default)]
    train_retrain_step: String,
    #[serde(default)]
    train_train_days: String,
    #[serde(default)]
    train_valid_days: String,
    #[serde(default)]
    train_label_embargo_days: String,
    #[serde(default)]
    train_model_profile: String,
    #[serde(default)]
    train_feature_profile: String,
    #[serde(default)]
    score_fusion_enabled: String,
}

#[derive(Debug, Clone)]
struct ArtifactJob {
    row: ArtifactRow,
    argv: Vec<String>,
    rust_command: Vec<String>,
    log_path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
struct JobSummary {
    matrix_id: String,
    train_id: String,
    backtest_profile_id: String,
    status: String,
    exit_code: i32,
    elapsed_seconds: f64,
    log_path: String,
    failure_reason: String,
    failure_detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct BatchSummaryJson {
    selected_tsv: String,
    output_root: String,
    log_dir: String,
    marker_dir: Option<String>,
    failed_tsv: Option<String>,
    summary_tsv: String,
    summary_json: String,
    artifact_level: String,
    parallel_jobs: usize,
    selected_jobs: usize,
    processed_jobs: usize,
    success_jobs: usize,
    skipped_jobs: usize,
    failed_jobs: usize,
    unprocessed_jobs: usize,
    elapsed_seconds: f64,
    jobs: Vec<JobSummary>,
}

#[derive(Debug)]
struct JobRunOutcome {
    index: usize,
    summary: JobSummary,
    failure: Option<(String, String)>,
}

#[derive(Debug)]
struct JobGroup {
    indexes: Vec<usize>,
}

pub(crate) fn run(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_batch_options(args)?;
    let jobs = build_jobs(&options)?;
    run_jobs(options, jobs)
}

fn parse_batch_options(args: &[String]) -> Result<BatchOptions, String> {
    let mut selected_tsv: Option<PathBuf> = None;
    let mut output_root: Option<PathBuf> = None;
    let mut log_dir: Option<PathBuf> = None;
    let mut marker_dir: Option<PathBuf> = None;
    let mut failed_tsv: Option<PathBuf> = None;
    let mut matrix_ids = Vec::new();
    let mut train_ids = Vec::new();
    let mut backtest_ids = Vec::new();
    let mut limit = 0usize;
    let mut start_after = String::new();
    let mut jobs = 1usize;
    let mut baseline_jobs = 1usize;
    let mut repo_root = env::var("AI4STOCK_REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let mut model = "lgbm".to_owned();
    let mut run_tag_prefix = "artifact-rebuild-lgbm".to_owned();
    let mut backtest_artifact_level = "full".to_owned();
    let mut save_predictions = false;
    let mut skip_reference_baselines = false;
    let mut skip_opportunity_diagnostics = false;
    let mut skip_backtest_plots = false;
    let mut skip_backtest_trace = false;
    let mut dry_run = false;
    let mut fail_fast = false;

    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--selected-tsv" => {
                selected_tsv = Some(PathBuf::from(next_arg(args, &mut idx, "--selected-tsv")?))
            }
            "--output-root" => {
                output_root = Some(PathBuf::from(next_arg(args, &mut idx, "--output-root")?))
            }
            "--log-dir" => log_dir = Some(PathBuf::from(next_arg(args, &mut idx, "--log-dir")?)),
            "--marker-dir" => {
                marker_dir = Some(PathBuf::from(next_arg(args, &mut idx, "--marker-dir")?))
            }
            "--failed-tsv" => {
                failed_tsv = Some(PathBuf::from(next_arg(args, &mut idx, "--failed-tsv")?))
            }
            "--matrix-id" => matrix_ids.push(next_arg(args, &mut idx, "--matrix-id")?),
            "--train-id" => train_ids.push(next_arg(args, &mut idx, "--train-id")?),
            "--backtest-id" => backtest_ids.push(next_arg(args, &mut idx, "--backtest-id")?),
            "--limit" => limit = parse_usize(&next_arg(args, &mut idx, "--limit")?),
            "--start-after" => start_after = next_arg(args, &mut idx, "--start-after")?,
            "--jobs" | "-j" => jobs = parse_positive_usize(&next_arg(args, &mut idx, "--jobs")?),
            "--baseline-jobs" => {
                baseline_jobs = parse_positive_usize(&next_arg(args, &mut idx, "--baseline-jobs")?)
            }
            "--repo-root" => repo_root = PathBuf::from(next_arg(args, &mut idx, "--repo-root")?),
            "--model" => model = next_arg(args, &mut idx, "--model")?,
            "--run-tag-prefix" => run_tag_prefix = next_arg(args, &mut idx, "--run-tag-prefix")?,
            "--backtest-artifact-level" => {
                backtest_artifact_level = next_arg(args, &mut idx, "--backtest-artifact-level")?
            }
            "--save-predictions" => save_predictions = true,
            "--skip-reference-baselines" => skip_reference_baselines = true,
            "--skip-opportunity-diagnostics" => skip_opportunity_diagnostics = true,
            "--skip-backtest-plots" => skip_backtest_plots = true,
            "--skip-backtest-trace" => skip_backtest_trace = true,
            "--dry-run" => dry_run = true,
            "--fail-fast" => fail_fast = true,
            "-h" | "--help" => return Err(batch_usage().to_owned()),
            other => {
                return Err(format!(
                    "unknown artifact-batch option: {other}\n\n{}",
                    batch_usage()
                ))
            }
        }
        idx += 1;
    }

    let selected_tsv = selected_tsv.ok_or_else(|| "--selected-tsv is required".to_owned())?;
    let parent = selected_tsv
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let output_root = output_root.unwrap_or_else(|| parent.join("lgbm_backtest_artifact_runs"));
    let log_dir = log_dir.unwrap_or_else(|| parent.join("lgbm_backtest_artifact_logs"));
    Ok(BatchOptions {
        selected_tsv,
        output_root,
        log_dir,
        marker_dir,
        failed_tsv,
        matrix_ids: split_allowlist(&matrix_ids),
        train_ids: split_allowlist(&train_ids),
        backtest_ids: split_allowlist(&backtest_ids),
        limit,
        start_after,
        jobs,
        baseline_jobs,
        repo_root,
        model,
        run_tag_prefix,
        backtest_artifact_level,
        save_predictions,
        skip_reference_baselines,
        skip_opportunity_diagnostics,
        skip_backtest_plots,
        skip_backtest_trace,
        dry_run,
        fail_fast,
    })
}

fn batch_usage() -> &'static str {
    "\
Usage:
  ai4stock-backtest artifact-batch --selected-tsv <PATH> [OPTIONS]

Main options:
  --output-root <PATH>
  --log-dir <PATH>
  --marker-dir <PATH>
  --failed-tsv <PATH>
  --limit <N>
  --start-after <ID>
  --jobs <N>
  --baseline-jobs <N>
  --matrix-id <ID>[,<ID>...]
  --train-id <ID>[,<ID>...]
  --backtest-id <ID>[,<ID>...]
  --repo-root <PATH>
  --model <NAME>
  --run-tag-prefix <TEXT>
  --backtest-artifact-level <full|reports|metrics>
  --save-predictions
  --skip-reference-baselines
  --skip-opportunity-diagnostics
  --skip-backtest-plots
  --skip-backtest-trace
  --dry-run
  --fail-fast
"
}

fn parse_usize(raw: &str) -> usize {
    raw.trim().parse::<usize>().unwrap_or(0)
}

fn parse_positive_usize(raw: &str) -> usize {
    parse_usize(raw).max(1)
}

fn split_allowlist(raw_values: &[String]) -> BTreeSet<String> {
    raw_values
        .iter()
        .flat_map(|raw| raw.split(','))
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect()
}

fn build_jobs(options: &BatchOptions) -> Result<Vec<ArtifactJob>, String> {
    let rows = select_rows(
        read_selected_rows(&options.selected_tsv)?,
        &options.matrix_ids,
        &options.train_ids,
        &options.backtest_ids,
        &options.start_after,
        options.limit,
    );
    rows.into_iter()
        .map(|row| {
            let argv = build_rolling_argv(&row, options);
            let rust_command = build_bundle_command(&argv, options)?;
            Ok(ArtifactJob {
                log_path: options.log_dir.join(format!("{}.log", row.matrix_id)),
                row,
                argv,
                rust_command,
            })
        })
        .collect()
}

fn read_selected_rows(path: &Path) -> Result<Vec<ArtifactRow>, String> {
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    let mut rows = Vec::new();
    for result in reader.deserialize() {
        let row: ArtifactRow =
            result.map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
        rows.push(row);
    }
    Ok(rows)
}

fn select_rows(
    rows: Vec<ArtifactRow>,
    matrix_ids: &BTreeSet<String>,
    train_ids: &BTreeSet<String>,
    backtest_ids: &BTreeSet<String>,
    start_after: &str,
    limit: usize,
) -> Vec<ArtifactRow> {
    let mut selected = Vec::new();
    let mut seen_start = start_after.is_empty();
    for row in rows {
        if !seen_start {
            if row.matrix_id == start_after || row.backtest_profile_id == start_after {
                seen_start = true;
            }
            continue;
        }
        if !row_matches(&row, matrix_ids, train_ids, backtest_ids) {
            continue;
        }
        selected.push(row);
        if limit > 0 && selected.len() >= limit {
            break;
        }
    }
    selected
}

fn row_matches(
    row: &ArtifactRow,
    matrix_ids: &BTreeSet<String>,
    train_ids: &BTreeSet<String>,
    backtest_ids: &BTreeSet<String>,
) -> bool {
    (matrix_ids.is_empty() || matrix_ids.contains(&row.matrix_id))
        && (train_ids.is_empty() || train_ids.contains(&row.train_id))
        && (backtest_ids.is_empty() || backtest_ids.contains(&row.backtest_profile_id))
}

fn build_rolling_argv(row: &ArtifactRow, options: &BatchOptions) -> Vec<String> {
    let mut argv = vec![
        "--config".to_owned(),
        row.config_snapshot.clone(),
        "--config-is-snapshot".to_owned(),
        "--run-tag".to_owned(),
        format!("{}-{}", options.run_tag_prefix, safe_tag(&row.matrix_id)),
        "--store-dir".to_owned(),
        path_to_string(&options.output_root),
        "--load-predictions-dir".to_owned(),
        row.primary_predictions_dir.clone(),
        "--model".to_owned(),
        options.model.clone(),
        "--signal-horizon".to_owned(),
        value_or(&row.train_signal_horizon, "20"),
        "--backtest-artifact-level".to_owned(),
        options.backtest_artifact_level.clone(),
    ];
    append_optional(&mut argv, "--retrain-step", &row.train_retrain_step);
    append_optional(&mut argv, "--train-days", &row.train_train_days);
    append_optional(&mut argv, "--valid-days", &row.train_valid_days);
    if has_value(&row.train_label_embargo_days) {
        argv.extend([
            "--set".to_owned(),
            format!(
                "rolling.label_embargo_days={}",
                row.train_label_embargo_days
            ),
        ]);
    }
    if has_value(&row.train_model_profile) {
        argv.extend([
            "--set".to_owned(),
            format!("model.profile={}", row.train_model_profile),
        ]);
    }
    if has_value(&row.train_feature_profile) {
        argv.extend([
            "--set".to_owned(),
            format!("features.profile={}", row.train_feature_profile),
        ]);
    }
    if truthy(&row.score_fusion_enabled) {
        argv.extend([
            "--set".to_owned(),
            format!(
                "strategy.score_fusion.secondary_predictions_dir={}",
                row.secondary_predictions_dir
            ),
        ]);
    }
    if options.save_predictions {
        argv.push("--save-predictions".to_owned());
    }
    if options.skip_reference_baselines {
        argv.push("--skip-reference-baselines".to_owned());
    }
    if options.skip_opportunity_diagnostics {
        argv.push("--skip-opportunity-diagnostics".to_owned());
    }
    if options.skip_backtest_plots {
        argv.push("--skip-backtest-plots".to_owned());
    }
    if options.skip_backtest_trace {
        argv.push("--skip-backtest-trace".to_owned());
    }
    argv
}

fn build_bundle_command(argv: &[String], options: &BatchOptions) -> Result<Vec<String>, String> {
    let exe =
        env::current_exe().map_err(|err| format!("failed to resolve current executable: {err}"))?;
    let mut command = vec![
        path_to_string(&exe),
        "bundle".to_owned(),
        "--repo-root".to_owned(),
        path_to_string(&options.repo_root),
        "--baseline-jobs".to_owned(),
        options.baseline_jobs.to_string(),
    ];
    command.push("--".to_owned());
    command.extend_from_slice(argv);
    Ok(command)
}

fn append_optional(argv: &mut Vec<String>, option: &str, value: &str) {
    if has_value(value) {
        argv.extend([option.to_owned(), value.to_owned()]);
    }
}

fn has_value(value: &str) -> bool {
    !value.is_empty() && value != NONE_MARKER
}

fn value_or(value: &str, default: &str) -> String {
    if has_value(value) {
        value.to_owned()
    } else {
        default.to_owned()
    }
}

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y" | "on"
    )
}

fn safe_tag(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn run_jobs(options: BatchOptions, jobs: Vec<ArtifactJob>) -> Result<ExitCode, String> {
    fs::create_dir_all(&options.output_root)
        .map_err(|err| format!("failed to create {}: {err}", options.output_root.display()))?;
    fs::create_dir_all(&options.log_dir)
        .map_err(|err| format!("failed to create {}: {err}", options.log_dir.display()))?;
    if let Some(marker_dir) = &options.marker_dir {
        fs::create_dir_all(marker_dir)
            .map_err(|err| format!("failed to create {}: {err}", marker_dir.display()))?;
    }
    if let Some(failed_tsv) = &options.failed_tsv {
        if let Some(parent) = failed_tsv.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
    }

    println!("[info] selected_jobs={}", jobs.len());
    println!("[info] output_root={}", options.output_root.display());
    println!("[info] log_dir={}", options.log_dir.display());
    println!("[info] artifact_level={}", options.backtest_artifact_level);
    println!("[info] jobs={}", options.jobs);
    println!("[info] baseline_jobs={}", options.baseline_jobs);
    println!("[info] rust_backtest=enabled");
    if let Some(marker_dir) = &options.marker_dir {
        println!("[info] marker_dir={}", marker_dir.display());
    }
    if let Some(failed_tsv) = &options.failed_tsv {
        println!("[info] failed_tsv={}", failed_tsv.display());
    }

    if options.dry_run {
        for job in &jobs {
            let (done_marker, _) = job_markers(options.marker_dir.as_deref(), job);
            if done_marker.as_ref().is_some_and(|path| path.is_file()) {
                println!("[dry-run skip] {} already marked done", job.row.matrix_id);
                continue;
            }
            println!("[dry-run] {}", job.row.matrix_id);
            println!("[bundle_args] {}", display_args(&job.argv));
            println!("[rust_cmd] {}", display_command(&job.rust_command));
        }
        return Ok(ExitCode::SUCCESS);
    }

    let batch_start = Instant::now();
    let summary_stamp = now_stamp();
    let summary_path = options
        .log_dir
        .join(format!("artifact_rebuild_summary_{summary_stamp}.tsv"));
    let summary_json_path = options
        .log_dir
        .join(format!("artifact_rebuild_summary_{summary_stamp}.json"));
    let mut summary_writer = WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&summary_path)
        .map_err(|err| format!("failed to create {}: {err}", summary_path.display()))?;
    summary_writer
        .write_record([
            "matrix_id",
            "train_id",
            "backtest_profile_id",
            "status",
            "exit_code",
            "elapsed_seconds",
            "log_path",
        ])
        .map_err(|err| format!("failed to write {}: {err}", summary_path.display()))?;

    let mut outcomes = run_jobs_parallel(&options, &jobs)?;
    outcomes.sort_by_key(|outcome| outcome.index);

    let mut failures = 0usize;
    let mut skipped = 0usize;
    let mut job_summaries = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        let job = &jobs[outcome.index];
        if outcome.summary.status == "skipped_done" {
            skipped += 1;
        }
        if outcome.summary.exit_code != 0 {
            failures += 1;
            if let Some((reason, detail)) = outcome.failure.as_ref() {
                append_failed_record(options.failed_tsv.as_deref(), job, reason, detail)?;
            }
        }
        write_summary_row(
            &mut summary_writer,
            job,
            &outcome.summary.status,
            outcome.summary.exit_code,
            outcome.summary.elapsed_seconds,
        )?;
        job_summaries.push(outcome.summary);
        summary_writer
            .flush()
            .map_err(|err| format!("failed to flush {}: {err}", summary_path.display()))?;
    }

    let success_jobs = job_summaries
        .iter()
        .filter(|summary| summary.status == "success")
        .count();
    let batch_summary = BatchSummaryJson {
        selected_tsv: path_to_string(&options.selected_tsv),
        output_root: path_to_string(&options.output_root),
        log_dir: path_to_string(&options.log_dir),
        marker_dir: options.marker_dir.as_ref().map(|path| path_to_string(path)),
        failed_tsv: options.failed_tsv.as_ref().map(|path| path_to_string(path)),
        summary_tsv: path_to_string(&summary_path),
        summary_json: path_to_string(&summary_json_path),
        artifact_level: options.backtest_artifact_level.clone(),
        parallel_jobs: options.jobs,
        selected_jobs: jobs.len(),
        processed_jobs: job_summaries.len(),
        success_jobs,
        skipped_jobs: skipped,
        failed_jobs: failures,
        unprocessed_jobs: jobs.len().saturating_sub(job_summaries.len()),
        elapsed_seconds: batch_start.elapsed().as_secs_f64(),
        jobs: job_summaries,
    };
    write_batch_summary_json(&summary_json_path, &batch_summary)?;

    println!(
        "[summary] skipped={} failures={} summary={} json={}",
        skipped,
        failures,
        summary_path.display(),
        summary_json_path.display()
    );
    Ok(if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn run_jobs_parallel(
    options: &BatchOptions,
    jobs: &[ArtifactJob],
) -> Result<Vec<JobRunOutcome>, String> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    env::set_current_dir(&options.repo_root).map_err(|err| {
        format!(
            "failed to enter repo root {}: {err}",
            options.repo_root.display()
        )
    })?;
    let groups = group_jobs_by_primary_bundle(jobs);
    if groups.len() == jobs.len() {
        return run_jobs_subprocess_parallel(options, jobs);
    }
    run_jobs_grouped_inprocess(options, jobs, groups)
}

fn run_jobs_grouped_inprocess(
    options: &BatchOptions,
    jobs: &[ArtifactJob],
    groups: Vec<JobGroup>,
) -> Result<Vec<JobRunOutcome>, String> {
    let worker_count = options.jobs.max(1).min(groups.len().max(1));
    let stop = AtomicBool::new(false);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|err| format!("failed to build Rayon thread pool: {err}"))?;
    pool.install(|| {
        groups
            .par_iter()
            .map(|group| execute_job_group_inprocess(group, jobs, jobs.len(), options, &stop))
            .try_reduce(Vec::new, |mut left, mut right| {
                left.append(&mut right);
                Ok(left)
            })
    })
}

fn run_jobs_subprocess_parallel(
    options: &BatchOptions,
    jobs: &[ArtifactJob],
) -> Result<Vec<JobRunOutcome>, String> {
    let worker_count = options.jobs.max(1).min(jobs.len());
    let stop = AtomicBool::new(false);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|err| format!("failed to build Rayon thread pool: {err}"))?;
    pool.install(|| {
        (0..jobs.len())
            .into_par_iter()
            .map(|index| {
                if options.fail_fast && stop.load(AtomicOrdering::Relaxed) {
                    return Ok(Vec::new());
                }
                let outcome = execute_job_subprocess(index, &jobs[index], jobs.len(), options);
                if outcome
                    .as_ref()
                    .map(|outcome| outcome.summary.exit_code != 0)
                    .unwrap_or(true)
                    && options.fail_fast
                {
                    stop.store(true, AtomicOrdering::Relaxed);
                }
                outcome.map(|outcome| vec![outcome])
            })
            .try_reduce(Vec::new, |mut left, mut right| {
                left.append(&mut right);
                Ok(left)
            })
    })
}

fn group_jobs_by_primary_bundle(jobs: &[ArtifactJob]) -> Vec<JobGroup> {
    let mut groups: Vec<JobGroup> = Vec::new();
    let mut group_index_by_primary: HashMap<String, usize> = HashMap::new();
    for (index, job) in jobs.iter().enumerate() {
        if let Some(&group_index) = group_index_by_primary.get(&job.row.primary_predictions_dir) {
            groups[group_index].indexes.push(index);
        } else {
            let group_index = groups.len();
            group_index_by_primary.insert(job.row.primary_predictions_dir.clone(), group_index);
            groups.push(JobGroup {
                indexes: vec![index],
            });
        }
    }
    groups
}

fn execute_job_group_inprocess(
    group: &JobGroup,
    jobs: &[ArtifactJob],
    total_jobs: usize,
    options: &BatchOptions,
    stop: &AtomicBool,
) -> Result<Vec<JobRunOutcome>, String> {
    let mut primary_bundle: Option<PredictionBundle> = None;
    let mut primary_bundle_dir: Option<PathBuf> = None;
    let mut secondary_cache = crate::run_bundle::PredictionBundleCache::new();
    let mut outcomes = Vec::with_capacity(group.indexes.len());
    for &index in &group.indexes {
        if options.fail_fast && stop.load(AtomicOrdering::Relaxed) {
            break;
        }
        let outcome = execute_job_inprocess(
            index,
            &jobs[index],
            total_jobs,
            options,
            &mut primary_bundle,
            &mut primary_bundle_dir,
            &mut secondary_cache,
        )?;
        if outcome.summary.exit_code != 0 && options.fail_fast {
            stop.store(true, AtomicOrdering::Relaxed);
        }
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

fn execute_job_inprocess(
    index: usize,
    job: &ArtifactJob,
    total_jobs: usize,
    options: &BatchOptions,
    primary_bundle: &mut Option<PredictionBundle>,
    primary_bundle_dir: &mut Option<PathBuf>,
    secondary_cache: &mut crate::run_bundle::PredictionBundleCache,
) -> Result<JobRunOutcome, String> {
    let (done_marker, failed_marker) = job_markers(options.marker_dir.as_deref(), job);
    if done_marker.as_ref().is_some_and(|path| path.is_file()) {
        println!("[skip] {} already marked done", job.row.matrix_id);
        return Ok(JobRunOutcome {
            index,
            summary: make_job_summary(job, "skipped_done", 0, 0.0, "", ""),
            failure: None,
        });
    }

    println!("[run] {}/{} {}", index + 1, total_jobs, job.row.matrix_id);
    let start = Instant::now();
    let preflight = preflight_job(job);
    let exit_code = run_job_inprocess(
        job,
        options,
        preflight.as_ref(),
        failed_marker.as_deref(),
        primary_bundle,
        primary_bundle_dir,
        secondary_cache,
    )?;
    let elapsed = start.elapsed().as_secs_f64();
    let status = if exit_code == 0 { "success" } else { "failed" };
    let failure = if exit_code == 0 {
        None
    } else {
        Some(if let Some(reason) = preflight.as_ref() {
            (reason.0.clone(), reason.1.clone())
        } else {
            (format!("exit_{exit_code}"), path_to_string(&job.log_path))
        })
    };
    if exit_code == 0 {
        if let Some(done_marker) = done_marker {
            File::create(done_marker)
                .map_err(|err| format!("failed to write done marker: {err}"))?;
        }
    }
    println!(
        "[{}] {} exit={} elapsed={:.2}s log={}",
        status,
        job.row.matrix_id,
        exit_code,
        elapsed,
        job.log_path.display()
    );
    let (failure_reason, failure_detail) = failure
        .as_ref()
        .map(|(reason, detail)| (reason.as_str(), detail.as_str()))
        .unwrap_or(("", ""));
    Ok(JobRunOutcome {
        index,
        summary: make_job_summary(
            job,
            status,
            exit_code,
            elapsed,
            failure_reason,
            failure_detail,
        ),
        failure,
    })
}

fn execute_job_subprocess(
    index: usize,
    job: &ArtifactJob,
    total_jobs: usize,
    options: &BatchOptions,
) -> Result<JobRunOutcome, String> {
    let (done_marker, failed_marker) = job_markers(options.marker_dir.as_deref(), job);
    if done_marker.as_ref().is_some_and(|path| path.is_file()) {
        println!("[skip] {} already marked done", job.row.matrix_id);
        return Ok(JobRunOutcome {
            index,
            summary: make_job_summary(job, "skipped_done", 0, 0.0, "", ""),
            failure: None,
        });
    }

    println!("[run] {}/{} {}", index + 1, total_jobs, job.row.matrix_id);
    let start = Instant::now();
    let preflight = preflight_job(job);
    let exit_code = run_job(job, options, preflight.as_ref(), failed_marker.as_deref())?;
    let elapsed = start.elapsed().as_secs_f64();
    let status = if exit_code == 0 { "success" } else { "failed" };
    let failure = if exit_code == 0 {
        None
    } else {
        Some(if let Some(reason) = preflight.as_ref() {
            (reason.0.clone(), reason.1.clone())
        } else {
            (format!("exit_{exit_code}"), path_to_string(&job.log_path))
        })
    };
    if exit_code == 0 {
        if let Some(done_marker) = done_marker {
            File::create(done_marker)
                .map_err(|err| format!("failed to write done marker: {err}"))?;
        }
    }
    println!(
        "[{}] {} exit={} elapsed={:.2}s log={}",
        status,
        job.row.matrix_id,
        exit_code,
        elapsed,
        job.log_path.display()
    );
    let (failure_reason, failure_detail) = failure
        .as_ref()
        .map(|(reason, detail)| (reason.as_str(), detail.as_str()))
        .unwrap_or(("", ""));
    Ok(JobRunOutcome {
        index,
        summary: make_job_summary(
            job,
            status,
            exit_code,
            elapsed,
            failure_reason,
            failure_detail,
        ),
        failure,
    })
}

fn make_job_summary(
    job: &ArtifactJob,
    status: &str,
    exit_code: i32,
    elapsed_seconds: f64,
    failure_reason: &str,
    failure_detail: &str,
) -> JobSummary {
    JobSummary {
        matrix_id: job.row.matrix_id.clone(),
        train_id: job.row.train_id.clone(),
        backtest_profile_id: job.row.backtest_profile_id.clone(),
        status: status.to_owned(),
        exit_code,
        elapsed_seconds,
        log_path: path_to_string(&job.log_path),
        failure_reason: failure_reason.to_owned(),
        failure_detail: failure_detail.to_owned(),
    }
}

fn write_batch_summary_json(path: &Path, summary: &BatchSummaryJson) -> Result<(), String> {
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, summary)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    writer
        .write_all(b"\n")
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn run_job(
    job: &ArtifactJob,
    options: &BatchOptions,
    preflight: Option<&(String, String)>,
    failed_marker: Option<&Path>,
) -> Result<i32, String> {
    if let Some(parent) = job.log_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let log_file = File::create(&job.log_path)
        .map_err(|err| format!("failed to create {}: {err}", job.log_path.display()))?;
    let mut header_writer = BufWriter::new(
        log_file
            .try_clone()
            .map_err(|err| format!("failed to clone log handle: {err}"))?,
    );
    write_job_header(&mut header_writer, job)?;
    if let Some((reason, detail)) = preflight {
        writeln!(header_writer, "[preflight] {reason}: {detail}")
            .map_err(|err| format!("failed to write {}: {err}", job.log_path.display()))?;
        header_writer
            .flush()
            .map_err(|err| format!("failed to flush {}: {err}", job.log_path.display()))?;
        if let Some(failed_marker) = failed_marker {
            fs::write(failed_marker, "1\n")
                .map_err(|err| format!("failed to write {}: {err}", failed_marker.display()))?;
        }
        return Ok(1);
    }
    header_writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", job.log_path.display()))?;

    if let Some(failed_marker) = failed_marker {
        if failed_marker.exists() {
            fs::remove_file(failed_marker)
                .map_err(|err| format!("failed to remove {}: {err}", failed_marker.display()))?;
        }
    }

    let mut command = Command::new(&job.rust_command[0]);
    command
        .args(&job.rust_command[1..])
        .current_dir(&options.repo_root)
        .stdout(Stdio::from(
            log_file
                .try_clone()
                .map_err(|err| format!("failed to clone log handle: {err}"))?,
        ))
        .stderr(Stdio::from(log_file));
    let status = command.status().map_err(|err| {
        format!(
            "failed to run {}: {err}",
            display_command(&job.rust_command)
        )
    })?;
    let exit_code = status.code().unwrap_or(1);
    if exit_code != 0 {
        if let Some(failed_marker) = failed_marker {
            fs::write(failed_marker, format!("{exit_code}\n"))
                .map_err(|err| format!("failed to write {}: {err}", failed_marker.display()))?;
        }
    }
    Ok(exit_code)
}

fn write_job_header(writer: &mut impl Write, job: &ArtifactJob) -> Result<(), String> {
    writeln!(writer, "[matrix_id] {}", job.row.matrix_id).map_err(|err| err.to_string())?;
    writeln!(writer, "[train_id] {}", job.row.train_id).map_err(|err| err.to_string())?;
    writeln!(
        writer,
        "[backtest_profile_id] {}",
        job.row.backtest_profile_id
    )
    .map_err(|err| err.to_string())?;
    writeln!(writer, "[execution_mode] rust").map_err(|err| err.to_string())?;
    writeln!(writer, "[start] {}", now_stamp()).map_err(|err| err.to_string())?;
    writeln!(writer, "[bundle_args] {}", display_args(&job.argv)).map_err(|err| err.to_string())?;
    writeln!(writer, "[rust_cmd] {}", display_command(&job.rust_command))
        .map_err(|err| err.to_string())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_job_inprocess(
    job: &ArtifactJob,
    options: &BatchOptions,
    preflight: Option<&(String, String)>,
    failed_marker: Option<&Path>,
    primary_bundle: &mut Option<PredictionBundle>,
    primary_bundle_dir: &mut Option<PathBuf>,
    secondary_cache: &mut crate::run_bundle::PredictionBundleCache,
) -> Result<i32, String> {
    if let Some(parent) = job.log_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let log_file = File::create(&job.log_path)
        .map_err(|err| format!("failed to create {}: {err}", job.log_path.display()))?;
    let mut writer = BufWriter::new(log_file);
    write_job_header(&mut writer, job)?;
    writeln!(writer, "[execution_mode_detail] rust_inprocess_grouped")
        .map_err(|err| format!("failed to write {}: {err}", job.log_path.display()))?;
    if let Some((reason, detail)) = preflight {
        writeln!(writer, "[preflight] {reason}: {detail}")
            .map_err(|err| format!("failed to write {}: {err}", job.log_path.display()))?;
        writer
            .flush()
            .map_err(|err| format!("failed to flush {}: {err}", job.log_path.display()))?;
        if let Some(failed_marker) = failed_marker {
            fs::write(failed_marker, "1\n")
                .map_err(|err| format!("failed to write {}: {err}", failed_marker.display()))?;
        }
        return Ok(1);
    }
    if let Some(failed_marker) = failed_marker {
        if failed_marker.exists() {
            fs::remove_file(failed_marker)
                .map_err(|err| format!("failed to remove {}: {err}", failed_marker.display()))?;
        }
    }

    let BundlePlan::Planned(mut plan) =
        bundle_entry::prepare_run(&job.argv, &options.repo_root, options.baseline_jobs)?;
    plan.execution.quiet = true;

    if primary_bundle_dir.as_ref() != Some(&plan.bundle_dir) {
        let bundle = read_prediction_bundle(&plan.bundle_dir)?;
        *primary_bundle = Some(bundle);
        *primary_bundle_dir = Some(plan.bundle_dir.clone());
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", job.log_path.display()))?;
    let code = bundle_entry::run_prepared_with_bundle_and_cache(
        plan,
        primary_bundle
            .as_ref()
            .ok_or_else(|| "internal primary bundle cache miss".to_owned())?,
        secondary_cache,
    )?;
    let exit_code = exit_code_value(code);
    writeln!(writer, "[exit] {exit_code}")
        .map_err(|err| format!("failed to write {}: {err}", job.log_path.display()))?;
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", job.log_path.display()))?;
    if exit_code != 0 {
        if let Some(failed_marker) = failed_marker {
            fs::write(failed_marker, format!("{exit_code}\n"))
                .map_err(|err| format!("failed to write {}: {err}", failed_marker.display()))?;
        }
    }
    Ok(exit_code)
}

fn exit_code_value(code: ExitCode) -> i32 {
    if code == ExitCode::SUCCESS {
        0
    } else {
        1
    }
}

fn preflight_job(job: &ArtifactJob) -> Option<(String, String)> {
    if job.row.primary_predictions_dir.is_empty()
        || !Path::new(&job.row.primary_predictions_dir).is_dir()
    {
        return Some((
            "missing_primary_predictions".to_owned(),
            job.row.primary_predictions_dir.clone(),
        ));
    }
    if truthy(&job.row.score_fusion_enabled)
        && (job.row.secondary_predictions_dir.is_empty()
            || job.row.secondary_predictions_dir == NONE_MARKER
            || !Path::new(&job.row.secondary_predictions_dir).is_dir())
    {
        return Some((
            "missing_secondary_predictions".to_owned(),
            job.row.secondary_predictions_dir.clone(),
        ));
    }
    None
}

fn job_markers(marker_dir: Option<&Path>, job: &ArtifactJob) -> (Option<PathBuf>, Option<PathBuf>) {
    match marker_dir {
        Some(dir) => (
            Some(dir.join(format!("{}.done", job.row.matrix_id))),
            Some(dir.join(format!("{}.failed", job.row.matrix_id))),
        ),
        None => (None, None),
    }
}

fn append_failed_record(
    path: Option<&Path>,
    job: &ArtifactJob,
    reason: &str,
    detail: &str,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    writeln!(
        file,
        "{}\t{}\t{}\t{}",
        job.row.matrix_id, job.row.train_id, reason, detail
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn write_summary_row(
    writer: &mut csv::Writer<File>,
    job: &ArtifactJob,
    status: &str,
    exit_code: i32,
    elapsed: f64,
) -> Result<(), String> {
    writer
        .write_record([
            job.row.matrix_id.as_str(),
            job.row.train_id.as_str(),
            job.row.backtest_profile_id.as_str(),
            status,
            &exit_code.to_string(),
            &format!("{elapsed:.6}"),
            &path_to_string(&job.log_path),
        ])
        .map_err(|err| format!("failed to write summary row: {err}"))
}

fn now_stamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_owned())
}

fn display_args(args: &[String]) -> String {
    args.iter()
        .map(|arg| shell_quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(matrix_id: &str, train_id: &str, backtest_id: &str) -> ArtifactRow {
        ArtifactRow {
            matrix_id: matrix_id.to_owned(),
            train_id: train_id.to_owned(),
            backtest_profile_id: backtest_id.to_owned(),
            config_snapshot: "configs/bt.yaml".to_owned(),
            primary_predictions_dir: "runs/train/prediction_artifacts".to_owned(),
            train_signal_horizon: "20".to_owned(),
            train_retrain_step: "10".to_owned(),
            train_train_days: "242".to_owned(),
            train_valid_days: "10".to_owned(),
            train_label_embargo_days: "21".to_owned(),
            train_model_profile: "lgbm_default_rankic".to_owned(),
            train_feature_profile: "core_v4".to_owned(),
            ..ArtifactRow::default()
        }
    }

    #[test]
    fn selects_rows_by_train_and_backtest_ids() {
        let rows = vec![
            row("train_a__bt_x", "train_a", "bt_x"),
            row("train_a__bt_y", "train_a", "bt_y"),
            row("train_b__bt_x", "train_b", "bt_x"),
        ];
        let selected = select_rows(
            rows,
            &BTreeSet::new(),
            &split_allowlist(&["train_a".to_owned()]),
            &split_allowlist(&["bt_y".to_owned()]),
            "",
            0,
        );
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].matrix_id, "train_a__bt_y");
    }

    #[test]
    fn builds_rolling_argv_like_python_batch_runner() {
        let options = BatchOptions {
            selected_tsv: PathBuf::from("selected.tsv"),
            output_root: PathBuf::from("artifact_runs"),
            log_dir: PathBuf::from("logs"),
            marker_dir: None,
            failed_tsv: None,
            matrix_ids: BTreeSet::new(),
            train_ids: BTreeSet::new(),
            backtest_ids: BTreeSet::new(),
            limit: 0,
            start_after: String::new(),
            jobs: 1,
            baseline_jobs: 1,
            repo_root: PathBuf::from("/repo"),
            model: "lgbm".to_owned(),
            run_tag_prefix: "artifact-rebuild-lgbm".to_owned(),
            backtest_artifact_level: "reports".to_owned(),
            save_predictions: false,
            skip_reference_baselines: true,
            skip_opportunity_diagnostics: true,
            skip_backtest_plots: true,
            skip_backtest_trace: true,
            dry_run: false,
            fail_fast: false,
        };
        let argv = build_rolling_argv(&row("train_a__bt_x", "train_a", "bt_x"), &options);
        assert!(argv.contains(&"--load-predictions-dir".to_owned()));
        assert_eq!(
            argv[argv
                .iter()
                .position(|arg| arg == "--backtest-artifact-level")
                .unwrap()
                + 1],
            "reports"
        );
        assert!(argv.contains(&"--skip-reference-baselines".to_owned()));
        assert!(argv.contains(&"rolling.label_embargo_days=21".to_owned()));
    }

    #[test]
    fn writes_batch_summary_json_with_job_status() {
        let dir = std::env::temp_dir().join(format!("ai4stock_batch_json_{}", now_stamp()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("summary.json");
        let job = ArtifactJob {
            row: row("train_a__bt_x", "train_a", "bt_x"),
            argv: Vec::new(),
            rust_command: Vec::new(),
            log_path: PathBuf::from("logs/train_a__bt_x.log"),
        };
        let summary = BatchSummaryJson {
            selected_tsv: "selected.tsv".to_owned(),
            output_root: "runs".to_owned(),
            log_dir: "logs".to_owned(),
            marker_dir: Some("markers".to_owned()),
            failed_tsv: Some("failed.tsv".to_owned()),
            summary_tsv: "logs/summary.tsv".to_owned(),
            summary_json: path_to_string(&path),
            artifact_level: "metrics".to_owned(),
            parallel_jobs: 1,
            selected_jobs: 1,
            processed_jobs: 1,
            success_jobs: 0,
            skipped_jobs: 0,
            failed_jobs: 1,
            unprocessed_jobs: 0,
            elapsed_seconds: 0.25,
            jobs: vec![make_job_summary(
                &job,
                "failed",
                1,
                0.25,
                "missing_primary_predictions",
                "runs/missing",
            )],
        };

        write_batch_summary_json(&path, &summary).unwrap();

        let payload: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(payload["selected_jobs"], 1);
        assert_eq!(payload["artifact_level"], "metrics");
        assert_eq!(payload["jobs"][0]["matrix_id"], "train_a__bt_x");
        assert_eq!(
            payload["jobs"][0]["failure_reason"],
            "missing_primary_predictions"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
