use ai4stock2_native::common::artifact::{
    json_to_csv_string as json_to_string, read_required_csv_rows as read_csv,
    read_required_json as read_json, write_json_pretty, write_json_rows_csv as write_rows, CsvRow,
    JsonRow,
};
use ai4stock2_native::common::cli::{next_arg, path_to_string, split_value};
use chrono::Local;
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

const CORE_REFERENCE_BASELINE_PREFIXES: &[&str] = &[
    "rank_avg_factor_baseline",
    "rank_ic_weighted_factor_baseline",
];
const CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES: &[&str] = &[
    "fixed_risk_rank_avg_factor_baseline",
    "fixed_risk_rank_ic_weighted_factor_baseline",
];
const REFERENCE_BASELINE_PREFIXES: &[&str] = &[
    "avg_factor_baseline",
    "sign_aligned_factor_baseline",
    "rank_avg_factor_baseline",
    "rank_ic_weighted_factor_baseline",
    "fixed_risk_avg_factor_baseline",
    "fixed_risk_sign_aligned_factor_baseline",
    "fixed_risk_rank_avg_factor_baseline",
    "fixed_risk_rank_ic_weighted_factor_baseline",
];
const REFERENCE_BASELINE_PROFILE_FIELDS: &[&str] = &[
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "information_ratio",
    "max_drawdown",
    "monthly_win_rate",
    "profitable_month_summary",
    "rebalance_win_rate",
    "profitable_rebalance_summary",
    "excess_annualized_return",
    "excess_information_ratio",
    "months_beating_pct",
    "months_beating_summary",
    "rebalances_beating_pct",
    "rebalances_beating_summary",
];
const PORTFOLIO_PROFILE_KEYS: &[&str] = &[
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "excess_annualized_return",
    "excess_information_ratio",
    "monthly_win_rate",
    "rebalance_win_rate",
    "turnover_mean",
];
const PORTFOLIO_KEYS: &[&str] = &[
    "annualized_return",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
    "excess_annualized_return",
    "excess_information_ratio",
    "monthly_win_rate",
    "rebalance_win_rate",
    "top_1_positive_month_share",
    "top_3_positive_month_share",
    "top_5_positive_month_share",
];
const TRAINING_SIGNAL_COLUMNS: &[&str] = &[
    "valid_topk_positive_rate",
    "valid_topk_excess_mean",
    "valid_topk_label_mean",
    "valid_topk_min_label_mean",
    "best_valid_daily_rank_ic",
];
#[derive(Debug, Clone)]
struct Options {
    runs: Vec<String>,
    candidate_root: Option<PathBuf>,
    no_sync_candidate_root: bool,
    output_dir: Option<PathBuf>,
    tag: String,
    top_bucket: i64,
    middle_buckets: BTreeSet<i64>,
}

#[derive(Debug, Clone)]
struct CandidateRoots {
    shortlist_root: Option<PathBuf>,
    candidate_dirs: BTreeMap<String, PathBuf>,
}

#[derive(Debug, Clone)]
struct Thresholds {
    min_annualized_return: f64,
    min_sharpe_ratio: f64,
    max_drawdown_floor: f64,
    max_top5_positive_share: f64,
    min_bucket_top_minus_bottom: f64,
    min_yearly_bucket_top_minus_bottom: f64,
    max_calibrated_best_bucket: i64,
    max_calibrated_top_bucket_rank: i64,
    min_validation_high_low_return_spread: f64,
    min_validation_high_bin_win_rate: f64,
    min_excess_annualized_vs_rank_avg_baseline: f64,
    min_rebalance_win_vs_rank_avg_baseline: f64,
    min_excess_annualized_vs_rank_ic_baseline: f64,
    min_rebalance_win_vs_rank_ic_baseline: f64,
    min_excess_annualized_vs_fixed_risk_rank_avg_baseline: f64,
    min_rebalance_win_vs_fixed_risk_rank_avg_baseline: f64,
    min_excess_annualized_vs_fixed_risk_rank_ic_baseline: f64,
    min_rebalance_win_vs_fixed_risk_rank_ic_baseline: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            min_annualized_return: 0.30,
            min_sharpe_ratio: 1.50,
            max_drawdown_floor: -0.15,
            max_top5_positive_share: 0.50,
            min_bucket_top_minus_bottom: 0.0,
            min_yearly_bucket_top_minus_bottom: -0.005,
            max_calibrated_best_bucket: 2,
            max_calibrated_top_bucket_rank: 2,
            min_validation_high_low_return_spread: 0.01,
            min_validation_high_bin_win_rate: 0.60,
            min_excess_annualized_vs_rank_avg_baseline: 0.0,
            min_rebalance_win_vs_rank_avg_baseline: 0.55,
            min_excess_annualized_vs_rank_ic_baseline: 0.0,
            min_rebalance_win_vs_rank_ic_baseline: 0.55,
            min_excess_annualized_vs_fixed_risk_rank_avg_baseline: 0.0,
            min_rebalance_win_vs_fixed_risk_rank_avg_baseline: 0.55,
            min_excess_annualized_vs_fixed_risk_rank_ic_baseline: 0.0,
            min_rebalance_win_vs_fixed_risk_rank_ic_baseline: 0.55,
        }
    }
}

pub(crate) fn run_candidate_pool_command(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    let runs = parse_runs(&options)?;
    let roots = resolve_candidate_dirs(&options)?;
    let output_dir = resolve_output_dir(&options);
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;

    let portfolio = summarize_portfolio(&runs)?;
    let yearly = summarize_yearly(&runs)?;
    let concentration = summarize_concentration(&runs)?;
    let (bucket_shape, yearly_bucket_shape) =
        summarize_buckets(&runs, options.top_bucket, &options.middle_buckets)?;
    let feature_families = summarize_feature_families(&runs)?;
    let validation_bins = summarize_validation_signal_bins(&runs)?;
    let candidate_profiles = build_candidate_profiles(
        &runs,
        options.top_bucket,
        &options.middle_buckets,
        &Thresholds::default(),
    )?;
    let candidate_profile_rows = flatten_candidate_profiles(&candidate_profiles);

    write_rows(&output_dir.join("portfolio_summary.csv"), &portfolio)?;
    write_rows(&output_dir.join("yearly_path_summary.csv"), &yearly)?;
    write_rows(
        &output_dir.join("positive_month_concentration.csv"),
        &concentration,
    )?;
    write_rows(&output_dir.join("bucket_shape_summary.csv"), &bucket_shape)?;
    write_rows(
        &output_dir.join("yearly_bucket_shape_summary.csv"),
        &yearly_bucket_shape,
    )?;
    write_rows(
        &output_dir.join("feature_family_importance.csv"),
        &feature_families,
    )?;
    write_rows(
        &output_dir.join("validation_signal_bins.csv"),
        &validation_bins,
    )?;
    write_rows(
        &output_dir.join("candidate_profiles.csv"),
        &candidate_profile_rows,
    )?;
    write_json(
        &output_dir.join("candidate_profiles.json"),
        &JsonValue::Array(candidate_profiles.clone()),
    )?;
    write_readme(&CandidateReadmeInput {
        output_dir: &output_dir,
        runs: &runs,
        portfolio: &portfolio,
        yearly: &yearly,
        concentration: &concentration,
        bucket_shape: &bucket_shape,
        yearly_bucket_shape: &yearly_bucket_shape,
        validation_bins: &validation_bins,
        candidate_profiles: &candidate_profile_rows,
    })?;

    if !options.no_sync_candidate_root {
        if let Some(shortlist_root) = roots.shortlist_root.as_ref() {
            sync_outputs_to_candidate_root(
                &output_dir,
                shortlist_root,
                &roots.candidate_dirs,
                &candidate_profiles,
                &candidate_profile_rows,
            )?;
        }
    }

    println!(
        "[+] Candidate-pool diagnostics saved to: {}",
        output_dir.display()
    );
    println!("    README: {}", output_dir.join("README.md").display());
    println!(
        "    portfolio: {}",
        output_dir.join("portfolio_summary.csv").display()
    );
    println!(
        "    yearly: {}",
        output_dir.join("yearly_path_summary.csv").display()
    );
    println!(
        "    bucket shape: {}",
        output_dir.join("bucket_shape_summary.csv").display()
    );
    println!(
        "    validation bins: {}",
        output_dir.join("validation_signal_bins.csv").display()
    );
    println!(
        "    candidate profiles: {}",
        output_dir.join("candidate_profiles.csv").display()
    );
    if !options.no_sync_candidate_root {
        if let Some(shortlist_root) = roots.shortlist_root {
            println!("    synced candidate root: {}", shortlist_root.display());
        }
    }
    Ok(())
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options {
        runs: Vec::new(),
        candidate_root: None,
        no_sync_candidate_root: false,
        output_dir: None,
        tag: String::new(),
        top_bucket: 1,
        middle_buckets: parse_bucket_ids("3,4,5,6,7")?,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--run" => options.runs.push(next_arg(args, &mut index, "--run")?),
            value if value.starts_with("--run=") => options.runs.push(split_value(value, "--run")?),
            "--candidate-root" => {
                options.candidate_root = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--candidate-root",
                )?))
            }
            value if value.starts_with("--candidate-root=") => {
                options.candidate_root =
                    Some(PathBuf::from(split_value(value, "--candidate-root")?))
            }
            "--no-sync-candidate-root" => options.no_sync_candidate_root = true,
            "--output-dir" => {
                options.output_dir =
                    Some(PathBuf::from(next_arg(args, &mut index, "--output-dir")?))
            }
            value if value.starts_with("--output-dir=") => {
                options.output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?))
            }
            "--tag" => options.tag = next_arg(args, &mut index, "--tag")?,
            value if value.starts_with("--tag=") => options.tag = split_value(value, "--tag")?,
            "--top-bucket" => {
                options.top_bucket =
                    parse_i64(&next_arg(args, &mut index, "--top-bucket")?, "--top-bucket")?
            }
            value if value.starts_with("--top-bucket=") => {
                options.top_bucket =
                    parse_i64(&split_value(value, "--top-bucket")?, "--top-bucket")?
            }
            "--middle-buckets" => {
                options.middle_buckets =
                    parse_bucket_ids(&next_arg(args, &mut index, "--middle-buckets")?)?
            }
            value if value.starts_with("--middle-buckets=") => {
                options.middle_buckets = parse_bucket_ids(&split_value(value, "--middle-buckets")?)?
            }
            other => {
                return Err(format!(
                    "unknown candidate-pool option: {other}\n\n{}",
                    usage()
                ))
            }
        }
        index += 1;
    }
    Ok(options)
}

fn usage() -> &'static str {
    "\
Usage:
  ai4stock-diagnostics candidate-pool (--run NAME=DIR | --candidate-root DIR) [options]

Options:
  --run NAME=DIR
  --candidate-root DIR
  --no-sync-candidate-root
  --output-dir DIR
  --tag TEXT
  --top-bucket N
  --middle-buckets A,B,C
"
}

fn parse_runs(options: &Options) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut runs = BTreeMap::new();
    if let Some(root) = options.candidate_root.as_ref() {
        let scan_root = if root.join("candidates").is_dir() {
            root.join("candidates")
        } else {
            root.clone()
        };
        if !scan_root.is_dir() {
            return Err(format!("Candidate root not found: {}", root.display()));
        }
        for entry in fs::read_dir(&scan_root)
            .map_err(|err| format!("failed to read {}: {err}", scan_root.display()))?
        {
            let path = entry.map_err(|err| err.to_string())?.path();
            let snapshot = path.join("snapshot");
            if path.is_dir() && snapshot.is_dir() {
                if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
                    runs.insert(name.to_owned(), snapshot);
                }
            }
        }
    }
    for raw in &options.runs {
        let Some((name, path)) = raw.split_once('=') else {
            return Err(format!("--run must use NAME=DIR format, got: {raw}"));
        };
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("--run name cannot be empty: {raw}"));
        }
        runs.insert(name.to_owned(), PathBuf::from(path.trim()));
    }
    if runs.is_empty() {
        return Err("Provide at least one --run or --candidate-root.".to_owned());
    }
    let missing = runs
        .iter()
        .filter(|(_, path)| !path.exists())
        .map(|(name, path)| format!("{name}={}", path.display()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!("Run directories not found: {}", missing.join(", ")));
    }
    Ok(runs)
}

fn resolve_candidate_dirs(options: &Options) -> Result<CandidateRoots, String> {
    let Some(root) = options.candidate_root.as_ref() else {
        return Ok(CandidateRoots {
            shortlist_root: None,
            candidate_dirs: BTreeMap::new(),
        });
    };
    let candidate_root = if root.join("candidates").is_dir() {
        root.join("candidates")
    } else {
        root.clone()
    };
    let shortlist_root =
        if candidate_root.file_name().and_then(|value| value.to_str()) == Some("candidates") {
            candidate_root
                .parent()
                .unwrap_or(&candidate_root)
                .to_path_buf()
        } else {
            candidate_root.clone()
        };
    let mut candidate_dirs = BTreeMap::new();
    if candidate_root.is_dir() {
        for entry in fs::read_dir(&candidate_root)
            .map_err(|err| format!("failed to read {}: {err}", candidate_root.display()))?
        {
            let path = entry.map_err(|err| err.to_string())?.path();
            if path.is_dir() && path.join("snapshot").is_dir() {
                if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
                    candidate_dirs.insert(name.to_owned(), path);
                }
            }
        }
    }
    Ok(CandidateRoots {
        shortlist_root: Some(shortlist_root),
        candidate_dirs,
    })
}

fn resolve_output_dir(options: &Options) -> PathBuf {
    if let Some(output_dir) = &options.output_dir {
        return output_dir.clone();
    }
    let suffix = if options.tag.trim().is_empty() {
        String::new()
    } else {
        format!("__{}", options.tag.trim())
    };
    PathBuf::from("results")
        .join("diagnostics")
        .join("candidate_pool")
        .join(format!(
            "{}{}",
            Local::now().format("%Y%m%d_%H%M%S"),
            suffix
        ))
}

fn summarize_portfolio(runs: &BTreeMap<String, PathBuf>) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in runs {
        let metrics = read_json(&run_dir.join("native_portfolio_metrics.json"))?;
        let mut row = JsonRow::new();
        insert_str(&mut row, "run", run_name);
        insert_str(&mut row, "run_dir", &path_to_string(run_dir));
        for key in PORTFOLIO_KEYS {
            row.insert((*key).to_owned(), metric_value(&metrics, key));
        }
        for prefix in CORE_REFERENCE_BASELINE_PREFIXES
            .iter()
            .chain(CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES.iter())
        {
            for field in reference_baseline_summary_fields(prefix) {
                row.insert(field.clone(), metric_value(&metrics, &field));
            }
        }
        row.insert(
            "profitable_month_summary".to_owned(),
            metrics
                .get("profitable_month_summary")
                .cloned()
                .unwrap_or(JsonValue::Null),
        );
        row.insert(
            "profitable_rebalance_summary".to_owned(),
            metrics
                .get("profitable_rebalance_summary")
                .cloned()
                .unwrap_or(JsonValue::Null),
        );
        rows.push(row);
    }
    Ok(rows)
}

fn summarize_yearly(runs: &BTreeMap<String, PathBuf>) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in runs {
        let monthly = read_csv(&run_dir.join("native_monthly_summary.csv"))?;
        for (year, group) in group_by_year(&monthly) {
            let returns = numeric_column(&group, "return");
            let bench = numeric_column(&group, "bench_return");
            let excess = numeric_column(&group, "excess_vs_benchmark");
            let mut row = JsonRow::new();
            insert_str(&mut row, "run", run_name);
            insert_str(&mut row, "year", &year);
            insert_i64(&mut row, "month_count", group.len() as i64);
            insert_f64(&mut row, "compound_return", compound_return(&returns));
            insert_f64(
                &mut row,
                "benchmark_compound_return",
                compound_return(&bench),
            );
            insert_f64(
                &mut row,
                "compound_excess_vs_benchmark",
                compound_return(&returns) - compound_return(&bench),
            );
            insert_i64(
                &mut row,
                "negative_months",
                returns.iter().filter(|v| **v < 0.0).count() as i64,
            );
            insert_i64(
                &mut row,
                "months_beating_benchmark",
                excess.iter().filter(|v| **v > 0.0).count() as i64,
            );
            insert_f64(&mut row, "worst_month_return", min_f64(&returns));
            insert_f64(&mut row, "best_month_return", max_f64(&returns));
            insert_f64(
                &mut row,
                "mean_monthly_turnover",
                mean(&numeric_column(&group, "avg_turnover")),
            );
            rows.push(row);
        }
    }
    Ok(rows)
}

fn summarize_concentration(runs: &BTreeMap<String, PathBuf>) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in runs {
        let monthly = read_csv(&run_dir.join("native_monthly_summary.csv"))?;
        let returns = numeric_column(&monthly, "return");
        let mut positives = returns
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > 0.0)
            .map(|(idx, value)| (idx, *value))
            .collect::<Vec<_>>();
        positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let positive_sum = positives.iter().map(|(_, value)| *value).sum::<f64>();
        let raw_sum = returns.iter().sum::<f64>();
        let mut row = JsonRow::new();
        insert_str(&mut row, "run", run_name);
        insert_i64(
            &mut row,
            "positive_month_count",
            returns.iter().filter(|v| **v > 0.0).count() as i64,
        );
        insert_i64(
            &mut row,
            "negative_month_count",
            returns.iter().filter(|v| **v < 0.0).count() as i64,
        );
        insert_f64(&mut row, "raw_monthly_return_sum", raw_sum);
        insert_f64(&mut row, "positive_month_return_sum", positive_sum);
        insert_f64_or_null(
            &mut row,
            "top3_positive_share",
            ratio(sum_top(&positives, 3), positive_sum),
        );
        insert_f64_or_null(
            &mut row,
            "top5_positive_share",
            ratio(sum_top(&positives, 5), positive_sum),
        );
        insert_f64_or_null(
            &mut row,
            "top3_raw_sum_share",
            ratio(sum_top(&positives, 3), raw_sum),
        );
        insert_f64_or_null(
            &mut row,
            "top5_raw_sum_share",
            ratio(sum_top(&positives, 5), raw_sum),
        );
        for (rank, (idx, value)) in positives.iter().take(5).enumerate() {
            insert_str(
                &mut row,
                &format!("best_month_{}", rank + 1),
                cell(&monthly[*idx], "period"),
            );
            insert_f64(&mut row, &format!("best_month_{}_return", rank + 1), *value);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn summarize_buckets(
    runs: &BTreeMap<String, PathBuf>,
    top_bucket: i64,
    middle_buckets: &BTreeSet<i64>,
) -> Result<(Vec<JsonRow>, Vec<JsonRow>), String> {
    let mut shape_rows = Vec::new();
    let mut yearly_rows = Vec::new();
    for (run_name, run_dir) in runs {
        let bucket = read_csv(&run_dir.join("native_score_bucket_report.csv"))?;
        let mut row = bucket_shape(&bucket, top_bucket, middle_buckets)?;
        insert_str(&mut row, "run", run_name);
        shape_rows.push(row);
        let yearly = read_csv(&run_dir.join("native_score_bucket_yearly_report.csv"))?;
        let mut by_year: BTreeMap<String, Vec<CsvRow>> = BTreeMap::new();
        for row in yearly {
            by_year
                .entry(cell(&row, "year").to_owned())
                .or_default()
                .push(row);
        }
        for (year, group) in by_year {
            let mut row = bucket_shape(&group, top_bucket, middle_buckets)?;
            insert_str(&mut row, "run", run_name);
            insert_str(&mut row, "year", &year);
            yearly_rows.push(row);
        }
    }
    Ok((shape_rows, yearly_rows))
}

fn summarize_feature_families(runs: &BTreeMap<String, PathBuf>) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in runs {
        let path = run_dir.join("feature_importance_gain_mean.csv");
        if !path.exists() {
            continue;
        }
        let frame = read_csv(&path)?;
        let mut by_family: BTreeMap<String, f64> = BTreeMap::new();
        for row in frame {
            let family = feature_family(cell(&row, "feature"));
            *by_family.entry(family).or_insert(0.0) +=
                parse_f64_cell(&row, "importance_gain").unwrap_or(0.0);
        }
        let total = by_family.values().sum::<f64>();
        for (family, importance) in by_family {
            let mut row = JsonRow::new();
            insert_str(&mut row, "run", run_name);
            insert_str(&mut row, "feature_family", &family);
            insert_f64(&mut row, "importance_gain", importance);
            insert_f64_or_null(&mut row, "importance_share", ratio(importance, total));
            rows.push(row);
        }
    }
    Ok(rows)
}

fn summarize_validation_signal_bins(
    runs: &BTreeMap<String, PathBuf>,
) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in runs {
        let bins = summarize_validation_bins(run_dir)?;
        for mut row in bins {
            insert_str(&mut row, "run", run_name);
            rows.push(row);
        }
    }
    Ok(rows)
}

fn build_candidate_profiles(
    runs: &BTreeMap<String, PathBuf>,
    top_bucket: i64,
    middle_buckets: &BTreeSet<i64>,
    thresholds: &Thresholds,
) -> Result<Vec<JsonValue>, String> {
    let mut profiles = Vec::new();
    for (run_name, run_dir) in runs {
        profiles.push(build_candidate_profile(
            run_name,
            run_dir,
            top_bucket,
            middle_buckets,
            thresholds,
        )?);
    }
    Ok(profiles)
}

fn build_candidate_profile(
    run_name: &str,
    run_dir: &Path,
    top_bucket: i64,
    middle_buckets: &BTreeSet<i64>,
    thresholds: &Thresholds,
) -> Result<JsonValue, String> {
    let metrics = read_json(&run_dir.join("native_portfolio_metrics.json"))?;
    let monthly = read_csv(&run_dir.join("native_monthly_summary.csv"))?;
    let bucket = read_csv(&run_dir.join("native_score_bucket_report.csv"))?;
    let yearly_bucket = read_csv(&run_dir.join("native_score_bucket_yearly_report.csv"))?;

    let portfolio = PORTFOLIO_PROFILE_KEYS
        .iter()
        .map(|key| ((*key).to_owned(), metric_value(&metrics, key)))
        .collect::<serde_json::Map<_, _>>();
    let reference_baselines = summarize_reference_baselines(&metrics);
    let yearly_path = summarize_yearly_path(&monthly);
    let concentration = summarize_concentration_inner(&monthly);
    let overall_bucket = row_to_json_object(bucket_shape(&bucket, top_bucket, middle_buckets)?);
    let mut yearly_buckets = Vec::new();
    for (year, group) in group_by_key(yearly_bucket, "year") {
        let mut row = bucket_shape(&group, top_bucket, middle_buckets)?;
        insert_str(&mut row, "year", &year);
        yearly_buckets.push(row_to_json_object(row));
    }
    let validation_bins = summarize_validation_bins(run_dir)?;
    let validation_edges = summarize_validation_edges(&validation_bins);
    let promotion_gates = build_promotion_gates(
        &JsonValue::Object(portfolio.clone()),
        &reference_baselines,
        &concentration,
        &JsonValue::Object(overall_bucket.clone()),
        &yearly_buckets,
        &validation_edges,
        thresholds,
    );
    let failed_gates = promotion_gates
        .iter()
        .filter(|(_, gate)| {
            !gate
                .get("passed")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
        })
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    let role = infer_candidate_role(&promotion_gates);
    let strong_years = yearly_path
        .iter()
        .filter(|row| {
            f64_from_json(row.get("compound_return")).unwrap_or(0.0) > 0.25
                && i64_from_json(row.get("months_beating_benchmark")).unwrap_or(0)
                    >= std::cmp::max(7, i64_from_json(row.get("month_count")).unwrap_or(0) / 2)
        })
        .filter_map(|row| {
            row.get("year")
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    let weak_years = yearly_path
        .iter()
        .filter(|row| {
            f64_from_json(row.get("compound_return")).unwrap_or(0.0) < 0.15
                || i64_from_json(row.get("negative_months")).unwrap_or(0)
                    >= i64_from_json(row.get("month_count")).unwrap_or(0) / 2
        })
        .filter_map(|row| {
            row.get("year")
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
        })
        .collect::<Vec<_>>();
    let weak_yearly_buckets = yearly_buckets
        .iter()
        .filter(|row| {
            f64_from_json(row.get("top_minus_bottom")).unwrap_or(0.0)
                < thresholds.min_yearly_bucket_top_minus_bottom
                || i64_from_json(row.get("top_bucket_label_rank")).unwrap_or(999)
                    > thresholds.max_calibrated_top_bucket_rank
        })
        .cloned()
        .collect::<Vec<_>>();
    let risk_profile = summarize_risk_profile(run_dir)?;
    Ok(serde_json::json!({
        "name": run_name,
        "run_dir": path_to_string(run_dir),
        "candidate_role": role,
        "portfolio": portfolio,
        "reference_baselines": reference_baselines,
        "regime_profile": {
            "strong_years": strong_years,
            "weak_years": weak_years,
            "yearly_path": yearly_path,
        },
        "concentration_profile": concentration,
        "calibration_profile": {
            "overall_bucket": overall_bucket,
            "yearly_buckets": yearly_buckets,
            "weak_yearly_buckets": weak_yearly_buckets,
        },
        "validation_profile": {
            "bins": validation_bins.into_iter().map(row_to_json_object).collect::<Vec<_>>(),
            "edges": validation_edges,
            "preferred_primary_metric": if validation_edges.contains_key("valid_topk_excess_mean") { JsonValue::String("valid_topk_excess_mean".to_owned()) } else { JsonValue::Null },
        },
        "feature_profile": {
            "family_importance": summarize_feature_families_for_run(run_dir)?,
        },
        "risk_profile": risk_profile,
        "promotion_gates": promotion_gates,
        "gate_summary": {
            "passed_count": promotion_gates.len() - failed_gates.len(),
            "total_count": promotion_gates.len(),
            "failed_gates": failed_gates,
        },
    }))
}

fn summarize_reference_baselines(metrics: &JsonValue) -> serde_json::Map<String, JsonValue> {
    let mut out = serde_json::Map::new();
    for prefix in REFERENCE_BASELINE_PREFIXES {
        let mut baseline = serde_json::Map::new();
        baseline.insert(
            "annualized_return".to_owned(),
            metric_value(metrics, &format!("{prefix}_annualized_return")),
        );
        baseline.insert(
            "annualized_volatility".to_owned(),
            metric_value(metrics, &format!("{prefix}_annualized_volatility")),
        );
        baseline.insert(
            "sharpe_ratio".to_owned(),
            metric_value(metrics, &format!("{prefix}_sharpe_ratio")),
        );
        baseline.insert(
            "information_ratio".to_owned(),
            metric_value(metrics, &format!("{prefix}_information_ratio")),
        );
        baseline.insert(
            "max_drawdown".to_owned(),
            metric_value(metrics, &format!("{prefix}_max_drawdown")),
        );
        baseline.insert(
            "monthly_win_rate".to_owned(),
            metric_value(metrics, &format!("{prefix}_monthly_win_rate")),
        );
        baseline.insert(
            "profitable_month_summary".to_owned(),
            metric_value(metrics, &format!("{prefix}_profitable_month_summary")),
        );
        baseline.insert(
            "rebalance_win_rate".to_owned(),
            metric_value(metrics, &format!("{prefix}_rebalance_win_rate")),
        );
        baseline.insert(
            "profitable_rebalance_summary".to_owned(),
            metric_value(metrics, &format!("{prefix}_profitable_rebalance_summary")),
        );
        baseline.insert(
            "excess_annualized_return".to_owned(),
            metric_value(metrics, &format!("{prefix}_excess_annualized_return")),
        );
        baseline.insert(
            "excess_information_ratio".to_owned(),
            metric_value(metrics, &format!("{prefix}_excess_information_ratio")),
        );
        baseline.insert(
            "months_beating_pct".to_owned(),
            metric_value(metrics, &format!("months_beating_{prefix}_pct")),
        );
        baseline.insert(
            "months_beating_summary".to_owned(),
            metric_value(metrics, &format!("months_beating_{prefix}_summary")),
        );
        baseline.insert(
            "rebalances_beating_pct".to_owned(),
            metric_value(metrics, &format!("rebalances_beating_{prefix}_pct")),
        );
        baseline.insert(
            "rebalances_beating_summary".to_owned(),
            metric_value(metrics, &format!("rebalances_beating_{prefix}_summary")),
        );
        if baseline.values().any(|value| !value.is_null()) {
            out.insert((*prefix).to_owned(), JsonValue::Object(baseline));
        }
    }
    out
}

fn build_promotion_gates(
    portfolio: &JsonValue,
    reference_baselines: &serde_json::Map<String, JsonValue>,
    concentration: &JsonValue,
    overall_bucket: &JsonValue,
    yearly_buckets: &[serde_json::Map<String, JsonValue>],
    validation_edges: &serde_json::Map<String, JsonValue>,
    thresholds: &Thresholds,
) -> serde_json::Map<String, JsonValue> {
    let valid_edge = validation_edges
        .get("valid_topk_excess_mean")
        .unwrap_or(&JsonValue::Null)
        .clone();
    let min_yearly_top_minus_bottom = yearly_buckets
        .iter()
        .filter_map(|row| f64_from_json(row.get("top_minus_bottom")))
        .reduce(f64::min);
    let mut gates = serde_json::Map::new();
    gates.insert(
        "return_quality".to_owned(),
        serde_json::json!({
            "passed": f64_path(portfolio, "annualized_return") >= thresholds.min_annualized_return
                && f64_path(portfolio, "sharpe_ratio") >= thresholds.min_sharpe_ratio,
            "annualized_return": value_path(portfolio, "annualized_return"),
            "sharpe_ratio": value_path(portfolio, "sharpe_ratio"),
        }),
    );
    gates.insert(
        "drawdown_control".to_owned(),
        serde_json::json!({
            "passed": f64_path(portfolio, "max_drawdown") >= thresholds.max_drawdown_floor,
            "max_drawdown": value_path(portfolio, "max_drawdown"),
            "floor": thresholds.max_drawdown_floor,
        }),
    );
    gates.insert(
        "concentration_control".to_owned(),
        serde_json::json!({
            "passed": f64_path(concentration, "top5_positive_share") <= thresholds.max_top5_positive_share,
            "top5_positive_share": value_path(concentration, "top5_positive_share"),
            "max_allowed": thresholds.max_top5_positive_share,
        }),
    );
    gates.insert(
        "bucket_separates_bad_tail".to_owned(),
        serde_json::json!({
            "passed": f64_path(overall_bucket, "top_minus_bottom") > thresholds.min_bucket_top_minus_bottom,
            "top_minus_bottom": value_path(overall_bucket, "top_minus_bottom"),
            "minimum": thresholds.min_bucket_top_minus_bottom,
        }),
    );
    gates.insert(
        "bucket_calibrated_for_sizing".to_owned(),
        serde_json::json!({
            "passed": i64_path(overall_bucket, "best_bucket") <= thresholds.max_calibrated_best_bucket
                && i64_path(overall_bucket, "top_bucket_label_rank") <= thresholds.max_calibrated_top_bucket_rank
                && f64_path(overall_bucket, "top_minus_middle_best") >= 0.0,
            "best_bucket": value_path(overall_bucket, "best_bucket"),
            "top_bucket_label_rank": value_path(overall_bucket, "top_bucket_label_rank"),
            "top_minus_middle_best": value_path(overall_bucket, "top_minus_middle_best"),
        }),
    );
    gates.insert(
        "yearly_bucket_not_inverted".to_owned(),
        serde_json::json!({
            "passed": min_yearly_top_minus_bottom.map(|value| value >= thresholds.min_yearly_bucket_top_minus_bottom).unwrap_or(false),
            "min_yearly_top_minus_bottom": min_yearly_top_minus_bottom,
            "minimum": thresholds.min_yearly_bucket_top_minus_bottom,
        }),
    );
    gates.insert(
        "validation_metric_supports_risk_gate".to_owned(),
        serde_json::json!({
            "passed": f64_path(&valid_edge, "high_low_return_spread") >= thresholds.min_validation_high_low_return_spread
                && f64_path(&valid_edge, "high_positive_rebalance_rate") >= thresholds.min_validation_high_bin_win_rate,
            "high_low_return_spread": value_path(&valid_edge, "high_low_return_spread"),
            "high_positive_rebalance_rate": value_path(&valid_edge, "high_positive_rebalance_rate"),
        }),
    );
    for prefix in candidate_gate_baseline_prefixes(reference_baselines) {
        if let Some(baseline) = reference_baselines.get(&prefix) {
            let (min_excess, min_rebalance) = baseline_thresholds(&prefix, thresholds);
            let excess = f64_path(baseline, "excess_annualized_return");
            let rebalance = f64_path(baseline, "rebalances_beating_pct");
            gates.insert(
                format!("beats_{prefix}"),
                serde_json::json!({
                    "passed": excess >= min_excess && rebalance >= min_rebalance,
                    "excess_annualized_return": value_path(baseline, "excess_annualized_return"),
                    "minimum_excess_annualized_return": min_excess,
                    "rebalances_beating_pct": value_path(baseline, "rebalances_beating_pct"),
                    "minimum_rebalances_beating_pct": min_rebalance,
                    "rebalances_beating_summary": value_path(baseline, "rebalances_beating_summary"),
                }),
            );
        }
    }
    gates
}

fn infer_candidate_role(gates: &serde_json::Map<String, JsonValue>) -> String {
    if !gate_passed(gates, "return_quality") {
        return "research_archive".to_owned();
    }
    let baseline_gates = gates
        .iter()
        .filter(|(name, _)| name.starts_with("beats_") && name.ends_with("_factor_baseline"))
        .map(|(_, gate)| {
            gate.get("passed")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if !baseline_gates.is_empty() && !baseline_gates.iter().all(|value| *value) {
        return "portfolio_candidate_requires_router".to_owned();
    }
    if gate_passed(gates, "bucket_calibrated_for_sizing")
        && gate_passed(gates, "validation_metric_supports_risk_gate")
    {
        return "ranker_and_sizer".to_owned();
    }
    if gate_passed(gates, "bucket_separates_bad_tail")
        && gate_passed(gates, "validation_metric_supports_risk_gate")
    {
        return "pool_selector_with_validation_gate".to_owned();
    }
    if gate_passed(gates, "bucket_separates_bad_tail") {
        return "pool_selector_only".to_owned();
    }
    "portfolio_candidate_requires_router".to_owned()
}

fn flatten_candidate_profiles(profiles: &[JsonValue]) -> Vec<JsonRow> {
    profiles.iter().map(flatten_candidate_profile).collect()
}

fn flatten_candidate_profile(profile: &JsonValue) -> JsonRow {
    let mut row = JsonRow::new();
    insert_json_path(&mut row, "name", profile, &["name"]);
    insert_json_path(&mut row, "candidate_role", profile, &["candidate_role"]);
    for key in [
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "excess_information_ratio",
        "monthly_win_rate",
        "rebalance_win_rate",
    ] {
        insert_json_path(&mut row, key, profile, &["portfolio", key]);
    }
    for prefix in REFERENCE_BASELINE_PREFIXES {
        for field in REFERENCE_BASELINE_PROFILE_FIELDS {
            insert_json_path(
                &mut row,
                &format!("{prefix}_{field}"),
                profile,
                &["reference_baselines", prefix, field],
            );
        }
    }
    insert_json_path(
        &mut row,
        "top5_positive_share",
        profile,
        &["concentration_profile", "top5_positive_share"],
    );
    for key in [
        "best_bucket",
        "top_bucket_label_rank",
        "top_minus_bottom",
        "top_minus_middle_best",
    ] {
        insert_json_path(
            &mut row,
            key,
            profile,
            &["calibration_profile", "overall_bucket", key],
        );
    }
    insert_json_path(
        &mut row,
        "validation_high_low_return_spread",
        profile,
        &[
            "validation_profile",
            "edges",
            "valid_topk_excess_mean",
            "high_low_return_spread",
        ],
    );
    insert_json_path(
        &mut row,
        "validation_high_positive_rebalance_rate",
        profile,
        &[
            "validation_profile",
            "edges",
            "valid_topk_excess_mean",
            "high_positive_rebalance_rate",
        ],
    );
    insert_str(
        &mut row,
        "strong_years",
        &json_string_list(profile, &["regime_profile", "strong_years"]).join("|"),
    );
    insert_str(
        &mut row,
        "weak_years",
        &json_string_list(profile, &["regime_profile", "weak_years"]).join("|"),
    );
    insert_json_path(
        &mut row,
        "max_drawdown_peak_date",
        profile,
        &["risk_profile", "max_drawdown_period", "peak_date"],
    );
    insert_json_path(
        &mut row,
        "max_drawdown_trough_date",
        profile,
        &["risk_profile", "max_drawdown_period", "trough_date"],
    );
    insert_json_path(
        &mut row,
        "passed_gate_count",
        profile,
        &["gate_summary", "passed_count"],
    );
    insert_json_path(
        &mut row,
        "total_gate_count",
        profile,
        &["gate_summary", "total_count"],
    );
    insert_str(
        &mut row,
        "failed_gates",
        &json_string_list(profile, &["gate_summary", "failed_gates"]).join("|"),
    );
    insert_json_path(&mut row, "run_dir", profile, &["run_dir"]);
    row
}

fn summarize_yearly_path(monthly: &[CsvRow]) -> Vec<serde_json::Map<String, JsonValue>> {
    group_by_year(monthly)
        .into_iter()
        .map(|(year, group)| {
            let returns = numeric_column(&group, "return");
            let benchmark = numeric_column(&group, "bench_return");
            let excess = numeric_column(&group, "excess_vs_benchmark");
            let mut row = serde_json::Map::new();
            row.insert("year".to_owned(), JsonValue::String(year));
            row.insert(
                "month_count".to_owned(),
                JsonValue::from(group.len() as i64),
            );
            row.insert(
                "compound_return".to_owned(),
                json_f64(compound_return(&returns)),
            );
            row.insert(
                "benchmark_compound_return".to_owned(),
                json_f64(compound_return(&benchmark)),
            );
            row.insert(
                "compound_excess_vs_benchmark".to_owned(),
                json_f64(compound_return(&returns) - compound_return(&benchmark)),
            );
            row.insert(
                "negative_months".to_owned(),
                JsonValue::from(returns.iter().filter(|value| **value < 0.0).count() as i64),
            );
            row.insert(
                "months_beating_benchmark".to_owned(),
                JsonValue::from(excess.iter().filter(|value| **value > 0.0).count() as i64),
            );
            row.insert("worst_month_return".to_owned(), json_f64(min_f64(&returns)));
            row.insert("best_month_return".to_owned(), json_f64(max_f64(&returns)));
            row.insert(
                "mean_monthly_turnover".to_owned(),
                json_f64(mean(&numeric_column(&group, "avg_turnover"))),
            );
            row
        })
        .collect()
}

fn summarize_concentration_inner(monthly: &[CsvRow]) -> JsonValue {
    let returns = numeric_column(monthly, "return");
    let mut positives = returns
        .iter()
        .enumerate()
        .filter(|(_, value)| **value > 0.0)
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    positives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut sorted_returns = returns
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    sorted_returns.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let positive_sum = positives.iter().map(|(_, value)| *value).sum::<f64>();
    let raw_sum = returns.iter().sum::<f64>();
    let best_months = positives
        .iter()
        .take(5)
        .map(|(idx, value)| {
            serde_json::json!({
                "period": cell(&monthly[*idx], "period"),
                "return": value,
            })
        })
        .collect::<Vec<_>>();
    let worst_months = sorted_returns
        .iter()
        .take(5)
        .map(|(idx, value)| {
            serde_json::json!({
                "period": cell(&monthly[*idx], "period"),
                "return": value,
                "bench_return": parse_f64_cell(&monthly[*idx], "bench_return"),
                "excess_vs_benchmark": parse_f64_cell(&monthly[*idx], "excess_vs_benchmark"),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "positive_month_count": returns.iter().filter(|value| **value > 0.0).count(),
        "negative_month_count": returns.iter().filter(|value| **value < 0.0).count(),
        "raw_monthly_return_sum": raw_sum,
        "positive_month_return_sum": positive_sum,
        "top3_positive_share": ratio(sum_top(&positives, 3), positive_sum),
        "top5_positive_share": ratio(sum_top(&positives, 5), positive_sum),
        "top3_raw_sum_share": ratio(sum_top(&positives, 3), raw_sum),
        "top5_raw_sum_share": ratio(sum_top(&positives, 5), raw_sum),
        "best_months": best_months,
        "worst_months": worst_months,
    })
}

fn summarize_validation_bins(run_dir: &Path) -> Result<Vec<JsonRow>, String> {
    let training = read_csv(&run_dir.join("training_summary.csv"))?;
    let rebalances = read_csv(&run_dir.join("native_rebalance_summary.csv"))?;
    if training
        .first()
        .is_none_or(|row| !row.contains_key("window_start"))
        || rebalances
            .first()
            .is_none_or(|row| !row.contains_key("period_start"))
    {
        return Ok(Vec::new());
    }
    let rebalances_by_start = rebalances
        .into_iter()
        .map(|row| (cell(&row, "period_start").to_owned(), row))
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::new();
    for signal_metric in TRAINING_SIGNAL_COLUMNS {
        if training
            .first()
            .is_none_or(|row| !row.contains_key(*signal_metric))
        {
            continue;
        }
        let mut aligned = Vec::<(f64, f64, f64)>::new();
        for train_row in &training {
            let period_start = cell(train_row, "window_start");
            let Some(rebalance_row) = rebalances_by_start.get(period_start) else {
                continue;
            };
            let Some(signal) = parse_f64_cell(train_row, signal_metric) else {
                continue;
            };
            let Some(ret) = parse_f64_cell(rebalance_row, "return") else {
                continue;
            };
            let Some(excess) = parse_f64_cell(rebalance_row, "excess_vs_benchmark") else {
                continue;
            };
            aligned.push((signal, ret, excess));
        }
        if aligned.len() < 9 {
            continue;
        }
        aligned.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        let thirds = aligned.len() / 3;
        let slices = [
            ("low", &aligned[..thirds]),
            ("mid", &aligned[thirds..2 * thirds]),
            ("high", &aligned[2 * thirds..]),
        ];
        for (bin_name, group) in slices {
            let mut row = JsonRow::new();
            insert_str(&mut row, "signal_metric", signal_metric);
            insert_str(&mut row, "bin", bin_name);
            insert_i64(&mut row, "window_count", group.len() as i64);
            insert_f64(
                &mut row,
                "signal_mean",
                mean(&group.iter().map(|item| item.0).collect::<Vec<_>>()),
            );
            insert_f64(
                &mut row,
                "rebalance_return_mean",
                mean(&group.iter().map(|item| item.1).collect::<Vec<_>>()),
            );
            insert_f64(
                &mut row,
                "rebalance_excess_mean",
                mean(&group.iter().map(|item| item.2).collect::<Vec<_>>()),
            );
            insert_i64(
                &mut row,
                "positive_rebalance_count",
                group.iter().filter(|item| item.1 > 0.0).count() as i64,
            );
            insert_f64(
                &mut row,
                "positive_rebalance_rate",
                group.iter().filter(|item| item.1 > 0.0).count() as f64 / group.len() as f64,
            );
            rows.push(row);
        }
    }
    Ok(rows)
}

fn summarize_validation_edges(rows: &[JsonRow]) -> serde_json::Map<String, JsonValue> {
    let mut grouped: BTreeMap<String, BTreeMap<String, &JsonRow>> = BTreeMap::new();
    for row in rows {
        grouped
            .entry(json_to_string(
                row.get("signal_metric").unwrap_or(&JsonValue::Null),
            ))
            .or_default()
            .insert(
                json_to_string(row.get("bin").unwrap_or(&JsonValue::Null)),
                row,
            );
    }
    let mut out = serde_json::Map::new();
    for (signal_metric, by_bin) in grouped {
        let (Some(high), Some(low)) = (by_bin.get("high"), by_bin.get("low")) else {
            continue;
        };
        out.insert(
            signal_metric,
            serde_json::json!({
                "high_low_return_spread": f64_from_json(high.get("rebalance_return_mean")) .unwrap_or(0.0)
                    - f64_from_json(low.get("rebalance_return_mean")).unwrap_or(0.0),
                "high_low_excess_spread": f64_from_json(high.get("rebalance_excess_mean")).unwrap_or(0.0)
                    - f64_from_json(low.get("rebalance_excess_mean")).unwrap_or(0.0),
                "high_positive_rebalance_rate": f64_from_json(high.get("positive_rebalance_rate")).unwrap_or(0.0),
                "low_positive_rebalance_rate": f64_from_json(low.get("positive_rebalance_rate")).unwrap_or(0.0),
            }),
        );
    }
    out
}

fn summarize_feature_families_for_run(run_dir: &Path) -> Result<Vec<JsonValue>, String> {
    let rows =
        summarize_feature_families(&BTreeMap::from([("run".to_owned(), run_dir.to_path_buf())]))?;
    Ok(rows
        .into_iter()
        .map(|mut row| {
            row.remove("run");
            row_to_json_object(row)
        })
        .map(JsonValue::Object)
        .collect())
}

fn summarize_risk_profile(run_dir: &Path) -> Result<JsonValue, String> {
    let daily = read_csv(&run_dir.join("native_daily_report.csv"))?;
    let mut by_year: BTreeMap<String, Vec<CsvRow>> = BTreeMap::new();
    for row in &daily {
        by_year
            .entry(cell(row, "datetime").chars().take(4).collect())
            .or_default()
            .push(row.clone());
    }
    let mut avg_risk_by_year = Vec::new();
    for (year, group) in by_year {
        let risk = numeric_column(&group, "risk_degree");
        let returns = numeric_column(&group, "return");
        avg_risk_by_year.push(serde_json::json!({
            "year": year,
            "avg_risk": mean(&risk),
            "min_risk": min_f64(&risk),
            "max_risk": max_f64(&risk),
            "negative_days": returns.iter().filter(|value| **value < 0.0).count(),
            "total_days": group.len(),
        }));
    }
    let values = numeric_column(&daily, "account_value");
    let dates = daily
        .iter()
        .map(|row| cell(row, "datetime").to_owned())
        .collect::<Vec<_>>();
    let mut peak = values.first().copied().unwrap_or(0.0);
    let mut peak_date = dates.first().cloned().unwrap_or_default();
    let mut max_drawdown = 0.0;
    let mut drawdown_start = peak_date.clone();
    let mut drawdown_trough = peak_date.clone();
    for (date, value) in dates.iter().zip(values.iter()) {
        if *value > peak {
            peak = *value;
            peak_date = date.clone();
        }
        let drawdown = if peak == 0.0 { 0.0 } else { value / peak - 1.0 };
        if drawdown < max_drawdown {
            max_drawdown = drawdown;
            drawdown_start = peak_date.clone();
            drawdown_trough = date.clone();
        }
    }
    Ok(serde_json::json!({
        "avg_risk_by_year": avg_risk_by_year,
        "max_drawdown_period": {
            "drawdown": max_drawdown,
            "peak_date": drawdown_start,
            "trough_date": drawdown_trough,
        },
    }))
}

fn bucket_shape(
    rows: &[CsvRow],
    top_bucket: i64,
    middle_buckets: &BTreeSet<i64>,
) -> Result<JsonRow, String> {
    let mut numeric = rows
        .iter()
        .filter_map(|row| {
            Some((
                parse_f64_cell(row, "bucket")? as i64,
                parse_f64_cell(row, "label_mean")?,
            ))
        })
        .collect::<Vec<_>>();
    numeric.sort_by_key(|(bucket, _)| *bucket);
    if numeric.is_empty() {
        return Err("Bucket report is empty after numeric conversion.".to_owned());
    }
    if !numeric.iter().any(|(bucket, _)| *bucket == top_bucket) {
        return Err(format!(
            "Top bucket {top_bucket} not found in bucket report."
        ));
    }
    let bottom_bucket = numeric
        .iter()
        .map(|(bucket, _)| *bucket)
        .max()
        .unwrap_or(top_bucket);
    let top_label = numeric
        .iter()
        .find(|(bucket, _)| *bucket == top_bucket)
        .map(|(_, label)| *label)
        .unwrap_or(0.0);
    let bottom_label = numeric
        .iter()
        .find(|(bucket, _)| *bucket == bottom_bucket)
        .map(|(_, label)| *label)
        .unwrap_or(0.0);
    let middle_values = numeric
        .iter()
        .filter(|(bucket, _)| middle_buckets.contains(bucket))
        .map(|(_, label)| *label)
        .collect::<Vec<_>>();
    let middle_mean = if middle_values.is_empty() {
        None
    } else {
        Some(mean(&middle_values))
    };
    let middle_best = if middle_values.is_empty() {
        None
    } else {
        Some(max_f64(&middle_values))
    };
    let best = numeric
        .iter()
        .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .copied()
        .unwrap();
    let worst = numeric
        .iter()
        .min_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        .copied()
        .unwrap();
    let mut ranked = numeric.clone();
    ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let top_rank = ranked
        .iter()
        .position(|(bucket, _)| *bucket == top_bucket)
        .map(|idx| idx as i64 + 1)
        .unwrap_or(0);
    let mut row = JsonRow::new();
    insert_i64(&mut row, "top_bucket", top_bucket);
    insert_i64(&mut row, "bottom_bucket", bottom_bucket);
    insert_f64(&mut row, "top_label_mean", top_label);
    insert_f64(&mut row, "bottom_label_mean", bottom_label);
    insert_f64(&mut row, "top_minus_bottom", top_label - bottom_label);
    insert_f64_or_null(&mut row, "middle_label_mean", middle_mean);
    insert_f64_or_null(&mut row, "middle_best_label_mean", middle_best);
    insert_f64_or_null(
        &mut row,
        "top_minus_middle_mean",
        middle_mean.map(|value| top_label - value),
    );
    insert_f64_or_null(
        &mut row,
        "top_minus_middle_best",
        middle_best.map(|value| top_label - value),
    );
    insert_i64(&mut row, "best_bucket", best.0);
    insert_f64(&mut row, "best_bucket_label_mean", best.1);
    insert_i64(&mut row, "worst_bucket", worst.0);
    insert_f64(&mut row, "worst_bucket_label_mean", worst.1);
    insert_i64(&mut row, "top_bucket_label_rank", top_rank);
    insert_f64_or_null(&mut row, "bucket_label_spearman", spearman(&numeric));
    Ok(row)
}

fn sync_outputs_to_candidate_root(
    output_dir: &Path,
    shortlist_root: &Path,
    candidate_dirs: &BTreeMap<String, PathBuf>,
    candidate_profiles: &[JsonValue],
    candidate_profile_rows: &[JsonRow],
) -> Result<(), String> {
    for filename in [
        "README.md",
        "portfolio_summary.csv",
        "yearly_path_summary.csv",
        "positive_month_concentration.csv",
        "bucket_shape_summary.csv",
        "yearly_bucket_shape_summary.csv",
        "feature_family_importance.csv",
        "validation_signal_bins.csv",
        "candidate_profiles.csv",
        "candidate_profiles.json",
    ] {
        let source = output_dir.join(filename);
        if source.exists() {
            fs::copy(&source, shortlist_root.join(filename)).map_err(|err| {
                format!(
                    "failed to copy {} -> {}: {err}",
                    source.display(),
                    shortlist_root.join(filename).display()
                )
            })?;
        }
    }
    let profiles_by_name = candidate_profiles
        .iter()
        .filter_map(|profile| {
            profile
                .get("name")
                .and_then(JsonValue::as_str)
                .map(|name| (name.to_owned(), profile))
        })
        .collect::<BTreeMap<_, _>>();
    for (candidate_name, candidate_dir) in candidate_dirs {
        let Some(profile) = profiles_by_name.get(candidate_name) else {
            continue;
        };
        write_json(&candidate_dir.join("candidate_profile.json"), profile)?;
        let rows = candidate_profile_rows
            .iter()
            .filter(|row| {
                row.get("name").and_then(JsonValue::as_str) == Some(candidate_name.as_str())
            })
            .cloned()
            .collect::<Vec<_>>();
        write_rows(&candidate_dir.join("candidate_profile.csv"), &rows)?;
    }
    Ok(())
}

struct CandidateReadmeInput<'a> {
    output_dir: &'a Path,
    runs: &'a BTreeMap<String, PathBuf>,
    portfolio: &'a [JsonRow],
    yearly: &'a [JsonRow],
    concentration: &'a [JsonRow],
    bucket_shape: &'a [JsonRow],
    yearly_bucket_shape: &'a [JsonRow],
    validation_bins: &'a [JsonRow],
    candidate_profiles: &'a [JsonRow],
}

fn write_readme(input: &CandidateReadmeInput<'_>) -> Result<(), String> {
    let output_dir = input.output_dir;
    let runs = input.runs;
    let portfolio = input.portfolio;
    let yearly = input.yearly;
    let concentration = input.concentration;
    let bucket_shape = input.bucket_shape;
    let yearly_bucket_shape = input.yearly_bucket_shape;
    let validation_bins = input.validation_bins;
    let candidate_profiles = input.candidate_profiles;
    let mut validation_display = validation_bins
        .iter()
        .filter(|row| {
            matches!(
                row.get("signal_metric").and_then(JsonValue::as_str),
                Some("valid_topk_positive_rate" | "valid_topk_excess_mean")
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    validation_display.sort_by(|left, right| {
        json_to_string(left.get("run").unwrap_or(&JsonValue::Null))
            .cmp(&json_to_string(
                right.get("run").unwrap_or(&JsonValue::Null),
            ))
            .then(
                json_to_string(left.get("signal_metric").unwrap_or(&JsonValue::Null)).cmp(
                    &json_to_string(right.get("signal_metric").unwrap_or(&JsonValue::Null)),
                ),
            )
            .then(bin_order(left).cmp(&bin_order(right)))
    });
    let mut weak_yearly = yearly_bucket_shape.to_vec();
    weak_yearly.sort_by(|left, right| {
        f64_from_json(left.get("top_minus_bottom"))
            .unwrap_or(f64::NAN)
            .partial_cmp(&f64_from_json(right.get("top_minus_bottom")).unwrap_or(f64::NAN))
            .unwrap_or(Ordering::Equal)
            .then(
                f64_from_json(left.get("bucket_label_spearman"))
                    .unwrap_or(f64::NAN)
                    .partial_cmp(
                        &f64_from_json(right.get("bucket_label_spearman")).unwrap_or(f64::NAN),
                    )
                    .unwrap_or(Ordering::Equal),
            )
    });
    let mut lines = vec![
        "# Candidate Pool Diagnostics".to_owned(),
        String::new(),
        "## Inputs".to_owned(),
        String::new(),
    ];
    for (name, path) in runs {
        lines.push(format!("- `{name}`: `{}`", path.display()));
    }
    lines.extend([
        String::new(),
        "## Portfolio Summary".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        portfolio,
        &[
            "run",
            "annualized_return",
            "sharpe_ratio",
            "max_drawdown",
            "excess_annualized_return",
            "excess_information_ratio",
            "monthly_win_rate",
        ],
        None,
    ));
    lines.extend([
        "".to_owned(),
        "## Same-Gate Reference Baseline Edges".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        portfolio,
        &[
            "run",
            "rank_avg_factor_baseline_excess_annualized_return",
            "rebalances_beating_rank_avg_factor_baseline_pct",
            "rank_ic_weighted_factor_baseline_excess_annualized_return",
            "rebalances_beating_rank_ic_weighted_factor_baseline_pct",
        ],
        None,
    ));
    lines.extend([
        "".to_owned(),
        "## Fixed-Risk Pure Reference Baseline Edges".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        portfolio,
        &[
            "run",
            "fixed_risk_rank_avg_factor_baseline_excess_annualized_return",
            "rebalances_beating_fixed_risk_rank_avg_factor_baseline_pct",
            "fixed_risk_rank_ic_weighted_factor_baseline_excess_annualized_return",
            "rebalances_beating_fixed_risk_rank_ic_weighted_factor_baseline_pct",
        ],
        None,
    ));
    lines.extend(["".to_owned(), "## Yearly Path".to_owned(), String::new()]);
    lines.extend(markdown_table(
        yearly,
        &[
            "run",
            "year",
            "compound_return",
            "benchmark_compound_return",
            "negative_months",
            "months_beating_benchmark",
            "worst_month_return",
        ],
        None,
    ));
    lines.extend(["".to_owned(), "## Concentration".to_owned(), String::new()]);
    lines.extend(markdown_table(
        concentration,
        &[
            "run",
            "positive_month_count",
            "top3_positive_share",
            "top5_positive_share",
            "top3_raw_sum_share",
            "top5_raw_sum_share",
        ],
        None,
    ));
    lines.extend(["".to_owned(), "## Bucket Shape".to_owned(), String::new()]);
    lines.extend(markdown_table(
        bucket_shape,
        &[
            "run",
            "top_minus_bottom",
            "top_minus_middle_best",
            "best_bucket",
            "top_bucket_label_rank",
            "bucket_label_spearman",
        ],
        None,
    ));
    lines.extend([
        "".to_owned(),
        "## Yearly Bucket Weak Spots".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        &weak_yearly,
        &[
            "run",
            "year",
            "top_minus_bottom",
            "top_minus_middle_best",
            "best_bucket",
            "top_bucket_label_rank",
            "bucket_label_spearman",
        ],
        Some(12),
    ));
    lines.extend([
        "".to_owned(),
        "## Validation Signal Bins".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        &validation_display,
        &[
            "run",
            "signal_metric",
            "bin",
            "window_count",
            "signal_mean",
            "rebalance_return_mean",
            "rebalance_excess_mean",
            "positive_rebalance_rate",
        ],
        None,
    ));
    lines.extend([
        "".to_owned(),
        "## Candidate Roles".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        candidate_profiles,
        &[
            "name",
            "candidate_role",
            "passed_gate_count",
            "total_gate_count",
            "failed_gates",
        ],
        None,
    ));
    lines.extend([
        String::new(),
        "## Reading".to_owned(),
        String::new(),
        "- Treat portfolio gains as real only when yearly path, concentration, bucket shape, and validation-bin behavior agree.".to_owned(),
        "- Negative top-minus-middle-best or weak yearly bucket shape means the score is useful as a selector but not yet calibrated enough for aggressive sizing.".to_owned(),
        "- Strong high-bin validation returns support using validation metrics as risk gates rather than fixed exposure.".to_owned(),
    ]);
    fs::write(
        output_dir.join("README.md"),
        lines.join("\n").trim().to_owned() + "\n",
    )
    .map_err(|err| format!("failed to write README.md: {err}"))
}

fn write_json(path: &Path, value: &JsonValue) -> Result<(), String> {
    write_json_pretty(path, value, true)
}

fn group_by_year(rows: &[CsvRow]) -> BTreeMap<String, Vec<CsvRow>> {
    let mut grouped = BTreeMap::new();
    for row in rows {
        grouped
            .entry(cell(row, "period").chars().take(4).collect())
            .or_insert_with(Vec::new)
            .push(row.clone());
    }
    grouped
}

fn group_by_key(rows: Vec<CsvRow>, key: &str) -> BTreeMap<String, Vec<CsvRow>> {
    let mut grouped = BTreeMap::new();
    for row in rows {
        grouped
            .entry(cell(&row, key).to_owned())
            .or_insert_with(Vec::new)
            .push(row);
    }
    grouped
}

fn numeric_column(rows: &[CsvRow], column: &str) -> Vec<f64> {
    rows.iter()
        .filter_map(|row| parse_f64_cell(row, column))
        .collect()
}

fn cell<'a>(row: &'a CsvRow, column: &str) -> &'a str {
    row.get(column).map(String::as_str).unwrap_or("")
}

fn parse_f64_cell(row: &CsvRow, column: &str) -> Option<f64> {
    cell(row, column)
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn metric_value(metrics: &JsonValue, key: &str) -> JsonValue {
    let value = metrics.get(key).cloned().unwrap_or(JsonValue::Null);
    if let Some(risk) = value.get("risk") {
        return risk.clone();
    }
    value
}

fn reference_baseline_summary_fields(prefix: &str) -> Vec<String> {
    [
        format!("{prefix}_annualized_return"),
        format!("{prefix}_annualized_volatility"),
        format!("{prefix}_sharpe_ratio"),
        format!("{prefix}_information_ratio"),
        format!("{prefix}_max_drawdown"),
        format!("{prefix}_monthly_win_rate"),
        format!("{prefix}_profitable_month_summary"),
        format!("{prefix}_rebalance_win_rate"),
        format!("{prefix}_profitable_rebalance_summary"),
        format!("{prefix}_excess_annualized_return"),
        format!("{prefix}_excess_information_ratio"),
        format!("months_beating_{prefix}_pct"),
        format!("months_beating_{prefix}_summary"),
        format!("rebalances_beating_{prefix}_pct"),
        format!("rebalances_beating_{prefix}_summary"),
    ]
    .to_vec()
}

fn feature_family(feature: &str) -> String {
    if feature.starts_with("TS_stock_vs_industry_") {
        "ts_stock_vs_industry".to_owned()
    } else if feature.starts_with("TS_industry_") {
        "ts_industry_state".to_owned()
    } else if feature.ends_with("_minus_industry") || feature.contains("_minus_industry_") {
        "ts_relative_value_quality".to_owned()
    } else if feature.starts_with("TS_dividend")
        || feature.starts_with("TS_bp")
        || feature.starts_with("TS_sp")
        || feature.starts_with("TS_ep")
    {
        "ts_valuation".to_owned()
    } else if feature.contains("amihud") || feature.contains("turnover") {
        "liquidity".to_owned()
    } else if feature.starts_with("LGBM_") {
        "lgbm".to_owned()
    } else if feature.starts_with("TEMP_") {
        "temporal".to_owned()
    } else if feature.starts_with("TECH_") {
        "technical".to_owned()
    } else {
        "other".to_owned()
    }
}

fn candidate_gate_baseline_prefixes(
    reference_baselines: &serde_json::Map<String, JsonValue>,
) -> Vec<String> {
    let pure = CANDIDATE_GATE_REFERENCE_BASELINE_PREFIXES
        .iter()
        .filter(|prefix| reference_baselines.contains_key(**prefix))
        .map(|prefix| (*prefix).to_owned())
        .collect::<Vec<_>>();
    if !pure.is_empty() {
        return pure;
    }
    CORE_REFERENCE_BASELINE_PREFIXES
        .iter()
        .filter(|prefix| reference_baselines.contains_key(**prefix))
        .map(|prefix| (*prefix).to_owned())
        .collect()
}

fn baseline_thresholds(prefix: &str, thresholds: &Thresholds) -> (f64, f64) {
    match prefix {
        "rank_avg_factor_baseline" => (
            thresholds.min_excess_annualized_vs_rank_avg_baseline,
            thresholds.min_rebalance_win_vs_rank_avg_baseline,
        ),
        "rank_ic_weighted_factor_baseline" => (
            thresholds.min_excess_annualized_vs_rank_ic_baseline,
            thresholds.min_rebalance_win_vs_rank_ic_baseline,
        ),
        "fixed_risk_rank_avg_factor_baseline" => (
            thresholds.min_excess_annualized_vs_fixed_risk_rank_avg_baseline,
            thresholds.min_rebalance_win_vs_fixed_risk_rank_avg_baseline,
        ),
        "fixed_risk_rank_ic_weighted_factor_baseline" => (
            thresholds.min_excess_annualized_vs_fixed_risk_rank_ic_baseline,
            thresholds.min_rebalance_win_vs_fixed_risk_rank_ic_baseline,
        ),
        _ => (0.0, 0.55),
    }
}

fn gate_passed(gates: &serde_json::Map<String, JsonValue>, name: &str) -> bool {
    gates
        .get(name)
        .and_then(|gate| gate.get("passed"))
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
}

fn row_to_json_object(row: JsonRow) -> serde_json::Map<String, JsonValue> {
    row.into_iter().collect()
}

fn value_path(value: &JsonValue, key: &str) -> JsonValue {
    value.get(key).cloned().unwrap_or(JsonValue::Null)
}

fn f64_path(value: &JsonValue, key: &str) -> f64 {
    f64_from_json(value.get(key)).unwrap_or(0.0)
}

fn i64_path(value: &JsonValue, key: &str) -> i64 {
    i64_from_json(value.get(key)).unwrap_or(999)
}

fn f64_from_json(value: Option<&JsonValue>) -> Option<f64> {
    match value? {
        JsonValue::Number(number) => number.as_f64(),
        JsonValue::String(text) => text.parse::<f64>().ok(),
        _ => None,
    }
}

fn i64_from_json(value: Option<&JsonValue>) -> Option<i64> {
    match value? {
        JsonValue::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|value| value as i64)),
        JsonValue::String(text) => text.parse::<i64>().ok(),
        _ => None,
    }
}

fn json_string_list(value: &JsonValue, path: &[&str]) -> Vec<String> {
    let mut current = value;
    for key in path {
        let Some(next) = current.get(*key) else {
            return Vec::new();
        };
        current = next;
    }
    current
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn insert_json_path(row: &mut JsonRow, key: &str, source: &JsonValue, path: &[&str]) {
    let mut current = source;
    for part in path {
        let Some(next) = current.get(*part) else {
            row.insert(key.to_owned(), JsonValue::Null);
            return;
        };
        current = next;
    }
    row.insert(key.to_owned(), current.clone());
}

fn insert_str(row: &mut JsonRow, key: &str, value: &str) {
    row.insert(key.to_owned(), JsonValue::String(value.to_owned()));
}

fn insert_i64(row: &mut JsonRow, key: &str, value: i64) {
    row.insert(key.to_owned(), JsonValue::from(value));
}

fn insert_f64(row: &mut JsonRow, key: &str, value: f64) {
    row.insert(key.to_owned(), json_f64(value));
}

fn insert_f64_or_null(row: &mut JsonRow, key: &str, value: Option<f64>) {
    row.insert(
        key.to_owned(),
        value.map(json_f64).unwrap_or(JsonValue::Null),
    );
}

fn json_f64(value: f64) -> JsonValue {
    serde_json::Number::from_f64(value)
        .map(JsonValue::Number)
        .unwrap_or(JsonValue::Null)
}

fn parse_bucket_ids(raw: &str) -> Result<BTreeSet<i64>, String> {
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<i64>()
                .map_err(|err| format!("invalid bucket id {part}: {err}"))
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if values.is_empty() {
        return Err("--middle-buckets must contain at least one bucket id.".to_owned());
    }
    Ok(values)
}

fn parse_i64(value: &str, option: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .map_err(|err| format!("invalid {option} {value}: {err}"))
}

fn compound_return(values: &[f64]) -> f64 {
    values.iter().fold(1.0, |acc, value| acc * (1.0 + value)) - 1.0
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn min_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::min).unwrap_or(f64::NAN)
}

fn max_f64(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(f64::NAN)
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 || !denominator.is_finite() {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn sum_top(values: &[(usize, f64)], n: usize) -> f64 {
    values.iter().take(n).map(|(_, value)| *value).sum()
}

fn spearman(values: &[(i64, f64)]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let x = values
        .iter()
        .map(|(bucket, _)| *bucket as f64)
        .collect::<Vec<_>>();
    let y = values.iter().map(|(_, label)| *label).collect::<Vec<_>>();
    pearson(&rank_values(&x), &rank_values(&y))
}

fn rank_values(values: &[f64]) -> Vec<f64> {
    let mut indexed = values.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
    let mut ranks = vec![0.0; values.len()];
    let mut idx = 0usize;
    while idx < indexed.len() {
        let start = idx;
        let value = indexed[idx].1;
        while idx < indexed.len() && indexed[idx].1 == value {
            idx += 1;
        }
        let rank = (start + 1 + idx) as f64 / 2.0;
        for item in &indexed[start..idx] {
            ranks[item.0] = rank;
        }
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut num = 0.0;
    let mut x_den = 0.0;
    let mut y_den = 0.0;
    for (left, right) in x.iter().zip(y.iter()) {
        num += (left - x_mean) * (right - y_mean);
        x_den += (left - x_mean).powi(2);
        y_den += (right - y_mean).powi(2);
    }
    let den = x_den.sqrt() * y_den.sqrt();
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}

fn markdown_table(rows: &[JsonRow], columns: &[&str], max_rows: Option<usize>) -> Vec<String> {
    let rows = max_rows.map(|n| &rows[..rows.len().min(n)]).unwrap_or(rows);
    if rows.is_empty() {
        return vec!["_No rows._".to_owned()];
    }
    let mut table = Vec::<Vec<String>>::new();
    table.push(columns.iter().map(|col| format!(" {col} ")).collect());
    for row in rows {
        table.push(
            columns
                .iter()
                .map(|column| {
                    format!(
                        " {} ",
                        fmt_value(row.get(*column).unwrap_or(&JsonValue::Null))
                    )
                })
                .collect(),
        );
    }
    let widths = (0..columns.len())
        .map(|idx| table.iter().map(|row| row[idx].len()).max().unwrap_or(0))
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    out.push(format!(
        "|{}|",
        table[0]
            .iter()
            .enumerate()
            .map(|(idx, value)| format!("{value:<width$}", width = widths[idx]))
            .collect::<Vec<_>>()
            .join("|")
    ));
    out.push(format!(
        "|{}|",
        widths
            .iter()
            .map(|width| "-".repeat(*width))
            .collect::<Vec<_>>()
            .join("|")
    ));
    for row in table.iter().skip(1) {
        out.push(format!(
            "|{}|",
            row.iter()
                .enumerate()
                .map(|(idx, value)| format!("{value:<width$}", width = widths[idx]))
                .collect::<Vec<_>>()
                .join("|")
        ));
    }
    out
}

fn fmt_value(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => String::new(),
        JsonValue::Number(number) => number
            .as_f64()
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| number.to_string()),
        JsonValue::String(text) => text.replace('|', "\\|"),
        JsonValue::Bool(value) => value.to_string(),
        other => other.to_string().replace('|', "\\|"),
    }
}

fn bin_order(row: &JsonRow) -> i32 {
    match row.get("bin").and_then(JsonValue::as_str) {
        Some("low") => 0,
        Some("mid") => 1,
        Some("high") => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn bucket_shape_ranks_top_bucket() {
        let rows = vec![
            row([("bucket", "1"), ("label_mean", "0.04")]),
            row([("bucket", "2"), ("label_mean", "0.02")]),
            row([("bucket", "3"), ("label_mean", "-0.01")]),
        ];
        let shape = bucket_shape(&rows, 1, &BTreeSet::from([2])).unwrap();
        assert_eq!(
            shape.get("best_bucket").and_then(JsonValue::as_i64),
            Some(1)
        );
        assert_eq!(
            shape
                .get("top_bucket_label_rank")
                .and_then(JsonValue::as_i64),
            Some(1)
        );
        assert!(
            shape
                .get("top_minus_bottom")
                .and_then(JsonValue::as_f64)
                .unwrap()
                > 0.04
        );
    }

    #[test]
    fn parses_run_and_bucket_options() {
        let options = parse_options(&[
            "--run=demo=/tmp/run".to_owned(),
            "--middle-buckets=2,3".to_owned(),
            "--top-bucket=1".to_owned(),
            "--no-sync-candidate-root".to_owned(),
        ])
        .unwrap();
        assert_eq!(options.runs, vec!["demo=/tmp/run"]);
        assert_eq!(options.middle_buckets, BTreeSet::from([2, 3]));
        assert!(options.no_sync_candidate_root);
    }

    #[test]
    fn candidate_pool_command_builds_profiles_and_syncs_candidate_root() {
        let root = temp_root("candidate_pool_sync");
        let shortlist_root = root.join("shortlist");
        let candidate_dir = shortlist_root.join("candidates").join("candidate_a");
        let snapshot = candidate_dir.join("snapshot");
        write_run_artifacts(&snapshot);
        let output_dir = root.join("diagnostics");

        run_candidate_pool_command(&[
            "--candidate-root".to_owned(),
            shortlist_root.to_string_lossy().into_owned(),
            "--output-dir".to_owned(),
            output_dir.to_string_lossy().into_owned(),
        ])
        .unwrap();

        assert!(output_dir.join("candidate_profiles.csv").exists());
        assert!(shortlist_root.join("portfolio_summary.csv").exists());
        assert!(candidate_dir.join("candidate_profile.csv").exists());
        let profile = read_json(&candidate_dir.join("candidate_profile.json")).unwrap();
        assert_eq!(
            profile.get("candidate_role").and_then(JsonValue::as_str),
            Some("ranker_and_sizer")
        );
        assert_eq!(
            profile
                .get("gate_summary")
                .and_then(|value| value.get("failed_gates"))
                .and_then(JsonValue::as_array)
                .map(Vec::len),
            Some(0)
        );
        let family = profile
            .get("feature_profile")
            .and_then(|value| value.get("family_importance"))
            .and_then(JsonValue::as_array)
            .and_then(|values| values.first())
            .and_then(JsonValue::as_object)
            .unwrap();
        assert!(!family.contains_key("run"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn summarize_yearly_compounds_excess_against_benchmark() {
        let root = temp_root("candidate_pool_yearly");
        let run_dir = root.join("run");
        fs::create_dir_all(&run_dir).unwrap();
        write_text(
            &run_dir.join("native_monthly_summary.csv"),
            "\
period,return,bench_return,excess_vs_benchmark,avg_turnover
2024-01,0.10,0.05,0.05,0.01
2024-02,0.10,0.00,0.10,0.01
",
        );
        let rows = summarize_yearly(&BTreeMap::from([("demo".to_owned(), run_dir)])).unwrap();
        let value = rows[0]
            .get("compound_excess_vs_benchmark")
            .and_then(JsonValue::as_f64)
            .unwrap();
        assert!((value - 0.16).abs() < 1e-12);
        fs::remove_dir_all(root).unwrap();
    }

    fn row<const N: usize>(items: [(&str, &str); N]) -> CsvRow {
        items
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect()
    }

    fn temp_root(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{name}_{}_{}", std::process::id(), stamp));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_run_artifacts(run_dir: &Path) {
        fs::create_dir_all(run_dir).unwrap();
        write_text(
            &run_dir.join("native_portfolio_metrics.json"),
            r#"{
  "annualized_return": {"risk": 0.42},
  "annualized_volatility": {"risk": 0.16},
  "sharpe_ratio": {"risk": 2.4},
  "max_drawdown": {"risk": -0.08},
  "profit_factor": {"risk": 1.7},
  "excess_annualized_return": {"risk": 0.41},
  "excess_information_ratio": {"risk": 1.8},
  "monthly_win_rate": {"risk": 0.75},
  "rebalance_win_rate": {"risk": 0.67},
  "turnover_mean": {"risk": 0.04},
  "fixed_risk_rank_avg_factor_baseline_excess_annualized_return": {"risk": 0.18},
  "fixed_risk_rank_avg_factor_baseline_excess_information_ratio": {"risk": 1.5},
  "rebalances_beating_fixed_risk_rank_avg_factor_baseline_pct": {"risk": 0.60},
  "rebalances_beating_fixed_risk_rank_avg_factor_baseline_summary": "7 / 12 (58.33%)",
  "fixed_risk_rank_ic_weighted_factor_baseline_excess_annualized_return": {"risk": 0.22},
  "fixed_risk_rank_ic_weighted_factor_baseline_excess_information_ratio": {"risk": 1.6},
  "rebalances_beating_fixed_risk_rank_ic_weighted_factor_baseline_pct": {"risk": 0.68},
  "rebalances_beating_fixed_risk_rank_ic_weighted_factor_baseline_summary": "8 / 12 (66.67%)"
}"#,
        );
        let monthly_rows = (1..=12)
            .map(|month| format!("2024-{month:02},0.02,0.00,0.02,0.04"))
            .collect::<Vec<_>>()
            .join("\n");
        write_text(
            &run_dir.join("native_monthly_summary.csv"),
            &format!(
                "period,return,bench_return,excess_vs_benchmark,avg_turnover\n{monthly_rows}\n"
            ),
        );
        write_text(
            &run_dir.join("native_score_bucket_report.csv"),
            "\
bucket,label_mean
1,0.04
2,0.03
3,0.02
4,0.01
5,-0.01
",
        );
        write_text(
            &run_dir.join("native_score_bucket_yearly_report.csv"),
            "\
year,bucket,label_mean
2024,1,0.04
2024,2,0.03
2024,3,0.02
2024,4,0.01
2024,5,-0.01
",
        );
        write_text(
            &run_dir.join("feature_importance_gain_mean.csv"),
            "\
feature,importance_gain
TS_industry_std_60,5.0
LGBM_ep_ttm,2.0
TS_dividend_yield_ttm,1.0
",
        );
        let training_rows = (0..9)
            .map(|idx| {
                let signal = idx as f64 / 10.0;
                format!(
                    "2024-01-{day:02},{signal},{signal},{signal},{signal},{signal}",
                    day = idx + 1
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        write_text(
            &run_dir.join("training_summary.csv"),
            &format!(
                "window_start,valid_topk_excess_mean,valid_topk_positive_rate,valid_topk_label_mean,valid_topk_min_label_mean,best_valid_daily_rank_ic\n{training_rows}\n"
            ),
        );
        let rebalance_rows = (0..9)
            .map(|idx| {
                let ret = if idx < 3 {
                    0.0
                } else if idx < 6 {
                    0.02
                } else {
                    0.06
                };
                format!("2024-01-{day:02},{ret},{ret}", day = idx + 1)
            })
            .collect::<Vec<_>>()
            .join("\n");
        write_text(
            &run_dir.join("native_rebalance_summary.csv"),
            &format!("period_start,return,excess_vs_benchmark\n{rebalance_rows}\n"),
        );
        write_text(
            &run_dir.join("native_daily_report.csv"),
            "\
datetime,return,risk_degree,account_value
2024-01-01,0.02,0.5,1.0
2024-01-02,-0.01,0.5,0.98
2024-01-03,0.03,0.7,1.03
",
        );
    }

    fn write_text(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, text).unwrap();
    }
}
