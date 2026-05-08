use ai4stock2_native::common::artifact::{
    json_to_csv_string as json_to_string, read_required_csv_rows as read_csv,
    read_required_json as read_json, write_json_rows_csv as write_rows, CsvRow, JsonRow,
};
use ai4stock2_native::common::cli::{next_arg, path_to_string, split_value};
use chrono::Local;
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

type BucketComparisonRows = (Vec<JsonRow>, Vec<JsonRow>, Vec<JsonRow>);

const METRIC_KEYS: &[&str] = &[
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "daily_win_rate",
    "monthly_win_rate",
    "profit_factor",
    "top_1_positive_month_share",
    "top_3_positive_month_share",
    "top_5_positive_month_share",
    "excess_annualized_return",
    "excess_information_ratio",
    "turnover_mean",
    "avg_factor_baseline_annualized_return",
    "avg_factor_baseline_information_ratio",
];

const TRAINING_COLUMNS: &[&str] = &[
    "best_valid_daily_ic",
    "best_valid_daily_rank_ic",
    "valid_top1_positive_rate",
    "valid_topk_positive_rate",
    "valid_topk_label_mean",
    "valid_topk_excess_mean",
    "best_iteration",
    "num_iterations",
];

const MONTHLY_KEEP_COLUMNS: &[&str] = &[
    "period",
    "return",
    "excess_vs_benchmark",
    "excess_vs_avg_factor_baseline",
    "max_drawdown",
    "win_rate",
    "profit_factor",
];

const DAILY_KEEP_COLUMNS: &[&str] = &[
    "datetime",
    "return",
    "bench",
    "risk_degree",
    "holdings",
    "turnover",
];

const TRACE_KEEP_COLUMNS: &[&str] = &[
    "datetime",
    "holdings_after",
    "net_return",
    "risk_degree",
    "buy_count",
    "sell_count",
];

#[derive(Debug, Clone)]
struct Options {
    candidate_run: PathBuf,
    baseline_run: PathBuf,
    candidate_name: String,
    baseline_name: String,
    output_dir: Option<PathBuf>,
    top_bucket: i64,
    middle_buckets: BTreeSet<i64>,
}

pub(crate) fn run_strategy_pair_command(args: &[String]) -> Result<(), String> {
    let options = parse_options(args)?;
    if !options.candidate_run.exists() {
        return Err(format!(
            "Candidate run directory not found: {}",
            options.candidate_run.display()
        ));
    }
    if !options.baseline_run.exists() {
        return Err(format!(
            "Baseline run directory not found: {}",
            options.baseline_run.display()
        ));
    }
    let output_dir = resolve_output_dir(&options);
    fs::create_dir_all(&output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;

    let run_dirs = BTreeMap::from([
        (
            options.candidate_name.clone(),
            options.candidate_run.clone(),
        ),
        (options.baseline_name.clone(), options.baseline_run.clone()),
    ]);
    let metrics = summarize_metrics(&run_dirs)?;
    let training = summarize_training(&run_dirs)?;
    let (buckets, bucket_shape, yearly_bucket_shape) =
        compare_buckets(&run_dirs, options.top_bucket, &options.middle_buckets)?;
    let (monthly_diff, yearly_monthly_diff) = compare_monthly(
        &options.candidate_name,
        &options.candidate_run,
        &options.baseline_name,
        &options.baseline_run,
    )?;
    let daily_diff = compare_daily(
        &options.candidate_name,
        &options.candidate_run,
        &options.baseline_name,
        &options.baseline_run,
    )?;
    let (feature_importance, family_importance) = compare_feature_importance(
        &options.candidate_name,
        &options.candidate_run,
        &options.baseline_name,
        &options.baseline_run,
    )?;
    let trace_overlap = compare_trace_holdings(
        &options.candidate_name,
        &options.candidate_run,
        &options.baseline_name,
        &options.baseline_run,
    )?;

    write_rows(&output_dir.join("portfolio_metrics_compare.csv"), &metrics)?;
    write_rows(&output_dir.join("training_metrics_compare.csv"), &training)?;
    write_rows(&output_dir.join("bucket_compare_long.csv"), &buckets)?;
    write_rows(&output_dir.join("bucket_shape_summary.csv"), &bucket_shape)?;
    write_rows(
        &output_dir.join("yearly_bucket_shape_summary.csv"),
        &yearly_bucket_shape,
    )?;
    write_rows(&output_dir.join("monthly_return_diff.csv"), &monthly_diff)?;
    write_rows(
        &output_dir.join("yearly_monthly_return_diff.csv"),
        &yearly_monthly_diff,
    )?;
    write_rows(&output_dir.join("daily_return_diff.csv"), &daily_diff)?;
    write_rows(
        &output_dir.join("feature_importance_diff.csv"),
        &feature_importance,
    )?;
    write_rows(
        &output_dir.join("feature_family_importance_diff.csv"),
        &family_importance,
    )?;
    write_rows(
        &output_dir.join("trace_holding_overlap.csv"),
        &trace_overlap,
    )?;
    write_rows(
        &output_dir.join("worst_trace_holding_diff.csv"),
        &take_sorted(&trace_overlap, "net_return_diff", 20, false),
    )?;
    write_rows(
        &output_dir.join("best_trace_holding_diff.csv"),
        &take_sorted(&trace_overlap, "net_return_diff", 20, true),
    )?;
    write_readme(&StrategyPairReadmeInput {
        output_dir: &output_dir,
        options: &options,
        metrics: &metrics,
        bucket_shape: &bucket_shape,
        monthly_diff: &monthly_diff,
        yearly_monthly_diff: &yearly_monthly_diff,
        family_importance: &family_importance,
        trace_overlap: &trace_overlap,
    })?;

    println!(
        "[+] Strategy-pair diagnostics saved to: {}",
        output_dir.display()
    );
    println!(
        "    portfolio metrics: {}",
        output_dir.join("portfolio_metrics_compare.csv").display()
    );
    println!(
        "    bucket shape: {}",
        output_dir.join("bucket_shape_summary.csv").display()
    );
    println!(
        "    monthly diff: {}",
        output_dir.join("monthly_return_diff.csv").display()
    );
    println!(
        "    holding overlap: {}",
        output_dir.join("trace_holding_overlap.csv").display()
    );
    println!("    README: {}", output_dir.join("README.md").display());
    Ok(())
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut candidate_run = None;
    let mut baseline_run = None;
    let mut candidate_name = "candidate".to_owned();
    let mut baseline_name = "baseline".to_owned();
    let mut output_dir = None;
    let mut top_bucket = 1;
    let mut middle_buckets = parse_bucket_ids("3,4,5,6,7")?;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--candidate-run" => {
                candidate_run = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--candidate-run",
                )?))
            }
            value if value.starts_with("--candidate-run=") => {
                candidate_run = Some(PathBuf::from(split_value(value, "--candidate-run")?))
            }
            "--baseline-run" => {
                baseline_run = Some(PathBuf::from(next_arg(args, &mut index, "--baseline-run")?))
            }
            value if value.starts_with("--baseline-run=") => {
                baseline_run = Some(PathBuf::from(split_value(value, "--baseline-run")?))
            }
            "--candidate-name" => candidate_name = next_arg(args, &mut index, "--candidate-name")?,
            value if value.starts_with("--candidate-name=") => {
                candidate_name = split_value(value, "--candidate-name")?
            }
            "--baseline-name" => baseline_name = next_arg(args, &mut index, "--baseline-name")?,
            value if value.starts_with("--baseline-name=") => {
                baseline_name = split_value(value, "--baseline-name")?
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_arg(args, &mut index, "--output-dir")?))
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?))
            }
            "--top-bucket" => {
                top_bucket =
                    parse_i64(&next_arg(args, &mut index, "--top-bucket")?, "--top-bucket")?
            }
            value if value.starts_with("--top-bucket=") => {
                top_bucket = parse_i64(&split_value(value, "--top-bucket")?, "--top-bucket")?
            }
            "--middle-buckets" => {
                middle_buckets = parse_bucket_ids(&next_arg(args, &mut index, "--middle-buckets")?)?
            }
            value if value.starts_with("--middle-buckets=") => {
                middle_buckets = parse_bucket_ids(&split_value(value, "--middle-buckets")?)?
            }
            other => {
                return Err(format!(
                    "unknown strategy-pair option: {other}\n\n{}",
                    usage()
                ))
            }
        }
        index += 1;
    }
    Ok(Options {
        candidate_run: candidate_run.ok_or_else(|| "--candidate-run is required".to_owned())?,
        baseline_run: baseline_run.ok_or_else(|| "--baseline-run is required".to_owned())?,
        candidate_name,
        baseline_name,
        output_dir,
        top_bucket,
        middle_buckets,
    })
}

fn usage() -> &'static str {
    "\
Usage:
  ai4stock-diagnostics strategy-pair --candidate-run DIR --baseline-run DIR [options]

Options:
  --candidate-name NAME
  --baseline-name NAME
  --output-dir DIR
  --top-bucket N
  --middle-buckets A,B,C
"
}

fn resolve_output_dir(options: &Options) -> PathBuf {
    if let Some(output_dir) = &options.output_dir {
        return output_dir.clone();
    }
    PathBuf::from("results")
        .join("diagnostics")
        .join("strategy_pair")
        .join(format!(
            "{}__{}_vs_{}",
            Local::now().format("%Y%m%d_%H%M%S"),
            options.candidate_name,
            options.baseline_name
        ))
}

fn summarize_metrics(run_dirs: &BTreeMap<String, PathBuf>) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in run_dirs {
        let metrics = read_json(&run_dir.join("native_portfolio_metrics.json"))?;
        let mut row = JsonRow::new();
        insert_str(&mut row, "run", run_name);
        insert_str(&mut row, "run_dir", &path_to_string(run_dir));
        for key in METRIC_KEYS {
            row.insert((*key).to_owned(), metric_value(&metrics, key));
        }
        for key in [
            "profitable_month_summary",
            "profitable_rebalance_summary",
            "months_beating_avg_factor_baseline_summary",
        ] {
            row.insert(
                key.to_owned(),
                metrics.get(key).cloned().unwrap_or(JsonValue::Null),
            );
        }
        rows.push(row);
    }
    Ok(rows)
}

fn summarize_training(run_dirs: &BTreeMap<String, PathBuf>) -> Result<Vec<JsonRow>, String> {
    let mut rows = Vec::new();
    for (run_name, run_dir) in run_dirs {
        let frame = read_csv(&run_dir.join("training_summary.csv"))?;
        let mut row = JsonRow::new();
        insert_str(&mut row, "run", run_name);
        insert_i64(&mut row, "window_count", frame.len() as i64);
        if has_column(&frame, "feature_count") {
            insert_f64_or_null(
                &mut row,
                "feature_count_median",
                median(numeric_column(&frame, "feature_count")),
            );
        }
        for column in TRAINING_COLUMNS {
            if !has_column(&frame, column) {
                continue;
            }
            let values = numeric_column(&frame, column);
            insert_f64_or_null(&mut row, &format!("{column}_mean"), mean(&values));
            insert_f64_or_null(
                &mut row,
                &format!("{column}_median"),
                median(values.clone()),
            );
            insert_f64_or_null(&mut row, &format!("{column}_min"), min_f64(&values));
            insert_f64_or_null(&mut row, &format!("{column}_max"), max_f64(&values));
        }
        rows.push(row);
    }
    Ok(rows)
}

fn compare_buckets(
    run_dirs: &BTreeMap<String, PathBuf>,
    top_bucket: i64,
    middle_buckets: &BTreeSet<i64>,
) -> Result<BucketComparisonRows, String> {
    let mut bucket_rows = Vec::new();
    let mut shape_rows = Vec::new();
    let mut yearly_shape_rows = Vec::new();
    for (run_name, run_dir) in run_dirs {
        let bucket = read_csv(&run_dir.join("native_score_bucket_report.csv"))?;
        for csv_row in &bucket {
            let mut row = csv_to_json_row(csv_row);
            insert_str(&mut row, "run", run_name);
            bucket_rows.push(row);
        }
        let mut shape = bucket_shape(&bucket, top_bucket, middle_buckets)?;
        insert_str(&mut shape, "run", run_name);
        shape_rows.push(shape);
        let yearly = read_csv(&run_dir.join("native_score_bucket_yearly_report.csv"))?;
        for (year, group) in group_by_key(yearly, "year") {
            let mut shape = bucket_shape(&group, top_bucket, middle_buckets)?;
            insert_str(&mut shape, "run", run_name);
            insert_str(&mut shape, "year", &year);
            yearly_shape_rows.push(shape);
        }
    }
    Ok((bucket_rows, shape_rows, yearly_shape_rows))
}

fn compare_monthly(
    candidate_name: &str,
    candidate_dir: &Path,
    baseline_name: &str,
    baseline_dir: &Path,
) -> Result<(Vec<JsonRow>, Vec<JsonRow>), String> {
    let candidate = read_csv(&candidate_dir.join("native_monthly_summary.csv"))?;
    let baseline = read_csv(&baseline_dir.join("native_monthly_summary.csv"))?;
    require_columns(
        &candidate,
        MONTHLY_KEEP_COLUMNS,
        "native_monthly_summary.csv",
    )?;
    require_columns(
        &baseline,
        MONTHLY_KEEP_COLUMNS,
        "native_monthly_summary.csv",
    )?;
    let baseline_by_period = baseline
        .iter()
        .map(|row| (cell(row, "period").to_owned(), row))
        .collect::<HashMap<_, _>>();
    let mut merged = Vec::new();
    for candidate_row in &candidate {
        let period = cell(candidate_row, "period");
        let Some(baseline_row) = baseline_by_period.get(period) else {
            continue;
        };
        let mut row = JsonRow::new();
        insert_str(&mut row, "period", period);
        for column in MONTHLY_KEEP_COLUMNS
            .iter()
            .filter(|column| **column != "period")
        {
            row.insert(
                format!("{candidate_name}_{column}"),
                csv_value(candidate_row, column),
            );
            row.insert(
                format!("{baseline_name}_{column}"),
                csv_value(baseline_row, column),
            );
        }
        insert_str(
            &mut row,
            "year",
            &period.chars().take(4).collect::<String>(),
        );
        let return_diff = parse_f64_cell(candidate_row, "return").unwrap_or(0.0)
            - parse_f64_cell(baseline_row, "return").unwrap_or(0.0);
        let bench_diff = parse_f64_cell(candidate_row, "excess_vs_benchmark").unwrap_or(0.0)
            - parse_f64_cell(baseline_row, "excess_vs_benchmark").unwrap_or(0.0);
        let factor_diff = parse_f64_cell(candidate_row, "excess_vs_avg_factor_baseline")
            .unwrap_or(0.0)
            - parse_f64_cell(baseline_row, "excess_vs_avg_factor_baseline").unwrap_or(0.0);
        insert_f64(&mut row, "return_diff", return_diff);
        insert_f64(&mut row, "excess_vs_benchmark_diff", bench_diff);
        insert_f64(&mut row, "excess_vs_avg_factor_diff", factor_diff);
        merged.push(row);
    }
    let mut by_year: BTreeMap<String, Vec<JsonRow>> = BTreeMap::new();
    for row in &merged {
        by_year
            .entry(json_to_string(row.get("year").unwrap_or(&JsonValue::Null)))
            .or_default()
            .push(row.clone());
    }
    let mut yearly = Vec::new();
    for (year, rows) in by_year {
        let candidate_returns = rows
            .iter()
            .filter_map(|row| f64_from_json(row.get(&format!("{candidate_name}_return"))))
            .collect::<Vec<_>>();
        let baseline_returns = rows
            .iter()
            .filter_map(|row| f64_from_json(row.get(&format!("{baseline_name}_return"))))
            .collect::<Vec<_>>();
        let diffs = rows
            .iter()
            .filter_map(|row| f64_from_json(row.get("return_diff")))
            .collect::<Vec<_>>();
        let mut row = JsonRow::new();
        insert_str(&mut row, "year", &year);
        insert_i64(&mut row, "month_count", rows.len() as i64);
        let candidate_compound = compound_return(&candidate_returns);
        let baseline_compound = compound_return(&baseline_returns);
        insert_f64(&mut row, "candidate_compound", candidate_compound);
        insert_f64(&mut row, "baseline_compound", baseline_compound);
        insert_f64(&mut row, "return_diff_sum", diffs.iter().sum());
        insert_f64_or_null(&mut row, "return_diff_mean", mean(&diffs));
        insert_f64_or_null(&mut row, "return_diff_median", median(diffs.clone()));
        insert_i64(
            &mut row,
            "candidate_win_months",
            diffs.iter().filter(|value| **value > 0.0).count() as i64,
        );
        insert_f64_or_null(&mut row, "worst_diff", min_f64(&diffs));
        insert_f64_or_null(&mut row, "best_diff", max_f64(&diffs));
        insert_f64(
            &mut row,
            "compound_diff",
            candidate_compound - baseline_compound,
        );
        yearly.push(row);
    }
    Ok((merged, yearly))
}

fn compare_daily(
    candidate_name: &str,
    candidate_dir: &Path,
    baseline_name: &str,
    baseline_dir: &Path,
) -> Result<Vec<JsonRow>, String> {
    let candidate = read_csv(&candidate_dir.join("native_daily_report.csv"))?;
    let baseline = read_csv(&baseline_dir.join("native_daily_report.csv"))?;
    require_columns(&candidate, DAILY_KEEP_COLUMNS, "native_daily_report.csv")?;
    require_columns(&baseline, DAILY_KEEP_COLUMNS, "native_daily_report.csv")?;
    let baseline_by_date = baseline
        .iter()
        .map(|row| (cell(row, "datetime").to_owned(), row))
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::new();
    for candidate_row in &candidate {
        let datetime = cell(candidate_row, "datetime");
        let Some(baseline_row) = baseline_by_date.get(datetime) else {
            continue;
        };
        let mut row = JsonRow::new();
        insert_str(&mut row, "datetime", datetime);
        for column in DAILY_KEEP_COLUMNS
            .iter()
            .filter(|column| **column != "datetime")
        {
            row.insert(
                format!("{candidate_name}_{column}"),
                csv_value(candidate_row, column),
            );
            row.insert(
                format!("{baseline_name}_{column}"),
                csv_value(baseline_row, column),
            );
        }
        insert_f64(
            &mut row,
            "return_diff",
            parse_f64_cell(candidate_row, "return").unwrap_or(0.0)
                - parse_f64_cell(baseline_row, "return").unwrap_or(0.0),
        );
        rows.push(row);
    }
    Ok(rows)
}

fn compare_feature_importance(
    candidate_name: &str,
    candidate_dir: &Path,
    baseline_name: &str,
    baseline_dir: &Path,
) -> Result<(Vec<JsonRow>, Vec<JsonRow>), String> {
    let candidate = read_csv(&candidate_dir.join("feature_importance_gain_mean.csv"))?;
    let baseline = read_csv(&baseline_dir.join("feature_importance_gain_mean.csv"))?;
    require_columns(
        &candidate,
        &["feature", "importance_gain"],
        "feature_importance_gain_mean.csv",
    )?;
    require_columns(
        &baseline,
        &["feature", "importance_gain"],
        "feature_importance_gain_mean.csv",
    )?;
    let candidate_ranks = importance_ranks(&candidate);
    let baseline_ranks = importance_ranks(&baseline);
    let candidate_by_feature = candidate
        .iter()
        .map(|row| (cell(row, "feature").to_owned(), row))
        .collect::<BTreeMap<_, _>>();
    let baseline_by_feature = baseline
        .iter()
        .map(|row| (cell(row, "feature").to_owned(), row))
        .collect::<BTreeMap<_, _>>();
    let features = candidate_by_feature
        .keys()
        .chain(baseline_by_feature.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut rows = Vec::new();
    let mut by_family: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    for feature in features {
        let candidate_gain = candidate_by_feature
            .get(&feature)
            .and_then(|row| parse_f64_cell(row, "importance_gain"))
            .unwrap_or(0.0);
        let baseline_gain = baseline_by_feature
            .get(&feature)
            .and_then(|row| parse_f64_cell(row, "importance_gain"))
            .unwrap_or(0.0);
        let family = feature_family(&feature);
        let mut row = JsonRow::new();
        insert_str(&mut row, "feature", &feature);
        insert_f64(
            &mut row,
            &format!("{candidate_name}_importance_gain"),
            candidate_gain,
        );
        insert_f64(
            &mut row,
            &format!("{baseline_name}_importance_gain"),
            baseline_gain,
        );
        insert_f64_or_null(
            &mut row,
            &format!("{candidate_name}_rank"),
            candidate_ranks.get(&feature).copied(),
        );
        insert_f64_or_null(
            &mut row,
            &format!("{baseline_name}_rank"),
            baseline_ranks.get(&feature).copied(),
        );
        insert_f64(&mut row, "importance_diff", candidate_gain - baseline_gain);
        insert_str(&mut row, "feature_family", &family);
        let entry = by_family.entry(family).or_insert((0.0, 0.0));
        entry.0 += candidate_gain;
        entry.1 += baseline_gain;
        rows.push(row);
    }
    rows.sort_by(|left, right| {
        f64_from_json(right.get("importance_diff"))
            .unwrap_or(0.0)
            .partial_cmp(&f64_from_json(left.get("importance_diff")).unwrap_or(0.0))
            .unwrap_or(Ordering::Equal)
    });
    let candidate_total = by_family.values().map(|(value, _)| *value).sum::<f64>();
    let baseline_total = by_family.values().map(|(_, value)| *value).sum::<f64>();
    let mut family_rows = Vec::new();
    for (family, (candidate_gain, baseline_gain)) in by_family {
        let candidate_share = ratio(candidate_gain, candidate_total);
        let baseline_share = ratio(baseline_gain, baseline_total);
        let mut row = JsonRow::new();
        insert_str(&mut row, "feature_family", &family);
        insert_f64(
            &mut row,
            &format!("{candidate_name}_importance_gain"),
            candidate_gain,
        );
        insert_f64(
            &mut row,
            &format!("{baseline_name}_importance_gain"),
            baseline_gain,
        );
        insert_f64(&mut row, "importance_diff", candidate_gain - baseline_gain);
        insert_f64_or_null(
            &mut row,
            &format!("{candidate_name}_share"),
            candidate_share,
        );
        insert_f64_or_null(&mut row, &format!("{baseline_name}_share"), baseline_share);
        insert_f64_or_null(
            &mut row,
            "share_diff",
            candidate_share
                .zip(baseline_share)
                .map(|(left, right)| left - right),
        );
        family_rows.push(row);
    }
    family_rows.sort_by(|left, right| {
        f64_from_json(right.get("share_diff"))
            .unwrap_or(0.0)
            .partial_cmp(&f64_from_json(left.get("share_diff")).unwrap_or(0.0))
            .unwrap_or(Ordering::Equal)
    });
    Ok((rows, family_rows))
}

fn compare_trace_holdings(
    candidate_name: &str,
    candidate_dir: &Path,
    baseline_name: &str,
    baseline_dir: &Path,
) -> Result<Vec<JsonRow>, String> {
    let candidate = read_csv(&candidate_dir.join("native_backtest_trace.csv"))?;
    let baseline = read_csv(&baseline_dir.join("native_backtest_trace.csv"))?;
    require_columns(&candidate, TRACE_KEEP_COLUMNS, "native_backtest_trace.csv")?;
    require_columns(&baseline, TRACE_KEEP_COLUMNS, "native_backtest_trace.csv")?;
    let baseline_by_date = baseline
        .iter()
        .map(|row| (cell(row, "datetime").to_owned(), row))
        .collect::<HashMap<_, _>>();
    let mut rows = Vec::new();
    for candidate_row in &candidate {
        let datetime = cell(candidate_row, "datetime");
        let Some(baseline_row) = baseline_by_date.get(datetime) else {
            continue;
        };
        let candidate_holdings = parse_mapping(cell(candidate_row, "holdings_after"));
        let baseline_holdings = parse_mapping(cell(baseline_row, "holdings_after"));
        let candidate_symbols = candidate_holdings.keys().cloned().collect::<BTreeSet<_>>();
        let baseline_symbols = baseline_holdings.keys().cloned().collect::<BTreeSet<_>>();
        let union = candidate_symbols
            .union(&baseline_symbols)
            .cloned()
            .collect::<BTreeSet<_>>();
        let overlap = candidate_symbols
            .intersection(&baseline_symbols)
            .cloned()
            .collect::<BTreeSet<_>>();
        let candidate_only = candidate_symbols
            .difference(&baseline_symbols)
            .cloned()
            .collect::<BTreeSet<_>>();
        let baseline_only = baseline_symbols
            .difference(&candidate_symbols)
            .cloned()
            .collect::<BTreeSet<_>>();
        let candidate_net = parse_f64_cell(candidate_row, "net_return").unwrap_or(0.0);
        let baseline_net = parse_f64_cell(baseline_row, "net_return").unwrap_or(0.0);
        let mut row = JsonRow::new();
        insert_str(&mut row, "datetime", datetime);
        insert_f64(
            &mut row,
            &format!("{candidate_name}_net_return"),
            candidate_net,
        );
        insert_f64(
            &mut row,
            &format!("{baseline_name}_net_return"),
            baseline_net,
        );
        insert_f64(&mut row, "net_return_diff", candidate_net - baseline_net);
        insert_f64_or_null(
            &mut row,
            &format!("{candidate_name}_risk_degree"),
            parse_f64_cell(candidate_row, "risk_degree"),
        );
        insert_f64_or_null(
            &mut row,
            &format!("{baseline_name}_risk_degree"),
            parse_f64_cell(baseline_row, "risk_degree"),
        );
        insert_f64_or_null(
            &mut row,
            &format!("{candidate_name}_buy_count"),
            parse_f64_cell(candidate_row, "buy_count"),
        );
        insert_f64_or_null(
            &mut row,
            &format!("{baseline_name}_buy_count"),
            parse_f64_cell(baseline_row, "buy_count"),
        );
        insert_f64_or_null(
            &mut row,
            &format!("{candidate_name}_sell_count"),
            parse_f64_cell(candidate_row, "sell_count"),
        );
        insert_f64_or_null(
            &mut row,
            &format!("{baseline_name}_sell_count"),
            parse_f64_cell(baseline_row, "sell_count"),
        );
        insert_i64(
            &mut row,
            "candidate_holding_count",
            candidate_symbols.len() as i64,
        );
        insert_i64(
            &mut row,
            "baseline_holding_count",
            baseline_symbols.len() as i64,
        );
        insert_i64(&mut row, "overlap_count", overlap.len() as i64);
        insert_i64(
            &mut row,
            "candidate_only_count",
            candidate_only.len() as i64,
        );
        insert_i64(&mut row, "baseline_only_count", baseline_only.len() as i64);
        insert_f64(
            &mut row,
            "holding_jaccard",
            if union.is_empty() {
                1.0
            } else {
                overlap.len() as f64 / union.len() as f64
            },
        );
        insert_str(
            &mut row,
            "candidate_only_symbols",
            &candidate_only.into_iter().collect::<Vec<_>>().join("|"),
        );
        insert_str(
            &mut row,
            "baseline_only_symbols",
            &baseline_only.into_iter().collect::<Vec<_>>().join("|"),
        );
        rows.push(row);
    }
    Ok(rows)
}

struct StrategyPairReadmeInput<'a> {
    output_dir: &'a Path,
    options: &'a Options,
    metrics: &'a [JsonRow],
    bucket_shape: &'a [JsonRow],
    monthly_diff: &'a [JsonRow],
    yearly_monthly_diff: &'a [JsonRow],
    family_importance: &'a [JsonRow],
    trace_overlap: &'a [JsonRow],
}

fn write_readme(input: &StrategyPairReadmeInput<'_>) -> Result<(), String> {
    let output_dir = input.output_dir;
    let options = input.options;
    let metrics = input.metrics;
    let bucket_shape = input.bucket_shape;
    let monthly_diff = input.monthly_diff;
    let yearly_monthly_diff = input.yearly_monthly_diff;
    let family_importance = input.family_importance;
    let trace_overlap = input.trace_overlap;
    let candidate_row = row_for_run(metrics, &options.candidate_name)
        .ok_or_else(|| format!("missing candidate metrics row: {}", options.candidate_name))?;
    let baseline_row = row_for_run(metrics, &options.baseline_name)
        .ok_or_else(|| format!("missing baseline metrics row: {}", options.baseline_name))?;
    let candidate_bucket = row_for_run(bucket_shape, &options.candidate_name)
        .ok_or_else(|| format!("missing candidate bucket row: {}", options.candidate_name))?;
    let baseline_bucket = row_for_run(bucket_shape, &options.baseline_name)
        .ok_or_else(|| format!("missing baseline bucket row: {}", options.baseline_name))?;
    let diffs = monthly_diff
        .iter()
        .filter_map(|row| f64_from_json(row.get("return_diff")))
        .collect::<Vec<_>>();
    let candidate_beats = diffs.iter().filter(|value| **value > 0.0).count();
    let worst = take_sorted(monthly_diff, "return_diff", 5, false);
    let best = take_sorted(monthly_diff, "return_diff", 5, true);
    let jaccard = trace_overlap
        .iter()
        .filter_map(|row| f64_from_json(row.get("holding_jaccard")))
        .collect::<Vec<_>>();
    let mut lines = vec![
        format!("# {} vs {}", options.candidate_name, options.baseline_name),
        String::new(),
        "## Inputs".to_owned(),
        String::new(),
        format!("- candidate_run: `{}`", options.candidate_run.display()),
        format!("- baseline_run: `{}`", options.baseline_run.display()),
        String::new(),
        "## Portfolio Summary".to_owned(),
        String::new(),
        format!(
            "- candidate annualized_return: `{}`",
            fmt_metric(candidate_row.get("annualized_return"))
        ),
        format!(
            "- baseline annualized_return: `{}`",
            fmt_metric(baseline_row.get("annualized_return"))
        ),
        format!(
            "- candidate sharpe: `{}`",
            fmt_metric(candidate_row.get("sharpe_ratio"))
        ),
        format!(
            "- baseline sharpe: `{}`",
            fmt_metric(baseline_row.get("sharpe_ratio"))
        ),
        format!(
            "- candidate max_drawdown: `{}`",
            fmt_metric(candidate_row.get("max_drawdown"))
        ),
        format!(
            "- baseline max_drawdown: `{}`",
            fmt_metric(baseline_row.get("max_drawdown"))
        ),
        String::new(),
        "## Bucket Shape".to_owned(),
        String::new(),
        format!(
            "- candidate top_minus_bottom: `{}`",
            fmt_metric(candidate_bucket.get("top_minus_bottom"))
        ),
        format!(
            "- baseline top_minus_bottom: `{}`",
            fmt_metric(baseline_bucket.get("top_minus_bottom"))
        ),
        format!(
            "- candidate top_minus_middle_best: `{}`",
            fmt_metric(candidate_bucket.get("top_minus_middle_best"))
        ),
        format!(
            "- baseline top_minus_middle_best: `{}`",
            fmt_metric(baseline_bucket.get("top_minus_middle_best"))
        ),
        format!(
            "- candidate best_bucket: `{}`",
            fmt_metric(candidate_bucket.get("best_bucket"))
        ),
        format!(
            "- baseline best_bucket: `{}`",
            fmt_metric(baseline_bucket.get("best_bucket"))
        ),
        String::new(),
        "## Monthly Difference".to_owned(),
        String::new(),
        format!(
            "- candidate beats baseline months: `{candidate_beats} / {}`",
            monthly_diff.len()
        ),
        format!("- mean monthly return_diff: `{}`", fmt_option(mean(&diffs))),
        format!(
            "- median monthly return_diff: `{}`",
            fmt_option(median(diffs.clone()))
        ),
        String::new(),
        "## Worst Candidate Relative Months".to_owned(),
        String::new(),
    ];
    lines.extend(worst.iter().map(|row| {
        format!(
            "- `{}`: `{}`",
            json_to_string(row.get("period").unwrap_or(&JsonValue::Null)),
            fmt_metric(row.get("return_diff"))
        )
    }));
    lines.extend([
        String::new(),
        "## Best Candidate Relative Months".to_owned(),
        String::new(),
    ]);
    lines.extend(best.iter().map(|row| {
        format!(
            "- `{}`: `{}`",
            json_to_string(row.get("period").unwrap_or(&JsonValue::Null)),
            fmt_metric(row.get("return_diff"))
        )
    }));
    lines.extend([
        String::new(),
        "## Yearly Monthly Difference".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        yearly_monthly_diff,
        &[
            "year",
            "month_count",
            "candidate_compound",
            "baseline_compound",
            "return_diff_mean",
            "candidate_win_months",
            "compound_diff",
        ],
    ));
    lines.extend([
        String::new(),
        "## Feature Family Importance".to_owned(),
        String::new(),
    ]);
    lines.extend(markdown_table(
        family_importance,
        &["feature_family", "importance_diff", "share_diff"],
    ));
    lines.extend([
        String::new(),
        "## Holding Overlap".to_owned(),
        String::new(),
        format!("- mean holding_jaccard: `{}`", fmt_option(mean(&jaccard))),
        format!(
            "- median holding_jaccard: `{}`",
            fmt_option(median(jaccard.clone()))
        ),
    ]);
    fs::write(
        output_dir.join("README.md"),
        lines.join("\n").trim().to_owned() + "\n",
    )
    .map_err(|err| format!("failed to write README.md: {err}"))
}

fn row_for_run<'a>(rows: &'a [JsonRow], run_name: &str) -> Option<&'a JsonRow> {
    rows.iter()
        .find(|row| row.get("run").and_then(JsonValue::as_str) == Some(run_name))
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
    let middle_mean = mean(&middle_values);
    let middle_best = max_f64(&middle_values);
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

fn require_columns(rows: &[CsvRow], columns: &[&str], artifact: &str) -> Result<(), String> {
    let Some(first) = rows.first() else {
        return Ok(());
    };
    for column in columns {
        if !first.contains_key(*column) {
            return Err(format!("{artifact} missing required column: {column}"));
        }
    }
    Ok(())
}

fn has_column(rows: &[CsvRow], column: &str) -> bool {
    rows.first()
        .map(|row| row.contains_key(column))
        .unwrap_or(false)
}

fn csv_to_json_row(row: &CsvRow) -> JsonRow {
    row.iter()
        .map(|(key, value)| (key.clone(), parse_json_scalar(value)))
        .collect()
}

fn csv_value(row: &CsvRow, column: &str) -> JsonValue {
    parse_json_scalar(cell(row, column))
}

fn parse_json_scalar(value: &str) -> JsonValue {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        JsonValue::Null
    } else if let Ok(value) = trimmed.parse::<f64>() {
        json_f64(value)
    } else {
        JsonValue::String(trimmed.to_owned())
    }
}

fn metric_value(metrics: &JsonValue, key: &str) -> JsonValue {
    let value = metrics.get(key).cloned().unwrap_or(JsonValue::Null);
    if let Some(risk) = value.get("risk") {
        return risk.clone();
    }
    value
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

fn importance_ranks(rows: &[CsvRow]) -> BTreeMap<String, f64> {
    let mut items = rows
        .iter()
        .map(|row| {
            (
                cell(row, "feature").to_owned(),
                parse_f64_cell(row, "importance_gain").unwrap_or(0.0),
            )
        })
        .collect::<Vec<_>>();
    items.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    let mut ranks = BTreeMap::new();
    let mut index = 0usize;
    while index < items.len() {
        let rank = index as f64 + 1.0;
        let value = items[index].1;
        while index < items.len() && items[index].1 == value {
            ranks.insert(items[index].0.clone(), rank);
            index += 1;
        }
    }
    ranks
}

fn parse_mapping(raw: &str) -> BTreeMap<String, f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return BTreeMap::new();
    }
    if let Ok(JsonValue::Object(map)) = serde_json::from_str::<JsonValue>(trimmed) {
        return map
            .into_iter()
            .filter_map(|(key, value)| value.as_f64().map(|number| (key, number)))
            .collect();
    }
    let body = trimmed
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .unwrap_or(trimmed);
    let mut out = BTreeMap::new();
    for part in body.split(',') {
        let Some((key, value)) = part.split_once(':') else {
            continue;
        };
        let key = key.trim().trim_matches('"').trim_matches('\'').to_owned();
        let Some(value) = value.trim().parse::<f64>().ok() else {
            continue;
        };
        if !key.is_empty() {
            out.insert(key, value);
        }
    }
    out
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

fn take_sorted(rows: &[JsonRow], key: &str, n: usize, descending: bool) -> Vec<JsonRow> {
    let mut sorted = rows.to_vec();
    sorted.sort_by(|left, right| {
        let left_value = f64_from_json(left.get(key)).unwrap_or(0.0);
        let right_value = f64_from_json(right.get(key)).unwrap_or(0.0);
        if descending {
            right_value
                .partial_cmp(&left_value)
                .unwrap_or(Ordering::Equal)
        } else {
            left_value
                .partial_cmp(&right_value)
                .unwrap_or(Ordering::Equal)
        }
    });
    sorted.truncate(n);
    sorted
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

fn numeric_column(rows: &[CsvRow], column: &str) -> Vec<f64> {
    rows.iter()
        .filter_map(|row| parse_f64_cell(row, column))
        .collect()
}

fn f64_from_json(value: Option<&JsonValue>) -> Option<f64> {
    match value? {
        JsonValue::Number(number) => number.as_f64(),
        JsonValue::String(text) => text.parse::<f64>().ok(),
        _ => None,
    }
}

fn compound_return(values: &[f64]) -> f64 {
    values.iter().fold(1.0, |acc, value| acc * (1.0 + value)) - 1.0
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

fn min_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::min)
}

fn max_f64(values: &[f64]) -> Option<f64> {
    values.iter().copied().reduce(f64::max)
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 || !denominator.is_finite() {
        None
    } else {
        Some(numerator / denominator)
    }
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
    let x_mean = mean(x)?;
    let y_mean = mean(y)?;
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

fn fmt_metric(value: Option<&JsonValue>) -> String {
    f64_from_json(value)
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| value.map(json_to_string).unwrap_or_default())
}

fn fmt_option(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}

fn markdown_table(rows: &[JsonRow], columns: &[&str]) -> Vec<String> {
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
                        fmt_table_value(row.get(*column).unwrap_or(&JsonValue::Null))
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

fn fmt_table_value(value: &JsonValue) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parses_strategy_pair_options() {
        let options = parse_options(&[
            "--candidate-run=/tmp/candidate".to_owned(),
            "--baseline-run".to_owned(),
            "/tmp/baseline".to_owned(),
            "--candidate-name".to_owned(),
            "cand".to_owned(),
            "--middle-buckets=2,3".to_owned(),
        ])
        .unwrap();
        assert_eq!(options.candidate_name, "cand");
        assert_eq!(options.baseline_run, PathBuf::from("/tmp/baseline"));
        assert_eq!(options.middle_buckets, BTreeSet::from([2, 3]));
    }

    #[test]
    fn strategy_pair_command_writes_core_outputs() {
        let root = temp_root("strategy_pair");
        let candidate = root.join("candidate");
        let baseline = root.join("baseline");
        write_run_artifacts(&candidate, 0.03, 3.0, "AAA", "BBB");
        write_run_artifacts(&baseline, 0.01, 1.0, "AAA", "CCC");
        let output_dir = root.join("diagnostics");

        run_strategy_pair_command(&[
            "--candidate-run".to_owned(),
            candidate.to_string_lossy().into_owned(),
            "--baseline-run".to_owned(),
            baseline.to_string_lossy().into_owned(),
            "--candidate-name".to_owned(),
            "cand".to_owned(),
            "--baseline-name".to_owned(),
            "base".to_owned(),
            "--output-dir".to_owned(),
            output_dir.to_string_lossy().into_owned(),
        ])
        .unwrap();

        assert!(output_dir.join("portfolio_metrics_compare.csv").exists());
        assert!(output_dir.join("trace_holding_overlap.csv").exists());
        assert!(output_dir.join("README.md").exists());
        let monthly = read_csv(&output_dir.join("monthly_return_diff.csv")).unwrap();
        assert_eq!(monthly.len(), 2);
        assert!(parse_f64_cell(&monthly[0], "return_diff").unwrap() > 0.0);
        let trace = read_csv(&output_dir.join("trace_holding_overlap.csv")).unwrap();
        assert_eq!(
            parse_f64_cell(&trace[0], "holding_jaccard"),
            Some(1.0 / 3.0)
        );

        fs::remove_dir_all(root).unwrap();
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

    fn write_run_artifacts(run_dir: &Path, ret: f64, importance: f64, shared: &str, unique: &str) {
        fs::create_dir_all(run_dir).unwrap();
        write_text(
            &run_dir.join("native_portfolio_metrics.json"),
            &format!(
                r#"{{
  "annualized_return": {{"risk": {annualized}}},
  "annualized_volatility": {{"risk": 0.12}},
  "sharpe_ratio": {{"risk": 2.0}},
  "max_drawdown": {{"risk": -0.05}},
  "daily_win_rate": {{"risk": 0.55}},
  "monthly_win_rate": {{"risk": 0.60}},
  "profit_factor": {{"risk": 1.5}},
  "excess_annualized_return": {{"risk": 0.20}},
  "excess_information_ratio": {{"risk": 1.2}},
  "turnover_mean": {{"risk": 0.04}},
  "avg_factor_baseline_annualized_return": {{"risk": 0.10}},
  "avg_factor_baseline_information_ratio": {{"risk": 0.9}}
}}"#,
                annualized = ret * 12.0
            ),
        );
        write_text(
            &run_dir.join("training_summary.csv"),
            "window_start,feature_count,best_valid_daily_ic,best_valid_daily_rank_ic,valid_topk_positive_rate,valid_topk_excess_mean,best_iteration,num_iterations\n2024-01-01,10,0.01,0.02,0.6,0.01,50,100\n2024-02-01,12,0.02,0.03,0.7,0.02,60,100\n",
        );
        write_text(
            &run_dir.join("native_score_bucket_report.csv"),
            "bucket,label_mean\n1,0.04\n2,0.02\n3,-0.01\n",
        );
        write_text(
            &run_dir.join("native_score_bucket_yearly_report.csv"),
            "year,bucket,label_mean\n2024,1,0.04\n2024,2,0.02\n2024,3,-0.01\n",
        );
        write_text(
            &run_dir.join("native_monthly_summary.csv"),
            &format!(
                "period,return,excess_vs_benchmark,excess_vs_avg_factor_baseline,max_drawdown,win_rate,profit_factor\n2024-01,{ret},0.01,0.01,-0.01,0.6,1.2\n2024-02,{ret},0.01,0.01,-0.01,0.6,1.2\n"
            ),
        );
        write_text(
            &run_dir.join("native_daily_report.csv"),
            &format!(
                "datetime,return,bench,risk_degree,holdings,turnover\n2024-01-01,{ret},0.0,0.5,2,0.1\n2024-01-02,{ret},0.0,0.5,2,0.1\n"
            ),
        );
        write_text(
            &run_dir.join("feature_importance_gain_mean.csv"),
            &format!("feature,importance_gain\nTS_industry_std_60,{importance}\nLGBM_ep_ttm,1.0\n"),
        );
        write_text(
            &run_dir.join("native_backtest_trace.csv"),
            &format!(
                "datetime,holdings_after,net_return,risk_degree,buy_count,sell_count\n2024-01-01,\"{{'{shared}': 0.5, '{unique}': 0.5}}\",{ret},0.5,2,0\n2024-01-02,\"{{'{shared}': 0.5, '{unique}': 0.5}}\",{ret},0.5,0,0\n"
            ),
        );
    }

    fn write_text(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, text).unwrap();
    }
}
