use crate::engine::{run_backtest_core_impl, BacktestParams, CoreInputs, OUT_COLS};
use crate::prediction_bundle::{read_prediction_bundle, MatrixFrame, PredictionBundle};
use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use csv::WriterBuilder;
use parquet::arrow::{
    arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder},
    ProjectionMask,
};
use rayon::prelude::*;
use serde::Serialize;
use serde_yaml::Value;
use std::cmp::{max, Ordering};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[path = "run_bundle/plots.rs"]
mod plots;

const WEIGHT_EQUAL: u8 = 0;
const WEIGHT_RANK: u8 = 1;
const WEIGHT_SCORE_SOFTMAX: u8 = 2;
const SCORE_TRANSFORM_NONE: u8 = 0;
const SCORE_TRANSFORM_RANK_PCT: u8 = 1;
const SCORE_TRANSFORM_ZSCORE_CLIP: u8 = 2;
const INTRAPERIOD_EXIT_DISABLED: u8 = 0;
const INTRAPERIOD_EXIT_SCORE_THRESHOLD: u8 = 1;
const INTRAPERIOD_EXIT_EXPECTED_RETURN: u8 = 2;
const INTRAPERIOD_SCORE_TRANSFORM_DISABLED: u8 = 255;
const INTRAPERIOD_SCORE_TRANSFORM_STRATEGY: u8 = 3;
const FUSION_TRANSFORM_RAW: u8 = 0;
const FUSION_TRANSFORM_RANK_PCT: u8 = 1;
const FUSION_MODE_MULTIPLY: u8 = 0;
const FUSION_MODE_BLEND: u8 = 1;
const FUSION_MODE_FILTER: u8 = 2;
const RISK_CURVE_LINEAR: u8 = 0;
const RISK_CURVE_CONVEX: u8 = 1;
const RISK_CURVE_CONCAVE: u8 = 2;
const RISK_CURVE_SIGMOID: u8 = 3;

const BASELINE_DISPLAY_NAMES: &[(&str, &str)] = &[
    ("avg_factor_baseline", "Avg Unique Factor Baseline"),
    (
        "sign_aligned_factor_baseline",
        "Sign-Aligned Factor Baseline",
    ),
    (
        "rank_avg_factor_baseline",
        "Rank-ZScore Average Factor Baseline",
    ),
    (
        "rank_ic_weighted_factor_baseline",
        "RankIC-Weighted Factor Baseline",
    ),
];

pub(crate) type PredictionBundleCache = HashMap<PathBuf, PredictionBundle>;

pub(crate) fn default_baseline_jobs() -> usize {
    env::var("AI4STOCK_RUST_BASELINE_JOBS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
        })
}

#[derive(Debug, Clone)]
struct RunBundleOptions {
    bundle_dir: PathBuf,
    config_path: PathBuf,
    output_dir: PathBuf,
    execution: RunBundleExecutionOptions,
}

#[derive(Debug, Clone)]
pub(crate) struct RunBundleExecutionOptions {
    pub(crate) skip_reference_baselines: bool,
    pub(crate) skip_backtest_plots: bool,
    pub(crate) baseline_jobs: usize,
    pub(crate) quiet: bool,
}

impl Default for RunBundleExecutionOptions {
    fn default() -> Self {
        Self {
            skip_reference_baselines: false,
            skip_backtest_plots: false,
            baseline_jobs: default_baseline_jobs(),
            quiet: false,
        }
    }
}

#[derive(Debug, Clone)]
struct NativeBacktestParams {
    topk: usize,
    n_drop: usize,
    rebalance_freq: usize,
    account: f64,
    risk_degree: f64,
    open_rate: f64,
    close_rate: f64,
    min_cost: f64,
    weighting_mode: u8,
    score_transform_mode: u8,
    zscore_clip: f64,
    max_weight: Option<f64>,
    max_industry_weight: Option<f64>,
    keep_top_n: Option<usize>,
    min_score: Option<f64>,
    desticky_threshold: Option<f64>,
    desticky_n_drop: Option<usize>,
    risk_control: RiskControlConfig,
    intraperiod_exit: Option<IntraperiodExitConfig>,
}

#[derive(Debug, Clone)]
struct RiskControlConfig {
    mode: String,
    risk_degree: f64,
    signal_metric: String,
    signal_source: String,
    validation_metric: String,
    secondary_validation_metric: Option<String>,
    min_signal: f64,
    max_signal: f64,
    min_signal_quantile: Option<f64>,
    max_signal_quantile: Option<f64>,
    min_risk: f64,
    max_risk: f64,
    secondary_min_signal: f64,
    secondary_max_signal: f64,
    secondary_min_signal_quantile: Option<f64>,
    secondary_max_signal_quantile: Option<f64>,
    secondary_min_risk: f64,
    secondary_max_risk: f64,
    risk_curve: String,
    risk_curve_power: f64,
    risk_curve_center: f64,
    risk_curve_steepness: f64,
    fast_window: usize,
    slow_window: usize,
    bull_risk: f64,
    neutral_risk: f64,
    bear_risk: f64,
}

#[derive(Debug, Clone)]
struct IntraperiodExitConfig {
    mode: u8,
    score_transform_mode: u8,
    threshold: f64,
    n_bins: usize,
    min_history: usize,
    price_confirm: Option<PriceConfirmConfig>,
}

#[derive(Debug, Clone)]
struct PriceConfirmConfig {
    ma_window: usize,
    min_remaining_steps: usize,
    force_exit_threshold: Option<f64>,
    signal_timing: String,
    execution_timing: String,
}

#[derive(Debug, Clone, Serialize)]
struct PortfolioMetrics {
    trading_days: usize,
    total_return: f64,
    annualized_return: f64,
    annualized_volatility: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
    avg_turnover: f64,
    final_account_value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BaselineSummary {
    prefix: String,
    display_name: String,
    fixed_risk_prefix: String,
    fixed_risk_display_name: String,
    portfolio_metrics: PortfolioMetrics,
    fixed_risk_portfolio_metrics: PortfolioMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct FusionSummary {
    fusion_mode: String,
    fusion_primary_transform: String,
    fusion_secondary_transform: String,
    fusion_primary_power: f64,
    fusion_secondary_power: f64,
    fusion_blend_weight: f64,
    fusion_filter_threshold: f64,
    fusion_overlap_rows: usize,
    fusion_overlap_dates: usize,
    fusion_primary_mean: f64,
    fusion_secondary_mean: f64,
    fusion_output_mean: f64,
    fusion_secondary_prediction_dir: String,
}

#[derive(Debug, Clone, Serialize)]
struct RunBundleSummary {
    bundle_dir: String,
    config_path: String,
    output_dir: String,
    daily_report_path: String,
    metrics_path: String,
    dates: usize,
    instruments: usize,
    skipped_all_nan_label_dates: usize,
    params: RunBundleSummaryParams,
    portfolio_metrics: PortfolioMetrics,
    baseline_metrics: Vec<BaselineSummary>,
    fusion: Option<FusionSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct RunBundleSummaryParams {
    topk: usize,
    n_drop: usize,
    rebalance_freq: usize,
    account: f64,
    risk_degree: f64,
    weighting: String,
    score_transform: String,
    baseline_jobs: usize,
}

#[derive(Debug, Clone)]
struct BaselineMatrix {
    prefix: String,
    display_name: String,
    matrix: MatrixFrame,
}

#[derive(Debug, Clone)]
struct PreparedMatrices {
    predictions: MatrixFrame,
    labels: MatrixFrame,
    baselines: Vec<BaselineMatrix>,
    fusion: Option<FusionSummary>,
}

#[derive(Debug, Clone)]
struct FixedBaselineMatrices {
    labels: MatrixFrame,
    baselines: Vec<BaselineMatrix>,
}

#[derive(Debug, Clone)]
struct BaselineRun {
    prefix: String,
    display_name: String,
    out: Vec<f64>,
    metrics: PortfolioMetrics,
    fixed_risk_dates_ns: Vec<i64>,
    fixed_risk_out: Vec<f64>,
    fixed_risk_metrics: PortfolioMetrics,
}

#[derive(Debug, Clone)]
struct ScoreFusionConfig {
    secondary_predictions_dir: PathBuf,
    mode: u8,
    mode_name: String,
    primary_transform: u8,
    primary_transform_name: String,
    secondary_transform: u8,
    secondary_transform_name: String,
    primary_power: f64,
    secondary_power: f64,
    blend_weight: f64,
    filter_threshold: f64,
    filter_value: f64,
}

#[derive(Debug, Clone)]
struct FusedMatrix {
    matrix: MatrixFrame,
    present_mask: Vec<bool>,
    summary: FusionSummary,
}

#[derive(Debug, Clone)]
struct FusionPair {
    date_row: usize,
    target_idx: usize,
    primary: f64,
    secondary: f64,
}

pub(crate) fn run(args: &[String]) -> Result<ExitCode, String> {
    let options = parse_options(args)?;
    let config = read_yaml(&options.config_path)?;
    run_with_config(
        &options.bundle_dir,
        &options.config_path,
        &options.output_dir,
        config,
        options.execution,
    )
}

pub(crate) fn run_with_config(
    bundle_dir: &Path,
    config_path: &Path,
    output_dir: &Path,
    config: Value,
    execution: RunBundleExecutionOptions,
) -> Result<ExitCode, String> {
    let bundle = read_prediction_bundle(bundle_dir)?;
    let mut secondary_cache = PredictionBundleCache::new();
    run_with_loaded_bundle_and_cache(
        &bundle,
        config_path,
        output_dir,
        config,
        execution,
        &mut secondary_cache,
    )
}

pub(crate) fn run_with_loaded_bundle_and_cache(
    bundle: &PredictionBundle,
    config_path: &Path,
    output_dir: &Path,
    config: Value,
    execution: RunBundleExecutionOptions,
    secondary_cache: &mut PredictionBundleCache,
) -> Result<ExitCode, String> {
    reject_unsupported_config(&config)?;
    let params = parse_backtest_params(&config)?;
    let matrices = build_prepared_matrices(bundle, &config, &execution, secondary_cache)?;
    if !matrices.predictions.axes_match(&matrices.labels) {
        return Err("prepared predictions and labels axes do not match".to_owned());
    }
    let prepared = prepare_core_inputs(&matrices.predictions, &matrices.labels);
    let n_dates = prepared.dates_ns.len();
    let n_instruments = matrices.predictions.instruments.len();
    let ctx = build_static_run_context(&config, bundle, &matrices.labels, &prepared, &params)?;
    let bench = align_benchmark_returns(&ctx.benchmark_returns, &prepared.dates_ns, Some(0.0));
    let engine_params = backtest_params_to_engine_params(
        &params,
        params.risk_control.risk_degree,
        params.max_industry_weight,
    );
    let strategy_inputs = build_run_matrices(
        &matrices.predictions,
        &matrices.labels,
        &prepared,
        &ctx,
        &params,
    )?;
    let out = run_core_backtest(
        &strategy_inputs.core_scores,
        &prepared.labels,
        n_dates,
        n_instruments,
        engine_params,
        &bench,
        &ctx.instrument_group_ids,
        &strategy_inputs.risk_values,
        &strategy_inputs.risk_signal_values,
        &ctx.price_confirm,
        &strategy_inputs.intraperiod_scores,
    )?;
    let fixed_risk_params =
        backtest_params_to_engine_params(&params, params.risk_degree, params.max_industry_weight);
    let fixed_baseline_matrices = build_fixed_risk_baseline_matrices(bundle, &execution)?;
    let fixed_baseline_context = if let Some(fixed) = fixed_baseline_matrices.as_ref() {
        let axis_prepared = prepare_core_inputs(&fixed.baselines[0].matrix, &fixed.labels);
        let fixed_ctx =
            build_static_run_context(&config, bundle, &fixed.labels, &axis_prepared, &params)?;
        let fixed_bench = align_benchmark_returns(
            &fixed_ctx.benchmark_returns,
            &axis_prepared.dates_ns,
            Some(0.0),
        );
        Some((fixed_ctx, fixed_bench))
    } else {
        None
    };
    let baseline_runs = run_baseline_backtests(
        &matrices.baselines,
        &matrices.labels,
        &prepared,
        n_dates,
        n_instruments,
        &ctx,
        &params,
        engine_params,
        &bench,
        fixed_risk_params,
        fixed_baseline_matrices.as_ref(),
        fixed_baseline_context.as_ref(),
        execution.baseline_jobs,
    )?;

    fs::create_dir_all(output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    let daily_report_path = output_dir.join("native_daily_report.csv");
    let metrics_path = output_dir.join("native_portfolio_metrics.json");
    write_daily_report(
        &daily_report_path,
        &prepared.dates_ns,
        &out,
        &baseline_runs,
        params.intraperiod_exit.as_ref(),
    )?;
    let metrics = compute_metrics(&out);
    write_metrics(&metrics_path, &metrics)?;
    copy_training_summary_if_present(bundle, output_dir)?;
    if !execution.skip_backtest_plots {
        let plot_paths =
            plots::write_backtest_plots(output_dir, &prepared.dates_ns, &out, &baseline_runs)?;
        if !execution.quiet {
            for path in plot_paths {
                println!("[plot] {}", path.display());
            }
        }
    }
    let baseline_metrics = baseline_runs
        .iter()
        .map(|run| BaselineSummary {
            prefix: run.prefix.clone(),
            display_name: run.display_name.clone(),
            fixed_risk_prefix: format!("fixed_risk_{}", run.prefix),
            fixed_risk_display_name: format!("Fixed-Risk {}", run.display_name),
            portfolio_metrics: run.metrics.clone(),
            fixed_risk_portfolio_metrics: run.fixed_risk_metrics.clone(),
        })
        .collect::<Vec<_>>();
    let summary = RunBundleSummary {
        bundle_dir: bundle.dir.to_string_lossy().into_owned(),
        config_path: config_path.to_string_lossy().into_owned(),
        output_dir: output_dir.to_string_lossy().into_owned(),
        daily_report_path: daily_report_path.to_string_lossy().into_owned(),
        metrics_path: metrics_path.to_string_lossy().into_owned(),
        dates: prepared.dates_ns.len(),
        instruments: matrices.predictions.instruments.len(),
        skipped_all_nan_label_dates: prepared.skipped_all_nan_label_dates,
        params: RunBundleSummaryParams {
            topk: params.topk,
            n_drop: params.n_drop,
            rebalance_freq: params.rebalance_freq,
            account: params.account,
            risk_degree: params.risk_degree,
            weighting: weighting_name(params.weighting_mode).to_owned(),
            score_transform: score_transform_name(params.score_transform_mode).to_owned(),
            baseline_jobs: execution.baseline_jobs.max(1),
        },
        portfolio_metrics: metrics,
        baseline_metrics,
        fusion: matrices.fusion,
    };
    let summary_path = output_dir.join("rust_bundle_summary.json");
    let summary_file = File::create(&summary_path)
        .map_err(|err| format!("failed to create {}: {err}", summary_path.display()))?;
    serde_json::to_writer_pretty(summary_file, &summary)
        .map_err(|err| format!("failed to write {}: {err}", summary_path.display()))?;
    if !execution.quiet {
        println!("[bundle] {}", summary.bundle_dir);
        println!("[daily_report] {}", daily_report_path.display());
        println!("[metrics] {}", metrics_path.display());
        println!("[summary] {}", summary_path.display());
        println!(
            "[result] days={} instruments={} total_return={:.6} ann_return={:.6} max_drawdown={:.6}",
            summary.dates,
            summary.instruments,
            summary.portfolio_metrics.total_return,
            summary.portfolio_metrics.annualized_return,
            summary.portfolio_metrics.max_drawdown
        );
    }
    Ok(ExitCode::SUCCESS)
}

#[allow(clippy::too_many_arguments)]
fn run_baseline_backtests(
    baselines: &[BaselineMatrix],
    labels: &MatrixFrame,
    prepared: &PreparedInputs,
    n_dates: usize,
    n_instruments: usize,
    ctx: &StaticRunContext,
    params: &NativeBacktestParams,
    engine_params: BacktestParams,
    bench: &[f64],
    fixed_risk_params: BacktestParams,
    fixed_baseline_matrices: Option<&FixedBaselineMatrices>,
    fixed_baseline_context: Option<&(StaticRunContext, Vec<f64>)>,
    baseline_jobs: usize,
) -> Result<Vec<BaselineRun>, String> {
    if baselines.is_empty() {
        return Ok(Vec::new());
    }
    let worker_count = baseline_jobs.max(1).min(baselines.len());
    if worker_count == 1 {
        return baselines
            .iter()
            .map(|baseline| {
                run_one_baseline_backtest(
                    baseline,
                    labels,
                    prepared,
                    n_dates,
                    n_instruments,
                    ctx,
                    params,
                    engine_params,
                    bench,
                    fixed_risk_params,
                    fixed_baseline_matrices,
                    fixed_baseline_context,
                )
            })
            .collect();
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|err| format!("failed to build Rayon thread pool: {err}"))?;
    pool.install(|| {
        baselines
            .par_iter()
            .enumerate()
            .map(|(index, baseline)| {
                run_one_baseline_backtest(
                    baseline,
                    labels,
                    prepared,
                    n_dates,
                    n_instruments,
                    ctx,
                    params,
                    engine_params,
                    bench,
                    fixed_risk_params,
                    fixed_baseline_matrices,
                    fixed_baseline_context,
                )
                .map(|run| (index, run))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|mut pairs| {
                pairs.sort_by_key(|(index, _)| *index);
                pairs.into_iter().map(|(_, run)| run).collect()
            })
    })
}

#[allow(clippy::too_many_arguments)]
fn run_one_baseline_backtest(
    baseline: &BaselineMatrix,
    labels: &MatrixFrame,
    prepared: &PreparedInputs,
    n_dates: usize,
    n_instruments: usize,
    ctx: &StaticRunContext,
    params: &NativeBacktestParams,
    engine_params: BacktestParams,
    bench: &[f64],
    fixed_risk_params: BacktestParams,
    fixed_baseline_matrices: Option<&FixedBaselineMatrices>,
    fixed_baseline_context: Option<&(StaticRunContext, Vec<f64>)>,
) -> Result<BaselineRun, String> {
    let baseline_prepared = prepare_core_inputs(&baseline.matrix, labels);
    if baseline_prepared.dates_ns != prepared.dates_ns {
        return Err(format!(
            "baseline {} did not prepare to the same date axis as the strategy",
            baseline.prefix
        ));
    }
    let baseline_inputs =
        build_run_matrices(&baseline.matrix, labels, &baseline_prepared, ctx, params)?;
    let baseline_out = run_core_backtest(
        &baseline_inputs.core_scores,
        &prepared.labels,
        n_dates,
        n_instruments,
        engine_params,
        bench,
        &ctx.instrument_group_ids,
        &baseline_inputs.risk_values,
        &baseline_inputs.risk_signal_values,
        &ctx.price_confirm,
        &baseline_inputs.intraperiod_scores,
    )?;
    let (fixed_risk_dates_ns, fixed_risk_out) =
        if let (Some(fixed), Some((fixed_ctx, fixed_bench))) =
            (fixed_baseline_matrices, fixed_baseline_context)
        {
            let fixed_baseline = fixed
                .baselines
                .iter()
                .find(|candidate| candidate.prefix == baseline.prefix)
                .ok_or_else(|| {
                    format!("fixed-risk baseline matrix missing for {}", baseline.prefix)
                })?;
            let fixed_prepared = prepare_core_inputs(&fixed_baseline.matrix, &fixed.labels);
            let fixed_inputs = build_run_matrices(
                &fixed_baseline.matrix,
                &fixed.labels,
                &fixed_prepared,
                fixed_ctx,
                params,
            )?;
            let out = run_core_backtest(
                &fixed_inputs.core_scores,
                &fixed_prepared.labels,
                fixed_prepared.dates_ns.len(),
                fixed_baseline.matrix.instruments.len(),
                fixed_risk_params,
                fixed_bench,
                &fixed_ctx.instrument_group_ids,
                &[],
                &[],
                &fixed_ctx.price_confirm,
                &fixed_inputs.intraperiod_scores,
            )?;
            (fixed_prepared.dates_ns, out)
        } else {
            let out = run_core_backtest(
                &baseline_inputs.core_scores,
                &prepared.labels,
                n_dates,
                n_instruments,
                fixed_risk_params,
                bench,
                &ctx.instrument_group_ids,
                &[],
                &[],
                &ctx.price_confirm,
                &baseline_inputs.intraperiod_scores,
            )?;
            (prepared.dates_ns.clone(), out)
        };
    let metrics = compute_metrics(&baseline_out);
    let fixed_risk_metrics = compute_metrics(&fixed_risk_out);
    Ok(BaselineRun {
        prefix: baseline.prefix.clone(),
        display_name: baseline.display_name.clone(),
        out: baseline_out,
        metrics,
        fixed_risk_dates_ns,
        fixed_risk_out,
        fixed_risk_metrics,
    })
}

fn backtest_params_to_engine_params(
    params: &NativeBacktestParams,
    default_risk_degree: f64,
    max_group_weight: Option<f64>,
) -> BacktestParams {
    let (intraperiod_exit_mode, intraperiod_score_transform_mode, threshold, n_bins, min_history) =
        if let Some(intraperiod) = &params.intraperiod_exit {
            (
                intraperiod.mode,
                intraperiod.score_transform_mode,
                Some(intraperiod.threshold),
                intraperiod.n_bins,
                intraperiod.min_history,
            )
        } else {
            (
                INTRAPERIOD_EXIT_DISABLED,
                INTRAPERIOD_SCORE_TRANSFORM_DISABLED,
                None,
                20,
                200,
            )
        };
    let (price_confirm_min_remaining_steps, price_confirm_force_exit_threshold) = params
        .intraperiod_exit
        .as_ref()
        .and_then(|intraperiod| intraperiod.price_confirm.as_ref())
        .map(|confirm| (confirm.min_remaining_steps, confirm.force_exit_threshold))
        .unwrap_or((0, None));
    BacktestParams {
        topk: params.topk,
        n_drop: params.n_drop,
        rebalance_freq: params.rebalance_freq,
        account: params.account,
        default_risk_degree,
        open_rate: params.open_rate,
        close_rate: params.close_rate,
        min_cost: params.min_cost,
        weighting_mode: params.weighting_mode,
        score_transform_mode: params.score_transform_mode,
        zscore_clip: params.zscore_clip,
        intraperiod_exit_mode,
        intraperiod_score_transform_mode,
        intraperiod_exit_threshold: threshold,
        intraperiod_expected_return_n_bins: n_bins,
        intraperiod_expected_return_min_history: min_history,
        price_confirm_min_remaining_steps,
        price_confirm_force_exit_threshold,
        max_weight: params.max_weight,
        max_group_weight,
        keep_top_n: params.keep_top_n,
        min_score: params.min_score,
        desticky_threshold: params.desticky_threshold,
        desticky_n_drop: params.desticky_n_drop,
    }
}

#[derive(Debug, Clone)]
struct PreparedInputs {
    dates_ns: Vec<i64>,
    scores: Vec<f64>,
    labels: Vec<f64>,
    skipped_all_nan_label_dates: usize,
}

fn prepare_core_inputs(predictions: &MatrixFrame, labels: &MatrixFrame) -> PreparedInputs {
    let n_cols = predictions.instruments.len();
    let mut dates_ns = Vec::with_capacity(predictions.dates_ns.len());
    let mut scores = Vec::with_capacity(predictions.values.len());
    let mut label_values = Vec::with_capacity(labels.values.len());
    let mut skipped = 0usize;
    for row in 0..predictions.dates_ns.len() {
        let start = row * n_cols;
        let end = start + n_cols;
        if !labels.values[start..end]
            .iter()
            .any(|value| value.is_finite())
        {
            skipped += 1;
            continue;
        }
        dates_ns.push(predictions.dates_ns[row]);
        scores.extend_from_slice(&predictions.values[start..end]);
        label_values.extend_from_slice(&labels.values[start..end]);
    }
    PreparedInputs {
        dates_ns,
        scores,
        labels: label_values,
        skipped_all_nan_label_dates: skipped,
    }
}

#[derive(Debug, Clone)]
struct StaticRunContext {
    benchmark_returns: Vec<(i64, f64)>,
    instrument_group_ids: Vec<i32>,
    price_confirm: Vec<u8>,
    risk_control: RiskControlConfig,
    training_summary_records: Vec<BTreeMap<String, String>>,
}

#[derive(Debug, Clone)]
struct RunMatrices {
    core_scores: Vec<f64>,
    intraperiod_scores: Vec<f64>,
    risk_values: Vec<f64>,
    risk_signal_values: Vec<f64>,
}

fn build_static_run_context(
    config: &Value,
    bundle: &PredictionBundle,
    labels: &MatrixFrame,
    prepared: &PreparedInputs,
    params: &NativeBacktestParams,
) -> Result<StaticRunContext, String> {
    let source_data_dir = resolve_source_parquet_dir(config)?;
    let benchmark_returns = load_benchmark_returns(config)?;
    let instrument_group_ids = if params.max_industry_weight.is_some() {
        load_instrument_group_ids(config, &labels.instruments)?
    } else {
        Vec::new()
    };
    let price_confirm = if let Some(intraperiod) = &params.intraperiod_exit {
        if intraperiod.price_confirm.is_some() {
            load_price_confirm_matrix(
                &source_data_dir,
                &prepared.dates_ns,
                &labels.instruments,
                intraperiod.price_confirm.as_ref().ok_or_else(|| {
                    "intraperiod exit price_confirm configuration missing".to_owned()
                })?,
            )?
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    Ok(StaticRunContext {
        benchmark_returns,
        instrument_group_ids,
        price_confirm,
        risk_control: params.risk_control.clone(),
        training_summary_records: bundle.training_summary_records.clone(),
    })
}

fn build_run_matrices(
    matrix: &MatrixFrame,
    _labels: &MatrixFrame,
    prepared: &PreparedInputs,
    ctx: &StaticRunContext,
    params: &NativeBacktestParams,
) -> Result<RunMatrices, String> {
    let core_scores = prepared.scores.clone();
    let intraperiod_scores = Vec::new();
    let (risk_values, risk_signal_values) =
        build_risk_schedule_values(matrix, prepared, ctx, params, &core_scores)?;
    Ok(RunMatrices {
        core_scores,
        intraperiod_scores,
        risk_values,
        risk_signal_values,
    })
}

fn rank_pct_transform_row(input: &[f64], out: &mut [f64]) {
    out.fill(f64::NAN);
    let mut order: Vec<usize> = (0..input.len())
        .filter(|&idx| input[idx].is_finite())
        .collect();
    let n = order.len();
    if n == 0 {
        return;
    }
    order.sort_by(|&left, &right| {
        input[left]
            .partial_cmp(&input[right])
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.cmp(&right))
    });
    let mut pos = 0usize;
    while pos < n {
        let mut end = pos + 1;
        while end < n && input[order[end]] == input[order[pos]] {
            end += 1;
        }
        let avg_rank = (pos + 1 + end) as f64 / 2.0;
        let pct_rank = avg_rank / n as f64;
        for idx in pos..end {
            out[order[idx]] = pct_rank;
        }
        pos = end;
    }
}

fn zscore_clip_transform_row(input: &[f64], clip: f64, out: &mut [f64]) {
    out.fill(f64::NAN);
    let mut count = 0usize;
    let mut sum = 0.0_f64;
    for &value in input {
        if value.is_finite() {
            count += 1;
            sum += value;
        }
    }
    if count == 0 {
        return;
    }
    let mean = sum / count as f64;
    let mut sq_sum = 0.0_f64;
    for &value in input {
        if value.is_finite() {
            let centered = value - mean;
            sq_sum += centered * centered;
        }
    }
    let std = (sq_sum / count as f64).sqrt();
    if !std.is_finite() || std.abs() <= 1e-12 {
        for (idx, &value) in input.iter().enumerate() {
            if value.is_finite() {
                out[idx] = 0.0;
            }
        }
        return;
    }
    let clip_value = clip.max(0.0);
    for (idx, &value) in input.iter().enumerate() {
        if value.is_finite() {
            let mut transformed = (value - mean) / std;
            if clip_value > 0.0 {
                transformed = transformed.clamp(-clip_value, clip_value);
            }
            out[idx] = transformed;
        }
    }
}

fn compute_signal_strength_row(values: &[f64], metric: &str, min_score: Option<f64>) -> f64 {
    let mut finite_values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if let Some(threshold) = min_score {
        finite_values.retain(|value| *value > threshold);
    }
    if finite_values.is_empty() {
        return f64::NAN;
    }
    match metric {
        "top1" => finite_values
            .into_iter()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(value)),
        "topk_sum" => finite_values.into_iter().sum(),
        _ => {
            let count = finite_values.len();
            finite_values.into_iter().sum::<f64>() / count as f64
        }
    }
}

fn build_signal_strength_series(
    values: &[f64],
    n_dates: usize,
    n_cols: usize,
    risk_control: &RiskControlConfig,
    topk: usize,
    min_score: Option<f64>,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_dates);
    for row in 0..n_dates {
        let row_values = &values[row * n_cols..(row + 1) * n_cols];
        let signal = match risk_control.signal_metric.as_str() {
            "top1" => compute_signal_strength_row(row_values, "top1", min_score),
            "topk_sum" => {
                let mut finite_values: Vec<f64> = row_values
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .collect();
                if let Some(threshold) = min_score {
                    finite_values.retain(|value| *value > threshold);
                }
                finite_values.sort_by(|left, right| {
                    right
                        .partial_cmp(left)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| Ordering::Equal)
                });
                finite_values.into_iter().take(max(1, topk)).sum()
            }
            _ => {
                let mut finite_values: Vec<f64> = row_values
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .collect();
                if let Some(threshold) = min_score {
                    finite_values.retain(|value| *value > threshold);
                }
                finite_values
                    .sort_by(|left, right| right.partial_cmp(left).unwrap_or(Ordering::Equal));
                if finite_values.is_empty() {
                    f64::NAN
                } else {
                    let k = max(1, topk).min(finite_values.len());
                    finite_values.into_iter().take(k).sum::<f64>() / k as f64
                }
            }
        };
        out.push(signal);
    }
    out
}

fn build_risk_schedule_values(
    matrix: &MatrixFrame,
    prepared: &PreparedInputs,
    ctx: &StaticRunContext,
    params: &NativeBacktestParams,
    _core_scores: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let risk = &ctx.risk_control;
    let n_dates = prepared.dates_ns.len();
    let n_cols = matrix.instruments.len();
    match risk.mode.as_str() {
        "fixed" => Ok((Vec::new(), Vec::new())),
        "signal_strength" => build_signal_risk_schedule_values(prepared, n_cols, ctx, params),
        "benchmark_ma" => Ok((
            build_benchmark_schedule_for_prepared(prepared, ctx, risk)?,
            Vec::new(),
        )),
        "benchmark_ma_signal_strength" => {
            let bench_schedule = build_benchmark_schedule_for_prepared(prepared, ctx, risk)?;
            let (signal_schedule, signal_values) =
                build_signal_risk_schedule_values(prepared, n_cols, ctx, params)?;
            if bench_schedule.len() != n_dates || signal_schedule.len() != n_dates {
                return Err("internal risk schedule length mismatch".to_owned());
            }
            let combined = bench_schedule
                .iter()
                .zip(signal_schedule.iter())
                .map(|(bench, signal)| bench.min(*signal))
                .collect::<Vec<_>>();
            Ok((combined, signal_values))
        }
        other => Err(format!(
            "Unsupported risk control mode: {other}. Supported: fixed, benchmark_ma, signal_strength, benchmark_ma_signal_strength"
        )),
    }
}

fn build_signal_risk_schedule_values(
    prepared: &PreparedInputs,
    n_cols: usize,
    ctx: &StaticRunContext,
    params: &NativeBacktestParams,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let risk = &ctx.risk_control;
    if risk.signal_source == "validation_metric" {
        let primary_signal = build_validation_metric_signal_series(
            &ctx.training_summary_records,
            &risk.validation_metric,
            &prepared.dates_ns,
        )
        .ok_or_else(|| {
            format!(
                "No rolling validation metric series available for risk_control.validation_metric={}",
                risk.validation_metric
            )
        })?;
        if let Some(secondary_metric) = &risk.secondary_validation_metric {
            let secondary_signal = build_validation_metric_signal_series(
                &ctx.training_summary_records,
                secondary_metric,
                &prepared.dates_ns,
            )
            .ok_or_else(|| {
                format!(
                    "No rolling validation metric series available for risk_control.secondary_validation_metric={secondary_metric}"
                )
            })?;
            let primary_schedule = build_signal_schedule_from_series(
                &primary_signal,
                risk.min_signal,
                risk.max_signal,
                risk.min_risk,
                risk.max_risk,
                risk.min_signal_quantile,
                risk.max_signal_quantile,
                risk,
            )?;
            let secondary_schedule = build_signal_schedule_from_series(
                &secondary_signal,
                risk.secondary_min_signal,
                risk.secondary_max_signal,
                risk.secondary_min_risk,
                risk.secondary_max_risk,
                risk.secondary_min_signal_quantile,
                risk.secondary_max_signal_quantile,
                risk,
            )?;
            let combined = primary_schedule
                .iter()
                .zip(secondary_schedule.iter())
                .map(|(primary, secondary)| primary.min(*secondary))
                .collect::<Vec<_>>();
            return Ok((combined, primary_signal));
        }
        let schedule = build_signal_schedule_from_series(
            &primary_signal,
            risk.min_signal,
            risk.max_signal,
            risk.min_risk,
            risk.max_risk,
            risk.min_signal_quantile,
            risk.max_signal_quantile,
            risk,
        )?;
        return Ok((schedule, primary_signal));
    }

    let transformed = transform_flat_values(
        &prepared.scores,
        prepared.dates_ns.len(),
        n_cols,
        params.score_transform_mode,
        params.zscore_clip,
    )?;
    let signal_values = build_signal_strength_series(
        &transformed,
        prepared.dates_ns.len(),
        n_cols,
        risk,
        params.topk,
        params.min_score,
    );
    let schedule = build_signal_schedule_from_series(
        &signal_values,
        risk.min_signal,
        risk.max_signal,
        risk.min_risk,
        risk.max_risk,
        risk.min_signal_quantile,
        risk.max_signal_quantile,
        risk,
    )?;
    Ok((schedule, signal_values))
}

fn build_validation_metric_signal_series(
    records: &[BTreeMap<String, String>],
    metric_name: &str,
    dates_ns: &[i64],
) -> Option<Vec<f64>> {
    if records.is_empty() || dates_ns.is_empty() {
        return None;
    }
    let mut out = vec![f64::NAN; dates_ns.len()];
    for record in records {
        let Some(raw_value) = record.get(metric_name) else {
            continue;
        };
        let Ok(value) = raw_value.parse::<f64>() else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let Some(start) = record
            .get("window_start")
            .and_then(|value| parse_datetime_ns(value).ok())
        else {
            continue;
        };
        let Some(end) = record
            .get("window_end")
            .and_then(|value| parse_datetime_ns(value).ok())
        else {
            continue;
        };
        for (idx, date_ns) in dates_ns.iter().enumerate() {
            if *date_ns >= start && *date_ns <= end {
                out[idx] = value;
            }
        }
    }
    out.iter().any(|value| value.is_finite()).then_some(out)
}

#[allow(clippy::too_many_arguments)]
fn build_signal_schedule_from_series(
    signal_values: &[f64],
    min_signal: f64,
    max_signal: f64,
    min_risk: f64,
    max_risk: f64,
    min_quantile: Option<f64>,
    max_quantile: Option<f64>,
    risk: &RiskControlConfig,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(signal_values.len());
    let mut history = Vec::with_capacity(signal_values.len());
    for &signal in signal_values {
        let mut min_threshold = min_signal;
        let mut max_threshold = max_signal;
        if let Some(q) = min_quantile {
            if !history.is_empty() {
                let value = linear_quantile(&history, q);
                if value.is_finite() {
                    min_threshold = value;
                }
            }
        }
        if let Some(q) = max_quantile {
            if !history.is_empty() {
                let value = linear_quantile(&history, q);
                if value.is_finite() {
                    max_threshold = value;
                }
            }
        }
        if max_threshold <= min_threshold {
            min_threshold = min_signal;
            max_threshold = max_signal;
        }
        out.push(signal_risk_value(
            signal,
            min_threshold,
            max_threshold,
            min_risk,
            max_risk,
            risk,
        )?);
        insert_sorted_finite(&mut history, signal);
    }
    Ok(out)
}

fn signal_risk_value(
    value: f64,
    min_signal: f64,
    max_signal: f64,
    min_risk: f64,
    max_risk: f64,
    risk: &RiskControlConfig,
) -> Result<f64, String> {
    let width = max_signal - min_signal;
    if width <= 0.0 {
        return Err(
            "risk_control.max_signal must be greater than risk_control.min_signal".to_owned(),
        );
    }
    let raw_scale = if value.is_nan() {
        0.0
    } else {
        ((value - min_signal) / width).clamp(0.0, 1.0)
    };
    let scale = apply_risk_curve(raw_scale, risk)?;
    Ok(min_risk + scale * (max_risk - min_risk))
}

fn apply_risk_curve(scale: f64, risk: &RiskControlConfig) -> Result<f64, String> {
    let clamped = scale.clamp(0.0, 1.0);
    match risk_curve_code(&risk.risk_curve)? {
        RISK_CURVE_LINEAR => Ok(clamped),
        RISK_CURVE_CONVEX => Ok(clamped.powf(risk.risk_curve_power).clamp(0.0, 1.0)),
        RISK_CURVE_CONCAVE => {
            Ok((1.0 - (1.0 - clamped).powf(risk.risk_curve_power)).clamp(0.0, 1.0))
        }
        RISK_CURVE_SIGMOID => {
            let centered = 1.0
                / (1.0 + (-risk.risk_curve_steepness * (clamped - risk.risk_curve_center)).exp());
            let lower =
                1.0 / (1.0 + (-risk.risk_curve_steepness * (0.0 - risk.risk_curve_center)).exp());
            let upper =
                1.0 / (1.0 + (-risk.risk_curve_steepness * (1.0 - risk.risk_curve_center)).exp());
            let denom = (upper - lower).max(1e-12);
            Ok(((centered - lower) / denom).clamp(0.0, 1.0))
        }
        other => Err(format!("unsupported risk curve code: {other}")),
    }
}

fn linear_quantile(sorted_values: &[f64], quantile: f64) -> f64 {
    if sorted_values.is_empty() {
        return f64::NAN;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let q = quantile.clamp(0.0, 1.0);
    let pos = q * (sorted_values.len() - 1) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        sorted_values[lower]
    } else {
        let weight = pos - lower as f64;
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight
    }
}

fn insert_sorted_finite(values: &mut Vec<f64>, value: f64) {
    if !value.is_finite() {
        return;
    }
    let pos = values
        .binary_search_by(|probe| probe.partial_cmp(&value).unwrap_or(Ordering::Equal))
        .unwrap_or_else(|idx| idx);
    values.insert(pos, value);
}

fn build_benchmark_schedule_for_prepared(
    prepared: &PreparedInputs,
    ctx: &StaticRunContext,
    risk: &RiskControlConfig,
) -> Result<Vec<f64>, String> {
    if ctx.benchmark_returns.is_empty() {
        let returns = cross_section_label_returns(&prepared.labels, prepared.dates_ns.len());
        return build_benchmark_ma_schedule(
            &returns,
            risk.fast_window,
            risk.slow_window,
            risk.bull_risk,
            risk.neutral_risk,
            risk.bear_risk,
            risk.risk_degree,
        );
    }
    let dates = ctx
        .benchmark_returns
        .iter()
        .map(|(date_ns, _)| *date_ns)
        .collect::<Vec<_>>();
    let returns = ctx
        .benchmark_returns
        .iter()
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    let schedule = build_benchmark_ma_schedule(
        &returns,
        risk.fast_window,
        risk.slow_window,
        risk.bull_risk,
        risk.neutral_risk,
        risk.bear_risk,
        risk.risk_degree,
    )?;
    let by_date = dates.into_iter().zip(schedule).collect::<BTreeMap<_, _>>();
    Ok(prepared
        .dates_ns
        .iter()
        .map(|date_ns| by_date.get(date_ns).copied().unwrap_or(risk.risk_degree))
        .collect())
}

fn cross_section_label_returns(labels: &[f64], n_dates: usize) -> Vec<f64> {
    if n_dates == 0 {
        return Vec::new();
    }
    let n_cols = labels.len() / n_dates;
    let mut out = Vec::with_capacity(n_dates);
    for row in 0..n_dates {
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &labels[row * n_cols..(row + 1) * n_cols] {
            if value.is_finite() {
                sum += *value;
                count += 1;
            }
        }
        out.push(if count == 0 { 0.0 } else { sum / count as f64 });
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_benchmark_ma_schedule(
    returns: &[f64],
    fast_window: usize,
    slow_window: usize,
    bull_risk: f64,
    neutral_risk: f64,
    bear_risk: f64,
    fallback_risk_degree: f64,
) -> Result<Vec<f64>, String> {
    let fast = fast_window.max(1);
    let slow = slow_window.max(1);
    if fast >= slow {
        return Err(
            "risk_control.fast_window must be smaller than risk_control.slow_window".to_owned(),
        );
    }
    let mut nav_values = Vec::with_capacity(returns.len());
    let mut nav = 1.0_f64;
    for &value in returns {
        nav *= 1.0 + if value.is_finite() { value } else { 0.0 };
        nav_values.push(nav);
    }
    let mut out = Vec::with_capacity(returns.len());
    let mut fast_sum = 0.0;
    let mut slow_sum = 0.0;
    let mut previous_raw = fallback_risk_degree;
    for idx in 0..nav_values.len() {
        let value = nav_values[idx];
        fast_sum += value;
        slow_sum += value;
        if idx >= fast {
            fast_sum -= nav_values[idx - fast];
        }
        if idx >= slow {
            slow_sum -= nav_values[idx - slow];
        }
        let fast_ma = fast_sum / (idx + 1).min(fast) as f64;
        let slow_ma = slow_sum / (idx + 1).min(slow) as f64;
        out.push(if idx == 0 {
            fallback_risk_degree
        } else {
            previous_raw
        });
        let mut raw = bear_risk;
        if value >= slow_ma {
            raw = neutral_risk;
        }
        if value >= fast_ma {
            raw = bull_risk;
        }
        previous_raw = raw;
    }
    Ok(out)
}

fn transform_flat_values(
    values: &[f64],
    n_dates: usize,
    n_cols: usize,
    mode: u8,
    zscore_clip: f64,
) -> Result<Vec<f64>, String> {
    let mut out = vec![f64::NAN; values.len()];
    for row in 0..n_dates {
        let start = row * n_cols;
        let end = start + n_cols;
        match mode {
            SCORE_TRANSFORM_NONE => out[start..end].copy_from_slice(&values[start..end]),
            SCORE_TRANSFORM_RANK_PCT => {
                rank_pct_transform_row(&values[start..end], &mut out[start..end])
            }
            SCORE_TRANSFORM_ZSCORE_CLIP => {
                zscore_clip_transform_row(&values[start..end], zscore_clip, &mut out[start..end])
            }
            other => return Err(format!("unsupported Rust score transform code: {other}")),
        }
    }
    Ok(out)
}

fn align_benchmark_returns(
    benchmark_returns: &[(i64, f64)],
    dates_ns: &[i64],
    fill_value: Option<f64>,
) -> Vec<f64> {
    if benchmark_returns.is_empty() {
        return Vec::new();
    }
    let by_date = benchmark_returns
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    dates_ns
        .iter()
        .map(|date_ns| {
            by_date
                .get(date_ns)
                .copied()
                .or(fill_value)
                .unwrap_or(f64::NAN)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_core_backtest(
    scores: &[f64],
    labels: &[f64],
    n_dates: usize,
    n_instruments: usize,
    params: BacktestParams,
    bench: &[f64],
    group_ids: &[i32],
    risk_values: &[f64],
    risk_signal_values: &[f64],
    price_confirm: &[u8],
    intraperiod_scores: &[f64],
) -> Result<Vec<f64>, String> {
    let mut out = vec![0.0_f64; n_dates * OUT_COLS];
    let core_inputs = CoreInputs {
        scores,
        labels,
        bench,
        group_ids,
        risk_values,
        risk_signal_values,
        price_confirm,
        intraperiod_scores,
        n_dates,
        n_instruments,
    };
    run_backtest_core_impl(core_inputs, params, &mut out)?;
    Ok(out)
}

fn build_prepared_matrices(
    bundle: &PredictionBundle,
    config: &Value,
    execution: &RunBundleExecutionOptions,
    secondary_cache: &mut PredictionBundleCache,
) -> Result<PreparedMatrices, String> {
    let baselines = if execution.skip_reference_baselines {
        Vec::new()
    } else {
        collect_baseline_matrices(bundle)
    };
    if let Some(fusion_cfg) = parse_score_fusion_config(config)? {
        let secondary_bundle =
            read_cached_prediction_bundle(&fusion_cfg.secondary_predictions_dir, secondary_cache)?;
        let fused = fuse_prediction_matrices(
            &bundle.final_predictions,
            &secondary_bundle.final_predictions,
            &fusion_cfg,
        )?;
        return project_fused_bundle(bundle, fused, baselines);
    }
    project_unfused_bundle(bundle, baselines)
}

fn read_cached_prediction_bundle<'a>(
    raw_path: &Path,
    cache: &'a mut PredictionBundleCache,
) -> Result<&'a PredictionBundle, String> {
    let key = raw_path.to_path_buf();
    if !cache.contains_key(&key) {
        let bundle = read_prediction_bundle(raw_path)?;
        cache.insert(key.clone(), bundle);
    }
    cache
        .get(&key)
        .ok_or_else(|| format!("internal bundle cache miss for {}", raw_path.display()))
}

fn build_fixed_risk_baseline_matrices(
    bundle: &PredictionBundle,
    execution: &RunBundleExecutionOptions,
) -> Result<Option<FixedBaselineMatrices>, String> {
    let baselines = if execution.skip_reference_baselines {
        Vec::new()
    } else {
        collect_baseline_matrices(bundle)
    };
    if baselines.is_empty() {
        return Ok(None);
    }
    let mut matrices = vec![&bundle.backtest_labels];
    matrices.extend(baselines.iter().map(|baseline| &baseline.matrix));
    let (dates_ns, instruments) = common_axes(&matrices)?;
    let present_mask = vec![true; dates_ns.len() * instruments.len()];
    let labels = project_matrix(
        &bundle.backtest_labels,
        &dates_ns,
        &instruments,
        &present_mask,
    );
    let baselines = baselines
        .into_iter()
        .map(|baseline| BaselineMatrix {
            prefix: baseline.prefix,
            display_name: baseline.display_name,
            matrix: project_matrix(&baseline.matrix, &dates_ns, &instruments, &present_mask),
        })
        .collect();
    Ok(Some(FixedBaselineMatrices { labels, baselines }))
}

fn collect_baseline_matrices(bundle: &PredictionBundle) -> Vec<BaselineMatrix> {
    bundle
        .baselines
        .iter()
        .filter_map(|baseline| {
            baseline.matrix.as_ref().map(|matrix| BaselineMatrix {
                prefix: baseline.name.clone(),
                display_name: baseline_display_name(&baseline.name).to_owned(),
                matrix: matrix.clone(),
            })
        })
        .collect()
}

fn project_unfused_bundle(
    bundle: &PredictionBundle,
    baselines: Vec<BaselineMatrix>,
) -> Result<PreparedMatrices, String> {
    let mut matrices = vec![&bundle.final_predictions, &bundle.backtest_labels];
    matrices.extend(baselines.iter().map(|baseline| &baseline.matrix));
    let (dates_ns, instruments) = common_axes(&matrices)?;
    let present_mask = vec![true; dates_ns.len() * instruments.len()];
    let predictions = project_matrix(
        &bundle.final_predictions,
        &dates_ns,
        &instruments,
        &present_mask,
    );
    let labels = project_matrix(
        &bundle.backtest_labels,
        &dates_ns,
        &instruments,
        &present_mask,
    );
    let baselines = baselines
        .into_iter()
        .map(|baseline| BaselineMatrix {
            prefix: baseline.prefix,
            display_name: baseline.display_name,
            matrix: project_matrix(&baseline.matrix, &dates_ns, &instruments, &present_mask),
        })
        .collect();
    Ok(PreparedMatrices {
        predictions,
        labels,
        baselines,
        fusion: None,
    })
}

fn project_fused_bundle(
    bundle: &PredictionBundle,
    fused: FusedMatrix,
    baselines: Vec<BaselineMatrix>,
) -> Result<PreparedMatrices, String> {
    let mut matrices = vec![&fused.matrix, &bundle.backtest_labels];
    matrices.extend(baselines.iter().map(|baseline| &baseline.matrix));
    let (dates_ns, instruments) = common_axes(&matrices)?;
    let target_mask = project_mask(
        &fused.matrix.dates_ns,
        &fused.matrix.instruments,
        &fused.present_mask,
        &dates_ns,
        &instruments,
    )?;
    let predictions = project_matrix(&fused.matrix, &dates_ns, &instruments, &target_mask);
    let labels = project_matrix(
        &bundle.backtest_labels,
        &dates_ns,
        &instruments,
        &target_mask,
    );
    let baselines = baselines
        .into_iter()
        .map(|baseline| BaselineMatrix {
            prefix: baseline.prefix,
            display_name: baseline.display_name,
            matrix: project_matrix(&baseline.matrix, &dates_ns, &instruments, &target_mask),
        })
        .collect();
    Ok(PreparedMatrices {
        predictions,
        labels,
        baselines,
        fusion: Some(fused.summary),
    })
}

fn common_axes(matrices: &[&MatrixFrame]) -> Result<(Vec<i64>, Vec<String>), String> {
    let Some((first, rest)) = matrices.split_first() else {
        return Err("no matrices supplied for axis alignment".to_owned());
    };
    let mut dates = first.dates_ns.clone();
    let mut instruments = first.instruments.clone();
    for matrix in rest {
        dates = intersect_i64(&dates, &matrix.dates_ns);
        instruments = intersect_strings(&instruments, &matrix.instruments);
    }
    if dates.is_empty() || instruments.is_empty() {
        return Err("prediction bundle matrices have no common date/instrument axes".to_owned());
    }
    Ok((dates, instruments))
}

fn intersect_i64(left: &[i64], right: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    while left_idx < left.len() && right_idx < right.len() {
        match left[left_idx].cmp(&right[right_idx]) {
            Ordering::Less => left_idx += 1,
            Ordering::Greater => right_idx += 1,
            Ordering::Equal => {
                out.push(left[left_idx]);
                left_idx += 1;
                right_idx += 1;
            }
        }
    }
    out
}

fn intersect_strings(left: &[String], right: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    while left_idx < left.len() && right_idx < right.len() {
        match left[left_idx].cmp(&right[right_idx]) {
            Ordering::Less => left_idx += 1,
            Ordering::Greater => right_idx += 1,
            Ordering::Equal => {
                out.push(left[left_idx].clone());
                left_idx += 1;
                right_idx += 1;
            }
        }
    }
    out
}

fn project_matrix(
    source: &MatrixFrame,
    dates_ns: &[i64],
    instruments: &[String],
    present_mask: &[bool],
) -> MatrixFrame {
    let n_cols = instruments.len();
    let source_n_cols = source.instruments.len();
    let date_pos = source
        .dates_ns
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value, idx))
        .collect::<BTreeMap<_, _>>();
    let instrument_pos = source
        .instruments
        .iter()
        .enumerate()
        .map(|(idx, value)| (value.as_str(), idx))
        .collect::<BTreeMap<_, _>>();
    let mut values = vec![f64::NAN; dates_ns.len() * n_cols];
    for (row, date_ns) in dates_ns.iter().enumerate() {
        let Some(&source_row) = date_pos.get(date_ns) else {
            continue;
        };
        for (col, instrument) in instruments.iter().enumerate() {
            let target_idx = row * n_cols + col;
            if !present_mask[target_idx] {
                continue;
            }
            let Some(&source_col) = instrument_pos.get(instrument.as_str()) else {
                continue;
            };
            values[target_idx] = source.values[source_row * source_n_cols + source_col];
        }
    }
    matrix_from_parts(
        dates_ns.to_vec(),
        instruments.to_vec(),
        values,
        present_mask,
    )
}

fn project_mask(
    source_dates: &[i64],
    source_instruments: &[String],
    source_mask: &[bool],
    target_dates: &[i64],
    target_instruments: &[String],
) -> Result<Vec<bool>, String> {
    let source_n_cols = source_instruments.len();
    let date_pos = source_dates
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value, idx))
        .collect::<BTreeMap<_, _>>();
    let instrument_pos = source_instruments
        .iter()
        .enumerate()
        .map(|(idx, value)| (value.as_str(), idx))
        .collect::<BTreeMap<_, _>>();
    let mut out = vec![false; target_dates.len() * target_instruments.len()];
    for (row, date_ns) in target_dates.iter().enumerate() {
        let source_row = *date_pos
            .get(date_ns)
            .ok_or_else(|| "target date missing from source mask".to_owned())?;
        for (col, instrument) in target_instruments.iter().enumerate() {
            let source_col = *instrument_pos
                .get(instrument.as_str())
                .ok_or_else(|| "target instrument missing from source mask".to_owned())?;
            out[row * target_instruments.len() + col] =
                source_mask[source_row * source_n_cols + source_col];
        }
    }
    Ok(out)
}

fn matrix_from_parts(
    dates_ns: Vec<i64>,
    instruments: Vec<String>,
    values: Vec<f64>,
    present_mask: &[bool],
) -> MatrixFrame {
    MatrixFrame {
        dates_ns,
        instruments,
        finite_values: values.iter().filter(|value| value.is_finite()).count(),
        input_rows: present_mask.iter().filter(|present| **present).count(),
        values,
    }
}

fn parse_score_fusion_config(config: &Value) -> Result<Option<ScoreFusionConfig>, String> {
    if !bool_at(config, &["strategy", "score_fusion", "enabled"], false) {
        return Ok(None);
    }
    let secondary_predictions_dir = raw_string_at(
        config,
        &["strategy", "score_fusion", "secondary_predictions_dir"],
    )
    .ok_or_else(|| {
        "strategy.score_fusion.secondary_predictions_dir is required when score_fusion is enabled"
            .to_owned()
    })?;
    if secondary_predictions_dir.is_empty() {
        return Err(
            "strategy.score_fusion.secondary_predictions_dir cannot be empty when score_fusion is enabled"
                .to_owned(),
        );
    }
    let mode_name = string_at(config, &["strategy", "score_fusion", "mode"])
        .unwrap_or_else(|| "multiply".to_owned());
    let primary_transform_name =
        string_at(config, &["strategy", "score_fusion", "primary_transform"])
            .unwrap_or_else(|| "raw".to_owned());
    let secondary_transform_name =
        string_at(config, &["strategy", "score_fusion", "secondary_transform"])
            .unwrap_or_else(|| "raw".to_owned());
    Ok(Some(ScoreFusionConfig {
        secondary_predictions_dir: PathBuf::from(secondary_predictions_dir),
        mode: fusion_mode_code(&mode_name)?,
        mode_name,
        primary_transform: fusion_transform_code(&primary_transform_name)?,
        primary_transform_name,
        secondary_transform: fusion_transform_code(&secondary_transform_name)?,
        secondary_transform_name,
        primary_power: f64_at(config, &["strategy", "score_fusion", "primary_power"], 1.0)?,
        secondary_power: f64_at(
            config,
            &["strategy", "score_fusion", "secondary_power"],
            1.0,
        )?,
        blend_weight: f64_at(config, &["strategy", "score_fusion", "blend_weight"], 0.5)?,
        filter_threshold: f64_at(
            config,
            &["strategy", "score_fusion", "filter_threshold"],
            0.5,
        )?,
        filter_value: f64_at(config, &["strategy", "score_fusion", "filter_value"], -1.0)?,
    }))
}

fn fuse_prediction_matrices(
    primary: &MatrixFrame,
    secondary: &MatrixFrame,
    config: &ScoreFusionConfig,
) -> Result<FusedMatrix, String> {
    let dates_ns = intersect_i64(&primary.dates_ns, &secondary.dates_ns);
    let common_instruments = intersect_strings(&primary.instruments, &secondary.instruments);
    if dates_ns.is_empty() || common_instruments.is_empty() {
        return Err(
            "No overlapping prediction index between primary and secondary bundles".to_owned(),
        );
    }

    let primary_date_pos = index_i64(&primary.dates_ns);
    let secondary_date_pos = index_i64(&secondary.dates_ns);
    let primary_instrument_pos = index_strings(&primary.instruments);
    let secondary_instrument_pos = index_strings(&secondary.instruments);
    let primary_n_cols = primary.instruments.len();
    let secondary_n_cols = secondary.instruments.len();

    let mut raw_pairs = Vec::new();
    let mut output_date_set = BTreeSet::new();
    let mut output_instrument_set = BTreeSet::new();
    for date_ns in &dates_ns {
        let primary_row = primary_date_pos[date_ns];
        let secondary_row = secondary_date_pos[date_ns];
        for instrument in &common_instruments {
            let primary_col = primary_instrument_pos[instrument.as_str()];
            let secondary_col = secondary_instrument_pos[instrument.as_str()];
            let primary_value = primary.values[primary_row * primary_n_cols + primary_col];
            let secondary_value =
                secondary.values[secondary_row * secondary_n_cols + secondary_col];
            if primary_value.is_nan() || secondary_value.is_nan() {
                continue;
            }
            raw_pairs.push((*date_ns, instrument.clone(), primary_value, secondary_value));
            output_date_set.insert(*date_ns);
            output_instrument_set.insert(instrument.clone());
        }
    }
    if raw_pairs.is_empty() {
        return Err(
            "No overlapping finite prediction pairs between primary and secondary bundles"
                .to_owned(),
        );
    }

    let output_dates = output_date_set.into_iter().collect::<Vec<_>>();
    let output_instruments = output_instrument_set.into_iter().collect::<Vec<_>>();
    let output_date_pos = index_i64(&output_dates);
    let output_instrument_pos = index_strings(&output_instruments);
    let output_n_cols = output_instruments.len();
    let mut pairs = Vec::with_capacity(raw_pairs.len());
    let mut present_mask = vec![false; output_dates.len() * output_n_cols];
    for (date_ns, instrument, primary_value, secondary_value) in raw_pairs {
        let row = output_date_pos[&date_ns];
        let col = output_instrument_pos[instrument.as_str()];
        let target_idx = row * output_n_cols + col;
        present_mask[target_idx] = true;
        pairs.push(FusionPair {
            date_row: row,
            target_idx,
            primary: primary_value,
            secondary: secondary_value,
        });
    }

    let primary_t = build_fusion_transform_values(&pairs, config.primary_transform, true)?;
    let secondary_t = build_fusion_transform_values(&pairs, config.secondary_transform, false)?;
    let mut fused_pair_values = vec![f64::NAN; pairs.len()];
    for idx in 0..pairs.len() {
        let primary_powered = primary_t[idx].powf(config.primary_power);
        let secondary_powered = secondary_t[idx].powf(config.secondary_power);
        fused_pair_values[idx] = match config.mode {
            FUSION_MODE_MULTIPLY => primary_powered * secondary_powered,
            FUSION_MODE_BLEND => {
                config.blend_weight * primary_powered
                    + (1.0 - config.blend_weight) * secondary_powered
            }
            FUSION_MODE_FILTER => {
                if secondary_t[idx] < config.filter_threshold {
                    config.filter_value
                } else {
                    primary_powered
                }
            }
            _ => return Err(format!("unsupported fusion mode code: {}", config.mode)),
        };
    }

    let mut output_values = vec![f64::NAN; output_dates.len() * output_n_cols];
    for (pair, value) in pairs.iter().zip(fused_pair_values.iter()) {
        output_values[pair.target_idx] = *value;
    }
    let summary = FusionSummary {
        fusion_mode: config.mode_name.clone(),
        fusion_primary_transform: config.primary_transform_name.clone(),
        fusion_secondary_transform: config.secondary_transform_name.clone(),
        fusion_primary_power: config.primary_power,
        fusion_secondary_power: config.secondary_power,
        fusion_blend_weight: config.blend_weight,
        fusion_filter_threshold: config.filter_threshold,
        fusion_overlap_rows: pairs.len(),
        fusion_overlap_dates: output_dates.len(),
        fusion_primary_mean: mean_skip_nan(&primary_t),
        fusion_secondary_mean: mean_skip_nan(&secondary_t),
        fusion_output_mean: mean_skip_nan(&fused_pair_values),
        fusion_secondary_prediction_dir: config
            .secondary_predictions_dir
            .to_string_lossy()
            .into_owned(),
    };
    let matrix = matrix_from_parts(
        output_dates,
        output_instruments,
        output_values,
        &present_mask,
    );
    Ok(FusedMatrix {
        matrix,
        present_mask,
        summary,
    })
}

fn build_fusion_transform_values(
    pairs: &[FusionPair],
    transform: u8,
    primary: bool,
) -> Result<Vec<f64>, String> {
    match transform {
        FUSION_TRANSFORM_RAW => Ok(pairs
            .iter()
            .map(|pair| {
                if primary {
                    pair.primary
                } else {
                    pair.secondary
                }
            })
            .collect()),
        FUSION_TRANSFORM_RANK_PCT => {
            let mut out = vec![f64::NAN; pairs.len()];
            let mut by_date: BTreeMap<usize, Vec<(f64, usize)>> = BTreeMap::new();
            for (idx, pair) in pairs.iter().enumerate() {
                let value = if primary {
                    pair.primary
                } else {
                    pair.secondary
                };
                if !value.is_nan() {
                    by_date.entry(pair.date_row).or_default().push((value, idx));
                }
            }
            for group in by_date.values_mut() {
                fill_average_rank_pct(group, &mut out);
            }
            Ok(out)
        }
        _ => Err(format!("unsupported fusion transform code: {transform}")),
    }
}

fn fill_average_rank_pct(pairs: &mut [(f64, usize)], out: &mut [f64]) {
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    let count = pairs.len();
    let mut pos = 0usize;
    while pos < count {
        let mut end = pos + 1;
        while end < count && pairs[end].0 == pairs[pos].0 {
            end += 1;
        }
        let rank_pct = (0.5 * ((pos + end - 1) as f64) + 1.0) / count as f64;
        for idx in pos..end {
            out[pairs[idx].1] = rank_pct;
        }
        pos = end;
    }
}

fn index_i64(values: &[i64]) -> BTreeMap<i64, usize> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value, idx))
        .collect()
}

fn index_strings(values: &[String]) -> BTreeMap<String, usize> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| (value.clone(), idx))
        .collect()
}

fn mean_skip_nan(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for value in values {
        if !value.is_nan() {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn resolve_source_parquet_dir(config: &Value) -> Result<PathBuf, String> {
    if let Some(path) = raw_string_at(config, &["data", "parquet_dir"]) {
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }
    let source = raw_string_at(config, &["data", "source"])
        .unwrap_or_else(|| "akshare".to_owned())
        .trim()
        .to_ascii_lowercase();
    match source.as_str() {
        "tushare" => Ok(PathBuf::from("data/tushare/source")),
        "akshare" | "eastmoney" | "em" => Ok(PathBuf::from("data/processed/combined")),
        other => Err(format!("Unsupported data.source for run-bundle: {other}")),
    }
}

fn resolve_data_source_name(config: &Value) -> Result<String, String> {
    let source = raw_string_at(config, &["data", "source"])
        .unwrap_or_else(|| "akshare".to_owned())
        .trim()
        .to_ascii_lowercase();
    match source.as_str() {
        "akshare" | "eastmoney" | "em" => Ok("akshare".to_owned()),
        "tushare" => Ok("tushare".to_owned()),
        other => Err(format!("Unsupported data.source for run-bundle: {other}")),
    }
}

fn load_benchmark_returns(config: &Value) -> Result<Vec<(i64, f64)>, String> {
    let mode = string_at(config, &["backtest", "benchmark", "mode"])
        .unwrap_or_else(|| "cross_section_mean".to_owned());
    if mode == "cross_section_mean" {
        return Ok(Vec::new());
    }
    if mode != "file" {
        return Err(format!(
            "Unsupported benchmark mode: {mode}. Supported: cross_section_mean, file"
        ));
    }
    let raw_path = raw_string_at(config, &["backtest", "benchmark", "path"]).ok_or_else(|| {
        "backtest.benchmark.path must be set when benchmark.mode == file".to_owned()
    })?;
    let path = PathBuf::from(raw_path);
    let date_column = raw_string_at(config, &["backtest", "benchmark", "date_column"])
        .unwrap_or_else(|| "date".to_owned());
    let value_column = raw_string_at(config, &["backtest", "benchmark", "value_column"])
        .unwrap_or_else(|| "close".to_owned());
    let value_type = string_at(config, &["backtest", "benchmark", "value_type"])
        .unwrap_or_else(|| "close".to_owned());
    let values = match path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
    {
        "csv" | "txt" => read_benchmark_csv(&path, &date_column, &value_column)?,
        "parquet" | "pq" => read_benchmark_parquet(&path, &date_column, &value_column)?,
        _ => {
            return Err(format!(
                "Unsupported benchmark file format: {}. Use .csv, .txt, .parquet, or .pq.",
                path.display()
            ))
        }
    };
    if values.is_empty() {
        return Err(format!(
            "Benchmark returned no usable rows from {}",
            path.display()
        ));
    }
    coerce_benchmark_returns(values, &value_type)
}

fn read_benchmark_csv(
    path: &Path,
    date_column: &str,
    value_column: &str,
) -> Result<Vec<(i64, f64)>, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|err| format!("failed to read benchmark CSV {}: {err}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|err| format!("failed to parse {} headers: {err}", path.display()))?
        .clone();
    let date_idx = headers
        .iter()
        .position(|name| name == date_column)
        .ok_or_else(|| format!("Benchmark file is missing date column: {date_column}"))?;
    let value_idx = headers
        .iter()
        .position(|name| name == value_column)
        .ok_or_else(|| format!("Benchmark file is missing value column: {value_column}"))?;
    let mut by_date = BTreeMap::new();
    for record in reader.records() {
        let record = record.map_err(|err| format!("failed to parse {}: {err}", path.display()))?;
        let Some(raw_date) = record.get(date_idx) else {
            continue;
        };
        let Some(raw_value) = record.get(value_idx) else {
            continue;
        };
        let Ok(date_ns) = parse_datetime_ns(raw_date) else {
            continue;
        };
        let Ok(value) = raw_value.trim().parse::<f64>() else {
            continue;
        };
        if value.is_finite() {
            by_date.insert(date_ns, value);
        }
    }
    Ok(by_date.into_iter().collect())
}

fn read_benchmark_parquet(
    path: &Path,
    date_column: &str,
    value_column: &str,
) -> Result<Vec<(i64, f64)>, String> {
    let reader = open_projected_parquet_reader(path, &[date_column, value_column])?;
    let mut by_date = BTreeMap::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let dates = required_column(&batch, &[date_column], path)?;
        let values = required_column(&batch, &[value_column], path)?;
        for row in 0..batch.num_rows() {
            if dates.is_null(row) || values.is_null(row) {
                continue;
            }
            let date_ns = datetime_ns_at_array(dates, row, path)?;
            let value = f64_at_array(values, row, path)?;
            if value.is_finite() {
                by_date.insert(date_ns, value);
            }
        }
    }
    Ok(by_date.into_iter().collect())
}

fn coerce_benchmark_returns(
    values: Vec<(i64, f64)>,
    value_type: &str,
) -> Result<Vec<(i64, f64)>, String> {
    match value_type {
        "return" => Ok(values),
        "close" => {
            let mut out = Vec::with_capacity(values.len());
            let mut previous: Option<f64> = None;
            for (date_ns, close) in values {
                let daily_return = if let Some(prev) = previous {
                    if prev != 0.0 {
                        close / prev - 1.0
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                out.push((date_ns, daily_return));
                previous = Some(close);
            }
            Ok(out)
        }
        other => Err(format!(
            "Unsupported benchmark value_type: {other}. Supported: return, close"
        )),
    }
}

fn load_instrument_group_ids(config: &Value, instruments: &[String]) -> Result<Vec<i32>, String> {
    let source = resolve_data_source_name(config)?;
    let path = PathBuf::from("data")
        .join(source)
        .join("raw")
        .join("meta")
        .join("symbol_cache.parquet");
    if instruments.is_empty() || !path.exists() {
        return Ok(Vec::new());
    }
    let rows = read_symbol_industry_rows(&path)?;
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let lookup = rows
        .into_iter()
        .filter(|(symbol, industry)| !symbol.is_empty() && !industry.is_empty())
        .map(|(symbol, industry)| (normalize_local_symbol(&symbol), industry))
        .collect::<BTreeMap<_, _>>();
    let mut mapped = 0usize;
    let mut group_to_id: BTreeMap<String, i32> = BTreeMap::new();
    let mut next_id = 0i32;
    let mut out = Vec::with_capacity(instruments.len());
    for instrument in instruments {
        let normalized = normalize_local_symbol(instrument);
        let group = if let Some(industry) = lookup.get(&normalized) {
            mapped += 1;
            industry.clone()
        } else {
            format!("__ungrouped__{instrument}")
        };
        let id = if let Some(id) = group_to_id.get(&group) {
            *id
        } else {
            let id = next_id;
            group_to_id.insert(group, id);
            next_id += 1;
            id
        };
        out.push(id);
    }
    if mapped == 0 {
        Ok(Vec::new())
    } else {
        Ok(out)
    }
}

fn read_symbol_industry_rows(path: &Path) -> Result<Vec<(String, String)>, String> {
    let reader = open_projected_parquet_reader(path, &["local_symbol", "symbol", "industry"])?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let Some(symbols) = optional_column(&batch, &["local_symbol", "symbol"]) else {
            return Ok(Vec::new());
        };
        let Some(industries) = optional_column(&batch, &["industry"]) else {
            return Ok(Vec::new());
        };
        for row in 0..batch.num_rows() {
            if symbols.is_null(row) {
                continue;
            }
            let symbol = string_at_array(symbols, row, path)?;
            let industry = if industries.is_null(row) {
                String::new()
            } else {
                string_at_array(industries, row, path)?
            };
            rows.push((symbol, industry));
        }
    }
    Ok(rows)
}

fn load_price_confirm_matrix(
    source_data_dir: &Path,
    dates_ns: &[i64],
    instruments: &[String],
    config: &PriceConfirmConfig,
) -> Result<Vec<u8>, String> {
    if dates_ns.is_empty() || instruments.is_empty() {
        return Ok(Vec::new());
    }
    let close_values = load_close_matrix(source_data_dir, dates_ns, instruments)?;
    Ok(build_price_confirm_values(
        &close_values,
        dates_ns.len(),
        instruments.len(),
        config.ma_window,
    ))
}

fn load_close_matrix(
    source_data_dir: &Path,
    dates_ns: &[i64],
    instruments: &[String],
) -> Result<Vec<f64>, String> {
    let n_dates = dates_ns.len();
    let n_cols = instruments.len();
    let mut out = vec![f64::NAN; n_dates * n_cols];
    let date_pos = dates_ns
        .iter()
        .enumerate()
        .map(|(idx, date_ns)| (*date_ns, idx))
        .collect::<BTreeMap<_, _>>();
    let min_date = *dates_ns.first().unwrap_or(&i64::MIN);
    let max_date = *dates_ns.last().unwrap_or(&i64::MAX);
    let mut instrument_pos = HashMap::new();
    let mut allowed_symbols = HashSet::new();
    for (idx, instrument) in instruments.iter().enumerate() {
        instrument_pos.insert(instrument.clone(), idx);
        instrument_pos.insert(normalize_local_symbol(instrument), idx);
        allowed_symbols.insert(instrument.clone());
        allowed_symbols.insert(normalize_local_symbol(instrument));
    }

    if source_data_dir.join("buckets").is_dir() && source_data_dir.join("manifest.parquet").exists()
    {
        let bucket_ids = load_source_bucket_ids(source_data_dir, &allowed_symbols)?;
        for bucket_id in bucket_ids {
            let path = source_data_dir
                .join("buckets")
                .join(format!("part-{bucket_id:04}.parquet"));
            if path.exists() {
                append_close_rows_from_parquet(
                    &path,
                    &date_pos,
                    &instrument_pos,
                    &allowed_symbols,
                    min_date,
                    max_date,
                    n_cols,
                    &mut out,
                )?;
            }
        }
        return Ok(out);
    }

    for instrument in instruments {
        let candidates = [
            source_data_dir.join(format!("{instrument}.parquet")),
            source_data_dir.join(format!("{}.parquet", normalize_local_symbol(instrument))),
        ];
        for path in candidates {
            if path.exists() {
                append_close_rows_from_parquet(
                    &path,
                    &date_pos,
                    &instrument_pos,
                    &allowed_symbols,
                    min_date,
                    max_date,
                    n_cols,
                    &mut out,
                )?;
                break;
            }
        }
    }
    Ok(out)
}

fn load_source_bucket_ids(
    source_data_dir: &Path,
    allowed_symbols: &HashSet<String>,
) -> Result<Vec<usize>, String> {
    let path = source_data_dir.join("manifest.parquet");
    let reader = open_projected_parquet_reader(
        &path,
        &["symbol", "local_symbol", "instrument", "bucket_id"],
    )?;
    let mut bucket_ids = BTreeSet::new();
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let symbols = required_column(&batch, &["symbol", "local_symbol", "instrument"], &path)?;
        let buckets = required_column(&batch, &["bucket_id"], &path)?;
        for row in 0..batch.num_rows() {
            if symbols.is_null(row) || buckets.is_null(row) {
                continue;
            }
            let symbol = string_at_array(symbols, row, &path)?;
            if allowed_symbols.contains(&symbol)
                || allowed_symbols.contains(&normalize_local_symbol(&symbol))
            {
                let bucket_id = usize_at_array(buckets, row, &path)?;
                bucket_ids.insert(bucket_id);
            }
        }
    }
    Ok(bucket_ids.into_iter().collect())
}

#[allow(clippy::too_many_arguments)]
fn append_close_rows_from_parquet(
    path: &Path,
    date_pos: &BTreeMap<i64, usize>,
    instrument_pos: &HashMap<String, usize>,
    allowed_symbols: &HashSet<String>,
    min_date: i64,
    max_date: i64,
    n_cols: usize,
    out: &mut [f64],
) -> Result<(), String> {
    let reader = open_projected_parquet_reader(
        path,
        &[
            "date",
            "datetime",
            "trade_date",
            "symbol",
            "instrument",
            "local_symbol",
            "close",
        ],
    )?;
    for batch in reader {
        let batch = batch.map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        let dates = required_column(&batch, &["date", "datetime", "trade_date"], path)?;
        let symbols = optional_column(&batch, &["symbol", "instrument", "local_symbol"]);
        let close = required_column(&batch, &["close"], path)?;
        for row in 0..batch.num_rows() {
            if dates.is_null(row) || close.is_null(row) {
                continue;
            }
            let date_ns = datetime_ns_at_array(dates, row, path)?;
            if date_ns < min_date || date_ns > max_date {
                continue;
            }
            let Some(&date_idx) = date_pos.get(&date_ns) else {
                continue;
            };
            let symbol = if let Some(symbols) = symbols {
                if symbols.is_null(row) {
                    continue;
                }
                string_at_array(symbols, row, path)?
            } else {
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("")
                    .to_owned()
            };
            let normalized = normalize_local_symbol(&symbol);
            if !allowed_symbols.contains(&symbol) && !allowed_symbols.contains(&normalized) {
                continue;
            }
            let Some(&col_idx) = instrument_pos
                .get(&symbol)
                .or_else(|| instrument_pos.get(&normalized))
            else {
                continue;
            };
            out[date_idx * n_cols + col_idx] = f64_at_array(close, row, path)?;
        }
    }
    Ok(())
}

fn build_price_confirm_values(
    close_values: &[f64],
    n_dates: usize,
    n_cols: usize,
    ma_window: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; close_values.len()];
    let window = ma_window.max(1);
    let mut running_sum = vec![0.0; n_cols];
    let mut running_count = vec![0usize; n_cols];
    for row in 0..n_dates {
        for col in 0..n_cols {
            let idx = row * n_cols + col;
            let value = close_values[idx];
            if value.is_finite() {
                running_sum[col] += value;
                running_count[col] += 1;
            }
            if row >= window {
                let old = close_values[(row - window) * n_cols + col];
                if old.is_finite() {
                    running_sum[col] -= old;
                    running_count[col] -= 1;
                }
            }
            if row + 1 >= window && running_count[col] == window {
                let mean = running_sum[col] / window as f64;
                out[idx] = u8::from(value.is_finite() && value < mean);
            }
        }
    }
    out
}

fn required_column<'a>(
    batch: &'a RecordBatch,
    candidates: &[&str],
    path: &Path,
) -> Result<&'a dyn Array, String> {
    optional_column(batch, candidates).ok_or_else(|| {
        format!(
            "{} is missing required column: {}",
            path.display(),
            candidates.join(" or ")
        )
    })
}

fn open_projected_parquet_reader(
    path: &Path,
    column_candidates: &[&str],
) -> Result<ParquetRecordBatchReader, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to open parquet {}: {err}", path.display()))?;
    let indices = column_candidates
        .iter()
        .filter_map(|name| builder.schema().index_of(name).ok())
        .collect::<BTreeSet<_>>();
    let mask = ProjectionMask::roots(builder.parquet_schema(), indices);
    builder
        .with_projection(mask)
        .with_batch_size(65_536)
        .build()
        .map_err(|err| {
            format!(
                "failed to build parquet reader for {}: {err}",
                path.display()
            )
        })
}

fn optional_column<'a>(batch: &'a RecordBatch, candidates: &[&str]) -> Option<&'a dyn Array> {
    candidates.iter().find_map(|name| {
        batch
            .schema()
            .index_of(name)
            .ok()
            .map(|idx| batch.column(idx).as_ref())
    })
}

fn parse_datetime_ns(raw: &str) -> Result<i64, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("empty datetime".to_owned());
    }
    if text.len() == 8 && text.chars().all(|ch| ch.is_ascii_digit()) {
        let date = NaiveDate::parse_from_str(text, "%Y%m%d")
            .map_err(|err| format!("failed to parse date {text}: {err}"))?;
        return naive_datetime_to_ns(
            date.and_hms_opt(0, 0, 0)
                .ok_or_else(|| format!("invalid date {text}"))?,
        );
    }
    if let Ok(date) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return naive_datetime_to_ns(
            date.and_hms_opt(0, 0, 0)
                .ok_or_else(|| format!("invalid date {text}"))?,
        );
    }
    for fmt in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(datetime) = NaiveDateTime::parse_from_str(text, fmt) {
            return naive_datetime_to_ns(datetime);
        }
    }
    if let Ok(datetime) = DateTime::parse_from_rfc3339(text) {
        return datetime
            .timestamp_nanos_opt()
            .ok_or_else(|| format!("datetime out of range: {text}"));
    }
    Err(format!("failed to parse datetime: {text}"))
}

fn naive_datetime_to_ns(datetime: NaiveDateTime) -> Result<i64, String> {
    datetime
        .and_utc()
        .timestamp_nanos_opt()
        .ok_or_else(|| format!("datetime out of range: {datetime}"))
}

fn datetime_ns_at_array(array: &dyn Array, row: usize, path: &Path) -> Result<i64, String> {
    if array.is_null(row) {
        return Err(format!("{} has null datetime at row {row}", path.display()));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return Ok(values.value(row) * 1_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return Ok(values.value(row) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return Ok(values.value(row) * 1_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return Ok(values.value(row) as i64 * 86_400_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date64Array>() {
        return Ok(values.value(row) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return parse_datetime_ns(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return parse_datetime_ns(values.value(row));
    }
    Err(format!(
        "{} has unsupported datetime type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn string_at_array(array: &dyn Array, row: usize, path: &Path) -> Result<String, String> {
    if array.is_null(row) {
        return Ok(String::new());
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(values.value(row).trim().to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(values.value(row).trim().to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(values.value(row).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row).to_string());
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(values.value(row).to_string());
    }
    Err(format!(
        "{} has unsupported string-like type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn f64_at_array(array: &dyn Array, row: usize, path: &Path) -> Result<f64, String> {
    if array.is_null(row) {
        return Ok(f64::NAN);
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(values.value(row) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return values.value(row).trim().parse::<f64>().map_err(|err| {
            format!(
                "{} has non-numeric value at row {row}: {err}",
                path.display()
            )
        });
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return values.value(row).trim().parse::<f64>().map_err(|err| {
            format!(
                "{} has non-numeric value at row {row}: {err}",
                path.display()
            )
        });
    }
    Err(format!(
        "{} has unsupported numeric type {:?}",
        path.display(),
        array.data_type()
    ))
}

fn usize_at_array(array: &dyn Array, row: usize, path: &Path) -> Result<usize, String> {
    let value = f64_at_array(array, row, path)?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "{} has invalid usize value at row {row}",
            path.display()
        ));
    }
    Ok(value as usize)
}

fn normalize_local_symbol(value: &str) -> String {
    let text = value.trim();
    if text.chars().all(|ch| ch.is_ascii_digit()) && text.len() <= 6 {
        format!("{text:0>6}")
    } else {
        text.to_owned()
    }
}

fn baseline_display_name(prefix: &str) -> &str {
    BASELINE_DISPLAY_NAMES
        .iter()
        .find_map(|(candidate, display)| (*candidate == prefix).then_some(*display))
        .unwrap_or(prefix)
}

fn fusion_transform_code(value: &str) -> Result<u8, String> {
    match value {
        "raw" => Ok(FUSION_TRANSFORM_RAW),
        "rank_pct" => Ok(FUSION_TRANSFORM_RANK_PCT),
        other => Err(format!(
            "unsupported strategy.score_fusion transform for run-bundle: {other}"
        )),
    }
}

fn fusion_mode_code(value: &str) -> Result<u8, String> {
    match value {
        "multiply" => Ok(FUSION_MODE_MULTIPLY),
        "blend" => Ok(FUSION_MODE_BLEND),
        "filter" => Ok(FUSION_MODE_FILTER),
        other => Err(format!(
            "unsupported strategy.score_fusion.mode for run-bundle: {other}"
        )),
    }
}

fn parse_options(args: &[String]) -> Result<RunBundleOptions, String> {
    let mut bundle_dir: Option<PathBuf> = None;
    let mut config_path: Option<PathBuf> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut execution = RunBundleExecutionOptions::default();
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--bundle" | "--bundle-dir" | "--load-predictions-dir" => {
                bundle_dir = Some(PathBuf::from(next_arg(args, &mut idx, "--bundle")?));
            }
            "--config" => {
                config_path = Some(PathBuf::from(next_arg(args, &mut idx, "--config")?));
            }
            "--output-dir" | "--results-dir" => {
                output_dir = Some(PathBuf::from(next_arg(args, &mut idx, "--output-dir")?));
            }
            "--skip-reference-baselines" => execution.skip_reference_baselines = true,
            "--skip-backtest-plots" => execution.skip_backtest_plots = true,
            "--baseline-jobs" => {
                execution.baseline_jobs =
                    parse_positive_usize(&next_arg(args, &mut idx, "--baseline-jobs")?)
            }
            "-h" | "--help" => return Err(usage().to_owned()),
            other => return Err(format!("unknown run-bundle option: {other}\n\n{}", usage())),
        }
        idx += 1;
    }
    let bundle_dir = bundle_dir.ok_or_else(|| format!("--bundle is required\n\n{}", usage()))?;
    let config_path = config_path.ok_or_else(|| format!("--config is required\n\n{}", usage()))?;
    let output_dir = output_dir.unwrap_or_else(|| PathBuf::from("results/rust_bundle_backtest"));
    Ok(RunBundleOptions {
        bundle_dir,
        config_path,
        output_dir,
        execution,
    })
}

fn usage() -> &'static str {
    "\
Usage:
  ai4stock-backtest run-bundle --bundle <PATH> --config <config_snapshot.yaml> [--output-dir <PATH>] [--skip-reference-baselines] [--skip-backtest-plots] [--baseline-jobs <N>]

Run the native Rust post-bundle backtest path from an existing prediction bundle.
"
}

fn next_arg(args: &[String], idx: &mut usize, option: &str) -> Result<String, String> {
    *idx += 1;
    args.get(*idx)
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn parse_positive_usize(raw: &str) -> usize {
    raw.trim().parse::<usize>().unwrap_or(1).max(1)
}

fn read_yaml(path: &Path) -> Result<Value, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    serde_yaml::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn reject_unsupported_config(config: &Value) -> Result<(), String> {
    let _ = config;
    Ok(())
}

fn parse_backtest_params(config: &Value) -> Result<NativeBacktestParams, String> {
    let topk = usize_at(config, &["strategy", "topk"], 30)?;
    let n_drop = usize_at(config, &["strategy", "n_drop"], 5)?;
    let rebalance_freq = usize_at(config, &["backtest", "rebalance_freq"], 1)?.max(1);
    let account = f64_at(config, &["backtest", "account"], 100_000_000.0)?;
    let risk_degree = validate_risk_degree_rs(
        f64_at(config, &["backtest", "risk_degree"], 0.95)?,
        "backtest.risk_degree",
    )?;
    let slippage = f64_at(config, &["backtest", "slippage"], 0.0)?;
    let buy = f64_at(config, &["backtest", "cost", "buy"], 0.001)?;
    let sell = f64_at(config, &["backtest", "cost", "sell"], 0.001)?;
    let weighting =
        string_at(config, &["strategy", "weighting"]).unwrap_or_else(|| "equal".to_owned());
    let score_transform =
        string_at(config, &["strategy", "score_transform"]).unwrap_or_else(|| "none".to_owned());
    let desticky_n_drop = optional_usize_at(config, &["strategy", "desticky_n_drop"])?;
    if let Some(value) = desticky_n_drop {
        if value >= topk {
            return Err("strategy.desticky_n_drop must be smaller than strategy.topk".to_owned());
        }
    }
    let risk_control = parse_risk_control_config(config, risk_degree)?;
    let intraperiod_exit = parse_intraperiod_exit_config(config)?;
    Ok(NativeBacktestParams {
        topk: topk.max(1),
        n_drop,
        rebalance_freq,
        account,
        risk_degree,
        open_rate: buy + slippage,
        close_rate: sell + slippage,
        min_cost: f64_at(config, &["backtest", "min_cost"], 5.0)?,
        weighting_mode: weighting_code(&weighting)?,
        score_transform_mode: score_transform_code(&score_transform)?,
        zscore_clip: f64_at(config, &["strategy", "score_zscore_clip"], 3.0)?.max(0.0),
        max_weight: optional_f64_at(config, &["strategy", "max_weight"])?,
        max_industry_weight: optional_f64_at(config, &["strategy", "max_industry_weight"])?
            .map(|value| validate_risk_degree_rs(value, "strategy.max_industry_weight"))
            .transpose()?,
        keep_top_n: optional_usize_at(config, &["strategy", "keep_top_n"])?,
        min_score: optional_f64_at(config, &["strategy", "min_score"])?,
        desticky_threshold: optional_f64_at(config, &["strategy", "desticky_signal_threshold"])?,
        desticky_n_drop,
        risk_control,
        intraperiod_exit,
    })
}

fn validate_risk_degree_rs(value: f64, field: &str) -> Result<f64, String> {
    if !(0.0..=1.0).contains(&value) {
        return Err(format!("{field} must be in [0, 1]"));
    }
    Ok(value)
}

fn validate_optional_quantile(value: Option<f64>, field: &str) -> Result<(), String> {
    if let Some(value) = value {
        if !(0.0..=1.0).contains(&value) {
            return Err(format!("{field} must be in [0, 1]"));
        }
    }
    Ok(())
}

fn parse_risk_control_config(
    config: &Value,
    fallback_risk_degree: f64,
) -> Result<RiskControlConfig, String> {
    let Some(_risk_cfg) = get_value(config, &["backtest", "risk_control"]) else {
        return Ok(RiskControlConfig {
            mode: "fixed".to_owned(),
            risk_degree: validate_risk_degree_rs(fallback_risk_degree, "risk_control.risk_degree")?,
            signal_metric: "topk_mean".to_owned(),
            signal_source: "score_strength".to_owned(),
            validation_metric: "valid_topk_label_mean".to_owned(),
            secondary_validation_metric: None,
            min_signal: 0.0,
            max_signal: 2.0,
            min_signal_quantile: None,
            max_signal_quantile: None,
            min_risk: 0.0,
            max_risk: validate_risk_degree_rs(fallback_risk_degree, "risk_control.max_risk")?,
            secondary_min_signal: 0.0,
            secondary_max_signal: 2.0,
            secondary_min_signal_quantile: None,
            secondary_max_signal_quantile: None,
            secondary_min_risk: 0.0,
            secondary_max_risk: validate_risk_degree_rs(
                fallback_risk_degree,
                "risk_control.secondary_max_risk",
            )?,
            risk_curve: "linear".to_owned(),
            risk_curve_power: 2.0,
            risk_curve_center: 0.5,
            risk_curve_steepness: 8.0,
            fast_window: 120,
            slow_window: 250,
            bull_risk: validate_risk_degree_rs(fallback_risk_degree, "risk_control.bull_risk")?,
            neutral_risk: validate_risk_degree_rs(
                fallback_risk_degree.min(0.5),
                "risk_control.neutral_risk",
            )?,
            bear_risk: 0.15,
        });
    };
    let mode = string_at(config, &["backtest", "risk_control", "mode"])
        .unwrap_or_else(|| "fixed".to_owned());
    if mode == "fixed" {
        let risk_degree = optional_f64_at(config, &["backtest", "risk_control", "risk_degree"])?
            .or_else(|| {
                optional_f64_at(config, &["backtest", "risk_degree"])
                    .ok()
                    .flatten()
            })
            .unwrap_or(fallback_risk_degree);
        return Ok(RiskControlConfig {
            mode,
            risk_degree: validate_risk_degree_rs(risk_degree, "risk_control.risk_degree")?,
            signal_metric: "topk_mean".to_owned(),
            signal_source: "score_strength".to_owned(),
            validation_metric: "valid_topk_label_mean".to_owned(),
            secondary_validation_metric: None,
            min_signal: 0.0,
            max_signal: 2.0,
            min_signal_quantile: None,
            max_signal_quantile: None,
            min_risk: validate_risk_degree_rs(
                fallback_risk_degree.min(0.3),
                "risk_control.min_risk",
            )?,
            max_risk: validate_risk_degree_rs(risk_degree, "risk_control.max_risk")?,
            secondary_min_signal: 0.0,
            secondary_max_signal: 2.0,
            secondary_min_signal_quantile: None,
            secondary_max_signal_quantile: None,
            secondary_min_risk: validate_risk_degree_rs(
                fallback_risk_degree.min(0.3),
                "risk_control.secondary_min_risk",
            )?,
            secondary_max_risk: validate_risk_degree_rs(
                risk_degree,
                "risk_control.secondary_max_risk",
            )?,
            risk_curve: "linear".to_owned(),
            risk_curve_power: 2.0,
            risk_curve_center: 0.5,
            risk_curve_steepness: 8.0,
            fast_window: 120,
            slow_window: 250,
            bull_risk: validate_risk_degree_rs(risk_degree, "risk_control.bull_risk")?,
            neutral_risk: validate_risk_degree_rs(
                fallback_risk_degree.min(0.5),
                "risk_control.neutral_risk",
            )?,
            bear_risk: 0.15,
        });
    }

    if mode == "signal_strength" || mode == "benchmark_ma_signal_strength" {
        let signal_metric = string_at(config, &["backtest", "risk_control", "signal_metric"])
            .unwrap_or_else(|| "topk_mean".to_owned());
        let signal_source = string_at(config, &["backtest", "risk_control", "signal_source"])
            .unwrap_or_else(|| "score_strength".to_owned());
        let min_signal = f64_at(config, &["backtest", "risk_control", "min_signal"], 0.0)?;
        let max_signal = f64_at(config, &["backtest", "risk_control", "max_signal"], 2.0)?;
        if max_signal <= min_signal {
            return Err(
                "risk_control.max_signal must be greater than risk_control.min_signal".to_owned(),
            );
        }
        let min_signal_quantile =
            optional_f64_at(config, &["backtest", "risk_control", "min_signal_quantile"])?;
        let max_signal_quantile =
            optional_f64_at(config, &["backtest", "risk_control", "max_signal_quantile"])?;
        validate_optional_quantile(min_signal_quantile, "risk_control.min_signal_quantile")?;
        validate_optional_quantile(max_signal_quantile, "risk_control.max_signal_quantile")?;
        if let (Some(left), Some(right)) = (min_signal_quantile, max_signal_quantile) {
            if right <= left {
                return Err("risk_control.max_signal_quantile must be greater than risk_control.min_signal_quantile".to_owned());
            }
        }
        let min_risk = validate_risk_degree_rs(
            f64_at(
                config,
                &["backtest", "risk_control", "min_risk"],
                fallback_risk_degree.min(0.3),
            )?,
            "risk_control.min_risk",
        )?;
        let max_risk = validate_risk_degree_rs(
            f64_at(
                config,
                &["backtest", "risk_control", "max_risk"],
                fallback_risk_degree,
            )?,
            "risk_control.max_risk",
        )?;
        if max_risk < min_risk {
            return Err("risk_control.max_risk must be >= risk_control.min_risk".to_owned());
        }
        let risk_curve = string_at(config, &["backtest", "risk_control", "risk_curve"])
            .unwrap_or_else(|| "linear".to_owned());
        risk_curve_code(&risk_curve)?;
        let risk_curve_power = f64_at(
            config,
            &["backtest", "risk_control", "risk_curve_power"],
            2.0,
        )?;
        if risk_curve_power <= 0.0 {
            return Err("risk_control.risk_curve_power must be > 0".to_owned());
        }
        let risk_curve_center = f64_at(
            config,
            &["backtest", "risk_control", "risk_curve_center"],
            0.5,
        )?;
        if !(0.0..=1.0).contains(&risk_curve_center) {
            return Err("risk_control.risk_curve_center must be in [0, 1]".to_owned());
        }
        let risk_curve_steepness = f64_at(
            config,
            &["backtest", "risk_control", "risk_curve_steepness"],
            8.0,
        )?;
        if risk_curve_steepness <= 0.0 {
            return Err("risk_control.risk_curve_steepness must be > 0".to_owned());
        }
        let mut out = RiskControlConfig {
            mode: mode.clone(),
            risk_degree: validate_risk_degree_rs(fallback_risk_degree, "risk_control.risk_degree")?,
            signal_metric,
            signal_source,
            validation_metric: "valid_topk_label_mean".to_owned(),
            secondary_validation_metric: None,
            min_signal,
            max_signal,
            min_signal_quantile,
            max_signal_quantile,
            min_risk,
            max_risk,
            secondary_min_signal: min_signal,
            secondary_max_signal: max_signal,
            secondary_min_signal_quantile: None,
            secondary_max_signal_quantile: None,
            secondary_min_risk: min_risk,
            secondary_max_risk: max_risk,
            risk_curve,
            risk_curve_power,
            risk_curve_center,
            risk_curve_steepness,
            fast_window: 120,
            slow_window: 250,
            bull_risk: fallback_risk_degree,
            neutral_risk: fallback_risk_degree.min(0.5),
            bear_risk: 0.15,
        };
        if out.signal_source == "validation_metric" {
            let validation_metric =
                string_at(config, &["backtest", "risk_control", "validation_metric"])
                    .unwrap_or_else(|| "valid_topk_label_mean".to_owned());
            out.validation_metric = validation_metric;
            if let Some(secondary_validation_metric) = string_at(
                config,
                &["backtest", "risk_control", "secondary_validation_metric"],
            ) {
                out.secondary_validation_metric = Some(secondary_validation_metric);
                out.secondary_min_signal = f64_at(
                    config,
                    &["backtest", "risk_control", "secondary_min_signal"],
                    min_signal,
                )?;
                out.secondary_max_signal = f64_at(
                    config,
                    &["backtest", "risk_control", "secondary_max_signal"],
                    max_signal,
                )?;
                if out.secondary_max_signal <= out.secondary_min_signal {
                    return Err("risk_control.secondary_max_signal must be greater than risk_control.secondary_min_signal".to_owned());
                }
                out.secondary_min_signal_quantile = optional_f64_at(
                    config,
                    &["backtest", "risk_control", "secondary_min_signal_quantile"],
                )?;
                out.secondary_max_signal_quantile = optional_f64_at(
                    config,
                    &["backtest", "risk_control", "secondary_max_signal_quantile"],
                )?;
                validate_optional_quantile(
                    out.secondary_min_signal_quantile,
                    "risk_control.secondary_min_signal_quantile",
                )?;
                validate_optional_quantile(
                    out.secondary_max_signal_quantile,
                    "risk_control.secondary_max_signal_quantile",
                )?;
                if let (Some(left), Some(right)) = (
                    out.secondary_min_signal_quantile,
                    out.secondary_max_signal_quantile,
                ) {
                    if right <= left {
                        return Err("risk_control.secondary_max_signal_quantile must be greater than risk_control.secondary_min_signal_quantile".to_owned());
                    }
                }
                out.secondary_min_risk = validate_risk_degree_rs(
                    f64_at(
                        config,
                        &["backtest", "risk_control", "secondary_min_risk"],
                        min_risk,
                    )?,
                    "risk_control.secondary_min_risk",
                )?;
                out.secondary_max_risk = validate_risk_degree_rs(
                    f64_at(
                        config,
                        &["backtest", "risk_control", "secondary_max_risk"],
                        max_risk,
                    )?,
                    "risk_control.secondary_max_risk",
                )?;
                if out.secondary_max_risk < out.secondary_min_risk {
                    return Err("risk_control.secondary_max_risk must be >= risk_control.secondary_min_risk".to_owned());
                }
            }
        }
        if mode == "signal_strength" {
            return Ok(out);
        }
        let fast_window = max(
            1,
            usize_at(config, &["backtest", "risk_control", "fast_window"], 120)?,
        );
        let slow_window = max(
            1,
            usize_at(config, &["backtest", "risk_control", "slow_window"], 250)?,
        );
        if fast_window >= slow_window {
            return Err(
                "risk_control.fast_window must be smaller than risk_control.slow_window".to_owned(),
            );
        }
        out.fast_window = fast_window;
        out.slow_window = slow_window;
        out.bull_risk = validate_risk_degree_rs(
            f64_at(
                config,
                &["backtest", "risk_control", "bull_risk"],
                fallback_risk_degree,
            )?,
            "risk_control.bull_risk",
        )?;
        out.neutral_risk = validate_risk_degree_rs(
            f64_at(
                config,
                &["backtest", "risk_control", "neutral_risk"],
                fallback_risk_degree.min(0.5),
            )?,
            "risk_control.neutral_risk",
        )?;
        out.bear_risk = validate_risk_degree_rs(
            f64_at(config, &["backtest", "risk_control", "bear_risk"], 0.15)?,
            "risk_control.bear_risk",
        )?;
        return Ok(out);
    }

    if mode != "benchmark_ma" {
        return Err(format!(
            "Unsupported risk control mode: {mode}. Supported: fixed, benchmark_ma, signal_strength, benchmark_ma_signal_strength"
        ));
    }
    let fast_window = max(
        1,
        usize_at(config, &["backtest", "risk_control", "fast_window"], 120)?,
    );
    let slow_window = max(
        1,
        usize_at(config, &["backtest", "risk_control", "slow_window"], 250)?,
    );
    if fast_window >= slow_window {
        return Err(
            "risk_control.fast_window must be smaller than risk_control.slow_window".to_owned(),
        );
    }
    Ok(RiskControlConfig {
        mode,
        risk_degree: validate_risk_degree_rs(fallback_risk_degree, "risk_control.risk_degree")?,
        signal_metric: "topk_mean".to_owned(),
        signal_source: "score_strength".to_owned(),
        validation_metric: "valid_topk_label_mean".to_owned(),
        secondary_validation_metric: None,
        min_signal: 0.0,
        max_signal: 2.0,
        min_signal_quantile: None,
        max_signal_quantile: None,
        min_risk: fallback_risk_degree.min(0.3),
        max_risk: validate_risk_degree_rs(fallback_risk_degree, "risk_control.max_risk")?,
        secondary_min_signal: 0.0,
        secondary_max_signal: 2.0,
        secondary_min_signal_quantile: None,
        secondary_max_signal_quantile: None,
        secondary_min_risk: fallback_risk_degree.min(0.3),
        secondary_max_risk: validate_risk_degree_rs(
            fallback_risk_degree,
            "risk_control.secondary_max_risk",
        )?,
        risk_curve: "linear".to_owned(),
        risk_curve_power: 2.0,
        risk_curve_center: 0.5,
        risk_curve_steepness: 8.0,
        fast_window,
        slow_window,
        bull_risk: validate_risk_degree_rs(fallback_risk_degree, "risk_control.bull_risk")?,
        neutral_risk: validate_risk_degree_rs(
            fallback_risk_degree.min(0.5),
            "risk_control.neutral_risk",
        )?,
        bear_risk: 0.15,
    })
}

fn parse_intraperiod_exit_config(config: &Value) -> Result<Option<IntraperiodExitConfig>, String> {
    let Some(exit_cfg) = get_value(config, &["backtest", "intraperiod_exit"]) else {
        return Ok(None);
    };
    let mode_name = string_at(config, &["backtest", "intraperiod_exit", "mode"])
        .unwrap_or_else(|| "none".to_owned());
    if mode_name == "none" {
        return Ok(None);
    }
    let score_source = string_at(config, &["backtest", "intraperiod_exit", "score_source"])
        .unwrap_or_else(|| "raw".to_owned());
    let threshold = f64_at(config, &["backtest", "intraperiod_exit", "threshold"], 0.0)?;
    let (mode, n_bins, min_history) = match mode_name.as_str() {
        "score_threshold" => (INTRAPERIOD_EXIT_SCORE_THRESHOLD, 20, 200),
        "expected_return_threshold" => {
            let calibration = string_at(config, &["backtest", "intraperiod_exit", "calibration"])
                .unwrap_or_else(|| "quantile_bins".to_owned());
            if calibration != "quantile_bins" {
                return Err(format!(
                    "Unsupported intraperiod exit calibration: {calibration}. Supported: quantile_bins"
                ));
            }
            (
                INTRAPERIOD_EXIT_EXPECTED_RETURN,
                max(
                    2,
                    usize_at(config, &["backtest", "intraperiod_exit", "n_bins"], 20)?,
                ),
                max(
                    1,
                    usize_at(
                        config,
                        &["backtest", "intraperiod_exit", "min_history"],
                        200,
                    )?,
                ),
            )
        }
        other => {
            return Err(format!(
                "Unsupported intraperiod exit mode: {other}. Supported: none, score_threshold, expected_return_threshold"
            ));
        }
    };
    let price_confirm = if get_value(exit_cfg, &["price_confirm"]).is_some() {
        let confirm_mode = string_at(
            config,
            &["backtest", "intraperiod_exit", "price_confirm", "mode"],
        )
        .unwrap_or_else(|| "close_below_ma".to_owned());
        if confirm_mode != "close_below_ma" {
            return Err(format!(
                "Unsupported intraperiod_exit.price_confirm.mode: {confirm_mode}. Supported: close_below_ma"
            ));
        }
        let signal_timing = string_at(
            config,
            &[
                "backtest",
                "intraperiod_exit",
                "price_confirm",
                "signal_timing",
            ],
        )
        .unwrap_or_else(|| "same_signal_date_close".to_owned());
        let execution_timing = string_at(
            config,
            &[
                "backtest",
                "intraperiod_exit",
                "price_confirm",
                "execution_timing",
            ],
        )
        .unwrap_or_else(|| "next_open".to_owned());
        if signal_timing != "same_signal_date_close" {
            return Err(
                "intraperiod_exit.price_confirm.signal_timing must be same_signal_date_close"
                    .to_owned(),
            );
        }
        if execution_timing != "next_open" {
            return Err(
                "intraperiod_exit.price_confirm.execution_timing must be next_open".to_owned(),
            );
        }
        Some(PriceConfirmConfig {
            ma_window: max(
                1,
                usize_at(
                    config,
                    &["backtest", "intraperiod_exit", "price_confirm", "ma_window"],
                    10,
                )?,
            ),
            min_remaining_steps: max(
                0,
                usize_at(
                    config,
                    &[
                        "backtest",
                        "intraperiod_exit",
                        "price_confirm",
                        "min_remaining_steps",
                    ],
                    0,
                )?,
            ),
            force_exit_threshold: optional_f64_at(
                config,
                &[
                    "backtest",
                    "intraperiod_exit",
                    "price_confirm",
                    "force_exit_threshold",
                ],
            )?,
            signal_timing,
            execution_timing,
        })
    } else {
        None
    };
    let score_transform_mode = intraperiod_score_transform_code(&score_source)?;
    let _ = exit_cfg;
    Ok(Some(IntraperiodExitConfig {
        mode,
        score_transform_mode,
        threshold,
        n_bins,
        min_history,
        price_confirm,
    }))
}

fn get_value<'a>(root: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = root;
    for key in path {
        current = current.get(*key)?;
    }
    if current.is_null() {
        None
    } else {
        Some(current)
    }
}

fn f64_at(root: &Value, path: &[&str], default: f64) -> Result<f64, String> {
    match get_value(root, path) {
        Some(value) => value_to_f64(value, path),
        None => Ok(default),
    }
}

fn optional_f64_at(root: &Value, path: &[&str]) -> Result<Option<f64>, String> {
    get_value(root, path)
        .map(|value| value_to_f64(value, path))
        .transpose()
}

fn value_to_f64(value: &Value, path: &[&str]) -> Result<f64, String> {
    if let Some(value) = value.as_f64() {
        return Ok(value);
    }
    if let Some(value) = value.as_i64() {
        return Ok(value as f64);
    }
    if let Some(value) = value.as_str() {
        return value
            .parse::<f64>()
            .map_err(|err| format!("{} must be a number: {err}", path.join(".")));
    }
    Err(format!("{} must be a number", path.join(".")))
}

fn usize_at(root: &Value, path: &[&str], default: usize) -> Result<usize, String> {
    match optional_usize_at(root, path)? {
        Some(value) => Ok(value),
        None => Ok(default),
    }
}

fn optional_usize_at(root: &Value, path: &[&str]) -> Result<Option<usize>, String> {
    let Some(value) = get_value(root, path) else {
        return Ok(None);
    };
    if let Some(value) = value.as_i64() {
        return if value >= 0 {
            Ok(Some(value as usize))
        } else {
            Err(format!("{} must be non-negative", path.join(".")))
        };
    }
    if let Some(value) = value.as_u64() {
        return Ok(Some(value as usize));
    }
    if let Some(value) = value.as_str() {
        return value
            .parse::<usize>()
            .map(Some)
            .map_err(|err| format!("{} must be a non-negative integer: {err}", path.join(".")));
    }
    Err(format!("{} must be a non-negative integer", path.join(".")))
}

fn string_at(root: &Value, path: &[&str]) -> Option<String> {
    get_value(root, path)
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase().replace('-', "_"))
}

fn raw_string_at(root: &Value, path: &[&str]) -> Option<String> {
    get_value(root, path)
        .and_then(Value::as_str)
        .map(|value| value.trim().to_owned())
}

fn bool_at(root: &Value, path: &[&str], default: bool) -> bool {
    get_value(root, path)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn weighting_code(value: &str) -> Result<u8, String> {
    match value {
        "equal" => Ok(WEIGHT_EQUAL),
        "rank" => Ok(WEIGHT_RANK),
        "score_softmax" => Ok(WEIGHT_SCORE_SOFTMAX),
        other => Err(format!(
            "unsupported strategy.weighting for run-bundle: {other}"
        )),
    }
}

fn score_transform_code(value: &str) -> Result<u8, String> {
    match value {
        "none" => Ok(SCORE_TRANSFORM_NONE),
        "rank_pct" => Ok(SCORE_TRANSFORM_RANK_PCT),
        "zscore_clip" => Ok(SCORE_TRANSFORM_ZSCORE_CLIP),
        other => Err(format!(
            "unsupported strategy.score_transform for run-bundle: {other}"
        )),
    }
}

fn risk_curve_code(value: &str) -> Result<u8, String> {
    match value {
        "linear" => Ok(RISK_CURVE_LINEAR),
        "convex" => Ok(RISK_CURVE_CONVEX),
        "concave" => Ok(RISK_CURVE_CONCAVE),
        "sigmoid" => Ok(RISK_CURVE_SIGMOID),
        other => Err(format!(
            "unsupported risk_control.risk_curve for run-bundle: {other}"
        )),
    }
}

fn intraperiod_score_transform_code(value: &str) -> Result<u8, String> {
    match value {
        "raw" => Ok(SCORE_TRANSFORM_NONE),
        "transformed" => Ok(INTRAPERIOD_SCORE_TRANSFORM_STRATEGY),
        "rank_pct" => Ok(SCORE_TRANSFORM_RANK_PCT),
        "zscore" => Ok(SCORE_TRANSFORM_ZSCORE_CLIP),
        other => Err(format!(
            "unsupported intraperiod exit score_source for run-bundle: {other}"
        )),
    }
}

fn weighting_name(code: u8) -> &'static str {
    match code {
        WEIGHT_RANK => "rank",
        WEIGHT_SCORE_SOFTMAX => "score_softmax",
        _ => "equal",
    }
}

fn score_transform_name(code: u8) -> &'static str {
    match code {
        SCORE_TRANSFORM_RANK_PCT => "rank_pct",
        SCORE_TRANSFORM_ZSCORE_CLIP => "zscore_clip",
        _ => "none",
    }
}

fn write_daily_report(
    path: &Path,
    dates_ns: &[i64],
    out: &[f64],
    baseline_runs: &[BaselineRun],
    intraperiod_exit: Option<&IntraperiodExitConfig>,
) -> Result<(), String> {
    let mut writer = WriterBuilder::new()
        .from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut header = vec![
        "datetime".to_owned(),
        "gross_return".to_owned(),
        "net_return".to_owned(),
        "return".to_owned(),
        "turnover".to_owned(),
        "cost".to_owned(),
        "bench".to_owned(),
        "buy_count".to_owned(),
        "sell_count".to_owned(),
        "holdings".to_owned(),
        "frozen_holdings".to_owned(),
        "account_value".to_owned(),
        "risk_degree".to_owned(),
        "effective_n_drop".to_owned(),
        "risk_control_signal".to_owned(),
        "desticky_active".to_owned(),
        "intraperiod_exit_count".to_owned(),
        "intraperiod_exit_score_candidate_count".to_owned(),
        "intraperiod_exit_price_confirm_required_count".to_owned(),
        "intraperiod_exit_price_confirm_blocked_count".to_owned(),
        "intraperiod_exit_price_confirm_bypassed_remaining_steps_count".to_owned(),
        "intraperiod_exit_price_confirm_bypassed_force_exit_count".to_owned(),
        "intraperiod_exit_price_confirm_signal_timing".to_owned(),
        "intraperiod_exit_price_confirm_execution_timing".to_owned(),
        "intraperiod_exit_remaining_steps".to_owned(),
        "intraperiod_exit_signal_mean".to_owned(),
        "intraperiod_exit_signal_min".to_owned(),
        "intraperiod_exit_residual_mean".to_owned(),
        "intraperiod_exit_residual_min".to_owned(),
        "intraperiod_exit_residual_max".to_owned(),
        "intraperiod_exit_saved_return".to_owned(),
        "intraperiod_exit_missed_return".to_owned(),
        "intraperiod_exit_beneficial_count".to_owned(),
        "intraperiod_exit_harmful_count".to_owned(),
        "cum_gross_return".to_owned(),
        "cum_net_return".to_owned(),
    ];
    for baseline in baseline_runs {
        header.push(format!("{}_return", baseline.prefix));
    }
    for baseline in baseline_runs {
        header.push(format!("fixed_risk_{}_return", baseline.prefix));
    }
    writer
        .write_record(&header)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    let mut cum_gross = 1.0;
    let mut cum_net = 1.0;
    let price_confirm_signal_timing = intraperiod_exit
        .and_then(|exit| exit.price_confirm.as_ref())
        .map(|confirm| confirm.signal_timing.as_str())
        .unwrap_or("");
    let price_confirm_execution_timing = intraperiod_exit
        .and_then(|exit| exit.price_confirm.as_ref())
        .map(|confirm| confirm.execution_timing.as_str())
        .unwrap_or("");
    let fixed_risk_returns_by_date = baseline_runs
        .iter()
        .map(|baseline| {
            baseline
                .fixed_risk_dates_ns
                .iter()
                .enumerate()
                .map(|(row, date_ns)| {
                    let base = row * OUT_COLS;
                    (*date_ns, baseline.fixed_risk_out[base + 1])
                })
                .collect::<BTreeMap<_, _>>()
        })
        .collect::<Vec<_>>();
    for (row, date_ns) in dates_ns.iter().enumerate() {
        let base = row * OUT_COLS;
        let gross = out[base];
        let net = out[base + 1];
        cum_gross *= 1.0 + gross;
        cum_net *= 1.0 + net;
        let mut record = vec![
            datetime_ns_to_date(*date_ns)?,
            format_float(gross),
            format_float(net),
            format_float(net),
            format_float(out[base + 2]),
            format_float(out[base + 3]),
            format_float(out[base + 4]),
            format_int(out[base + 5]),
            format_int(out[base + 6]),
            format_int(out[base + 7]),
            format_int(out[base + 8]),
            format_float(out[base + 9]),
            format_float(out[base + 10]),
            format_int(out[base + 11]),
            format_float(out[base + 12]),
            if out[base + 13] != 0.0 {
                "true".to_owned()
            } else {
                "false".to_owned()
            },
            format_int(out[base + 14]),
            format_int(out[base + 15]),
            format_int(out[base + 16]),
            format_int(out[base + 17]),
            format_int(out[base + 18]),
            format_int(out[base + 19]),
            price_confirm_signal_timing.to_owned(),
            price_confirm_execution_timing.to_owned(),
            format_int(out[base + 20]),
            format_float(out[base + 21]),
            format_float(out[base + 22]),
            format_float(out[base + 23]),
            format_float(out[base + 24]),
            format_float(out[base + 25]),
            format_float(out[base + 26]),
            format_float(out[base + 27]),
            format_int(out[base + 28]),
            format_int(out[base + 29]),
            format_float(cum_gross),
            format_float(cum_net),
        ];
        for baseline in baseline_runs {
            let baseline_base = row * OUT_COLS;
            record.push(format_float(baseline.out[baseline_base + 1]));
        }
        for fixed_returns in &fixed_risk_returns_by_date {
            record.push(format_float(
                fixed_returns.get(date_ns).copied().unwrap_or(0.0),
            ));
        }
        writer
            .write_record(&record)
            .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn write_metrics(path: &Path, metrics: &PortfolioMetrics) -> Result<(), String> {
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    serde_json::to_writer_pretty(file, metrics)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn copy_training_summary_if_present(
    bundle: &PredictionBundle,
    output_dir: &Path,
) -> Result<(), String> {
    let source = bundle.dir.join("training_summary.csv");
    if !source.is_file() {
        return Ok(());
    }
    let target = output_dir.join("training_summary.csv");
    if source == target {
        return Ok(());
    }
    fs::copy(&source, &target).map(|_| ()).map_err(|err| {
        format!(
            "failed to copy {} to {}: {err}",
            source.display(),
            target.display()
        )
    })
}

fn compute_metrics(out: &[f64]) -> PortfolioMetrics {
    let n_rows = out.len() / OUT_COLS;
    let returns = (0..n_rows)
        .map(|row| out[row * OUT_COLS + 1])
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let turnovers = (0..n_rows)
        .map(|row| out[row * OUT_COLS + 2])
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let mean_return = mean(&returns);
    let std = sample_std(&returns, mean_return);
    let annualized_return = mean_return * 242.0;
    let annualized_volatility = std * 242.0_f64.sqrt();
    let sharpe_ratio = if annualized_volatility > 0.0 {
        annualized_return / annualized_volatility
    } else {
        0.0
    };
    let total_return = returns.iter().fold(1.0, |acc, value| acc * (1.0 + value)) - 1.0;
    let final_account_value = if n_rows == 0 {
        f64::NAN
    } else {
        out[(n_rows - 1) * OUT_COLS + 9]
    };
    let win_rate = if returns.is_empty() {
        0.0
    } else {
        returns.iter().filter(|value| **value > 0.0).count() as f64 / returns.len() as f64
    };
    PortfolioMetrics {
        trading_days: returns.len(),
        total_return,
        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        max_drawdown: max_drawdown(&returns),
        win_rate,
        avg_turnover: mean(&turnovers),
        final_account_value,
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn sample_std(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

fn max_drawdown(returns: &[f64]) -> f64 {
    let mut cum = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut max_dd = 0.0_f64;
    for value in returns {
        cum *= 1.0 + value;
        peak = peak.max(cum);
        if peak > 0.0 {
            max_dd = max_dd.min(cum / peak - 1.0);
        }
    }
    max_dd
}

fn datetime_ns_to_date(value: i64) -> Result<String, String> {
    let secs = value.div_euclid(1_000_000_000);
    let nanos = value.rem_euclid(1_000_000_000) as u32;
    let datetime = DateTime::<Utc>::from_timestamp(secs, nanos)
        .ok_or_else(|| format!("invalid timestamp ns: {value}"))?;
    Ok(datetime.date_naive().format("%Y-%m-%d").to_string())
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12}")
    } else {
        String::new()
    }
}

fn format_int(value: f64) -> String {
    if value.is_finite() {
        format!("{}", value.round() as i64)
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_dates_without_any_label() {
        let predictions = MatrixFrame {
            dates_ns: vec![10, 20],
            instruments: vec!["a".to_owned(), "b".to_owned()],
            values: vec![1.0, 2.0, 3.0, 4.0],
            input_rows: 4,
            finite_values: 4,
        };
        let labels = MatrixFrame {
            dates_ns: vec![10, 20],
            instruments: vec!["a".to_owned(), "b".to_owned()],
            values: vec![f64::NAN, f64::NAN, 0.01, f64::NAN],
            input_rows: 4,
            finite_values: 1,
        };
        let prepared = prepare_core_inputs(&predictions, &labels);
        assert_eq!(prepared.dates_ns, [20]);
        assert_eq!(prepared.scores, [3.0, 4.0]);
        assert_eq!(prepared.labels.len(), 2);
        assert_eq!(prepared.labels[0], 0.01);
        assert!(prepared.labels[1].is_nan());
        assert_eq!(prepared.skipped_all_nan_label_dates, 1);
    }

    #[test]
    fn computes_basic_return_metrics() {
        let mut out = vec![0.0_f64; OUT_COLS * 3];
        out[1] = 0.10;
        out[OUT_COLS + 1] = -0.05;
        out[2 * OUT_COLS + 1] = 0.02;
        out[9] = 110.0;
        out[OUT_COLS + 9] = 104.5;
        out[2 * OUT_COLS + 9] = 106.59;
        let metrics = compute_metrics(&out);
        assert_eq!(metrics.trading_days, 3);
        assert!((metrics.total_return - 0.0659).abs() < 1e-12);
        assert!(metrics.max_drawdown < 0.0);
        assert_eq!(metrics.win_rate, 2.0 / 3.0);
    }

    #[test]
    fn fuses_rank_pct_predictions_on_finite_overlap() {
        let primary = MatrixFrame {
            dates_ns: vec![10, 20],
            instruments: vec!["a".to_owned(), "b".to_owned()],
            values: vec![2.0, 4.0, 1.0, f64::NAN],
            input_rows: 4,
            finite_values: 3,
        };
        let secondary = MatrixFrame {
            dates_ns: vec![10, 20],
            instruments: vec!["a".to_owned(), "b".to_owned()],
            values: vec![10.0, 5.0, 3.0, 4.0],
            input_rows: 4,
            finite_values: 4,
        };
        let config = ScoreFusionConfig {
            secondary_predictions_dir: PathBuf::from("secondary"),
            mode: FUSION_MODE_MULTIPLY,
            mode_name: "multiply".to_owned(),
            primary_transform: FUSION_TRANSFORM_RANK_PCT,
            primary_transform_name: "rank_pct".to_owned(),
            secondary_transform: FUSION_TRANSFORM_RANK_PCT,
            secondary_transform_name: "rank_pct".to_owned(),
            primary_power: 1.0,
            secondary_power: 1.0,
            blend_weight: 0.5,
            filter_threshold: 0.5,
            filter_value: -1.0,
        };

        let fused = fuse_prediction_matrices(&primary, &secondary, &config).unwrap();
        assert_eq!(fused.matrix.dates_ns, [10, 20]);
        assert_eq!(fused.matrix.instruments, ["a", "b"]);
        assert_eq!(fused.present_mask, [true, true, true, false]);
        assert!((fused.matrix.values[0] - 0.5).abs() < 1e-12);
        assert!((fused.matrix.values[1] - 0.5).abs() < 1e-12);
        assert!((fused.matrix.values[2] - 1.0).abs() < 1e-12);
        assert!(fused.matrix.values[3].is_nan());
        assert_eq!(fused.summary.fusion_overlap_rows, 3);
        assert_eq!(fused.summary.fusion_overlap_dates, 2);
    }

    #[test]
    fn project_matrix_respects_present_mask() {
        let source = MatrixFrame {
            dates_ns: vec![10, 20],
            instruments: vec!["a".to_owned(), "b".to_owned()],
            values: vec![0.1, 0.2, 0.3, 0.4],
            input_rows: 4,
            finite_values: 4,
        };
        let projected = project_matrix(
            &source,
            &[10, 20],
            &["a".to_owned(), "b".to_owned()],
            &[true, false, true, true],
        );
        assert_eq!(projected.input_rows, 3);
        assert_eq!(projected.values[0], 0.1);
        assert!(projected.values[1].is_nan());
        assert_eq!(projected.values[2], 0.3);
        assert_eq!(projected.values[3], 0.4);
    }
}
