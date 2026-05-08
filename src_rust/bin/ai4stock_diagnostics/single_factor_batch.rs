use ai4stock2_native::common::artifact::write_json_pretty;
use ai4stock2_native::common::cli::{display_command, next_arg, path_to_string, split_value};
use ai4stock2_native::common::yaml::{deep_merge_yaml, read_yaml_file, write_yaml_file};
use ai4stock2_native::feature_prefilter::{
    run_build_prefilter_profile, run_build_robust_profile, PrefilterProfileBuildOptions,
    PrefilterThresholds, ProfileBuildSummary, RobustProfileBuildOptions,
};
use ai4stock2_native::single_factor_diagnostics::{
    run_single_factor_diagnostics, BenchmarkMode, BenchmarkOptions, BenchmarkValueType,
    DiagnosticLabelSpace, SegmentSpec, SingleFactorOptions,
};
use chrono::{Datelike, Local, NaiveDate, TimeZone, Utc};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::Instant;

const SUMMARY_HEADERS: [&str; 23] = [
    "case_name",
    "feature_profile",
    "baseline_feature_profile",
    "diagnostic_label_space",
    "diagnostic_threshold",
    "feature_count",
    "incremental_feature_count",
    "row_count",
    "top_feature",
    "top_rank_ic_mean",
    "top_rank_ic_abs_mean",
    "top_rank_ic_ir",
    "top_monotonicity_mean",
    "top_monthly_directional_hit",
    "incremental_top_feature",
    "incremental_top_rank_ic_mean",
    "incremental_top_rank_ic_abs_mean",
    "incremental_top_rank_ic_ir",
    "incremental_top_monotonicity_mean",
    "incremental_top_monthly_directional_hit",
    "incremental_n_abs_rankic_ge_0p03",
    "incremental_n_abs_rankic_ge_0p05",
    "output_dir",
];

const MANIFEST_HEADERS: [&str; 11] = [
    "case_name",
    "feature_profile",
    "baseline_feature_profile",
    "diagnostic_label_space",
    "diagnostic_threshold",
    "output_dir",
    "readme_path",
    "summary_csv",
    "incremental_summary_csv",
    "segment_comparison_csv",
    "incremental_segment_comparison_csv",
];

#[derive(Debug, Clone)]
struct BatchOptions {
    config: PathBuf,
    config_is_snapshot: bool,
    experiment_profile: Option<String>,
    feature_profile: Option<String>,
    model_profile: Option<String>,
    data_source: Option<String>,
    set_overrides: Vec<String>,
    topk: Option<usize>,
    n_drop: Option<usize>,
    run_tag: Option<String>,
    store_dir: Option<String>,
    disable_local_store: bool,
    rebalance_freq: Option<usize>,
    signal_horizon: Option<usize>,
    period: String,
    date_start: Option<String>,
    date_end: Option<String>,
    all_features: bool,
    quantile_bins: usize,
    top_n: usize,
    no_detail_artifacts: bool,
    segment_scheme: String,
    segments: Option<String>,
    base_output_dir: Option<PathBuf>,
    summary_path: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    cases: Vec<DiagnosticsBatchCase>,
    industry_neutral: bool,
    feature_chunk_size: usize,
    batch_size: usize,
    dry_run: bool,
}

#[derive(Debug, Clone)]
struct DiagnosticsBatchCase {
    name: String,
    feature_profile: String,
    baseline_feature_profile: Option<String>,
    diagnostic_label_space: String,
    diagnostic_threshold: f64,
    output_dir: Option<PathBuf>,
    run_tag: Option<String>,
}

#[derive(Debug, Clone)]
struct ResolvedRuntimeConfig {
    cfg: YamlValue,
    factor_store: PathBuf,
    data_source: String,
    signal_horizon: usize,
    label_column: String,
    universe_name: String,
    universe_dir: PathBuf,
}

#[derive(Debug, Clone)]
struct FeatureProfileResolution {
    source_columns: Option<Vec<String>>,
    factor_store_name: String,
}

#[derive(Debug, Clone)]
struct CsvTable {
    headers: Vec<String>,
    rows: Vec<BTreeMap<String, String>>,
}

pub(crate) fn run_single_factor_batch_command(args: &[String]) -> Result<(), String> {
    let started = Instant::now();
    let options = parse_batch_options(args)?;
    let resolved = resolve_runtime_config(&options)?;
    let (date_start, date_end) = resolve_period_dates(&resolved.cfg, &options)?;
    let base_output_dir = options.base_output_dir.clone().unwrap_or_else(|| {
        PathBuf::from("results")
            .join("diagnostics")
            .join("single_factor_batch")
            .join(Local::now().format("%Y%m%d_%H%M%S").to_string())
    });
    let summary_path = options
        .summary_path
        .clone()
        .unwrap_or_else(|| base_output_dir.join("batch_summary.tsv"));
    let manifest_path = options
        .manifest_path
        .clone()
        .unwrap_or_else(|| base_output_dir.join("batch_manifest.tsv"));

    init_tsv(&summary_path, &SUMMARY_HEADERS)?;
    init_tsv(&manifest_path, &MANIFEST_HEADERS)?;

    let factor_store_meta = load_factor_store_metadata(&resolved.factor_store)?;
    let all_feature_names = read_meta_feature_names(&factor_store_meta)?;
    let mut case_feature_map = BTreeMap::new();
    let mut case_incremental_feature_map = BTreeMap::new();
    for case in &options.cases {
        let feature_names = if options.all_features {
            all_feature_names.clone()
        } else {
            resolve_feature_profile_source_columns(
                &resolved.cfg,
                &factor_store_meta,
                &case.feature_profile,
            )?
        };
        if feature_names.is_empty() {
            return Err(format!("Case '{}' resolved no features", case.name));
        }
        let baseline_feature_names = if let Some(profile) = &case.baseline_feature_profile {
            resolve_feature_profile_source_columns(&resolved.cfg, &factor_store_meta, profile)?
        } else {
            Vec::new()
        };
        let incremental_feature_names =
            resolve_incremental_feature_names(&feature_names, &baseline_feature_names);
        case_feature_map.insert(case.name.clone(), feature_names);
        case_incremental_feature_map.insert(case.name.clone(), incremental_feature_names);
    }

    let segments = resolve_segments(&resolved.cfg, &options, &date_start, &date_end)?;
    println!(
        "[*] Single-factor diagnostics batch: engine=rust, cases={}, date_start={}, date_end={}",
        options.cases.len(),
        date_start,
        date_end
    );

    for (case_index, case) in options.cases.iter().enumerate() {
        let case_started = Instant::now();
        let feature_names = case_feature_map
            .get(&case.name)
            .ok_or_else(|| format!("missing feature map for case {}", case.name))?;
        let incremental_feature_names = case_incremental_feature_map
            .get(&case.name)
            .ok_or_else(|| format!("missing incremental feature map for case {}", case.name))?;
        let output_dir = resolve_case_output_dir(&base_output_dir, case);
        let mut case_cfg = resolved.cfg.clone();
        set_yaml_dotted(
            &mut case_cfg,
            "features.profile",
            YamlValue::String(case.feature_profile.clone()),
        )?;
        let case_context = BatchCaseContext {
            resolved: &resolved,
            options: &options,
            case,
            date_range: DateRangeRef {
                start: &date_start,
                end: &date_end,
            },
        };
        let metadata = build_case_metadata(
            &case_context,
            feature_names,
            incremental_feature_names,
            segments.len(),
        );
        let prepared = prepare_single_factor_inputs(
            &case_cfg,
            &resolved,
            &options,
            case,
            &output_dir,
            feature_names,
            &metadata,
        )?;
        let command = build_display_single_factor_command(
            &case_context,
            &output_dir,
            &prepared.features_path,
            &prepared.metadata_path,
            &prepared.config_path,
            &segments,
        );
        if options.dry_run {
            println!("[dry-run] {}", display_command(&command));
            continue;
        }

        println!(
            "[*] [{}/{}] Rust diagnostics: {}",
            case_index + 1,
            options.cases.len(),
            case.name
        );
        let diagnostic_summary = run_single_factor_diagnostics(&SingleFactorOptions {
            factor_store: resolved.factor_store.clone(),
            output_dir: output_dir.clone(),
            feature_names: feature_names.clone(),
            label_column: resolved.label_column.clone(),
            signal_horizon: resolved.signal_horizon,
            date_start: Some(date_start.clone()),
            date_end: Some(date_end.clone()),
            universe_name: resolved.universe_name.clone(),
            universe_dir: resolved.universe_dir.clone(),
            quantile_bins: options.quantile_bins.max(2),
            top_n: options.top_n.max(1),
            include_details: !options.no_detail_artifacts,
            segments: segments.clone(),
            diagnostic_label_space: DiagnosticLabelSpace::parse(&case.diagnostic_label_space)?,
            diagnostic_threshold: case.diagnostic_threshold,
            industry_neutral: options.industry_neutral,
            industry_map_path: industry_map_path(&resolved.cfg),
            feature_chunk_size: options.feature_chunk_size.max(1),
            batch_size: options.batch_size.max(1),
            metadata_json_path: Some(prepared.metadata_path),
            config_snapshot_path: Some(prepared.config_path),
            benchmark: benchmark_options(&resolved.cfg, &case.diagnostic_label_space)?,
        })?;

        let summary_csv = output_dir.join("single_factor_summary.csv");
        let summary = read_csv_table(&summary_csv)?;
        let incremental_summary = filter_table_for_features(&summary, incremental_feature_names);
        let segment_comparison_path = output_dir.join("single_factor_segment_comparison.csv");
        let segment_comparison = if segment_comparison_path.exists() {
            Some(read_csv_table(&segment_comparison_path)?)
        } else {
            None
        };
        let incremental_artifacts = write_single_factor_subset_artifacts(
            &incremental_summary,
            &output_dir,
            "incremental",
            options.top_n.max(1),
            segment_comparison.as_ref(),
            incremental_feature_names,
        )?;
        merge_artifacts_into_manifest(&output_dir, &incremental_artifacts)?;

        let top = summary.rows.first();
        let incremental_top = incremental_summary.rows.first();
        let readme_path = output_dir.join("README.md");
        append_tsv_row(
            &summary_path,
            vec![
                case.name.clone(),
                case.feature_profile.clone(),
                case.baseline_feature_profile.clone().unwrap_or_default(),
                case.diagnostic_label_space.clone(),
                case.diagnostic_threshold.to_string(),
                feature_names.len().to_string(),
                incremental_feature_names.len().to_string(),
                diagnostic_summary.row_count.to_string(),
                row_value(top, "feature"),
                row_value(top, "rank_ic_mean"),
                row_value(top, "rank_ic_abs_mean"),
                row_value(top, "rank_ic_ir"),
                row_value(top, "monotonicity_mean"),
                row_value(top, "monthly_rank_ic_directional_hit_rate"),
                row_value(incremental_top, "feature"),
                row_value(incremental_top, "rank_ic_mean"),
                row_value(incremental_top, "rank_ic_abs_mean"),
                row_value(incremental_top, "rank_ic_ir"),
                row_value(incremental_top, "monotonicity_mean"),
                row_value(incremental_top, "monthly_rank_ic_directional_hit_rate"),
                count_abs_metric_ge(&incremental_summary, "rank_ic_mean", 0.03).to_string(),
                count_abs_metric_ge(&incremental_summary, "rank_ic_mean", 0.05).to_string(),
                path_to_string(&output_dir),
            ],
        )?;
        append_tsv_row(
            &manifest_path,
            vec![
                case.name.clone(),
                case.feature_profile.clone(),
                case.baseline_feature_profile.clone().unwrap_or_default(),
                case.diagnostic_label_space.clone(),
                case.diagnostic_threshold.to_string(),
                path_to_string(&output_dir),
                path_to_string(&readme_path),
                path_to_string(&summary_csv),
                incremental_artifacts
                    .get("incremental_summary_csv")
                    .cloned()
                    .unwrap_or_default(),
                if segment_comparison_path.exists() {
                    path_to_string(&segment_comparison_path)
                } else {
                    String::new()
                },
                incremental_artifacts
                    .get("incremental_segment_comparison_csv")
                    .cloned()
                    .unwrap_or_default(),
            ],
        )?;
        println!(
            "[{}/{}] {}: rows={}, features={}, incremental_features={}, elapsed={:.2}s",
            case_index + 1,
            options.cases.len(),
            case.name,
            diagnostic_summary.row_count,
            feature_names.len(),
            incremental_feature_names.len(),
            case_started.elapsed().as_secs_f64()
        );
    }

    println!("[+] Batch outputs: {}", base_output_dir.display());
    println!("    summary: {}", summary_path.display());
    println!("    manifest: {}", manifest_path.display());
    println!("    total elapsed: {:.2}s", started.elapsed().as_secs_f64());
    Ok(())
}

pub(crate) fn run_single_factor_profile_command(args: &[String]) -> Result<(), String> {
    let started = Instant::now();
    let profile_options = parse_single_factor_profile_options(args)?;
    let mut batch_options = profile_options.batch_options;
    let resolved = resolve_runtime_config(&batch_options)?;
    let (date_start, date_end) = resolve_period_dates(&resolved.cfg, &batch_options)?;
    let factor_store_meta = load_factor_store_metadata(&resolved.factor_store)?;
    let all_feature_names = read_meta_feature_names(&factor_store_meta)?;
    let feature_profile = yaml_path_string(&resolved.cfg, &["features", "profile"])
        .unwrap_or_else(|| "all".to_owned());
    let feature_names = if batch_options.all_features {
        all_feature_names
    } else {
        resolve_feature_profile_source_columns(&resolved.cfg, &factor_store_meta, &feature_profile)?
    };
    if feature_names.is_empty() {
        return Err("No features resolved for diagnostics.".to_owned());
    }
    let segments = resolve_segments(&resolved.cfg, &batch_options, &date_start, &date_end)?;
    let output_dir = profile_options.output_dir.unwrap_or_else(|| {
        default_single_factor_output_dir(
            &resolved,
            &batch_options,
            &feature_profile,
            &profile_options.diagnostic_label_space,
        )
    });
    batch_options.cases = vec![DiagnosticsBatchCase {
        name: feature_profile.clone(),
        feature_profile: feature_profile.clone(),
        baseline_feature_profile: None,
        diagnostic_label_space: profile_options.diagnostic_label_space.clone(),
        diagnostic_threshold: profile_options.diagnostic_threshold,
        output_dir: Some(output_dir.clone()),
        run_tag: batch_options.run_tag.clone(),
    }];
    let case = batch_options
        .cases
        .first()
        .ok_or_else(|| "internal error: missing single-factor case".to_owned())?;
    println!(
        "[*] Single-factor diagnostics: period={}, date_start={}, date_end={}, features={}",
        batch_options.period,
        date_start,
        date_end,
        feature_names.len()
    );
    let metadata = serde_json::json!({
        "data_source": resolved.data_source,
        "universe": resolved.universe_name,
        "feature_profile": feature_profile,
        "factor_store_dir": resolved.factor_store,
        "signal_horizon": resolved.signal_horizon,
        "period": batch_options.period,
        "date_start": date_start,
        "date_end": date_end,
        "diagnostic_label_space": profile_options.diagnostic_label_space,
        "diagnostic_threshold": profile_options.diagnostic_threshold,
        "industry_neutral": batch_options.industry_neutral,
        "neutralized_feature_count": if batch_options.industry_neutral { feature_names.len() } else { 0 },
        "feature_count": feature_names.len(),
        "quantile_bins": batch_options.quantile_bins.max(2),
        "detail_artifacts": !batch_options.no_detail_artifacts,
        "segment_scheme": batch_options.segment_scheme,
        "segment_count": segments.len(),
        "engine": "rust",
    });
    let mut case_cfg = resolved.cfg.clone();
    set_yaml_dotted(
        &mut case_cfg,
        "features.profile",
        YamlValue::String(feature_profile.clone()),
    )?;
    let prepared = prepare_single_factor_inputs(
        &case_cfg,
        &resolved,
        &batch_options,
        case,
        &output_dir,
        &feature_names,
        &metadata,
    )?;
    let case_context = BatchCaseContext {
        resolved: &resolved,
        options: &batch_options,
        case,
        date_range: DateRangeRef {
            start: &date_start,
            end: &date_end,
        },
    };
    let command = build_display_single_factor_command(
        &case_context,
        &output_dir,
        &prepared.features_path,
        &prepared.metadata_path,
        &prepared.config_path,
        &segments,
    );
    if batch_options.dry_run {
        println!("[dry-run] {}", display_command(&command));
        return Ok(());
    }
    println!(
        "[*] Delegating single-factor diagnostics to Rust: {}",
        display_command(&command)
    );
    let summary = run_single_factor_diagnostics(&SingleFactorOptions {
        factor_store: resolved.factor_store.clone(),
        output_dir: output_dir.clone(),
        feature_names,
        label_column: resolved.label_column.clone(),
        signal_horizon: resolved.signal_horizon,
        date_start: Some(date_start),
        date_end: Some(date_end),
        universe_name: resolved.universe_name.clone(),
        universe_dir: resolved.universe_dir.clone(),
        quantile_bins: batch_options.quantile_bins.max(2),
        top_n: batch_options.top_n.max(1),
        include_details: !batch_options.no_detail_artifacts,
        segments,
        diagnostic_label_space: DiagnosticLabelSpace::parse(
            &profile_options.diagnostic_label_space,
        )?,
        diagnostic_threshold: profile_options.diagnostic_threshold,
        industry_neutral: batch_options.industry_neutral,
        industry_map_path: industry_map_path(&resolved.cfg),
        feature_chunk_size: batch_options.feature_chunk_size.max(1),
        batch_size: batch_options.batch_size.max(1),
        metadata_json_path: Some(prepared.metadata_path),
        config_snapshot_path: Some(prepared.config_path),
        benchmark: benchmark_options(&resolved.cfg, &profile_options.diagnostic_label_space)?,
    })?;
    let summary_path = output_dir.join("single_factor_summary.csv");
    println!(
        "[+] Single-factor diagnostics saved to: {}",
        output_dir.display()
    );
    println!("    engine: rust");
    println!("    total={:.2}s", started.elapsed().as_secs_f64());
    println!("    rows={}", summary.row_count);
    println!("    summary: {}", summary_path.display());
    println!(
        "    top abs RankIC: {}",
        output_dir
            .join("single_factor_top_abs_rankic.csv")
            .display()
    );
    println!("    readme: {}", output_dir.join("README.md").display());
    print_summary_preview(&summary_path)?;
    Ok(())
}

pub(crate) fn run_prefilter_profile_runtime_command(args: &[String]) -> Result<(), String> {
    let options = parse_profile_runtime_options(args, ProfileRuntimeKind::Prefilter)?;
    let resolved = resolve_runtime_config(&options.batch_options)?;
    let (date_start, date_end) = resolve_period_dates(&resolved.cfg, &options.batch_options)?;
    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| default_profile_output_dir("prefilter_profiles", &options.profile_name));
    let safety_warning = if options.write_config_profile {
        profile_write_safety_warning(
            &resolved.cfg,
            &date_start,
            &date_end,
            options.allow_unsafe_profile_write,
            "run_build_prefiltered_profile.py",
            &[
                options.diagnostics_summary.as_ref(),
                options.segment_comparison.as_ref(),
            ],
        )?
    } else {
        None
    };
    let factor_store_name = factor_store_name_for_current_profile(&resolved.cfg)?;
    let settings = build_prefilter_profile_settings(
        &options,
        &resolved,
        &date_start,
        &date_end,
        safety_warning.is_none(),
    );
    if options.dry_run {
        println!(
            "[dry-run] ai4stock-diagnostics build-prefilter-profile --diagnostics-summary {} --factor-store {} --output-dir {} --profile-name {}",
            options
                .diagnostics_summary
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            resolved.factor_store.display(),
            output_dir.display(),
            options.profile_name
        );
        return Ok(());
    }
    let summary = run_build_prefilter_profile(&PrefilterProfileBuildOptions {
        diagnostics_summary: options
            .diagnostics_summary
            .clone()
            .ok_or_else(|| "--diagnostics-summary is required".to_owned())?,
        segment_comparison: options.segment_comparison.clone(),
        output_dir,
        thresholds: options.thresholds.clone(),
        factor_store: resolved.factor_store.clone(),
        date_start: Some(date_start),
        date_end: Some(date_end),
        universe_name: resolved.universe_name.clone(),
        universe_dir: resolved.universe_dir.clone(),
        corr_threshold: options.max_abs_corr,
        use_cross_sectional_rank: options.use_cross_sectional_rank,
        batch_size: options.batch_size,
        profile_name: options.profile_name.clone(),
        max_features: options.max_features,
        write_config_profile: options.write_config_profile,
        config_profile_path: options.config_profile_path.clone().or_else(|| {
            options.write_config_profile.then(|| {
                PathBuf::from("configs")
                    .join("features")
                    .join(format!("{}.yaml", options.profile_name))
            })
        }),
        factor_store_name,
        settings,
        safety_warning,
    })?;
    print_profile_build_summary("Prefilter", &summary, options.json)?;
    Ok(())
}

pub(crate) fn run_robust_profile_runtime_command(args: &[String]) -> Result<(), String> {
    let options = parse_profile_runtime_options(args, ProfileRuntimeKind::Robust)?;
    let resolved = resolve_runtime_config(&options.batch_options)?;
    let (date_start, date_end) = resolve_period_dates(&resolved.cfg, &options.batch_options)?;
    let output_dir = options
        .output_dir
        .clone()
        .unwrap_or_else(|| default_profile_output_dir("robust_profiles", &options.profile_name));
    let safety_warning = if options.write_config_profile {
        profile_write_safety_warning(
            &resolved.cfg,
            &date_start,
            &date_end,
            options.allow_unsafe_profile_write,
            "run_build_robust_factor_profile.py",
            &[
                options.raw_summary.as_ref(),
                options.raw_segment_comparison.as_ref(),
                options.neutral_summary.as_ref(),
                options.neutral_segment_comparison.as_ref(),
            ],
        )?
    } else {
        None
    };
    let factor_store_name = factor_store_name_for_current_profile(&resolved.cfg)?;
    let settings = build_robust_profile_settings(
        &options,
        &resolved,
        &date_start,
        &date_end,
        safety_warning.is_none(),
    );
    if options.dry_run {
        println!(
            "[dry-run] ai4stock-diagnostics build-robust-profile --raw-summary {} --neutral-summary {} --factor-store {} --output-dir {} --profile-name {}",
            options
                .raw_summary
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            options
                .neutral_summary
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_default(),
            resolved.factor_store.display(),
            output_dir.display(),
            options.profile_name
        );
        return Ok(());
    }
    let summary = run_build_robust_profile(&RobustProfileBuildOptions {
        raw_summary: options
            .raw_summary
            .clone()
            .ok_or_else(|| "--raw-summary is required".to_owned())?,
        neutral_summary: options
            .neutral_summary
            .clone()
            .ok_or_else(|| "--neutral-summary is required".to_owned())?,
        raw_segment_comparison: options.raw_segment_comparison.clone(),
        neutral_segment_comparison: options.neutral_segment_comparison.clone(),
        output_dir,
        thresholds: options.thresholds.clone(),
        factor_store: resolved.factor_store.clone(),
        date_start: Some(date_start),
        date_end: Some(date_end),
        universe_name: resolved.universe_name.clone(),
        universe_dir: resolved.universe_dir.clone(),
        corr_threshold: options.max_abs_corr,
        use_cross_sectional_rank: options.use_cross_sectional_rank,
        batch_size: options.batch_size,
        profile_name: options.profile_name.clone(),
        max_features: options.max_features,
        write_config_profile: options.write_config_profile,
        config_profile_path: options.config_profile_path.clone().or_else(|| {
            options.write_config_profile.then(|| {
                PathBuf::from("configs")
                    .join("features")
                    .join(format!("{}.yaml", options.profile_name))
            })
        }),
        factor_store_name,
        settings,
        safety_warning,
    })?;
    print_profile_build_summary("Robust-profile", &summary, options.json)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct PreparedSingleFactorInputs {
    features_path: PathBuf,
    metadata_path: PathBuf,
    config_path: PathBuf,
}

#[derive(Debug, Clone)]
struct SingleFactorProfileOptions {
    batch_options: BatchOptions,
    output_dir: Option<PathBuf>,
    diagnostic_label_space: String,
    diagnostic_threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileRuntimeKind {
    Prefilter,
    Robust,
}

#[derive(Debug, Clone)]
struct ProfileRuntimeOptions {
    batch_options: BatchOptions,
    diagnostics_summary: Option<PathBuf>,
    segment_comparison: Option<PathBuf>,
    raw_summary: Option<PathBuf>,
    neutral_summary: Option<PathBuf>,
    raw_segment_comparison: Option<PathBuf>,
    neutral_segment_comparison: Option<PathBuf>,
    thresholds: PrefilterThresholds,
    max_abs_corr: f64,
    use_cross_sectional_rank: bool,
    max_features: Option<usize>,
    profile_name: String,
    write_config_profile: bool,
    allow_unsafe_profile_write: bool,
    config_profile_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    batch_size: usize,
    dry_run: bool,
    json: bool,
}

fn parse_single_factor_profile_options(
    args: &[String],
) -> Result<SingleFactorProfileOptions, String> {
    let mut forwarded = Vec::new();
    let mut output_dir = None;
    let mut diagnostic_label_space = "raw_return".to_owned();
    let mut diagnostic_threshold = 0.0f64;
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_arg(args, &mut index, "--output-dir")?));
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--diagnostic-label-space" => {
                diagnostic_label_space = next_arg(args, &mut index, "--diagnostic-label-space")?
                    .trim()
                    .to_ascii_lowercase();
            }
            value if value.starts_with("--diagnostic-label-space=") => {
                diagnostic_label_space = split_value(value, "--diagnostic-label-space")?
                    .trim()
                    .to_ascii_lowercase();
            }
            "--diagnostic-threshold" => {
                diagnostic_threshold = parse_f64(
                    next_arg(args, &mut index, "--diagnostic-threshold")?,
                    "--diagnostic-threshold",
                )?;
            }
            value if value.starts_with("--diagnostic-threshold=") => {
                diagnostic_threshold = parse_f64(
                    split_value(value, "--diagnostic-threshold")?,
                    "--diagnostic-threshold",
                )?;
            }
            "--case" => {
                return Err("--case is only valid for single-factor-batch".to_owned());
            }
            value if value.starts_with("--case=") => {
                return Err("--case is only valid for single-factor-batch".to_owned());
            }
            other => forwarded.push(other.to_owned()),
        }
        index += 1;
    }
    DiagnosticLabelSpace::parse(&diagnostic_label_space)?;
    let mut batch_options = parse_batch_options(
        &[
            forwarded,
            vec![
                "--case".to_owned(),
                "name=single_factor".to_owned(),
                "feature_profile=single_factor".to_owned(),
            ],
        ]
        .concat(),
    )?;
    batch_options.cases.clear();
    Ok(SingleFactorProfileOptions {
        batch_options,
        output_dir,
        diagnostic_label_space,
        diagnostic_threshold,
    })
}

fn parse_profile_runtime_options(
    args: &[String],
    kind: ProfileRuntimeKind,
) -> Result<ProfileRuntimeOptions, String> {
    let mut forwarded = Vec::new();
    let mut diagnostics_summary = None;
    let mut segment_comparison = None;
    let mut raw_summary = None;
    let mut neutral_summary = None;
    let mut raw_segment_comparison = None;
    let mut neutral_segment_comparison = None;
    let mut thresholds = if kind == ProfileRuntimeKind::Robust {
        PrefilterThresholds {
            min_segment_directional_hit_mean: Some(0.55),
            max_segment_rank_ic_mean_range: Some(0.14),
            ..Default::default()
        }
    } else {
        PrefilterThresholds::default()
    };
    let mut max_abs_corr = 0.97f64;
    let mut use_cross_sectional_rank = true;
    let mut max_features = None;
    let mut profile_name = None;
    let mut write_config_profile = false;
    let mut allow_unsafe_profile_write = false;
    let mut config_profile_path = None;
    let mut output_dir = None;
    let mut batch_size = 65_536usize;
    let mut dry_run = false;
    let mut json = false;

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(super::usage().to_owned()),
            "--diagnostics-summary" => {
                diagnostics_summary = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--diagnostics-summary",
                )?));
            }
            value if value.starts_with("--diagnostics-summary=") => {
                diagnostics_summary =
                    Some(PathBuf::from(split_value(value, "--diagnostics-summary")?));
            }
            "--segment-comparison" => {
                segment_comparison = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--segment-comparison",
                )?));
            }
            value if value.starts_with("--segment-comparison=") => {
                segment_comparison =
                    Some(PathBuf::from(split_value(value, "--segment-comparison")?));
            }
            "--raw-summary" => {
                raw_summary = Some(PathBuf::from(next_arg(args, &mut index, "--raw-summary")?));
            }
            value if value.starts_with("--raw-summary=") => {
                raw_summary = Some(PathBuf::from(split_value(value, "--raw-summary")?));
            }
            "--neutral-summary" => {
                neutral_summary = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--neutral-summary",
                )?));
            }
            value if value.starts_with("--neutral-summary=") => {
                neutral_summary = Some(PathBuf::from(split_value(value, "--neutral-summary")?));
            }
            "--raw-segment-comparison" => {
                raw_segment_comparison = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
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
                neutral_segment_comparison = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--neutral-segment-comparison",
                )?));
            }
            value if value.starts_with("--neutral-segment-comparison=") => {
                neutral_segment_comparison = Some(PathBuf::from(split_value(
                    value,
                    "--neutral-segment-comparison",
                )?));
            }
            "--min-coverage-pct" => {
                thresholds.min_coverage_pct = parse_f64(
                    next_arg(args, &mut index, "--min-coverage-pct")?,
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
                thresholds.min_abs_rank_ic = parse_f64(
                    next_arg(args, &mut index, "--min-abs-rank-ic")?,
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
                thresholds.min_abs_rank_ic_ir = parse_f64(
                    next_arg(args, &mut index, "--min-abs-rank-ic-ir")?,
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
                thresholds.min_monthly_positive_rate = parse_f64(
                    next_arg(args, &mut index, "--min-monthly-positive-rate")?,
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
                thresholds.min_segment_directional_hit_mean = Some(parse_f64(
                    next_arg(args, &mut index, "--min-segment-directional-hit-mean")?,
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
                thresholds.max_segment_rank_ic_mean_range = Some(parse_f64(
                    next_arg(args, &mut index, "--max-segment-rank-ic-mean-range")?,
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
            "--max-abs-corr" | "--corr-threshold" => {
                max_abs_corr = parse_f64(
                    next_arg(args, &mut index, "--max-abs-corr")?,
                    "--max-abs-corr",
                )?;
            }
            value if value.starts_with("--max-abs-corr=") => {
                max_abs_corr = parse_f64(split_value(value, "--max-abs-corr")?, "--max-abs-corr")?;
            }
            value if value.starts_with("--corr-threshold=") => {
                max_abs_corr =
                    parse_f64(split_value(value, "--corr-threshold")?, "--corr-threshold")?;
            }
            "--no-cross-sectional-rank-corr" | "--no-cross-sectional-rank" | "--raw-values" => {
                use_cross_sectional_rank = false;
            }
            "--cross-sectional-rank" => use_cross_sectional_rank = true,
            "--max-features" => {
                max_features = Some(parse_usize(
                    next_arg(args, &mut index, "--max-features")?,
                    "--max-features",
                )?);
            }
            value if value.starts_with("--max-features=") => {
                max_features = Some(parse_usize(
                    split_value(value, "--max-features")?,
                    "--max-features",
                )?);
            }
            "--profile-name" => profile_name = Some(next_arg(args, &mut index, "--profile-name")?),
            value if value.starts_with("--profile-name=") => {
                profile_name = Some(split_value(value, "--profile-name")?);
            }
            "--write-config-profile" => write_config_profile = true,
            "--allow-unsafe-profile-write" => allow_unsafe_profile_write = true,
            "--config-profile-path" => {
                config_profile_path = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--config-profile-path",
                )?));
            }
            value if value.starts_with("--config-profile-path=") => {
                config_profile_path =
                    Some(PathBuf::from(split_value(value, "--config-profile-path")?));
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(next_arg(args, &mut index, "--output-dir")?))
            }
            value if value.starts_with("--output-dir=") => {
                output_dir = Some(PathBuf::from(split_value(value, "--output-dir")?));
            }
            "--batch-size" => {
                batch_size =
                    parse_usize(next_arg(args, &mut index, "--batch-size")?, "--batch-size")?;
            }
            value if value.starts_with("--batch-size=") => {
                batch_size = parse_usize(split_value(value, "--batch-size")?, "--batch-size")?;
            }
            "--summary-engine" | "--correlation-engine" => {
                let option = args[index].clone();
                let _ = next_arg(args, &mut index, &option)?;
            }
            value
                if value.starts_with("--summary-engine=")
                    || value.starts_with("--correlation-engine=") => {}
            "--dry-run" => dry_run = true,
            "--json" => json = true,
            other => forwarded.push(other.to_owned()),
        }
        index += 1;
    }

    match kind {
        ProfileRuntimeKind::Prefilter => {
            if diagnostics_summary.is_none() {
                return Err("--diagnostics-summary is required".to_owned());
            }
        }
        ProfileRuntimeKind::Robust => {
            if raw_summary.is_none() {
                return Err("--raw-summary is required".to_owned());
            }
            if neutral_summary.is_none() {
                return Err("--neutral-summary is required".to_owned());
            }
        }
    }
    let profile_name = profile_name.ok_or_else(|| "--profile-name is required".to_owned())?;
    let mut batch_options = parse_batch_options(
        &[
            forwarded,
            vec![
                "--case".to_owned(),
                "name=profile_builder".to_owned(),
                "feature_profile=profile_builder".to_owned(),
            ],
        ]
        .concat(),
    )?;
    batch_options.cases.clear();
    Ok(ProfileRuntimeOptions {
        batch_options,
        diagnostics_summary,
        segment_comparison,
        raw_summary,
        neutral_summary,
        raw_segment_comparison,
        neutral_segment_comparison,
        thresholds,
        max_abs_corr,
        use_cross_sectional_rank,
        max_features,
        profile_name,
        write_config_profile,
        allow_unsafe_profile_write,
        config_profile_path,
        output_dir,
        batch_size,
        dry_run,
        json,
    })
}

fn default_single_factor_output_dir(
    resolved: &ResolvedRuntimeConfig,
    options: &BatchOptions,
    feature_profile: &str,
    diagnostic_label_space: &str,
) -> PathBuf {
    let tag_suffix = options
        .run_tag
        .as_ref()
        .map(|tag| tag.trim())
        .filter(|tag| !tag.is_empty())
        .map(|tag| format!("__{tag}"))
        .unwrap_or_default();
    PathBuf::from("results")
        .join("diagnostics")
        .join("single_factor")
        .join(format!(
            "{}__{}__{}__{}__h{}__{}__{}{}",
            Local::now().format("%Y%m%d_%H%M%S"),
            resolved.data_source,
            resolved.universe_name,
            feature_profile,
            resolved.signal_horizon,
            diagnostic_label_space,
            options.period,
            tag_suffix,
        ))
}

fn default_profile_output_dir(kind_dir: &str, profile_name: &str) -> PathBuf {
    PathBuf::from("results")
        .join("diagnostics")
        .join(kind_dir)
        .join(format!(
            "{}__{}",
            Local::now().format("%Y%m%d_%H%M%S"),
            profile_name
        ))
}

fn factor_store_name_for_current_profile(cfg: &YamlValue) -> Result<String, String> {
    let profile_name = yaml_path_string(cfg, &["features", "profile"]);
    if let Some(profile_name) = profile_name {
        return Ok(resolve_feature_profile(&profile_name)?.factor_store_name);
    }
    Ok("full_factor_space".to_owned())
}

fn build_prefilter_profile_settings(
    options: &ProfileRuntimeOptions,
    resolved: &ResolvedRuntimeConfig,
    date_start: &str,
    date_end: &str,
    profile_write_safe: bool,
) -> Vec<(String, String)> {
    vec![
        setting_path("diagnostics_summary", options.diagnostics_summary.as_ref()),
        setting_path("segment_comparison", options.segment_comparison.as_ref()),
        ("data_source".to_owned(), resolved.data_source.clone()),
        ("universe".to_owned(), resolved.universe_name.clone()),
        ("period".to_owned(), options.batch_options.period.clone()),
        ("date_start".to_owned(), date_start.to_owned()),
        ("date_end".to_owned(), date_end.to_owned()),
        (
            "write_config_profile".to_owned(),
            options.write_config_profile.to_string(),
        ),
        (
            "profile_write_safety".to_owned(),
            if profile_write_safe {
                "training_only"
            } else {
                "unsafe_override"
            }
            .to_owned(),
        ),
        (
            "min_coverage_pct".to_owned(),
            options.thresholds.min_coverage_pct.to_string(),
        ),
        (
            "min_abs_rank_ic".to_owned(),
            options.thresholds.min_abs_rank_ic.to_string(),
        ),
        (
            "min_abs_rank_ic_ir".to_owned(),
            options.thresholds.min_abs_rank_ic_ir.to_string(),
        ),
        (
            "min_monthly_positive_rate".to_owned(),
            options.thresholds.min_monthly_positive_rate.to_string(),
        ),
        (
            "min_segment_directional_hit_mean".to_owned(),
            option_f64_string(options.thresholds.min_segment_directional_hit_mean),
        ),
        (
            "max_segment_rank_ic_mean_range".to_owned(),
            option_f64_string(options.thresholds.max_segment_rank_ic_mean_range),
        ),
        (
            "exclude_direction_flip".to_owned(),
            options.thresholds.exclude_direction_flip.to_string(),
        ),
        ("max_abs_corr".to_owned(), options.max_abs_corr.to_string()),
        (
            "max_features".to_owned(),
            option_usize_string(options.max_features),
        ),
        (
            "correlation_space".to_owned(),
            if options.use_cross_sectional_rank {
                "cross_sectional_rank"
            } else {
                "raw"
            }
            .to_owned(),
        ),
    ]
}

fn build_robust_profile_settings(
    options: &ProfileRuntimeOptions,
    resolved: &ResolvedRuntimeConfig,
    date_start: &str,
    date_end: &str,
    profile_write_safe: bool,
) -> Vec<(String, String)> {
    let mut settings = vec![
        setting_path("raw_summary", options.raw_summary.as_ref()),
        setting_path(
            "raw_segment_comparison",
            options.raw_segment_comparison.as_ref(),
        ),
        setting_path("neutral_summary", options.neutral_summary.as_ref()),
        setting_path(
            "neutral_segment_comparison",
            options.neutral_segment_comparison.as_ref(),
        ),
    ];
    settings.extend(
        build_prefilter_profile_settings(
            options,
            resolved,
            date_start,
            date_end,
            profile_write_safe,
        )
        .into_iter()
        .filter(|(key, _)| key != "diagnostics_summary" && key != "segment_comparison"),
    );
    settings
}

fn setting_path(key: &str, value: Option<&PathBuf>) -> (String, String) {
    (
        key.to_owned(),
        value.map(|path| path_to_string(path)).unwrap_or_default(),
    )
}

fn option_f64_string(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn option_usize_string(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn profile_write_safety_warning(
    cfg: &YamlValue,
    date_start: &str,
    date_end: &str,
    allow_unsafe: bool,
    tool_name: &str,
    diagnostics_paths: &[Option<&PathBuf>],
) -> Result<Option<String>, String> {
    let mut issues = Vec::new();
    if !is_training_date_range(cfg, date_start, date_end)? {
        issues.push(format!(
            "filter date_start={date_start}, date_end={date_end} is outside the training range"
        ));
    }
    issues.extend(check_diagnostics_provenance_safety(cfg, diagnostics_paths)?);
    if issues.is_empty() {
        return Ok(None);
    }
    let train_range = yaml_path(cfg, &["time", "train"])
        .and_then(YamlValue::as_sequence)
        .map(|seq| {
            seq.iter()
                .map(|item| yaml_string_scalar(item).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let message = format!(
        "{tool_name} refuses to write a config feature profile from unsafe diagnostics evidence. Issues: {}. Use a range within train=[{train_range}], or pass --allow-unsafe-profile-write to record this as research-selection leakage.",
        issues.join("; ")
    );
    if allow_unsafe {
        Ok(Some(message))
    } else {
        Err(message)
    }
}

fn check_diagnostics_provenance_safety(
    cfg: &YamlValue,
    diagnostics_paths: &[Option<&PathBuf>],
) -> Result<Vec<String>, String> {
    let mut issues = Vec::new();
    let mut seen_manifests = HashSet::new();
    for raw_path in diagnostics_paths.iter().flatten() {
        let manifest_path = raw_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("manifest.json");
        let manifest_key = manifest_path
            .canonicalize()
            .unwrap_or_else(|_| manifest_path.clone());
        if !seen_manifests.insert(manifest_key) {
            continue;
        }
        if !manifest_path.exists() {
            issues.push(format!(
                "{} has no sibling manifest.json",
                raw_path.display()
            ));
            continue;
        }
        let manifest = match File::open(&manifest_path)
            .map_err(|err| format!("failed to open {}: {err}", manifest_path.display()))
            .and_then(|file| {
                serde_json::from_reader::<_, JsonValue>(file)
                    .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))
            }) {
            Ok(value) => value,
            Err(err) => {
                issues.push(format!(
                    "{} manifest cannot be read: {err}",
                    raw_path.display()
                ));
                continue;
            }
        };
        let metadata = manifest.get("metadata").and_then(JsonValue::as_object);
        let Some(metadata) = metadata else {
            issues.push(format!(
                "{} manifest has no metadata object",
                raw_path.display()
            ));
            continue;
        };
        let period = metadata
            .get("period")
            .and_then(JsonValue::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if period != "train" {
            issues.push(format!(
                "{} diagnostics period={} is not train",
                raw_path.display(),
                if period.is_empty() {
                    "<missing>"
                } else {
                    &period
                }
            ));
        }
        let manifest_start = metadata.get("date_start").and_then(JsonValue::as_str);
        let manifest_end = metadata.get("date_end").and_then(JsonValue::as_str);
        match (manifest_start, manifest_end) {
            (Some(start), Some(end)) => {
                if !is_training_date_range(cfg, start, end)? {
                    issues.push(format!(
                        "{} diagnostics date_start={}, date_end={} is outside the training range",
                        raw_path.display(),
                        start,
                        end
                    ));
                }
            }
            _ => issues.push(format!(
                "{} manifest has no date_start/date_end",
                raw_path.display()
            )),
        }
    }
    Ok(issues)
}

fn is_training_date_range(
    cfg: &YamlValue,
    date_start: &str,
    date_end: &str,
) -> Result<bool, String> {
    let (train_start, train_end) = time_split(cfg, "train")?;
    let start = parse_ymd(date_start)?;
    let end = parse_ymd(date_end)?;
    Ok(start >= parse_ymd(&train_start)? && end <= parse_ymd(&train_end)?)
}

fn parse_ymd(value: &str) -> Result<NaiveDate, String> {
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .map_err(|err| format!("invalid date {value}: {err}"))
}

fn print_profile_build_summary(
    label: &str,
    summary: &ProfileBuildSummary,
    json: bool,
) -> Result<(), String> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(summary)
                .map_err(|err| format!("failed to encode JSON summary: {err}"))?
        );
        return Ok(());
    }
    if let Some(path) = &summary.profile_artifacts.config_profile_path {
        println!("[*] Config profile written: {path}");
    }
    println!("[+] {label} artifacts saved to: {}", summary.output_dir);
    if label.starts_with("Robust") {
        println!(
            "    robust summary: {}",
            summary
                .prefilter_summary
                .robust_summary_path
                .as_deref()
                .unwrap_or("")
        );
    }
    println!("    kept summary: {}", summary.prefilter_summary.kept_path);
    println!(
        "    exact-duplicate kept: {}",
        summary.prefilter_summary.exact_kept_path
    );
    println!(
        "    exact-duplicate drops: {}",
        summary.prefilter_summary.exact_dropped_path
    );
    println!(
        "    corr-pruned drops: {}",
        summary.corr_prune_summary.dropped_path
    );
    println!(
        "    profile yaml: {}",
        summary.profile_artifacts.profile_path
    );
    println!(
        "[*] Selected features ({}):",
        summary.profile_artifacts.selected_feature_count
    );
    for name in &summary.profile_artifacts.selected_features {
        println!("    - {name}");
    }
    Ok(())
}

fn print_summary_preview(summary_path: &Path) -> Result<(), String> {
    if !summary_path.exists() {
        return Ok(());
    }
    let table = read_csv_table(summary_path)?;
    if table.rows.is_empty() {
        return Ok(());
    }
    let columns = [
        "feature",
        "rank_ic_mean",
        "rank_ic_ir",
        "coverage_pct",
        "monotonicity_mean",
        "monthly_rank_ic_directional_hit_rate",
    ];
    println!("\nTop factors by absolute RankIC:");
    println!("{}", columns.join("\t"));
    for row in table.rows.iter().take(10) {
        println!(
            "{}",
            columns
                .iter()
                .map(|column| row.get(*column).map(String::as_str).unwrap_or(""))
                .collect::<Vec<_>>()
                .join("\t")
        );
    }
    Ok(())
}

fn parse_batch_options(args: &[String]) -> Result<BatchOptions, String> {
    let mut options = BatchOptions {
        config: PathBuf::from("configs/config.yaml"),
        config_is_snapshot: false,
        experiment_profile: None,
        feature_profile: None,
        model_profile: None,
        data_source: None,
        set_overrides: Vec::new(),
        topk: None,
        n_drop: None,
        run_tag: None,
        store_dir: None,
        disable_local_store: false,
        rebalance_freq: None,
        signal_horizon: None,
        period: "train".to_owned(),
        date_start: None,
        date_end: None,
        all_features: false,
        quantile_bins: 5,
        top_n: 50,
        no_detail_artifacts: false,
        segment_scheme: "none".to_owned(),
        segments: None,
        base_output_dir: None,
        summary_path: None,
        manifest_path: None,
        cases: Vec::new(),
        industry_neutral: false,
        feature_chunk_size: 64,
        batch_size: 65_536,
        dry_run: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(super::usage().to_owned()),
            "--config" => options.config = PathBuf::from(next_arg(args, &mut index, "--config")?),
            value if value.starts_with("--config=") => {
                options.config = PathBuf::from(split_value(value, "--config")?)
            }
            "--config-is-snapshot" => options.config_is_snapshot = true,
            "--experiment-profile" => {
                options.experiment_profile =
                    Some(next_arg(args, &mut index, "--experiment-profile")?)
            }
            value if value.starts_with("--experiment-profile=") => {
                options.experiment_profile = Some(split_value(value, "--experiment-profile")?)
            }
            "--feature-profile" | "--profile" => {
                options.feature_profile = Some(next_arg(args, &mut index, "--feature-profile")?)
            }
            value if value.starts_with("--feature-profile=") || value.starts_with("--profile=") => {
                options.feature_profile = Some(value.split_once('=').unwrap().1.to_owned())
            }
            "--model-profile" => {
                options.model_profile = Some(next_arg(args, &mut index, "--model-profile")?)
            }
            value if value.starts_with("--model-profile=") => {
                options.model_profile = Some(split_value(value, "--model-profile")?)
            }
            "--data-source" => {
                options.data_source = Some(next_arg(args, &mut index, "--data-source")?)
            }
            value if value.starts_with("--data-source=") => {
                options.data_source = Some(split_value(value, "--data-source")?)
            }
            "--set" => options
                .set_overrides
                .push(next_arg(args, &mut index, "--set")?),
            value if value.starts_with("--set=") => {
                options.set_overrides.push(split_value(value, "--set")?)
            }
            "--topk" => {
                options.topk = Some(parse_usize(
                    next_arg(args, &mut index, "--topk")?,
                    "--topk",
                )?)
            }
            value if value.starts_with("--topk=") => {
                options.topk = Some(parse_usize(split_value(value, "--topk")?, "--topk")?)
            }
            "--n-drop" => {
                options.n_drop = Some(parse_usize(
                    next_arg(args, &mut index, "--n-drop")?,
                    "--n-drop",
                )?)
            }
            value if value.starts_with("--n-drop=") => {
                options.n_drop = Some(parse_usize(split_value(value, "--n-drop")?, "--n-drop")?)
            }
            "--run-tag" => options.run_tag = Some(next_arg(args, &mut index, "--run-tag")?),
            value if value.starts_with("--run-tag=") => {
                options.run_tag = Some(split_value(value, "--run-tag")?)
            }
            "--store-dir" => options.store_dir = Some(next_arg(args, &mut index, "--store-dir")?),
            value if value.starts_with("--store-dir=") => {
                options.store_dir = Some(split_value(value, "--store-dir")?)
            }
            "--disable-local-store" => options.disable_local_store = true,
            "--rebalance-freq" => {
                options.rebalance_freq = Some(parse_usize(
                    next_arg(args, &mut index, "--rebalance-freq")?,
                    "--rebalance-freq",
                )?)
            }
            value if value.starts_with("--rebalance-freq=") => {
                options.rebalance_freq = Some(parse_usize(
                    split_value(value, "--rebalance-freq")?,
                    "--rebalance-freq",
                )?)
            }
            "--signal-horizon" | "--label-horizon" => {
                options.signal_horizon = Some(parse_usize(
                    next_arg(args, &mut index, "--signal-horizon")?,
                    "--signal-horizon",
                )?)
            }
            value
                if value.starts_with("--signal-horizon=")
                    || value.starts_with("--label-horizon=") =>
            {
                let flag = value.split_once('=').unwrap().0;
                options.signal_horizon = Some(parse_usize(
                    value.split_once('=').unwrap().1.to_owned(),
                    flag,
                )?)
            }
            "--period" => options.period = next_arg(args, &mut index, "--period")?,
            value if value.starts_with("--period=") => {
                options.period = split_value(value, "--period")?
            }
            "--date-start" => {
                options.date_start = Some(next_arg(args, &mut index, "--date-start")?)
            }
            value if value.starts_with("--date-start=") => {
                options.date_start = Some(split_value(value, "--date-start")?)
            }
            "--date-end" => options.date_end = Some(next_arg(args, &mut index, "--date-end")?),
            value if value.starts_with("--date-end=") => {
                options.date_end = Some(split_value(value, "--date-end")?)
            }
            "--all-features" => options.all_features = true,
            "--quantile-bins" => {
                options.quantile_bins = parse_usize(
                    next_arg(args, &mut index, "--quantile-bins")?,
                    "--quantile-bins",
                )?
            }
            value if value.starts_with("--quantile-bins=") => {
                options.quantile_bins =
                    parse_usize(split_value(value, "--quantile-bins")?, "--quantile-bins")?
            }
            "--top-n" => {
                options.top_n = parse_usize(next_arg(args, &mut index, "--top-n")?, "--top-n")?
            }
            value if value.starts_with("--top-n=") => {
                options.top_n = parse_usize(split_value(value, "--top-n")?, "--top-n")?
            }
            "--no-detail-artifacts" => options.no_detail_artifacts = true,
            "--segment-scheme" => {
                options.segment_scheme = next_arg(args, &mut index, "--segment-scheme")?
            }
            value if value.starts_with("--segment-scheme=") => {
                options.segment_scheme = split_value(value, "--segment-scheme")?
            }
            "--segments" => options.segments = Some(next_arg(args, &mut index, "--segments")?),
            value if value.starts_with("--segments=") => {
                options.segments = Some(split_value(value, "--segments")?)
            }
            "--base-output-dir" => {
                options.base_output_dir = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--base-output-dir",
                )?))
            }
            value if value.starts_with("--base-output-dir=") => {
                options.base_output_dir =
                    Some(PathBuf::from(split_value(value, "--base-output-dir")?))
            }
            "--summary-path" => {
                options.summary_path =
                    Some(PathBuf::from(next_arg(args, &mut index, "--summary-path")?))
            }
            value if value.starts_with("--summary-path=") => {
                options.summary_path = Some(PathBuf::from(split_value(value, "--summary-path")?))
            }
            "--manifest-path" => {
                options.manifest_path = Some(PathBuf::from(next_arg(
                    args,
                    &mut index,
                    "--manifest-path",
                )?))
            }
            value if value.starts_with("--manifest-path=") => {
                options.manifest_path = Some(PathBuf::from(split_value(value, "--manifest-path")?))
            }
            "--case" => {
                let mut group = Vec::new();
                while index + 1 < args.len() && !args[index + 1].starts_with("--") {
                    index += 1;
                    group.push(args[index].clone());
                }
                if group.is_empty() {
                    return Err("--case requires at least one KEY=VALUE item".to_owned());
                }
                options.cases.push(parse_case(&group)?);
            }
            value if value.starts_with("--case=") => options
                .cases
                .push(parse_case(&[split_value(value, "--case")?])?),
            "--industry-neutral" => options.industry_neutral = true,
            "--engine" => {
                let _ = next_arg(args, &mut index, "--engine")?;
            }
            value if value.starts_with("--engine=") => {
                let _ = split_value(value, "--engine")?;
            }
            "--feature-chunk-size" => {
                options.feature_chunk_size = parse_usize(
                    next_arg(args, &mut index, "--feature-chunk-size")?,
                    "--feature-chunk-size",
                )?
            }
            value if value.starts_with("--feature-chunk-size=") => {
                options.feature_chunk_size = parse_usize(
                    split_value(value, "--feature-chunk-size")?,
                    "--feature-chunk-size",
                )?
            }
            "--batch-size" => {
                options.batch_size =
                    parse_usize(next_arg(args, &mut index, "--batch-size")?, "--batch-size")?
            }
            value if value.starts_with("--batch-size=") => {
                options.batch_size =
                    parse_usize(split_value(value, "--batch-size")?, "--batch-size")?
            }
            "--dry-run" => options.dry_run = true,
            other => return Err(format!("unknown option for single-factor-batch: {other}")),
        }
        index += 1;
    }
    if options.cases.is_empty() {
        return Err("At least one --case is required".to_owned());
    }
    if !matches!(options.period.as_str(), "train" | "valid" | "test" | "all") {
        return Err("--period must be train, valid, test, or all".to_owned());
    }
    if !matches!(
        options.segment_scheme.as_str(),
        "none" | "config_split" | "yearly"
    ) {
        return Err("--segment-scheme must be none, config_split, or yearly".to_owned());
    }
    Ok(options)
}

fn parse_case(raw_group: &[String]) -> Result<DiagnosticsBatchCase, String> {
    let mut payload = BTreeMap::new();
    for raw in raw_group {
        let (key, value) = parse_key_value_arg(raw, "Case")?;
        payload.insert(key, value);
    }
    let feature_profile = yaml_string_value(payload.get("feature_profile"))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Each --case must define feature_profile=...".to_owned())?;
    let name = yaml_string_value(payload.get("name"))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| feature_profile.clone());
    let baseline_feature_profile = yaml_string_value(payload.get("baseline_feature_profile"))
        .or_else(|| yaml_string_value(payload.get("compare_to_feature_profile")))
        .or_else(|| yaml_string_value(payload.get("baseline_profile")))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    let diagnostic_label_space = yaml_string_value(payload.get("diagnostic_label_space"))
        .unwrap_or_else(|| "raw_return".to_owned())
        .trim()
        .to_ascii_lowercase();
    let diagnostic_threshold = payload
        .get("diagnostic_threshold")
        .and_then(yaml_f64_value)
        .unwrap_or(0.0);
    Ok(DiagnosticsBatchCase {
        name,
        feature_profile,
        baseline_feature_profile,
        diagnostic_label_space,
        diagnostic_threshold,
        output_dir: yaml_string_value(payload.get("output_dir"))
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .map(PathBuf::from),
        run_tag: yaml_string_value(payload.get("run_tag"))
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
    })
}

fn resolve_runtime_config(options: &BatchOptions) -> Result<ResolvedRuntimeConfig, String> {
    let mut cfg = if options.config_is_snapshot {
        read_yaml_file(&options.config)?
    } else {
        load_composed_config(options)?
    };
    set_yaml_dotted(
        &mut cfg,
        "runtime.config_path",
        YamlValue::String(path_to_string(&options.config)),
    )?;
    if let Some(data_source) = &options.data_source {
        set_yaml_dotted(
            &mut cfg,
            "data.source",
            YamlValue::String(data_source.clone()),
        )?;
    }
    if let Some(feature_profile) = &options.feature_profile {
        set_yaml_dotted(
            &mut cfg,
            "features.profile",
            YamlValue::String(feature_profile.clone()),
        )?;
    }
    for override_arg in &options.set_overrides {
        let (key, value) = parse_key_value_arg(override_arg, "Override")?;
        set_yaml_dotted(&mut cfg, &key, value)?;
    }
    if let Some(topk) = options.topk {
        set_yaml_dotted(&mut cfg, "strategy.topk", yaml_usize(topk))?;
    }
    if let Some(n_drop) = options.n_drop {
        set_yaml_dotted(&mut cfg, "strategy.n_drop", yaml_usize(n_drop))?;
    }
    if let Some(rebalance_freq) = options.rebalance_freq {
        set_yaml_dotted(
            &mut cfg,
            "backtest.rebalance_freq",
            yaml_usize(rebalance_freq),
        )?;
    }
    if let Some(signal_horizon) = options.signal_horizon {
        set_yaml_dotted(&mut cfg, "label.signal_horizon", yaml_usize(signal_horizon))?;
    }
    if let Some(store_dir) = &options.store_dir {
        set_yaml_dotted(
            &mut cfg,
            "artifacts.store_dir",
            YamlValue::String(store_dir.clone()),
        )?;
    }
    if options.disable_local_store {
        set_yaml_dotted(
            &mut cfg,
            "artifacts.enable_local_store",
            YamlValue::Bool(false),
        )?;
    }

    let data_source = normalize_data_source(
        yaml_path_string(&cfg, &["data", "source"])
            .as_deref()
            .unwrap_or("akshare"),
    )?;
    let signal_horizon = yaml_path_usize(&cfg, &["label", "signal_horizon"]).unwrap_or(20);
    let label_column = format!("label_{signal_horizon}d");
    let feature_profile = yaml_path_string(&cfg, &["features", "profile"]);
    let factor_store_name = if let Some(profile_name) = feature_profile {
        resolve_feature_profile(&profile_name)?.factor_store_name
    } else {
        "full_factor_space".to_owned()
    };
    let factor_store = yaml_path_string(&cfg, &["features", "factor_store_dir"])
        .or_else(|| yaml_path_string(&cfg, &["features", "cache_dir"]))
        .map(PathBuf::from)
        .unwrap_or_else(|| default_factor_store_dir(&data_source, &factor_store_name));
    let universe_name = yaml_path_string(&cfg, &["universe"]).unwrap_or_else(|| "all".to_owned());
    let universe_dir = yaml_path_string(&cfg, &["native", "universe_dir"])
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/universes"));
    Ok(ResolvedRuntimeConfig {
        cfg,
        factor_store,
        data_source,
        signal_horizon,
        label_column,
        universe_name,
        universe_dir,
    })
}

fn load_composed_config(options: &BatchOptions) -> Result<YamlValue, String> {
    let mut cfg = read_yaml_file(&options.config)?;
    ensure_mapping(&mut cfg);
    let experiment_name = options
        .experiment_profile
        .clone()
        .or_else(|| yaml_path_string(&cfg, &["experiment", "profile"]))
        .ok_or_else(|| {
            "Experiment profile must be specified explicitly. Pass --experiment-profile or set experiment.profile in the loaded config."
                .to_owned()
        })?;
    let experiment = resolve_profile_with_path(
        Path::new("configs/experiment_profiles.yaml"),
        &experiment_name,
        "experiment",
    )?;
    let mut experiment_cfg = experiment.config;
    remove_yaml_mapping_key(&mut experiment_cfg, "name");
    remove_yaml_mapping_key(&mut experiment_cfg, "path");
    remove_yaml_mapping_key(&mut experiment_cfg, "sweep");
    deep_merge_yaml(&mut cfg, experiment_cfg);
    set_yaml_dotted(
        &mut cfg,
        "experiment.profile",
        YamlValue::String(experiment_name),
    )?;
    set_yaml_dotted(
        &mut cfg,
        "experiment.profile_path",
        YamlValue::String(experiment.path),
    )?;

    let model_name = options
        .model_profile
        .clone()
        .or_else(|| yaml_path_string(&cfg, &["model", "profile"]))
        .or_else(|| yaml_path_string(&cfg, &["runtime", "default_model_profile"]))
        .or_else(default_model_profile)
        .ok_or_else(|| "model profile could not be resolved".to_owned())?;
    let model = resolve_model_profile(&model_name)?;
    deep_merge_yaml(&mut cfg, model.config);
    set_yaml_dotted(&mut cfg, "model.profile", YamlValue::String(model_name))?;
    set_yaml_dotted(
        &mut cfg,
        "model.profile_path",
        YamlValue::String(model.path),
    )?;
    Ok(cfg)
}

struct ResolvedProfile {
    config: YamlValue,
    path: String,
}

fn resolve_model_profile(profile_name: &str) -> Result<ResolvedProfile, String> {
    let profile_data = read_yaml_file("configs/model_profiles.yaml")?;
    let profiles = yaml_path(&profile_data, &["profiles"])
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| "configs/model_profiles.yaml missing profiles mapping".to_owned())?;
    let entry = profiles
        .get(YamlValue::String(profile_name.to_owned()))
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| format!("Unknown model profile: {profile_name}"))?;
    let path = entry
        .get(YamlValue::String("path".to_owned()))
        .and_then(YamlValue::as_str)
        .ok_or_else(|| format!("model profile {profile_name} missing path"))?;
    let resolved_path = resolve_relative_to_repo(Path::new("configs/model_profiles.yaml"), path);
    Ok(ResolvedProfile {
        config: read_yaml_file(&resolved_path)?,
        path: path_to_string(&resolved_path),
    })
}

fn resolve_profile_with_path(
    profile_config_path: &Path,
    profile_name: &str,
    profile_kind: &str,
) -> Result<ResolvedProfile, String> {
    let profile_data = read_yaml_file(profile_config_path)?;
    let profiles = yaml_path(&profile_data, &["profiles"])
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| format!("{} missing profiles mapping", profile_config_path.display()))?;
    let config = resolve_profile_definition(
        profile_name,
        profiles,
        profile_config_path,
        profile_kind,
        &mut Vec::new(),
    )?;
    let path = yaml_path_string(&config, &["path"])
        .unwrap_or_else(|| format!("{}::{}", profile_config_path.display(), profile_name));
    Ok(ResolvedProfile { config, path })
}

fn resolve_profile_definition(
    profile_name: &str,
    profiles: &YamlMapping,
    profile_config_path: &Path,
    profile_kind: &str,
    stack: &mut Vec<String>,
) -> Result<YamlValue, String> {
    if stack.iter().any(|item| item == profile_name) {
        stack.push(profile_name.to_owned());
        return Err(format!(
            "{} profile inheritance cycle detected: {}",
            profile_kind,
            stack.join(" -> ")
        ));
    }
    let raw_entry = profiles
        .get(YamlValue::String(profile_name.to_owned()))
        .ok_or_else(|| format!("Unknown {profile_kind} profile: {profile_name}"))?;
    let mut entry = raw_entry
        .as_mapping()
        .cloned()
        .ok_or_else(|| format!("{profile_kind} profile {profile_name} must be a mapping"))?;
    let source_path;
    let mut loaded = if let Some(path_value) = entry.remove(YamlValue::String("path".to_owned())) {
        let path = path_value.as_str().ok_or_else(|| {
            format!("{profile_kind} profile {profile_name} path must be a string")
        })?;
        let resolved_path = resolve_relative_to_repo(profile_config_path, path);
        source_path = path_to_string(&resolved_path);
        read_yaml_file(&resolved_path)?
    } else {
        source_path = format!("{}::{}", profile_config_path.display(), profile_name);
        YamlValue::Mapping(YamlMapping::new())
    };
    let extends = entry
        .remove(YamlValue::String("extends".to_owned()))
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default();
    deep_merge_yaml(&mut loaded, YamlValue::Mapping(entry));
    let mut merged = if extends.trim().is_empty() {
        loaded
    } else {
        stack.push(profile_name.to_owned());
        let mut parent = resolve_profile_definition(
            &extends,
            profiles,
            profile_config_path,
            profile_kind,
            stack,
        )?;
        stack.pop();
        remove_yaml_mapping_key(&mut parent, "name");
        remove_yaml_mapping_key(&mut parent, "path");
        deep_merge_yaml(&mut parent, loaded);
        parent
    };
    set_yaml_dotted(
        &mut merged,
        "name",
        YamlValue::String(profile_name.to_owned()),
    )?;
    set_yaml_dotted(&mut merged, "path", YamlValue::String(source_path))?;
    Ok(merged)
}

fn resolve_feature_profile(profile_name: &str) -> Result<FeatureProfileResolution, String> {
    let profile_data = read_yaml_file("configs/feature_profiles.yaml")?;
    let profiles = yaml_path(&profile_data, &["profiles"])
        .and_then(YamlValue::as_mapping)
        .ok_or_else(|| "configs/feature_profiles.yaml missing profiles mapping".to_owned())?;
    let mut profile = resolve_feature_profile_definition(
        profile_name,
        profiles,
        Path::new("configs/feature_profiles.yaml"),
        &mut Vec::new(),
    )?;
    let factor_store_name = yaml_path_string(&profile, &["factor_store_name"])
        .unwrap_or_else(|| "full_factor_space".to_owned());
    let selected_columns = yaml_sequence_strings(yaml_path(&profile, &["selected_columns"]))?;
    let (_selected_columns, source_columns) = if let Some(columns) = selected_columns {
        let repeat_columns = yaml_repeat_columns(yaml_path(&profile, &["repeat_columns"]))?;
        let mut expanded = Vec::new();
        let mut sources = Vec::new();
        for column in columns {
            let repeat_count = repeat_columns.get(&column).copied().unwrap_or(1);
            for repeat_index in 0..repeat_count {
                if repeat_index == 0 {
                    expanded.push(column.clone());
                } else {
                    expanded.push(format!("{}__rep{}", column, repeat_index + 1));
                }
                sources.push(column.clone());
            }
        }
        (Some(expanded), Some(sources))
    } else {
        (None, None)
    };
    remove_yaml_mapping_key(&mut profile, "name");
    Ok(FeatureProfileResolution {
        source_columns,
        factor_store_name,
    })
}

fn resolve_feature_profile_definition(
    profile_name: &str,
    profiles: &YamlMapping,
    profile_config_path: &Path,
    stack: &mut Vec<String>,
) -> Result<YamlValue, String> {
    if stack.iter().any(|item| item == profile_name) {
        stack.push(profile_name.to_owned());
        return Err(format!(
            "feature profile inheritance cycle detected: {}",
            stack.join(" -> ")
        ));
    }
    let raw_entry = profiles
        .get(YamlValue::String(profile_name.to_owned()))
        .ok_or_else(|| format!("Unknown feature profile: {profile_name}"))?;
    let mut entry = raw_entry
        .as_mapping()
        .cloned()
        .ok_or_else(|| format!("feature profile {profile_name} must be a mapping"))?;
    let source_path;
    let mut loaded = if let Some(path_value) = entry.remove(YamlValue::String("path".to_owned())) {
        let path = path_value
            .as_str()
            .ok_or_else(|| format!("feature profile {profile_name} path must be a string"))?;
        let resolved_path = resolve_relative_to_repo(profile_config_path, path);
        source_path = path_to_string(&resolved_path);
        read_yaml_file(&resolved_path)?
    } else {
        source_path = format!("{}::{}", profile_config_path.display(), profile_name);
        YamlValue::Mapping(YamlMapping::new())
    };
    let extends = entry
        .remove(YamlValue::String("extends".to_owned()))
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default();
    let drop_columns = yaml_sequence_strings(
        entry
            .remove(YamlValue::String("drop_columns".to_owned()))
            .as_ref(),
    )?
    .unwrap_or_default();
    let add_columns = yaml_sequence_strings(
        entry
            .remove(YamlValue::String("add_columns".to_owned()))
            .as_ref(),
    )?
    .unwrap_or_default();
    let repeat_columns = entry.remove(YamlValue::String("repeat_columns".to_owned()));
    deep_merge_yaml(&mut loaded, YamlValue::Mapping(entry));
    let mut merged = if extends.trim().is_empty() {
        loaded
    } else {
        stack.push(profile_name.to_owned());
        let mut parent =
            resolve_feature_profile_definition(&extends, profiles, profile_config_path, stack)?;
        stack.pop();
        remove_yaml_mapping_key(&mut parent, "name");
        remove_yaml_mapping_key(&mut parent, "path");
        deep_merge_yaml(&mut parent, loaded);
        parent
    };
    if !drop_columns.is_empty() || !add_columns.is_empty() {
        let base_columns = yaml_sequence_strings(yaml_path(&merged, &["selected_columns"]))?
            .ok_or_else(|| format!("Feature profile '{profile_name}' uses drop/add column mutations but does not resolve to selected_columns"))?;
        let drop_set = drop_columns.into_iter().collect::<HashSet<_>>();
        let mut mutated = base_columns
            .into_iter()
            .filter(|item| !drop_set.contains(item))
            .collect::<Vec<_>>();
        let mut existing = mutated.iter().cloned().collect::<HashSet<_>>();
        for item in add_columns {
            if existing.insert(item.clone()) {
                mutated.push(item);
            }
        }
        set_yaml_dotted(
            &mut merged,
            "selected_columns",
            YamlValue::Sequence(mutated.into_iter().map(YamlValue::String).collect()),
        )?;
    }
    if let Some(repeat_columns) = repeat_columns {
        set_yaml_dotted(&mut merged, "repeat_columns", repeat_columns)?;
    }
    set_yaml_dotted(
        &mut merged,
        "name",
        YamlValue::String(profile_name.to_owned()),
    )?;
    set_yaml_dotted(&mut merged, "path", YamlValue::String(source_path))?;
    Ok(merged)
}

fn resolve_feature_profile_source_columns(
    cfg: &YamlValue,
    factor_store_meta: &JsonValue,
    feature_profile: &str,
) -> Result<Vec<String>, String> {
    let feature_names = read_meta_feature_names(factor_store_meta)?;
    let selected_override =
        yaml_sequence_strings(yaml_path(cfg, &["features", "selected_columns"]))?;
    let source_columns = if let Some(columns) = selected_override {
        columns
    } else {
        let profile = resolve_feature_profile(feature_profile)?;
        profile
            .source_columns
            .unwrap_or_else(|| feature_names.clone())
    };
    if source_columns.is_empty() {
        return Ok(feature_names);
    }
    let duplicate_source_map = exact_duplicate_feature_source_map();
    let resolved_sources = source_columns
        .into_iter()
        .map(|name| duplicate_source_map.get(&name).cloned().unwrap_or(name))
        .collect::<Vec<_>>();
    let feature_set = feature_names.into_iter().collect::<BTreeSet<_>>();
    let missing = resolved_sources
        .iter()
        .filter(|name| !feature_set.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "Selected feature columns not found in cache metadata: {:?}",
            missing
        ));
    }
    Ok(dedup_preserve_order(resolved_sources))
}

fn resolve_incremental_feature_names(
    feature_names: &[String],
    baseline_feature_names: &[String],
) -> Vec<String> {
    let baseline = baseline_feature_names.iter().collect::<HashSet<_>>();
    feature_names
        .iter()
        .filter(|feature| !baseline.contains(*feature))
        .cloned()
        .collect()
}

fn prepare_single_factor_inputs(
    cfg: &YamlValue,
    resolved: &ResolvedRuntimeConfig,
    options: &BatchOptions,
    case: &DiagnosticsBatchCase,
    output_dir: &Path,
    feature_names: &[String],
    metadata: &JsonValue,
) -> Result<PreparedSingleFactorInputs, String> {
    fs::create_dir_all(output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    let features_path = output_dir.join("_rust_selected_features.json");
    let metadata_path = output_dir.join("_rust_metadata_input.json");
    let config_path = output_dir.join("config_snapshot.yaml");
    write_json_file(
        &features_path,
        &serde_json::json!({ "selected_features": feature_names }),
    )?;
    write_json_file(&metadata_path, metadata)?;
    write_yaml_file(&config_path, cfg)?;
    let _ = (resolved, options, case);
    Ok(PreparedSingleFactorInputs {
        features_path,
        metadata_path,
        config_path,
    })
}

struct DateRangeRef<'a> {
    start: &'a str,
    end: &'a str,
}

struct BatchCaseContext<'a> {
    resolved: &'a ResolvedRuntimeConfig,
    options: &'a BatchOptions,
    case: &'a DiagnosticsBatchCase,
    date_range: DateRangeRef<'a>,
}

fn build_case_metadata(
    context: &BatchCaseContext<'_>,
    feature_names: &[String],
    incremental_feature_names: &[String],
    segment_count: usize,
) -> JsonValue {
    let resolved = context.resolved;
    let options = context.options;
    let case = context.case;
    serde_json::json!({
        "data_source": resolved.data_source,
        "universe": resolved.universe_name,
        "feature_profile": case.feature_profile,
        "baseline_feature_profile": case.baseline_feature_profile.clone().unwrap_or_default(),
        "factor_store_dir": resolved.factor_store,
        "signal_horizon": resolved.signal_horizon,
        "period": options.period,
        "date_start": context.date_range.start,
        "date_end": context.date_range.end,
        "diagnostic_label_space": case.diagnostic_label_space,
        "diagnostic_threshold": case.diagnostic_threshold,
        "industry_neutral": options.industry_neutral,
        "neutralized_feature_count": if options.industry_neutral { feature_names.len() } else { 0 },
        "feature_count": feature_names.len(),
        "incremental_feature_count": incremental_feature_names.len(),
        "incremental_features": incremental_feature_names,
        "quantile_bins": options.quantile_bins.max(2),
        "detail_artifacts": !options.no_detail_artifacts,
        "segment_scheme": options.segment_scheme,
        "segment_count": segment_count,
        "engine": "rust",
        "run_tag": case.run_tag.clone().or_else(|| options.run_tag.clone()).unwrap_or_default(),
    })
}

fn build_display_single_factor_command(
    context: &BatchCaseContext<'_>,
    output_dir: &Path,
    features_path: &Path,
    metadata_path: &Path,
    config_path: &Path,
    segments: &[SegmentSpec],
) -> Vec<String> {
    let resolved = context.resolved;
    let options = context.options;
    let case = context.case;
    let mut command = vec![
        "ai4stock-diagnostics".to_owned(),
        "single-factor".to_owned(),
        "--factor-store".to_owned(),
        path_to_string(&resolved.factor_store),
        "--output-dir".to_owned(),
        path_to_string(output_dir),
        "--label-column".to_owned(),
        resolved.label_column.clone(),
        "--signal-horizon".to_owned(),
        resolved.signal_horizon.to_string(),
        "--features-json".to_owned(),
        path_to_string(features_path),
        "--date-start".to_owned(),
        context.date_range.start.to_owned(),
        "--date-end".to_owned(),
        context.date_range.end.to_owned(),
        "--universe-name".to_owned(),
        resolved.universe_name.clone(),
        "--universe-dir".to_owned(),
        path_to_string(&resolved.universe_dir),
        "--quantile-bins".to_owned(),
        options.quantile_bins.max(2).to_string(),
        "--top-n".to_owned(),
        options.top_n.max(1).to_string(),
        "--diagnostic-label-space".to_owned(),
        case.diagnostic_label_space.clone(),
        "--diagnostic-threshold".to_owned(),
        case.diagnostic_threshold.to_string(),
        "--feature-chunk-size".to_owned(),
        options.feature_chunk_size.max(1).to_string(),
        "--batch-size".to_owned(),
        options.batch_size.max(1).to_string(),
        "--metadata-json".to_owned(),
        path_to_string(metadata_path),
        "--config-snapshot".to_owned(),
        path_to_string(config_path),
    ];
    if options.no_detail_artifacts {
        command.push("--no-detail-artifacts".to_owned());
    }
    if options.industry_neutral {
        command.push("--industry-neutral".to_owned());
    }
    if matches!(case.diagnostic_label_space.as_str(), "industry_excess") || options.industry_neutral
    {
        if let Some(path) = industry_map_path(&resolved.cfg) {
            command.extend(["--industry-map".to_owned(), path_to_string(&path)]);
        }
    }
    if case.diagnostic_label_space == "benchmark_excess" {
        append_benchmark_display_options(&mut command, &resolved.cfg);
    }
    for segment in segments {
        command.extend(["--segment".to_owned(), display_segment(segment)]);
    }
    command
}

fn append_benchmark_display_options(command: &mut Vec<String>, cfg: &YamlValue) {
    let mode = yaml_path_string(cfg, &["backtest", "benchmark", "mode"])
        .unwrap_or_else(|| "cross_section_mean".to_owned());
    command.extend(["--benchmark-mode".to_owned(), mode.clone()]);
    if mode == "file" {
        if let Some(path) = yaml_path_string(cfg, &["backtest", "benchmark", "path"]) {
            command.extend(["--benchmark-path".to_owned(), path]);
        }
        command.extend([
            "--benchmark-date-column".to_owned(),
            yaml_path_string(cfg, &["backtest", "benchmark", "date_column"])
                .unwrap_or_else(|| "date".to_owned()),
            "--benchmark-value-column".to_owned(),
            yaml_path_string(cfg, &["backtest", "benchmark", "value_column"])
                .unwrap_or_else(|| "close".to_owned()),
            "--benchmark-value-type".to_owned(),
            yaml_path_string(cfg, &["backtest", "benchmark", "value_type"])
                .unwrap_or_else(|| "close".to_owned()),
        ]);
    }
}

fn benchmark_options(cfg: &YamlValue, label_space: &str) -> Result<BenchmarkOptions, String> {
    let mut options = BenchmarkOptions::default();
    if label_space != "benchmark_excess" {
        return Ok(options);
    }
    let mode = yaml_path_string(cfg, &["backtest", "benchmark", "mode"])
        .unwrap_or_else(|| "cross_section_mean".to_owned());
    options.mode = BenchmarkMode::parse(&mode)?;
    options.path = yaml_path_string(cfg, &["backtest", "benchmark", "path"]).map(PathBuf::from);
    options.date_column = yaml_path_string(cfg, &["backtest", "benchmark", "date_column"])
        .unwrap_or_else(|| "date".to_owned());
    options.value_column = yaml_path_string(cfg, &["backtest", "benchmark", "value_column"])
        .unwrap_or_else(|| "close".to_owned());
    let value_type = yaml_path_string(cfg, &["backtest", "benchmark", "value_type"])
        .unwrap_or_else(|| "close".to_owned());
    options.value_type = BenchmarkValueType::parse(&value_type)?;
    Ok(options)
}

fn industry_map_path(cfg: &YamlValue) -> Option<PathBuf> {
    let data_source =
        yaml_path_string(cfg, &["data", "source"]).unwrap_or_else(|| "akshare".to_owned());
    Some(
        PathBuf::from("data")
            .join(data_source)
            .join("raw")
            .join("meta")
            .join("symbol_cache.parquet"),
    )
}

fn resolve_period_dates(
    cfg: &YamlValue,
    options: &BatchOptions,
) -> Result<(String, String), String> {
    if options.date_start.is_some() || options.date_end.is_some() {
        return match (&options.date_start, &options.date_end) {
            (Some(start), Some(end)) => Ok((start.clone(), end.clone())),
            _ => Err(
                "Provide both --date-start and --date-end when overriding the diagnostics range."
                    .to_owned(),
            ),
        };
    }
    if options.period == "all" {
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for split in ["train", "valid", "test"] {
            let (start, end) = time_split(cfg, split)?;
            starts.push(start);
            ends.push(end);
        }
        starts.sort();
        ends.sort();
        return Ok((starts[0].clone(), ends[ends.len() - 1].clone()));
    }
    time_split(cfg, &options.period)
}

fn resolve_segments(
    cfg: &YamlValue,
    options: &BatchOptions,
    main_start: &str,
    main_end: &str,
) -> Result<Vec<SegmentSpec>, String> {
    let mut raw_segments = Vec::<(String, String, String)>::new();
    if options.segment_scheme == "yearly" {
        let start = NaiveDate::parse_from_str(main_start, "%Y-%m-%d")
            .map_err(|err| format!("invalid main start date {main_start}: {err}"))?;
        let end = NaiveDate::parse_from_str(main_end, "%Y-%m-%d")
            .map_err(|err| format!("invalid main end date {main_end}: {err}"))?;
        for year in start.year()..=end.year() {
            let seg_start = start.max(NaiveDate::from_ymd_opt(year, 1, 1).unwrap());
            let seg_end = end.min(NaiveDate::from_ymd_opt(year, 12, 31).unwrap());
            if seg_start <= seg_end {
                raw_segments.push((
                    format!("y{year}"),
                    seg_start.to_string(),
                    seg_end.to_string(),
                ));
            }
        }
    }
    if options.segment_scheme == "config_split" {
        let main_start_date = NaiveDate::parse_from_str(main_start, "%Y-%m-%d")
            .map_err(|err| format!("invalid main start date {main_start}: {err}"))?;
        let main_end_date = NaiveDate::parse_from_str(main_end, "%Y-%m-%d")
            .map_err(|err| format!("invalid main end date {main_end}: {err}"))?;
        for split in ["train", "valid", "test"] {
            let (start, end) = time_split(cfg, split)?;
            let start = NaiveDate::parse_from_str(&start, "%Y-%m-%d")
                .map_err(|err| format!("invalid {split} start date: {err}"))?;
            let end = NaiveDate::parse_from_str(&end, "%Y-%m-%d")
                .map_err(|err| format!("invalid {split} end date: {err}"))?;
            let clipped_start = start.max(main_start_date);
            let clipped_end = end.min(main_end_date);
            if clipped_start <= clipped_end {
                raw_segments.push((
                    split.to_owned(),
                    clipped_start.to_string(),
                    clipped_end.to_string(),
                ));
            }
        }
    }
    if let Some(custom) = &options.segments {
        for item in custom.split(';') {
            let text = item.trim();
            if text.is_empty() {
                continue;
            }
            let parts = text.split(':').map(str::trim).collect::<Vec<_>>();
            if parts.len() != 3 {
                return Err(format!(
                    "Custom segments must use name:start:end format. Got: {text}"
                ));
            }
            raw_segments.push((
                parts[0].to_owned(),
                parts[1].to_owned(),
                parts[2].to_owned(),
            ));
        }
    }
    let mut seen = HashSet::new();
    let mut segments = Vec::new();
    for (name, start, end) in raw_segments {
        if seen.insert(name.clone()) {
            segments.push(SegmentSpec::parse(&format!("{name}:{start}:{end}"))?);
        }
    }
    Ok(segments)
}

fn time_split(cfg: &YamlValue, split: &str) -> Result<(String, String), String> {
    let sequence = yaml_path(cfg, &["time", split])
        .and_then(YamlValue::as_sequence)
        .ok_or_else(|| format!("config missing time.{split} range"))?;
    if sequence.len() < 2 {
        return Err(format!("time.{split} must have [start, end]"));
    }
    Ok((
        yaml_string_scalar(&sequence[0])?,
        yaml_string_scalar(&sequence[1])?,
    ))
}

fn load_factor_store_metadata(factor_store: &Path) -> Result<JsonValue, String> {
    let meta_path = factor_store.join("meta.json");
    let file = File::open(&meta_path).map_err(|err| {
        format!(
            "Parquet factor store metadata missing: {}: {err}",
            meta_path.display()
        )
    })?;
    serde_json::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", meta_path.display()))
}

fn read_meta_feature_names(meta: &JsonValue) -> Result<Vec<String>, String> {
    meta.get("feature_names")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "Cache metadata missing non-empty feature_names.".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "feature_names entries must be strings".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|items| {
            if items.is_empty() {
                Err("Cache metadata missing non-empty feature_names.".to_owned())
            } else {
                Ok(items)
            }
        })
}

fn write_single_factor_subset_artifacts(
    summary: &CsvTable,
    output_dir: &Path,
    prefix: &str,
    top_n: usize,
    segment_comparison: Option<&CsvTable>,
    feature_names: &[String],
) -> Result<BTreeMap<String, String>, String> {
    fs::create_dir_all(output_dir)
        .map_err(|err| format!("failed to create {}: {err}", output_dir.display()))?;
    let summary_path = output_dir.join(format!("single_factor_{prefix}_summary.csv"));
    let top_abs_rankic_path = output_dir.join(format!("single_factor_{prefix}_top_abs_rankic.csv"));
    let top_rankic_path = output_dir.join(format!("single_factor_{prefix}_top_rankic.csv"));
    let top_icir_path = output_dir.join(format!("single_factor_{prefix}_top_rankic_ir.csv"));
    write_csv_table(&summary_path, summary)?;
    write_csv_table(
        &top_abs_rankic_path,
        &sort_top_factors(
            summary,
            &[("rank_ic_abs_mean", false), ("coverage_pct", false)],
            top_n,
        ),
    )?;
    write_csv_table(
        &top_rankic_path,
        &sort_top_factors(
            summary,
            &[("rank_ic_mean", false), ("coverage_pct", false)],
            top_n,
        ),
    )?;
    write_csv_table(
        &top_icir_path,
        &sort_top_factors(
            summary,
            &[("rank_ic_ir", false), ("coverage_pct", false)],
            top_n,
        ),
    )?;
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        format!("{prefix}_summary_csv"),
        path_to_string(&summary_path),
    );
    artifacts.insert(
        format!("{prefix}_top_abs_rankic_csv"),
        path_to_string(&top_abs_rankic_path),
    );
    artifacts.insert(
        format!("{prefix}_top_rankic_csv"),
        path_to_string(&top_rankic_path),
    );
    artifacts.insert(
        format!("{prefix}_top_rankic_ir_csv"),
        path_to_string(&top_icir_path),
    );
    if let Some(segment_comparison) = segment_comparison {
        if segment_comparison
            .headers
            .iter()
            .any(|header| header == "feature")
        {
            let subset = filter_table_for_features(segment_comparison, feature_names);
            let path = output_dir.join(format!("single_factor_{prefix}_segment_comparison.csv"));
            write_csv_table(&path, &subset)?;
            artifacts.insert(
                format!("{prefix}_segment_comparison_csv"),
                path_to_string(&path),
            );
        }
    }
    Ok(artifacts)
}

fn read_csv_table(path: &Path) -> Result<CsvTable, String> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let headers = reader
        .headers()
        .map_err(|err| format!("failed to read {} headers: {err}", path.display()))?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for record in reader.records() {
        let record =
            record.map_err(|err| format!("failed to read {} row: {err}", path.display()))?;
        rows.push(
            headers
                .iter()
                .cloned()
                .zip(record.iter().map(str::to_owned))
                .collect::<BTreeMap<_, _>>(),
        );
    }
    Ok(CsvTable { headers, rows })
}

fn write_csv_table(path: &Path, table: &CsvTable) -> Result<(), String> {
    let mut writer = csv::Writer::from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record(&table.headers)
        .map_err(|err| format!("failed to write {} headers: {err}", path.display()))?;
    for row in &table.rows {
        writer
            .write_record(
                table
                    .headers
                    .iter()
                    .map(|header| row.get(header).map(String::as_str).unwrap_or("")),
            )
            .map_err(|err| format!("failed to write {} row: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn filter_table_for_features(table: &CsvTable, feature_names: &[String]) -> CsvTable {
    if table.rows.is_empty() {
        return table.clone();
    }
    if feature_names.is_empty() {
        return CsvTable {
            headers: table.headers.clone(),
            rows: Vec::new(),
        };
    }
    let feature_set = feature_names.iter().cloned().collect::<HashSet<_>>();
    CsvTable {
        headers: table.headers.clone(),
        rows: table
            .rows
            .iter()
            .filter(|row| {
                row.get("feature")
                    .map(|feature| feature_set.contains(feature))
                    .unwrap_or(false)
            })
            .cloned()
            .collect(),
    }
}

fn sort_top_factors(table: &CsvTable, keys: &[(&str, bool)], top_n: usize) -> CsvTable {
    if table.rows.is_empty()
        || keys
            .iter()
            .any(|(key, _)| !table.headers.iter().any(|header| header == key))
    {
        return CsvTable {
            headers: table.headers.clone(),
            rows: table.rows.iter().take(top_n).cloned().collect(),
        };
    }
    let mut rows = table.rows.clone();
    rows.sort_by(|left, right| {
        for (key, ascending) in keys {
            let left_value = parse_sort_float(left.get(*key));
            let right_value = parse_sort_float(right.get(*key));
            let ordering = match (left_value, right_value) {
                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };
            let ordering = if *ascending {
                ordering
            } else {
                ordering.reverse()
            };
            if ordering != std::cmp::Ordering::Equal {
                return ordering;
            }
        }
        std::cmp::Ordering::Equal
    });
    rows.truncate(top_n);
    CsvTable {
        headers: table.headers.clone(),
        rows,
    }
}

fn parse_sort_float(value: Option<&String>) -> Option<f64> {
    value
        .and_then(|text| text.parse::<f64>().ok())
        .filter(|value| value.is_finite())
}

fn count_abs_metric_ge(table: &CsvTable, column: &str, threshold: f64) -> usize {
    table
        .rows
        .iter()
        .filter_map(|row| row.get(column))
        .filter_map(|value| value.parse::<f64>().ok())
        .filter(|value| value.abs() >= threshold)
        .count()
}

fn merge_artifacts_into_manifest(
    output_dir: &Path,
    artifacts: &BTreeMap<String, String>,
) -> Result<(), String> {
    if artifacts.is_empty() {
        return Ok(());
    }
    let manifest_path = output_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(());
    }
    let file = File::open(&manifest_path)
        .map_err(|err| format!("failed to open {}: {err}", manifest_path.display()))?;
    let mut manifest: JsonValue = serde_json::from_reader(file)
        .map_err(|err| format!("failed to parse {}: {err}", manifest_path.display()))?;
    if !manifest.get("artifacts").is_some_and(JsonValue::is_object) {
        manifest["artifacts"] = serde_json::json!({});
    }
    let artifact_obj = manifest["artifacts"]
        .as_object_mut()
        .ok_or_else(|| "manifest artifacts field is not an object".to_owned())?;
    for (key, value) in artifacts {
        artifact_obj.insert(key.clone(), JsonValue::String(value.clone()));
    }
    write_json_file(&manifest_path, &manifest)
}

fn init_tsv(path: &Path, headers: &[&str]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    writer
        .write_record(headers)
        .map_err(|err| format!("failed to write {} headers: {err}", path.display()))?;
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn append_tsv_row(path: &Path, row: Vec<String>) -> Result<(), String> {
    let file = fs::OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_writer(file);
    writer
        .write_record(row)
        .map_err(|err| format!("failed to append {}: {err}", path.display()))?;
    writer
        .flush()
        .map_err(|err| format!("failed to flush {}: {err}", path.display()))
}

fn row_value(row: Option<&BTreeMap<String, String>>, key: &str) -> String {
    row.and_then(|row| row.get(key))
        .cloned()
        .unwrap_or_default()
}

fn resolve_case_output_dir(base_output_dir: &Path, case: &DiagnosticsBatchCase) -> PathBuf {
    case.output_dir.clone().unwrap_or_else(|| {
        base_output_dir.join(format!(
            "{}__{}__{}",
            slugify(&case.name),
            slugify(&case.feature_profile),
            slugify(&case.diagnostic_label_space)
        ))
    })
}

fn slugify(value: &str) -> String {
    let mut safe = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>();
    while safe.contains("--") {
        safe = safe.replace("--", "-");
    }
    let safe = safe.trim_matches('-').to_owned();
    if safe.is_empty() {
        "case".to_owned()
    } else {
        safe
    }
}

fn write_json_file(path: &Path, value: &JsonValue) -> Result<(), String> {
    write_json_pretty(path, value, false)
}

fn ensure_mapping(value: &mut YamlValue) {
    if !value.is_mapping() {
        *value = YamlValue::Mapping(YamlMapping::new());
    }
}

fn set_yaml_dotted(cfg: &mut YamlValue, dotted_key: &str, value: YamlValue) -> Result<(), String> {
    ensure_mapping(cfg);
    let parts = dotted_key
        .split('.')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err("override key must be non-empty".to_owned());
    }
    let mut current = cfg;
    for part in &parts[..parts.len() - 1] {
        ensure_mapping(current);
        let map = current.as_mapping_mut().unwrap();
        current = map
            .entry(YamlValue::String((*part).to_owned()))
            .or_insert_with(|| YamlValue::Mapping(YamlMapping::new()));
    }
    ensure_mapping(current);
    current
        .as_mapping_mut()
        .unwrap()
        .insert(YamlValue::String(parts[parts.len() - 1].to_owned()), value);
    Ok(())
}

fn remove_yaml_mapping_key(value: &mut YamlValue, key: &str) {
    if let Some(map) = value.as_mapping_mut() {
        map.remove(YamlValue::String(key.to_owned()));
    }
}

fn yaml_path<'a>(value: &'a YamlValue, path: &[&str]) -> Option<&'a YamlValue> {
    let mut current = value;
    for key in path {
        current = current
            .as_mapping()?
            .get(YamlValue::String((*key).to_owned()))?;
    }
    Some(current)
}

fn yaml_path_string(value: &YamlValue, path: &[&str]) -> Option<String> {
    yaml_path(value, path).and_then(|value| value.as_str().map(str::to_owned))
}

fn yaml_path_usize(value: &YamlValue, path: &[&str]) -> Option<usize> {
    yaml_path(value, path).and_then(|value| match value {
        YamlValue::Number(number) => number.as_u64().map(|value| value as usize),
        YamlValue::String(text) => text.parse::<usize>().ok(),
        _ => None,
    })
}

fn yaml_sequence_strings(value: Option<&YamlValue>) -> Result<Option<Vec<String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let sequence = value
        .as_sequence()
        .ok_or_else(|| "expected a list of strings".to_owned())?;
    sequence
        .iter()
        .map(yaml_string_scalar)
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn yaml_repeat_columns(value: Option<&YamlValue>) -> Result<HashMap<String, usize>, String> {
    let Some(value) = value else {
        return Ok(HashMap::new());
    };
    let mapping = value
        .as_mapping()
        .ok_or_else(|| "repeat_columns must be a mapping".to_owned())?;
    let mut out = HashMap::new();
    for (key, value) in mapping {
        let name = yaml_string_scalar(key)?;
        let count = match value {
            YamlValue::Number(number) => number.as_u64().map(|value| value as usize),
            YamlValue::String(text) => text.parse::<usize>().ok(),
            _ => None,
        }
        .ok_or_else(|| format!("repeat_columns[{name}] must be a positive integer"))?;
        if count == 0 {
            return Err(format!("repeat_columns[{name}] must be a positive integer"));
        }
        out.insert(name, count);
    }
    Ok(out)
}

fn yaml_string_scalar(value: &YamlValue) -> Result<String, String> {
    match value {
        YamlValue::String(text) => Ok(text.clone()),
        YamlValue::Number(number) => Ok(number.to_string()),
        YamlValue::Bool(value) => Ok(value.to_string()),
        _ => Err("expected scalar value".to_owned()),
    }
}

fn yaml_string_value(value: Option<&YamlValue>) -> Option<String> {
    value.and_then(|value| yaml_string_scalar(value).ok())
}

fn yaml_f64_value(value: &YamlValue) -> Option<f64> {
    match value {
        YamlValue::Number(number) => number.as_f64(),
        YamlValue::String(text) => text.parse::<f64>().ok(),
        _ => None,
    }
}

fn yaml_usize(value: usize) -> YamlValue {
    YamlValue::Number(serde_yaml::Number::from(value as u64))
}

fn parse_key_value_arg(raw: &str, label: &str) -> Result<(String, YamlValue), String> {
    let Some((key, value)) = raw.trim().split_once('=') else {
        return Err(format!("{label} must be in key=value form, got: {raw}"));
    };
    let key = key.trim().to_owned();
    if key.is_empty() {
        return Err(format!("{label} key must be non-empty, got: {raw}"));
    }
    let parsed = serde_yaml::from_str::<YamlValue>(value.trim())
        .unwrap_or_else(|_| YamlValue::String(value.trim().to_owned()));
    Ok((key, parsed))
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

fn default_model_profile() -> Option<String> {
    let profile_data = read_yaml_file("configs/model_profiles.yaml").ok()?;
    yaml_path_string(&profile_data, &["default_profile"])
}

fn resolve_relative_to_repo(profile_config_path: &Path, raw_path: &str) -> PathBuf {
    let path = PathBuf::from(raw_path);
    if path.is_absolute() {
        return path;
    }
    profile_config_path
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

fn normalize_data_source(value: &str) -> Result<String, String> {
    let source = match value.trim().to_ascii_lowercase().as_str() {
        "" => "akshare".to_owned(),
        "eastmoney" | "em" => "akshare".to_owned(),
        "akshare" => "akshare".to_owned(),
        "tushare" => "tushare".to_owned(),
        other => {
            return Err(format!(
                "Unsupported data source: {other}. Available: akshare, tushare"
            ))
        }
    };
    Ok(source)
}

fn default_factor_store_dir(data_source: &str, factor_store_name: &str) -> PathBuf {
    if data_source == "akshare" {
        PathBuf::from("data/factor_store").join(factor_store_name)
    } else {
        PathBuf::from("data/factor_store").join(format!("{data_source}_{factor_store_name}"))
    }
}

fn exact_duplicate_feature_source_map() -> HashMap<String, String> {
    let mut source_map = HashMap::new();
    for group in known_exact_duplicate_feature_groups() {
        let canonical = group[0].clone();
        source_map.insert(canonical.clone(), canonical.clone());
        for feature_name in group.into_iter().skip(1) {
            source_map.insert(feature_name, canonical.clone());
        }
    }
    source_map
}

fn known_exact_duplicate_feature_groups() -> Vec<Vec<String>> {
    let mut groups = Vec::new();
    for window in [5, 10, 20, 30, 60] {
        groups.push(vec![format!("RSV{window}"), format!("TEMP_rsv_{window}")]);
        groups.push(vec![
            format!("CORR{window}"),
            format!("TEMP_corr_cv_{window}"),
        ]);
    }
    for window in [20, 60] {
        groups.push(vec![
            format!("LGBM_ret_{window}"),
            format!("TEMP_ret_{window}"),
        ]);
    }
    for window in [20, 60, 120] {
        groups.push(vec![
            format!("LGBM_dist_ma{window}"),
            format!("TEMP_ma_gap_{window}"),
        ]);
    }
    groups.extend([
        vec!["LGBM_std_60".to_owned(), "TEMP_std_60".to_owned()],
        vec!["LGBM_amihud_20".to_owned(), "TEMP_amihud_20".to_owned()],
        vec![
            "LGBM_turnover_20".to_owned(),
            "TEMP_turnover_mean_20".to_owned(),
        ],
        vec![
            "LGBM_dist_high_20".to_owned(),
            "TEMP_high_gap_20".to_owned(),
        ],
        vec!["LGBM_dist_low_20".to_owned(), "TEMP_low_gap_20".to_owned()],
    ]);
    groups
}

fn dedup_preserve_order(items: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            out.push(item);
        }
    }
    out
}

fn display_segment(segment: &SegmentSpec) -> String {
    format!(
        "{}:{}:{}",
        segment.name,
        ns_to_date(segment.start_ns),
        ns_to_date(segment.end_ns)
    )
}

fn ns_to_date(ns: i64) -> String {
    let seconds = ns.div_euclid(1_000_000_000);
    let nanos = ns.rem_euclid(1_000_000_000) as u32;
    Utc.timestamp_opt(seconds, nanos)
        .single()
        .map(|dt| dt.date_naive().to_string())
        .unwrap_or_else(|| ns.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_feature_names_preserve_case_order() {
        let incremental = resolve_incremental_feature_names(
            &[
                "base_a".to_owned(),
                "layer_b".to_owned(),
                "base_c".to_owned(),
                "layer_d".to_owned(),
            ],
            &[
                "base_a".to_owned(),
                "base_c".to_owned(),
                "missing_from_case".to_owned(),
            ],
        );
        assert_eq!(incremental, vec!["layer_b", "layer_d"]);
    }

    #[test]
    fn filter_table_for_features_keeps_summary_order() {
        let table = CsvTable {
            headers: vec!["feature".to_owned(), "rank_ic_abs_mean".to_owned()],
            rows: vec![
                row([("feature", "base_a"), ("rank_ic_abs_mean", "0.09")]),
                row([("feature", "layer_b"), ("rank_ic_abs_mean", "0.07")]),
                row([("feature", "layer_c"), ("rank_ic_abs_mean", "0.03")]),
            ],
        };
        let filtered =
            filter_table_for_features(&table, &["layer_c".to_owned(), "layer_b".to_owned()]);
        assert_eq!(
            filtered
                .rows
                .iter()
                .map(|row| row.get("feature").unwrap().as_str())
                .collect::<Vec<_>>(),
            vec!["layer_b", "layer_c"]
        );
    }

    #[test]
    fn parses_batch_case_aliases() {
        let case = parse_case(&[
            "feature_profile=core".to_owned(),
            "compare_to_feature_profile=base".to_owned(),
            "diagnostic_label_space=industry_excess".to_owned(),
            "diagnostic_threshold=0.01".to_owned(),
        ])
        .unwrap();
        assert_eq!(case.name, "core");
        assert_eq!(case.baseline_feature_profile.as_deref(), Some("base"));
        assert_eq!(case.diagnostic_label_space, "industry_excess");
        assert_eq!(case.diagnostic_threshold, 0.01);
    }

    #[test]
    fn diagnostics_provenance_accepts_train_manifest() {
        let root = temp_test_dir("diagnostics_provenance_accepts_train_manifest");
        fs::create_dir_all(&root).unwrap();
        let summary_path = root.join("single_factor_summary.csv");
        fs::write(&summary_path, "feature,rank_ic_mean\nalpha,0.03\n").unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{"metadata":{"period":"train","date_start":"2020-01-01","date_end":"2020-12-31"}}"#,
        )
        .unwrap();
        let cfg = train_cfg("2020-01-01", "2020-12-31");

        let issues = check_diagnostics_provenance_safety(&cfg, &[Some(&summary_path)]).unwrap();

        assert!(issues.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_write_safety_rejects_non_train_diagnostics_manifest() {
        let root = temp_test_dir("profile_write_safety_rejects_non_train_diagnostics_manifest");
        fs::create_dir_all(&root).unwrap();
        let summary_path = root.join("single_factor_summary.csv");
        fs::write(&summary_path, "feature,rank_ic_mean\nalpha,0.03\n").unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{"metadata":{"period":"test","date_start":"2021-01-01","date_end":"2021-12-31"}}"#,
        )
        .unwrap();
        let cfg = train_cfg("2020-01-01", "2020-12-31");

        let error = profile_write_safety_warning(
            &cfg,
            "2020-01-01",
            "2020-12-31",
            false,
            "run_build_prefiltered_profile.py",
            &[Some(&summary_path)],
        )
        .unwrap_err();

        assert!(error.contains("diagnostics period=test is not train"));
        assert!(error.contains("diagnostics date_start=2021-01-01, date_end=2021-12-31"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn profile_write_safety_rejects_missing_manifest() {
        let root = temp_test_dir("profile_write_safety_rejects_missing_manifest");
        fs::create_dir_all(&root).unwrap();
        let summary_path = root.join("single_factor_summary.csv");
        fs::write(&summary_path, "feature,rank_ic_mean\nalpha,0.03\n").unwrap();
        let cfg = train_cfg("2020-01-01", "2020-12-31");

        let error = profile_write_safety_warning(
            &cfg,
            "2020-01-01",
            "2020-12-31",
            false,
            "run_build_prefiltered_profile.py",
            &[Some(&summary_path)],
        )
        .unwrap_err();

        assert!(error.contains("has no sibling manifest.json"));
        fs::remove_dir_all(root).unwrap();
    }

    fn train_cfg(start: &str, end: &str) -> YamlValue {
        serde_yaml::from_str(&format!(
            "time:\n  train: [{start}, {end}]\n  valid: [2021-01-01, 2021-12-31]\n  test: [2022-01-01, 2022-12-31]\n"
        ))
        .unwrap()
    }

    fn temp_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ai4stock2_{name}_{}_{}",
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap()
        ))
    }

    fn row<const N: usize>(items: [(&str, &str); N]) -> BTreeMap<String, String> {
        items
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect()
    }
}
