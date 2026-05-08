use arrow_array::{
    Array, ArrayRef, Date32Array, Date64Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{Local, NaiveDate};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const DEFAULT_TOKEN_ENV: &str = "TUSHARE_TOKEN";
const DEFAULT_WORKERS: usize = 8;
const RATE_LIMIT_COOLDOWN_SECONDS: u64 = 60;
const NS_PER_DAY: i64 = 86_400_000_000_000;
const EPS: f64 = 1e-12;
const TS_ROOT: &str = "data/tushare";
const TUSHARE_SOURCE_DIR: &str = "data/tushare/source";
const TUSHARE_SOURCE_BUCKET_DIR: &str = "data/tushare/source/buckets";
const TUSHARE_SOURCE_MANIFEST_PATH: &str = "data/tushare/source/manifest.parquet";
const TUSHARE_SOURCE_META_PATH: &str = "data/tushare/source/meta.json";
const TUSHARE_SYMBOL_CACHE_PATH: &str = "data/tushare/raw/meta/symbol_cache.parquet";
const TUSHARE_INDUSTRY_CONTEXT_PATH: &str = "data/tushare/raw/meta/industry_context.parquet";
const TUSHARE_EVENT_AVAILABILITY_POLICY: &str = "strict_next_trading_day_after_ann_date";
const TUSHARE_INDUSTRY_MAPPING_POLICY: &str = "static_symbol_cache_current_classification";
const PACKED_SOURCE_DEPENDENCY_SIGNATURE_POLICY: &str =
    "processed_sidecars_industry_context_file_signatures_v1";
const DEFAULT_PACKED_SOURCE_BUCKET_COUNT: usize = 128;
const DEFAULT_UNIVERSE_DIR: &str = "data/universes";
const DEFAULT_STAGES: &[&str] = &["daily", "daily_basic", "adj_factor", "stk_limit"];
const ALL_STAGES: &[&str] = &[
    "daily",
    "daily_basic",
    "adj_factor",
    "stk_limit",
    "fina_indicator",
    "dividend",
    "forecast",
    "express",
    "processed",
];

fn usage() -> &'static str {
    "\
ai4stock-collect: native collection orchestration for AI4Stock2

Usage:
  ai4stock-collect tushare [options]
  ai4stock-collect akshare [options]
  ai4stock-collect universes [options]

Options:
  Tushare:
  --symbols <CSV>              Comma-separated symbols, supports bare code or ts_code.
  --update                     Resolve locally known/cached Tushare symbols.
  --rebuild-processed          Rebuild processed parquet from local raw files.
  --rebuild-packed-source-only Rebuild Tushare packed source buckets from local processed files and exit.
  --stages <CSV|all>           Stages to update. Default: daily,daily_basic,adj_factor,stk_limit.
  --refresh-symbols            Refresh symbol cache before resolving --update symbols.
  --refresh-symbols-only       Refresh symbol cache and exit.
  --refresh-benchmarks         Refresh Tushare benchmark files before other work.
  --refresh-benchmarks-only    Refresh Tushare benchmark files and exit.
  --no-rebuild-packed-source   Skip packed source rebuild after local updates.
  --workers <N>                Worker count. Default: 8.
  --end-date <DATE>            Target end date in YYYY-MM-DD. Default: today.
  --token-env <NAME>           Env var containing Tushare token. Default: TUSHARE_TOKEN.

  AkShare:
  --all                        Process all symbols from the cached/live stock list.
  --adjust <hfq|qfq|raw>       Eastmoney adjustment mode. Default: hfq.
  --network-backend <NAME>     cookie or proxy_patch. Default: cookie.
  --proxy-auth-token <TOKEN>   Proxy auth token for proxy_patch backend.
  --refresh-stock-list         Refresh stock-list cache while resolving symbols.
  --refresh-stock-list-only    Refresh stock-list cache and exit.

  Universes:
  --universes <CSV>            Comma-separated universe names. Default: csi300,csi500,zz1000.
  --output-dir <PATH>          Universe output directory. Default: data/universes.
  --allow-static-membership    Allow current/static membership if PIT intervals are unavailable.

  Common:
  --json                       Print machine-readable JSON summary.
  -h, --help                   Show this help.
"
}

#[derive(Debug, Clone)]
struct TushareCollectOptions {
    symbols_csv: Option<String>,
    update: bool,
    rebuild_processed: bool,
    rebuild_packed_source_only: bool,
    stages: Vec<String>,
    refresh_symbols: bool,
    refresh_symbols_only: bool,
    refresh_benchmarks: bool,
    refresh_benchmarks_only: bool,
    rebuild_packed_source: bool,
    workers: usize,
    end_date: String,
    token_env: String,
    json: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct StageResult {
    symbol: String,
    stage_name: String,
    ok: bool,
    detail: String,
    #[serde(default)]
    changed: bool,
}

#[derive(Debug, Serialize)]
struct CollectSummary {
    data_source: String,
    end_date: String,
    latest_trading_date: String,
    symbols: usize,
    stages: Vec<String>,
    ok: usize,
    failed: usize,
    changed: usize,
}

#[derive(Debug, Clone)]
struct AkshareCollectOptions {
    symbols_csv: Option<String>,
    all: bool,
    update: bool,
    rebuild_processed: bool,
    refresh_stock_list: bool,
    refresh_stock_list_only: bool,
    workers: usize,
    end_date: String,
    adjust: String,
    network_backend: String,
    proxy_auth_token: String,
    json: bool,
}

#[derive(Debug, Clone)]
struct UniverseBuildOptions {
    universes: Vec<String>,
    output_dir: PathBuf,
    allow_static_membership: bool,
    network_backend: String,
    proxy_auth_token: String,
    json: bool,
}

#[derive(Debug, Clone, Serialize)]
struct UniverseFileSummary {
    universe: String,
    index_code: String,
    path: String,
    rows: usize,
    static_membership: bool,
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
        "tushare" => run_tushare(&args[1..]),
        "akshare" => run_akshare(&args[1..]),
        "universes" | "universe" => run_universes(&args[1..]),
        other => Err(format!("unknown command: {other}\n\n{}", usage())),
    }
}

fn run_tushare(args: &[String]) -> Result<(), String> {
    let options = parse_tushare_options(args)?;
    let latest_trading_date = if options.refresh_symbols_only
        || options.rebuild_processed
        || options.rebuild_packed_source_only
    {
        options.end_date.clone()
    } else {
        call_python_json(
            "resolve_tushare_latest_trading_date",
            &[
                ("end_date", options.end_date.clone()),
                ("token_env", options.token_env.clone()),
            ],
        )?
        .get("latest_trading_date")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            "resolve_tushare_latest_trading_date returned no latest_trading_date".to_owned()
        })?
        .to_owned()
    };

    if options.refresh_symbols_only {
        let payload = call_python_json(
            "refresh_tushare_symbol_cache",
            &[("token_env", options.token_env.clone())],
        )?;
        print_simple_json_or_message(
            options.json,
            &payload,
            "Tushare symbol cache refresh completed",
        )?;
        return Ok(());
    }

    if options.refresh_benchmarks_only {
        let payload = call_python_json(
            "refresh_tushare_benchmarks",
            &[
                ("end_date", latest_trading_date.clone()),
                ("token_env", options.token_env.clone()),
            ],
        )?;
        print_simple_json_or_message(
            options.json,
            &payload,
            "Tushare benchmark refresh completed",
        )?;
        return Ok(());
    }

    if options.refresh_benchmarks {
        call_python_json(
            "refresh_tushare_benchmarks",
            &[
                ("end_date", latest_trading_date.clone()),
                ("token_env", options.token_env.clone()),
            ],
        )?;
    }

    if options.rebuild_packed_source_only {
        let metadata = rebuild_tushare_packed_source_from_local(options.workers)?;
        print_simple_json_or_message(
            options.json,
            &metadata,
            "Tushare packed source rebuild completed",
        )?;
        return Ok(());
    }

    let symbols = resolve_symbols(&options)?;
    if symbols.is_empty() {
        return Err("no Tushare symbols to process".to_owned());
    }

    let stages = if options.rebuild_processed {
        vec!["processed".to_owned()]
    } else {
        options.stages.clone()
    };
    println!(
        "[collect] tushare latest_trading_date={} symbols={} stages={} workers={}",
        latest_trading_date,
        symbols.len(),
        stages.join(","),
        options.workers
    );

    let mut all_results = Vec::new();
    for stage_name in &stages {
        let stage_results = run_stage_with_retry(
            &symbols,
            stage_name,
            &latest_trading_date,
            &options.token_env,
            options.workers,
        )?;
        all_results.extend(stage_results);
    }
    if should_rebuild_processed_after_raw(&stages)
        && !stages.iter().any(|stage| stage == "processed")
    {
        let failed_raw_symbols = all_results
            .iter()
            .filter(|item| !item.ok && is_processed_dependency_stage(&item.stage_name))
            .map(|item| item.symbol.clone())
            .collect::<BTreeSet<_>>();
        let processed_symbols = symbols
            .iter()
            .filter(|symbol| !failed_raw_symbols.contains(*symbol))
            .cloned()
            .collect::<Vec<_>>();
        let stage_results = run_stage_with_retry(
            &processed_symbols,
            "processed",
            &latest_trading_date,
            &options.token_env,
            options.workers,
        )?;
        all_results.extend(stage_results);
    }

    if options.rebuild_packed_source && should_rebuild_packed_source(&stages) {
        println!("[collect] rebuilding packed Tushare source buckets");
        call_python_packed_source(options.workers)?;
    }

    let failed = all_results.iter().filter(|item| !item.ok).count();
    let ok = all_results.len().saturating_sub(failed);
    let changed = all_results.iter().filter(|item| item.changed).count();
    let summary = CollectSummary {
        data_source: "tushare".to_owned(),
        end_date: options.end_date,
        latest_trading_date,
        symbols: symbols.len(),
        stages,
        ok,
        failed,
        changed,
    };
    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode summary: {err}"))?
        );
    } else {
        println!(
            "[collect] done ok={} failed={} changed={}",
            summary.ok, summary.failed, summary.changed
        );
    }
    if failed > 0 {
        return Err(format!(
            "Tushare collection finished with {failed} failed stage tasks"
        ));
    }
    Ok(())
}

fn run_akshare(args: &[String]) -> Result<(), String> {
    let options = parse_akshare_options(args)?;
    let latest_trading_date = options.end_date.clone();

    if options.refresh_stock_list_only {
        prepare_akshare_runtime(&options)?;
        let payload = call_python_json(
            "refresh_akshare_stock_list",
            &[
                ("network_backend", options.network_backend.clone()),
                ("proxy_auth_token", options.proxy_auth_token.clone()),
            ],
        )?;
        print_simple_json_or_message(
            options.json,
            &payload,
            "AkShare stock-list cache refresh completed",
        )?;
        return Ok(());
    }

    let symbols = resolve_akshare_symbols(&options)?;
    if symbols.is_empty() {
        return Err("no AkShare symbols to process".to_owned());
    }

    println!(
        "[collect] akshare end_date={} symbols={} workers={} rebuild_processed={}",
        latest_trading_date,
        symbols.len(),
        options.workers,
        options.rebuild_processed
    );

    let results = if options.rebuild_processed {
        run_akshare_processed_rebuild(&symbols, options.workers)?
    } else {
        prepare_akshare_runtime(&options)?;
        let pending_symbols = if options.symbols_csv.is_none() {
            precheck_akshare_pending_symbols(&symbols, &latest_trading_date, options.workers)?
        } else {
            symbols.clone()
        };
        if pending_symbols.is_empty() {
            println!("[collect] akshare all symbols already complete");
            Vec::new()
        } else {
            run_akshare_update_symbols(&pending_symbols, &options)?
        }
    };

    let failed = results.iter().filter(|item| !item.ok).count();
    let ok = results.len().saturating_sub(failed);
    let changed = results.iter().filter(|item| item.changed).count();
    let summary = CollectSummary {
        data_source: "akshare".to_owned(),
        end_date: options.end_date,
        latest_trading_date,
        symbols: symbols.len(),
        stages: if options.rebuild_processed {
            vec!["processed".to_owned()]
        } else {
            vec![
                "daily".to_owned(),
                "valuation".to_owned(),
                "processed".to_owned(),
            ]
        },
        ok,
        failed,
        changed,
    };
    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| format!("failed to encode summary: {err}"))?
        );
    } else {
        println!(
            "[collect] done ok={} failed={} changed={}",
            summary.ok, summary.failed, summary.changed
        );
    }
    if failed > 0 {
        return Err(format!(
            "AkShare collection finished with {failed} failed symbol tasks"
        ));
    }
    Ok(())
}

fn run_universes(args: &[String]) -> Result<(), String> {
    let options = parse_universe_options(args)?;
    clear_proxy_env();
    fs::create_dir_all(&options.output_dir).map_err(|err| {
        format!(
            "failed to create universe output dir {}: {err}",
            options.output_dir.display()
        )
    })?;

    let mut summaries = Vec::new();
    for universe in &options.universes {
        let index_code = universe_index_code(universe)?;
        let records = fetch_universe_constituents(universe, index_code, &options)?;
        let rows = normalize_universe_records(&records, options.allow_static_membership)?;
        if rows.is_empty() {
            return Err(format!("universe {universe} produced no valid symbol rows"));
        }
        let path = options.output_dir.join(format!("{universe}.txt"));
        write_universe_rows(&path, &rows)?;
        println!(
            "[collect] universe {} index={} rows={} saved={}",
            universe,
            index_code,
            rows.len(),
            path.display()
        );
        summaries.push(UniverseFileSummary {
            universe: universe.clone(),
            index_code: index_code.to_owned(),
            path: path.display().to_string(),
            rows: rows.len(),
            static_membership: records_used_static_membership(&rows),
        });
    }

    if options.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "ok": true,
                "output_dir": options.output_dir.display().to_string(),
                "universes": summaries,
            }))
            .map_err(|err| format!("failed to encode universe summary: {err}"))?
        );
    }
    Ok(())
}

fn parse_tushare_options(args: &[String]) -> Result<TushareCollectOptions, String> {
    let mut options = TushareCollectOptions {
        symbols_csv: None,
        update: false,
        rebuild_processed: false,
        rebuild_packed_source_only: false,
        stages: DEFAULT_STAGES
            .iter()
            .map(|item| (*item).to_owned())
            .collect(),
        refresh_symbols: false,
        refresh_symbols_only: false,
        refresh_benchmarks: false,
        refresh_benchmarks_only: false,
        rebuild_packed_source: true,
        workers: DEFAULT_WORKERS,
        end_date: Local::now().date_naive().to_string(),
        token_env: DEFAULT_TOKEN_ENV.to_owned(),
        json: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--symbols" => {
                index += 1;
                options.symbols_csv = Some(require_value(args, index, "--symbols")?.to_owned());
            }
            "--update" => options.update = true,
            "--rebuild-processed" => options.rebuild_processed = true,
            "--rebuild-packed-source-only" => options.rebuild_packed_source_only = true,
            "--stages" => {
                index += 1;
                options.stages = parse_stages(require_value(args, index, "--stages")?)?;
            }
            "--refresh-symbols" => options.refresh_symbols = true,
            "--refresh-symbols-only" => options.refresh_symbols_only = true,
            "--refresh-benchmarks" => options.refresh_benchmarks = true,
            "--refresh-benchmarks-only" => options.refresh_benchmarks_only = true,
            "--no-rebuild-packed-source" => options.rebuild_packed_source = false,
            "--workers" => {
                index += 1;
                options.workers = require_value(args, index, "--workers")?
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --workers: {err}"))?
                    .max(1);
            }
            "--end-date" => {
                index += 1;
                let value = require_value(args, index, "--end-date")?;
                NaiveDate::parse_from_str(value, "%Y-%m-%d")
                    .map_err(|err| format!("invalid --end-date: {err}"))?;
                options.end_date = value.to_owned();
            }
            "--token-env" => {
                index += 1;
                options.token_env = require_value(args, index, "--token-env")?.to_owned();
            }
            "--json" => options.json = true,
            other => {
                return Err(format!(
                    "unknown option for tushare: {other}\n\n{}",
                    usage()
                ))
            }
        }
        index += 1;
    }
    if !options.rebuild_processed
        && !options.update
        && !options.rebuild_packed_source_only
        && options
            .symbols_csv
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        && !options.refresh_symbols_only
        && !options.refresh_benchmarks_only
    {
        return Err("provide one of --update, --rebuild-processed, --symbols, --refresh-symbols-only, or --refresh-benchmarks-only".to_owned());
    }
    Ok(options)
}

fn parse_akshare_options(args: &[String]) -> Result<AkshareCollectOptions, String> {
    let mut options = AkshareCollectOptions {
        symbols_csv: None,
        all: false,
        update: false,
        rebuild_processed: false,
        refresh_stock_list: false,
        refresh_stock_list_only: false,
        workers: 4,
        end_date: Local::now().date_naive().to_string(),
        adjust: "hfq".to_owned(),
        network_backend: "cookie".to_owned(),
        proxy_auth_token: String::new(),
        json: false,
    };
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--symbols" => {
                index += 1;
                options.symbols_csv = Some(require_value(args, index, "--symbols")?.to_owned());
            }
            "--all" => options.all = true,
            "--update" => options.update = true,
            "--rebuild-processed" => options.rebuild_processed = true,
            "--refresh-stock-list" => options.refresh_stock_list = true,
            "--refresh-stock-list-only" => options.refresh_stock_list_only = true,
            "--workers" => {
                index += 1;
                options.workers = require_value(args, index, "--workers")?
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --workers: {err}"))?
                    .max(1);
            }
            "--end-date" => {
                index += 1;
                let value = require_value(args, index, "--end-date")?;
                NaiveDate::parse_from_str(value, "%Y-%m-%d")
                    .map_err(|err| format!("invalid --end-date: {err}"))?;
                options.end_date = value.to_owned();
            }
            "--adjust" => {
                index += 1;
                let value = require_value(args, index, "--adjust")?;
                options.adjust = if value == "raw" {
                    String::new()
                } else {
                    value.to_owned()
                };
                if !["hfq", "qfq", ""].contains(&options.adjust.as_str()) {
                    return Err("invalid --adjust; expected hfq, qfq, or raw".to_owned());
                }
            }
            "--network-backend" => {
                index += 1;
                options.network_backend =
                    require_value(args, index, "--network-backend")?.to_owned();
                if !["cookie", "proxy_patch"].contains(&options.network_backend.as_str()) {
                    return Err(
                        "invalid --network-backend; expected cookie or proxy_patch".to_owned()
                    );
                }
            }
            "--proxy-auth-token" => {
                index += 1;
                options.proxy_auth_token =
                    require_value(args, index, "--proxy-auth-token")?.to_owned();
            }
            "--json" => options.json = true,
            other => {
                return Err(format!(
                    "unknown option for akshare: {other}\n\n{}",
                    usage()
                ))
            }
        }
        index += 1;
    }
    if !options.rebuild_processed
        && !options.update
        && !options.all
        && options
            .symbols_csv
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        && !options.refresh_stock_list_only
    {
        return Err("provide one of --all, --update, --rebuild-processed, --symbols, or --refresh-stock-list-only".to_owned());
    }
    Ok(options)
}

fn parse_universe_options(args: &[String]) -> Result<UniverseBuildOptions, String> {
    let mut universes = vec![
        "csi300".to_owned(),
        "csi500".to_owned(),
        "zz1000".to_owned(),
    ];
    let mut output_dir = PathBuf::from(DEFAULT_UNIVERSE_DIR);
    let mut allow_static_membership = false;
    let mut network_backend = "cookie".to_owned();
    let mut proxy_auth_token = String::new();
    let mut json = false;

    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "-h" | "--help" => return Err(usage().to_owned()),
            "--universes" => {
                index += 1;
                universes = parse_universe_names(require_value(args, index, "--universes")?)?;
            }
            "--output-dir" => {
                index += 1;
                output_dir = PathBuf::from(require_value(args, index, "--output-dir")?);
            }
            "--allow-static-membership" => allow_static_membership = true,
            "--network-backend" => {
                index += 1;
                network_backend = require_value(args, index, "--network-backend")?.to_owned();
                if !["cookie", "proxy_patch"].contains(&network_backend.as_str()) {
                    return Err(
                        "invalid --network-backend; expected cookie or proxy_patch".to_owned()
                    );
                }
            }
            "--proxy-auth-token" => {
                index += 1;
                proxy_auth_token = require_value(args, index, "--proxy-auth-token")?.to_owned();
            }
            "--json" => json = true,
            other => {
                return Err(format!(
                    "unknown option for universes: {other}\n\n{}",
                    usage()
                ))
            }
        }
        index += 1;
    }

    for universe in &universes {
        universe_index_code(universe)?;
    }

    Ok(UniverseBuildOptions {
        universes,
        output_dir,
        allow_static_membership,
        network_backend,
        proxy_auth_token,
        json,
    })
}

fn parse_universe_names(raw: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for name in raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let value = name.to_ascii_lowercase();
        universe_index_code(&value)?;
        if !out.iter().any(|existing| existing == &value) {
            out.push(value);
        }
    }
    if out.is_empty() {
        return Err("empty --universes".to_owned());
    }
    Ok(out)
}

fn parse_stages(raw: &str) -> Result<Vec<String>, String> {
    if raw.trim().eq_ignore_ascii_case("all") {
        return Ok(ALL_STAGES.iter().map(|item| (*item).to_owned()).collect());
    }
    let mut out = Vec::new();
    for item in raw
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        if !ALL_STAGES.contains(&item) {
            return Err(format!(
                "unsupported stage: {item}. Available: {}",
                ALL_STAGES.join(",")
            ));
        }
        if !out.iter().any(|existing| existing == item) {
            out.push(item.to_owned());
        }
    }
    if out.is_empty() {
        return Err("empty --stages".to_owned());
    }
    Ok(out)
}

fn require_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, String> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn resolve_symbols(options: &TushareCollectOptions) -> Result<Vec<String>, String> {
    if options.rebuild_processed {
        if let Some(symbols_csv) = &options.symbols_csv {
            return parse_symbols_json(call_python_json(
                "resolve_tushare_symbols",
                &[
                    ("symbols_csv", symbols_csv.clone()),
                    ("update", "false".to_owned()),
                    ("refresh_symbols", "false".to_owned()),
                    ("token_env", options.token_env.clone()),
                ],
            )?);
        }
        return parse_symbols_json(call_python_json("list_tushare_local_symbols", &[])?);
    }
    parse_symbols_json(call_python_json(
        "resolve_tushare_symbols",
        &[
            (
                "symbols_csv",
                options.symbols_csv.clone().unwrap_or_default(),
            ),
            ("update", options.update.to_string()),
            ("refresh_symbols", options.refresh_symbols.to_string()),
            ("token_env", options.token_env.clone()),
        ],
    )?)
}

fn parse_symbols_json(payload: JsonValue) -> Result<Vec<String>, String> {
    let symbols = payload
        .get("symbols")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "symbol resolver returned no symbols array".to_owned())?;
    Ok(symbols
        .iter()
        .filter_map(JsonValue::as_str)
        .map(str::to_owned)
        .collect())
}

fn resolve_akshare_symbols(options: &AkshareCollectOptions) -> Result<Vec<String>, String> {
    if options.rebuild_processed {
        if let Some(symbols_csv) = &options.symbols_csv {
            return parse_symbols_json(call_python_json(
                "resolve_akshare_symbols",
                &[
                    ("symbols_csv", symbols_csv.clone()),
                    ("all", "false".to_owned()),
                    ("update", "false".to_owned()),
                    ("refresh_stock_list", "false".to_owned()),
                    ("network_backend", options.network_backend.clone()),
                    ("proxy_auth_token", options.proxy_auth_token.clone()),
                ],
            )?);
        }
        return parse_symbols_json(call_python_json("list_akshare_local_symbols", &[])?);
    }
    parse_symbols_json(call_python_json(
        "resolve_akshare_symbols",
        &[
            (
                "symbols_csv",
                options.symbols_csv.clone().unwrap_or_default(),
            ),
            ("all", options.all.to_string()),
            ("update", options.update.to_string()),
            ("refresh_stock_list", options.refresh_stock_list.to_string()),
            ("network_backend", options.network_backend.clone()),
            ("proxy_auth_token", options.proxy_auth_token.clone()),
        ],
    )?)
}

fn prepare_akshare_runtime(options: &AkshareCollectOptions) -> Result<(), String> {
    let payload = call_python_json(
        "prepare_akshare_runtime",
        &[
            ("network_backend", options.network_backend.clone()),
            ("proxy_auth_token", options.proxy_auth_token.clone()),
        ],
    )?;
    if payload
        .get("ok")
        .and_then(JsonValue::as_bool)
        .unwrap_or(false)
    {
        Ok(())
    } else {
        Err(format!("prepare_akshare_runtime failed: {payload}"))
    }
}

fn clear_proxy_env() {
    for key in [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ] {
        env::remove_var(key);
    }
}

fn run_akshare_processed_rebuild(
    symbols: &[String],
    workers: usize,
) -> Result<Vec<StageResult>, String> {
    run_symbol_tasks(symbols, workers, |symbol| {
        rebuild_akshare_processed_symbol(symbol)
    })
}

fn run_akshare_update_symbols(
    symbols: &[String],
    options: &AkshareCollectOptions,
) -> Result<Vec<StageResult>, String> {
    let end_date = options.end_date.clone();
    let adjust = options.adjust.clone();
    let network_backend = options.network_backend.clone();
    let proxy_auth_token = options.proxy_auth_token.clone();
    run_symbol_tasks(symbols, options.workers, move |symbol| {
        let payload = call_python_json(
            "run_akshare_raw_update",
            &[
                ("symbol", symbol.to_owned()),
                ("end_date", end_date.clone()),
                ("adjust", adjust.clone()),
                ("network_backend", network_backend.clone()),
                ("proxy_auth_token", proxy_auth_token.clone()),
            ],
        )?;
        let ok = payload
            .get("ok")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        if !ok {
            return Ok(StageResult {
                symbol: symbol.to_owned(),
                stage_name: "raw".to_owned(),
                ok: false,
                detail: payload
                    .get("detail")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("raw update failed")
                    .to_owned(),
                changed: false,
            });
        }
        let processed = rebuild_akshare_processed_symbol(symbol)?;
        let detail = format!(
            "{}; {}",
            payload
                .get("detail")
                .and_then(JsonValue::as_str)
                .unwrap_or("raw updated"),
            processed.detail
        );
        Ok(StageResult {
            symbol: symbol.to_owned(),
            stage_name: "akshare".to_owned(),
            ok: true,
            detail,
            changed: payload
                .get("changed")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true)
                || processed.changed,
        })
    })
}

fn run_symbol_tasks<F>(
    symbols: &[String],
    workers: usize,
    worker: F,
) -> Result<Vec<StageResult>, String>
where
    F: Fn(&str) -> Result<StageResult, String> + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .map_err(|err| format!("failed to build worker pool: {err}"))?;
    let results = pool.install(|| {
        symbols
            .par_iter()
            .map(|symbol| worker(symbol))
            .collect::<Vec<_>>()
    });
    let mut out = Vec::with_capacity(results.len());
    for result in results {
        let result = result?;
        if result.ok {
            println!(
                "[collect] stage={} symbol={} ok detail={}",
                result.stage_name, result.symbol, result.detail
            );
        } else {
            println!(
                "[collect] stage={} symbol={} failed detail={}",
                result.stage_name, result.symbol, result.detail
            );
        }
        out.push(result);
    }
    Ok(out)
}

fn precheck_akshare_pending_symbols(
    symbols: &[String],
    target_end_date: &str,
    workers: usize,
) -> Result<Vec<String>, String> {
    let target_ns = parse_date_ns(target_end_date)
        .ok_or_else(|| format!("invalid target end date: {target_end_date}"))?;
    let (daily_dir, valuation_dir, processed_dir) = akshare_raw_dirs();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .map_err(|err| format!("failed to build worker pool: {err}"))?;
    let states = pool.install(|| {
        symbols
            .par_iter()
            .map(|symbol| {
                let daily_latest = infer_parquet_latest_date(
                    &daily_dir.join(format!("{symbol}.parquet")),
                    "date",
                )?;
                let valuation_latest = infer_parquet_latest_date(
                    &valuation_dir.join(format!("{symbol}.parquet")),
                    "date",
                )?;
                let processed_latest = infer_parquet_latest_date(
                    &processed_dir.join(format!("{symbol}.parquet")),
                    "date",
                )?;
                Ok::<_, String>((
                    symbol.clone(),
                    daily_latest,
                    valuation_latest,
                    processed_latest,
                ))
            })
            .collect::<Vec<_>>()
    });
    let mut pending = Vec::new();
    let mut completed = 0usize;
    for state in states {
        let (symbol, daily_latest, valuation_latest, processed_latest) = state?;
        let complete = matches!(daily_latest, Some(value) if value >= target_ns)
            && matches!(valuation_latest, Some(value) if value >= target_ns)
            && matches!((processed_latest, daily_latest), (Some(processed), Some(daily)) if processed >= daily);
        if complete {
            completed += 1;
        } else {
            pending.push(symbol);
        }
    }
    println!(
        "[collect] akshare precheck completed={} pending={}",
        completed,
        pending.len()
    );
    Ok(pending)
}

fn run_stage_with_retry(
    symbols: &[String],
    stage_name: &str,
    end_date: &str,
    token_env: &str,
    workers: usize,
) -> Result<Vec<StageResult>, String> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .map_err(|err| format!("failed to build worker pool: {err}"))?;
    let mut queue = VecDeque::from(symbols.to_vec());
    let mut results = Vec::new();
    while !queue.is_empty() {
        let batch_size = workers.min(queue.len()).max(1);
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            if let Some(symbol) = queue.pop_front() {
                batch.push(symbol);
            }
        }
        let batch_results = pool.install(|| {
            batch
                .par_iter()
                .map(|symbol| call_python_stage(symbol, stage_name, end_date, token_env))
                .collect::<Vec<_>>()
        });
        let mut retry = Vec::new();
        for result in batch_results {
            let result = result?;
            if !result.ok && is_tushare_rate_limit(&result.detail) {
                retry.push(result.symbol.clone());
            } else {
                if result.ok {
                    println!(
                        "[collect] stage={} symbol={} ok detail={}",
                        stage_name, result.symbol, result.detail
                    );
                } else {
                    println!(
                        "[collect] stage={} symbol={} failed detail={}",
                        stage_name, result.symbol, result.detail
                    );
                }
                results.push(result);
            }
        }
        if !retry.is_empty() {
            for symbol in retry.into_iter().rev() {
                queue.push_front(symbol);
            }
            println!(
                "[collect] stage={} hit Tushare rate limit; sleeping {}s pending={}",
                stage_name,
                RATE_LIMIT_COOLDOWN_SECONDS,
                queue.len()
            );
            thread::sleep(Duration::from_secs(RATE_LIMIT_COOLDOWN_SECONDS));
        }
        println!(
            "[collect] stage={} progress done={} pending={}",
            stage_name,
            results.len(),
            queue.len()
        );
    }
    Ok(results)
}

fn call_python_stage(
    symbol: &str,
    stage_name: &str,
    end_date: &str,
    token_env: &str,
) -> Result<StageResult, String> {
    if stage_name == "processed" {
        return rebuild_tushare_processed_symbol(symbol);
    }
    let payload = call_python_json(
        "run_tushare_stage",
        &[
            ("symbol", symbol.to_owned()),
            ("stage_name", stage_name.to_owned()),
            ("end_date", end_date.to_owned()),
            ("token_env", token_env.to_owned()),
        ],
    )?;
    serde_json::from_value(payload).map_err(|err| format!("invalid stage result JSON: {err}"))
}

fn universe_index_code(universe: &str) -> Result<&'static str, String> {
    match universe {
        "csi300" => Ok("000300"),
        "csi500" => Ok("000905"),
        "zz1000" => Ok("000852"),
        other => Err(format!(
            "unknown universe '{other}'. Available: csi300,csi500,zz1000"
        )),
    }
}

fn fetch_universe_constituents(
    universe: &str,
    index_code: &str,
    options: &UniverseBuildOptions,
) -> Result<Vec<BTreeMap<String, JsonValue>>, String> {
    let payload = call_python_json(
        "fetch_akshare_index_constituents",
        &[
            ("index_code", index_code.to_owned()),
            ("network_backend", options.network_backend.clone()),
            ("proxy_auth_token", options.proxy_auth_token.clone()),
        ],
    )?;
    let records = payload
        .get("records")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            format!("AkShare constituent resolver returned no records for {universe}")
        })?;
    if records.is_empty() {
        return Err(format!("No constituents returned for universe {universe}"));
    }
    records
        .iter()
        .map(|record| {
            record
                .as_object()
                .ok_or_else(|| format!("invalid constituent record for {universe}: {record}"))
                .map(|object| {
                    object
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<BTreeMap<_, _>>()
                })
        })
        .collect()
}

fn normalize_universe_records(
    records: &[BTreeMap<String, JsonValue>],
    allow_static_membership: bool,
) -> Result<Vec<(String, String, String)>, String> {
    let symbol_col = find_first_present_column(
        records,
        &[
            "成分券代码",
            "指数代码",
            "stock_code",
            "品种代码",
            "证券代码",
        ],
    )
    .ok_or_else(|| {
        format!(
            "Could not find symbol column in constituent table: {:?}",
            sorted_record_columns(records)
        )
    })?;
    let start_col = find_first_present_column(records, &["纳入日期", "开始日期", "start_date"]);
    let end_col = find_first_present_column(records, &["剔除日期", "结束日期", "end_date"]);
    if (start_col.is_none() || end_col.is_none()) && !allow_static_membership {
        return Err(
            "Index constituent table does not expose point-in-time membership intervals. \
Refusing to build a universe file that would silently use static/current membership. \
Pass --allow-static-membership only for explicitly labelled research controls."
                .to_owned(),
        );
    }

    let mut unique = BTreeSet::new();
    for record in records {
        let symbol = normalize_symbol(value_to_string(record.get(&symbol_col)).as_deref());
        if symbol.is_empty() {
            continue;
        }
        let start_date = match &start_col {
            Some(column) => normalize_date_or_default(record.get(column), "2005-01-01")?,
            None => "2005-01-01".to_owned(),
        };
        let end_date = match &end_col {
            Some(column) => normalize_date_or_default(record.get(column), "2099-12-31")?,
            None => "2099-12-31".to_owned(),
        };
        unique.insert((symbol, start_date, end_date));
    }
    Ok(unique.into_iter().collect())
}

fn find_first_present_column(
    records: &[BTreeMap<String, JsonValue>],
    candidates: &[&str],
) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| {
            records
                .iter()
                .any(|record| record.contains_key(**candidate))
        })
        .map(|candidate| (*candidate).to_owned())
}

fn sorted_record_columns(records: &[BTreeMap<String, JsonValue>]) -> Vec<String> {
    let mut columns = BTreeSet::new();
    for record in records {
        for key in record.keys() {
            columns.insert(key.clone());
        }
    }
    columns.into_iter().collect()
}

fn normalize_symbol(raw: Option<&str>) -> String {
    let Some(value) = raw else {
        return String::new();
    };
    let digits: String = value.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.len() >= 6 {
        digits[digits.len() - 6..].to_owned()
    } else {
        digits
    }
}

fn normalize_date_or_default(value: Option<&JsonValue>, default: &str) -> Result<String, String> {
    let Some(raw) = value_to_string(value) else {
        return Ok(default.to_owned());
    };
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "none" | "null" | "nat" | "nan"
        )
    {
        return Ok(default.to_owned());
    }
    parse_date_string(trimmed)
        .map(|date| date.to_string())
        .ok_or_else(|| format!("invalid universe membership date: {trimmed}"))
}

fn parse_date_string(raw: &str) -> Option<NaiveDate> {
    let head = raw.split_whitespace().next().unwrap_or(raw);
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"] {
        if let Ok(date) = NaiveDate::parse_from_str(head, fmt) {
            return Some(date);
        }
    }
    let digits: String = raw.chars().filter(|ch| ch.is_ascii_digit()).collect();
    if digits.len() >= 8 {
        return NaiveDate::parse_from_str(&digits[..8], "%Y%m%d").ok();
    }
    None
}

fn value_to_string(value: Option<&JsonValue>) -> Option<String> {
    match value? {
        JsonValue::Null => None,
        JsonValue::String(value) => Some(value.clone()),
        JsonValue::Number(value) => Some(value.to_string()),
        JsonValue::Bool(value) => Some(value.to_string()),
        other => Some(other.to_string()),
    }
}

fn write_universe_rows(path: &Path, rows: &[(String, String, String)]) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|err| format!("failed to create universe file {}: {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for (symbol, start_date, end_date) in rows {
        writeln!(writer, "{symbol}\t{start_date}\t{end_date}")
            .map_err(|err| format!("failed to write universe file {}: {err}", path.display()))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush universe file {}: {err}", path.display()))
}

fn records_used_static_membership(rows: &[(String, String, String)]) -> bool {
    rows.iter()
        .all(|(_, start_date, end_date)| start_date == "2005-01-01" && end_date == "2099-12-31")
}

fn should_rebuild_processed_after_raw(stages: &[String]) -> bool {
    stages
        .iter()
        .any(|stage| is_processed_dependency_stage(stage))
}

fn is_processed_dependency_stage(stage: &str) -> bool {
    matches!(stage, "daily" | "daily_basic" | "adj_factor" | "stk_limit")
}

fn should_rebuild_packed_source(stages: &[String]) -> bool {
    stages.iter().any(|stage| {
        matches!(
            stage.as_str(),
            "daily"
                | "daily_basic"
                | "adj_factor"
                | "stk_limit"
                | "fina_indicator"
                | "dividend"
                | "forecast"
                | "express"
                | "processed"
        )
    })
}

#[derive(Debug, Clone)]
struct DailyRawRow {
    date_ns: i64,
    ts_code: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    pre_close: f64,
    volume: f64,
    amount: f64,
}

#[derive(Debug, Clone, Default)]
struct DailyBasicRow {
    turnover_rate: f64,
    turnover_rate_f: f64,
    volume_ratio: f64,
    pe: f64,
    pe_ttm: f64,
    pb: f64,
    ps: f64,
    ps_ttm: f64,
    dv_ratio: f64,
    dv_ttm: f64,
    total_share: f64,
    float_share: f64,
    free_share: f64,
    total_mv: f64,
    circ_mv: f64,
}

#[derive(Debug, Clone, Default)]
struct AdjFactorRow {
    adj_factor: f64,
}

#[derive(Debug, Clone, Default)]
struct StkLimitRow {
    limit_pre_close: f64,
    up_limit: f64,
    down_limit: f64,
}

#[derive(Debug, Clone)]
struct ProcessedRow {
    date_ns: i64,
    symbol: String,
    ts_code: String,
    values: BTreeMap<&'static str, f64>,
}

#[derive(Debug, Clone)]
struct AkshareDailyRow {
    values: BTreeMap<&'static str, f64>,
}

#[derive(Debug, Clone)]
struct AkshareValuationRow {
    values: BTreeMap<&'static str, f64>,
}

#[derive(Debug, Clone)]
struct AkshareProcessedRow {
    date_ns: i64,
    symbol: String,
    values: BTreeMap<&'static str, f64>,
}

#[derive(Debug, Clone)]
struct SourceRow {
    date_ns: i64,
    symbol: String,
    ts_code: String,
    values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
struct SidecarEvent {
    ann_day: i64,
    available_ns: i64,
    values: BTreeMap<&'static str, f64>,
}

#[derive(Debug, Clone, Default)]
struct IndustryAccum {
    count: usize,
    ret_values: Vec<f64>,
    sums: BTreeMap<&'static str, (f64, usize)>,
}

#[derive(Debug, Clone)]
struct IndustryDailyRow {
    date_ns: i64,
    industry: String,
    values: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize)]
struct PackedSourceManifestRow {
    symbol: String,
    bucket_id: usize,
    source_path: String,
    source_size: i64,
    source_mtime_ns: i64,
    dependency_signature: String,
    row_count: usize,
    min_date: String,
    max_date: String,
}

fn rebuild_tushare_processed_symbol(symbol: &str) -> Result<StageResult, String> {
    let symbol = normalize_local_symbol(symbol);
    let daily_path = PathBuf::from(TS_ROOT)
        .join("raw/daily")
        .join(format!("{symbol}.parquet"));
    let daily_basic_path = PathBuf::from(TS_ROOT)
        .join("raw/daily_basic")
        .join(format!("{symbol}.parquet"));
    let adj_factor_path = PathBuf::from(TS_ROOT)
        .join("raw/adj_factor")
        .join(format!("{symbol}.parquet"));
    let stk_limit_path = PathBuf::from(TS_ROOT)
        .join("raw/stk_limit")
        .join(format!("{symbol}.parquet"));
    let processed_path = PathBuf::from(TS_ROOT)
        .join("processed/combined")
        .join(format!("{symbol}.parquet"));

    let daily = read_daily_raw_rows(&daily_path, &symbol)?;
    if daily.is_empty() {
        return Err(format!("missing Tushare daily parquet rows for {symbol}"));
    }
    let daily_basic = read_daily_basic_rows(&daily_basic_path)?;
    let adj_factor = read_adj_factor_rows(&adj_factor_path)?;
    let stk_limit = read_stk_limit_rows(&stk_limit_path)?;
    let processed = build_processed_rows(&symbol, daily, daily_basic, adj_factor, stk_limit);
    write_processed_rows(&processed_path, &processed)?;
    let latest = processed
        .last()
        .map(|row| format_date(row.date_ns))
        .unwrap_or_else(|| "n/a".to_owned());
    Ok(StageResult {
        symbol,
        stage_name: "processed".to_owned(),
        ok: true,
        detail: format!("processed rows={} last_date={latest}", processed.len()),
        changed: true,
    })
}

fn akshare_raw_dirs() -> (PathBuf, PathBuf, PathBuf) {
    (
        PathBuf::from("data/raw/daily"),
        PathBuf::from("data/raw/valuation"),
        PathBuf::from("data/processed/combined"),
    )
}

fn rebuild_akshare_processed_symbol(symbol: &str) -> Result<StageResult, String> {
    let (daily_dir, valuation_dir, processed_dir) = akshare_raw_dirs();
    let daily_path = daily_dir.join(format!("{symbol}.parquet"));
    let valuation_path = valuation_dir.join(format!("{symbol}.parquet"));
    let processed_path = processed_dir.join(format!("{symbol}.parquet"));

    let daily = read_akshare_daily_rows(&daily_path)?;
    if daily.is_empty() {
        return Err(format!("missing AkShare daily parquet rows for {symbol}"));
    }
    let valuation = read_akshare_valuation_rows(&valuation_path)?;
    let processed = build_akshare_processed_rows(symbol, daily, valuation);
    write_akshare_processed_rows(&processed_path, &processed)?;
    let latest = processed
        .last()
        .map(|row| format_date(row.date_ns))
        .unwrap_or_else(|| "n/a".to_owned());
    Ok(StageResult {
        symbol: symbol.to_owned(),
        stage_name: "processed".to_owned(),
        ok: true,
        detail: format!("processed rows={} last_date={latest}", processed.len()),
        changed: true,
    })
}

fn read_akshare_daily_rows(path: &Path) -> Result<BTreeMap<i64, AkshareDailyRow>, String> {
    if !path.exists() {
        return Err(format!("missing AkShare daily parquet: {}", path.display()));
    }
    let mut rows = BTreeMap::new();
    for batch in read_parquet_batches(path)? {
        let date_array = required_array(&batch, "date", path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(date_array.as_ref(), row_index) else {
                continue;
            };
            let mut values = BTreeMap::new();
            for name in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "amplitude",
                "pct_chg",
                "change",
                "turnover",
            ] {
                values.insert(name, numeric_column_value(&batch, name, row_index));
            }
            rows.insert(date_ns, AkshareDailyRow { values });
        }
    }
    Ok(rows)
}

fn read_akshare_valuation_rows(path: &Path) -> Result<BTreeMap<i64, AkshareValuationRow>, String> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let mut rows = BTreeMap::new();
    for batch in read_parquet_batches(path)? {
        let date_array = required_array(&batch, "date", path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(date_array.as_ref(), row_index) else {
                continue;
            };
            let mut values = BTreeMap::new();
            for name in [
                "v_close",
                "val_pct_chg",
                "total_mv",
                "circ_mv",
                "total_share",
                "circ_share",
                "pe_ttm",
                "pe_static",
                "pb",
                "peg",
                "pcf",
                "ps",
            ] {
                values.insert(name, numeric_column_value(&batch, name, row_index));
            }
            rows.insert(date_ns, AkshareValuationRow { values });
        }
    }
    Ok(rows)
}

fn build_akshare_processed_rows(
    symbol: &str,
    daily: BTreeMap<i64, AkshareDailyRow>,
    valuation: BTreeMap<i64, AkshareValuationRow>,
) -> Vec<AkshareProcessedRow> {
    let mut rows = Vec::with_capacity(daily.len());
    for (date_ns, daily_row) in daily {
        let val = valuation.get(&date_ns).cloned();
        let mut values = BTreeMap::new();
        let mut close = *daily_row.values.get("close").unwrap_or(&f64::NAN);
        if !close.is_finite() {
            close = val
                .as_ref()
                .and_then(|row| row.values.get("v_close").copied())
                .unwrap_or(f64::NAN);
        }
        let open = *daily_row.values.get("open").unwrap_or(&f64::NAN);
        let high = *daily_row.values.get("high").unwrap_or(&f64::NAN);
        let low = *daily_row.values.get("low").unwrap_or(&f64::NAN);
        let volume = *daily_row.values.get("volume").unwrap_or(&f64::NAN);
        let amount = *daily_row.values.get("amount").unwrap_or(&f64::NAN);
        let amplitude = *daily_row.values.get("amplitude").unwrap_or(&f64::NAN);
        let pct_chg = *daily_row.values.get("pct_chg").unwrap_or(&f64::NAN);
        let change = *daily_row.values.get("change").unwrap_or(&f64::NAN);
        let turnover = *daily_row.values.get("turnover").unwrap_or(&f64::NAN);
        values.insert("open", open);
        values.insert("high", high);
        values.insert("low", low);
        values.insert("close", close);
        values.insert("volume", volume);
        values.insert("amount", amount);
        values.insert("amplitude", amplitude);
        values.insert("pct_chg", pct_chg);
        values.insert("change", change);
        values.insert("turnover", turnover);
        for name in [
            "val_pct_chg",
            "total_mv",
            "circ_mv",
            "total_share",
            "circ_share",
            "pe_ttm",
            "pe_static",
            "pb",
            "peg",
            "pcf",
            "ps",
        ] {
            let value = val
                .as_ref()
                .and_then(|row| row.values.get(name).copied())
                .unwrap_or(f64::NAN);
            values.insert(name, value);
        }
        rows.push(AkshareProcessedRow {
            date_ns,
            symbol: symbol.to_owned(),
            values,
        });
    }
    rows
}

fn write_akshare_processed_rows(path: &Path, rows: &[AkshareProcessedRow]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let schema = Arc::new(Schema::new(
        akshare_processed_columns()
            .iter()
            .map(|column| match *column {
                "date" => Field::new(
                    "date",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                "symbol" => Field::new("symbol", DataType::Utf8, true),
                _ => Field::new(*column, DataType::Float64, true),
            })
            .collect::<Vec<_>>(),
    ));
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for column in akshare_processed_columns() {
        match *column {
            "date" => arrays.push(Arc::new(TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|row| Some(row.date_ns / 1_000))
                    .collect::<Vec<_>>(),
            ))),
            "symbol" => arrays.push(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| Some(row.symbol.as_str()))
                    .collect::<Vec<_>>(),
            ))),
            name => arrays.push(Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| {
                        let value = row.values.get(name).copied().unwrap_or(f64::NAN);
                        value.is_finite().then_some(value)
                    })
                    .collect::<Vec<_>>(),
            ))),
        }
    }
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|err| format!("failed to build processed record batch: {err}"))?;
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|err| format!("failed to create parquet writer {}: {err}", path.display()))?;
    writer
        .write(&batch)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    writer
        .close()
        .map_err(|err| format!("failed to close {}: {err}", path.display()))?;
    Ok(())
}

fn read_daily_raw_rows(path: &Path, symbol: &str) -> Result<BTreeMap<i64, DailyRawRow>, String> {
    if !path.exists() {
        return Err(format!("missing daily parquet: {}", path.display()));
    }
    let mut rows = BTreeMap::new();
    for batch in read_parquet_batches(path)? {
        let trade_date = required_array(&batch, "trade_date", path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(trade_date.as_ref(), row_index) else {
                continue;
            };
            let ts_code = string_column_value(&batch, "ts_code", row_index)
                .unwrap_or_else(|| local_symbol_to_ts(symbol));
            rows.insert(
                date_ns,
                DailyRawRow {
                    date_ns,
                    ts_code,
                    open: numeric_column_value(&batch, "open", row_index),
                    high: numeric_column_value(&batch, "high", row_index),
                    low: numeric_column_value(&batch, "low", row_index),
                    close: numeric_column_value(&batch, "close", row_index),
                    pre_close: numeric_column_value(&batch, "pre_close", row_index),
                    volume: numeric_column_value_fallback(&batch, &["volume", "vol"], row_index),
                    amount: numeric_column_value(&batch, "amount", row_index),
                },
            );
        }
    }
    Ok(rows)
}

fn read_daily_basic_rows(path: &Path) -> Result<BTreeMap<i64, DailyBasicRow>, String> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let mut rows = BTreeMap::new();
    for batch in read_parquet_batches(path)? {
        let trade_date = required_array(&batch, "trade_date", path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(trade_date.as_ref(), row_index) else {
                continue;
            };
            rows.insert(
                date_ns,
                DailyBasicRow {
                    turnover_rate: numeric_column_value(&batch, "turnover_rate", row_index),
                    turnover_rate_f: numeric_column_value(&batch, "turnover_rate_f", row_index),
                    volume_ratio: numeric_column_value(&batch, "volume_ratio", row_index),
                    pe: numeric_column_value(&batch, "pe", row_index),
                    pe_ttm: numeric_column_value(&batch, "pe_ttm", row_index),
                    pb: numeric_column_value(&batch, "pb", row_index),
                    ps: numeric_column_value(&batch, "ps", row_index),
                    ps_ttm: numeric_column_value(&batch, "ps_ttm", row_index),
                    dv_ratio: numeric_column_value(&batch, "dv_ratio", row_index),
                    dv_ttm: numeric_column_value(&batch, "dv_ttm", row_index),
                    total_share: numeric_column_value(&batch, "total_share", row_index),
                    float_share: numeric_column_value(&batch, "float_share", row_index),
                    free_share: numeric_column_value(&batch, "free_share", row_index),
                    total_mv: numeric_column_value(&batch, "total_mv", row_index),
                    circ_mv: numeric_column_value(&batch, "circ_mv", row_index),
                },
            );
        }
    }
    Ok(rows)
}

fn read_adj_factor_rows(path: &Path) -> Result<BTreeMap<i64, AdjFactorRow>, String> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let mut rows = BTreeMap::new();
    for batch in read_parquet_batches(path)? {
        let trade_date = required_array(&batch, "trade_date", path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(trade_date.as_ref(), row_index) else {
                continue;
            };
            rows.insert(
                date_ns,
                AdjFactorRow {
                    adj_factor: numeric_column_value(&batch, "adj_factor", row_index),
                },
            );
        }
    }
    Ok(rows)
}

fn read_stk_limit_rows(path: &Path) -> Result<BTreeMap<i64, StkLimitRow>, String> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let mut rows = BTreeMap::new();
    for batch in read_parquet_batches(path)? {
        let trade_date = required_array(&batch, "trade_date", path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(trade_date.as_ref(), row_index) else {
                continue;
            };
            rows.insert(
                date_ns,
                StkLimitRow {
                    limit_pre_close: numeric_column_value(&batch, "pre_close", row_index),
                    up_limit: numeric_column_value(&batch, "up_limit", row_index),
                    down_limit: numeric_column_value(&batch, "down_limit", row_index),
                },
            );
        }
    }
    Ok(rows)
}

fn build_processed_rows(
    symbol: &str,
    daily: BTreeMap<i64, DailyRawRow>,
    daily_basic: BTreeMap<i64, DailyBasicRow>,
    adj_factor: BTreeMap<i64, AdjFactorRow>,
    stk_limit: BTreeMap<i64, StkLimitRow>,
) -> Vec<ProcessedRow> {
    let mut rows = Vec::with_capacity(daily.len());
    let mut previous_close_adj = f64::NAN;
    for (date_ns, daily_row) in daily {
        let basic = daily_basic.get(&date_ns).cloned().unwrap_or_default();
        let adj = adj_factor.get(&date_ns).cloned().unwrap_or(AdjFactorRow {
            adj_factor: f64::NAN,
        });
        let limit = stk_limit.get(&date_ns).cloned().unwrap_or_default();
        let factor = if adj.adj_factor.is_finite() {
            adj.adj_factor
        } else {
            1.0
        };
        let open = daily_row.open * factor;
        let high = daily_row.high * factor;
        let low = daily_row.low * factor;
        let close = daily_row.close * factor;
        let fallback_pre_close = daily_row.pre_close * factor;
        let pre_close = if previous_close_adj.is_finite() {
            previous_close_adj
        } else {
            fallback_pre_close
        };
        previous_close_adj = close;
        let change = close - pre_close;
        let pct_chg = if pre_close > 0.0 {
            change / pre_close * 100.0
        } else {
            f64::NAN
        };
        let amplitude = if pre_close > 0.0 {
            (high - low) / pre_close * 100.0
        } else {
            f64::NAN
        };
        let mut values = BTreeMap::new();
        values.insert("open", open);
        values.insert("high", high);
        values.insert("low", low);
        values.insert("close", close);
        values.insert("pre_close", pre_close);
        values.insert("volume", daily_row.volume);
        values.insert("amount", daily_row.amount * 1000.0);
        values.insert("amplitude", amplitude);
        values.insert("pct_chg", pct_chg);
        values.insert("change", change);
        values.insert("turnover", basic.turnover_rate);
        values.insert("turnover_free", basic.turnover_rate_f);
        values.insert("volume_ratio", basic.volume_ratio);
        values.insert("total_mv", basic.total_mv * 10000.0);
        values.insert("circ_mv", basic.circ_mv * 10000.0);
        values.insert("total_share", basic.total_share * 10000.0);
        values.insert("circ_share", basic.float_share * 10000.0);
        values.insert("free_share", basic.free_share * 10000.0);
        values.insert("pe", basic.pe);
        values.insert("pe_ttm", basic.pe_ttm);
        values.insert("pb", basic.pb);
        values.insert("ps", basic.ps);
        values.insert("ps_ttm", basic.ps_ttm);
        values.insert("dv_ratio", basic.dv_ratio);
        values.insert("dv_ttm", basic.dv_ttm);
        values.insert("limit_pre_close", limit.limit_pre_close * factor);
        values.insert("up_limit", limit.up_limit * factor);
        values.insert("down_limit", limit.down_limit * factor);
        values.insert("adj_factor", factor);
        values.insert("raw_open", daily_row.open);
        values.insert("raw_high", daily_row.high);
        values.insert("raw_low", daily_row.low);
        values.insert("raw_close", daily_row.close);
        values.insert("raw_pre_close", daily_row.pre_close);
        rows.push(ProcessedRow {
            date_ns: daily_row.date_ns,
            symbol: symbol.to_owned(),
            ts_code: daily_row.ts_code,
            values,
        });
    }
    rows
}

fn write_processed_rows(path: &Path, rows: &[ProcessedRow]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let schema = Arc::new(Schema::new(
        processed_columns()
            .iter()
            .map(|column| match *column {
                "date" => Field::new(
                    "date",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                "symbol" | "ts_code" => Field::new(*column, DataType::Utf8, true),
                _ => Field::new(*column, DataType::Float64, true),
            })
            .collect::<Vec<_>>(),
    ));
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for column in processed_columns() {
        match *column {
            "date" => arrays.push(Arc::new(TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|row| Some(row.date_ns / 1_000))
                    .collect::<Vec<_>>(),
            ))),
            "symbol" => arrays.push(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| Some(row.symbol.as_str()))
                    .collect::<Vec<_>>(),
            ))),
            "ts_code" => arrays.push(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| Some(row.ts_code.as_str()))
                    .collect::<Vec<_>>(),
            ))),
            name => arrays.push(Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| {
                        let value = row.values.get(name).copied().unwrap_or(f64::NAN);
                        value.is_finite().then_some(value)
                    })
                    .collect::<Vec<_>>(),
            ))),
        }
    }
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|err| format!("failed to build processed record batch: {err}"))?;
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|err| format!("failed to create parquet writer {}: {err}", path.display()))?;
    writer
        .write(&batch)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    writer
        .close()
        .map_err(|err| format!("failed to close {}: {err}", path.display()))?;
    Ok(())
}

fn read_parquet_batches(path: &Path) -> Result<Vec<RecordBatch>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|err| format!("failed to open parquet {}: {err}", path.display()))?
        .with_batch_size(65_536)
        .build()
        .map_err(|err| format!("failed to build parquet reader {}: {err}", path.display()))?;
    reader
        .map(|batch| batch.map_err(|err| format!("failed to read {}: {err}", path.display())))
        .collect()
}

fn infer_parquet_latest_date(path: &Path, date_column: &str) -> Result<Option<i64>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let mut latest: Option<i64> = None;
    for batch in read_parquet_batches(path)? {
        let Some(array) = batch.column_by_name(date_column) else {
            return Ok(None);
        };
        for row_index in 0..batch.num_rows() {
            if let Some(date_ns) = date_value_ns(array.as_ref(), row_index) {
                latest = Some(latest.map_or(date_ns, |current| current.max(date_ns)));
            }
        }
    }
    Ok(latest)
}

fn required_array(batch: &RecordBatch, name: &str, path: &Path) -> Result<ArrayRef, String> {
    batch
        .column_by_name(name)
        .cloned()
        .ok_or_else(|| format!("{} missing required column {name}", path.display()))
}

fn numeric_column_value(batch: &RecordBatch, name: &str, row_index: usize) -> f64 {
    batch
        .column_by_name(name)
        .and_then(|array| numeric_value(array.as_ref(), row_index))
        .unwrap_or(f64::NAN)
}

fn numeric_column_value_fallback(batch: &RecordBatch, names: &[&str], row_index: usize) -> f64 {
    for name in names {
        if let Some(array) = batch.column_by_name(name) {
            return numeric_value(array.as_ref(), row_index).unwrap_or(f64::NAN);
        }
    }
    f64::NAN
}

fn string_column_value(batch: &RecordBatch, name: &str, row_index: usize) -> Option<String> {
    batch
        .column_by_name(name)
        .and_then(|array| string_value(array.as_ref(), row_index))
}

fn numeric_value(array: &dyn Array, idx: usize) -> Option<f64> {
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return values.is_valid(idx).then(|| values.value(idx));
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return values.is_valid(idx).then(|| values.value(idx) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return values.is_valid(idx).then(|| values.value(idx) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return values.is_valid(idx).then(|| values.value(idx) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return values.is_valid(idx).then(|| values.value(idx) as f64);
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return values.is_valid(idx).then(|| values.value(idx) as f64);
    }
    None
}

fn string_value(array: &dyn Array, idx: usize) -> Option<String> {
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        return values.is_valid(idx).then(|| values.value(idx).to_owned());
    }
    if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
        return values.is_valid(idx).then(|| values.value(idx).to_owned());
    }
    None
}

fn date_value_ns(array: &dyn Array, idx: usize) -> Option<i64> {
    if let Some(values) = array.as_any().downcast_ref::<TimestampNanosecondArray>() {
        return values.is_valid(idx).then(|| values.value(idx));
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMicrosecondArray>() {
        return values.is_valid(idx).then(|| values.value(idx) * 1_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampMillisecondArray>() {
        return values.is_valid(idx).then(|| values.value(idx) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<TimestampSecondArray>() {
        return values
            .is_valid(idx)
            .then(|| values.value(idx) * 1_000_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date64Array>() {
        return values.is_valid(idx).then(|| values.value(idx) * 1_000_000);
    }
    if let Some(values) = array.as_any().downcast_ref::<Date32Array>() {
        return values
            .is_valid(idx)
            .then(|| values.value(idx) as i64 * 86_400_000_000_000);
    }
    if let Some(value) = string_value(array, idx) {
        return parse_date_ns(&value);
    }
    None
}

fn parse_date_ns(value: &str) -> Option<i64> {
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .and_then(|dt| dt.and_utc().timestamp_nanos_opt());
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y%m%d") {
        return date
            .and_hms_opt(0, 0, 0)
            .and_then(|dt| dt.and_utc().timestamp_nanos_opt());
    }
    None
}

fn normalize_local_symbol(symbol: &str) -> String {
    symbol.split('.').next().unwrap_or(symbol).trim().to_owned()
}

fn local_symbol_to_ts(symbol: &str) -> String {
    if symbol.starts_with("600")
        || symbol.starts_with("601")
        || symbol.starts_with("603")
        || symbol.starts_with("605")
        || symbol.starts_with("688")
    {
        format!("{symbol}.SH")
    } else {
        format!("{symbol}.SZ")
    }
}

fn format_date(date_ns: i64) -> String {
    chrono::DateTime::from_timestamp(date_ns / 1_000_000_000, 0)
        .map(|dt| dt.date_naive().to_string())
        .unwrap_or_else(|| "n/a".to_owned())
}

fn processed_columns() -> &'static [&'static str] {
    &[
        "date",
        "symbol",
        "ts_code",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "amplitude",
        "pct_chg",
        "change",
        "turnover",
        "turnover_free",
        "volume_ratio",
        "total_mv",
        "circ_mv",
        "total_share",
        "circ_share",
        "free_share",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "limit_pre_close",
        "up_limit",
        "down_limit",
        "adj_factor",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_pre_close",
    ]
}

fn akshare_processed_columns() -> &'static [&'static str] {
    &[
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "amplitude",
        "pct_chg",
        "change",
        "turnover",
        "val_pct_chg",
        "total_mv",
        "circ_mv",
        "total_share",
        "circ_share",
        "pe_ttm",
        "pe_static",
        "pb",
        "peg",
        "pcf",
        "ps",
    ]
}

fn rebuild_tushare_packed_source_from_local(workers: usize) -> Result<JsonValue, String> {
    let processed_dir = PathBuf::from(TS_ROOT).join("processed/combined");
    let mut symbols = sorted_parquet_stems(&processed_dir)?;
    symbols.retain(|symbol| processed_dir.join(format!("{symbol}.parquet")).exists());
    if symbols.is_empty() {
        return Err(format!(
            "No processed Tushare parquet files found in {}",
            processed_dir.display()
        ));
    }

    fs::create_dir_all(TUSHARE_SOURCE_BUCKET_DIR)
        .map_err(|err| format!("failed to create {TUSHARE_SOURCE_BUCKET_DIR}: {err}"))?;
    fs::create_dir_all(
        PathBuf::from(TUSHARE_INDUSTRY_CONTEXT_PATH)
            .parent()
            .unwrap(),
    )
    .map_err(|err| format!("failed to create Tushare raw meta dir: {err}"))?;

    println!(
        "[collect] building Rust Tushare industry context for {} symbols",
        symbols.len()
    );
    let industry_map = read_tushare_industry_map()?;
    let industry_context = build_tushare_industry_context(&symbols, &industry_map)?;
    write_tushare_industry_context(&industry_context)?;
    let context_lookup = Arc::new(build_industry_context_lookup(&industry_context));
    let industry_map = Arc::new(industry_map);

    let bucket_count = DEFAULT_PACKED_SOURCE_BUCKET_COUNT;
    let mut bucket_to_symbols: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for symbol in &symbols {
        bucket_to_symbols
            .entry(stable_bucket_id_rust(symbol, bucket_count))
            .or_default()
            .push(symbol.clone());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers.max(1))
        .build()
        .map_err(|err| format!("failed to build packed-source worker pool: {err}"))?;
    let bucket_jobs = bucket_to_symbols
        .iter()
        .map(|(bucket_id, bucket_symbols)| (*bucket_id, bucket_symbols.clone()))
        .collect::<Vec<_>>();
    let bucket_results = pool.install(|| {
        bucket_jobs
            .par_iter()
            .map(|(bucket_id, bucket_symbols)| {
                build_tushare_packed_source_bucket(
                    *bucket_id,
                    bucket_symbols,
                    &industry_map,
                    &context_lookup,
                )
            })
            .collect::<Vec<_>>()
    });

    let mut manifest_rows = Vec::new();
    let mut available_dates = BTreeSet::new();
    for result in bucket_results {
        let (rows, dates) = result?;
        manifest_rows.extend(rows);
        available_dates.extend(dates);
    }
    if manifest_rows.is_empty() {
        return Err("Rust packed-source rebuild produced no manifest rows".to_owned());
    }
    manifest_rows.sort_by(|left, right| {
        left.bucket_id
            .cmp(&right.bucket_id)
            .then_with(|| left.symbol.cmp(&right.symbol))
    });
    write_packed_source_manifest(&manifest_rows)?;
    remove_inactive_bucket_files(&bucket_to_symbols)?;

    let active_bucket_ids = bucket_to_symbols.keys().copied().collect::<Vec<_>>();
    let schema_columns = source_columns()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let metadata = serde_json::json!({
        "storage_format": "parquet",
        "storage_layout": "bucket_shards",
        "source_kind": "tushare_packed_source",
        "data_source": "tushare",
        "processed_dir": processed_dir.display().to_string(),
        "source_dir": TUSHARE_SOURCE_DIR,
        "buckets_dir": TUSHARE_SOURCE_BUCKET_DIR,
        "manifest_path": TUSHARE_SOURCE_MANIFEST_PATH,
        "bucket_count": bucket_count,
        "bucket_ids": active_bucket_ids,
        "num_symbols": manifest_rows.iter().map(|row| row.symbol.as_str()).collect::<BTreeSet<_>>().len(),
        "num_rows": manifest_rows.iter().map(|row| row.row_count).sum::<usize>(),
        "available_dates": available_dates.into_iter().collect::<Vec<_>>(),
        "schema_columns": schema_columns,
        "source_layout_assumptions": {
            "tushare_event_availability_policy": TUSHARE_EVENT_AVAILABILITY_POLICY,
            "tushare_industry_mapping": TUSHARE_INDUSTRY_MAPPING_POLICY,
            "packed_source_dependency_signature_policy": PACKED_SOURCE_DEPENDENCY_SIGNATURE_POLICY,
        },
        "incremental": {
            "enabled": false,
            "reused_buckets": 0,
            "rebuilt_buckets": bucket_to_symbols.len(),
            "reused_symbols": 0,
            "rebuilt_symbols": symbols.len(),
        },
    });
    fs::write(
        TUSHARE_SOURCE_META_PATH,
        serde_json::to_string_pretty(&metadata)
            .map_err(|err| format!("failed to encode packed-source metadata: {err}"))?,
    )
    .map_err(|err| format!("failed to write {TUSHARE_SOURCE_META_PATH}: {err}"))?;
    Ok(metadata)
}

fn sorted_parquet_stems(dir: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let entries =
        fs::read_dir(dir).map_err(|err| format!("failed to read {}: {err}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|err| format!("failed to read directory entry in {}: {err}", dir.display()))?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("parquet") {
            if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
                out.push(stem.to_owned());
            }
        }
    }
    out.sort();
    Ok(out)
}

fn read_tushare_industry_map() -> Result<HashMap<String, String>, String> {
    let path = PathBuf::from(TUSHARE_SYMBOL_CACHE_PATH);
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let mut map = HashMap::new();
    for batch in read_parquet_batches(&path)? {
        let Some(symbol_array) = batch
            .column_by_name("local_symbol")
            .or_else(|| batch.column_by_name("symbol"))
        else {
            continue;
        };
        let Some(industry_array) = batch.column_by_name("industry") else {
            continue;
        };
        for row_index in 0..batch.num_rows() {
            let Some(symbol) = string_value(symbol_array.as_ref(), row_index) else {
                continue;
            };
            let industry = string_value(industry_array.as_ref(), row_index)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "UNKNOWN".to_owned());
            let local = normalize_local_symbol(&symbol);
            if !local.is_empty() {
                map.insert(format!("{local:0>6}"), industry);
            }
        }
    }
    Ok(map)
}

fn build_tushare_industry_context(
    symbols: &[String],
    industry_map: &HashMap<String, String>,
) -> Result<Vec<IndustryDailyRow>, String> {
    let mut accum: BTreeMap<(i64, String), IndustryAccum> = BTreeMap::new();
    let mut market_returns: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for symbol in symbols {
        let frame = read_source_rows_from_processed(symbol)?;
        if frame.is_empty() {
            continue;
        }
        let dates = frame.iter().map(|row| row.date_ns).collect::<Vec<_>>();
        let fi_sidecar = load_sidecar_series(
            symbol,
            &dates,
            "fina_indicator",
            tushare_fina_indicator_pairs(),
            None,
        )?;
        let div_sidecar =
            load_sidecar_series(symbol, &dates, "dividend", tushare_dividend_pairs(), None)?;
        let industry = industry_map
            .get(symbol)
            .cloned()
            .unwrap_or_else(|| "UNKNOWN".to_owned());
        let mut previous_close = f64::NAN;
        for row in frame {
            let close = row.values.get("close").copied().unwrap_or(f64::NAN);
            let ret = if previous_close.is_finite() && previous_close.abs() > EPS {
                close / previous_close - 1.0
            } else {
                f64::NAN
            };
            previous_close = close;
            if !ret.is_finite() {
                continue;
            }
            market_returns.entry(row.date_ns).or_default().push(ret);
            let mut metrics = BTreeMap::new();
            let amount_abs = row.values.get("amount").copied().unwrap_or(f64::NAN).abs();
            let amihud = ret.abs() / (amount_abs + EPS);
            let downside_amihud = if ret < 0.0 { amihud } else { f64::NAN };
            let hit_up_limit = hit_limit(
                close,
                row.values.get("up_limit").copied().unwrap_or(f64::NAN),
                true,
            );
            let hit_down_limit = hit_limit(
                close,
                row.values.get("down_limit").copied().unwrap_or(f64::NAN),
                false,
            );
            metrics.insert("ret", ret);
            metrics.insert(
                "turnover",
                row.values.get("turnover").copied().unwrap_or(f64::NAN),
            );
            metrics.insert(
                "turnover_free",
                row.values.get("turnover_free").copied().unwrap_or(f64::NAN),
            );
            metrics.insert(
                "volume_ratio",
                row.values.get("volume_ratio").copied().unwrap_or(f64::NAN),
            );
            metrics.insert(
                "amplitude",
                row.values.get("amplitude").copied().unwrap_or(f64::NAN),
            );
            metrics.insert("amihud", amihud);
            metrics.insert("downside_amihud", downside_amihud);
            metrics.insert("hit_up_limit", hit_up_limit);
            metrics.insert("hit_down_limit", hit_down_limit);
            metrics.insert(
                "ep",
                positive_inverse_value(row.values.get("pe").copied().unwrap_or(f64::NAN))
                    .unwrap_or(-1.0),
            );
            metrics.insert(
                "sp",
                positive_inverse_value(row.values.get("ps").copied().unwrap_or(f64::NAN))
                    .unwrap_or(-1.0),
            );
            metrics.insert(
                "sp_ttm",
                positive_inverse_value(row.values.get("ps_ttm").copied().unwrap_or(f64::NAN))
                    .unwrap_or(-1.0),
            );
            metrics.insert(
                "bp",
                positive_inverse_value(row.values.get("pb").copied().unwrap_or(f64::NAN))
                    .unwrap_or(-1.0),
            );
            metrics.insert(
                "ep_clean",
                positive_inverse_value(row.values.get("pe").copied().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            metrics.insert(
                "sp_clean",
                positive_inverse_value(row.values.get("ps").copied().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            metrics.insert(
                "sp_ttm_clean",
                positive_inverse_value(row.values.get("ps_ttm").copied().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            metrics.insert(
                "bp_clean",
                positive_inverse_value(row.values.get("pb").copied().unwrap_or(f64::NAN))
                    .unwrap_or(f64::NAN),
            );
            metrics.insert(
                "dividend_yield",
                row.values.get("dv_ratio").copied().unwrap_or(f64::NAN),
            );
            metrics.insert(
                "dividend_yield_ttm",
                row.values.get("dv_ttm").copied().unwrap_or(f64::NAN),
            );
            let fi = fi_sidecar.get(&row.date_ns);
            let div = div_sidecar.get(&row.date_ns);
            let fi_eps = sidecar_value(fi, "fi_eps");
            let fi_ocfps = sidecar_value(fi, "fi_ocfps");
            let dividend_cash = sidecar_value(div, "div_cash_div");
            metrics.insert(
                "dividend_cash_to_eps",
                positive_ratio_value(dividend_cash, fi_eps),
            );
            metrics.insert(
                "dividend_cash_to_ocfps",
                positive_ratio_value(dividend_cash, fi_ocfps),
            );
            metrics.insert(
                "dividend_cash_yield_proxy",
                if close > EPS {
                    dividend_cash / close
                } else {
                    f64::NAN
                },
            );
            metrics.insert("fi_ocf_to_eps", positive_ratio_value(fi_ocfps, fi_eps));
            metrics.insert("fi_ocfps_minus_eps", fi_ocfps - fi_eps);
            metrics.insert(
                "fi_roe_quality_gap",
                sidecar_value(fi, "fi_roe_dt") - sidecar_value(fi, "fi_roe"),
            );
            metrics.insert(
                "fi_margin_quality",
                sidecar_value(fi, "fi_grossprofit_margin")
                    - sidecar_value(fi, "fi_netprofit_margin"),
            );
            let entry = accum.entry((row.date_ns, industry.clone())).or_default();
            entry.count += 1;
            entry.ret_values.push(ret);
            for (name, value) in metrics {
                if value.is_finite() {
                    let item = entry.sums.entry(name).or_insert((0.0, 0));
                    item.0 += value;
                    item.1 += 1;
                }
            }
        }
    }
    let market_daily = market_returns
        .into_iter()
        .map(|(date_ns, values)| (date_ns, mean_values(&values)))
        .collect::<BTreeMap<_, _>>();
    let mut by_industry: BTreeMap<String, Vec<IndustryDailyRow>> = BTreeMap::new();
    for ((date_ns, industry), entry) in accum {
        let mut values = BTreeMap::new();
        let ind_daily_ret = mean_values(&entry.ret_values);
        values.insert("ind_member_count".to_owned(), entry.count as f64);
        values.insert("ind_daily_ret".to_owned(), ind_daily_ret);
        values.insert(
            "ind_excess_daily_ret".to_owned(),
            ind_daily_ret - market_daily.get(&date_ns).copied().unwrap_or(f64::NAN),
        );
        values.insert(
            "ind_daily_pos_rate".to_owned(),
            mean_values(
                &entry
                    .ret_values
                    .iter()
                    .map(|value| if *value > 0.0 { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            ),
        );
        values.insert(
            "ind_daily_dispersion".to_owned(),
            sample_std_values(&entry.ret_values),
        );
        for (source_name, output_name) in industry_daily_mean_mappings() {
            let value = entry
                .sums
                .get(source_name)
                .and_then(|(sum, count)| (*count > 0).then_some(*sum / *count as f64))
                .unwrap_or(f64::NAN);
            values.insert((*output_name).to_owned(), value);
        }
        by_industry
            .entry(industry.clone())
            .or_default()
            .push(IndustryDailyRow {
                date_ns,
                industry,
                values,
            });
    }
    let mut output = Vec::new();
    for (_industry, mut rows) in by_industry {
        rows.sort_by_key(|row| row.date_ns);
        add_industry_rolling_context(&mut rows, &market_daily);
        output.extend(rows);
    }
    output.sort_by(|left, right| {
        left.industry
            .cmp(&right.industry)
            .then_with(|| left.date_ns.cmp(&right.date_ns))
    });
    Ok(output)
}

fn add_industry_rolling_context(rows: &mut [IndustryDailyRow], market_daily: &BTreeMap<i64, f64>) {
    let dates = rows.iter().map(|row| row.date_ns).collect::<Vec<_>>();
    let daily_ret = rows
        .iter()
        .map(|row| row.values.get("ind_daily_ret").copied().unwrap_or(f64::NAN))
        .collect::<Vec<_>>();
    let daily_pos = rows
        .iter()
        .map(|row| {
            row.values
                .get("ind_daily_pos_rate")
                .copied()
                .unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();
    let daily_dispersion = rows
        .iter()
        .map(|row| {
            row.values
                .get("ind_daily_dispersion")
                .copied()
                .unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();
    for &window in &[5usize, 20, 60] {
        let industry_ret = rolling_compound_return(&daily_ret, window);
        let industry_std = rolling_sample_std(&daily_ret, window);
        let pos_rate = rolling_mean_vec(&daily_pos, window);
        let dispersion = rolling_mean_vec(&daily_dispersion, window);
        for idx in 0..rows.len() {
            let market_window = rolling_market_return(market_daily, dates[idx], window);
            rows[idx]
                .values
                .insert(format!("ind_ret_{window}"), industry_ret[idx]);
            rows[idx]
                .values
                .insert(format!("ind_std_{window}"), industry_std[idx]);
            rows[idx].values.insert(
                format!("ind_excess_ret_{window}"),
                industry_ret[idx] - market_window,
            );
            rows[idx]
                .values
                .insert(format!("ind_pos_rate_{window}"), pos_rate[idx]);
            rows[idx]
                .values
                .insert(format!("ind_dispersion_{window}"), dispersion[idx]);
        }
    }
    for &window in &[20usize, 60] {
        for (output_suffix, source_col) in relative_industry_context_sources() {
            let source = rows
                .iter()
                .map(|row| row.values.get(*source_col).copied().unwrap_or(f64::NAN))
                .collect::<Vec<_>>();
            let rolled = rolling_mean_vec(&source, window);
            for idx in 0..rows.len() {
                rows[idx]
                    .values
                    .insert(format!("ind_{output_suffix}_{window}"), rolled[idx]);
            }
        }
    }
}

fn build_tushare_packed_source_bucket(
    bucket_id: usize,
    symbols: &[String],
    industry_map: &HashMap<String, String>,
    context_lookup: &HashMap<String, Vec<IndustryDailyRow>>,
) -> Result<(Vec<PackedSourceManifestRow>, BTreeSet<String>), String> {
    let mut rows = Vec::new();
    let mut manifest = Vec::new();
    let mut available_dates = BTreeSet::new();
    for symbol in symbols {
        let processed_path = PathBuf::from(TS_ROOT)
            .join("processed/combined")
            .join(format!("{symbol}.parquet"));
        let mut symbol_rows = read_source_rows_from_processed(symbol)?;
        if symbol_rows.is_empty() {
            continue;
        }
        augment_tushare_symbol_rows(symbol, &mut symbol_rows, industry_map, context_lookup)?;
        let min_date = symbol_rows
            .first()
            .map(|row| format_date(row.date_ns))
            .unwrap_or_else(|| "n/a".to_owned());
        let max_date = symbol_rows
            .last()
            .map(|row| format_date(row.date_ns))
            .unwrap_or_else(|| "n/a".to_owned());
        for row in &symbol_rows {
            available_dates.insert(format_date(row.date_ns));
        }
        let stat = processed_path
            .metadata()
            .map_err(|err| format!("failed to stat {}: {err}", processed_path.display()))?;
        manifest.push(PackedSourceManifestRow {
            symbol: symbol.clone(),
            bucket_id,
            source_path: processed_path
                .canonicalize()
                .unwrap_or_else(|_| processed_path.clone())
                .display()
                .to_string(),
            source_size: stat.len() as i64,
            source_mtime_ns: file_mtime_ns(&stat),
            dependency_signature: packed_source_dependency_signature(symbol, &processed_path)?,
            row_count: symbol_rows.len(),
            min_date,
            max_date,
        });
        rows.extend(symbol_rows);
    }
    rows.sort_by(|left, right| {
        left.symbol
            .cmp(&right.symbol)
            .then_with(|| left.date_ns.cmp(&right.date_ns))
    });
    if !rows.is_empty() {
        write_source_rows(
            &PathBuf::from(TUSHARE_SOURCE_BUCKET_DIR).join(format!("part-{bucket_id:04}.parquet")),
            &rows,
        )?;
    }
    Ok((manifest, available_dates))
}

fn read_source_rows_from_processed(symbol: &str) -> Result<Vec<SourceRow>, String> {
    let path = PathBuf::from(TS_ROOT)
        .join("processed/combined")
        .join(format!("{symbol}.parquet"));
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut rows = Vec::new();
    for batch in read_parquet_batches(&path)? {
        let date_array = required_array(&batch, "date", &path)?;
        for row_index in 0..batch.num_rows() {
            let Some(date_ns) = date_value_ns(date_array.as_ref(), row_index) else {
                continue;
            };
            let row_symbol = string_column_value(&batch, "symbol", row_index)
                .unwrap_or_else(|| symbol.to_owned());
            let ts_code = string_column_value(&batch, "ts_code", row_index)
                .unwrap_or_else(|| local_symbol_to_ts(symbol));
            let mut values = BTreeMap::new();
            for column in processed_columns() {
                if matches!(*column, "date" | "symbol" | "ts_code") {
                    continue;
                }
                values.insert(
                    (*column).to_owned(),
                    numeric_column_value(&batch, column, row_index),
                );
            }
            rows.push(SourceRow {
                date_ns,
                symbol: row_symbol,
                ts_code,
                values,
            });
        }
    }
    rows.sort_by_key(|row| row.date_ns);
    Ok(rows)
}

fn augment_tushare_symbol_rows(
    symbol: &str,
    rows: &mut [SourceRow],
    industry_map: &HashMap<String, String>,
    context_lookup: &HashMap<String, Vec<IndustryDailyRow>>,
) -> Result<(), String> {
    let dates = rows.iter().map(|row| row.date_ns).collect::<Vec<_>>();
    let sidecar_specs = [
        ("fina_indicator", tushare_fina_indicator_pairs(), None),
        ("dividend", tushare_dividend_pairs(), None),
        (
            "forecast",
            tushare_forecast_pairs(),
            Some("fc_days_since_ann"),
        ),
        (
            "express",
            tushare_express_pairs(),
            Some("exp_days_since_ann"),
        ),
    ];
    for (stage, pairs, days_col) in sidecar_specs {
        let series = load_sidecar_series(symbol, &dates, stage, pairs, days_col)?;
        for row in rows.iter_mut() {
            if let Some(values) = series.get(&row.date_ns) {
                for (name, value) in values {
                    row.values.insert((*name).to_owned(), *value);
                }
            }
        }
    }
    let industry = industry_map
        .get(symbol)
        .cloned()
        .unwrap_or_else(|| "UNKNOWN".to_owned());
    if let Some(context_rows) = context_lookup.get(&industry) {
        for row in rows.iter_mut() {
            if let Some(context) = asof_industry_context(context_rows, row.date_ns) {
                for name in industry_context_cols() {
                    row.values.insert(
                        (*name).to_owned(),
                        context.values.get(*name).copied().unwrap_or(f64::NAN),
                    );
                }
            }
        }
    }
    for row in rows.iter_mut() {
        for name in source_numeric_columns() {
            row.values.entry(name.to_owned()).or_insert(f64::NAN);
        }
    }
    Ok(())
}

fn load_sidecar_series(
    symbol: &str,
    trading_dates: &[i64],
    stage_name: &str,
    pairs: &[(&'static str, &'static str)],
    days_since_ann_column: Option<&'static str>,
) -> Result<BTreeMap<i64, BTreeMap<&'static str, f64>>, String> {
    let path = PathBuf::from(TS_ROOT)
        .join("raw")
        .join(stage_name)
        .join(format!("{symbol}.parquet"));
    let mut output = BTreeMap::new();
    if !path.exists() || trading_dates.is_empty() {
        return Ok(output);
    }
    let mut events = Vec::new();
    for batch in read_parquet_batches(&path)? {
        let Some(ann_array) = batch.column_by_name("ann_date") else {
            continue;
        };
        for row_index in 0..batch.num_rows() {
            let Some(ann_ns) = date_value_ns(ann_array.as_ref(), row_index) else {
                continue;
            };
            let insert_pos = trading_dates.partition_point(|date_ns| *date_ns <= ann_ns);
            let Some(&available_ns) = trading_dates.get(insert_pos) else {
                continue;
            };
            let mut values = BTreeMap::new();
            for (source, target) in pairs {
                values.insert(*target, numeric_column_value(&batch, source, row_index));
            }
            if let Some(column) = days_since_ann_column {
                values.insert(column, f64::NAN);
            }
            events.push(SidecarEvent {
                ann_day: ann_ns / NS_PER_DAY,
                available_ns,
                values,
            });
        }
    }
    events.sort_by_key(|event| event.available_ns);
    let mut event_idx = 0usize;
    let mut current: Option<&SidecarEvent> = None;
    for &date_ns in trading_dates {
        while event_idx < events.len() && events[event_idx].available_ns <= date_ns {
            current = Some(&events[event_idx]);
            event_idx += 1;
        }
        if let Some(event) = current {
            let mut values = event.values.clone();
            if let Some(column) = days_since_ann_column {
                values.insert(column, (date_ns / NS_PER_DAY - event.ann_day) as f64);
            }
            output.insert(date_ns, values);
        }
    }
    Ok(output)
}

fn write_tushare_industry_context(rows: &[IndustryDailyRow]) -> Result<(), String> {
    let path = PathBuf::from(TUSHARE_INDUSTRY_CONTEXT_PATH);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let mut columns = vec!["date", "industry"];
    columns.extend(industry_context_cols());
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| match *column {
                "date" => Field::new(
                    "date",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                "industry" => Field::new("industry", DataType::Utf8, true),
                _ => Field::new(*column, DataType::Float64, true),
            })
            .collect::<Vec<_>>(),
    ));
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for column in columns {
        match column {
            "date" => arrays.push(Arc::new(TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|row| Some(row.date_ns / 1_000))
                    .collect::<Vec<_>>(),
            ))),
            "industry" => arrays.push(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| Some(row.industry.as_str()))
                    .collect::<Vec<_>>(),
            ))),
            name => arrays.push(Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| {
                        let value = row.values.get(name).copied().unwrap_or(f64::NAN);
                        value.is_finite().then_some(value)
                    })
                    .collect::<Vec<_>>(),
            ))),
        }
    }
    write_record_batch_parquet(&path, schema, arrays)
}

fn write_source_rows(path: &Path, rows: &[SourceRow]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let columns = source_columns();
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| match *column {
                "date" => Field::new(
                    "date",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                ),
                "symbol" | "ts_code" => Field::new(*column, DataType::Utf8, true),
                _ => Field::new(*column, DataType::Float64, true),
            })
            .collect::<Vec<_>>(),
    ));
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for column in columns {
        match column {
            "date" => arrays.push(Arc::new(TimestampMicrosecondArray::from(
                rows.iter()
                    .map(|row| Some(row.date_ns / 1_000))
                    .collect::<Vec<_>>(),
            ))),
            "symbol" => arrays.push(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| Some(row.symbol.as_str()))
                    .collect::<Vec<_>>(),
            ))),
            "ts_code" => arrays.push(Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| Some(row.ts_code.as_str()))
                    .collect::<Vec<_>>(),
            ))),
            name => arrays.push(Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| {
                        let value = row.values.get(name).copied().unwrap_or(f64::NAN);
                        value.is_finite().then_some(value)
                    })
                    .collect::<Vec<_>>(),
            ))),
        }
    }
    write_record_batch_parquet(path, schema, arrays)
}

fn write_packed_source_manifest(rows: &[PackedSourceManifestRow]) -> Result<(), String> {
    let path = PathBuf::from(TUSHARE_SOURCE_MANIFEST_PATH);
    let schema = Arc::new(Schema::new(vec![
        Field::new("symbol", DataType::Utf8, true),
        Field::new("bucket_id", DataType::Int64, true),
        Field::new("source_path", DataType::Utf8, true),
        Field::new("source_size", DataType::Int64, true),
        Field::new("source_mtime_ns", DataType::Int64, true),
        Field::new("dependency_signature", DataType::Utf8, true),
        Field::new("row_count", DataType::Int64, true),
        Field::new("min_date", DataType::Utf8, true),
        Field::new("max_date", DataType::Utf8, true),
    ]));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| Some(row.symbol.as_str()))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| Some(row.bucket_id as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| Some(row.source_path.as_str()))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| Some(row.source_size))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| Some(row.source_mtime_ns))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| Some(row.dependency_signature.as_str()))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter()
                .map(|row| Some(row.row_count as i64))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| Some(row.min_date.as_str()))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter()
                .map(|row| Some(row.max_date.as_str()))
                .collect::<Vec<_>>(),
        )),
    ];
    write_record_batch_parquet(&path, schema, arrays)
}

fn write_record_batch_parquet(
    path: &Path,
    schema: Arc<Schema>,
    arrays: Vec<ArrayRef>,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|err| format!("failed to build record batch for {}: {err}", path.display()))?;
    let file =
        File::create(path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)
        .map_err(|err| format!("failed to create parquet writer {}: {err}", path.display()))?;
    writer
        .write(&batch)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    writer
        .close()
        .map_err(|err| format!("failed to close {}: {err}", path.display()))?;
    Ok(())
}

fn build_industry_context_lookup(
    rows: &[IndustryDailyRow],
) -> HashMap<String, Vec<IndustryDailyRow>> {
    let mut lookup: HashMap<String, Vec<IndustryDailyRow>> = HashMap::new();
    for row in rows {
        lookup
            .entry(row.industry.clone())
            .or_default()
            .push(row.clone());
    }
    for rows in lookup.values_mut() {
        rows.sort_by_key(|row| row.date_ns);
    }
    lookup
}

fn asof_industry_context(rows: &[IndustryDailyRow], date_ns: i64) -> Option<&IndustryDailyRow> {
    let idx = rows.partition_point(|row| row.date_ns <= date_ns);
    idx.checked_sub(1).and_then(|pos| rows.get(pos))
}

fn remove_inactive_bucket_files(
    bucket_to_symbols: &BTreeMap<usize, Vec<String>>,
) -> Result<(), String> {
    let active = bucket_to_symbols.keys().copied().collect::<BTreeSet<_>>();
    let bucket_dir = PathBuf::from(TUSHARE_SOURCE_BUCKET_DIR);
    if !bucket_dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(&bucket_dir)
        .map_err(|err| format!("failed to read {}: {err}", bucket_dir.display()))?
    {
        let path = entry
            .map_err(|err| format!("failed to read bucket dir entry: {err}"))?
            .path();
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        let Some(suffix) = stem.strip_prefix("part-") else {
            continue;
        };
        let Ok(bucket_id) = suffix.parse::<usize>() else {
            continue;
        };
        if !active.contains(&bucket_id) {
            fs::remove_file(&path).map_err(|err| {
                format!("failed to remove inactive bucket {}: {err}", path.display())
            })?;
        }
    }
    Ok(())
}

fn packed_source_dependency_signature(
    symbol: &str,
    processed_path: &Path,
) -> Result<String, String> {
    let sidecar = |stage: &str| {
        file_signature_json(
            &PathBuf::from(TS_ROOT)
                .join("raw")
                .join(stage)
                .join(format!("{symbol}.parquet")),
        )
    };
    let payload = serde_json::json!({
        "policy": PACKED_SOURCE_DEPENDENCY_SIGNATURE_POLICY,
        "processed": file_signature_json(processed_path)?,
        "sidecars": {
            "fina_indicator": sidecar("fina_indicator")?,
            "dividend": sidecar("dividend")?,
            "forecast": sidecar("forecast")?,
            "express": sidecar("express")?,
        },
        "industry_context": file_signature_json(&PathBuf::from(TUSHARE_INDUSTRY_CONTEXT_PATH))?,
    });
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to encode dependency signature: {err}"))
}

fn file_signature_json(path: &Path) -> Result<JsonValue, String> {
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !path.exists() {
        return Ok(serde_json::json!({
            "path": resolved.display().to_string(),
            "exists": false,
            "size": -1,
            "mtime_ns": -1,
        }));
    }
    let stat = path
        .metadata()
        .map_err(|err| format!("failed to stat {}: {err}", path.display()))?;
    Ok(serde_json::json!({
        "path": resolved.display().to_string(),
        "exists": true,
        "size": stat.len() as i64,
        "mtime_ns": file_mtime_ns(&stat),
    }))
}

#[cfg(unix)]
fn file_mtime_ns(stat: &fs::Metadata) -> i64 {
    use std::os::unix::fs::MetadataExt;
    stat.mtime() * 1_000_000_000 + stat.mtime_nsec()
}

#[cfg(not(unix))]
fn file_mtime_ns(stat: &fs::Metadata) -> i64 {
    stat.modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos() as i64)
        .unwrap_or(-1)
}

fn stable_bucket_id_rust(symbol: &str, bucket_count: usize) -> usize {
    crc32_ieee(symbol.as_bytes()) as usize % bucket_count.max(1)
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn mean_values(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            sum += *value;
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        f64::NAN
    }
}

fn sample_std_values(values: &[f64]) -> f64 {
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return f64::NAN;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let var = finite
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (finite.len() - 1) as f64;
    var.sqrt()
}

fn rolling_mean_vec(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|idx| {
            let start = idx.saturating_add(1).saturating_sub(window.max(1));
            mean_values(&values[start..=idx])
        })
        .collect()
}

fn rolling_sample_std(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|idx| {
            let start = idx.saturating_add(1).saturating_sub(window.max(1));
            sample_std_values(&values[start..=idx])
        })
        .collect()
}

fn rolling_compound_return(values: &[f64], window: usize) -> Vec<f64> {
    (0..values.len())
        .map(|idx| {
            let start = idx.saturating_add(1).saturating_sub(window.max(1));
            let mut log_sum = 0.0;
            let mut observed = false;
            for value in &values[start..=idx] {
                if value.is_finite() {
                    log_sum += value.max(-0.999999).ln_1p();
                    observed = true;
                }
            }
            if observed {
                log_sum.exp_m1()
            } else {
                f64::NAN
            }
        })
        .collect()
}

fn rolling_market_return(market_daily: &BTreeMap<i64, f64>, date_ns: i64, window: usize) -> f64 {
    let values = market_daily
        .range(..=date_ns)
        .rev()
        .take(window.max(1))
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    let mut log_sum = 0.0;
    let mut observed = false;
    for value in values.iter().rev() {
        if value.is_finite() {
            log_sum += value.max(-0.999999).ln_1p();
            observed = true;
        }
    }
    if observed {
        log_sum.exp_m1()
    } else {
        f64::NAN
    }
}

fn hit_limit(close: f64, limit: f64, upper: bool) -> f64 {
    if !close.is_finite() || !limit.is_finite() {
        return f64::NAN;
    }
    if upper {
        (close >= limit * (1.0 - 1e-6)) as i32 as f64
    } else {
        (close <= limit * (1.0 + 1e-6)) as i32 as f64
    }
}

fn positive_inverse_value(value: f64) -> Option<f64> {
    (value.is_finite() && value > EPS).then_some(1.0 / value)
}

fn positive_ratio_value(numerator: f64, denominator: f64) -> f64 {
    if numerator.is_finite() && denominator.is_finite() && denominator > EPS {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn sidecar_value(values: Option<&BTreeMap<&'static str, f64>>, name: &'static str) -> f64 {
    values
        .and_then(|item| item.get(name).copied())
        .unwrap_or(f64::NAN)
}

fn tushare_fina_indicator_pairs() -> &'static [(&'static str, &'static str)] {
    &[
        ("eps", "fi_eps"),
        ("dt_eps", "fi_dt_eps"),
        ("bps", "fi_bps"),
        ("ocfps", "fi_ocfps"),
        ("roe", "fi_roe"),
        ("roe_dt", "fi_roe_dt"),
        ("roa", "fi_roa"),
        ("grossprofit_margin", "fi_grossprofit_margin"),
        ("netprofit_margin", "fi_netprofit_margin"),
        ("debt_to_assets", "fi_debt_to_assets"),
        ("q_eps", "fi_q_eps"),
        ("q_dtprofit", "fi_q_dtprofit"),
        ("q_roe", "fi_q_roe"),
        ("q_dt_roe", "fi_q_dt_roe"),
        ("tr_yoy", "fi_tr_yoy"),
        ("or_yoy", "fi_or_yoy"),
        ("op_yoy", "fi_op_yoy"),
        ("netprofit_yoy", "fi_netprofit_yoy"),
        ("ocf_yoy", "fi_ocf_yoy"),
    ]
}

fn tushare_dividend_pairs() -> &'static [(&'static str, &'static str)] {
    &[
        ("cash_div", "div_cash_div"),
        ("cash_div_tax", "div_cash_div_tax"),
        ("stk_div", "div_stk_div"),
        ("stk_bo_rate", "div_stk_bo_rate"),
        ("stk_co_rate", "div_stk_co_rate"),
        ("base_share", "div_base_share"),
    ]
}

fn tushare_forecast_pairs() -> &'static [(&'static str, &'static str)] {
    &[
        ("p_change_min", "fc_p_change_min"),
        ("p_change_max", "fc_p_change_max"),
        ("net_profit_min", "fc_net_profit_min"),
        ("net_profit_max", "fc_net_profit_max"),
        ("last_parent_net", "fc_last_parent_net"),
    ]
}

fn tushare_express_pairs() -> &'static [(&'static str, &'static str)] {
    &[
        ("revenue", "exp_revenue"),
        ("operate_profit", "exp_operate_profit"),
        ("total_profit", "exp_total_profit"),
        ("n_income", "exp_n_income"),
        ("total_assets", "exp_total_assets"),
        ("diluted_eps", "exp_diluted_eps"),
        ("diluted_roe", "exp_diluted_roe"),
        ("yoy_sales", "exp_yoy_sales"),
        ("yoy_op", "exp_yoy_op"),
        ("yoy_tp", "exp_yoy_tp"),
        ("yoy_dedu_np", "exp_yoy_dedu_np"),
        ("yoy_eps", "exp_yoy_eps"),
        ("yoy_roe", "exp_yoy_roe"),
        ("growth_assets", "exp_growth_assets"),
        ("yoy_assets", "exp_yoy_assets"),
    ]
}

fn industry_daily_mean_mappings() -> &'static [(&'static str, &'static str)] {
    &[
        ("turnover", "ind_daily_turnover"),
        ("turnover_free", "ind_daily_free_turnover"),
        ("volume_ratio", "ind_daily_volume_ratio"),
        ("amihud", "ind_daily_amihud"),
        ("downside_amihud", "ind_daily_downside_amihud"),
        ("amplitude", "ind_daily_amplitude"),
        ("hit_up_limit", "ind_daily_hit_up_limit_rate"),
        ("hit_down_limit", "ind_daily_hit_down_limit_rate"),
        ("ep", "ind_ep_mean"),
        ("sp", "ind_sp_mean"),
        ("sp_ttm", "ind_sp_ttm_mean"),
        ("bp", "ind_bp_mean"),
        ("ep_clean", "ind_ep_clean_mean"),
        ("sp_clean", "ind_sp_clean_mean"),
        ("sp_ttm_clean", "ind_sp_ttm_clean_mean"),
        ("bp_clean", "ind_bp_clean_mean"),
        ("dividend_yield", "ind_dividend_yield_mean"),
        ("dividend_yield_ttm", "ind_dividend_yield_ttm_mean"),
        ("dividend_cash_to_eps", "ind_dividend_cash_to_eps_mean"),
        ("dividend_cash_to_ocfps", "ind_dividend_cash_to_ocfps_mean"),
        (
            "dividend_cash_yield_proxy",
            "ind_dividend_cash_yield_proxy_mean",
        ),
        ("fi_ocf_to_eps", "ind_fi_ocf_to_eps_mean"),
        ("fi_ocfps_minus_eps", "ind_fi_ocfps_minus_eps_mean"),
        ("fi_roe_quality_gap", "ind_fi_roe_quality_gap_mean"),
        ("fi_margin_quality", "ind_fi_margin_quality_mean"),
    ]
}

fn relative_industry_context_sources() -> &'static [(&'static str, &'static str)] {
    &[
        ("turnover_mean", "ind_daily_turnover"),
        ("free_turnover_mean", "ind_daily_free_turnover"),
        ("volume_ratio_mean", "ind_daily_volume_ratio"),
        ("amihud_mean", "ind_daily_amihud"),
        ("downside_amihud_mean", "ind_daily_downside_amihud"),
        ("amplitude_mean", "ind_daily_amplitude"),
        ("hit_up_limit_rate", "ind_daily_hit_up_limit_rate"),
        ("hit_down_limit_rate", "ind_daily_hit_down_limit_rate"),
    ]
}

fn industry_context_cols() -> &'static [&'static str] {
    &[
        "ind_member_count",
        "ind_daily_ret",
        "ind_excess_daily_ret",
        "ind_ret_5",
        "ind_std_5",
        "ind_excess_ret_5",
        "ind_pos_rate_5",
        "ind_dispersion_5",
        "ind_ret_20",
        "ind_std_20",
        "ind_excess_ret_20",
        "ind_pos_rate_20",
        "ind_dispersion_20",
        "ind_ret_60",
        "ind_std_60",
        "ind_excess_ret_60",
        "ind_pos_rate_60",
        "ind_dispersion_60",
        "ind_turnover_mean_20",
        "ind_free_turnover_mean_20",
        "ind_volume_ratio_mean_20",
        "ind_amihud_mean_20",
        "ind_downside_amihud_mean_20",
        "ind_amplitude_mean_20",
        "ind_hit_up_limit_rate_20",
        "ind_hit_down_limit_rate_20",
        "ind_turnover_mean_60",
        "ind_free_turnover_mean_60",
        "ind_volume_ratio_mean_60",
        "ind_amihud_mean_60",
        "ind_downside_amihud_mean_60",
        "ind_amplitude_mean_60",
        "ind_hit_up_limit_rate_60",
        "ind_hit_down_limit_rate_60",
        "ind_ep_mean",
        "ind_sp_mean",
        "ind_sp_ttm_mean",
        "ind_bp_mean",
        "ind_ep_clean_mean",
        "ind_sp_clean_mean",
        "ind_sp_ttm_clean_mean",
        "ind_bp_clean_mean",
        "ind_dividend_yield_mean",
        "ind_dividend_yield_ttm_mean",
        "ind_dividend_cash_to_eps_mean",
        "ind_dividend_cash_to_ocfps_mean",
        "ind_dividend_cash_yield_proxy_mean",
        "ind_fi_ocf_to_eps_mean",
        "ind_fi_ocfps_minus_eps_mean",
        "ind_fi_roe_quality_gap_mean",
        "ind_fi_margin_quality_mean",
    ]
}

fn source_columns() -> Vec<&'static str> {
    let mut columns = Vec::new();
    columns.extend(processed_columns());
    for (_, target) in tushare_fina_indicator_pairs() {
        columns.push(*target);
    }
    for (_, target) in tushare_dividend_pairs() {
        columns.push(*target);
    }
    for (_, target) in tushare_forecast_pairs() {
        columns.push(*target);
    }
    columns.push("fc_days_since_ann");
    for (_, target) in tushare_express_pairs() {
        columns.push(*target);
    }
    columns.push("exp_days_since_ann");
    columns.extend(industry_context_cols());
    let mut seen = BTreeSet::new();
    columns
        .into_iter()
        .filter(|name| seen.insert(*name))
        .collect()
}

fn source_numeric_columns() -> Vec<&'static str> {
    source_columns()
        .into_iter()
        .filter(|name| !matches!(*name, "date" | "symbol" | "ts_code"))
        .collect()
}

fn call_python_packed_source(workers: usize) -> Result<(), String> {
    rebuild_tushare_packed_source_from_local(workers).map(|metadata| {
        println!(
            "[collect] packed source rebuilt buckets={} symbols={} rows={}",
            metadata
                .get("bucket_ids")
                .and_then(JsonValue::as_array)
                .map(Vec::len)
                .unwrap_or(0),
            metadata
                .get("num_symbols")
                .and_then(JsonValue::as_u64)
                .unwrap_or(0),
            metadata
                .get("num_rows")
                .and_then(JsonValue::as_u64)
                .unwrap_or(0)
        );
    })
}

fn call_python_json(function_name: &str, kwargs: &[(&str, String)]) -> Result<JsonValue, String> {
    Python::attach(|python| -> PyResult<String> {
        prepare_python_path(python)?;
        let module = python.import("src.rust_collector_bridge")?;
        let py_kwargs = PyDict::new(python);
        for (key, value) in kwargs {
            match *key {
                "all" | "update" | "refresh_symbols" | "refresh_stock_list" => {
                    py_kwargs.set_item(*key, value == "true")?
                }
                _ => py_kwargs.set_item(*key, value)?,
            }
        }
        module
            .getattr(function_name)?
            .call((), Some(&py_kwargs))?
            .extract()
    })
    .map_err(|err| format!("Python {function_name} failed: {err}"))
    .and_then(|raw| {
        serde_json::from_str(&raw).map_err(|err| {
            format!("Python {function_name} returned invalid JSON: {err}; raw={raw}")
        })
    })
}

fn prepare_python_path(python: Python<'_>) -> PyResult<()> {
    let repo_root = env::current_dir()
        .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
    let conda_prefix = env::var("CONDA_PREFIX").ok();
    let site_packages_dir = conda_prefix
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root.join(".pixi/envs/default"))
        .join("lib/python3.12/site-packages");
    let sys_module = python.import("sys")?;
    sys_module
        .getattr("path")?
        .call_method1("insert", (0, repo_root.as_os_str()))?;
    sys_module
        .getattr("path")?
        .call_method1("insert", (0, site_packages_dir.as_os_str()))?;
    Ok(())
}

fn is_tushare_rate_limit(detail: &str) -> bool {
    detail.contains("每分钟最多访问该接口")
}

fn print_simple_json_or_message(
    json: bool,
    payload: &JsonValue,
    message: &str,
) -> Result<(), String> {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(payload)
                .map_err(|err| format!("failed to encode JSON: {err}"))?
        );
    } else {
        println!("[collect] {message}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_bucket_matches_python_zlib_examples() {
        assert_eq!(crc32_ieee(b"000001"), 1_312_896_368);
        assert_eq!(crc32_ieee(b"600000"), 4_011_845_371);
        assert_eq!(stable_bucket_id_rust("000001", 128), 112);
        assert_eq!(stable_bucket_id_rust("600000", 128), 123);
    }

    #[test]
    fn universe_symbol_normalization_matches_python_wrapper() {
        assert_eq!(normalize_symbol(Some("sh600000")), "600000");
        assert_eq!(normalize_symbol(Some("000001.SZ")), "000001");
        assert_eq!(normalize_symbol(Some("300750")), "300750");
    }

    #[test]
    fn universe_records_require_point_in_time_dates_by_default() {
        let mut record = BTreeMap::new();
        record.insert(
            "成分券代码".to_owned(),
            JsonValue::String("000001".to_owned()),
        );
        let err = normalize_universe_records(&[record], false).unwrap_err();
        assert!(err.contains("point-in-time membership intervals"));
    }

    #[test]
    fn universe_records_allow_explicit_static_control() {
        let mut record = BTreeMap::new();
        record.insert(
            "成分券代码".to_owned(),
            JsonValue::String("000001".to_owned()),
        );
        let rows = normalize_universe_records(&[record], true).unwrap();
        assert_eq!(
            rows,
            vec![(
                "000001".to_owned(),
                "2005-01-01".to_owned(),
                "2099-12-31".to_owned()
            )]
        );
    }

    #[test]
    fn universe_records_parse_membership_dates() {
        let mut record = BTreeMap::new();
        record.insert(
            "成分券代码".to_owned(),
            JsonValue::String("000001.SZ".to_owned()),
        );
        record.insert(
            "纳入日期".to_owned(),
            JsonValue::String("20240102".to_owned()),
        );
        record.insert(
            "剔除日期".to_owned(),
            JsonValue::String("2099-12-31".to_owned()),
        );
        let rows = normalize_universe_records(&[record], false).unwrap();
        assert_eq!(rows[0].0, "000001");
        assert_eq!(rows[0].1, "2024-01-02");
        assert_eq!(rows[0].2, "2099-12-31");
    }

    #[test]
    fn rolling_compound_return_uses_observed_window_values() {
        let values = vec![0.10, f64::NAN, -0.05, 0.02];
        let out = rolling_compound_return(&values, 3);
        assert!((out[0] - 0.10).abs() < 1e-12);
        assert!((out[2] - 0.045).abs() < 1e-12);
        assert!((out[3] - ((1.0_f64 - 0.05) * 1.02 - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn strict_sidecar_availability_uses_next_trading_day_position() {
        let trading_dates = [
            parse_date_ns("2024-01-02").unwrap(),
            parse_date_ns("2024-01-03").unwrap(),
            parse_date_ns("2024-01-05").unwrap(),
        ];
        let ann_date = parse_date_ns("2024-01-03").unwrap();
        let insert_pos = trading_dates.partition_point(|date_ns| *date_ns <= ann_date);
        assert_eq!(
            trading_dates[insert_pos],
            parse_date_ns("2024-01-05").unwrap()
        );
    }
}
