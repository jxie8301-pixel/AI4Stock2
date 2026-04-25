# AI4Stock2 Progress And TODO

## Current Status

The project has completed the runtime transition to a native pipeline.

What is true now:

- `qlib` runtime code has been removed from the active project
- `pyqlib` has been removed from project dependencies
- The main runnable workflows are:
  - `main.py`
  - `run_native_rolling.py`
  - `src/gen_feature.py`
- Native feature caches are built from Parquet source data
- `gen_feature.py` now defaults to a unified all-factor cache
- Feature profiles are now treated as subset presets over the unified cache
- The unified cache now includes a systematic temporal factor family with shared windows
- Alpha360 has been removed from the default full-factor space
- Training-time feature subset selection is supported through `features.selected_columns`
- Local experiment storage is enabled through `results/experiments/`
- Parquet factor-store migration design is documented in `docs/PARQUET_FACTOR_STORE_DESIGN.md`
- Canonical config layering is documented in `docs/CONFIG_PROFILE_ARCHITECTURE.md`
- Eastmoney fused parquet schema has been audited in `data/processed/combined/*.parquet`
- Current raw/processed parquet schemas are intended to be normalized to ASCII-only field names
- Current combined parquet columns are intended to be: `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `amount`, `amplitude`, `pct_chg`, `change`, `turnover`, `val_pct_chg`, `total_mv`, `circ_mv`, `total_share`, `circ_share`, `pe_ttm`, `pe_static`, `pb`, `peg`, `pcf`, `ps`
- The collector now preserves the full useful daily history fields from Eastmoney, including `amplitude`, `pct_chg`, and `change`
- Valuation, share-count, and `当日涨跌幅` coverage begins around `2018-01-02` in the current Eastmoney parquet sample
- The current collector downcasts `int64 -> int32` blindly; some Eastmoney share-count fields already overflow to negative values and must be fixed before direct use
- The current research default should stay on `hfq`; `qfq` should not be the default because it can leak future corporate-action adjustments into historical rows
- A first GM data path should preserve full raw endpoint fields first, then derive normalized parquet as a second step
- A first Tushare-native collector now exists under `src/collector_tushare.py`
- The Tushare path currently stores symbol cache, trade calendar, raw market tables, and a first-pass normalized `hfq` combined parquet under `data/tushare/`
- The Tushare collector now supports symbol-level incremental updates, lifecycle-aware completion checks, segmented long-history backfill, and stage cooldown after rate-limit errors
- The native training/feature pipeline now accepts `data.source: tushare` and stores its factor cache under `data/factor_store/tushare_*`
- The latest diagnosed Tushare full-factor store contains `432` factors, including `173` `TS_*` factors across flow/event, industry, dividend, valuation, quality, forecast, and express themes
- The first two relative-factor batches expand generated Tushare full-factor space to `459` factors, including `200` `TS_*` factors, after the factor store is rebuilt
- A first Tushare-specific feature family now exists, covering涨跌停结构, 自由流通占比, 自由换手, 市销率倒数, and股息率
- Tushare side-input raw stages now include `fina_indicator`, `dividend`, `forecast`, and `express`
- Tushare side-input features now include latest announced snapshots from `fina_indicator`, `dividend`, `forecast`, and `express`
- Single-factor diagnostics now support yearly regime slices, optional industry neutralization, and detailed artifact exports for bucket returns, top-bottom spreads, monthly RankIC, and missingness by year
- Semantic `TS_sem_*` factors have been added as an explicit factor-engineering layer; the latest useful subset is concentrated in dividend quality, low-volatility value reversal, value-quality, profitability resilience, and industry-strength low-volatility combinations
- The current `core_v7_semantic_alpha_strong7_v1` profile is useful as a semantic-factor test profile, but it should not be promoted to `candidate` yet because the default rolling LGBM backtest underperformed its parent feature profile
- The latest semantic diagnostics suggest these factors may be more useful as a separate linear / rank-score sleeve or overlay than as raw extra columns inside the same LGBM setup
- Absolute factor-baseline performance is now a real research constraint: `rankic_weighted_factor` and `rank_zscore_avg_factor` must be treated as hard non-ML baselines, not only as auxiliary diagnostics
- Latest full-space diagnostics show that `TS_dividend_*` and `TS_stock_vs_industry_*` retain signal best after industry neutralization
- Latest full-space diagnostics also show that most pure `TS_industry_*` features and several absolute flow/event and valuation signals are largely industry / style exposure rather than clean within-industry stock alpha
- The next factor-strength source batch now supports dividend cash payout / coverage, industry-relative dividend cash yield, relative crowding, relative liquidity-stress, and low-volatility/liquidity composite factors; packed Tushare source buckets and the factor store still need to be rebuilt before these factors are diagnosable
- Several `TS_exp_*` / `TS_latest_exp_*` features still have zero effective coverage in the current diagnostics range and should not be treated as validated alpha candidates yet
- `src/probe_tushare.py` can be used to inspect real Tushare endpoint columns and latency before integrating new tables into the formal pipeline

## Current Recommended Workflow

1. Update or refresh Tushare raw / processed parquet with `src/collector_tushare.py`
2. Generate the Tushare-backed unified factor store with `src/gen_feature.py`
3. Run rolling `lgbm` baselines with `run_native_rolling.py`
4. Run `run_single_factor_diagnostics.py` before broad model / strategy sweeps
5. Compare experiments through `results/experiments/experiment_index.csv`

Active migration note:

- `src/collector_tushare.py` is now the active research data path for the native pipeline.
- `akshare` and `gm` should remain selectable comparison / fallback sources until the shared collector utility layer is cleaned up.

## Current Technical Priorities

1. Split the research target into layered tasks instead of forcing one score to solve everything at once.
   - market / universe opportunity
   - industry relative opportunity
   - within-industry stock selection
2. Treat industry-relative opportunity as the first genuinely learnable task on the current stack.
3. Stop using portfolio-layer tuning as the main research loop until the model-side task definition is cleaner.
4. Expand the data range before broadening the feature / strategy search space.
5. Add diagnostics that separate "learned better signal" from "mapped the same signal into holdings more cleverly".

## Current Research Baselines

The project now needs four baseline concepts instead of one:

1. Feature baseline
   - `core_v4_techlite`
   - Use this as the default reference feature set for ablation and diagnostics.
2. Offensive production baseline
   - `core_v4_techlite_tushare_plus_industry_nostruct_v1` + `core_v4_lgbm_ranker_default_10x20x10`
   - Current default offensive reference:
     `top8_drop2 + score_softmax + rank_pct + intraperiod_exit(rank_pct<=0.45) + validation posrate overlay + conditional desticky(d4 @ 0.35)`
   - Candidate reference: `offensive_industry_cond_desticky_d4_t035_t8`
   - Use this as the portfolio-conversion benchmark, not as the default label-design benchmark.
3. Stable production baseline
   - `core_v4_techlite_tushare_plus_industry_nostruct_v1` + `core_v4_lgbm_ranker_default_10x20x10`
   - Current default stable reference:
     `top10_drop2 + score_softmax + rank_pct + intraperiod_exit(rank_pct<=0.45) + validation posrate overlay`
   - Candidate reference: `stable_industry_nostruct_t10`
   - Use this as the portfolio-conversion benchmark for stable shaping, not as proof that the core task is already correct.
4. Research-only defensive branch
   - `selective-price-confirm` variants remain research branches only
   - Best observed shape so far: `top10_drop2 + rank_pct exit 0.45 + pcma5_mr5`
   - This branch improved local exit quality but did not beat the stable baseline at the portfolio level, so it should not be promoted.
5. Layered-learning research baseline
   - Primary learnable task candidate:
     `industry_excess_hard_none` buyability / secondary head
   - Key references:
     `results/two_stage_excess_focused_batch_20260410_000611.tsv`
     `results/two_stage_industry_blend_tune_batch_20260410_042559.tsv`
   - Interpretation:
     the current stack appears to learn relative industry opportunity more cleanly than absolute future return magnitude.
6. Non-ML factor baseline hard gate
   - `rankic_weighted_factor` and `rank_zscore_avg_factor` are now promotion gates for future ML candidates.
   - A model that improves headline annualized return but does not beat the non-ML factor sleeves on drawdown, monthly win rate, and rebalance win rate should remain a research result, not a candidate.
   - Conservative formula style baselines should be added next: low volatility, dividend / payout, value-quality, profitability quality, and simple trend confirmation.

Important interpretation:

- Industry context and de-stickiness are now treated as support layers, not the primary research target.
- Conditional de-stickiness, overlays, and exit refinements improved portfolio conversion, but did not materially change the underlying learned signal after the `industry_nostruct` jump.
- The next meaningful leap should come from improving what the model learns, not from another round of portfolio-rule polishing.
- Current evidence suggests the most learnable signal is ordinal / relative opportunity, especially industry-relative opportunity, rather than raw future-return magnitude.

The old `top30 + equal weight + no score transform + no intraperiod exit` rolling setup
should remain available as a simple historical reference, but it is no longer the main
system baseline for feature research.

## Immediate TODO

### 0. Task Decomposition First

- [ ] Reframe the main research question from "improve one final score" to "which subtask is actually learnable?"
- [ ] Separate the problem into three explicit targets:
  - market / universe opportunity
  - industry relative opportunity
  - within-industry stock selection
- [ ] Freeze broad portfolio-rule tuning while these task definitions are being tested
- [ ] Keep cumulative return as a downstream check, not the primary gate for model-side ideas
- [ ] Promote future research results only after asking:
  - did the learned ordering improve?
  - did the selected positive-rate improve?
  - did the improvement survive regime slices?

### 0A. Factor Research Focus

- [ ] Treat full-space single-factor diagnostics plus industry-neutral diagnostics as the default gate before promoting new feature families
- [ ] Stop using "raw single-factor looks strong" as sufficient evidence; require at least:
  - acceptable coverage
  - non-trivial yearly stability
  - a readable top-bottom spread
  - a clear answer for whether the signal survives industry neutralization
- [ ] Promote three feature themes as the current primary factor-engineering track:
  - dividend and dividend-quality factors
  - stock-vs-industry relative factors
  - relative crowding / relative-liquidity factors
- [ ] Deprioritize broad expansion of:
  - pure `TS_industry_*` features
  - additional absolute valuation-level features
  - more same-family temporal / technical window variants
- [ ] Repair or quarantine zero-coverage `TS_exp_*` / `TS_latest_exp_*` fields before drawing further conclusions from the express family
- [ ] Keep feature research aimed at "what stock can make money inside its current context", not at adding more proxies for the same price-state signal

### 0B. Semantic Factors And Non-ML Baselines

- [ ] Do not promote `core_v7_semantic_alpha_strong7_v1` to `candidate` unless it beats the parent profile under the same strategy family and not only in single-factor diagnostics
- [ ] Re-test the strong semantic subset under the current candidate-style conversion family:
  - `top8` / `top10`
  - `score_softmax`
  - `rank_pct`
  - intraperiod exit / validation overlay variants only after the raw comparison is understood
- [ ] Test semantic factors as a score-level sleeve rather than only as LGBM input columns:
  - standalone `rankic_weighted` semantic sleeve
  - parent model score plus semantic sleeve overlay
  - conservative formula sleeve using low-volatility, dividend quality, value-quality, profitability quality, and trend confirmation
- [x] Add absolute baseline metrics to experiment manifests; current `*_baseline_excess_annualized_return` fields are model-minus-baseline deltas and are easy to misread
  - include baseline annualized return, volatility, Sharpe / information ratio, max drawdown, monthly win rate, and rebalance win rate in both manifest and `experiment_index.csv`
- [ ] Export monthly and rebalance-period summaries for each factor baseline, especially `rankic_weighted_factor` and `rank_zscore_avg_factor`
- [ ] Treat the non-ML factor sleeves as minimum bars for future model promotion:
  - annualized return and Sharpe must improve after costs
  - max drawdown must not deteriorate without a clear return tradeoff
  - monthly win rate and rebalance win rate must be reported directly
- [ ] Fix or quarantine `TS_sem_express_growth_quality_fresh` before using the full semantic profile; current diagnostics indicate the value path can collapse to all-zero / NaN-effective behavior
- [ ] Keep `results/` ignored; preserve reproducibility through profile/config/source changes and use copied summary artifacts only when promoting to `candidate`

### 0C. Code Quality And Reliability Cleanup

- [ ] Treat this cleanup pass as reliability / maintainability work, not performance work
- [ ] Remove research-critical silent fallback paths:
  - baseline reconstruction failures should be recorded in manifest warnings or fail in strict mode
  - opportunity-label derivation failures should not silently disable buyability diagnostics
  - missing industry mapping should be explicit when `industry_excess` or `max_industry_weight` is configured
- [ ] Extract duplicated domain helpers into shared modules:
  - forward compound return construction
  - industry group loading
  - parquet dataset scan helpers
  - profile inheritance / inline-path resolution helpers where practical
- [ ] Decide the long-term status of `main.py`:
  - either mark it as a legacy single-run wrapper
  - or refactor it to reuse the rolling runtime / evaluation / artifact helpers
- [ ] Fix semantic-factor missingness semantics:
  - avoid treating missing components as neutral zero by default
  - add minimum observed-component gates or coverage outputs for composite semantic factors
  - split invalid valuation sentinels from actual economic factor values
- [ ] Make Tushare bucket-source factor generation preconditions explicit:
  - validate that packed source shards contain required sidecar / industry context columns
  - record source-layout assumptions in factor-store metadata
- [ ] Move diagnostics script helper logic out of runnable scripts and into `src/` modules
- [ ] Split collector common parquet / lifecycle / symbol-cache utilities out of data-source-specific collectors
- [ ] Clean compatibility and dead-code surfaces after the shared helpers are in place:
  - hidden legacy CLI flags
  - wrapper-only re-exports
  - unused imports
  - duplicated artifact readers and feature-family classifiers
- [ ] Add a lightweight lint gate later, after source cleanup is complete; do not introduce broad formatting churn during the cleanup pass

Current status:

- [x] Shared forward-return, industry-group, parquet-scan, profile-resolution, and single-factor runtime helpers are extracted
- [x] Rolling evaluation now records structured warnings for non-fatal fallback paths
- [x] `main.py` is explicitly marked as the legacy single-window entrypoint
- [x] Semantic Tushare composites preserve missingness and require minimum observed component weight
- [x] Legacy valuation sentinel columns are preserved, with clean NaN-preserving variants and invalid flags added for migration
- [x] Tushare bucket-source precondition validation records sidecar/context schema assumptions in metadata
- [x] Collector common parquet write/read and numeric-dtype utilities are extracted behind compatibility wrappers
- [x] Candidate diagnostics share artifact readers, metric extraction, bucket-shape, and feature-family helpers
- [x] Config validation supported-mode constants are centralized while keeping `validate_training_config` as the public entrypoint
- [x] Backtest wrapper/report boundaries use shared native-to-legacy return helpers while keeping engine `net_return` canonical
- [x] `run_native_rolling.py` no longer re-exports underscored artifact/baseline helpers only for tests
- [ ] Hidden legacy CLI flag cleanup is deferred until compatibility aliases are no longer needed
- [x] Low-risk unused imports are removed without adding a lint dependency
- [ ] Lint gate remains deferred until code movement stabilizes

Detailed implementation plan:

1. Silent fallback cleanup
   - files: `src/rolling_evaluate.py`, `src/rolling_train.py`
   - collect baseline reconstruction failures and opportunity-label failures into structured warnings
   - expose warning records in run artifacts / manifests instead of only printing
   - keep non-critical baseline reconstruction non-fatal at first, but make configured required paths explicit
2. Industry group loading cleanup
   - files: `src/rolling_train.py`, `src/rolling_evaluate.py`, `run_single_factor_diagnostics.py`
   - add `src/industry_groups.py`
   - centralize symbol-cache path resolution, industry parquet loading, and required-vs-optional behavior
   - fail-fast when industry-neutral diagnostics are requested and groups cannot be loaded
3. Forward horizon return cleanup
   - files: `src/rolling_train.py`, `src/rolling_evaluate.py`, `run_single_factor_diagnostics.py`
   - add `src/return_horizon.py`
   - centralize `t+1 ... t+horizon` compound-return construction
   - add tests for short input, NaN windows, index sorting, and horizon-one behavior
4. Parquet dataset scan cleanup
   - files: `src/factor_store.py`, `src/source_store.py`
   - add `src/parquet_io.py`
   - move shared Arrow dataset scan logic without changing scan parameters
   - keep layout and manifest-specific decisions in factor/source store modules
5. Profile resolution cleanup
   - files: `src/feature_profiles.py`, `src/experiment_profiles.py`
   - add a small shared profile helper only for inline path / inheritance mechanics
   - leave feature-specific add/drop/repeat mutation local in `feature_profiles.py`
6. `main.py` status cleanup
   - files: `main.py`, `src/rolling_evaluate.py`
   - first mark `main.py` as a legacy single-run entrypoint
   - then extract common backtest kwargs / plot-report helpers so single and rolling paths do not drift
7. Semantic factor missingness cleanup
   - file: `src/feature_value_core.py`
   - replace missing-as-neutral-zero composite components with explicit observed-component gating
   - avoid adding new feature columns in the first pass unless tests show the coverage signal is necessary
8. Valuation sentinel cleanup
   - files: `src/feature_value_core.py`, `src/gen_feature.py`
   - keep legacy `-1.0` columns initially for reproducibility
   - add explicit invalid flags and clean NaN-preserving valuation variants before moving profiles to the clean variants
9. Tushare bucket-source precondition cleanup
   - file: `src/gen_feature.py`
   - validate packed source schema before bucket-source factor generation
   - record source-layout and sidecar assumptions in factor-store metadata
10. Single-factor diagnostics runtime cleanup
    - files: `run_single_factor_diagnostics.py`, `run_single_factor_diagnostics_batch.py`
    - move period/segment/label-space/industry-neutral helpers into a `src/` module
    - stop importing private helper functions from runnable scripts
11. Candidate diagnostics shared utilities
    - files: `run_candidate_pool_diagnostics.py`, `run_strategy_pair_diagnostics.py`, `src/candidate_profiles.py`
    - centralize artifact readers, metric extraction, bucket shape, and feature-family classification
12. Collector common parquet utilities
    - files: `src/collector_tushare.py`, `src/collector_gm.py`, `src/collector_akshare.py`
    - first extract only schema-neutral parquet / atomic-write helpers
    - defer lifecycle and symbol-cache extraction until the low-level utility layer is stable
13. Config validation split
    - file: `src/config_validation.py`
    - first centralize supported-mode constants
    - later split domain validators while preserving `validate_training_config` as the public mutating entrypoint
14. Backtest wrapper boundary cleanup
    - files: `src/backtest_engine.py`, `src/native_backtest.py`, `src/backtest.py`, `main.py`, `src/rolling_evaluate.py`
    - keep wrappers for compatibility
    - keep canonical engine return column as `net_return`
    - restrict legacy `return` aliasing to wrapper/report-preparation boundaries
15. Legacy CLI / compatibility import cleanup
    - files: `run_native_rolling.py`, `src/runtime_cli.py`, tests
    - migrate tests away from underscored `run_native_rolling.py` aliases
    - keep hidden legacy flags until test imports and documented commands no longer depend on them
16. Unused import cleanup
    - clean low-risk imports after shared helpers land
    - do not run broad auto-formatting in this pass
17. Lint gate
    - defer dependency changes
    - add `ruff` later only after code movement stabilizes
    - start with unused import / broad exception / import sorting checks before considering format enforcement

### 1. Relative-Opportunity Research Track

- [ ] Promote `industry_excess` style labels from side experiment to first-class research track
- [ ] Build a formal label family for relative opportunity:
  - `ret_20d > industry_ret_20d`
  - `ret_20d > benchmark_ret_20d`
  - `ret_20d > 0`
  - `ret_20d > cost_buffer`
- [ ] Explicitly compare which task is most learnable using:
  - selected positive-rate
  - opportunity-rate monotonicity
  - return monotonicity
  - bucket spread
- [ ] Treat raw future-return magnitude as optional / secondary until it proves learnable on the same stack
- [ ] Add a within-industry evaluation view so we can tell whether the model is selecting the right industry, the right stock inside an industry, or both

### 2. Two-Layer Modeling Direction

- [ ] Build a research prototype where the primary target is no longer forced to be absolute return ranking
- [ ] Test a layered scoring structure:
  - layer A: industry / relative-opportunity score
  - layer B: within-industry stock-selection score
  - layer C: optional buyability / tail-risk gate
- [ ] Compare layered scoring against the current single-score mainline on the same dates and universe
- [ ] Record whether improvements come from:
  - better industry allocation
  - better stock selection inside chosen industries
  - better avoidance of no-opportunity windows
- [ ] Do not optimize blend weights first; optimize the task split first

### 3. Diagnostics That Distinguish Learning From Mapping

- [ ] Add diagnostics for whether the model actually learned something useful:
  - score bucket monotonicity for realized return
  - score bucket monotonicity for positive-rate
  - calibration of predicted buyability vs realized buyability
  - regime-sliced diagnostics by year / quarter / bear windows
  - industry-level hit-rate diagnostics for selected names
  - between-industry vs within-industry attribution
- [ ] Add "same signal, different strategy" comparison views so portfolio-layer changes cannot masquerade as alpha improvements
- [ ] Add explicit flags in summaries for:
  - model-side change
  - feature-side change
  - portfolio-conversion-side change

### 4. Regime Memory And Data Range

- [ ] Run controlled memory-length experiments on the current stable and offensive baselines:
  - `train_days = 242`
  - `train_days = 360`
  - `train_days = 480`
- [ ] Compare weighting schemes that keep long-window context while emphasizing recent data
  - current half-life weighting
  - generalized exp-with-floor weighting
  - piecewise / smooth weighting with stronger recent emphasis but non-trivial remote weight
- [ ] Expand the evaluation range to cover more independent regimes before widening universe
  Preferred next target: extend analysis to at least `2018-2025`, ideally `2016-2025` if data coverage is acceptable
- [ ] Treat global backtest range and rolling train window as separate levers; do not conflate them
- [ ] Add regime-level summaries so weak-signal / weak-opportunity periods can be studied directly instead of inferred from the final equity curve
- [ ] Only revisit wider-universe research after the longer-horizon regime study is stable enough to compare task learnability

### 5. Mixed Feature Representation

- [ ] Add a mixed feature-transform path instead of ranking every feature column by default
- [ ] Keep cross-sectional-rank versions for stock-selection features that benefit from relative comparison
- [ ] Preserve raw or lightly normalized absolute versions for regime-sensitive inputs such as:
  - benchmark trend / drawdown / volatility
  - market breadth
  - industry absolute trend / excess return
  - selected stock-level absolute scale features
- [ ] Compare full-rank vs mixed-rank/raw pipelines on the same baselines
- [ ] Make sure experiment manifests record which features stayed absolute and which were ranked

### 5A. Factor-Store Layout And WSL2 I/O

- [ ] Keep the current `bucket_shards` path as the stable baseline, but stop assuming it is the final training-optimal layout
- [ ] Add a second training-oriented factor-store layout for evaluation:
  - date-major or year-partitioned panel layout
  - explicitly tuned Parquet row-group size
  - same logical columns and metadata contract as the current store
- [ ] Separate incremental-update storage from training-read storage instead of forcing one layout to solve both jobs
- [ ] Verify whether date-window pruning actually works on the training store; the current bucket files should not remain a single row group if we expect date filters to save I/O
- [ ] Re-check universe filtering cost on the current factor store and remove redundant Python-side masking when Arrow-level symbol filtering is already sufficient
- [ ] Add one reproducible local benchmark that compares:
  - current `bucket_shards`
  - proposed date-major training layout
  - representative read patterns: `all`, `csi300`, short/long date windows, 1-column vs 40-column loads
- [ ] Only promote a new layout after verifying end-to-end rolling runtime, not just synthetic file-read speed

### 6. Tushare Migration And Pipeline Cleanup

- [x] Finalize the first canonical Tushare market-data normalized schema for `combined` parquet
- [x] Wire the Tushare collector into the formal native feature/training workflow as an optional `data.source`
- [~] Add Tushare financial/event raw tables to the formal collector path: `income`, `balancesheet`, `cashflow`, `fina_indicator`, `forecast`, `express`, `dividend`, `fina_audit`, `fina_mainbz`
  Current state: `fina_indicator`, `dividend`, `forecast`, `express` are already wired; next batch should be `income`, `balancesheet`, `cashflow`, then `fina_audit`, `fina_mainbz`
- [ ] Design the second Tushare factor layer around financial/event tables instead of only daily market tables
- [x] Finish a full `--stages all` Tushare backfill to a stable converged state and verify repeated reruns are purely incremental for the currently landed stage set
- [x] Build the Tushare factor store after raw convergence: `uv run python -m src.gen_feature --data-source tushare --workers 16 --incremental`
- [x] Run the first Tushare-native rolling baseline before adding more tables
- [ ] Deduplicate `main.py` and `run_native_rolling.py` into shared training/prediction/evaluation helpers before continuing to add more run modes
- [ ] Refactor `gen_feature.py` into smaller modules: factor definitions, label builder, and factor-store builder
- [ ] Extract collector-common parquet/lifecycle/update helpers so `collector_akshare.py`, `collector_gm.py`, and `collector_tushare.py` stop diverging
- [ ] Upgrade training-time transforms from ad-hoc functions to a real fit/transform pipeline before adding more transform combinations

### 7. LightGBM Feature Research

- [x] Add a native `lgbm_purified_v1` feature profile inspired by the strongest old-project factors
- [x] Add native support for valuation/style factors such as `ep_ttm`, `bp`, `log_mcap`, `is_loss`
- [x] Add liquidity and microstructure factors such as `amihud_20`, `turnover_20`, `vwap_ratio`
- [x] Add a first-pass unified temporal factor family with systematic windows
- [ ] Compare `alpha158_compact_v1` vs `alpha158_full` vs `lgbm_purified_v1`
- [x] Remove `A360_*` from the default full-factor space
- [ ] Add causal relative-strength / residual factors versus market and universe benchmark
- [ ] Add richer volatility-shape factors such as downside vol, range vol, gap shock, skew, kurtosis
- [ ] Add richer amount/turnover flow factors such as amount shock, turnover z-score, signed flow persistence
- [ ] Expand valuation/share-based factors from Eastmoney parquet fields: `ps`, `pcf`, `peg`, `circ_share`, `total_share`, float ratio
- [ ] Decide whether and how to use the retained Eastmoney daily fields such as `amplitude`, `pct_chg`, and `change`
- [ ] Expand Tushare-only factors from already-landed market columns: limit-band width/position dynamics, free-float turnover shocks, share-structure drift, and valuation/dividend change factors
- [ ] Evaluate whether the newly added Tushare event-side features (`fina_indicator`, `dividend`, `forecast`, `express`) actually improve the rolling baseline before expanding to more statement tables
- [x] Add a rolling single-factor diagnostics report: IC, RankIC, coverage, monotonicity, stability
- [x] Extend single-factor diagnostics with yearly slices, industry-neutral mode, and detailed daily / monthly artifacts
- [~] Add automated prefiltering by minimum coverage plus minimum rolling IC / RankIC threshold
  Current state: standalone diagnostics-based prefilter tooling now exists; it still needs promotion into a repeatable experiment workflow and selection policy.
- [~] Add redundancy pruning on the selected feature set using correlation clustering before model training
  Current state: standalone greedy correlation-pruning tooling now exists for diagnostics-selected candidates; it is not yet integrated into the training path.
- [ ] Separate factor research into three layers: raw factor generation, stable profile curation, optional training-time auto-filter
- [ ] Add a formal promotion policy for new factors based on:
  - raw diagnostics
  - industry-neutral diagnostics
  - yearly stability
  - coverage / missingness
  - redundancy against already-promoted factors
- [~] Build the next priority factor batch around stock-vs-industry relative structure instead of adding more absolute state features
  Current state: first code batch is implemented for `20` / `60` day relative turnover, free-turnover, volume-ratio, Amihud, downside-Amihud, amplitude, and limit-hit deviation. The second source batch adds relative crowding, relative liquidity-stress, and low-volatility/liquidity composites; packed source and factor store still need to be rebuilt and diagnosed.
  First batch should target:
  - stock-vs-industry turnover ratio
  - stock-vs-industry free-turnover ratio
  - stock-vs-industry amihud ratio
  - stock-vs-industry volume-ratio deviation
  - stock-vs-industry limit-hit / crowding deviation
  Implementation design:
  - extend the Tushare industry-context cache first; the current cache only has industry return, volatility, positive-rate, and dispersion fields
  - add industry-context columns for each selected window, initially `20` and `60`:
    - `ind_turnover_mean_{w}`
    - `ind_free_turnover_mean_{w}`
    - `ind_volume_ratio_mean_{w}`
    - `ind_amihud_mean_{w}`
    - `ind_downside_amihud_mean_{w}`
    - `ind_amplitude_mean_{w}`
    - `ind_hit_up_limit_rate_{w}`
    - `ind_hit_down_limit_rate_{w}`
  - then add stock-level relative factor outputs:
    - `stock_vs_industry_turnover_ratio_{w}`
    - `stock_vs_industry_free_turnover_ratio_{w}`
    - `stock_vs_industry_volume_ratio_gap_{w}`
    - `stock_vs_industry_amihud_ratio_{w}`
    - `stock_vs_industry_downside_amihud_ratio_{w}`
    - `stock_vs_industry_amplitude_ratio_{w}`
    - `stock_vs_industry_hit_up_limit_gap_{w}`
    - `stock_vs_industry_hit_down_limit_gap_{w}`
  - use ratio form for strictly positive intensity variables and gap form for rate / bounded variables
  - after changing industry-context columns, rebuild packed Tushare source buckets before rebuilding the factor store; otherwise the bucket-source path will keep old `ind_*` columns
  - validate this batch using full-space raw diagnostics and full-space industry-neutral diagnostics before adding any model sweep
- [~] Build the next priority dividend-quality batch instead of only using dividend yield level
  Current state: first feasible source batch is implemented for dividend cash payout / coverage, industry-relative dividend cash yield, dividend cash yield surprise, spread z-scores, and a conservative dividend cash-quality composite; packed source and factor store still need to be rebuilt and diagnosed.
  First batch should target:
  - dividend-to-OCF
  - dividend-to-net-profit
  - dividend consistency / stability
  - dividend cut / resume flags
  - multi-year dividend growth
  Implementation design:
  - current sidecar data has `dividend` plus latest `fina_indicator`, but it does not yet preserve enough multi-year dividend history in factor values
  - first add only factors that are feasible from latest snapshot fields:
    - `dividend_cash_to_eps`
    - `dividend_cash_to_ocfps`
    - `dividend_cash_yield_proxy`
    - `dividend_yield_minus_industry_20`
  - defer true multi-year consistency / cut / resume factors until dividend event history is carried as an event-series feature instead of only a latest snapshot
- [~] Replace absolute valuation emphasis with industry-relative valuation factors
  Current state: first code batch is implemented for industry-relative `ep`, `sp`, `sp_ttm`, `bp`, `dividend_yield`, and `dividend_yield_ttm`; packed source and factor store still need to be rebuilt and diagnosed.
  First batch should target:
  - `ep - industry_ep`
  - `sp - industry_sp`
  - `bp - industry_bp`
  - valuation change minus industry valuation change
  - dividend yield minus industry dividend yield
  Implementation design:
  - extend industry context with industry cross-sectional means / medians for valuation variables:
    - `ind_ep_mean`
    - `ind_sp_mean`
    - `ind_sp_ttm_mean`
    - `ind_bp_mean`
    - `ind_dividend_yield_mean`
    - `ind_dividend_yield_ttm_mean`
  - add relative valuation outputs:
    - `ep_minus_industry_ep`
    - `sp_minus_industry_sp`
    - `sp_ttm_minus_industry_sp_ttm`
    - `bp_minus_industry_bp`
    - `dividend_yield_minus_industry`
    - `dividend_yield_ttm_minus_industry`
  - add change-relative variants only after the level-relative variants pass diagnostics
- [~] Expand financial-quality factors beyond simple first-order ratios
  Current state: first feasible code batch is implemented for `fi_ocfps_minus_eps` and industry-relative `fi_ocf_to_eps`, `fi_ocfps_minus_eps`, `fi_roe_quality_gap`, and `fi_margin_quality`; statement-table-dependent factors are still deferred.
  First batch should target:
  - accrual ratio
  - receivable growth minus revenue growth
  - inventory growth minus revenue growth
  - cash-conversion quality
  - margin / ROE stability
  Implementation design:
  - this requires statement tables beyond the currently wired `fina_indicator` snapshot
  - do not implement accrual / receivable / inventory features until `income`, `balancesheet`, and `cashflow` are part of the formal Tushare source path
  - near-term feasible additions from current fields are limited to:
    - `ocfps_minus_eps`
    - `roe_dt_minus_roe`
    - `gross_margin_minus_net_margin`
    - industry-relative versions of those three
- [ ] Rework forecast / express factors around surprise structure rather than raw latest snapshots
  First batch should target:
  - forecast midpoint vs last actual
  - forecast width / confidence
  - forecast revision acceleration
  - forecast / express surprise relative to industry
  - post-event drift features
  Implementation design:
  - first fix `TS_exp_*` coverage gaps; several express-derived fields currently have zero effective coverage
  - add diagnostic coverage checks before adding more express formulas
  - first feasible forecast additions from current fields:
    - `forecast_mid_vs_last_parent_net`
    - `forecast_width_to_abs_mid`
    - `forecast_positive_confidence_decay`
    - `forecast_days_since_ann_decay`
  - defer revision acceleration and post-event drift until event history is available, not just latest as-of snapshots
- [ ] Fix the current `TS_exp_*` coverage failures before expanding the express family further
- [ ] Avoid broad new `TEMP_*` / `TECH_*` feature expansion until the relative / dividend / quality themes above are tested
- [ ] Add market / universe / industry opportunity features rather than only stock-level predictors
  First batch should target:
  - universe positive-rate / breadth proxies
  - benchmark trend and volatility regime
  - industry dispersion and rotation-intensity signals
  - industry-vs-benchmark absolute opportunity signals

### 8. Training-Time Transforms

- [x] Add optional daily cross-sectional rank transform before model training for LightGBM
- [ ] Add optional feature winsorization / clipping transform in the training path
- [ ] Add optional label de-meaning for LightGBM experiments
- [ ] Record applied transforms in experiment manifests
- [ ] Add optional cross-sectional z-score transform and make it composable with rank / clipping
- [ ] Add an optional training-time feature decorrelation path based on correlation pruning; avoid PCA as the default path
- [ ] Add a mixed transform policy so selected columns can bypass cross-sectional rank
- [ ] Support transform policies by feature group instead of only one global toggle
- [ ] Add label transforms targeted at buyability / positive-rate modeling instead of only ranking-friendly transforms

### 9. Native Data Quality

- [x] Stand up a GM-native collector that stores full raw endpoint fields under `data/gm/raw/` before any schema reduction
- [x] Add GM lifecycle-aware precheck with `effective_end_date` so delisted symbols do not stay pending forever
- [x] Harden collector parquet writes with atomic replace and safe reads for broken/zero-byte files
- [ ] Define the canonical GM-to-native mapping for `circ_mv`, `circ_share`, `pb`, `pcf`, and `ps`
- [ ] Compare GM vs Eastmoney coverage, freshness, and field stability over the same sample
- [ ] Add a feature coverage report during cache generation
- [ ] Add per-feature NaN / inf diagnostics to `meta.json`
- [ ] Add a cache validation command for shape, names, coverage, and label sanity
- [ ] Review whether valuation fields are complete enough across the full sample
- [ ] Reduce full-factor cache footprint without changing the current training read path
- [ ] Fix collector dtype downcast so Eastmoney share-count fields do not overflow when written to parquet
- [x] Audit Eastmoney fused parquet schema and confirm the current combined columns retained by the collector
- [x] Normalize retained collector fields to ASCII names before factor generation
- [x] Document the default price-adjustment choice; keep `hfq` in the current research path and avoid `qfq` leakage
- [ ] Port lifecycle/effective-end-date handling to the Eastmoney collector path
- [ ] Add collector performance telemetry: request count, empty-return count, write time, throughput by stage
- [ ] Split collector common helpers into shared parquet/lifecycle utilities instead of duplicating logic in both collectors

### 10. Backtest Realism

- [ ] Add explicit tradability flags for suspension / invalid rows where possible
- [ ] Evaluate whether limit-up / limit-down blocking should be modeled in native backtest
- [ ] Add higher-slippage sensitivity experiments
- [~] Add risk control experiments for lower turnover and lower drawdown
  Current state: signal-strength and benchmark-aware risk control modes are implemented and already in the current shortlist; realism and attribution work still remains.
- [ ] Add embargo / gap controls between train, valid, and test windows to reduce boundary leakage in rolling runs

### 11. Strategy Layer

- [x] Add score-weighted portfolio construction instead of pure equal weight
- [x] Promote the shortlist winner set into the current production baseline definition
- [ ] Add sector / style exposure diagnostics
- [ ] Add market-regime comparison for rebalance frequency
- [ ] Compare `topk` / `n_drop` combinations systematically
- [ ] Add a volatility-aware position sizing baseline before considering full mean-variance optimization
- [ ] Add lightweight attribution: benchmark beta, size/value proxy exposure, and turnover decomposition
- [ ] Keep strategy experiments subordinate to model-learning questions
  Promotion rule:
  - if the change only reshapes holdings but does not improve learning diagnostics, treat it as secondary
  - if the change improves learning diagnostics but needs portfolio help, keep it in the alpha pipeline
- [ ] Avoid broad new sweeps in this section until the layered-task roadmap above has a first answer

### 12. Native Model Roadmap

- [ ] Decide whether native LSTM should remain supported as a secondary path
- [ ] If sequence models remain in scope, build a real native Transformer implementation
- [ ] Add a unified save/load contract shared by all native models
- [ ] Add CatBoost as the first non-LGBM tabular baseline
- [ ] Add a simple linear baseline such as Ridge / ElasticNet for signal sanity checks
- [~] Evaluate whether rank objectives should be added before broadening to more tree models
  Current state: ranking losses are already wired into `NativeLGBM`; what remains is systematic head-to-head experimentation and promotion criteria.
- [~] Add a LightGBM ranking experiment path grouped by trade date and compare against regression on the same profiles
  Current state: grouped ranking support exists in the model wrapper and config profiles; a formal research sweep and result comparison are still pending.
- [ ] Add model-profile level control for objective family, evaluation metric, and regularization regime
- [ ] Defer AutoEncoder / GNN / multi-task sequence models until tabular baselines and ranking objectives are exhausted
- [ ] Add a first explicit buyability / profit-probability objective path before adding more model families
- [ ] Compare ranking-only vs buyability-only vs two-stage ranking+buyability on the same feature/profile stack
- [ ] Add model-side promotion criteria based on calibration, monotonicity, and regime robustness, not only IC and backtest return

### 13. Tooling And UX

- [x] Split feature-set definitions into dedicated files under `configs/features/`
- [x] Upgrade model presets into first-class model profiles under `configs/models/`
- [x] Add experiment-level config composition so a run can reference named feature/model/experiment profiles
- [x] Add automatic LightGBM feature-importance export for single and rolling runs
- [ ] Make native model saving opt-out rather than opt-in where practical
- [ ] Add a script to compare runs directly from `experiment_index.csv`
- [ ] Add a script to summarize the best run per model/profile/tag
- [ ] Add richer manifest metadata for selected features and transforms
- [ ] Improve README with a short native quickstart
- [ ] Unify canonical return naming across backtest/evaluation (`gross`, `net`, `cost`, `bench`) and keep compatibility only at wrapper boundaries

## Research Priority Recommendation

If only one direction should be pursued next, it should be:

1. native LightGBM
2. layered task decomposition: market / industry / stock
3. explicit relative-opportunity modeling before raw magnitude modeling
4. longer regime coverage plus mixed absolute/ranked feature representation
5. diagnostics that verify the model learned what can make money
6. only then more feature expansion and portfolio-layer tuning

This is more likely to produce a model that truly knows what can make money than
another round of backtest-driven strategy polishing, blend-weight tuning, or another deep
model right now.
