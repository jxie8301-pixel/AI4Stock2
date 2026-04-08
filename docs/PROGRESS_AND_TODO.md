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
- A first Tushare-specific feature family now exists, covering涨跌停结构, 自由流通占比, 自由换手, 市销率倒数, and股息率
- Tushare side-input raw stages now include `fina_indicator`, `dividend`, `forecast`, and `express`
- Tushare side-input features now include latest announced snapshots from `fina_indicator`, `dividend`, `forecast`, and `express`
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

1. Shift research toward learning absolute profit opportunity, not only cross-sectional ranking.
2. Preserve more regime information in the feature pipeline instead of fully collapsing everything into daily rank space.
3. Extend rolling-memory experiments (`train_days`, weighting function) before widening the research universe.
4. Add model diagnostics that answer whether the model truly learned what can make money: probability calibration, score monotonicity, regime slices, and industry hit-rate.
5. Keep pipeline cleanup and backtest realism work moving, but stop treating portfolio-layer tuning as the main alpha research path.

## Current Research Baselines

The project now needs three baseline concepts instead of one:

1. Feature baseline
   - `core_v4_techlite`
   - Use this as the default reference feature set for ablation and diagnostics.
2. Offensive production baseline
   - `core_v4_techlite_tushare_plus_industry_nostruct_v1` + `core_v4_lgbm_ranker_default_10x20x10`
   - Current default offensive reference:
     `top8_drop2 + score_softmax + rank_pct + intraperiod_exit(rank_pct<=0.45) + validation posrate overlay + conditional desticky(d4 @ 0.35)`
   - Candidate reference: `offensive_industry_cond_desticky_d4_t035_t8`
   - Use this as the main offensive benchmark for opportunity-modeling, training-memory, and mixed absolute/ranked feature research.
3. Stable production baseline
   - `core_v4_techlite_tushare_plus_industry_nostruct_v1` + `core_v4_lgbm_ranker_default_10x20x10`
   - Current default stable reference:
     `top10_drop2 + score_softmax + rank_pct + intraperiod_exit(rank_pct<=0.45) + validation posrate overlay`
   - Candidate reference: `stable_industry_nostruct_t10`
   - Use this as the primary benchmark for new objectives, absolute-opportunity filtering, and regime-adaptation experiments.
4. Research-only defensive branch
   - `selective-price-confirm` variants remain research branches only
   - Best observed shape so far: `top10_drop2 + rank_pct exit 0.45 + pcma5_mr5`
   - This branch improved local exit quality but did not beat the stable baseline at the portfolio level, so it should not be promoted.

Important interpretation:

- Industry context and de-stickiness are now treated as support layers, not the primary research target.
- The next meaningful leap should come from improving what the model learns, not from another round of portfolio-rule polishing.

The old `top30 + equal weight + no score transform + no intraperiod exit` rolling setup
should remain available as a simple historical reference, but it is no longer the main
system baseline for feature research.

## Immediate TODO

### 0. Alpha-First Research Direction

- [ ] Add a formal "absolute opportunity" research track alongside the current ranker track
- [ ] Add training labels that directly answer whether a stock is worth buying over the holding horizon, not only how it ranks cross-sectionally
  First batch should include:
  - `ret_20d > 0`
  - `ret_20d > cost_buffer`
  - `ret_20d > benchmark_ret_20d`
  - `ret_20d > industry_ret_20d`
- [ ] Build a simple two-stage scoring prototype:
  - stage A: current ranker score
  - stage B: profit-probability or buyability score
  - merge rule: probability filter and/or `rank_score * profit_prob`
- [ ] Add diagnostics for whether the model actually learned something useful:
  - score bucket monotonicity for realized return
  - score bucket monotonicity for positive-rate
  - calibration of predicted buyability vs realized buyability
  - regime-sliced diagnostics by year / quarter / bear windows
  - industry-level hit-rate diagnostics for selected names
- [ ] Stop using cumulative return as the primary promotion criterion for new model-side ideas
- [ ] Keep candidate promotion portfolio-first, but keep alpha research model-first

### 1. Regime Memory And Absolute Context

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

### 2. Mixed Feature Representation

- [ ] Add a mixed feature-transform path instead of ranking every feature column by default
- [ ] Keep cross-sectional-rank versions for stock-selection features that benefit from relative comparison
- [ ] Preserve raw or lightly normalized absolute versions for regime-sensitive inputs such as:
  - benchmark trend / drawdown / volatility
  - market breadth
  - industry absolute trend / excess return
  - selected stock-level absolute scale features
- [ ] Compare full-rank vs mixed-rank/raw pipelines on the same baselines
- [ ] Make sure experiment manifests record which features stayed absolute and which were ranked

### 3. Tushare Migration And Pipeline Cleanup

- [x] Finalize the first canonical Tushare market-data normalized schema for `combined` parquet
- [x] Wire the Tushare collector into the formal native feature/training workflow as an optional `data.source`
- [~] Add Tushare financial/event raw tables to the formal collector path: `income`, `balancesheet`, `cashflow`, `fina_indicator`, `forecast`, `express`, `dividend`, `fina_audit`, `fina_mainbz`
  Current state: `fina_indicator`, `dividend`, `forecast`, `express` are already wired; next batch should be `income`, `balancesheet`, `cashflow`, then `fina_audit`, `fina_mainbz`
- [ ] Design the second Tushare factor layer around financial/event tables instead of only daily market tables
- [x] Finish a full `--stages all` Tushare backfill to a stable converged state and verify repeated reruns are purely incremental for the currently landed stage set
- [x] Build the Tushare factor store after raw convergence: `uv run python src/gen_feature.py --data-source tushare --workers 16 --incremental`
- [x] Run the first Tushare-native rolling baseline before adding more tables
- [ ] Deduplicate `main.py` and `run_native_rolling.py` into shared training/prediction/evaluation helpers before continuing to add more run modes
- [ ] Refactor `gen_feature.py` into smaller modules: factor definitions, label builder, and factor-store builder
- [ ] Extract collector-common parquet/lifecycle/update helpers so `collector_akshare.py`, `collector_gm.py`, and `collector_tushare.py` stop diverging
- [ ] Upgrade training-time transforms from ad-hoc functions to a real fit/transform pipeline before adding more transform combinations

### 4. LightGBM Feature Research

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
- [~] Add automated prefiltering by minimum coverage plus minimum rolling IC / RankIC threshold
  Current state: standalone diagnostics-based prefilter tooling now exists; it still needs promotion into a repeatable experiment workflow and selection policy.
- [~] Add redundancy pruning on the selected feature set using correlation clustering before model training
  Current state: standalone greedy correlation-pruning tooling now exists for diagnostics-selected candidates; it is not yet integrated into the training path.
- [ ] Separate factor research into three layers: raw factor generation, stable profile curation, optional training-time auto-filter
- [ ] Add market / universe / industry opportunity features rather than only stock-level predictors
  First batch should target:
  - universe positive-rate / breadth proxies
  - benchmark trend and volatility regime
  - industry dispersion and rotation-intensity signals
  - industry-vs-benchmark absolute opportunity signals

### 5. Training-Time Transforms

- [x] Add optional daily cross-sectional rank transform before model training for LightGBM
- [ ] Add optional feature winsorization / clipping transform in the training path
- [ ] Add optional label de-meaning for LightGBM experiments
- [ ] Record applied transforms in experiment manifests
- [ ] Add optional cross-sectional z-score transform and make it composable with rank / clipping
- [ ] Add an optional training-time feature decorrelation path based on correlation pruning; avoid PCA as the default path
- [ ] Add a mixed transform policy so selected columns can bypass cross-sectional rank
- [ ] Support transform policies by feature group instead of only one global toggle
- [ ] Add label transforms targeted at buyability / positive-rate modeling instead of only ranking-friendly transforms

### 6. Native Data Quality

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

### 7. Backtest Realism

- [ ] Add explicit tradability flags for suspension / invalid rows where possible
- [ ] Evaluate whether limit-up / limit-down blocking should be modeled in native backtest
- [ ] Add higher-slippage sensitivity experiments
- [~] Add risk control experiments for lower turnover and lower drawdown
  Current state: signal-strength and benchmark-aware risk control modes are implemented and already in the current shortlist; realism and attribution work still remains.
- [ ] Add embargo / gap controls between train, valid, and test windows to reduce boundary leakage in rolling runs

### 8. Strategy Layer

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

### 9. Native Model Roadmap

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

### 10. Tooling And UX

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
2. explicit absolute-opportunity modeling on top of the current ranker baseline
3. mixed absolute/ranked feature representation plus longer rolling memory
4. diagnostics that verify the model learned what can make money
5. only then more feature expansion and portfolio-layer tuning

This is more likely to produce a model that truly knows what can make money than
another round of backtest-driven strategy polishing or another deep model right now.
