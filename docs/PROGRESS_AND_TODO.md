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
- `src/probe_tushare.py` can be used to inspect real Tushare endpoint columns and latency before integrating new tables into the formal pipeline

## Current Recommended Workflow

1. Update or refresh Parquet data with `src/collector_akshare.py`
2. Generate the unified all-factor cache once
3. Train `lgbm` on rolling windows with `run_native_rolling.py`
4. Compare experiments through `results/experiments/experiment_index.csv`

Active migration note:

- `src/collector_tushare.py` is already wired into the formal native pipeline as an optional `data.source`.
- It should remain a selectable source until financial/event tables and the next feature layer are added.

## Immediate TODO

### 0. Tushare Migration And Pipeline Cleanup

- [x] Finalize the first canonical Tushare market-data normalized schema for `combined` parquet
- [x] Wire the Tushare collector into the formal native feature/training workflow as an optional `data.source`
- [ ] Add Tushare financial/event raw tables to the formal collector path: `income`, `balancesheet`, `cashflow`, `fina_indicator`, `forecast`, `express`, `dividend`, `fina_audit`, `fina_mainbz`
- [ ] Design the second Tushare factor layer around financial/event tables instead of only daily market tables
- [ ] Deduplicate `main.py` and `run_native_rolling.py` into shared training/prediction/evaluation helpers before continuing to add more run modes
- [ ] Refactor `gen_feature.py` into smaller modules: factor definitions, label builder, and factor-store builder
- [ ] Extract collector-common parquet/lifecycle/update helpers so `collector_akshare.py`, `collector_gm.py`, and `collector_tushare.py` stop diverging
- [ ] Upgrade training-time transforms from ad-hoc functions to a real fit/transform pipeline before adding more transform combinations

### 1. LightGBM Feature Research

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
- [ ] Add a rolling single-factor diagnostics report: IC, RankIC, coverage, monotonicity, stability
- [ ] Add automated prefiltering by minimum coverage plus minimum rolling IC / RankIC threshold
- [ ] Add redundancy pruning on the selected feature set using correlation clustering before model training
- [ ] Separate factor research into three layers: raw factor generation, stable profile curation, optional training-time auto-filter

### 2. Training-Time Transforms

- [x] Add optional daily cross-sectional rank transform before model training for LightGBM
- [ ] Add optional feature winsorization / clipping transform in the training path
- [ ] Add optional label de-meaning for LightGBM experiments
- [ ] Record applied transforms in experiment manifests
- [ ] Add optional cross-sectional z-score transform and make it composable with rank / clipping
- [ ] Add an optional training-time feature decorrelation path based on correlation pruning; avoid PCA as the default path

### 3. Native Data Quality

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

### 4. Backtest Realism

- [ ] Add explicit tradability flags for suspension / invalid rows where possible
- [ ] Evaluate whether limit-up / limit-down blocking should be modeled in native backtest
- [ ] Add higher-slippage sensitivity experiments
- [ ] Add risk control experiments for lower turnover and lower drawdown
- [ ] Add embargo / gap controls between train, valid, and test windows to reduce boundary leakage in rolling runs

### 5. Strategy Layer

- [ ] Add score-weighted portfolio construction instead of pure equal weight
- [ ] Add sector / style exposure diagnostics
- [ ] Add market-regime comparison for rebalance frequency
- [ ] Compare `topk` / `n_drop` combinations systematically
- [ ] Add a volatility-aware position sizing baseline before considering full mean-variance optimization
- [ ] Add lightweight attribution: benchmark beta, size/value proxy exposure, and turnover decomposition

### 6. Native Model Roadmap

- [ ] Decide whether native LSTM should remain supported as a secondary path
- [ ] If sequence models remain in scope, build a real native Transformer implementation
- [ ] Add a unified save/load contract shared by all native models
- [ ] Add CatBoost as the first non-LGBM tabular baseline
- [ ] Add a simple linear baseline such as Ridge / ElasticNet for signal sanity checks
- [ ] Evaluate whether rank objectives should be added before broadening to more tree models
- [ ] Add a LightGBM ranking experiment path grouped by trade date and compare against regression on the same profiles
- [ ] Add model-profile level control for objective family, evaluation metric, and regularization regime
- [ ] Defer AutoEncoder / GNN / multi-task sequence models until tabular baselines and ranking objectives are exhausted

### 7. Tooling And UX

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
2. better factor set
3. training-time cross-sectional transforms

This is more likely to improve results than adding another deep model right now.
