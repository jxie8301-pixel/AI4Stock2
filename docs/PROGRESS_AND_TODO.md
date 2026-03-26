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
- Training-time feature subset selection is supported through `features.selected_columns`
- Local experiment storage is enabled through `results/experiments/`

## Current Recommended Workflow

1. Update or refresh Parquet data with `src/collector_akshare.py`
2. Generate a cache with the desired feature profile
3. Train `lgbm` on rolling windows with `run_native_rolling.py`
4. Compare experiments through `results/experiments/experiment_index.csv`

## Immediate TODO

### 1. LightGBM Feature Research

- [ ] Add a native `lgbm_purified_v1` feature profile inspired by the strongest old-project factors
- [ ] Add native support for valuation/style factors such as `ep_ttm`, `bp`, `log_mcap`, `is_loss`
- [ ] Add liquidity and microstructure factors such as `amihud_20`, `turnover_20`, `vwap_ratio`
- [ ] Compare `alpha158_compact_v1` vs `alpha158_full` vs `lgbm_purified_v1`

### 2. Training-Time Transforms

- [ ] Add optional daily cross-sectional rank transform before model training
- [ ] Add optional feature winsorization / clipping transform in the training path
- [ ] Add optional label de-meaning for LightGBM experiments
- [ ] Record applied transforms in experiment manifests

### 3. Native Data Quality

- [ ] Add a feature coverage report during cache generation
- [ ] Add per-feature NaN / inf diagnostics to `meta.json`
- [ ] Add a cache validation command for shape, names, coverage, and label sanity
- [ ] Review whether valuation fields are complete enough across the full sample

### 4. Backtest Realism

- [ ] Add explicit tradability flags for suspension / invalid rows where possible
- [ ] Evaluate whether limit-up / limit-down blocking should be modeled in native backtest
- [ ] Add higher-slippage sensitivity experiments
- [ ] Add risk control experiments for lower turnover and lower drawdown

### 5. Strategy Layer

- [ ] Add score-weighted portfolio construction instead of pure equal weight
- [ ] Add sector / style exposure diagnostics
- [ ] Add market-regime comparison for rebalance frequency
- [ ] Compare `topk` / `n_drop` combinations systematically

### 6. Native Model Roadmap

- [ ] Decide whether native LSTM should remain supported as a secondary path
- [ ] If sequence models remain in scope, build a real native Transformer implementation
- [ ] Add a unified save/load contract shared by all native models

### 7. Tooling And UX

- [ ] Add a script to compare runs directly from `experiment_index.csv`
- [ ] Add a script to summarize the best run per model/profile/tag
- [ ] Add richer manifest metadata for selected features and transforms
- [ ] Improve README with a short native quickstart

## Research Priority Recommendation

If only one direction should be pursued next, it should be:

1. native LightGBM
2. better factor set
3. training-time cross-sectional transforms

This is more likely to improve results than adding another deep model right now.
