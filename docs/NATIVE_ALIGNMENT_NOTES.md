# Native/Qlib Alignment Notes

Updated: 2026-03-12

## Goal

Current priority is compatibility with the existing Qlib pipeline first, then full removal of the runtime dependency.

## Fixed in this round

### 1. Native backtest now honors `n_drop`

The old native backtest rebuilt the full Top-K basket on each rebalance day.
That was not equivalent to `qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy`.

`src/native_backtest.py` now mirrors the Qlib selection flow:

- rank current holdings by the latest score
- select non-held candidates from the top of the cross-section
- combine current holdings and candidates
- sell only the bottom `n_drop` names from the combined list
- buy only enough names to replace the sold names and refill to `topk`

### 2. Native cost model is closer to Qlib

The old native backtest approximated costs as:

`turnover * (buy_cost + sell_cost)`

That under-estimated both initial entry costs and any asymmetric buy/sell path.

The backtest now simulates:

- account cash
- per-position market value
- buy and sell legs separately
- `slippage`
- `min_cost`
- `risk_degree`

Default config values now match the Qlib wrapper:

- `slippage: 0.0005`
- `min_cost: 5`
- `account: 100000000`
- `risk_degree: 0.95`

### 3. Native model selection now uses daily IC

The old native LSTM early stopping logic used one global Pearson correlation over
the entire validation split.

That was not aligned with the project's reporting target, which is daily
cross-sectional IC.

The native model path now does the following:

- `NativeStockDataset` carries sample dates
- native LSTM validation computes mean daily IC
- native LightGBM reports `daily_ic` as an auxiliary validation metric while
  early stopping uses a stable regression metric (`l2`/`l1`)

### 4. Native rolling/main now read project-owned universe files

The old native path still read:

`data/qlib_data_cn/instruments/<universe>.txt`

That dependency has been replaced with project-owned native assets under:

`data/universes/`

The native loader now supports symbol membership windows with start/end dates,
so the project can keep PIT-aware universe files without relying on Qlib's
instrument directory structure.

## Still not fully aligned

### 1. Tradability rules are still missing

Native backtest still does not model:

- limit-up / limit-down blocking
- suspension
- round-lot sizing
- exact exchange order validation

So native results are now closer, but still not a byte-for-byte replacement for `backtest_daily`.

### 2. Feature/processor semantics still differ

The Qlib handler path still includes:

- valuation features
- `RobustZScoreNorm`
- `Fillna`
- `DropnaLabel`
- `CSRankNorm`

The native cache path does not yet reproduce those processors exactly.
So model training is still not apples-to-apples with the legacy Qlib path.

## Recommended next fixes

1. Reproduce the Qlib processor chain in native mode before training.
2. Decide whether native training should also optimize rank-based metrics, not only Pearson-based daily IC.
3. Add a regression fixture comparing native and Qlib backtest outputs on a tiny deterministic panel.
4. Decide whether native universe files should become part of the data generation pipeline instead of being copied manually.
