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

## Still not fully aligned

### 1. Tradability rules are still missing

Native backtest still does not model:

- limit-up / limit-down blocking
- suspension
- round-lot sizing
- exact exchange order validation

So native results are now closer, but still not a byte-for-byte replacement for `backtest_daily`.

### 2. Native universe still depends on Qlib instrument files

`main.py` and `run_native_rolling.py` still read:

`data/qlib_data_cn/instruments/<universe>.txt`

This blocks full de-Qlib execution and can silently fall back to the full market if the file is missing.

### 3. Feature/processor semantics still differ

The Qlib handler path still includes:

- valuation features
- `RobustZScoreNorm`
- `Fillna`
- `DropnaLabel`
- `CSRankNorm`

The native cache path does not yet reproduce those processors exactly.
So model training is still not apples-to-apples with the legacy Qlib path.

### 4. Validation metric is still mismatched

Native LSTM early stopping still uses a global Pearson correlation over the whole validation set.
Final reporting uses daily IC / RankIC.
This can bias checkpoint selection away from the real production objective.

## Recommended next fixes

1. Replace Qlib universe files with a native universe source and fail fast if the configured universe is unavailable.
2. Reproduce the Qlib processor chain in native mode before training.
3. Align the validation objective with daily IC / RankIC.
4. Add a regression fixture comparing native and Qlib backtest outputs on a tiny deterministic panel.
