# AI4Stock2 Config And Profile Architecture

## Purpose

This document defines the canonical configuration architecture for AI4Stock2.
It is the source of truth for how runtime config, factor-space generation,
feature subsets, model hyperparameters, and experiment semantics are separated.

The design goal is simple:

- `gen_feature.py` generates one unified full-factor store
- training consumes subsets from that store
- model hyperparameters are never hardcoded in Python when they should be profiles
- experiment semantics are named and reproducible

## Four Layers

AI4Stock2 should be configured through four explicit layers.

### 1. Runtime Config

Runtime config describes machine-local or environment-local settings.
It must not define the research experiment itself.

Typical contents:

- local storage roots
- universe file directory
- factor-store root / shared cache paths
- factor-store materialization options such as label horizons to persist

Runtime config should live in:

- `configs/config.yaml`

Switch the active source with `data.source` or `--data-source`.

### 2. Feature Profile

Feature profiles define which columns are selected from the unified factor store.
They do not define model hyperparameters, backtest rules, or signal horizons.

Typical contents:

- selected factor columns
- optional factor-store name override if a profile uses a non-default store
- optional profile metadata

Feature profile index:

- `configs/feature_profiles.yaml`

Feature profile definitions:

- `configs/features/*.yaml`
- derived feature profiles may also live inline in `configs/feature_profiles.yaml` via `extends` / `drop_columns` / `add_columns`

### 3. Model Profile

Model profiles define model type and training hyperparameters.
They do not define factor subsets or strategy semantics.

Typical contents:

- `model.name`
- `model.batch_size`
- `model.n_jobs`
- `lgbm.learning_rate`
- `lgbm.num_boost_round`
- `lgbm.early_stop`
- `lgbm.early_stopping_min_delta`
- `lgbm.log_evaluation_period`
- `lgbm.num_leaves`
- `lgbm.max_depth`
- `lgbm.min_data_in_leaf`
- `lgbm.subsample`
- `lgbm.colsample_bytree`
- `lgbm.lambda_l1`
- `lgbm.lambda_l2`
- `lgbm.seed`

Model profile index:

- `configs/model_profiles.yaml`

Model profile definitions:

- `configs/models/*.yaml`

### 4. Experiment Profile

Experiment profiles define the research semantics of a run.
This is the layer that answers "what experiment are we running?"

Typical contents:

- `features.profile`
- `model.profile`
- `universe`
- `time.train`
- `time.valid`
- `time.test`
- `label.signal_horizon`
- `rolling.retrain_step`
- `rolling.train_days`
- `rolling.valid_days`
- `strategy.topk`
- `strategy.n_drop`
- `backtest.rebalance_freq`
- trading costs
- optional feature transforms that belong to experiment semantics

Experiment profile index:

- `configs/experiment_profiles.yaml`

Experiment profile definitions:

- `configs/experiments/*.yaml`
- derived experiment profiles may also live inline in `configs/experiment_profiles.yaml` via `extends`

## Naming Rules

To remove ambiguity, the following names are canonical:

- `signal_horizon`: how far ahead the model predicts
- `retrain_step`: how often the rolling model is retrained
- `rebalance_freq`: how often the portfolio is rebalanced

The name `horizon` by itself is ambiguous and should not be used in new configs,
new CLI flags, manifests, or docs.

## Label Semantics

Signal evaluation and portfolio backtest use different data semantics:

- signal evaluation uses `label.signal_horizon`
- backtest always uses realized `1d` returns

This distinction is mandatory.
Multi-day forward labels must never be compounded as daily realized returns.

## Composition Order

The canonical merge order is:

1. code defaults
2. runtime config
3. experiment profile
4. model profile
5. CLI overrides

Feature profiles are not merged as generic config blocks.
They are resolved separately and used only for factor-store selection.

## Command-Line Interface

The preferred entry points should be:

```bash
uv run python -m src.gen_feature
uv run python run_native_rolling.py --experiment-profile core_v4_lgbm_default_10x20x10
uv run python main.py --experiment-profile core_v4_lgbm_default_10x20x10
```

`--experiment-profile` should be treated as mandatory for train/backtest entry
points. There should be no implicit default experiment profile.

Optional overrides are allowed, but named profiles should be the default workflow:

```bash
uv run python run_native_rolling.py \
  --experiment-profile core_v4_lgbm_default_10x20x10 \
  --feature-profile core_v4_techlite \
  --model-profile lgbm_fast
```

For one-off comparisons, prefer generic dotted overrides over cloning many yaml files:

```bash
uv run python run_native_rolling.py \
  --experiment-profile core_v4_lgbm_default_10x20x10 \
  --set strategy.topk=20 \
  --set rolling.retrain_step=5
```

For batch sweeps, prefer one base experiment plus a sweep runner:

```bash
uv run python run_experiment_batch.py \
  --pipeline rolling \
  --experiment-profile core_v4_lgbm_default_10x20x10 \
  --sweep 'rolling.retrain_step=[5,10,15]'
```

## Recommended Baselines

The recommended baseline stack is:

- factor store: unified full-factor space
- feature profile: `core_v4_techlite`
- model profile: `lgbm_default`
- experiment profile: `core_v4_lgbm_default_10x20x10`

## Migration Notes

The intended migration path is:

1. keep one unified factor store
2. move model hyperparameters fully into model profiles
3. move experiment semantics out of `config.yaml`
4. switch entry scripts to `--experiment-profile`
5. keep only narrowly scoped CLI overrides for comparisons

## Non-Goals

This architecture is intentionally avoiding:

- one-off experiments encoded only in long CLI strings
- feature profiles that also carry model hyperparameters
- hardcoded LightGBM training-loop knobs such as fixed `num_boost_round`
- hidden coupling between signal labels and backtest realized returns
