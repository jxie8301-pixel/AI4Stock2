# AI4Stock2 Context Map

## Current State

AI4Stock2 is now a native A-share research pipeline built around:

- Parquet market data in normalized `combined/` directories
- Local feature-cache generation through Rust `ai4stock-gen-feature`
- Native rolling LightGBM training through Rust `ai4stock-train rolling-lgbm`
- Native post-bundle backtest and evaluation through Rust `ai4stock-backtest run-bundle`

The stable research path today is:

1. Update raw/processed Parquet data with `src/collector_akshare.py`
2. Generate one unified full-factor store with `src/gen_feature.py`
3. Train a native LightGBM model from a named experiment profile
4. Evaluate signal quality and run the Rust post-bundle backtest
5. Archive metrics, plots, models, and config snapshots into the local experiment store

At the same time, the data layer is actively being migrated:

1. `src/collector_tushare.py` is the leading replacement candidate for daily market data
2. Tushare is not yet the default research source because the canonical normalized schema and formal workflow wiring are still in progress

## Active Workflow

### Data Layer

- Stable source of truth today: `data/processed/combined/*.parquet`
- Stable updater today: Rust `ai4stock-collect akshare`, with `src/collector_akshare.py` kept as a compatibility wrapper.
- Universe membership builder: Rust `ai4stock-collect universes`, with `src/build_universes.py` kept as a compatibility wrapper and the AkShare provider adapter kept behind the PyO3 bridge.
- Isolated Tushare path: `data/tushare/raw/*` -> `data/tushare/processed/combined/*.parquet`
- Migration target: promote one canonical Tushare-normalized `combined` schema into the formal research workflow
- Native feature-cache builder: Rust `ai4stock-gen-feature`, with `src/gen_feature.py` kept as config/profile-resolving compatibility wrapper and pandas reference path
- Native rolling LightGBM entry: Rust `ai4stock-train rolling-lgbm` owns output-dir resolution, config/profile/date overrides, prediction-bundle creation, and post-bundle backtest delegation; `run_native_rolling.py` is a compatibility wrapper only
- Native post-bundle backtest: Rust `ai4stock-backtest run-bundle` owns backtest execution, benchmark/reference-baseline reports, plots, trace artifacts, and backtest label safety validation. The old Python backtest engine path has been removed.
- Diagnostics/profile prefilter: Rust `ai4stock-diagnostics single-factor`, `single-factor-profile`, `single-factor-batch`, `full-space-single-factor`, `quality-event-flow-single-factor`, `prefilter-summary`, `robust-prefilter-summary`, `corr-prune`, `write-profile`, `build-prefilter-profile`, `build-robust-profile`, `build-prefilter-profile-runtime`, `build-robust-profile-runtime`, `candidate-pool`, and `strategy-pair`; Python diagnostics/profile builders are compatibility wrappers only
- LGBM artifact-rebuild batches: Rust `ai4stock-backtest artifact-batch`, with `run_lgbm_backtest_artifacts.py` kept only as a compatibility wrapper that delegates to Rust
- Experiment sweep batches: Rust `ai4stock-experiment batch` owns sweep/case expansion, command generation, dry-run output, sequential child execution, and prediction-bundle dedupe/replay; `run_experiment_batch.py` is a compatibility wrapper

Current collector roles:

- `src/collector_akshare.py`: compatibility wrapper plus provider adapter for the current default Eastmoney-compatible dataset
- `src/collector_tushare.py`: compatibility wrapper plus Tushare provider adapter; Rust owns scheduling, processed rebuild, packed source, sidecar lagging, and industry context materialization
- `src/probe_tushare.py`: endpoint probe for schema/latency inspection before formal integration

### Feature Layer

- Feature profile index: `configs/feature_profiles.yaml`
- Feature profile definitions: `configs/features/*.yaml`
- Profile resolver: `src/feature_profiles.py`
- Training-time feature subset selection: `src/feature_selection.py`

The intended workflow is:

- Generate one unified full-factor store once
- Resolve a feature profile at training time
- Use `features.selected_columns` only as a narrow explicit override when needed

### Model Layer

- Model profile index: `configs/model_profiles.yaml`
- Model profile definitions: `configs/models/*.yaml`
- Native LightGBM: `src/models/pure_lightgbm.py`
- Native LSTM: `src/models/pure_pytorch_lstm.py`

Current default model is `lgbm`.

### Experiment Layer

- Experiment profile index: `configs/experiment_profiles.yaml`
- Experiment profile definitions: `configs/experiments/*.yaml`
- Rolling entry: Rust `ai4stock-train rolling-lgbm`; compatibility wrapper `run_native_rolling.py`
- Experiment/model archive: `src/experiment_store.py`

### Evaluation Layer

- Training-side validation metrics and benchmark helpers: Rust `ai4stock-train rolling-lgbm`
- Post-bundle score fusion, backtest, reports, and plots: Rust `ai4stock-backtest run-bundle`

## Key Files

```text
AI4Stock2/
├── configs/
│   ├── config.yaml
│   ├── feature_profiles.yaml
│   ├── model_profiles.yaml
│   ├── experiment_profiles.yaml
│   ├── features/
│   ├── models/
│   └── experiments/
├── data/
│   ├── processed/combined/
│   └── factor_store/
├── docs/
│   ├── AI_CONTEXT_MAP.md
│   ├── CONFIG_PROFILE_ARCHITECTURE.md
│   ├── PROGRESS_AND_TODO.md
│   └── USER_GUIDE.md
├── src/
│   ├── collector_akshare.py
│   ├── collector_tushare.py
│   ├── config_loader.py
│   ├── data_source.py
│   ├── experiment_profiles.py
│   ├── gen_feature.py
│   ├── rust_lgbm_bridge.py
│   ├── feature_profiles.py
│   ├── feature_selection.py
│   ├── model_profiles.py
│   ├── override_utils.py
│   ├── runtime_cli.py
│   ├── probe_tushare.py
│   └── models/
│       ├── pure_lightgbm.py
│       └── pure_pytorch_lstm.py
├── src_rust/
│   ├── lib.rs
│   ├── gen_feature.rs
│   ├── feature_kernels.rs
│   └── bin/
│       ├── ai4stock_backtest.rs
│       ├── ai4stock_gen_feature.rs
│       └── ai4stock_train.rs
├── run_experiment_batch.py
└── run_native_rolling.py
```

## Current Defaults

- Backend: native only
- Default model: `lgbm`
- Default feature profile: `core_v4_techlite`
- Stable default data path: AkShare / Eastmoney-compatible `data/processed/combined`
- Tushare is the active migration target, but not yet the formal default
- Experiments must now be selected explicitly via `--experiment-profile`
- Default universe: `csi300`
- Supported generated universes: `csi300`, `csi500`, `zz1000`
- Backtest always uses realized `1d` returns
- Signal evaluation uses the configured `signal_horizon`

## Practical Notes

- If you change experiment profile, you do not rebuild the cache.
- If you change feature profile, you do not rebuild the cache.
- If you only change model profile, you do not rebuild the cache.
- `gen_feature.py` should remain detached from feature/model/experiment profiles.
- When validating Tushare, keep its normalized parquet under its own directory until the canonical schema switch is finished.
- Use this document for structure.
- Use `docs/CONFIG_PROFILE_ARCHITECTURE.md` for the canonical layering rules.
- Use `docs/USER_GUIDE.md` for commands.
- Use `docs/PROGRESS_AND_TODO.md` for next research tasks.
