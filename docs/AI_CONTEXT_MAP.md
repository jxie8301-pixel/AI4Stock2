# AI4Stock2 Context Map

## Current State

AI4Stock2 is now a native A-share research pipeline built around:

- Parquet market data in `data/processed/combined/`
- Local feature-cache generation in `src/gen_feature.py`
- Native rolling training in `run_native_rolling.py`
- Native backtest and evaluation in `src/native_backtest.py` and `src/evaluate.py`

The default research path is:

1. Update raw/processed Parquet data with `src/collector_akshare.py`
2. Generate a feature cache from the selected feature profile
3. Train a native LightGBM model on rolling windows
4. Evaluate signal quality and run the native backtest
5. Archive metrics, plots, models, and config snapshots into the local experiment store

## Active Workflow

### Data Layer

- Source of truth: `data/processed/combined/*.parquet`
- Updater: `src/collector_akshare.py`
- Native feature-cache builder: `src/gen_feature.py`

### Feature Layer

- Feature profiles: `configs/feature_profiles.yaml`
- Profile resolver: `src/feature_profiles.py`
- Training-time feature subset selection: `src/feature_selection.py`

The intended workflow is:

- Generate a full profile cache once
- Use `features.selected_columns` in config to train on subsets without rebuilding the cache

### Model Layer

- Native LightGBM: `src/models/pure_lightgbm.py`
- Native LSTM: `src/models/pure_pytorch_lstm.py`

Current default model is `lgbm`.

### Experiment Layer

- Single-run entry: `main.py`
- Rolling entry: `run_native_rolling.py`
- Experiment/model archive: `src/experiment_store.py`

### Evaluation Layer

- Signal metrics and plots: `src/evaluate.py`
- Native backtest engine: `src/native_backtest.py`
- Backtest wrapper: `src/backtest.py`

## Key Files

```text
AI4Stock2/
├── configs/
│   ├── config.yaml
│   ├── config_baseline.yaml
│   └── feature_profiles.yaml
├── data/
│   ├── processed/combined/
│   └── cache/
├── docs/
│   ├── AI_CONTEXT_MAP.md
│   ├── PROGRESS_AND_TODO.md
│   └── USER_GUIDE.md
├── src/
│   ├── collector_akshare.py
│   ├── gen_feature.py
│   ├── feature_profiles.py
│   ├── feature_selection.py
│   ├── native_backtest.py
│   ├── backtest.py
│   ├── evaluate.py
│   └── models/
│       ├── pure_lightgbm.py
│       └── pure_pytorch_lstm.py
├── main.py
└── run_native_rolling.py
```

## Current Defaults

- Backend: native only
- Default model: `lgbm`
- Default profile: `alpha158_compact_v1`
- Default universe: `csi300_real`
- Default label: `open_{t+2} / open_{t+1} - 1`

## Practical Notes

- If you change `features.profile`, rebuild the cache.
- If you only change `features.selected_columns`, do not rebuild the cache.
- Use this document for structure.
- Use `docs/USER_GUIDE.md` for commands.
- Use `docs/PROGRESS_AND_TODO.md` for next research tasks.
