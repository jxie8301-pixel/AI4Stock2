# AI4Stock2

Native A-share research pipeline built around Parquet data, local feature caches, native LightGBM/LSTM training, and a native backtest engine.

## Quickstart

Generate the factor store for the active config:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python src/gen_feature.py --workers 8
```

Run a rolling LightGBM experiment by explicitly naming the experiment profile:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python run_native_rolling.py --config configs/config.yaml --experiment-profile core_v4_lgbm_default_10x20x10
```

Run a single native experiment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --config configs/config.yaml --experiment-profile core_v4_lgbm_default_10x20x10
```

## Core Docs

- `docs/USER_GUIDE.md`: usage and command examples
- `docs/AI_CONTEXT_MAP.md`: current architecture and file map
- `docs/CONFIG_PROFILE_ARCHITECTURE.md`: canonical config/profile layering
- `docs/PROGRESS_AND_TODO.md`: current status and next research tasks
