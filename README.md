# AI4Stock2

Native A-share research pipeline built around Parquet data, local feature caches, native LightGBM/LSTM training, and a native backtest engine.

## Quickstart

Generate the cache for the active config:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python src/gen_feature.py --workers 8
```

Run the default rolling LightGBM experiment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python run_native_rolling.py --config configs/config.yaml --model lgbm --horizon 20 --run-tag compact_lgbm
```

Run a single native experiment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py --config configs/config.yaml --model lgbm --run-tag single_lgbm
```

## Core Docs

- `docs/USER_GUIDE.md`: usage and command examples
- `docs/AI_CONTEXT_MAP.md`: current architecture and file map
- `docs/PROGRESS_AND_TODO.md`: current status and next research tasks
