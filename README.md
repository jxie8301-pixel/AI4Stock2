# AI4Stock2

Native A-share research pipeline built around Parquet data, local feature caches, native LightGBM/LSTM training, and a native backtest engine.

## Quickstart

Before running training or factor generation, prepare one local normalized parquet data source first.
The current stable path is still AkShare / Eastmoney-compatible data; GM and Tushare usage is documented in `docs/USER_GUIDE.md`.

Generate the factor store for the active config:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.gen_feature --workers 8
```

Run a rolling LightGBM experiment by explicitly naming the experiment profile:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python run_native_rolling.py --config configs/config.yaml --experiment-profile core_v4_lgbm_default_10x20x10
```

The old single-window training entrypoint has been removed; use rolling runs for research and reporting.

## Core Docs

- `docs/USER_GUIDE.md`: usage and command examples
- `docs/AI_CONTEXT_MAP.md`: current architecture and file map
- `docs/CONFIG_PROFILE_ARCHITECTURE.md`: canonical config/profile layering
- `docs/PROGRESS_AND_TODO.md`: current status and next research tasks
