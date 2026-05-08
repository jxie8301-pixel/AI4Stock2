# AI4Stock2

Native A-share research pipeline built around Parquet data, local feature caches, native LightGBM training, and a native backtest engine.

## Quickstart

Before running training or factor generation, prepare one local normalized parquet data source first.
The active research path is Tushare; AkShare / Eastmoney-compatible data remains available for compatibility and is documented in `docs/USER_GUIDE.md`.

Generate the factor store for the active config:

```bash
pixi run cargo run --bin ai4stock-gen-feature -- generate \
  --parquet-dir data/tushare/source \
  --output-dir data/factor_store/tushare_full_factor_space \
  --data-source tushare \
  --workers 8
```

Run a rolling LightGBM experiment by explicitly naming the experiment profile:

```bash
pixi run cargo run --bin ai4stock-train -- rolling-lgbm \
  --config configs/config.yaml \
  --experiment-profile core_v4_lgbm_default_10x20x10
```

Run Rust binaries through `pixi run cargo ...` so PyO3 embeds the pixi Python used by LightGBM and provider adapters. The old single-window training entrypoint has been removed; use rolling runs for research and reporting.

## Core Docs

- `docs/USER_GUIDE.md`: usage and command examples
- `docs/AI_CONTEXT_MAP.md`: current architecture and file map
- `docs/CONFIG_PROFILE_ARCHITECTURE.md`: canonical config/profile layering
- `docs/PROGRESS_AND_TODO.md`: current status and next research tasks
