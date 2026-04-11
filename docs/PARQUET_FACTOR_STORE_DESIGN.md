# Parquet Factor Store Design

## Goal

Move the project from a single dense `X.npy` memmap cache to a Parquet-based factor store
that is friendlier to the real workload:

- WSL environment with weak I/O throughput
- training on a selected subset of factors, not the full factor space
- rolling windows sliced by date range
- repeated experiments over the same date range with different selected columns

The objective is not to optimize for the fastest possible full-matrix scan.
The objective is to minimize bytes read for the columns and dates actually used.

## Why The Current Format Is Not Optimal

Current cache layout:

- `data/cache/all_factors_panel/X.npy`
- `data/cache/all_factors_panel/y.npy`
- `data/cache/all_factors_panel/date.npy`
- `data/cache/all_factors_panel/symbol.npy`
- `data/cache/all_factors_panel/meta.json`

Current issues:

1. The store is row-major and dense.
   Training often uses 20-40 columns out of the full factor space, but the physical layout
   is optimized for scanning the entire feature matrix.

2. The row order is symbol-major.
   Rolling training and evaluation slice by date windows, so the current physical layout
   works against the actual access pattern.

3. The store is uncompressed `float32`.
   On WSL, I/O is the bottleneck. Reading fewer bytes matters more than avoiding light decompression.

4. Incremental generation still ends with a full monolithic matrix.
   That preserves old training assumptions, but does not align with the desired store shape.

## Target Design

Use a Parquet factor store as the single source of truth for generated factors.

### Store Root

Recommended root:

- `data/factor_store/full_factor_space/`

### Physical Layout

Two layers:

1. Symbol shards for incremental recomputation

- `data/factor_store/full_factor_space/shards/<symbol>.parquet`

Each shard contains:

- `date`
- `symbol`
- `label`
- all factor columns

Properties:

- sorted by `date`
- one symbol per file
- used only for incremental rebuild / recomputation

2. Date-major compact panel dataset for training and evaluation

- `data/factor_store/full_factor_space/panel/year=YYYY/part-*.parquet`

Each row contains:

- `date`
- `symbol`
- `label`
- selected factor columns for the full factor space

Properties:

- globally sorted by `date`, then `symbol`
- partitioned by `year`
- written with `zstd`
- row groups sized for date-range pruning

This gives:

- cheap symbol-level incremental recomputation
- efficient date-range and column-pruned reads during training

## Why This Fits WSL Better

For WSL, the important metric is not "fastest possible sequential dense scan".
The important metric is "smallest amount of data read for the actual experiment".

Parquet helps because:

- only requested factor columns need to be read
- date partitions and row-group statistics reduce unnecessary reads
- compressed column data lowers total bytes read from disk
- experiments with different selected columns no longer pay for the entire factor matrix

## Read Path Design

Training should no longer open `X.npy`, `y.npy`, `date.npy`, `symbol.npy` directly.

Instead, introduce a dataset loader module, for example:

- `src/factor_store.py`

Core loader responsibilities:

1. Resolve factor store root from config
2. Read only:
   - selected factor columns
   - `date`
   - `symbol`
   - `label`
3. Apply date filter as early as possible
4. Apply universe filter after loading `date` and `symbol`
5. Return a compact pandas frame or column arrays

Recommended API:

```python
load_factor_frame(
    store_dir: str | Path,
    columns: list[str],
    date_start: str | pd.Timestamp,
    date_end: str | pd.Timestamp,
    universe_name: str,
    universe_dir: str | Path,
) -> pd.DataFrame
```

Required columns in the returned frame:

- `date`
- `symbol`
- `label`
- selected factor columns

## Rolling Pipeline Strategy

The rolling pipeline should not re-read the whole dataset window by window.

Recommended strategy:

1. Resolve selected factor columns
2. Compute the earliest train start and latest test end across all rolling windows
3. Load that date span once from the Parquet panel
4. Build rolling masks in memory from the loaded frame

This preserves the current rolling logic while drastically reducing on-disk bytes read.

## Generation Pipeline Strategy

`src/gen_feature.py` should be refactored into two steps:

1. `build_symbol_shards`
2. `compact_panel_dataset`

Recommended CLI evolution:

```bash
uv run python -m src.gen_feature --workers 24
```

still does both steps by default.

Optional advanced modes:

```bash
uv run python -m src.gen_feature --mode shards --incremental
uv run python -m src.gen_feature --mode compact
```

## File-Level Change Plan

### New Modules

- `src/factor_store.py`
  - Parquet read helpers
  - date-range pruning
  - selected-column loading
  - optional universe filtering helper

### Existing Modules To Change

- `src/gen_feature.py`
  - write symbol shards as Parquet
  - compact shards into date-major Parquet panel
  - stop writing monolithic `X.npy/y.npy/date.npy/symbol.npy`

- `src/feature_profiles.py`
  - `cache_dir` concept should evolve toward `factor_store_dir`
  - keep backward-compatible field alias during migration

- `main.py`
  - replace memmap loading with `factor_store.load_factor_frame`
  - build train/valid/test splits from the loaded frame

- `run_native_rolling.py`
  - replace memmap loading with one preloaded date-range frame
  - keep current rolling-window logic on the in-memory frame

- `src/feature_selection.py`
  - keep profile/subset logic
  - no major semantic change required

- `src/experiment_store.py`
  - record factor store path / format in manifest

### Tests To Add

- factor store column-pruning read test
- factor store date-range read test
- rolling pipeline smoke test on Parquet-backed factor store

## Migration Phases

### Phase 1

- Add Parquet factor store writer in parallel with current memmap output
- Add `src/factor_store.py`
- Add read-path tests

### Phase 2

- Switch `main.py` and `run_native_rolling.py` to read the Parquet factor store
- Keep memmap generation behind a fallback flag only if needed

### Phase 3

- Remove memmap generation and old `data/cache/all_factors_panel`
- Rename config fields from `cache_dir` to `factor_store_dir`

## Old Data That Can Be Deleted After Migration

After the Parquet factor store is fully adopted and validated, these old artifacts can be removed:

- `data/cache/all_factors_panel/X.npy`
- `data/cache/all_factors_panel/y.npy`
- `data/cache/all_factors_panel/date.npy`
- `data/cache/all_factors_panel/symbol.npy`
- `data/cache/all_factors_panel/meta.json`

If `data/cache/` has no other active contents after migration:

- `data/cache/`

These should be kept:

- `data/raw/`
- `data/processed/`
- `data/universes/`
- `data/cookies.json`

## Deletion Commands After Migration

Only after the Parquet path becomes the active runtime:

```bash
rm -rf data/cache/all_factors_panel
rmdir data/cache 2>/dev/null || true
```

## Recommendation

Do not migrate to `npz`.

For this project and this environment:

- Parquet is the right main factor-store format
- selected-column reads are the critical optimization
- WSL makes byte reduction more valuable than dense full-matrix scan speed
