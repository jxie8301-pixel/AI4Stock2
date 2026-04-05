"""Unified feature pipeline without qlib.

This module remains the public entrypoint for:
1) Feature family definitions.
2) Feature value computation.
3) Tushare sidecar augmentation.
4) Unified Parquet factor-store generation.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml

try:
    from src.data_source import (
        SUPPORTED_DATA_SOURCES,
        get_default_factor_store_dir,
        resolve_data_source_name,
        resolve_source_parquet_dir,
    )
except ModuleNotFoundError:
    from data_source import (  # type: ignore
        SUPPORTED_DATA_SOURCES,
        get_default_factor_store_dir,
        resolve_data_source_name,
        resolve_source_parquet_dir,
    )

try:
    from src.label_utils import (
        get_label_column_name,
        get_label_definition,
        get_legacy_label_column_name,
        resolve_label_horizons,
    )
except ModuleNotFoundError:
    from label_utils import (  # type: ignore
        get_label_column_name,
        get_label_definition,
        get_legacy_label_column_name,
        resolve_label_horizons,
    )

from src.feature_name_registry import (
    ALL_FACTORS_ALPHA360_PREFIX,
    ALL_FACTORS_LGBM_PREFIX,
    DEFAULT_ALPHA158_CONFIG,
    DEFAULT_FULL_FACTOR_STORE_DIR,
    DEFAULT_LGBM_PURIFIED_CONFIG,
    DEFAULT_TECHNICAL_FACTOR_CONFIG,
    DEFAULT_TEMPORAL_FACTOR_CONFIG,
    DEFAULT_TUSHARE_FACTOR_CONFIG,
    EPS,
    FULL_FACTOR_SPACE_NAME,
    TECHNICAL_FACTOR_PREFIX,
    TEMPORAL_FACTOR_PREFIX,
    TUSHARE_FACTOR_PREFIX,
    get_all_factor_feature_names,
    get_alpha158_feature_config,
    get_alpha360_feature_config,
    get_exact_duplicate_feature_source_map,
    get_full_factor_space_feature_names,
    get_known_exact_duplicate_feature_groups,
    get_lgbm_purified_feature_names,
    get_technical_factor_feature_names,
    get_temporal_factor_feature_names,
    get_tushare_factor_feature_names,
    deduplicate_exact_feature_names,
    validate_default_dimensions,
)
from src.feature_value_core import (
    _build_open_to_open_label_from_base,
    _index_to_epoch_ns,
    _prepare_ohlcv,
    _rolling_corr,
    _rolling_rank_pct,
    _rolling_regression_stats,
    _rolling_resi,
    _rolling_rsquare,
    _rolling_slope,
    _to_panel_arrays,
    build_open_to_open_label,
    build_open_to_open_labels,
    compute_all_factor_features,
    compute_alpha158,
    compute_alpha360,
    compute_lgbm_purified_features,
    compute_technical_factor_features,
    compute_temporal_factor_features,
    compute_tushare_factor_features,
)


TUSHARE_RAW_FINA_INDICATOR_DIR = Path("data/tushare/raw/fina_indicator")
TUSHARE_RAW_DIVIDEND_DIR = Path("data/tushare/raw/dividend")
TUSHARE_RAW_FORECAST_DIR = Path("data/tushare/raw/forecast")
TUSHARE_RAW_EXPRESS_DIR = Path("data/tushare/raw/express")
SHARD_DIRNAME = "_shards"

TUSHARE_FINA_INDICATOR_FEATURE_PAIRS = [
    ("eps", "fi_eps"),
    ("dt_eps", "fi_dt_eps"),
    ("bps", "fi_bps"),
    ("ocfps", "fi_ocfps"),
    ("roe", "fi_roe"),
    ("roe_dt", "fi_roe_dt"),
    ("roa", "fi_roa"),
    ("grossprofit_margin", "fi_grossprofit_margin"),
    ("netprofit_margin", "fi_netprofit_margin"),
    ("debt_to_assets", "fi_debt_to_assets"),
    ("q_eps", "fi_q_eps"),
    ("q_dtprofit", "fi_q_dtprofit"),
    ("q_roe", "fi_q_roe"),
    ("q_dt_roe", "fi_q_dt_roe"),
    ("tr_yoy", "fi_tr_yoy"),
    ("or_yoy", "fi_or_yoy"),
    ("op_yoy", "fi_op_yoy"),
    ("netprofit_yoy", "fi_netprofit_yoy"),
    ("ocf_yoy", "fi_ocf_yoy"),
]
TUSHARE_FINA_INDICATOR_FEATURE_COLS = [target for _, target in TUSHARE_FINA_INDICATOR_FEATURE_PAIRS]

TUSHARE_DIVIDEND_FEATURE_PAIRS = [
    ("cash_div", "div_cash_div"),
    ("cash_div_tax", "div_cash_div_tax"),
    ("stk_div", "div_stk_div"),
    ("stk_bo_rate", "div_stk_bo_rate"),
    ("stk_co_rate", "div_stk_co_rate"),
    ("base_share", "div_base_share"),
]
TUSHARE_DIVIDEND_FEATURE_COLS = [target for _, target in TUSHARE_DIVIDEND_FEATURE_PAIRS]

TUSHARE_FORECAST_FEATURE_PAIRS = [
    ("p_change_min", "fc_p_change_min"),
    ("p_change_max", "fc_p_change_max"),
    ("net_profit_min", "fc_net_profit_min"),
    ("net_profit_max", "fc_net_profit_max"),
    ("last_parent_net", "fc_last_parent_net"),
]
TUSHARE_FORECAST_FEATURE_COLS = [target for _, target in TUSHARE_FORECAST_FEATURE_PAIRS]

TUSHARE_EXPRESS_FEATURE_PAIRS = [
    ("revenue", "exp_revenue"),
    ("operate_profit", "exp_operate_profit"),
    ("total_profit", "exp_total_profit"),
    ("n_income", "exp_n_income"),
    ("total_assets", "exp_total_assets"),
    ("diluted_eps", "exp_diluted_eps"),
    ("diluted_roe", "exp_diluted_roe"),
    ("yoy_sales", "exp_yoy_sales"),
    ("yoy_op", "exp_yoy_op"),
    ("yoy_tp", "exp_yoy_tp"),
    ("yoy_dedu_np", "exp_yoy_dedu_np"),
    ("yoy_eps", "exp_yoy_eps"),
    ("yoy_roe", "exp_yoy_roe"),
    ("growth_assets", "exp_growth_assets"),
    ("yoy_assets", "exp_yoy_assets"),
]
TUSHARE_EXPRESS_FEATURE_COLS = [target for _, target in TUSHARE_EXPRESS_FEATURE_PAIRS]


def _load_tushare_sidecar_features(
    symbol: str,
    date_index: pd.DatetimeIndex,
    *,
    raw_dir: Path,
    column_pairs: list[tuple[str, str]],
) -> pd.DataFrame | None:
    path = raw_dir / f"{symbol}.parquet"
    if not path.exists():
        return None

    output_cols = [target for _, target in column_pairs]
    available_pairs: list[tuple[str, str]]

    try:
        schema_names = set(pq.read_schema(path).names)
        if "ann_date" not in schema_names:
            return None
        available_pairs = [(source, target) for source, target in column_pairs if source in schema_names]
        if not available_pairs:
            return None
        frame = pd.read_parquet(path, columns=["ann_date", *(source for source, _ in available_pairs)])
    except Exception:
        frame = pd.read_parquet(path)
        if frame.empty or "ann_date" not in frame.columns:
            return None
        available_pairs = [(source, target) for source, target in column_pairs if source in frame.columns]
        if not available_pairs:
            return None

    if frame.empty:
        return None

    frame = frame.copy()
    frame["ann_date"] = pd.to_datetime(frame["ann_date"], errors="coerce")
    frame = frame.dropna(subset=["ann_date"]).sort_values("ann_date")
    if frame.empty:
        return None

    source_cols = [source for source, _ in available_pairs]
    right = frame[["ann_date", *source_cols]].copy()
    for col in source_cols:
        right[col] = pd.to_numeric(right[col], errors="coerce")
    right = right.rename(columns=dict(available_pairs))

    left = pd.DataFrame({"date": pd.to_datetime(date_index)}).sort_values("date")
    merged = pd.merge_asof(
        left,
        right.sort_values("ann_date"),
        left_on="date",
        right_on="ann_date",
        direction="backward",
    )
    merged = merged.drop(columns=["ann_date"], errors="ignore").set_index("date")
    for col in output_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    return merged.reindex(columns=output_cols)


def _load_tushare_fina_indicator_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_FINA_INDICATOR_DIR,
        column_pairs=TUSHARE_FINA_INDICATOR_FEATURE_PAIRS,
    )


def _load_tushare_dividend_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_DIVIDEND_DIR,
        column_pairs=TUSHARE_DIVIDEND_FEATURE_PAIRS,
    )


def _load_tushare_forecast_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_FORECAST_DIR,
        column_pairs=TUSHARE_FORECAST_FEATURE_PAIRS,
    )


def _load_tushare_express_features(symbol: str, date_index: pd.DatetimeIndex) -> pd.DataFrame | None:
    return _load_tushare_sidecar_features(
        symbol,
        date_index,
        raw_dir=TUSHARE_RAW_EXPRESS_DIR,
        column_pairs=TUSHARE_EXPRESS_FEATURE_PAIRS,
    )


def _augment_tushare_symbol_frame(
    df: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        return out
    date_index = pd.DatetimeIndex(pd.to_datetime(out["date"]))
    fina_indicator = _load_tushare_fina_indicator_features(symbol, date_index)
    if fina_indicator is not None and not fina_indicator.empty:
        aligned = fina_indicator.reindex(date_index)
        out[aligned.columns] = aligned.to_numpy(copy=False)
    dividend = _load_tushare_dividend_features(symbol, date_index)
    if dividend is not None and not dividend.empty:
        aligned = dividend.reindex(date_index)
        out[aligned.columns] = aligned.to_numpy(copy=False)
    forecast = _load_tushare_forecast_features(symbol, date_index)
    if forecast is not None and not forecast.empty:
        aligned = forecast.reindex(date_index)
        out[aligned.columns] = aligned.to_numpy(copy=False)
    express = _load_tushare_express_features(symbol, date_index)
    if express is not None and not express.empty:
        aligned = express.reindex(date_index)
        out[aligned.columns] = aligned.to_numpy(copy=False)
    return out


def _compute_symbol_feat_label(
    file_path: str,
    *,
    data_source: str | None = None,
) -> tuple[str, pd.DataFrame, pd.Series]:
    df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    if data_source == "tushare":
        df = _augment_tushare_symbol_frame(df, symbol=symbol)
    base = _prepare_ohlcv(df)
    feat = compute_all_factor_features(df, data_source=data_source, _base=base)
    label = _build_open_to_open_label_from_base(base, horizon_days=1)
    return symbol, feat, label


def _compute_symbol_feat_labels(
    file_path: str,
    *,
    label_horizons: list[int],
    data_source: str | None = None,
) -> tuple[str, pd.DataFrame, dict[str, pd.Series]]:
    df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    if data_source == "tushare":
        df = _augment_tushare_symbol_frame(df, symbol=symbol)
    base = _prepare_ohlcv(df)
    feat = compute_all_factor_features(df, data_source=data_source, _base=base)
    labels = {
        get_label_column_name(horizon): _build_open_to_open_label_from_base(base, horizon_days=horizon)
        for horizon in label_horizons
    }
    labels[get_legacy_label_column_name()] = labels[get_label_column_name(1)]
    return symbol, feat, labels


def _count_file_worker(
    file_path: str,
    data_source: str | None = None,
) -> tuple[str, int, int]:
    symbol = Path(file_path).stem
    try:
        meta = pq.read_metadata(file_path)
        n_rows = int(meta.num_rows)
    except Exception:
        n_rows = int(len(pd.read_parquet(file_path, columns=["date"])))
    return file_path, symbol, len(get_full_factor_space_feature_names(data_source=data_source)), n_rows


def _build_file_payload_worker(
    file_path: str,
    data_source: str | None = None,
) -> tuple[str, int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    symbol, feat, label = _compute_symbol_feat_label(file_path, data_source=data_source)
    payload = _to_panel_arrays(feat, label)
    return symbol, feat.shape[1], payload


def _write_panel_file_slice_process(
    file_path: str,
    start: int,
    count: int,
    symbol_id: int,
    x_path: str,
    y_path: str,
    date_path: str,
    symbol_path: str,
    total_rows: int,
    n_feat: int,
    data_source: str | None = None,
) -> int:
    symbol, feat, label = _compute_symbol_feat_label(file_path, data_source=data_source)
    x_arr, y_arr, d_arr = _to_panel_arrays(feat, label)
    if x_arr.shape[0] != count:
        raise RuntimeError(
            f"Row count mismatch for {file_path}: counted={count}, computed={x_arr.shape[0]} (symbol={symbol})"
        )

    x_store = np.lib.format.open_memmap(x_path, mode="r+", dtype=np.float32, shape=(total_rows, n_feat))
    y_store = np.lib.format.open_memmap(y_path, mode="r+", dtype=np.float32, shape=(total_rows,))
    date_store = np.lib.format.open_memmap(date_path, mode="r+", dtype=np.int64, shape=(total_rows,))
    symbol_store = np.lib.format.open_memmap(symbol_path, mode="r+", dtype=np.int32, shape=(total_rows,))

    end = start + count
    x_store[start:end] = x_arr
    y_store[start:end] = y_arr
    date_store[start:end] = d_arr
    symbol_store[start:end] = symbol_id
    return count


def _json_dumps_canonical(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _source_file_signature(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _shard_base_name(file_path: str | Path) -> str:
    return Path(file_path).stem


def _shard_paths(shard_root: Path, shard_meta_root: Path, file_path: str | Path) -> tuple[Path, Path]:
    base = _shard_base_name(file_path)
    return shard_root / f"{base}.parquet", shard_meta_root / f"{base}.json"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_reusable_shard_meta(
    *,
    shard_root: Path,
    shard_meta_root: Path,
    file_path: str | Path,
    feature_names: list[str],
    label_columns: list[str],
) -> dict[str, Any] | None:
    shard_path, meta_path = _shard_paths(shard_root, shard_meta_root, file_path)
    shard_meta = _load_json(meta_path)
    if shard_meta is None or not shard_path.exists():
        return None
    source_sig = _source_file_signature(file_path)
    if shard_meta.get("source") != source_sig:
        return None
    if shard_meta.get("factor_space") != FULL_FACTOR_SPACE_NAME:
        return None
    if shard_meta.get("feature_names") != feature_names:
        return None
    if shard_meta.get("label_columns") != label_columns:
        return None
    row_count = shard_meta.get("row_count")
    if not isinstance(row_count, int) or row_count < 0:
        return None
    return shard_meta


def _save_shard(
    *,
    shard_root: Path,
    shard_meta_root: Path,
    file_path: str | Path,
    symbol: str,
    feature_names: list[str],
    label_columns: list[str],
    shard_frame: pd.DataFrame,
) -> dict[str, Any]:
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_meta_root.mkdir(parents=True, exist_ok=True)
    shard_path, meta_path = _shard_paths(shard_root, shard_meta_root, file_path)
    shard_frame.to_parquet(shard_path, index=False, engine="pyarrow", compression="zstd")
    shard_meta = {
        "symbol": symbol,
        "row_count": int(len(shard_frame)),
        "num_features": len(feature_names),
        "factor_space": FULL_FACTOR_SPACE_NAME,
        "source": _source_file_signature(file_path),
        "feature_names": feature_names,
        "label_columns": label_columns,
        "min_date": str(pd.to_datetime(shard_frame["date"]).min().date()) if not shard_frame.empty else "",
        "max_date": str(pd.to_datetime(shard_frame["date"]).max().date()) if not shard_frame.empty else "",
        "shard_path": str(shard_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(shard_meta, f, ensure_ascii=False, indent=2)
    return shard_meta


def _build_shard_frame(
    file_path: str | Path,
    *,
    label_horizons: list[int],
    data_source: str | None = None,
) -> tuple[str, pd.DataFrame]:
    symbol, feat, labels = _compute_symbol_feat_labels(
        str(file_path),
        label_horizons=label_horizons,
        data_source=data_source,
    )
    frame = feat.copy()
    frame = frame.astype(np.float32)
    frame.insert(0, "date", pd.to_datetime(frame.index))
    frame.insert(1, "symbol", symbol)
    insert_at = 2
    for label_column in [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]:
        if label_column in frame.columns:
            continue
        frame.insert(insert_at, label_column, labels[label_column].reindex(frame.index).astype(np.float32))
        insert_at += 1
    return symbol, frame.reset_index(drop=True)


def _write_factor_shard_worker(
    file_path: str,
    shard_path: str,
    meta_path: str,
    feature_names: list[str],
    label_horizons: list[int],
    data_source: str | None = None,
) -> dict[str, Any]:
    symbol, shard_frame = _build_shard_frame(
        file_path,
        label_horizons=label_horizons,
        data_source=data_source,
    )
    shard_root = Path(shard_path).parent
    shard_meta_root = Path(meta_path).parent
    return _save_shard(
        shard_root=shard_root,
        shard_meta_root=shard_meta_root,
        file_path=file_path,
        symbol=symbol,
        feature_names=feature_names,
        label_columns=[get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)],
        shard_frame=shard_frame,
    )


def _remove_orphan_shards(
    *,
    shard_root: Path,
    shard_meta_root: Path,
    source_files: list[Path],
) -> None:
    valid_names = {_shard_base_name(path) for path in source_files}
    for meta_path in shard_meta_root.glob("*.json"):
        if meta_path.stem in valid_names:
            continue
        shard_path = shard_root / f"{meta_path.stem}.parquet"
        if shard_path.exists():
            shard_path.unlink()
        meta_path.unlink()


def _collect_shard_metas(shard_meta_root: Path) -> list[dict[str, Any]]:
    shard_metas: list[dict[str, Any]] = []
    for meta_path in sorted(shard_meta_root.glob("*.json")):
        shard_meta = _load_json(meta_path)
        if shard_meta is not None:
            shard_metas.append(shard_meta)
    return shard_metas


def _compute_available_dates_from_shards(shard_root: Path) -> list[str]:
    dataset = ds.dataset(shard_root, format="parquet")
    table = dataset.to_table(columns=["date"])
    if table.num_rows == 0:
        return []
    dates = pd.to_datetime(table.column("date").to_pandas()).drop_duplicates().sort_values()
    return [str(pd.Timestamp(value).date()) for value in dates]


def generate_factor_store(
    parquet_dir: str = "data/processed/combined",
    output_dir: str = DEFAULT_FULL_FACTOR_STORE_DIR,
    workers: int = 1,
    incremental: bool = False,
    label_horizons: list[int] | None = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    pdir = Path(parquet_dir)
    out_root = Path(output_dir)
    shard_root = out_root / "shards"
    shard_meta_root = out_root / "shard_meta"
    out_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_meta_root.mkdir(parents=True, exist_ok=True)

    files = sorted(pdir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pdir}")

    _remove_orphan_shards(shard_root=shard_root, shard_meta_root=shard_meta_root, source_files=files)

    feature_names = get_full_factor_space_feature_names(data_source=data_source)
    label_horizons = resolve_label_horizons({"label": {"horizons": label_horizons}} if label_horizons is not None else {})
    label_columns = [get_legacy_label_column_name(), *(get_label_column_name(h) for h in label_horizons)]
    workers = max(1, int(workers))
    files_to_recompute: list[Path] = []
    reused_shard_metas: list[dict[str, Any]] = []
    written_shard_metas: list[dict[str, Any]] = []
    reused_files = 0

    print(f"[1/3] Planning factor-store build from {len(files)} parquet files (workers={workers})...")
    t0 = time.perf_counter()
    pbar = tqdm(files, desc="planning", total=len(files), unit="file")
    for idx, fp in enumerate(pbar, start=1):
        reusable = (
            _load_reusable_shard_meta(
                shard_root=shard_root,
                shard_meta_root=shard_meta_root,
                file_path=fp,
                feature_names=feature_names,
                label_columns=label_columns,
            )
            if incremental
            else None
        )
        if reusable is None:
            files_to_recompute.append(fp)
        else:
            reused_files += 1
            reused_shard_metas.append(reusable)
        elapsed = time.perf_counter() - t0
        speed = idx / elapsed if elapsed > 0 else 0.0
        eta = (len(files) - idx) / speed if speed > 0 else float("inf")
        pbar.set_postfix(reused=reused_files, rebuild=len(files_to_recompute), eta_m=f"{eta/60:.1f}")
    pbar.close()

    print(f"Shard plan: {reused_files} reused, {len(files_to_recompute)} recomputed")

    if files_to_recompute:
        print(f"[2/3] Writing Parquet shards (workers={workers})...")
        t1 = time.perf_counter()
        if workers == 1:
            pbar = tqdm(files_to_recompute, desc="shards", total=len(files_to_recompute), unit="file")
            for idx, fp in enumerate(pbar, start=1):
                shard_path, meta_path = _shard_paths(shard_root, shard_meta_root, fp)
                shard_meta = _write_factor_shard_worker(
                    str(fp),
                    str(shard_path),
                    str(meta_path),
                    feature_names,
                    label_horizons,
                    data_source,
                )
                written_shard_metas.append(shard_meta)
                elapsed = time.perf_counter() - t1
                speed = idx / elapsed if elapsed > 0 else 0.0
                eta = (len(files_to_recompute) - idx) / speed if speed > 0 else float("inf")
                pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
            pbar.close()
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for fp in files_to_recompute:
                    shard_path, meta_path = _shard_paths(shard_root, shard_meta_root, fp)
                    futures.append(
                        executor.submit(
                            _write_factor_shard_worker,
                            str(fp),
                            str(shard_path),
                            str(meta_path),
                            feature_names,
                            label_horizons,
                            data_source,
                        )
                    )
                pbar = tqdm(total=len(files_to_recompute), desc="shards", unit="file")
                for idx, fut in enumerate(as_completed(futures), start=1):
                    written_shard_metas.append(fut.result())
                    pbar.update(1)
                    elapsed = time.perf_counter() - t1
                    speed = idx / elapsed if elapsed > 0 else 0.0
                    eta = (len(files_to_recompute) - idx) / speed if speed > 0 else float("inf")
                    pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
                pbar.close()
    else:
        print("[2/3] Writing Parquet shards skipped: all source files reused.")

    print("[3/3] Finalizing factor-store metadata...")
    meta_by_source_path = {
        str(item.get("source", {}).get("path", "")): item
        for item in (reused_shard_metas + written_shard_metas)
    }
    source_paths = [str(fp.resolve()) for fp in files]
    shard_metas = [meta_by_source_path[path] for path in source_paths if path in meta_by_source_path]
    if len(shard_metas) != len(files):
        print("Metadata finalize fallback: reloading shard meta files from disk.")
        shard_metas = _collect_shard_metas(shard_meta_root)
    total_rows = sum(int(item.get("row_count", 0)) for item in shard_metas)
    recomputed_source_paths = {str(fp.resolve()) for fp in files_to_recompute}

    metadata = {
        "storage_format": "parquet",
        "storage_layout": "symbol_shards",
        "factor_space": FULL_FACTOR_SPACE_NAME,
        "data_source": data_source or "",
        "source_parquet_dir": str(pdir),
        "num_features": len(feature_names),
        "num_rows": total_rows,
        "shape": [total_rows, len(feature_names)],
        "feature_names": feature_names,
        "label": get_label_definition(1),
        "default_label_column": get_legacy_label_column_name(),
        "label_columns": [
            {
                "column": get_legacy_label_column_name(),
                "horizon": 1,
                "definition": get_label_definition(1),
                "legacy_alias": True,
            },
            *[
                {
                    "column": get_label_column_name(horizon),
                    "horizon": int(horizon),
                    "definition": get_label_definition(horizon),
                    "legacy_alias": False,
                }
                for horizon in label_horizons
            ],
        ],
        "factor_store_dir": str(out_root),
        "shards_dir": str(shard_root),
        "available_dates": _compute_available_dates_from_shards(shard_root),
        "incremental": {
            "enabled": incremental,
            "shard_dir": str(shard_root),
            "reused_files": reused_files,
            "recomputed_files": len(files_to_recompute),
        },
        "source_files": [
            {
                "file_path": item.get("source", {}).get("path", ""),
                "symbol": item.get("symbol", ""),
                "row_count": item.get("row_count", 0),
                "source": item.get("source", {}),
                "reused_shard": item.get("source", {}).get("path", "") not in recomputed_source_paths,
                "shard_path": item.get("shard_path", ""),
            }
            for item in shard_metas
        ],
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[3/3] Done. Parquet factor store saved to: {out_root}")
    return metadata


def generate_panel_cache(
    parquet_dir: str = "data/processed/combined",
    output_dir: str = DEFAULT_FULL_FACTOR_STORE_DIR,
    workers: int = 1,
    incremental: bool = False,
    label_horizons: list[int] | None = None,
    data_source: str | None = None,
) -> dict[str, Any]:
    return generate_factor_store(
        parquet_dir=parquet_dir,
        output_dir=output_dir,
        workers=workers,
        incremental=incremental,
        label_horizons=label_horizons,
        data_source=data_source,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the unified full-factor Parquet store from parquet.")
    parser.add_argument("--config", default="configs/config.yaml", help="Experiment config path for factor-store output settings.")
    parser.add_argument(
        "--data-source",
        choices=SUPPORTED_DATA_SOURCES,
        help="Named data source for default parquet/factor-store resolution.",
    )
    parser.add_argument("--parquet-dir", default=None, help="Input parquet directory.")
    parser.add_argument("--output-dir", default=None, help="Output factor-store directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for counting/writing.")
    parser.add_argument(
        "--label-horizons",
        help="Comma-separated label horizons to materialize, for example '1,5,10,20'. If omitted, use config/defaults.",
    )
    parser.set_defaults(incremental=True)
    parser.add_argument(
        "--incremental",
        dest="incremental",
        action="store_true",
        help="Reuse unchanged per-symbol feature shards and only recompute changed parquet files (default).",
    )
    parser.add_argument(
        "--full-rebuild",
        dest="incremental",
        action="store_false",
        help="Ignore reusable shards and rebuild all per-symbol feature shards.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_default_dimensions()
    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if args.data_source:
        cfg.setdefault("data", {})
        cfg["data"]["source"] = args.data_source
    label_horizons = resolve_label_horizons(cfg)
    if args.label_horizons:
        label_horizons = resolve_label_horizons(
            {"label": {"horizons": [int(item.strip()) for item in args.label_horizons.split(",") if item.strip()]}}
        )
    data_source = resolve_data_source_name(cfg)
    parquet_dir = args.parquet_dir or resolve_source_parquet_dir(cfg)
    out_dir = (
        args.output_dir
        or cfg.get("features", {}).get("factor_store_dir")
        or cfg.get("features", {}).get("cache_dir")
        or get_default_factor_store_dir(data_source, FULL_FACTOR_SPACE_NAME)
    )
    feature_names = get_full_factor_space_feature_names(data_source=data_source)
    alpha158_count = len(get_alpha158_feature_config()[1])
    lgbm_count = len(get_lgbm_purified_feature_names())
    temporal_count = len(get_temporal_factor_feature_names())
    technical_count = len(get_technical_factor_feature_names())
    tushare_count = len(get_tushare_factor_feature_names()) if data_source == "tushare" else 0

    print(
        "storage_format=parquet, "
        f"data_source={data_source}, "
        f"parquet_dir={parquet_dir}, "
        f"factor_space={FULL_FACTOR_SPACE_NAME}, "
        f"output={out_dir}, "
        f"incremental={args.incremental}, "
        f"label_horizons={label_horizons}"
    )
    print(
        "factor_groups="
        f"legacy158:{alpha158_count}, "
        f"lgbm_purified:{lgbm_count}, "
        f"temporal:{temporal_count}, "
        f"technical:{technical_count}, "
        f"tushare:{tushare_count}, "
        f"total:{len(feature_names)}"
    )
    generate_factor_store(
        parquet_dir=parquet_dir,
        output_dir=out_dir,
        workers=max(1, int(args.workers)),
        incremental=args.incremental,
        label_horizons=label_horizons,
        data_source=data_source,
    )


if __name__ == "__main__":
    main()
