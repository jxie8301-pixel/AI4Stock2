"""Parquet-backed factor store loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

from src.label_utils import get_label_column_name, get_legacy_label_column_name, sanitize_label_series
from src.native_universe import build_universe_frame_mask, load_universe_table
from src.parquet_io import load_dataset_frame


DEFAULT_FACTOR_STORE_DIR = Path("data/factor_store/full_factor_space")


def load_factor_store_metadata(store_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(store_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Parquet factor store metadata missing: {meta_path}. "
            "Please run `uv run python -m src.gen_feature --workers 24` first."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_storage_layout(meta: dict[str, Any]) -> str:
    return str(meta.get("storage_layout") or "symbol_shards")


def _get_symbol_shards_dir(store_dir: str | Path) -> Path:
    shards_dir = Path(store_dir) / "shards"
    if not shards_dir.exists():
        raise FileNotFoundError(
            f"Parquet factor store shard directory missing: {shards_dir}. "
            "Please run `uv run python -m src.gen_feature --workers 24` first."
        )
    return shards_dir


def _get_bucket_shards_dir(store_dir: str | Path) -> Path:
    bucket_dir = Path(store_dir) / "buckets"
    if not bucket_dir.exists():
        raise FileNotFoundError(
            f"Parquet factor store bucket directory missing: {bucket_dir}. "
            "Please run `uv run python -m src.gen_feature --workers 24` first."
        )
    return bucket_dir


def _get_bucket_manifest_path(store_dir: str | Path) -> Path:
    manifest_path = Path(store_dir) / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Parquet factor store bucket manifest missing: {manifest_path}. "
            "Please run `uv run python -m src.gen_feature --workers 24` first."
        )
    return manifest_path


def _load_bucket_manifest(
    store_dir: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    manifest_path = _get_bucket_manifest_path(store_dir)
    return pd.read_parquet(manifest_path, columns=columns)


def _filter_available_dates_by_universe(
    dates: pd.DatetimeIndex,
    *,
    universe_name: str,
    universe_dir: str | Path,
) -> pd.DatetimeIndex:
    if universe_name == "all" or dates.empty:
        return dates

    table = load_universe_table(universe_name, universe_dir=universe_dir)
    mask = pd.Series(False, index=pd.RangeIndex(len(dates)))
    dates_series = pd.Series(dates)
    for _, row in table.iterrows():
        mask |= (dates_series >= row["start_date"]) & (dates_series <= row["end_date"])
    return pd.DatetimeIndex(dates_series.loc[mask].drop_duplicates().sort_values())


def _resolve_requested_label_column(meta: dict[str, Any], requested_label_column: str | None) -> str:
    requested = str(requested_label_column or get_legacy_label_column_name())
    available_label_columns = {
        str(item.get("column"))
        for item in meta.get("label_columns", [])
        if isinstance(item, dict) and item.get("column")
    }
    if not available_label_columns:
        available_label_columns = {str(meta.get("default_label_column") or get_legacy_label_column_name())}

    if requested in available_label_columns:
        return requested

    if requested == get_label_column_name(1) and get_legacy_label_column_name() in available_label_columns:
        return get_legacy_label_column_name()

    available_preview = ", ".join(sorted(available_label_columns))
    raise ValueError(
        f"Requested label column '{requested}' is not available in factor store. "
        f"Available label columns: {available_preview}. "
        "If you changed label horizons, regenerate the factor store."
    )


def load_factor_frame(
    *,
    store_dir: str | Path,
    columns: list[str],
    label_column: str | None = None,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
    universe_name: str = "all",
    universe_dir: str | Path = "data/universes",
    sort_by: tuple[str, str] = ("date", "symbol"),
    progress_desc: str | None = None,
) -> pd.DataFrame:
    meta = load_factor_store_metadata(store_dir)
    storage_layout = _get_storage_layout(meta)
    actual_label_column = _resolve_requested_label_column(meta, label_column)
    selected_columns = ["date", "symbol", actual_label_column, *columns]
    allowed_symbols = None
    if universe_name != "all":
        universe_table = load_universe_table(universe_name, universe_dir=universe_dir)
        allowed_symbols = set(universe_table["symbol"].astype(str))

    filters = []
    if date_start is not None:
        filters.append(ds.field("date") >= pd.Timestamp(date_start))
    if date_end is not None:
        filters.append(ds.field("date") <= pd.Timestamp(date_end))

    scan_filter = None
    if filters:
        scan_filter = filters[0]
        for extra_filter in filters[1:]:
            scan_filter = scan_filter & extra_filter

    dataset_root: Path
    fragment_paths: list[str] | None = None
    if storage_layout == "bucket_shards":
        dataset_root = _get_bucket_shards_dir(store_dir)
        if allowed_symbols is not None:
            manifest = _load_bucket_manifest(store_dir, columns=["symbol", "bucket_id"])
            manifest["symbol"] = manifest["symbol"].astype(str)
            matched = manifest.loc[manifest["symbol"].isin(allowed_symbols), "bucket_id"]
            if matched.empty:
                return pd.DataFrame(columns=selected_columns)
            bucket_ids = sorted({int(value) for value in matched.tolist()})
            symbol_filter = ds.field("symbol").isin(sorted(allowed_symbols))
            scan_filter = symbol_filter if scan_filter is None else scan_filter & symbol_filter
        else:
            bucket_ids = [int(value) for value in meta.get("bucket_ids", [])]
            if not bucket_ids:
                bucket_ids = sorted(
                    int(path.stem.split("-")[-1]) for path in dataset_root.glob("part-*.parquet") if path.stem.split("-")[-1].isdigit()
                )
        fragment_paths = [
            str(dataset_root / f"part-{bucket_id:04d}.parquet")
            for bucket_id in bucket_ids
            if (dataset_root / f"part-{bucket_id:04d}.parquet").exists()
        ]
    else:
        dataset_root = _get_symbol_shards_dir(store_dir)

    frame = load_dataset_frame(
        dataset_root=dataset_root,
        selected_columns=selected_columns,
        scan_filter=scan_filter,
        progress_desc=progress_desc,
        fragment_paths=fragment_paths,
        allowed_symbols=allowed_symbols,
    )
    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"])
    if actual_label_column != get_legacy_label_column_name():
        frame = frame.rename(columns={actual_label_column: get_legacy_label_column_name()})
    frame[get_legacy_label_column_name()] = sanitize_label_series(frame[get_legacy_label_column_name()])

    if universe_name != "all":
        mask = build_universe_frame_mask(
            dates=frame["date"],
            symbols=frame["symbol"],
            universe_name=universe_name,
            universe_dir=universe_dir,
        )
        frame = frame.loc[mask].copy()

    if frame.empty:
        return frame

    frame = frame.sort_values(list(sort_by)).reset_index(drop=True)
    return frame


def load_available_dates(
    *,
    store_dir: str | Path,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
    universe_name: str = "all",
    universe_dir: str | Path = "data/universes",
    progress_desc: str | None = None,
) -> pd.DatetimeIndex:
    meta = load_factor_store_metadata(store_dir)
    cached_dates = meta.get("available_dates")
    if cached_dates:
        dates = pd.DatetimeIndex(pd.to_datetime(cached_dates))
        if date_start is not None:
            dates = dates[dates >= pd.Timestamp(date_start)]
        if date_end is not None:
            dates = dates[dates <= pd.Timestamp(date_end)]
        return _filter_available_dates_by_universe(
            dates,
            universe_name=universe_name,
            universe_dir=universe_dir,
        )

    frame = load_factor_frame(
        store_dir=store_dir,
        columns=[],
        date_start=date_start,
        date_end=date_end,
        universe_name=universe_name,
        universe_dir=universe_dir,
        sort_by=("date", "symbol"),
        progress_desc=progress_desc,
    )
    if frame.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(sorted(frame["date"].drop_duplicates()))
