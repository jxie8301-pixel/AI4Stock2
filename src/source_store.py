"""Parquet-backed source-store loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from zlib import crc32

from src.parquet_io import load_dataset_frame


SOURCE_BUCKET_DIRNAME = "buckets"
SOURCE_BUCKET_MANIFEST_FILENAME = "manifest.parquet"
SOURCE_META_FILENAME = "meta.json"
DEFAULT_SOURCE_STORAGE_LAYOUT = "symbol_shards"


def stable_bucket_id(symbol: str, bucket_count: int) -> int:
    return crc32(str(symbol).encode("utf-8")) % max(1, int(bucket_count))


def bucket_path(store_dir: str | Path, bucket_id: int) -> Path:
    return Path(store_dir) / SOURCE_BUCKET_DIRNAME / f"part-{int(bucket_id):04d}.parquet"


def extract_bucket_id_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    suffix = stem.split("-")[-1]
    if not suffix.isdigit():
        raise ValueError(f"Cannot parse bucket id from path: {path}")
    return int(suffix)


def load_source_store_metadata(store_dir: str | Path) -> dict[str, Any] | None:
    meta_path = Path(store_dir) / SOURCE_META_FILENAME
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_source_storage_layout(store_dir: str | Path) -> str:
    store_path = Path(store_dir)
    meta = load_source_store_metadata(store_path)
    layout = str((meta or {}).get("storage_layout") or "").strip()
    if layout:
        return layout
    if (store_path / SOURCE_BUCKET_DIRNAME).exists() and (store_path / SOURCE_BUCKET_MANIFEST_FILENAME).exists():
        return "bucket_shards"
    return DEFAULT_SOURCE_STORAGE_LAYOUT


def uses_bucket_shards(store_dir: str | Path) -> bool:
    return detect_source_storage_layout(store_dir) == "bucket_shards"


def get_bucket_dir(store_dir: str | Path) -> Path:
    bucket_dir = Path(store_dir) / SOURCE_BUCKET_DIRNAME
    if not bucket_dir.exists():
        raise FileNotFoundError(f"Source bucket directory does not exist: {bucket_dir}")
    return bucket_dir


def get_manifest_path(store_dir: str | Path) -> Path:
    manifest_path = Path(store_dir) / SOURCE_BUCKET_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest does not exist: {manifest_path}")
    return manifest_path


def load_source_manifest(
    store_dir: str | Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    return pd.read_parquet(get_manifest_path(store_dir), columns=columns)


def list_bucket_paths(store_dir: str | Path) -> list[Path]:
    store_path = Path(store_dir)
    meta = load_source_store_metadata(store_path) or {}
    bucket_ids = [int(value) for value in meta.get("bucket_ids", []) if str(value).strip()]
    if bucket_ids:
        return [
            bucket_path(store_path, bucket_id)
            for bucket_id in sorted(bucket_ids)
            if bucket_path(store_path, bucket_id).exists()
        ]
    return sorted(get_bucket_dir(store_path).glob("part-*.parquet"))


def _load_symbol_shard_frame(
    *,
    store_dir: Path,
    selected_columns: list[str],
    date_start: str | pd.Timestamp | None,
    date_end: str | pd.Timestamp | None,
    allowed_symbols: set[str] | None,
) -> pd.DataFrame:
    symbols = sorted(allowed_symbols) if allowed_symbols is not None else sorted(path.stem for path in store_dir.glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        path = store_dir / f"{symbol}.parquet"
        if not path.exists():
            continue
        schema_names = set(pq.read_schema(path).names)
        if "date" not in schema_names:
            raise ValueError(f"Source shard missing required column 'date': {path}")
        missing_columns = [
            col
            for col in selected_columns
            if col != "symbol" and col not in schema_names
        ]
        if missing_columns:
            raise ValueError(f"Source shard missing requested columns {missing_columns}: {path}")
        read_columns = [col for col in selected_columns if col != "symbol"]
        frame = pd.read_parquet(path, columns=read_columns)
        if frame.empty:
            continue
        if "symbol" not in frame.columns:
            frame["symbol"] = str(symbol)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        if date_start is not None:
            frame = frame.loc[frame["date"] >= pd.Timestamp(date_start)]
        if date_end is not None:
            frame = frame.loc[frame["date"] <= pd.Timestamp(date_end)]
        if frame.empty:
            continue
        frame["symbol"] = frame["symbol"].astype(str)
        for col in selected_columns:
            if col not in frame.columns:
                frame[col] = pd.NA
        frames.append(frame[selected_columns].copy())
    if not frames:
        return pd.DataFrame(columns=selected_columns)
    return pd.concat(frames, ignore_index=True)


def load_source_frame(
    *,
    store_dir: str | Path,
    columns: list[str],
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
    symbols: list[str] | set[str] | None = None,
    sort_by: tuple[str, str] = ("date", "symbol"),
    progress_desc: str | None = None,
) -> pd.DataFrame:
    store_path = Path(store_dir)
    if not store_path.exists():
        raise FileNotFoundError(f"Source store does not exist: {store_path}")

    selected_columns = list(dict.fromkeys(["date", "symbol", *columns]))
    allowed_symbols = None if symbols is None else {str(symbol) for symbol in symbols if str(symbol).strip()}

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

    layout = detect_source_storage_layout(store_path)
    if layout == "bucket_shards":
        fragment_paths: list[str] | None = None
        if allowed_symbols is not None:
            manifest = load_source_manifest(store_path, columns=["symbol", "bucket_id"])
            if manifest.empty:
                return pd.DataFrame(columns=selected_columns)
            manifest["symbol"] = manifest["symbol"].astype(str)
            matched = manifest.loc[manifest["symbol"].isin(allowed_symbols), "bucket_id"]
            if matched.empty:
                return pd.DataFrame(columns=selected_columns)
            bucket_ids = sorted({int(value) for value in matched.tolist()})
            symbol_filter = ds.field("symbol").isin(sorted(allowed_symbols))
            scan_filter = symbol_filter if scan_filter is None else scan_filter & symbol_filter
        else:
            bucket_ids = [extract_bucket_id_from_path(path) for path in list_bucket_paths(store_path)]
        fragment_paths = [
            str(bucket_path(store_path, bucket_id))
            for bucket_id in bucket_ids
            if bucket_path(store_path, bucket_id).exists()
        ]
        frame = load_dataset_frame(
            dataset_root=get_bucket_dir(store_path),
            selected_columns=selected_columns,
            scan_filter=scan_filter,
            fragment_paths=fragment_paths,
            progress_desc=progress_desc,
        )
    else:
        frame = _load_symbol_shard_frame(
            store_dir=store_path,
            selected_columns=selected_columns,
            date_start=date_start,
            date_end=date_end,
            allowed_symbols=allowed_symbols,
        )

    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(list(sort_by)).reset_index(drop=True)
