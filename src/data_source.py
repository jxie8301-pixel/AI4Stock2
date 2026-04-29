"""Data-source resolution for native parquet and factor-store paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_DATA_SOURCE = "akshare"
SUPPORTED_DATA_SOURCES = ("akshare", "tushare")

SOURCE_PARQUET_DIRS = {
    "akshare": "data/processed/combined",
    "tushare": "data/tushare/source",
}


def normalize_data_source_name(value: str | None) -> str:
    source = str(value or DEFAULT_DATA_SOURCE).strip().lower()
    alias_map = {
        "eastmoney": "akshare",
        "em": "akshare",
    }
    source = alias_map.get(source, source)
    if source not in SUPPORTED_DATA_SOURCES:
        raise ValueError(
            f"Unsupported data source: {source}. "
            f"Available: {', '.join(SUPPORTED_DATA_SOURCES)}"
        )
    return source


def get_default_parquet_dir(data_source: str) -> str:
    normalized = normalize_data_source_name(data_source)
    return SOURCE_PARQUET_DIRS[normalized]


def get_default_factor_store_dir(
    data_source: str,
    factor_store_name: str = "full_factor_space",
) -> str:
    normalized = normalize_data_source_name(data_source)
    factor_store_name = str(factor_store_name).strip() or "full_factor_space"
    if normalized == "akshare":
        return str(Path("data/factor_store") / factor_store_name)
    return str(Path("data/factor_store") / f"{normalized}_{factor_store_name}")


def resolve_data_source_name(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or {}
    data_cfg = cfg.get("data", {}) or {}
    return normalize_data_source_name(data_cfg.get("source"))


def resolve_source_parquet_dir(cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or {}
    data_cfg = cfg.get("data", {}) or {}
    explicit = str(data_cfg.get("parquet_dir") or "").strip()
    if explicit:
        return explicit
    return get_default_parquet_dir(resolve_data_source_name(cfg))
