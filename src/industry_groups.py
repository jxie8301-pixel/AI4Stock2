"""Industry-group loading helpers shared by training, evaluation, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_source import resolve_data_source_name


@dataclass(frozen=True)
class IndustryGroupLoadResult:
    path: Path
    status: str
    groups: pd.Series | None
    requested_count: int = 0
    mapped_count: int = 0
    missing_count: int = 0
    error: str | None = None


def resolve_symbol_cache_path(cfg: dict[str, Any] | None = None) -> Path:
    data_source = resolve_data_source_name(cfg)
    return Path("data") / data_source / "raw" / "meta" / "symbol_cache.parquet"


def _normalize_local_symbol(value: object) -> str:
    text = str(value).strip()
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def load_instrument_industry_group_result(
    cfg: dict[str, Any] | None,
    *,
    instruments: pd.Index,
    symbol_cache_path: str | Path | None = None,
) -> IndustryGroupLoadResult:
    path = Path(symbol_cache_path) if symbol_cache_path is not None else resolve_symbol_cache_path(cfg)
    requested = pd.Index([str(value).strip() for value in instruments], dtype=object)
    requested = pd.Index([value for value in requested if value], dtype=object)
    if requested.empty:
        return IndustryGroupLoadResult(path=path, status="empty_instruments", groups=None)

    if not path.exists():
        return IndustryGroupLoadResult(path=path, status="missing_file", groups=None, requested_count=len(requested))

    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return IndustryGroupLoadResult(
            path=path,
            status="read_error",
            groups=None,
            requested_count=len(requested),
            error=f"{type(exc).__name__}: {exc}",
        )

    if frame.empty:
        return IndustryGroupLoadResult(path=path, status="empty_file", groups=None, requested_count=len(requested))

    symbol_column = "local_symbol" if "local_symbol" in frame.columns else "symbol" if "symbol" in frame.columns else ""
    required_columns = {symbol_column, "industry"} if symbol_column else {"local_symbol", "industry"}
    missing_columns = sorted(col for col in required_columns if col and col not in frame.columns)
    if not symbol_column or missing_columns:
        return IndustryGroupLoadResult(
            path=path,
            status="missing_columns",
            groups=None,
            requested_count=len(requested),
            error=", ".join(missing_columns or ["local_symbol"]),
        )

    normalized_symbols = frame[symbol_column].map(_normalize_local_symbol)
    industries = frame["industry"].fillna("").replace("", pd.NA)
    lookup = (
        pd.DataFrame({"symbol": normalized_symbols, "industry": industries})
        .dropna(subset=["symbol"])
        .drop_duplicates("symbol", keep="last")
        .set_index("symbol")["industry"]
    )
    normalized_requested = pd.Index([_normalize_local_symbol(value) for value in requested], dtype=object)
    groups = pd.Series(
        lookup.reindex(normalized_requested).to_numpy(copy=False),
        index=requested,
        name="industry",
        dtype=object,
    )
    mapped_count = int(groups.notna().sum())
    missing_count = int(len(groups) - mapped_count)
    if mapped_count <= 0:
        return IndustryGroupLoadResult(
            path=path,
            status="no_mapped_groups",
            groups=None,
            requested_count=len(requested),
            mapped_count=0,
            missing_count=missing_count,
        )
    return IndustryGroupLoadResult(
        path=path,
        status="ok",
        groups=groups,
        requested_count=len(requested),
        mapped_count=mapped_count,
        missing_count=missing_count,
    )


def load_instrument_industry_groups(
    cfg: dict[str, Any] | None,
    *,
    instruments: pd.Index,
    required: bool = False,
    symbol_cache_path: str | Path | None = None,
) -> pd.Series | None:
    result = load_instrument_industry_group_result(
        cfg,
        instruments=instruments,
        symbol_cache_path=symbol_cache_path,
    )
    if result.groups is not None:
        return result.groups
    if required:
        detail = f": {result.error}" if result.error else ""
        raise ValueError(f"Industry group mapping unavailable ({result.status}) at {result.path}{detail}")
    return None
