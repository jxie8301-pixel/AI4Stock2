"""Native universe loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE_DIR = Path("data/universes")


def _normalize_symbol(symbol: str) -> str:
    return "".join(ch for ch in str(symbol) if ch.isdigit())


def resolve_universe_path(universe_name: str, universe_dir: str | Path = DEFAULT_UNIVERSE_DIR) -> Path:
    base_dir = Path(universe_dir)
    candidates = [
        base_dir / universe_name,
        base_dir / f"{universe_name}.txt",
        base_dir / f"{universe_name}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Native universe file not found for '{universe_name}' under {base_dir}")


def load_universe_table(universe_name: str, universe_dir: str | Path = DEFAULT_UNIVERSE_DIR) -> pd.DataFrame:
    path = resolve_universe_path(universe_name, universe_dir=universe_dir)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, dtype=str)
    else:
        table = pd.read_csv(path, sep="\t", header=None, dtype=str)

    if table.shape[1] == 1:
        table.columns = ["symbol"]
        table["start_date"] = pd.Timestamp.min.normalize()
        table["end_date"] = pd.Timestamp.max.normalize()
    elif table.shape[1] >= 3:
        table = table.iloc[:, :3].copy()
        table.columns = ["symbol", "start_date", "end_date"]
    else:
        raise ValueError(f"Unsupported universe format in {path}")

    table["symbol"] = table["symbol"].map(_normalize_symbol)
    table = table[table["symbol"] != ""].copy()
    table["start_date"] = pd.to_datetime(table["start_date"], errors="coerce").fillna(pd.Timestamp.min.normalize())
    table["end_date"] = pd.to_datetime(table["end_date"], errors="coerce").fillna(pd.Timestamp.max.normalize())
    return table


def build_universe_mask(
    dates_ns: np.ndarray,
    symbol_ids: np.ndarray,
    symbol_to_id: dict[str, int],
    universe_name: str,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
) -> np.ndarray:
    if universe_name == "all":
        return np.ones(len(symbol_ids), dtype=bool)

    table = load_universe_table(universe_name, universe_dir=universe_dir)
    normalized_symbol_to_ids: dict[str, list[int]] = {}
    for sym_key, sid in symbol_to_id.items():
        normalized_symbol_to_ids.setdefault(_normalize_symbol(sym_key), []).append(int(sid))

    intervals_by_sid: dict[int, list[tuple[int, int]]] = {}
    for _, row in table.iterrows():
        symbol = row["symbol"]
        start_ns = int(pd.Timestamp(row["start_date"]).value)
        end_ns = int(pd.Timestamp(row["end_date"]).value)
        for sid in normalized_symbol_to_ids.get(symbol, []):
            intervals_by_sid.setdefault(sid, []).append((start_ns, end_ns))

    if not intervals_by_sid:
        raise ValueError(f"No symbols from universe '{universe_name}' matched the native cache symbol map.")

    mask = np.zeros(len(symbol_ids), dtype=bool)
    for sid, intervals in intervals_by_sid.items():
        sid_mask = symbol_ids == sid
        if not np.any(sid_mask):
            continue
        sid_dates = dates_ns[sid_mask]
        sid_valid = np.zeros(sid_dates.shape[0], dtype=bool)
        for start_ns, end_ns in intervals:
            sid_valid |= (sid_dates >= start_ns) & (sid_dates <= end_ns)
        mask[sid_mask] = sid_valid

    return mask


def build_universe_frame_mask(
    dates: pd.Series | pd.DatetimeIndex,
    symbols: pd.Series | list[str],
    universe_name: str,
    universe_dir: str | Path = DEFAULT_UNIVERSE_DIR,
) -> np.ndarray:
    if universe_name == "all":
        return np.ones(len(symbols), dtype=bool)

    table = load_universe_table(universe_name, universe_dir=universe_dir)
    intervals_by_symbol: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for _, row in table.iterrows():
        intervals_by_symbol.setdefault(str(row["symbol"]), []).append((row["start_date"], row["end_date"]))

    dates_arr = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    symbol_arr = pd.Series(symbols).astype(str).map(_normalize_symbol).reset_index(drop=True)
    mask = np.zeros(len(symbol_arr), dtype=bool)

    for symbol, intervals in intervals_by_symbol.items():
        symbol_mask = symbol_arr == symbol
        if not np.any(symbol_mask):
            continue
        symbol_dates = dates_arr[symbol_mask]
        symbol_valid = np.zeros(symbol_dates.shape[0], dtype=bool)
        for start_date, end_date in intervals:
            symbol_valid |= (symbol_dates >= start_date) & (symbol_dates <= end_date)
        mask[np.where(symbol_mask)[0]] = symbol_valid

    if not np.any(mask):
        raise ValueError(f"No rows matched universe '{universe_name}' in the loaded factor frame.")

    return mask
