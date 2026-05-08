"""Parity harness for factor-engine implementations.

The current production factor engine is the pandas implementation in
``src.gen_feature``.  This module makes that implementation an explicit
reference target so future Rust kernels can write a candidate parquet/CSV and
be compared against the same date/symbol/column semantics before promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_source import SUPPORTED_DATA_SOURCES
from src.gen_feature import _build_shard_frame_from_frame
from src.label_utils import get_label_column_name, get_legacy_label_column_name, resolve_label_horizons


KEY_COLUMNS = ["date", "symbol"]


def _read_frame(path: str | Path) -> pd.DataFrame:
    frame_path = Path(path)
    if frame_path.suffix.lower() == ".csv":
        return pd.read_csv(frame_path)
    return pd.read_parquet(frame_path)


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or None


def _resolve_symbol_frame(frame: pd.DataFrame, symbol: str | None) -> tuple[str, pd.DataFrame]:
    if "symbol" not in frame.columns:
        if not symbol:
            symbol = "UNKNOWN"
        resolved = frame.copy()
        resolved["symbol"] = str(symbol)
        return str(symbol), resolved

    symbols = sorted(str(value) for value in frame["symbol"].dropna().astype(str).unique())
    if symbol:
        selected = frame.loc[frame["symbol"].astype(str) == str(symbol)].copy()
        if selected.empty:
            raise ValueError(f"Symbol {symbol!r} is not present in the input frame.")
        return str(symbol), selected
    if len(symbols) != 1:
        raise ValueError("Input contains multiple symbols; pass --symbol for a single-symbol parity check.")
    return symbols[0], frame.copy()


def _sort_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    sorted_frame = frame.copy()
    if "date" not in sorted_frame.columns:
        raise ValueError("Feature parity frames must contain a date column.")
    if "symbol" not in sorted_frame.columns:
        raise ValueError("Feature parity frames must contain a symbol column.")
    sorted_frame["date"] = pd.to_datetime(sorted_frame["date"], errors="coerce")
    sorted_frame["symbol"] = sorted_frame["symbol"].astype(str)
    sorted_frame = sorted_frame.dropna(subset=["date", "symbol"])
    return sorted_frame.sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_python_feature_snapshot(
    frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    label_horizons: list[int] | None = None,
    data_source: str | None = None,
    feature_subset: list[str] | None = None,
) -> pd.DataFrame:
    resolved_symbol, symbol_frame = _resolve_symbol_frame(frame, symbol)
    horizons = resolve_label_horizons({"label": {"horizons": label_horizons}} if label_horizons is not None else {})
    built = _build_shard_frame_from_frame(
        symbol_frame,
        symbol=resolved_symbol,
        label_horizons=horizons,
        data_source=data_source,
    )
    if feature_subset:
        label_columns = [get_legacy_label_column_name(), *(get_label_column_name(horizon) for horizon in horizons)]
        requested_columns = [*KEY_COLUMNS, *label_columns, *feature_subset]
        missing = [column for column in requested_columns if column not in built.columns]
        if missing:
            raise ValueError(f"Reference Python snapshot is missing requested columns: {missing}")
        built = built.loc[:, requested_columns]
    return _sort_snapshot(built)


def compare_feature_snapshots(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    compare_columns: list[str] | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> dict[str, Any]:
    left = _sort_snapshot(reference)
    right = _sort_snapshot(candidate)
    result: dict[str, Any] = {
        "passed": False,
        "rows_reference": int(len(left)),
        "rows_candidate": int(len(right)),
        "columns_checked": 0,
        "missing_reference_columns": [],
        "missing_candidate_columns": [],
        "extra_reference_columns": [],
        "extra_candidate_columns": [],
        "key_mismatch": False,
        "mismatched_columns": [],
        "max_abs_diff": 0.0,
        "nan_mismatch_count": 0,
    }
    if len(left) != len(right):
        result["key_mismatch"] = True
        result["reason"] = "row_count_mismatch"
        return result

    left_keys = left[KEY_COLUMNS].copy()
    right_keys = right[KEY_COLUMNS].copy()
    if not left_keys.equals(right_keys):
        result["key_mismatch"] = True
        result["reason"] = "date_symbol_key_mismatch"
        return result

    reference_columns = [column for column in left.columns if column not in KEY_COLUMNS]
    candidate_columns = [column for column in right.columns if column not in KEY_COLUMNS]
    if compare_columns is None:
        columns = reference_columns
    else:
        columns = list(compare_columns)
    missing_reference = [column for column in columns if column not in left.columns]
    missing_candidate = [column for column in columns if column not in right.columns]
    result["missing_reference_columns"] = missing_reference
    result["missing_candidate_columns"] = missing_candidate
    result["extra_reference_columns"] = [column for column in reference_columns if column not in columns]
    result["extra_candidate_columns"] = [column for column in candidate_columns if column not in columns]
    if missing_reference or missing_candidate:
        result["reason"] = "missing_columns"
        return result

    mismatched_columns: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    nan_mismatch_count = 0
    for column in columns:
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=np.float64)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=np.float64)
        left_nan = np.isnan(left_values)
        right_nan = np.isnan(right_values)
        nan_mismatch = left_nan != right_nan
        finite_mask = ~(left_nan | right_nan)
        if finite_mask.any():
            abs_diff = np.abs(left_values[finite_mask] - right_values[finite_mask])
            column_max_abs_diff = float(np.max(abs_diff)) if abs_diff.size else 0.0
            close = np.isclose(
                left_values[finite_mask],
                right_values[finite_mask],
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
            finite_mismatch_count = int((~close).sum())
        else:
            column_max_abs_diff = 0.0
            finite_mismatch_count = 0
        column_nan_mismatch_count = int(nan_mismatch.sum())
        if column_nan_mismatch_count or finite_mismatch_count:
            mismatched_columns.append(
                {
                    "column": column,
                    "finite_mismatch_count": finite_mismatch_count,
                    "nan_mismatch_count": column_nan_mismatch_count,
                    "max_abs_diff": column_max_abs_diff,
                }
            )
        max_abs_diff = max(max_abs_diff, column_max_abs_diff)
        nan_mismatch_count += column_nan_mismatch_count

    result["columns_checked"] = int(len(columns))
    result["mismatched_columns"] = mismatched_columns
    result["max_abs_diff"] = float(max_abs_diff)
    result["nan_mismatch_count"] = int(nan_mismatch_count)
    result["passed"] = not mismatched_columns
    if not result["passed"]:
        result["reason"] = "value_mismatch"
    return result


def run_feature_engine_parity(
    *,
    input_parquet: str | Path,
    candidate_path: str | Path | None = None,
    symbol: str | None = None,
    label_horizons: list[int] | None = None,
    data_source: str | None = None,
    feature_subset: list[str] | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> dict[str, Any]:
    source_frame = _read_frame(input_parquet)
    reference = build_python_feature_snapshot(
        source_frame,
        symbol=symbol,
        label_horizons=label_horizons,
        data_source=data_source,
        feature_subset=feature_subset,
    )
    candidate = _read_frame(candidate_path) if candidate_path else build_python_feature_snapshot(
        source_frame,
        symbol=symbol,
        label_horizons=label_horizons,
        data_source=data_source,
        feature_subset=feature_subset,
    )
    compare_columns = feature_subset or [
        column
        for column in reference.columns
        if column not in KEY_COLUMNS
    ]
    result = compare_feature_snapshots(
        reference,
        candidate,
        compare_columns=compare_columns,
        atol=atol,
        rtol=rtol,
    )
    result["reference_engine"] = "python"
    result["candidate_engine"] = "file" if candidate_path else "python"
    result["input_parquet"] = str(input_parquet)
    result["candidate_path"] = "" if candidate_path is None else str(candidate_path)
    result["data_source"] = data_source or ""
    result["symbol"] = symbol or ""
    return result


def _parse_label_horizons(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare factor-engine output against the Python reference engine.")
    parser.add_argument("--input-parquet", required=True, help="Single-symbol source parquet or bucket shard parquet.")
    parser.add_argument("--candidate-path", help="Candidate feature snapshot parquet/CSV. If omitted, checks Python determinism.")
    parser.add_argument("--symbol", help="Symbol to select when the input parquet contains multiple symbols.")
    parser.add_argument("--label-horizons", help="Comma-separated label horizons, for example '1,5,10,20'.")
    parser.add_argument("--data-source", choices=SUPPORTED_DATA_SOURCES, help="Optional data source semantics.")
    parser.add_argument("--feature-subset", help="Comma-separated feature columns to compare.")
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance for numeric comparison.")
    parser.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance for numeric comparison.")
    parser.add_argument("--output-json", help="Optional path for the parity result JSON.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = run_feature_engine_parity(
        input_parquet=args.input_parquet,
        candidate_path=args.candidate_path,
        symbol=args.symbol,
        label_horizons=_parse_label_horizons(args.label_horizons),
        data_source=args.data_source,
        feature_subset=_parse_csv_list(args.feature_subset),
        atol=args.atol,
        rtol=args.rtol,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
