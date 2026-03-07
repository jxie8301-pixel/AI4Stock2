"""Unified feature pipeline without qlib.

This module combines:
1) Alpha158/Alpha360 definitions.
2) Alpha158/Alpha360 value computation on pandas dataframe.
3) Parquet -> memmap(.npy) cache generation.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import percentileofscore
from tqdm import tqdm

EPS = 1e-12


DEFAULT_ALPHA158_CONFIG: dict[str, Any] = {
    "kbar": {},
    "price": {
        "windows": [0],
        "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
    },
    "rolling": {},
}


def get_alpha158_feature_config(config: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Return Alpha158 expression fields and names (qlib-compatible)."""
    cfg = deepcopy(DEFAULT_ALPHA158_CONFIG)
    if config is not None:
        cfg.update(config)

    fields: list[str] = []
    names: list[str] = []

    if "kbar" in cfg:
        fields += [
            "($close-$open)/$open",
            "($high-$low)/$open",
            "($close-$open)/($high-$low+1e-12)",
            "($high-Greater($open, $close))/$open",
            "($high-Greater($open, $close))/($high-$low+1e-12)",
            "(Less($open, $close)-$low)/$open",
            "(Less($open, $close)-$low)/($high-$low+1e-12)",
            "(2*$close-$high-$low)/$open",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
        ]
        names += ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]

    if "price" in cfg:
        windows = cfg["price"].get("windows", range(5))
        feature = cfg["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
        for field in feature:
            field = field.lower()
            fields += [f"Ref(${field}, {d})/$close" if d != 0 else f"${field}/$close" for d in windows]
            names += [field.upper() + str(d) for d in windows]

    if "volume" in cfg:
        windows = cfg["volume"].get("windows", range(5))
        fields += [f"Ref($volume, {d})/($volume+1e-12)" if d != 0 else "$volume/($volume+1e-12)" for d in windows]
        names += [f"VOLUME{d}" for d in windows]

    if "rolling" in cfg:
        windows = cfg["rolling"].get("windows", [5, 10, 20, 30, 60])
        include = cfg["rolling"].get("include", None)
        exclude = cfg["rolling"].get("exclude", [])

        def use(op: str) -> bool:
            return op not in exclude and (include is None or op in include)

        if use("ROC"):
            fields += [f"Ref($close, {d})/$close" for d in windows]
            names += [f"ROC{d}" for d in windows]
        if use("MA"):
            fields += [f"Mean($close, {d})/$close" for d in windows]
            names += [f"MA{d}" for d in windows]
        if use("STD"):
            fields += [f"Std($close, {d})/$close" for d in windows]
            names += [f"STD{d}" for d in windows]
        if use("BETA"):
            fields += [f"Slope($close, {d})/$close" for d in windows]
            names += [f"BETA{d}" for d in windows]
        if use("RSQR"):
            fields += [f"Rsquare($close, {d})" for d in windows]
            names += [f"RSQR{d}" for d in windows]
        if use("RESI"):
            fields += [f"Resi($close, {d})/$close" for d in windows]
            names += [f"RESI{d}" for d in windows]
        if use("MAX"):
            fields += [f"Max($high, {d})/$close" for d in windows]
            names += [f"MAX{d}" for d in windows]
        if use("LOW"):
            fields += [f"Min($low, {d})/$close" for d in windows]
            names += [f"MIN{d}" for d in windows]
        if use("QTLU"):
            fields += [f"Quantile($close, {d}, 0.8)/$close" for d in windows]
            names += [f"QTLU{d}" for d in windows]
        if use("QTLD"):
            fields += [f"Quantile($close, {d}, 0.2)/$close" for d in windows]
            names += [f"QTLD{d}" for d in windows]
        if use("RANK"):
            fields += [f"Rank($close, {d})" for d in windows]
            names += [f"RANK{d}" for d in windows]
        if use("RSV"):
            fields += [f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)" for d in windows]
            names += [f"RSV{d}" for d in windows]
        if use("IMAX"):
            fields += [f"IdxMax($high, {d})/{d}" for d in windows]
            names += [f"IMAX{d}" for d in windows]
        if use("IMIN"):
            fields += [f"IdxMin($low, {d})/{d}" for d in windows]
            names += [f"IMIN{d}" for d in windows]
        if use("IMXD"):
            fields += [f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}" for d in windows]
            names += [f"IMXD{d}" for d in windows]
        if use("CORR"):
            fields += [f"Corr($close, Log($volume+1), {d})" for d in windows]
            names += [f"CORR{d}" for d in windows]
        if use("CORD"):
            fields += [f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})" for d in windows]
            names += [f"CORD{d}" for d in windows]
        if use("CNTP"):
            fields += [f"Mean($close>Ref($close, 1), {d})" for d in windows]
            names += [f"CNTP{d}" for d in windows]
        if use("CNTN"):
            fields += [f"Mean($close<Ref($close, 1), {d})" for d in windows]
            names += [f"CNTN{d}" for d in windows]
        if use("CNTD"):
            fields += [f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})" for d in windows]
            names += [f"CNTD{d}" for d in windows]
        if use("SUMP"):
            fields += [
                f"Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"SUMP{d}" for d in windows]
        if use("SUMN"):
            fields += [
                f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"SUMN{d}" for d in windows]
        if use("SUMD"):
            fields += [
                f"(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))"
                f"/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"SUMD{d}" for d in windows]
        if use("VMA"):
            fields += [f"Mean($volume, {d})/($volume+1e-12)" for d in windows]
            names += [f"VMA{d}" for d in windows]
        if use("VSTD"):
            fields += [f"Std($volume, {d})/($volume+1e-12)" for d in windows]
            names += [f"VSTD{d}" for d in windows]
        if use("WVMA"):
            fields += [
                f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
                for d in windows
            ]
            names += [f"WVMA{d}" for d in windows]
        if use("VSUMP"):
            fields += [
                f"Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"VSUMP{d}" for d in windows]
        if use("VSUMN"):
            fields += [
                f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"VSUMN{d}" for d in windows]
        if use("VSUMD"):
            fields += [
                f"(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))"
                f"/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)"
                for d in windows
            ]
            names += [f"VSUMD{d}" for d in windows]

    return fields, names


def get_alpha360_feature_config() -> tuple[list[str], list[str]]:
    """Return Alpha360 expression fields and names (qlib-compatible)."""
    fields: list[str] = []
    names: list[str] = []

    for i in range(59, 0, -1):
        fields.append(f"Ref($close, {i})/$close")
        names.append(f"CLOSE{i}")
    fields.append("$close/$close")
    names.append("CLOSE0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($open, {i})/$close")
        names.append(f"OPEN{i}")
    fields.append("$open/$close")
    names.append("OPEN0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($high, {i})/$close")
        names.append(f"HIGH{i}")
    fields.append("$high/$close")
    names.append("HIGH0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($low, {i})/$close")
        names.append(f"LOW{i}")
    fields.append("$low/$close")
    names.append("LOW0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($vwap, {i})/$close")
        names.append(f"VWAP{i}")
    fields.append("$vwap/$close")
    names.append("VWAP0")

    for i in range(59, 0, -1):
        fields.append(f"Ref($volume, {i})/($volume+1e-12)")
        names.append(f"VOLUME{i}")
    fields.append("$volume/($volume+1e-12)")
    names.append("VOLUME0")

    return fields, names


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").set_index("date")
    else:
        out = out.sort_index()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "vwap" not in out.columns:
        if "amount" in out.columns:
            vol = out["volume"].replace(0, np.nan)
            out["vwap"] = (out["amount"] / vol).replace([np.inf, -np.inf], np.nan)
            out["vwap"] = out["vwap"].fillna(out["close"])
        else:
            out["vwap"] = out["close"]

    return out


def _rolling_rank_pct(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window, min_periods=1)

    def _rank(x: np.ndarray) -> float:
        if np.isnan(x[-1]):
            return np.nan
        x1 = x[~np.isnan(x)]
        if x1.size == 0:
            return np.nan
        return percentileofscore(x1, x1[-1]) / 100.0

    return roll.apply(_rank, raw=True)


def _rolling_idxmax(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(lambda x: float(np.argmax(x) + 1), raw=True)


def _rolling_idxmin(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(lambda x: float(np.argmin(x) + 1), raw=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(y: np.ndarray) -> float:
        mask = ~np.isnan(y)
        yv = y[mask]
        if yv.size < 2:
            return np.nan
        xv = np.arange(1, yv.size + 1, dtype=np.float64)
        xm = xv.mean()
        ym = yv.mean()
        denom = np.sum((xv - xm) ** 2)
        if np.isclose(denom, 0):
            return np.nan
        return float(np.sum((xv - xm) * (yv - ym)) / denom)

    return series.rolling(window, min_periods=1).apply(_slope, raw=True)


def _rolling_rsquare(series: pd.Series, window: int) -> pd.Series:
    def _rsq(y: np.ndarray) -> float:
        mask = ~np.isnan(y)
        yv = y[mask]
        if yv.size < 2:
            return np.nan
        xv = np.arange(1, yv.size + 1, dtype=np.float64)
        sx = xv.std(ddof=0)
        sy = yv.std(ddof=0)
        if np.isclose(sx, 0) or np.isclose(sy, 0):
            return np.nan
        r = np.corrcoef(xv, yv)[0, 1]
        return float(r * r)

    res = series.rolling(window, min_periods=1).apply(_rsq, raw=True)
    std = series.rolling(window, min_periods=1).std()
    res[np.isclose(std, 0, atol=2e-5)] = np.nan
    return res


def _rolling_resi(series: pd.Series, window: int) -> pd.Series:
    def _resi(y: np.ndarray) -> float:
        mask = ~np.isnan(y)
        yv = y[mask]
        if yv.size < 2:
            return np.nan
        xv = np.arange(1, yv.size + 1, dtype=np.float64)
        xm = xv.mean()
        ym = yv.mean()
        denom = np.sum((xv - xm) ** 2)
        if np.isclose(denom, 0):
            return np.nan
        beta = np.sum((xv - xm) * (yv - ym)) / denom
        alpha = ym - beta * xm
        return float(yv[-1] - (alpha + beta * xv[-1]))

    return series.rolling(window, min_periods=1).apply(_resi, raw=True)


def _rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    res = left.rolling(window, min_periods=1).corr(right)
    lstd = left.rolling(window, min_periods=1).std()
    rstd = right.rolling(window, min_periods=1).std()
    res[np.isclose(lstd, 0, atol=2e-5) | np.isclose(rstd, 0, atol=2e-5)] = np.nan
    return res


def compute_alpha360(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Alpha360 values (360 features) for one instrument."""
    base = _prepare_ohlcv(df)
    close = base["close"]
    volume = base["volume"]
    out: dict[str, pd.Series] = {}

    for i in range(59, 0, -1):
        out[f"CLOSE{i}"] = close.shift(i) / close
    out["CLOSE0"] = close / close

    for i in range(59, 0, -1):
        out[f"OPEN{i}"] = base["open"].shift(i) / close
    out["OPEN0"] = base["open"] / close

    for i in range(59, 0, -1):
        out[f"HIGH{i}"] = base["high"].shift(i) / close
    out["HIGH0"] = base["high"] / close

    for i in range(59, 0, -1):
        out[f"LOW{i}"] = base["low"].shift(i) / close
    out["LOW0"] = base["low"] / close

    for i in range(59, 0, -1):
        out[f"VWAP{i}"] = base["vwap"].shift(i) / close
    out["VWAP0"] = base["vwap"] / close

    for i in range(59, 0, -1):
        out[f"VOLUME{i}"] = volume.shift(i) / (volume + EPS)
    out["VOLUME0"] = volume / (volume + EPS)

    feat = pd.DataFrame(out, index=base.index)
    if feat.shape[1] != 360:
        raise ValueError(f"Alpha360 feature count mismatch: {feat.shape[1]}")
    return feat


def compute_alpha158(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Compute Alpha158 values for one instrument."""
    cfg = deepcopy(DEFAULT_ALPHA158_CONFIG)
    if config is not None:
        cfg.update(config)

    base = _prepare_ohlcv(df)
    open_ = base["open"]
    high = base["high"]
    low = base["low"]
    close = base["close"]
    vwap = base["vwap"]
    volume = base["volume"]

    out: dict[str, pd.Series] = {}

    if "kbar" in cfg:
        hl_range = high - low + EPS
        out["KMID"] = (close - open_) / open_
        out["KLEN"] = (high - low) / open_
        out["KMID2"] = (close - open_) / hl_range
        out["KUP"] = (high - np.maximum(open_, close)) / open_
        out["KUP2"] = (high - np.maximum(open_, close)) / hl_range
        out["KLOW"] = (np.minimum(open_, close) - low) / open_
        out["KLOW2"] = (np.minimum(open_, close) - low) / hl_range
        out["KSFT"] = (2 * close - high - low) / open_
        out["KSFT2"] = (2 * close - high - low) / hl_range

    if "price" in cfg:
        windows = cfg["price"].get("windows", range(5))
        features = cfg["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
        series_map = {"open": open_, "high": high, "low": low, "close": close, "vwap": vwap}
        for field in features:
            key = field.lower()
            s = series_map[key]
            for d in windows:
                out[f"{key.upper()}{d}"] = (s.shift(d) if d != 0 else s) / close

    if "volume" in cfg:
        windows = cfg["volume"].get("windows", range(5))
        for d in windows:
            out[f"VOLUME{d}"] = (volume.shift(d) if d != 0 else volume) / (volume + EPS)

    if "rolling" in cfg:
        windows = cfg["rolling"].get("windows", [5, 10, 20, 30, 60])
        include = cfg["rolling"].get("include", None)
        exclude = cfg["rolling"].get("exclude", [])

        def use(op: str) -> bool:
            return op not in exclude and (include is None or op in include)

        ref_close1 = close.shift(1)
        ref_vol1 = volume.shift(1)
        close_ret = close / ref_close1
        vol_ret_log = np.log(volume / ref_vol1 + 1)
        close_delta = close - ref_close1
        vol_delta = volume - ref_vol1

        for d in windows:
            if use("ROC"):
                out[f"ROC{d}"] = close.shift(d) / close
            if use("MA"):
                out[f"MA{d}"] = close.rolling(d, min_periods=1).mean() / close
            if use("STD"):
                out[f"STD{d}"] = close.rolling(d, min_periods=1).std() / close
            if use("BETA"):
                out[f"BETA{d}"] = _rolling_slope(close, d) / close
            if use("RSQR"):
                out[f"RSQR{d}"] = _rolling_rsquare(close, d)
            if use("RESI"):
                out[f"RESI{d}"] = _rolling_resi(close, d) / close
            if use("MAX"):
                out[f"MAX{d}"] = high.rolling(d, min_periods=1).max() / close
            if use("LOW"):
                out[f"MIN{d}"] = low.rolling(d, min_periods=1).min() / close
            if use("QTLU"):
                out[f"QTLU{d}"] = close.rolling(d, min_periods=1).quantile(0.8) / close
            if use("QTLD"):
                out[f"QTLD{d}"] = close.rolling(d, min_periods=1).quantile(0.2) / close
            if use("RANK"):
                out[f"RANK{d}"] = _rolling_rank_pct(close, d)
            if use("RSV"):
                out[f"RSV{d}"] = (close - low.rolling(d, min_periods=1).min()) / (
                    high.rolling(d, min_periods=1).max() - low.rolling(d, min_periods=1).min() + EPS
                )
            if use("IMAX"):
                out[f"IMAX{d}"] = _rolling_idxmax(high, d) / d
            if use("IMIN"):
                out[f"IMIN{d}"] = _rolling_idxmin(low, d) / d
            if use("IMXD"):
                out[f"IMXD{d}"] = (_rolling_idxmax(high, d) - _rolling_idxmin(low, d)) / d
            if use("CORR"):
                out[f"CORR{d}"] = _rolling_corr(close, np.log(volume + 1), d)
            if use("CORD"):
                out[f"CORD{d}"] = _rolling_corr(close_ret, vol_ret_log, d)
            if use("CNTP"):
                out[f"CNTP{d}"] = (close > ref_close1).rolling(d, min_periods=1).mean()
            if use("CNTN"):
                out[f"CNTN{d}"] = (close < ref_close1).rolling(d, min_periods=1).mean()
            if use("CNTD"):
                out[f"CNTD{d}"] = (
                    (close > ref_close1).rolling(d, min_periods=1).mean()
                    - (close < ref_close1).rolling(d, min_periods=1).mean()
                )
            if use("SUMP"):
                out[f"SUMP{d}"] = (
                    np.maximum(close_delta, 0).rolling(d, min_periods=1).sum()
                    / (np.abs(close_delta).rolling(d, min_periods=1).sum() + EPS)
                )
            if use("SUMN"):
                out[f"SUMN{d}"] = (
                    np.maximum(ref_close1 - close, 0).rolling(d, min_periods=1).sum()
                    / (np.abs(close_delta).rolling(d, min_periods=1).sum() + EPS)
                )
            if use("SUMD"):
                out[f"SUMD{d}"] = (
                    (
                        np.maximum(close_delta, 0).rolling(d, min_periods=1).sum()
                        - np.maximum(ref_close1 - close, 0).rolling(d, min_periods=1).sum()
                    )
                    / (np.abs(close_delta).rolling(d, min_periods=1).sum() + EPS)
                )
            if use("VMA"):
                out[f"VMA{d}"] = volume.rolling(d, min_periods=1).mean() / (volume + EPS)
            if use("VSTD"):
                out[f"VSTD{d}"] = volume.rolling(d, min_periods=1).std() / (volume + EPS)
            if use("WVMA"):
                w = np.abs(close_ret - 1) * volume
                out[f"WVMA{d}"] = w.rolling(d, min_periods=1).std() / (w.rolling(d, min_periods=1).mean() + EPS)
            if use("VSUMP"):
                out[f"VSUMP{d}"] = (
                    np.maximum(vol_delta, 0).rolling(d, min_periods=1).sum()
                    / (np.abs(vol_delta).rolling(d, min_periods=1).sum() + EPS)
                )
            if use("VSUMN"):
                out[f"VSUMN{d}"] = (
                    np.maximum(ref_vol1 - volume, 0).rolling(d, min_periods=1).sum()
                    / (np.abs(vol_delta).rolling(d, min_periods=1).sum() + EPS)
                )
            if use("VSUMD"):
                out[f"VSUMD{d}"] = (
                    (
                        np.maximum(vol_delta, 0).rolling(d, min_periods=1).sum()
                        - np.maximum(ref_vol1 - volume, 0).rolling(d, min_periods=1).sum()
                    )
                    / (np.abs(vol_delta).rolling(d, min_periods=1).sum() + EPS)
                )

    feat = pd.DataFrame(out, index=base.index)
    _, ordered_names = get_alpha158_feature_config(cfg)
    feat = feat.reindex(columns=ordered_names)

    if config is None and feat.shape[1] != 158:
        raise ValueError(f"Alpha158 feature count mismatch: {feat.shape[1]}")
    return feat


def build_open_to_open_label(df: pd.DataFrame) -> pd.Series:
    """Label: open_{t+2}/open_{t+1} - 1."""
    base = _prepare_ohlcv(df)
    return base["open"].shift(-2) / base["open"].shift(-1) - 1


def _index_to_epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert datetime index to int64 nanoseconds since epoch."""
    idx_ns = index.astype("datetime64[ns]")
    return idx_ns.view("i8")


def _to_panel_arrays(feat: pd.DataFrame, label: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert per-symbol feature frame into 2D panel rows."""
    y = label.reindex(feat.index)
    x2d = feat.to_numpy(dtype=np.float32, copy=False)
    y1d = y.to_numpy(dtype=np.float32, copy=False)
    date_ns = _index_to_epoch_ns(feat.index).astype(np.int64, copy=False)
    return x2d, y1d, date_ns


def _alpha_feature_count(alpha: str) -> int:
    if alpha == "158":
        return 158
    if alpha == "360":
        return 360
    raise ValueError(f"Unknown alpha: {alpha}")


def _compute_symbol_feat_label(file_path: str, alpha: str) -> tuple[str, pd.DataFrame, pd.Series]:
    """Load one parquet and compute features + label."""
    df = pd.read_parquet(file_path)
    symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) > 0 else Path(file_path).stem
    feat = compute_alpha158(df) if alpha == "158" else compute_alpha360(df)
    label = build_open_to_open_label(df)
    return symbol, feat, label


def _count_file_worker(
    file_path: str,
    alpha: str,
) -> tuple[str, int, int]:
    """Worker for panel-cache counting pass (fast metadata path)."""
    symbol = Path(file_path).stem
    try:
        meta = pq.read_metadata(file_path)
        n_rows = int(meta.num_rows)
    except Exception:
        # Fallback for corrupted/unusual parquet metadata.
        n_rows = int(len(pd.read_parquet(file_path, columns=["date"])))
    return file_path, symbol, _alpha_feature_count(alpha), n_rows


def _build_file_payload_worker(
    file_path: str,
    alpha: str,
) -> tuple[str, int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Worker for panel-cache payload pass."""
    symbol, feat, label = _compute_symbol_feat_label(file_path, alpha)
    payload = _to_panel_arrays(feat, label)
    return symbol, feat.shape[1], payload


def _write_panel_file_slice_process(
    file_path: str,
    alpha: str,
    start: int,
    count: int,
    symbol_id: int,
    x_path: str,
    y_path: str,
    date_path: str,
    symbol_path: str,
    total_rows: int,
    n_feat: int,
) -> int:
    """Process worker: compute one file and write to fixed memmap slice."""
    symbol, feat, label = _compute_symbol_feat_label(file_path, alpha)
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


def generate_panel_cache(
    parquet_dir: str = "data/processed/combined",
    output_dir: str = "data/cache/alpha158_panel",
    alpha: str = "158",
    workers: int = 1,
) -> dict[str, Any]:
    """Generate 2D panel cache arrays with no lookback/split."""
    pdir = Path(parquet_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(pdir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {pdir}")

    workers = max(1, int(workers))
    total_files = len(files)
    symbols: set[str] = set()
    n_feat: int | None = None
    total_count = 0
    file_rows: dict[str, tuple[str, int]] = {}

    print(f"[1/3] Counting panel rows from {total_files} parquet files (workers={workers})...")
    t0 = time.perf_counter()
    if workers == 1:
        pbar = tqdm(files, desc="counting", total=total_files, unit="file")
        for idx, fp in enumerate(pbar, start=1):
            file_path, symbol, file_n_feat, cnt = _count_file_worker(str(fp), alpha)
            symbols.add(symbol)
            n_feat = file_n_feat if n_feat is None else n_feat
            if n_feat != file_n_feat:
                raise RuntimeError(f"Feature dim mismatch: expected {n_feat}, got {file_n_feat} for {fp.name}")
            total_count += cnt
            file_rows[file_path] = (symbol, cnt)
            elapsed = time.perf_counter() - t0
            speed = idx / elapsed if elapsed > 0 else 0.0
            eta = (total_files - idx) / speed if speed > 0 else float("inf")
            pbar.set_postfix(rows=total_count, speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
    else:
        def _run_parallel_count(executor) -> None:
            nonlocal n_feat, total_count
            futures = [executor.submit(_count_file_worker, str(fp), alpha) for fp in files]
            pbar = tqdm(total=total_files, desc="counting", unit="file")
            for idx, fut in enumerate(as_completed(futures), start=1):
                file_path, symbol, file_n_feat, cnt = fut.result()
                symbols.add(symbol)
                n_feat = file_n_feat if n_feat is None else n_feat
                if n_feat != file_n_feat:
                    raise RuntimeError(f"Feature dim mismatch: expected {n_feat}, got {file_n_feat}")
                total_count += cnt
                file_rows[file_path] = (symbol, cnt)
                pbar.update(1)
                elapsed = time.perf_counter() - t0
                speed = idx / elapsed if elapsed > 0 else 0.0
                eta = (total_files - idx) / speed if speed > 0 else float("inf")
                pbar.set_postfix(rows=total_count, speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
            pbar.close()

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                print("  counting backend: process")
                _run_parallel_count(executor)
        except Exception as e:
            print(f"  process backend unavailable ({type(e).__name__}: {e}), fallback to thread backend")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                print("  counting backend: thread")
                _run_parallel_count(executor)

    if n_feat is None:
        raise RuntimeError("Failed to infer feature count.")

    print(f"Panel row count: {total_count}")
    symbol_to_id = {s: i for i, s in enumerate(sorted(symbols))}

    file_layout: list[tuple[str, str, int, int]] = []
    cursor = 0
    for fp in files:
        key = str(fp)
        symbol, cnt = file_rows[key]
        file_layout.append((key, symbol, cursor, cnt))
        cursor += cnt
    if cursor != total_count:
        raise RuntimeError(f"Layout row count mismatch: layout={cursor}, total={total_count}")

    x_store = np.lib.format.open_memmap(
        out_root / "X.npy", mode="w+", dtype=np.float32, shape=(total_count, n_feat)
    )
    y_store = np.lib.format.open_memmap(
        out_root / "y.npy", mode="w+", dtype=np.float32, shape=(total_count,)
    )
    date_store = np.lib.format.open_memmap(
        out_root / "date.npy", mode="w+", dtype=np.int64, shape=(total_count,)
    )
    symbol_store = np.lib.format.open_memmap(
        out_root / "symbol.npy", mode="w+", dtype=np.int32, shape=(total_count,)
    )

    print(f"[2/3] Building panel arrays (workers={workers})...")
    t1 = time.perf_counter()
    if workers == 1:
        pbar = tqdm(file_layout, desc="processing", total=total_files, unit="file")
        for idx, (file_path, symbol, start, cnt) in enumerate(pbar, start=1):
            symbol2, file_n_feat, payload = _build_file_payload_worker(file_path, alpha)
            if file_n_feat != n_feat:
                raise RuntimeError(f"Feature dim mismatch in pass2: expected {n_feat}, got {file_n_feat}")
            if symbol2 != symbol:
                raise RuntimeError(f"Symbol mismatch in pass2: layout={symbol}, computed={symbol2}, file={file_path}")
            sid = symbol_to_id[symbol]
            x_arr, y_arr, d_arr = payload
            if y_arr.shape[0] != cnt:
                raise RuntimeError(
                    f"Row count mismatch for {file_path}: counted={cnt}, computed={y_arr.shape[0]}"
                )
            end = start + cnt
            x_store[start:end] = x_arr
            y_store[start:end] = y_arr
            date_store[start:end] = d_arr
            symbol_store[start:end] = sid
            elapsed = time.perf_counter() - t1
            speed = idx / elapsed if elapsed > 0 else 0.0
            eta = (total_files - idx) / speed if speed > 0 else float("inf")
            pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
    else:
        def _run_thread_write() -> None:
            def _thread_job(file_path: str, symbol: str, start: int, cnt: int) -> int:
                symbol2, file_n_feat, payload = _build_file_payload_worker(file_path, alpha)
                if file_n_feat != n_feat:
                    raise RuntimeError(f"Feature dim mismatch in pass2: expected {n_feat}, got {file_n_feat}")
                if symbol2 != symbol:
                    raise RuntimeError(f"Symbol mismatch in pass2: layout={symbol}, computed={symbol2}, file={file_path}")
                x_arr, y_arr, d_arr = payload
                if y_arr.shape[0] != cnt:
                    raise RuntimeError(
                        f"Row count mismatch for {file_path}: counted={cnt}, computed={y_arr.shape[0]}"
                    )
                end = start + cnt
                x_store[start:end] = x_arr
                y_store[start:end] = y_arr
                date_store[start:end] = d_arr
                symbol_store[start:end] = symbol_to_id[symbol]
                return cnt

            futures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for file_path, symbol, start, cnt in file_layout:
                    futures.append(
                        executor.submit(
                            _thread_job,
                            file_path,
                            symbol,
                            start,
                            cnt,
                        )
                    )
                pbar = tqdm(total=total_files, desc="processing", unit="file")
                for idx, fut in enumerate(as_completed(futures), start=1):
                    fut.result()
                    pbar.update(1)
                    elapsed = time.perf_counter() - t1
                    speed = idx / elapsed if elapsed > 0 else 0.0
                    eta = (total_files - idx) / speed if speed > 0 else float("inf")
                    pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
                pbar.close()

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = []
                print("  processing backend: process")
                for file_path, symbol, start, cnt in file_layout:
                    sid = symbol_to_id[symbol]
                    futures.append(
                        executor.submit(
                            _write_panel_file_slice_process,
                            file_path,
                            alpha,
                            start,
                            cnt,
                            sid,
                            str(out_root / "X.npy"),
                            str(out_root / "y.npy"),
                            str(out_root / "date.npy"),
                            str(out_root / "symbol.npy"),
                            total_count,
                            n_feat,
                        )
                    )
                pbar = tqdm(total=total_files, desc="processing", unit="file")
                for idx, fut in enumerate(as_completed(futures), start=1):
                    fut.result()
                    pbar.update(1)
                    elapsed = time.perf_counter() - t1
                    speed = idx / elapsed if elapsed > 0 else 0.0
                    eta = (total_files - idx) / speed if speed > 0 else float("inf")
                    pbar.set_postfix(speed=f"{speed:.2f}/s", eta_m=f"{eta/60:.1f}")
                pbar.close()
        except Exception as e:
            print(f"  process backend unavailable ({type(e).__name__}: {e}), fallback to thread backend")
            print("  processing backend: thread")
            _run_thread_write()

    metadata = {
        "cache_mode": "panel_2d",
        "alpha": alpha,
        "date_unit": "ns",
        "num_features": n_feat,
        "num_rows": total_count,
        "shape": [total_count, n_feat],
        "feature_names": get_alpha158_feature_config()[1] if alpha == "158" else get_alpha360_feature_config()[1],
        "label": "open_t+2 / open_t+1 - 1",
        "symbol_to_id": symbol_to_id,
        "row_order": "symbol-major (input file order), date ascending within symbol",
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[3/3] Done. Panel cache saved to: {out_root}")
    return metadata


def validate_default_dimensions() -> dict[str, int]:
    f158, n158 = get_alpha158_feature_config()
    f360, n360 = get_alpha360_feature_config()
    if len(f158) != 158 or len(n158) != 158:
        raise ValueError(f"Alpha158 mismatch: {len(f158)}, {len(n158)}")
    if len(f360) != 360 or len(n360) != 360:
        raise ValueError(f"Alpha360 mismatch: {len(f360)}, {len(n360)}")
    return {"alpha158": 158, "alpha360": 360}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D panel cache from parquet (no qlib).")
    parser.add_argument("--parquet-dir", default="data/processed/combined", help="Input parquet directory.")
    parser.add_argument("--output-dir", default=None, help="Output cache directory.")
    parser.add_argument("--alpha", choices=["158", "360"], default="158", help="Alpha feature set.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for counting/writing.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_default_dimensions()
    out_dir = args.output_dir or f"data/cache/alpha{args.alpha}_panel"
    print(f"cache_mode=panel_2d, alpha={args.alpha}, output={out_dir}")
    generate_panel_cache(
        parquet_dir=args.parquet_dir,
        output_dir=out_dir,
        alpha=args.alpha,
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    main()
