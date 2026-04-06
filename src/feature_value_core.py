"""Pure feature-value computation helpers for the unified factor pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.data_source import normalize_data_source_name
    from src.label_utils import (
        DEFAULT_LABEL_ABS_CAP,
        DEFAULT_LABEL_HORIZON,
        get_label_column_name,
        get_legacy_label_column_name,
    )
except ModuleNotFoundError:
    from data_source import normalize_data_source_name  # type: ignore
    from label_utils import (  # type: ignore
        DEFAULT_LABEL_ABS_CAP,
        DEFAULT_LABEL_HORIZON,
        get_label_column_name,
        get_legacy_label_column_name,
    )

from src.feature_name_registry import (
    ALL_FACTORS_LGBM_PREFIX,
    DEFAULT_ALPHA158_CONFIG,
    DEFAULT_LGBM_PURIFIED_CONFIG,
    DEFAULT_TECHNICAL_FACTOR_CONFIG,
    DEFAULT_TEMPORAL_FACTOR_CONFIG,
    DEFAULT_TUSHARE_FACTOR_CONFIG,
    EPS,
    TECHNICAL_FACTOR_PREFIX,
    TEMPORAL_FACTOR_PREFIX,
    TUSHARE_FACTOR_PREFIX,
    get_all_factor_feature_names,
    get_alpha158_feature_config,
    get_lgbm_purified_feature_names,
    get_technical_factor_feature_names,
    get_temporal_factor_feature_names,
    get_tushare_factor_feature_names,
)


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
    return series.rolling(window, min_periods=1).rank(method="average", pct=True)


def _rolling_idxmax(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(lambda x: float(np.argmax(x) + 1), raw=True)


def _rolling_idxmin(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).apply(lambda x: float(np.argmin(x) + 1), raw=True)


def _rolling_window_view(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.full(values.size + window - 1, np.nan, dtype=np.float64)
    padded[window - 1 :] = values
    return np.lib.stride_tricks.sliding_window_view(padded, window)


def _rolling_regression_stats(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = series.to_numpy(dtype=np.float64, copy=False)
    if values.size == 0:
        empty = pd.Series(index=series.index, dtype=float)
        return empty, empty.copy(), empty.copy()

    windows = _rolling_window_view(values, int(window))
    mask = np.isfinite(windows)
    counts = mask.sum(axis=1).astype(np.float64)
    ordered_pos = np.cumsum(mask, axis=1, dtype=np.float64) * mask
    clean_windows = np.where(mask, windows, 0.0)

    sy = clean_windows.sum(axis=1)
    syy = (clean_windows * clean_windows).sum(axis=1)
    sx = counts * (counts + 1.0) / 2.0
    sxx = counts * (counts + 1.0) * (2.0 * counts + 1.0) / 6.0
    sxy = (ordered_pos * clean_windows).sum(axis=1)

    numer = counts * sxy - sx * sy
    denom_x = counts * sxx - sx * sx
    valid = (counts >= 2.0) & ~np.isclose(denom_x, 0.0)

    slope = np.full(values.shape, np.nan, dtype=np.float64)
    slope[valid] = numer[valid] / denom_x[valid]

    denom_y = counts * syy - sy * sy
    rsquare = np.full(values.shape, np.nan, dtype=np.float64)
    rsquare_valid = valid & (denom_y > 0.0)
    rsquare[rsquare_valid] = np.clip(
        (numer[rsquare_valid] * numer[rsquare_valid]) / (denom_x[rsquare_valid] * denom_y[rsquare_valid]),
        0.0,
        1.0,
    )

    alpha = np.full(values.shape, np.nan, dtype=np.float64)
    alpha[valid] = (sy[valid] - slope[valid] * sx[valid]) / counts[valid]
    last_pos = np.where(mask, np.arange(windows.shape[1], dtype=np.int64), -1).max(axis=1)
    residual = np.full(values.shape, np.nan, dtype=np.float64)
    residual_valid = valid & (last_pos >= 0)
    residual_rows = np.flatnonzero(residual_valid)
    last_values = windows[residual_rows, last_pos[residual_rows]]
    residual[residual_rows] = last_values - (alpha[residual_rows] + slope[residual_rows] * counts[residual_rows])

    return (
        pd.Series(slope, index=series.index),
        pd.Series(rsquare, index=series.index),
        pd.Series(residual, index=series.index),
    )


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    slope, _, _ = _rolling_regression_stats(series, window)
    return slope


def _rolling_rsquare(series: pd.Series, window: int) -> pd.Series:
    _, rsquare, _ = _rolling_regression_stats(series, window)
    return rsquare


def _rolling_resi(series: pd.Series, window: int) -> pd.Series:
    _, _, residual = _rolling_regression_stats(series, window)
    return residual


def _rolling_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    res = left.rolling(window, min_periods=1).corr(right)
    lstd = left.rolling(window, min_periods=1).std()
    rstd = right.rolling(window, min_periods=1).std()
    res[np.isclose(lstd, 0, atol=2e-5) | np.isclose(rstd, 0, atol=2e-5)] = np.nan
    return res


def compute_alpha360(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_alpha158(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    _base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = deepcopy(DEFAULT_ALPHA158_CONFIG)
    if config is not None:
        cfg.update(config)

    base = _base if _base is not None else _prepare_ohlcv(df)
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
        close_delta_pos = close_delta.clip(lower=0)
        close_delta_neg = (ref_close1 - close).clip(lower=0)
        vol_delta_pos = vol_delta.clip(lower=0)
        vol_delta_neg = (ref_vol1 - volume).clip(lower=0)
        log_volume = np.log(volume + 1)

        for d in windows:
            close_roll = close.rolling(d, min_periods=1)
            high_roll = high.rolling(d, min_periods=1)
            low_roll = low.rolling(d, min_periods=1)
            volume_roll = volume.rolling(d, min_periods=1)
            close_abs_delta_roll = np.abs(close_delta).rolling(d, min_periods=1)
            vol_abs_delta_roll = np.abs(vol_delta).rolling(d, min_periods=1)
            slope = rsquare = residual = None
            if use("BETA") or use("RSQR") or use("RESI"):
                slope, rsquare, residual = _rolling_regression_stats(close, d)
            if use("ROC"):
                out[f"ROC{d}"] = close.shift(d) / close
            if use("MA"):
                out[f"MA{d}"] = close_roll.mean() / close
            if use("STD"):
                out[f"STD{d}"] = close_roll.std() / close
            if use("BETA"):
                out[f"BETA{d}"] = slope / close
            if use("RSQR"):
                out[f"RSQR{d}"] = rsquare
            if use("RESI"):
                out[f"RESI{d}"] = residual / close
            if use("MAX"):
                out[f"MAX{d}"] = high_roll.max() / close
            if use("LOW"):
                out[f"MIN{d}"] = low_roll.min() / close
            if use("QTLU"):
                out[f"QTLU{d}"] = close_roll.quantile(0.8) / close
            if use("QTLD"):
                out[f"QTLD{d}"] = close_roll.quantile(0.2) / close
            if use("RANK"):
                out[f"RANK{d}"] = _rolling_rank_pct(close, d)
            if use("RSV"):
                rolling_low = low_roll.min()
                rolling_high = high_roll.max()
                out[f"RSV{d}"] = (close - rolling_low) / (rolling_high - rolling_low + EPS)
            imax = imin = None
            if use("IMAX") or use("IMXD"):
                imax = _rolling_idxmax(high, d)
            if use("IMIN") or use("IMXD"):
                imin = _rolling_idxmin(low, d)
            if use("IMAX"):
                out[f"IMAX{d}"] = imax / d
            if use("IMIN"):
                out[f"IMIN{d}"] = imin / d
            if use("IMXD"):
                out[f"IMXD{d}"] = (imax - imin) / d
            if use("CORR"):
                out[f"CORR{d}"] = _rolling_corr(close, log_volume, d)
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
            close_pos_sum = close_neg_sum = None
            if use("SUMP") or use("SUMN") or use("SUMD"):
                close_pos_sum = close_delta_pos.rolling(d, min_periods=1).sum()
                close_neg_sum = close_delta_neg.rolling(d, min_periods=1).sum()
            if use("SUMP"):
                out[f"SUMP{d}"] = close_pos_sum / (close_abs_delta_roll.sum() + EPS)
            if use("SUMN"):
                out[f"SUMN{d}"] = close_neg_sum / (close_abs_delta_roll.sum() + EPS)
            if use("SUMD"):
                out[f"SUMD{d}"] = (close_pos_sum - close_neg_sum) / (close_abs_delta_roll.sum() + EPS)
            if use("VMA"):
                out[f"VMA{d}"] = volume_roll.mean() / (volume + EPS)
            if use("VSTD"):
                out[f"VSTD{d}"] = volume_roll.std() / (volume + EPS)
            if use("WVMA"):
                w = np.abs(close_ret - 1) * volume
                out[f"WVMA{d}"] = w.rolling(d, min_periods=1).std() / (w.rolling(d, min_periods=1).mean() + EPS)
            vol_pos_sum = vol_neg_sum = None
            if use("VSUMP") or use("VSUMN") or use("VSUMD"):
                vol_pos_sum = vol_delta_pos.rolling(d, min_periods=1).sum()
                vol_neg_sum = vol_delta_neg.rolling(d, min_periods=1).sum()
            if use("VSUMP"):
                out[f"VSUMP{d}"] = vol_pos_sum / (vol_abs_delta_roll.sum() + EPS)
            if use("VSUMN"):
                out[f"VSUMN{d}"] = vol_neg_sum / (vol_abs_delta_roll.sum() + EPS)
            if use("VSUMD"):
                out[f"VSUMD{d}"] = (vol_pos_sum - vol_neg_sum) / (vol_abs_delta_roll.sum() + EPS)

    feat = pd.DataFrame(out, index=base.index)
    _, ordered_names = get_alpha158_feature_config(cfg)
    feat = feat.reindex(columns=ordered_names)
    if config is None and feat.shape[1] != 158:
        raise ValueError(f"Alpha158 feature count mismatch: {feat.shape[1]}")
    return feat


def _calculate_atr(base: pd.DataFrame, period: int) -> pd.Series:
    high_low = base["high"] - base["low"]
    high_close = (base["high"] - base["close"].shift()).abs()
    low_close = (base["low"] - base["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr / base["close"].replace(0, np.nan)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _calculate_macd(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss > EPS, 100.0)
    rsi = rsi.where((avg_gain > EPS) | (avg_loss > EPS), 50.0)
    return rsi


def _calculate_bollinger(close: pd.Series, window: int, num_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    pos = (close - ma) / (num_std * std + EPS)
    width = (2.0 * num_std * std) / (ma.abs() + EPS)
    zscore = (close - ma) / (std + EPS)
    return pos, width, zscore


def _calculate_adx(base: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = base["high"]
    low = base["low"]
    close = base["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=base.index, dtype=float)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=base.index, dtype=float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / (atr + EPS)
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / (atr + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = _wilder_smooth(dx, period)
    return adx, plus_di, minus_di


def _calculate_mfi(base: pd.DataFrame, period: int) -> pd.Series:
    typical_price = (base["high"] + base["low"] + base["close"]) / 3.0
    raw_money_flow = typical_price * base["volume"].fillna(0.0)
    tp_delta = typical_price.diff()
    pos_flow = raw_money_flow.where(tp_delta > 0, 0.0)
    neg_flow = raw_money_flow.where(tp_delta < 0, 0.0).abs()
    pos_sum = pos_flow.rolling(period, min_periods=period).sum()
    neg_sum = neg_flow.rolling(period, min_periods=period).sum()
    money_ratio = pos_sum / (neg_sum + EPS)
    return 100.0 - (100.0 / (1.0 + money_ratio))


def _calculate_cci(base: pd.DataFrame, period: int) -> pd.Series:
    typical_price = (base["high"] + base["low"] + base["close"]) / 3.0
    tp_ma = typical_price.rolling(period, min_periods=period).mean()
    tp_mad = typical_price.rolling(period, min_periods=period).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))),
        raw=True,
    )
    return (typical_price - tp_ma) / (0.015 * tp_mad + EPS)


def _calculate_willr(base: pd.DataFrame, period: int) -> pd.Series:
    highest_high = base["high"].rolling(period, min_periods=period).max()
    lowest_low = base["low"].rolling(period, min_periods=period).min()
    return -100.0 * (highest_high - base["close"]) / (highest_high - lowest_low + EPS)


def _rolling_periods_since_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    def _periods_since(x: np.ndarray) -> float:
        if len(x) == 0 or np.all(np.isnan(x)):
            return np.nan
        xv = np.asarray(x, dtype=float)
        valid_idx = np.where(~np.isnan(xv))[0]
        if valid_idx.size == 0:
            return np.nan
        valid_values = xv[valid_idx]
        chosen_idx = valid_idx[np.argmax(valid_values)] if mode == "max" else valid_idx[np.argmin(valid_values)]
        return float(len(xv) - 1 - chosen_idx)

    return series.rolling(window, min_periods=window).apply(_periods_since, raw=True)


def _calculate_aroon(base: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    periods_since_high = _rolling_periods_since_extreme(base["high"], period, mode="max")
    periods_since_low = _rolling_periods_since_extreme(base["low"], period, mode="min")
    aroon_up = 100.0 * (period - periods_since_high) / period
    aroon_down = 100.0 * (period - periods_since_low) / period
    return aroon_up, aroon_down, aroon_up - aroon_down


def _calculate_trix(close: pd.Series, period: int, signal_period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema1 = close.ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix = ema3.pct_change(fill_method=None) * 100.0
    signal = trix.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = trix - signal
    return trix, signal, hist


def _calculate_obv_flow(base: pd.DataFrame, window: int) -> pd.Series:
    close_diff = base["close"].diff().fillna(0.0)
    signed_volume = pd.Series(np.sign(close_diff) * base["volume"].fillna(0.0), index=base.index, dtype=float)
    total_volume = base["volume"].fillna(0.0).rolling(window, min_periods=window).sum()
    return signed_volume.rolling(window, min_periods=window).sum() / (total_volume + EPS)


def compute_lgbm_purified_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    _base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = deepcopy(DEFAULT_LGBM_PURIFIED_CONFIG)
    if config is not None:
        cfg.update(config)

    base = _base if _base is not None else _prepare_ohlcv(df)
    close = base["close"].replace(0, np.nan)
    volume = base["volume"].replace(0, np.nan)
    amount = base["amount"] if "amount" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    out: dict[str, pd.Series] = {}
    close_ret = close.pct_change(fill_method=None)

    for window in cfg["momentum_windows"]:
        out[f"ret_{window}"] = close.pct_change(window, fill_method=None)
    for window in cfg["ma_windows"]:
        ma = close.rolling(window, min_periods=1).mean().replace(0, np.nan)
        out[f"dist_ma{window}"] = close / ma - 1.0
    vol_window = int(cfg["vol_window"])
    out["std_60"] = close_ret.rolling(vol_window, min_periods=1).std()
    out["atr_14"] = _calculate_atr(base, int(cfg["atr_window"]))
    amihud_window = int(cfg["amihud_window"])
    daily_ret_abs = close_ret.abs()
    out["amihud_20"] = (daily_ret_abs / (amount.abs() + EPS)).rolling(amihud_window, min_periods=1).mean()
    volume_window = int(cfg["volume_window"])
    vol_ma = volume.rolling(volume_window, min_periods=1).mean().replace(0, np.nan)
    out["vol_ratio_20"] = volume / vol_ma
    corr_window = int(cfg["corr_window"])
    out["corr_cv_20"] = close.rolling(corr_window, min_periods=1).corr(base["volume"])
    vwap = base["vwap"].replace(0, np.nan)
    out["vwap_ratio"] = close / vwap - 1.0
    out["log_mcap"] = np.log1p(base["circ_mv"].clip(lower=0)) if "circ_mv" in base.columns else pd.Series(np.nan, index=base.index)
    if "pe_ttm" in base.columns:
        out["ep_ttm"] = np.where(base["pe_ttm"] > 0, 1.0 / base["pe_ttm"], -1.0)
        out["is_loss"] = (base["pe_ttm"] <= 0).astype(float)
    else:
        out["ep_ttm"] = pd.Series(np.nan, index=base.index)
        out["is_loss"] = pd.Series(np.nan, index=base.index)
    out["bp"] = np.where(base["pb"] > 0, 1.0 / base["pb"], -1.0) if "pb" in base.columns else pd.Series(np.nan, index=base.index)
    turnover_window = int(cfg["turnover_window"])
    out["turnover_20"] = base["turnover"].rolling(turnover_window, min_periods=1).mean() if "turnover" in base.columns else pd.Series(np.nan, index=base.index)
    extreme_window = int(cfg["extreme_window"])
    rolling_high = base["high"].rolling(extreme_window, min_periods=1).max().replace(0, np.nan)
    rolling_low = base["low"].rolling(extreme_window, min_periods=1).min().replace(0, np.nan)
    out["dist_high_20"] = close / rolling_high - 1.0
    out["dist_low_20"] = close / rolling_low - 1.0

    feat = pd.DataFrame(out, index=base.index)
    return feat.reindex(columns=get_lgbm_purified_feature_names(cfg))


def compute_technical_factor_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    _base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = deepcopy(DEFAULT_TECHNICAL_FACTOR_CONFIG)
    if config is not None:
        cfg.update(config)

    base = _base if _base is not None else _prepare_ohlcv(df)
    close = base["close"].replace(0, np.nan)
    out: dict[str, pd.Series] = {}

    for setup in cfg["macd_setups"]:
        fast = int(setup["fast"])
        slow = int(setup["slow"])
        signal = int(setup["signal"])
        macd_line, signal_line, hist = _calculate_macd(close, fast, slow, signal)
        out[f"macd_line_{fast}_{slow}_{signal}"] = macd_line
        out[f"macd_signal_{fast}_{slow}_{signal}"] = signal_line
        out[f"macd_hist_{fast}_{slow}_{signal}"] = hist
    for window in cfg["rsi_windows"]:
        out[f"rsi_{int(window)}"] = _calculate_rsi(close, int(window))
    num_std = float(cfg["boll_num_std"])
    num_std_label = int(num_std)
    for window in cfg["boll_windows"]:
        window = int(window)
        pos, width, zscore = _calculate_bollinger(close, window, num_std)
        out[f"boll_pos_{window}_{num_std_label}"] = pos
        out[f"boll_width_{window}_{num_std_label}"] = width
        out[f"boll_zscore_{window}_{num_std_label}"] = zscore
    for window in cfg["adx_windows"]:
        window = int(window)
        adx, plus_di, minus_di = _calculate_adx(base, window)
        out[f"adx_{window}"] = adx
        out[f"plus_di_{window}"] = plus_di
        out[f"minus_di_{window}"] = minus_di
    for window in cfg["mfi_windows"]:
        out[f"mfi_{int(window)}"] = _calculate_mfi(base, int(window))
    for window in cfg["cci_windows"]:
        out[f"cci_{int(window)}"] = _calculate_cci(base, int(window))
    for window in cfg["willr_windows"]:
        out[f"willr_{int(window)}"] = _calculate_willr(base, int(window))
    for window in cfg["aroon_windows"]:
        window = int(window)
        aroon_up, aroon_down, aroon_osc = _calculate_aroon(base, window)
        out[f"aroon_up_{window}"] = aroon_up
        out[f"aroon_down_{window}"] = aroon_down
        out[f"aroon_osc_{window}"] = aroon_osc
    trix_signal = int(cfg["trix_signal"])
    for window in cfg["trix_windows"]:
        window = int(window)
        trix, signal, hist = _calculate_trix(close, window, trix_signal)
        out[f"trix_{window}"] = trix
        out[f"trix_signal_{window}_{trix_signal}"] = signal
        out[f"trix_hist_{window}_{trix_signal}"] = hist
    for window in cfg["obv_windows"]:
        out[f"obv_flow_{int(window)}"] = _calculate_obv_flow(base, int(window))

    feat = pd.DataFrame(out, index=base.index)
    return feat.reindex(columns=get_technical_factor_feature_names(cfg))


def compute_temporal_factor_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    _base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = deepcopy(DEFAULT_TEMPORAL_FACTOR_CONFIG)
    if config is not None:
        cfg.update(config)

    base = _base if _base is not None else _prepare_ohlcv(df)
    close = base["close"].replace(0, np.nan)
    high = base["high"].replace(0, np.nan)
    low = base["low"].replace(0, np.nan)
    volume = base["volume"].replace(0, np.nan)
    amount = base["amount"] if "amount" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    turnover = base["turnover"] if "turnover" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)

    close_ret = close.pct_change(fill_method=None)
    close_ret_abs = close_ret.abs()
    log_volume = np.log(volume + 1.0)
    amount_abs = amount.abs()
    groups = set(cfg["groups"])
    out: dict[str, pd.Series] = {}

    for window in cfg["windows"]:
        close_roll = close.rolling(window, min_periods=1)
        high_roll = high.rolling(window, min_periods=1)
        low_roll = low.rolling(window, min_periods=1)
        volume_roll = volume.rolling(window, min_periods=1)
        turnover_roll = turnover.rolling(window, min_periods=1)
        if "ret" in groups:
            out[f"ret_{window}"] = close.pct_change(window, fill_method=None)
        if "ma_gap" in groups:
            ma = close_roll.mean().replace(0, np.nan)
            out[f"ma_gap_{window}"] = close / ma - 1.0
        if "std" in groups:
            out[f"std_{window}"] = close_ret.rolling(window, min_periods=1).std()
        rolling_low = rolling_high = None
        if "rsv" in groups or "high_gap" in groups or "low_gap" in groups:
            rolling_low = low_roll.min()
            rolling_high = high_roll.max()
        if "rsv" in groups:
            out[f"rsv_{window}"] = (close - rolling_low) / (rolling_high - rolling_low + EPS)
        if "price_rank" in groups:
            out[f"price_rank_{window}"] = _rolling_rank_pct(close, window)
        if "volume_ratio" in groups:
            volume_ma = volume_roll.mean().replace(0, np.nan)
            out[f"volume_ratio_{window}"] = volume / volume_ma - 1.0
        if "turnover_mean" in groups:
            out[f"turnover_mean_{window}"] = turnover_roll.mean()
        if "amihud" in groups:
            out[f"amihud_{window}"] = (close_ret_abs / (amount_abs + EPS)).rolling(window, min_periods=1).mean()
        if "high_gap" in groups:
            out[f"high_gap_{window}"] = close / rolling_high.replace(0, np.nan) - 1.0
        if "low_gap" in groups:
            out[f"low_gap_{window}"] = close / rolling_low.replace(0, np.nan) - 1.0
        if "corr_cv" in groups:
            out[f"corr_cv_{window}"] = close.rolling(window, min_periods=1).corr(log_volume)

    feat = pd.DataFrame(out, index=base.index)
    return feat.reindex(columns=get_temporal_factor_feature_names(cfg))


def compute_tushare_factor_features(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    _base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    cfg = deepcopy(DEFAULT_TUSHARE_FACTOR_CONFIG)
    if config is not None:
        cfg.update(config)

    base = _base if _base is not None else _prepare_ohlcv(df)
    close = base["close"].replace(0, np.nan)
    up_limit = base["up_limit"] if "up_limit" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    down_limit = base["down_limit"] if "down_limit" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    limit_pre_close = base["limit_pre_close"] if "limit_pre_close" in base.columns else base.get("pre_close", pd.Series(np.nan, index=base.index, dtype=float))
    limit_pre_close = limit_pre_close.replace(0, np.nan)
    total_share = base["total_share"] if "total_share" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    circ_share = base["circ_share"] if "circ_share" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    free_share = base["free_share"] if "free_share" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    turnover = base["turnover"] if "turnover" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    turnover_free = base["turnover_free"] if "turnover_free" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    volume_ratio = base["volume_ratio"] if "volume_ratio" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    total_mv = base["total_mv"] if "total_mv" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    circ_mv = base["circ_mv"] if "circ_mv" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    pe = base["pe"] if "pe" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    pe_ttm = base["pe_ttm"] if "pe_ttm" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    ps = base["ps"] if "ps" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    ps_ttm = base["ps_ttm"] if "ps_ttm" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    dv_ratio = base["dv_ratio"] if "dv_ratio" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    dv_ttm = base["dv_ttm"] if "dv_ttm" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    amplitude = base["amplitude"] if "amplitude" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    pct_chg = base["pct_chg"] if "pct_chg" in base.columns else pd.Series(np.nan, index=base.index, dtype=float)
    def _series(name: str) -> pd.Series:
        return base[name] if name in base.columns else pd.Series(np.nan, index=base.index, dtype=float)

    fi_eps = _series("fi_eps")
    fi_dt_eps = _series("fi_dt_eps")
    fi_bps = _series("fi_bps")
    fi_ocfps = _series("fi_ocfps")
    fi_roe = _series("fi_roe")
    fi_roe_dt = _series("fi_roe_dt")
    fi_roa = _series("fi_roa")
    fi_gpm = _series("fi_grossprofit_margin")
    fi_npm = _series("fi_netprofit_margin")
    fi_debt_to_assets = _series("fi_debt_to_assets")
    fi_q_eps = _series("fi_q_eps")
    fi_q_dtprofit = _series("fi_q_dtprofit")
    fi_q_roe = _series("fi_q_roe")
    fi_q_dt_roe = _series("fi_q_dt_roe")
    fi_tr_yoy = _series("fi_tr_yoy")
    fi_or_yoy = _series("fi_or_yoy")
    fi_op_yoy = _series("fi_op_yoy")
    fi_netprofit_yoy = _series("fi_netprofit_yoy")
    fi_ocf_yoy = _series("fi_ocf_yoy")
    div_cash_div = _series("div_cash_div")
    div_cash_div_tax = _series("div_cash_div_tax")
    div_stk_div = _series("div_stk_div")
    div_stk_bo_rate = _series("div_stk_bo_rate")
    div_stk_co_rate = _series("div_stk_co_rate")
    div_base_share = _series("div_base_share")
    fc_p_change_min = _series("fc_p_change_min")
    fc_p_change_max = _series("fc_p_change_max")
    fc_net_profit_min = _series("fc_net_profit_min")
    fc_net_profit_max = _series("fc_net_profit_max")
    fc_last_parent_net = _series("fc_last_parent_net")
    exp_revenue = _series("exp_revenue")
    exp_operate_profit = _series("exp_operate_profit")
    exp_total_profit = _series("exp_total_profit")
    exp_n_income = _series("exp_n_income")
    exp_total_assets = _series("exp_total_assets")
    exp_diluted_eps = _series("exp_diluted_eps")
    exp_diluted_roe = _series("exp_diluted_roe")
    exp_yoy_sales = _series("exp_yoy_sales")
    exp_yoy_op = _series("exp_yoy_op")
    exp_yoy_tp = _series("exp_yoy_tp")
    exp_yoy_dedu_np = _series("exp_yoy_dedu_np")
    exp_yoy_eps = _series("exp_yoy_eps")
    exp_yoy_roe = _series("exp_yoy_roe")
    exp_growth_assets = _series("exp_growth_assets")
    exp_yoy_assets = _series("exp_yoy_assets")
    ind_member_count = _series("ind_member_count")
    ind_daily_ret = _series("ind_daily_ret")
    ind_excess_daily_ret = _series("ind_excess_daily_ret")

    out: dict[str, pd.Series | np.ndarray] = {}
    out["gap_up_limit"] = up_limit / close - 1.0
    out["gap_down_limit"] = close / down_limit - 1.0
    out["limit_band_pct"] = (up_limit - down_limit) / (limit_pre_close + EPS)
    out["limit_band_pos"] = (close - down_limit) / (up_limit - down_limit + EPS)
    hit_up_limit = pd.Series(np.where(np.isfinite(close) & np.isfinite(up_limit), (close >= up_limit * (1.0 - 1e-6)).astype(float), np.nan), index=base.index)
    hit_down_limit = pd.Series(np.where(np.isfinite(close) & np.isfinite(down_limit), (close <= down_limit * (1.0 + 1e-6)).astype(float), np.nan), index=base.index)
    out["hit_up_limit"] = hit_up_limit
    out["hit_down_limit"] = hit_down_limit
    total_share_safe = total_share.replace(0, np.nan)
    circ_share_safe = circ_share.replace(0, np.nan)
    out["free_float_ratio"] = free_share / total_share_safe
    out["circ_float_ratio"] = circ_share / total_share_safe
    out["free_to_circ_ratio"] = free_share / circ_share_safe
    turnover_safe = turnover.replace(0, np.nan)
    out["free_turnover_ratio"] = turnover_free / turnover_safe
    out["free_turnover_spread"] = turnover_free - turnover
    for window in cfg["free_turnover_windows"]:
        window = int(window)
        out[f"free_turnover_mean_{window}"] = turnover_free.rolling(window, min_periods=1).mean()
    out["volume_ratio_raw"] = volume_ratio
    total_mv_safe = total_mv.replace(0, np.nan)
    out["float_mv_ratio"] = circ_mv / total_mv_safe
    out["ep"] = np.where(pe > 0, 1.0 / pe, -1.0)
    out["sp"] = np.where(ps > 0, 1.0 / ps, -1.0)
    out["sp_ttm"] = np.where(ps_ttm > 0, 1.0 / ps_ttm, -1.0)
    ep_ttm = np.where(pe_ttm > 0, 1.0 / pe_ttm, -1.0)
    out["ep_ttm_gap"] = out["ep"] - ep_ttm
    out["dividend_yield"] = dv_ratio
    out["dividend_yield_ttm"] = dv_ttm
    out["has_dividend"] = np.where(np.isfinite(dv_ttm), (dv_ttm > 0).astype(float), np.nan)
    out["industry_member_count"] = ind_member_count
    out["industry_daily_ret"] = ind_daily_ret
    out["industry_excess_daily_ret"] = ind_excess_daily_ret
    for window in cfg["limit_stat_windows"]:
        window = int(window)
        out[f"limit_band_pct_mean_{window}"] = pd.Series(out["limit_band_pct"], index=base.index).rolling(window, min_periods=1).mean()
        out[f"limit_band_pos_mean_{window}"] = pd.Series(out["limit_band_pos"], index=base.index).rolling(window, min_periods=1).mean()
        out[f"gap_up_limit_mean_{window}"] = pd.Series(out["gap_up_limit"], index=base.index).rolling(window, min_periods=1).mean()
        out[f"gap_down_limit_mean_{window}"] = pd.Series(out["gap_down_limit"], index=base.index).rolling(window, min_periods=1).mean()
        out[f"hit_up_limit_count_{window}"] = hit_up_limit.rolling(window, min_periods=1).sum()
        out[f"hit_down_limit_count_{window}"] = hit_down_limit.rolling(window, min_periods=1).sum()
    for window in cfg["amplitude_windows"]:
        out[f"amplitude_mean_{int(window)}"] = amplitude.rolling(int(window), min_periods=1).mean()
    for window in cfg["pct_chg_windows"]:
        out[f"pct_chg_mean_{int(window)}"] = pct_chg.rolling(int(window), min_periods=1).mean()
    for window in cfg["ratio_change_windows"]:
        window = int(window)
        out[f"free_float_ratio_change_{window}"] = pd.Series(out["free_float_ratio"], index=base.index).pct_change(window, fill_method=None)
        out[f"free_to_circ_ratio_change_{window}"] = pd.Series(out["free_to_circ_ratio"], index=base.index).pct_change(window, fill_method=None)
        out[f"float_mv_ratio_change_{window}"] = pd.Series(out["float_mv_ratio"], index=base.index).pct_change(window, fill_method=None)
    for window in cfg["valuation_change_windows"]:
        window = int(window)
        out[f"sp_ttm_change_{window}"] = pd.Series(out["sp_ttm"], index=base.index).pct_change(window, fill_method=None)
        out[f"dividend_yield_ttm_change_{window}"] = pd.Series(out["dividend_yield_ttm"], index=base.index).pct_change(window, fill_method=None)
    for window in cfg["industry_windows"]:
        window = int(window)
        industry_ret = _series(f"ind_ret_{window}")
        industry_std = _series(f"ind_std_{window}")
        industry_excess_ret = _series(f"ind_excess_ret_{window}")
        own_ret = close.pct_change(window, fill_method=None)
        out[f"industry_ret_{window}"] = industry_ret
        out[f"industry_std_{window}"] = industry_std
        out[f"industry_excess_ret_{window}"] = industry_excess_ret
        out[f"industry_rel_ret_{window}"] = own_ret - industry_ret
    zscore_window = int(cfg["zscore_window"])
    amplitude_mean = amplitude.rolling(zscore_window, min_periods=1).mean()
    amplitude_std = amplitude.rolling(zscore_window, min_periods=1).std()
    pct_chg_mean = pct_chg.rolling(zscore_window, min_periods=1).mean()
    pct_chg_std = pct_chg.rolling(zscore_window, min_periods=1).std()
    free_turnover_ratio_series = pd.Series(out["free_turnover_ratio"], index=base.index)
    free_turnover_mean = free_turnover_ratio_series.rolling(zscore_window, min_periods=1).mean()
    free_turnover_std = free_turnover_ratio_series.rolling(zscore_window, min_periods=1).std()
    volume_ratio_mean = volume_ratio.rolling(zscore_window, min_periods=1).mean()
    volume_ratio_std = volume_ratio.rolling(zscore_window, min_periods=1).std()
    out[f"amplitude_zscore_{zscore_window}"] = (amplitude - amplitude_mean) / (amplitude_std + EPS)
    out[f"pct_chg_zscore_{zscore_window}"] = (pct_chg - pct_chg_mean) / (pct_chg_std + EPS)
    out[f"free_turnover_ratio_zscore_{zscore_window}"] = (free_turnover_ratio_series - free_turnover_mean) / (free_turnover_std + EPS)
    out[f"volume_ratio_raw_zscore_{zscore_window}"] = (volume_ratio - volume_ratio_mean) / (volume_ratio_std + EPS)
    out["latest_eps"] = fi_eps
    out["latest_dt_eps"] = fi_dt_eps
    out["latest_bps"] = fi_bps
    out["latest_ocfps"] = fi_ocfps
    out["latest_roe"] = fi_roe
    out["latest_roe_dt"] = fi_roe_dt
    out["latest_roa"] = fi_roa
    out["latest_grossprofit_margin"] = fi_gpm
    out["latest_netprofit_margin"] = fi_npm
    out["latest_debt_to_assets"] = fi_debt_to_assets
    out["latest_q_eps"] = fi_q_eps
    out["latest_q_dtprofit"] = fi_q_dtprofit
    out["latest_q_roe"] = fi_q_roe
    out["latest_q_dt_roe"] = fi_q_dt_roe
    out["latest_tr_yoy"] = fi_tr_yoy
    out["latest_or_yoy"] = fi_or_yoy
    out["latest_op_yoy"] = fi_op_yoy
    out["latest_netprofit_yoy"] = fi_netprofit_yoy
    out["latest_ocf_yoy"] = fi_ocf_yoy
    out["latest_div_cash"] = div_cash_div
    out["latest_div_cash_tax"] = div_cash_div_tax
    out["latest_div_stock"] = div_stk_div
    out["latest_div_bo_rate"] = div_stk_bo_rate
    out["latest_div_co_rate"] = div_stk_co_rate
    out["latest_div_base_share"] = div_base_share
    out["latest_div_cash_yield_proxy"] = div_cash_div / (close + EPS)
    out["latest_div_stock_ratio"] = div_stk_div + div_stk_bo_rate + div_stk_co_rate
    out["has_stock_dividend"] = np.where(np.isfinite(out["latest_div_stock_ratio"]), (out["latest_div_stock_ratio"] > 0).astype(float), np.nan)
    out["latest_fc_p_change_min"] = fc_p_change_min
    out["latest_fc_p_change_max"] = fc_p_change_max
    out["latest_fc_net_profit_min"] = fc_net_profit_min
    out["latest_fc_net_profit_max"] = fc_net_profit_max
    out["latest_fc_last_parent_net"] = fc_last_parent_net
    out["latest_exp_revenue"] = exp_revenue
    out["latest_exp_operate_profit"] = exp_operate_profit
    out["latest_exp_total_profit"] = exp_total_profit
    out["latest_exp_n_income"] = exp_n_income
    out["latest_exp_total_assets"] = exp_total_assets
    out["latest_exp_diluted_eps"] = exp_diluted_eps
    out["latest_exp_diluted_roe"] = exp_diluted_roe
    out["latest_exp_yoy_sales"] = exp_yoy_sales
    out["latest_exp_yoy_op"] = exp_yoy_op
    out["latest_exp_yoy_tp"] = exp_yoy_tp
    out["latest_exp_yoy_dedu_np"] = exp_yoy_dedu_np
    out["latest_exp_yoy_eps"] = exp_yoy_eps
    out["latest_exp_yoy_roe"] = exp_yoy_roe
    out["latest_exp_growth_assets"] = exp_growth_assets
    out["latest_exp_yoy_assets"] = exp_yoy_assets

    feat = pd.DataFrame(out, index=base.index)
    return feat.reindex(columns=get_tushare_factor_feature_names(cfg))


def compute_all_factor_features(
    df: pd.DataFrame,
    alpha158_config: dict[str, Any] | None = None,
    lgbm_purified_config: dict[str, Any] | None = None,
    technical_config: dict[str, Any] | None = None,
    data_source: str | None = None,
    tushare_config: dict[str, Any] | None = None,
    _base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = _base if _base is not None else _prepare_ohlcv(df)
    alpha158_feat = compute_alpha158(df, config=alpha158_config, _base=base)
    lgbm_feat = compute_lgbm_purified_features(df, config=lgbm_purified_config, _base=base).rename(columns=lambda name: f"{ALL_FACTORS_LGBM_PREFIX}{name}")
    temporal_feat = compute_temporal_factor_features(df, _base=base).rename(columns=lambda name: f"{TEMPORAL_FACTOR_PREFIX}{name}")
    technical_feat = compute_technical_factor_features(df, config=technical_config, _base=base).rename(columns=lambda name: f"{TECHNICAL_FACTOR_PREFIX}{name}")
    parts = [alpha158_feat, lgbm_feat, temporal_feat, technical_feat]
    normalized_data_source = normalize_data_source_name(data_source) if data_source is not None else None
    if normalized_data_source == "tushare":
        tushare_feat = compute_tushare_factor_features(df, config=tushare_config, _base=base).rename(columns=lambda name: f"{TUSHARE_FACTOR_PREFIX}{name}")
        parts.append(tushare_feat)
    feat = pd.concat(parts, axis=1)
    ordered_names = get_all_factor_feature_names(
        alpha158_config,
        lgbm_purified_config,
        technical_config,
        data_source=normalized_data_source,
        tushare_config=tushare_config,
    )
    return feat.reindex(columns=ordered_names)


def build_open_to_open_label(df: pd.DataFrame, horizon_days: int = DEFAULT_LABEL_HORIZON) -> pd.Series:
    base = _prepare_ohlcv(df)
    return _build_open_to_open_label_from_base(base, horizon_days=horizon_days)


def _build_open_to_open_label_from_base(
    base: pd.DataFrame,
    *,
    horizon_days: int = DEFAULT_LABEL_HORIZON,
) -> pd.Series:
    next_open = base["open"].shift(-1)
    exit_open = base["open"].shift(-(1 + int(horizon_days)))
    label = exit_open / next_open - 1
    valid = np.isfinite(next_open) & np.isfinite(exit_open) & (next_open > 0) & (exit_open > 0)
    if "volume" in base.columns:
        next_vol = base["volume"].shift(-1)
        exit_vol = base["volume"].shift(-(1 + int(horizon_days)))
        valid &= np.isfinite(next_vol) & np.isfinite(exit_vol) & (next_vol > 0) & (exit_vol > 0)
    if "amount" in base.columns:
        next_amt = base["amount"].shift(-1)
        exit_amt = base["amount"].shift(-(1 + int(horizon_days)))
        valid &= np.isfinite(next_amt) & np.isfinite(exit_amt) & (next_amt > 0) & (exit_amt > 0)
    label = label.where(valid)
    label = label.where(label.abs() <= DEFAULT_LABEL_ABS_CAP)
    return label


def build_open_to_open_labels(
    df: pd.DataFrame,
    horizons: list[int],
) -> dict[str, pd.Series]:
    base = _prepare_ohlcv(df)
    labels = {
        get_label_column_name(horizon): _build_open_to_open_label_from_base(base, horizon_days=horizon)
        for horizon in horizons
    }
    labels[get_legacy_label_column_name()] = labels[get_label_column_name(1)]
    return labels


def _index_to_epoch_ns(index: pd.DatetimeIndex) -> np.ndarray:
    idx_ns = index.astype("datetime64[ns]")
    return idx_ns.view("i8")


def _to_panel_arrays(feat: pd.DataFrame, label: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = label.reindex(feat.index)
    x2d = feat.to_numpy(dtype=np.float32, copy=False)
    y1d = y.to_numpy(dtype=np.float32, copy=False)
    date_ns = _index_to_epoch_ns(feat.index).astype(np.int64, copy=False)
    return x2d, y1d, date_ns
