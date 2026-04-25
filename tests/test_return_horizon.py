from __future__ import annotations

import numpy as np
import pandas as pd

from src.return_horizon import build_forward_compound_return_series


def test_build_forward_compound_return_series_uses_future_window() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    returns = pd.Series([0.01, 0.02, -0.01, 0.05], index=dates)

    out = build_forward_compound_return_series(returns, horizon=2)

    assert np.isclose(out.loc[dates[0]], (1.02 * 0.99) - 1.0)
    assert np.isclose(out.loc[dates[1]], (0.99 * 1.05) - 1.0)
    assert pd.isna(out.loc[dates[2]])
    assert pd.isna(out.loc[dates[3]])


def test_build_forward_compound_return_series_rejects_nan_windows() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    returns = pd.Series([0.01, np.nan, 0.03, 0.04], index=dates)

    out = build_forward_compound_return_series(returns, horizon=1)

    assert pd.isna(out.loc[dates[0]])
    assert np.isclose(out.loc[dates[1]], 0.03)
    assert np.isclose(out.loc[dates[2]], 0.04)
    assert pd.isna(out.loc[dates[3]])


def test_build_forward_compound_return_series_sorts_index_and_handles_short_input() -> None:
    dates = pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"])
    returns = pd.Series([0.03, 0.01, 0.02], index=dates)

    out = build_forward_compound_return_series(returns, horizon=5)

    assert list(out.index) == sorted(dates)
    assert bool(out.isna().all())
