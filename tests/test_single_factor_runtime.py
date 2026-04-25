from __future__ import annotations

import argparse

import pandas as pd

from src.single_factor_runtime import derive_diagnostic_label_series, resolve_period_dates, resolve_segments


def test_resolve_period_dates_uses_explicit_range() -> None:
    args = argparse.Namespace(date_start="2024-01-01", date_end="2024-01-31", period="test")

    assert resolve_period_dates({}, args) == ("2024-01-01", "2024-01-31")


def test_resolve_segments_combines_yearly_and_custom_segments() -> None:
    args = argparse.Namespace(
        segment_scheme="yearly",
        segments="custom:2024-02-01:2024-02-29",
    )

    segments = resolve_segments({}, args, main_start="2023-12-15", main_end="2024-02-29")

    assert ("y2023", "2023-12-15", "2023-12-31") in segments
    assert ("y2024", "2024-01-01", "2024-02-29") in segments
    assert ("custom", "2024-02-01", "2024-02-29") in segments


def test_derive_diagnostic_label_series_raw_return() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["A", "B"],
            "label": [0.01, -0.02],
        }
    )

    labels = derive_diagnostic_label_series(
        frame,
        cfg={},
        signal_horizon=1,
        diagnostic_label_space="raw_return",
        diagnostic_threshold=0.0,
    )

    assert labels.index.names == ["datetime", "instrument"]
    assert labels.loc[(pd.Timestamp("2024-01-02"), "A")] == 0.01
    assert labels.loc[(pd.Timestamp("2024-01-02"), "B")] == -0.02
