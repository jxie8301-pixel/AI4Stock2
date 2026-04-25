"""Backtest report boundary helpers.

Native backtest reports use ``net_return`` as the canonical engine column.
Legacy plotting/reporting code still consumes ``return``. Keep that aliasing
explicit and localized through this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

NATIVE_RETURN_COLUMN = "net_return"
LEGACY_RETURN_COLUMN = "return"


def get_backtest_return_series(report: pd.DataFrame) -> pd.Series | None:
    """Return the semantic net-return series from a native or legacy report."""
    if LEGACY_RETURN_COLUMN in report.columns:
        return report[LEGACY_RETURN_COLUMN].astype(float)
    if NATIVE_RETURN_COLUMN in report.columns:
        return report[NATIVE_RETURN_COLUMN].astype(float)
    return None


def to_legacy_return_report(native_report: pd.DataFrame) -> pd.DataFrame:
    """Convert a native-engine report to the legacy reporting shape."""
    if NATIVE_RETURN_COLUMN not in native_report.columns:
        raise KeyError(f"native backtest report must contain {NATIVE_RETURN_COLUMN!r}")
    if LEGACY_RETURN_COLUMN in native_report.columns:
        raise ValueError(
            f"native backtest report already contains {LEGACY_RETURN_COLUMN!r}; "
            "legacy aliasing should happen only at reporting boundaries"
        )

    report = native_report.rename(columns={NATIVE_RETURN_COLUMN: LEGACY_RETURN_COLUMN})
    report.attrs = dict(getattr(native_report, "attrs", {}) or {})
    return report


def attach_native_baseline_returns(
    report: pd.DataFrame,
    baseline_reports: Mapping[str, tuple[str, pd.DataFrame]],
) -> None:
    """Attach native baseline reports to a legacy reporting frame in place."""
    for prefix, (display_name, baseline_report) in baseline_reports.items():
        if NATIVE_RETURN_COLUMN not in baseline_report.columns:
            raise KeyError(f"baseline report {prefix!r} must contain {NATIVE_RETURN_COLUMN!r}")
        report[f"{prefix}_return"] = (
            baseline_report[NATIVE_RETURN_COLUMN].reindex(report.index).fillna(0.0).to_numpy()
        )
        report.attrs[f"{prefix}_name"] = display_name
