import unittest

import pandas as pd

from src.backtest_report import (
    attach_native_baseline_returns,
    get_backtest_return_series,
    to_legacy_return_report,
)


class BacktestReportBoundaryTest(unittest.TestCase):
    def test_to_legacy_return_report_converts_net_return_only_at_boundary(self):
        native_report = pd.DataFrame(
            {"net_return": [0.01], "turnover": [0.2]},
            index=pd.to_datetime(["2024-01-02"]),
        )
        native_report.attrs["benchmark_name"] = "Benchmark"

        report = to_legacy_return_report(native_report)

        self.assertIn("return", report.columns)
        self.assertNotIn("net_return", report.columns)
        self.assertEqual(report.attrs["benchmark_name"], "Benchmark")
        self.assertAlmostEqual(float(report.iloc[0]["return"]), 0.01, places=8)

    def test_to_legacy_return_report_rejects_double_aliasing(self):
        native_report = pd.DataFrame(
            {"net_return": [0.01], "return": [0.01]},
            index=pd.to_datetime(["2024-01-02"]),
        )

        with self.assertRaises(ValueError):
            to_legacy_return_report(native_report)

    def test_get_backtest_return_series_accepts_native_and_legacy_shapes(self):
        index = pd.to_datetime(["2024-01-02"])
        native_report = pd.DataFrame({"net_return": [0.01]}, index=index)
        legacy_report = pd.DataFrame({"return": [0.02]}, index=index)

        self.assertAlmostEqual(float(get_backtest_return_series(native_report).iloc[0]), 0.01, places=8)
        self.assertAlmostEqual(float(get_backtest_return_series(legacy_report).iloc[0]), 0.02, places=8)

    def test_attach_native_baseline_returns_uses_net_return_source(self):
        report = pd.DataFrame(
            {"return": [0.01, 0.02]},
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        baseline = pd.DataFrame(
            {"net_return": [0.03]},
            index=pd.to_datetime(["2024-01-02"]),
        )

        attach_native_baseline_returns(report, {"avg_factor_baseline": ("Avg Factor", baseline)})

        self.assertEqual(report.attrs["avg_factor_baseline_name"], "Avg Factor")
        self.assertEqual(report["avg_factor_baseline_return"].tolist(), [0.03, 0.0])


if __name__ == "__main__":
    unittest.main()
