import unittest

import pandas as pd

from src.backtest import run_backtest


class BacktestWrapperTest(unittest.TestCase):
    def test_run_backtest_returns_legacy_compatible_tuple(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A"],
            ],
            names=["datetime", "instrument"],
        )
        predictions = pd.Series([1.0], index=index)
        labels = pd.Series([0.1], index=index)

        report, indicator = run_backtest(
            predictions=predictions,
            labels=labels,
            topk=1,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
        )

        self.assertIsNone(indicator)
        self.assertIn("return", report.columns)
        self.assertIn("bench", report.columns)
        self.assertNotIn("net_return", report.columns)
        self.assertAlmostEqual(float(report.iloc[0]["return"]), 0.1, places=8)
        self.assertAlmostEqual(float(report.iloc[0]["bench"]), 0.1, places=8)

    def test_run_backtest_can_return_trace(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A"],
            ],
            names=["datetime", "instrument"],
        )
        predictions = pd.Series([1.0], index=index)
        labels = pd.Series([0.1], index=index)

        portfolio_metric, trace = run_backtest(
            predictions=predictions,
            labels=labels,
            topk=1,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-02")},
        )
        report, indicator = portfolio_metric

        self.assertIsNone(indicator)
        self.assertEqual(report.index.tolist(), [pd.Timestamp("2024-01-02")])
        self.assertEqual(trace.index.tolist(), [pd.Timestamp("2024-01-02")])
        self.assertIn("holdings_before", trace.columns)
        self.assertEqual(trace.iloc[0]["buy_list"], ["A"])


if __name__ == "__main__":
    unittest.main()
