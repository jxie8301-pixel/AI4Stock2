import unittest

import pandas as pd

from src.native_backtest import run_native_backtest


class NativeBacktestTest(unittest.TestCase):
    def test_n_drop_only_replaces_limited_positions(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B", "C", "D"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series(
            [
                4.0, 3.0, 2.0, 1.0,
                3.0, 2.0, 4.0, 1.0,
            ],
            index=index,
        )
        labels = pd.Series(0.0, index=index)

        report = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=2,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
        )

        self.assertEqual(int(report.iloc[0]["buy_count"]), 2)
        self.assertEqual(int(report.iloc[1]["buy_count"]), 1)
        self.assertEqual(int(report.iloc[1]["sell_count"]), 1)
        self.assertEqual(int(report.iloc[1]["holdings"]), 2)

    def test_min_cost_is_applied_on_initial_entry(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0], index=index)
        labels = pd.Series(0.0, index=index)

        report = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=2,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=10.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
        )

        self.assertAlmostEqual(report.iloc[0]["cost"], 0.02, places=8)
        self.assertAlmostEqual(report.iloc[0]["net_return"], -0.02, places=8)
        self.assertEqual(int(report.iloc[0]["buy_count"]), 2)


if __name__ == "__main__":
    unittest.main()
