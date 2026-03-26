import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.native_backtest import run_native_backtest
from reference_backtest import run_reference_backtest


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

    def test_nan_label_rows_are_dropped_before_backtest(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 2.0, 1.0], index=index)
        labels = pd.Series([0.01, -0.01, float("nan"), float("nan")], index=index)

        report = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=1,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
        )

        self.assertEqual(report.index.tolist(), [pd.Timestamp("2024-01-02")])

    def test_missing_label_on_held_stock_is_frozen_without_replacement(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 2.0, 1.0], index=index)
        labels = pd.Series([0.1, 0.0, float("nan"), -0.2], index=index)

        report = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=1,
            n_drop=0,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
        )

        self.assertEqual(int(report.iloc[1]["frozen_holdings"]), 1)
        self.assertEqual(int(report.iloc[1]["holdings"]), 1)
        self.assertEqual(int(report.iloc[1]["buy_count"]), 0)
        self.assertEqual(int(report.iloc[1]["sell_count"]), 0)
        self.assertAlmostEqual(float(report.iloc[1]["net_return"]), 0.0, places=8)

    def test_return_trace_captures_daily_state_transition(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 1.0, 2.0], index=index)
        labels = pd.Series([0.1, 0.0, 0.0, 0.2], index=index)

        report, trace = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=1,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(report.index.tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
        self.assertEqual(trace.index.tolist(), [pd.Timestamp("2024-01-03")])
        self.assertEqual(trace.iloc[0]["sell_list"], ["A"])
        self.assertEqual(trace.iloc[0]["buy_list"], ["B"])
        self.assertEqual(trace.iloc[0]["holdings_before"], {"A": 1100.0})
        self.assertEqual(trace.iloc[0]["holdings_after"], {"B": 1320.0})
        self.assertAlmostEqual(float(trace.iloc[0]["start_value"]), 1100.0, places=8)
        self.assertAlmostEqual(float(trace.iloc[0]["end_value"]), 1320.0, places=8)
        self.assertEqual(int(trace.iloc[0]["buy_count"]), 1)
        self.assertEqual(int(trace.iloc[0]["sell_count"]), 1)

    def test_reference_backtest_matches_native_report(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B", "C"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series(
            [
                3.0, 2.0, 1.0,
                3.0, 1.0, 4.0,
                1.0, 5.0, 4.0,
            ],
            index=index,
        )
        labels = pd.Series(
            [
                0.1, 0.0, -0.05,
                float("nan"), -0.05, 0.1,
                0.0, 0.2, 0.05,
            ],
            index=index,
        )

        native_report = run_native_backtest(
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
        reference_report = run_reference_backtest(
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

        cols = [
            "gross_return",
            "net_return",
            "turnover",
            "cost",
            "bench",
            "buy_count",
            "sell_count",
            "holdings",
            "frozen_holdings",
            "account_value",
            "cum_gross_return",
            "cum_net_return",
        ]
        assert_frame_equal(
            native_report[cols],
            reference_report[cols],
            check_dtype=False,
            atol=1e-10,
            rtol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
