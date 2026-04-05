import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.native_backtest import run_native_backtest
from reference_backtest import run_reference_backtest


class NativeBacktestTest(unittest.TestCase):
    def test_equal_weighting_rebalances_existing_holdings_to_equal_targets(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 2.0, 1.0], index=index)
        labels = pd.Series([0.1, -0.1, 0.0, 0.0], index=index)

        report, trace = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=2,
            n_drop=0,
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
        self.assertEqual(trace.iloc[0]["weighting"], "equal")
        self.assertEqual(trace.iloc[0]["sell_list"], [])
        self.assertEqual(trace.iloc[0]["buy_list"], [])
        self.assertEqual(trace.iloc[0]["trade_sell_list"], ["A"])
        self.assertEqual(trace.iloc[0]["trade_buy_list"], ["B"])
        self.assertAlmostEqual(trace.iloc[0]["target_weights"]["A"], 0.5, places=8)
        self.assertAlmostEqual(trace.iloc[0]["target_weights"]["B"], 0.5, places=8)
        self.assertAlmostEqual(trace.iloc[0]["holdings_after"]["A"], 500.0, places=8)
        self.assertAlmostEqual(trace.iloc[0]["holdings_after"]["B"], 500.0, places=8)

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
        self.assertEqual(trace.iloc[0]["trade_sell_list"], ["A"])
        self.assertEqual(trace.iloc[0]["trade_buy_list"], ["B"])
        self.assertEqual(trace.iloc[0]["holdings_before"], {"A": 1100.0})
        self.assertEqual(trace.iloc[0]["holdings_after"], {"B": 1320.0})
        self.assertAlmostEqual(float(trace.iloc[0]["start_value"]), 1100.0, places=8)
        self.assertAlmostEqual(float(trace.iloc[0]["end_value"]), 1320.0, places=8)
        self.assertEqual(int(trace.iloc[0]["buy_count"]), 1)
        self.assertEqual(int(trace.iloc[0]["sell_count"]), 1)

    def test_rank_weighting_allocates_more_capital_to_higher_rank(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0], index=index)
        labels = pd.Series([0.0, 0.0], index=index)

        report, trace = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=2,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=900.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
            weighting="rank",
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-02")},
        )

        self.assertEqual(report.index.tolist(), [pd.Timestamp("2024-01-02")])
        self.assertAlmostEqual(trace.iloc[0]["target_weights"]["A"], 2.0 / 3.0, places=8)
        self.assertAlmostEqual(trace.iloc[0]["target_weights"]["B"], 1.0 / 3.0, places=8)
        self.assertAlmostEqual(trace.iloc[0]["holdings_after"]["A"], 600.0, places=8)
        self.assertAlmostEqual(trace.iloc[0]["holdings_after"]["B"], 300.0, places=8)

    def test_score_softmax_respects_max_weight_cap(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A", "B", "C"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([10.0, 1.0, 0.0], index=index)
        labels = pd.Series([0.0, 0.0, 0.0], index=index)

        _, trace = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=3,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
            weighting="score_softmax",
            max_weight=0.5,
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-02")},
        )

        target_weights = trace.iloc[0]["target_weights"]
        self.assertLessEqual(target_weights["A"], 0.5 + 1e-12)
        self.assertAlmostEqual(sum(target_weights.values()), 1.0, places=8)
        self.assertGreater(target_weights["A"], target_weights["B"])
        self.assertGreater(target_weights["B"], target_weights["C"])

    def test_rank_pct_score_transform_changes_scale_but_preserves_softmax_order(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A", "B", "C"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([10.0, 2.0, -1.0], index=index)
        labels = pd.Series([0.0, 0.0, 0.0], index=index)

        _, trace = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=3,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
            weighting="score_softmax",
            score_transform="rank_pct",
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-02")},
        )

        weights = trace.iloc[0]["target_weights"]
        self.assertEqual(trace.iloc[0]["score_transform"], "rank_pct")
        self.assertGreater(weights["A"], weights["B"])
        self.assertGreater(weights["B"], weights["C"])

    def test_zscore_clip_score_transform_supports_positive_score_floor(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02"]),
                ["A", "B", "C"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([10.0, 0.0, -10.0], index=index)
        labels = pd.Series([0.0, 0.0, 0.0], index=index)

        _, trace = run_native_backtest(
            preds=preds,
            labels=labels,
            topk=3,
            n_drop=1,
            cost_buy=0.0,
            cost_sell=0.0,
            min_cost=0.0,
            account=1000.0,
            risk_degree=1.0,
            slippage=0.0,
            rebalance_freq=1,
            weighting="rank",
            score_transform="zscore_clip",
            score_zscore_clip=1.0,
            min_score=0.0,
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-02")},
        )

        self.assertEqual(trace.iloc[0]["buy_list"], ["A"])
        self.assertEqual(trace.iloc[0]["target_weights"], {"A": 1.0})

    def test_min_score_zero_liquidates_negative_score_holdings(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, -0.2], index=index)
        labels = pd.Series([0.1, 0.0, 0.0, 0.0], index=index)

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
            min_score=0.0,
            score_transform="none",
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(report.index.tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
        self.assertEqual(trace.iloc[0]["sell_list"], ["A"])
        self.assertEqual(trace.iloc[0]["buy_list"], [])
        self.assertEqual(trace.iloc[0]["trade_sell_list"], ["A"])
        self.assertEqual(trace.iloc[0]["trade_buy_list"], [])
        self.assertEqual(trace.iloc[0]["holdings_after"], {})
        self.assertEqual(trace.iloc[0]["target_weights"], {})

    def test_keep_top_n_buffers_ranked_holdings_against_forced_rotation(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B", "C"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([3.0, 2.0, 1.0, 2.0, 1.0, 3.0], index=index)
        labels = pd.Series([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)

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
            keep_top_n=2,
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(report.index.tolist(), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
        self.assertEqual(trace.iloc[0]["sell_list"], [])
        self.assertEqual(trace.iloc[0]["buy_list"], [])
        self.assertEqual(trace.iloc[0]["trade_sell_list"], [])
        self.assertEqual(trace.iloc[0]["trade_buy_list"], [])
        self.assertEqual(trace.iloc[0]["keep_top_n"], 2)
        self.assertEqual(set(trace.iloc[0]["holdings_after"].keys()), {"A"})

    def test_risk_control_benchmark_ma_uses_lagged_schedule(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 2.0, 1.0, 2.0, 1.0], index=index)
        labels = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=index)
        benchmark_returns = pd.Series(
            [0.0, -0.5, 0.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

        report, trace = run_native_backtest(
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
            benchmark_returns=benchmark_returns,
            risk_control={
                "mode": "benchmark_ma",
                "fast_window": 2,
                "slow_window": 3,
                "bull_risk": 1.0,
                "neutral_risk": 0.5,
                "bear_risk": 0.0,
            },
            return_trace=True,
            trace_dates={
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-04"),
            },
        )

        self.assertEqual(report["risk_degree"].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(trace["risk_control_mode"].tolist(), ["benchmark_ma", "benchmark_ma", "benchmark_ma"])
        self.assertAlmostEqual(float(trace.loc[pd.Timestamp("2024-01-04"), "risk_degree"]), 0.0, places=8)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-04"), "holdings_after"], {})

    def test_legacy_dynamic_risk_alias_still_works(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 2.0, 1.0], index=index)
        labels = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)
        benchmark_returns = pd.Series(
            [0.0, -0.2],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )

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
            benchmark_returns=benchmark_returns,
            dynamic_risk={
                "mode": "benchmark_ma",
                "fast_window": 2,
                "slow_window": 3,
                "bull_risk": 1.0,
                "neutral_risk": 0.5,
                "bear_risk": 0.0,
            },
        )

        self.assertIn("risk_degree", report.columns)

    def test_signal_strength_risk_control_scales_exposure_from_scores(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, 0.5, 0.4], index=index)
        labels = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)

        report, trace = run_native_backtest(
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
            risk_control={
                "mode": "signal_strength",
                "signal_metric": "topk_mean",
                "min_signal": 0.0,
                "max_signal": 2.0,
                "min_risk": 0.2,
                "max_risk": 1.0,
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")},
        )

        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-02"), "risk_degree"]), 1.0, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "risk_degree"]), 0.4, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-02"), "risk_control_signal"]), 2.0, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "risk_control_signal"]), 0.5, places=8)
        self.assertEqual(trace["risk_control_mode"].tolist(), ["signal_strength", "signal_strength"])
        self.assertAlmostEqual(trace.loc[pd.Timestamp("2024-01-03"), "holdings_after"]["A"], 400.0, places=8)

    def test_signal_strength_quantiles_use_lagged_history(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([1.0, 0.0, 3.0, 0.0, 2.0, 0.0], index=index)
        labels = pd.Series(0.0, index=index)

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
            risk_control={
                "mode": "signal_strength",
                "signal_metric": "top1",
                "min_signal": 0.0,
                "max_signal": 4.0,
                "min_signal_quantile": 0.25,
                "max_signal_quantile": 0.75,
                "min_risk": 0.2,
                "max_risk": 1.0,
            },
        )

        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-02"), "risk_degree"]), 0.4, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "risk_degree"]), 0.8, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-04"), "risk_degree"]), 0.6, places=8)

    def test_benchmark_ma_signal_strength_caps_signal_risk(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([3.0, 0.0, 3.0, 0.0, 3.0, 0.0], index=index)
        labels = pd.Series(0.0, index=index)
        benchmark_returns = pd.Series(
            [0.0, -0.5, 0.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

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
            benchmark_returns=benchmark_returns,
            risk_control={
                "mode": "benchmark_ma_signal_strength",
                "fast_window": 2,
                "slow_window": 3,
                "bull_risk": 1.0,
                "neutral_risk": 0.5,
                "bear_risk": 0.0,
                "signal_metric": "top1",
                "min_signal": 0.0,
                "max_signal": 3.0,
                "min_risk": 0.4,
                "max_risk": 1.0,
            },
        )

        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-02"), "risk_degree"]), 1.0, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "risk_degree"]), 1.0, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-04"), "risk_degree"]), 0.0, places=8)

    def test_intraperiod_exit_sells_negative_score_holdings_between_rebalances(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, 0.2, 0.5, 0.1], index=index)
        labels = pd.Series([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], index=index)

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "raw",
                "threshold": 0.0,
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "net_return"]), 0.0, places=8)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "sell_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_count"]), 1)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_residual_mean"]), -0.2, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_saved_return"]), 0.2, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_missed_return"]), 0.0, places=8)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_beneficial_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_harmful_count"]), 0)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "holdings"]), 0)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "trade_sell_list"], ["A"])
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "holdings_after"], {})
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_mode"], "score_threshold")
        self.assertAlmostEqual(
            float(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_residual_values"]["A"]),
            -0.2,
            places=8,
        )
        self.assertAlmostEqual(
            float(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["saved_return_contribution"]),
            0.2,
            places=8,
        )

    def test_intraperiod_exit_rank_pct_uses_cross_sectional_threshold(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03"]),
                ["A", "B", "C"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([3.0, 2.0, 1.0, 1.0, 2.0, 3.0], index=index)
        labels = pd.Series([0.0, 0.0, 0.0, -0.2, 0.0, 0.0], index=index)

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "rank_pct",
                "threshold": 0.5,
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_count"]), 1)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "trade_sell_list"], ["A"])
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "holdings_after"], {})
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_score_source"], "rank_pct")

    def test_expected_return_intraperiod_exit_uses_calibrated_future_return(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-08",
                        "2024-01-09",
                    ]
                ),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series(
            [
                2.0, 1.0,
                0.2, 0.8,
                0.3, 0.7,
                3.0, 1.0,
                0.25, 0.9,
                0.2, 1.0,
            ],
            index=index,
        )
        labels = pd.Series(
            [
                0.0, 0.0,
                -0.05, 0.02,
                -0.05, 0.02,
                0.0, 0.0,
                -0.10, 0.01,
                -0.05, 0.01,
            ],
            index=index,
        )

        report, trace = run_native_backtest(
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
            rebalance_freq=3,
            intraperiod_exit={
                "mode": "expected_return_threshold",
                "score_source": "raw",
                "threshold": 0.0,
                "calibration": "quantile_bins",
                "n_bins": 2,
                "min_history": 2,
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-08")},
        )

        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-08"), "intraperiod_exit_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-08"), "sell_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-08"), "holdings"]), 0)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-08"), "net_return"]), 0.0, places=8)
        self.assertLess(float(report.loc[pd.Timestamp("2024-01-08"), "intraperiod_exit_signal_min"]), 0.0)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-08"), "trade_sell_list"], ["A"])
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-08"), "intraperiod_exit_mode"], "expected_return_threshold")
        self.assertLess(float(trace.loc[pd.Timestamp("2024-01-08"), "intraperiod_exit_signal_values"]["A"]), 0.0)
        self.assertLess(float(trace.loc[pd.Timestamp("2024-01-08"), "intraperiod_exit_residual_values"]["A"]), 0.0)

    def test_intraperiod_exit_tracks_missed_gain_when_threshold_sells_winner(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, 0.2, 0.5, 0.1], index=index)
        labels = pd.Series([0.0, 0.0, 0.1, 0.0, 0.0, 0.0], index=index)

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "raw",
                "threshold": 0.0,
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_residual_mean"]), 0.1, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_saved_return"]), 0.0, places=8)
        self.assertAlmostEqual(float(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_missed_return"]), 0.1, places=8)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_beneficial_count"]), 0)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_harmful_count"]), 1)
        self.assertAlmostEqual(
            float(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["missed_return_contribution"]),
            0.1,
            places=8,
        )

    def test_intraperiod_exit_price_confirm_blocks_score_only_exit_without_close_break(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, 0.2, 0.5, 0.1], index=index)
        labels = pd.Series([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], index=index)
        market_data = pd.DataFrame(
            {
                "close": [100.0, 50.0, 100.5, 51.0, 98.0, 52.0],
            },
            index=index,
        )

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            market_data=market_data,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "raw",
                "threshold": 0.0,
                "price_confirm": {
                    "mode": "close_below_ma",
                    "ma_window": 2,
                },
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_score_candidate_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_required_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_blocked_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_count"]), 0)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "holdings"]), 1)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "trade_sell_list"], [])
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_mode"], "close_below_ma")
        self.assertEqual(int(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_required_count"]), 1)
        self.assertEqual(int(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_blocked_count"]), 1)

    def test_intraperiod_exit_price_confirm_allows_exit_after_close_breaks_ma(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, 0.2, 0.5, 0.1], index=index)
        labels = pd.Series([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], index=index)
        market_data = pd.DataFrame(
            {
                "close": [100.0, 50.0, 94.0, 51.0, 98.0, 52.0],
            },
            index=index,
        )

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            market_data=market_data,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "raw",
                "threshold": 0.0,
                "price_confirm": {
                    "mode": "close_below_ma",
                    "ma_window": 2,
                },
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_score_candidate_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_required_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_blocked_count"]), 0)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_count"]), 1)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "trade_sell_list"], ["A"])
        self.assertTrue(bool(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["price_confirm_required"]))
        self.assertTrue(bool(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["price_confirm_passed"]))

    def test_intraperiod_exit_price_confirm_bypasses_on_force_exit_threshold(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, 0.2, 0.5, 0.1], index=index)
        labels = pd.Series([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], index=index)
        market_data = pd.DataFrame(
            {
                "close": [100.0, 50.0, 100.5, 51.0, 98.0, 52.0],
            },
            index=index,
        )

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            market_data=market_data,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "raw",
                "threshold": 0.0,
                "price_confirm": {
                    "mode": "close_below_ma",
                    "ma_window": 2,
                    "force_exit_threshold": -0.05,
                },
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_score_candidate_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_required_count"]), 0)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_bypassed_force_exit_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_count"]), 1)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "trade_sell_list"], ["A"])
        self.assertFalse(bool(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["price_confirm_required"]))
        self.assertEqual(
            trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["price_confirm_bypass_reason"],
            "force_exit_threshold",
        )

    def test_intraperiod_exit_price_confirm_bypasses_near_rebalance(self):
        index = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                ["A", "B"],
            ],
            names=["datetime", "instrument"],
        )
        preds = pd.Series([2.0, 1.0, -0.1, 0.2, 0.5, 0.1], index=index)
        labels = pd.Series([0.0, 0.0, -0.2, 0.0, 0.0, 0.0], index=index)
        market_data = pd.DataFrame(
            {
                "close": [100.0, 50.0, 100.5, 51.0, 98.0, 52.0],
            },
            index=index,
        )

        report, trace = run_native_backtest(
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
            rebalance_freq=2,
            market_data=market_data,
            intraperiod_exit={
                "mode": "score_threshold",
                "score_source": "raw",
                "threshold": 0.0,
                "price_confirm": {
                    "mode": "close_below_ma",
                    "ma_window": 2,
                    "min_remaining_steps": 2,
                },
            },
            return_trace=True,
            trace_dates={pd.Timestamp("2024-01-03")},
        )

        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_score_candidate_count"]), 1)
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_required_count"]), 0)
        self.assertEqual(
            int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_price_confirm_bypassed_remaining_steps_count"]),
            1,
        )
        self.assertEqual(int(report.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_count"]), 1)
        self.assertEqual(trace.loc[pd.Timestamp("2024-01-03"), "trade_sell_list"], ["A"])
        self.assertFalse(bool(trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["price_confirm_required"]))
        self.assertEqual(
            trace.loc[pd.Timestamp("2024-01-03"), "intraperiod_exit_events"][0]["price_confirm_bypass_reason"],
            "remaining_steps",
        )

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
