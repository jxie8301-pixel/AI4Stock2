import unittest

import pandas as pd

from src.backtest_trace import parse_trace_dates_arg, select_trace_dates


class BacktestTraceTest(unittest.TestCase):
    def test_parse_trace_dates_arg_handles_empty_and_csv(self):
        self.assertEqual(parse_trace_dates_arg(None), set())
        self.assertEqual(parse_trace_dates_arg(""), set())

        parsed = parse_trace_dates_arg("2024-01-02, 2024-01-03")

        self.assertEqual(parsed, {pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")})

    def test_select_trace_dates_combines_extremes_and_drawdown(self):
        report = pd.DataFrame(
            {
                "return": [0.01, -0.20, 0.03, -0.04],
                "turnover": [0.1, 0.05, 0.4, 0.2],
                "cost": [0.001, 0.003, 0.002, 0.02],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        )

        dates = select_trace_dates(report, top_n=1)

        self.assertEqual(
            dates,
            [
                pd.Timestamp("2024-01-03"),  # largest abs return and deepest drawdown
                pd.Timestamp("2024-01-04"),  # largest turnover
                pd.Timestamp("2024-01-05"),  # largest cost
            ],
        )


if __name__ == "__main__":
    unittest.main()
