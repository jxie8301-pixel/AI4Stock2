import unittest

import pandas as pd

from src.gen_feature import (
    TUSHARE_FACTOR_PREFIX,
    compute_all_factor_features,
    compute_tushare_factor_features,
    get_all_factor_feature_names,
    get_tushare_factor_feature_names,
)


class TushareFeatureTest(unittest.TestCase):
    def test_tushare_feature_names_are_appended_only_for_tushare_source(self):
        default_names = get_all_factor_feature_names()
        tushare_names = get_all_factor_feature_names(data_source="tushare")

        self.assertEqual(len(default_names), 279)
        self.assertEqual(len(tushare_names), len(default_names) + len(get_tushare_factor_feature_names()))
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", tushare_names)
        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", default_names)
        self.assertEqual(len(tushare_names), len(set(tushare_names)))

    def test_compute_tushare_factor_features_uses_processed_market_fields(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"]),
                "symbol": ["000001"] * 3,
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.8, 10.8, 11.8],
                "close": [10.0, 11.0, 12.0],
                "pre_close": [9.5, 10.0, 11.0],
                "volume": [100.0, 120.0, 140.0],
                "amount": [1000.0, 1400.0, 1800.0],
                "turnover": [1.0, 2.0, 4.0],
                "turnover_free": [1.5, 3.0, 6.0],
                "volume_ratio": [1.2, 1.3, 1.4],
                "total_mv": [1000.0, 1100.0, 1200.0],
                "circ_mv": [800.0, 880.0, 960.0],
                "total_share": [100.0, 100.0, 100.0],
                "circ_share": [80.0, 80.0, 80.0],
                "free_share": [60.0, 60.0, 60.0],
                "pe": [10.0, 20.0, -5.0],
                "pe_ttm": [8.0, 16.0, 32.0],
                "ps": [2.0, 4.0, 5.0],
                "ps_ttm": [2.5, 5.0, 10.0],
                "dv_ratio": [0.5, 0.0, 0.2],
                "dv_ttm": [0.8, 0.0, 0.4],
                "amplitude": [7.0, 8.0, 9.0],
                "pct_chg": [5.0, 10.0, -3.0],
                "limit_pre_close": [10.0, 11.0, 12.0],
                "up_limit": [11.0, 12.1, 13.2],
                "down_limit": [9.0, 9.9, 10.8],
            }
        )

        feat = compute_tushare_factor_features(df)

        self.assertEqual(feat.columns.tolist(), get_tushare_factor_feature_names())
        self.assertAlmostEqual(float(feat.iloc[0]["free_float_ratio"]), 0.6, places=6)
        self.assertAlmostEqual(float(feat.iloc[1]["gap_up_limit"]), 0.1, places=6)
        self.assertAlmostEqual(float(feat.iloc[1]["gap_down_limit"]), 11.0 / 9.9 - 1.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[1]["free_turnover_ratio"]), 1.5, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["free_turnover_mean_5"]), (1.5 + 3.0 + 6.0) / 3.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[0]["float_mv_ratio"]), 0.8, places=6)
        self.assertAlmostEqual(float(feat.iloc[0]["ep"]), 0.1, places=6)
        self.assertAlmostEqual(float(feat.iloc[1]["ep_ttm_gap"]), 0.05 - 0.0625, places=6)
        self.assertAlmostEqual(float(feat.iloc[0]["sp"]), 0.5, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["sp_ttm"]), 0.1, places=6)
        self.assertEqual(float(feat.iloc[0]["has_dividend"]), 1.0)
        self.assertEqual(float(feat.iloc[1]["has_dividend"]), 0.0)
        self.assertAlmostEqual(float(feat.iloc[2]["limit_band_pct_mean_5"]), 0.2, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["limit_band_pos_mean_5"]), 0.5, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["gap_up_limit_mean_5"]), 0.1, places=6)
        self.assertAlmostEqual(float(feat.iloc[0]["hit_up_limit_count_5"]), 0.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["amplitude_mean_5"]), 8.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["pct_chg_mean_5"]), 4.0, places=6)
        self.assertTrue(pd.notna(feat.iloc[2]["amplitude_zscore_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["pct_chg_zscore_20"]))
        self.assertTrue(pd.isna(feat.iloc[0]["free_float_ratio_change_20"]))
        self.assertTrue(pd.isna(feat.iloc[0]["sp_ttm_change_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["free_turnover_ratio_zscore_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["volume_ratio_raw_zscore_20"]))

    def test_compute_all_factor_features_adds_tushare_columns_only_for_tushare_source(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-30", "2026-03-31"]),
                "symbol": ["000001", "000001"],
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.8, 10.8],
                "close": [10.0, 11.0],
                "pre_close": [9.5, 10.0],
                "volume": [100.0, 120.0],
                "amount": [1000.0, 1400.0],
                "turnover": [1.0, 2.0],
                "turnover_free": [1.5, 3.0],
                "volume_ratio": [1.2, 1.3],
                "total_mv": [1000.0, 1100.0],
                "total_share": [100.0, 100.0],
                "circ_share": [80.0, 80.0],
                "free_share": [60.0, 60.0],
                "pe": [10.0, 11.0],
                "pe_ttm": [10.0, 11.0],
                "pb": [1.0, 1.1],
                "ps": [2.0, 4.0],
                "ps_ttm": [2.5, 5.0],
                "dv_ratio": [0.5, 0.0],
                "dv_ttm": [0.8, 0.0],
                "circ_mv": [1000.0, 1100.0],
                "amplitude": [7.0, 8.0],
                "pct_chg": [5.0, 10.0],
                "limit_pre_close": [10.0, 11.0],
                "up_limit": [11.0, 12.1],
                "down_limit": [9.0, 9.9],
            }
        )

        default_feat = compute_all_factor_features(df)
        tushare_feat = compute_all_factor_features(df, data_source="tushare")

        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", default_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}dividend_yield_ttm", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}float_mv_ratio", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}amplitude_mean_5", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}limit_band_pos_mean_5", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}free_turnover_ratio_zscore_20", tushare_feat.columns)


if __name__ == "__main__":
    unittest.main()
