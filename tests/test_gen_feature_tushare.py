import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.gen_feature import (
    TUSHARE_FACTOR_PREFIX,
    TUSHARE_INDUSTRY_CONTEXT_PATH,
    TUSHARE_RAW_DIVIDEND_DIR,
    TUSHARE_RAW_EXPRESS_DIR,
    TUSHARE_RAW_FINA_INDICATOR_DIR,
    TUSHARE_RAW_FORECAST_DIR,
    TUSHARE_SYMBOL_CACHE_PATH,
    _clear_tushare_context_caches,
    _augment_tushare_symbol_frame,
    compute_all_factor_features,
    compute_tushare_factor_features,
    get_all_factor_feature_names,
    get_tushare_factor_feature_names,
)


class TushareFeatureTest(unittest.TestCase):
    def test_tushare_feature_names_are_appended_only_for_tushare_source(self):
        default_names = get_all_factor_feature_names()
        tushare_names = get_all_factor_feature_names(data_source="tushare")

        self.assertEqual(len(default_names), 259)
        self.assertEqual(len(tushare_names), len(default_names) + len(get_tushare_factor_feature_names()))
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", tushare_names)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_ret_20", tushare_names)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}turnover_accel_5_20", tushare_names)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_20", tushare_names)
        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", default_names)
        self.assertNotIn("TEMP_ret_20", default_names)
        self.assertNotIn("TEMP_corr_cv_20", default_names)
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
                "pb": [1.0, 1.1, 1.2],
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
                "ind_member_count": [6.0, 6.0, 6.0],
                "ind_daily_ret": [0.01, 0.02, -0.01],
                "ind_excess_daily_ret": [0.005, 0.015, -0.005],
                "ind_ret_5": [0.03, 0.04, 0.05],
                "ind_ret_20": [0.10, 0.11, 0.12],
                "ind_ret_60": [0.20, 0.21, 0.22],
                "ind_std_5": [0.02, 0.02, 0.02],
                "ind_std_20": [0.03, 0.03, 0.03],
                "ind_std_60": [0.04, 0.04, 0.04],
                "ind_excess_ret_5": [0.01, 0.015, 0.02],
                "ind_excess_ret_20": [0.02, 0.025, 0.03],
                "ind_excess_ret_60": [0.04, 0.045, 0.05],
                "ind_pos_rate_5": [0.5, 0.55, 0.6],
                "ind_pos_rate_20": [0.52, 0.57, 0.62],
                "ind_pos_rate_60": [0.54, 0.59, 0.64],
                "ind_dispersion_5": [0.01, 0.011, 0.012],
                "ind_dispersion_20": [0.02, 0.021, 0.022],
                "ind_dispersion_60": [0.03, 0.031, 0.032],
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
        self.assertAlmostEqual(float(feat.iloc[0]["industry_member_count"]), 6.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[1]["industry_daily_ret"]), 0.02, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["industry_ret_20"]), 0.12, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["industry_excess_ret_60"]), 0.05, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["industry_std_20"]), 0.03, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["industry_pos_rate_20"]), 0.62, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["industry_dispersion_60"]), 0.032, places=6)
        self.assertTrue(pd.isna(feat.iloc[2]["industry_rel_ret_5"]))
        self.assertAlmostEqual(float(feat.iloc[2]["turnover_accel_5_20"]), 0.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["free_turnover_accel_5_20"]), 0.0, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["amihud_term_5_20"]), 0.0, places=6)
        self.assertTrue(pd.isna(feat.iloc[2]["downside_amihud_20"]))
        self.assertAlmostEqual(float(feat.iloc[2]["stock_vs_industry_std_ratio_20"]), 0.214275, places=6)
        self.assertTrue(pd.notna(feat.iloc[2]["amplitude_zscore_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["pct_chg_zscore_20"]))
        self.assertTrue(pd.isna(feat.iloc[0]["free_float_ratio_change_20"]))
        self.assertTrue(pd.isna(feat.iloc[0]["sp_ttm_change_20"]))
        self.assertTrue(pd.isna(feat.iloc[0]["ep_ttm_change_20"]))
        self.assertTrue(pd.isna(feat.iloc[0]["bp_change_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["free_turnover_ratio_zscore_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["free_turnover_spread_zscore_20"]))
        self.assertTrue(pd.notna(feat.iloc[2]["volume_ratio_raw_zscore_20"]))
        self.assertAlmostEqual(float(feat.iloc[1]["dividend_yield_ttm_surprise_20"]), -0.8, places=6)
        self.assertAlmostEqual(float(feat.iloc[2]["dividend_yield_ttm_surprise_20"]), 0.0, places=6)

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
                "ind_member_count": [6.0, 6.0],
                "ind_daily_ret": [0.01, 0.02],
                "ind_excess_daily_ret": [0.005, 0.015],
                "ind_ret_5": [0.03, 0.04],
                "ind_ret_20": [0.10, 0.11],
                "ind_ret_60": [0.20, 0.21],
                "ind_std_5": [0.02, 0.02],
                "ind_std_20": [0.03, 0.03],
                "ind_std_60": [0.04, 0.04],
                "ind_excess_ret_5": [0.01, 0.015],
                "ind_excess_ret_20": [0.02, 0.025],
                "ind_excess_ret_60": [0.04, 0.045],
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
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_ret_20", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_rel_ret_5", tushare_feat.columns)

    def test_augment_tushare_symbol_frame_loads_fina_indicator_sidecar(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"]),
                "symbol": ["000001", "000001", "000001"],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.8, 10.8, 11.8],
                "close": [10.0, 11.0, 12.0],
                "volume": [100.0, 120.0, 140.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            sidecar = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "ann_date": ["20260330", "20260331"],
                    "end_date": ["20251231", "20260331"],
                    "roe": [8.0, 9.0],
                    "roe_dt": [7.5, 8.5],
                    "or_yoy": [10.0, 12.0],
                    "netprofit_yoy": [15.0, 18.0],
                }
            )
            sidecar.to_parquet(raw_dir / "000001.parquet", index=False)

            from src import gen_feature as gen_feature_module

            original_dir = gen_feature_module.TUSHARE_RAW_FINA_INDICATOR_DIR
            gen_feature_module.TUSHARE_RAW_FINA_INDICATOR_DIR = raw_dir
            try:
                out = _augment_tushare_symbol_frame(df, symbol="000001")
            finally:
                gen_feature_module.TUSHARE_RAW_FINA_INDICATOR_DIR = original_dir

        self.assertAlmostEqual(float(out.loc[0, "fi_roe"]), 8.0, places=6)
        self.assertAlmostEqual(float(out.loc[2, "fi_roe"]), 9.0, places=6)
        self.assertAlmostEqual(float(out.loc[2, "fi_roe_dt"]), 8.5, places=6)
        self.assertAlmostEqual(float(out.loc[2, "fi_or_yoy"]), 12.0, places=6)
        self.assertAlmostEqual(float(out.loc[2, "fi_netprofit_yoy"]), 18.0, places=6)

    def test_compute_all_factor_features_uses_side_loaded_fina_indicator_columns(self):
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
                "circ_mv": [1000.0, 1100.0],
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
                "amplitude": [7.0, 8.0],
                "pct_chg": [5.0, 10.0],
                "limit_pre_close": [10.0, 11.0],
                "up_limit": [11.0, 12.1],
                "down_limit": [9.0, 9.9],
                "fi_roe": [8.0, 9.0],
                "fi_roe_dt": [7.5, 8.5],
                "fi_or_yoy": [10.0, 12.0],
                "fi_netprofit_yoy": [15.0, 18.0],
            }
        )

        tushare_feat = compute_all_factor_features(df, data_source="tushare")

        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}latest_roe", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}latest_or_yoy", tushare_feat.columns)
        self.assertAlmostEqual(float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}latest_roe"]), 9.0, places=6)
        self.assertAlmostEqual(float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}latest_or_yoy"]), 12.0, places=6)

    def test_augment_tushare_symbol_frame_loads_dividend_sidecar(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"]),
                "symbol": ["000001", "000001", "000001"],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.8, 10.8, 11.8],
                "close": [10.0, 11.0, 12.0],
                "volume": [100.0, 120.0, 140.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            sidecar = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "ann_date": ["20260330", "20260331"],
                    "cash_div": [0.3, 0.5],
                    "cash_div_tax": [0.25, 0.4],
                    "stk_div": [0.0, 0.1],
                    "stk_bo_rate": [0.0, 0.2],
                    "stk_co_rate": [0.0, 0.0],
                    "base_share": [100.0, 110.0],
                }
            )
            sidecar.to_parquet(raw_dir / "000001.parquet", index=False)

            from src import gen_feature as gen_feature_module

            original_dir = gen_feature_module.TUSHARE_RAW_DIVIDEND_DIR
            gen_feature_module.TUSHARE_RAW_DIVIDEND_DIR = raw_dir
            try:
                out = _augment_tushare_symbol_frame(df, symbol="000001")
            finally:
                gen_feature_module.TUSHARE_RAW_DIVIDEND_DIR = original_dir

        self.assertAlmostEqual(float(out.loc[0, "div_cash_div"]), 0.3, places=6)
        self.assertAlmostEqual(float(out.loc[2, "div_cash_div"]), 0.5, places=6)
        self.assertAlmostEqual(float(out.loc[2, "div_stk_div"]), 0.1, places=6)
        self.assertAlmostEqual(float(out.loc[2, "div_stk_bo_rate"]), 0.2, places=6)

    def test_compute_all_factor_features_uses_side_loaded_dividend_columns(self):
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
                "circ_mv": [1000.0, 1100.0],
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
                "amplitude": [7.0, 8.0],
                "pct_chg": [5.0, 10.0],
                "limit_pre_close": [10.0, 11.0],
                "up_limit": [11.0, 12.1],
                "down_limit": [9.0, 9.9],
                "div_cash_div": [0.3, 0.5],
                "div_cash_div_tax": [0.25, 0.4],
                "div_stk_div": [0.0, 0.1],
                "div_stk_bo_rate": [0.0, 0.2],
                "div_stk_co_rate": [0.0, 0.0],
                "div_base_share": [100.0, 110.0],
            }
        )

        tushare_feat = compute_all_factor_features(df, data_source="tushare")

        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}latest_div_cash", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}latest_div_cash_yield_proxy", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}has_stock_dividend", tushare_feat.columns)
        self.assertAlmostEqual(float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}latest_div_cash"]), 0.5, places=6)
        self.assertAlmostEqual(
            float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}latest_div_cash_yield_proxy"]),
            0.5 / 11.0,
            places=6,
        )
        self.assertEqual(float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}has_stock_dividend"]), 1.0)

    def test_augment_tushare_symbol_frame_loads_forecast_and_express_sidecars(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"]),
                "symbol": ["000001", "000001", "000001"],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.8, 10.8, 11.8],
                "close": [10.0, 11.0, 12.0],
                "volume": [100.0, 120.0, 140.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            forecast_dir = Path(tmpdir) / "forecast"
            express_dir = Path(tmpdir) / "express"
            forecast_dir.mkdir()
            express_dir.mkdir()
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "ann_date": ["20260330", "20260331"],
                    "p_change_min": [10.0, 20.0],
                    "p_change_max": [30.0, 40.0],
                    "net_profit_min": [100.0, 200.0],
                    "net_profit_max": [300.0, 400.0],
                    "last_parent_net": [50.0, 60.0],
                }
            ).to_parquet(forecast_dir / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ"],
                    "ann_date": ["20260330", "20260331"],
                    "revenue": [1000.0, 1100.0],
                    "operate_profit": [100.0, 120.0],
                    "n_income": [80.0, 90.0],
                    "yoy_sales": [15.0, 18.0],
                }
            ).to_parquet(express_dir / "000001.parquet", index=False)

            from src import gen_feature as gen_feature_module

            original_forecast = gen_feature_module.TUSHARE_RAW_FORECAST_DIR
            original_express = gen_feature_module.TUSHARE_RAW_EXPRESS_DIR
            gen_feature_module.TUSHARE_RAW_FORECAST_DIR = forecast_dir
            gen_feature_module.TUSHARE_RAW_EXPRESS_DIR = express_dir
            try:
                out = _augment_tushare_symbol_frame(df, symbol="000001")
            finally:
                gen_feature_module.TUSHARE_RAW_FORECAST_DIR = original_forecast
                gen_feature_module.TUSHARE_RAW_EXPRESS_DIR = original_express

        self.assertAlmostEqual(float(out.loc[2, "fc_p_change_max"]), 40.0, places=6)
        self.assertAlmostEqual(float(out.loc[2, "fc_net_profit_min"]), 200.0, places=6)
        self.assertAlmostEqual(float(out.loc[2, "exp_revenue"]), 1100.0, places=6)
        self.assertAlmostEqual(float(out.loc[2, "exp_yoy_sales"]), 18.0, places=6)

    def test_augment_tushare_symbol_frame_loads_industry_context_sidecar(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-03-30", "2026-03-31", "2026-04-01"]),
                "symbol": ["000001", "000001", "000001"],
                "open": [10.0, 11.0, 12.0],
                "high": [10.5, 11.5, 12.5],
                "low": [9.8, 10.8, 11.8],
                "close": [10.0, 11.0, 12.0],
                "volume": [100.0, 120.0, 140.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir)
            symbol_cache = pd.DataFrame({"local_symbol": ["000001"], "industry": ["银行"]})
            industry_context = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-03-30", "2026-03-31"]),
                    "industry": ["银行", "银行"],
                    "ind_member_count": [24.0, 24.0],
                    "ind_daily_ret": [0.01, 0.02],
                    "ind_excess_daily_ret": [0.005, 0.015],
                    "ind_ret_5": [0.03, 0.04],
                    "ind_std_5": [0.02, 0.02],
                    "ind_excess_ret_5": [0.01, 0.015],
                    "ind_pos_rate_5": [0.45, 0.50],
                    "ind_dispersion_5": [0.01, 0.011],
                    "ind_ret_20": [0.10, 0.11],
                    "ind_std_20": [0.03, 0.03],
                    "ind_excess_ret_20": [0.02, 0.025],
                    "ind_pos_rate_20": [0.55, 0.60],
                    "ind_dispersion_20": [0.02, 0.021],
                    "ind_ret_60": [0.20, 0.21],
                    "ind_std_60": [0.04, 0.04],
                    "ind_excess_ret_60": [0.04, 0.045],
                    "ind_pos_rate_60": [0.65, 0.70],
                    "ind_dispersion_60": [0.03, 0.031],
                }
            )
            symbol_cache_path = meta_dir / "symbol_cache.parquet"
            industry_context_path = meta_dir / "industry_context.parquet"
            symbol_cache.to_parquet(symbol_cache_path, index=False)
            industry_context.to_parquet(industry_context_path, index=False)

            from src import gen_feature as gen_feature_module

            original_symbol_cache_path = gen_feature_module.TUSHARE_SYMBOL_CACHE_PATH
            original_industry_context_path = gen_feature_module.TUSHARE_INDUSTRY_CONTEXT_PATH
            gen_feature_module.TUSHARE_SYMBOL_CACHE_PATH = symbol_cache_path
            gen_feature_module.TUSHARE_INDUSTRY_CONTEXT_PATH = industry_context_path
            gen_feature_module._clear_tushare_context_caches()
            try:
                out = _augment_tushare_symbol_frame(df, symbol="000001")
            finally:
                gen_feature_module.TUSHARE_SYMBOL_CACHE_PATH = original_symbol_cache_path
                gen_feature_module.TUSHARE_INDUSTRY_CONTEXT_PATH = original_industry_context_path
                gen_feature_module._clear_tushare_context_caches()

        self.assertAlmostEqual(float(out.loc[0, "ind_member_count"]), 24.0, places=6)
        self.assertAlmostEqual(float(out.loc[1, "ind_daily_ret"]), 0.02, places=6)
        self.assertAlmostEqual(float(out.loc[2, "ind_ret_20"]), 0.11, places=6)
        self.assertAlmostEqual(float(out.loc[2, "ind_excess_ret_60"]), 0.045, places=6)
        self.assertAlmostEqual(float(out.loc[2, "ind_pos_rate_20"]), 0.60, places=6)
        self.assertAlmostEqual(float(out.loc[2, "ind_dispersion_60"]), 0.031, places=6)

    def test_compute_all_factor_features_uses_side_loaded_forecast_and_express_columns(self):
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
                "circ_mv": [1000.0, 1100.0],
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
                "amplitude": [7.0, 8.0],
                "pct_chg": [5.0, 10.0],
                "limit_pre_close": [10.0, 11.0],
                "up_limit": [11.0, 12.1],
                "down_limit": [9.0, 9.9],
                "fc_p_change_min": [10.0, 20.0],
                "fc_p_change_max": [30.0, 40.0],
                "fc_net_profit_min": [100.0, 200.0],
                "fc_net_profit_max": [300.0, 400.0],
                "fc_last_parent_net": [50.0, 60.0],
                "exp_revenue": [1000.0, 1100.0],
                "exp_operate_profit": [100.0, 120.0],
                "exp_n_income": [80.0, 90.0],
                "exp_yoy_sales": [15.0, 18.0],
                "ind_member_count": [24.0, 24.0],
                "ind_daily_ret": [0.01, 0.02],
                "ind_excess_daily_ret": [0.005, 0.015],
                "ind_ret_5": [0.03, 0.04],
                "ind_ret_20": [0.10, 0.11],
                "ind_ret_60": [0.20, 0.21],
                "ind_std_5": [0.02, 0.02],
                "ind_std_20": [0.03, 0.03],
                "ind_std_60": [0.04, 0.04],
                "ind_excess_ret_5": [0.01, 0.015],
                "ind_excess_ret_20": [0.02, 0.025],
                "ind_excess_ret_60": [0.04, 0.045],
            }
        )

        tushare_feat = compute_all_factor_features(df, data_source="tushare")

        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}latest_fc_p_change_max", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}latest_exp_revenue", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_member_count", tushare_feat.columns)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_rel_ret_20", tushare_feat.columns)
        self.assertAlmostEqual(float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}latest_fc_p_change_max"]), 40.0, places=6)
        self.assertAlmostEqual(float(tushare_feat.iloc[1][f"{TUSHARE_FACTOR_PREFIX}latest_exp_revenue"]), 1100.0, places=6)


if __name__ == "__main__":
    unittest.main()
