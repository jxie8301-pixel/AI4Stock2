import unittest

from src.feature_profiles import resolve_feature_profile
from src.gen_feature import (
    TECHNICAL_FACTOR_PREFIX,
    TEMPORAL_FACTOR_PREFIX,
    get_all_factor_feature_names,
    get_alpha158_feature_config,
    get_lgbm_purified_feature_names,
    get_technical_factor_feature_names,
    get_temporal_factor_feature_names,
)


class FeatureProfilesTest(unittest.TestCase):
    def test_default_profile_is_core_v1(self):
        profile = resolve_feature_profile({})

        self.assertEqual(profile["alpha"], "all_factors")
        self.assertEqual(profile["generation_space"], "full_factor_space")
        self.assertEqual(profile["factor_store_dir"], "data/factor_store/full_factor_space")
        self.assertEqual(len(profile["selected_columns"]), 41)
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/core_v1.yaml"))
        self.assertEqual(profile["selected_columns"][0], "KMID")
        self.assertEqual(profile["selected_columns"][-1], "TEMP_corr_cv_20")
        self.assertEqual(len(get_all_factor_feature_names()), 279)
        self.assertIn(f"{TEMPORAL_FACTOR_PREFIX}ret_120", get_all_factor_feature_names())
        self.assertIn(f"{TECHNICAL_FACTOR_PREFIX}macd_hist_12_26_9", get_all_factor_feature_names())

    def test_full_profile_preserves_legacy_cache_dir(self):
        cfg = {
            "alpha_version": 158,
            "features": {
                "profile": "alpha158_full",
            },
        }

        profile = resolve_feature_profile(cfg)

        self.assertEqual(profile["alpha"], "158")
        self.assertEqual(profile["generation_space"], "full_factor_space")
        self.assertEqual(profile["factor_store_dir"], "data/factor_store/full_factor_space")
        self.assertIsNone(profile["alpha158_config"])
        self.assertEqual(len(profile["selected_columns"]), 158)
        self.assertEqual(profile["selected_columns"][0], "KMID")
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/alpha158_full.yaml"))

    def test_compact_profile_reduces_feature_count(self):
        profile = resolve_feature_profile(
            {
                "features": {
                    "profile": "alpha158_compact_v1",
                }
            }
        )

        compact_count = len(get_alpha158_feature_config(profile["alpha158_config"])[1])
        full_count = len(get_alpha158_feature_config()[1])

        self.assertEqual(profile["factor_store_dir"], "data/factor_store/full_factor_space")
        self.assertEqual(profile["selected_columns"], get_alpha158_feature_config(profile["alpha158_config"])[1])
        self.assertLess(compact_count, full_count)
        self.assertEqual(compact_count, 82)
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/alpha158_compact_v1.yaml"))

    def test_lgbm_purified_profile_has_expected_feature_family(self):
        profile = resolve_feature_profile(
            {
                "features": {
                    "profile": "lgbm_purified_v1",
                }
            }
        )

        feature_names = get_lgbm_purified_feature_names(profile["raw"].get("lgbm_purified"))

        self.assertEqual(profile["alpha"], "lgbm_purified")
        self.assertEqual(profile["generation_space"], "full_factor_space")
        self.assertEqual(profile["factor_store_dir"], "data/factor_store/full_factor_space")
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/lgbm_purified_v1.yaml"))
        self.assertEqual(len(feature_names), 18)
        self.assertEqual(profile["selected_columns"], [f"LGBM_{name}" for name in feature_names])
        self.assertEqual(
            feature_names,
            [
                "ret_20",
                "ret_60",
                "dist_ma20",
                "dist_ma60",
                "dist_ma120",
                "std_60",
                "atr_14",
                "amihud_20",
                "vol_ratio_20",
                "corr_cv_20",
                "vwap_ratio",
                "log_mcap",
                "ep_ttm",
                "is_loss",
                "bp",
                "turnover_20",
                "dist_high_20",
                "dist_low_20",
            ],
        )

    def test_temporal_factor_family_has_systematic_windows(self):
        names = get_temporal_factor_feature_names()

        self.assertEqual(len(names), 77)
        self.assertIn("ret_1", names)
        self.assertIn("ma_gap_120", names)
        self.assertIn("corr_cv_60", names)

    def test_technical_factor_family_has_classic_indicators(self):
        names = get_technical_factor_feature_names()

        self.assertEqual(len(names), 26)
        self.assertIn("macd_line_12_26_9", names)
        self.assertIn("rsi_14", names)
        self.assertIn("boll_width_20_2", names)
        self.assertIn("adx_14", names)
        self.assertIn("obv_flow_60", names)

    def test_technical_core_profile_has_expected_feature_family(self):
        profile = resolve_feature_profile(
            {
                "features": {
                    "profile": "technical_core_v1",
                }
            }
        )

        self.assertEqual(profile["alpha"], "all_factors")
        self.assertEqual(profile["generation_space"], "full_factor_space")
        self.assertEqual(profile["factor_store_dir"], "data/factor_store/full_factor_space")
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/technical_core_v1.yaml"))
        self.assertEqual(profile["selected_columns"], [f"{TECHNICAL_FACTOR_PREFIX}{name}" for name in get_technical_factor_feature_names()])

    def test_removed_alpha360_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_feature_profile({"features": {"profile": "alpha360_full"}})


if __name__ == "__main__":
    unittest.main()
