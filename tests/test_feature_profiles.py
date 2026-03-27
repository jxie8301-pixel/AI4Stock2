import unittest

from src.feature_profiles import resolve_feature_profile
from src.gen_feature import get_all_factor_feature_names, get_alpha158_feature_config, get_lgbm_purified_feature_names


class FeatureProfilesTest(unittest.TestCase):
    def test_default_profile_is_all_factors_full(self):
        profile = resolve_feature_profile({})

        self.assertEqual(profile["alpha"], "all_factors")
        self.assertEqual(profile["generation_alpha"], "all_factors")
        self.assertEqual(profile["cache_dir"], "data/cache/all_factors_panel")
        self.assertIsNone(profile["selected_columns"])
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/all_factors_full.yaml"))
        self.assertEqual(len(get_all_factor_feature_names()), 536)

    def test_full_profile_preserves_legacy_cache_dir(self):
        cfg = {
            "alpha_version": 158,
            "features": {
                "profile": "alpha158_full",
            },
        }

        profile = resolve_feature_profile(cfg)

        self.assertEqual(profile["alpha"], "158")
        self.assertEqual(profile["generation_alpha"], "all_factors")
        self.assertEqual(profile["cache_dir"], "data/cache/all_factors_panel")
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

        self.assertEqual(profile["cache_dir"], "data/cache/all_factors_panel")
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
        self.assertEqual(profile["generation_alpha"], "all_factors")
        self.assertEqual(profile["cache_dir"], "data/cache/all_factors_panel")
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

    def test_alpha360_profile_maps_to_prefixed_all_factor_columns(self):
        profile = resolve_feature_profile({"features": {"profile": "alpha360_full"}})

        self.assertEqual(profile["alpha"], "360")
        self.assertEqual(profile["generation_alpha"], "all_factors")
        self.assertEqual(profile["cache_dir"], "data/cache/all_factors_panel")
        self.assertEqual(len(profile["selected_columns"]), 360)
        self.assertEqual(profile["selected_columns"][0], "A360_CLOSE59")
        self.assertEqual(profile["selected_columns"][-1], "A360_VOLUME0")


if __name__ == "__main__":
    unittest.main()
