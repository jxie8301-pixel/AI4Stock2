import unittest

from src.feature_profiles import resolve_feature_profile
from src.gen_feature import get_alpha158_feature_config


class FeatureProfilesTest(unittest.TestCase):
    def test_full_profile_preserves_legacy_cache_dir(self):
        cfg = {
            "alpha_version": 158,
            "features": {
                "profile": "alpha158_full",
            },
        }

        profile = resolve_feature_profile(cfg)

        self.assertEqual(profile["alpha"], "158")
        self.assertEqual(profile["cache_dir"], "data/cache/alpha158_panel")
        self.assertIsNone(profile["alpha158_config"])

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

        self.assertEqual(profile["cache_dir"], "data/cache/alpha158_compact_v1_panel")
        self.assertLess(compact_count, full_count)
        self.assertEqual(compact_count, 82)


if __name__ == "__main__":
    unittest.main()
