import unittest
import tempfile
from pathlib import Path

import yaml

from src.feature_profiles import resolve_feature_profile
from src.gen_feature import (
    TECHNICAL_FACTOR_PREFIX,
    TEMPORAL_FACTOR_PREFIX,
    TUSHARE_FACTOR_PREFIX,
    get_all_factor_feature_names,
    get_alpha158_feature_config,
    get_known_exact_duplicate_feature_groups,
    get_lgbm_purified_feature_names,
    get_technical_factor_feature_names,
    get_temporal_factor_feature_names,
)


class FeatureProfilesTest(unittest.TestCase):
    def test_default_profile_is_core_v4_techlite(self):
        profile = resolve_feature_profile({})

        self.assertEqual(profile["alpha"], "all_factors")
        self.assertEqual(profile["data_source"], "akshare")
        self.assertEqual(profile["generation_space"], "full_factor_space")
        self.assertEqual(profile["factor_store_dir"], "data/factor_store/full_factor_space")
        self.assertEqual(len(profile["selected_columns"]), 46)
        self.assertEqual(len(profile["load_columns"]), 37)
        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/core_v4_techlite.yaml"))
        self.assertEqual(profile["selected_columns"][0], "KMID")
        self.assertEqual(profile["selected_columns"][-1], f"{TECHNICAL_FACTOR_PREFIX}mfi_14")
        self.assertEqual(len(get_all_factor_feature_names()), 259)
        self.assertIn(f"{TEMPORAL_FACTOR_PREFIX}ret_120", get_all_factor_feature_names())
        self.assertIn(f"{TECHNICAL_FACTOR_PREFIX}macd_hist_12_26_9", get_all_factor_feature_names())
        self.assertIn("CORR20__rep2", profile["selected_columns"])
        self.assertIn("LGBM_ret_20__rep2", profile["selected_columns"])
        self.assertNotIn(f"{TEMPORAL_FACTOR_PREFIX}ret_20", profile["load_columns"])
        self.assertNotIn(f"{TEMPORAL_FACTOR_PREFIX}rsv_20", profile["load_columns"])

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

    def test_tushare_source_switches_default_factor_store_dir(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite"},
            }
        )

        self.assertEqual(profile["data_source"], "tushare")
        self.assertEqual(profile["factor_store_dir"], "data/factor_store/tushare_full_factor_space")

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

    def test_core_v3_techlite_profiles_are_registered(self):
        techlite = resolve_feature_profile({"features": {"profile": "core_v3_techlite"}})
        techlite_pruned = resolve_feature_profile({"features": {"profile": "core_v3_techlite_pruned"}})

        self.assertTrue(str(techlite["profile_path"]).endswith("configs/features/core_v3_techlite.yaml"))
        self.assertTrue(str(techlite_pruned["profile_path"]).endswith("configs/features/core_v3_techlite_pruned.yaml"))
        self.assertEqual(len(techlite["selected_columns"]), 47)
        self.assertEqual(len(techlite_pruned["selected_columns"]), 41)
        self.assertIn(f"{TECHNICAL_FACTOR_PREFIX}boll_pos_20_2", techlite["selected_columns"])
        self.assertIn(f"{TECHNICAL_FACTOR_PREFIX}macd_hist_12_26_9", techlite_pruned["selected_columns"])

    def test_core_v4_ablation_profiles_are_registered(self):
        no_ret120 = resolve_feature_profile({"features": {"profile": "core_v4_techlite_no_ret120"}})

        self.assertTrue(str(no_ret120["profile_path"]).endswith("configs/feature_profiles.yaml::core_v4_techlite_no_ret120"))
        self.assertNotIn(f"{TEMPORAL_FACTOR_PREFIX}ret_120", no_ret120["load_columns"])
        self.assertEqual(len(no_ret120["selected_columns"]), 45)
        self.assertEqual(len(no_ret120["load_columns"]), 36)

    def test_core_v4_tushare_plus_profile_adds_ts_columns(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus"},
            }
        )

        self.assertTrue(
            str(profile["profile_path"]).endswith("configs/feature_profiles.yaml::core_v4_techlite_tushare_plus")
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}gap_up_limit", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}free_turnover_ratio", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}dividend_yield_ttm", profile["selected_columns"])

    def test_core_v4_tushare_plus_industry_profile_adds_industry_columns(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_industry_v1"},
            }
        )

        self.assertTrue(
            str(profile["profile_path"]).endswith("configs/feature_profiles.yaml::core_v4_techlite_tushare_plus_industry_v1")
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_member_count", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_ret_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_excess_ret_60", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_rel_ret_5", profile["selected_columns"])

    def test_core_v4_tushare_plus_excess_flow_value_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_excess_flow_value_v1"},
            }
        )

        self.assertTrue(
            str(profile["profile_path"]).endswith(
                "configs/feature_profiles.yaml::core_v4_techlite_tushare_plus_excess_flow_value_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}turnover_accel_5_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}free_turnover_spread_zscore_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}downside_amihud_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}ep_ttm_change_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}dividend_yield_ttm_surprise_20", profile["selected_columns"])

    def test_core_v4_tushare_plus_industry_excess_flow_value_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_industry_excess_flow_value_v1"},
            }
        )

        self.assertTrue(
            str(profile["profile_path"]).endswith(
                "configs/feature_profiles.yaml::core_v4_techlite_tushare_plus_industry_excess_flow_value_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_dispersion_60", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_std_ratio_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_std_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}turnover_accel_5_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_20", profile["cross_sectional_rank_exclude_columns"])

    def test_core_v4_tushare_plus_quality_event_flow_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_quality_event_flow_v1"},
            }
        )

        self.assertTrue(
            str(profile["profile_path"]).endswith(
                "configs/feature_profiles.yaml::core_v4_techlite_tushare_plus_quality_event_flow_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}fi_ocf_to_eps", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}fc_surprise_fresh", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}exp_growth_fresh", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}up_down_turnover_ratio_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_breadth_accel_20_60", profile["selected_columns"])

    def test_core_v4_tushare_plus_industry_excess_flow_value_slim_profiles_are_registered(self):
        slim_a = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_a_v1"},
            }
        )
        slim_b = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_v1"},
            }
        )

        self.assertTrue(
            str(slim_a["profile_path"]).endswith(
                "configs/feature_profiles.yaml::core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_a_v1"
            )
        )
        self.assertTrue(
            str(slim_b["profile_path"]).endswith(
                "configs/feature_profiles.yaml::core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}free_turnover_mean_20", slim_a["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}downside_amihud_20", slim_a["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}bp_change_20", slim_a["selected_columns"])
        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}industry_dispersion_60", slim_a["selected_columns"])

        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_dispersion_60", slim_b["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_std_ratio_20", slim_b["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_std_20", slim_b["cross_sectional_rank_exclude_columns"])

    def test_core_v4_tushare_plus_industry_excess_flow_value_slim_b_family_profiles_are_registered(self):
        slim_b_drop_rel = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {
                    "profile": (
                        "core_v4_techlite_tushare_plus_industry_excess_flow_value_"
                        "slim_b_drop_rel_midlong_v1"
                    )
                },
            }
        )
        slim_b_breadth = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_plus_breadth_v1"},
            }
        )
        slim_b_support = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {
                    "profile": "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_plus_flow_value_support_v1"
                },
            }
        )

        self.assertTrue(
            str(slim_b_drop_rel["profile_path"]).endswith(
                "configs/feature_profiles.yaml::"
                "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_drop_rel_midlong_v1"
            )
        )
        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}industry_rel_ret_20", slim_b_drop_rel["selected_columns"])
        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}industry_rel_ret_60", slim_b_drop_rel["selected_columns"])
        self.assertNotIn(
            f"{TUSHARE_FACTOR_PREFIX}industry_rel_ret_20",
            slim_b_drop_rel["cross_sectional_rank_exclude_columns"],
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_ret_60", slim_b_drop_rel["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_excess_ret_60", slim_b_drop_rel["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_dispersion_60", slim_b_drop_rel["selected_columns"])

        self.assertTrue(
            str(slim_b_breadth["profile_path"]).endswith(
                "configs/feature_profiles.yaml::"
                "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_plus_breadth_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_20", slim_b_breadth["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_60", slim_b_breadth["selected_columns"])
        self.assertIn(
            f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_20",
            slim_b_breadth["cross_sectional_rank_exclude_columns"],
        )

        self.assertTrue(
            str(slim_b_support["profile_path"]).endswith(
                "configs/feature_profiles.yaml::"
                "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_plus_flow_value_support_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}free_turnover_spread", slim_b_support["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}turnover_accel_5_20", slim_b_support["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}ep_ttm_change_20", slim_b_support["selected_columns"])
        self.assertNotIn(f"{TUSHARE_FACTOR_PREFIX}industry_pos_rate_20", slim_b_support["selected_columns"])

    def test_core_v4_slim_b_plus_relative_alpha_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {
                    "profile": (
                        "core_v4_techlite_tushare_plus_industry_excess_flow_value_"
                        "slim_b_plus_relative_alpha_v1"
                    )
                },
            }
        )

        self.assertTrue(
            str(profile["profile_path"]).endswith(
                "configs/feature_profiles.yaml::"
                "core_v4_techlite_tushare_plus_industry_excess_flow_value_slim_b_plus_relative_alpha_v1"
            )
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_dispersion_60", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_std_ratio_60", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}dividend_yield_minus_industry", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}bp_minus_industry_bp", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_amplitude_ratio_20", profile["selected_columns"])
        self.assertIn(
            f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_downside_amihud_ratio_60",
            profile["selected_columns"],
        )
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}industry_std_60", profile["cross_sectional_rank_exclude_columns"])

    def test_core_v5_diag_prefilter_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v5_diag_prefilter_v1"},
            }
        )

        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/core_v5_diag_prefilter_v1.yaml"))
        self.assertEqual(profile["data_source"], "tushare")
        self.assertEqual(len(profile["selected_columns"]), 8)
        self.assertEqual(profile["selected_columns"][0], "CORR20")
        self.assertIn("LGBM_bp", profile["selected_columns"])

    def test_core_v5_diag_prefilter_v2_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v5_diag_prefilter_v2"},
            }
        )

        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/core_v5_diag_prefilter_v2.yaml"))
        self.assertEqual(profile["data_source"], "tushare")
        self.assertEqual(len(profile["selected_columns"]), 10)
        self.assertEqual(profile["selected_columns"][0], "CORR20")
        self.assertIn("LGBM_amihud_20", profile["selected_columns"])
        self.assertNotIn("TEMP_corr_cv_20", profile["selected_columns"])

    def test_core_v6_relative_alpha_profile_is_registered(self):
        profile = resolve_feature_profile(
            {
                "data": {"source": "tushare"},
                "features": {"profile": "core_v6_relative_alpha_v1"},
            }
        )

        self.assertTrue(str(profile["profile_path"]).endswith("configs/features/core_v6_relative_alpha_v1.yaml"))
        self.assertEqual(profile["data_source"], "tushare")
        self.assertEqual(len(profile["selected_columns"]), 12)
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}dividend_yield_minus_industry", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}bp_minus_industry_bp", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_amplitude_ratio_20", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}stock_vs_industry_std_ratio_60", profile["selected_columns"])
        self.assertIn(f"{TUSHARE_FACTOR_PREFIX}fi_ocfps_minus_eps_minus_industry", profile["selected_columns"])

    def test_known_exact_duplicate_groups_cover_core_overlaps(self):
        duplicate_groups = get_known_exact_duplicate_feature_groups()

        self.assertIn(("CORR20", "TEMP_corr_cv_20"), duplicate_groups)
        self.assertIn(("RSV20", "TEMP_rsv_20"), duplicate_groups)
        self.assertIn(("LGBM_ret_20", "TEMP_ret_20"), duplicate_groups)
        self.assertIn(("LGBM_dist_ma60", "TEMP_ma_gap_60"), duplicate_groups)

    def test_inline_feature_profile_extends_with_drop_and_add_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "feature_profiles.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "default_profile": "base",
                        "profiles": {
                            "base": {
                                "alpha": "all_factors",
                                "generation_space": "full_factor_space",
                                "factor_store_name": "full_factor_space",
                                "selected_columns": ["A", "B", "C"],
                            },
                            "derived": {
                                "extends": "base",
                                "drop_columns": ["B"],
                                "add_columns": ["D", "A"],
                            },
                        },
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            profile = resolve_feature_profile(
                {"features": {"profile": "derived"}},
                profile_config_path=str(config_path),
            )

            self.assertEqual(profile["selected_columns"], ["A", "C", "D"])
            self.assertTrue(str(profile["profile_path"]).endswith("feature_profiles.yaml::derived"))

    def test_removed_duplicate_v3_no_rank20_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_feature_profile({"features": {"profile": "core_v3_techlite_no_price_rank20"}})

    def test_removed_alpha360_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_feature_profile({"features": {"profile": "alpha360_full"}})


if __name__ == "__main__":
    unittest.main()
