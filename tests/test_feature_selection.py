import unittest

import numpy as np
import pandas as pd

from src.gen_feature import get_all_factor_feature_names
from src.feature_selection import (
    apply_cross_sectional_rank,
    apply_feature_transforms,
    compute_finite_feature_mask,
    materialize_selected_feature_frame,
    resolve_selected_features,
)


class FeatureSelectionTest(unittest.TestCase):
    def test_resolve_selected_features_returns_all_when_omitted(self):
        meta = {"feature_names": ["A", "B", "C"]}
        cfg = {"features": {"profile": "all_factors_full"}}

        idx, names = resolve_selected_features(meta, cfg)

        self.assertEqual(idx, [0, 1, 2])
        self.assertEqual(names, ["A", "B", "C"])

    def test_resolve_selected_features_preserves_order_for_inline_override(self):
        meta = {"feature_names": ["A", "B", "C"]}
        cfg = {"features": {"selected_columns": ["C", "A", "C"]}}

        idx, names = resolve_selected_features(meta, cfg)

        self.assertEqual(idx, [2, 0, 2])
        self.assertEqual(names, ["C", "A", "C"])

    def test_resolve_selected_features_uses_profile_subset_when_columns_omitted(self):
        lgbm_cols = [f"LGBM_{name}" for name in [
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
            "ep_ttm_clean",
            "ep_ttm_invalid",
            "bp",
            "bp_clean",
            "bp_invalid",
        ]]
        meta = {
            "feature_names": [
                "KMID",
                "MA5",
                "TEMP_ret_20",
                *lgbm_cols,
                "LGBM_turnover_20",
                "LGBM_dist_high_20",
                "LGBM_dist_low_20",
            ]
        }
        cfg = {"features": {"profile": "lgbm_purified_v1"}}

        idx, names = resolve_selected_features(meta, cfg)

        self.assertEqual(
            idx,
            list(range(3, 3 + len(lgbm_cols))) + [len(meta["feature_names"]) - 3, len(meta["feature_names"]) - 2, len(meta["feature_names"]) - 1],
        )
        self.assertEqual(
            names,
            lgbm_cols + ["LGBM_turnover_20", "LGBM_dist_high_20", "LGBM_dist_low_20"],
        )

    def test_compute_finite_feature_mask_uses_selected_columns_only(self):
        X = np.array(
            [
                [1.0, np.inf, 2.0],
                [1.0, 3.0, np.inf],
                [1.0, 3.0, 2.0],
            ],
            dtype=np.float32,
        )

        mask = compute_finite_feature_mask(X, [0, 1], 3)

        self.assertEqual(mask.tolist(), [False, True, True])

    def test_apply_cross_sectional_rank_ranks_within_each_date(self):
        frame = pd.DataFrame(
            {
                "f1": [10.0, 20.0, 30.0, 40.0],
                "f2": [1.0, 3.0, 4.0, 2.0],
            }
        )
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])

        ranked = apply_cross_sectional_rank(frame, dates)

        self.assertEqual(ranked["f1"].round(3).tolist(), [0.5, 1.0, 0.5, 1.0])
        self.assertEqual(ranked["f2"].round(3).tolist(), [0.5, 1.0, 1.0, 0.5])

    def test_apply_cross_sectional_rank_preserves_excluded_columns(self):
        frame = pd.DataFrame(
            {
                "f1": [10.0, 20.0, 30.0, 40.0],
                "raw_market": [100.0, 90.0, 80.0, 70.0],
            }
        )
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])

        ranked = apply_cross_sectional_rank(frame, dates, exclude_columns={"raw_market"})

        self.assertEqual(ranked["f1"].round(3).tolist(), [0.5, 1.0, 0.5, 1.0])
        self.assertEqual(ranked["raw_market"].tolist(), frame["raw_market"].tolist())

    def test_apply_feature_transforms_respects_config_flag(self):
        frame = pd.DataFrame({"f1": [10.0, 20.0]})
        dates = pd.to_datetime(["2024-01-02", "2024-01-02"])

        transformed = apply_feature_transforms(frame, dates, {"features": {"transforms": {"cross_sectional_rank": True}}})
        untouched = apply_feature_transforms(frame, dates, {"features": {}})

        self.assertEqual(transformed["f1"].tolist(), [0.5, 1.0])
        self.assertEqual(untouched["f1"].tolist(), [10.0, 20.0])

    def test_materialize_selected_feature_frame_creates_alias_columns(self):
        frame = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})

        expanded = materialize_selected_feature_frame(
            frame,
            selected_columns=["A", "A__rep2", "B"],
            source_columns=["A", "A", "B"],
        )

        self.assertEqual(expanded["A__rep2"].tolist(), [1.0, 2.0])
        self.assertEqual(expanded["B"].tolist(), [3.0, 4.0])

    def test_all_factor_feature_names_are_unique(self):
        feature_names = get_all_factor_feature_names()

        self.assertEqual(len(feature_names), len(set(feature_names)))
        self.assertIn("KMID", feature_names)
        self.assertIn("LGBM_ret_20", feature_names)
        self.assertIn("TEMP_ret_120", feature_names)
        self.assertNotIn("TEMP_ret_20", feature_names)
        self.assertNotIn("TEMP_corr_cv_20", feature_names)


if __name__ == "__main__":
    unittest.main()
