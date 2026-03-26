import unittest

import numpy as np

from src.feature_selection import compute_finite_feature_mask, resolve_selected_features


class FeatureSelectionTest(unittest.TestCase):
    def test_resolve_selected_features_returns_all_when_omitted(self):
        meta = {"feature_names": ["A", "B", "C"]}
        cfg = {"features": {}}

        idx, names = resolve_selected_features(meta, cfg)

        self.assertEqual(idx, [0, 1, 2])
        self.assertEqual(names, ["A", "B", "C"])

    def test_resolve_selected_features_preserves_order_and_dedupes(self):
        meta = {"feature_names": ["A", "B", "C"]}
        cfg = {"features": {"selected_columns": ["C", "A", "C"]}}

        idx, names = resolve_selected_features(meta, cfg)

        self.assertEqual(idx, [2, 0])
        self.assertEqual(names, ["C", "A"])

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


if __name__ == "__main__":
    unittest.main()
