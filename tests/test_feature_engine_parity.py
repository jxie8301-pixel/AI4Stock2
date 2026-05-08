import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.feature_engine_parity import (
    build_python_feature_snapshot,
    compare_feature_snapshots,
    run_feature_engine_parity,
)
from src.gen_feature import get_full_factor_space_feature_names


def _minimal_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "symbol": ["000001"] * 4,
            "open": [10.0, 10.5, 11.0, 10.8],
            "high": [10.3, 10.8, 11.2, 11.1],
            "low": [9.8, 10.2, 10.7, 10.6],
            "close": [10.1, 10.6, 11.1, 10.9],
            "volume": [100.0, 120.0, 140.0, 130.0],
            "amount": [1000.0, 1200.0, 1400.0, 1300.0],
        }
    )


class FeatureEngineParityTest(unittest.TestCase):
    def test_python_reference_is_self_consistent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "000001.parquet"
            _minimal_source_frame().to_parquet(source_path, index=False)
            feature = get_full_factor_space_feature_names()[0]

            result = run_feature_engine_parity(
                input_parquet=source_path,
                symbol="000001",
                label_horizons=[1],
                feature_subset=[feature],
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["reference_engine"], "python")
        self.assertEqual(result["candidate_engine"], "python")
        self.assertEqual(result["columns_checked"], 1)

    def test_compare_feature_snapshots_reports_value_mismatch(self):
        feature = get_full_factor_space_feature_names()[0]
        reference = build_python_feature_snapshot(
            _minimal_source_frame(),
            symbol="000001",
            label_horizons=[1],
            feature_subset=[feature],
        )
        candidate = reference.copy()
        finite_index = candidate[feature].first_valid_index()
        self.assertIsNotNone(finite_index)
        candidate.loc[finite_index, feature] = float(candidate.loc[finite_index, feature]) + 0.01

        result = compare_feature_snapshots(reference, candidate, compare_columns=[feature])

        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "value_mismatch")
        self.assertEqual(result["mismatched_columns"][0]["column"], feature)


if __name__ == "__main__":
    unittest.main()
