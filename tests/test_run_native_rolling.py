import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import run_native_rolling


class RunNativeRollingTest(unittest.TestCase):
    def test_prediction_bundle_round_trip(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-03"), "B"),
            ],
            names=["datetime", "instrument"],
        )
        bundle = run_native_rolling.PredictionBundle(
            final_predictions=pd.Series([0.1, 0.2], index=index, name="prediction"),
            label_series=pd.Series([0.01, 0.02], index=index, name="label"),
            backtest_label_series=pd.Series([0.001, 0.002], index=index, name="label"),
            selected_feature_names=["f1", "f2"],
            metadata={"signal_horizon": 20, "test_start": "2024-01-02", "test_end": "2024-01-03"},
            feature_importance_frames=[],
            training_summary_records=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / run_native_rolling.PREDICTION_ARTIFACT_DIRNAME
            run_native_rolling._write_prediction_bundle(bundle, artifact_dir)
            loaded = run_native_rolling.load_prediction_bundle(artifact_dir)

        pd.testing.assert_series_equal(loaded.final_predictions, bundle.final_predictions, check_names=False)
        pd.testing.assert_series_equal(loaded.label_series, bundle.label_series, check_names=False)
        pd.testing.assert_series_equal(loaded.backtest_label_series, bundle.backtest_label_series, check_names=False)
        self.assertEqual(loaded.selected_feature_names, ["f1", "f2"])
        self.assertEqual(int(loaded.metadata["signal_horizon"]), 20)

    def test_resolve_prediction_artifact_dir_accepts_parent_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            artifact_dir = run_dir / run_native_rolling.PREDICTION_ARTIFACT_DIRNAME
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with open(artifact_dir / run_native_rolling.PREDICTION_METADATA_FILENAME, "w", encoding="utf-8") as f:
                json.dump({"selected_features": []}, f)

            resolved = run_native_rolling._resolve_prediction_artifact_dir(run_dir)

        self.assertEqual(resolved, artifact_dir)


if __name__ == "__main__":
    unittest.main()
