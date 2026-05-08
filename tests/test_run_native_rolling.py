import os
import unittest

from run_native_rolling import build_delegated_command


class RunNativeRollingTest(unittest.TestCase):
    def test_rust_wrapper_delegates_rolling_runtime_to_rust(self):
        command = build_delegated_command(
            [
                "--config",
                "configs/config.yaml",
                "--experiment-profile",
                "core_v4_lgbm_default_10x20x10",
                "--model-profile",
                "lgbm_fast",
                "--feature-profile",
                "core_v4_techlite",
                "--run-tag",
                "demo",
                "--save-models",
                "--set",
                "lgbm.num_boost_round=2",
            ]
        )

        self.assertIn("ai4stock-train", " ".join(command))
        self.assertIn("rolling-lgbm", command)
        self.assertNotIn("run_native_rolling.py", command)
        self.assertIn("--experiment-profile", command)
        self.assertIn("core_v4_lgbm_default_10x20x10", command)
        self.assertIn("--save-models", command)
        self.assertIn("lgbm.num_boost_round=2", command)

    def test_rust_wrapper_honors_train_binary_override(self):
        original = os.environ.get("AI4STOCK_TRAIN_BIN")
        os.environ["AI4STOCK_TRAIN_BIN"] = "/tmp/ai4stock-train --flag"
        try:
            command = build_delegated_command(["--dry-run"])
        finally:
            if original is None:
                os.environ.pop("AI4STOCK_TRAIN_BIN", None)
            else:
                os.environ["AI4STOCK_TRAIN_BIN"] = original

        self.assertEqual(command[:3], ["/tmp/ai4stock-train", "--flag", "rolling-lgbm"])
        self.assertIn("--dry-run", command)

    def test_rust_wrapper_forwards_load_predictions_backtest_args(self):
        command = build_delegated_command(
            [
                "--config",
                "snapshot.yaml",
                "--config-is-snapshot",
                "--load-predictions-dir",
                "run/prediction_artifacts",
                "--skip-reference-baselines",
                "--backtest-artifact-level",
                "reports",
                "--baseline-jobs",
                "2",
            ]
        )

        self.assertIn("rolling-lgbm", command)
        self.assertIn("--load-predictions-dir", command)
        self.assertIn("--skip-reference-baselines", command)
        self.assertIn("--backtest-artifact-level", command)
        self.assertIn("--baseline-jobs", command)
        self.assertIn("2", command)


if __name__ == "__main__":
    unittest.main()
