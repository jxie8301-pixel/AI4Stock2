import tempfile
import unittest
from pathlib import Path

from src.config_loader import load_config
from src.config_validation import validate_training_config


class ConfigValidationTest(unittest.TestCase):
    def test_validate_training_config_accepts_valid_profile_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            factor_store_dir = tmp_path / "factor_store" / "full_factor_space"
            factor_store_dir.mkdir(parents=True, exist_ok=True)
            universe_dir = tmp_path / "universes"
            universe_dir.mkdir(parents=True, exist_ok=True)
            (universe_dir / "csi300.txt").write_text("000001.SZ\n", encoding="utf-8")

            cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
            cfg["features"]["factor_store_dir"] = str(factor_store_dir)
            cfg["native"]["universe_dir"] = str(universe_dir)

            validated = validate_training_config(cfg, check_paths=True)

            self.assertEqual(validated["experiment"]["profile"], "core_v4_lgbm_default_10x20x10")

    def test_validate_training_config_rejects_invalid_strategy_shape(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["topk"] = 5
        cfg["strategy"]["n_drop"] = 5

        with self.assertRaisesRegex(ValueError, "strategy.n_drop must be smaller than strategy.topk"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_unknown_keys(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["unknown_knob"] = 1

        with self.assertRaisesRegex(ValueError, "Unknown config keys"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_unknown_data_source(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["data"]["source"] = "demo"

        with self.assertRaisesRegex(ValueError, "Unsupported data source"):
            validate_training_config(cfg, check_paths=False)


if __name__ == "__main__":
    unittest.main()
