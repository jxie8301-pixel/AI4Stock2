import unittest

from src.config_loader import load_config, resolve_model_preset


class ConfigLoaderTest(unittest.TestCase):
    def test_resolve_model_preset_loads_external_yaml(self):
        preset = resolve_model_preset({"model": {"preset": "lgbm_default"}})

        self.assertEqual(preset["name"], "lgbm_default")
        self.assertTrue(preset["path"].endswith("configs/models/lgbm_default.yaml"))
        self.assertEqual(preset["config"]["model"]["name"], "lgbm")

    def test_load_config_merges_model_preset(self):
        cfg = load_config("configs/config.yaml")

        self.assertEqual(cfg["model"]["preset"], "lgbm_default")
        self.assertEqual(cfg["model"]["name"], "lgbm")
        self.assertEqual(cfg["lgbm"]["learning_rate"], 0.05)
        self.assertTrue(cfg["model"]["preset_path"].endswith("configs/models/lgbm_default.yaml"))


if __name__ == "__main__":
    unittest.main()
