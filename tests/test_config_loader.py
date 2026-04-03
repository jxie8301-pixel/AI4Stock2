import unittest

from src.config_loader import load_config, load_runtime_config
from src.experiment_profiles import resolve_experiment_profile
from src.model_profiles import resolve_model_profile


class ConfigLoaderTest(unittest.TestCase):
    def test_resolve_model_profile_loads_external_yaml(self):
        profile = resolve_model_profile({"model": {"profile": "lgbm_default"}})

        self.assertEqual(profile["name"], "lgbm_default")
        self.assertTrue(profile["path"].endswith("configs/models/lgbm_default.yaml"))
        self.assertEqual(profile["config"]["model"]["name"], "lgbm")

    def test_resolve_experiment_profile_loads_external_yaml(self):
        profile = resolve_experiment_profile({}, profile_name="core_v4_lgbm_default_10x20x10")

        self.assertEqual(profile["name"], "core_v4_lgbm_default_10x20x10")
        self.assertTrue(profile["path"].endswith("configs/experiments/core_v4_lgbm_default_10x20x10.yaml"))
        self.assertEqual(profile["config"]["features"]["profile"], "core_v4_techlite")

    def test_load_config_merges_experiment_and_model_profiles(self):
        runtime_cfg = load_runtime_config("configs/config.yaml")
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")

        self.assertEqual(cfg["data"]["source"], runtime_cfg["data"]["source"])
        self.assertEqual(cfg["experiment"]["profile"], "core_v4_lgbm_default_10x20x10")
        self.assertTrue(cfg["experiment"]["profile_path"].endswith("configs/experiments/core_v4_lgbm_default_10x20x10.yaml"))
        self.assertEqual(cfg["features"]["profile"], "core_v4_techlite")
        self.assertEqual(cfg["model"]["profile"], "lgbm_default")
        self.assertTrue(cfg["model"]["profile_path"].endswith("configs/models/lgbm_default.yaml"))
        self.assertEqual(cfg["model"]["name"], "lgbm")
        self.assertEqual(cfg["lgbm"]["learning_rate"], 0.05)
        self.assertEqual(cfg["label"]["signal_horizon"], 20)
        self.assertEqual(cfg["rolling"]["retrain_step"], 10)
        self.assertEqual(cfg["backtest"]["rebalance_freq"], 10)

    def test_load_config_supports_recent_lgbm_profile(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_recent_10x20x10")

        self.assertEqual(cfg["rolling"]["train_days"], 180)
        self.assertEqual(cfg["rolling"]["valid_days"], 20)
        self.assertEqual(cfg["lgbm"]["train_weight_half_life"], 60)

    def test_resolve_ranker_model_profile(self):
        profile = resolve_model_profile({"model": {"profile": "lgbm_ranker_default"}})

        self.assertEqual(profile["name"], "lgbm_ranker_default")
        self.assertTrue(profile["path"].endswith("configs/models/lgbm_ranker_default.yaml"))
        self.assertEqual(profile["config"]["lgbm"]["loss"], "rank_xendcg")
        self.assertEqual(profile["config"]["lgbm"]["ranking_num_bins"], 5)

    def test_load_config_requires_explicit_experiment_profile(self):
        with self.assertRaises(ValueError):
            load_config("configs/config.yaml")


if __name__ == "__main__":
    unittest.main()
