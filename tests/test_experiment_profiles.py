import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiment_profiles import resolve_experiment_profile


class ExperimentProfilesTest(unittest.TestCase):
    def test_resolve_experiment_profile_separates_sweep_from_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "configs"
            exp_dir = config_dir / "experiments"
            exp_dir.mkdir(parents=True, exist_ok=True)

            profile_path = exp_dir / "demo.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "features": {"profile": "core_v4_techlite"},
                        "model": {"profile": "lgbm_default"},
                        "universe": "csi300",
                        "sweep": {
                            "strategy.topk": [10, 20],
                            "rolling": {"retrain_step": [5, 10]},
                        },
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            index_path = config_dir / "experiment_profiles.yaml"
            index_path.write_text(
                yaml.safe_dump(
                    {"profiles": {"demo": {"path": "configs/experiments/demo.yaml"}}},
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            profile = resolve_experiment_profile(
                {},
                profile_name="demo",
                profile_config_path=str(index_path),
            )

            self.assertEqual(profile["name"], "demo")
            self.assertNotIn("sweep", profile["config"])
            self.assertIn("sweep", profile["raw"])
            self.assertEqual(profile["sweep"]["strategy.topk"], [10, 20])
            self.assertEqual(profile["sweep"]["rolling"]["retrain_step"], [5, 10])

    def test_resolve_experiment_profile_supports_inline_extends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)

            index_path = config_dir / "experiment_profiles.yaml"
            index_path.write_text(
                yaml.safe_dump(
                    {
                        "profiles": {
                            "base": {
                                "features": {"profile": "core_v4_techlite"},
                                "model": {"profile": "lgbm_default"},
                                "universe": "csi300",
                                "time": {
                                    "train": ["2016-01-01", "2020-12-31"],
                                    "valid": ["2021-01-01", "2021-12-31"],
                                    "test": ["2022-01-01", "2025-12-31"],
                                },
                                "label": {"signal_horizon": 20},
                                "rolling": {"retrain_step": 10, "train_days": 242, "valid_days": 10},
                                "strategy": {"topk": 30, "n_drop": 5},
                                "backtest": {
                                    "rebalance_freq": 10,
                                    "cost": {"buy": 0.001, "sell": 0.001},
                                    "slippage": 0.0,
                                    "min_cost": 5,
                                    "account": 100000000,
                                    "risk_degree": 0.95,
                                },
                            },
                            "derived": {
                                "extends": "base",
                                "model": {"profile": "lgbm_fast"},
                                "rolling": {"retrain_step": 20},
                            },
                        }
                    },
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

            profile = resolve_experiment_profile(
                {},
                profile_name="derived",
                profile_config_path=str(index_path),
            )

            self.assertEqual(profile["name"], "derived")
            self.assertTrue(profile["path"].endswith("experiment_profiles.yaml::derived"))
            self.assertEqual(profile["config"]["model"]["profile"], "lgbm_fast")
            self.assertEqual(profile["config"]["rolling"]["retrain_step"], 20)
            self.assertEqual(profile["config"]["strategy"]["topk"], 30)


if __name__ == "__main__":
    unittest.main()
