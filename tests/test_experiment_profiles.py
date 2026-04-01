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


if __name__ == "__main__":
    unittest.main()
