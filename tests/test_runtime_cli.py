import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.runtime_cli import apply_common_runtime_overrides, load_validated_config_from_args


class RuntimeCliTest(unittest.TestCase):
    def _build_args(self, **overrides):
        base = {
            "model": None,
            "data_source": None,
            "set_overrides": None,
            "feature_profile": None,
            "profile": None,
            "topk": None,
            "n_drop": None,
            "rebalance_freq": None,
            "signal_horizon": None,
            "label_horizon": None,
            "retrain_step": None,
            "horizon": None,
            "train_days": None,
            "valid_days": None,
            "config": "configs/config.yaml",
            "config_is_snapshot": False,
            "experiment_profile": None,
            "model_profile": None,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_apply_common_runtime_overrides_updates_shared_fields(self):
        cfg = {
            "model": {"name": "lgbm"},
            "features": {"profile": "core_v4_techlite"},
            "strategy": {"topk": 30, "n_drop": 5},
            "backtest": {"rebalance_freq": 10},
            "label": {"signal_horizon": 20},
        }
        args = self._build_args(
            model="lstm",
            data_source="tushare",
            set_overrides=["strategy.topk=25"],
            feature_profile="alpha158_full",
            topk=20,
            n_drop=3,
            rebalance_freq=5,
            signal_horizon=10,
        )

        apply_common_runtime_overrides(cfg, args, argparse.ArgumentParser(add_help=False))

        self.assertEqual(cfg["model"]["name"], "lstm")
        self.assertEqual(cfg["data"]["source"], "tushare")
        self.assertEqual(cfg["features"]["profile"], "alpha158_full")
        self.assertEqual(cfg["strategy"]["topk"], 20)
        self.assertEqual(cfg["strategy"]["n_drop"], 3)
        self.assertEqual(cfg["backtest"]["rebalance_freq"], 5)
        self.assertEqual(cfg["label"]["signal_horizon"], 10)

    def test_apply_common_runtime_overrides_supports_rolling_aliases(self):
        cfg = {}
        args = self._build_args(horizon=15, train_days=242, valid_days=10)

        apply_common_runtime_overrides(
            cfg,
            args,
            argparse.ArgumentParser(add_help=False),
            allow_rolling_overrides=True,
        )

        self.assertEqual(cfg["rolling"]["retrain_step"], 15)
        self.assertEqual(cfg["rolling"]["train_days"], 242)
        self.assertEqual(cfg["rolling"]["valid_days"], 10)

    def test_apply_common_runtime_overrides_rejects_conflicting_rolling_aliases(self):
        cfg = {}
        args = self._build_args(retrain_step=10, horizon=20)

        with self.assertRaises(SystemExit):
            apply_common_runtime_overrides(
                cfg,
                args,
                argparse.ArgumentParser(add_help=False),
                allow_rolling_overrides=True,
            )

    def test_load_validated_config_from_args_can_use_config_snapshot_directly(self):
        snapshot = {
            "experiment": {"profile": "core_v4_lgbm_default_10x20x10"},
            "model": {"name": "lgbm", "profile": "lgbm_default"},
            "strategy": {"topk": 8, "n_drop": 2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "config_snapshot.yaml"
            snapshot_path.write_text(yaml.safe_dump(snapshot), encoding="utf-8")
            args = self._build_args(
                config=str(snapshot_path),
                config_is_snapshot=True,
                topk=None,
            )

            with patch("src.runtime_cli.validate_training_config") as validate_mock:
                cfg = load_validated_config_from_args(
                    args,
                    argparse.ArgumentParser(add_help=False),
                    check_paths=False,
                )

        self.assertEqual(cfg["strategy"]["topk"], 8)
        self.assertEqual(cfg["runtime"]["config_path"], str(snapshot_path))
        validate_mock.assert_called_once_with(cfg, check_paths=False)


if __name__ == "__main__":
    unittest.main()
