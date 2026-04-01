import argparse
import unittest

from src.runtime_cli import apply_common_runtime_overrides


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


if __name__ == "__main__":
    unittest.main()
