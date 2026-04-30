import argparse
import unittest

import run_experiment_batch


class RunExperimentBatchTest(unittest.TestCase):
    def test_resolve_explicit_cases_groups_overrides_per_run(self):
        args = argparse.Namespace(
            case_overrides=[
                ["strategy.topk=5", "strategy.n_drop=1"],
                ["strategy.topk=10", "strategy.n_drop=2"],
            ]
        )

        cases = run_experiment_batch._resolve_explicit_cases(args)

        self.assertEqual(
            cases,
            [
                {"strategy.topk": 5, "strategy.n_drop": 1},
                {"strategy.topk": 10, "strategy.n_drop": 2},
            ],
        )

    def test_resolve_explicit_cases_returns_empty_list_when_absent(self):
        args = argparse.Namespace(case_overrides=None)

        cases = run_experiment_batch._resolve_explicit_cases(args)

        self.assertEqual(cases, [])

    def test_prediction_fingerprint_ignores_replay_only_n_drop(self):
        base_cfg = {
            "data": {"source": "tushare"},
            "features": {"profile": "core_v4_techlite"},
            "model": {"name": "lgbm"},
            "lgbm": {"validation_topk": 8},
            "strategy": {"topk": 30, "n_drop": 5},
            "backtest": {"rebalance_freq": 10},
            "label": {"signal_horizon": 20},
            "rolling": {"retrain_step": 20, "train_days": 242, "valid_days": 10},
            "time": {"test": ["2022-01-01", "2022-02-07"]},
        }
        replay_cfg = {
            **base_cfg,
            "strategy": {"topk": 30, "n_drop": 4},
        }

        self.assertEqual(
            run_experiment_batch._prediction_fingerprint(base_cfg),
            run_experiment_batch._prediction_fingerprint(replay_cfg),
        )

    def test_prediction_fingerprint_includes_effective_lgbm_validation_topk(self):
        base_cfg = {
            "data": {"source": "tushare"},
            "features": {"profile": "core_v4_techlite"},
            "model": {"name": "lgbm"},
            "lgbm": {},
            "strategy": {"topk": 30, "n_drop": 5},
            "backtest": {"rebalance_freq": 10},
            "label": {"signal_horizon": 20},
            "rolling": {"retrain_step": 20, "train_days": 242, "valid_days": 10},
            "time": {"test": ["2022-01-01", "2022-02-07"]},
        }
        changed_cfg = {
            **base_cfg,
            "strategy": {"topk": 20, "n_drop": 5},
        }

        self.assertNotEqual(
            run_experiment_batch._prediction_fingerprint(base_cfg),
            run_experiment_batch._prediction_fingerprint(changed_cfg),
        )


if __name__ == "__main__":
    unittest.main()
