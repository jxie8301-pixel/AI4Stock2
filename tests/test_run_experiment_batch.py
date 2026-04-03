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


if __name__ == "__main__":
    unittest.main()
