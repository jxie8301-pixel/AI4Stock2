import unittest

from src.override_utils import (
    apply_dotted_override,
    build_override_tag,
    expand_sweep_grid,
    flatten_sweep_mapping,
    parse_override_arg,
    parse_sweep_arg,
)


class OverrideUtilsTest(unittest.TestCase):
    def test_parse_override_arg_parses_scalar_types(self):
        key, value = parse_override_arg("strategy.topk=15")
        self.assertEqual(key, "strategy.topk")
        self.assertEqual(value, 15)

    def test_apply_dotted_override_builds_nested_mappings(self):
        cfg = {}
        apply_dotted_override(cfg, "strategy.topk", 20)
        apply_dotted_override(cfg, "data.source", "tushare")

        self.assertEqual(cfg, {"strategy": {"topk": 20}, "data": {"source": "tushare"}})

    def test_parse_sweep_arg_accepts_yaml_list_syntax(self):
        key, values = parse_sweep_arg("rolling.retrain_step=[5,10,15]")
        self.assertEqual(key, "rolling.retrain_step")
        self.assertEqual(values, [5, 10, 15])

    def test_flatten_sweep_mapping_accepts_nested_dicts(self):
        flattened = flatten_sweep_mapping(
            {
                "strategy": {"topk": [10, 20]},
                "rolling.retrain_step": [5, 10],
            }
        )

        self.assertEqual(flattened["strategy.topk"], [10, 20])
        self.assertEqual(flattened["rolling.retrain_step"], [5, 10])

    def test_expand_sweep_grid_builds_cartesian_product(self):
        runs = expand_sweep_grid(
            {
                "strategy.topk": [10, 20],
                "rolling.retrain_step": [5, 10],
            }
        )

        self.assertEqual(len(runs), 4)
        self.assertIn({"strategy.topk": 10, "rolling.retrain_step": 5}, runs)
        self.assertIn({"strategy.topk": 20, "rolling.retrain_step": 10}, runs)

    def test_build_override_tag_is_stable(self):
        tag = build_override_tag({"rolling.retrain_step": 10, "strategy.topk": 20})
        self.assertEqual(tag, "rolling-retrain-step-10__strategy-topk-20")


if __name__ == "__main__":
    unittest.main()
