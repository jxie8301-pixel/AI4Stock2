import unittest
import json
import tempfile
from pathlib import Path

from src.research_safety import (
    check_config_profile_write_safety,
    check_diagnostics_provenance_safety,
    is_training_date_range,
)


class ResearchSafetyTest(unittest.TestCase):
    def test_training_date_range_accepts_subset_of_train_period(self):
        cfg = {"time": {"train": ["2020-01-01", "2021-12-31"]}}

        self.assertTrue(is_training_date_range(cfg, "2020-06-01", "2021-06-30"))

    def test_profile_write_safety_rejects_non_training_range(self):
        cfg = {"time": {"train": ["2020-01-01", "2021-12-31"]}}

        with self.assertRaisesRegex(ValueError, "refuses to write"):
            check_config_profile_write_safety(
                cfg,
                date_start="2022-01-01",
                date_end="2022-12-31",
                allow_unsafe=False,
                tool_name="test_tool",
            )

    def test_profile_write_safety_records_unsafe_override(self):
        cfg = {"time": {"train": ["2020-01-01", "2021-12-31"]}}

        safe, warning = check_config_profile_write_safety(
            cfg,
            date_start="2022-01-01",
            date_end="2022-12-31",
            allow_unsafe=True,
            tool_name="test_tool",
        )

        self.assertFalse(safe)
        self.assertIsNotNone(warning)
        self.assertIn("--allow-unsafe-profile-write", warning or "")

    def test_diagnostics_provenance_rejects_test_period_summary(self):
        cfg = {"time": {"train": ["2020-01-01", "2021-12-31"]}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary_path = output_dir / "single_factor_summary.csv"
            summary_path.write_text("feature,rank_ic_mean\nf1,0.1\n", encoding="utf-8")
            with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "metadata": {
                            "period": "test",
                            "date_start": "2022-01-01",
                            "date_end": "2022-12-31",
                        }
                    },
                    fh,
                )

            issues = check_diagnostics_provenance_safety(cfg, diagnostics_paths=[summary_path])

        self.assertTrue(any("period=test" in issue for issue in issues))

    def test_profile_write_safety_rejects_unsafe_diagnostics_even_when_filter_range_is_train(self):
        cfg = {"time": {"train": ["2020-01-01", "2021-12-31"]}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary_path = output_dir / "single_factor_summary.csv"
            summary_path.write_text("feature,rank_ic_mean\nf1,0.1\n", encoding="utf-8")
            with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "metadata": {
                            "period": "all",
                            "date_start": "2020-01-01",
                            "date_end": "2022-12-31",
                        }
                    },
                    fh,
                )

            with self.assertRaisesRegex(ValueError, "diagnostics period=all"):
                check_config_profile_write_safety(
                    cfg,
                    date_start="2020-01-01",
                    date_end="2021-12-31",
                    allow_unsafe=False,
                    tool_name="test_tool",
                    diagnostics_paths=[summary_path],
                )

    def test_profile_write_safety_rejects_diagnostics_without_manifest(self):
        cfg = {"time": {"train": ["2020-01-01", "2021-12-31"]}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "single_factor_summary.csv"
            summary_path.write_text("feature,rank_ic_mean\nf1,0.1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no sibling manifest"):
                check_config_profile_write_safety(
                    cfg,
                    date_start="2020-01-01",
                    date_end="2021-12-31",
                    allow_unsafe=False,
                    tool_name="test_tool",
                    diagnostics_paths=[summary_path],
                )


if __name__ == "__main__":
    unittest.main()
