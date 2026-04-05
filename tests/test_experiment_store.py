import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiment_store import finalize_run_store, prepare_run_store


class ExperimentStoreTest(unittest.TestCase):
    def _build_cfg(self, store_dir: str) -> dict:
        return {
            "data": {
                "source": "tushare",
            },
            "artifacts": {
                "enable_local_store": True,
                "store_dir": store_dir,
            },
            "universe": "csi300",
            "strategy": {
                "topk": 30,
                "n_drop": 5,
                "weighting": "rank",
                "score_transform": "zscore_clip",
                "score_zscore_clip": 2.5,
                "max_weight": 0.2,
                "keep_top_n": 50,
                "min_score": 0.0,
            },
            "backtest": {
                "rebalance_freq": 3,
            },
            "time": {
                "train": ["2016-01-01", "2022-12-31"],
                "valid": ["2023-01-01", "2023-12-31"],
                "test": ["2024-01-01", "2025-12-31"],
            },
        }

    def test_prepare_run_store_builds_tagged_default_model_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._build_cfg(tmpdir)
            args = argparse.Namespace(
                rebalance_freq=None,
                run_tag="Top 30 / A",
                store_dir=None,
                disable_local_store=False,
            )

            store = prepare_run_store(
                cfg,
                args,
                backend="native",
                pipeline="single",
                model_name="lstm",
                model_ext=".pt",
            )

            self.assertTrue(store.enabled)
            self.assertIsNotNone(store.default_model_path)
            self.assertEqual(store.default_model_path.suffix, ".pt")
            self.assertIn("top-30-a", store.run_id)

    def test_finalize_run_store_archives_outputs_and_updates_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = self._build_cfg(str(tmp_path / "experiments"))
            args = argparse.Namespace(
                rebalance_freq=None,
                run_tag="compare-a",
                store_dir=None,
                disable_local_store=False,
            )
            results_dir = tmp_path / "results"
            results_dir.mkdir()
            (results_dir / "cumulative_return.png").write_text("png", encoding="utf-8")
            (results_dir / "monthly_report.csv").write_text("date,ret\n", encoding="utf-8")

            source_model_path = tmp_path / "lstm.pt"
            source_model_path.write_text("weights", encoding="utf-8")

            store = prepare_run_store(
                cfg,
                args,
                backend="native",
                pipeline="single",
                model_name="lstm",
                model_ext=".pt",
            )

            manifest_path = finalize_run_store(
                store,
                cfg=cfg,
                args=args,
                backend="native",
                pipeline="single",
                model_name="lstm",
                results_dir=results_dir,
                signal_metrics={"IC_mean": 0.12, "ICIR": 1.5, "Rank_IC_mean": 0.08, "Rank_ICIR": 1.2},
                portfolio_metrics={
                    "benchmark_name": "CSI300",
                    "benchmark_annualized_return": {"risk": 0.08},
                    "excess_annualized_return": {"risk": 0.17},
                    "excess_information_ratio": {"risk": 0.9},
                    "monthly_win_rate": {"risk": 0.6},
                    "top_3_positive_month_share": {"risk": 0.4},
                    "top_5_positive_month_share": {"risk": 0.75},
                    "rebalance_win_rate": {"risk": 0.55},
                    "top_3_positive_rebalance_share": {"risk": 0.5},
                    "top_5_positive_rebalance_share": {"risk": 0.8},
                    "payoff_ratio": {"risk": 1.3},
                    "profit_factor": {"risk": 1.6},
                    "avg_factor_baseline_excess_annualized_return": {"risk": 0.03},
                    "avg_factor_baseline_excess_information_ratio": {"risk": 0.4},
                    "months_beating_avg_factor_baseline_pct": {"risk": 0.625},
                    "months_beating_avg_factor_baseline_summary": "30 / 48 = 62.50%",
                    "rebalances_beating_avg_factor_baseline_pct": {"risk": 0.58},
                    "rebalances_beating_avg_factor_baseline_summary": "56 / 97 = 57.73%",
                    "annualized_return": {"risk": 0.25},
                    "information_ratio": {"risk": 1.1},
                    "max_drawdown": {"risk": -0.15},
                },
                model_path=source_model_path,
            )

            self.assertIsNotNone(manifest_path)
            self.assertTrue(manifest_path.exists())
            self.assertTrue((store.artifacts_dir / "cumulative_return.png").exists())
            self.assertTrue((store.models_dir / "lstm.pt").exists())

            config_snapshot = yaml.safe_load((store.run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config_snapshot["strategy"]["topk"], 30)

            index_path = Path(cfg["artifacts"]["store_dir"]) / "experiment_index.csv"
            with open(index_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data_source"], "tushare")
            self.assertEqual(rows[0]["run_tag"], "compare-a")
            self.assertEqual(rows[0]["signal_ic_mean"], "0.12")
            self.assertEqual(rows[0]["benchmark_name"], "CSI300")
            self.assertEqual(rows[0]["benchmark_annualized_return"], "0.08")
            self.assertEqual(rows[0]["excess_annualized_return"], "0.17")
            self.assertEqual(rows[0]["monthly_win_rate"], "0.6")
            self.assertEqual(rows[0]["top_3_positive_month_share"], "0.4")
            self.assertEqual(rows[0]["top_5_positive_month_share"], "0.75")
            self.assertEqual(rows[0]["rebalance_win_rate"], "0.55")
            self.assertEqual(rows[0]["top_3_positive_rebalance_share"], "0.5")
            self.assertEqual(rows[0]["top_5_positive_rebalance_share"], "0.8")
            self.assertEqual(rows[0]["payoff_ratio"], "1.3")
            self.assertEqual(rows[0]["profit_factor"], "1.6")
            self.assertEqual(rows[0]["avg_factor_baseline_excess_annualized_return"], "0.03")
            self.assertEqual(rows[0]["avg_factor_baseline_excess_information_ratio"], "0.4")
            self.assertEqual(rows[0]["months_beating_avg_factor_baseline_pct"], "0.625")
            self.assertEqual(rows[0]["months_beating_avg_factor_baseline_summary"], "30 / 48 = 62.50%")
            self.assertEqual(rows[0]["rebalances_beating_avg_factor_baseline_pct"], "0.58")
            self.assertEqual(rows[0]["rebalances_beating_avg_factor_baseline_summary"], "56 / 97 = 57.73%")
            self.assertEqual(rows[0]["portfolio_annualized_return"], "0.25")
            self.assertEqual(rows[0]["weighting"], "rank")
            self.assertEqual(rows[0]["score_transform"], "zscore_clip")
            self.assertEqual(rows[0]["score_zscore_clip"], "2.5")
            self.assertEqual(rows[0]["max_weight"], "0.2")
            self.assertEqual(rows[0]["keep_top_n"], "50")
            self.assertEqual(rows[0]["min_score"], "0.0")


if __name__ == "__main__":
    unittest.main()
