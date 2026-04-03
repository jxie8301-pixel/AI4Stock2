import tempfile
import unittest
from pathlib import Path

from src.config_loader import load_config
from src.config_validation import validate_training_config


class ConfigValidationTest(unittest.TestCase):
    def test_validate_training_config_accepts_valid_profile_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            factor_store_dir = tmp_path / "factor_store" / "full_factor_space"
            factor_store_dir.mkdir(parents=True, exist_ok=True)
            universe_dir = tmp_path / "universes"
            universe_dir.mkdir(parents=True, exist_ok=True)
            (universe_dir / "csi300.txt").write_text("000001.SZ\n", encoding="utf-8")
            benchmark_dir = tmp_path / "benchmarks" / "tushare"
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            (benchmark_dir / "csi300.parquet").write_text("stub", encoding="utf-8")

            cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
            cfg["features"]["factor_store_dir"] = str(factor_store_dir)
            cfg["native"]["universe_dir"] = str(universe_dir)
            cfg["backtest"]["benchmark"]["path"] = str(benchmark_dir / "csi300.parquet")

            validated = validate_training_config(cfg, check_paths=True)

            self.assertEqual(validated["experiment"]["profile"], "core_v4_lgbm_default_10x20x10")

    def test_validate_training_config_rejects_invalid_strategy_shape(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["topk"] = 5
        cfg["strategy"]["n_drop"] = 5

        with self.assertRaisesRegex(ValueError, "strategy.n_drop must be smaller than strategy.topk"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_unknown_weighting_mode(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["weighting"] = "demo"

        with self.assertRaisesRegex(ValueError, "strategy.weighting must be one of"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_unknown_score_transform(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["score_transform"] = "demo"

        with self.assertRaisesRegex(ValueError, "strategy.score_transform must be one of"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_invalid_max_weight(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["max_weight"] = 1.5

        with self.assertRaisesRegex(ValueError, "strategy.max_weight must be in"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_keep_top_n_smaller_than_topk(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["keep_top_n"] = 10

        with self.assertRaisesRegex(ValueError, "strategy.keep_top_n must be >="):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_nonpositive_score_zscore_clip(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["strategy"]["score_zscore_clip"] = 0

        with self.assertRaisesRegex(ValueError, "strategy.score_zscore_clip must be > 0"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_excessive_ranking_bins(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["lgbm"]["loss"] = "rank_xendcg"
        cfg["lgbm"]["ranking_num_bins"] = 32

        with self.assertRaisesRegex(ValueError, "lgbm.ranking_num_bins must be <= 31"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_nonpositive_train_weight_half_life(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg.setdefault("lgbm", {})
        cfg["lgbm"]["train_weight_half_life"] = 0

        with self.assertRaisesRegex(ValueError, "lgbm.train_weight_half_life must be > 0"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_unknown_keys(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["unknown_knob"] = 1

        with self.assertRaisesRegex(ValueError, "Unknown config keys"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_unknown_data_source(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["data"]["source"] = "demo"

        with self.assertRaisesRegex(ValueError, "Unsupported data source"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_accepts_file_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            factor_store_dir = tmp_path / "factor_store" / "full_factor_space"
            factor_store_dir.mkdir(parents=True, exist_ok=True)
            universe_dir = tmp_path / "universes"
            universe_dir.mkdir(parents=True, exist_ok=True)
            (universe_dir / "csi300.txt").write_text("000001.SZ\n", encoding="utf-8")
            benchmark_path = tmp_path / "csi300.csv"
            benchmark_path.write_text("date,close\n2024-01-02,100\n", encoding="utf-8")

            cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
            cfg["features"]["factor_store_dir"] = str(factor_store_dir)
            cfg["native"]["universe_dir"] = str(universe_dir)
            cfg["backtest"]["benchmark"] = {
                "mode": "file",
                "path": str(benchmark_path),
                "date_column": "date",
                "value_column": "close",
                "value_type": "close",
                "name": "CSI300",
            }

            validated = validate_training_config(cfg, check_paths=True)

            self.assertEqual(validated["backtest"]["benchmark"]["mode"], "file")

    def test_validate_training_config_rejects_invalid_benchmark_mode(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["benchmark"] = {"mode": "demo"}

        with self.assertRaisesRegex(ValueError, "backtest.benchmark.mode must be one of"):
            validate_training_config(cfg, check_paths=False)


if __name__ == "__main__":
    unittest.main()
