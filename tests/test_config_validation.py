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

    def test_validate_training_config_accepts_risk_control_benchmark_ma(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {
            "mode": "benchmark_ma",
            "fast_window": 120,
            "slow_window": 250,
            "bull_risk": 0.95,
            "neutral_risk": 0.5,
            "bear_risk": 0.15,
        }

        validated = validate_training_config(cfg, check_paths=False)

        self.assertEqual(validated["backtest"]["risk_control"]["mode"], "benchmark_ma")

    def test_validate_training_config_accepts_risk_control_signal_strength(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {
            "mode": "signal_strength",
            "signal_metric": "topk_mean",
            "min_signal": 0.0,
            "max_signal": 2.0,
            "min_risk": 0.30,
            "max_risk": 0.95,
        }

        validated = validate_training_config(cfg, check_paths=False)

        self.assertEqual(validated["backtest"]["risk_control"]["mode"], "signal_strength")

    def test_validate_training_config_accepts_risk_control_benchmark_ma_signal_strength(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {
            "mode": "benchmark_ma_signal_strength",
            "fast_window": 120,
            "slow_window": 250,
            "bull_risk": 0.95,
            "neutral_risk": 0.80,
            "bear_risk": 0.50,
            "signal_metric": "topk_mean",
            "min_signal": 0.0,
            "max_signal": 2.0,
            "min_risk": 0.40,
            "max_risk": 0.95,
            "min_signal_quantile": 0.2,
            "max_signal_quantile": 0.8,
        }

        validated = validate_training_config(cfg, check_paths=False)

        self.assertEqual(validated["backtest"]["risk_control"]["mode"], "benchmark_ma_signal_strength")

    def test_validate_training_config_rejects_invalid_signal_strength_bounds(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {
            "mode": "signal_strength",
            "min_signal": 2.0,
            "max_signal": 1.0,
        }

        with self.assertRaisesRegex(ValueError, "backtest.risk_control.max_signal must be greater"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_invalid_signal_strength_quantiles(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {
            "mode": "signal_strength",
            "min_signal_quantile": 0.8,
            "max_signal_quantile": 0.2,
        }

        with self.assertRaisesRegex(ValueError, "backtest.risk_control.max_signal_quantile must be greater"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_accepts_legacy_dynamic_risk_alias(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["dynamic_risk"] = {
            "mode": "benchmark_ma",
            "fast_window": 120,
            "slow_window": 250,
        }

        validated = validate_training_config(cfg, check_paths=False)

        self.assertEqual(validated["backtest"]["risk_control"]["mode"], "benchmark_ma")

    def test_validate_training_config_accepts_intraperiod_exit(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["intraperiod_exit"] = {
            "mode": "score_threshold",
            "score_source": "rank_pct",
            "threshold": 0.0,
        }

        validated = validate_training_config(cfg, check_paths=False)

        self.assertEqual(validated["backtest"]["intraperiod_exit"]["mode"], "score_threshold")

    def test_validate_training_config_accepts_expected_return_intraperiod_exit(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["intraperiod_exit"] = {
            "mode": "expected_return_threshold",
            "score_source": "raw",
            "threshold": 0.0,
            "calibration": "quantile_bins",
            "n_bins": 8,
            "min_history": 16,
        }

        validated = validate_training_config(cfg, check_paths=False)

        self.assertEqual(validated["backtest"]["intraperiod_exit"]["mode"], "expected_return_threshold")
        self.assertEqual(validated["backtest"]["intraperiod_exit"]["n_bins"], 8)
        self.assertEqual(validated["backtest"]["intraperiod_exit"]["min_history"], 16)

    def test_validate_training_config_rejects_invalid_intraperiod_exit_mode(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["intraperiod_exit"] = {"mode": "demo"}

        with self.assertRaisesRegex(ValueError, "backtest.intraperiod_exit.mode must be one of"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_invalid_intraperiod_exit_calibration(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["intraperiod_exit"] = {
            "mode": "expected_return_threshold",
            "calibration": "demo",
        }

        with self.assertRaisesRegex(ValueError, "backtest.intraperiod_exit.calibration must be one of"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_invalid_risk_control_windows(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {
            "mode": "benchmark_ma",
            "fast_window": 250,
            "slow_window": 120,
        }

        with self.assertRaisesRegex(ValueError, "backtest.risk_control.fast_window must be smaller"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_invalid_risk_control_mode(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {"mode": "demo"}

        with self.assertRaisesRegex(ValueError, "backtest.risk_control.mode must be one of"):
            validate_training_config(cfg, check_paths=False)

    def test_validate_training_config_rejects_both_risk_configs(self):
        cfg = load_config("configs/config.yaml", experiment_profile_name="core_v4_lgbm_default_10x20x10")
        cfg["backtest"]["risk_control"] = {"mode": "fixed", "risk_degree": 0.95}
        cfg["backtest"]["dynamic_risk"] = {"mode": "benchmark_ma"}

        with self.assertRaisesRegex(ValueError, "Use either backtest.risk_control or backtest.dynamic_risk"):
            validate_training_config(cfg, check_paths=False)


if __name__ == "__main__":
    unittest.main()
