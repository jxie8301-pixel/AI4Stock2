import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import run_native_rolling
from src.rolling_runtime import load_source_market_data_frame


class RunNativeRollingTest(unittest.TestCase):
    def test_prediction_bundle_round_trip(self):
        index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-02"), "A"),
                (pd.Timestamp("2024-01-03"), "B"),
            ],
            names=["datetime", "instrument"],
        )
        bundle = run_native_rolling.PredictionBundle(
            final_predictions=pd.Series([0.1, 0.2], index=index, name="prediction"),
            label_series=pd.Series([0.01, 0.02], index=index, name="label"),
            backtest_label_series=pd.Series([0.001, 0.002], index=index, name="label"),
            avg_factor_baseline_predictions=pd.Series([1.1, 1.2], index=index, name="prediction"),
            sign_aligned_factor_baseline_predictions=pd.Series([0.9, 0.8], index=index, name="prediction"),
            selected_feature_names=["f1", "f2"],
            metadata={"signal_horizon": 20, "test_start": "2024-01-02", "test_end": "2024-01-03"},
            feature_importance_frames=[],
            training_summary_records=[
                {
                    "window_start": "2024-01-02",
                    "window_end": "2024-01-03",
                    "valid_topk_label_mean": 0.12,
                    "best_valid_daily_rank_ic": 0.34,
                }
            ],
            rank_avg_factor_baseline_predictions=pd.Series([0.7, 0.6], index=index, name="prediction"),
            rank_ic_weighted_factor_baseline_predictions=pd.Series([0.5, 0.4], index=index, name="prediction"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / run_native_rolling.PREDICTION_ARTIFACT_DIRNAME
            run_native_rolling._write_prediction_bundle(bundle, artifact_dir)
            loaded = run_native_rolling.load_prediction_bundle(artifact_dir)

        pd.testing.assert_series_equal(loaded.final_predictions, bundle.final_predictions, check_names=False)
        pd.testing.assert_series_equal(loaded.label_series, bundle.label_series, check_names=False)
        pd.testing.assert_series_equal(loaded.backtest_label_series, bundle.backtest_label_series, check_names=False)
        pd.testing.assert_series_equal(
            loaded.avg_factor_baseline_predictions,
            bundle.avg_factor_baseline_predictions,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            loaded.sign_aligned_factor_baseline_predictions,
            bundle.sign_aligned_factor_baseline_predictions,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            loaded.rank_avg_factor_baseline_predictions,
            bundle.rank_avg_factor_baseline_predictions,
            check_names=False,
        )
        pd.testing.assert_series_equal(
            loaded.rank_ic_weighted_factor_baseline_predictions,
            bundle.rank_ic_weighted_factor_baseline_predictions,
            check_names=False,
        )
        self.assertEqual(loaded.selected_feature_names, ["f1", "f2"])
        self.assertEqual(int(loaded.metadata["signal_horizon"]), 20)
        self.assertEqual(len(loaded.training_summary_records), 1)
        self.assertAlmostEqual(float(loaded.training_summary_records[0]["valid_topk_label_mean"]), 0.12, places=8)

    def test_average_factor_baseline_uses_unique_source_columns(self):
        runtime_data = run_native_rolling.RollingRuntimeData(
            factor_frame=pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                    "symbol": ["A", "B"],
                    "f1": [1.0, 3.0],
                    "f1__rep2": [1.0, 3.0],
                    "f2": [5.0, 7.0],
                }
            ),
            dt_index=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-02"])),
            y=np.array([0.1, 0.2], dtype=np.float32),
            backtest_y=np.array([0.01, 0.02], dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-02"])),
            test_start=pd.Timestamp("2024-01-02"),
            test_end=pd.Timestamp("2024-01-02"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-02"])),
            selected_feature_names=["f1", "f1__rep2", "f2"],
            selected_feature_sources=["f1", "f1", "f2"],
            finite_feature_mask=np.array([True, True]),
            lookback=20,
            batch_size=2,
        )

        preds = run_native_rolling._build_average_factor_baseline_predictions(runtime_data)

        expected = pd.Series(
            [3.0, 5.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
                names=["datetime", "instrument"],
            ),
            name="prediction",
        )
        pd.testing.assert_series_equal(preds, expected)

    def test_sign_aligned_factor_baseline_flips_negative_train_ic_features(self):
        dates = pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2 + ["2024-01-03"] * 2)
        runtime_data = run_native_rolling.RollingRuntimeData(
            factor_frame=pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ["A", "B", "A", "B", "A", "B"],
                    "good": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                    "bad": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                }
            ),
            dt_index=pd.Series(dates),
            y=np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3], dtype=np.float32),
            backtest_y=np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.03], dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            test_start=pd.Timestamp("2024-01-03"),
            test_end=pd.Timestamp("2024-01-03"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-03"])),
            selected_feature_names=["good", "bad"],
            selected_feature_sources=["good", "bad"],
            finite_feature_mask=np.array([True] * 6),
            lookback=20,
            batch_size=2,
        )

        preds = run_native_rolling._build_sign_aligned_factor_baseline_predictions(runtime_data)
        values = preds.to_numpy(dtype=float)
        self.assertGreater(values[0], values[1])

    def test_rank_average_factor_baseline_uses_cross_sectional_rank_zscores(self):
        dates = pd.to_datetime(["2024-01-02"] * 3)
        runtime_data = run_native_rolling.RollingRuntimeData(
            factor_frame=pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ["A", "B", "C"],
                    "f1": [1.0, 2.0, 3.0],
                    "f1__rep2": [1.0, 2.0, 3.0],
                    "f2": [10.0, 20.0, 30.0],
                }
            ),
            dt_index=pd.Series(dates),
            y=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            backtest_y=np.array([0.01, 0.02, 0.03], dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-02"])),
            test_start=pd.Timestamp("2024-01-02"),
            test_end=pd.Timestamp("2024-01-02"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-02"])),
            selected_feature_names=["f1", "f1__rep2", "f2"],
            selected_feature_sources=["f1", "f1", "f2"],
            finite_feature_mask=np.array([True] * 3),
            lookback=20,
            batch_size=3,
        )

        preds = run_native_rolling._build_rank_average_factor_baseline_predictions(runtime_data)

        expected = pd.Series(
            [-1.0, 0.0, 1.0],
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02"), "A"),
                    (pd.Timestamp("2024-01-02"), "B"),
                    (pd.Timestamp("2024-01-02"), "C"),
                ],
                names=["datetime", "instrument"],
            ),
            name="prediction",
        )
        pd.testing.assert_series_equal(preds, expected)

    def test_rank_ic_weighted_factor_baseline_uses_train_rank_ic_signs(self):
        dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3)
        runtime_data = run_native_rolling.RollingRuntimeData(
            factor_frame=pd.DataFrame(
                {
                    "date": dates,
                    "symbol": ["A", "B", "C"] * 3,
                    "good": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
                    "bad": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                }
            ),
            dt_index=pd.Series(dates),
            y=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32),
            backtest_y=np.zeros(9, dtype=np.float32),
            full_calendar=pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])),
            test_start=pd.Timestamp("2024-01-03"),
            test_end=pd.Timestamp("2024-01-03"),
            test_calendar=pd.Series(pd.to_datetime(["2024-01-03"])),
            selected_feature_names=["good", "bad"],
            selected_feature_sources=["good", "bad"],
            finite_feature_mask=np.array([True] * 9),
            lookback=20,
            batch_size=3,
        )

        preds = run_native_rolling._build_rank_ic_weighted_factor_baseline_predictions(runtime_data)

        values = preds.to_numpy(dtype=float)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_resolve_prediction_artifact_dir_accepts_parent_run_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            artifact_dir = run_dir / run_native_rolling.PREDICTION_ARTIFACT_DIRNAME
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with open(artifact_dir / run_native_rolling.PREDICTION_METADATA_FILENAME, "w", encoding="utf-8") as f:
                json.dump({"selected_features": []}, f)

            resolved = run_native_rolling._resolve_prediction_artifact_dir(run_dir)

        self.assertEqual(resolved, artifact_dir)

    def test_load_source_market_data_frame_reads_processed_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_dir = Path(tmpdir) / "processed"
            parquet_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                    "symbol": ["A", "A", "A"],
                    "close": [10.0, 11.0, 12.0],
                }
            ).to_parquet(parquet_dir / "A.parquet", index=False)

            runtime_data = run_native_rolling.RollingRuntimeData(
                factor_frame=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                        "symbol": ["A", "A"],
                        "f1": [1.0, 2.0],
                    }
                ),
                dt_index=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
                y=np.array([0.1, 0.2], dtype=np.float32),
                backtest_y=np.array([0.01, 0.02], dtype=np.float32),
                full_calendar=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
                test_start=pd.Timestamp("2024-01-02"),
                test_end=pd.Timestamp("2024-01-03"),
                test_calendar=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
                selected_feature_names=["f1"],
                selected_feature_sources=["f1"],
                finite_feature_mask=np.array([True, True]),
                lookback=20,
                batch_size=2,
            )
            cfg = {"data": {"source": "tushare", "parquet_dir": str(parquet_dir)}}

            market_data = load_source_market_data_frame(cfg, runtime_data, columns=["close"])

        self.assertEqual(list(market_data.columns), ["close"])
        self.assertEqual(list(market_data.index.get_level_values("symbol").unique()), ["A"])
        self.assertEqual(market_data["close"].tolist(), [10.0, 11.0])

    def test_load_source_market_data_frame_reads_bucket_source_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            buckets = source_dir / "buckets"
            buckets.mkdir(parents=True, exist_ok=True)
            with open(source_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "storage_layout": "bucket_shards",
                        "bucket_ids": [3],
                    },
                    f,
                )
            pd.DataFrame(
                {
                    "symbol": ["A"],
                    "bucket_id": [3],
                    "source_path": ["/tmp/A.parquet"],
                    "source_size": [1],
                    "source_mtime_ns": [1],
                    "row_count": [2],
                    "min_date": ["2024-01-02"],
                    "max_date": ["2024-01-03"],
                }
            ).to_parquet(source_dir / "manifest.parquet", index=False)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                    "symbol": ["A", "A", "A"],
                    "close": [10.0, 11.0, 12.0],
                }
            ).to_parquet(buckets / "part-0003.parquet", index=False)

            runtime_data = run_native_rolling.RollingRuntimeData(
                factor_frame=pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                        "symbol": ["A", "A"],
                        "f1": [1.0, 2.0],
                    }
                ),
                dt_index=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
                y=np.array([0.1, 0.2], dtype=np.float32),
                backtest_y=np.array([0.01, 0.02], dtype=np.float32),
                full_calendar=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
                test_start=pd.Timestamp("2024-01-02"),
                test_end=pd.Timestamp("2024-01-03"),
                test_calendar=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
                selected_feature_names=["f1"],
                selected_feature_sources=["f1"],
                finite_feature_mask=np.array([True, True]),
                lookback=20,
                batch_size=2,
            )
            cfg = {"data": {"source": "tushare", "parquet_dir": str(source_dir)}}

            market_data = load_source_market_data_frame(cfg, runtime_data, columns=["close"])

        self.assertEqual(list(market_data.columns), ["close"])
        self.assertEqual(list(market_data.index.get_level_values("symbol").unique()), ["A"])
        self.assertEqual(market_data["close"].tolist(), [10.0, 11.0])


if __name__ == "__main__":
    unittest.main()
