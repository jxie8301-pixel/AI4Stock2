import unittest
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.pure_lightgbm import (
    NativeLGBM,
    _build_direct_ranking_relevance_labels,
    _build_ranking_relevance_labels,
    _compute_time_decay_weights,
    _compute_ranking_groups,
    _daily_ic_metric,
    _daily_rank_ic_metric_from_labels,
    _valid_topk_excess_mean_metric_from_labels,
    _valid_topk_label_mean_metric_from_labels,
    _should_use_direct_ranking_relevance_labels,
)


class NativeModelMetricsTest(unittest.TestCase):
    def test_lightgbm_daily_ic_metric_uses_mean_of_daily_cross_sections(self):
        predictions = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )
        dataset = lgb.Dataset(np.zeros((4, 1), dtype=np.float32), label=labels)

        metric_name, metric_value, higher_is_better = _daily_ic_metric(predictions, dataset, dates)

        self.assertEqual(metric_name, "daily_ic")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.0, places=8)

    def test_native_lgbm_defaults_to_stable_eval_metric(self):
        mse_model = NativeLGBM(loss="mse")
        mae_model = NativeLGBM(loss="mae")
        huber_model = NativeLGBM(loss="huber")
        rank_model = NativeLGBM(loss="rank_xendcg")
        custom_model = NativeLGBM(loss="mse", eval_metric="rmse")

        self.assertEqual(mse_model.params["metric"], "l2")
        self.assertEqual(mae_model.params["metric"], "l1")
        self.assertEqual(huber_model.params["metric"], "rmse")
        self.assertEqual(huber_model.params["objective"], "huber")
        self.assertEqual(rank_model.params["metric"], "ndcg")
        self.assertEqual(rank_model.params["objective"], "rank_xendcg")
        self.assertEqual(custom_model.params["metric"], "rmse")

    def test_native_lgbm_passes_device_params_to_lightgbm(self):
        model = NativeLGBM(
            loss="mse",
            device_type="cuda",
            max_bin=63,
            gpu_device_id=0,
            num_gpu=1,
            is_enable_sparse=False,
            num_threads=4,
        )

        self.assertEqual(model.params["device_type"], "cuda")
        self.assertEqual(model.params["max_bin"], 63)
        self.assertEqual(model.params["gpu_device_id"], 0)
        self.assertEqual(model.params["num_gpu"], 1)
        self.assertIs(model.params["is_enable_sparse"], False)
        self.assertEqual(model.params["num_threads"], 4)

    def test_daily_rank_ic_metric_uses_cross_sectional_spearman_mean(self):
        predictions = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 2.0, 1.0], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )

        metric_name, metric_value, higher_is_better = _daily_rank_ic_metric_from_labels(predictions, labels, dates)

        self.assertEqual(metric_name, "daily_rank_ic")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.0, places=8)

    def test_lightgbm_metrics_ignore_nat_dates(self):
        predictions = np.array([1.0, 2.0, 100.0, 1.0, 2.0], dtype=np.float32)
        labels = np.array([1.0, 2.0, 100.0, 2.0, 1.0], dtype=np.float32)
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", pd.NaT, "2024-01-03", "2024-01-03"])

        metric_name, metric_value, higher_is_better = _daily_rank_ic_metric_from_labels(predictions, labels, dates)

        self.assertEqual(metric_name, "daily_rank_ic")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.0, places=8)

    def test_valid_topk_label_mean_metric_uses_selected_labels(self):
        predictions = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2], dtype=np.float32)
        labels = np.array([0.10, -0.05, -0.10, 0.03, 0.02, -0.04], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )

        metric_name, metric_value, higher_is_better = _valid_topk_label_mean_metric_from_labels(
            predictions,
            labels,
            dates,
            topk=2,
        )

        self.assertEqual(metric_name, "valid_topk_label_mean")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.025, places=8)

    def test_valid_topk_label_mean_preserves_boundary_tie_order(self):
        predictions = np.array([0.9, 0.8, 0.8, 0.1], dtype=np.float32)
        labels = np.array([1.0, 2.0, 100.0, 0.0], dtype=np.float32)
        dates = np.array(["2024-01-02"] * 4, dtype="datetime64[ns]")

        metric_name, metric_value, higher_is_better = _valid_topk_label_mean_metric_from_labels(
            predictions,
            labels,
            dates,
            topk=2,
        )

        self.assertEqual(metric_name, "valid_topk_label_mean")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 1.5, places=8)

    def test_valid_topk_excess_mean_metric_uses_cross_section_excess(self):
        predictions = np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.2], dtype=np.float32)
        labels = np.array([0.10, -0.05, -0.10, 0.03, 0.02, -0.04], dtype=np.float32)
        dates = np.array(
            ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"],
            dtype="datetime64[ns]",
        )

        metric_name, metric_value, higher_is_better = _valid_topk_excess_mean_metric_from_labels(
            predictions,
            labels,
            dates,
            topk=2,
        )

        self.assertEqual(metric_name, "valid_topk_excess_mean")
        self.assertTrue(higher_is_better)
        self.assertAlmostEqual(metric_value, 0.031666667, places=8)

    def test_native_lgbm_custom_early_stopping_metric_disables_builtin_metric(self):
        model = NativeLGBM(loss="huber", early_stopping_metric="daily_rank_ic")

        self.assertEqual(model.params["metric"], "None")
        self.assertEqual(model.early_stopping_metric, "daily_rank_ic")

    def test_native_lgbm_rejects_unknown_early_stopping_metric(self):
        with self.assertRaisesRegex(ValueError, "early_stopping_metric must be one of"):
            NativeLGBM(loss="huber", early_stopping_metric="demo")

    def test_native_lgbm_accepts_topk_based_early_stopping_metrics(self):
        label_model = NativeLGBM(loss="rank_xendcg", early_stopping_metric="valid_topk_label_mean")
        excess_model = NativeLGBM(loss="rank_xendcg", early_stopping_metric="valid_topk_excess_mean")

        self.assertEqual(label_model.early_stopping_metric, "valid_topk_label_mean")
        self.assertEqual(excess_model.early_stopping_metric, "valid_topk_excess_mean")

    def test_native_lgbm_can_export_feature_importance_csv(self):
        model = NativeLGBM(loss="mse", early_stop=0, num_threads=1)
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])

        model.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.csv"
            model.save_feature_importance(save_path)
            exported = pd.read_csv(save_path)

        self.assertEqual(exported.columns.tolist(), ["feature", "gain"])
        self.assertEqual(set(exported["feature"]), {"f1", "f2"})

    def test_native_lgbm_accepts_numpy_feature_matrices(self):
        model = NativeLGBM(loss="mse", early_stop=0, num_threads=1)
        X_train = np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float64,
        )
        y_train = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)

        model.fit(X_train, y_train, feature_names=["f1", "f2"])
        preds = model.predict(X_train, feature_names=["f1", "f2"])

        self.assertEqual(preds.shape, (4,))
        self.assertEqual(set(model.model.feature_name()), {"f1", "f2"})

    def test_compute_time_decay_weights_emphasizes_recent_rows(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-31"])

        weights = _compute_time_decay_weights(dates, half_life=10)

        self.assertEqual(weights.shape[0], 3)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))

    def test_compute_time_decay_weights_floor_preserves_far_history_weight(self):
        dates = pd.to_datetime(["2024-01-01", "2024-01-11", "2024-01-31"])

        pure_exp = _compute_time_decay_weights(dates, half_life=10, floor=0.0)
        floored = _compute_time_decay_weights(dates, half_life=10, floor=0.2)

        self.assertAlmostEqual(float(floored.mean()), 1.0, places=6)
        self.assertLess(float(floored[0]), float(floored[1]))
        self.assertLess(float(floored[1]), float(floored[2]))
        self.assertGreater(float(floored[0]), float(pure_exp[0]))
        self.assertLess(float(floored[2]), float(pure_exp[2]))

    def test_native_lgbm_accepts_time_decay_training_weights(self):
        model = NativeLGBM(loss="mse", early_stop=0, num_threads=1, train_weight_half_life=10)
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"])

        model.fit(X_train, y_train, train_dates=train_dates)

        self.assertIsNotNone(model.model)

    def test_native_lgbm_accepts_time_decay_floor_training_weights(self):
        model = NativeLGBM(
            loss="mse",
            early_stop=0,
            num_threads=1,
            train_weight_half_life=10,
            train_weight_floor=0.2,
        )
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"])

        model.fit(X_train, y_train, train_dates=train_dates)

        self.assertIsNotNone(model.model)

    def test_build_ranking_relevance_labels_is_cross_sectional(self):
        labels = np.array([0.10, 0.00, -0.10, 0.20, 0.10, 0.00], dtype=np.float32)
        dates = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-03",
                "2024-01-03",
                "2024-01-03",
            ]
        )

        rel = _build_ranking_relevance_labels(labels, dates, num_bins=3)
        groups = _compute_ranking_groups(dates)

        self.assertEqual(groups.tolist(), [3, 3])
        self.assertEqual(rel.tolist()[:3], [2, 1, 0])
        self.assertEqual(rel.tolist()[3:], [2, 1, 0])

    def test_should_use_direct_ranking_relevance_labels_detects_bucket_labels(self):
        labels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        self.assertTrue(_should_use_direct_ranking_relevance_labels(labels, max_unique_values=5))

    def test_should_use_direct_ranking_relevance_labels_rejects_continuous_returns(self):
        labels = np.array([0.01, 0.02, -0.03, 0.04], dtype=np.float32)

        self.assertFalse(_should_use_direct_ranking_relevance_labels(labels, max_unique_values=5))

    def test_build_direct_ranking_relevance_labels_shifts_to_zero_based(self):
        labels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        rel = _build_direct_ranking_relevance_labels(labels)

        self.assertEqual(rel.tolist(), [0, 1, 2, 3, 4])

    def test_native_lgbm_accepts_ranking_objective(self):
        model = NativeLGBM(loss="rank_xendcg", early_stop=0, num_threads=1, ranking_num_bins=3)
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.3, 0.2])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])

        model.fit(X_train, y_train, train_dates=train_dates)

        self.assertIsNotNone(model.model)
        preds = model.predict(X_train)
        self.assertEqual(preds.shape[0], len(X_train))

    def test_native_lgbm_custom_early_stopping_records_rank_metrics(self):
        model = NativeLGBM(
            loss="rank_xendcg",
            early_stop=3,
            num_boost_round=20,
            num_threads=1,
            ranking_num_bins=3,
            early_stopping_metric="daily_rank_ic",
            log_evaluation_period=1000,
        )
        X_train = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "f2": [2.0, 1.0, 0.0, 1.5, 0.5, -0.5],
            }
        )
        y_train = pd.Series([0.0, 0.1, 0.3, -0.1, 0.2, 0.4])
        train_dates = pd.to_datetime(
            ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"]
        )
        X_valid = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "f2": [1.8, 0.8, -0.2, 1.3, 0.3, -0.7],
            }
        )
        y_valid = pd.Series([0.0, 0.05, 0.2, -0.2, 0.1, 0.3])
        valid_dates = pd.to_datetime(
            ["2024-01-03", "2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04", "2024-01-04"]
        )

        model.fit(
            X_train,
            y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            train_dates=train_dates,
            valid_dates=valid_dates,
        )

        history = model.get_training_history_frame()
        self.assertIn("valid_daily_rank_ic", history.columns)
        self.assertIn("valid_daily_ic", history.columns)
        self.assertIsNotNone(model.best_iteration_)

    def test_native_lgbm_custom_early_stopping_requires_valid_dates(self):
        model = NativeLGBM(loss="huber", early_stop=5, num_threads=1, early_stopping_metric="daily_rank_ic")
        X_train = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 1.0, 0.0, 0.0]})
        y_train = pd.Series([0.0, 0.1, 0.2, 0.3])
        X_valid = pd.DataFrame({"f1": [0.5, 1.5], "f2": [1.0, 0.0]})
        y_valid = pd.Series([0.05, 0.15])
        train_dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-20"])

        with self.assertRaisesRegex(ValueError, "valid_dates is required"):
            model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, train_dates=train_dates)

if __name__ == "__main__":
    unittest.main()
