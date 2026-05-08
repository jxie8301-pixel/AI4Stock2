import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.factor_store import load_factor_frame
from src.gen_feature import (
    DEFAULT_FACTOR_GENERATION_TIMING_FILENAME,
    TUSHARE_EVENT_AVAILABILITY_POLICY,
    _get_tushare_bucket_source_required_columns,
    _resolve_factor_generation_runtime,
    _write_factor_bucket_from_source_bucket_worker,
    generate_factor_store,
    get_factor_family_counts,
    get_full_factor_space_feature_names,
)


def _write_bucket_source(
    root: Path,
    frame: pd.DataFrame,
    *,
    event_policy: str = TUSHARE_EVENT_AVAILABILITY_POLICY,
) -> Path:
    source_dir = root / "source"
    buckets = source_dir / "buckets"
    buckets.mkdir(parents=True, exist_ok=True)
    with open(source_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "storage_layout": "bucket_shards",
                "bucket_ids": [5],
                "bucket_count": 16,
                "source_layout_assumptions": {
                    "tushare_event_availability_policy": event_policy,
                    "tushare_industry_mapping": "static_symbol_cache_current_classification",
                },
            },
            f,
        )
    pd.DataFrame(
        {
            "symbol": ["000001"],
            "bucket_id": [5],
            "source_path": ["/tmp/000001.parquet"],
            "source_size": [1],
            "source_mtime_ns": [1],
            "row_count": [len(frame)],
            "min_date": [str(pd.to_datetime(frame["date"]).min().date())],
            "max_date": [str(pd.to_datetime(frame["date"]).max().date())],
        }
    ).to_parquet(source_dir / "manifest.parquet", index=False)
    frame.to_parquet(buckets / "part-0005.parquet", index=False)
    return source_dir


def _minimal_bucket_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "symbol": ["000001", "000001", "000001"],
            "open": [10.0, 10.5, 11.0],
            "high": [10.3, 10.8, 11.2],
            "low": [9.8, 10.2, 10.7],
            "close": [10.1, 10.6, 11.1],
            "volume": [100.0, 120.0, 140.0],
            "amount": [1000.0, 1200.0, 1400.0],
        }
    )


def _minimal_tushare_bucket_frame() -> pd.DataFrame:
    frame = _minimal_bucket_frame()
    missing_columns = {
        column: np.nan
        for column in _get_tushare_bucket_source_required_columns()
        if column not in frame.columns
    }
    if missing_columns:
        frame = pd.concat([frame, pd.DataFrame(missing_columns, index=frame.index)], axis=1)
    frame["turnover"] = [1.0, 1.2, 1.4]
    frame["turnover_free"] = [1.5, 1.7, 1.9]
    frame["volume_ratio"] = [1.0, 1.1, 1.2]
    frame["pb"] = [1.0, 1.1, 1.2]
    frame["pe"] = [10.0, 11.0, 12.0]
    frame["pe_ttm"] = [10.0, 11.0, 12.0]
    frame["ps"] = [2.0, 2.2, 2.4]
    frame["ps_ttm"] = [2.0, 2.2, 2.4]
    frame["dv_ratio"] = [0.2, 0.2, 0.2]
    frame["dv_ttm"] = [0.3, 0.3, 0.3]
    frame["amplitude"] = [5.0, 6.0, 7.0]
    frame["pct_chg"] = [1.0, 2.0, 3.0]
    frame["limit_pre_close"] = [10.0, 10.5, 11.0]
    frame["up_limit"] = [11.0, 11.55, 12.1]
    frame["down_limit"] = [9.0, 9.45, 9.9]
    return frame


class GenFeatureStoreTest(unittest.TestCase):
    def test_factor_family_counts_come_from_registry(self):
        default_counts = get_factor_family_counts()
        tushare_counts = get_factor_family_counts(data_source="tushare")

        self.assertEqual(default_counts["legacy158"], 158)
        self.assertEqual(default_counts["lgbm_purified"], 22)
        self.assertEqual(default_counts["temporal"], 77)
        self.assertEqual(default_counts["technical"], 26)
        self.assertNotIn("tushare", default_counts)
        self.assertIn("tushare", tushare_counts)
        self.assertEqual(default_counts["total"], len(get_full_factor_space_feature_names()))
        self.assertEqual(tushare_counts["total"], len(get_full_factor_space_feature_names(data_source="tushare")))

    def test_factor_generation_runtime_uses_shared_config_resolution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "data:",
                        "  source: tushare",
                        f"  parquet_dir: {root / 'source'}",
                        "features:",
                        f"  cache_dir: {root / 'legacy_cache_dir'}",
                        "label:",
                        "  horizons: [1, 10]",
                    ]
                ),
                encoding="utf-8",
            )
            args = Namespace(
                config=str(config_path),
                data_source=None,
                feature_profile=None,
                parquet_dir=None,
                output_dir=None,
                workers=1,
                label_horizons=None,
                set_overrides=[
                    f"features.factor_store_dir={root / 'factor_store'}",
                    "label.horizons=[1,5,20]",
                ],
                incremental=True,
            )

            runtime = _resolve_factor_generation_runtime(args)

        self.assertEqual(runtime.data_source, "tushare")
        self.assertEqual(runtime.parquet_dir, str(root / "source"))
        self.assertEqual(runtime.output_dir, str(root / "factor_store"))
        self.assertEqual(runtime.label_horizons, [1, 5, 10, 20])

    def test_generate_factor_store_can_read_bucket_sharded_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "factor_store"
            source_dir = _write_bucket_source(root, _minimal_bucket_frame())

            meta = generate_factor_store(
                parquet_dir=str(source_dir),
                output_dir=str(output_dir),
                workers=1,
                incremental=False,
                label_horizons=[1],
            )
            loaded = load_factor_frame(store_dir=output_dir, columns=[meta["feature_names"][0]])

        self.assertEqual(meta["storage_layout"], "bucket_shards")
        self.assertEqual(meta["source_storage_layout"], "bucket_shards")
        self.assertEqual(len(loaded), 3)
        self.assertIn(meta["feature_names"][0], loaded.columns)

    def test_bucket_source_worker_streams_one_row_group_per_symbol(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = _minimal_bucket_frame()
            second = _minimal_bucket_frame().copy()
            second["symbol"] = "000002"
            source_dir = _write_bucket_source(root, pd.concat([first, second], ignore_index=True))
            output_bucket_root = root / "factor_store" / "buckets"
            source_bucket_path = source_dir / "buckets" / "part-0005.parquet"

            result = _write_factor_bucket_from_source_bucket_worker(
                str(source_bucket_path),
                output_bucket_root=str(output_bucket_root),
                label_horizons=[1],
                feature_names=get_full_factor_space_feature_names(),
            )
            output_path = output_bucket_root / "part-0005.parquet"
            loaded = pd.read_parquet(output_path)
            parquet_file = pq.ParquetFile(output_path)

        self.assertEqual(len(result.manifest_rows), 2)
        self.assertEqual(parquet_file.num_row_groups, 2)
        self.assertEqual(len(loaded), 6)
        self.assertEqual(sorted(loaded["symbol"].unique().tolist()), ["000001", "000002"])

    def test_generate_factor_store_writes_timing_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "factor_store"
            source_dir = _write_bucket_source(root, _minimal_bucket_frame())
            timing_path = root / "timing" / "factor_timing.json"

            meta = generate_factor_store(
                parquet_dir=str(source_dir),
                output_dir=str(output_dir),
                workers=1,
                incremental=False,
                label_horizons=[1],
                timing_output_path=timing_path,
            )

            with open(timing_path, encoding="utf-8") as f:
                timing = json.load(f)

        self.assertEqual(meta["timing"]["timing_path"], str(timing_path))
        self.assertEqual(timing["artifact"]["default_filename"], DEFAULT_FACTOR_GENERATION_TIMING_FILENAME)
        self.assertIn("factor_compute", timing["phases"])
        self.assertIn("read_source_bucket", timing["phases"])
        self.assertGreater(timing["phases"]["factor_compute"]["seconds"], 0.0)
        self.assertEqual(timing["phases"]["factor_compute"]["symbols"], 1)

    def test_tushare_bucket_source_requires_prejoined_context_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = _write_bucket_source(root, _minimal_bucket_frame())

            with self.assertRaisesRegex(ValueError, "Tushare bucket source is missing required sidecar/context columns"):
                generate_factor_store(
                    parquet_dir=str(source_dir),
                    output_dir=str(root / "factor_store"),
                    workers=1,
                    incremental=False,
                    label_horizons=[1],
                    data_source="tushare",
                )

    def test_tushare_bucket_source_schema_validation_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = _write_bucket_source(root, _minimal_tushare_bucket_frame())

            meta = generate_factor_store(
                parquet_dir=str(source_dir),
                output_dir=str(root / "factor_store"),
                workers=1,
                incremental=False,
                label_horizons=[1],
                data_source="tushare",
            )

        self.assertTrue(meta["source_schema_validation"]["validated"])
        self.assertIn("ind_bp_clean_mean", meta["source_schema_validation"]["required_columns"])
        self.assertEqual(
            meta["source_schema_validation"]["tushare_event_availability_policy"],
            TUSHARE_EVENT_AVAILABILITY_POLICY,
        )
        self.assertTrue(meta["source_layout_assumptions"]["tushare_bucket_source_requires_sidecar_context_columns"])

    def test_full_rebuild_refreshes_stale_default_tushare_bucket_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = _write_bucket_source(root, _minimal_bucket_frame())
            processed_dir = root / "processed"
            processed_dir.mkdir()
            _minimal_bucket_frame().to_parquet(processed_dir / "000001.parquet", index=False)
            calls: list[dict[str, object]] = []

            def rebuild_packed_source(
                symbols: list[str] | None = None,
                *,
                bucket_count: int = 128,
                workers: int = 8,
                incremental: bool = True,
            ) -> dict[str, object]:
                calls.append(
                    {
                        "symbols": symbols,
                        "bucket_count": bucket_count,
                        "workers": workers,
                        "incremental": incremental,
                    }
                )
                _write_bucket_source(root, _minimal_tushare_bucket_frame())
                return {"incremental": {"rebuilt_buckets": 1, "reused_buckets": 0}}

            from src import collector_tushare as collector_module

            with (
                patch.object(collector_module, "PACKED_SOURCE_DIR", source_dir),
                patch.object(collector_module, "PROCESSED_DIR", processed_dir),
                patch.object(collector_module, "rebuild_packed_source_from_local", rebuild_packed_source),
            ):
                meta = generate_factor_store(
                    parquet_dir=str(source_dir),
                    output_dir=str(root / "factor_store"),
                    workers=1,
                    incremental=False,
                    label_horizons=[1],
                    data_source="tushare",
                )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["symbols"], ["000001"])
        self.assertTrue(calls[0]["incremental"])
        self.assertTrue(meta["source_schema_validation"]["validated"])
        self.assertTrue(meta["source_schema_validation"]["auto_rebuilt_stale_source"])

    def test_full_rebuild_refreshes_default_tushare_bucket_source_on_event_policy_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = _write_bucket_source(
                root,
                _minimal_tushare_bucket_frame(),
                event_policy="legacy_same_day_ann_date",
            )
            processed_dir = root / "processed"
            processed_dir.mkdir()
            _minimal_bucket_frame().to_parquet(processed_dir / "000001.parquet", index=False)
            calls: list[dict[str, object]] = []

            def rebuild_packed_source(
                symbols: list[str] | None = None,
                *,
                bucket_count: int = 128,
                workers: int = 8,
                incremental: bool = True,
            ) -> dict[str, object]:
                calls.append(
                    {
                        "symbols": symbols,
                        "bucket_count": bucket_count,
                        "workers": workers,
                        "incremental": incremental,
                    }
                )
                _write_bucket_source(root, _minimal_tushare_bucket_frame())
                return {"incremental": {"rebuilt_buckets": 1, "reused_buckets": 0}}

            from src import collector_tushare as collector_module

            with (
                patch.object(collector_module, "PACKED_SOURCE_DIR", source_dir),
                patch.object(collector_module, "PROCESSED_DIR", processed_dir),
                patch.object(collector_module, "rebuild_packed_source_from_local", rebuild_packed_source),
            ):
                meta = generate_factor_store(
                    parquet_dir=str(source_dir),
                    output_dir=str(root / "factor_store"),
                    workers=1,
                    incremental=False,
                    label_horizons=[1],
                    data_source="tushare",
                )

        self.assertEqual(len(calls), 1)
        self.assertTrue(meta["source_schema_validation"]["auto_rebuilt_stale_source"])
        self.assertEqual(
            meta["source_schema_validation"]["tushare_event_availability_policy"],
            TUSHARE_EVENT_AVAILABILITY_POLICY,
        )


if __name__ == "__main__":
    unittest.main()
