import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.factor_store import load_factor_frame
from src.gen_feature import _get_tushare_bucket_source_required_columns, generate_factor_store


def _write_bucket_source(root: Path, frame: pd.DataFrame) -> Path:
    source_dir = root / "source"
    buckets = source_dir / "buckets"
    buckets.mkdir(parents=True, exist_ok=True)
    with open(source_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "storage_layout": "bucket_shards",
                "bucket_ids": [5],
                "bucket_count": 16,
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


if __name__ == "__main__":
    unittest.main()
