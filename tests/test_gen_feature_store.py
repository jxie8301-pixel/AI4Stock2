import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.factor_store import load_factor_frame
from src.gen_feature import generate_factor_store


class GenFeatureStoreTest(unittest.TestCase):
    def test_generate_factor_store_can_read_bucket_sharded_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source_dir = root / "source"
            buckets = source_dir / "buckets"
            output_dir = root / "factor_store"
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
                    "row_count": [3],
                    "min_date": ["2024-01-02"],
                    "max_date": ["2024-01-04"],
                }
            ).to_parquet(source_dir / "manifest.parquet", index=False)
            pd.DataFrame(
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
            ).to_parquet(buckets / "part-0005.parquet", index=False)

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


if __name__ == "__main__":
    unittest.main()
