import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.source_store import (
    detect_source_storage_layout,
    load_source_frame,
    load_source_store_metadata,
)


class SourceStoreTest(unittest.TestCase):
    def test_load_source_frame_supports_symbol_shards_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "symbol": ["000001", "000001"],
                    "close": [10.0, 11.0],
                }
            ).to_parquet(root / "000001.parquet", index=False)

            loaded = load_source_frame(
                store_dir=root,
                columns=["close"],
                date_end="2024-01-02",
                symbols=["000001"],
            )

            self.assertEqual(detect_source_storage_layout(root), "symbol_shards")
            self.assertEqual(list(loaded.columns), ["date", "symbol", "close"])
            self.assertEqual(loaded["close"].tolist(), [10.0])

    def test_load_source_frame_supports_bucket_shards_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            buckets = root / "buckets"
            buckets.mkdir(parents=True, exist_ok=True)
            manifest = pd.DataFrame(
                {
                    "symbol": ["000001", "000002"],
                    "bucket_id": [7, 7],
                    "source_path": ["/tmp/a.parquet", "/tmp/b.parquet"],
                    "source_size": [1, 1],
                    "source_mtime_ns": [1, 1],
                    "row_count": [2, 1],
                    "min_date": ["2024-01-02", "2024-01-03"],
                    "max_date": ["2024-01-03", "2024-01-03"],
                }
            )
            manifest.to_parquet(root / "manifest.parquet", index=False)
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "storage_layout": "bucket_shards",
                        "bucket_count": 128,
                        "bucket_ids": [7],
                    },
                    f,
                )
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
                    "symbol": ["000001", "000001", "000002"],
                    "close": [10.0, 11.0, 20.0],
                    "open": [9.5, 10.5, 19.5],
                }
            ).to_parquet(buckets / "part-0007.parquet", index=False)

            meta = load_source_store_metadata(root)
            loaded = load_source_frame(
                store_dir=root,
                columns=["close"],
                date_start="2024-01-03",
                symbols=["000002"],
            )

            self.assertEqual(meta["storage_layout"], "bucket_shards")
            self.assertEqual(detect_source_storage_layout(root), "bucket_shards")
            self.assertEqual(list(loaded.columns), ["date", "symbol", "close"])
            self.assertEqual(loaded["symbol"].tolist(), ["000002"])
            self.assertEqual(loaded["close"].tolist(), [20.0])

    def test_load_source_frame_symbol_shards_require_date(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pd.DataFrame({"symbol": ["000001"], "close": [10.0]}).to_parquet(root / "000001.parquet", index=False)

            with self.assertRaisesRegex(ValueError, "missing required column 'date'"):
                load_source_frame(
                    store_dir=root,
                    columns=["close"],
                    symbols=["000001"],
                )

    def test_load_source_frame_symbol_shards_require_requested_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02"]),
                    "symbol": ["000001"],
                    "open": [9.5],
                }
            ).to_parquet(root / "000001.parquet", index=False)

            with self.assertRaisesRegex(ValueError, "missing requested columns"):
                load_source_frame(
                    store_dir=root,
                    columns=["close"],
                    symbols=["000001"],
                )


if __name__ == "__main__":
    unittest.main()
