import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.factor_store import load_available_dates, load_factor_frame, load_factor_store_metadata


class FactorStoreTest(unittest.TestCase):
    def test_load_factor_frame_prunes_columns_and_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shards = root / "shards"
            shards.mkdir(parents=True, exist_ok=True)
            meta = {
                "feature_names": ["F1", "F2"],
                "factor_store_dir": str(root),
            }
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                    "symbol": ["000001", "000001", "000001"],
                    "label": [0.1, 0.2, 0.3],
                    "F1": [1.0, 2.0, 3.0],
                    "F2": [10.0, 20.0, 30.0],
                }
            ).to_parquet(shards / "000001.parquet", index=False)

            loaded = load_factor_frame(
                store_dir=root,
                columns=["F2"],
                date_start="2024-01-03",
                date_end="2024-01-04",
            )

            self.assertEqual(list(loaded.columns), ["date", "symbol", "label", "F2"])
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded["F2"].tolist(), [20.0, 30.0])

    def test_load_available_dates_returns_unique_sorted_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shards = root / "shards"
            shards.mkdir(parents=True, exist_ok=True)
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"feature_names": []}, f)

            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
                    "symbol": ["000001", "000001"],
                    "label": [0.1, 0.2],
                }
            ).to_parquet(shards / "000001.parquet", index=False)

            dates = load_available_dates(store_dir=root)
            self.assertEqual([str(x.date()) for x in dates], ["2024-01-02", "2024-01-03"])

    def test_load_available_dates_prefers_cached_meta_calendar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "shards").mkdir(parents=True, exist_ok=True)
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_names": [],
                        "available_dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    },
                    f,
                )

            dates = load_available_dates(store_dir=root, date_start="2024-01-03", date_end="2024-01-04")
            self.assertEqual([str(x.date()) for x in dates], ["2024-01-03", "2024-01-04"])

    def test_load_available_dates_cached_meta_can_filter_universe_ranges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            universe_dir = root / "universes"
            universe_dir.mkdir(parents=True, exist_ok=True)
            (root / "shards").mkdir(parents=True, exist_ok=True)
            (universe_dir / "demo.txt").write_text("000001\t2024-01-03\t2024-01-03\n", encoding="utf-8")
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_names": [],
                        "available_dates": ["2024-01-02", "2024-01-03", "2024-01-04"],
                    },
                    f,
                )

            dates = load_available_dates(store_dir=root, universe_name="demo", universe_dir=universe_dir)
            self.assertEqual([str(x.date()) for x in dates], ["2024-01-03"])

    def test_load_factor_store_metadata_reads_meta_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            meta = {"feature_names": ["F1"], "num_rows": 10}
            with open(root / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

            loaded_meta = load_factor_store_metadata(root)
            self.assertEqual(loaded_meta["feature_names"], ["F1"])
            self.assertEqual(loaded_meta["num_rows"], 10)


if __name__ == "__main__":
    unittest.main()
