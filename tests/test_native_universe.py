import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.native_universe import build_universe_frame_mask, build_universe_mask, load_universe_table, resolve_universe_path


class NativeUniverseTest(unittest.TestCase):
    def test_build_universe_mask_respects_membership_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            universe_path = Path(tmp_dir) / "demo.txt"
            universe_path.write_text(
                "000001\t2024-01-01\t2024-01-31\n"
                "000002\t2024-02-01\t2024-02-29\n",
                encoding="utf-8",
            )

            dates = np.array(
                [
                    np.datetime64("2024-01-15", "ns").astype("int64"),
                    np.datetime64("2024-02-15", "ns").astype("int64"),
                    np.datetime64("2024-01-15", "ns").astype("int64"),
                    np.datetime64("2024-02-15", "ns").astype("int64"),
                ],
                dtype=np.int64,
            )
            symbol_ids = np.array([0, 0, 1, 1], dtype=np.int32)
            symbol_to_id = {"000001": 0, "000002": 1}

            mask = build_universe_mask(
                dates_ns=dates,
                symbol_ids=symbol_ids,
                symbol_to_id=symbol_to_id,
                universe_name="demo",
                universe_dir=tmp_dir,
            )

            self.assertEqual(mask.tolist(), [True, False, False, True])

    def test_resolve_universe_path_prefers_native_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            universe_path = Path(tmp_dir) / "csi300.txt"
            universe_path.write_text("000001\n", encoding="utf-8")

            resolved = resolve_universe_path("csi300", universe_dir=tmp_dir)

            self.assertEqual(resolved, universe_path)

    def test_build_universe_frame_mask_respects_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            universe_path = Path(tmp_dir) / "demo.txt"
            universe_path.write_text("000001\t2024-01-01\t2024-01-31\n", encoding="utf-8")

            mask = build_universe_frame_mask(
                dates=pd.to_datetime(["2024-01-15", "2024-02-01"]),
                symbols=["000001", "000001"],
                universe_name="demo",
                universe_dir=tmp_dir,
            )

            self.assertEqual(mask.tolist(), [True, False])

    def test_one_column_universe_uses_unbounded_date_interval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            universe_path = Path(tmp_dir) / "demo.txt"
            universe_path.write_text("000001\n", encoding="utf-8")

            table = load_universe_table("demo", universe_dir=tmp_dir)
            mask = build_universe_frame_mask(
                dates=pd.to_datetime(["1800-01-01", "2024-01-15", "2262-04-11"]),
                symbols=["000001", "000001", "000001"],
                universe_name="demo",
                universe_dir=tmp_dir,
            )

            self.assertTrue(pd.isna(table.loc[0, "start_date"]))
            self.assertTrue(pd.isna(table.loc[0, "end_date"]))
            self.assertEqual(mask.tolist(), [True, True, True])

    def test_open_ended_universe_bounds_match_before_and_after_dates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            universe_path = Path(tmp_dir) / "demo.txt"
            universe_path.write_text(
                "000001\t\t2024-01-31\n"
                "000002\t2024-02-01\t\n",
                encoding="utf-8",
            )

            mask = build_universe_frame_mask(
                dates=pd.to_datetime(["2023-12-31", "2024-02-01", "2024-01-31", "2024-03-01"]),
                symbols=["000001", "000001", "000002", "000002"],
                universe_name="demo",
                universe_dir=tmp_dir,
            )

            self.assertEqual(mask.tolist(), [True, False, False, True])


if __name__ == "__main__":
    unittest.main()
