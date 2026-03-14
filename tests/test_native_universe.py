import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.native_universe import build_universe_mask, resolve_universe_path


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
            universe_path = Path(tmp_dir) / "csi300_real.txt"
            universe_path.write_text("000001\n", encoding="utf-8")

            resolved = resolve_universe_path("csi300_real", universe_dir=tmp_dir)

            self.assertEqual(resolved, universe_path)


if __name__ == "__main__":
    unittest.main()
