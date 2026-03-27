import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.build_universes import _normalize_symbol, build_universe_file


class BuildUniversesTest(unittest.TestCase):
    def test_normalize_symbol_extracts_last_six_digits(self):
        self.assertEqual(_normalize_symbol("sh600000"), "600000")
        self.assertEqual(_normalize_symbol("000001.SZ"), "000001")
        self.assertEqual(_normalize_symbol(300750), "300750")

    def test_build_universe_file_writes_tab_separated_membership(self):
        sample = pd.DataFrame(
            {
                "symbol": ["000001", "600000"],
                "start_date": ["2005-01-01", "2010-01-01"],
                "end_date": ["2099-12-31", "2099-12-31"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("src.build_universes.fetch_universe_members", return_value=sample):
                path = build_universe_file("csi300", output_dir=tmp_dir)

            self.assertEqual(path, Path(tmp_dir) / "csi300.txt")
            content = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(content[0], "000001\t2005-01-01\t2099-12-31")
            self.assertEqual(content[1], "600000\t2010-01-01\t2099-12-31")


if __name__ == "__main__":
    unittest.main()
