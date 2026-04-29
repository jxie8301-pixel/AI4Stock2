import unittest

from src.data_source import (
    get_default_factor_store_dir,
    get_default_parquet_dir,
    normalize_data_source_name,
    resolve_data_source_name,
    resolve_source_parquet_dir,
)


class DataSourceTest(unittest.TestCase):
    def test_default_data_source_is_akshare(self):
        self.assertEqual(resolve_data_source_name({}), "akshare")
        self.assertEqual(resolve_source_parquet_dir({}), "data/processed/combined")

    def test_aliases_normalize_to_akshare(self):
        self.assertEqual(normalize_data_source_name("eastmoney"), "akshare")
        self.assertEqual(normalize_data_source_name("em"), "akshare")

    def test_tushare_defaults_resolve_expected_paths(self):
        cfg = {"data": {"source": "tushare"}}
        self.assertEqual(resolve_data_source_name(cfg), "tushare")
        self.assertEqual(resolve_source_parquet_dir(cfg), "data/tushare/source")
        self.assertEqual(
            get_default_factor_store_dir("tushare", "full_factor_space"),
            "data/factor_store/tushare_full_factor_space",
        )

    def test_gm_is_no_longer_a_supported_data_source(self):
        with self.assertRaisesRegex(ValueError, "Unsupported data source"):
            get_default_parquet_dir("gm")
        with self.assertRaisesRegex(ValueError, "Unsupported data source"):
            get_default_factor_store_dir("gm", "full_factor_space")


if __name__ == "__main__":
    unittest.main()
