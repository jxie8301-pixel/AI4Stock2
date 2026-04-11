"""Backward-compatible wrapper around unified feature module."""

from src.gen_feature import (  # noqa: F401
    compute_alpha158,
    compute_alpha360,
)


if __name__ == "__main__":
    import pandas as pd

    sample = pd.read_parquet("data/processed/combined/000001.parquet")
    a158 = compute_alpha158(sample)
    a360 = compute_alpha360(sample)
    print(f"alpha158 shape: {a158.shape}")
    print(f"alpha360 shape: {a360.shape}")
