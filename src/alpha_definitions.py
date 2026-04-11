"""Backward-compatible wrapper around unified feature module."""

from src.gen_feature import (  # noqa: F401
    DEFAULT_ALPHA158_CONFIG,
    get_alpha158_feature_config,
    get_alpha360_feature_config,
    validate_default_dimensions,
)


if __name__ == "__main__":
    dims = validate_default_dimensions()
    print(f"Alpha158 fields: {dims['alpha158']}")
    print(f"Alpha360 fields: {dims['alpha360']}")
