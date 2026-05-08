"""Compatibility wrapper for the Rust rolling LightGBM pipeline."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def _rust_binary_command() -> list[str]:
    env_value = os.environ.get("AI4STOCK_TRAIN_BIN")
    if env_value:
        return shlex.split(env_value)
    return ["cargo", "run", "--bin", "ai4stock-train", "--"]


def build_delegated_command(argv: list[str] | None = None) -> list[str]:
    return [*_rust_binary_command(), "rolling-lgbm", *(argv if argv is not None else sys.argv[1:])]


def run_rolling_pipeline(argv: list[str] | None = None) -> None:
    completed = subprocess.run(build_delegated_command(argv), check=False)
    raise SystemExit(int(completed.returncode))


if __name__ == "__main__":
    try:
        run_rolling_pipeline()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
