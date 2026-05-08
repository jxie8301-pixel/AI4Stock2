"""Compatibility wrapper for the Rust quality/event/flow diagnostics preset."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def _rust_binary_command() -> list[str]:
    env_value = os.environ.get("AI4STOCK_DIAGNOSTICS_BIN")
    if env_value:
        return shlex.split(env_value)
    return ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]


def build_delegated_command(argv: list[str] | None = None) -> list[str]:
    return [*_rust_binary_command(), "quality-event-flow-single-factor", *(argv if argv is not None else sys.argv[1:])]


def main() -> None:
    completed = subprocess.run(build_delegated_command(), check=False)
    raise SystemExit(int(completed.returncode))


if __name__ == "__main__":
    main()
