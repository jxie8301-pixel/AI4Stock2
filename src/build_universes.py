"""Build native universe membership files from index constituents."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_UNIVERSE_DIR = Path("data/universes")
INDEX_CODE_BY_UNIVERSE = {
    "csi300": "000300",
    "csi500": "000905",
    "zz1000": "000852",
}


def _clear_proxy_env() -> None:
    for key in [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ]:
        os.environ.pop(key, None)


def _normalize_symbol(value: object) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits


def fetch_universe_members(index_code: str, *, allow_static_membership: bool = False) -> pd.DataFrame:
    import akshare as ak

    df = ak.index_stock_cons_csindex(symbol=index_code)
    if df is None or df.empty:
        raise ValueError(f"No constituents returned for index {index_code}")

    symbol_col = None
    for candidate in ["成分券代码", "指数代码", "stock_code", "品种代码", "证券代码"]:
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        raise ValueError(f"Could not find symbol column in constituent table: {list(df.columns)}")

    start_col = None
    end_col = None
    for candidate in ["纳入日期", "开始日期", "start_date"]:
        if candidate in df.columns:
            start_col = candidate
            break
    for candidate in ["剔除日期", "结束日期", "end_date"]:
        if candidate in df.columns:
            end_col = candidate
            break
    if (start_col is None or end_col is None) and not allow_static_membership:
        raise ValueError(
            "Index constituent table does not expose point-in-time membership intervals. "
            "Refusing to build a universe file that would silently use static/current membership. "
            "Pass --allow-static-membership only for explicitly labelled research controls."
        )

    out = pd.DataFrame()
    out["symbol"] = df[symbol_col].map(_normalize_symbol)
    out = out[out["symbol"] != ""].copy()
    out["start_date"] = (
        pd.to_datetime(df[start_col], errors="coerce").fillna(pd.Timestamp("2005-01-01"))
        if start_col is not None
        else pd.Timestamp("2005-01-01")
    )
    out["end_date"] = (
        pd.to_datetime(df[end_col], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
        if end_col is not None
        else pd.Timestamp("2099-12-31")
    )
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.strftime("%Y-%m-%d")
    out["end_date"] = pd.to_datetime(out["end_date"]).dt.strftime("%Y-%m-%d")
    return out.sort_values(["symbol", "start_date", "end_date"]).drop_duplicates().reset_index(drop=True)


def build_universe_file(
    universe_name: str,
    output_dir: str | Path = DEFAULT_UNIVERSE_DIR,
    *,
    allow_static_membership: bool = False,
) -> Path:
    if universe_name not in INDEX_CODE_BY_UNIVERSE:
        raise ValueError(
            f"Unknown universe '{universe_name}'. Available: {', '.join(sorted(INDEX_CODE_BY_UNIVERSE))}"
        )

    _clear_proxy_env()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    index_code = INDEX_CODE_BY_UNIVERSE[universe_name]
    table = fetch_universe_members(index_code, allow_static_membership=allow_static_membership)
    output_path = output_root / f"{universe_name}.txt"
    table.to_csv(output_path, sep="\t", index=False, header=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build native universe files from index constituents.")
    parser.add_argument(
        "--universes",
        default="csi300,csi500,zz1000",
        help="Comma-separated universe names to build (default: csi300,csi500,zz1000).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_UNIVERSE_DIR), help="Universe output directory.")
    parser.add_argument(
        "--allow-static-membership",
        action="store_true",
        help="Allow fallback wide membership intervals when the source endpoint lacks PIT start/end dates.",
    )
    return parser.parse_args()


def _rust_collector_command() -> list[str]:
    env_value = os.environ.get("AI4STOCK_COLLECT_BIN")
    if env_value:
        return shlex.split(env_value)
    return ["cargo", "run", "--bin", "ai4stock-collect", "--"]


def _rust_collector_env() -> dict[str, str]:
    env = os.environ.copy()
    pixi_lib = Path(".pixi/envs/default/lib")
    if pixi_lib.exists():
        current = env.get("LD_LIBRARY_PATH", "")
        parts = [str(pixi_lib), *([current] if current else [])]
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def main() -> None:
    if os.environ.get("AI4STOCK_PY_BUILD_UNIVERSES_RUNTIME", "").strip() != "1":
        command = _rust_collector_command()
        command.extend(["universes", *sys.argv[1:]])
        print(f"[run] {shlex.join(command)}", flush=True)
        completed = subprocess.run(command, check=False, env=_rust_collector_env())
        raise SystemExit(int(completed.returncode))

    args = _parse_args()
    universe_names = [name.strip() for name in args.universes.split(",") if name.strip()]
    for universe_name in universe_names:
        output_path = build_universe_file(
            universe_name,
            output_dir=args.output_dir,
            allow_static_membership=bool(args.allow_static_membership),
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
