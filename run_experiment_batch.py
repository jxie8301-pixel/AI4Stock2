"""Batch runner for native single/rolling experiments with sweep expansion."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.config_loader import load_runtime_config
from src.experiment_profiles import resolve_experiment_profile
from src.override_utils import (
    build_override_tag,
    expand_sweep_grid,
    flatten_sweep_mapping,
    parse_override_arg,
    parse_sweep_arg,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native experiments in batch from sweep definitions.")
    parser.add_argument("--config", default="configs/config.yaml", help="Runtime config path")
    parser.add_argument("--pipeline", choices=["single", "rolling"], default="rolling", help="Target pipeline")
    parser.add_argument("--experiment-profile", required=True, help="Base experiment profile")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--feature-profile", help="Override feature profile")
    parser.add_argument("--data-source", choices=["akshare", "gm", "tushare"], help="Override data source")
    parser.add_argument(
        "--set",
        action="append",
        dest="set_overrides",
        help="Fixed dotted override in key=value form, applied to every run",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        nargs="+",
        dest="sweep_overrides",
        help="Sweep override in key=[a,b,c] form. Quote it in zsh to avoid shell glob expansion.",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs="+",
        dest="case_overrides",
        help=(
            "Explicit grouped overrides for one run. "
            "Example: --case strategy.topk=5 strategy.n_drop=1 "
            "--case strategy.topk=10 strategy.n_drop=2"
        ),
    )
    parser.add_argument("--run-tag-prefix", help="Optional run-tag prefix added before per-run sweep suffix")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded commands without executing them")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed child run")
    return parser


def _build_base_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    target_script = repo_root / ("main.py" if args.pipeline == "single" else "run_native_rolling.py")
    cmd = [sys.executable, str(target_script), "--config", args.config, "--experiment-profile", args.experiment_profile]
    if args.model_profile:
        cmd += ["--model-profile", args.model_profile]
    if args.feature_profile:
        cmd += ["--feature-profile", args.feature_profile]
    if args.data_source:
        cmd += ["--data-source", args.data_source]
    for item in args.set_overrides or []:
        cmd += ["--set", item]
    return cmd


def _resolve_sweep_map(args: argparse.Namespace) -> dict[str, list[object]]:
    runtime_cfg = load_runtime_config(args.config)
    experiment_profile = resolve_experiment_profile(runtime_cfg, profile_name=args.experiment_profile)
    sweep_map = flatten_sweep_mapping(experiment_profile.get("sweep", {}) or {})
    raw_sweeps = [item for group in (args.sweep_overrides or []) for item in group]
    grouped_cli_sweeps: dict[str, list[object]] = {}
    for raw in raw_sweeps:
        key, values = parse_sweep_arg(raw)
        grouped_cli_sweeps.setdefault(key, []).extend(values)
    for key, values in grouped_cli_sweeps.items():
        sweep_map[key] = values
    return sweep_map


def _resolve_explicit_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for group in args.case_overrides or []:
        case: dict[str, object] = {}
        for raw in group:
            key, value = parse_override_arg(raw)
            case[key] = value
        if case:
            cases.append(case)
    return cases


def _build_run_tag(prefix: str | None, overrides: dict[str, object]) -> str | None:
    suffix = build_override_tag(overrides)
    if prefix and suffix:
        return f"{prefix}__{suffix}"
    if prefix:
        return prefix
    if suffix:
        return suffix
    return None


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    explicit_cases = _resolve_explicit_cases(args)
    if explicit_cases:
        sweep_map: dict[str, list[object]] = {}
        runs = explicit_cases
    else:
        sweep_map = _resolve_sweep_map(args)
        runs = expand_sweep_grid(sweep_map)
    base_cmd = _build_base_command(args)

    print(f"[*] Batch pipeline: {args.pipeline}")
    print(f"[*] Base experiment profile: {args.experiment_profile}")
    print(f"[*] Expanded runs: {len(runs)}")
    if explicit_cases:
        print("[*] Explicit cases:")
        for idx, case in enumerate(explicit_cases, start=1):
            rendered = ", ".join(f"{key}={value}" for key, value in sorted(case.items()))
            print(f"    case {idx}: {rendered}")
    elif sweep_map:
        for key, values in sweep_map.items():
            print(f"    {key} -> {values}")

    failures: list[tuple[int, int]] = []
    for idx, overrides in enumerate(runs, start=1):
        cmd = list(base_cmd)
        for key, value in overrides.items():
            cmd += ["--set", f"{key}={value}"]
        run_tag = _build_run_tag(args.run_tag_prefix, overrides)
        if run_tag:
            cmd += ["--run-tag", run_tag]

        rendered = " ".join(subprocess.list2cmdline([part]) if " " in part else part for part in cmd)
        print(f"\n[{idx}/{len(runs)}] {rendered}")
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures.append((idx, completed.returncode))
            if args.fail_fast:
                raise SystemExit(completed.returncode)

    if args.dry_run:
        print("\n[+] Dry run completed.")
        return
    if failures:
        print("\n[!] Batch completed with failures:")
        for idx, returncode in failures:
            print(f"    run {idx}: exit_code={returncode}")
        raise SystemExit(1)
    print("\n[+] Batch completed successfully.")


if __name__ == "__main__":
    main()
