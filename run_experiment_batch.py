"""Batch runner for native single/rolling experiments with sweep expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from src.config_loader import load_config, load_runtime_config
from src.experiment_profiles import resolve_experiment_profile
from src.experiment_store import _slugify
from src.model_config import get_lgbm_config
from src.override_utils import (
    apply_dotted_override,
    build_override_tag,
    expand_sweep_grid,
    flatten_sweep_mapping,
    parse_override_arg,
    parse_sweep_arg,
)
from src.rolling_types import PREDICTION_ARTIFACT_DIRNAME, PREDICTION_METADATA_FILENAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native experiments in batch from sweep definitions.")
    parser.add_argument("--config", default="configs/config.yaml", help="Runtime config path")
    parser.add_argument("--pipeline", choices=["rolling"], default="rolling", help="Target pipeline")
    parser.add_argument("--experiment-profile", required=True, help="Base experiment profile")
    parser.add_argument("--model-profile", help="Override model profile")
    parser.add_argument("--feature-profile", help="Override feature profile")
    parser.add_argument("--data-source", choices=["akshare", "tushare"], help="Override data source")
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
    parser.add_argument("--store-dir", help="Override local experiment store root")
    parser.add_argument(
        "--dedupe-predictions",
        action="store_true",
        help=(
            "Train once for runs with identical prediction-producing config, "
            "then replay compatible later runs from the saved prediction bundle."
        ),
    )
    parser.add_argument(
        "--skip-reference-baselines",
        action="store_true",
        help="Forward --skip-reference-baselines to child rolling runs for training-speed batches.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print expanded commands without executing them")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed child run")
    return parser


def _build_base_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parent
    target_script = repo_root / "run_native_rolling.py"
    cmd = [sys.executable, str(target_script), "--config", args.config, "--experiment-profile", args.experiment_profile]
    if args.model_profile:
        cmd += ["--model-profile", args.model_profile]
    if args.feature_profile:
        cmd += ["--feature-profile", args.feature_profile]
    if args.data_source:
        cmd += ["--data-source", args.data_source]
    if args.store_dir:
        cmd += ["--store-dir", args.store_dir]
    if getattr(args, "skip_reference_baselines", False):
        cmd += ["--skip-reference-baselines"]
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


def _resolve_run_config(args: argparse.Namespace, overrides: dict[str, object]) -> dict:
    cfg = load_config(
        args.config,
        experiment_profile_name=args.experiment_profile,
        model_profile_name=args.model_profile,
    )
    if args.feature_profile:
        cfg.setdefault("features", {})
        cfg["features"]["profile"] = args.feature_profile
    if args.data_source:
        cfg.setdefault("data", {})
        cfg["data"]["source"] = args.data_source
    if args.store_dir:
        cfg.setdefault("artifacts", {})
        cfg["artifacts"]["store_dir"] = args.store_dir
    for raw in args.set_overrides or []:
        key, value = parse_override_arg(raw)
        apply_dotted_override(cfg, key, value)
    for key, value in overrides.items():
        apply_dotted_override(cfg, key, value)
    return cfg


def _prediction_fingerprint(cfg: dict) -> str:
    """Hash the parts of config that can change generated prediction artifacts."""
    model_name = str(cfg.get("model", {}).get("name", "") or "").strip().lower()
    relevant = {
        "data": cfg.get("data", {}),
        "native": cfg.get("native", {}),
        "universe": cfg.get("universe", ""),
        "time": cfg.get("time", {}),
        "features": cfg.get("features", {}),
        "model": cfg.get("model", {}),
        "label": cfg.get("label", {}),
        "rolling": cfg.get("rolling", {}),
        "prediction_fusion": cfg.get("prediction_fusion", {}),
        "score_fusion": cfg.get("score_fusion", {}),
        "backtest_benchmark": (cfg.get("backtest", {}) or {}).get("benchmark"),
    }
    if model_name == "lgbm":
        relevant["effective_lgbm"] = get_lgbm_config(cfg)
    elif model_name == "formula_score":
        relevant["formula_score"] = cfg.get("formula_score", {})

    encoded = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_run_command(
    base_cmd: list[str],
    *,
    overrides: dict[str, object],
    run_tag: str | None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = list(base_cmd)
    for key, value in overrides.items():
        cmd += ["--set", f"{key}={value}"]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def _render_command(cmd: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) if " " in part else part for part in cmd)


def _resolve_store_dir(cfg: dict, args: argparse.Namespace) -> Path:
    return Path(args.store_dir or cfg.get("artifacts", {}).get("store_dir", "results/experiments"))


def _find_prediction_artifact_dir(
    *,
    cfg: dict,
    args: argparse.Namespace,
    run_tag: str | None,
    started_at: float,
) -> Path:
    model_name = str(cfg.get("model", {}).get("name", "") or "").strip() or "lgbm"
    run_root = _resolve_store_dir(cfg, args) / "native" / "rolling" / model_name
    if not run_root.exists():
        raise FileNotFoundError(f"Run store directory not found after training: {run_root}")

    suffix = f"__{_slugify(run_tag)}" if run_tag else ""
    candidates: list[tuple[float, Path]] = []
    for run_dir in run_root.iterdir():
        if not run_dir.is_dir():
            continue
        if suffix and not run_dir.name.endswith(suffix):
            continue
        artifact_dir = run_dir / PREDICTION_ARTIFACT_DIRNAME
        metadata_path = artifact_dir / PREDICTION_METADATA_FILENAME
        if not metadata_path.exists():
            continue
        metadata_mtime = metadata_path.stat().st_mtime
        if metadata_mtime + 1.0 < started_at:
            continue
        candidates.append((metadata_mtime, artifact_dir))
    if not candidates:
        raise FileNotFoundError(
            "Saved prediction artifact not found after training. "
            f"run_root={run_root}, run_tag={run_tag or ''}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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
    prediction_artifacts_by_fingerprint: dict[str, Path] = {}
    for idx, overrides in enumerate(runs, start=1):
        run_tag = _build_run_tag(args.run_tag_prefix, overrides)
        extra_args: list[str] = []
        cfg = None
        fingerprint = ""
        artifact_dir = None
        if args.dedupe_predictions:
            cfg = _resolve_run_config(args, overrides)
            fingerprint = _prediction_fingerprint(cfg)
            artifact_dir = prediction_artifacts_by_fingerprint.get(fingerprint)
            if artifact_dir is None:
                extra_args.append("--save-predictions")
            else:
                extra_args += ["--load-predictions-dir", str(artifact_dir)]

        cmd = _build_run_command(
            base_cmd,
            overrides=overrides,
            run_tag=run_tag,
            extra_args=extra_args,
        )

        rendered = _render_command(cmd)
        print(f"\n[{idx}/{len(runs)}] {rendered}")
        if args.dry_run:
            if args.dedupe_predictions and artifact_dir is None and fingerprint:
                prediction_artifacts_by_fingerprint[fingerprint] = Path(
                    f"<prediction_artifact_from_run_{idx}>"
                )
            continue
        started_at = time.time()
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures.append((idx, completed.returncode))
            if args.fail_fast:
                raise SystemExit(completed.returncode)
        if args.dedupe_predictions and artifact_dir is None and cfg is not None and fingerprint:
            prediction_artifacts_by_fingerprint[fingerprint] = _find_prediction_artifact_dir(
                cfg=cfg,
                args=args,
                run_tag=run_tag,
                started_at=started_at,
            )

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
