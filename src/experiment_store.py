"""Local experiment/model store helpers for reproducible comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml

from src.data_source import resolve_data_source_name
from src.label_utils import DEFAULT_SIGNAL_HORIZON, resolve_signal_horizon


SUMMARY_FIELDS = [
    "created_at",
    "run_id",
    "backend",
    "pipeline",
    "model",
    "data_source",
    "experiment_profile",
    "model_profile",
    "run_tag",
    "feature_profile",
    "universe",
    "topk",
    "n_drop",
    "weighting",
    "max_weight",
    "rebalance_freq",
    "retrain_step",
    "signal_horizon",
    "train_days",
    "valid_days",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
    "selected_feature_count",
    "signal_ic_mean",
    "signal_icir",
    "signal_rank_ic_mean",
    "signal_rank_icir",
    "portfolio_annualized_return",
    "portfolio_information_ratio",
    "portfolio_max_drawdown",
    "results_dir",
    "archived_artifacts_dir",
    "model_source_path",
    "archived_model_path",
    "models_source_dir",
    "archived_models_dir",
    "load_model_path",
    "feature_importance_path",
    "training_summary_path",
]


@dataclass(frozen=True)
class RunStore:
    enabled: bool
    root_dir: Path | None = None
    run_id: str | None = None
    run_dir: Path | None = None
    artifacts_dir: Path | None = None
    models_dir: Path | None = None
    default_model_path: Path | None = None


def _slugify(value: str | None) -> str:
    if not value:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _copy_file_if_needed(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _copy_tree_contents(source_dir: Path, target_dir: Path, *, exclude_names: set[str] | None = None) -> None:
    exclude_names = exclude_names or set()
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        if item.name in exclude_names:
            continue
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_rebalance_freq(cfg: dict, args) -> int:
    return getattr(args, "rebalance_freq", None) or cfg.get("backtest", {}).get("rebalance_freq", 5)


def resolve_retrain_step(cfg: dict, args) -> int:
    retrain_step = getattr(args, "retrain_step", None)
    return int(retrain_step or cfg.get("rolling", {}).get("retrain_step", 10))


def resolve_signal_horizon_for_run(cfg: dict, args) -> int:
    signal_horizon = getattr(args, "signal_horizon", None)
    if signal_horizon is None:
        signal_horizon = getattr(args, "label_horizon", None)
    if signal_horizon is not None:
        return int(signal_horizon)
    return int(resolve_signal_horizon(cfg) or DEFAULT_SIGNAL_HORIZON)


def resolve_feature_profile_name(cfg: dict, args) -> str:
    profile = (
        getattr(args, "feature_profile", None)
        or getattr(args, "profile", None)
        or cfg.get("features", {}).get("profile")
    )
    return str(profile or "")


def resolve_model_profile_name(cfg: dict, args) -> str:
    profile = getattr(args, "model_profile", None) or cfg.get("model", {}).get("profile")
    return str(profile or "")


def resolve_experiment_profile_name(cfg: dict, args) -> str:
    profile = getattr(args, "experiment_profile", None) or cfg.get("experiment", {}).get("profile")
    return str(profile or "")


def prepare_run_store(
    cfg: dict,
    args,
    *,
    backend: str,
    pipeline: str,
    model_name: str,
    model_ext: str,
) -> RunStore:
    artifacts_cfg = cfg.get("artifacts", {})
    enabled = artifacts_cfg.get("enable_local_store", True) and not getattr(args, "disable_local_store", False)
    if not enabled:
        return RunStore(enabled=False)

    root_dir = Path(getattr(args, "store_dir", None) or artifacts_cfg.get("store_dir", "results/experiments"))
    created_at = datetime.now()
    strategy_slug = (
        f"top{cfg.get('strategy', {}).get('topk', 'na')}"
        f"_drop{cfg.get('strategy', {}).get('n_drop', 'na')}"
        f"_w{cfg.get('strategy', {}).get('weighting', 'equal')}"
        f"_reb{resolve_rebalance_freq(cfg, args)}"
    )
    tag_slug = _slugify(getattr(args, "run_tag", None))
    experiment_slug = _slugify(resolve_experiment_profile_name(cfg, args))
    profile_slug = _slugify(resolve_feature_profile_name(cfg, args))
    run_id_parts = [
        created_at.strftime("%Y%m%d_%H%M%S"),
        backend,
        pipeline,
        model_name,
        strategy_slug,
    ]
    if not tag_slug and experiment_slug:
        run_id_parts.append(experiment_slug)
    elif not tag_slug and profile_slug:
        run_id_parts.append(profile_slug)
    if tag_slug:
        run_id_parts.append(tag_slug)
    run_id = "__".join(run_id_parts)

    run_dir = root_dir / backend / pipeline / model_name / run_id
    artifacts_dir = run_dir
    models_dir = run_dir / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return RunStore(
        enabled=True,
        root_dir=root_dir,
        run_id=run_id,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        default_model_path=models_dir / f"{model_name}{model_ext}",
    )


def finalize_run_store(
    store: RunStore,
    *,
    cfg: dict,
    args,
    backend: str,
    pipeline: str,
    model_name: str,
    results_dir: str | Path,
    signal_metrics: dict | None = None,
    portfolio_metrics: dict | None = None,
    model_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    load_model_path: str | Path | None = None,
    extra_context: dict | None = None,
) -> Path | None:
    if not store.enabled or not store.run_dir or not store.root_dir or not store.artifacts_dir or not store.models_dir:
        return None

    results_dir = Path(results_dir)
    if results_dir.exists() and results_dir.resolve() != store.artifacts_dir.resolve():
        _copy_tree_contents(results_dir, store.artifacts_dir, exclude_names={"models"})

    archived_model_path = None
    if model_path:
        source_model_path = Path(model_path)
        if source_model_path.exists():
            archived_model_path = _copy_file_if_needed(source_model_path, store.models_dir / source_model_path.name)

    archived_models_dir = None
    if models_dir:
        source_models_dir = Path(models_dir)
        if source_models_dir.exists() and source_models_dir.resolve() != store.models_dir.resolve():
            _copy_tree_contents(source_models_dir, store.models_dir)
            archived_models_dir = store.models_dir
        elif source_models_dir.exists():
            archived_models_dir = store.models_dir

    config_snapshot_path = store.run_dir / "config_snapshot.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    extra_context = extra_context or {}
    flat_metrics = flatten_metrics(signal_metrics=signal_metrics, portfolio_metrics=portfolio_metrics)
    summary_row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": store.run_id,
        "backend": backend,
        "pipeline": pipeline,
        "model": model_name,
        "data_source": resolve_data_source_name(cfg),
        "experiment_profile": resolve_experiment_profile_name(cfg, args),
        "model_profile": resolve_model_profile_name(cfg, args),
        "run_tag": getattr(args, "run_tag", None) or "",
        "feature_profile": resolve_feature_profile_name(cfg, args),
        "universe": cfg.get("universe", ""),
        "topk": cfg.get("strategy", {}).get("topk"),
        "n_drop": cfg.get("strategy", {}).get("n_drop"),
        "weighting": cfg.get("strategy", {}).get("weighting", "equal"),
        "max_weight": cfg.get("strategy", {}).get("max_weight", ""),
        "rebalance_freq": resolve_rebalance_freq(cfg, args),
        "retrain_step": extra_context.get("retrain_step", resolve_retrain_step(cfg, args)),
        "signal_horizon": extra_context.get("signal_horizon", resolve_signal_horizon_for_run(cfg, args)),
        "train_days": extra_context.get("train_days", ""),
        "valid_days": extra_context.get("valid_days", ""),
        "train_start": extra_context.get("train_start", cfg.get("time", {}).get("train", ["", ""])[0]),
        "train_end": extra_context.get("train_end", cfg.get("time", {}).get("train", ["", ""])[1]),
        "valid_start": extra_context.get("valid_start", cfg.get("time", {}).get("valid", ["", ""])[0]),
        "valid_end": extra_context.get("valid_end", cfg.get("time", {}).get("valid", ["", ""])[1]),
        "test_start": extra_context.get("test_start", cfg.get("time", {}).get("test", ["", ""])[0]),
        "test_end": extra_context.get("test_end", cfg.get("time", {}).get("test", ["", ""])[1]),
        "selected_feature_count": len(extra_context.get("selected_features", [])),
        "results_dir": str(results_dir),
        "archived_artifacts_dir": str(store.artifacts_dir),
        "model_source_path": str(model_path) if model_path else "",
        "archived_model_path": str(archived_model_path) if archived_model_path else "",
        "models_source_dir": str(models_dir) if models_dir else "",
        "archived_models_dir": str(archived_models_dir) if archived_models_dir else "",
        "load_model_path": str(load_model_path) if load_model_path else "",
        "feature_importance_path": str(extra_context.get("feature_importance_path", "")),
        "training_summary_path": str(extra_context.get("training_summary_path", "")),
        **flat_metrics,
    }

    manifest = {
        "summary": summary_row,
        "config_snapshot_path": str(config_snapshot_path),
        "args": vars(args),
        "signal_metrics": signal_metrics or {},
        "portfolio_metrics": portfolio_metrics or {},
        "extra_context": extra_context,
    }
    manifest_path = store.run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    append_summary_row(store.root_dir / "experiment_index.csv", summary_row)
    return manifest_path


def flatten_metrics(signal_metrics: dict | None, portfolio_metrics: dict | None) -> dict:
    signal_metrics = signal_metrics or {}
    portfolio_metrics = portfolio_metrics or {}
    return {
        "signal_ic_mean": _safe_float(signal_metrics.get("IC_mean")),
        "signal_icir": _safe_float(signal_metrics.get("ICIR")),
        "signal_rank_ic_mean": _safe_float(signal_metrics.get("Rank_IC_mean")),
        "signal_rank_icir": _safe_float(signal_metrics.get("Rank_ICIR")),
        "portfolio_annualized_return": _safe_float(
            portfolio_metrics.get("annualized_return", {}).get("risk")
        ),
        "portfolio_information_ratio": _safe_float(
            portfolio_metrics.get("information_ratio", {}).get("risk")
        ),
        "portfolio_max_drawdown": _safe_float(
            portfolio_metrics.get("max_drawdown", {}).get("risk")
        ),
    }


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_row = {field: row.get(field, "") for field in SUMMARY_FIELDS}

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(normalized_row)
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    if existing_fields != SUMMARY_FIELDS:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow({field: existing_row.get(field, "") for field in SUMMARY_FIELDS})
            writer.writerow(normalized_row)
        return

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writerow(normalized_row)
