"""Research safety checks for diagnostics-driven profile generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def is_training_date_range(cfg: dict[str, Any], date_start: str, date_end: str) -> bool:
    time_cfg = cfg.get("time", {}) if isinstance(cfg, dict) else {}
    train_range = time_cfg.get("train")
    if not isinstance(train_range, (list, tuple)) or len(train_range) != 2:
        return False
    start = pd.Timestamp(date_start)
    end = pd.Timestamp(date_end)
    train_start = pd.Timestamp(train_range[0])
    train_end = pd.Timestamp(train_range[1])
    return bool(start >= train_start and end <= train_end)


def check_config_profile_write_safety(
    cfg: dict[str, Any],
    *,
    date_start: str,
    date_end: str,
    allow_unsafe: bool,
    tool_name: str,
    diagnostics_paths: list[str | Path | None] | tuple[str | Path | None, ...] | None = None,
) -> tuple[bool, str | None]:
    """Validate that a persisted feature profile is built only from training-period evidence."""
    issues: list[str] = []
    if not is_training_date_range(cfg, date_start, date_end):
        issues.append(f"filter date_start={date_start}, date_end={date_end} is outside the training range")

    diagnostics_issues = check_diagnostics_provenance_safety(
        cfg,
        diagnostics_paths=diagnostics_paths or [],
    )
    issues.extend(diagnostics_issues)

    if not issues:
        return True, None

    time_cfg = cfg.get("time", {}) if isinstance(cfg, dict) else {}
    train_range = time_cfg.get("train", ["", ""])
    rendered_issues = "; ".join(issues)
    message = (
        f"{tool_name} refuses to write a config feature profile from unsafe diagnostics evidence. "
        f"Issues: {rendered_issues}. "
        f"Use a range within train={train_range}, or pass --allow-unsafe-profile-write "
        "to record this as research-selection leakage."
    )
    if allow_unsafe:
        return False, message
    raise ValueError(message)


def check_diagnostics_provenance_safety(
    cfg: dict[str, Any],
    *,
    diagnostics_paths: list[str | Path | None] | tuple[str | Path | None, ...],
) -> list[str]:
    """Return unsafe diagnostics provenance reasons for summary/segment artifacts."""
    issues: list[str] = []
    seen_manifests: set[Path] = set()
    for raw_path in diagnostics_paths:
        if raw_path is None or str(raw_path).strip() == "":
            continue
        artifact_path = Path(raw_path)
        manifest_path = (artifact_path.parent / "manifest.json").resolve()
        if manifest_path in seen_manifests:
            continue
        seen_manifests.add(manifest_path)
        if not manifest_path.exists():
            issues.append(f"{artifact_path} has no sibling manifest.json")
            continue
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            issues.append(f"{artifact_path} manifest cannot be read: {type(exc).__name__}: {exc}")
            continue
        metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
        if not isinstance(metadata, dict):
            issues.append(f"{artifact_path} manifest has no metadata object")
            continue
        period = str(metadata.get("period") or "").strip().lower()
        if period != "train":
            issues.append(f"{artifact_path} diagnostics period={period or '<missing>'} is not train")
        manifest_start = metadata.get("date_start")
        manifest_end = metadata.get("date_end")
        if not manifest_start or not manifest_end:
            issues.append(f"{artifact_path} manifest has no date_start/date_end")
        elif not is_training_date_range(cfg, str(manifest_start), str(manifest_end)):
            issues.append(
                f"{artifact_path} diagnostics date_start={manifest_start}, date_end={manifest_end} "
                "is outside the training range"
            )
    return issues
