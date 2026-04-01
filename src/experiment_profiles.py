"""Experiment-profile resolution for native training/backtest pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config_utils import deep_update, load_yaml_file


DEFAULT_EXPERIMENT_PROFILE_CONFIG_PATH = "configs/experiment_profiles.yaml"


def load_experiment_profiles(config_path: str = DEFAULT_EXPERIMENT_PROFILE_CONFIG_PATH) -> dict[str, Any]:
    data = load_yaml_file(config_path)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No experiment profiles found in {config_path}")
    return data


def resolve_experiment_profile(
    cfg: dict[str, Any] | None = None,
    *,
    profile_name: str | None = None,
    profile_config_path: str = DEFAULT_EXPERIMENT_PROFILE_CONFIG_PATH,
) -> dict[str, Any]:
    cfg = cfg or {}
    profile_index = load_experiment_profiles(profile_config_path)
    profiles = profile_index["profiles"]

    experiment_cfg = cfg.get("experiment", {})
    resolved_name = profile_name or experiment_cfg.get("profile")
    if not resolved_name:
        raise ValueError(
            "Experiment profile must be specified explicitly. "
            "Pass --experiment-profile or set experiment.profile in the loaded config."
        )
    if resolved_name not in profiles:
        raise ValueError(
            f"Unknown experiment profile: {resolved_name}. "
            f"Available: {', '.join(sorted(profiles))}"
        )

    profile_cfg = _resolve_profile_definition(
        resolved_name,
        profiles=profiles,
        profile_config_path=profile_config_path,
    )
    config = deepcopy(profile_cfg)
    config.pop("name", None)
    resolved_path = str(config.pop("path"))
    sweep = deepcopy(config.pop("sweep", {}))
    return {
        "name": resolved_name,
        "path": resolved_path,
        "config": config,
        "sweep": sweep,
        "raw": deepcopy(profile_cfg),
    }


def _build_inline_profile_path(profile_name: str, profile_config_path: str) -> str:
    return f"{Path(profile_config_path).resolve()}::{profile_name}"


def _resolve_profile_definition(
    profile_name: str,
    *,
    profiles: dict[str, Any],
    profile_config_path: str,
    stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if profile_name in stack:
        cycle = " -> ".join([*stack, profile_name])
        raise ValueError(f"Experiment profile inheritance cycle detected: {cycle}")
    if profile_name not in profiles:
        raise ValueError(f"Unknown experiment profile: {profile_name}")

    profile_entry = deepcopy(profiles[profile_name])
    source_path = _build_inline_profile_path(profile_name, profile_config_path)
    if "path" not in profile_entry:
        loaded_profile = {}
    else:
        repo_root = Path(profile_config_path).resolve().parent.parent
        profile_path = Path(profile_entry.pop("path"))
        if not profile_path.is_absolute():
            profile_path = repo_root / profile_path
        loaded_profile = load_yaml_file(profile_path)
        source_path = str(profile_path)

    extends_name = str(profile_entry.pop("extends", "") or "").strip()
    merged = deepcopy(loaded_profile)
    deep_update(merged, profile_entry)
    if extends_name:
        parent_profile = _resolve_profile_definition(
            extends_name,
            profiles=profiles,
            profile_config_path=profile_config_path,
            stack=(*stack, profile_name),
        )
        parent_profile = deepcopy(parent_profile)
        parent_profile.pop("name", None)
        parent_profile.pop("path", None)
        merged_profile = parent_profile
        deep_update(merged_profile, merged)
        merged = merged_profile

    merged["name"] = profile_name
    merged["path"] = source_path
    return merged
