"""Experiment-profile resolution for native training/backtest pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config_utils import load_yaml_file


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

    repo_root = Path(profile_config_path).resolve().parent.parent
    profile_path = Path(profiles[resolved_name]["path"])
    if not profile_path.is_absolute():
        profile_path = repo_root / profile_path

    profile_cfg = load_yaml_file(profile_path)
    return {
        "name": resolved_name,
        "path": str(profile_path),
        "config": deepcopy(profile_cfg),
    }
