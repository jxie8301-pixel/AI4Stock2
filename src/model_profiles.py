"""Model-profile resolution for native training pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config_utils import load_yaml_file


DEFAULT_MODEL_PROFILE_CONFIG_PATH = "configs/model_profiles.yaml"


def load_model_profiles(config_path: str = DEFAULT_MODEL_PROFILE_CONFIG_PATH) -> dict[str, Any]:
    data = load_yaml_file(config_path)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No model profiles found in {config_path}")
    return data


def resolve_model_profile(
    cfg: dict[str, Any] | None = None,
    *,
    profile_name: str | None = None,
    profile_config_path: str = DEFAULT_MODEL_PROFILE_CONFIG_PATH,
) -> dict[str, Any]:
    cfg = cfg or {}
    profile_index = load_model_profiles(profile_config_path)
    profiles = profile_index["profiles"]

    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    resolved_name = (
        profile_name
        or model_cfg.get("profile")
        or runtime_cfg.get("default_model_profile")
        or profile_index.get("default_profile")
    )
    if resolved_name not in profiles:
        raise ValueError(
            f"Unknown model profile: {resolved_name}. "
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
