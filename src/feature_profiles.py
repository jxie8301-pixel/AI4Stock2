"""Feature-profile resolution for native cache generation/training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROFILE_CONFIG_PATH = "configs/feature_profiles.yaml"


def load_feature_profiles(config_path: str = DEFAULT_PROFILE_CONFIG_PATH) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No feature profiles found in {config_path}")
    return data


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_profile_definition(
    profile_name: str,
    profile_entry: dict[str, Any],
    *,
    profile_config_path: str,
) -> dict[str, Any]:
    if "path" not in profile_entry:
        return deepcopy(profile_entry)

    repo_root = Path(profile_config_path).resolve().parent.parent
    feature_path = Path(profile_entry["path"])
    if not feature_path.is_absolute():
        feature_path = repo_root / feature_path

    profile = _load_yaml_file(feature_path)
    profile["path"] = str(feature_path)
    profile["name"] = profile_name
    return profile


def resolve_feature_profile(
    cfg: dict | None = None,
    *,
    profile_name: str | None = None,
    profile_config_path: str = DEFAULT_PROFILE_CONFIG_PATH,
) -> dict[str, Any]:
    cfg = cfg or {}
    features_cfg = cfg.get("features", {})
    profile_data = load_feature_profiles(profile_config_path)
    profiles = profile_data["profiles"]

    resolved_profile_name = profile_name or features_cfg.get("profile") or profile_data.get("default_profile")
    if resolved_profile_name not in profiles:
        raise ValueError(
            f"Unknown feature profile: {resolved_profile_name}. "
            f"Available: {', '.join(sorted(profiles))}"
        )

    profile = _resolve_profile_definition(
        resolved_profile_name,
        deepcopy(profiles[resolved_profile_name]),
        profile_config_path=profile_config_path,
    )
    alpha = str(profile.get("alpha", features_cfg.get("alpha", cfg.get("alpha_version", 158))))
    cache_name = profile.get("cache_name")
    if not cache_name:
        cache_name = f"alpha{alpha}_panel" if resolved_profile_name.endswith("_full") else f"{resolved_profile_name}_panel"

    return {
        "name": resolved_profile_name,
        "alpha": alpha,
        "cache_dir": features_cfg.get("cache_dir") or str(Path("data/cache") / cache_name),
        "alpha158_config": deepcopy(profile.get("alpha158")),
        "raw": profile,
        "profile_config_path": profile_config_path,
        "profile_path": profile.get("path"),
    }


def get_native_cache_dir(cfg: dict) -> str:
    return resolve_feature_profile(cfg)["cache_dir"]
