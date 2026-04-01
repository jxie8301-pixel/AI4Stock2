"""Feature-profile resolution for native cache generation/training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

try:
    from src.data_source import get_default_factor_store_dir, resolve_data_source_name
except ModuleNotFoundError:
    from data_source import get_default_factor_store_dir, resolve_data_source_name  # type: ignore


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
    try:
        from src.gen_feature import (
            ALL_FACTORS_LGBM_PREFIX,
            get_alpha158_feature_config,
            get_lgbm_purified_feature_names,
        )
    except ModuleNotFoundError:
        from gen_feature import (  # type: ignore
            ALL_FACTORS_LGBM_PREFIX,
            get_alpha158_feature_config,
            get_lgbm_purified_feature_names,
        )

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
    generation_space = str(profile.get("generation_space", profile.get("generation_alpha", "full_factor_space")))
    factor_store_name = profile.get("factor_store_name") or "full_factor_space"
    data_source = resolve_data_source_name(cfg)
    factor_store_dir = features_cfg.get("factor_store_dir") or get_default_factor_store_dir(
        data_source,
        factor_store_name=factor_store_name,
    )
    if "selected_columns" in profile:
        profile_selected_columns = list(profile["selected_columns"])
    elif alpha == "158":
        profile_selected_columns = get_alpha158_feature_config(profile.get("alpha158"))[1]
    elif alpha == "lgbm_purified":
        profile_selected_columns = [
            f"{ALL_FACTORS_LGBM_PREFIX}{name}"
            for name in get_lgbm_purified_feature_names(profile.get("lgbm_purified"))
        ]
    else:
        profile_selected_columns = None

    return {
        "name": resolved_profile_name,
        "data_source": data_source,
        "alpha": alpha,
        "generation_space": generation_space,
        "factor_store_dir": factor_store_dir,
        "cache_dir": factor_store_dir,
        "alpha158_config": deepcopy(profile.get("alpha158")),
        "selected_columns": deepcopy(profile_selected_columns),
        "raw": profile,
        "profile_config_path": profile_config_path,
        "profile_path": profile.get("path"),
    }


def get_native_cache_dir(cfg: dict) -> str:
    return resolve_feature_profile(cfg)["factor_store_dir"]


def get_native_factor_store_dir(cfg: dict) -> str:
    return resolve_feature_profile(cfg)["factor_store_dir"]
