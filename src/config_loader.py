"""Layered runtime config loading with experiment/model profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.config_utils import deep_update, load_yaml_file
from src.experiment_profiles import resolve_experiment_profile
from src.model_profiles import resolve_model_profile


def load_runtime_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    cfg = load_yaml_file(config_path)
    cfg.setdefault("runtime", {})
    return cfg


def load_config(
    config_path: str = "configs/config.yaml",
    *,
    experiment_profile_name: str | None = None,
    model_profile_name: str | None = None,
) -> dict[str, Any]:
    merged = deepcopy(load_runtime_config(config_path))

    experiment_profile = resolve_experiment_profile(merged, profile_name=experiment_profile_name)
    deep_update(merged, experiment_profile["config"])

    model_profile = resolve_model_profile(merged, profile_name=model_profile_name)
    deep_update(merged, model_profile["config"])

    merged.setdefault("runtime", {})
    merged["runtime"]["config_path"] = config_path
    merged.setdefault("experiment", {})
    merged["experiment"]["profile"] = experiment_profile["name"]
    merged["experiment"]["profile_path"] = experiment_profile["path"]
    merged.setdefault("model", {})
    merged["model"]["profile"] = model_profile["name"]
    merged["model"]["profile_path"] = model_profile["path"]
    return merged
