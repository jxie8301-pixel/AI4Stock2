"""Runtime config loading with preset composition."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_PRESET_CONFIG_PATH = "configs/model_presets.yaml"


def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def resolve_model_preset(
    cfg: dict[str, Any] | None = None,
    *,
    preset_name: str | None = None,
    preset_config_path: str = DEFAULT_MODEL_PRESET_CONFIG_PATH,
) -> dict[str, Any]:
    cfg = cfg or {}
    preset_index = _load_yaml_file(preset_config_path)
    presets = preset_index.get("presets", {})
    if not isinstance(presets, dict) or not presets:
        raise ValueError(f"No model presets found in {preset_config_path}")

    model_cfg = cfg.get("model", {})
    resolved_preset_name = preset_name or model_cfg.get("preset") or preset_index.get("default_preset")
    if resolved_preset_name not in presets:
        raise ValueError(
            f"Unknown model preset: {resolved_preset_name}. "
            f"Available: {', '.join(sorted(presets))}"
        )

    repo_root = Path(preset_config_path).resolve().parent.parent
    preset_path = Path(presets[resolved_preset_name]["path"])
    if not preset_path.is_absolute():
        preset_path = repo_root / preset_path

    preset = _load_yaml_file(preset_path)
    return {
        "name": resolved_preset_name,
        "path": str(preset_path),
        "config": preset,
    }


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    cfg = _load_yaml_file(config_path)
    preset = resolve_model_preset(cfg)

    merged = deepcopy(cfg)
    _deep_update(merged, preset["config"])

    merged.setdefault("model", {})
    merged["model"]["preset"] = preset["name"]
    merged["model"]["preset_path"] = preset["path"]
    return merged
