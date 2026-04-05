"""Feature-profile resolution for native cache generation/training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from src.config_utils import deep_update, load_yaml_file
    from src.data_source import get_default_factor_store_dir, resolve_data_source_name
except ModuleNotFoundError:
    from config_utils import deep_update, load_yaml_file  # type: ignore
    from data_source import get_default_factor_store_dir, resolve_data_source_name  # type: ignore


DEFAULT_PROFILE_CONFIG_PATH = "configs/feature_profiles.yaml"


def load_feature_profiles(config_path: str = DEFAULT_PROFILE_CONFIG_PATH) -> dict[str, Any]:
    data = load_yaml_file(config_path)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No feature profiles found in {config_path}")
    return data


def _build_inline_profile_path(profile_name: str, profile_config_path: str) -> str:
    return f"{Path(profile_config_path).resolve()}::{profile_name}"


def _normalize_profile_column_mutation(
    value: Any,
    *,
    field_name: str,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    return list(value)


def _normalize_profile_repeat_columns(
    value: Any,
    *,
    field_name: str,
) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping of feature_name -> positive integer")
    out: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{field_name} keys must be non-empty strings")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{name!r}] must be a positive integer") from exc
        if count <= 0:
            raise ValueError(f"{field_name}[{name!r}] must be a positive integer")
        out[name] = count
    return out


def _materialize_profile_selected_columns(profile: dict[str, Any]) -> list[str] | None:
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

    if "selected_columns" in profile:
        selected_columns = profile.get("selected_columns")
        if selected_columns is None:
            return None
        if not isinstance(selected_columns, list) or not all(isinstance(item, str) and item for item in selected_columns):
            raise ValueError("feature profile selected_columns must be a list of non-empty strings")
        return list(selected_columns)

    alpha = str(profile.get("alpha", 158))
    if alpha == "158":
        return list(get_alpha158_feature_config(profile.get("alpha158"))[1])
    if alpha == "lgbm_purified":
        return [
            f"{ALL_FACTORS_LGBM_PREFIX}{name}"
            for name in get_lgbm_purified_feature_names(profile.get("lgbm_purified"))
        ]
    return None


def _expand_profile_selected_columns(profile: dict[str, Any]) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    selected_columns = _materialize_profile_selected_columns(profile)
    if selected_columns is None:
        return None, None, None

    repeat_columns = _normalize_profile_repeat_columns(
        profile.get("repeat_columns"),
        field_name=f"feature profile '{profile.get('name', '<inline>')}'.repeat_columns",
    )
    expanded_columns: list[str] = []
    source_columns: list[str] = []
    for column_name in selected_columns:
        repeat_count = repeat_columns.get(column_name, 1)
        for repeat_idx in range(repeat_count):
            expanded_name = column_name if repeat_idx == 0 else f"{column_name}__rep{repeat_idx + 1}"
            expanded_columns.append(expanded_name)
            source_columns.append(column_name)

    load_columns = list(dict.fromkeys(source_columns))
    return expanded_columns, source_columns, load_columns


def _resolve_profile_definition(
    profile_name: str,
    *,
    profiles: dict[str, Any],
    profile_config_path: str,
    stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if profile_name in stack:
        cycle = " -> ".join([*stack, profile_name])
        raise ValueError(f"Feature profile inheritance cycle detected: {cycle}")
    if profile_name not in profiles:
        raise ValueError(f"Unknown feature profile: {profile_name}")

    profile_entry = deepcopy(profiles[profile_name])
    source_path = _build_inline_profile_path(profile_name, profile_config_path)
    if "path" not in profile_entry:
        loaded_profile = {}
    else:
        repo_root = Path(profile_config_path).resolve().parent.parent
        feature_path = Path(profile_entry.pop("path"))
        if not feature_path.is_absolute():
            feature_path = repo_root / feature_path
        loaded_profile = load_yaml_file(feature_path)
        source_path = str(feature_path)

    extends_name = str(profile_entry.pop("extends", "") or "").strip()
    drop_columns = _normalize_profile_column_mutation(
        profile_entry.pop("drop_columns", None),
        field_name=f"feature profile '{profile_name}'.drop_columns",
    )
    add_columns = _normalize_profile_column_mutation(
        profile_entry.pop("add_columns", None),
        field_name=f"feature profile '{profile_name}'.add_columns",
    )
    repeat_columns = _normalize_profile_repeat_columns(
        profile_entry.pop("repeat_columns", None),
        field_name=f"feature profile '{profile_name}'.repeat_columns",
    )

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

    if drop_columns or add_columns:
        base_columns = _materialize_profile_selected_columns(merged)
        if base_columns is None:
            raise ValueError(
                f"Feature profile '{profile_name}' uses drop/add column mutations but does not resolve to selected_columns"
            )
        drop_set = set(drop_columns)
        mutated_columns = [item for item in base_columns if item not in drop_set]
        existing = set(mutated_columns)
        for item in add_columns:
            if item not in existing:
                mutated_columns.append(item)
                existing.add(item)
        merged["selected_columns"] = mutated_columns

    if repeat_columns:
        merged["repeat_columns"] = repeat_columns

    merged["path"] = source_path
    merged["name"] = profile_name
    return merged


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
        profiles=profiles,
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
    profile_selected_columns, profile_source_columns, profile_load_columns = _expand_profile_selected_columns(profile)

    return {
        "name": resolved_profile_name,
        "data_source": data_source,
        "alpha": alpha,
        "generation_space": generation_space,
        "factor_store_dir": factor_store_dir,
        "cache_dir": factor_store_dir,
        "alpha158_config": deepcopy(profile.get("alpha158")),
        "selected_columns": deepcopy(profile_selected_columns),
        "source_columns": deepcopy(profile_source_columns),
        "load_columns": deepcopy(profile_load_columns),
        "raw": profile,
        "profile_config_path": profile_config_path,
        "profile_path": profile.get("path"),
    }


def get_native_cache_dir(cfg: dict) -> str:
    return resolve_feature_profile(cfg)["factor_store_dir"]


def get_native_factor_store_dir(cfg: dict) -> str:
    return resolve_feature_profile(cfg)["factor_store_dir"]
