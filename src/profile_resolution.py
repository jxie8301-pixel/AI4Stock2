"""Shared helpers for profile inheritance and external profile files."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config_utils import deep_update, load_yaml_file


@dataclass(frozen=True)
class LoadedProfileEntry:
    profile_entry: dict[str, Any]
    loaded_profile: dict[str, Any]
    source_path: str


def build_inline_profile_path(profile_name: str, profile_config_path: str) -> str:
    return f"{Path(profile_config_path).resolve()}::{profile_name}"


def load_profile_entry(
    profile_name: str,
    *,
    profiles: dict[str, Any],
    profile_config_path: str,
    profile_kind: str,
    stack: tuple[str, ...] = (),
) -> LoadedProfileEntry:
    display_kind = profile_kind[:1].upper() + profile_kind[1:]
    if profile_name in stack:
        cycle = " -> ".join([*stack, profile_name])
        raise ValueError(f"{display_kind} profile inheritance cycle detected: {cycle}")
    if profile_name not in profiles:
        raise ValueError(f"Unknown {profile_kind} profile: {profile_name}")

    profile_entry = deepcopy(profiles[profile_name])
    source_path = build_inline_profile_path(profile_name, profile_config_path)
    if "path" not in profile_entry:
        loaded_profile: dict[str, Any] = {}
    else:
        repo_root = Path(profile_config_path).resolve().parent.parent
        profile_path = Path(profile_entry.pop("path"))
        if not profile_path.is_absolute():
            profile_path = repo_root / profile_path
        loaded_profile = load_yaml_file(profile_path)
        source_path = str(profile_path)
    return LoadedProfileEntry(
        profile_entry=profile_entry,
        loaded_profile=loaded_profile,
        source_path=source_path,
    )


def merge_parent_profile(parent_profile: dict[str, Any], child_profile: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(parent_profile)
    merged.pop("name", None)
    merged.pop("path", None)
    deep_update(merged, child_profile)
    return merged
