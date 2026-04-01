"""Helpers for dotted-key overrides and sweep expansion."""

from __future__ import annotations

from itertools import product
from typing import Any

import yaml


def parse_override_value(raw: str) -> Any:
    text = str(raw).strip()
    if not text:
        return ""
    return yaml.safe_load(text)


def parse_override_arg(raw: str) -> tuple[str, Any]:
    text = str(raw).strip()
    if "=" not in text:
        raise ValueError(f"Override must be in key=value form, got: {raw}")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Override key must be non-empty, got: {raw}")
    return key, parse_override_value(value)


def parse_sweep_arg(raw: str) -> tuple[str, list[Any]]:
    text = str(raw).strip()
    if "=" not in text:
        raise ValueError(f"Sweep must be in key=value form, got: {raw}")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Sweep key must be non-empty, got: {raw}")
    values = parse_sweep_values(value)
    if not values:
        raise ValueError(f"Sweep values for {key} must be non-empty")
    return key, values


def apply_dotted_override(cfg: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    parts = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key}")
    cursor: dict[str, Any] = cfg
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            cursor[part] = {}
        elif not isinstance(current, dict):
            raise ValueError(f"Cannot override nested key through non-mapping path: {dotted_key}")
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return cfg


def apply_override_args(cfg: dict[str, Any], override_args: list[str] | None) -> dict[str, Any]:
    if not override_args:
        return cfg
    for raw in override_args:
        key, value = parse_override_arg(raw)
        apply_dotted_override(cfg, key, value)
    return cfg


def _parse_brace_sweep(text: str) -> list[Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    return [parse_override_value(part) for part in inner.split(",")]


def parse_sweep_values(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("{") and text.endswith("}"):
            return _parse_brace_sweep(text)
        if text.startswith("[") and text.endswith("]"):
            parsed = yaml.safe_load(text)
            if isinstance(parsed, list):
                return parsed
        return [parse_override_value(text)]
    return [raw]


def flatten_sweep_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for key, value in mapping.items():
        dotted_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_sweep_mapping(value, dotted_key))
        else:
            values = parse_sweep_values(value)
            if not values:
                raise ValueError(f"Sweep values for {dotted_key} must be non-empty")
            out[dotted_key] = values
    return out


def expand_sweep_grid(sweep_map: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not sweep_map:
        return [{}]
    keys = list(sweep_map)
    value_grid = [sweep_map[key] for key in keys]
    runs: list[dict[str, Any]] = []
    for combo in product(*value_grid):
        runs.append({key: value for key, value in zip(keys, combo)})
    return runs


def slugify_override_key(dotted_key: str) -> str:
    return dotted_key.strip().replace(".", "-").replace("_", "-")


def slugify_override_value(value: Any) -> str:
    text = str(value).strip().lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in text)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "na"


def build_override_tag(overrides: dict[str, Any]) -> str:
    if not overrides:
        return ""
    parts = [
        f"{slugify_override_key(key)}-{slugify_override_value(value)}"
        for key, value in sorted(overrides.items())
    ]
    return "__".join(parts)
