"""YAML config loading and command-line overrides."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from rom_bench.paths import resolve_path


Config = dict[str, Any]


def load_yaml(path: str | Path) -> Config:
    """Load a YAML config file."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return data


def _coerce_value(value: str) -> Any:
    """Convert a CLI string override to a Python value."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply Hydra-like key=value overrides such as model.rank=8."""
    resolved = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        target = resolved
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise TypeError(f"Cannot set nested override below non-dict key {part!r}")
        target[parts[-1]] = _coerce_value(raw_value)
    return resolved


def save_yaml(config: Config, path: str | Path) -> None:
    """Save a YAML config."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def parse_config_args(description: str) -> tuple[Config, argparse.Namespace]:
    """Parse --config and trailing key=value overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--force", action="store_true", help="Recompute even when cached outputs exist")
    args, overrides = parser.parse_known_args()
    config = apply_overrides(load_yaml(args.config), overrides)
    config.setdefault("_config_path", args.config)
    return config, args
