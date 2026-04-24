"""JSON/TOML configuration loading for command inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import tomllib


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or TOML config file into a dictionary."""

    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    else:
        raise ValueError("config files must use .json or .toml")
    if not isinstance(payload, dict):
        raise ValueError("config file root must be an object/table")
    return dict(payload)


def merge_config_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return config with non-None override values applied shallowly."""

    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
