from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a mapping.")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def get_value(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        current = current[part]
    return current


def set_value(config: dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Set a YAML-parsed command line override such as train.steps=1000."""
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = yaml.safe_load(raw_value)


def with_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        set_value(result, key, value)
    return result


def json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def json_load(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
