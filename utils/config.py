"""Config loading and merging utilities."""

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file into a dict.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed config dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base.

    Override values take precedence. Both dicts are not mutated.

    Args:
        base: Base configuration dict.
        override: Override configuration dict.

    Returns:
        Merged configuration dict.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_and_merge_config(base_path: str, override_path: str | None = None) -> dict[str, Any]:
    """Load base config and optionally merge with an override config.

    If override_path is not the same as base_path, the override is merged on top.
    This allows passing a specific config that only overrides a few keys.

    Args:
        base_path: Path to base.yaml.
        override_path: Path to override YAML (e.g. taxi_noise8.yaml). May be None.

    Returns:
        Merged configuration dict.
    """
    # Find the repo base.yaml relative to the override if override is provided
    base_dir = Path(base_path).parent
    base_yaml = base_dir / "base.yaml"

    if base_yaml.exists() and str(base_yaml) != base_path:
        cfg = load_yaml_config(str(base_yaml))
        override = load_yaml_config(base_path)
        cfg = deep_merge(cfg, override)
    else:
        cfg = load_yaml_config(base_path)

    if override_path and override_path != base_path:
        override = load_yaml_config(override_path)
        cfg = deep_merge(cfg, override)

    return cfg


def resolve_config(config_path: str) -> dict[str, Any]:
    """Load a config, automatically merging with base.yaml if needed.

    Looks for base.yaml in the same directory as config_path. If found and
    config_path is not base.yaml, merges base + config_path.

    Args:
        config_path: Path to config YAML file.

    Returns:
        Fully resolved config dictionary.
    """
    config_path = str(config_path)
    cfg_dir = Path(config_path).parent
    base_yaml = cfg_dir / "base.yaml"

    if base_yaml.exists() and Path(config_path).name != "base.yaml":
        base_cfg = load_yaml_config(str(base_yaml))
        override_cfg = load_yaml_config(config_path)
        return deep_merge(base_cfg, override_cfg)
    else:
        return load_yaml_config(config_path)


def save_resolved_config(cfg: dict[str, Any], output_path: str) -> None:
    """Save a resolved config dict to YAML.

    Args:
        cfg: Configuration dictionary.
        output_path: Destination YAML file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
