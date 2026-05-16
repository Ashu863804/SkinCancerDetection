"""
Configuration loader: YAML defaults + environment overrides + runtime overrides.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR = "SKIN_CANCER_ENV"
_DEFAULT_CONFIG_REL = "configs/default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    """Set config['a']['b'] from dotted_key 'a.b'."""
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _parse_overrides(overrides: dict[str, Any] | None) -> dict:
    """Convert flat dotted overrides to nested dict."""
    if not overrides:
        return {}
    nested: dict = {}
    for key, value in overrides.items():
        keys = key.split(".")
        current = nested
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return nested


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from start to find directory containing configs/default.yaml."""
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / _DEFAULT_CONFIG_REL).exists():
            return path
    return current


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load configuration from YAML, apply environment-specific paths, then overrides.

    Parameters
    ----------
    config_path : optional path to YAML; defaults to configs/default.yaml under project root
    overrides : flat dict with dotted keys, e.g. {"train.epochs": 5}
    """
    root = find_project_root()
    if config_path is None:
        config_path = root / _DEFAULT_CONFIG_REL
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment: SKIN_CANCER_ENV or config.env
    env = os.environ.get(_ENV_VAR, config.get("env", "local")).lower()
    config["env"] = env
    config["paths"]["project_root"] = str(root)

    # Resolve data paths for Kaggle vs local
    if env == "kaggle":
        config["data"]["metadata_csv"] = config["data"]["kaggle_metadata_csv"]
        config["data"]["images_dirs"] = list(config["data"]["kaggle_images_dirs"])
    else:
        # Make local paths absolute relative to project root
        meta = config["data"]["metadata_csv"]
        if not Path(meta).is_absolute():
            config["data"]["metadata_csv"] = str(root / meta)
        images_dirs = []
        for d in config["data"]["images_dirs"]:
            p = Path(d)
            images_dirs.append(str(p if p.is_absolute() else root / p))
        config["data"]["images_dirs"] = images_dirs

    # Apply dotted overrides
    if overrides:
        nested = _parse_overrides(overrides)
        config = _deep_merge(config, nested)

    # Ensure output directories exist
    paths = config["paths"]
    for key in ("saved_models_dir", "outputs_dir", "logs_dir", "reports_dir",
                "figures_dir", "gradcam_dir", "splits_dir", "debug_dir"):
        rel = paths.get(key, "")
        if rel and not Path(rel).is_absolute():
            paths[key] = str(root / rel)

    return config


def get_device(config: dict) -> str:
    """Resolve device string from config."""
    import torch

    device_cfg = config.get("project", {}).get("device", "auto")
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def save_config_snapshot(config: dict, path: str | Path) -> None:
    """Save resolved config for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
