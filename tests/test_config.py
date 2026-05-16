"""Tests for configuration loading."""

from pathlib import Path

import pytest

from src.config import find_project_root, load_config


def test_find_project_root():
    root = find_project_root()
    assert (root / "configs" / "default.yaml").exists()


def test_load_config_local():
    config = load_config()
    assert config["env"] in ("local", "kaggle")
    assert "data" in config
    assert config["model"]["num_classes"] == 7


def test_load_config_overrides():
    config = load_config(overrides={"training.epochs": 5})
    assert config["training"]["epochs"] == 5


def test_paths_created_keys():
    config = load_config()
    assert "saved_models_dir" in config["paths"]
    assert Path(config["paths"]["project_root"]).exists()
