"""
Path resolution for HAM10000 images across multiple directories (Kaggle/local).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_image_path(
    image_id: str,
    images_dirs: list[str],
    extension: str = ".jpg",
) -> Path | None:
    """
    Find image file for image_id in one of images_dirs.

    HAM10000 stores files as {image_id}.jpg in part_1 or part_2 folders.
    """
    filename = f"{image_id}{extension}" if not image_id.endswith(extension) else image_id
    for directory in images_dirs:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_image_path_index(
    images_dirs: list[str],
    extension: str = ".jpg",
) -> dict[str, str]:
    """
    Build a mapping image_id -> absolute path by scanning image directories.
    Faster for repeated lookups when metadata is large.
    """
    index: dict[str, str] = {}
    ext = extension if extension.startswith(".") else f".{extension}"
    for directory in images_dirs:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.warning("Image directory not found: %s", directory)
            continue
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() == ext.lower() and file_path.is_file():
                image_id = file_path.stem
                if image_id not in index:
                    index[image_id] = str(file_path.resolve())
    return index


def ensure_output_dirs(config: dict) -> None:
    """Create all configured output directories."""
    paths = config.get("paths", {})
    for key in (
        "saved_models_dir",
        "outputs_dir",
        "logs_dir",
        "reports_dir",
        "figures_dir",
        "gradcam_dir",
        "splits_dir",
        "debug_dir",
    ):
        p = paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)
