"""
Public visualization API: training curves, Grad-CAM driver.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import load_config
from src.evaluate.plots import plot_training_curves
from src.inference.predictor import SkinLesionPredictor
from src.utils.logging import setup_logger
from src.utils.paths import ensure_output_dirs

logger = logging.getLogger(__name__)


def plot_metrics(config_path: str | None = None, overrides: dict[str, Any] | None = None) -> str:
    """Plot training curves from saved history JSON."""
    config = load_config(config_path=config_path, overrides=overrides)
    history_path = Path(config["paths"]["logs_dir"]) / "training_history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}. Run train_model() first.")

    save_path = Path(config["paths"]["figures_dir"]) / "training_curves.png"
    plot_training_curves(history_path, save_path)
    logger.info("Saved training curves to %s", save_path)
    return str(save_path)


def run_gradcam(
    image_path_or_id: str,
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    class_idx: int | None = None,
) -> dict:
    """
    Generate Grad-CAM visualization for one image.

    Parameters
    ----------
    image_path_or_id : full path to image OR image_id (looks up in dataset paths)
    class_idx : target class index for Grad-CAM; None = predicted class
    """
    config = load_config(config_path=config_path, overrides=overrides)
    ensure_output_dirs(config)
    setup_logger("skin_cancer")

    path = Path(image_path_or_id)
    if not path.is_file():
        # Resolve image_id via path index
        from src.utils.paths import build_image_path_index
        ext = config["data"].get("image_extension", ".jpg")
        index = build_image_path_index(config["data"]["images_dirs"], extension=ext)
        if image_path_or_id in index:
            path = Path(index[image_path_or_id])
        else:
            raise FileNotFoundError(f"Image not found: {image_path_or_id}")

    predictor = SkinLesionPredictor(config=config, checkpoint_path=checkpoint_path)
    result = predictor.predict(path, return_gradcam=True, target_class=class_idx)
    logger.info(
        "Grad-CAM saved | pred=%s | path=%s",
        result.get("predicted_class"),
        result.get("gradcam_save_path"),
    )
    return result


def predict_image(
    image_path: str,
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    with_gradcam: bool = False,
) -> dict:
    """Public inference API for notebook and scripts."""
    config = load_config(config_path=config_path, overrides=overrides)
    predictor = SkinLesionPredictor(config=config, checkpoint_path=checkpoint_path)
    return predictor.predict(image_path, return_gradcam=with_gradcam)
