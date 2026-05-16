"""
Inference predictor for single-image and batch skin lesion classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_device, load_config
from src.constants import CLASS_NAMES, IDX_TO_LABEL
from src.data.augmentations import AugmentationPipeline
from src.models.factory import build_model, load_checkpoint

logger = logging.getLogger(__name__)


class SkinLesionPredictor:
    """
    Load trained checkpoint and run inference with optional Grad-CAM.
    """

    def __init__(
        self,
        config: dict | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ):
        if config is None:
            config = load_config()
        self.config = config
        self.device = device or get_device(config)

        model_cfg = config.get("model", {})
        self.model, self.gradcam_layer_name = build_model(
            name=model_cfg.get("name", "efficientnet_v2_s"),
            num_classes=model_cfg.get("num_classes", 7),
            pretrained=False,
            dropout=model_cfg.get("dropout", 0.3),
        )

        ckpt = checkpoint_path or config.get("inference", {}).get(
            "checkpoint_path",
            str(Path(config["paths"]["saved_models_dir"]) / "best.pt"),
        )
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        load_checkpoint(self.model, ckpt, device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.aug_pipeline = AugmentationPipeline(config)
        self.image_size = config.get("augmentation", {}).get("image_size", 224)
        self.top_k = config.get("inference", {}).get("top_k", 3)

    @torch.no_grad()
    def predict(
        self,
        image_path: str | Path,
        return_gradcam: bool = False,
        target_class: int | None = None,
    ) -> dict[str, Any]:
        """
        Predict class for a single image path.

        Returns dict with predicted_class, probabilities, top_k predictions.
        """
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise OSError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.predict_array(image_rgb, return_gradcam=return_gradcam, target_class=target_class, image_id=image_path.stem)

    def predict_array(
        self,
        image_rgb: np.ndarray,
        return_gradcam: bool = False,
        target_class: int | None = None,
        image_id: str = "image",
    ) -> dict[str, Any]:
        """Predict from RGB numpy array."""
        # Use eval transform via pipeline (label dummy 0)
        out = self.aug_pipeline(image_rgb.copy(), label=0, is_train=False)
        tensor = out["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)
            pred_idx = int(proba.argmax())
        pred_class = IDX_TO_LABEL[pred_idx]

        top_k = min(self.top_k, len(CLASS_NAMES))
        top_indices = proba.argsort()[::-1][:top_k]
        top_predictions = [
            {
                "class": IDX_TO_LABEL[i],
                "probability": float(proba[i]),
            }
            for i in top_indices
        ]

        result: dict[str, Any] = {
            "image_id": image_id,
            "predicted_class": pred_class,
            "predicted_index": pred_idx,
            "confidence": float(proba[pred_idx]),
            "probabilities": {IDX_TO_LABEL[i]: float(proba[i]) for i in range(len(CLASS_NAMES))},
            "top_predictions": top_predictions,
        }

        if return_gradcam:
            from src.explain.gradcam import GradCAM, overlay_heatmap, save_gradcam_visualization
            from src.models.factory import get_gradcam_target_layer

            target_layer = get_gradcam_target_layer(self.model, self.gradcam_layer_name)
            cam = GradCAM(self.model, target_layer)
            try:
                heatmap = cam.generate(tensor, target_class=target_class if target_class is not None else pred_idx)
                overlay = overlay_heatmap(
                    cv2.resize(image_rgb, (self.image_size, self.image_size)),
                    heatmap,
                )
                result["gradcam_heatmap"] = heatmap
                result["gradcam_overlay"] = overlay
                result["gradcam_target_class"] = IDX_TO_LABEL[target_class or pred_idx]

                save_dir = Path(self.config["paths"]["gradcam_dir"])
                save_path = save_dir / f"{image_id}_gradcam.png"
                save_gradcam_visualization(
                    cv2.resize(image_rgb, (self.image_size, self.image_size)),
                    heatmap,
                    save_path,
                    title=f"{result['gradcam_target_class']} | conf={result['confidence']:.2f}",
                )
                result["gradcam_save_path"] = str(save_path)
            finally:
                cam.remove_hooks()

        return result
