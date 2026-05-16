"""
Grad-CAM explainability for multiclass skin lesion CNNs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import CLASS_NAMES, IDX_TO_LABEL
from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from src.models.factory import get_gradcam_target_layer


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN feature maps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

        self._handles.append(target_layer.register_forward_hook(self._forward_hook))
        self._handles.append(target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for one image tensor (1, C, H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        score = logits[0, target_class]
        score.backward()

        activations = self.activations
        gradients = self.gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        # Global average pooling of gradients over spatial dims
        if gradients.dim() == 4:
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
        else:
            weights = gradients.mean(dim=-1, keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Resize to input image size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay heatmap on RGB image."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != heatmap.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (image_rgb.shape[1], image_rgb.shape[0]))
    blended = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended


def save_gradcam_visualization(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    save_path: str | Path,
    title: str = "",
) -> Path:
    """Save side-by-side original and Grad-CAM overlay."""
    import matplotlib.pyplot as plt

    overlay = overlay_heatmap(original_rgb, heatmap)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(title or "Grad-CAM")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def preprocess_for_model(
    image_rgb: np.ndarray,
    image_size: int = 224,
) -> torch.Tensor:
    """Apply eval-style normalize for Grad-CAM (without albumentations dependency in minimal path)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return transform(image=image_rgb)["image"]
