"""
Model factory: transfer learning backbones with replaced classification heads.
Supports EfficientNetV2-S (default), ConvNeXt-Tiny, ResNet50.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# Grad-CAM target layer names per architecture (child module name path)
GRADCAM_TARGET_LAYERS: dict[str, str] = {
    "efficientnet_v2_s": "features",
    "convnext_tiny": "features",
    "resnet50": "layer4",
}


def _replace_classifier(head: nn.Module, in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    """Standard head: dropout + linear."""
    layers: list[nn.Module] = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_features, num_classes))
    return nn.Sequential(*layers)


def build_efficientnet_v2_s(num_classes: int, pretrained: bool, dropout: float) -> tuple[nn.Module, str]:
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = _replace_classifier(model.classifier, in_features, num_classes, dropout)
    return model, "features"


def build_convnext_tiny(num_classes: int, pretrained: bool, dropout: float) -> tuple[nn.Module, str]:
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        *_replace_classifier(model.classifier[2], in_features, num_classes, dropout),
    )
    return model, "features"


def build_resnet50(num_classes: int, pretrained: bool, dropout: float) -> tuple[nn.Module, str]:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = _replace_classifier(model.fc, in_features, num_classes, dropout)
    return model, "layer4"


_BUILDERS = {
    "efficientnet_v2_s": build_efficientnet_v2_s,
    "convnext_tiny": build_convnext_tiny,
    "resnet50": build_resnet50,
}


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> tuple[nn.Module, str]:
    """
    Build model and return (model, gradcam_target_layer_name).

    Parameters
    ----------
    freeze_backbone : if True, freeze all parameters except classifier head
    """
    name = name.lower().strip()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(_BUILDERS.keys())}")

    model, target_layer = _BUILDERS[name](num_classes, pretrained, dropout)

    if freeze_backbone:
        _freeze_backbone(model, name)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built %s | trainable params: %s | Grad-CAM layer: %s", name, f"{n_params:,}", target_layer)
    return model, target_layer


def _freeze_backbone(model: nn.Module, name: str) -> None:
    """Freeze feature extractor; keep classifier trainable."""
    if name == "efficientnet_v2_s":
        for param in model.features.parameters():
            param.requires_grad = False
    elif name == "convnext_tiny":
        for param in model.features.parameters():
            param.requires_grad = False
    elif name == "resnet50":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True


def get_gradcam_target_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Retrieve submodule for Grad-CAM hooks."""
    if not hasattr(model, layer_name):
        raise AttributeError(f"Model has no attribute '{layer_name}' for Grad-CAM")
    return getattr(model, layer_name)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters (e.g. after warmup epochs)."""
    for param in model.parameters():
        param.requires_grad = True


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model weights from checkpoint dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(ckpt, strict=strict)
    return ckpt if isinstance(ckpt, dict) else {}
