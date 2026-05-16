"""Smoke tests for model factory and loss."""

import torch

from src.losses.focal import FocalLoss, build_criterion
from src.models.factory import build_model, get_gradcam_target_layer


def test_efficientnet_forward():
    model, layer_name = build_model("efficientnet_v2_s", num_classes=7, pretrained=False, dropout=0.3)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)
    layer = get_gradcam_target_layer(model, layer_name)
    assert layer is not None


def test_convnext_forward():
    model, _ = build_model("convnext_tiny", num_classes=7, pretrained=False)
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 7)


def test_focal_loss_backward():
    criterion = FocalLoss(gamma=2.0)
    logits = torch.randn(4, 7, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    loss = criterion(logits, targets)
    loss.backward()
    assert logits.grad is not None


def test_build_criterion():
    config = {
        "loss": {"name": "focal", "focal_gamma": 2.0},
        "training": {"use_class_weights": True, "label_smoothing": 0.0},
    }
    weights = torch.ones(7)
    criterion = build_criterion(config, class_weights=weights, device="cpu")
    logits = torch.randn(2, 7)
    loss = criterion(logits, torch.tensor([0, 6]))
    assert loss.item() > 0
