"""
Focal Loss for imbalanced multiclass classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: per-class weights tensor of shape (num_classes,) or None
    gamma: focusing parameter (default 2.0)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (N, C) logits
        targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    config: dict,
    class_weights: torch.Tensor | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Build loss function from config."""
    loss_cfg = config.get("loss", {})
    train_cfg = config.get("training", {})
    name = loss_cfg.get("name", "focal").lower()
    label_smoothing = train_cfg.get("label_smoothing", 0.0)

    alpha = None
    if loss_cfg.get("focal_alpha") is not None:
        alpha = torch.tensor(loss_cfg["focal_alpha"], dtype=torch.float32, device=device)
    elif train_cfg.get("use_class_weights", True) and class_weights is not None:
        alpha = class_weights.to(device)

    if name == "focal":
        return FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss: {name}")
