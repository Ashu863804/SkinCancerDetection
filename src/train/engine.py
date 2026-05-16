"""
Training and validation epoch loops with mixed precision and metric tracking.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.evaluate.metrics import compute_epoch_metrics

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler | None = None,
    use_amp: bool = True,
    gradient_clip_norm: float | None = 1.0,
    log_interval: int = 10,
    epoch: int = 0,
) -> dict:
    """Run one training epoch; return aggregated metrics."""
    model.train()
    all_preds: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.startswith("cuda"):
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        if batch_idx % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_epoch_metrics(
        np.array(all_labels),
        np.array(all_preds),
        prefix="train",
    )
    metrics["train_loss"] = running_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True,
    epoch: int = 0,
    split_name: str = "val",
) -> dict:
    """Run validation/test epoch."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"{split_name.capitalize()} Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if use_amp and device.startswith("cuda"):
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    prefix = split_name
    metrics = compute_epoch_metrics(
        np.array(all_labels),
        np.array(all_preds),
        prefix=prefix,
    )
    metrics[f"{prefix}_loss"] = running_loss / max(n_batches, 1)
    return metrics


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """AdamW optimizer from config."""
    train_cfg = config.get("training", {})
    lr = train_cfg.get("learning_rate", 3e-4)
    wd = train_cfg.get("weight_decay", 1e-4)
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Cosine annealing or OneCycle scheduler."""
    train_cfg = config.get("training", {})
    scheduler_name = train_cfg.get("scheduler", "cosine").lower()
    epochs = train_cfg.get("epochs", 30)

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg.get("learning_rate", 3e-4),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {scheduler_name}")
