"""
Public training API: train_model()
Entry point for local scripts and Kaggle notebook.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.config import get_device, load_config, save_config_snapshot
from src.data.datamodule import HAM10000DataModule
from src.losses.focal import build_criterion
from src.models.factory import build_model, unfreeze_all
from src.train.callbacks import EarlyStopping, ModelCheckpoint
from src.train.engine import (
    build_optimizer,
    build_scheduler,
    train_one_epoch,
    validate_one_epoch,
)
from src.utils.logging import setup_logger
from src.utils.paths import ensure_output_dirs
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def train_model(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    config_path : optional path to YAML config
    overrides : flat dict with dotted keys, e.g. {"training.epochs": 5}

    Returns
    -------
    dict with training history and best metrics
    """
    config = load_config(config_path=config_path, overrides=overrides)
    ensure_output_dirs(config)

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    log_dir = Path(config["paths"]["logs_dir"])
    setup_logger("skin_cancer", log_file=log_dir / "train.log")
    logger.info("Starting training | env=%s", config.get("env"))

    save_config_snapshot(config, log_dir / "config_snapshot.yaml")

    device = get_device(config)
    logger.info("Using device: %s", device)

    # Data
    datamodule = HAM10000DataModule(config)
    datamodule.setup(fit=True)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Model
    model_cfg = config.get("model", {})
    model, gradcam_layer = build_model(
        name=model_cfg.get("name", "efficientnet_v2_s"),
        num_classes=model_cfg.get("num_classes", 7),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
        freeze_backbone=model_cfg.get("freeze_backbone_epochs", 0) > 0,
    )
    config["model"]["gradcam_target_layer"] = gradcam_layer
    model = model.to(device)

    # Loss & optimizer
    class_weights = datamodule.class_weights
    criterion = build_criterion(config, class_weights=class_weights, device=device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    train_cfg = config.get("training", {})
    use_amp = train_cfg.get("mixed_precision", True) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Callbacks
    ckpt_path = Path(config["paths"]["saved_models_dir"]) / "best.pt"
    checkpoint = ModelCheckpoint(
        filepath=ckpt_path,
        monitor=train_cfg.get("checkpoint_metric", "val_macro_f1"),
        mode=train_cfg.get("checkpoint_mode", "max"),
    )
    early_stop = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 7),
        monitor=train_cfg.get("checkpoint_metric", "val_macro_f1"),
        mode=train_cfg.get("checkpoint_mode", "max"),
    )

    epochs = train_cfg.get("epochs", 30)
    freeze_epochs = model_cfg.get("freeze_backbone_epochs", 0)
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            logger.info("Unfreezing backbone at epoch %d", epoch)
            unfreeze_all(model)
            optimizer = build_optimizer(model, config)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip_norm=train_cfg.get("gradient_clip_norm"),
            log_interval=config.get("logging", {}).get("log_interval", 10),
            epoch=epoch,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            split_name="val",
        )

        epoch_metrics = {**train_metrics, **val_metrics}
        history.append(epoch_metrics)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_macro_f1=%.4f | val_bal_acc=%.4f",
            epoch,
            epochs,
            train_metrics["train_loss"],
            val_metrics["val_loss"],
            val_metrics["val_macro_f1"],
            val_metrics["val_balanced_accuracy"],
        )

        checkpoint.step(epoch_metrics, model, optimizer, epoch, config)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                pass  # stepped per batch in onecycle; skip here for cosine
            else:
                scheduler.step()

        if early_stop.step(epoch_metrics):
            break

    # Save training history
    history_path = log_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if config.get("logging", {}).get("save_training_curves", True):
        from src.evaluate.plots import plot_training_curves
        figures_dir = Path(config["paths"]["figures_dir"])
        plot_training_curves(history_path, figures_dir / "training_curves.png")

    # Save last checkpoint
    last_path = Path(config["paths"]["saved_models_dir"]) / "last.pt"
    torch.save(
        {
            "epoch": len(history),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        last_path,
    )

    result = {
        "history": history,
        "best_epoch": checkpoint.best_epoch,
        "best_score": checkpoint.best_score,
        "checkpoint_path": str(ckpt_path),
        "history_path": str(history_path),
    }
    logger.info("Training complete | best %s=%s at epoch %d", checkpoint.monitor, checkpoint.best_score, checkpoint.best_epoch)
    return result
