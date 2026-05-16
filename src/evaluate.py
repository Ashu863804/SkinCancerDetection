"""
Public evaluation API: evaluate_model()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import get_device, load_config
from src.data.datamodule import HAM10000DataModule
from src.evaluate.metrics import compute_full_metrics, save_metrics_report
from src.evaluate.plots import plot_confusion_matrix, plot_per_class_metrics
from src.models.factory import build_model, load_checkpoint
from src.utils.logging import setup_logger
from src.utils.paths import ensure_output_dirs
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def _collect_predictions(model, dataloader, device, use_amp=True):
    """Run inference and collect labels, preds, probabilities."""
    model.eval()
    all_labels, all_preds, all_proba, all_ids = [], [], [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        if use_amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                logits = model(images)
        else:
            logits = model(images)

        proba = F.softmax(logits, dim=1).cpu().numpy()
        preds = proba.argmax(axis=1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_proba.extend(proba.tolist())
        all_ids.extend(batch.get("image_id", [""] * len(labels)))

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_proba),
        all_ids,
    )


def evaluate_model(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    checkpoint_path: str | None = None,
    split: str | None = None,
) -> dict:
    """
    Evaluate model on val or test split.

    Parameters
    ----------
    checkpoint_path : path to .pt file; defaults to saved_models/best.pt from config
    split : 'val' or 'test'; defaults to config evaluation.split
    """
    config = load_config(config_path=config_path, overrides=overrides)
    ensure_output_dirs(config)
    set_seed(config.get("project", {}).get("seed", 42))

    log_dir = Path(config["paths"]["logs_dir"])
    setup_logger("skin_cancer", log_file=log_dir / "evaluate.log")

    device = get_device(config)
    eval_cfg = config.get("evaluation", {})
    split = split or eval_cfg.get("split", "test")

    datamodule = HAM10000DataModule(config)
    datamodule.setup(fit=False)

    if split == "val":
        loader = datamodule.val_dataloader()
    elif split == "test":
        loader = datamodule.test_dataloader()
    else:
        raise ValueError(f"split must be 'val' or 'test', got {split}")

    model_cfg = config.get("model", {})
    model, _ = build_model(
        name=model_cfg.get("name", "efficientnet_v2_s"),
        num_classes=model_cfg.get("num_classes", 7),
        pretrained=False,
        dropout=model_cfg.get("dropout", 0.3),
    )

    ckpt_path = checkpoint_path or str(Path(config["paths"]["saved_models_dir"]) / "best.pt")
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train the model first.")

    load_checkpoint(model, ckpt_path, device=device)
    model = model.to(device)

    use_amp = config.get("training", {}).get("mixed_precision", True)
    y_true, y_pred, y_proba, image_ids = _collect_predictions(model, loader, device, use_amp)

    metrics = compute_full_metrics(
        y_true,
        y_pred,
        y_proba=y_proba,
        compute_roc_auc=eval_cfg.get("compute_roc_auc", True),
    )
    metrics["split"] = split
    metrics["checkpoint"] = ckpt_path
    metrics["n_samples"] = len(y_true)

    reports_dir = Path(config["paths"]["reports_dir"])
    save_metrics_report(metrics, reports_dir / f"metrics_{split}.json")

    cm = np.array(metrics["confusion_matrix"])
    figures_dir = Path(config["paths"]["figures_dir"])
    plot_confusion_matrix(cm, figures_dir / f"confusion_matrix_{split}.png")
    plot_per_class_metrics(metrics["per_class"], figures_dir / f"per_class_f1_{split}.png", metric="f1")
    plot_per_class_metrics(metrics["per_class"], figures_dir / f"per_class_recall_{split}.png", metric="recall")

    if eval_cfg.get("save_predictions", True):
        preds_path = reports_dir / f"predictions_{split}.json"
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"image_id": iid, "true": int(t), "pred": int(p), "proba": prob}
                    for iid, t, p, prob in zip(image_ids, y_true, y_pred, y_proba)
                ],
                f,
                indent=2,
            )

    logger.info(
        "Evaluation [%s] | accuracy=%.4f | balanced_acc=%.4f | macro_f1=%.4f",
        split,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    )

    # Print classification report to log
    from src.constants import CLASS_NAMES
    report = metrics["classification_report"]
    for name in CLASS_NAMES:
        if name in report:
            r = report[name]
            logger.info(
                "  %s | P=%.3f R=%.3f F1=%.3f support=%d",
                name,
                r["precision"],
                r["recall"],
                r["f1-score"],
                r["support"],
            )

    return metrics
