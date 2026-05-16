"""
Evaluation metrics for imbalanced multiclass skin lesion classification.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.constants import CLASS_NAMES, NUM_CLASSES


def compute_epoch_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "val",
) -> dict:
    """Compute standard metrics for one epoch."""
    metrics = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f"{prefix}_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    compute_roc_auc: bool = True,
) -> dict:
    """
    Full evaluation: confusion matrix, per-class metrics, classification report, ROC-AUC OVR.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if compute_roc_auc and y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            auc_ovr = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=list(range(NUM_CLASSES)),
            )
            result["roc_auc_macro_ovr"] = float(auc_ovr)
            per_class_auc = {}
            for i, name in enumerate(CLASS_NAMES):
                y_binary = (y_true == i).astype(int)
                if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                    per_class_auc[name] = float(roc_auc_score(y_binary, y_proba[:, i]))
                else:
                    per_class_auc[name] = None
            result["roc_auc_per_class"] = per_class_auc
        except ValueError as e:
            result["roc_auc_error"] = str(e)

    return result


def save_metrics_report(metrics: dict, path: str | Path) -> None:
    """Save metrics dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
