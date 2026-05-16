"""
Visualization: confusion matrix, training curves, per-class metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.constants import CLASS_DISPLAY_NAMES, CLASS_NAMES


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str | Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    labels = [CLASS_DISPLAY_NAMES.get(c, c) for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_metrics(
    per_class: dict,
    save_path: str | Path,
    metric: str = "f1",
) -> None:
    """Bar chart of per-class precision/recall/F1."""
    names = CLASS_NAMES
    values = [per_class.get(n, {}).get(metric, 0) for n in names]
    display = [CLASS_DISPLAY_NAMES.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(display, values, color="steelblue", edgecolor="navy")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Per-class {metric.upper()}")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_path: str | Path,
    save_path: str | Path,
) -> None:
    """Plot loss and macro-F1 curves from training history JSON."""
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = list(range(1, len(history) + 1))
    train_loss = [h.get("train_loss", 0) for h in history]
    val_loss = [h.get("val_loss", 0) for h in history]
    train_f1 = [h.get("train_macro_f1", 0) for h in history]
    val_f1 = [h.get("val_macro_f1", 0) for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training & Validation Loss")

    axes[1].plot(epochs, train_f1, label="Train Macro F1")
    axes[1].plot(epochs, val_f1, label="Val Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend()
    axes[1].set_title("Macro F1 Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
