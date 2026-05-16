"""
Training callbacks: checkpointing and early stopping on validation macro-F1.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Save best model when monitored metric improves."""

    def __init__(
        self,
        filepath: str | Path,
        monitor: str = "val_macro_f1",
        mode: str = "max",
        save_last: bool = True,
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best_score: float | None = None
        self.best_epoch = -1

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def step(
        self,
        metrics: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: dict,
    ) -> bool:
        """
        Check metrics and save if improved. Returns True if new best.
        """
        score = metrics.get(self.monitor)
        if score is None:
            logger.warning("Checkpoint monitor '%s' not in metrics", self.monitor)
            return False

        if not self._is_improvement(score):
            return False

        self.best_score = score
        self.best_epoch = epoch
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": copy.deepcopy(metrics),
            "monitor": self.monitor,
            "best_score": score,
            "config": config,
        }
        torch.save(checkpoint, self.filepath)
        logger.info(
            "Saved best checkpoint | %s=%.4f | epoch=%d | path=%s",
            self.monitor,
            score,
            epoch,
            self.filepath,
        )
        return True


class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 7, monitor: str = "val_macro_f1", mode: str = "max"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def step(self, metrics: dict) -> bool:
        """Returns True if training should stop."""
        score = metrics.get(self.monitor)
        if score is None:
            return False

        if self.best_score is None:
            self.best_score = score
            return False

        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered | no improvement in %s for %d epochs",
                    self.monitor,
                    self.patience,
                )
        return self.should_stop
