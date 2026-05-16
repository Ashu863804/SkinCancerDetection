"""
Reproducibility utilities for Python, NumPy, and PyTorch.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducible experiments.

    Note: deterministic=True may reduce GPU throughput slightly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_generator(seed: int) -> torch.Generator:
    """Return a PyTorch Generator for DataLoader reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
