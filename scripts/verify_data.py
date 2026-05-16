#!/usr/bin/env python
"""
Verify HAM10000 data paths, metadata, and lesion-level splits.
Run from project root: python scripts/verify_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.datamodule import HAM10000DataModule
from src.data.splits import assert_no_lesion_leakage
from src.utils.logging import setup_logger


def main():
    setup_logger("verify_data")
    config = load_config()
    dm = HAM10000DataModule(config)
    dm.setup(fit=True)

    assert_no_lesion_leakage(dm.train_df, dm.val_df, dm.test_df)

    print("=== Data verification OK ===")
    print(f"Train images: {len(dm.train_df)} | lesions: {dm.train_df['lesion_id'].nunique()}")
    print(f"Val images:   {len(dm.val_df)} | lesions: {dm.val_df['lesion_id'].nunique()}")
    print(f"Test images:  {len(dm.test_df)} | lesions: {dm.test_df['lesion_id'].nunique()}")
    print(f"Resolved paths: {len(dm.image_path_map)}")
    print("Train class distribution:", dm.get_class_distribution("train"))


if __name__ == "__main__":
    main()
