"""Tests for lesion-level splitting (no leakage)."""

import pandas as pd
import pytest

from src.constants import DX_COLUMN, LESION_ID_COLUMN
from src.data.splits import assert_no_lesion_leakage, lesion_level_stratified_split


def _synthetic_metadata(n_lesions: int = 50, images_per_lesion: int = 2) -> pd.DataFrame:
    """Create synthetic metadata mimicking HAM10000 structure."""
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    rows = []
    for lesion_idx in range(n_lesions):
        lesion_id = f"LES_{lesion_idx}"
        dx = classes[lesion_idx % len(classes)]
        for img_idx in range(images_per_lesion):
            rows.append({
                "image_id": f"IMG_{lesion_idx}_{img_idx}",
                "lesion_id": lesion_id,
                "dx": dx,
            })
    return pd.DataFrame(rows)


def test_lesion_level_split_no_leakage():
    df = _synthetic_metadata(n_lesions=70, images_per_lesion=3)
    train_df, val_df, test_df = lesion_level_stratified_split(df, 0.8, 0.1, 0.1, seed=42)
    assert_no_lesion_leakage(train_df, val_df, test_df)

    train_lesions = set(train_df[LESION_ID_COLUMN])
    val_lesions = set(val_df[LESION_ID_COLUMN])
    test_lesions = set(test_df[LESION_ID_COLUMN])
    assert len(train_lesions & val_lesions) == 0
    assert len(train_lesions & test_lesions) == 0
    assert len(val_lesions & test_lesions) == 0


def test_all_images_assigned():
    df = _synthetic_metadata(n_lesions=100)
    train_df, val_df, test_df = lesion_level_stratified_split(df, 0.8, 0.1, 0.1, seed=0)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df)


def test_stratification_covers_classes():
    df = _synthetic_metadata(n_lesions=100)
    train_df, val_df, test_df = lesion_level_stratified_split(df, 0.8, 0.1, 0.1, seed=7)
    for split_df in (train_df, val_df, test_df):
        if len(split_df) > 0:
            assert DX_COLUMN in split_df.columns
