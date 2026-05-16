"""
Fixed constants for HAM10000 seven-class skin lesion classification.
Class order is fixed across dataset, model, metrics, and visualization.
"""

from __future__ import annotations

# Canonical class order (index 0 .. 6)
CLASS_NAMES: list[str] = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

NUM_CLASSES: int = len(CLASS_NAMES)

LABEL_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL: dict[int, str] = {idx: name for name, idx in LABEL_TO_IDX.items()}

# Human-readable names for reports and plots
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "akiec": "Actinic keratoses / IEC",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Required metadata columns
METADATA_REQUIRED_COLUMNS: list[str] = [
    "image_id",
    "lesion_id",
    "dx",
]

# Image ID column may appear as image_id in HAM10000 CSV
IMAGE_ID_COLUMN: str = "image_id"
LESION_ID_COLUMN: str = "lesion_id"
DX_COLUMN: str = "dx"
