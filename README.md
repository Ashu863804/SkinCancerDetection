# SkinCancerDetection — HAM10000 Multiclass Classification

Production-grade, modular deep learning pipeline for **7-class skin lesion classification** on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) dataset.

Designed for: **Cursor IDE → GitHub → Kaggle GPU** workflow.

> **Disclaimer:** This project is for research and education only. It is **not** a medical device and must not be used for clinical diagnosis.

---

## Features

- Lesion-level stratified train/val/test splits (no data leakage)
- EfficientNetV2-S transfer learning (ConvNeXt-Tiny, ResNet50 also supported)
- Class imbalance: Focal Loss + weighted sampler + class weights
- Albumentations augmentation with minority-class boost
- Mixed precision training, cosine scheduler, early stopping
- Full metrics: macro-F1, balanced accuracy, confusion matrix, ROC-AUC OVR
- Grad-CAM explainability
- Thin Kaggle notebook controller over modular `src/` package

---

## Project structure

```
SkinCancerDetection/
├── configs/default.yaml      # All hyperparameters and paths
├── notebooks/
│   └── ham10000_pipeline.ipynb
├── src/
│   ├── config.py             # YAML loader + Kaggle/local paths
│   ├── constants.py          # 7 class names and label maps
│   ├── data/                 # Dataset, DataModule, augmentations
│   ├── models/               # Model factory
│   ├── losses/               # Focal loss
│   ├── train/                # Engine + callbacks
│   ├── evaluate/             # Metrics + plots
│   ├── explain/              # Grad-CAM
│   ├── inference/            # Predictor
│   ├── train.py              # train_model() API
│   ├── evaluate.py           # evaluate_model() API
│   └── visualize.py          # run_gradcam(), predict_image()
├── tests/
├── saved_models/             # Checkpoints (gitignored)
└── outputs/                  # Logs, figures, reports (gitignored)
```

---

## Setup (local)

```bash
cd SkinCancerDetection
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Dataset layout (local)

Place HAM10000 under `data/raw/`:

```
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

Or set paths in `configs/default.yaml`.

---

## Quick start (Python)

```python
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import run_gradcam, predict_image, plot_metrics

# Train
train_model()

# Evaluate on test set
evaluate_model(split="test")

# Plot training curves
plot_metrics()

# Grad-CAM on one image
run_gradcam("data/raw/HAM10000_images_part_1/ISIC_0024306.jpg")

# Inference
predict_image("path/to/lesion.jpg")
```

### Debug smoke run (small subset)

```python
train_model(overrides={
    "data.debug_max_samples": 200,
    "training.epochs": 2,
    "training.batch_size": 16,
})
```

---

## Kaggle workflow

### 1. Push this repo to GitHub

### 2. Create a Kaggle notebook with GPU enabled

### 3. Manual setup cells (you add these first)

```python
!git clone https://github.com/YOUR_USERNAME/SkinCancerDetection.git
%cd SkinCancerDetection
!pip install -q -r requirements.txt
```

### 4. Add HAM10000 dataset to the notebook

Attach the [HAM10000 Kaggle dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). Update slug in `configs/default.yaml` if your path differs.

### 5. Run pipeline cells from `notebooks/ham10000_pipeline.ipynb`

The notebook sets `SKIN_CANCER_ENV=kaggle` and calls `train_model()`, `evaluate_model()`, etc.

---

## Configuration

Edit [`configs/default.yaml`](configs/default.yaml) or pass overrides:

```python
import os
os.environ["SKIN_CANCER_ENV"] = "kaggle"

train_model(overrides={
    "training.epochs": 30,
    "training.batch_size": 32,
    "model.name": "efficientnet_v2_s",
})
```

**Environment variable:** `SKIN_CANCER_ENV=local|kaggle` switches data paths automatically.

---

## Primary metrics

| Metric | Why |
|--------|-----|
| **Macro F1** | Checkpoint selection; treats all classes equally |
| **Balanced accuracy** | Corrects for class imbalance |
| **Per-class recall** | Critical for melanoma (`mel`) and rare classes |
| Accuracy | Reported but not used for model selection |

---

## Tests

```bash
cd SkinCancerDetection
pytest tests/ -v
```

---

## License

MIT (add your license as needed). HAM10000 dataset has its own terms on Kaggle.
