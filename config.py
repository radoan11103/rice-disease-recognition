"""Central configuration for the Rice Disease cross-year recognition project.

Every stage of the pipeline (preprocessing, training, evaluation, Grad-CAM,
analysis) reads from this single file. That is what guarantees the research
requirement: *all models train on exactly the same data and are tested on
exactly the same data*.

Paths can be overridden with environment variables (handy on Windows):
    set RICE_RAW_DIR=R:\\Anti gravity\\Rice Disease Project\\Rice Disease
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder that directly contains the year subfolders (2021, 2022, ... 2026).
# See directory_structure.txt for the expected layout.
RAW_DATA_DIR = Path(os.environ.get(
    "RICE_RAW_DIR",
    PROJECT_ROOT / "data" / "Rice Disease",
))

# Where the cleaned, cross-year dataset is written (train/ val/ test/).
PROCESSED_DIR = Path(os.environ.get(
    "RICE_PROCESSED_DIR",
    PROJECT_ROOT / "data" / "RiceCrossYear",
))

# Where every training run, Grad-CAM figure and the final report are saved.
RESULTS_DIR = Path(os.environ.get(
    "RICE_RESULTS_DIR",
    PROJECT_ROOT / "results",
))

# ---------------------------------------------------------------------------
# CLASSES   (raw folder name  ->  canonical label)
# ---------------------------------------------------------------------------
CLASS_MAPPING = {
    "Healthy Leaf": "healthy",
    "Rice Blast":   "rice_blast",
    "Brown Spot":   "brown_spot",
    "Rice Tungro":  "rice_tungro",
}
# Fixed, sorted class order used everywhere (ImageFolder also sorts like this).
CLASS_NAMES = sorted(CLASS_MAPPING.values())
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# CROSS-YEAR SPLIT  --  the heart of the research question:
#   train on the PAST (2021-2025), predict the FUTURE (2026)
# ---------------------------------------------------------------------------
TRAIN_YEARS = ["2021", "2022", "2023", "2024", "2025"]
TEST_YEARS = ["2026"]

VAL_SPLIT = 0.15               # fraction of the 2021-2025 pool kept for validation
TEST_IMAGES_PER_CLASS = 1000   # balanced future-year test set (capped per class)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------------------------

def experiment_dir() -> Path:
    """Directory for the current experiment."""
    experiment_name = os.environ.get(
        "RICE_EXPERIMENT_NAME",
        f"experiment_seed_{RANDOM_SEED}",
    )
    return RESULTS_DIR / experiment_name

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# PREPROCESSING  (Standard SOTA: EXIF fix -> denoise -> CLAHE -> crop -> resize)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224               # every model trains and tests at this resolution
JPEG_QUALITY = 95
CLAHE_CLIP = 2.0               # CLAHE contrast-limit (LAB L-channel)
CLAHE_GRID = 8                 # CLAHE tile grid (GRID x GRID)
BILATERAL_D = 5                # edge-preserving denoise neighbourhood
BILATERAL_SIGMA = 50           # bilateral colour/space sigma
PREPROCESS_WORKERS = int(os.environ.get("RICE_PP_WORKERS", "4"))

# ---------------------------------------------------------------------------
# TRAINING  --  tuned for an RTX 3050 Mobile (4 GB VRAM) + 16 GB RAM laptop
# ---------------------------------------------------------------------------
MODELS = ["resnet50", "densenet121", "efficientnet_b0", "dinov2"]
FINETUNE_MODES = ["full", "partial"]   # full = all weights, partial = head only

EPOCHS = 15
EARLY_STOP_PATIENCE = 5        # stop if val macro-F1 stalls this many epochs
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

LR_FULL = 1e-4                 # learning rate when fine-tuning the whole network
LR_PARTIAL = 1e-3              # learning rate when only the head is trained

# Per-model batch sizes kept small so they fit in 4 GB of VRAM.
# Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS. Lower these if you hit
# "CUDA out of memory".
BATCH_SIZE = {
    "resnet50":        16,
    "densenet121":     12,
    "efficientnet_b0": 24,
    "dinov2":          16,
}
GRAD_ACCUM_STEPS = 2

# DataLoader workers. 4 keeps the 6-core/12-thread CPU busy without exhausting
# the 16 GB of RAM. Drop to 2 if the machine starts swapping.
NUM_WORKERS = int(os.environ.get("RICE_NUM_WORKERS", "4"))

USE_AMP = True                 # mixed precision -- essential to fit 4 GB of VRAM

# ---------------------------------------------------------------------------
# GRAD-CAM / ANALYSIS
# ---------------------------------------------------------------------------
GRADCAM_IMAGES_PER_CLASS = 5   # sample images visualised per class
GRADCAM_SUMMARY_IMAGES = 5     # images in the single gradcam_summary.png grid
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def run_dir(model: str, mode: str) -> Path:
    """Directory that holds every artefact for one (model, mode) training run."""
    return experiment_dir() / f"{model}_{mode}"