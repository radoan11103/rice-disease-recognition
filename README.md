# Rice Disease Recognition - Cross-Year Generalization Study

This project studies a practical deployment question: can a rice-leaf disease classifier trained on historical images generalize to future-year field images?

The experiment trains deep models on rice disease data from 2021-2025 and evaluates them on a held-out 2026 test set. The current research pipeline is multi-seed: each model/mode is trained across 10 random seeds so the final comparison is based on mean and variation, not a single lucky run.

## Study Design

Classes:

- brown_spot
- healthy
- rice_blast
- rice_tungro

Temporal split:

| Years | Usage |
|---|---|
| 2021-2025 | Train + validation |
| 2026 | Future-year test |

Models:

| Model | Type | Pretraining |
|---|---|---|
| ResNet50 | CNN | ImageNet |
| DenseNet121 | CNN | ImageNet |
| EfficientNet-B0 | CNN | ImageNet |
| DINOv2 ViT-S/14 | Transformer | Self-supervised |

Fine-tuning modes:

| Mode | Description |
|---|---|
| full | Update the entire network |
| partial | Freeze the backbone and train only the classifier head |

Current experiment count:

```text
10 seeds x 4 models x 2 fine-tuning modes = 80 training runs
```

The 10 seeds used in the current results are:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

## Main Findings From The 10-Seed Run

The multi-seed summary is written to `results/MULTI_SEED_SUMMARY.md` and `results/multi_seed_summary.csv`.

| Model | Mode | N | Test Macro-F1 | Val Macro-F1 | Gap Macro-F1 | Retention |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | full | 10 | 43.82 +/- 1.37 | 93.02 +/- 0.32 | 49.20 +/- 1.35 | 47.11 +/- 1.45 |
| ResNet50 | partial | 10 | 39.24 +/- 0.47 | 80.83 +/- 0.33 | 41.59 +/- 0.55 | 48.54 +/- 0.59 |
| DenseNet121 | full | 10 | 44.94 +/- 1.75 | 92.86 +/- 0.24 | 47.92 +/- 1.80 | 48.40 +/- 1.91 |
| DenseNet121 | partial | 10 | 36.48 +/- 1.52 | 74.12 +/- 0.55 | 37.64 +/- 1.69 | 49.22 +/- 2.13 |
| EfficientNet-B0 | full | 10 | 44.64 +/- 0.99 | 92.30 +/- 0.28 | 47.65 +/- 0.97 | 48.37 +/- 1.05 |
| EfficientNet-B0 | partial | 10 | 33.32 +/- 0.69 | 78.96 +/- 0.37 | 45.63 +/- 0.93 | 42.21 +/- 0.98 |
| DINOv2 | full | 10 | 38.03 +/- 0.98 | 85.04 +/- 0.94 | 47.01 +/- 1.18 | 44.73 +/- 1.13 |
| DINOv2 | partial | 10 | 47.11 +/- 1.26 | 81.74 +/- 0.61 | 34.63 +/- 1.71 | 57.65 +/- 1.86 |

Key observations:

- Validation Macro-F1 on historical data is high for fully fine-tuned CNNs, but future-year Macro-F1 drops sharply on 2026 images.
- DINOv2 with partial fine-tuning has the strongest future-year Macro-F1 and future retention across the 10-seed experiment.
- Multi-seed reporting matters because some per-seed values move by several Macro-F1 points.

## Repository Layout

```text
config.py                 central paths, classes, hyperparameters
run_pipeline.py           multi-seed pipeline and aggregation entry point
run_all.bat / run_all.ps1 Windows wrappers around run_pipeline.py
src/preprocess.py         builds the cross-year dataset
src/train.py              trains one model/mode/seed
src/gradcam.py            Grad-CAM comparison for full vs partial checkpoints
src/analyze.py            per-seed report and plots
src/evaluate.py           evaluation-only helper for existing checkpoints
src/select_difficulty_subset.py optional difficulty-stratified subset selector
```

Ignored local folders:

```text
data/          processed dataset
Rice Disease/  raw dataset
.venv/         virtual environment
```

## Installation

Recommended environment:

- Windows
- Python 3.10 or 3.11
- NVIDIA GPU, tested on an RTX 3050 Laptop GPU with 4 GB VRAM

Setup:

```bat
setup.bat
```

Manual setup:

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Dataset

Expected raw-data structure:

```text
Rice Disease/
  2021/
  2022/
  2023/
  2024/
  2025/
  2026/
```

Recognized raw class folder names:

- Brown Spot
- Healthy Leaf
- Rice Blast
- Rice Tungro

Set `RICE_RAW_DIR` in `run_all.bat` or `run_all.ps1` so it points to the folder that directly contains the year folders.

## Running The Current 10-Seed Pipeline

Full 10-seed run:

```bat
run_all.bat --runs 10 --start-seed 42
```

Equivalent explicit seed command:

```bat
python run_pipeline.py --seeds 42 43 44 45 46 47 48 49 50 51
```

If the dataset has already been preprocessed:

```bat
python run_pipeline.py --skip-preprocess --seeds 42 43 44 45 46 47 48 49 50 51
```

To rebuild only the multi-seed summary from completed seed folders, without training anything:

```bat
python run_pipeline.py --aggregate-only --seeds 42 43 44 45 46 47 48 49 50 51
```

This is useful when an older seed run already exists and you want to include it in the final statistics.

## Output Layout

Each seed is isolated in its own result folder:

```text
results/
  experiment_seed_42/
    resnet50_full/
    resnet50_partial/
    densenet121_full/
    densenet121_partial/
    efficientnet_b0_full/
    efficientnet_b0_partial/
    dinov2_full/
    dinov2_partial/
    analysis/
    gradcam/
    SUMMARY.md
  experiment_seed_43/
  ...
  experiment_seed_51/
  MULTI_SEED_SUMMARY.md
  multi_seed_summary.csv
  multi_seed_summary.json
```

Important files:

| File | Purpose |
|---|---|
| `results/MULTI_SEED_SUMMARY.md` | Human-readable 10-seed summary |
| `results/multi_seed_summary.csv` | Per-seed metrics table for statistical analysis |
| `results/multi_seed_summary.json` | Full machine-readable aggregate |
| `results/experiment_seed_<seed>/SUMMARY.md` | Per-seed report |
| `results/experiment_seed_<seed>/<model>_<mode>/summary.json` | Metrics for one trained run |
| `results/experiment_seed_<seed>/<model>_<mode>/best.pth` | Best checkpoint for that run |

## Individual Commands

Preprocess only:

```bat
python -m src.preprocess
```

Train one model/mode/seed:

```bat
set RICE_RESULTS_DIR=results\experiment_seed_42
set RICE_EXPERIMENT_NAME=.
python -m src.train --model resnet50 --mode full --seed 42
```

Run Grad-CAM for one model inside a seed folder:

```bat
set RICE_RESULTS_DIR=results\experiment_seed_42
set RICE_EXPERIMENT_NAME=.
python -m src.gradcam --model resnet50
```

Analyze one seed folder:

```bat
set RICE_RESULTS_DIR=results\experiment_seed_42
set RICE_EXPERIMENT_NAME=.
python -m src.analyze
```

## Method Summary

Preprocessing:

- EXIF orientation correction
- bilateral denoising
- CLAHE contrast enhancement
- center crop
- resize to 224 x 224
- JPEG quality 95

Training:

- AdamW optimizer
- cosine learning-rate schedule
- mixed precision training
- gradient accumulation
- class-weighted cross entropy
- label smoothing
- early stopping
- checkpoint selection by validation Macro-F1
- deterministic seeding per run

Metrics:

- accuracy
- balanced accuracy
- Macro-F1
- precision and recall
- per-class F1
- confusion matrix
- generalization gap
- future retention

## GitHub Notes

Large datasets, virtual environments, checkpoints, and full generated experiment folders should not be pushed to GitHub unless you intentionally use Git LFS or a release artifact. For normal repository updates, commit the source code, documentation, and compact summary files.

Recommended files to commit for the current update:

```text
README.md
HOW_TO_RUN.md
config.py
run_pipeline.py
src/train.py
src/utils.py
results/MULTI_SEED_SUMMARY.md
results/multi_seed_summary.csv
results/multi_seed_summary.json
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `run_pipeline.py` not found | Make sure the file exists in the project root and run the command from that root folder. |
| CUDA out of memory | Reduce the relevant value in `BATCH_SIZE` in `config.py`. |
| GPU not detected | Update the NVIDIA driver or pass `--allow-cpu` for a slow CPU run. |
| Raw dataset not found | Fix `RICE_RAW_DIR` in `run_all.bat` or `run_all.ps1`. |
| Need to include an old seed run | Put it under `results/experiment_seed_<seed>/` and run `python run_pipeline.py --aggregate-only --seeds ...`. |
