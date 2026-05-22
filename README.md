````markdown
# Rice Disease Recognition — Cross-Year Generalization Study

> ⚠️ Models achieving very high historical validation accuracy may still fail severely on future-year field data.

Can a rice-leaf disease classifier **trained on the past predict the future?**

This project evaluates how well deep-learning models trained on historical rice disease data (**2021–2025**) generalize to a completely unseen future-year dataset (**2026**). The study focuses on **temporal robustness** — how much performance is lost when models encounter future real-world field images.

The project compares:

- **Full fine-tuning** vs **partial fine-tuning**
- **CNNs** vs **self-supervised transformers**
- Cross-year generalization behavior
- Grad-CAM attention shifts between training strategies

The pipeline trains 4 architectures in 2 fine-tuning modes:

| Model | Type | Pretraining |
|---|---|---|
| ResNet50 | CNN | ImageNet |
| DenseNet121 | CNN | ImageNet |
| EfficientNet-B0 | CNN | ImageNet |
| DINOv2 (ViT-S/14) | Transformer | Self-supervised |

Total experiments:

```text
4 models × 2 tuning modes = 8 training runs
```

Classes:
- brown_spot
- healthy
- rice_blast
- rice_tungro

---

# TL;DR

CNNs achieved over **92% validation Macro-F1** on historical data but lost more than half of their effective performance on unseen future-year images.  
Partial fine-tuning of **DINOv2** achieved the strongest future robustness and the highest future retention.

---

# Why This Matters

Real agricultural deployments face changing environmental conditions over time:
- different lighting
- different devices
- seasonal variation
- geographic variation
- changing disease appearance

Models that perform well on historical benchmark datasets may still fail under future real-world field conditions. This project studies that temporal reliability problem directly using a strict cross-year evaluation protocol.

---

# Key Experimental Findings

The experiments revealed a severe temporal generalization gap between historical rice disease data (**2021–2025**) and future-year field data (**2026**).

| Model | Mode | Validation Macro-F1 | Future Macro-F1 | Future Retention |
|---|---|---:|---:|---:|
| ResNet50 | Full | 92.46 | 45.12 | 48.8% |
| DenseNet121 | Full | 93.03 | 44.49 | 47.8% |
| EfficientNet-B0 | Full | 92.25 | 45.25 | 49.0% |
| DINOv2 | Partial | 82.31 | **47.86** | **58.1%** |

## Main Observations

- Models achieved very high validation performance on historical data but experienced substantial degradation on unseen future-year images.
- Fully fine-tuned CNNs produced the highest validation scores but did not generalize best to future-year data.
- Partial fine-tuning of DINOv2 achieved the strongest future robustness.
- The results suggest that frozen self-supervised transformer representations preserve more transferable features under temporal distribution shift than fully fine-tuned CNNs.
- Average future retention across all experiments was approximately **48.6%**, highlighting the difficulty of real-world deployment across years.

---

# Example Outputs

## Generalization Gap Analysis

```text
results/analysis/generalization_gap.png
```

## Grad-CAM Comparison

```text
results/gradcam/dinov2/gradcam_summary.png
```

## Per-Class Future Robustness

```text
results/analysis/per_class_f1.png
```

---

# Research Questions

## 1. Cross-Year Generalization

Train on:

```text
2021–2025
```

Test on:

```text
2026
```

The project measures:
- generalization gap
- future retention
- temporal robustness
- per-class degradation

---

## 2. Full vs Partial Fine-Tuning

Each model is trained twice:

| Mode | Description |
|---|---|
| Full | Entire network is updated |
| Partial | Backbone frozen, classifier head trained |

The study compares:
- validation accuracy
- future-year robustness
- Grad-CAM attention behavior
- feature transferability

---

# Dataset Structure

Expected raw-data structure:

```text
Rice Disease/
├── 2021/
├── 2022/
├── 2023/
├── 2024/
├── 2025/
└── 2026/
```

Recognized classes:
- Brown Spot
- Healthy Leaf
- Rice Blast
- Rice Tungro

Temporal split:

| Years | Usage |
|---|---|
| 2021–2025 | Train + Validation |
| 2026 | Future Test |

---

# Hardware

The pipeline was designed and optimized for:

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) |
| CPU | Ryzen 5 5600H |
| RAM | 16 GB |
| OS | Windows |

Optimizations:
- mixed precision (AMP)
- gradient accumulation
- reduced batch sizes
- sequential subprocess execution

---

# Project Structure

```text
RICE/
├── config.py
├── run_pipeline.py
├── run_all.bat
├── run_all.ps1
├── setup.bat
├── requirements.txt
├── README.md
├── HOW_TO_RUN.md
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── models.py
│   ├── engine.py
│   ├── metrics.py
│   ├── train.py
│   ├── gradcam.py
│   ├── analyze.py
│   └── utils.py
├── data/                  # ignored from GitHub
└── results/               # experiment outputs
```

---

# Installation

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU recommended
- Updated NVIDIA driver

---

## Setup

```bat
setup.bat
```

This:
- creates `.venv`
- installs the CUDA PyTorch build
- installs dependencies
- checks CUDA availability

Manual alternative:

```bat
python -m venv .venv
.\.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

# Running the Pipeline

## Full Pipeline

```bat
run_all.bat
```

Pipeline stages:

```text
Preprocessing
→ Training
→ Grad-CAM
→ Analysis
→ Final Report
```

---

## Useful Options

Skip preprocessing:

```bat
run_all.bat --skip-preprocess
```

Run only selected models:

```bat
run_all.bat --models resnet50 dinov2
```

Run only one tuning mode:

```bat
run_all.bat --modes full
```

Continue after failures:

```bat
run_all.bat --continue-on-error
```

---

# Individual Stages

Preprocess dataset:

```bat
python -m src.preprocess
```

Train one model:

```bat
python -m src.train --model resnet50 --mode full
```

Run Grad-CAM:

```bat
python -m src.gradcam --model resnet50
```

Generate analysis report:

```bat
python -m src.analyze
```

---

# Methodology

## Preprocessing Pipeline

The preprocessing stage includes:
- EXIF orientation correction
- bilateral denoising
- CLAHE contrast enhancement
- center crop
- resize to 224×224
- JPEG quality 95

---

## Training Strategy

Techniques used:
- AdamW optimizer
- cosine learning-rate scheduling
- mixed precision (AMP)
- gradient accumulation
- class-weighted loss
- label smoothing
- early stopping
- macro-F1 checkpoint selection

---

## Evaluation Metrics

The project evaluates:
- Accuracy
- Balanced Accuracy
- Macro-F1
- Precision / Recall
- Per-class F1
- Confusion matrices
- Generalization gap
- Future retention

---

# Grad-CAM Analysis

Grad-CAM compares how full and partial fine-tuning strategies focus on leaf regions.

Each visualization includes:

```text
Input Image
→ Partial Attention
→ Full Attention
→ Difference Map
```

Metrics:
- CAM correlation
- Hot-region IoU
- MAE
- centroid shift

Interesting observation:

| Model | CAM Correlation |
|---|---:|
| ResNet50 | 0.459 |
| DenseNet121 | 0.447 |
| EfficientNet-B0 | 0.321 |
| DINOv2 | 0.044 |

DINOv2 showed the largest attention redistribution between tuning modes.

---

# Outputs

```text
results/
├── SUMMARY.md
├── summary.csv
├── analysis/
├── gradcam/
├── confusion matrices
├── training curves
└── per-model experiment folders
```

Important outputs:

| File | Description |
|---|---|
| SUMMARY.md | Final report |
| summary.csv | All experiment metrics |
| generalization_gap.png | Temporal degradation |
| partial_vs_full_f1.png | Tuning comparison |
| per_class_f1.png | Per-class robustness |
| gradcam_summary.png | Attention comparison |

---

# Key Conclusions

The experiments demonstrate that:

- High validation accuracy does not guarantee future robustness.
- Temporal distribution shift causes severe performance degradation.
- CNNs strongly overfit historical distributions under full fine-tuning.
- Frozen self-supervised transformer representations generalize better to future-year field data.
- Future-year robustness should be considered explicitly in agricultural AI deployment.

---

# Troubleshooting

| Problem | Fix |
|---|---|
| CUDA out of memory | Reduce batch size in `config.py` |
| GPU not detected | Update NVIDIA driver |
| Dataset not found | Fix `RICE_RAW_DIR` |
| Machine slows down | Reduce `RICE_NUM_WORKERS=2` |
| Rebuild skipped | Use `--skip-preprocess` |

---

# Citation

If you use this project or build upon it, please cite or reference the repository.

---

# License

This project is intended for research and educational purposes.
````
