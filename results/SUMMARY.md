# Rice Disease Recognition -- Cross-Year Results

**Research question:** *Training on the past (2021-2025), how well can the models predict the future (2026)?*

- **VAL** = held-out 15% of 2021-2025 (in-distribution).
- **TEST** = year 2026, never seen in training (the future).
- **Generalization gap** = VAL macro-F1 - TEST macro-F1.
- **Future retention** = TEST macro-F1 / VAL macro-F1.

## Headline

- Best future-year model: **dinov2 (partial)** -- TEST macro-F1 **47.86%**.
- Average generalization gap across all runs: **43.68** macro-F1 points.
- Average future retention: **48.6%** of in-distribution performance.

## All runs

| Model | Mode | VAL F1 | TEST F1 | TEST acc | Gen. gap | Retention | Train min |
|---|---|---|---|---|---|---|---|
| densenet121 | full | 93.03 | 44.49 | 47.17 | +48.54 | 47.8% | 20.8 |
| densenet121 | partial | 73.97 | 35.99 | 36.58 | +37.98 | 48.7% | 8.9 |
| dinov2 | full | 85.58 | 37.92 | 39.27 | +47.66 | 44.3% | 15.6 |
| dinov2 | partial | 82.31 | 47.86 | 50.88 | +34.45 | 58.1% | 4.8 |
| efficientnet_b0 | full | 92.25 | 45.25 | 48.00 | +47.00 | 49.0% | 9.7 |
| efficientnet_b0 | partial | 78.81 | 34.44 | 37.38 | +44.37 | 43.7% | 5.9 |
| resnet50 | full | 92.46 | 45.12 | 47.35 | +47.35 | 48.8% | 12.7 |
| resnet50 | partial | 81.34 | 39.25 | 39.95 | +42.09 | 48.3% | 6.2 |

## Full vs partial fine-tuning (TEST macro-F1)

| Model | Full | Partial | Full - Partial |
|---|---|---|---|
| densenet121 | 44.49 | 35.99 | +8.50 |
| dinov2 | 37.92 | 47.86 | -9.94 |
| efficientnet_b0 | 45.25 | 34.44 | +10.81 |
| resnet50 | 45.12 | 39.25 | +5.87 |

## Confusion-matrix shift -- biggest change per model

Row-normalised TEST confusion matrix, full minus partial. A positive value means full fine-tuning sends more of that true class to that prediction.

| Model | True class | Predicted as | Change |
|---|---|---|---|
| resnet50 | brown_spot | brown_spot | +0.19 |
| densenet121 | healthy | healthy | +0.33 |
| efficientnet_b0 | healthy | healthy | +0.25 |
| dinov2 | brown_spot | brown_spot | -0.45 |

The highlighted (green box) cell in each `analysis/cm_delta_<model>.png` marks this change.

## Grad-CAM: how far attention moves (partial -> full)

| Model | CAM correlation | Hot-region IoU | Centroid shift |
|---|---|---|---|
| resnet50 | 0.459 | 0.373 | 0.085 |
| densenet121 | 0.447 | 0.400 | 0.091 |
| efficientnet_b0 | 0.321 | 0.331 | 0.144 |
| dinov2 | 0.044 | 0.159 | 0.103 |

- **CAM correlation** near 1.0 means full and partial fine-tuning look at the same regions; lower means fine-tuning the backbone substantially relocated the model's focus.
- **Hot-region IoU** = overlap of the top-25% most attended pixels.
- **Centroid shift** = how far the attention centre of mass moved (fraction of the image diagonal).
- See `gradcam/<model>/gradcam_summary.png` for the 5-image partial -> full progression.

## Figures

**Performance**
- `analysis/val_vs_test_f1.png` -- in-distribution vs future.
- `analysis/generalization_gap.png` -- accuracy lost on 2026.
- `analysis/partial_vs_full_f1.png` -- fine-tuning strategies.
- `analysis/per_class_f1.png` -- per-class F1 for every run.
- `<model>_<mode>/training_curves.png` -- loss & F1 per epoch.
- `<model>_<mode>/confusion_matrix_test.png` -- per run.

**Fine-tuning & attention**
- `analysis/cm_delta_<model>.png` -- confusion-matrix shift, biggest-change cell highlighted.
- `analysis/attention_change.png` -- how far Grad-CAM focus moves, partial -> full.
- `gradcam/<model>/gradcam_summary.png` -- 5 sample images, partial -> full Grad-CAM progression.
- `gradcam/<model>/NN_<class>_<img>.png` -- per-image input | partial | full | change map.
