# Statistical Analysis of Multi-Seed Experiments

## Analysis Overview

Statistical analysis was performed using the aggregated multi-seed result file:

```text
results/multi_seed_summary.csv
```

The analysis used the cross-year test macro F1 score (`test_f1`) as the primary performance metric. The experiment contains 10 seed-matched runs, with each seed evaluated under 8 model/mode conditions:

```text
10 seeds x 8 conditions = 80 total result rows
```

Each condition therefore has `n = 10` paired observations. The seeds used were:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

Because the same seeds were used for every model/mode condition, paired statistical tests were used for direct comparisons.

## Methods

For each condition, the mean, standard deviation, and 95% confidence interval were calculated over the 10 seed-level test macro F1 scores.

An overall Friedman test was used to evaluate whether statistically significant differences existed across the 8 paired model/mode conditions. The Friedman test is appropriate here because the same seeds were reused across all conditions.

Pairwise comparisons were then performed using the Wilcoxon signed-rank test. This non-parametric paired test was selected because the sample size is small (`n = 10`) and normality should not be assumed. Holm-Bonferroni correction was applied to control the family-wise error rate across multiple pairwise comparisons.

Effect size was reported using Cohen's dz for paired comparisons:

```text
Cohen's dz = mean paired difference / standard deviation of paired differences
```

## Descriptive Statistics

Primary metric: cross-year test macro F1 score.

| Condition | n | Mean F1 | SD | 95% CI |
|---|---:|---:|---:|---:|
| ResNet50 full | 10 | 43.822 | 1.443 | 42.790-44.854 |
| ResNet50 partial | 10 | 39.236 | 0.494 | 38.883-39.590 |
| DenseNet121 full | 10 | 44.941 | 1.846 | 43.620-46.261 |
| DenseNet121 partial | 10 | 36.480 | 1.608 | 35.330-37.630 |
| EfficientNet-B0 full | 10 | 44.642 | 1.042 | 43.896-45.388 |
| EfficientNet-B0 partial | 10 | 33.323 | 0.723 | 32.806-33.841 |
| DINOv2 full | 10 | 38.035 | 1.029 | 37.299-38.770 |
| DINOv2 partial | 10 | **47.112** | 1.324 | **46.165-48.060** |

DINOv2 partial achieved the highest mean cross-year test macro F1 score.

## Overall Statistical Test

A Friedman test was conducted across all 8 paired conditions.

| Test | Statistic | p-value |
|---|---:|---:|
| Friedman test | 64.0000 | 2.388e-11 |

The Friedman test indicates a statistically significant difference among the 8 model/mode conditions.

## Pairwise Statistical Comparisons

Pairwise comparisons were performed using the Wilcoxon signed-rank test. Holm-Bonferroni correction was applied to the Wilcoxon p-values.

| Comparison | Mean Difference | Wilcoxon p | Holm-corrected p | Paired t-test p | Cohen's dz |
|---|---:|---:|---:|---:|---:|
| DINOv2 partial vs ResNet50 partial | +7.876 | 0.001953 | 0.013672 | 2.051e-09 | 7.483 |
| DINOv2 partial vs DenseNet121 partial | +10.632 | 0.001953 | 0.013672 | 9.152e-09 | 6.319 |
| DINOv2 partial vs EfficientNet-B0 partial | +13.789 | 0.001953 | 0.013672 | 4.101e-11 | 11.606 |
| DINOv2 partial vs DINOv2 full | +9.078 | 0.001953 | 0.013672 | 6.897e-09 | 6.525 |
| ResNet50 full vs ResNet50 partial | +4.586 | 0.001953 | 0.013672 | 7.413e-06 | 2.896 |
| DenseNet121 full vs DenseNet121 partial | +8.460 | 0.001953 | 0.013672 | 5.008e-06 | 3.037 |
| EfficientNet-B0 full vs EfficientNet-B0 partial | +11.319 | 0.001953 | 0.013672 | 7.496e-10 | 8.380 |

All listed pairwise comparisons remained statistically significant after Holm-Bonferroni correction.

## Generalization Gap Analysis

The F1 generalization gap was calculated as the difference between validation macro F1 and cross-year test macro F1. Lower values indicate better retention of performance under the cross-year distribution shift.

| Condition | Mean F1 Gap | SD |
|---|---:|---:|
| ResNet50 full | 49.202 | 1.421 |
| ResNet50 partial | 41.592 | 0.577 |
| DenseNet121 full | 47.917 | 1.900 |
| DenseNet121 partial | 37.638 | 1.778 |
| EfficientNet-B0 full | 47.654 | 1.028 |
| EfficientNet-B0 partial | 45.634 | 0.979 |
| DINOv2 full | 47.006 | 1.246 |
| DINOv2 partial | **34.626** | 1.806 |

DINOv2 partial produced the lowest mean F1 generalization gap, indicating the strongest cross-year retention among the tested conditions.

## Suggested Thesis/Report Wording

Across 10 seed-matched runs, DINOv2 partial achieved the highest cross-year macro F1 score, with a mean of 47.112 and a standard deviation of 1.324. The 95% confidence interval was 46.165-48.060. A Friedman test showed significant differences across the 8 model/mode conditions, chi-square = 64.000, p < 0.001. Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction confirmed that DINOv2 partial significantly outperformed the other partial-training models. DINOv2 partial also showed the lowest mean F1 generalization gap, 34.626 +/- 1.806, supporting its stronger cross-year generalization performance.

## Interpretation

The results indicate that model/mode choice has a statistically significant effect on cross-year macro F1 performance. Among the tested settings, DINOv2 partial was the strongest condition for cross-year evaluation. Its advantage was statistically significant in the key seed-paired comparisons and was supported by a lower generalization gap.

