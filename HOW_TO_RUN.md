# How to Run - Quick Start

This guide matches the current multi-seed research pipeline. The current reported experiment uses 10 seeds:

```text
42, 43, 44, 45, 46, 47, 48, 49, 50, 51
```

That gives:

```text
10 seeds x 4 models x 2 fine-tuning modes = 80 training runs
```

For full context, see `README.md`.

## 1. Setup

From the project root:

```bat
setup.bat
```

This creates `.venv`, installs PyTorch and the project dependencies, and checks CUDA.

## 2. Point the project at the dataset

Open `run_all.bat` and set `RICE_RAW_DIR` to the folder that directly contains the year folders:

```bat
set RICE_RAW_DIR=R:\Anti gravity\Rice Disease Project\Rice Disease
```

Expected layout:

```text
Rice Disease/
  2021/
  2022/
  2023/
  2024/
  2025/
  2026/
```

## 3. Smoke test

Run one small training path before launching the long experiment:

```bat
run_all.bat --skip-gradcam --skip-analyze --models efficientnet_b0 --modes partial --seeds 42
```

## 4. Run the full 10-seed experiment

```bat
run_all.bat --runs 10 --start-seed 42
```

Equivalent explicit form:

```bat
python run_pipeline.py --seeds 42 43 44 45 46 47 48 49 50 51
```

If preprocessing is already done:

```bat
python run_pipeline.py --skip-preprocess --seeds 42 43 44 45 46 47 48 49 50 51
```

## 5. Rebuild only the 10-seed summary

Use this when the seed folders already exist and you only need to regenerate the aggregate files:

```bat
python run_pipeline.py --aggregate-only --seeds 42 43 44 45 46 47 48 49 50 51
```

This does not train, evaluate, or run Grad-CAM. It only reads:

```text
results\experiment_seed_<seed>\<model>_<mode>\summary.json
```

and rewrites:

```text
results\MULTI_SEED_SUMMARY.md
results\multi_seed_summary.csv
results\multi_seed_summary.json
```

## 6. Output locations

Each seed has its own folder:

```text
results\experiment_seed_42\
results\experiment_seed_43\
...
results\experiment_seed_51\
```

Main aggregate files:

| File | Shows |
|---|---|
| `results\MULTI_SEED_SUMMARY.md` | Mean +/- std across the 10 seeds |
| `results\multi_seed_summary.csv` | Per-seed metrics for statistical analysis |
| `results\multi_seed_summary.json` | Full machine-readable aggregate |

Per-seed files:

| File | Shows |
|---|---|
| `results\experiment_seed_<seed>\SUMMARY.md` | Report for one seed |
| `results\experiment_seed_<seed>\summary.csv` | Per-seed table across all 8 model/mode runs |
| `results\experiment_seed_<seed>\<model>_<mode>\summary.json` | Metrics for one training run |
| `results\experiment_seed_<seed>\<model>_<mode>\best.pth` | Best checkpoint |

## Useful variations

Run only selected models:

```bat
python run_pipeline.py --skip-preprocess --models resnet50 dinov2 --seeds 42 43
```

Run only one fine-tuning mode:

```bat
python run_pipeline.py --skip-preprocess --modes partial --seeds 42 43 44
```

Continue even if one step fails:

```bat
python run_pipeline.py --continue-on-error --seeds 42 43 44 45 46 47 48 49 50 51
```

## Updating GitHub

Before pushing, check what changed:

```bat
git status
```

Recommended source/doc summary commit:

```bat
git add README.md HOW_TO_RUN.md config.py run_pipeline.py src\train.py src\utils.py results\MULTI_SEED_SUMMARY.md results\multi_seed_summary.csv results\multi_seed_summary.json
git commit -m "Document 10-seed cross-year experiment pipeline"
git push origin main
```

Avoid committing the full `results\experiment_seed_*` folders unless you intentionally want to upload thousands of generated files and checkpoints. Use Git LFS or a separate release/archive for large artifacts.

## Troubleshooting

| Problem | Fix |
|---|---|
| `run_pipeline.py` not found | Restore `run_pipeline.py` in the project root and run commands from the project root. |
| CUDA out of memory | Lower batch sizes in `config.py`. |
| No CUDA GPU detected | Update NVIDIA driver or pass `--allow-cpu`. |
| Raw data folder not found | Fix `RICE_RAW_DIR` in `run_all.bat` or `run_all.ps1`. |
| Old seed missing from summary | Put its results in `results\experiment_seed_<seed>\` and run `--aggregate-only`. |
