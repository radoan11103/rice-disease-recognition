# How to Run — Quick Start

A short, copy-paste guide for running the rice-disease cross-year pipeline.
For the full documentation see [`README.md`](README.md).

**What this does:** trains 4 models (ResNet-50, DenseNet-121, EfficientNet-B0,
DINOv2) on rice-leaf images from **2021–2025** and tests them on **2026** to
measure how well "training on the past" predicts the future — each model in
both **full** and **partial** fine-tuning modes, with Grad-CAM analysis.

---

## Requirements

- A **Windows** PC with an **NVIDIA GPU** (built/tuned for an RTX 3050, 4 GB).
- **Python 3.10 or 3.11 (64-bit)** — install from python.org, tick
  *"Add Python to PATH"*.
- A recent **NVIDIA driver** (no separate CUDA Toolkit needed).
- The **`Rice Disease`** dataset folder (year subfolders `2021` … `2026`).

---

## Step 1 — Get the code

```bat
git clone https://github.com/radoan11103/rice-disease-recognition.git
cd rice-disease-recognition
```

## Step 2 — Set up the environment (one time)

In a **Command Prompt**, inside the project folder:

```bat
setup.bat
```

This creates a `.venv`, installs PyTorch (CUDA build) + all dependencies, and
prints the GPU it found. You want to see `CUDA available: True`.
If it says `False`, update the NVIDIA driver and run `setup.bat` again.

## Step 3 — Point it at the dataset

Open **`run_all.bat`** in Notepad and edit this one line so it points at your
`Rice Disease` folder (no quotes, even with spaces):

```bat
set RICE_RAW_DIR=R:\Anti gravity\Rice Disease Project\Rice Disease
```

The folder must directly contain the year subfolders — see
`directory_structure.txt` for the expected layout.

## Step 4 — Smoke test (~5–10 min, recommended)

Confirm everything works before the long run:

```bat
run_all.bat --skip-gradcam --skip-analyze --models efficientnet_b0 --modes partial
```

If that finishes without errors, you're good.

## Step 5 — Run the full pipeline

```bat
run_all.bat
```

Runs everything in order: preprocessing → 8 training runs → Grad-CAM →
final report. Expect **~3–6 hours** on an RTX 3050. Keep the window open;
progress is also written to `results\pipeline.log`.

## Step 6 — Look at the results

Open **`results\SUMMARY.md`** — the main report. Also useful:

| File | Shows |
|---|---|
| `results\SUMMARY.md` | Full report + headline numbers |
| `results\analysis\generalization_gap.png` | Accuracy lost going 2025 → 2026 |
| `results\analysis\attention_change.png` | How attention moves: partial → full |
| `results\analysis\per_class_f1.png` | Per-class F1 for every run |
| `results\gradcam\<model>\gradcam_summary.png` | 5-image Grad-CAM comparison |
| `results\<model>_<mode>\training_curves.png` | Loss & F1 curves |
| `results\<model>_<mode>\confusion_matrix_test.png` | Confusion matrix on 2026 |

---

## Re-run evaluation only

`src/evaluate.py` scores already-trained checkpoints on a test directory —
**no training, no weights changed**. Use it when you have the `best.pth` files
and just want fresh metrics.

**You need first:** the trained checkpoints at `results\<model>_<mode>\best.pth`.

**Run it:**

1. Open `run_eval.bat` and check the variables at the top:
   - `TEST_DIR` — the test image folder (one subfolder per class);
   - `VAL_DIR` — optional, used for the generalization gap (skipped if absent);
   - `OUT_DIR` — where results are written.
2. Run it:

   ```bat
   run_eval.bat
   ```

   That evaluates all 8 checkpoints. To narrow it down, extra args pass through:

   ```bat
   run_eval.bat --model resnet50 --mode full
   ```

**Outputs** (in `results\evaluation\` — kept separate, nothing overwritten):

| File | Shows |
|---|---|
| `summary.csv` | every run — val/test accuracy, macro-F1, generalization gap, retention |
| `<model>_<mode>\test_metrics.json` | per-run metrics (accuracy, per-class F1, confusion matrix) |
| `<model>_<mode>\confusion_matrix_test.png` | per-run confusion matrix |
| `<model>_<mode>\eval_summary.json` | full per-run summary |

---

## Optional — difficulty-stratified subset selection

`src/select_difficulty_subset.py` takes one trained checkpoint and, for each
class, selects the window of images whose accuracy falls in a target band
(default 70–80%), ranking images by the model's prediction confidence.

- **What it is for:** difficulty-aware data selection — curriculum learning,
  active-learning pools, error analysis.
- **What it is _not_:** a test set or a performance benchmark. The band
  accuracy is a property of the selection, not of the model — the model's real
  accuracy is recorded in the output `manifest.json`. A subset defined by one
  model's scores must not be used to compare other models.

**You need first:**

- a trained checkpoint at `results\<model>_<mode>\best.pth` (run Step 5 first);
- a candidate pool with **more than `PER_CLASS` images per class** — the
  processed `test\` set is capped at 1000/class, so point `DATA_DIR` at a
  larger pool.

**Run it:**

1. Open `run_select_subset.bat` and edit the variables at the top
   (`CHECKPOINT`, `DATA_DIR`, `PER_CLASS`, `TARGET_LO`/`TARGET_HI`, `OUT_DIR`,
   `COPY_IMAGES`).
2. Run it:

   ```bat
   run_select_subset.bat
   ```

   Or call the script directly:

   ```bat
   python -m src.select_difficulty_subset --checkpoint results\resnet50_full\best.pth --data-dir <pool> --copy-images
   ```

**Outputs** (in `results\difficulty_subset\` by default):

| File | Shows |
|---|---|
| `selected_images.csv` | the chosen images — path, confidence, predicted class |
| `manifest.json` | model, target band, the model's **true** accuracy, caveats |
| `selection_overview.png` | per class: every window's accuracy, selected window marked |
| `images\<class>\` | copies of the selected files (only when `COPY_IMAGES=yes`) |

---

## Handy variations

```bat
run_all.bat --skip-preprocess        REM dataset already processed
run_all.bat --models resnet50 dinov2 REM only some models
run_all.bat --continue-on-error      REM keep going if one run fails
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower that model's batch size in `config.py` (`BATCH_SIZE`). |
| PC slows down / swaps | Add `set RICE_NUM_WORKERS=2` near the top of `run_all.bat`. |
| `no CUDA GPU detected` | Update the NVIDIA driver; re-run `setup.bat`. |
| `Raw data folder not found` | Fix the `RICE_RAW_DIR` path in `run_all.bat`. |
| `python` not recognized | Reinstall Python with *"Add to PATH"* ticked. |

Full details, methodology, and how to read the figures: see [`README.md`](README.md).
