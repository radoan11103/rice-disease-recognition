"""Difficulty-stratified subset selection for a *fixed* trained model.

Given one trained checkpoint (e.g. results/<model>_<mode>/best.pth), this tool
scores every image in a candidate pool by the model's prediction confidence --
a standard per-sample difficulty signal -- and, for each class, extracts the
contiguous ``--per-class`` window whose accuracy naturally falls inside a
target band (default 70-80%). That window is the model's "medium-difficulty"
stratum for that class.

WHAT THIS IS FOR
    Difficulty-aware data selection: curriculum learning, active-learning
    pools, sample-efficiency or robustness studies, error analysis.

WHAT THIS IS *NOT* FOR  (please read)
    * It is NOT a performance measurement. The ~75% accuracy on the selected
      window is a property of the *selection* -- you asked for that band --
      not of the model. The model's REAL accuracy on the full pool is printed
      and stored in manifest.json; use that number if you need to state
      performance.
    * It must NOT be used as a benchmark to compare *other* models. The subset
      is defined by THIS model's scores, so it is biased toward this model:
      another model's accuracy on it measures agreement with this model, not
      real skill. To compare models, evaluate every model on the same
      untouched test set instead.

The tool only writes into its own output directory; it never modifies the
existing results/ figures or summary.csv.

Run:
    python -m src.select_difficulty_subset --checkpoint results/resnet50_full/best.pth
    python -m src.select_difficulty_subset --checkpoint <pth> --data-dir <pool> \\
        --per-class 1000 --target-lo 0.70 --target-hi 0.80 --copy-images
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from torch.amp import autocast                                 # noqa: E402
from torch.utils.data import DataLoader                        # noqa: E402
from torchvision import datasets                               # noqa: E402

import config                                                  # noqa: E402
from src.dataset import build_transforms                       # noqa: E402
from src.models import create_model                            # noqa: E402
from src.utils import get_device, get_logger, save_json, set_seed  # noqa: E402


# ===========================================================================
# MODEL + INFERENCE
# ===========================================================================
def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load a checkpoint saved by src/train.py. Returns (model, classes,
    model_name, mode)."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt_path}\n"
            "Place the .pth there (the trainer writes results/<model>_<mode>/"
            "best.pth) or pass the right path with --checkpoint.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for key in ("model_state", "model_name", "classes"):
        if key not in ckpt:
            raise SystemExit(
                f"Checkpoint {ckpt_path} is missing '{key}'. This tool expects "
                "a checkpoint saved by src/train.py (a dict with model_state, "
                "model_name, mode, classes).")

    model_name = ckpt["model_name"]
    classes = list(ckpt["classes"])
    model = create_model(model_name, num_classes=len(classes),
                         pretrained=False, image_size=config.IMAGE_SIZE)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, classes, model_name, ckpt.get("mode", "?")


@torch.no_grad()
def score_pool(model, data_dir: Path, expected_classes, device,
               batch_size: int, num_workers: int, logger):
    """Run the model over every image in ``data_dir`` (ImageFolder layout).

    Returns a list of per-image record dicts -- one per image -- carrying the
    path, true/predicted class, prediction confidence and correctness.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Candidate pool not found: {data_dir}")

    _, eval_transform = build_transforms(config.IMAGE_SIZE)
    dataset = datasets.ImageFolder(data_dir, eval_transform)

    # CORRECTNESS GUARD: the pool's class folders must match the checkpoint's
    # class order, otherwise label indices would silently disagree.
    if list(dataset.classes) != list(expected_classes):
        raise SystemExit(
            "Class mismatch between the checkpoint and the data pool.\n"
            f"  checkpoint classes : {list(expected_classes)}\n"
            f"  data-dir classes   : {list(dataset.classes)}\n"
            "The class subfolder names (and their sorted order) must match.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    use_amp = device.type == "cuda"

    confidences, preds, prob_true = [], [], []
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1).cpu()
        conf, pred = probs.max(dim=1)
        confidences.append(conf)
        preds.append(pred)
        prob_true.append(probs.gather(1, labels.view(-1, 1)).squeeze(1))
        if batch_idx % 20 == 0:
            logger.info(f"  scored batch {batch_idx + 1}/{len(loader)}")

    confidences = torch.cat(confidences).numpy()
    preds = torch.cat(preds).numpy()
    prob_true = torch.cat(prob_true).numpy()

    records = []
    for i, (path, true_idx) in enumerate(dataset.samples):
        records.append(dict(
            path=path,
            true_class=dataset.classes[true_idx],
            pred_class=dataset.classes[int(preds[i])],
            confidence=float(confidences[i]),
            prob_true=float(prob_true[i]),
            correct=bool(int(preds[i]) == int(true_idx)),
        ))
    logger.info(f"Scored {len(records)} images from {data_dir}")
    return records


# ===========================================================================
# DIFFICULTY-BAND SELECTION
# ===========================================================================
def select_window(class_records, per_class: int, lo: float, hi: float) -> dict:
    """Pick the contiguous, confidence-ranked window of ``per_class`` images
    whose accuracy is closest to the centre of the [lo, hi] band.

    Images are ranked easy -> hard by prediction confidence. A sliding window
    over that ranking gives a smooth difficulty sweep; the window landing in
    the target band is a genuine difficulty stratum (unlike a random search
    that simply stops when a number is hit).
    """
    ordered = sorted(class_records, key=lambda r: (-r["confidence"], r["path"]))
    n = len(ordered)
    if n < per_class:
        return dict(ok=False,
                    reason=(f"pool has only {n} images for this class; "
                            f"need at least --per-class={per_class}"))

    correct = np.array([1 if r["correct"] else 0 for r in ordered])
    prefix = np.concatenate([[0], np.cumsum(correct)])
    # Accuracy of every candidate window of width per_class.
    curve = (prefix[per_class:] - prefix[:-per_class]) / per_class

    mid = 0.5 * (lo + hi)
    best_start = int(np.argmin(np.abs(curve - mid)))
    best_acc = float(curve[best_start])
    window = ordered[best_start:best_start + per_class]

    return dict(
        ok=True,
        in_band=bool(lo <= best_acc <= hi),
        accuracy=best_acc,
        window_start=best_start,
        pool_size=n,
        confidence_hi=float(window[0]["confidence"]),
        confidence_lo=float(window[-1]["confidence"]),
        curve=curve,                      # kept for the overview plot only
        records=window,
    )


# ===========================================================================
# OUTPUTS
# ===========================================================================
def write_csv(rows, path: Path) -> None:
    """Write the selected images to a CSV."""
    fields = ["class", "in_band", "image_path", "confidence", "prob_true",
              "predicted_class", "correct"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(per_class_result, classes, lo: float, hi: float,
                  model_name: str, path: Path) -> None:
    """One subplot per class: accuracy of every candidate window, the target
    band shaded, and the window that was selected marked."""
    cols = 2
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i, cls in enumerate(classes):
        ax = axes[i // cols][i % cols]
        ax.axis("on")
        res = per_class_result[cls]
        if not res.get("ok"):
            ax.text(0.5, 0.5, f"{cls}\n{res.get('reason', 'not selectable')}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        curve = res["curve"] * 100
        ax.plot(np.arange(len(curve)), curve, color="#2c7fb8", lw=1.5,
                label="window accuracy")
        ax.axhspan(lo * 100, hi * 100, color="#7fbf7b", alpha=0.25,
                   label=f"target {int(lo * 100)}-{int(hi * 100)}%")
        ax.axvline(res["window_start"], color="#d95f02", lw=2,
                   label="selected window")
        flag = "" if res["in_band"] else "  -- OUT OF BAND"
        ax.set_title(f"{cls}  (pool {res['pool_size']}, "
                     f"selected {res['accuracy'] * 100:.1f}%{flag})",
                     fontsize=10)
        ax.set_xlabel("window start (images ranked easy -> hard)")
        ax.set_ylabel("model accuracy on window (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Difficulty-stratified selection -- model: {model_name}\n"
                 "accuracy of each candidate window; the band you asked for "
                 "is shaded", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a difficulty-stratified per-class image subset for "
                    "a fixed trained model. This is a difficulty stratum, not "
                    "a performance benchmark -- see the module docstring.")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="path to a best.pth saved by src/train.py")
    parser.add_argument("--data-dir", type=Path,
                        default=config.PROCESSED_DIR / "test",
                        help="candidate pool (ImageFolder layout: one subfolder "
                             "per class). MUST hold MORE than --per-class "
                             "images per class for selection to be meaningful.")
    parser.add_argument("--per-class", type=int,
                        default=config.TEST_IMAGES_PER_CLASS,
                        help="images to select per class (default 1000)")
    parser.add_argument("--target-lo", type=float, default=0.70,
                        help="lower bound of the accuracy band, as a fraction")
    parser.add_argument("--target-hi", type=float, default=0.80,
                        help="upper bound of the accuracy band, as a fraction")
    parser.add_argument("--out", type=Path,
                        default=config.RESULTS_DIR / "difficulty_subset",
                        help="output directory (its own folder; the existing "
                             "results/ figures are never touched)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="inference batch size (default: per-model)")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--copy-images", action="store_true",
                        help="also copy the selected files into "
                             "<out>/images/<class>/")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="run on CPU if no GPU is available (slow)")
    args = parser.parse_args()

    if not 0.0 <= args.target_lo < args.target_hi <= 1.0:
        raise SystemExit("Require 0 <= --target-lo < --target-hi <= 1.")

    set_seed(config.RANDOM_SEED)
    device = get_device(allow_cpu=args.allow_cpu)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("select_difficulty_subset", out_dir / "selection.log")

    logger.info(f"Checkpoint : {args.checkpoint}")
    model, classes, model_name, mode = load_checkpoint(args.checkpoint, device)
    logger.info(f"Model      : {model_name} [{mode}]   classes={classes}")

    batch_size = args.batch_size or config.BATCH_SIZE.get(model_name, 16)
    logger.info(f"Pool       : {args.data_dir}  (batch_size={batch_size})")
    records = score_pool(model, args.data_dir, classes, device,
                         batch_size, args.num_workers, logger)

    # ----- the honest anchor: the model's REAL accuracy on the full pool ----
    overall_true_acc = float(np.mean([r["correct"] for r in records]))
    by_class: dict[str, list] = {c: [] for c in classes}
    for record in records:
        by_class[record["true_class"]].append(record)
    true_acc_per_class = {
        c: (float(np.mean([r["correct"] for r in recs])) if recs else None)
        for c, recs in by_class.items()
    }
    logger.info(f"TRUE accuracy on the full pool: {overall_true_acc * 100:.2f}% "
                "(this is the model's real performance)")

    # ----- per-class difficulty-band selection -----------------------------
    per_class_result = {}
    for cls in classes:
        res = select_window(by_class[cls], args.per_class,
                            args.target_lo, args.target_hi)
        res["true_accuracy"] = true_acc_per_class[cls]
        per_class_result[cls] = res

    # ----- write the selected subset (CSV) ---------------------------------
    csv_rows, selected_total = [], 0
    for cls in classes:
        res = per_class_result[cls]
        if not res.get("ok"):
            continue
        for r in res["records"]:
            csv_rows.append({
                "class": cls,
                "in_band": res["in_band"],
                "image_path": r["path"],
                "confidence": round(r["confidence"], 6),
                "prob_true": round(r["prob_true"], 6),
                "predicted_class": r["pred_class"],
                "correct": r["correct"],
            })
        selected_total += len(res["records"])
    write_csv(csv_rows, out_dir / "selected_images.csv")

    # ----- optionally copy the image files ---------------------------------
    if args.copy_images:
        img_root = out_dir / "images"
        for cls in classes:
            res = per_class_result[cls]
            if not res.get("ok"):
                continue
            dst_dir = img_root / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for r in res["records"]:
                shutil.copy2(r["path"], dst_dir / Path(r["path"]).name)
        logger.info(f"Copied {selected_total} selected images -> {img_root}")

    # ----- manifest (with the integrity caveats baked in) ------------------
    band = f"{int(args.target_lo * 100)}-{int(args.target_hi * 100)}%"
    notes = [
        "This subset was built by ranking each class's images by THIS model's "
        "prediction confidence and taking the window whose accuracy lands in "
        "the target band. It is a difficulty stratum, not a measurement.",
        f"The {band} accuracy on this subset is a property of the selection, "
        "not of the model -- see 'true_accuracy_overall' for the model's real "
        "accuracy on the full pool.",
        "Do NOT use this subset to compare other models: it is biased toward "
        "the model that defined it. Compare models on the same untouched test "
        "set instead.",
        "Valid uses: curriculum learning, active-learning pools, "
        "sample-efficiency / robustness studies, error analysis.",
    ]
    manifest = dict(
        tool="src/select_difficulty_subset.py",
        checkpoint=str(args.checkpoint),
        model_name=model_name,
        finetune_mode=mode,
        classes=classes,
        data_pool=str(args.data_dir),
        per_class=args.per_class,
        target_band=[args.target_lo, args.target_hi],
        true_accuracy_overall=overall_true_acc,
        true_accuracy_per_class=true_acc_per_class,
        selected_total=selected_total,
        per_class_selection={
            cls: {k: v for k, v in res.items()
                  if k not in ("records", "curve")}
            for cls, res in per_class_result.items()
        },
        notes=notes,
    )
    save_json(manifest, out_dir / "manifest.json")

    plot_overview(per_class_result, classes, args.target_lo, args.target_hi,
                  model_name, out_dir / "selection_overview.png")

    # ----- console summary -------------------------------------------------
    logger.info("=" * 72)
    logger.info(f"Model REAL accuracy on the pool : {overall_true_acc * 100:.2f}"
                "%   <- use THIS to state performance")
    for cls in classes:
        res = per_class_result[cls]
        if res.get("ok"):
            flag = "" if res["in_band"] else "   *** OUT OF BAND ***"
            logger.info(f"  {cls:<14} selected {args.per_class} imgs | "
                        f"window acc {res['accuracy'] * 100:5.1f}% | "
                        f"true acc {res['true_accuracy'] * 100:5.1f}%{flag}")
        else:
            logger.info(f"  {cls:<14} NOT SELECTABLE -- {res['reason']}")

    not_ok = [c for c in classes if not per_class_result[c].get("ok")]
    out_of_band = [c for c in classes if per_class_result[c].get("ok")
                   and not per_class_result[c]["in_band"]]
    if not_ok or out_of_band:
        logger.info("-" * 72)
        if not_ok:
            logger.info(f"Could not select for {not_ok}: the pool needs MORE "
                        f"than --per-class={args.per_class} images per class. "
                        "Point --data-dir at a larger pool.")
        if out_of_band:
            logger.info(f"No window hit the {band} band for {out_of_band}; the "
                        "closest window was used. Use a larger / more varied "
                        "pool, or widen --target-lo / --target-hi.")
    logger.info("-" * 72)
    logger.info("REMINDER: the band accuracy is a selection artefact, not a "
                "model score, and this subset must not be used to compare "
                "other models. See manifest.json -> notes.")
    logger.info(f"Outputs -> {out_dir}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
