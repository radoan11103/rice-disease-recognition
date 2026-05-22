"""Stand-alone evaluation -- score trained checkpoints on the test set.

src/train.py evaluates a model only at the end of a fresh training run. This
script re-runs the evaluation ONLY: it loads existing ``best.pth`` checkpoints
and scores them on a test directory (and a val directory, if given). No
training happens and no weights are changed.

Outputs are written to a separate folder (results/evaluation/ by default) so
the original per-run results/<model>_<mode>/ files are left untouched.

Run:
    python -m src.evaluate                              (all checkpoints found)
    python -m src.evaluate --model resnet50 --mode full
    python -m src.evaluate --test-dir path/to/test --val-dir path/to/val
    python -m src.evaluate --checkpoint path/to/best.pth --model resnet50 --mode full
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                                   # noqa: E402
import torch.nn as nn                                          # noqa: E402
from torch.utils.data import DataLoader                        # noqa: E402
from torchvision import datasets                               # noqa: E402

import config                                                  # noqa: E402
from src.dataset import build_transforms                       # noqa: E402
from src.engine import evaluate as run_eval                    # noqa: E402
from src.metrics import compute_metrics, plot_confusion_matrix  # noqa: E402
from src.models import create_model                            # noqa: E402
from src.utils import get_device, get_logger, save_json, set_seed  # noqa: E402


def build_split_loader(split_dir: Path, batch_size: int, num_workers: int):
    """Build a deterministic eval DataLoader over one ImageFolder directory.

    Returns (loader, classes), or (None, None) if the directory is absent.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return None, None
    _, eval_transform = build_transforms(config.IMAGE_SIZE)
    dataset = datasets.ImageFolder(split_dir, eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return loader, list(dataset.classes)


def evaluate_checkpoint(model_name: str, mode: str, checkpoint_path: Path,
                        test_dir: Path, val_dir: Path, out_dir: Path, device,
                        batch_size: int, num_workers: int, logger) -> dict | None:
    """Evaluate one checkpoint on the test directory (and val, if present).

    Returns the summary dict, or None if the checkpoint is missing.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"  SKIP {model_name} [{mode}] -- no checkpoint at "
                       f"{checkpoint_path}")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = list(ckpt["classes"])
    ckpt_model = ckpt.get("model_name", model_name)
    if ckpt_model != model_name:
        logger.warning(f"  {model_name} [{mode}]: checkpoint says model is "
                       f"'{ckpt_model}' -- building that architecture.")

    model = create_model(ckpt_model, num_classes=len(classes),
                         pretrained=False, image_size=config.IMAGE_SIZE)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    use_amp = device.type == "cuda"
    criterion = nn.CrossEntropyLoss()

    # --- TEST (required) ---------------------------------------------------
    test_loader, test_classes = build_split_loader(
        test_dir, batch_size, num_workers)
    if test_loader is None:
        raise SystemExit(f"Test directory not found: {test_dir}")
    if test_classes != classes:
        raise SystemExit(
            f"{model_name} [{mode}]: class mismatch between checkpoint and "
            f"test directory.\n  checkpoint: {classes}\n  test: {test_classes}")

    # --- VAL (optional) ----------------------------------------------------
    val_loader, val_classes = build_split_loader(
        val_dir, batch_size, num_workers)
    if val_loader is not None and val_classes != classes:
        logger.warning(f"  {model_name} [{mode}]: val classes differ from the "
                       "checkpoint -- skipping val.")
        val_loader = None

    run_out = Path(out_dir) / f"{model_name}_{mode}"
    run_out.mkdir(parents=True, exist_ok=True)

    _, _, _, te_preds, te_labels = run_eval(
        model, test_loader, criterion, device, use_amp=use_amp)
    test_metrics = compute_metrics(te_labels, te_preds, classes)
    save_json(test_metrics, run_out / "test_metrics.json")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], classes,
        f"{model_name} [{mode}] - TEST", run_out / "confusion_matrix_test.png")

    val_metrics = None
    if val_loader is not None:
        _, _, _, va_preds, va_labels = run_eval(
            model, val_loader, criterion, device, use_amp=use_amp)
        val_metrics = compute_metrics(va_labels, va_preds, classes)
        save_json(val_metrics, run_out / "val_metrics.json")
        plot_confusion_matrix(
            val_metrics["confusion_matrix"], classes,
            f"{model_name} [{mode}] - VAL",
            run_out / "confusion_matrix_val.png")

    summary = dict(
        model=model_name, mode=mode,
        checkpoint=str(checkpoint_path),
        evaluated_on=str(test_dir),
        test=test_metrics, val=val_metrics,
    )
    if val_metrics is not None:
        # Same research numbers train.py reports: how much is lost past->future.
        summary["generalization_gap_acc"] = round(
            val_metrics["accuracy"] - test_metrics["accuracy"], 3)
        summary["generalization_gap_f1"] = round(
            val_metrics["macro_f1"] - test_metrics["macro_f1"], 3)
        summary["future_retention_pct"] = round(
            100 * test_metrics["macro_f1"]
            / max(val_metrics["macro_f1"], 1e-6), 2)
    save_json(summary, run_out / "eval_summary.json")

    line = (f"  {model_name} [{mode}]: TEST acc {test_metrics['accuracy']:.2f} "
            f"macro-F1 {test_metrics['macro_f1']:.2f}")
    if val_metrics is not None:
        line += (f"  | VAL acc {val_metrics['accuracy']:.2f} "
                 f"macro-F1 {val_metrics['macro_f1']:.2f}")
    logger.info(line)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run evaluation only: score trained checkpoints on a "
                    "test directory. No training, no weights changed.")
    parser.add_argument("--model", default="all",
                        choices=config.MODELS + ["all"])
    parser.add_argument("--mode", default="all",
                        choices=config.FINETUNE_MODES + ["all"])
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="evaluate this exact .pth (requires a single "
                             "--model and --mode)")
    parser.add_argument("--test-dir", type=Path,
                        default=config.PROCESSED_DIR / "test",
                        help="the test image directory (ImageFolder layout: "
                             "one subfolder per class)")
    parser.add_argument("--val-dir", type=Path,
                        default=config.PROCESSED_DIR / "val",
                        help="optional val directory, for the generalization "
                             "gap; skipped automatically if it does not exist")
    parser.add_argument("--out", type=Path,
                        default=config.RESULTS_DIR / "evaluation",
                        help="output dir (kept separate from results/<run>/)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="run on CPU if no GPU is available (slow)")
    args = parser.parse_args()

    models = config.MODELS if args.model == "all" else [args.model]
    modes = config.FINETUNE_MODES if args.mode == "all" else [args.mode]
    if args.checkpoint is not None and (len(models) != 1 or len(modes) != 1):
        raise SystemExit("--checkpoint requires a single --model and --mode.")

    set_seed(config.RANDOM_SEED)
    device = get_device(allow_cpu=args.allow_cpu)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("evaluate", out_dir / "evaluate.log")
    logger.info(f"Device: {device.type}  |  test directory: {args.test_dir}")

    rows = []
    for model_name in models:
        for mode in modes:
            batch_size = args.batch_size or config.BATCH_SIZE.get(model_name, 16)
            ckpt_path = (args.checkpoint if args.checkpoint is not None
                         else config.run_dir(model_name, mode) / "best.pth")
            summary = evaluate_checkpoint(
                model_name, mode, ckpt_path, args.test_dir, args.val_dir,
                out_dir, device, batch_size, args.num_workers, logger)
            if summary is None:
                continue
            test, val = summary["test"], summary["val"]
            rows.append({
                "model": model_name,
                "mode": mode,
                "val_acc": round(val["accuracy"], 2) if val else "",
                "val_f1": round(val["macro_f1"], 2) if val else "",
                "test_acc": round(test["accuracy"], 2),
                "test_f1": round(test["macro_f1"], 2),
                "test_balanced_acc": round(test["balanced_accuracy"], 2),
                "gen_gap_f1": summary.get("generalization_gap_f1", ""),
                "retention_pct": summary.get("future_retention_pct", ""),
            })

    if not rows:
        raise SystemExit(
            "No checkpoints were evaluated. Expected best.pth files at "
            "results/<model>_<mode>/best.pth -- train the models first, or "
            "pass --checkpoint with --model/--mode.")

    summary_csv = out_dir / "summary.csv"
    fields = ["model", "mode", "val_acc", "val_f1", "test_acc", "test_f1",
              "test_balanced_acc", "gen_gap_f1", "retention_pct"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=" * 64)
    logger.info(f"Evaluated {len(rows)} checkpoint(s).")
    logger.info(f"Per-run metrics + confusion matrices -> {out_dir}")
    logger.info(f"Combined table -> {summary_csv}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
