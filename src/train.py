"""Stage 2 -- train ONE model in ONE fine-tuning mode.

The pipeline calls this eight times (4 models x {full, partial}). Every run
uses the identical dataset, transforms, optimiser family, epochs and seed, so
the results are directly comparable.

Run:  python -m src.train --model resnet50 --mode full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.amp import GradScaler

import config
from src.dataset import build_dataloaders, compute_class_weights
from src.engine import evaluate, train_one_epoch
from src.metrics import (compute_metrics, plot_confusion_matrix,
                         plot_training_curves)
from src.models import (count_parameters, create_model, get_classifier,
                        set_finetune_mode, trainable_parameters)
from src.utils import (Timer, describe_device, get_device, get_logger,
                       save_json, set_seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one model in one fine-tuning mode.")
    parser.add_argument("--model", required=True, choices=config.MODELS)
    parser.add_argument("--mode", required=True, choices=config.FINETUNE_MODES)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="run on CPU if no GPU is available (very slow)")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    out_dir = config.run_dir(args.model, args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"train.{args.model}.{args.mode}",
                        out_dir / "train.log")

    device = get_device(allow_cpu=args.allow_cpu)
    use_amp = config.USE_AMP and device.type == "cuda"
    logger.info(f"Run: model={args.model}  mode={args.mode}  "
                f"epochs={args.epochs}")
    logger.info(f"Device: {describe_device(device)} | AMP={use_amp}")

    if not (config.PROCESSED_DIR / "train").exists():
        raise SystemExit(
            f"Processed data not found at {config.PROCESSED_DIR}. "
            "Run 'python -m src.preprocess' first.")

    # ----- data -----------------------------------------------------------
    batch_size = config.BATCH_SIZE[args.model]
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        config.PROCESSED_DIR, config.IMAGE_SIZE, batch_size,
        config.NUM_WORKERS, config.RANDOM_SEED)
    logger.info(f"Classes: {classes}")
    logger.info(f"Batches/epoch: train={len(train_loader)} "
                f"val={len(val_loader)} test={len(test_loader)} "
                f"(batch_size={batch_size}, accum={config.GRAD_ACCUM_STEPS})")

    # ----- model ----------------------------------------------------------
    model = create_model(args.model, config.NUM_CLASSES, pretrained=True,
                         image_size=config.IMAGE_SIZE).to(device)
    set_finetune_mode(model, args.model, args.mode)
    head = get_classifier(model, args.model)
    total_params, trainable_params_count = count_parameters(model)
    logger.info(f"Parameters: total={total_params:,}  "
                f"trainable={trainable_params_count:,}  "
                f"({100 * trainable_params_count / total_params:.2f}%)")

    # ----- loss / optimiser ----------------------------------------------
    # Class-weighted + label-smoothed loss handles the imbalanced past years.
    class_weights = compute_class_weights(
        train_loader.dataset, config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=config.LABEL_SMOOTHING)
    learning_rate = config.LR_FULL if args.mode == "full" else config.LR_PARTIAL
    optimizer = torch.optim.AdamW(trainable_parameters(model),
                                  lr=learning_rate,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    # ----- training loop --------------------------------------------------
    best_f1, best_epoch, stale_epochs = -1.0, -1, 0
    history: list[dict] = []
    ckpt_path = out_dir / "best.pth"

    with Timer() as timer:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc, tr_f1 = train_one_epoch(
                model, head, train_loader, criterion, optimizer, scaler,
                device, mode=args.mode, use_amp=use_amp,
                accum_steps=config.GRAD_ACCUM_STEPS)
            va_loss, va_acc, va_f1, _, _ = evaluate(
                model, val_loader, criterion, device, use_amp=use_amp)
            scheduler.step()

            history.append(dict(
                epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                train_f1=tr_f1, val_loss=va_loss, val_acc=va_acc,
                val_f1=va_f1, lr=optimizer.param_groups[0]["lr"]))
            logger.info(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:5.2f} f1 {tr_f1:5.2f} | "
                f"val loss {va_loss:.4f} acc {va_acc:5.2f} f1 {va_f1:5.2f}")

            # Select the best model on validation macro-F1 (robust to imbalance).
            if va_f1 > best_f1:
                best_f1, best_epoch, stale_epochs = va_f1, epoch, 0
                torch.save({
                    "model_state": model.state_dict(),
                    "model_name": args.model, "mode": args.mode,
                    "classes": classes, "epoch": epoch, "val_f1": va_f1,
                }, ckpt_path)
                logger.info(f"  -> new best (val macro-F1 {va_f1:.2f}) saved")
            else:
                stale_epochs += 1
                if stale_epochs >= config.EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch} "
                                f"(no val-F1 gain for {stale_epochs} epochs).")
                    break

    train_minutes = timer.elapsed / 60
    logger.info(f"Training done in {train_minutes:.1f} min. "
                f"Best epoch {best_epoch} (val macro-F1 {best_f1:.2f}).")

    # ----- final evaluation with the best checkpoint ----------------------
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    _, _, _, va_preds, va_labels = evaluate(
        model, val_loader, criterion, device, use_amp=use_amp)
    _, _, _, te_preds, te_labels = evaluate(
        model, test_loader, criterion, device, use_amp=use_amp)

    val_metrics = compute_metrics(va_labels, va_preds, classes)
    test_metrics = compute_metrics(te_labels, te_preds, classes)

    # The research numbers: how much performance is lost PAST -> FUTURE.
    gap_acc = val_metrics["accuracy"] - test_metrics["accuracy"]
    gap_f1 = val_metrics["macro_f1"] - test_metrics["macro_f1"]
    retention = 100 * test_metrics["macro_f1"] / max(val_metrics["macro_f1"], 1e-6)

    summary = dict(
        model=args.model, mode=args.mode,
        total_params=total_params, trainable_params=trainable_params_count,
        best_epoch=best_epoch, epochs_run=len(history),
        train_minutes=round(train_minutes, 2),
        val=val_metrics, test=test_metrics,
        generalization_gap_acc=round(gap_acc, 3),
        generalization_gap_f1=round(gap_f1, 3),
        future_retention_pct=round(retention, 2),
    )
    save_json(summary, out_dir / "summary.json")
    save_json(history, out_dir / "history.json")
    save_json(val_metrics, out_dir / "val_metrics.json")
    save_json(test_metrics, out_dir / "test_metrics.json")

    plot_confusion_matrix(
        val_metrics["confusion_matrix"], classes,
        f"{args.model} [{args.mode}] - VAL (2021-2025 holdout)",
        out_dir / "confusion_matrix_val.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], classes,
        f"{args.model} [{args.mode}] - TEST (2026 future)",
        out_dir / "confusion_matrix_test.png")
    plot_training_curves(
        history, best_epoch,
        f"{args.model} [{args.mode}] - training curves",
        out_dir / "training_curves.png")

    logger.info("=" * 64)
    logger.info(f"VAL  (2021-2025 holdout): acc {val_metrics['accuracy']:.2f}  "
                f"macro-F1 {val_metrics['macro_f1']:.2f}")
    logger.info(f"TEST (2026 future)      : acc {test_metrics['accuracy']:.2f}  "
                f"macro-F1 {test_metrics['macro_f1']:.2f}")
    logger.info(f"Generalization gap (macro-F1): {gap_f1:.2f} points")
    logger.info(f"Future retention            : {retention:.1f}%")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
