"""Evaluation metrics and confusion-matrix plotting."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # headless backend -- no display needed
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
from sklearn.metrics import (         # noqa: E402
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score)


def compute_metrics(labels, preds, class_names) -> dict:
    """Return a full metrics dict (accuracy, balanced acc, macro-F1, per-class,
    confusion matrix) ready to be saved as JSON."""
    label_ids = list(range(len(class_names)))
    matrix = confusion_matrix(labels, preds, labels=label_ids)
    report = classification_report(
        labels, preds, labels=label_ids, target_names=class_names,
        output_dict=True, zero_division=0)

    per_class = {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }
        for cls in class_names
    }
    return {
        "accuracy": float(accuracy_score(labels, preds) * 100),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds) * 100),
        "macro_f1": float(f1_score(labels, preds, average="macro",
                                   zero_division=0) * 100),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classes": list(class_names),
    }


def plot_confusion_matrix(matrix, class_names, title: str, path: Path,
                          normalize: bool = True) -> None:
    """Save a confusion-matrix heatmap to ``path``."""
    matrix = np.asarray(matrix, dtype=float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        display = matrix / row_sums
        fmt, vmax = ".2f", 1.0
    else:
        display = matrix
        fmt, vmax = ".0f", display.max() if display.size else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = display.max() / 2 if display.size else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(display[i, j], fmt), ha="center", va="center",
                    color="white" if display[i, j] > threshold else "black",
                    fontsize=9)

    fig.colorbar(image, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list, best_epoch: int, title: str,
                         path: Path) -> None:
    """Save an intuitive 2-panel training figure: loss and macro-F1 per epoch.

    The loss panel makes over-fitting obvious -- when the orange validation
    loss starts rising while the blue training loss keeps falling, the model
    has begun memorising the past years instead of generalising.
    """
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_f1"] for h in history]
    val_f1 = [h["val_f1"] for h in history]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- loss panel -------------------------------------------------------
    ax_loss.plot(epochs, train_loss, "o-", color="#4C72B0", label="train loss")
    ax_loss.plot(epochs, val_loss, "s-", color="#DD8452", label="val loss")
    ax_loss.axvline(best_epoch, color="#55A868", linestyle="--", linewidth=1.5,
                    label=f"best epoch ({best_epoch})")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss  (val rising above train = over-fitting)")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    # --- macro-F1 panel ---------------------------------------------------
    ax_f1.plot(epochs, train_f1, "o-", color="#4C72B0", label="train macro-F1")
    ax_f1.plot(epochs, val_f1, "s-", color="#DD8452", label="val macro-F1")
    ax_f1.axvline(best_epoch, color="#55A868", linestyle="--", linewidth=1.5,
                  label=f"best epoch ({best_epoch})")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("macro-F1 (%)")
    ax_f1.set_ylim(0, 100)
    ax_f1.set_title("Macro-F1")
    ax_f1.grid(alpha=0.3)
    ax_f1.legend()

    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
