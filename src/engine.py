"""Training / evaluation loops with mixed precision and gradient accumulation.

Mixed precision (AMP) and a small batch + gradient accumulation are what let
heavy models train inside the RTX 3050's 4 GB of VRAM.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def _apply_train_mode(model: nn.Module, head: nn.Module, mode: str) -> None:
    """Set module train/eval flags correctly for the fine-tuning mode.

    In "partial" mode the whole model is kept in eval() so the frozen
    backbone's BatchNorm stats do not drift; only the trainable head is
    switched to train() (this matters for the head's Dropout).
    """
    if mode == "full":
        model.train()
    else:
        model.eval()
        head.train()


def train_one_epoch(model: nn.Module, head: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer, scaler: GradScaler,
                    device: torch.device, *, mode: str, use_amp: bool,
                    accum_steps: int):
    """Run one training epoch. Returns (avg_loss, accuracy, macro_f1)."""
    _apply_train_mode(model, head, mode)

    total_loss, total_seen = 0.0, 0
    all_preds, all_labels = [], []
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scale the loss for gradient accumulation, then back-propagate.
        scaler.scale(loss / accum_steps).backward()

        last_batch = (step + 1) == num_batches
        if (step + 1) % accum_steps == 0 or last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        total_seen += images.size(0)
        all_preds.append(outputs.detach().argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_labels).numpy()
    accuracy = float((preds == targets).mean() * 100)
    macro_f1 = float(f1_score(targets, preds, average="macro",
                              zero_division=0) * 100)
    return total_loss / total_seen, accuracy, macro_f1


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, *, use_amp: bool):
    """Evaluate the model. Returns (avg_loss, accuracy, macro_f1, preds, labels)."""
    model.eval()
    total_loss, total_seen = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_seen += images.size(0)
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_labels).numpy()
    accuracy = float((preds == targets).mean() * 100)
    macro_f1 = float(f1_score(targets, preds, average="macro",
                              zero_division=0) * 100)
    return total_loss / total_seen, accuracy, macro_f1, preds, targets
