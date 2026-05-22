"""Shared data loading -- guarantees every model sees the *same* data.

`build_dataloaders` is the single function used by training, evaluation and
Grad-CAM, so the train/val/test images and the class->index mapping are
identical across all eight runs.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from src.utils import seed_worker


def build_transforms(image_size: int):
    """Return (train_transform, eval_transform).

    Training uses light augmentation; evaluation is deterministic. Both
    normalise with ImageNet statistics because all backbones are pretrained.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05),
                                scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return train_transform, eval_transform


def build_dataloaders(processed_dir: Path, image_size: int, batch_size: int,
                      num_workers: int, seed: int):
    """Create the train/val/test DataLoaders and return the class list."""
    processed_dir = Path(processed_dir)
    train_transform, eval_transform = build_transforms(image_size)

    train_set = datasets.ImageFolder(processed_dir / "train", train_transform)
    val_set = datasets.ImageFolder(processed_dir / "val", eval_transform)
    test_set = datasets.ImageFolder(processed_dir / "test", eval_transform)

    # CORRECTNESS GUARD: every split must expose the same classes in the same
    # order, otherwise label indices would silently disagree.
    if not (train_set.classes == val_set.classes == test_set.classes):
        raise RuntimeError(
            "Class mismatch across splits -- "
            f"train={train_set.classes} val={val_set.classes} "
            f"test={test_set.classes}")

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        # drop_last avoids a size-1 final batch crashing BatchNorm in training.
        drop_last=True, generator=generator, **common)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, **common)

    return train_loader, val_loader, test_loader, list(train_set.classes)


def compute_class_weights(dataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for the imbalanced 2021-2025 pool."""
    counts = torch.bincount(torch.tensor(dataset.targets),
                            minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    return counts.sum() / (num_classes * counts)


def sample_paths_per_class(split_dir: Path, n_per_class: int, classes,
                           seed: int):
    """Return [(image_path, class_name), ...] sampled for Grad-CAM."""
    rng = random.Random(seed)
    picks = []
    for cls in classes:
        files = sorted((Path(split_dir) / cls).glob("*.jpg"))
        if not files:
            continue
        for path in rng.sample(files, min(n_per_class, len(files))):
            picks.append((path, cls))
    return picks
