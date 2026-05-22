"""Model factory + fine-tuning control + Grad-CAM target layers.

Four architectures share one interface so the trainer is model-agnostic:
    resnet50, densenet121, efficientnet_b0  (CNNs)
    dinov2                                  (ViT-S/14, transformer)

Fine-tuning modes:
    "full"    -- every weight is updated.
    "partial" -- the pretrained backbone is frozen; only the classification
                 head is trained (linear probing).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision import models as tvm

import timm


# ===========================================================================
# MODEL CREATION
# ===========================================================================
def create_model(name: str, num_classes: int, pretrained: bool = True,
                 image_size: int = 224) -> nn.Module:
    """Build one model with its classifier head resized to ``num_classes``."""
    weights = "DEFAULT" if pretrained else None

    if name == "resnet50":
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "densenet121":
        model = tvm.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "dinov2":
        # DINOv2 ViT-S/14. img_size=224 makes timm interpolate the position
        # embeddings -- far lighter on a 4 GB GPU than the native 518.
        model = timm.create_model(
            "vit_small_patch14_dinov2", pretrained=pretrained,
            num_classes=num_classes, img_size=image_size)

    else:
        raise ValueError(f"unknown model '{name}'")

    return model


def get_classifier(model: nn.Module, name: str) -> nn.Module:
    """Return the classification head module for ``name``."""
    if name == "resnet50":
        return model.fc
    if name == "densenet121":
        return model.classifier
    if name == "efficientnet_b0":
        return model.classifier          # Sequential(Dropout, Linear)
    if name == "dinov2":
        return model.get_classifier()    # timm -> model.head
    raise ValueError(f"unknown model '{name}'")


# ===========================================================================
# FINE-TUNING MODE
# ===========================================================================
def set_finetune_mode(model: nn.Module, name: str, mode: str) -> nn.Module:
    """Configure ``requires_grad`` for full or partial fine-tuning."""
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif mode == "partial":
        for param in model.parameters():           # freeze the whole backbone
            param.requires_grad = False
        for param in get_classifier(model, name).parameters():
            param.requires_grad = True             # train only the head

    else:
        raise ValueError(f"unknown finetune mode '{mode}'")
    return model


def trainable_parameters(model: nn.Module):
    """Iterable of parameters that should be passed to the optimizer."""
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===========================================================================
# GRAD-CAM TARGET LAYERS
# ===========================================================================
def _vit_reshape(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT token sequence (B, tokens, C) -> (B, C, H, W) for Grad-CAM.

    The patch tokens are the trailing ``H*W`` entries; any leading class /
    register tokens are dropped automatically.
    """
    num_tokens = tensor.size(1)
    side = int(round(math.sqrt(num_tokens)))
    while side * side > num_tokens:
        side -= 1
    patches = tensor[:, num_tokens - side * side:, :]
    result = patches.reshape(tensor.size(0), side, side, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def get_gradcam_layers(model: nn.Module, name: str):
    """Return (target_layers, reshape_transform) for pytorch-grad-cam."""
    if name == "resnet50":
        return [model.layer4[-1]], None
    if name == "densenet121":
        return [model.features[-1]], None
    if name == "efficientnet_b0":
        return [model.features[-1]], None
    if name == "dinov2":
        # Normalisation layer of the last transformer block.
        return [model.blocks[-1].norm1], _vit_reshape
    raise ValueError(f"unknown model '{name}'")
