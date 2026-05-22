"""Stage 3 -- Grad-CAM analysis: how does fine-tuning move the model's focus?

For one model this loads BOTH trained checkpoints (full + partial), runs
Grad-CAM on the same future-year (2026) sample images, and shows -- visually
and numerically -- how the focus shifts when moving from partial to full
fine-tuning.

Outputs (in results/gradcam/<model>/):
    NN_<class>_<img>.png  -- per image: input | partial | full | change map
    gradcam_summary.png   -- one figure, 5 sample images, partial -> full
    attention_shift.json  -- per-image and averaged shift metrics

Run:  python -m src.gradcam --model resnet50
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from PIL import Image                    # noqa: E402
from pytorch_grad_cam import GradCAM     # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image          # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

import config                            # noqa: E402
from src.dataset import build_transforms, sample_paths_per_class    # noqa: E402
from src.models import create_model, get_gradcam_layers            # noqa: E402
from src.utils import (get_device, get_logger, save_json,          # noqa: E402
                       set_seed)


def load_trained(model_name: str, mode: str, device: torch.device):
    """Load a trained checkpoint for (model, mode). Returns (model, classes)."""
    ckpt_path = config.run_dir(model_name, mode) / "best.pth"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}\n"
                         f"Train it first: python -m src.train "
                         f"--model {model_name} --mode {mode}")
    model = create_model(model_name, config.NUM_CLASSES, pretrained=False,
                         image_size=config.IMAGE_SIZE)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["classes"]


def attention_shift(cam_partial: np.ndarray, cam_full: np.ndarray) -> dict:
    """Quantify how much the saliency map moved from partial to full FT.

    Both maps are HxW float arrays in [0, 1].
    """
    a = cam_partial.astype(np.float64)
    b = cam_full.astype(np.float64)
    flat_a, flat_b = a.ravel(), b.ravel()

    # Pearson correlation of the two maps (1 = identical focus).
    if flat_a.std() < 1e-8 or flat_b.std() < 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flat_a, flat_b)[0, 1])

    # Mean absolute difference.
    mae = float(np.abs(flat_a - flat_b).mean())

    # IoU of the "hot" regions (top 25% most-attended pixels).
    hot_a = a >= np.quantile(a, 0.75)
    hot_b = b >= np.quantile(b, 0.75)
    union = np.logical_or(hot_a, hot_b).sum()
    hot_iou = float(np.logical_and(hot_a, hot_b).sum() / union) if union else 0.0

    # Distance between the attention centroids (fraction of the diagonal).
    def centroid(cam: np.ndarray) -> np.ndarray:
        h, w = cam.shape
        ys, xs = np.mgrid[0:h, 0:w]
        total = cam.sum()
        if total < 1e-8:
            return np.array([h / 2.0, w / 2.0])
        return np.array([(ys * cam).sum() / total, (xs * cam).sum() / total])

    diagonal = math.hypot(*a.shape)
    centroid_shift = float(
        np.linalg.norm(centroid(a) - centroid(b)) / diagonal)

    return dict(correlation=correlation, mae=mae, hot_region_iou=hot_iou,
                centroid_shift=centroid_shift)


def save_detail_figure(model_name: str, item: dict, classes, path: Path) -> None:
    """Per-image figure: input | partial CAM | full CAM | attention-change map."""
    rgb = item["rgb"]
    cams = item["cams"]
    preds = item["preds"]
    shift = item["shift"]
    diff = cams["full"] - cams["partial"]
    limit = max(abs(float(diff.min())), abs(float(diff.max())), 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"input\n(true: {item['true_class']})")
    axes[1].imshow(show_cam_on_image(rgb, cams["partial"], use_rgb=True))
    axes[1].set_title(f"partial fine-tune\npred: {classes[preds['partial']]}")
    axes[2].imshow(show_cam_on_image(rgb, cams["full"], use_rgb=True))
    axes[2].set_title(f"full fine-tune\npred: {classes[preds['full']]}")
    change = axes[3].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[3].set_title("attention change\n(red: full looks MORE)")
    fig.colorbar(change, ax=axes[3], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{model_name}  Grad-CAM   "
                 f"corr={shift['correlation']:.2f}  "
                 f"hot-IoU={shift['hot_region_iou']:.2f}  "
                 f"centroid-shift={shift['centroid_shift']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_summary(collected: list, classes, n_images: int) -> list:
    """Pick representative images for the summary grid: the most-changed image
    per class first, then fill up to ``n_images`` with the next-most-changed."""
    by_class: dict[str, list] = {}
    for item in collected:
        by_class.setdefault(item["true_class"], []).append(item)

    chosen, chosen_ids = [], set()
    for cls in classes:                              # one per class first
        items = sorted(by_class.get(cls, []),
                       key=lambda d: d["shift"]["centroid_shift"], reverse=True)
        if items:
            chosen.append(items[0])
            chosen_ids.add(id(items[0]))

    for item in sorted(collected, key=lambda d: d["shift"]["centroid_shift"],
                       reverse=True):                # fill with biggest changes
        if len(chosen) >= n_images:
            break
        if id(item) not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(id(item))
    return chosen[:n_images]


def save_summary_grid(model_name: str, items: list, classes, path: Path) -> None:
    """One figure -- the 5 sample images, showing the partial -> full progression.

    Rows are sample images; columns are input, partial Grad-CAM, full Grad-CAM
    and the attention-change map. A shared scale lets the change maps be
    compared directly across rows.
    """
    n_rows = len(items)
    if n_rows == 0:
        return
    # Shared colour scale for all change maps so rows are comparable.
    global_limit = max(
        (max(abs(float((it["cams"]["full"] - it["cams"]["partial"]).min())),
             abs(float((it["cams"]["full"] - it["cams"]["partial"]).max())))
         for it in items), default=1e-6)
    global_limit = max(global_limit, 1e-6)

    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["input", "partial fine-tune", "full fine-tune",
                  "attention change (full - partial)"]

    change = None
    for row, item in enumerate(items):
        rgb = item["rgb"]
        cams = item["cams"]
        preds = item["preds"]
        diff = cams["full"] - cams["partial"]

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_ylabel(item["true_class"], fontsize=11)
        axes[row, 1].imshow(show_cam_on_image(rgb, cams["partial"],
                                              use_rgb=True))
        axes[row, 1].set_xlabel(f"pred: {classes[preds['partial']]}",
                                fontsize=9)
        axes[row, 2].imshow(show_cam_on_image(rgb, cams["full"], use_rgb=True))
        axes[row, 2].set_xlabel(f"pred: {classes[preds['full']]}", fontsize=9)
        change = axes[row, 3].imshow(diff, cmap="RdBu_r",
                                     vmin=-global_limit, vmax=global_limit)
        axes[row, 3].set_xlabel(
            f"centroid shift {item['shift']['centroid_shift']:.2f} | "
            f"corr {item['shift']['correlation']:.2f}", fontsize=9)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, text in enumerate(col_titles):
        axes[0, col].set_title(text, fontsize=12)

    if change is not None:
        fig.colorbar(change, ax=axes[:, 3].tolist(), fraction=0.025, pad=0.02,
                     label="red: full looks more   /   blue: full looks less")
    fig.suptitle(
        f"{model_name}: Grad-CAM progression, partial -> full fine-tuning "
        f"({n_rows} sample images)", fontsize=14)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grad-CAM partial-vs-full comparison for one model.")
    parser.add_argument("--model", required=True, choices=config.MODELS)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    device = get_device(allow_cpu=args.allow_cpu)
    out_dir = config.RESULTS_DIR / "gradcam" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"gradcam.{args.model}", out_dir / "gradcam.log")
    logger.info(f"Grad-CAM analysis for {args.model} on device {device.type}")

    # Load both fine-tuned variants of this model.
    models_by_mode = {}
    classes = None
    for mode in config.FINETUNE_MODES:
        model, classes = load_trained(args.model, mode, device)
        models_by_mode[mode] = model

    _, eval_transform = build_transforms(config.IMAGE_SIZE)
    picks = sample_paths_per_class(
        config.PROCESSED_DIR / "test", config.GRADCAM_IMAGES_PER_CLASS,
        classes, config.RANDOM_SEED)
    logger.info(f"{len(picks)} sample images "
                f"({config.GRADCAM_IMAGES_PER_CLASS} per class), "
                f"modes={list(models_by_mode)}")

    # One Grad-CAM object per fine-tuning mode.
    cam_by_mode = {}
    for mode, model in models_by_mode.items():
        layers, reshape = get_gradcam_layers(model, args.model)
        cam_by_mode[mode] = GradCAM(model=model, target_layers=layers,
                                    reshape_transform=reshape)

    collected: list[dict] = []
    for idx, (path, cls) in enumerate(picks):
        pil = Image.open(path).convert("RGB").resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE))
        rgb = np.asarray(pil).astype(np.float32) / 255.0
        cls_idx = classes.index(cls)
        targets = [ClassifierOutputTarget(cls_idx)]

        cams, preds = {}, {}
        for mode, model in models_by_mode.items():
            # requires_grad on the input keeps the autograd graph connected
            # even when the backbone is fully frozen (partial mode).
            tensor = eval_transform(pil).unsqueeze(0).to(device)
            tensor.requires_grad_(True)
            cams[mode] = cam_by_mode[mode](input_tensor=tensor,
                                           targets=targets)[0]
            with torch.no_grad():
                preds[mode] = int(model(tensor.detach()).argmax(1).item())

        item = dict(idx=idx, image=path.name, true_class=cls, rgb=rgb,
                    cams=cams, preds=preds,
                    shift=attention_shift(cams["partial"], cams["full"]))
        collected.append(item)
        save_detail_figure(args.model, item, classes,
                           out_dir / f"{idx:02d}_{cls}_{path.stem}.png")

    # The single 5-image summary grid.
    summary_items = select_summary(collected, classes,
                                   config.GRADCAM_SUMMARY_IMAGES)
    save_summary_grid(args.model, summary_items, classes,
                      out_dir / "gradcam_summary.png")
    logger.info(f"summary grid: {len(summary_items)} images -> "
                f"gradcam_summary.png")

    # Aggregate the attention-shift statistics.
    keys = ["correlation", "mae", "hot_region_iou", "centroid_shift"]
    mean = {k: float(np.mean([it["shift"][k] for it in collected]))
            for k in keys}
    std = {k: float(np.std([it["shift"][k] for it in collected])) for k in keys}
    records = [
        dict(image=it["image"], true_class=it["true_class"],
             pred_partial=classes[it["preds"]["partial"]],
             pred_full=classes[it["preds"]["full"]], **it["shift"])
        for it in collected
    ]
    save_json(dict(model=args.model, n_images=len(collected),
                   mean=mean, std=std, per_image=records),
              out_dir / "attention_shift.json")

    logger.info("=" * 64)
    logger.info(f"Partial -> full attention change ({len(collected)} images):")
    logger.info(f"  CAM correlation : {mean['correlation']:.3f}  "
                f"(1.0 = identical focus)")
    logger.info(f"  hot-region IoU  : {mean['hot_region_iou']:.3f}")
    logger.info(f"  centroid shift  : {mean['centroid_shift']:.3f}  "
                f"(fraction of image diagonal)")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
