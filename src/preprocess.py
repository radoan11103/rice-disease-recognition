"""Stage 1 -- state-of-the-art preprocessing + cross-year dataset builder.

What this does
--------------
1. Scans the raw dataset (see directory_structure.txt for the layout).
2. Splits it for the research question:
       * 2021-2025  ->  train / validation pool   (the "past")
       * 2026       ->  test set                  (the "future")
   The test set is stratified-sampled to a balanced TEST_IMAGES_PER_CLASS.
3. Cleans every image with a Standard-SOTA pipeline:
       EXIF orientation fix  ->  edge-preserving bilateral denoise
       ->  CLAHE contrast enhancement (LAB L-channel)
       ->  centre square crop  ->  high-quality resize to IMAGE_SIZE.
4. Writes one shared dataset to PROCESSED_DIR so that *every* model later
   trains and tests on exactly the same images.

Run:  python -m src.preprocess           (add --force to rebuild)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Allow running both as `python -m src.preprocess` and `python src/preprocess.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from src.utils import get_logger, save_json

# OpenCV must not spawn its own thread pool inside each worker process.
cv2.setNumThreads(1)


# ===========================================================================
# IMAGE ENHANCEMENT
# ===========================================================================
def enhance_image(pil_image: Image.Image, size: int) -> Image.Image:
    """Apply the Standard-SOTA cleaning pipeline and return a square RGB image."""
    # 1. Respect the camera EXIF orientation, then force 3-channel RGB.
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    rgb = np.asarray(pil_image)                       # H x W x 3, uint8, RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 2. Edge-preserving denoise -- removes sensor noise while keeping the
    #    sharp boundaries of disease lesions intact.
    bgr = cv2.bilateralFilter(
        bgr, d=config.BILATERAL_D,
        sigmaColor=config.BILATERAL_SIGMA, sigmaSpace=config.BILATERAL_SIGMA)

    # 3. CLAHE on the L channel in LAB space -- boosts local contrast so that
    #    brown spots / blast lesions stand out without blowing out colour.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP,
        tileGridSize=(config.CLAHE_GRID, config.CLAHE_GRID))
    l_chan = clahe.apply(l_chan)
    bgr = cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 4. Centre square crop (keeps the leaf, drops letterbox borders).
    height, width = rgb.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    rgb = rgb[top:top + side, left:left + side]

    # 5. Resize -- INTER_AREA gives the cleanest down-scaling.
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(rgb)


def _process_one(task: tuple[str, str]) -> tuple[bool, str, str]:
    """Worker: read one image, enhance it, save it. Returns (ok, src, error)."""
    src_path, dst_path = task
    try:
        # First pass: detect truncated / corrupt files.
        with Image.open(src_path) as probe:
            probe.verify()
        # Second pass: actually decode and process (verify() leaves it unusable).
        with Image.open(src_path) as image:
            image.load()
            processed = enhance_image(image, config.IMAGE_SIZE)
            processed.save(dst_path, format="JPEG",
                           quality=config.JPEG_QUALITY, optimize=True,
                           subsampling=0)
        return (True, src_path, "")
    except Exception as exc:                          # noqa: BLE001 - report all
        return (False, src_path, repr(exc))


# ===========================================================================
# DATASET SCANNING + SPLITTING
# ===========================================================================
def scan_dataset(logger) -> tuple[list, list]:
    """Walk RAW_DATA_DIR and return (trainval_items, test_items).

    Each item is (absolute_path, canonical_class, year).
    """
    if not config.RAW_DATA_DIR.exists():
        raise SystemExit(
            f"Raw data folder not found: {config.RAW_DATA_DIR}\n"
            "Set RICE_RAW_DIR or place the data there. See README.md.")

    trainval, test = [], []
    known_years = set(config.TRAIN_YEARS) | set(config.TEST_YEARS)

    for root, _dirs, files in os.walk(config.RAW_DATA_DIR):
        parts = Path(root).parts

        # Detect the class and year from anywhere in the path.
        detected_class = next(
            (config.CLASS_MAPPING[p] for p in parts if p in config.CLASS_MAPPING),
            None)
        detected_year = next((p for p in parts if p in known_years), None)
        if detected_class is None or detected_year is None:
            continue

        for name in files:
            if Path(name).suffix.lower() not in config.VALID_EXTENSIONS:
                continue
            item = (os.path.join(root, name), detected_class, detected_year)
            if detected_year in config.TRAIN_YEARS:
                trainval.append(item)
            else:
                test.append(item)

    logger.info(f"Scanned {config.RAW_DATA_DIR}")
    logger.info(f"  found {len(trainval)} images in train years "
                f"{config.TRAIN_YEARS}")
    logger.info(f"  found {len(test)} images in test years "
                f"{config.TEST_YEARS}")
    if not trainval or not test:
        raise SystemExit(
            "No images found for one of the splits. Check RICE_RAW_DIR and "
            "that the folder names match directory_structure.txt.")
    return trainval, test


def build_splits(trainval: list, test: list, logger) -> dict[str, list]:
    """Make balanced future test set + stratified train/val split."""
    rng = random.Random(config.RANDOM_SEED)

    # --- TEST: stratified, balanced sample from the future year ------------
    test_by_class: dict[str, list] = defaultdict(list)
    for item in test:
        test_by_class[item[1]].append(item)

    test_data: list = []
    for cls in config.CLASS_NAMES:
        available = test_by_class.get(cls, [])
        take = min(config.TEST_IMAGES_PER_CLASS, len(available))
        if take < config.TEST_IMAGES_PER_CLASS:
            logger.warning(f"  test class '{cls}': only {len(available)} images "
                           f"available (< {config.TEST_IMAGES_PER_CLASS})")
        test_data.extend(rng.sample(available, take))

    # --- TRAIN / VAL: stratified split of the past years -------------------
    paths = [it[0] for it in trainval]
    labels = [it[1] for it in trainval]
    years = [it[2] for it in trainval]
    (tr_p, va_p, tr_l, va_l, tr_y, va_y) = train_test_split(
        paths, labels, years,
        test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED,
        shuffle=True, stratify=labels)

    splits = {
        "train": list(zip(tr_p, tr_l, tr_y)),
        "val": list(zip(va_p, va_l, va_y)),
        "test": test_data,
    }
    total = sum(len(v) for v in splits.values())
    logger.info("Split sizes:")
    for name, data in splits.items():
        logger.info(f"  {name:5s}: {len(data):6d}  ({100 * len(data) / total:.1f}%)")
    return splits


# ===========================================================================
# MAIN
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cross-year dataset.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if PROCESSED_DIR already exists")
    parser.add_argument("--workers", type=int, default=config.PREPROCESS_WORKERS,
                        help="parallel worker processes")
    args = parser.parse_args()

    logger = get_logger("preprocess", config.RESULTS_DIR / "preprocess.log")

    done_marker = config.PROCESSED_DIR / "manifest.json"
    if done_marker.exists() and not args.force:
        logger.info(f"Processed dataset already present at {config.PROCESSED_DIR}")
        logger.info("Nothing to do (use --force to rebuild).")
        return

    trainval, test = scan_dataset(logger)
    splits = build_splits(trainval, test, logger)

    # Build the full task list with collision-free destination names.
    tasks: list[tuple[str, str]] = []
    class_counts: dict[str, dict[str, int]] = {}
    for split_name, items in splits.items():
        per_class: dict[str, int] = defaultdict(int)
        for index, (src, cls, _year) in enumerate(items):
            dst_dir = config.PROCESSED_DIR / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Running index guarantees a unique name even when different
            # source datasets contain files with the same stem.
            dst = dst_dir / f"{split_name}_{cls}_{index:06d}.jpg"
            tasks.append((src, str(dst)))
            per_class[cls] += 1
        class_counts[split_name] = dict(per_class)

    logger.info(f"Processing {len(tasks)} images with {args.workers} workers...")
    failures: list[tuple[str, str]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_process_one, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures),
                               ncols=100, desc="enhance"):
                ok, src, err = future.result()
                if not ok:
                    failures.append((src, err))
    else:
        for task in tqdm(tasks, ncols=100, desc="enhance"):
            ok, src, err = _process_one(task)
            if not ok:
                failures.append((src, err))

    for src, err in failures:
        logger.warning(f"FAILED {src} :: {err}")

    manifest = {
        "image_size": config.IMAGE_SIZE,
        "classes": config.CLASS_NAMES,
        "train_years": config.TRAIN_YEARS,
        "test_years": config.TEST_YEARS,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "class_counts": class_counts,
        "processed": len(tasks) - len(failures),
        "failed": len(failures),
        "preprocessing": "exif + bilateral denoise + CLAHE(LAB) + square crop + resize",
    }
    save_json(manifest, done_marker)

    logger.info("=" * 64)
    logger.info(f"Preprocessing complete: {manifest['processed']} ok, "
                f"{manifest['failed']} failed.")
    logger.info(f"Cross-year dataset saved to: {config.PROCESSED_DIR}")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
