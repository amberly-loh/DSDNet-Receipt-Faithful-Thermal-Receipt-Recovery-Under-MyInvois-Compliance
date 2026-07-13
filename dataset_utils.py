import csv
import gc
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


SEVERITY_CONFIG = {
    "light": {
        "blur": 1,
        "fade": 0.10,
        "contrast": 0.90,
        "noise": 0.02,
        "abrasion": 0.02,
    },
    "moderate": {
        "blur": 2,
        "fade": 0.20,
        "contrast": 0.75,
        "noise": 0.05,
        "abrasion": 0.05,
    },
    "severe": {
        "blur": 3,
        "fade": 0.30,
        "contrast": 0.60,
        "noise": 0.10,
        "abrasion": 0.10,
    },
}


def resize_if_too_large(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def apply_fade(image: np.ndarray, intensity: float) -> np.ndarray:
    overlay = np.full_like(image, 255, dtype=np.uint8)
    return cv2.addWeighted(image, 1.0 - intensity, overlay, intensity, 0)


def apply_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image
    return cv2.GaussianBlur(image, (radius * 2 + 1, radius * 2 + 1), 0)


def apply_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    pil = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil)
    return np.array(enhancer.enhance(factor))


def apply_noise(image: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image
    # Use int16 to avoid huge float64 memory allocations
    noise = np.random.normal(0, amount * 255, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_uneven_illumination(image: np.ndarray, intensity: float) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), max(h, w) // 2, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.5, sigmaY=h * 0.5)
    mask = cv2.normalize(mask, None, 1 - intensity, 1.0, cv2.NORM_MINMAX)
    if len(image.shape) == 3:
        mask = mask[:, :, None]
    result = image.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_abrasion(image: np.ndarray, amount: float) -> np.ndarray:
    h, w = image.shape[:2]
    torn = image.copy()
    num_shapes = max(1, int(amount * 20))
    for _ in range(num_shapes):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        x1 = min(w, x0 + random.randint(10, max(10, int(w * amount))))
        y1 = min(h, y0 + random.randint(5, max(5, int(h * amount))))
        cv2.rectangle(torn, (x0, y0), (x1, y1), (255, 255, 255), -1)
    return torn


def build_trace_mask(image: np.ndarray, amount: float) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = mask.shape
    num_regions = max(1, int(amount * 10))
    for _ in range(num_regions):
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)
        max_dx = max(10, int(w * amount))
        max_dy = max(10, int(h * amount))
        x1 = min(w, x0 + random.randint(10, max_dx))
        y1 = min(h, y0 + random.randint(10, max_dy))
        mask[y0:y1, x0:x1] = 255
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)
    return (mask > 128).astype(np.uint8) * 255


def degrade_image(image: np.ndarray, severity: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = SEVERITY_CONFIG[severity]
    degraded = image.copy()
    degraded = apply_fade(degraded, cfg["fade"])
    degraded = apply_blur(degraded, cfg["blur"])
    degraded = apply_contrast(degraded, cfg["contrast"])
    degraded = apply_uneven_illumination(degraded, cfg["fade"])
    degraded = apply_abrasion(degraded, cfg["abrasion"])
    degraded = apply_noise(degraded, cfg["noise"])
    trace_mask = build_trace_mask(degraded, cfg["abrasion"] + cfg["noise"])
    return degraded, trace_mask


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_synthetic_dataset(
    clean_dir: str,
    output_dir: str,
    target_count: int = 350,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean receipt folder not found: {clean_path.resolve()}")

    records = []
    images = sorted([f for f in clean_path.glob("**/*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff"]])
    selected = images[:target_count]
    if len(selected) == 0:
        raise ValueError(f"No clean images found inside: {clean_path.resolve()}")

    total_images = len(selected)
    print(f"Starting synthetic generation for {total_images} clean receipts...", flush=True)

    for idx, img_path in enumerate(selected, start=1):
        try:
            pil_img = Image.open(img_path).convert("RGB")
            image = np.array(pil_img)
            
            # Prevent RAM spike by resizing massive phone scans
            image = resize_if_too_large(image, max_dim=1280)

            severity = random.choice(list(SEVERITY_CONFIG.keys()))
            degraded, trace_mask = degrade_image(image, severity)
            stem = img_path.stem

            clean_out = output_path / "clean" / f"{stem}.png"
            degraded_out = output_path / "degraded" / f"{stem}_{severity}.png"
            mask_out = output_path / "masks" / f"{stem}_{severity}_mask.png"

            save_image(clean_out, image)
            save_image(degraded_out, degraded)
            save_image(mask_out, trace_mask)

            records.append(
                {
                    "image_id": stem,
                    "clean_path": str(clean_out),
                    "degraded_path": str(degraded_out),
                    "mask_path": str(mask_out),
                    "severity": severity,
                    "source_path": str(img_path),
                }
            )

            # Print progress every 10 images
            if idx % 10 == 0 or idx == total_images:
                print(f"[{idx}/{total_images}] Generated: {stem} ({severity})", flush=True)

            # Cleanup memory each iteration
            del image, degraded, trace_mask
            gc.collect()

        except Exception as e:
            print(f"Failed to process {img_path.name}: {e}", flush=True)

    metadata_path = output_path / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "clean_path", "degraded_path", "mask_path", "severity", "source_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSynthetic dataset successfully generated: {len(records)} pairs", flush=True)
    print(f"Metadata saved to {metadata_path.resolve()}", flush=True)


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    INPUT_CLEAN_DIR = "/workspaces/DSDNet-Receipt-Faithful-Thermal-Receipt-Recovery-Under-MyInvois-Compliance/Datasets/Real life/clean"
    OUTPUT_DATASET_DIR = "/workspaces/DSDNet-Receipt-Faithful-Thermal-Receipt-Recovery-Under-MyInvois-Compliance/Datasets/Real life/synthetic"

    build_synthetic_dataset(
        clean_dir=INPUT_CLEAN_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        target_count=350,
        seed=42,
    )