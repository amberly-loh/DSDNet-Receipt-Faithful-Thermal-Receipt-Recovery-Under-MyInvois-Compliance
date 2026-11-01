from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(root: str | Path) -> list[Path]:
    base = Path(root)
    return sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_gray(path: str | Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ext = target.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Unable to encode image: {target}")
    encoded.tofile(str(target))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: np.ndarray, size: int = 8) -> int:
    resized = cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_quality(image: np.ndarray) -> dict[str, float]:
    normalized = image.astype(np.float32) / 255.0
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    contrast = float(normalized.std())
    exposure = float(1.0 - abs(normalized.mean() - 0.75) / 0.75)
    threshold, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground_ratio = float((image < threshold).mean())
    foreground_score = max(0.0, 1.0 - abs(foreground_ratio - 0.15) / 0.15)
    score = (
        0.35 * min(sharpness / 1000.0, 1.0)
        + 0.30 * min(contrast / 0.25, 1.0)
        + 0.20 * max(0.0, exposure)
        + 0.15 * foreground_score
    )
    return {
        "quality_score": float(score),
        "sharpness": sharpness,
        "contrast": contrast,
        "exposure_score": exposure,
        "foreground_ratio": foreground_ratio,
    }

