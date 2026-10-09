from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import write_json
from .images import read_gray, write_image


def variant_seed(base_seed: int, source_id: str, variant_id: str, severity: str) -> int:
    value = f"{base_seed}|{source_id}|{variant_id}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big", signed=False)


def foreground_mask(clean: np.ndarray) -> np.ndarray:
    threshold, _ = cv2.threshold(clean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean < threshold


def illumination_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    grid = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
    grid -= float(grid.mean())
    maximum = float(np.abs(grid).max())
    if maximum > 0:
        grid /= maximum
    return cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def abrasion_mask(
    rng: np.random.Generator, foreground: np.ndarray, target_fraction: float
) -> np.ndarray:
    height, width = foreground.shape
    target = min(int(round(target_fraction * height * width)), int(foreground.sum()))
    mask = np.zeros((height, width), dtype=np.uint8)
    locations = np.argwhere(foreground)
    if target <= 0 or len(locations) == 0:
        return mask.astype(bool)
    directions = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    attempts = 0
    covered = 0
    maximum_attempts = max(50, min(5000, target // 8 + 50))
    while covered < target and attempts < maximum_attempts:
        row, col = locations[int(rng.integers(0, len(locations)))]
        walk = int(rng.integers(12, 80))
        points = [(int(col), int(row))]
        for _ in range(walk):
            dr, dc = directions[int(rng.integers(0, len(directions)))]
            step = int(rng.integers(1, 5))
            row = int(np.clip(row + dr * step, 0, height - 1))
            col = int(np.clip(col + dc * step, 0, width - 1))
            points.append((int(col), int(row)))
        cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 1, thickness=int(rng.integers(2, 7)))
        covered = int(np.count_nonzero(mask & foreground.astype(np.uint8)))
        attempts += 1
    return mask.astype(bool) & foreground


def degrade(
    clean_uint8: np.ndarray,
    severity: str,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    ranges = config["degradation"]["severities"][severity]
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = {name: float(rng.uniform(bounds[0], bounds[1])) for name, bounds in ranges.items()}
    abrasion_fraction = float(rng.uniform(*config["degradation"]["abrasion_area"]))
    clean = clean_uint8.astype(np.float32) / 255.0
    foreground = foreground_mask(clean_uint8)
    background = cv2.GaussianBlur(clean, (0, 0), sigmaX=15.0)

    blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=sampled["blur_sigma"])
    median = float(np.median(blurred))
    transformed = median + sampled["contrast"] * (blurred - median)
    illumination = illumination_field(rng, clean.shape)
    transformed *= np.clip(1.0 + sampled["illumination_amplitude"] * illumination, 0.0, 2.0)

    foreground_indices = np.argwhere(foreground)
    dropout = np.zeros_like(foreground, dtype=bool)
    dropout_count = int(round(sampled["foreground_dropout"] * len(foreground_indices)))
    if dropout_count > 0 and len(foreground_indices) > 0:
        chosen = rng.choice(len(foreground_indices), size=min(dropout_count, len(foreground_indices)), replace=False)
        pixels = foreground_indices[chosen]
        dropout[pixels[:, 0], pixels[:, 1]] = True

    abrasion = abrasion_mask(rng, foreground, abrasion_fraction)
    removed = dropout | abrasion
    transformed[removed] = background[removed]
    noise = rng.normal(0.0, sampled["noise_sigma"], clean.shape).astype(np.float32)
    degraded = np.clip(transformed + noise, 0.0, 1.0)

    observable = np.abs(degraded - background) >= float(config["degradation"]["visibility_threshold"])
    support = foreground & observable
    masks = {
        "clean_foreground": foreground,
        "degradation_removal": removed,
        "observable_trace": observable,
        "support_target": support,
    }
    parameters = {
        "severity": severity,
        "seed": int(seed),
        **sampled,
        "abrasion_fraction": abrasion_fraction,
        "transformation_order": [
            "gaussian_blur",
            "contrast_reduction",
            "low_frequency_multiplicative_illumination",
            "foreground_dropout",
            "abrasion",
            "gaussian_noise",
        ],
    }
    return (degraded * 255.0 + 0.5).astype(np.uint8), masks, parameters


def generate_variant(
    source_path: str | Path,
    source_id: str,
    variant_id: str,
    severity: str,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = read_gray(source_path)
    maximum = int(config["preprocessing"]["max_long_edge"])
    scale = min(1.0, maximum / max(source.shape))
    if scale < 1.0:
        source = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    seed = variant_seed(config["base_seed"], source_id, variant_id, severity)
    degraded, masks, parameters = degrade(source, severity, config, seed)
    target = Path(output_dir) / source_id / f"{variant_id}_{severity}"
    write_image(target / "clean_reference.png", source)
    write_image(target / "degraded.png", degraded)
    for name, mask in masks.items():
        write_image(target / f"{name}.png", mask.astype(np.uint8) * 255)
    write_json(target / "parameters.json", parameters)
    return {
        "source_id": source_id,
        "variant_id": variant_id,
        "severity": severity,
        "seed": seed,
        "source_path": str(Path(source_path).resolve()),
        "output_path": str((target / "degraded.png").resolve()),
        **parameters,
    }
