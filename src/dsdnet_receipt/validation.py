from __future__ import annotations

import numpy as np


def stroke_fidelity_metrics(
    clean: np.ndarray,
    recovered: np.ndarray,
    clean_foreground: np.ndarray,
    *,
    minimum_ink: float = 8.0,
    retention_fraction: float = 0.35,
    false_ink_delta: float = 20.0,
) -> dict[str, float]:
    """Measure retained, deleted and falsely-created dark strokes.

    The supplied foreground mask comes from the clean reference, not OCR.  A
    foreground pixel is retained when the recovered image still contains a
    conservative fraction of its clean-reference darkness.  A background
    pixel is false ink only when recovery makes it materially darker than the
    clean reference.  These definitions make the safety trade-off explicit
    instead of allowing PSNR/SSIM to hide small but important text damage.
    """
    if clean.shape != recovered.shape or clean.shape != clean_foreground.shape:
        raise ValueError("clean, recovered and clean_foreground must share a shape")

    clean_f = clean.astype(np.float32)
    recovered_f = recovered.astype(np.float32)
    foreground = clean_foreground.astype(bool)
    background = ~foreground
    clean_ink = 255.0 - clean_f
    recovered_ink = 255.0 - recovered_f
    required_ink = np.maximum(minimum_ink, clean_ink * retention_fraction)
    retained = foreground & (recovered_ink >= required_ink)
    deleted = foreground & ~retained
    false_ink = background & ((recovered_ink - clean_ink) >= false_ink_delta)

    foreground_count = max(int(foreground.sum()), 1)
    background_count = max(int(background.sum()), 1)
    return {
        "stroke_recall": float(retained.sum() / foreground_count),
        "deleted_stroke_rate": float(deleted.sum() / foreground_count),
        "false_stroke_rate": float(false_ink.sum() / background_count),
        "retained_stroke_pixels": int(retained.sum()),
        "deleted_stroke_pixels": int(deleted.sum()),
        "false_stroke_pixels": int(false_ink.sum()),
        "foreground_pixels": int(foreground.sum()),
        "background_pixels": int(background.sum()),
    }


def masked_residual_darkness(
    clean: np.ndarray, recovered: np.ndarray, artifact_mask: np.ndarray
) -> float:
    """Return added darkness inside a known synthetic artifact region."""
    if clean.shape != recovered.shape or clean.shape != artifact_mask.shape:
        raise ValueError("clean, recovered and artifact_mask must share a shape")
    mask = artifact_mask.astype(bool)
    if not mask.any():
        return 0.0
    added_darkness = np.maximum(
        clean.astype(np.float32) - recovered.astype(np.float32), 0.0
    )
    return float(added_darkness[mask].mean() / 255.0)
