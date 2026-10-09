from __future__ import annotations

import numpy as np

from dsdnet_receipt.baselines import otsu, perceptual_filter, sauvola
from dsdnet_receipt.degradation import degrade, variant_seed
from dsdnet_receipt.metrics import cer, psnr, ssim, wer


def test_seed_is_stable() -> None:
    assert variant_seed(56, "SRC-0001", "V01", "light") == variant_seed(
        56, "SRC-0001", "V01", "light"
    )


def test_degradation_is_reproducible() -> None:
    image = np.full((128, 128), 255, dtype=np.uint8)
    image[30:90, 50:55] = 0
    config = {
        "degradation": {
            "severities": {
                "light": {
                    "blur_sigma": [0.3, 0.6],
                    "contrast": [0.8, 1.0],
                    "noise_sigma": [0.0, 0.02],
                    "foreground_dropout": [0.0, 0.05],
                    "illumination_amplitude": [0.0, 0.1],
                }
            },
            "abrasion_area": [0.0, 0.05],
            "visibility_threshold": 0.05,
        }
    }
    first, first_masks, _ = degrade(image, "light", config, 123)
    second, second_masks, _ = degrade(image, "light", config, 123)
    assert np.array_equal(first, second)
    assert all(np.array_equal(first_masks[key], second_masks[key]) for key in first_masks)


def test_baselines_preserve_shape() -> None:
    image = np.arange(4096, dtype=np.uint8).reshape(64, 64)
    assert otsu(image).shape == image.shape
    assert sauvola(image).shape == image.shape
    assert perceptual_filter(image).shape == image.shape


def test_metrics_identity() -> None:
    image = np.full((32, 32), 127, dtype=np.uint8)
    assert psnr(image, image) == float("inf")
    assert abs(ssim(image, image) - 1.0) < 1e-9
    assert cer("abc", "abc") == 0.0
    assert wer("a b", "a b") == 0.0

