from __future__ import annotations

import numpy as np
import cv2

from dsdnet_receipt.baselines import otsu, perceptual_filter, sauvola
from dsdnet_receipt.degradation import degrade, variant_seed
from dsdnet_receipt.metrics import cer, psnr, ssim, wer
from dsdnet_receipt.fields import extract_fields, route_fields
from dsdnet_receipt.preprocessing import preprocess


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


def test_target_field_extraction() -> None:
    text = "DOCUMENT NO: TD01167104\nDATE: 25/12/2018\nTOTAL: RM 9.00"
    fields = extract_fields(text, [{"text": text, "confidence": 0.91}])
    assert fields["transaction_date"]["value"] == "25/12/2018"
    assert fields["receipt_invoice_number"]["value"] == "TD01167104"
    assert fields["total_amount"]["value"] == "9.00"


def test_natural_receipt_never_auto_accepts_without_support() -> None:
    extracted = {
        "transaction_date": {"value": "25/12/2018", "confidence": 0.99},
        "receipt_invoice_number": {"value": "TD01167104", "confidence": 0.99},
        "total_amount": {"value": "9.00", "confidence": 0.99},
    }
    decisions = route_fields(extracted, support_available=False, model_available=True)
    assert all(item.decision == "Human decision required" for item in decisions)
    assert all(item.support_status == "unavailable" for item in decisions)


def test_preprocessing_corrects_quarter_turn_receipt() -> None:
    receipt = np.full((600, 300), 245, dtype=np.uint8)
    for y in range(80, 540, 35):
        cv2.line(receipt, (35, y), (265, y), 20, 3)
    sideways = cv2.rotate(receipt, cv2.ROTATE_90_CLOCKWISE)
    processed, record = preprocess(sideways, {"preprocessing": {"max_long_edge": 1600, "boundary_detection": False, "orientation_correction": True}})
    assert record["orientation_correction"]["coarse_rotation_degrees"] == 90
    assert processed.shape[0] > processed.shape[1]
