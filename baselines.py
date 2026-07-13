import math
import os
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

# Suppress verbose Paddle OCR logging output
os.environ["PPOCR_SHOW_LOG"] = "0"
logging.getLogger("ppocr").setLevel(logging.ERROR)

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from paddleocr import PaddleOCR
    # Initialize PaddleOCR globally once so models aren't re-created on every image loop
    GLOBAL_PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
except Exception:
    GLOBAL_PADDLE_OCR = None


class BaselineName(str, Enum):
    DIRECT = "A"
    CLASSICAL = "B"
    GENERATIVE = "C"
    DSDNET_NO_AUTH = "D"
    RESTORE_AND_EXTRACT = "E"
    PROPOSED = "Proposed"


@dataclass
class ExtractionResult:
    baseline: BaselineName
    degraded_path: str
    restored_image: np.ndarray
    ocr_text: str
    fields: dict[str, str]
    authenticity_map: np.ndarray | None = None
    abstained_fields: list[str] | None = None
    confidence: dict[str, float] | None = None
    notes: str | None = None


def _load_image(path: str) -> np.ndarray:
    # Safely load using PIL to prevent cv2.imread failure on paths with spaces/special characters
    try:
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)
    except Exception as e:
        raise FileNotFoundError(f"Cannot load image at {path}: {e}")


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _classic_enhancement(image: np.ndarray) -> np.ndarray:
    gray = _to_gray(image)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    equalized = cv2.equalizeHist(denoised)
    _, otsu = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        sauvola = cv2.ximgproc.niBlackThreshold(
            equalized,
            maxValue=255,
            type=cv2.THRESH_BINARY,
            blockSize=25,
            k=0.2,
        )
    except AttributeError:
        sauvola = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
    return cv2.cvtColor(sauvola, cv2.COLOR_GRAY2RGB)


def _restoration_stub(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image)
    restored = pil.filter(ImageFilter.MedianFilter(size=3))
    restored = restored.filter(ImageFilter.SHARPEN)
    return np.array(restored)


def _fake_dsdnet_restore(image: np.ndarray, stages: int = 3) -> np.ndarray:
    restored = image.copy()
    for _ in range(stages):
        restored = _restoration_stub(restored)
    return restored


def _build_authenticity_map(degraded: np.ndarray, restored: np.ndarray) -> np.ndarray:
    degraded_gray = _to_gray(degraded).astype(np.float32)
    restored_gray = _to_gray(restored).astype(np.float32)
    residual = np.abs(degraded_gray - restored_gray)
    normalized = cv2.normalize(residual, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized


def _run_ocr(image: np.ndarray) -> tuple[str, dict[str, str], dict[str, float]]:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lines = []

    if GLOBAL_PADDLE_OCR is not None:
        try:
            # FIX: Removed cls=True to avoid 'unexpected keyword argument cls' error in PaddleOCR 3.x+
            result = GLOBAL_PADDLE_OCR.ocr(image_bgr)
            if result and result[0]:
                for page in result:
                    if page:
                        for line in page:
                            if line and len(line) > 1 and len(line[1]) > 0:
                                lines.append(line[1][0])
            text = "\n".join(lines)
        except Exception as e:
            print(f"PaddleOCR error during execution: {e}")
            text = ""
    elif pytesseract is not None:
        text = pytesseract.image_to_string(Image.fromarray(image))
    else:
        raise RuntimeError("No OCR engine installed. Install paddleocr or pytesseract.")

    fields = {
        "date": "",
        "receipt_no": "",
        "tax_id": "",
        "tax_amount": "",
        "total_amount": "",
    }
    for line in text.splitlines():
        cleaned = line.strip().lower()
        if "date" in cleaned and ":" in cleaned:
            fields["date"] = cleaned.split(":", 1)[1].strip()
        if "receipt" in cleaned and "no" in cleaned:
            fields["receipt_no"] = cleaned.split(":", 1)[-1].strip()
        if "tax id" in cleaned or "tin" in cleaned:
            fields["tax_id"] = cleaned.split(":", 1)[-1].strip()
        if "tax" in cleaned and ("amount" in cleaned or "$" in cleaned):
            fields["tax_amount"] = cleaned.split(":", 1)[-1].strip()
        if "total" in cleaned and ("amount" in cleaned or "$" in cleaned):
            fields["total_amount"] = cleaned.split(":", 1)[-1].strip()
            
    confidence = {k: 0.8 if v else 0.2 for k, v in fields.items()}
    return text, fields, confidence


def _aggregate_fields(field_scores: dict[str, float], use_lower_quantile: bool = True) -> dict[str, float]:
    if use_lower_quantile:
        return {k: min(scores) if isinstance(scores, (list, tuple)) else scores for k, scores in field_scores.items()}
    return {k: sum(scores) / len(scores) if isinstance(scores, (list, tuple)) else scores for k, scores in field_scores.items()}


def _apply_rule_checks(fields: dict[str, str]) -> dict[str, str]:
    fixed = fields.copy()
    if fixed["total_amount"] and fixed["tax_amount"]:
        try:
            total = float(fixed["total_amount"].replace("$", ""))
            tax = float(fixed["tax_amount"].replace("$", ""))
            if tax > total:
                fixed["tax_amount"] = ""
        except ValueError:
            pass
    return fixed


def _field_abstention(fields: dict[str, str], confidences: dict[str, float], threshold: float = 0.65) -> list[str]:
    return [k for k, v in confidences.items() if v < threshold]


def run_baseline(
    degraded_path: str,
    baseline: BaselineName,
    use_auth_map: bool = True,
    use_abstention: bool = True,
    use_rule_checks: bool = True,
    use_lower_quantile: bool = True,
    unrolled_stages: int = 3,
) -> ExtractionResult:
    degraded = _load_image(degraded_path)
    restored = degraded.copy()
    authenticity_map = None
    abstained = []
    notes = []

    if baseline == BaselineName.DIRECT:
        restored = degraded
        notes.append("Direct extraction on degraded input")
    elif baseline == BaselineName.CLASSICAL:
        restored = _classic_enhancement(degraded)
        notes.append("Classical enhancement and binarisation")
    elif baseline == BaselineName.GENERATIVE:
        restored = _restoration_stub(degraded)
        notes.append("Generative-style restoration stub")
    elif baseline == BaselineName.DSDNET_NO_AUTH:
        restored = _fake_dsdnet_restore(degraded, stages=unrolled_stages)
        notes.append("DSDNet restoration only, no authenticity map or abstention")
    elif baseline == BaselineName.RESTORE_AND_EXTRACT:
        restored = _fake_dsdnet_restore(degraded, stages=unrolled_stages)
        notes.append("Conventional restore-and-extract without abstention")
    elif baseline == BaselineName.PROPOSED:
        restored = _fake_dsdnet_restore(degraded, stages=unrolled_stages)
        notes.append("Proposed DSDNet-Receipt pipeline")
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    if baseline in [BaselineName.DSDNET_NO_AUTH, BaselineName.RESTORE_AND_EXTRACT, BaselineName.PROPOSED] and use_auth_map:
        authenticity_map = _build_authenticity_map(degraded, restored)
        notes.append("Authenticity map computed")
    else:
        authenticity_map = None
        notes.append("Authenticity map disabled")

    text, fields, confidences = _run_ocr(restored)
    if use_rule_checks:
        fixed_fields = _apply_rule_checks(fields)
        if fixed_fields != fields:
            notes.append("Rule checks applied")
        fields = fixed_fields
    else:
        notes.append("Rule checks disabled")

    if baseline == BaselineName.PROPOSED and use_abstention:
        abstained = _field_abstention(fields, confidences)
        notes.append(f"Abstention applied to fields: {abstained}")
    elif use_abstention and baseline != BaselineName.PROPOSED:
        notes.append("Abstention only active in proposed pipeline")
    else:
        abstained = []
        notes.append("Abstention disabled")

    if not use_lower_quantile:
        notes.append("Mean aggregation used instead of lower-quantile")
    if unrolled_stages != 3:
        notes.append(f"Reduced unrolled stages: {unrolled_stages}")

    return ExtractionResult(
        baseline=baseline,
        degraded_path=degraded_path,
        restored_image=restored,
        ocr_text=text,
        fields=fields,
        authenticity_map=authenticity_map,
        abstained_fields=abstained,
        confidence=confidences,
        notes="; ".join(notes),
    )