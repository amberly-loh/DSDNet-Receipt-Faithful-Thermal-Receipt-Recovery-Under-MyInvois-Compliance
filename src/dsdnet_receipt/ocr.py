from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OCRResult:
    text: str
    lines: list[dict[str, Any]]
    engine_version: str
    status: str
    error: str | None = None


class PaddleOCRAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.version = importlib.metadata.version("paddleocr")
        self.required_version = str(config["ocr"]["required_version"])
        self._engine = None

    @property
    def version_matches_protocol(self) -> bool:
        return self.version == self.required_version

    def _load(self) -> Any:
        if self._engine is None:
            from paddleocr import PaddleOCR

            if self.version.startswith("2.7"):
                self._engine = PaddleOCR(
                    lang=self.config["ocr"]["language"],
                    ocr_version=self.config["ocr"]["ocr_version"],
                    use_angle_cls=False,
                    show_log=False,
                )
            else:
                self._engine = PaddleOCR(
                    lang=self.config["ocr"]["language"],
                    use_doc_orientation_classify=self.config["ocr"]["use_doc_orientation_classify"],
                    use_doc_unwarping=self.config["ocr"]["use_doc_unwarping"],
                    use_textline_orientation=self.config["ocr"]["use_textline_orientation"],
                )
        return self._engine

    def run(self, image: np.ndarray) -> OCRResult:
        try:
            if image.ndim == 2:
                import cv2

                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            engine = self._load()
            if self.version.startswith("2.7"):
                prediction = engine.ocr(image, cls=False)
                lines: list[dict[str, Any]] = []
                for page in prediction or []:
                    for item in page or []:
                        box, recognition = item
                        text, score = recognition
                        lines.append({"text": str(text), "confidence": float(score), "box": box})
                return OCRResult(
                    text="\n".join(line["text"] for line in lines),
                    lines=lines,
                    engine_version=self.version,
                    status="ok" if self.version_matches_protocol else "protocol_version_mismatch",
                )
            prediction = engine.predict(image)
            lines: list[dict[str, Any]] = []
            for page in prediction:
                payload = page.json if hasattr(page, "json") else {}
                result = payload.get("res", payload) if isinstance(payload, dict) else {}
                texts = result.get("rec_texts", [])
                scores = result.get("rec_scores", [])
                boxes = result.get("rec_boxes", [])
                for index, text in enumerate(texts):
                    lines.append(
                        {
                            "text": str(text),
                            "confidence": float(scores[index]) if index < len(scores) else None,
                            "box": boxes[index].tolist() if index < len(boxes) and hasattr(boxes[index], "tolist") else (boxes[index] if index < len(boxes) else None),
                        }
                    )
            return OCRResult(
                text="\n".join(line["text"] for line in lines),
                lines=lines,
                engine_version=self.version,
                status="ok" if self.version_matches_protocol else "protocol_version_mismatch",
            )
        except Exception as error:
            return OCRResult("", [], self.version, "failed", f"{type(error).__name__}: {error}")

    def run_views(self, views: list[tuple[str, np.ndarray]]) -> OCRResult:
        """Run the fixed OCR configuration on faithful views and merge duplicates."""
        merged: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for view_name, image in views:
            result = self.run(image)
            if result.error:
                errors.append(f"{view_name}: {result.error}")
            for line in result.lines:
                normalized = "".join(ch.lower() for ch in str(line.get("text", "")) if ch.isalnum())
                if not normalized:
                    continue
                candidate = dict(line)
                candidate["source_view"] = view_name
                cy, cx = _line_position(candidate.get("box"))
                # Preserve the same value when it occurs in different receipt
                # columns (for example unit price and line amount), while
                # merging repeated readings of the same physical text line.
                key = f"{normalized}:{round(cx / 24)}:{round(cy / 12)}"
                old = merged.get(key)
                if old is None or float(candidate.get("confidence") or 0) > float(old.get("confidence") or 0):
                    merged[key] = candidate
        lines = sorted(merged.values(), key=lambda item: _line_position(item.get("box")))
        return OCRResult("\n".join(str(line["text"]) for line in lines), lines, self.version,
                         "ok" if lines else "failed", "; ".join(errors) if errors and not lines else None)


def _line_position(box: Any) -> tuple[float, float]:
    try:
        return (min(float(point[1]) for point in box), min(float(point[0]) for point in box))
    except (TypeError, ValueError, IndexError):
        return (float("inf"), float("inf"))
