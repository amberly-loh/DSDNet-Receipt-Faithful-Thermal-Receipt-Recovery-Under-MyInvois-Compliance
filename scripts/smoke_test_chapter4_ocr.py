from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402
from dsdnet_receipt.ocr import PaddleOCRAdapter  # noqa: E402

DATASET = REPO.parent / "FYP datasets" / "External" / "SROIE-Malaysia"

def main() -> None:
    config, _ = load_config(REPO / "configs" / "default.json")
    sample = sorted((DATASET / "images" / "test").glob("*"))[0]
    image = read_gray(sample)
    result = PaddleOCRAdapter(config).run(image)
    report = {
        "status": "PASS" if result.status == "ok" and result.lines else "FAIL",
        "sample": str(sample),
        "python": sys.version,
        "opencv": cv2.__version__,
        "paddlepaddle": importlib.metadata.version("paddlepaddle"),
        "paddleocr": result.engine_version,
        "ocr_protocol_status": result.status,
        "recognized_lines": len(result.lines),
        "recognized_text_preview": result.text[:500],
        "error": result.error,
        "known_dependency_override": "PaddleOCR 2.7.0.3 metadata requests OpenCV<=4.6.0.66; Chapter 4 records OpenCV 4.8.1.78.",
    }
    target = REPO / "artifacts" / "readiness" / "chapter4_ocr_smoke_test.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
