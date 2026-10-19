from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config
from dsdnet_receipt.ocr import PaddleOCRAdapter


def main() -> None:
    config, _ = load_config(ROOT / "configs" / "default.json")
    candidates = sorted((Path(config["artifacts_root"]) / "user_runs").glob("*/input.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates: raise SystemExit("No saved natural receipt is available for the OCR smoke test.")
    image = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    result = PaddleOCRAdapter(config).run(image)
    report = {
        "input": str(candidates[0]), "status": result.status, "engine_version": result.engine_version,
        "required_version": config["ocr"]["required_version"], "lines": len(result.lines), "error": result.error,
        "pass": result.status == "ok" and not result.error and len(result.lines) > 0,
    }
    output = Path(config["artifacts_root"]) / "system_verification" / "ocr_smoke_test.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
