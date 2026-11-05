from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.restoration import make_conservative_creased_view
from dsdnet_receipt.config import load_config
from dsdnet_receipt.ocr import PaddleOCRAdapter


def main() -> None:
    source_root = Path(r"C:\Users\Amberly\Desktop\FYP\Faithful_media")
    output_root = ROOT / "artifacts" / "feedback_receipt_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(ROOT / "configs" / "default.json")
    ocr = PaddleOCRAdapter(config)
    comparisons: list[np.ndarray] = []
    for row in range(1, 15):
        source_path = source_root / f"image{row * 3}.png"
        source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise FileNotFoundError(source_path)
        recognition = ocr.run(source)
        recovered = make_conservative_creased_view(source, None, recognition.lines)
        print(f"row {row:02d}: OCR {recognition.status}, lines={len(recognition.lines)}")
        cv2.imwrite(str(output_root / f"row_{row:02d}_original.png"), source)
        cv2.imwrite(str(output_root / f"row_{row:02d}_recovered.png"), recovered)
        target_height = 430
        source_preview = cv2.resize(
            source,
            (max(1, int(source.shape[1] * target_height / source.shape[0])), target_height),
            interpolation=cv2.INTER_AREA,
        )
        recovered_preview = cv2.resize(
            recovered,
            (max(1, int(recovered.shape[1] * target_height / recovered.shape[0])), target_height),
            interpolation=cv2.INTER_AREA,
        )
        cell_width = max(source_preview.shape[1], recovered_preview.shape[1])
        pair = np.full((target_height + 36, cell_width * 2 + 8), 255, np.uint8)
        pair[36:, : source_preview.shape[1]] = source_preview
        pair[36:, cell_width + 8 : cell_width + 8 + recovered_preview.shape[1]] = recovered_preview
        cv2.putText(pair, f"ROW {row} ORIGINAL", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1)
        cv2.putText(pair, "RECOVERED", (cell_width + 13, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, 0, 1)
        comparisons.append(pair)
    for page, start in enumerate((0, 7), start=1):
        group = comparisons[start : start + 7]
        sheet_width = max(item.shape[1] for item in group)
        sheet = np.full((sum(item.shape[0] for item in group) + 6 * (len(group) - 1), sheet_width), 238, np.uint8)
        top = 0
        for item in group:
            sheet[top : top + item.shape[0], : item.shape[1]] = item
            top += item.shape[0] + 6
        cv2.imwrite(str(output_root / f"contact_sheet_{page}.png"), sheet)
    print(output_root)


if __name__ == "__main__":
    main()
