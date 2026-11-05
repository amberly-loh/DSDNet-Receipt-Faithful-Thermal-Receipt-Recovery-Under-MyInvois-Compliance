from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.restoration import (
    _thin_document_structure_mask,
    detect_source_text_mask,
    select_faithful_display_candidate,
)

DATABASE = ROOT / "artifacts" / "app_data" / "receipts.sqlite3"
OUTPUT = ROOT / "artifacts" / "saved_faithful_audit"


def preview(image: np.ndarray, height: int = 520) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT id, public_id, source_name, input_path, restored_path, audit_json "
        "FROM receipts ORDER BY id DESC"
    ).fetchall()
    selected = []
    seen = set()
    for row in rows:
        if row["source_name"] in seen:
            continue
        run_root = Path(row["restored_path"]).parent
        faithful = run_root / "faithful_content.png"
        original = Path(row["input_path"])
        if faithful.is_file() and original.is_file():
            selected.append((row, original, faithful))
            seen.add(row["source_name"])
        if len(selected) >= 24:
            break

    records = []
    panels = []
    for index, (row, original_path, faithful_path) in enumerate(selected, start=1):
        original = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        faithful = cv2.imread(str(faithful_path), cv2.IMREAD_GRAYSCALE)
        if original is None or faithful is None:
            continue
        run_root = faithful_path.parent
        rectified = cv2.imread(str(run_root / "rectified_input.png"), cv2.IMREAD_GRAYSCALE)
        if rectified is None:
            rectified = cv2.resize(original, (faithful.shape[1], faithful.shape[0]))
        audit = json.loads(row["audit_json"] or "{}")
        projected = select_faithful_display_candidate(
            rectified, faithful, audit.get("ocr_lines", [])
        )
        selector_source = rectified
        if selector_source.shape != faithful.shape:
            selector_source = cv2.resize(
                selector_source, (faithful.shape[1], faithful.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        source_text = detect_source_text_mask(selector_source, protection="wide")
        source_structure = _thin_document_structure_mask(selector_source)
        candidate_background = cv2.GaussianBlur(faithful, (0, 0), 5.0)
        candidate_ink = (
            candidate_background.astype(np.int16) - faithful.astype(np.int16)
        ) >= 1.5
        source_background = cv2.GaussianBlur(selector_source, (0, 0), 7.0)
        source_dark = (
            source_background.astype(np.int16) - selector_source.astype(np.int16)
        ) >= 2
        support_selector = cv2.dilate(
            source_dark.astype(np.uint8), np.ones((3, 3), np.uint8)
        ) > 0
        selector_metrics = {
            "text_pixels": int(np.count_nonzero(source_text)),
            "text_retention": float(np.count_nonzero(candidate_ink & source_text) / max(np.count_nonzero(source_text), 1)),
            "structure_pixels": int(np.count_nonzero(source_structure)),
            "structure_retention": float(np.count_nonzero(candidate_ink & source_structure) / max(np.count_nonzero(source_structure), 1)),
            "unsupported_ink": float(np.mean(candidate_ink & ~support_selector)),
            "changed": bool(projected.shape != faithful.shape or np.any(projected != faithful)),
        }
        cv2.imwrite(str(OUTPUT / f"{index:02d}_projected.png"), projected)
        original_metric = cv2.resize(original, (projected.shape[1], projected.shape[0]))
        faithful = projected
        original_bg = cv2.GaussianBlur(original_metric, (0, 0), 9)
        faithful_bg = cv2.GaussianBlur(faithful, (0, 0), 9)
        original_dark = np.maximum(original_bg.astype(np.int16) - original_metric.astype(np.int16), 0)
        faithful_dark = np.maximum(faithful_bg.astype(np.int16) - faithful.astype(np.int16), 0)
        source_ink = original_dark >= 4
        recovered_ink = faithful_dark >= 4
        support = cv2.dilate(source_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        unsupported = float(np.mean(recovered_ink & ~support))
        retention = float(np.count_nonzero(recovered_ink & source_ink) / max(np.count_nonzero(source_ink), 1))
        darkness_values = faithful_dark[recovered_ink]
        darkness_median = float(np.median(darkness_values)) if darkness_values.size else 0.0
        darkness_cv = float(np.std(darkness_values) / max(np.mean(darkness_values), 1.0)) if darkness_values.size else 0.0
        sharpness = float(cv2.Laplacian(faithful, cv2.CV_32F).var())
        records.append({
            "index": index,
            "id": row["id"],
            "run_id": row["public_id"],
            "source_name": row["source_name"],
            "source_ink_retention": retention,
            "unsupported_dark_fraction": unsupported,
            "median_local_darkness": darkness_median,
            "darkness_cv": darkness_cv,
            "laplacian_variance": sharpness,
            "ocr_lines": len(audit.get("ocr_lines", [])),
            "preprocessing": audit.get("preprocessing", {}),
            "selector": selector_metrics,
        })

        left, current, right = preview(original), preview(cv2.imread(str(faithful_path), cv2.IMREAD_GRAYSCALE)), preview(projected)
        cell_width = max(left.shape[1], current.shape[1], right.shape[1])
        panel = np.full((570, cell_width * 3 + 24, 3), 255, np.uint8)
        panel[50:570, : left.shape[1]] = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        panel[50:570, cell_width + 12 : cell_width + 12 + current.shape[1]] = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
        panel[50:570, cell_width * 2 + 24 : cell_width * 2 + 24 + right.shape[1]] = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        title = f"{index:02d}  ID {row['id']}  {row['source_name']}"
        cv2.putText(panel, title[:80], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "ORIGINAL", (6, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        cv2.putText(panel, "CURRENT", (cell_width + 18, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        cv2.putText(panel, "PROJECTED", (cell_width * 2 + 30, 43), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        panels.append(panel)

    (OUTPUT / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    for page, start in enumerate(range(0, len(panels), 6), start=1):
        group = panels[start : start + 6]
        width = max(panel.shape[1] for panel in group)
        sheet = np.full((sum(panel.shape[0] for panel in group) + 8 * (len(group) - 1), width, 3), 232, np.uint8)
        top = 0
        for panel in group:
            sheet[top : top + panel.shape[0], : panel.shape[1]] = panel
            top += panel.shape[0] + 8
        cv2.imwrite(str(OUTPUT / f"sheet_{page}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps({"receipts": len(records), "sheets": (len(panels) + 5) // 6, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
