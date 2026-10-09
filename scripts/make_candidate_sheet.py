from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


project = Path(__file__).resolve().parents[1]
task_root = project.parent
records = json.loads(
    (project / "artifacts/pilot_degradation/candidate_clean_sources.json").read_text(encoding="utf-8")
)
tiles = []
for index, record in enumerate(records, start=1):
    path = task_root / "FYP datasets/Real life" / record["relative_path"]
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    scale = min(320 / width, 360 / height)
    image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    canvas = np.full((420, 360, 3), 255, np.uint8)
    top = (360 - image.shape[0]) // 2
    left = (360 - image.shape[1]) // 2
    canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
    cv2.putText(canvas, f"{index}. {record['source_id']}", (8, 382), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, record["file_name"][:38], (8, 407), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    tiles.append(canvas)

rows = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
sheet = np.vstack(rows)
output = project / "artifacts/pilot_degradation/candidate_contact_sheet.jpg"
cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(output))
print(output)
