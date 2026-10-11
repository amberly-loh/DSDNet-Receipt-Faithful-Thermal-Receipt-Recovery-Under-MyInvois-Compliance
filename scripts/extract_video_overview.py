from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

TASK = Path(__file__).resolve().parents[2]
VIDEO = TASK / "DSDNet-Receipt Thermal Receipt Recovery and Evidence-Awareness.mp4"
OUTPUT = TASK / "video_overview_frames"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {VIDEO}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = count / fps
    timestamps = np.linspace(0, max(duration - 0.2, 0), 24)
    thumbs = []
    records = []
    for index, seconds in enumerate(timestamps, start=1):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(seconds * 1000))
        ok, frame = capture.read()
        if not ok:
            continue
        frame_path = OUTPUT / f"frame_{index:02d}_{seconds:07.2f}s.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        thumb = cv2.resize(frame, (320, round(320 * height / width)), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{seconds:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
        records.append({"index": index, "seconds": round(float(seconds), 3), "path": str(frame_path)})
    capture.release()
    columns = 4
    rows = int(np.ceil(len(thumbs) / columns))
    cell_h = max(image.shape[0] for image in thumbs)
    sheet = np.full((rows * cell_h, columns * 320, 3), 255, dtype=np.uint8)
    for index, image in enumerate(thumbs):
        row, column = divmod(index, columns)
        sheet[row * cell_h : row * cell_h + image.shape[0], column * 320 : (column + 1) * 320] = image
    cv2.imwrite(str(OUTPUT / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    metadata = {"video": str(VIDEO), "fps": fps, "frames": count, "width": width, "height": height, "duration_seconds": duration, "samples": records}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("fps", "frames", "width", "height", "duration_seconds")}, indent=2))


if __name__ == "__main__":
    main()
