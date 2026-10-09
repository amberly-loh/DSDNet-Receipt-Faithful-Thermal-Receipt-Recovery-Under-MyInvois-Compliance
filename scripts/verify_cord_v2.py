from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "FYP datasets" / "External" / "CORD-v2"

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main() -> None:
    with (ROOT / "cord_v2_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    errors: list[str] = []
    for row in rows:
        image_path = ROOT / row["image_path"]
        annotation_path = ROOT / row["annotation_path"]
        if not image_path.is_file() or not annotation_path.is_file():
            errors.append(f"missing files: {row['source_id']}")
            continue
        if digest(image_path) != row["sha256"]:
            errors.append(f"hash mismatch: {row['source_id']}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid payload: {row['source_id']}: {exc}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "rows": len(rows),
        "unique_sources": len({row["source_id"] for row in rows}),
        "official_split_counts": dict(Counter(row["official_split"] for row in rows)),
        "project_partition_counts": dict(Counter(row["project_partition"] for row in rows)),
        "hashes_checked": len(rows),
        "errors": errors,
    }
    target = Path(__file__).resolve().parents[1] / "artifacts" / "readiness" / "cord_v2_verification.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
