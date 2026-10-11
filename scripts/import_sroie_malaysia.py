from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "FYP datasets" / "External" / "SROIE-Malaysia"
PARQUET_DIR = DATASET / "data"
IMAGE_DIR = DATASET / "images"
ANNOTATION_DIR = DATASET / "annotations"
MANIFEST = DATASET / "sroie_malaysia_manifest.csv"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for parquet_path in sorted(PARQUET_DIR.glob("*.parquet")):
        split = parquet_path.name.split("-", 1)[0]
        split_image_dir = IMAGE_DIR / split
        split_image_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = ANNOTATION_DIR / f"{split}.jsonl"
        table = pq.read_table(parquet_path)

        with annotation_path.open("w", encoding="utf-8") as annotation_file:
            for row_index, row in enumerate(table.to_pylist()):
                key = str(row["key"])
                original_path = str(row["image"].get("path") or "")
                suffix = Path(original_path).suffix.lower() or ".jpg"
                image_name = f"{key}{suffix}"
                image_path = split_image_dir / image_name
                image_bytes = row["image"]["bytes"]
                image_path.write_bytes(image_bytes)
                digest = hashlib.sha256(image_bytes).hexdigest()

                record = {
                    "source_id": f"SROIE-MY-{split.upper()}-{row_index + 1:04d}",
                    "key": key,
                    "split": split,
                    "image": image_path.relative_to(DATASET).as_posix(),
                    "image_size": row["image_size"],
                    "entities": row["entities"],
                    "words": row["words"],
                    "bboxes": row["bboxes"],
                    "sha256": digest,
                }
                annotation_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_rows.append(
                    {
                        "source_id": record["source_id"],
                        "key": key,
                        "split": split,
                        "relative_path": record["image"],
                        "country_context": "Malaysia",
                        "source_type": "scanned_receipt",
                        "company": row["entities"]["company"],
                        "date": row["entities"]["date"],
                        "total": row["entities"]["total"],
                        "width": row["image_size"]["width"],
                        "height": row["image_size"]["height"],
                        "sha256": digest,
                        "license": "CC BY 4.0",
                    }
                )

    fieldnames = list(manifest_rows[0])
    with MANIFEST.open("w", encoding="utf-8-sig", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Extracted {len(manifest_rows)} Malaysian SROIE receipts")
    print(IMAGE_DIR)
    print(MANIFEST)


if __name__ == "__main__":
    main()
