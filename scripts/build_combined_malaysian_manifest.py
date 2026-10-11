from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "receipt_recovery/artifacts/readiness/dataset_manifest.csv"
SROIE = ROOT / "FYP datasets/External/SROIE-Malaysia/sroie_malaysia_manifest.csv"
OUTPUT = ROOT / "FYP datasets/combined_malaysian_receipts_manifest.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    local, sroie = read(LOCAL), read(SROIE)
    hashes = Counter(row["sha256"] for row in local + sroie)
    rows = []
    for row in local:
        rows.append({
            "source_id": f"LOCAL-{row['source_id']}",
            "dataset": "provided Malaysian real-life receipts",
            "relative_path": f"Real life/{row['relative_path']}",
            "original_split": row["assigned_partition"],
            "country_context": "Malaysia",
            "sha256": row["sha256"],
            "exact_duplicate_count": hashes[row["sha256"]],
            "license_or_permission": row["consent_usage_permission"],
            "annotation_status": row["annotation_status"],
        })
    for row in sroie:
        rows.append({
            "source_id": row["source_id"],
            "dataset": "ICDAR2019 SROIE Malaysian receipts",
            "relative_path": f"External/SROIE-Malaysia/{row['relative_path']}",
            "original_split": row["split"],
            "country_context": "Malaysia",
            "sha256": row["sha256"],
            "exact_duplicate_count": hashes[row["sha256"]],
            "license_or_permission": row["license"],
            "annotation_status": "OCR boxes, words and key entities available",
        })
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Combined rows: {len(rows)}")
    print(f"Exact unique images: {len(hashes)}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
