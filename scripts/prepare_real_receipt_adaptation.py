from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.restoration import assess_receipt_degradation  # noqa: E402


def pixel_digest(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable receipt: {path}")
    canonical = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deduplicated real-receipt adaptation partitions")
    parser.add_argument("--database", type=Path, default=REPO / "artifacts/app_data/receipts.sqlite3")
    parser.add_argument("--output", type=Path, default=REPO / "artifacts/real_receipt_adaptation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute("SELECT public_id, source_name, input_path, created_at FROM receipts ORDER BY created_at").fetchall()
    unique: dict[str, dict] = {}
    rejected = []
    for row in rows:
        path = Path(row["input_path"])
        try:
            digest = pixel_digest(path)
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            profile = assess_receipt_degradation(image)
        except Exception as error:
            rejected.append({"public_id": row["public_id"], "reason": str(error)})
            continue
        record = unique.setdefault(digest, {
            "source_id": row["public_id"],
            "source_name": row["source_name"],
            "clean_reference": str(path.resolve()),
            "degraded_path": str(path.resolve()),
            "variant_id": "REAL-AUG",
            "parameters": {"blur_sigma": 0.6},
            "pixel_sha256": digest,
            "degradation_profile": profile["profile"],
            "degradation_flags": profile["flags"],
            "duplicate_run_ids": [],
        })
        record["duplicate_run_ids"].append(row["public_id"])

    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in unique.values():
        buckets[record["degradation_profile"]].append(record)
    # Stable, profile-balanced allocation. Every unique source appears exactly
    # once; validation and test are never read by the training loop.
    ordered = []
    for profile in sorted(buckets):
        bucket = sorted(buckets[profile], key=lambda item: item["pixel_sha256"])
        for index, record in enumerate(bucket):
            slot = index % 10
            record["project_partition"] = "train" if slot < 7 else ("validation" if slot < 9 else "test")
            ordered.append(record)

    manifest = args.output / "generation_manifest_real_receipts.jsonl"
    manifest.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered), encoding="utf-8")
    summary = {
        "saved_runs": len(rows),
        "unique_pixel_sources": len(ordered),
        "duplicates_removed": len(rows) - len(ordered),
        "partition_counts": {name: sum(r["project_partition"] == name for r in ordered) for name in ("train", "validation", "test")},
        "profile_counts": {name: len(bucket) for name, bucket in sorted(buckets.items())},
        "rejected": rejected,
        "manifest": str(manifest.resolve()),
        "training_role": "real-scan domain adaptation target with additional synthetic degradation as input",
        "clean_reference_limitation": "The target is the observed scan, not an unknown pristine receipt. Formal restoration claims require separate clean-reference evaluation.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
