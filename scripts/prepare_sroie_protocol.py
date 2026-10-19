from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO.parent / "FYP datasets" / "External" / "SROIE-Malaysia"
SEED = 56


def main() -> None:
    records = []
    for split in ("train", "test"):
        path = DATASET / "annotations" / f"{split}.jsonl"
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    records.sort(key=lambda row: row["source_id"])
    order = list(range(len(records)))
    random.Random(SEED).shuffle(order)
    train_n = round(len(records) * 0.70)
    validation_n = round(len(records) * 0.15)
    partition = {}
    for position, index in enumerate(order):
        name = "train" if position < train_n else "validation" if position < train_n + validation_n else "test"
        partition[records[index]["source_id"]] = name
    fields = [
        "source_id", "relative_path", "publisher_split", "project_partition", "company",
        "transaction_date", "total_amount", "receipt_invoice_number", "ocr_reference_text", "sha256",
    ]
    destination = DATASET / "sroie_protocol_manifest.csv"
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            words = row.get("words") or []
            writer.writerow({
                "source_id": row["source_id"],
                "relative_path": row["image"],
                "publisher_split": row["split"],
                "project_partition": partition[row["source_id"]],
                "company": row.get("entities", {}).get("company", ""),
                "transaction_date": row.get("entities", {}).get("date", ""),
                "total_amount": row.get("entities", {}).get("total", ""),
                "receipt_invoice_number": "",
                "ocr_reference_text": "\n".join(words),
                "sha256": row.get("sha256", ""),
            })
    counts = {name: sum(value == name for value in partition.values()) for name in ("train", "validation", "test")}
    print(json.dumps({"manifest": str(destination), "sources": len(records), "partitions": counts, "seed": SEED}, indent=2))


if __name__ == "__main__":
    main()
