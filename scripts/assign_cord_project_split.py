from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "FYP datasets" / "External" / "CORD-v2"
MANIFEST = ROOT / "cord_v2_manifest.csv"

def main() -> None:
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000 or len({row["source_id"] for row in rows}) != 1000:
        raise RuntimeError("CORD manifest must contain 1,000 unique source identifiers")
    source_ids = [row["source_id"] for row in rows]
    random.Random(56).shuffle(source_ids)
    assignments = {source_id: "train" if i < 700 else "validation" if i < 850 else "test" for i, source_id in enumerate(source_ids)}
    for row in rows:
        row["project_partition"] = assignments[row["source_id"]]
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance_path = ROOT / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["project_partition_counts"] = dict(Counter(assignments.values()))
    provenance["project_partition_seed"] = 56
    provenance["role_boundary"] = "External controlled development data. It is not Malaysian natural-domain evidence and does not replace the self-collected robustness set. The publisher split is retained as provenance; the Chapter 3 project partition is 70:15:15."
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(dict(Counter(assignments.values())))

if __name__ == "__main__":
    main()
