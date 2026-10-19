from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATH = REPO / "artifacts" / "controlled_degradation_sroie" / "generation_manifest.jsonl"


def main() -> None:
    unique = {}
    for line in PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            unique[(row["source_id"], row["severity"])] = row
    rows = sorted(unique.values(), key=lambda row: (row["source_id"], row["variant_id"]))
    PATH.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"unique_variants": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
