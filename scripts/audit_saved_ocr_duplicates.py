from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config
from dsdnet_receipt.ocr import deduplicate_lines


def main() -> None:
    config, _ = load_config(ROOT / "configs" / "default.json")
    records = []
    for path in sorted((Path(config["artifacts_root"]) / "user_runs").glob("*/audit.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        original = audit.get("ocr_lines", [])
        cleaned = deduplicate_lines(original)
        records.append({
            "run_id": audit.get("run_id") or path.parent.name,
            "original_lines": len(original),
            "unique_physical_lines": len(cleaned),
            "duplicates_removed": len(original) - len(cleaned),
        })
    report = {
        "scope": "saved audit OCR lines; same normalized text is merged only when spatially co-located",
        "runs_checked": len(records),
        "original_lines": sum(row["original_lines"] for row in records),
        "unique_physical_lines": sum(row["unique_physical_lines"] for row in records),
        "duplicates_removed": sum(row["duplicates_removed"] for row in records),
        "records": records,
    }
    output = Path(config["artifacts_root"]) / "system_verification" / "saved_ocr_duplicate_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
