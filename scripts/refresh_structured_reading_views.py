from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import hashlib
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config
from dsdnet_receipt.images import write_image
from dsdnet_receipt.restoration import (
    make_faithful_content_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh stored Faithful views only")
    parser.add_argument("--run", help="Refresh one saved run by directory name")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config, _ = load_config(ROOT / "configs" / "default.json")
    refreshed, skipped = 0, 0
    runs_root = Path(config["artifacts_root"]) / "user_runs"
    runs = [runs_root / args.run] if args.run else list(runs_root.iterdir())
    database = ROOT / "artifacts" / "app_data" / "receipts.sqlite3"
    connection = sqlite3.connect(database)
    saved_runs = {row[0] for row in connection.execute("SELECT public_id FROM receipts")}
    runs = sorted((run for run in runs if run.name in saved_runs), reverse=True)
    if args.limit:
        runs = runs[:args.limit]
    for run in runs:
        source_path = run / "rectified_input.png"
        if not source_path.is_file():
            source_path = run / "input.png"
        recovery_path = run / "ocr_input.png"
        audit_path, target_path = run / "audit.json", run / "restored.png"
        if not source_path.is_file() or not recovery_path.is_file() or not audit_path.is_file():
            skipped += 1; continue
        source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        recovery = cv2.imread(str(recovery_path), cv2.IMREAD_GRAYSCALE)
        if source is None or recovery is None:
            skipped += 1; continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        backup_root = run / "before_measured_content_refresh"
        backup_root.mkdir(exist_ok=True)
        for name in ("faithful_content.png", "restored.png", "audit.json"):
            original = run / name
            if original.is_file() and not (backup_root / name).exists():
                shutil.copy2(original, backup_root / name)
        backup = run / "restored_before_structured_cleanup.png"
        if target_path.is_file() and not backup.exists(): shutil.copy2(target_path, backup)
        lines = audit.get("ocr_lines", [])
        mask_config = config.get("artifact_mask", {})
        faithful = make_faithful_content_view(
            source,
            recovery,
            lines,
            mask_config.get("checkpoint") if mask_config.get("enabled") else None,
            float(mask_config.get("threshold", 0.95)),
        )
        write_image(run / "faithful_content.png", faithful)
        audit["faithful_display_policy"] = (
            "background-normalised image-derived content with conservative "
            "source-supported character-gap repair; no OCR glyph rendering"
        )
        audit["stored_file_sha256"] = {
            **audit.get("stored_file_sha256", {}),
            "faithful_content": hashlib.sha256((run / "faithful_content.png").read_bytes()).hexdigest(),
        }
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        connection.execute("UPDATE receipts SET audit_json=? WHERE public_id=?", (json.dumps(audit), run.name))
        connection.commit()
        refreshed += 1
        print(json.dumps({"refreshed": refreshed, "run": run.name}), flush=True)
    connection.close()
    print({"refreshed": refreshed, "skipped": skipped})


if __name__ == "__main__":
    main()
