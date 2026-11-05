"""Backfill safe reviewed-field memory and conservative tags.

This does not guess missing accounting values. A date is copied only from a
reviewed receipt belonging to the same user when either the source SHA-256 is
identical or merchant + receipt number + total match exactly and all reviewed
matches agree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.fields import infer_receipt_tags
from dsdnet_receipt.storage import ReceiptStore


STORE = ReceiptStore(ROOT / "artifacts" / "app_data" / "receipts.sqlite3")


def main() -> None:
    with STORE.connect() as db:
        rows = [dict(row) for row in db.execute(
            "SELECT * FROM receipts WHERE deleted_at IS NULL ORDER BY id"
        )]
    changed_dates = 0
    changed_tags = 0
    for row in rows:
        changes: dict[str, object] = {}
        memory = STORE.verified_field_suggestions(
            row["user_id"],
            content_sha256=row.get("content_sha256"),
            merchant=row.get("merchant"),
            receipt_number=row.get("receipt_number"),
            total_amount=row.get("total_amount"),
        )
        if not row.get("transaction_date") and memory.get("transaction_date"):
            changes["transaction_date"] = memory["transaction_date"]
            changed_dates += 1
        existing_tags = json.loads(row.get("tags") or "[]")
        inferred_tags = infer_receipt_tags(row.get("ocr_text") or "", row.get("merchant"))
        if not existing_tags and inferred_tags:
            changes["tags"] = inferred_tags
            changed_tags += 1
        if not changes:
            continue
        STORE.update_receipt(row["user_id"], row["public_id"], changes)
        try:
            audit = json.loads(row.get("audit_json") or "{}")
        except json.JSONDecodeError:
            audit = {}
        audit["verified_field_memory"] = memory
        if inferred_tags:
            audit["inferred_tags"] = inferred_tags
        STORE.update_receipt_audit(row["user_id"], row["public_id"], audit)
        STORE.log_event(
            row["user_id"], "receipt.safe_backfill", "receipt", row["public_id"],
            details={"fields": sorted(changes), "verified_memory_method": memory.get("method")},
        )
    print(json.dumps({
        "records_scanned": len(rows),
        "dates_backfilled": changed_dates,
        "tag_sets_backfilled": changed_tags,
    }, indent=2))


if __name__ == "__main__":
    main()
