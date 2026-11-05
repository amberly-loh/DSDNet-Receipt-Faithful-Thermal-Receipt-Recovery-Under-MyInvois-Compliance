"""Backfill semantic line items and accounting evidence for saved receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsdnet_receipt.semantic import extract_accounting_fields, extract_line_items
from dsdnet_receipt.storage import ReceiptStore, utc_now


def run(database: Path) -> dict[str, int]:
    store = ReceiptStore(database)
    counts = {"examined": 0, "updated": 0, "with_line_items": 0}
    with store.connect() as db:
        rows = db.execute("SELECT * FROM receipts WHERE deleted_at IS NULL").fetchall()
        for raw in rows:
            row = dict(raw); counts["examined"] += 1
            audit = json.loads(row.get("audit_json") or "{}")
            lines = audit.get("ocr_lines") or []
            primary = {"merchant": row.get("merchant"), "transaction_date": row.get("transaction_date"),
                       "receipt_invoice_number": row.get("receipt_number"), "total_amount": row.get("total_amount")}
            items = extract_line_items(lines)
            accounting = extract_accounting_fields(row.get("ocr_text") or "", lines, primary)
            audit["semantic_extraction"] = {"method": "layout_aware_receipt_semantics_v2",
                                              "line_items": items, "accounting_fields": accounting}
            db.execute("UPDATE receipts SET line_items_json=?,accounting_fields_json=?,audit_json=?,updated_at=? WHERE id=?",
                       (json.dumps(items), json.dumps(accounting), json.dumps(audit), utc_now(), row["id"]))
            audit_path = Path(row["restored_path"]).parent / "audit.json"
            if audit_path.parent.is_dir():
                audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
            counts["updated"] += 1
            counts["with_line_items"] += bool(items)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.database), indent=2))
