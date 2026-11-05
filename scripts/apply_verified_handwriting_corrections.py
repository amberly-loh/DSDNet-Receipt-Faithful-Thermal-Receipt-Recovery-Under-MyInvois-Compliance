"""Apply user-verified handwritten receipt fields with an auditable backup.

This script does not infer labels.  It applies only the values recorded in the
user-verified annotation CSV and preserves the previous values in audit_json.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "artifacts" / "app_data" / "receipts.sqlite3"
ANNOTATIONS = ROOT / "artifacts" / "handwriting_datasets" / "user_verified_fields_20261103.csv"


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = DATABASE.with_name(f"receipts.before_handwriting_corrections.{timestamp}.sqlite3")
    shutil.copy2(DATABASE, backup)

    with ANNOTATIONS.open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))

    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    changed: list[dict[str, str]] = []
    try:
        for label in labels:
            rows = connection.execute(
                "SELECT public_id, source_name, transaction_date, receipt_number, total_amount, audit_json "
                "FROM receipts WHERE source_name = ?",
                (label["source_name"],),
            ).fetchall()
            for row in rows:
                before = {
                    "transaction_date": row["transaction_date"],
                    "receipt_number": row["receipt_number"],
                    "total_amount": row["total_amount"],
                }
                after = {
                    "transaction_date": label["transaction_date"],
                    "receipt_number": label["receipt_number"],
                    "total_amount": label["total_amount"],
                }
                audit = json.loads(row["audit_json"] or "{}")
                already_recorded = any(
                    item.get("source") == ANNOTATIONS.name and item.get("after") == after
                    for item in audit.get("verified_field_corrections", [])
                )
                if already_recorded and before == after:
                    continue
                audit.setdefault("verified_field_corrections", []).append(
                    {
                        "verified_at": label["verified_at"],
                        "verification_status": label["verification_status"],
                        "source": ANNOTATIONS.name,
                        "before": before,
                        "after": after,
                    }
                )
                connection.execute(
                    "UPDATE receipts SET transaction_date = ?, receipt_number = ?, total_amount = ?, "
                    "status = 'reviewed', updated_at = ?, audit_json = ? WHERE public_id = ?",
                    (
                        after["transaction_date"],
                        after["receipt_number"],
                        after["total_amount"],
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps(audit, ensure_ascii=False),
                        row["public_id"],
                    ),
                )
                changed.append({"public_id": row["public_id"], "source_name": row["source_name"], **after})
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(json.dumps({"backup": str(backup), "updated": changed}, indent=2))


if __name__ == "__main__":
    main()
