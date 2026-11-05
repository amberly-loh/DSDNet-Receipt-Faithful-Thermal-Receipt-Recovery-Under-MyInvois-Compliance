import json
import sqlite3
from pathlib import Path


database = Path(__file__).resolve().parents[1] / "artifacts" / "app_data" / "receipts.sqlite3"
connection = sqlite3.connect(database)
connection.row_factory = sqlite3.Row

rows = connection.execute(
    """
    SELECT id, source_name, receipt_number, transaction_date, total_amount, status
    FROM receipts
    WHERE receipt_number IN ('001624', '100311', '3603', '62271', '61569', '61036')
    ORDER BY source_name, id
    """
).fetchall()
print(json.dumps([dict(row) for row in rows], indent=2))

copy_summary = connection.execute(
    """
    SELECT COUNT(*) AS total,
           SUM(
               CASE
                   WHEN receipt_number = '61036'
                    AND transaction_date = '2024-03-19'
                    AND ROUND(total_amount, 2) = 340.00
                    AND status = 'reviewed'
                   THEN 1 ELSE 0
               END
           ) AS corrected
    FROM receipts
    WHERE source_name = 'Document_20240511_0008 - Copy.jpg'
    """
).fetchone()
print("copy_summary", dict(copy_summary))

print("columns", [row[1] for row in connection.execute("PRAGMA table_info(receipts)")])
for row in connection.execute("SELECT * FROM receipts ORDER BY id DESC LIMIT 30"):
    record = dict(row)
    if "USER-20261103-215104-7b137b" in json.dumps(record, default=str):
        print("current_detail", json.dumps(record, indent=2, default=str))
