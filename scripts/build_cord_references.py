from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "FYP datasets" / "External" / "CORD-v2"

def scalar_values(value: object) -> list[str]:
    if isinstance(value, dict):
        values: list[str] = []
        for child in value.values():
            values.extend(scalar_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(scalar_values(child))
        return values
    return [str(value)] if value is not None else []

def main() -> None:
    with (ROOT / "cord_v2_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        manifest = list(csv.DictReader(stream))
    records = []
    category_counts: Counter[str] = Counter()
    for row in manifest:
        annotation = json.loads((ROOT / row["annotation_path"]).read_text(encoding="utf-8"))
        valid_lines = annotation.get("valid_line", [])
        lines = []
        for line in valid_lines:
            category_counts[str(line.get("category", "unlabelled"))] += 1
            text = " ".join(str(word.get("text", "")).strip() for word in line.get("words", [])).strip()
            if text:
                lines.append(text)
        total = annotation.get("gt_parse", {}).get("total", {}).get("total_price", "")
        if isinstance(total, list):
            total = " | ".join(scalar_values(total))
        records.append({
            "source_id": row["source_id"],
            "project_partition": row["project_partition"],
            "ocr_reference_text": "\n".join(lines),
            "transaction_date": "",
            "receipt_invoice_number": "",
            "total_amount": str(total),
            "transaction_date_status": "manual verification required",
            "receipt_invoice_number_status": "manual verification required",
            "total_amount_status": "mapped from gt_parse.total.total_price" if total else "ineligible: absent",
        })
    output = ROOT / "evaluation_references.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        "sources": len(records),
        "full_text_references": sum(bool(row["ocr_reference_text"]) for row in records),
        "total_amount_mapped": sum(bool(row["total_amount"]) for row in records),
        "transaction_date_automatically_mapped": 0,
        "receipt_invoice_number_automatically_mapped": 0,
        "manual_review_required": ["transaction_date", "receipt_invoice_number"],
        "valid_line_category_counts": dict(category_counts.most_common()),
        "interpretation": "Full-text OCR and total amount can be evaluated after validation. Date and receipt/invoice number require manual source-grounded annotation; absent fields are ineligible, not OCR failures.",
    }
    target = Path(__file__).resolve().parents[1] / "artifacts" / "readiness" / "cord_annotation_coverage.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("sources", "full_text_references", "total_amount_mapped", "manual_review_required")}, indent=2))

if __name__ == "__main__":
    main()
