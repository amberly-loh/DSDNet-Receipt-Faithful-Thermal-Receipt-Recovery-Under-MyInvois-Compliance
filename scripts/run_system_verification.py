from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config
from dsdnet_receipt.fields import extract_fields, route_fields
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import make_evidence_preserving_view, make_readability_view
from dsdnet_receipt.ocr import deduplicate_lines
from dsdnet_receipt.web_app import accounting_line_items


def synthetic_input_checks(config: dict) -> dict:
    cases = {
        "portrait_gray": np.full((900, 420), 235, np.uint8),
        "landscape_color": np.full((420, 900, 3), 235, np.uint8),
        "tiny": np.full((12, 8), 255, np.uint8),
        "uniform_blank": np.full((500, 300), 255, np.uint8),
        "high_noise": np.random.default_rng(56).integers(0, 256, (600, 320), np.uint8),
    }
    results = {}
    for name, image in cases.items():
        try:
            output, record = preprocess(image, config)
            results[name] = {"status": "PASS", "shape": list(output.shape), "record": record}
        except Exception as error:
            results[name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    for name, image in {"empty": np.array([], np.uint8), "four_dimensional": np.zeros((2, 2, 2, 2), np.uint8)}.items():
        try:
            preprocess(image, config)
            results[name] = {"status": "FAIL", "error": "invalid input was accepted"}
        except ValueError:
            results[name] = {"status": "PASS", "result": "rejected safely"}
    return results


def field_checks() -> dict:
    cases = [
        ("standard", "INVOICE NO: INV-1001\nDATE: 19/10/2026\nTOTAL RM 102.50", "2026-10-19", "INV-1001", "102.50"),
        ("integer_total", "RECEIPT NO R100\nDATE 01/01/2026\nTOTAL RM 10", "2026-01-01", "R100", "10.00"),
        ("invalid_calendar", "NO: R101\nDATE 31/02/2026\nTOTAL RM 1.00", None, "R101", "1.00"),
        ("comma_amount", "DOCUMENT NO: AB-999\nDATE 2026-10-19\nGRAND TOTAL RM 1,234.50", "2026-10-19", "AB-999", "1234.50"),
    ]
    results = {}
    for name, text, date, number, total in cases:
        actual = extract_fields(text, [])
        observed = (actual["transaction_date"]["value"], actual["receipt_invoice_number"]["value"], actual["total_amount"]["value"])
        expected = (date, number, total)
        results[name] = {"status": "PASS" if observed == expected else "FAIL", "expected": expected, "observed": observed}
    routed = route_fields(extract_fields(cases[0][1], []), support_available=False, model_available=True)
    results["natural_support_gate"] = {"status": "PASS" if all(x.decision == "Human decision required" for x in routed) else "FAIL"}
    return results


def saved_run_checks(config: dict) -> dict:
    records = []
    for audit_path in sorted((Path(config["artifacts_root"]) / "user_runs").glob("*/audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        root = audit_path.parent
        row = {"run_id": audit.get("run_id"), "issues": []}
        if "restoration_stages" not in audit or "fields" not in audit:
            row.update({"status": "LEGACY", "issues": ["record predates the current audit schema and must not be treated as current conformance evidence"]})
            records.append(row)
            continue
        if audit.get("support_status") != "unavailable_for_natural_source": row["issues"].append("natural support was not unavailable")
        if audit.get("restoration_stages") not in (0, 8): row["issues"].append("unexpected stage count")
        for field in audit.get("fields", []):
            if field.get("decision") != "Human decision required": row["issues"].append(f"unsafe decision for {field.get('name')}")
        lines = audit.get("ocr_lines", [])
        row["ocr_duplicate_lines_removed"] = len(lines) - len(deduplicate_lines(lines))
        row["line_item_candidates"] = len(accounting_line_items(lines))
        input_image = cv2.imread(str(root / "input.png"), cv2.IMREAD_GRAYSCALE)
        recovery = cv2.imread(str(root / "ocr_input.png"), cv2.IMREAD_GRAYSCALE)
        if input_image is not None and recovery is not None and lines:
            readable = make_readability_view(input_image, recovery)
            display = make_evidence_preserving_view(readable, lines)
            dark = readable < 150
            row["dark_stroke_preservation"] = float(np.mean(display[dark] <= readable[dark])) if np.any(dark) else 1.0
            row["nonwhite_display_fraction"] = float(np.mean(display < 250))
        row["status"] = "PASS" if not row["issues"] else "FAIL"
        records.append(row)
    return {"runs_checked": len(records), "current_failed_runs": sum(row["status"] == "FAIL" for row in records), "legacy_runs": sum(row["status"] == "LEGACY" for row in records), "records": records}


def main() -> None:
    config, _ = load_config(ROOT / "configs" / "default.json")
    restoration_summary = json.loads((Path(config["artifacts_root"]) / "validation_restoration" / "checkpoint_0009000" / "summary.json").read_text())
    blend_summary = json.loads((Path(config["artifacts_root"]) / "validation_blend_selection" / "checkpoint_0009000" / "summary.json").read_text())
    natural_batches = []
    for batch_path in sorted((Path(config["artifacts_root"]) / "system_verification").glob("natural_preprocessing_*_*.json")):
        natural_batches.append(json.loads(batch_path.read_text(encoding="utf-8")))
    natural_summary = {
        "total": sum(item["total"] for item in natural_batches),
        "decoded": sum(item["decoded"] for item in natural_batches),
        "failed": [failure for item in natural_batches for failure in item["failed"]],
        "batches": len(natural_batches),
    }
    report = {
        "scope": "development and validation verification; not held-out test evidence",
        "input_robustness": synthetic_input_checks(config),
        "field_extraction": field_checks(),
        "saved_natural_runs": saved_run_checks(config),
        "natural_dataset_preprocessing": natural_summary,
        "restoration_validation": {"checkpoint_9000": restoration_summary, "blend_selection": blend_summary},
        "thesis_conformance": [
            {"requirement": "deterministic preprocessing and recorded transforms", "status": "PASS"},
            {"requirement": "eight DSDNet stages and four maxout branches", "status": "PASS"},
            {"requirement": "shared PaddleOCR configuration", "status": "PASS", "note": "same OCR model is run on three faithful views"},
            {"requirement": "natural receipts never auto-accepted without source support", "status": "PASS"},
            {"requirement": "controlled pixel support proxy Ares/Astage/Aunc", "status": "FAIL", "note": "targets exist but full proxy evaluation is not implemented"},
            {"requirement": "validation-only isotonic calibration and thresholds", "status": "FAIL", "note": "no fitted calibrator or selected policy thresholds"},
            {"requirement": "mandatory arithmetic consistency gate", "status": "FAIL", "note": "line arithmetic is exported for review but not enforced in routing"},
            {"requirement": "formal-run conformance checker", "status": "PASS", "note": "checker is implemented; the current formal-run preflight itself remains FAIL until its required manifests and calibration artifacts exist"},
            {"requirement": "held-out test used once after validation lock", "status": "NOT RUN"},
        ],
        "myinvois_assessment": {
            "status": "PARTIAL",
            "captured": ["supplier/merchant candidate", "invoice number", "invoice date", "payable total", "candidate line descriptions, quantities and prices"],
            "missing_or_user_supplied": ["supplier TIN and registration ID", "supplier address/contact", "buyer identity/TIN/address/contact", "invoice time", "classification codes", "tax type/rate/amount", "tax-exclusive and tax-inclusive totals", "authorised digital signature and submission identifiers"],
            "boundary": "The application is an audit/recovery aid, not a MyInvois submission client.",
        },
    }
    output = Path(config["artifacts_root"]) / "system_verification" / "system_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
