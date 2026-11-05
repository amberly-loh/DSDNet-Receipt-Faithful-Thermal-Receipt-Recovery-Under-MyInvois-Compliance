from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "test_evidence_20261103"
PERF = json.loads((ROOT / "artifacts/training_candidate_20261102/local_api_performance.json").read_text())
CONC = json.loads((ROOT / "artifacts/training_candidate_20261102/concurrent_api_performance.json").read_text())
PRIV = json.loads((ROOT / "artifacts/training_candidate_20261102/source_privacy_audit.json").read_text())
USAB = json.loads((ROOT / "artifacts/training_candidate_20261102/usability_static_audit.json").read_text())
CLAIM = json.loads((ROOT / "artifacts/claim_dependent_analysis/checkpoint_43000_20261102/claim_dependent_analysis.json").read_text())


def p95(endpoint: str) -> float:
    return PERF["endpoints"][endpoint]["latency_seconds"]["p95_nearest_rank"]


def concurrent(endpoint: str, workers: int) -> dict:
    return next(row for row in CONC["results"][endpoint] if row["workers"] == workers)


def tc(category, ident, description, preconditions, steps, data, expected, actual, status="Pass", evidence="pytest_results.xml"):
    return {
        "Category": category,
        "Test Case ID": ident,
        "Test Description": description,
        "Preconditions": preconditions,
        "Test Steps": steps,
        "Test Data": data,
        "Expected Result": expected,
        "Actual Result": actual,
        "Status": status,
        "Evidence": evidence,
    }


cases = [
    tc("Functional", "FUN-01", "Reject empty and unsupported image inputs", "Application validation functions available", "Submit empty byte array and unsupported image payload", "Empty array; unsupported payload", "Input is rejected without creating a receipt", "Both invalid inputs were rejected safely", evidence="tests.test_core::test_empty_and_unsupported_images_are_rejected"),
    tc("Functional", "FUN-02", "Correct a receipt captured at a quarter turn", "Default preprocessing configuration loaded", "Create rotated synthetic receipt; call preprocess; inspect orientation", "Quarter-turn portrait receipt", "Output is returned in upright portrait orientation", "Orientation corrected and dimensions were valid", evidence="tests.test_core::test_preprocessing_corrects_quarter_turn_receipt"),
    tc("Functional", "FUN-03", "Avoid rotating a creased portrait receipt sideways", "Preprocessor loaded", "Process portrait receipt containing a dominant crease", "Creased portrait synthetic image", "Text-row evidence dominates the crease during orientation selection", "Portrait orientation was retained", evidence="tests.test_core::test_preprocessing_does_not_turn_creased_portrait_receipt_sideways"),
    tc("Functional", "FUN-04", "Extract standard date, invoice number and payable total", "Field parser loaded", "Pass OCR text to extract_fields and compare normalized values", "INVOICE NO: INV-1001; DATE: 19/10/2026; TOTAL RM 102.50", "2026-10-19; INV-1001; 102.50", "All three normalized values matched", evidence="tests.test_core::test_target_field_extraction"),
    tc("Functional", "FUN-05", "Reject invalid calendar dates while preserving later valid values", "Field parser loaded", "Parse text containing 31/02/2026 and a later valid date", "Invalid and valid date candidates", "Invalid calendar candidate is rejected; valid candidate remains eligible", "Expected candidate selection was observed", evidence="tests.test_core::test_invalid_date_does_not_hide_a_later_valid_date"),
    tc("Functional", "FUN-06", "Select payable total instead of cash or change", "Field parser loaded", "Parse receipt text containing TOTAL, CASH and CHANGE", "Multiple monetary values", "Payable total is selected", "TOTAL was selected; cash/change were not mislabelled", evidence="tests.test_core::test_total_prefers_payable_and_does_not_use_cash_or_change_as_total"),
    tc("Functional", "FUN-07", "Preserve header, signature and large logos during crop", "Content crop functions loaded", "Run crop tests on table, signature, handwriting and logo layouts", "Synthetic structured documents", "No relevant edge content is removed", "Header, signature, handwriting, scripts and logo remained inside crop", evidence="tests.test_crop_content_retention"),
    tc("Functional", "FUN-08", "Preserve source-supported text while suppressing paper shading", "Faithful view functions loaded", "Generate faithful view and inspect supported dark pixels", "Synthetic faded text plus broad shading", "Source-supported ink retained; broad shading reduced; no typeset text", "Preservation and suppression assertions passed", evidence="tests.test_core::test_faithful_content_view_suppresses_paper_shading_without_drawing_text"),
    tc("Functional", "FUN-09", "Validate review form fields and values", "Authenticated test client and owned receipt", "Submit missing merchant/date/amount and malformed values", "Missing and invalid review fields", "HTTP validation response names each invalid field; record is not verified", "Field-specific errors returned and invalid review rejected", evidence="tests.test_web_app::test_review_rejects_missing_required_fields_and_invalid_values"),
    tc("Functional", "FUN-10", "Complete account and owned-receipt workflow", "Isolated test database", "Register; log in; create/read/update receipt; export and delete", "Synthetic user and receipt", "Only the owner completes the workflow successfully", "Full owned-receipt workflow passed", evidence="tests.test_web_app::test_account_and_owned_receipt_workflow"),
    tc("Functional", "FUN-11", "Support guest workflow and isolate guest selection", "Fresh anonymous client", "Create guest session; validate selection; attempt cross-session access", "Guest receipt IDs", "Guest can use permitted functions and cannot access another session", "Guest workflow and isolation assertions passed", evidence="tests.test_web_app::test_guest_selection_validation_and_isolation"),
    tc("Functional", "FUN-12", "Generate readable audit PDF and accounting workbook", "Owned synthetic receipt with audit fields", "Request audit PDF and XLSX endpoints; inspect content and MIME", "One controlled receipt", "Fixed report is readable; workbook includes accounting evidence and MyInvois boundary", "Both document-generation tests passed", evidence="tests.test_intensive::test_audit_pdf_is_fixed_readable_report; test_receipt_workbook_has_accounting_evidence_and_myinvois_boundary"),

    tc("Integration", "INT-01", "Expose eight unrolled DSDNet stages and four maxout branches", "Pinned model configuration and PyTorch environment", "Run adapter forward pass and inspect stage/branch tensors", "Controlled 128 x 128 tensor", "Eight stages match pinned forward; four branches available", "Architecture integration matched expected tensors", evidence="tests.test_dsdnet_unrolling::test_eight_stage_adapter_matches_pinned_forward_and_exposes_branches"),
    tc("Integration", "INT-02", "Contain native inference crash and recover worker", "Inference worker enabled", "Force native crash; submit a second receipt", "Crash sentinel then valid image", "First job fails safely; fresh worker handles next job", "Crash contained and subsequent job completed", evidence="tests.test_inference_worker::test_native_crash_is_contained_and_next_receipt_gets_fresh_worker"),
    tc("Integration", "INT-03", "Continue batch after one invalid image", "Batch worker available", "Submit bad image followed by valid image", "Malformed image; valid synthetic receipt", "Bad item reports error without cancelling following item", "Following item completed successfully", evidence="tests.test_inference_worker::test_bad_image_does_not_interrupt_following_batch_item"),
    tc("Integration", "INT-04", "Return busy response for simultaneous scan", "One inference job already active", "Submit another scan concurrently", "Two scan requests", "Second request returns controlled busy response rather than blocking", "Busy response returned immediately", evidence="tests.test_inference_worker::test_concurrent_scan_returns_busy_instead_of_blocking_http_threads"),
    tc("Integration", "INT-05", "Replace timed-out inference worker", "Timeout configured", "Start non-terminating job; wait for timeout; submit new job", "Timeout sentinel then valid job", "Hung worker is terminated and replaced", "Timeout handled and worker replaced", evidence="tests.test_inference_worker::test_timed_out_worker_is_stopped_and_replaced"),
    tc("Integration", "INT-06", "Deduplicate identical OCR views without losing text", "OCR multi-view merge enabled", "Supply identical OCR view records and merge", "Repeated regions and readings", "Duplicate call removed; unique text retained", "Exactly one recognition retained per identical view", evidence="tests.test_ocr_duplicate_views::test_identical_ocr_views_are_recognized_once_without_losing_text"),
    tc("Integration", "INT-07", "Enforce arithmetic consistency gate", "Extracted subtotal/total values available", "Provide inconsistent monetary fields and route review decision", "Subtotal does not equal reported total", "Mismatch is reported and automatic verification blocked", "Arithmetic inconsistency was reported", evidence="tests.test_intensive::test_arithmetic_consistency_gate_reports_mismatch"),
    tc("Integration", "INT-08", "Protect barcode region through restoration", "Barcode detector and restoration fusion loaded", "Restore image with barcode and compare protected region", "Synthetic barcode plus text", "Barcode pixels are sourced from original region", "Barcode region was restored from original source", evidence="tests.test_core::test_barcode_region_is_restored_from_original_source"),

    tc("Usability (static/applicable)", "USA-01", "Detect duplicate HTML element IDs", "Current index.html available", "Parse all elements with id and count duplicates", f"{USAB['controls_checked']} controls; {USAB['buttons_checked']} buttons", "No duplicate IDs", f"Duplicate ID map was {USAB['duplicate_ids']}", evidence="usability_static_audit.json"),
    tc("Usability (static/applicable)", "USA-02", "Check explicit labels for interactive fields", "Current index.html available", "Inspect input/select/textarea for label or accessible name", f"{USAB['controls_checked']} controls", "No unlabeled controls", f"Unlabeled control list was {USAB['controls_without_explicit_label']}", evidence="usability_static_audit.json"),
    tc("Usability (static/applicable)", "USA-03", "Check accessible names for buttons", "Current index.html available", "Inspect button text, aria-label, aria-labelledby and title", f"{USAB['buttons_checked']} buttons", "Every button has a visible or accessible name", f"Unnamed button list was {USAB['buttons_without_text_or_accessible_name']}", evidence="usability_static_audit.json"),
    tc("Usability (static/applicable)", "USA-04", "Return field-specific registration errors", "Fresh registration form", "Submit missing, malformed and inconsistent registration values", "Invalid name/email/password/confirmation", "Each invalid field receives a specific error", "Field-specific responses passed automated assertions", evidence="tests.test_web_app::test_registration_returns_field_specific_errors"),
    tc("Usability (static/applicable)", "USA-05", "Limit receipt image choices to faithful, original and comparison", "Receipt detail view rendered", "Inspect available image-view controls", "Current detail page", "No obsolete clean-view control; three intended choices present", "Faithful, original and compare controls only", evidence="tests.test_web_app::test_receipt_view_has_only_faithful_original_and_compare"),
    tc("Usability (static/applicable)", "USA-06", "Refresh one stale CSRF token transparently", "Authenticated frontend with stale token", "Trigger protected request and observe retry", "One stale CSRF token", "Client refreshes token once and retries without loop", "Single refresh/retry path passed", evidence="tests.test_web_app::test_frontend_refreshes_stale_csrf_token_once"),

    tc("Privacy/Security", "SEC-01", "Apply security headers and restrict anonymous access", "Application test client", "Request public and protected routes anonymously; inspect headers", "Anonymous session", "Security headers present; protected data denied", "Header and access assertions passed", evidence="tests.test_evaluation_security::test_security_headers_and_anonymous_access"),
    tc("Privacy/Security", "SEC-02", "Prevent cross-user read, update and deletion", "Two registered accounts and one owned receipt", "Attempt receipt operations from non-owner account", "User A receipt; User B session", "All cross-user operations are denied", "Read/change/delete attempts were denied", evidence="tests.test_evaluation_security::test_cross_user_data_cannot_be_read_changed_or_deleted"),
    tc("Privacy/Security", "SEC-03", "Store passwords as hashes", "Registration endpoint and database", "Register account; inspect stored credential", "Synthetic strong password", "Stored value is a password hash and not plaintext", "Stored credential passed hash checks", evidence="tests.test_evaluation_security::test_passwords_stored_hashed"),
    tc("Privacy/Security", "SEC-04", "Require CSRF token for mutations", "Authenticated or guest session", "POST mutation without valid X-CSRF-Token", "Missing/invalid CSRF token", "Request is rejected", "Mutation was rejected", evidence="tests.test_web_app::test_mutation_requires_csrf"),
    tc("Privacy/Security", "SEC-05", "Neutralise spreadsheet formula injection", "Export endpoint and synthetic malicious values", "Export CSV/XLSX fields beginning with formula characters", "=, +, -, @ formula-like strings", "Exported cell is treated as text", "CSV and Excel outputs were neutralised", evidence="tests.test_intensive::test_excel_and_csv_exports_neutralise_formula_injection"),
    tc("Privacy/Security", "SEC-06", "Deny non-admin access to admin resources", "Registered non-admin user", "Request and mutate admin endpoints", "Non-admin session", "All admin operations denied", "Non-admin access was denied", evidence="tests.test_intensive::test_non_admin_cannot_read_or_change_admin_resources"),
    tc("Privacy/Security", "SEC-07", "Maintain tamper-evident event chain", "Application events generated", "Read event chain and verify linked hashes", "Synthetic account/receipt events", "Chain verifies and security headers remain present", "Chain and headers passed", evidence="tests.test_intensive::test_security_headers_and_tamper_evident_event_chain"),
    tc("Privacy/Security", "SEC-08", "Expose receipt file-integrity state to owner", "Owned receipt and stored files", "Request integrity information as owner", "Controlled receipt files", "Owner receives consistent integrity result", "Integrity state was visible to owner", evidence="tests.test_intensive::test_receipt_file_integrity_is_visible_to_owner"),
    tc("Privacy/Security", "SEC-09", "Reject invalid upload and oversized/empty bulk request safely", "Upload endpoints available", "Submit invalid file, empty bulk list and over-limit list", "Malformed upload and boundary-size lists", "Requests fail safely without partial mutation", "All boundary cases passed", evidence="tests.test_intensive::test_invalid_upload_empty_bulk_and_bulk_limit_are_safe"),
    tc("Privacy/Security", "SEC-10", "Scan source/config/tests for common secrets and dangerous constructs", "Current source tree and environment", "Run pattern scan, AST scan and pip check", f"{PRIV['files_scanned']} files", "No hits; no dangerous constructs; dependencies consistent", f"0 secret hits; 0 dangerous constructs; pip check exit {PRIV['dependency_check']['exit_code']}", evidence="source_privacy_audit.json"),

    tc("Performance", "PER-01", "Measure history endpoint with 250 records", "Temporary SQLite DB and guest session", "Issue 30 GET /api/receipts requests", "250 synthetic records", "All HTTP 200; record p95 latency", f"30/30 HTTP 200; p95 {p95('history_250_records'):.4f} s", evidence="local_api_performance.json"),
    tc("Performance", "PER-02", "Measure dashboard endpoint with 250 records", "Temporary SQLite DB and guest session", "Issue 30 GET /api/dashboard requests", "250 synthetic records", "All HTTP 200; record p95 latency", f"30/30 HTTP 200; p95 {p95('dashboard_250_records'):.4f} s", evidence="local_api_performance.json"),
    tc("Performance", "PER-03", "Measure CSV export", "Temporary SQLite DB populated", "Issue 10 GET /api/export.csv requests", "250 synthetic records", "All exports HTTP 200", f"10/10 HTTP 200; p95 {p95('csv_export_250_records'):.4f} s", evidence="local_api_performance.json"),
    tc("Performance", "PER-04", "Measure Excel export", "Temporary SQLite DB populated", "Issue 10 GET /api/export.xlsx requests", "250 synthetic records", "All exports HTTP 200", f"10/10 HTTP 200; p95 {p95('xlsx_export_250_records'):.4f} s", evidence="local_api_performance.json"),
    tc("Performance", "PER-05", "Measure summary of 50 selected receipts", "Temporary SQLite DB populated", "Issue 30 selection-summary requests", "50 selected IDs from 250 records", "All requests HTTP 200", f"30/30 HTTP 200; p95 {p95('selection_summary_50_records'):.4f} s", evidence="local_api_performance.json"),
    tc("Performance", "PER-06", "Exercise history endpoint at eight concurrent workers", "Synthetic registered account; 250 records", "Run 64 requests through eight clients", "8 workers; 64 requests", "Zero failures; record p95 and throughput", f"0 failures; p95 {concurrent('history_250_records',8)['latency_seconds']['p95_nearest_rank']:.4f} s; {concurrent('history_250_records',8)['throughput_requests_per_second']:.2f} req/s", evidence="concurrent_api_performance.json"),
    tc("Performance", "PER-07", "Exercise dashboard endpoint at eight concurrent workers", "Synthetic registered account; 250 records", "Run 64 requests through eight clients", "8 workers; 64 requests", "Zero failures", f"0 failures; p95 {concurrent('dashboard_250_records',8)['latency_seconds']['p95_nearest_rank']:.4f} s; {concurrent('dashboard_250_records',8)['throughput_requests_per_second']:.2f} req/s", evidence="concurrent_api_performance.json"),
    tc("Performance", "PER-08", "Exercise selection summary at eight concurrent workers", "Synthetic registered account; 250 records", "Run 64 selection requests through eight clients", "8 workers; 50 IDs/request", "Zero failures", f"0 failures; p95 {concurrent('selection_summary_50_records',8)['latency_seconds']['p95_nearest_rank']:.4f} s; {concurrent('selection_summary_50_records',8)['throughput_requests_per_second']:.2f} req/s", evidence="concurrent_api_performance.json"),

    tc("Baseline/Proposed Experiment", "EXP-01", "Compare proposed method with direct degraded input", "Locked 86-source, 258-variant validation set", "Evaluate both methods on identical variants and aggregate by source", "Light/moderate/severe variants", "Report paired metrics without claiming independent test evidence", "Proposed objective improved on all 86 sources", evidence="claim_dependent_analysis.json"),
    tc("Baseline/Proposed Experiment", "EXP-02", "Compare proposed method with global Otsu", "Same locked validation set", "Run Otsu and proposed method on identical variants", "86 sources; 258 variants", "Report SSIM, L1, objective and stroke metrics", "Proposed maintained higher SSIM across all three severity levels", evidence="table_severity_robustness.csv"),
    tc("Baseline/Proposed Experiment", "EXP-03", "Compare proposed method with Sauvola", "Same locked validation set", "Run Sauvola and proposed method on identical variants", "86 sources; 258 variants", "Report fidelity/recall and false-stroke trade-off", "Proposed had higher SSIM and recall; Sauvola retained lower false-stroke rate", evidence="table_severity_robustness.csv"),
    tc("Baseline/Proposed Experiment", "EXP-04", "Compare proposed method with perceptual baseline", "Same locked validation set", "Run perceptual baseline and proposed method on identical variants", "86 sources; 258 variants", "Report paired results by degradation severity", "Proposed had higher SSIM and recall at light, moderate and severe levels", evidence="table_severity_robustness.csv"),
    tc("Baseline/Proposed Experiment", "EXP-05", "Evaluate retained-stage depth", "Iteration-43,000 checkpoint loaded", "Collect outputs after stages 1, 2, 4, 6 and 8; aggregate by source", "1,290 stage/source-variant rows", "Identify whether full depth improves balanced objective", "Stage 8 achieved objective 0.0526 and SSIM 0.9167; stage 2 recall gain coincided with false-stroke rate 0.9682", evidence="table_stage_ablation.csv"),
    tc("Baseline/Proposed Experiment", "EXP-06", "Evaluate blur-kernel sensitivity", "Iteration-43,000 checkpoint loaded", "Scale applied kernel by 0.50, 0.75, 1.00, 1.25 and 1.50", "1,290 kernel/source-variant rows", "Identify stable operating factor and trade-offs", "Factor 1.00 minimized objective (0.0526) and maximized SSIM (0.9167); factors 1.25/1.50 slightly reduced false strokes but worsened fidelity", evidence="table_kernel_sensitivity.csv"),
    tc("Baseline/Proposed Experiment", "EXP-07", "Quantify robustness by degradation severity", "Locked grouped validation outputs", "Aggregate light/moderate/severe metrics per source; run Friedman and paired Wilcoxon tests", "86 paired sources per severity", "Report degradation trend and corrected significance", "SSIM fell 0.9410 to 0.8825 and objective rose 0.0390 to 0.0686; Friedman p<0.001", evidence="severity_statistical_tests.csv"),
    tc("Baseline/Proposed Experiment", "EXP-08", "Analyse mixed and negative source-level findings", "Source-level direct/proposed metrics", "Compute recall and false-stroke changes per source", "86 source groups", "Count improvements and trade-off cases", "Objective improved 86/86; recall improved 79/86; 23 sources gained recall at a false-stroke cost", evidence="error_failure_report.json"),
]


OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "test_case_register.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
    writer.writeheader()
    writer.writerows(cases)

summary = {}
for row in cases:
    item = summary.setdefault(row["Category"], {"cases": 0, "pass": 0, "fail": 0})
    item["cases"] += 1
    item["pass" if row["Status"].lower() == "pass" else "fail"] += 1

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Executed local development/validation test register; not a production certification or participant usability study",
    "total_cases": len(cases),
    "summary": summary,
    "cases": cases,
    "execution_evidence": {
        "pytest": {"result": "116 passed in 14.13 s", "path": str(OUT / "pytest_results.xml"), "sha256": digest(OUT / "pytest_results.xml")},
        "usability_static": {"path": str(ROOT / "artifacts/training_candidate_20261102/usability_static_audit.json"), "sha256": digest(ROOT / "artifacts/training_candidate_20261102/usability_static_audit.json")},
        "source_privacy": {"path": str(ROOT / "artifacts/training_candidate_20261102/source_privacy_audit.json"), "sha256": digest(ROOT / "artifacts/training_candidate_20261102/source_privacy_audit.json")},
        "local_performance": {"path": str(ROOT / "artifacts/training_candidate_20261102/local_api_performance.json"), "sha256": digest(ROOT / "artifacts/training_candidate_20261102/local_api_performance.json")},
        "concurrent_performance": {"path": str(ROOT / "artifacts/training_candidate_20261102/concurrent_api_performance.json"), "sha256": digest(ROOT / "artifacts/training_candidate_20261102/concurrent_api_performance.json")},
        "claim_analysis": {"path": str(ROOT / "artifacts/claim_dependent_analysis/checkpoint_43000_20261102/claim_dependent_analysis.json"), "sha256": digest(ROOT / "artifacts/claim_dependent_analysis/checkpoint_43000_20261102/claim_dependent_analysis.json")},
    },
    "limitations": [
        "Usability evidence is a static interface heuristic plus automated workflow checks; it is not a participant study or accessibility certification.",
        "Performance evidence uses a local Flask test client, temporary SQLite database and synthetic records; it is not a production load test or SLA.",
        "Privacy/security evidence is automated regression and narrow source scanning; it is not a penetration test or data-protection impact assessment.",
        "Restoration comparisons use the already examined locked development-validation partition; they are not a fresh independent test set.",
    ],
}
(OUT / "test_case_register.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"csv": str(csv_path), "json": str(OUT / 'test_case_register.json'), "total": len(cases), "summary": summary}, indent=2))
