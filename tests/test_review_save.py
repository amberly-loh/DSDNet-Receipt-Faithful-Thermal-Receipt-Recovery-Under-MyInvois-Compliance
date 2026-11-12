from __future__ import annotations

import json
from pathlib import Path

from dsdnet_receipt.storage import utc_now
from dsdnet_receipt.web_app import create_app


def _setup(tmp_path: Path):
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    r = client.post("/api/register", json={"name": "Test User", "email": "test@valid-domain.my", "password": "long-test-password-1",
                                           "password_confirmation": "long-test-password-1", "policy_accepted": True},
                    headers={"X-CSRF-Token": token})
    token, user = r.get_json()["csrf_token"], r.get_json()["user"]
    (tmp_path / "o.png").write_bytes(b"x")
    lines = [{"text": "KEDAI ABC", "confidence": 0.9}, {"text": "Receipt No: R-555", "confidence": 0.9},
             {"text": "Date: 19/10/2026", "confidence": 0.9}, {"text": "Coffee 1 x 5.00 5.00", "confidence": 0.9},
             {"text": "TOTAL RM 5.00", "confidence": 0.9}]
    now = utc_now()
    app.extensions["receipt_store"].create_receipt({
        "public_id": "SAVE-1", "user_id": user["id"], "source_name": "r.png", "created_at": now, "updated_at": now,
        "status": "needs_review", "merchant": "KEDAI ABC", "transaction_date": None, "receipt_number": None,
        "total_amount": "5.00", "tags": "[]", "notes": "", "ocr_text": "\n".join(l["text"] for l in lines),
        "audit_json": json.dumps({"run_id": "SAVE-1", "ocr_lines": lines}), "input_path": str(tmp_path / "o.png"),
        "restored_path": str(tmp_path / "o.png"), "runtime_seconds": 0.1, "restoration_status": "ok", "support_status": "unavailable"})
    return client, token


def test_save_returns_the_full_page_record(tmp_path: Path) -> None:
    client, token = _setup(tmp_path)
    saved = client.patch("/api/receipts/SAVE-1", headers={"X-CSRF-Token": token},
                         json={"merchant": "Kedai ABC Sdn Bhd", "transaction_date": "2026-10-19", "receipt_number": "R-555",
                               "total_amount": "5.00", "status": "reviewed", "accounting_fields": {"currency": "MYR"}}).get_json()
    assert saved["merchant"] == "Kedai ABC Sdn Bhd"
    assert saved["accounting_fields"]["currency"] == "MYR"          # parsed object, not a JSON string
    assert saved["review_readiness"]["state"] == "reviewed"          # page status pill can refresh from the save
    assert "integrity" in saved and "ocr_suggestions" in saved["audit"]


def test_values_the_user_cleared_stay_cleared(tmp_path: Path) -> None:
    client, token = _setup(tmp_path)
    # Opening the unsaved record may fill the missing number/date from the scan.
    first = client.get("/api/receipts/SAVE-1").get_json()
    assert first["receipt_number"] == "R-555"
    # The user deliberately clears the number and removes the line items.
    client.patch("/api/receipts/SAVE-1", headers={"X-CSRF-Token": token},
                 json={"receipt_number": "", "line_items": [], "notes": "no number on this one"})
    again = client.get("/api/receipts/SAVE-1").get_json()
    assert again["receipt_number"] == ""
    assert again["line_items"] == []
    assert again["notes"] == "no number on this one"
