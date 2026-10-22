from __future__ import annotations

import json
from pathlib import Path

from dsdnet_receipt.storage import utc_now
from dsdnet_receipt.web_app import create_app


def csrf(client) -> str:
    return client.get("/api/auth").get_json()["csrf_token"]


def test_account_and_owned_receipt_workflow(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client()

    token = csrf(client)
    response = client.post(
        "/api/register",
        json={"name": "Test User", "email": "test@valid-domain.my", "password": "long-test-password-1", "password_confirmation": "long-test-password-1", "policy_accepted": True},
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 201
    token = response.get_json()["csrf_token"]
    user = response.get_json()["user"]

    source = tmp_path / "original.png"
    restored = tmp_path / "restored.png"
    source.write_bytes(b"test image")
    restored.write_bytes(b"test image")
    now = utc_now()
    store = app.extensions["receipt_store"]
    store.create_receipt(
        {
            "public_id": "USER-TEST-001",
            "user_id": user["id"],
            "source_name": "receipt.png",
            "created_at": now,
            "updated_at": now,
            "status": "needs_review",
            "merchant": None,
            "transaction_date": "2026-10-19",
            "receipt_number": "R-101",
            "total_amount": "12.50",
            "tags": "[]",
            "notes": "",
            "ocr_text": "TOTAL RM 12.50",
            "audit_json": json.dumps({"run_id": "USER-TEST-001"}),
            "input_path": str(source),
            "restored_path": str(restored),
            "runtime_seconds": 0.1,
            "restoration_status": "ok",
            "support_status": "unavailable",
        }
    )

    assert client.get("/api/dashboard").get_json()["stats"]["total"] == 1
    update = client.patch(
        "/api/receipts/USER-TEST-001",
        json={"merchant": "Local Shop", "status": "reviewed", "tags": ["claim"], "line_items": [{"item":"1","description":"Office paper","quantity":"2","unit":"pack","unit_price":"5.00","discount":"0.00","amount":"10.00","review_note":"Verified"}]},
        headers={"X-CSRF-Token": token},
    )
    assert update.status_code == 200
    assert update.get_json()["merchant"] == "Local Shop"
    assert update.get_json()["status"] == "reviewed"
    assert update.get_json()["line_items"][0]["description"] == "Office paper"
    assert client.get("/api/export.csv").status_code == 200

    assert client.post("/api/logout", headers={"X-CSRF-Token": token}).status_code == 200
    assert client.get("/api/dashboard").status_code == 401


def test_mutation_requires_csrf(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client()
    csrf(client)
    response = client.post(
        "/api/register",
        json={"name": "Test User", "email": "test@valid-domain.my", "password": "long-test-password-1", "password_confirmation": "long-test-password-1", "policy_accepted": True},
    )
    assert response.status_code == 403


def test_registration_returns_field_specific_errors(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client()
    response = client.post(
        "/api/register",
        json={"name": "A", "email": "fake@invalid.t", "password": "short", "password_confirmation": "different", "policy_accepted": False},
        headers={"X-CSRF-Token": csrf(client)},
    )
    assert response.status_code == 400
    assert set(response.get_json()["fields"]) == {"name", "email", "password", "confirmation", "policy"}


def test_duplicate_registration_is_rejected(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    first = app.test_client()
    payload = {
        "name": "Duplicate User", "email": "duplicate@valid-domain.my",
        "password": "a long unique passphrase 2026", "password_confirmation": "a long unique passphrase 2026",
        "policy_accepted": True,
    }
    assert first.post("/api/register", json=payload, headers={"X-CSRF-Token": csrf(first)}).status_code == 201
    second = app.test_client()
    response = second.post("/api/register", json=payload, headers={"X-CSRF-Token": csrf(second)})
    assert response.status_code == 409
    assert "email" in response.get_json()["fields"]


def test_review_rejects_missing_required_fields_and_invalid_values(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client(); token = csrf(client)
    response = client.post("/api/register", json={"name":"Review User","email":"review@valid-domain.my","password":"a secure review passphrase 2026","password_confirmation":"a secure review passphrase 2026","policy_accepted":True}, headers={"X-CSRF-Token":token})
    token = response.get_json()["csrf_token"]; user = response.get_json()["user"]; store = app.extensions["receipt_store"]; now = utc_now()
    source = tmp_path / "a.png"; source.write_bytes(b"x")
    store.create_receipt({"public_id":"R-INVALID","user_id":user["id"],"source_name":"a.png","created_at":now,"updated_at":now,"status":"needs_review","merchant":None,"transaction_date":None,"receipt_number":None,"total_amount":None,"tags":"[]","notes":"","ocr_text":"","audit_json":"{}","input_path":str(source),"restored_path":str(source),"runtime_seconds":0,"restoration_status":"ok","support_status":"unavailable"})
    result = client.patch("/api/receipts/R-INVALID", json={"transaction_date":"2025-02-31","total_amount":"-2","status":"reviewed"}, headers={"X-CSRF-Token":token})
    assert result.status_code == 400
    assert set(result.get_json()["fields"]) == {"transaction_date", "total_amount", "status"}


def test_receipt_view_defaults_to_faithful_recovery_and_separates_reading_copy(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'class="comparison improved-mode"' in html
    assert 'id="faithfulViewButton" class="active"' in html
    assert 'id="readingViewButton"' in html
    assert 'Faithful DSDNet recovery · thesis output' in html
    assert 'Optional reading copy' not in html  # runtime label depends on the receipt
    assert 'id="compareViewButton"' in html
