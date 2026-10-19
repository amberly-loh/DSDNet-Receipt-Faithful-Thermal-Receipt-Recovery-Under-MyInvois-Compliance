from __future__ import annotations

import io
import json
from pathlib import Path

from openpyxl import load_workbook

from dsdnet_receipt.storage import utc_now
from dsdnet_receipt.web_app import create_app


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"


def auth_token(client) -> str:
    return client.get("/api/auth").get_json()["csrf_token"]


def register(client, email: str, name: str = "Audit User") -> tuple[str, dict]:
    response = client.post("/api/register", json={
        "name": name, "email": email, "password": "A long unique audit passphrase 2026!",
        "password_confirmation": "A long unique audit passphrase 2026!", "policy_accepted": True,
    }, headers={"X-CSRF-Token": auth_token(client)})
    assert response.status_code == 201
    return response.get_json()["csrf_token"], response.get_json()["user"]


def add_receipt(app, tmp_path: Path, user: dict, receipt_id: str = "OWNER-001", merchant: str = "Local Shop") -> None:
    original, restored = tmp_path / f"{receipt_id}-o.png", tmp_path / f"{receipt_id}-r.png"
    original.write_bytes(b"png"); restored.write_bytes(b"png")
    now = utc_now()
    app.extensions["receipt_store"].create_receipt({
        "public_id": receipt_id, "user_id": user["id"], "source_name": "receipt.png", "created_at": now,
        "updated_at": now, "status": "needs_review", "merchant": merchant, "transaction_date": "2026-10-20",
        "receipt_number": "INV-001", "total_amount": "10.00", "tags": "[]", "notes": "", "ocr_text": "",
        "audit_json": json.dumps({"run_id": receipt_id, "ocr_lines": []}), "input_path": str(original),
        "restored_path": str(restored), "runtime_seconds": .1, "restoration_status": "ok", "support_status": "unavailable",
    })


def test_all_protected_get_routes_reject_anonymous_user(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    for path in ("/api/dashboard", "/api/receipts", "/api/export.csv", "/api/export.xlsx", "/api/admin/overview", "/api/admin/receipts", "/api/admin/events"):
        assert client.get(path).status_code == 401, path


def test_receipt_ownership_isolation_covers_detail_images_and_exports(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    owner, outsider = app.test_client(), app.test_client()
    _, owner_user = register(owner, "owner@example.my", "Owner User")
    register(outsider, "outsider@example.my", "Other User")
    add_receipt(app, tmp_path, owner_user)
    for path in ("/api/receipts/OWNER-001", "/api/receipts/OWNER-001/image/original", "/api/receipts/OWNER-001/image/restored", "/api/receipts/OWNER-001/audit.json", "/api/receipts/OWNER-001/ocr.xlsx"):
        assert outsider.get(path).status_code == 404, path


def test_non_admin_cannot_read_or_change_admin_resources(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    first, user_client = app.test_client(), app.test_client()
    register(first, "admin@example.my", "Install Owner")
    token, user = register(user_client, "user@example.my", "Normal User")
    assert user["role"] == "user"
    assert user_client.get("/api/admin/overview").status_code == 403
    assert user_client.patch(f"/api/admin/users/{user['public_id']}", json={"role":"admin"}, headers={"X-CSRF-Token":token}).status_code == 403


def test_invalid_upload_empty_bulk_and_bulk_limit_are_safe(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); token, _ = register(client, "upload@example.my")
    assert client.post("/api/analyze", data={}, headers={"X-CSRF-Token":token}).status_code == 400
    bad = {"receipt": (io.BytesIO(b"not an image"), "receipt.txt")}
    response = client.post("/api/analyze", data=bad, headers={"X-CSRF-Token":token}, content_type="multipart/form-data")
    assert response.status_code == 400 and "unreadable" in response.get_json()["error"]
    assert client.post("/api/bulk-analyze", data={}, headers={"X-CSRF-Token":token}).status_code == 400
    too_many = {"receipts": [(io.BytesIO(b"x"), f"r{i}.png") for i in range(26)]}
    response = client.post("/api/bulk-analyze", data=too_many, headers={"X-CSRF-Token":token}, content_type="multipart/form-data")
    assert response.status_code == 400 and "25" in response.get_json()["error"]


def test_excel_and_csv_exports_neutralise_formula_injection(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); _, user = register(client, "export@example.my")
    add_receipt(app, tmp_path, user, merchant="=HYPERLINK(\"https://invalid\")")
    workbook_response = client.get("/api/export.xlsx")
    assert workbook_response.status_code == 200
    workbook = load_workbook(io.BytesIO(workbook_response.data), data_only=False)
    assert workbook["Receipts"]["D2"].value.startswith("'=")
    csv_response = client.get("/api/export.csv")
    assert csv_response.status_code == 200 and b"'=HYPERLINK" in csv_response.data


def test_health_policy_and_unknown_resources_are_consistent(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); token, _ = register(client, "status@example.my")
    assert client.get("/api/health").get_json()["status"] == "ok"
    policy = client.get("/api/policy").get_json()
    assert "device" in policy["storage"].lower() and "own records" in policy["access"].lower()
    assert client.get("/api/receipts/MISSING").status_code == 404
    assert client.patch("/api/receipts/MISSING", json={}, headers={"X-CSRF-Token":token}).status_code == 404
    assert client.delete("/api/receipts/MISSING", headers={"X-CSRF-Token":token}).status_code == 404


def test_receipt_workbook_has_accounting_evidence_and_myinvois_boundary(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); _, user = register(client, "workbook@example.my")
    add_receipt(app, tmp_path, user, receipt_id="BOOK-001")
    response = client.get("/api/receipts/BOOK-001/ocr.xlsx")
    assert response.status_code == 200
    workbook = load_workbook(io.BytesIO(response.data), data_only=False)
    assert workbook.sheetnames == ["Receipt Summary", "Line Items", "OCR Evidence", "MyInvois Coverage"]
    assert "not a MyInvois submission file" in workbook["MyInvois Coverage"]["B1"].value


def test_google_login_without_configuration_fails_safely(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False); monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    response = app.test_client().get("/auth/google")
    assert response.status_code == 302 and "google_not_configured" in response.location
