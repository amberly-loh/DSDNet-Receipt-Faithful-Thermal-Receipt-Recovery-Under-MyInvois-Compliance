from __future__ import annotations

import io
import json
import hashlib
import sqlite3
from pathlib import Path

from openpyxl import load_workbook
from pypdf import PdfReader

from dsdnet_receipt.storage import utc_now
from dsdnet_receipt.web_app import arithmetic_consistency, create_app


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
    for path in ("/api/dashboard", "/api/receipts", "/api/export.csv", "/api/export.xlsx"):
        assert client.get(path).status_code == 401, path


def test_receipt_ownership_isolation_covers_detail_images_and_exports(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    owner, outsider = app.test_client(), app.test_client()
    _, owner_user = register(owner, "owner@example.my", "Owner User")
    register(outsider, "outsider@example.my", "Other User")
    add_receipt(app, tmp_path, owner_user)
    for path in ("/api/receipts/OWNER-001", "/api/receipts/OWNER-001/image/original", "/api/receipts/OWNER-001/image/restored", "/api/receipts/OWNER-001/audit.json", "/api/receipts/OWNER-001/audit.pdf", "/api/receipts/OWNER-001/ocr.xlsx"):
        assert outsider.get(path).status_code == 404, path


def test_there_is_no_admin_side(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    first, user_client = app.test_client(), app.test_client()
    register(first, "admin@example.my", "Install Owner")
    token, user = register(user_client, "user@example.my", "Normal User")
    assert user["role"] == "user"
    for client in (first, user_client):
        assert client.get("/api/admin/overview").status_code == 404
        assert client.get("/api/admin/receipts").status_code == 404
        assert client.patch(f"/api/admin/users/{user['public_id']}", json={"role":"admin"}, headers={"X-CSRF-Token":token}).status_code in {403, 404}
    assert b'adminPage' not in first.get("/").data


def test_invalid_upload_empty_bulk_and_bulk_limit_are_safe(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); token, _ = register(client, "upload@example.my")
    assert client.post("/api/analyze", data={}, headers={"X-CSRF-Token":token}).status_code == 400
    bad = {"receipt": (io.BytesIO(b"not an image"), "receipt.txt")}
    response = client.post("/api/analyze", data=bad, headers={"X-CSRF-Token":token}, content_type="multipart/form-data")
    assert response.status_code == 400 and ("unreadable" in response.get_json()["error"] or "upload a JPEG" in response.get_json()["error"])
    assert client.post("/api/bulk-analyze", data={}, headers={"X-CSRF-Token":token}).status_code == 400
    too_many = {"receipts": [(io.BytesIO(b"x"), f"r{i}.png") for i in range(101)]}
    response = client.post("/api/bulk-analyze", data=too_many, headers={"X-CSRF-Token":token}, content_type="multipart/form-data")
    assert response.status_code == 400 and "100" in response.get_json()["error"]


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


def test_security_headers_and_tamper_evident_event_chain(tmp_path: Path) -> None:
    database = tmp_path / "db.sqlite3"
    app = create_app(CONFIG, testing=True, database_path=database)
    client = app.test_client(); register(client, "chain@example.my")
    response = client.get("/api/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.get_json()["audit_chain"] == "verified"
    with sqlite3.connect(database) as db:
        db.execute("UPDATE audit_events SET action='tampered' WHERE id=(SELECT MIN(id) FROM audit_events)")
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.get_json()["audit_chain"] == "failed"
    restarted = create_app(CONFIG, testing=True, database_path=database).test_client()
    assert restarted.get("/api/health").status_code == 503


def test_receipt_file_integrity_is_visible_to_owner(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); _, user = register(client, "integrity@example.my")
    add_receipt(app, tmp_path, user, receipt_id="HASH-001")
    store = app.extensions["receipt_store"]
    receipt = store.get_receipt(user["id"], "HASH-001")
    hashes = {
        "original": hashlib.sha256(Path(receipt["input_path"]).read_bytes()).hexdigest(),
        "recovered": hashlib.sha256(Path(receipt["restored_path"]).read_bytes()).hexdigest(),
    }
    with store.connect() as db:
        audit = receipt["audit"]; audit["stored_file_sha256"] = hashes
        db.execute("UPDATE receipts SET audit_json=? WHERE public_id='HASH-001'", (json.dumps(audit),))
    assert client.get("/api/receipts/HASH-001").get_json()["integrity"]["status"] == "verified"
    Path(receipt["restored_path"]).write_bytes(b"changed")
    assert client.get("/api/receipts/HASH-001").get_json()["integrity"]["status"] == "failed"


def test_receipt_workbook_has_accounting_evidence_and_myinvois_boundary(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); _, user = register(client, "workbook@example.my")
    add_receipt(app, tmp_path, user, receipt_id="BOOK-001")
    response = client.get("/api/receipts/BOOK-001/ocr.xlsx")
    assert response.status_code == 200
    workbook = load_workbook(io.BytesIO(response.data), data_only=False)
    assert workbook.sheetnames == ["Receipt Summary", "Line Items", "OCR Evidence", "MyInvois Coverage"]
    assert "not a MyInvois submission file" in workbook["MyInvois Coverage"]["B1"].value


def test_audit_pdf_is_fixed_readable_report(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client(); _, user = register(client, "pdf@example.my")
    add_receipt(app, tmp_path, user, receipt_id="PDF-001")
    response = client.get("/api/receipts/PDF-001/audit.pdf")
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    reader = PdfReader(io.BytesIO(response.data))
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for expected in ("Receipt Audit Report", "PDF-001", "Local Shop", "Accounting evidence", "Audit trail"):
        assert expected in text


def test_google_sign_in_is_not_offered(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    assert client.get("/auth/google").status_code == 404
    assert b"Continue with Google" not in client.get("/").data


def test_this_computer_opens_without_sign_in_until_sign_in_is_required(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    state = client.get("/api/auth").get_json()
    assert state["user"] is None and state["workspace_available"] and not state["sign_in_required"]
    assert client.get("/api/dashboard").status_code == 401  # nothing opens by itself on the server
    opened = client.post("/api/workspace", headers={"X-CSRF-Token": state["csrf_token"]})
    assert opened.status_code == 200 and opened.get_json()["user"]["is_workspace"]
    token = opened.get_json()["csrf_token"]
    assert client.get("/api/dashboard").status_code == 200
    # the same workspace (and its receipts) opens again later
    again = app.test_client(); t2 = again.get("/api/auth").get_json()["csrf_token"]
    assert again.post("/api/workspace", headers={"X-CSRF-Token": t2}).get_json()["user"]["id"] == opened.get_json()["user"]["id"]
    # requiring sign-in first gives the workspace an email and password
    weak = client.patch("/api/access", json={"require_sign_in": True}, headers={"X-CSRF-Token": token})
    assert weak.status_code == 400 and "email" in weak.get_json()["fields"]
    done = client.patch("/api/access", headers={"X-CSRF-Token": token}, json={
        "require_sign_in": True, "name": "Siti Aminah", "email": "siti@example.my",
        "password": "Receipt-Ledger-2026", "password_confirmation": "Receipt-Ledger-2026"})
    assert done.status_code == 200 and done.get_json()["sign_in_required"]
    fresh = app.test_client(); state = fresh.get("/api/auth").get_json()
    assert not state["workspace_available"]
    assert fresh.post("/api/workspace", headers={"X-CSRF-Token": state["csrf_token"]}).status_code == 403
    login = fresh.post("/api/login", json={"email": "siti@example.my", "password": "Receipt-Ledger-2026"}, headers={"X-CSRF-Token": state["csrf_token"]})
    assert login.status_code == 200 and login.get_json()["user"]["id"] == opened.get_json()["user"]["id"]
    # turning it off again opens that account without a password
    token = login.get_json()["csrf_token"]
    assert fresh.patch("/api/access", json={"require_sign_in": False}, headers={"X-CSRF-Token": token}).get_json()["workspace_available"]

def test_arithmetic_consistency_gate_reports_mismatch() -> None:
    lines = [
        {"text":"ITEM", "box":[[10,10],[35,10],[35,20],[10,20]]},
        {"text":"DESCRIPTION", "box":[[50,10],[140,10],[140,20],[50,20]]},
        {"text":"QUANTITY", "box":[[180,10],[230,10],[230,20],[180,20]]},
        {"text":"UNIT PRICE", "box":[[260,10],[320,10],[320,20],[260,20]]},
        {"text":"AMOUNT", "box":[[350,10],[400,10],[400,20],[350,20]]},
        {"text":"Service", "box":[[60,50],[130,50],[130,65],[60,65]]},
        {"text":"2.00", "box":[[190,50],[225,50],[225,65],[190,65]]},
        {"text":"10.00", "box":[[270,50],[315,50],[315,65],[270,65]]},
        {"text":"30.00", "box":[[355,50],[400,50],[400,65],[355,65]]},
        {"text":"TOTAL PAYABLE", "box":[[240,90],[330,90],[330,105],[240,105]]},
    ]
    result = arithmetic_consistency(lines, "30.00")
    assert result["status"] == "inconsistent"
    assert result["failed_checks"] >= 1


def test_phone_pairing_sends_photos_only_with_a_live_code(tmp_path: Path, monkeypatch) -> None:
    import io
    import dsdnet_receipt.web_app as web_app
    monkeypatch.setattr(web_app, "lan_addresses", lambda: ["192.168.1.20"])
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    desk = app.test_client()
    token = desk.get("/api/auth").get_json()["csrf_token"]
    token = desk.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    pairing = desk.post("/api/phone/pair", headers={"X-Phone-Token": "", "X-CSRF-Token": token}).get_json()
    assert pairing["url"].startswith("http://192.168.1.20:") and "/phone#" in pairing["url"]
    assert pairing["qr_svg"].startswith("<svg") and pairing["expires_in"] > 0
    code = pairing["url"].split("#", 1)[1]
    phone = app.test_client(); phone.environ_base["REMOTE_ADDR"] = "192.168.1.31"
    assert phone.get("/phone").status_code == 200
    # the phone has no session: nothing but the paired endpoints is open to it
    assert phone.get("/api/dashboard").status_code == 401
    assert phone.post("/api/workspace", headers={"X-CSRF-Token": phone.get("/api/auth").get_json()["csrf_token"]}).status_code == 403
    assert phone.get("/api/phone/check", headers={"X-Phone-Token": "wrong"}).status_code == 401
    assert phone.get("/api/phone/check", headers={"X-Phone-Token": code}).status_code == 200
    bad = {"receipt": (io.BytesIO(b"not an image"), "receipt.jpg")}
    response = phone.post("/api/phone/upload", data=bad, headers={"X-Phone-Token": code}, content_type="multipart/form-data")
    assert response.status_code == 400  # reached processing without a browser session or CSRF token
    assert desk.get("/api/phone/pair").get_json()["pairing"]["failed"] == 1
    desk.delete("/api/phone/pair", headers={"X-CSRF-Token": token})
    assert phone.get("/api/phone/check", headers={"X-Phone-Token": code}).status_code == 401


def test_welcome_is_shown_once(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    opened = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()
    assert opened["user"]["welcome_seen"] is False
    client.post("/api/welcome", headers={"X-CSRF-Token": opened["csrf_token"]})
    assert client.get("/api/auth").get_json()["user"]["welcome_seen"] is True
