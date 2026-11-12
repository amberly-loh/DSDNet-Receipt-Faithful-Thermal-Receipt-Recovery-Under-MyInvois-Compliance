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
        json={"merchant": "Local Shop", "status": "reviewed", "tags": ["claim"], "line_items": [{"item":"1","description":"Office paper","quantity":"2","unit":"pack","unit_price":"5.00","discount":"0.00","amount":"10.00","review_note":"Verified"}], "accounting_fields": {"document_type":"Tax Invoice","currency":"MYR","supplier_registration_number":"206259-X","supplier_address":"No. 2, Jalan Jati Utama 2","supplier_phone":"03-60573046","payment_method":"Cash","subtotal":"12.50","tax_amount":"0.00"}},
        headers={"X-CSRF-Token": token},
    )
    assert update.status_code == 200
    assert update.get_json()["merchant"] == "Local Shop"
    assert update.get_json()["status"] == "reviewed"
    assert update.get_json()["line_items"][0]["description"] == "Office paper"
    assert update.get_json()["accounting_fields"]["document_type"] == "Tax Invoice"
    detail = client.get("/api/receipts/USER-TEST-001")
    assert detail.status_code == 200
    assert detail.get_json()["accounting_fields"]["supplier_registration_number"] == "206259-X"
    assert detail.get_json()["accounting_fields"]["currency"] == "MYR"
    assert client.get("/api/export.csv").status_code == 200
    assessment = client.post(
        "/api/receipts/USER-TEST-001/quality-assessment",
        json={"receipt_type":"thermal","source_condition":"moderate","faithful_rating":"better","readability_score":4,"issues":["texture_remaining"],"notes":"Fold remains near total"},
        headers={"X-CSRF-Token": token},
    )
    assert assessment.status_code == 200
    assert assessment.get_json()["faithful_rating"] == "better"
    evaluation = client.get("/api/quality-evaluation.csv")
    assert evaluation.status_code == 200
    assert b"thermal" in evaluation.data and b"better" in evaluation.data

    assert client.post("/api/logout", headers={"X-CSRF-Token": token}).status_code == 200
    assert client.get("/api/dashboard").status_code == 401


def test_verified_receipt_memory_requires_consensus_and_strong_match(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    store = app.extensions["receipt_store"]
    user = store.create_guest()
    now = utc_now()
    common = {
        "user_id": user["id"], "source_name": "receipt.jpg", "created_at": now,
        "updated_at": now, "status": "reviewed", "merchant": "CHOP HOCK CHUAN HIN",
        "receipt_number": "61036", "total_amount": "340.00", "tags": "[]",
        "notes": "", "ocr_text": "", "audit_json": "{}", "input_path": "a.png",
        "restored_path": "b.png", "runtime_seconds": 0, "restoration_status": "ok",
        "support_status": "unavailable", "content_sha256": "same-hash",
    }
    store.create_receipt({**common, "public_id": "VERIFIED-1", "transaction_date": "2024-03-19"})
    exact = store.verified_field_suggestions(
        user["id"], content_sha256="same-hash", merchant=None,
        receipt_number=None, total_amount=None,
    )
    assert exact["transaction_date"] == "2024-03-19"
    assert exact["method"] == "global_exact_source_sha256"
    reusable = store.verified_field_suggestions(
        user["id"], content_sha256="different", merchant="CHOP HOCK CHUAN HIN",
        receipt_number="61036", total_amount="340",
    )
    assert reusable["transaction_date"] == "2024-03-19"
    weak = store.verified_field_suggestions(
        user["id"], content_sha256="different", merchant="CHOP HOCK CHUAN HIN",
        receipt_number=None, total_amount="340",
    )
    assert weak["transaction_date"] is None


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


def test_receipt_view_has_only_faithful_original_and_compare(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'class="comparison improved-mode"' in html
    assert 'id="faithfulViewButton" class="active"' in html
    assert 'id="readingViewButton"' not in html
    assert 'Clean view' not in html
    assert 'Optional reading copy' not in html  # runtime label depends on the receipt
    assert 'id="compareViewButton"' in html
    assert 'id="originalViewButton"' in html
    assert 'id="qualityForm"' not in html
    assert 'Accounting and MyInvois evidence' in html
    assert '/api/quality-evaluation.csv' in html


def test_frontend_refreshes_stale_csrf_token_once() -> None:
    script = (Path(__file__).resolve().parents[1] / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "allowSessionRetry = true" in script
    assert "fetch('/api/auth', {cache: 'no-store'})" in script
    assert "return api(path, options, false)" in script


def test_frontend_supports_large_additive_batch_queue() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "web" / "static" / "app.js").read_text(encoding="utf-8")
    html = (root / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert ".slice(0, 100)" in script
    assert "100 receipt images" in html
    assert "timeoutMs: 330000" in script


def test_guest_selection_validation_and_isolation(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    guest_client = app.test_client()
    guest_response = guest_client.post('/api/guest', headers={'X-CSRF-Token': csrf(guest_client)})
    assert guest_response.status_code == 201
    guest = guest_response.get_json()['user']
    token = guest_response.get_json()['csrf_token']
    assert guest['is_guest'] and guest['role'] == 'user'
    store = app.extensions['receipt_store']; now = utc_now()
    source = tmp_path / 'source.png'; source.write_bytes(b'x')
    for index, amount in enumerate(['0.10', '0.20', None]):
        store.create_receipt({'public_id': f'G-{index}', 'user_id': guest['id'], 'source_name': 'a.png', 'created_at': now, 'updated_at': now, 'status': 'needs_review', 'merchant': 'Shop', 'transaction_date': '2026-01-01', 'receipt_number': None, 'total_amount': amount, 'tags': '[]', 'notes': '', 'ocr_text': '', 'audit_json': '{}', 'input_path': str(source), 'restored_path': str(source), 'runtime_seconds': 0, 'restoration_status': 'ok', 'support_status': 'unavailable'})
    result = guest_client.post('/api/receipts/selection-summary', json={'ids': ['G-0', 'G-1', 'G-2']}, headers={'X-CSRF-Token': token})
    assert result.status_code == 200
    assert result.get_json()['total'] == '0.30'
    assert result.get_json()['missing_amounts'] == 1
    assert result.get_json()['not_reviewed'] == 3
    invalid = guest_client.patch('/api/receipts/G-0', json={'total_amount': '1.234'}, headers={'X-CSRF-Token': token})
    assert invalid.status_code == 400
    valid = guest_client.patch('/api/receipts/G-0', json={'status': 'reviewed'}, headers={'X-CSRF-Token': token})
    assert valid.status_code == 200
    other = app.test_client()
    registration = other.post('/api/register', json={'name': 'Owner User', 'email': 'owner@valid-domain.my', 'password': 'long-test-password-1', 'password_confirmation': 'long-test-password-1', 'policy_accepted': True}, headers={'X-CSRF-Token': csrf(other)})
    assert registration.status_code == 201 and registration.get_json()['user']['role'] == 'admin'
    other_token = registration.get_json()['csrf_token']
    assert other.post('/api/receipts/selection-summary', json={'ids': ['G-0']}, headers={'X-CSRF-Token': other_token}).status_code == 404
    assert guest_client.post('/api/logout', headers={'X-CSRF-Token': token}).status_code == 200
    assert guest_client.get('/api/dashboard').status_code == 401


def test_finance_report_combines_receipts_manual_expenses_and_export(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client(); login = client.post('/api/guest', headers={'X-CSRF-Token': csrf(client)})
    token = login.get_json()['csrf_token']; user = login.get_json()['user']; store = app.extensions['receipt_store']; now = utc_now()
    image = tmp_path / 'receipt.png'; image.write_bytes(b'x')
    store.create_receipt({'public_id':'FIN-R1','user_id':user['id'],'source_name':'fuel.png','created_at':now,'updated_at':now,'status':'reviewed','merchant':'PETRONAS','transaction_date':'2026-10-01','receipt_number':'F1','total_amount':'50.00','tags':'["fuel"]','notes':'','ocr_text':'','audit_json':'{}','input_path':str(image),'restored_path':str(image),'runtime_seconds':0,'restoration_status':'ok','support_status':'unavailable','accounting_fields_json':'{"currency":"MYR","payment_method":"Card"}'})
    created = client.post('/api/finance/expenses', json={'expense_date':'2026-10-02','supplier':'Office Shop','description':'Paper','expense_type':'office','amount':'12.50','currency':'MYR','tax_amount':'0.50'}, headers={'X-CSRF-Token':token})
    assert created.status_code == 201
    report = client.get('/api/finance?from=2026-10-01&to=2026-10-31').get_json()
    assert report['summary']['total'] == '62.50' and report['summary']['count'] == 2
    assert {row['code'] for row in report['by_type']} == {'fuel', 'office'}
    assert client.patch('/api/finance/receipts/FIN-R1', json={'expense_type':'travel'}, headers={'X-CSRF-Token':token}).status_code == 200
    assert client.get('/api/finance?category=travel').get_json()['summary']['total'] == '50.00'
    export = client.get('/api/finance/export.xlsx')
    assert export.status_code == 200 and export.headers['Content-Type'].startswith('application/vnd.openxmlformats')
    assert client.delete('/api/finance/expenses/' + created.get_json()['public_id'], headers={'X-CSRF-Token':token}).status_code == 200


def test_finance_expense_validation_and_frontend_module(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client(); login = client.post('/api/guest', headers={'X-CSRF-Token': csrf(client)})
    token = login.get_json()['csrf_token']
    invalid = client.post('/api/finance/expenses', json={'expense_date':'2026-02-31','supplier':'','amount':'-1','expense_type':'unknown'}, headers={'X-CSRF-Token':token})
    assert invalid.status_code == 400
    assert {'expense_date','supplier','amount','expense_type'} <= set(invalid.get_json()['fields'])
    html = client.get('/').get_data(as_text=True)
    assert 'data-route="finance"' in html and 'id="financePage"' in html
    assert '/api/finance/export.xlsx' in html
