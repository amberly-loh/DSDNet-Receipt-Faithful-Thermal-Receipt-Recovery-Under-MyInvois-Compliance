from __future__ import annotations

from pathlib import Path

from dsdnet_receipt.web_app import TERMS_VERSION, create_app

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"
ROOT = Path(__file__).resolve().parents[1]


def _workspace(tmp_path: Path):
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    opened = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()
    return app, client, opened["csrf_token"], opened["user"]


def test_terms_must_be_accepted_once_and_are_audited(tmp_path: Path) -> None:
    app, client, token, user = _workspace(tmp_path)
    assert user["terms_accepted"] is False
    stale = client.post("/api/terms", json={"version": "0.9"}, headers={"X-CSRF-Token": token})
    assert stale.status_code == 409
    accepted = client.post("/api/terms", json={"version": TERMS_VERSION}, headers={"X-CSRF-Token": token})
    assert accepted.status_code == 200
    assert client.get("/api/auth").get_json()["user"]["terms_accepted"] is True
    store = app.extensions["receipt_store"]
    assert store.app_setting(f"terms_accepted:{user['id']}").startswith(f"{TERMS_VERSION}|")


def test_terms_require_a_session_and_csrf(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    assert client.post("/api/terms", json={"version": TERMS_VERSION}, headers={"X-CSRF-Token": token}).status_code == 401
    _, client, _, _ = _workspace(tmp_path / "other")
    assert client.post("/api/terms", json={"version": TERMS_VERSION}).status_code in {400, 403}


def test_registration_records_terms_acceptance(tmp_path: Path) -> None:
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    response = client.post("/api/register", json={"name": "Test User", "email": "test@valid-domain.my", "password": "long-test-password-1",
                                                  "password_confirmation": "long-test-password-1", "policy_accepted": True},
                           headers={"X-CSRF-Token": token})
    assert response.status_code == 201
    assert client.get("/api/auth").get_json()["user"]["terms_accepted"] is True


def test_legal_text_is_bilingual_and_states_key_limits() -> None:
    html = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="legalEn"' in html and 'id="legalMs"' in html and 'id="legalOss"' in html
    assert "Notis Privasi" in html and "Privacy Notice" in html
    assert "not affiliated with or endorsed by LHDN" in html
    assert "Income Tax Act 1967" in html and "Personal Data Protection Act 2010" in html
    assert "ResitTrace" not in html and "<title>ClearReceipt</title>" in html
    script = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "confirm(" not in script.replace("askConfirm(", "")
