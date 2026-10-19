from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from openpyxl import Workbook

from .config import load_config, write_json
from .fields import extract_fields, route_fields
from .images import write_image
from .ocr import PaddleOCRAdapter
from .preprocessing import preprocess
from .restoration import RestorationEngine
from .storage import ReceiptStore, utc_now


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
BLOCKED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "mailinator.com", "guerrillamail.com", "tempmail.com"}
COMMON_PASSWORDS = {
    "passwordpassword", "password123456", "qwertyuiop12345", "123456789012345",
    "letmeinletmein123", "adminadminadmin", "receipt-recovery", "thermalreceipt",
}


def _valid_email(value: str) -> bool:
    email = unicodedata.normalize("NFKC", value).strip().lower()
    if len(email) > 254 or email.count("@") != 1 or any(ch.isspace() for ch in email):
        return False
    local, domain = email.rsplit("@", 1)
    if not local or len(local) > 64 or local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    try:
        domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    suffix = domain.rsplit(".", 1)[-1] if "." in domain else ""
    canonical = f"{local}@{domain}"
    return bool(EMAIL_PATTERN.fullmatch(canonical)) and len(suffix) >= 2 and domain not in BLOCKED_EMAIL_DOMAINS and not domain.endswith((".test", ".invalid", ".localhost"))


def _password_error(password: str, name: str, email: str) -> str | None:
    normalized = unicodedata.normalize("NFC", password)
    compact = normalized.casefold().replace(" ", "")
    if len(normalized) < 15:
        return "Use at least 15 characters. A memorable passphrase works well."
    if len(normalized) > 128:
        return "Use no more than 128 characters."
    contextual = {name.casefold().replace(" ", ""), email.split("@", 1)[0].casefold()}
    if compact in COMMON_PASSWORDS or compact in contextual or compact.isdigit() or len(set(compact)) < 5:
        return "Choose a less common password that is not based on your name or email."
    return None


def _data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Image preview could not be encoded")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _secret_key(root: Path) -> bytes:
    path = root / "artifacts" / "app_data" / ".session_key"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def create_app(
    config_path: str | Path,
    testing: bool = False,
    database_path: str | Path | None = None,
) -> Flask:
    root = Path(__file__).resolve().parents[2]
    app = Flask(__name__, template_folder=str(root / "web" / "templates"), static_folder=str(root / "web" / "static"))
    app.config.update(
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        SECRET_KEY=_secret_key(root),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
        TESTING=testing,
    )
    config, _ = load_config(config_path)
    store = ReceiptStore(database_path or (Path(config["artifacts_root"]) / "app_data" / "receipts.sqlite3"))
    app.extensions["receipt_store"] = store
    restoration_engine = RestorationEngine(config)
    ocr_engine = PaddleOCRAdapter(config)
    inference_lock = threading.Lock()

    def current_user():
        user_id = session.get("user_id")
        return store.user_by_id(int(user_id)) if user_id else None

    def require_auth(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                return jsonify({"error": "Sign in to continue."}), 401
            if user.get("status") != "active":
                session.clear()
                return jsonify({"error": "This account is awaiting approval or has been suspended."}), 403
            return fn(*args, **kwargs)
        return wrapped

    def require_admin(fn):
        @wraps(fn)
        @require_auth
        def wrapped(*args, **kwargs):
            if current_user().get("role") != "admin":
                return jsonify({"error": "Administrator access is required."}), 403
            return fn(*args, **kwargs)
        return wrapped

    @app.before_request
    def csrf_guard():
        session.setdefault("csrf_token", secrets.token_urlsafe(24))
        if request.method in {"POST", "PATCH", "DELETE"} and request.endpoint not in {"static"}:
            supplied = request.headers.get("X-CSRF-Token", "")
            if not secrets.compare_digest(supplied, session["csrf_token"]):
                return jsonify({"error": "Your session changed. Refresh the page and try again."}), 403

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "receipt-recovery", "database": "ready"})

    @app.get("/api/auth")
    def auth_state():
        return jsonify({"user": current_user(), "csrf_token": session["csrf_token"]})

    @app.get("/api/auth/providers")
    def auth_providers():
        return jsonify({"google": {"enabled": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")), "login_url": "/auth/google"}})

    @app.post("/api/register")
    def register():
        payload = request.get_json(silent=True) or {}
        name, email, password = str(payload.get("name", "")).strip(), str(payload.get("email", "")).strip().lower(), str(payload.get("password", ""))
        errors = {}
        confirmation = str(payload.get("password_confirmation", ""))
        if not 2 <= len(name) <= 80 or any(ord(ch) < 32 for ch in name): errors["name"] = "Enter your name using 2 to 80 characters."
        if not _valid_email(email): errors["email"] = "Enter a valid email address, such as name@example.my."
        password_error = _password_error(password, name, email)
        if password_error: errors["password"] = password_error
        if confirmation != password: errors["confirmation"] = "The passwords do not match."
        if payload.get("policy_accepted") is not True: errors["policy"] = "Accept the local data-handling notice."
        if errors:
            return jsonify({"error": "Check the highlighted fields.", "fields": errors}), 400
        if store.user_by_email(email):
            return jsonify({"error": "An account already exists for this email address.", "fields": {"email": "Use a different email or sign in instead."}}), 409
        try:
            user = store.create_user(uuid.uuid4().hex, name, email, generate_password_hash(unicodedata.normalize("NFC", password), method="scrypt"))
        except sqlite3.IntegrityError:
            return jsonify({"error": "An account already exists for this email address.", "fields": {"email": "Use a different email or sign in instead."}}), 409
        store.log_event(user["id"], "account.register", "user", user["public_id"], details={"status": user["status"], "policy_accepted": True})
        session.clear(); session["user_id"] = user["id"]; session["csrf_token"] = secrets.token_urlsafe(24); session.permanent = True
        return jsonify({"user": user, "csrf_token": session["csrf_token"]}), 201

    @app.get("/auth/google")
    def google_login():
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        if not client_id or not os.getenv("GOOGLE_CLIENT_SECRET"):
            return redirect("/?auth_error=google_not_configured")
        state = secrets.token_urlsafe(32); session["google_oauth_state"] = state
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or url_for("google_callback", _external=True)
        query = urlencode({"client_id": client_id, "redirect_uri": redirect_uri, "response_type": "code", "scope": "openid email profile", "state": state, "prompt": "select_account"})
        return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{query}")

    @app.get("/auth/google/callback")
    def google_callback():
        expected, supplied = session.pop("google_oauth_state", None), request.args.get("state")
        if not expected or not supplied or not secrets.compare_digest(expected, supplied):
            return redirect("/?auth_error=google_state")
        code = request.args.get("code")
        if not code: return redirect("/?auth_error=google_denied")
        redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or url_for("google_callback", _external=True)
        try:
            token_response = requests.post("https://oauth2.googleapis.com/token", data={"code": code, "client_id": os.environ["GOOGLE_CLIENT_ID"], "client_secret": os.environ["GOOGLE_CLIENT_SECRET"], "redirect_uri": redirect_uri, "grant_type": "authorization_code"}, timeout=10)
            token_response.raise_for_status(); access_token = token_response.json()["access_token"]
            profile_response = requests.get("https://openidconnect.googleapis.com/v1/userinfo", headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
            profile_response.raise_for_status(); profile = profile_response.json()
            if not profile.get("email_verified"): return redirect("/?auth_error=google_unverified")
            user = store.user_by_email(profile["email"])
            if not user:
                user = store.create_user(uuid.uuid4().hex, profile.get("name") or profile["email"].split("@")[0], profile["email"], generate_password_hash(secrets.token_urlsafe(48), method="scrypt"))
                store.log_event(user["id"], "account.register_google", "user", user["public_id"], details={"status": user["status"], "consent": "google_oauth_and_local_notice"})
            if user.get("status") == "suspended": return redirect("/?auth_error=account_suspended")
            store.record_login(user["id"], True); store.log_event(user["id"], "account.login_google", "user", user["public_id"])
            session.clear(); session["user_id"] = user["id"]; session["csrf_token"] = secrets.token_urlsafe(24); session.permanent = True
            return redirect("/#overview")
        except Exception:
            return redirect("/?auth_error=google_failed")

    @app.post("/api/login")
    def login():
        payload = request.get_json(silent=True) or {}
        user = store.user_by_email(str(payload.get("email", "")))
        if not user or not check_password_hash(user["password_hash"], str(payload.get("password", ""))):
            if user: store.record_login(user["id"], False)
            return jsonify({"error": "The email or password is incorrect."}), 401
        if user.get("status") != "active":
            store.log_event(user["id"], "account.login", "user", user["public_id"], outcome="denied", details={"status": user.get("status")})
            return jsonify({"error": "This account is awaiting administrator approval or has been suspended."}), 403
        store.record_login(user["id"], True)
        store.log_event(user["id"], "account.login", "user", user["public_id"])
        session.clear(); session["user_id"] = user["id"]; session["csrf_token"] = secrets.token_urlsafe(24); session.permanent = True
        public = {k: user[k] for k in ("id", "public_id", "name", "email", "created_at", "role", "status")}
        return jsonify({"user": public, "csrf_token": session["csrf_token"]})

    @app.post("/api/logout")
    def logout():
        session.clear()
        return jsonify({"ok": True})

    @app.get("/api/dashboard")
    @require_auth
    def dashboard():
        user = current_user()
        recent = store.list_receipts(user["id"])[:6]
        return jsonify({"stats": store.stats(user["id"]), "recent": recent})

    @app.get("/api/policy")
    def policy():
        return jsonify({
            "storage": "Receipt images, extracted fields and audit records are stored on this device.",
            "primary_store": "SQLite is the primary record; JSON and spreadsheet files are derived exports.",
            "access": "Users access their own records. Administrators can suspend accounts and inspect audit records.",
            "retention": "The configured default retention period is seven years; records should be deleted earlier when no longer required and no legal requirement applies.",
            "verification": "Google verifies Google identities. Password accounts use format, duplicate and password-safety checks; a deployable public version should also send an email-verification link.",
        })

    @app.get("/api/admin/overview")
    @require_admin
    def admin_overview():
        return jsonify({"stats": store.admin_stats(), "users": store.list_users(), "events": store.list_events(100)})

    @app.patch("/api/admin/users/<user_id>")
    @require_admin
    def admin_update_user(user_id: str):
        payload = request.get_json(silent=True) or {}
        actor = current_user()
        if user_id == actor["public_id"] and payload.get("status") == "suspended":
            return jsonify({"error": "You cannot suspend your own administrator account."}), 400
        updated = store.update_user(user_id, payload.get("status"), payload.get("role"))
        if not updated: return jsonify({"error": "Account not found or no valid change supplied."}), 404
        store.log_event(actor["id"], "admin.user_update", "user", user_id, details={"status": updated["status"], "role": updated["role"]})
        return jsonify(updated)

    @app.get("/api/admin/receipts")
    @require_admin
    def admin_receipts():
        store.log_event(current_user()["id"], "admin.receipt_list", "receipt")
        return jsonify({"receipts": store.all_receipts()})

    @app.get("/api/admin/events")
    @require_admin
    def admin_events():
        return jsonify({"events": store.list_events(500)})

    @app.get("/api/receipts")
    @require_auth
    def receipts():
        user = current_user()
        return jsonify({"receipts": store.list_receipts(user["id"], request.args.get("q", ""), request.args.get("status", "all"))})

    @app.get("/api/receipts/<receipt_id>")
    @require_auth
    def receipt_detail(receipt_id: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if receipt: store.log_event(current_user()["id"], "receipt.view", "receipt", receipt_id)
        return jsonify(receipt) if receipt else (jsonify({"error": "Receipt not found."}), 404)

    @app.patch("/api/receipts/<receipt_id>")
    @require_auth
    def update_receipt(receipt_id: str):
        payload = request.get_json(silent=True) or {}
        if payload.get("status") not in {None, "needs_review", "reviewed", "archived"}:
            return jsonify({"error": "Invalid review status."}), 400
        receipt = store.update_receipt(current_user()["id"], receipt_id, payload)
        if receipt: store.log_event(current_user()["id"], "receipt.review_update", "receipt", receipt_id, details={"status": receipt["status"]})
        return jsonify(receipt) if receipt else (jsonify({"error": "Receipt not found."}), 404)

    @app.delete("/api/receipts/<receipt_id>")
    @require_auth
    def delete_receipt(receipt_id: str):
        if not store.delete_receipt(current_user()["id"], receipt_id):
            return jsonify({"error": "Receipt not found."}), 404
        store.log_event(current_user()["id"], "receipt.delete_verified", "receipt", receipt_id)
        return jsonify({"ok": True})

    @app.get("/api/receipts/<receipt_id>/image/<kind>")
    @require_auth
    def receipt_image(receipt_id: str, kind: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt or kind not in {"original", "restored"}:
            return jsonify({"error": "Image not found."}), 404
        store.log_event(current_user()["id"], "receipt.image_view", "receipt", receipt_id, details={"kind": kind})
        return send_file(receipt["input_path"] if kind == "original" else receipt["restored_path"], mimetype="image/png", conditional=True)

    @app.get("/api/export.csv")
    @require_auth
    def export_csv():
        rows = store.list_receipts(current_user()["id"])
        store.log_event(current_user()["id"], "receipt.export_csv", "receipt_collection", details={"count": len(rows)})
        output = io.StringIO(); writer = csv.writer(output)
        writer.writerow(["receipt_id", "created_at", "status", "merchant", "transaction_date", "receipt_number", "total_amount", "tags", "notes"])
        for row in rows:
            writer.writerow([row["public_id"], row["created_at"], row["status"], row.get("merchant"), row.get("transaction_date"), row.get("receipt_number"), row.get("total_amount"), ", ".join(row["tags"]), row.get("notes")])
        data = io.BytesIO(output.getvalue().encode("utf-8-sig"))
        return send_file(data, mimetype="text/csv", as_attachment=True, download_name="receipt-history.csv")

    @app.get("/api/export.xlsx")
    @require_auth
    def export_xlsx():
        rows = store.list_receipts(current_user()["id"])
        workbook = Workbook(); sheet = workbook.active; sheet.title = "Receipts"
        headers = ["Receipt ID", "Processed at", "Review status", "Merchant", "Transaction date", "Receipt number", "Total amount", "Tags", "Notes", "Restoration status", "Support status", "Retention until"]
        sheet.append(headers)
        for row in rows:
            sheet.append([row["public_id"], row["created_at"], row["status"], row.get("merchant"), row.get("transaction_date"), row.get("receipt_number"), row.get("total_amount"), ", ".join(row["tags"]), row.get("notes"), row.get("restoration_status"), row.get("support_status"), row.get("retention_until")])
        for cell in sheet[1]: cell.font = cell.font.copy(bold=True)
        sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            width = min(42, max(12, max(len(str(cell.value or "")) for cell in column) + 2)); sheet.column_dimensions[column[0].column_letter].width = width
        output = io.BytesIO(); workbook.save(output); output.seek(0)
        store.log_event(current_user()["id"], "receipt.export_xlsx", "receipt_collection", details={"count": len(rows)})
        return send_file(output, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name="receipt-register.xlsx")

    @app.get("/api/receipts/<receipt_id>/audit.json")
    @require_auth
    def export_audit(receipt_id: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        store.log_event(current_user()["id"], "receipt.export_audit", "receipt", receipt_id)
        data = io.BytesIO(json.dumps(receipt["audit"], indent=2).encode("utf-8"))
        return send_file(data, mimetype="application/json", as_attachment=True, download_name=f"{receipt_id}-audit.json")

    def process_receipt(upload, user):
        raw_bytes = upload.read()
        raw = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f"{upload.filename}: unreadable receipt image")
        started = time.perf_counter(); processed, preprocessing_record = preprocess(image, config)
        with inference_lock:
            restoration = restoration_engine.run(processed); ocr = ocr_engine.run(restoration.image)
        fields = route_fields(extract_fields(ocr.text, ocr.lines), support_available=False, model_available=restoration.model_available)
        field_map = {field.name: field.value for field in fields}
        run_id = f"USER-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        audit_root = Path(config["artifacts_root"]) / "user_runs" / run_id
        input_path, restored_path = audit_root / "input.png", audit_root / "restored.png"
        write_image(input_path, processed); write_image(restored_path, restoration.image)
        runtime = time.perf_counter() - started
        audit = {"run_id": run_id, "source": upload.filename, "content_sha256": hashlib.sha256(raw_bytes).hexdigest(), "runtime_seconds": runtime, "restoration_status": restoration.status, "restoration_error": restoration.error, "checkpoint": restoration.checkpoint, "ocr_status": ocr.status, "ocr_version": ocr.engine_version, "support_status": "unavailable", "fields": [field.as_dict() for field in fields], "preprocessing": preprocessing_record, "formal_evidence": False}
        write_json(audit_root / "audit.json", audit)
        now = utc_now(); retention_days = int(config.get("app", {}).get("retention_days", 2555))
        retention_until = (datetime.now(timezone.utc) + timedelta(days=retention_days)).date().isoformat()
        record = store.create_receipt({
            "public_id": run_id, "user_id": user["id"], "source_name": upload.filename,
            "created_at": now, "updated_at": now, "status": "needs_review", "merchant": None,
            "transaction_date": field_map.get("transaction_date"), "receipt_number": field_map.get("receipt_invoice_number"),
            "total_amount": field_map.get("total_amount"), "tags": "[]", "notes": "", "ocr_text": ocr.text,
            "audit_json": json.dumps(audit), "input_path": str(input_path), "restored_path": str(restored_path),
            "runtime_seconds": runtime, "restoration_status": restoration.status, "support_status": "unavailable",
            "retention_until": retention_until, "content_sha256": audit["content_sha256"],
        })
        store.log_event(user["id"], "receipt.process", "receipt", run_id, details={"source_name": upload.filename, "restoration_status": restoration.status})
        return {"run_id": run_id, "receipt": record, "original": _data_url(processed), "restored": _data_url(restoration.image), "ocr_text": ocr.text, "fields": [field.as_dict() for field in fields], "restoration_status": restoration.status, "support_status": "unavailable", "runtime_seconds": runtime, "audit": audit}

    @app.post("/api/analyze")
    @require_auth
    def analyze():
        upload = request.files.get("receipt")
        if upload is None or not upload.filename:
            return jsonify({"error": "Choose a receipt image before continuing."}), 400
        try: return jsonify(process_receipt(upload, current_user()))
        except ValueError as error: return jsonify({"error": str(error)}), 400

    @app.post("/api/bulk-analyze")
    @require_auth
    def bulk_analyze():
        uploads = [item for item in request.files.getlist("receipts") if item.filename]
        if not uploads: return jsonify({"error": "Choose one or more receipt images."}), 400
        if len(uploads) > 25: return jsonify({"error": "Process at most 25 receipts in one batch."}), 400
        completed, failed = [], []
        for upload in uploads:
            try: completed.append(process_receipt(upload, current_user())["receipt"])
            except Exception as error: failed.append({"filename": upload.filename, "error": str(error)})
        store.log_event(current_user()["id"], "receipt.bulk_process", "receipt_collection", details={"requested": len(uploads), "completed": len(completed), "failed": len(failed)})
        return jsonify({"completed": completed, "failed": failed, "summary": {"requested": len(uploads), "completed": len(completed), "failed": len(failed)}})

    @app.errorhandler(413)
    def too_large(_error): return jsonify({"error": "The image is larger than 20 MB. Choose a smaller image."}), 413

    return app


def run(config_path: str | Path, port: int = 5055) -> None:
    create_app(config_path).run(host="127.0.0.1", port=port, debug=False, threaded=True)
