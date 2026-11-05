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
from copy import copy
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urlencode
from xml.sax.saxutils import escape

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from openpyxl import Workbook
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import load_config, write_json
from .fields import extract_fields, infer_receipt_tags, route_fields
from .images import write_image
from .language_assistance import suggestions_for_lines
from .ocr import PaddleOCRAdapter, deduplicate_lines
from .preprocessing import preprocess
from .restoration import RestorationEngine
from .semantic import LINE_ITEM_KEYS, extract_accounting_fields, extract_line_items
from .storage import ReceiptStore, utc_now


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
BLOCKED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "mailinator.com", "guerrillamail.com", "tempmail.com"}
COMMON_PASSWORDS = {
    "passwordpassword", "password123456", "qwertyuiop12345", "123456789012345",
    "letmeinletmein123", "adminadminadmin", "receipt-recovery", "thermalreceipt",
}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/bmp", "image/tiff"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def _pdf_fonts() -> tuple[str, str]:
    """Use a Unicode-capable Windows font when available."""
    candidates = [
        (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/msyhbd.ttc"), "AuditYaHei"),
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf"), "AuditArial"),
    ]
    for regular, bold, family in candidates:
        if regular.is_file() and bold.is_file():
            try:
                if family not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont(family, str(regular)))
                if family + "Bold" not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont(family + "Bold", str(bold)))
                return family, family + "Bold"
            except Exception:
                continue
    return "Helvetica", "Helvetica-Bold"


def build_audit_pdf(receipt: dict, reviewer: dict, integrity: dict, events: list[dict]) -> io.BytesIO:
    """Create a fixed, human-readable audit report; JSON remains the technical record."""
    regular, bold = _pdf_fonts()
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Receipt audit report - {receipt['public_id']}", author="Receipt Recovery Audit Application")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("AuditTitle", parent=styles["Title"], fontName=bold, fontSize=19, leading=23, textColor=colors.HexColor("#103C36"), spaceAfter=3 * mm)
    heading = ParagraphStyle("AuditHeading", parent=styles["Heading2"], fontName=bold, fontSize=11, leading=14, textColor=colors.HexColor("#176B5D"), spaceBefore=4 * mm, spaceAfter=2 * mm)
    body = ParagraphStyle("AuditBody", parent=styles["BodyText"], fontName=regular, fontSize=8.5, leading=11, textColor=colors.HexColor("#263633"))
    small = ParagraphStyle("AuditSmall", parent=body, fontSize=7, leading=9, textColor=colors.HexColor("#526864"))
    table_header = ParagraphStyle("AuditTableHeader", parent=small, fontName=bold, textColor=colors.white)
    right = ParagraphStyle("AuditRight", parent=small, alignment=TA_RIGHT)
    center = ParagraphStyle("AuditCenter", parent=body, alignment=TA_CENTER)

    def p(value, style=body):
        return Paragraph(escape(str(value if value not in (None, "") else "Not confirmed")), style)

    def footer(canvas, document):
        canvas.saveState(); canvas.setFont(regular, 7); canvas.setFillColor(colors.HexColor("#61736F"))
        canvas.drawString(16 * mm, 9 * mm, f"Receipt audit report - {receipt['public_id']}")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {document.page}"); canvas.restoreState()

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    primary_complete = all(receipt.get(key) for key in ("merchant", "transaction_date", "receipt_number", "total_amount"))
    status = status_label_for_export(receipt.get("status"))
    integrity_label = {"verified": "Verified", "failed": "FAILED", "legacy_unverified": "Not available for legacy record"}.get(integrity.get("status"), "Not available")
    lines = deduplicate_lines(receipt.get("audit", {}).get("ocr_lines", []))
    arithmetic = arithmetic_consistency(lines, receipt.get("total_amount"))
    items = receipt.get("line_items") or []
    if not items:
        keys = ("item", "description", "quantity", "unit", "unit_price", "discount", "amount", "review_note")
        items = [dict(zip(keys, row)) for row in accounting_line_items(lines)]
    accounting = receipt.get("accounting_fields") or extract_accounting_fields(
        receipt.get("ocr_text", ""), lines,
        {"merchant": receipt.get("merchant"), "transaction_date": receipt.get("transaction_date"),
         "receipt_invoice_number": receipt.get("receipt_number"), "total_amount": receipt.get("total_amount")},
    )
    if status != "Reviewed" and primary_complete and items:
        status = "Ready to confirm"

    story = [Paragraph("Receipt Audit Report", title),
             Table([[p("Record ID", small), p(receipt["public_id"], small), p("Generated", small), p(generated_at, right)]], colWidths=[24*mm, 58*mm, 22*mm, 72*mm],
                   style=TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#E9F4F1")),("BOX",(0,0),(-1,-1),.5,colors.HexColor("#B8D4CE")),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])),
             Paragraph("Audit conclusion", heading)]
    missing = [label for key, label in (("merchant","merchant"),("transaction_date","transaction date"),("receipt_number","receipt / invoice number"),("total_amount","total amount")) if not receipt.get(key)]
    conclusion = ("This record was reviewed and its key fields are complete." if status == "Reviewed" and not missing else "This record still requires human review" + (f"; missing: {', '.join(missing)}." if missing else "."))
    story.append(Table([[p("Review status", small), p(status), p("File integrity", small), p(integrity_label)], [p("Arithmetic check", small), p(arithmetic["status"].replace("_", " ").title()), p("Reviewer", small), p(reviewer.get("name") or reviewer.get("email"))], [p("Conclusion", small), Paragraph(escape(conclusion), body), "", ""]], colWidths=[30*mm,58*mm,30*mm,58*mm], style=TableStyle([("GRID",(0,0),(-1,1),.4,colors.HexColor("#CBD8D5")),("SPAN",(1,2),(-1,2)),("BOX",(0,2),(-1,2),.4,colors.HexColor("#CBD8D5")),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F6F5")),("BACKGROUND",(2,0),(2,1),colors.HexColor("#F1F6F5")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])))
    story += [Paragraph("Verified receipt details", heading), Table([[p("Merchant", small), p(receipt.get("merchant")), p("Transaction date", small), p(receipt.get("transaction_date"))], [p("Receipt / invoice no.", small), p(receipt.get("receipt_number")), p("Total amount", small), p("RM " + str(receipt.get("total_amount")) if receipt.get("total_amount") else None)], [p("Tags", small), p(", ".join(receipt.get("tags", [])) or None), p("Source file", small), p(receipt.get("source_name"))], [p("Reviewer notes", small), p(receipt.get("notes")), p("Last updated", small), p(receipt.get("updated_at"))]], colWidths=[30*mm,58*mm,30*mm,58*mm], style=TableStyle([("GRID",(0,0),(-1,-1),.4,colors.HexColor("#CBD8D5")),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F6F5")),("BACKGROUND",(2,0),(2,-1),colors.HexColor("#F1F6F5")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])),
              Paragraph("Accounting and MyInvois supporting evidence", heading),
              Table([[p("Document type", small), p(accounting.get("document_type")), p("Currency", small), p(accounting.get("currency"))],
                     [p("Supplier registration", small), p(accounting.get("supplier_registration_number")), p("Supplier tax no.", small), p(accounting.get("supplier_tax_number"))],
                     [p("Supplier address", small), p(accounting.get("supplier_address")), p("Supplier phone / email", small), p(" / ".join(filter(None, (accounting.get("supplier_phone"), accounting.get("supplier_email")))) or None)],
                     [p("Payment method", small), p(accounting.get("payment_method")), p("Subtotal / tax / discount", small), p(" / ".join(str(accounting.get(key)) for key in ("subtotal","tax_amount","discount") if accounting.get(key)) or None)]],
                    colWidths=[30*mm,58*mm,30*mm,58*mm], style=TableStyle([("GRID",(0,0),(-1,-1),.4,colors.HexColor("#CBD8D5")),("BACKGROUND",(0,0),(0,-1),colors.HexColor("#F1F6F5")),("BACKGROUND",(2,0),(2,-1),colors.HexColor("#F1F6F5")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])),
              Paragraph("Receipt evidence", heading)]
    image_cells = []
    for label, path in (("Original audit source", receipt.get("input_path")), ("Clean document view", receipt.get("restored_path"))):
        try:
            from PIL import Image as PilImage
            with PilImage.open(path) as im: width, height = im.size
            max_w, max_h = 84 * mm, 108 * mm; scale = min(max_w / width, max_h / height)
            image_cells.append([PdfImage(path, width=width*scale, height=height*scale), p(label, center)])
        except Exception:
            image_cells.append([p("Image unavailable in this test record", center), p(label, center)])
    story.append(Table([[image_cells[0][0], image_cells[1][0]], [image_cells[0][1], image_cells[1][1]]], colWidths=[88*mm,88*mm], style=TableStyle([("BOX",(0,0),(-1,0),.4,colors.HexColor("#CBD8D5")),("INNERGRID",(0,0),(-1,0),.4,colors.HexColor("#CBD8D5")),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])))
    story += [PageBreak(), Paragraph("Accounting evidence", title), Paragraph("Line items", heading)]
    item_data = [[p(x, table_header) for x in ("Item", "Description", "Qty", "Unit", "Unit price", "Discount", "Amount", "Review note")]]
    for row in items: item_data.append([p(row.get(key), small) for key in ("item","description","quantity","unit","unit_price","discount","amount","review_note")])
    if len(item_data) == 1: item_data.append([p("-",small), p("No line items confirmed",small), p("-",small), p("-",small), p("-",small), p("-",small), p("-",small), p("Verify the receipt",small)])
    story.append(Table(item_data, repeatRows=1, colWidths=[10*mm,45*mm,13*mm,13*mm,20*mm,17*mm,19*mm,39*mm], style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#176B5D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#C7D5D2")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)])))
    story += [Paragraph("Validation findings", heading), Table([[p("Check", table_header), p("Result", table_header), p("Meaning", table_header)], [p("Key fields", small), p("Complete" if not missing else "Incomplete", small), p("All four primary fields are present." if not missing else "Human input required for: " + ", ".join(missing), small)], [p("Line arithmetic", small), p(arithmetic["status"].replace("_"," ").title(), small), p(f"{arithmetic.get('checks_run', 0)} calculation(s) checked; {arithmetic.get('failed_checks', 0)} failed.", small)], [p("Stored files", small), p(integrity_label, small), p("Hashes confirm stored evidence is unchanged." if integrity.get("status") == "verified" else "Do not rely on the recovered view without checking the original.", small)]], colWidths=[36*mm,38*mm,102*mm], style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#176B5D")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#C7D5D2")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)])), Paragraph("Audit trail", heading)]
    event_data = [[p("Time", table_header), p("Action", table_header), p("Actor", table_header), p("Outcome", table_header)]]
    for event in sorted(events, key=lambda item: item.get("occurred_at", ""))[-12:]: event_data.append([p(event.get("occurred_at"), small), p(event.get("action", "").replace("_", " "), small), p(event.get("actor_name") or "System", small), p(event.get("outcome"), small)])
    if len(event_data) == 1: event_data.append([p(receipt.get("created_at"),small),p("receipt record created",small),p(reviewer.get("name"),small),p("success",small)])
    story.append(Table(event_data, repeatRows=1, colWidths=[48*mm,66*mm,42*mm,20*mm], style=TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#176B5D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#C7D5D2")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)])))
    story += [Spacer(1, 6*mm), Paragraph("Important: this report documents receipt recovery and human review. It does not replace the original receipt, constitute an independent audit opinion, or submit an e-Invoice to MyInvois.", small)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer); output.seek(0)
    return output


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


def status_label_for_export(status: str | None) -> str:
    return {"needs_review": "Needs review", "reviewed": "Reviewed", "archived": "Archived"}.get(status or "", status or "Unknown")


def safe_excel_value(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file(): return None
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _receipt_integrity(receipt: dict) -> dict:
    audit = receipt.get("audit", {})
    expected = audit.get("stored_file_sha256", {})
    if not expected:
        return {"status": "legacy_unverified", "message": "This record predates stored-file integrity hashes."}
    root = Path(receipt.get("input_path", "")).parent
    paths = {
        "original": Path(receipt.get("input_path", "")),
        "recovered": Path(receipt.get("restored_path", "")),
        "source_upload": root / "source_upload.bin",
        "rectified_input": root / "rectified_input.png",
        "faithful_content": root / "faithful_content.png",
        "faithful_reference": root / "faithful_reference.png",
    }
    # Compare every declared evidence file with the corresponding stored file.
    # The earlier implementation loaded only three files, so a valid
    # faithful_content hash was incorrectly reported as a failure.
    matches = {key: bool(paths.get(key) and paths[key].is_file()) and _file_sha256(paths[key]) == value
               for key, value in expected.items() if key in paths}
    return {"status": "verified" if matches and all(matches.values()) else "failed", "files": matches}


def _ocr_center(line: dict) -> tuple[float, float]:
    try:
        points = line.get("box", [])
        return sum(float(p[0]) for p in points) / len(points), sum(float(p[1]) for p in points) / len(points)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0, 0.0


def accounting_line_items(lines: list[dict]) -> list[list]:
    """Compatibility wrapper around the layout-aware semantic extractor."""
    return [[row.get(key, "") for key in LINE_ITEM_KEYS] for row in extract_line_items(lines)]


def arithmetic_consistency(lines: list[dict], total_amount: str | None) -> dict:
    """Check OCR arithmetic without inventing missing quantities or amounts."""
    rows = accounting_line_items(lines)
    checks = []
    for row in rows:
        quantity, unit_price, discount, amount = row[2], row[4], row[5], row[6]
        if not quantity or not unit_price or not amount: continue
        try:
            expected = Decimal(str(quantity)) * Decimal(str(unit_price))
            if discount: expected -= Decimal(str(discount))
            observed = Decimal(str(amount))
        except InvalidOperation:
            continue
        checks.append({"description": row[1], "expected": f"{expected:.2f}", "observed": f"{observed:.2f}", "passed": abs(expected - observed) <= Decimal("0.02")})
    total_check = None
    amounts = []
    for row in rows:
        try:
            if row[6]: amounts.append(Decimal(str(row[6])))
        except InvalidOperation: pass
    if total_amount and amounts:
        try:
            observed_total = Decimal(str(total_amount)); expected_total = sum(amounts, Decimal("0"))
            total_check = {"expected": f"{expected_total:.2f}", "observed": f"{observed_total:.2f}", "passed": abs(expected_total - observed_total) <= Decimal("0.02")}
        except InvalidOperation: pass
    evaluated = len(checks) + (1 if total_check else 0)
    failures = sum(not check["passed"] for check in checks) + (1 if total_check and not total_check["passed"] else 0)
    status = "not_evaluable" if not evaluated else "inconsistent" if failures else "consistent"
    return {"status": status, "evaluated_checks": evaluated, "failed_checks": failures, "line_checks": checks, "total_check": total_check}


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
    isolated_worker: bool = False,
) -> Flask:
    root = Path(__file__).resolve().parents[2]
    app = Flask(__name__, template_folder=str(root / "web" / "templates"), static_folder=str(root / "web" / "static"))
    app.config.update(
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        SECRET_KEY=_secret_key(root),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("RECEIPT_HTTPS", "0") == "1",
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
        user = store.user_by_id(int(user_id)) if user_id else None
        if user:
            user["is_guest"] = user["email"].endswith("@guest.local.invalid")
            if user["is_guest"]: user["email"] = ""
        return user

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

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), geolocation=(), microphone=()"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        # The SPA shell and JavaScript must be updated together.  Keeping an
        # older cached DOM with newer JavaScript caused null-field errors after
        # schema/UI releases, so local app assets are deliberately not cached.
        response.headers["Cache-Control"] = "no-store"
        if app.config["SESSION_COOKIE_SECURE"]:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        chain = store.verify_event_chain()
        status = "ok" if chain["status"] == "verified" else "degraded"
        return jsonify({"status": status, "service": "receipt-recovery", "database": "ready", "audit_chain": chain["status"]}), (200 if status == "ok" else 503)

    @app.get("/api/auth")
    def auth_state():
        return jsonify({"user": current_user(), "csrf_token": session["csrf_token"]})

    @app.get("/api/auth/providers")
    def auth_providers():
        return jsonify({"google": {"enabled": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")), "login_url": "/auth/google"}})

    @app.post("/api/guest")
    def guest_login():
        # Guest workspaces are confined to local sessions. A public deployment
        # requires a separate privacy, abuse-prevention and lifecycle design.
        import ipaddress
        try: local = ipaddress.ip_address(request.remote_addr or "").is_loopback
        except ValueError: local = False
        if not local or os.getenv("RECEIPT_HOST", "127.0.0.1") not in {"127.0.0.1", "::1", "localhost"}:
            return jsonify({"error": "Guest mode is available only on this device."}), 403
        guest = store.create_guest()
        store.log_event(guest["id"], "account.guest_start", "user", guest["public_id"])
        session.clear(); session["user_id"] = guest["id"]; session["csrf_token"] = secrets.token_urlsafe(24)
        guest["is_guest"] = True; guest["email"] = ""
        return jsonify({"user": guest, "csrf_token": session["csrf_token"]}), 201

    @app.post("/api/register")
    def register():
        payload = request.get_json(silent=True) or {}
        name, email, password = str(payload.get("name", "")).strip(), str(payload.get("email", "")).strip().lower(), str(payload.get("password", ""))
        errors = {}
        confirmation = str(payload.get("password_confirmation", ""))
        if not 2 <= len(name) <= 80 or any(ord(ch) < 32 for ch in name): errors["name"] = "Enter your name using 2 to 80 characters."
        if not _valid_email(email): errors["email"] = "Enter a valid email address, such as name@example.my."
        if email.endswith("@guest.local.invalid"): errors["email"] = "This email domain is reserved."
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

    @app.post("/api/receipts/selection-summary")
    @require_auth
    def selected_receipt_summary():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict): return jsonify({"error": "Invalid selection."}), 400
        ids = payload.get("ids")
        if not isinstance(ids, list) or len(ids) > 500 or any(not isinstance(item, str) or len(item) > 100 for item in ids) or len(set(ids)) != len(ids):
            return jsonify({"error": "Select up to 500 distinct receipts."}), 400
        values = []
        for identifier in ids:
            record = store.get_receipt(current_user()["id"], identifier)
            if not record: return jsonify({"error": "A selected receipt is unavailable. Refresh history and try again."}), 404
            values.append(record)
        total = Decimal('0'); missing = 0; unreviewed = 0; hashes = set(); duplicate_images = 0
        for record in values:
            raw = record.get("total_amount")
            try:
                value = Decimal(str(raw)); valid = value.is_finite() and value >= 0 and value.as_tuple().exponent >= -2
            except (InvalidOperation, ValueError, TypeError): valid = False
            if valid: total += value
            else: missing += 1
            if record.get("status") != "reviewed": unreviewed += 1
            digest = record.get("content_sha256")
            if digest and digest in hashes: duplicate_images += 1
            if digest: hashes.add(digest)
        return jsonify({"count": len(values), "total": f"{total:.2f}", "missing_amounts": missing,
                        "not_reviewed": unreviewed, "duplicate_images": duplicate_images, "currency": "MYR assumed"})

    @app.get("/api/receipts/<receipt_id>")
    @require_auth
    def receipt_detail(receipt_id: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        # Keep the public API stable for records created by older builds that
        # exposed the database JSON column instead of the parsed object.
        if receipt and not isinstance(receipt.get("accounting_fields"), dict):
            raw_accounting = receipt.pop("accounting_fields_json", "{}")
            try:
                receipt["accounting_fields"] = json.loads(raw_accounting or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                receipt["accounting_fields"] = {}
        elif receipt:
            receipt.pop("accounting_fields_json", None)
        if receipt and receipt.get("audit", {}).get("ocr_lines"):
            receipt["audit"]["ocr_lines"] = deduplicate_lines(receipt["audit"]["ocr_lines"])
            # Re-run the current evidence-based parser for saved records made
            # by older builds. Only fill missing values, except when an old
            # receipt number is demonstrably a ROC/company-registration value.
            # This repairs legacy records without requiring another scan.
            reparsed = extract_fields(receipt.get("ocr_text", ""), receipt["audit"]["ocr_lines"])
            detected_date = reparsed.get("transaction_date", {}).get("value")
            detected_number = reparsed.get("receipt_invoice_number", {}).get("value")
            current_number = str(receipt.get("receipt_number") or "").strip()
            registration_selected = bool(current_number and any(
                re.search(
                    rf"(?i)\b(?:roc|company|co\.?\s*reg|registration|reg\.?)\b[^\n]*{re.escape(current_number)}",
                    str(line.get("text", "")),
                )
                for line in receipt["audit"]["ocr_lines"]
            ))
            repaired_values = {}
            if detected_date and not receipt.get("transaction_date"):
                repaired_values["transaction_date"] = detected_date
            if detected_number and (not current_number or registration_selected):
                repaired_values["receipt_number"] = detected_number
            if repaired_values:
                repaired_accounting = dict(receipt.get("accounting_fields") or {})
                if repaired_values.get("transaction_date"):
                    repaired_accounting["transaction_date"] = repaired_values["transaction_date"]
                if repaired_values.get("receipt_number"):
                    repaired_accounting["document_number"] = repaired_values["receipt_number"]
                repaired_values["accounting_fields"] = repaired_accounting
                receipt = store.update_receipt(current_user()["id"], receipt_id, repaired_values)
                receipt["audit"]["ocr_lines"] = deduplicate_lines(receipt.get("audit", {}).get("ocr_lines", []))
            receipt["audit"]["ocr_suggestions"] = suggestions_for_lines(receipt["audit"]["ocr_lines"])
            receipt["audit"]["arithmetic_consistency"] = arithmetic_consistency(receipt["audit"]["ocr_lines"], receipt.get("total_amount"))
            if not receipt.get("line_items"):
                receipt["line_items"] = extract_line_items(receipt["audit"]["ocr_lines"])
            if not receipt.get("accounting_fields"):
                receipt["accounting_fields"] = extract_accounting_fields(
                    receipt.get("ocr_text", ""), receipt["audit"]["ocr_lines"],
                    {"merchant": receipt.get("merchant"), "transaction_date": receipt.get("transaction_date"),
                     "receipt_invoice_number": receipt.get("receipt_number"), "total_amount": receipt.get("total_amount")},
                )
        if receipt:
            duplicates = []
            for candidate in store.list_receipts(current_user()["id"]):
                if candidate["public_id"] == receipt_id: continue
                exact_source = bool(receipt.get("content_sha256") and candidate.get("content_sha256") == receipt.get("content_sha256"))
                same_business_record = all(receipt.get(key) and receipt.get(key) == candidate.get(key) for key in ("merchant", "transaction_date", "receipt_number", "total_amount"))
                if exact_source or same_business_record:
                    duplicates.append({"public_id": candidate["public_id"], "exact_image": exact_source, "created_at": candidate["created_at"]})
            receipt["possible_duplicates"] = duplicates
        if receipt:
            receipt["integrity"] = _receipt_integrity(receipt)
            primary_complete = all(str(receipt.get(key) or "").strip() for key in ("merchant", "transaction_date", "receipt_number", "total_amount"))
            if receipt.get("status") == "reviewed":
                state, label = "reviewed", "Reviewed"
            elif primary_complete and receipt.get("line_items"):
                state, label = "ready_to_confirm", "Ready to confirm"
            else:
                state, label = "needs_attention", "Needs attention"
            receipt["review_readiness"] = {"state": state, "label": label, "primary_fields_complete": primary_complete,
                                              "line_items_detected": len(receipt.get("line_items") or []),
                                              "integrity": receipt["integrity"].get("status")}
        if receipt: store.log_event(current_user()["id"], "receipt.view", "receipt", receipt_id)
        return jsonify(receipt) if receipt else (jsonify({"error": "Receipt not found."}), 404)

    @app.patch("/api/receipts/<receipt_id>")
    @require_auth
    def update_receipt(receipt_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict): return jsonify({"error": "Invalid receipt details."}), 400
        if not isinstance(payload.get("status"), (str, type(None))) or payload.get("status") not in {None, "needs_review", "reviewed", "archived"}:
            return jsonify({"error": "Invalid review status."}), 400
        existing = store.get_receipt(current_user()["id"], receipt_id)
        if not existing: return jsonify({"error": "Receipt not found."}), 404
        proposed = {**existing, **payload}
        errors = {}
        for key, limit in (("merchant", 160), ("receipt_number", 100), ("transaction_date", 10), ("total_amount", 18), ("notes", 4000)):
            if key in payload and (not isinstance(payload[key], str) or len(payload[key]) > limit or any(ord(ch) < 32 and ch not in '\n\r\t' for ch in payload[key])):
                errors[key] = f"Enter valid text of at most {limit} characters."
        if payload.get("transaction_date") and "transaction_date" not in errors:
            try:
                value = payload["transaction_date"]
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value): raise ValueError
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError: errors["transaction_date"] = "Use a valid date in YYYY-MM-DD format."
        if payload.get("total_amount") and "total_amount" not in errors:
            value = payload["total_amount"]
            if not re.fullmatch(r"\d{1,12}(?:\.\d{1,2})?", value): errors["total_amount"] = "Enter a non-negative amount with up to two decimal places."
        if "tags" in payload and (not isinstance(payload["tags"], list) or len(payload["tags"]) > 30 or any(not isinstance(tag, str) or len(tag) > 60 for tag in payload["tags"])):
            errors["tags"] = "Use up to 30 tags of at most 60 characters each."
        if payload.get("status") == "reviewed":
            missing = [key for key in ("merchant", "transaction_date", "total_amount") if not str(proposed.get(key) or "").strip()]
            if missing: errors["status"] = "Complete merchant, date and total before marking this receipt reviewed."
        if "line_items" in payload:
            if not isinstance(payload["line_items"], list) or len(payload["line_items"]) > 200:
                errors["line_items"] = "Line items must be a list with no more than 200 rows."
            else:
                allowed_item_keys = {"item", "description", "quantity", "unit", "unit_price", "discount", "amount", "review_note"}
                for row in payload["line_items"]:
                    if not isinstance(row, dict) or any(not isinstance(value, str) or len(value) > 300 for value in row.values()):
                        errors["line_items"] = "Each line item must contain text values of at most 300 characters."; break
                    for key in ("unit_price", "discount", "amount"):
                        if row.get(key) and not re.fullmatch(r"\d{1,12}(?:\.\d{1,2})?", row[key]):
                            errors["line_items"] = "Line amounts must be non-negative, with up to two decimal places."
                    if row.get("quantity") and not re.fullmatch(r"\d{1,9}(?:\.\d{1,3})?", row["quantity"]):
                        errors["line_items"] = "Quantity must be non-negative, with up to three decimal places."
                if "line_items" not in errors:
                    payload["line_items"] = [{key: row.get(key, "") for key in allowed_item_keys} for row in payload["line_items"]]
        if "accounting_fields" in payload:
            permitted = {"supplier_name", "supplier_registration_number", "supplier_tax_number", "supplier_address",
                         "supplier_phone", "supplier_email", "document_type", "document_number", "transaction_date",
                         "currency", "payment_method", "subtotal", "discount", "tax_amount", "payable_amount", "line_item_count"}
            fields = payload["accounting_fields"]
            if not isinstance(fields, dict) or any(key not in permitted for key in fields):
                errors["accounting_fields"] = "Accounting evidence contains unsupported fields."
            elif any(value is not None and (not isinstance(value, (str, int)) or len(str(value)) > 500) for value in fields.values()):
                errors["accounting_fields"] = "Accounting evidence values must be at most 500 characters."
            else:
                payload["accounting_fields"] = {key: fields.get(key) for key in permitted}
        if errors: return jsonify({"error": "Correct the highlighted receipt details.", "fields": errors}), 400
        receipt = store.update_receipt(current_user()["id"], receipt_id, payload)
        if receipt: store.log_event(current_user()["id"], "receipt.review_update", "receipt", receipt_id, details={"status": receipt["status"]})
        return jsonify(receipt) if receipt else (jsonify({"error": "Receipt not found."}), 404)

    @app.post("/api/receipts/<receipt_id>/quality-assessment")
    @require_auth
    def quality_assessment(receipt_id: str):
        user = current_user()
        receipt = store.get_receipt(user["id"], receipt_id)
        if not receipt:
            return jsonify({"error": "Receipt not found."}), 404
        payload = request.get_json(silent=True) or {}
        receipt_types = {"printed", "thermal", "invoice", "photograph", "other"}
        conditions = {"clear", "moderate", "faded", "severe"}
        ratings = {"better", "same", "worse"}
        permitted_issues = {"missing_text", "false_strokes", "logo_damage", "layout_shift", "texture_remaining", "overcleaned"}
        errors = {}
        if payload.get("receipt_type") not in receipt_types: errors["receipt_type"] = "Choose a receipt type."
        if payload.get("source_condition") not in conditions: errors["source_condition"] = "Choose the original condition."
        if payload.get("faithful_rating") not in ratings: errors["faithful_rating"] = "Choose better, the same or worse."
        try:
            readability = int(payload.get("readability_score"))
            if readability not in range(1, 6): raise ValueError
        except (TypeError, ValueError):
            errors["readability_score"] = "Choose a readability score from 1 to 5."
            readability = 0
        issues = payload.get("issues", [])
        if not isinstance(issues, list) or any(issue not in permitted_issues for issue in issues):
            errors["issues"] = "One or more quality issues are invalid."
        if errors:
            return jsonify({"error": "Complete the receipt quality assessment.", "fields": errors}), 400
        assessment = {
            "receipt_type": payload["receipt_type"],
            "source_condition": payload["source_condition"],
            "faithful_rating": payload["faithful_rating"],
            "readability_score": readability,
            "issues": sorted(set(issues)),
            "notes": str(payload.get("notes") or "").strip()[:1000],
            "assessed_at": utc_now(),
            "assessor_user_id": user["public_id"],
            "checkpoint": receipt.get("audit", {}).get("checkpoint"),
        }
        audit = receipt.get("audit", {})
        audit["quality_assessment"] = assessment
        if not store.update_receipt_audit(user["id"], receipt_id, audit):
            return jsonify({"error": "The assessment could not be saved."}), 500
        audit_path = Path(receipt["restored_path"]).parent / "audit.json"
        write_json(audit_path, audit)
        store.log_event(user["id"], "receipt.quality_assessment", "receipt", receipt_id, details=assessment)
        return jsonify(assessment)

    @app.post("/api/receipts/<receipt_id>/reextract")
    @require_auth
    def reextract_receipt(receipt_id: str):
        user = current_user()
        existing = store.get_receipt(user["id"], receipt_id)
        if not existing: return jsonify({"error": "Receipt not found."}), 404
        lines = deduplicate_lines(existing.get("audit", {}).get("ocr_lines", []))
        if not lines: return jsonify({"error": "No stored scanner readings are available for this receipt."}), 400
        text = "\n".join(str(line.get("text", "")) for line in lines)
        extracted = extract_fields(text, lines)
        values = {
            "merchant": extracted["merchant"]["value"],
            "transaction_date": extracted["transaction_date"]["value"],
            "receipt_number": extracted["receipt_invoice_number"]["value"],
            "total_amount": extracted["total_amount"]["value"],
            "status": "needs_review",
        }
        semantic_items = extract_line_items(lines)
        values["line_items"] = semantic_items
        values["accounting_fields"] = extract_accounting_fields(text, lines, {
            "merchant": values["merchant"], "transaction_date": values["transaction_date"],
            "receipt_invoice_number": values["receipt_number"], "total_amount": values["total_amount"],
        })
        updated = store.update_receipt(user["id"], receipt_id, values)
        # Rebuild the reading copy too. This allows older faint thermal
        # receipts to benefit from the adaptive renderer without re-uploading.
        from .restoration import assess_receipt_degradation, make_adaptive_document_view, make_faithful_content_view, make_readability_view, overlay_supported_ocr_text, region_adaptive_source_fusion, select_faithful_recovery
        run_root = Path(existing["restored_path"]).parent
        source_path = run_root / "rectified_input.png"
        recovery_path = run_root / "ocr_input.png"
        source_image = cv2.imread(str(source_path if source_path.is_file() else existing["input_path"]), cv2.IMREAD_GRAYSCALE)
        model_path = run_root / "model_output_x8.png"
        recovery_image = cv2.imread(str(model_path if model_path.is_file() else recovery_path), cv2.IMREAD_GRAYSCALE)
        if source_image is not None and recovery_image is not None:
            degradation_profile = assess_receipt_degradation(source_image)
            fused_recovery, fusion_record = region_adaptive_source_fusion(source_image, recovery_image)
            faithfulness = select_faithful_recovery(source_image, fused_recovery)
            recovery_image = faithfulness.image
            mask_config = config.get("artifact_mask", {})
            native_record = {}
            registered_source = cv2.imread(str(run_root / 'registered_color.png'))
            if registered_source is not None and registered_source.shape[:2] != source_image.shape:
                registered_source = None
            faithful_content = make_faithful_content_view(
                source_image, recovery_image, lines,
                mask_config.get("checkpoint") if mask_config.get("enabled") else None,
                float(mask_config.get("threshold", 0.95)),
                degradation_profile,
                native_refinement=config.get('native_refinement'),
                refinement_audit=native_record,
                source_color=registered_source,
            )
            faithful_path = run_root / "faithful_content.png"
            write_image(faithful_path, faithful_content)
            reading_base = cv2.cvtColor(faithful_content, cv2.COLOR_BGR2GRAY) if faithful_content.ndim == 3 else faithful_content
            reading_copy, rendered_readings = overlay_supported_ocr_text(
                reading_base, lines, minimum_confidence=0.70
            )
            reading_path = run_root / "reading_copy.png"
            write_image(reading_path, reading_copy)
            display_mode = 'faithful_recovery'
            write_image(existing["restored_path"], faithful_content)
            audit = existing["audit"]
            audit["display_image"] = display_mode
            audit["display_component_policy"] = ({
                "text": "continuous grayscale source evidence with locally amplified measured strokes",
                "background": "locally normalised",
                "selection": "low-contrast thermal receipt",
                "synthetic_text_rendering": False,
            } if display_mode.startswith("faint_thermal") else {
                "text": "OCR-localised source pixels", "rules": "thin linear structures",
                "marks": "dark connected source components", "background": "white",
            })
            audit.setdefault("stored_file_sha256", {})["recovered"] = _file_sha256(existing["restored_path"])
            audit["stored_file_sha256"]["faithful_content"] = _file_sha256(faithful_path)
            audit["stored_file_sha256"]["reading_copy"] = _file_sha256(reading_path)
            audit["reading_copy"] = {
                "image": "reading_copy.png",
                "synthetic_text_rendering": True,
                "rendered_ocr_regions": rendered_readings,
                "minimum_confidence": 0.70,
                "role": "optional reconstructed reading; excluded from faithful evidence and restoration metrics",
            }
            audit["faithful_display_image"] = "faithful_content.png"
            audit["faithful_display_policy"] = "background-normalised image-derived content; no OCR glyph rendering"
            audit["faithfulness_gate"] = {
                "accepted": faithfulness.accepted,
                "reason": faithfulness.reason,
                "metrics": faithfulness.metrics,
                "fallback": None if faithfulness.accepted else "rectified_original",
            }
            audit["region_adaptive_fusion"] = fusion_record
            audit["degradation_profile"] = degradation_profile
            audit['native_refinement'] = native_record
            audit["semantic_extraction"] = {"method": "layout_aware_receipt_semantics_v2", "line_items": semantic_items,
                                              "accounting_fields": values["accounting_fields"]}
            reference = registered_source if registered_source is not None else source_image
            reference = cv2.resize(reference, (faithful_content.shape[1], faithful_content.shape[0]), interpolation=cv2.INTER_CUBIC)
            write_image(run_root / 'faithful_reference.png', reference)
            audit['stored_file_sha256']['faithful_reference'] = _file_sha256(run_root / 'faithful_reference.png')
            write_json(run_root / "audit.json", audit)
            store.update_receipt_audit(user["id"], receipt_id, audit)
        store.log_event(user["id"], "receipt.fields_reextracted", "receipt", receipt_id, details=values)
        return jsonify(store.get_receipt(user["id"], receipt_id) or updated)

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
        if not receipt or kind not in {"original", "restored", "faithful", "reading", "faithful_reference", "diagnostic"}:
            return jsonify({"error": "Image not found."}), 404
        store.log_event(current_user()["id"], "receipt.image_view", "receipt", receipt_id, details={"kind": kind})
        path = receipt["input_path"] if kind == "original" else receipt["restored_path"]
        if kind == "faithful":
            candidate = Path(receipt["restored_path"]).with_name("faithful_content.png")
            if not candidate.is_file():
                candidate = Path(receipt["restored_path"]).with_name("ocr_input.png")
            path = str(candidate if candidate.is_file() else Path(receipt["restored_path"]))
        if kind == "reading":
            root = Path(receipt["restored_path"]).parent
            candidate = root / "reading_copy.png"
            if not candidate.is_file():
                from .restoration import overlay_supported_ocr_text
                base_path = root / "faithful_content.png"
                base = cv2.imread(str(base_path if base_path.is_file() else Path(receipt["restored_path"])), cv2.IMREAD_GRAYSCALE)
                lines = (receipt.get("audit") or {}).get("ocr_lines") or []
                if base is not None and lines:
                    reading_copy, _ = overlay_supported_ocr_text(base, lines, minimum_confidence=0.70)
                    write_image(candidate, reading_copy)
            path = str(candidate)
        if kind == "diagnostic": path = str(Path(receipt["restored_path"]).with_name("diagnostic.png"))
        if kind == "faithful_reference":
            root = Path(receipt["restored_path"]).parent
            paths = [root / 'faithful_reference.png', root / 'registered_color.png', root / 'rectified_input.png', Path(receipt['input_path'])]
            path = str(next(item for item in paths if item.is_file()))
        if not Path(path).is_file(): return jsonify({"error": "Image not found."}), 404
        return send_file(path, mimetype="image/png", conditional=True)

    @app.get("/api/export.csv")
    @require_auth
    def export_csv():
        rows = store.list_receipts(current_user()["id"])
        store.log_event(current_user()["id"], "receipt.export_csv", "receipt_collection", details={"count": len(rows)})
        output = io.StringIO(); writer = csv.writer(output)
        writer.writerow(["receipt_id", "created_at", "status", "merchant", "transaction_date", "receipt_number", "total_amount", "tags", "notes"])
        for row in rows:
            writer.writerow([safe_excel_value(value) for value in [row["public_id"], row["created_at"], row["status"], row.get("merchant"), row.get("transaction_date"), row.get("receipt_number"), row.get("total_amount"), ", ".join(row["tags"]), row.get("notes")]])
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
            sheet.append([safe_excel_value(value) for value in [row["public_id"], row["created_at"], row["status"], row.get("merchant"), row.get("transaction_date"), row.get("receipt_number"), row.get("total_amount"), ", ".join(row["tags"]), row.get("notes"), row.get("restoration_status"), row.get("support_status"), row.get("retention_until")]])
            try:
                amount = Decimal(str(row.get("total_amount")))
                if amount.is_finite() and 0 <= amount < Decimal('1000000000000') and amount.as_tuple().exponent >= -2:
                    sheet.cell(sheet.max_row, 7).value = float(amount)
                    sheet.cell(sheet.max_row, 7).number_format = '0.00'
            except (InvalidOperation, ValueError, TypeError):
                pass
        for cell in sheet[1]:
            font = copy(cell.font); font.bold = True; cell.font = font
        sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
        for column in sheet.columns:
            width = min(42, max(12, max(len(str(cell.value or "")) for cell in column) + 2)); sheet.column_dimensions[column[0].column_letter].width = width
        output = io.BytesIO(); workbook.save(output); output.seek(0)
        store.log_event(current_user()["id"], "receipt.export_xlsx", "receipt_collection", details={"count": len(rows)})
        return send_file(output, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name="receipt-register.xlsx")

    @app.get("/api/quality-evaluation.csv")
    @require_auth
    def export_quality_evaluation():
        rows = []
        user_id = current_user()["id"]
        for summary in store.list_receipts(user_id):
            receipt = store.get_receipt(user_id, summary["public_id"])
            assessment = receipt.get("audit", {}).get("quality_assessment")
            if not assessment:
                continue
            rows.append({
                "receipt_id": receipt["public_id"], "source_name": receipt["source_name"],
                "receipt_type": assessment.get("receipt_type", ""),
                "source_condition": assessment.get("source_condition", ""),
                "faithful_rating": assessment.get("faithful_rating", ""),
                "readability_score": assessment.get("readability_score", ""),
                "issues": ";".join(assessment.get("issues", [])),
                "notes": assessment.get("notes", ""),
                "assessed_at": assessment.get("assessed_at", ""),
                "checkpoint": assessment.get("checkpoint", ""),
            })
        stream = io.StringIO()
        fields = ["receipt_id", "source_name", "receipt_type", "source_condition", "faithful_rating", "readability_score", "issues", "notes", "assessed_at", "checkpoint"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        store.log_event(current_user()["id"], "receipt.export_quality_evaluation", "receipt_collection", details={"assessments": len(rows)})
        return send_file(io.BytesIO(stream.getvalue().encode("utf-8-sig")), mimetype="text/csv", as_attachment=True, download_name="receipt-quality-evaluation.csv")

    @app.get("/api/receipts/<receipt_id>/audit.json")
    @require_auth
    def export_audit(receipt_id: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        store.log_event(current_user()["id"], "receipt.export_audit", "receipt", receipt_id)
        data = io.BytesIO(json.dumps(receipt["audit"], indent=2).encode("utf-8"))
        return send_file(data, mimetype="application/json", as_attachment=True, download_name=f"{receipt_id}-audit.json")

    @app.get("/api/receipts/<receipt_id>/audit.pdf")
    @require_auth
    def export_audit_pdf(receipt_id: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        events = [event for event in store.list_events(limit=500) if event.get("object_type") == "receipt" and event.get("object_id") == receipt_id]
        data = build_audit_pdf(receipt, current_user(), _receipt_integrity(receipt), events)
        store.log_event(current_user()["id"], "receipt.export_audit_pdf", "receipt", receipt_id)
        return send_file(data, mimetype="application/pdf", as_attachment=True, download_name=f"{receipt_id}-audit-report.pdf")

    @app.get("/api/receipts/<receipt_id>/ocr.xlsx")
    @require_auth
    def export_ocr(receipt_id: str):
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        lines = deduplicate_lines(receipt["audit"].get("ocr_lines", []))
        suggestion_by_line = {int(item["line_index"]): item for item in suggestions_for_lines(lines)}
        workbook = Workbook(); summary = workbook.active; summary.title = "Receipt Summary"
        summary.append(["Receipt audit and accounting record"]); title_font = copy(summary["A1"].font); title_font.bold = True; title_font.size = 14; summary["A1"].font = title_font
        summary.append([])
        summary_rows = [
            ("Merchant", receipt.get("merchant")), ("Transaction date", receipt.get("transaction_date")),
            ("Invoice / receipt number", receipt.get("receipt_number")), ("Total amount (RM)", receipt.get("total_amount")),
            ("Review status", status_label_for_export(receipt.get("status"))), ("Record ID", receipt.get("public_id")),
            ("Source file", receipt.get("source_name")), ("Processed at", receipt.get("created_at")),
            ("Reviewer notes", receipt.get("notes")), ("Tags", ", ".join(receipt.get("tags", []))),
            ("Arithmetic consistency", arithmetic_consistency(lines, receipt.get("total_amount"))["status"].replace("_", " ").title()),
        ]
        for label, value in summary_rows: summary.append([label, safe_excel_value(value or "Not confirmed")])
        summary.column_dimensions["A"].width = 28; summary.column_dimensions["B"].width = 72
        for row in summary.iter_rows(min_row=3, max_row=summary.max_row, min_col=1, max_col=1):
            font = copy(row[0].font); font.bold = True; row[0].font = font

        items = workbook.create_sheet("Line Items")
        items.append(["Item", "Description", "Quantity", "Unit", "Unit price (RM)", "Discount", "Amount (RM)", "Review note"])
        saved_items = receipt.get("line_items") or []
        export_items = [[row.get(key, "") for key in ("item", "description", "quantity", "unit", "unit_price", "discount", "amount", "review_note")] for row in saved_items] if saved_items else accounting_line_items(lines)
        for row in export_items: items.append([safe_excel_value(value) for value in row])
        if items.max_row == 1: items.append([None, "No line items were extracted reliably", None, None, None, None, None, "Check the receipt image"])
        for cell in items[1]:
            font = copy(cell.font); font.bold = True; cell.font = font
        items.freeze_panes = "A2"; items.auto_filter.ref = items.dimensions
        for column, width in {"A":10,"B":42,"C":12,"D":12,"E":18,"F":14,"G":16,"H":25}.items(): items.column_dimensions[column].width = width

        evidence = workbook.create_sheet("OCR Evidence")
        evidence.append(["Reading order", "Recognised text", "Suggested reading", "Suggestion status", "Confidence (%)", "Evidence source"])
        friendly_views = {"rectified_input":"Original receipt", "source_preserving_recovery":"Recovered receipt", "readability_view":"Clean reading copy"}
        for index, line in enumerate(lines, 1):
            confidence = line.get("confidence")
            suggestion = suggestion_by_line.get(index - 1)
            evidence.append([
                index, safe_excel_value(line.get("text", "")),
                safe_excel_value(suggestion["suggested"]) if suggestion else "",
                "Verify against original" if suggestion else "No language suggestion",
                round(float(confidence) * 100, 1) if confidence is not None else None,
                friendly_views.get(line.get("source_view"), "Receipt image"),
            ])
        for cell in evidence[1]:
            font = copy(cell.font); font.bold = True; cell.font = font
        evidence.freeze_panes = "A2"; evidence.auto_filter.ref = evidence.dimensions
        evidence.column_dimensions["A"].width = 15; evidence.column_dimensions["B"].width = 72
        evidence.column_dimensions["C"].width = 72; evidence.column_dimensions["D"].width = 26
        evidence.column_dimensions["E"].width = 18; evidence.column_dimensions["F"].width = 24

        coverage = workbook.create_sheet("MyInvois Coverage")
        coverage.append(["Important", "This workbook supports review and accounting. It is not a MyInvois submission file.", ""])
        coverage.append(["MyInvois field", "Current value", "Coverage status"])
        line_rows = export_items
        coverage_rows = [
            ("Supplier name", receipt.get("merchant"), "Captured" if receipt.get("merchant") else "Requires user input"),
            ("Supplier TIN", None, "Requires user input"), ("Supplier registration / ID", None, "Requires user input"),
            ("Supplier address and contact", None, "Requires user input"), ("Buyer name, TIN, ID, address and contact", None, "Requires user input"),
            ("e-Invoice code / number", receipt.get("receipt_number"), "Captured" if receipt.get("receipt_number") else "Requires user input"),
            ("Invoice date", receipt.get("transaction_date"), "Captured" if receipt.get("transaction_date") else "Requires user input"),
            ("Invoice time", None, "Requires user input"), ("Invoice currency code", "MYR", "Application default; verify"),
            ("Invoice line descriptions, quantities and prices", f"{len(line_rows)} candidate line(s)", "Review Line Items sheet" if line_rows else "Requires user input"),
            ("Product / service classification", None, "Requires user input"), ("Tax type, rate and amount", None, "Requires user input"),
            ("Tax-exclusive / tax-inclusive / payable totals", receipt.get("total_amount"), "Payable total captured; remaining totals require input"),
            ("Digital signature and submission identifiers", None, "Created only by the authorised MyInvois workflow"),
        ]
        for row in coverage_rows: coverage.append([safe_excel_value(value) for value in row])
        for cell in coverage[2]:
            font = copy(cell.font); font.bold = True; cell.font = font
        coverage.freeze_panes = "A3"; coverage.column_dimensions["A"].width = 48; coverage.column_dimensions["B"].width = 55; coverage.column_dimensions["C"].width = 42
        output = io.BytesIO(); workbook.save(output); output.seek(0)
        store.log_event(current_user()["id"], "receipt.export_ocr_xlsx", "receipt", receipt_id)
        return send_file(output, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name=f"{receipt_id}-audit-accounting.xlsx")

    # Native OCR predictors must stay on the same thread between requests.
    # A lock prevents overlap but still allows Waitress to switch threads.
    from concurrent.futures import ThreadPoolExecutor
    receipt_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="receipt-inference")
    isolated_inference = None
    if not testing and not isolated_worker:
        from .inference_worker import IsolatedReceiptWorker
        isolated_inference = IsolatedReceiptWorker(timeout=300)
        app.extensions['isolated_receipt_worker'] = isolated_inference

    def process_receipt(upload, user):
        if isolated_inference is not None:
            data = upload.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError('The image exceeds the 20 MB limit')
            return isolated_inference.run({'bytes':data,'filename':upload.filename,'mimetype':upload.mimetype,
                'user':user,'config':str(Path(config_path).resolve()),
                'database':str(Path(database_path or (Path(config['artifacts_root'])/'app_data'/'receipts.sqlite3')).resolve())})
        return receipt_worker.submit(_process_receipt, upload, user).result()

    def _process_receipt(upload, user):
        if upload.mimetype and upload.mimetype.casefold() not in ALLOWED_IMAGE_TYPES:
            raise ValueError(f"{upload.filename}: upload a JPEG, PNG, WebP, HEIC or HEIF receipt image")
        raw_bytes = upload.read()
        if not raw_bytes: raise ValueError(f"{upload.filename}: the uploaded file is empty")
        if len(raw_bytes) > MAX_UPLOAD_BYTES: raise ValueError(f"{upload.filename}: the image exceeds the 20 MB limit")
        raw = np.frombuffer(raw_bytes, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f"{upload.filename}: unreadable receipt image")
        if int(image.shape[0]) * int(image.shape[1]) > MAX_IMAGE_PIXELS:
            raise ValueError(f"{upload.filename}: image dimensions are too large")
        started = time.perf_counter()
        registered_color, preprocessing_record = preprocess(image, config, preserve_color=True)
        processed = cv2.cvtColor(registered_color, cv2.COLOR_BGR2GRAY) if registered_color.ndim == 3 else registered_color
        faithful_refinement = {}
        from .restoration import assess_receipt_degradation, make_adaptive_document_view, make_faithful_content_view, make_readability_view, overlay_supported_ocr_text, region_adaptive_source_fusion, select_faithful_recovery
        with inference_lock:
            restoration = restoration_engine.run(processed)
            degradation_profile = assess_receipt_degradation(processed)
            fused_recovery, fusion_record = region_adaptive_source_fusion(processed, restoration.image)
            faithfulness = select_faithful_recovery(processed, fused_recovery)
            selected_recovery = faithfulness.image
            readability = make_readability_view(processed, selected_recovery)
            ocr = ocr_engine.run_views([
                ("rectified_input", processed),
                ("source_preserving_recovery", selected_recovery),
                ("readability_view", readability),
            ])
            if not ocr.lines:
                raise ValueError(
                    "No recoverable receipt text was detected. Use a clearer image; "
                    "the app did not create a restoration from background noise."
                )
            mask_config = config.get("artifact_mask", {})
            faithful_content = make_faithful_content_view(
                processed, selected_recovery, ocr.lines,
                mask_config.get("checkpoint") if mask_config.get("enabled") else None,
                float(mask_config.get("threshold", 0.95)),
                degradation_profile,
                source_color=registered_color if registered_color.ndim == 3 else None,
                native_refinement=config.get('native_refinement'),
                refinement_audit=faithful_refinement,
            )
            reading_base = cv2.cvtColor(faithful_content, cv2.COLOR_BGR2GRAY) if faithful_content.ndim == 3 else faithful_content
            reading_copy, rendered_readings = overlay_supported_ocr_text(
                reading_base, ocr.lines, minimum_confidence=0.70
            )
            document_view = faithful_content
            display_mode = "faithful_recovery"
        fields = route_fields(extract_fields(ocr.text, ocr.lines), support_available=False, model_available=restoration.model_available)
        field_map = {field.name: field.value for field in fields}
        content_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        verified_memory = store.verified_field_suggestions(
            user["id"],
            content_sha256=content_sha256,
            merchant=field_map.get("merchant"),
            receipt_number=field_map.get("receipt_invoice_number"),
            total_amount=field_map.get("total_amount"),
        )
        if not field_map.get("transaction_date") and verified_memory.get("transaction_date"):
            field_map["transaction_date"] = verified_memory["transaction_date"]
            for field in fields:
                if field.name == "transaction_date":
                    field.value = verified_memory["transaction_date"]
                    field.mandatory_rule_passed = True
                    field.rule_status = "passed_from_verified_memory"
                    field.reason = "Recovered from the user's prior reviewed receipt with an exact evidence match."
                    break
        inferred_tags = infer_receipt_tags(ocr.text, field_map.get("merchant"))
        semantic_items = extract_line_items(ocr.lines)
        accounting_fields = extract_accounting_fields(ocr.text, ocr.lines, field_map)
        run_id = f"USER-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        audit_root = Path(config["artifacts_root"]) / "user_runs" / run_id
        input_path, restored_path, diagnostic_path = audit_root / "input.png", audit_root / "restored.png", audit_root / "diagnostic.png"
        source_upload_path, rectified_path = audit_root / "source_upload.bin", audit_root / "rectified_input.png"
        # The raw network output is retained separately for reproducibility,
        # but is not suitable as the default user view: it can amplify paper
        # grain and folds.  Show the conservative, non-binary reading view.
        display_restoration = document_view
        audit_root.mkdir(parents=True, exist_ok=True)
        source_upload_path.write_bytes(raw_bytes)
        write_image(input_path, image); write_image(rectified_path, processed); write_image(restored_path, display_restoration)
        write_image(audit_root / "registered_color.png", registered_color)
        write_image(audit_root / "ocr_input.png", selected_recovery)
        write_image(audit_root / "faithful_content.png", faithful_content)
        write_image(audit_root / "reading_copy.png", reading_copy)
        if faithful_refinement.get('applied'):
            reference = cv2.resize(registered_color, (faithful_content.shape[1], faithful_content.shape[0]), interpolation=cv2.INTER_CUBIC)
            write_image(audit_root / 'faithful_reference.png', reference)
        if restoration.model_output is not None: write_image(audit_root / "model_output_x8.png", restoration.model_output)
        if restoration.diagnostic_map is not None: write_image(diagnostic_path, restoration.diagnostic_map)
        runtime = time.perf_counter() - started
        policy = ({
            "text": "high-confidence OCR readings rendered over a source-derived document view",
            "background": "locally normalised", "selection": "low-contrast thermal receipt",
            "synthetic_text_rendering": True, "rendered_ocr_regions": rendered_readings,
            "role": "optional reading copy; excluded from restoration and evidence evaluation",
        } if display_mode == "ocr_assisted_reading_copy" else {
            "text": "OCR-localised source pixels", "rules": "thin linear structures",
            "marks": "dark connected source components", "background": "white",
            "synthetic_text_rendering": False,
            "role": "source-derived reading view; excluded from formal restoration metrics",
        })
        audit = {"run_id": run_id, "source": upload.filename, "content_sha256": hashlib.sha256(raw_bytes).hexdigest(), "raw_source_file": "source_upload.bin", "rectified_input_file": "rectified_input.png", "stored_file_sha256": {"source_upload": _file_sha256(source_upload_path), "original": _file_sha256(input_path), "recovered": _file_sha256(restored_path), "faithful_content": _file_sha256(audit_root / "faithful_content.png")}, "runtime_seconds": runtime, "restoration_status": restoration.status, "restoration_error": restoration.error, "checkpoint": restoration.checkpoint, "restoration_stages": restoration.stage_count, "degradation_profile": degradation_profile, "region_adaptive_fusion": fusion_record, "faithfulness_gate": {"accepted": faithfulness.accepted, "reason": faithfulness.reason, "metrics": faithfulness.metrics, "fallback": None if faithfulness.accepted else "rectified_original"}, "display_image": display_mode, "faithful_display_image": "faithful_content.png", "faithful_display_policy": "degradation-adaptive, region-level source-supported contrast; exact rectified original used when the faithfulness gate rejects recovery", "model_output_image": "model_output_x8.png" if restoration.model_output is not None else None, "source_preserving_recovery_image": "ocr_input.png", "ocr_input_images": ["rectified_input", "source_preserving_recovery", "readability_view"], "display_component_policy": policy, "restoration_diagnostics": restoration.diagnostics, "diagnostic_map": str(diagnostic_path) if restoration.diagnostic_map is not None else None, "ocr_status": ocr.status, "ocr_version": ocr.engine_version, "ocr_lines": ocr.lines, "ocr_suggestions": suggestions_for_lines(ocr.lines), "language_assistance_role": "post_recognition_review_only_not_used_for_fixed_ocr_evaluation", "semantic_extraction": {"method": "layout_aware_receipt_semantics_v2", "line_items": semantic_items, "accounting_fields": accounting_fields}, "arithmetic_consistency": arithmetic_consistency(ocr.lines, field_map.get("total_amount")), "support_status": "unavailable_for_natural_source", "fields": [field.as_dict() for field in fields], "preprocessing": preprocessing_record, "formal_evidence": False}
        audit["stored_file_sha256"]["reading_copy"] = _file_sha256(audit_root / "reading_copy.png")
        audit["reading_copy"] = {
            "image": "reading_copy.png",
            "synthetic_text_rendering": True,
            "rendered_ocr_regions": rendered_readings,
            "minimum_confidence": 0.70,
            "role": "optional reconstructed reading; excluded from faithful evidence and restoration metrics",
        }
        from .restoration import complete_document_preservation_mode
        faithful_mode = complete_document_preservation_mode(registered_color)
        if faithful_mode:
            audit["faithful_display_policy"] = faithful_mode
            audit["faithful_preprocessing"] = preprocessing_record
            audit["faithful_processing"] = {
                "method": "continuous source luminance enhancement",
                "model_pixels_used": False,
                "synthetic_text_rendering": False,
                "ocr_used_for_foreground_deletion": False,
                "color_retained": faithful_content.ndim == 3,
            }
        audit['native_refinement'] = faithful_refinement
        audit['verified_field_memory'] = verified_memory
        audit['inferred_tags'] = inferred_tags
        if faithful_refinement.get('applied'):
            audit['stored_file_sha256']['faithful_reference'] = _file_sha256(audit_root / 'faithful_reference.png')
        write_json(audit_root / "audit.json", audit)
        now = utc_now(); retention_days = int(config.get("app", {}).get("retention_days", 2555))
        retention_until = (datetime.now(timezone.utc) + timedelta(days=retention_days)).date().isoformat()
        record = store.create_receipt({
            "public_id": run_id, "user_id": user["id"], "source_name": upload.filename,
            "created_at": now, "updated_at": now, "status": "needs_review", "merchant": field_map.get("merchant"),
            "transaction_date": field_map.get("transaction_date"), "receipt_number": field_map.get("receipt_invoice_number"),
            "total_amount": field_map.get("total_amount"), "tags": json.dumps(inferred_tags), "notes": "", "ocr_text": ocr.text,
            "line_items_json": json.dumps(semantic_items), "accounting_fields_json": json.dumps(accounting_fields),
            "audit_json": json.dumps(audit), "input_path": str(input_path), "restored_path": str(restored_path),
            "runtime_seconds": runtime, "restoration_status": restoration.status, "support_status": "unavailable_for_natural_source",
            "retention_until": retention_until, "content_sha256": audit["content_sha256"],
        })
        store.log_event(user["id"], "receipt.process", "receipt", run_id, details={"source_name": upload.filename, "restoration_status": restoration.status})
        return {"run_id": run_id, "receipt": record, "original": _data_url(image), "restored": _data_url(display_restoration), "diagnostic": _data_url(restoration.diagnostic_map) if restoration.diagnostic_map is not None else None, "ocr_text": ocr.text, "fields": [field.as_dict() for field in fields], "restoration_status": restoration.status, "support_status": "unavailable_for_natural_source", "runtime_seconds": runtime, "audit": audit}

    app.extensions['receipt_process_inline'] = _process_receipt

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
        maximum = int(config.get("app", {}).get("max_bulk_files", 100))
        if len(uploads) > maximum: return jsonify({"error": f"Process at most {maximum} receipts in one batch."}), 400
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


def _has_meaningful_document_color(image: np.ndarray) -> bool:
    if image.ndim != 3:
        return False
    channels = image.astype(np.int16)
    chroma = channels.max(axis=2) - channels.min(axis=2)
    return float(np.percentile(chroma, 90)) >= 12.0 and float(np.mean(chroma >= 8)) >= 0.03


def _make_color_preserving_clean_view(image: np.ndarray) -> np.ndarray:
    """Normalize luminance gently while preserving measured document color."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    paper = cv2.GaussianBlur(lightness, (0, 0), 19.0)
    normalized = cv2.divide(lightness, np.maximum(paper, 1), scale=242)
    normalized = cv2.addWeighted(lightness, 0.42, normalized, 0.58, 0)
    restored = cv2.cvtColor(
        cv2.merge((normalized, a_channel, b_channel)), cv2.COLOR_LAB2BGR
    )
    blurred = cv2.GaussianBlur(restored, (0, 0), 0.65)
    return cv2.addWeighted(restored, 1.08, blurred, -0.08, 0)
