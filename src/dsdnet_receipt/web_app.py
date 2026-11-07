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
from xml.sax.saxutils import escape

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file, session
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
from .receipt_context import reading_copy_lines
from .phone import PhonePairings, lan_addresses, qr_svg
from .storage import WORKSPACE_EMAIL, ReceiptStore, utc_now


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
BLOCKED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "mailinator.com", "guerrillamail.com", "tempmail.com"}
COMMON_PASSWORDS = {
    "passwordpassword", "password123456", "qwertyuiop12345", "123456789012345",
    "letmeinletmein123", "adminadminadmin", "receipt-recovery", "thermalreceipt",
}
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "image/bmp", "image/tiff"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

EXPENSE_TYPES = {
    "inventory": "Inventory / cost of goods", "office": "Office supplies", "fuel": "Fuel & transport",
    "meals": "Meals & entertainment", "utilities": "Utilities", "maintenance": "Repairs & maintenance",
    "travel": "Travel & accommodation", "professional": "Professional services", "rent": "Rent & premises",
    "tax": "Tax, licences & government fees", "medical": "Medical & welfare", "other": "Other expense",
}


def _money(value: Any) -> Decimal | None:
    try:
        amount = Decimal(str(value).replace(",", "").replace("RM", "").strip())
        return amount if amount.is_finite() and amount >= 0 and amount.as_tuple().exponent >= -2 else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def _expense_type(receipt: dict[str, Any]) -> str:
    accounting = receipt.get("accounting_fields") or {}
    explicit = str(accounting.get("expense_type") or "").strip().lower()
    if explicit in EXPENSE_TYPES:
        return explicit
    suggested = str(accounting.get("suggested_expense_type") or "").strip().lower()
    if suggested in EXPENSE_TYPES:
        return suggested  # recognised from the receipt's own words (brand, fuel grade, items)
    haystack = " ".join([
        str(receipt.get("merchant") or ""), " ".join(receipt.get("tags") or []),
        " ".join(str(line.get("description") or "") for line in receipt.get("line_items") or []),
    ]).lower()
    rules = {
        "fuel": ("petron", "shell", "bhpetrol", "caltex", "fuel", "diesel", "petrol", "toll", "grab"),
        "office": ("stationery", "paper", "printer", "office", "ink", "toner"),
        "meals": ("restaurant", "cafe", "coffee", "food", "mart", "mcd", "kfc"),
        "utilities": ("electric", "water", "telekom", "internet", "utility"),
        "maintenance": ("hardware", "d.i.y", "diy", "repair", "maintenance", "tools"),
        "travel": ("hotel", "airasia", "airlines", "flight", "booking"),
        "professional": ("consult", "legal", "accounting", "audit", "service fee"),
        "rent": ("rent", "rental", "premises"), "tax": ("licence", "license", "government", "assessment"),
        "medical": ("clinic", "pharmacy", "medical", "hospital"),
        "inventory": ("supplier", "trading", "wholesale", "stock", "inventory"),
    }
    for category, terms in rules.items():
        if any(term in haystack for term in terms):
            return category
    return "other"


def _finance_entries(store: ReceiptStore, user_id: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for receipt in store.list_receipts(user_id):
        amount = _money(receipt.get("total_amount"))
        accounting = receipt.get("accounting_fields") or {}
        entries.append({
            "id": receipt["public_id"], "source": "receipt", "receipt_id": receipt["public_id"],
            "date": receipt.get("transaction_date") or str(receipt.get("created_at") or "")[:10],
            "supplier": receipt.get("merchant") or receipt.get("source_name") or "Unknown supplier",
            "description": "; ".join(str(line.get("description") or "") for line in receipt.get("line_items") or [] if line.get("description")) or accounting.get("purchase_summary") or "Receipt expense",
            "expense_type": _expense_type(receipt), "amount": f"{amount:.2f}" if amount is not None else None,
            "currency": str(accounting.get("currency") or "MYR").upper(),
            "payment_method": accounting.get("payment_method") or "", "tax_amount": accounting.get("tax_amount"),
            "status": receipt.get("status"), "notes": receipt.get("notes") or "", "has_amount": amount is not None,
        })
    for record in store.list_manual_expenses(user_id):
        amount = _money(record.get("amount"))
        entries.append({
            "id": record["public_id"], "source": "manual", "receipt_id": None, "date": record["expense_date"],
            "supplier": record["supplier"], "description": record["description"], "expense_type": record["expense_type"],
            "amount": f"{amount:.2f}" if amount is not None else None, "currency": record["currency"],
            "payment_method": record["payment_method"], "tax_amount": record["tax_amount"],
            "status": "recorded", "notes": record["notes"], "has_amount": amount is not None,
        })
    return entries


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


def second_reading(ocr_engine, extracted, faithful_content, source_shape):
    """Read the cleaned faithful view only when the date, total or receipt
    number could not be read from the first views, and fill only those."""
    keys = ("transaction_date", "total_amount", "receipt_invoice_number")
    missing = [key for key in keys if not (extracted.get(key) or {}).get("value")]
    record = {"run": bool(missing), "missing_before": missing, "filled": []}
    if not missing:
        return record
    view = faithful_content if faithful_content.ndim == 2 else cv2.cvtColor(faithful_content, cv2.COLOR_BGR2GRAY)
    try:
        second = ocr_engine.run_views([("faithful_recovery", view)])
    except Exception as error:  # a second reading must never fail the upload
        record["error"] = type(error).__name__
        return record
    if not second.lines:
        return record
    scale = view.shape[1] / float(source_shape[1]) if source_shape[1] else 1.0
    found = extract_fields(second.text, second.lines)
    for key in missing:
        payload = found.get(key) or {}
        if not payload.get("value"):
            continue
        box = payload.get("box")
        if box is not None and scale and abs(scale - 1.0) > 1e-3:
            try:
                box = (np.asarray(box, dtype=float) / scale).tolist()
            except (TypeError, ValueError):
                box = None
        extracted[key] = dict(payload, box=box)
        record["filled"].append({"field": key, "value": payload["value"]})
    return record


_MONTH_NAMES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def date_strip_views(source: np.ndarray, box, from_right: bool = False) -> list[np.ndarray]:
    """Enlarged copies of the strip that holds a date, from the date label (or,
    with from_right, from the end of the read part) to the right edge of the
    receipt, where a faint or stained year often is. Print is measured against
    its own surroundings at stroke scale, so a stain behind the digits becomes
    background. Nothing is drawn."""
    gray = source if source.ndim == 2 else cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    points = np.asarray(box, dtype=float)
    x0, y0 = points.min(axis=0); x1, y1 = points.max(axis=0)
    height = max(6.0, y1 - y0)
    top, bottom = int(max(0, y0 - 0.45 * height)), int(min(gray.shape[0], y1 + 0.45 * height))
    # from_right: start back at the month name (the reader anchors on it) and stop
    # a little past the box, before any paper edge or stain band at the margin
    left = int(max(0, (x1 - 2.4 * height) if from_right else (x0 - 0.3 * height)))
    right = int(min(gray.shape[1], x1 + 2.2 * height)) if from_right else gray.shape[1]
    strip = gray[top:bottom, left:right]
    if strip.size == 0 or strip.shape[0] < 4 or strip.shape[1] < 8:
        return []
    scale = float(np.clip(52.0 / height, 2.0, 6.0))
    big = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.fastNlMeansDenoising(big, None, h=5, templateWindowSize=7, searchWindowSize=21)
    text_height = height * scale
    views = []
    for factor in (0.3, 0.42):
        kernel = int(max(7, factor * text_height)) | 1
        evidence = cv2.morphologyEx(big, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))).astype(np.float32)
        low, high = np.percentile(evidence, 60), np.percentile(evidence, 99.3)
        tone = np.clip((evidence - low) / max(high - low, 1.0), 0, 1)
        views.append(cv2.copyMakeBorder((255 - 255 * tone).astype(np.uint8), 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255))
    # paper flattened by its own local brightest level: a stain over the digits is
    # divided out, so the print under it keeps its own shape
    raw = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    kernel = int(max(9, 0.48 * text_height)) | 1
    paper = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    ratio = raw / np.maximum(paper, 1.0)
    low, high = np.percentile(ratio, 1), np.percentile(ratio, 75)
    flat = (255 * np.clip((ratio - low) / max(high - low, 1e-3), 0, 1)).astype(np.uint8)
    views.insert(0, cv2.copyMakeBorder(flat, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255))
    return views


def year_from_strip_text(text: str, hint: str) -> str | None:
    """Full date when the strip text shows the hinted day/month with a 4-digit year."""
    from datetime import date
    try:
        day, month_name = hint.split()
        month = _MONTH_NAMES.index(month_name[:3].upper()) + 1
        day = int(day)
    except (ValueError, AttributeError):
        return None
    cleaned = text.upper()
    # the usual reader confusions inside a year: O/D/Q/C for 0, I/L for 1, Z for 2, S for 5, B for 8
    pattern = _MONTH_NAMES[month - 1] + r"[A-Z]*\W{0,3}([12Z][09OQDC][0-9OQDCILZSB]{2})(?![0-9A-Z])"
    match = re.search(pattern, cleaned)
    if not match:
        return None
    year = match.group(1).translate(str.maketrans("OQDCILZSB", "000011258"))
    try:
        found = date(int(year), month, day)
    except ValueError:
        return None
    if found > date.today() or found.year < 2000:
        return None
    return found.isoformat()


def _year_only(text: str, hint: str) -> str | None:
    """A strip read from just after the month: the year alone."""
    match = re.match(r"\W*([12Z][09OQDC][0-9OQDCILZSB]{2})(?![0-9A-Z])", text.upper().strip())
    return year_from_strip_text(f"{hint.split()[1]} {match.group(1)}", hint) if match and len(hint.split()) == 2 else None


def _month_and_year(text: str, hint: str) -> str | None:
    """Strip read from the month onward; the bare year is also accepted."""
    return year_from_strip_text(text, hint) or _year_only(text, hint)


def date_year_reading(ocr_engine, extracted, source: np.ndarray) -> dict:
    """When only the day and month were read, re-read just the date strip:
    first with line detection, then the strip as one line, then only the
    part after the month. A year is accepted only when all four digits are
    read; otherwise the year is left for the user."""
    payload = extracted.get("transaction_date") or {}
    record = {"run": False}
    if payload.get("value") or not payload.get("hint") or payload.get("box") is None:
        return record
    record = {"run": True, "hint": payload["hint"], "readings": []}
    hint, box = payload["hint"], payload["box"]
    attempts = []
    for view in date_strip_views(source, box):
        attempts.append(("detect", view, year_from_strip_text))
        attempts.append(("line", view, year_from_strip_text))
    for view in date_strip_views(source, box, from_right=True):
        attempts.append(("after_month", view, _month_and_year))
        attempts.append(("after_month_detect", view, _month_and_year))
    for mode, view, parse in attempts:
        try:
            if mode in ("line", "after_month") and hasattr(ocr_engine, "recognize"):
                text, _ = ocr_engine.recognize(view)
            else:
                text = " ".join(line.get("text", "") for line in ocr_engine.run(view).lines)
        except Exception as error:
            record["error"] = type(error).__name__
            continue
        record["readings"].append(f"{mode}: {text[:60]}")
        value = parse(text, hint)
        if value:
            extracted["transaction_date"] = dict(payload, value=value, hint=None)
            record["filled"] = value
            record["method"] = mode
            break
    else:
        suggested = _partly_read_year(record["readings"], hint)
        if suggested:
            extracted["transaction_date"] = dict(payload, suggested=suggested)
            record["suggested"] = suggested
    return record


def _partly_read_year(readings: list[str], hint: str) -> str | None:
    """A year whose century digit is hidden (stain, fold) but whose first and last
    two digits were read after the month: only one year from 2000 to today fits,
    so it is offered as a suggestion for the user to accept, never filled in.
    Two different candidates mean no suggestion."""
    month = hint.split()[1][:3].upper() if len(hint.split()) == 2 else ""
    if len(month) != 3:
        return None
    # the month with at most one misread letter (FEB, FED, F3B ...)
    month_pattern = "(?:" + "|".join(re.escape(month[:i]) + "." + re.escape(month[i + 1:]) for i in range(3)) + ")"
    candidates = set()
    for reading in readings:
        if not reading.startswith("after_month"):
            continue
        text = reading.split(":", 1)[1].upper()
        match = re.search(month_pattern + r"\W{0,2}([0-9OQDCILZSB]{3,4})(?![0-9A-Z])", text)
        if not match:
            continue
        digits = match.group(1).translate(str.maketrans("OQDCILZSB", "000011258"))
        if len(digits) == 4:
            year = digits if digits[:2] == "20" else None
        else:
            year = "20" + digits[1:] if digits[0] == "2" else None
        if year:
            candidates.add(year_from_strip_text(f"{hint.split()[1]} {year}", hint))
    candidates.discard(None)
    return candidates.pop() if len(candidates) == 1 else None


def expense_category(receipt: dict) -> str:
    """Expense type label shared with the finance view (user-editable)."""
    return EXPENSE_TYPES[_expense_type(receipt)]


def _alnum(value) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(value or "").upper())


def possible_duplicate(a: dict, b: dict) -> bool:
    """Same receipt uploaded twice, tolerant of OCR differences: identical
    image, or same receipt number and total, or same date and total with a
    similar supplier name."""
    from difflib import SequenceMatcher
    if a.get("content_sha256") and a.get("content_sha256") == b.get("content_sha256"):
        return True
    def amount(r):
        try:
            return Decimal(str(r.get("total_amount"))).quantize(Decimal("0.01"))
        except (InvalidOperation, ValueError, TypeError):
            return None
    total = amount(a)
    if total is None or total != amount(b):
        return False
    number_a, number_b = _alnum(a.get("receipt_number")), _alnum(b.get("receipt_number"))
    if len(number_a) >= 3 and number_a == number_b:
        return True
    if a.get("transaction_date") and a.get("transaction_date") == b.get("transaction_date"):
        name_a, name_b = _alnum(a.get("merchant")), _alnum(b.get("merchant"))
        if name_a and name_b and SequenceMatcher(None, name_a, name_b).ratio() >= 0.6:
            return True
    return False


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


def upright_by_reading(color, ocr):
    """A phone photo can be cut out and stood upright, but the cut alone cannot
    tell whether the print reads upright or upside down. Read a small copy both
    ways and keep the way that reads better. Scans never come through here."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
    factor = min(1.0, 1100.0 / max(gray.shape))
    small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else gray

    def score(view):
        result = ocr.run(view)
        return sum(float(line.get("confidence") or 0) * len(str(line.get("text") or "").strip())
                   for line in (result.lines or []) if float(line.get("confidence") or 0) >= 0.5)

    upright, flipped = score(small), score(cv2.rotate(small, cv2.ROTATE_180))
    turn = flipped > 1.15 * upright + 5
    record = {"upright_score": round(upright, 1), "upside_down_score": round(flipped, 1), "turned_180": bool(turn)}
    return (cv2.rotate(color, cv2.ROTATE_180) if turn else color), record


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
    phone_pairings = PhonePairings()
    app.config["PHONE_SCANNING"] = os.getenv("RECEIPT_PHONE", "0") == "1" or testing

    def current_user():
        user_id = session.get("user_id")
        user = store.user_by_id(int(user_id)) if user_id else None
        if user:
            user["is_guest"] = user["email"].endswith("@guest.local.invalid")
            user["is_workspace"] = user["email"] == WORKSPACE_EMAIL
            if user["is_guest"] or user["is_workspace"]: user["email"] = ""
            user["welcome_seen"] = user["is_guest"] or store.app_setting(f"welcome_seen:{user['id']}") is not None
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
        # phone uploads carry their own pairing code instead of a browser session
        if request.method in {"POST", "PATCH", "DELETE"} and request.endpoint not in {"static", "phone_upload"}:
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
        return jsonify({"user": current_user(), "csrf_token": session["csrf_token"], **_access_state()})

    def _local_request() -> bool:
        import ipaddress
        try:
            local = ipaddress.ip_address(request.remote_addr or "").is_loopback
        except ValueError:
            local = False
        return local and os.getenv("RECEIPT_HOST", "127.0.0.1") in {"127.0.0.1", "::1", "localhost"}

    def sign_in_required() -> bool:
        return store.app_setting("require_sign_in", "0") == "1"

    def _access_state() -> dict[str, Any]:
        return {"sign_in_required": sign_in_required(), "workspace_available": (not sign_in_required()) and _local_request()}

    @app.post("/api/workspace")
    def open_workspace():
        """Open this computer's receipts without a password. Only from this
        computer, and only while "Require sign-in" is off."""
        if sign_in_required():
            return jsonify({"error": "Sign-in is required on this computer."}), 403
        if not _local_request():
            return jsonify({"error": "The workspace opens only on the computer running the app."}), 403
        user = store.workspace_user()
        store.log_event(user["id"], "account.workspace_open", "user", user["public_id"])
        session.clear(); session["user_id"] = user["id"]; session["csrf_token"] = secrets.token_urlsafe(24); session.permanent = True
        return jsonify({"user": current_user(), "csrf_token": session["csrf_token"]})

    @app.get("/api/access")
    @require_auth
    def access_settings():
        user = current_user()
        return jsonify({**_access_state(), "has_password": store.user_has_password(user["id"]), "email": user.get("email") or "",
                        "name": user.get("name") or ""})

    @app.patch("/api/access")
    @require_auth
    def update_access_settings():
        """Turn "Require sign-in" on or off for this computer.

        Turning it on from the password-free workspace first gives that
        workspace an email and password, so its receipts stay reachable.
        Turning it off makes the signed-in account the one that opens
        without a password."""
        user = current_user()
        if user.get("is_guest"):
            return jsonify({"error": "Guest sessions cannot change sign-in settings."}), 403
        payload = request.get_json(silent=True) or {}
        required = payload.get("require_sign_in")
        if not isinstance(required, bool):
            return jsonify({"error": "Choose whether sign-in is required."}), 400
        if required:
            if not store.user_has_password(user["id"]):
                name = str(payload.get("name") or user.get("name") or "").strip()
                email = str(payload.get("email", "")).strip().lower()
                password = str(payload.get("password", ""))
                errors = {}
                if not 2 <= len(name) <= 80: errors["name"] = "Enter a name using 2 to 80 characters."
                if not _valid_email(email) or email.endswith(".local.invalid"): errors["email"] = "Enter a valid email address."
                password_error = _password_error(password, name, email)
                if password_error: errors["password"] = password_error
                if str(payload.get("password_confirmation", "")) != password: errors["confirmation"] = "The passwords do not match."
                other = store.user_by_email(email) if email else None
                if other and other["id"] != user["id"]: errors["email"] = "Another account already uses this email."
                if errors:
                    return jsonify({"error": "Set an email and password first, so these receipts stay yours.", "fields": errors}), 400
                store.set_user_credentials(user["id"], name, email,
                                           generate_password_hash(unicodedata.normalize("NFC", password), method="scrypt"))
            store.set_app_setting("require_sign_in", "1")
        else:
            if not store.user_has_password(user["id"]):
                return jsonify({"error": "Sign in with an account to change this."}), 403
            store.set_app_setting("require_sign_in", "0")
            store.set_app_setting("workspace_user_id", str(user["id"]))
        store.log_event(user["id"], "settings.require_sign_in", "app", None, details={"require_sign_in": required})
        return jsonify({**_access_state(), "has_password": store.user_has_password(user["id"]), "user": current_user()})

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

    @app.get("/api/finance")
    @require_auth
    def finance_report():
        user_id = current_user()["id"]
        entries = _finance_entries(store, user_id)
        date_from, date_to = request.args.get("from", ""), request.args.get("to", "")
        category, source = request.args.get("category", "all"), request.args.get("source", "all")
        if date_from: entries = [item for item in entries if item["date"] and item["date"] >= date_from]
        if date_to: entries = [item for item in entries if item["date"] and item["date"] <= date_to]
        if category != "all": entries = [item for item in entries if item["expense_type"] == category]
        if source != "all": entries = [item for item in entries if item["source"] == source]
        entries.sort(key=lambda item: (item.get("date") or "", item["id"]), reverse=True)
        totals: dict[str, Decimal] = {}; monthly: dict[str, Decimal] = {}; total = Decimal("0")
        missing = unreviewed = 0
        for item in entries:
            amount = _money(item.get("amount"))
            if amount is None: missing += 1; continue
            total += amount
            totals[item["expense_type"]] = totals.get(item["expense_type"], Decimal("0")) + amount
            month = (item.get("date") or "Unknown")[:7]
            monthly[month] = monthly.get(month, Decimal("0")) + amount
            if item["source"] == "receipt" and item["status"] != "reviewed": unreviewed += 1
        return jsonify({
            "entries": entries, "types": EXPENSE_TYPES,
            "summary": {"total": f"{total:.2f}", "count": len(entries), "missing_amounts": missing,
                        "needs_review": unreviewed, "average": f"{(total / (len(entries)-missing)):.2f}" if len(entries) > missing else "0.00"},
            "by_type": [{"code": key, "label": EXPENSE_TYPES[key], "total": f"{value:.2f}"} for key, value in sorted(totals.items(), key=lambda pair: pair[1], reverse=True)],
            "monthly": [{"month": key, "total": f"{monthly[key]:.2f}"} for key in sorted(monthly)],
        })

    def validate_expense(payload: dict[str, Any], partial: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
        clean: dict[str, Any] = {}; errors: dict[str, str] = {}
        required = ("expense_date", "supplier", "amount")
        for key in required:
            if key in payload or not partial:
                value = str(payload.get(key, "")).strip()
                if not value: errors[key] = "This field is required."
                else: clean[key] = value
        if clean.get("expense_date"):
            try: datetime.strptime(clean["expense_date"], "%Y-%m-%d")
            except ValueError: errors["expense_date"] = "Use a valid date."
        if "amount" in clean:
            amount = _money(clean["amount"])
            if amount is None or amount <= 0: errors["amount"] = "Enter a positive amount with up to two decimals."
            else: clean["amount"] = f"{amount:.2f}"
        category = str(payload.get("expense_type", "other")).strip().lower()
        if "expense_type" in payload or not partial:
            if category not in EXPENSE_TYPES: errors["expense_type"] = "Choose a valid expense type."
            else: clean["expense_type"] = category
        for key, limit in (("description", 500), ("payment_method", 100), ("notes", 2000)):
            if key in payload: clean[key] = str(payload.get(key, "")).strip()[:limit]
        if "currency" in payload or not partial:
            currency = str(payload.get("currency", "MYR")).strip().upper()
            if not re.fullmatch(r"[A-Z]{3}", currency): errors["currency"] = "Use a three-letter currency code."
            else: clean["currency"] = currency
        if "tax_amount" in payload:
            raw_tax = str(payload.get("tax_amount") or "").strip()
            tax = _money(raw_tax) if raw_tax else None
            if raw_tax and tax is None: errors["tax_amount"] = "Enter a valid tax amount."
            else: clean["tax_amount"] = f"{tax:.2f}" if tax is not None else None
        return clean, errors

    @app.post("/api/finance/expenses")
    @require_auth
    def create_finance_expense():
        payload = request.get_json(silent=True) or {}
        clean, errors = validate_expense(payload)
        if errors: return jsonify({"error": "Check the expense fields.", "fields": errors}), 400
        record = store.create_manual_expense(current_user()["id"], clean)
        store.log_event(current_user()["id"], "finance.expense_create", "expense", record["public_id"], details={"amount": record["amount"], "type": record["expense_type"]})
        return jsonify(record), 201

    @app.patch("/api/finance/expenses/<expense_id>")
    @require_auth
    def update_finance_expense(expense_id: str):
        payload = request.get_json(silent=True) or {}; clean, errors = validate_expense(payload, partial=True)
        if errors: return jsonify({"error": "Check the expense fields.", "fields": errors}), 400
        record = store.update_manual_expense(current_user()["id"], expense_id, clean)
        if not record: return jsonify({"error": "Expense not found or no valid change supplied."}), 404
        store.log_event(current_user()["id"], "finance.expense_update", "expense", expense_id)
        return jsonify(record)

    @app.delete("/api/finance/expenses/<expense_id>")
    @require_auth
    def delete_finance_expense(expense_id: str):
        if not store.delete_manual_expense(current_user()["id"], expense_id): return jsonify({"error": "Expense not found."}), 404
        store.log_event(current_user()["id"], "finance.expense_delete", "expense", expense_id)
        return jsonify({"ok": True})

    @app.patch("/api/finance/receipts/<receipt_id>")
    @require_auth
    def classify_receipt_expense(receipt_id: str):
        category = str((request.get_json(silent=True) or {}).get("expense_type", "")).lower()
        if category not in EXPENSE_TYPES: return jsonify({"error": "Choose a valid expense type."}), 400
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        accounting = dict(receipt.get("accounting_fields") or {}); accounting["expense_type"] = category
        updated = store.update_receipt(current_user()["id"], receipt_id, {"accounting_fields": accounting})
        store.log_event(current_user()["id"], "finance.receipt_classify", "receipt", receipt_id, details={"expense_type": category})
        return jsonify({"receipt": updated, "expense_type": category})

    @app.get("/api/finance/export.xlsx")
    @require_auth
    def finance_export():
        entries = _finance_entries(store, current_user()["id"])
        workbook = Workbook(); ledger = workbook.active; ledger.title = "Expense ledger"
        ledger.append(["Date", "Supplier", "Description", "Expense type", "Amount", "Currency", "Payment method", "Tax", "Source", "Review status", "Record ID"])
        for item in sorted(entries, key=lambda row: row.get("date") or ""):
            ledger.append([item["date"], safe_excel_value(item["supplier"]), safe_excel_value(item["description"]), EXPENSE_TYPES[item["expense_type"]], float(item["amount"]) if item["amount"] else None, item["currency"], safe_excel_value(item["payment_method"]), float(item["tax_amount"]) if _money(item["tax_amount"]) is not None else None, item["source"], item["status"], item["id"]])
        for cell in ledger[1]:
            font = copy(cell.font); font.bold = True; cell.font = font
        for row in ledger.iter_rows(min_row=2): row[4].number_format = row[7].number_format = "0.00"
        for column, width in {"A": 14, "B": 28, "C": 42, "D": 28, "E": 14, "F": 10, "G": 18, "H": 12, "I": 12, "J": 16, "K": 28}.items(): ledger.column_dimensions[column].width = width
        by_type = workbook.create_sheet("By expense type"); by_type.append(["Expense type", "Records", "Total (RM)"])
        grouped: dict[str, list[Decimal]] = {}
        for item in entries:
            amount = _money(item.get("amount"))
            if amount is not None: grouped.setdefault(item["expense_type"], []).append(amount)
        for category, amounts in sorted(grouped.items()): by_type.append([EXPENSE_TYPES[category], len(amounts), float(sum(amounts, Decimal("0")))])
        for cell in by_type[1]:
            font = copy(cell.font); font.bold = True; cell.font = font
        by_type.column_dimensions["A"].width = 32; by_type.column_dimensions["B"].width = 12; by_type.column_dimensions["C"].width = 18
        stream = io.BytesIO(); workbook.save(stream); stream.seek(0)
        store.log_event(current_user()["id"], "finance.export", "expense", details={"records": len(entries)})
        return send_file(stream, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", as_attachment=True, download_name="expense-report.xlsx")

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

    @app.get("/api/receipts/<receipt_id>/reading-copy")
    @require_auth
    def receipt_reading_copy(receipt_id: str):
        """The scanner's text, one tidy reading per printed place, for the reading copy."""
        receipt = store.get_receipt(current_user()["id"], receipt_id)
        if not receipt:
            return jsonify({"error": "Receipt not found."}), 404
        return jsonify({"lines": reading_copy_lines((receipt.get("audit") or {}).get("ocr_lines") or [])})

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
                if possible_duplicate(receipt, candidate):
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
                         "currency", "payment_method", "subtotal", "discount", "tax_amount", "payable_amount", "line_item_count",
                         "expense_type", "suggested_expense_type", "brand", "business_type", "purchase_summary", "recognised_from"}
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
        # Rebuild the Faithful proposed-method output so older receipts can
        # benefit from the current recovery without being uploaded again.
        from .restoration import assess_receipt_degradation, make_adaptive_document_view, make_faithful_content_view, make_readability_view, region_adaptive_source_fusion, select_faithful_recovery
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
                "ocr_role": "localisation only; recognised strings are not rendered",
                "model_role": "independent source-trace support gate",
            })
            audit.setdefault("stored_file_sha256", {})["recovered"] = _file_sha256(existing["restored_path"])
            audit["stored_file_sha256"]["faithful_content"] = _file_sha256(faithful_path)
            audit["faithful_display_image"] = "faithful_content.png"
            audit["faithful_display_policy"] = (
                "background-normalised image-derived content with OCR-localised, "
                "source-and-DSDNet-supported stroke regularisation; recognised strings are not rendered"
            )
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
        if not receipt or kind not in {"original", "restored", "faithful", "faithful_reference", "diagnostic"}:
            return jsonify({"error": "Image not found."}), 404
        store.log_event(current_user()["id"], "receipt.image_view", "receipt", receipt_id, details={"kind": kind})
        path = receipt["input_path"] if kind == "original" else receipt["restored_path"]
        if kind == "faithful":
            candidate = Path(receipt["restored_path"]).with_name("faithful_content.png")
            if not candidate.is_file():
                candidate = Path(receipt["restored_path"]).with_name("ocr_input.png")
            path = str(candidate if candidate.is_file() else Path(receipt["restored_path"]))
        if kind == "diagnostic": path = str(Path(receipt["restored_path"]).with_name("diagnostic.png"))
        if kind == "faithful_reference":
            root = Path(receipt["restored_path"]).parent
            paths = [root / 'faithful_reference.png', root / 'registered_color.png', root / 'rectified_input.png', Path(receipt['input_path'])]
            path = str(next(item for item in paths if item.is_file()))
        if not path or not Path(path).is_file(): return jsonify({"error": "Image not found."}), 404
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

    def _accountant_records(user, start="", end=""):
        from datetime import date
        from .myinvois import assess_purchase_receipt
        today = date.today().isoformat()
        records = []
        for row in store.list_receipts(user["id"]):
            receipt = store.get_receipt(user["id"], row["public_id"])
            if not receipt:
                continue
            assessed = assess_purchase_receipt(receipt, today=today)
            when = assessed["essentials"]["transaction_date"] or ""
            if (start and (not when or when < start)) or (end and (not when or when > end)):
                continue
            records.append((receipt, assessed))
        records.sort(key=lambda item: (item[1]["essentials"]["transaction_date"] or "9999", item[1]["essentials"]["supplier_name"] or ""))
        return records

    def _pack_file_name(receipt, assessed, suffix):
        e = assessed["essentials"]
        supplier = re.sub(r"[^A-Za-z0-9]+", "-", (e["supplier_name"] or "unknown"))[:30].strip("-") or "unknown"
        total = (e["total"] or "na").replace(".", "_")
        return f"{e['transaction_date'] or 'undated'}_{supplier}_{total}_{receipt['public_id'][-6:]}_{suffix}.png"

    def _accountant_workbook(records, start="", end="", with_images=False):
        workbook = Workbook()
        sheet = workbook.active; sheet.title = "Expenses"
        headers = ["Date", "Supplier", "Receipt no.", "Total (RM)", "Tax (RM)", "Payment", "Expense type",
                   "e-Invoice status", "Request e-invoice by", "Check before use", "Possible duplicate of",
                   "Review status", "Receipt ID", "Receipt image"]
        sheet.append(headers)
        def number(value):
            try:
                return float(Decimal(str(value))) if value not in (None, "") else None
            except (InvalidOperation, ValueError):
                return None
        receipts_only = [receipt for receipt, _ in records]
        reviewed = 0
        for receipt, assessed in records:
            e = assessed["essentials"]; req = assessed["request"]
            # a request date only matters when the receipt invites it or the sale is too large to consolidate
            deadline = req["deadline"] if (assessed["status"] == "supplier_must_issue" or (req["possible"] and req["printed_on_receipt"])) else None
            twins = [other["public_id"] for other in receipts_only if other is not receipt and possible_duplicate(receipt, other)]
            reviewed += receipt.get("status") == "reviewed"
            image_name = _pack_file_name(receipt, assessed, "recovered") if with_images else None
            sheet.append([safe_excel_value(v) for v in [
                e["transaction_date"], e["supplier_name"], e["receipt_number"], None, None, e["payment_mode"],
                expense_category(receipt), assessed["label"], deadline, ", ".join(assessed["needs_review"]),
                ", ".join(twins), status_label_for_export(receipt.get("status")), receipt["public_id"], image_name]])
            row = sheet.max_row
            for column, value in ((4, e["total"]), (5, e["tax"])):
                amount = number(value)
                if amount is not None:
                    sheet.cell(row, column).value = amount
                    sheet.cell(row, column).number_format = "0.00"
            if image_name:
                cell = sheet.cell(row, 14); cell.hyperlink = f"receipts/{image_name}"; cell.style = "Hyperlink"
        if records:
            sheet.append([]); sheet.append(["", "Total", "", f"=SUM(D2:D{len(records) + 1})", f"=SUM(E2:E{len(records) + 1})"])
            sheet.cell(sheet.max_row, 4).number_format = "0.00"; sheet.cell(sheet.max_row, 5).number_format = "0.00"
        summary = workbook.create_sheet("By supplier")
        summary.append(["Month", "Supplier", "Receipts", "Total (RM)", "Needs supplier e-invoice"])
        groups: dict[tuple[str, str], list] = {}
        for receipt, assessed in records:
            e = assessed["essentials"]
            key = ((e["transaction_date"] or "")[:7] or "undated", e["supplier_name"] or "unknown")
            groups.setdefault(key, []).append(assessed)
        for (month, supplier), items in sorted(groups.items()):
            total = sum((number(i["essentials"]["total"]) or 0.0) for i in items)
            summary.append([month, safe_excel_value(supplier), len(items), round(total, 2),
                            sum(1 for i in items if i["status"] == "supplier_must_issue")])
            summary.cell(summary.max_row, 4).number_format = "0.00"
        categories = workbook.create_sheet("By expense type")
        categories.append(["Expense type", "Receipts", "Total (RM)"])
        by_category: dict[str, list] = {}
        for receipt, assessed in records:
            by_category.setdefault(expense_category(receipt), []).append(number(assessed["essentials"]["total"]) or 0.0)
        for name, amounts in sorted(by_category.items()):
            categories.append([name, len(amounts), round(sum(amounts), 2)])
            categories.cell(categories.max_row, 3).number_format = "0.00"
        notes = workbook.create_sheet("Read me")
        period = f"{start or 'first receipt'} to {end or 'latest receipt'}"
        for line in [
            "How to read this pack",
            f"Period: {period}. Receipts: {len(records)}. Reviewed by the user: {reviewed}; still to verify: {len(records) - reviewed}.",
            "These are purchase receipts. Under MyInvois the supplier issues the e-invoice; the buyer does not submit purchase receipts to LHDN.",
            "Keep as supporting document: LHDN accepts normal receipts/bills to support tax deductions until the legislation is amended. Keep records for 7 years.",
            "Supplier must issue an e-invoice: sales above RM10,000 cannot be consolidated, so obtain the supplier's validated e-invoice.",
            "Request e-invoice by: shown only when the receipt prints a request rule or the sale is above RM10,000. Otherwise no action is needed.",
            "Expense type comes from the type set on the receipt or, if none, from its tags and shop name; the accountant decides the final account code.",
            "Possible duplicate of: another receipt with the same image, the same receipt number and total, or the same date, total and a similar supplier name. Check before claiming both.",
            "Check before use: fields that could not be read from the receipt. Rows still marked 'To verify' have not been checked by the user.",
            "Receipt image: the recovered image in the receipts folder of this pack (the original upload is next to it).",
            "Supplier TIN, MSIC and SST numbers are the supplier's e-invoice fields; they are not needed to record a purchase.",
            "Self-billed e-invoices are only issued by the buyer in LHDN's self-billed cases; use the per-receipt draft in the app for those.",
        ]:
            notes.append([line])
        for ws in (sheet, summary, categories):
            for cell in ws[1]:
                font = copy(cell.font); font.bold = True; cell.font = font
            ws.freeze_panes = "A2"
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width = min(48, max(10, max(len(str(c.value or "")) for c in column) + 2))
        notes.column_dimensions["A"].width = 120
        return workbook

    def _pack_range():
        month = request.args.get("month") or ""
        start, end = request.args.get("from") or "", request.args.get("to") or ""
        if re.fullmatch(r"\d{4}-\d{2}", month):
            import calendar
            year, mon = int(month[:4]), int(month[5:])
            start, end = f"{month}-01", f"{month}-{calendar.monthrange(year, mon)[1]:02d}"
        for value in (start, end):
            if value and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                abort(400)
        return start, end

    @app.get("/api/accountant-pack.xlsx")
    @require_auth
    def export_accountant_pack():
        """All purchase receipts in one workbook for the accountant."""
        user = current_user(); start, end = _pack_range()
        records = _accountant_records(user, start, end)
        output = io.BytesIO(); _accountant_workbook(records, start, end).save(output); output.seek(0)
        store.log_event(user["id"], "receipt.export_accountant_pack", "receipt_collection", details={"count": len(records), "format": "xlsx"})
        return send_file(output, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         as_attachment=True, download_name="accountant-pack.xlsx")

    @app.get("/api/accountant-pack.zip")
    @require_auth
    def export_accountant_pack_zip():
        """Workbook plus every receipt image: what an accountant needs as
        supporting documents, named by date, supplier and amount."""
        import zipfile
        user = current_user(); start, end = _pack_range()
        records = _accountant_records(user, start, end)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            book = io.BytesIO(); _accountant_workbook(records, start, end, with_images=True).save(book)
            archive.writestr("accountant-pack.xlsx", book.getvalue())
            for receipt, assessed in records:
                root = Path(receipt["restored_path"]).parent
                recovered = next((item for item in (root / "faithful_content.png", Path(receipt["restored_path"])) if item.is_file()), None)
                if recovered:
                    archive.write(recovered, f"receipts/{_pack_file_name(receipt, assessed, 'recovered')}")
                original = Path(receipt["input_path"])
                if original.is_file():
                    archive.write(original, f"receipts/{_pack_file_name(receipt, assessed, 'original')}")
        output.seek(0)
        label = request.args.get("month") or (f"{start}_to_{end}" if start or end else "all")
        store.log_event(user["id"], "receipt.export_accountant_pack", "receipt_collection", details={"count": len(records), "format": "zip"})
        return send_file(output, mimetype="application/zip", as_attachment=True,
                         download_name=f"accountant-pack-{re.sub(r'[^0-9A-Za-z_-]', '', label)}.zip")

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

    # ------------------------------------------------------------ MyInvois
    def _myinvois_state(user, receipt, saved=None):
        from .myinvois import merge_draft, suggest_from_receipt, supplier_profile_key, validate_draft
        base = suggest_from_receipt(receipt)
        if saved is None:
            saved = store.get_myinvois_draft(user["id"], receipt["public_id"]) or {}
        supplier = base["supplier"]
        supplier_profile = (store.get_myinvois_profile(user["id"], "supplier", supplier_profile_key(supplier.get("name"), supplier.get("id_value")))
                            or store.get_myinvois_profile(user["id"], "supplier", supplier_profile_key(supplier.get("name"))))
        buyer_profile = store.get_myinvois_profile(user["id"], "buyer", "self")
        draft = merge_draft(base, saved or None, ("supplier", supplier_profile), ("buyer", buyer_profile))
        return draft, validate_draft(draft)

    @app.get("/api/settings")
    @require_auth
    def get_user_settings():
        from .settings import PRESETS, SETTINGS, resolve
        return jsonify({"values": resolve(store.get_settings(current_user()["id"])), "schema": SETTINGS, "presets": PRESETS})

    @app.patch("/api/settings")
    @require_auth
    def save_user_settings():
        from .settings import clean, resolve
        user = current_user()
        payload = request.get_json(silent=True) or {}
        merged = dict(store.get_settings(user["id"])); merged.update(clean(payload))
        store.save_settings(user["id"], merged)
        store.log_event(user["id"], "settings.update", "user", str(user["id"]), details={"keys": sorted(clean(payload))})
        return jsonify({"values": resolve(merged)})

    def _purchase_state(receipt):
        from datetime import date
        from .myinvois import assess_purchase_receipt
        return assess_purchase_receipt(receipt, today=date.today().isoformat())

    def _clean_myinvois_payload(raw):
        """Accept only the draft structure, with bounded sizes and plain values."""
        if not isinstance(raw, dict):
            return None, "The MyInvois draft must be an object."
        if len(json.dumps(raw)) > 200_000:
            return None, "The MyInvois draft is too large."
        def scalar(value):
            if value is None or isinstance(value, (bool, int, float)):
                return value
            if isinstance(value, str):
                return value.strip()[:1000]
            raise ValueError
        def clean(value, depth=0):
            if depth > 4:
                raise ValueError
            if isinstance(value, dict):
                return {str(k)[:60]: clean(v, depth + 1) for k, v in list(value.items())[:80]}
            if isinstance(value, list):
                return [clean(v, depth + 1) for v in value[:500]]
            return scalar(value)
        try:
            draft = {key: clean(raw[key]) for key in ("mode", "invoice", "supplier", "buyer", "lines", "totals", "provenance") if key in raw}
        except ValueError:
            return None, "The MyInvois draft contains unsupported values."
        if draft.get("mode") not in (None, "individual", "consolidated"):
            return None, "Mode must be individual or consolidated."
        if "lines" in draft and not all(isinstance(row, dict) for row in draft["lines"]):
            return None, "Each invoice line must be an object."
        return draft, None

    @app.get("/api/receipts/<receipt_id>/myinvois")
    @require_auth
    def myinvois_draft(receipt_id: str):
        from .myinvois import form_schema
        user = current_user()
        receipt = store.get_receipt(user["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        draft, validation = _myinvois_state(user, receipt)
        return jsonify({"draft": draft, "validation": validation, "schema": form_schema(),
                        "purchase": _purchase_state(receipt),
                        "scope": "Prepares and validates an unsigned MyInvois Invoice v1.0 document. Signing and submission are done by the authorised issuer."})

    @app.patch("/api/receipts/<receipt_id>/myinvois")
    @require_auth
    def save_myinvois_draft(receipt_id: str):
        from .myinvois import form_schema, profile_from_draft, supplier_profile_key
        user = current_user()
        receipt = store.get_receipt(user["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        payload = request.get_json(silent=True) or {}
        saved, error = _clean_myinvois_payload(payload.get("draft"))
        if error: return jsonify({"error": error}), 400
        store.save_myinvois_draft(user["id"], receipt_id, saved)
        draft, validation = _myinvois_state(user, receipt, saved)
        if payload.get("remember_supplier"):
            profile = profile_from_draft(draft, "supplier")
            for key in {supplier_profile_key(profile.get("name"), profile.get("id_value")), supplier_profile_key(profile.get("name"))} - {None}:
                store.save_myinvois_profile(user["id"], "supplier", key, profile)
        if payload.get("remember_buyer") and draft.get("mode") != "consolidated":
            store.save_myinvois_profile(user["id"], "buyer", "self", profile_from_draft(draft, "buyer"))
        store.log_event(user["id"], "receipt.myinvois_draft_saved", "receipt", receipt_id,
                        details={"ready": validation["ready"], "errors": validation["error_count"]})
        return jsonify({"draft": draft, "validation": validation, "schema": form_schema(),
                        "purchase": _purchase_state(receipt)})

    @app.get("/api/receipts/<receipt_id>/myinvois.json")
    @require_auth
    def export_myinvois(receipt_id: str):
        from .myinvois import build_ubl_json
        user = current_user()
        receipt = store.get_receipt(user["id"], receipt_id)
        if not receipt: return jsonify({"error": "Receipt not found."}), 404
        draft, validation = _myinvois_state(user, receipt)
        if not validation["ready"]:
            return jsonify({"error": "Complete the highlighted MyInvois fields before exporting.", "validation": validation}), 422
        document = build_ubl_json(draft)
        store.log_event(user["id"], "receipt.export_myinvois_json", "receipt", receipt_id)
        data = io.BytesIO(json.dumps(document, indent=2, ensure_ascii=False).encode("utf-8"))
        return send_file(data, mimetype="application/json", as_attachment=True,
                         download_name=f"{receipt_id}-myinvois-invoice-v1.0.json")

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
        friendly_views = {"rectified_input":"Original receipt", "source_preserving_recovery":"Recovered receipt", "readability_view":"Clean reading copy", "dsdnet_restoration":"Restored receipt"}
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
        coverage.append(["Important", "This workbook is not a MyInvois submission file. Fields are validated against the MyInvois Invoice v1.0 structure; download the MyInvois JSON from the receipt page. Signing and submission are done by the authorised issuer.", ""])
        coverage.append(["MyInvois field", "Current value", "Coverage status"])
        line_rows = export_items
        from .myinvois import FIELD_SPEC as _MI_FIELDS
        _mi_draft, _mi_validation = _myinvois_state(current_user(), receipt)
        _mi_issues = {}
        for _issue in _mi_validation["issues"]:
            _mi_issues.setdefault(_issue["path"], _issue["message"])
        _origin = {"receipt": "From receipt", "profile": "From saved profile", "default": "Rule default; verify",
                   "consolidated_rule": "Consolidated e-Invoice rule", "user": "Entered by user"}
        coverage_rows = []
        for _spec in _MI_FIELDS:
            _value = _mi_draft
            for _part in _spec["path"].split("."):
                _value = _value.get(_part) if isinstance(_value, dict) else None
            _status = _mi_issues.get(_spec["path"]) or _origin.get(_mi_draft["provenance"].get(_spec["path"]), "Captured" if _value else ("Optional" if _spec["rule"] != "M" else "Requires user input"))
            coverage_rows.append((_spec["label"], _value, _status))
        coverage_rows.append(("Invoice lines", f"{len(_mi_draft['lines'])} line(s)", "; ".join(i["message"] for i in _mi_validation["issues"] if i["path"].startswith("lines")) or "Complete"))
        coverage_rows.append(("Overall", f"{_mi_validation['mandatory_filled']}/{_mi_validation['mandatory_total']} mandatory values", "Ready to export" if _mi_validation["ready"] else f"{_mi_validation['error_count']} item(s) to complete"))
        coverage_rows.append(("Digital signature and submission identifiers", None, "Applied by the authorised issuer when submitting"))
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
        if (preprocessing_record.get("boundary_detection") or {}).get("method") == "photographed_paper":
            registered_color, preprocessing_record["upright_check"] = upright_by_reading(registered_color, ocr_engine)
        processed = cv2.cvtColor(registered_color, cv2.COLOR_BGR2GRAY) if registered_color.ndim == 3 else registered_color
        faithful_refinement = {}
        from .restoration import assess_receipt_degradation, make_adaptive_document_view, make_faithful_content_view, make_readability_view, region_adaptive_source_fusion, select_faithful_recovery
        from .settings import resolve as resolve_settings
        user_settings = resolve_settings(store.get_settings(user["id"]))
        stage_seconds: dict[str, float] = {}
        def _lap(name, since):
            stage_seconds[name] = round(time.perf_counter() - since, 2)
            return time.perf_counter()
        with inference_lock:
            lap = time.perf_counter()
            restoration = restoration_engine.run(processed)
            lap = _lap("restoration_model", lap)
            degradation_profile = assess_receipt_degradation(processed)
            fused_recovery, fusion_record = region_adaptive_source_fusion(processed, restoration.image)
            faithfulness = select_faithful_recovery(processed, fused_recovery)
            selected_recovery = faithfulness.image
            ocr_views = [("rectified_input", processed), ("source_preserving_recovery", selected_recovery)]
            # Also read the network's own restoration: where it sharpens faint
            # print the most, it reads best; the reading copy and the field rules
            # then keep the best reading of each printed place across all views.
            if (os.environ.get("RECEIPT_MODEL_VIEW", "1") != "0" and restoration.model_output is not None
                    and restoration.model_output.shape[:2] == processed.shape[:2]):
                ocr_views.append(("dsdnet_restoration", restoration.model_output))
            third_pass = user_settings["extra_ocr_view"]
            if third_pass == "always" or (third_pass == "auto" and {"faded", "blurred"} & set(degradation_profile.get("flags", []))):
                ocr_views.append(("readability_view", make_readability_view(processed, selected_recovery)))
            lap = _lap("fusion", lap)
            ocr = ocr_engine.run_views(ocr_views)
            lap = _lap(f"text_reading_{len(ocr_views)}_views", lap)
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
                finishing_options=user_settings,
            )
            lap = _lap("faithful_view_and_cleanup", lap)
            document_view = faithful_content
            display_mode = "faithful_recovery"
            extracted = extract_fields(ocr.text, ocr.lines)
            if user_settings["second_reading"]:
                second_reading_record = second_reading(ocr_engine, extracted, faithful_content, processed.shape)
                second_reading_record["date_strip"] = date_year_reading(ocr_engine, extracted, processed)
                lap = _lap("second_reading", lap)
            else:
                second_reading_record = {"run": False, "disabled": True}
        fields = route_fields(extracted, support_available=False, model_available=restoration.model_available)
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
        if faithful_refinement.get('applied'):
            reference = cv2.resize(registered_color, (faithful_content.shape[1], faithful_content.shape[0]), interpolation=cv2.INTER_CUBIC)
            write_image(audit_root / 'faithful_reference.png', reference)
        if restoration.model_output is not None: write_image(audit_root / "model_output_x8.png", restoration.model_output)
        if restoration.diagnostic_map is not None: write_image(diagnostic_path, restoration.diagnostic_map)
        runtime = time.perf_counter() - started
        policy = {
            "text": "OCR-localised source pixels", "rules": "thin linear structures",
            "marks": "dark connected source components", "background": "white",
            "synthetic_text_rendering": False,
            "ocr_role": "localisation only; recognised strings are not rendered",
            "model_role": "independent source-trace support gate",
            "role": "Faithful proposed-method output",
        }
        audit = {"run_id": run_id, "source": upload.filename, "content_sha256": hashlib.sha256(raw_bytes).hexdigest(), "raw_source_file": "source_upload.bin", "rectified_input_file": "rectified_input.png", "stored_file_sha256": {"source_upload": _file_sha256(source_upload_path), "original": _file_sha256(input_path), "recovered": _file_sha256(restored_path), "faithful_content": _file_sha256(audit_root / "faithful_content.png")}, "runtime_seconds": runtime, "restoration_status": restoration.status, "restoration_error": restoration.error, "checkpoint": restoration.checkpoint, "restoration_stages": restoration.stage_count, "degradation_profile": degradation_profile, "region_adaptive_fusion": fusion_record, "faithfulness_gate": {"accepted": faithfulness.accepted, "reason": faithfulness.reason, "metrics": faithfulness.metrics, "fallback": None if faithfulness.accepted else "rectified_original"}, "display_image": display_mode, "faithful_display_image": "faithful_content.png", "faithful_display_policy": "degradation-adaptive, OCR-localised source-and-DSDNet-supported stroke regularisation; recognised strings are not rendered; exact rectified original used when the faithfulness gate rejects recovery", "model_output_image": "model_output_x8.png" if restoration.model_output is not None else None, "source_preserving_recovery_image": "ocr_input.png", "ocr_input_images": [name for name, _ in ocr_views], "display_component_policy": policy, "restoration_diagnostics": restoration.diagnostics, "diagnostic_map": str(diagnostic_path) if restoration.diagnostic_map is not None else None, "ocr_status": ocr.status, "ocr_version": ocr.engine_version, "ocr_lines": ocr.lines, "ocr_suggestions": suggestions_for_lines(ocr.lines), "language_assistance_role": "localisation_only_for_faithful_stroke_regularisation; recognised_strings_not_rendered", "second_reading": second_reading_record, "stage_seconds": stage_seconds, "processing_settings": {k: user_settings[k] for k in ("extra_ocr_view", "second_reading", "severe_cleanup", "damage_separation", "stroke_joining")}, "semantic_extraction": {"method": "layout_aware_receipt_semantics_v2", "line_items": semantic_items, "accounting_fields": accounting_fields}, "arithmetic_consistency": arithmetic_consistency(ocr.lines, field_map.get("total_amount")), "support_status": "unavailable_for_natural_source", "fields": [field.as_dict() for field in fields], "preprocessing": preprocessing_record, "formal_evidence": False}
        from .restoration import complete_document_preservation_mode
        faithful_mode = complete_document_preservation_mode(registered_color)
        if faithful_mode:
            audit["faithful_display_policy"] = faithful_mode
            audit["faithful_preprocessing"] = preprocessing_record
            audit["faithful_processing"] = {
                "method": "continuous source luminance enhancement with evidence-gated stroke regularisation",
                "model_role": "independent support gate; recognised glyphs are not rendered",
                "synthetic_text_rendering": False,
                "ocr_role": "line localisation only",
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

    @app.post("/api/welcome")
    @require_auth
    def welcome_seen():
        store.set_app_setting(f"welcome_seen:{current_user()['id']}", utc_now())
        return jsonify({"ok": True})

    # ------------------------------------------------------------ phone scanner
    @app.post("/api/phone/pair")
    @require_auth
    def phone_pair():
        user = current_user()
        if user.get("is_guest"):
            return jsonify({"error": "Phone scanning is not available in a guest session."}), 403
        if not app.config.get("PHONE_SCANNING"):
            return jsonify({"error": "Phone scanning is switched off. Restart the app with phone scanning on.", "reason": "off"}), 409
        addresses = lan_addresses()
        if not addresses:
            return jsonify({"error": "This computer is not on a Wi-Fi or office network. Connect it to the same Wi-Fi as the phone.", "reason": "no_network"}), 409
        port = request.host.rsplit(":", 1)[1] if ":" in request.host else "80"
        pairing = phone_pairings.start(user["id"])
        url = f"http://{addresses[0]}:{port}/phone#{pairing.pop('token')}"
        store.log_event(user["id"], "phone.pair", "user", user["public_id"], details={"address": addresses[0]})
        return jsonify({**pairing, "url": url, "address": addresses[0], "other_addresses": addresses[1:4], "qr_svg": qr_svg(url)})

    @app.get("/api/phone/pair")
    @require_auth
    def phone_pair_status():
        return jsonify({"pairing": phone_pairings.status(current_user()["id"])})

    @app.delete("/api/phone/pair")
    @require_auth
    def phone_pair_end():
        phone_pairings.end(current_user()["id"])
        return jsonify({"ok": True})

    @app.get("/phone")
    def phone_page():
        return render_template("phone.html")

    def _phone_user():
        token = request.headers.get("X-Phone-Token", "")
        user_id = phone_pairings.user_for(token) if token else None
        user = store.user_by_id(user_id) if user_id else None
        if not user or user.get("status") != "active":
            return None, token
        user["is_guest"] = False
        user["is_workspace"] = user["email"] == WORKSPACE_EMAIL
        return user, token

    @app.get("/api/phone/check")
    def phone_check():
        user, _ = _phone_user()
        if not user:
            return jsonify({"error": "This pairing has ended. Scan the QR code on the computer again."}), 401
        return jsonify({"ok": True, "name": user.get("name")})

    @app.post("/api/phone/upload")
    def phone_upload():
        user, token = _phone_user()
        if not user:
            return jsonify({"error": "This pairing has ended. Scan the QR code on the computer again."}), 401
        upload = request.files.get("receipt")
        if upload is None or not upload.filename:
            return jsonify({"error": "No photo was received."}), 400
        phone_pairings.note(token, busy=1)
        try:
            result = process_receipt(upload, user)
        except ValueError as error:
            phone_pairings.note(token, failed=True, busy=-1)
            return jsonify({"error": str(error)}), 400
        except Exception:
            phone_pairings.note(token, failed=True, busy=-1)
            raise
        receipt = result.get("receipt") or {}
        phone_pairings.note(token, receipt=receipt, busy=-1)
        store.log_event(user["id"], "phone.upload", "receipt", receipt.get("public_id"))
        return jsonify({"receipt": {key: receipt.get(key) for key in ("public_id", "merchant", "total_amount", "transaction_date", "review_status")}})

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
