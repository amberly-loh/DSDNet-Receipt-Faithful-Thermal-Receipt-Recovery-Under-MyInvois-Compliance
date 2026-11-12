"""MyInvois (LHDN e-Invoice) draft preparation and validation.

Builds a complete e-Invoice draft from a reviewed receipt, records where
every value came from (receipt, saved profile, rule default or user entry),
validates it against the LHDN MyInvois SDK Invoice v1.0 data structure and
code lists, and serialises a valid draft as UBL 2.1 JSON.

Scope: this module prepares and validates data. It does not apply the
issuer's digital signature (required for document version 1.1) and does not
call the MyInvois API; submission remains with the authorised issuer.

Field rules follow https://sdk.myinvois.hasil.gov.my/documents/invoice-v1-0/
and the SDK code lists (state, payment mode, tax type, classification).
"""
from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

# --------------------------------------------------------------- code lists
STATE_CODES = {
    "01": "Johor", "02": "Kedah", "03": "Kelantan", "04": "Melaka", "05": "Negeri Sembilan",
    "06": "Pahang", "07": "Pulau Pinang", "08": "Perak", "09": "Perlis", "10": "Selangor",
    "11": "Terengganu", "12": "Sabah", "13": "Sarawak", "14": "Wilayah Persekutuan Kuala Lumpur",
    "15": "Wilayah Persekutuan Labuan", "16": "Wilayah Persekutuan Putrajaya", "17": "Not Applicable",
}
PAYMENT_MODES = {
    "01": "Cash", "02": "Cheque", "03": "Bank Transfer", "04": "Credit Card", "05": "Debit Card",
    "06": "e-Wallet / Digital Wallet", "07": "Digital Bank", "08": "Others",
}
TAX_TYPES = {
    "01": "Sales Tax", "02": "Service Tax", "03": "Tourism Tax", "04": "High-Value Goods Tax",
    "05": "Sales Tax on Low Value Goods", "06": "Not Applicable", "E": "Tax exemption (where applicable)",
}
CLASSIFICATION_CODES = {
    "001": "Breastfeeding equipment", "002": "Child care centres and kindergartens fees",
    "003": "Computer, smartphone or tablet", "004": "Consolidated e-Invoice",
    "005": "Construction materials (Fourth Schedule, CIDB Act 1994)", "006": "Disbursement",
    "007": "Donation", "008": "e-Commerce - e-Invoice to buyer / purchaser",
    "009": "e-Commerce - Self-billed e-Invoice to seller, logistics, etc.", "010": "Education fees",
    "011": "Goods on consignment (Consignor)", "012": "Goods on consignment (Consignee)",
    "013": "Gym membership", "014": "Insurance - Education and medical benefits",
    "015": "Insurance - Takaful or life insurance", "016": "Interest and financing expenses",
    "017": "Internet subscription", "018": "Land and building",
    "019": "Medical examination for learning disabilities / early intervention",
    "020": "Medical examination or vaccination expenses", "021": "Medical expenses for serious diseases",
    "022": "Others", "023": "Petroleum operations", "024": "Private retirement scheme or deferred annuity scheme",
    "025": "Motor vehicle", "026": "Subscription of books / journals / magazines / newspapers",
    "027": "Reimbursement", "028": "Rental of motor vehicle", "029": "EV charging facilities",
    "030": "Repair and maintenance", "031": "Research and development", "032": "Foreign income",
    "033": "Self-billed - Betting and gaming", "034": "Self-billed - Importation of goods",
    "035": "Self-billed - Importation of services", "036": "Self-billed - Others",
    "037": "Self-billed - Monetary payment to agents, dealers or distributors",
    "038": "Sports equipment, facilities, competitions or training fees",
    "039": "Supporting equipment for disabled person", "040": "Voluntary contribution to approved provident fund",
    "041": "Dental examination or treatment", "042": "Fertility treatment",
    "043": "Treatment and home care nursing, daycare and residential care centres",
    "044": "Vouchers, gift cards, loyalty points, etc",
    "045": "Self-billed - Non-monetary payment to agents, dealers or distributors",
}
ID_TYPES = {"BRN": 20, "NRIC": 12, "PASSPORT": 12, "ARMY": 12}
GENERAL_PUBLIC_TIN = "EI00000000010"
FOREIGN_BUYER_TIN = "EI00000000020"
# Common receipt units -> UN/ECE Recommendation 20 codes accepted by MyInvois.
UNIT_CODES = {
    "PCS": "H87", "PC": "H87", "PCE": "H87", "PIECE": "H87", "PIECES": "H87", "UNIT": "C62", "UNITS": "C62",
    "EA": "EA", "EACH": "EA", "SET": "SET", "SETS": "SET", "BOX": "BX", "BOXES": "BX", "PKT": "PA",
    "PACK": "PA", "KG": "KGM", "G": "GRM", "L": "LTR", "LTR": "LTR", "LITRE": "LTR", "ML": "MLT",
    "M": "MTR", "MTR": "MTR", "METER": "MTR", "LENGTH": "MTR", "LENGHT": "MTR", "ROLL": "RO", "PAIR": "PR",
    "HR": "HUR", "HOUR": "HUR", "DAY": "DAY", "MONTH": "MON", "DOZEN": "DZN", "BTL": "BO", "BOTTLE": "BO",
}
_STATE_WORDS = [
    (r"kuala\s+lumpur|\bk\.?\s*l\.?\b|w\.?p\.?\s*kuala", "14"), (r"putrajaya", "16"), (r"labuan", "15"),
    (r"selangor|petaling|shah\s+alam|klang|subang|batu\s+caves|selayang|kajang|ampang|puchong|rawang|sepang", "10"),
    (r"johor", "01"), (r"kedah", "02"), (r"kelantan", "03"), (r"melaka|malacca", "04"),
    (r"negeri\s+sembilan|seremban", "05"), (r"pahang|kuantan", "06"), (r"pulau\s+pinang|penang|george\s*town", "07"),
    (r"perak|ipoh", "08"), (r"perlis", "09"), (r"terengganu", "11"), (r"sabah|kota\s+kinabalu", "12"),
    (r"sarawak|kuching", "13"),
]
# Malaysian postcode ranges (first two digits) used only when no state word is printed.
_POSTCODE_STATES = [
    (range(1, 3), "09"), (range(5, 10), "02"), (range(10, 15), "07"), (range(15, 19), "03"),
    (range(20, 25), "11"), (range(25, 29), "06"), (range(30, 37), "08"), (range(39, 40), "06"),
    (range(40, 49), "10"), (range(50, 61), "14"), (range(62, 63), "16"), (range(63, 69), "10"),
    (range(69, 70), "06"), (range(70, 74), "05"), (range(75, 79), "04"), (range(79, 87), "01"),
    (range(87, 88), "15"), (range(88, 92), "12"), (range(93, 99), "13"),
]

# ------------------------------------------------------------- field schema
# path, label, group, rule ("M" mandatory, "C" conditional, "O" optional), max length, hint
FIELD_SPEC: list[dict[str, Any]] = [
    {"path": "invoice.number", "label": "e-Invoice code / number", "group": "Invoice", "rule": "M", "max": 50},
    {"path": "invoice.type_code", "label": "e-Invoice type", "group": "Invoice", "rule": "M", "max": 2,
     "options": {"01": "Invoice", "02": "Credit note", "03": "Debit note", "04": "Refund note"}},
    {"path": "invoice.currency", "label": "Invoice currency code", "group": "Invoice", "rule": "M", "max": 3},
    {"path": "invoice.exchange_rate", "label": "Currency exchange rate (foreign currency only)", "group": "Invoice", "rule": "C", "max": 15},
    {"path": "invoice.transaction_date", "label": "Transaction date (receipt)", "group": "Invoice", "rule": "M", "max": 10,
     "hint": "The e-Invoice issue date/time is set to the current UTC time when the draft is exported."},
    {"path": "invoice.transaction_time", "label": "Transaction time (receipt)", "group": "Invoice", "rule": "O", "max": 8},
    {"path": "invoice.payment_mode", "label": "Payment mode", "group": "Invoice", "rule": "O", "max": 2, "options": PAYMENT_MODES},
    {"path": "invoice.payment_terms", "label": "Payment terms", "group": "Invoice", "rule": "O", "max": 300},
    {"path": "invoice.bill_reference", "label": "Bill reference number", "group": "Invoice", "rule": "O", "max": 150},
    {"path": "supplier.name", "label": "Supplier name", "group": "Supplier", "rule": "M", "max": 300},
    {"path": "supplier.tin", "label": "Supplier TIN", "group": "Supplier", "rule": "M", "max": 14},
    {"path": "supplier.id_type", "label": "Supplier ID type", "group": "Supplier", "rule": "M", "max": 8,
     "options": {"BRN": "Business registration (BRN)", "NRIC": "MyKad / MyPR / MyKAS", "PASSPORT": "Passport", "ARMY": "MyTentera"}},
    {"path": "supplier.id_value", "label": "Supplier registration / ID number", "group": "Supplier", "rule": "M", "max": 20},
    {"path": "supplier.sst", "label": "Supplier SST registration no. (NA if not registered)", "group": "Supplier", "rule": "M", "max": 35},
    {"path": "supplier.ttx", "label": "Supplier tourism tax no. (NA if not registered)", "group": "Supplier", "rule": "M", "max": 17},
    {"path": "supplier.msic", "label": "Supplier MSIC code", "group": "Supplier", "rule": "M", "max": 5},
    {"path": "supplier.activity", "label": "Supplier business activity description", "group": "Supplier", "rule": "M", "max": 300},
    {"path": "supplier.phone", "label": "Supplier contact number", "group": "Supplier", "rule": "M", "max": 20},
    {"path": "supplier.email", "label": "Supplier e-mail", "group": "Supplier", "rule": "O", "max": 320},
    {"path": "supplier.address.line0", "label": "Supplier address line", "group": "Supplier", "rule": "M", "max": 150},
    {"path": "supplier.address.line1", "label": "Supplier address line 2", "group": "Supplier", "rule": "O", "max": 150},
    {"path": "supplier.address.line2", "label": "Supplier address line 3", "group": "Supplier", "rule": "O", "max": 150},
    {"path": "supplier.address.postal", "label": "Supplier postcode", "group": "Supplier", "rule": "O", "max": 50},
    {"path": "supplier.address.city", "label": "Supplier city", "group": "Supplier", "rule": "M", "max": 50},
    {"path": "supplier.address.state", "label": "Supplier state", "group": "Supplier", "rule": "M", "max": 2, "options": STATE_CODES},
    {"path": "supplier.address.country", "label": "Supplier country (ISO 3166-1 alpha-3)", "group": "Supplier", "rule": "M", "max": 3},
    {"path": "buyer.name", "label": "Buyer name", "group": "Buyer", "rule": "M", "max": 300},
    {"path": "buyer.tin", "label": "Buyer TIN", "group": "Buyer", "rule": "M", "max": 14},
    {"path": "buyer.id_type", "label": "Buyer ID type", "group": "Buyer", "rule": "M", "max": 8,
     "options": {"BRN": "Business registration (BRN)", "NRIC": "MyKad / MyPR / MyKAS", "PASSPORT": "Passport", "ARMY": "MyTentera"}},
    {"path": "buyer.id_value", "label": "Buyer registration / ID number (NA for consolidated)", "group": "Buyer", "rule": "M", "max": 20},
    {"path": "buyer.sst", "label": "Buyer SST registration no. (NA if none)", "group": "Buyer", "rule": "M", "max": 35},
    {"path": "buyer.phone", "label": "Buyer contact number (NA for consolidated)", "group": "Buyer", "rule": "M", "max": 20},
    {"path": "buyer.email", "label": "Buyer e-mail", "group": "Buyer", "rule": "O", "max": 320},
    {"path": "buyer.address.line0", "label": "Buyer address line (NA for consolidated)", "group": "Buyer", "rule": "M", "max": 150},
    {"path": "buyer.address.line1", "label": "Buyer address line 2", "group": "Buyer", "rule": "O", "max": 150},
    {"path": "buyer.address.line2", "label": "Buyer address line 3", "group": "Buyer", "rule": "O", "max": 150},
    {"path": "buyer.address.postal", "label": "Buyer postcode", "group": "Buyer", "rule": "O", "max": 50},
    {"path": "buyer.address.city", "label": "Buyer city", "group": "Buyer", "rule": "M", "max": 50},
    {"path": "buyer.address.state", "label": "Buyer state", "group": "Buyer", "rule": "M", "max": 2, "options": STATE_CODES},
    {"path": "buyer.address.country", "label": "Buyer country (ISO 3166-1 alpha-3)", "group": "Buyer", "rule": "M", "max": 3},
    {"path": "totals.total_excl_tax", "label": "Total excluding tax", "group": "Totals", "rule": "M", "max": 26},
    {"path": "totals.tax_amount", "label": "Total tax amount", "group": "Totals", "rule": "M", "max": 26},
    {"path": "totals.total_incl_tax", "label": "Total including tax", "group": "Totals", "rule": "M", "max": 26},
    {"path": "totals.discount", "label": "Invoice additional discount", "group": "Totals", "rule": "O", "max": 26},
    {"path": "totals.charge", "label": "Invoice additional fee / charge", "group": "Totals", "rule": "O", "max": 26},
    {"path": "totals.rounding", "label": "Rounding amount", "group": "Totals", "rule": "O", "max": 26},
    {"path": "totals.payable", "label": "Total payable amount", "group": "Totals", "rule": "M", "max": 26},
]
LINE_SPEC: list[dict[str, Any]] = [
    {"key": "classification", "label": "Classification", "rule": "M", "options": CLASSIFICATION_CODES},
    {"key": "description", "label": "Description", "rule": "M", "max": 300},
    {"key": "quantity", "label": "Qty", "rule": "O"},
    {"key": "unit_code", "label": "Unit (UN/ECE)", "rule": "O", "max": 3},
    {"key": "unit_price", "label": "Unit price", "rule": "M"},
    {"key": "discount", "label": "Discount", "rule": "O"},
    {"key": "subtotal", "label": "Subtotal", "rule": "M"},
    {"key": "total_excl_tax", "label": "Total excl. tax", "rule": "M"},
    {"key": "tax_type", "label": "Tax type", "rule": "M", "options": TAX_TYPES},
    {"key": "tax_rate", "label": "Tax rate %", "rule": "C"},
    {"key": "tax_amount", "label": "Tax amount", "rule": "M"},
    {"key": "exemption_reason", "label": "Exemption reason", "rule": "C", "max": 300},
]


# ------------------------------------------------------------------ helpers
def _get(data: dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _dec(value: Any) -> Decimal | None:
    if _blank(value):
        return None
    try:
        return Decimal(str(value).replace(",", "").replace("RM", "").strip())
    except InvalidOperation:
        return None


def _money(value: Decimal | None) -> str | None:
    return None if value is None else str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def normalise_phone(value: str | None) -> str | None:
    """Malaysian numbers to E.164 (+60...). Returns None if not a phone number."""
    if _blank(value):
        return None
    first = re.split(r"[,/;]|\bfax\b", str(value), flags=re.I)[0]
    digits = re.sub(r"\D", "", first)
    if str(value).strip().startswith("+") and 8 <= len(digits) <= 15:
        return "+" + digits
    if digits.startswith("60") and 10 <= len(digits) <= 12:
        return "+" + digits
    if digits.startswith("0") and 9 <= len(digits) <= 11:
        return "+60" + digits[1:]
    return None


def normalise_date(value: str | None) -> str | None:
    if _blank(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    match = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", text)
    if match:
        day, month, year = (int(part) for part in match.groups())
        year = year + 2000 if year < 100 else year
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None
    for pattern in ("%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d %b %y"):
        try:
            return datetime.strptime(text.title(), pattern).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _state_from_text(text: str | None) -> str | None:
    if _blank(text):
        return None
    lower = str(text).lower()
    for pattern, code in _STATE_WORDS:
        if re.search(pattern, lower):
            return code
    match = re.search(r"\b(\d{5})\b", lower)
    if match:
        prefix = int(match.group(1)[:2])
        for prefixes, code in _POSTCODE_STATES:
            if prefix in prefixes:
                return code
    return None


def parse_address(address: str | None) -> dict[str, str | None]:
    """Split a printed Malaysian address into MyInvois address components."""
    result: dict[str, str | None] = {"line0": None, "line1": None, "line2": None, "postal": None,
                                     "city": None, "state": None, "country": "MYS"}
    if _blank(address):
        return result
    text = re.sub(r"\s+", " ", str(address)).strip(" ,.")
    result["state"] = _state_from_text(text)
    postcode = re.search(r"\b(\d{5})\b", text)
    before = text
    if postcode:
        result["postal"] = postcode.group(1)
        before = text[:postcode.start()].strip(" ,.")
        after = text[postcode.end():].strip(" ,.")
        city = re.split(r",", after)[0].strip(" ,.") if after else ""
        city = re.split(r"(?i)\.\s|\bno\b\.?\s*:|\btel\b|\bfax\b|\bemail\b", city)[0]
        city = re.sub(r"(?i)\b(selangor|darul\s+ehsan|johor|darul\s+takzim|w\.?p\.?)\b.*$", "", city).strip(" ,.")
        if city:
            result["city"] = city.title()[:50]
    if not result["city"]:
        for pattern, label in ((r"kuala\s+lumpur", "Kuala Lumpur"), (r"putrajaya", "Putrajaya"), (r"labuan", "Labuan")):
            if re.search(pattern, text, re.I):
                result["city"] = label
    parts = [part.strip() for part in re.split(r",", before) if part.strip()]
    lines: list[str] = []
    for part in parts:
        if lines and len(lines[-1]) + len(part) + 2 <= 60:
            lines[-1] = f"{lines[-1]}, {part}"
        else:
            lines.append(part)
    if len(lines) > 3:
        lines = lines[:2] + [", ".join(lines[2:])]
    for index, line in enumerate(lines[:3]):
        result[f"line{index}"] = line[:150]
    return result


def _registration(value: str | None, text: str) -> tuple[str | None, str | None]:
    """Prefer the 12-digit SSM number (new format), else the old company number."""
    for source in (value or "", text or ""):
        match = re.search(r"(?<!\d)((?:19|20)\d{2}0[1-6]\d{6})(?!\d)", source)
        if match:
            return "BRN", match.group(1)
    if not _blank(value):
        cleaned = re.sub(r"[^A-Z0-9-]", "", str(value).upper())
        if 4 <= len(cleaned) <= 20:
            return "BRN", cleaned
    return None, None


def _find(pattern: str, text: str, flags: int = re.I) -> str | None:
    match = re.search(pattern, text or "", flags)
    return match.group(1).strip() if match else None


def _payment_mode(value: str | None, text: str) -> str | None:
    evidence = f"{value or ''} {text or ''}".lower()
    for pattern, code in ((r"credit\s*card|visa|master\s*card|amex", "04"), (r"debit\s*card|mydebit", "05"),
                          (r"e-?wallet|touch\s*'?n\s*go|tng|grab\s*pay|boost|shopee\s*pay|duitnow\s*qr", "06"),
                          (r"cheque|check\s+no", "02"), (r"bank\s*transfer|online\s*transfer|ibg|duitnow\s*transfer", "03"),
                          (r"\bcash\b", "01")):
        if re.search(pattern, evidence):
            return code
    return None


def _unit_code(value: str | None) -> str | None:
    if _blank(value):
        return None
    key = re.sub(r"[^A-Z]", "", str(value).upper())
    return UNIT_CODES.get(key)


def _clean_supplier_name(value: str | None) -> str | None:
    if _blank(value):
        return None
    name = re.sub(r"\s*\(?\b(?:19|20)\d{10}\b\)?", "", str(value))          # 12-digit SSM number
    name = re.sub(r"\s*\(\s*[0-9]{4,}[- ]?[A-Z]?\s*\)", "", name, flags=re.I)  # (268499-X)
    name = re.sub(r"\s*\(\s*[0-9]{4,}[- ]?[A-Z]\s*$", "", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip(" .,-") or None


def _clean_email(value: str | None) -> str | None:
    if _blank(value):
        return None
    match = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", str(value))
    if not match:
        return None
    return re.sub(r"(?i)^e-?mail[:.]?", "", match.group(1))


def _first_phone(value: str | None) -> str | None:
    if _blank(value):
        return None
    match = re.search(r"(?:\+?60[- ]?|0)\d{1,2}[- ]?\d{3,4}[- ]?\d{4}", str(value))
    return normalise_phone(match.group(0)) if match else normalise_phone(value)


def _box_bounds(box: Any) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(point[0]) for point in box]; ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _row_numbers(description: str, ocr_lines: list[dict[str, Any]]) -> list[Decimal]:
    """Numbers printed on the same physical row as an item description, left to right."""
    if _blank(description) or not ocr_lines:
        return []
    probe = re.sub(r"[^a-z0-9]", "", description.lower().split("/")[0])[:10]
    if len(probe) < 3:
        return []
    anchors = [line for line in ocr_lines
               if probe[:6] in re.sub(r"[^a-z0-9]", "", str(line.get("text", "")).lower()) and _box_bounds(line.get("box"))]
    if not anchors:
        return []
    x0, y0, x1, y1 = _box_bounds(anchors[0]["box"])
    centre, height = (y0 + y1) / 2, max(4.0, y1 - y0)
    found: list[tuple[float, Decimal]] = []
    for line in ocr_lines:
        bounds = _box_bounds(line.get("box"))
        if not bounds or abs((bounds[1] + bounds[3]) / 2 - centre) > 0.6 * height:
            continue
        text = str(line.get("text", ""))
        for match in re.finditer(r"(?<![A-Za-z0-9])(\d{1,7}(?:,\d{3})*(?:\.\d{1,3})?)(?![A-Za-z0-9])", text):
            value = _dec(match.group(1))
            if value is None:
                continue
            # position of the number inside the OCR box, proportional to characters
            x = bounds[0] + (bounds[2] - bounds[0]) * match.start() / max(1, len(text))
            if not any(abs(x - other_x) < 0.6 * height and value == other for other_x, other in found):
                found.append((x, value))
    return [value for _, value in sorted(found, key=lambda item: item[0])]


def _reconcile_line(row: dict[str, Any], ocr_lines: list[dict[str, Any]]) -> dict[str, Decimal | None]:
    """Choose quantity, unit price, discount and amount that agree arithmetically."""
    quantity, unit_price = _dec(row.get("quantity")), _dec(row.get("unit_price"))
    discount, amount = _dec(row.get("discount")), _dec(row.get("amount"))
    if quantity is not None and unit_price is not None and amount is not None \
            and abs(quantity * unit_price - (discount or 0) - amount) <= Decimal("0.01"):
        return {"quantity": quantity, "unit_price": unit_price, "discount": discount, "amount": amount}
    numbers = _row_numbers(str(row.get("description") or ""), ocr_lines)
    known = [value for value in (quantity, unit_price, discount, amount) if value is not None]
    pool = numbers or known
    best = None
    for k, a in enumerate(pool):
        for i, q in enumerate(pool[:k]):
            for j in range(i + 1, k):
                p = pool[j]
                if q <= 0 or p <= 0 or q > 100000:
                    continue
                if abs(q * p - a) <= Decimal("0.01"):
                    score = (k == len(pool) - 1, -abs(j - i - 1), k)
                    if best is None or score > best[0]:
                        best = (score, q, p, None, a)
                for d in pool:
                    if 0 < d < q * p and abs(q * p - d - a) <= Decimal("0.01") and d not in (q, p, a):
                        score = (k == len(pool) - 1, -1, k)
                        if best is None or score > best[0]:
                            best = (score, q, p, d, a)
    if best:
        return {"quantity": best[1], "unit_price": best[2], "discount": best[3], "amount": best[4]}
    if amount is None and unit_price is not None and quantity is not None:
        amount = quantity * unit_price - (discount or 0)
    if quantity is None and amount is not None and unit_price is not None and unit_price != 0:
        ratio = amount / unit_price
        if ratio == ratio.to_integral_value() and ratio > 0:
            quantity = ratio
    if unit_price is None and amount is not None and quantity:
        unit_price = (amount + (discount or 0)) / quantity
    if discount is not None and amount is not None and quantity is not None and unit_price is not None \
            and abs(quantity * unit_price - discount - amount) > Decimal("0.01"):
        discount = None   # a mis-read column (e.g. the amount read as a discount)
    return {"quantity": quantity, "unit_price": unit_price, "discount": discount, "amount": amount}


# --------------------------------------------------------------- suggestion
def empty_draft() -> dict[str, Any]:
    address = {"line0": None, "line1": None, "line2": None, "postal": None, "city": None, "state": None, "country": "MYS"}
    return {
        "version": 1,
        "mode": "individual",
        "invoice": {"number": None, "type_code": "01", "currency": "MYR", "exchange_rate": None,
                    "transaction_date": None, "transaction_time": None, "payment_mode": None,
                    "payment_terms": None, "bill_reference": None},
        "supplier": {"name": None, "tin": None, "id_type": None, "id_value": None, "sst": None, "ttx": None,
                     "msic": None, "activity": None, "phone": None, "email": None, "address": dict(address)},
        "buyer": {"name": None, "tin": None, "id_type": None, "id_value": None, "sst": None,
                  "phone": None, "email": None, "address": dict(address)},
        "lines": [],
        "totals": {"total_excl_tax": None, "tax_amount": None, "total_incl_tax": None, "discount": None,
                   "charge": None, "rounding": None, "payable": None},
        "provenance": {},
    }


def suggest_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Build a draft from the reviewed receipt fields, line items and OCR text."""
    draft = empty_draft()
    source: dict[str, str] = draft["provenance"]
    accounting = receipt.get("accounting_fields") or {}
    text = receipt.get("ocr_text") or ""
    if not text and isinstance(receipt.get("audit"), dict):
        text = "\n".join(str(line.get("text", "")) for line in receipt["audit"].get("ocr_lines", []))

    def put(path: str, value: Any, origin: str = "receipt") -> None:
        if not _blank(value):
            _set(draft, path, value)
            source[path] = origin

    number = receipt.get("receipt_number") or accounting.get("document_number")
    if number and re.search(r"(?i)^(?:roc|co\.?\s*no|reg)|(?:19|20)\d{10}", str(number)):
        # a company registration number mis-read as the document number
        number = _find(r"\bNo\.?\s*[:.]\s*([A-Z]{1,4}[-.]?\s?\d{3,}[-/]?\d*)", text) or None
        number = number.replace(" ", "") if number else None
    put("invoice.number", number)
    put("invoice.transaction_date", normalise_date(receipt.get("transaction_date") or accounting.get("transaction_date")))
    time_match = re.search(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\s*([AaPp][Mm])?", text)
    if time_match:
        hour, minute, second, meridiem = time_match.groups()
        hour_value = int(hour)
        if meridiem:
            hour_value = hour_value % 12 + (12 if meridiem.lower() == "pm" else 0)
        put("invoice.transaction_time", f"{hour_value:02d}:{minute}:{second or '00'}")
    currency = (accounting.get("currency") or "MYR").upper()
    put("invoice.currency", currency if re.fullmatch(r"[A-Z]{3}", currency) else "MYR",
        "receipt" if accounting.get("currency") else "default")
    boilerplate_free = "\n".join(line for line in text.splitlines()
                                  if not re.search(r"(?i)crossed|payable\s+to|cheques?\s+should|returnable|refundable", line))
    put("invoice.payment_mode", _payment_mode(accounting.get("payment_method"), "") or _payment_mode(None, boilerplate_free))
    terms = _find(r"\bterms?\s*[:.]?\s*([A-Z][A-Z .]{1,20})", text)
    if terms and not re.search(r"(?i)condition|sale", terms):
        put("invoice.payment_terms", terms.strip(" ."))

    put("supplier.name", _clean_supplier_name(accounting.get("supplier_name") or receipt.get("merchant")))
    tin = _find(r"\b(?:TIN|tax\s*identification\s*(?:no|number)?|no\.?\s*cukai)\s*[:.]?\s*((?:IG|C|CS|D|E|F|FA|PT|TA|TC|TN|TR|TP|J|LE)\d{8,11})\b", text)
    put("supplier.tin", tin)
    registration = accounting.get("supplier_registration_number")
    if registration and not re.search(r"\d", str(registration)):
        registration = None   # OCR picked up a non-number after the label
    id_type, id_value = _registration(registration, text)
    if id_value:
        put("supplier.id_type", id_type)
        put("supplier.id_value", id_value)
    sst = _find(r"\b([A-Z]\d{2}-\d{4}-\d{8})\b", text, 0) or _find(r"\bSST\s*(?:reg(?:istration)?\.?\s*)?(?:no\.?|id)?\s*[:.]?\s*([A-Z0-9][A-Z0-9-]{8,})", text)
    if sst and len(re.findall(r"\d", sst)) < 6:
        sst = None   # SST numbers are mostly digits; this is OCR text after the label
    if sst:
        put("supplier.sst", sst.upper())
    put("supplier.phone", _first_phone(accounting.get("supplier_phone")))
    put("supplier.email", _clean_email(accounting.get("supplier_email")))
    for key, value in parse_address(accounting.get("supplier_address")).items():
        if key == "country" and not accounting.get("supplier_address"):
            continue
        put(f"supplier.address.{key}", value, "receipt" if key != "country" else "default")
    if _blank(draft["supplier"]["address"].get("state")):
        put("supplier.address.state", _state_from_text(text))

    # Buyer: a printed buyer block is rare on cash receipts; "cash sales" pads
    # name a walk-in customer. Leave buyer identity to the user/profile.
    buyer_name = _find(r"(?m)^\s*(?:M/S|Bill\s*To|Sold\s*To|Customer)\s*[:.]?\s*(.{3,80})$", text)
    if buyer_name and not re.search(r"(?i)cash\s+sale|walk.?in|general|cash\b", buyer_name):
        put("buyer.name", buyer_name.strip())

    items = receipt.get("line_items") or []
    ocr_lines = (receipt.get("audit") or {}).get("ocr_lines") or [] if isinstance(receipt.get("audit"), dict) else []
    for index, row in enumerate(items, start=1):
        fixed = _reconcile_line(row, ocr_lines)
        quantity, unit_price, amount, discount = fixed["quantity"], fixed["unit_price"], fixed["amount"], fixed["discount"]
        subtotal = (quantity * unit_price) if (quantity is not None and unit_price is not None) else amount
        excl = amount if amount is not None else (subtotal - (discount or 0) if subtotal is not None else None)
        line = {
            "classification": "022",
            "description": (row.get("description") or row.get("item") or "").strip()[:300] or None,
            "quantity": None if quantity is None else format(quantity.normalize(), "f"),
            "unit_code": _unit_code(row.get("unit")),
            "unit_price": _money(unit_price),
            "discount": _money(discount),
            "subtotal": _money(subtotal),
            "total_excl_tax": _money(excl),
            "tax_type": None, "tax_rate": None, "tax_amount": None, "exemption_reason": None,
        }
        draft["lines"].append(line)
        for key, value in line.items():
            if not _blank(value):
                source[f"lines.{index - 1}.{key}"] = "default" if key == "classification" else "receipt"

    tax = _dec(accounting.get("tax_amount"))
    payable = _dec(receipt.get("total_amount") or accounting.get("payable_amount"))
    discount = _dec(accounting.get("discount"))
    rounding = _dec(_find(r"(?:round(?:ing)?\s*adj(?:ustment)?|rounding)\s*[:.]?\s*(?:RM\s*)?(-?\d+\.\d{2})", text))
    put("totals.payable", _money(payable))
    line_values = [_dec(line["total_excl_tax"]) for line in draft["lines"]]
    if discount is not None and payable is not None and line_values and all(v is not None for v in line_values):
        if abs(sum(line_values, Decimal(0)) - discount - payable + (tax or 0)) > Decimal("0.01"):
            discount = None   # e.g. the 'Discount' column header row; it does not reconcile
    put("totals.discount", _money(discount))
    put("totals.rounding", _money(rounding))
    if tax is not None:
        put("totals.tax_amount", _money(tax))
    line_sum = sum((_dec(line["total_excl_tax"]) or Decimal(0)) for line in draft["lines"]) if draft["lines"] else None
    if tax is None and not draft["supplier"].get("sst"):
        # No SST evidence on the receipt: tax not applicable is the rule default.
        put("totals.tax_amount", "0.00", "default")
        for index, line in enumerate(draft["lines"]):
            line["tax_type"] = "06"; line["tax_amount"] = "0.00"
            source[f"lines.{index}.tax_type"] = "default"; source[f"lines.{index}.tax_amount"] = "default"
        put("supplier.sst", "NA", "default")
    if not draft["supplier"].get("ttx"):
        put("supplier.ttx", "NA", "default")
    recompute_totals(draft, keep_payable=True)
    if line_sum is None and payable is not None and not draft["lines"]:
        # No item rows: a single summarised line keeps the draft structurally complete.
        draft["lines"].append({"classification": "022", "description": "Goods / services per receipt",
                               "quantity": "1", "unit_code": "C62", "unit_price": _money(payable - (tax or 0)),
                               "discount": None, "subtotal": _money(payable - (tax or 0)),
                               "total_excl_tax": _money(payable - (tax or 0)), "tax_type": "06" if not tax else None,
                               "tax_rate": None, "tax_amount": "0.00" if not tax else None, "exemption_reason": None})
        for key in draft["lines"][0]:
            if not _blank(draft["lines"][0][key]):
                source[f"lines.0.{key}"] = "default"
        recompute_totals(draft, keep_payable=True)
    return draft


def recompute_totals(draft: dict[str, Any], keep_payable: bool = False) -> None:
    """Derive the invoice totals from line values where they are complete."""
    totals = draft.setdefault("totals", {})
    lines = draft.get("lines") or []
    values = [_dec(line.get("total_excl_tax")) for line in lines]
    taxes = [_dec(line.get("tax_amount")) for line in lines]
    if lines and all(v is not None for v in values):
        net = sum(values, Decimal(0))
        excl = net - (_dec(totals.get("discount")) or 0) + (_dec(totals.get("charge")) or 0)
        totals["total_excl_tax"] = _money(excl)
        tax = sum(taxes, Decimal(0)) if all(t is not None for t in taxes) else _dec(totals.get("tax_amount"))
        if tax is not None:
            totals["tax_amount"] = _money(tax)
            incl = excl + tax
            totals["total_incl_tax"] = _money(incl)
            computed_payable = incl + (_dec(totals.get("rounding")) or 0)
            if not keep_payable or _blank(totals.get("payable")):
                totals["payable"] = _money(computed_payable)


def apply_consolidated_rules(draft: dict[str, Any]) -> None:
    """LHDN consolidated e-Invoice buyer values (General Public)."""
    buyer = draft["buyer"]
    buyer.update({"name": "General Public", "tin": GENERAL_PUBLIC_TIN, "id_type": "BRN", "id_value": "NA",
                  "sst": "NA", "phone": "NA", "email": None})
    buyer["address"] = {"line0": "NA", "line1": None, "line2": None, "postal": None, "city": "NA",
                        "state": "17", "country": "MYS"}
    for line in draft["lines"]:
        line["classification"] = "004"
    for path in ("buyer.name", "buyer.tin", "buyer.id_type", "buyer.id_value", "buyer.sst", "buyer.phone",
                 "buyer.address.line0", "buyer.address.city", "buyer.address.state", "buyer.address.country"):
        draft["provenance"][path] = "consolidated_rule"


def merge_draft(base: dict[str, Any], saved: dict[str, Any] | None, *profiles: tuple[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Saved user values win; then profiles (supplier / buyer); then receipt suggestions."""
    draft = copy.deepcopy(base)
    for prefix, profile in profiles:
        if not profile:
            continue
        for spec in FIELD_SPEC:
            if not spec["path"].startswith(prefix + "."):
                continue
            value = _get({prefix: profile}, spec["path"])
            if not _blank(value) and (_blank(_get(draft, spec["path"])) or draft["provenance"].get(spec["path"]) == "default"):
                _set(draft, spec["path"], value)
                draft["provenance"][spec["path"]] = "profile"
    if saved:
        user_paths = {path for path, origin in (saved.get("provenance") or {}).items() if origin == "user"}
        for key in ("mode", "invoice", "supplier", "buyer", "totals"):
            if key in saved:
                if isinstance(saved[key], dict):
                    _deep_update(draft[key], saved[key], key, user_paths)
                else:
                    draft[key] = saved[key]
        if isinstance(saved.get("lines"), list):
            draft["lines"] = saved["lines"]
        draft["provenance"].update(saved.get("provenance") or {})
    if draft.get("mode") == "consolidated":
        apply_consolidated_rules(draft)
    return draft


def _deep_update(target: dict[str, Any], update: dict[str, Any], prefix: str = "", user_paths: set[str] | None = None) -> None:
    """Apply saved values; a blank only clears a value the user cleared explicitly."""
    for key, value in update.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value, path, user_paths)
        elif not _blank(value) or (user_paths is not None and path in user_paths) or user_paths is None:
            target[key] = value


# ---------------------------------------------------------------- validation
_TIN = re.compile(r"^(?:IG\d{9,11}|(?:C|CS|D|E|F|FA|PT|TA|TC|TN|TR|TP|J|LE)\d{8,11}|EI\d{11})$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_LIMITED = re.compile(r"^[A-Za-z0-9 ,.()&/'-]*$")


def validate_draft(draft: dict[str, Any]) -> dict[str, Any]:
    """Return issues (errors block export; warnings ask for a check) and readiness."""
    issues: list[dict[str, str]] = []
    consolidated = draft.get("mode") == "consolidated"

    def issue(path: str, message: str, severity: str = "error") -> None:
        issues.append({"path": path, "severity": severity, "message": message})

    for spec in FIELD_SPEC:
        value = _get(draft, spec["path"])
        if spec["rule"] == "M" and _blank(value):
            issue(spec["path"], f"{spec['label']} is mandatory.")
            continue
        if not _blank(value) and spec.get("max") and len(str(value)) > spec["max"]:
            issue(spec["path"], f"{spec['label']} must be at most {spec['max']} characters.")
        if not _blank(value) and spec.get("options") and str(value) not in spec["options"]:
            issue(spec["path"], f"{spec['label']} must be a MyInvois code.")

    invoice, supplier, buyer, totals = draft["invoice"], draft["supplier"], draft["buyer"], draft["totals"]
    if not _blank(invoice.get("currency")) and not re.fullmatch(r"[A-Z]{3}", str(invoice["currency"])):
        issue("invoice.currency", "Currency must be an ISO 4217 code such as MYR.")
    if invoice.get("currency") not in (None, "", "MYR") and _blank(invoice.get("exchange_rate")):
        issue("invoice.exchange_rate", "A foreign-currency invoice needs the exchange rate to MYR.")
    if not _blank(invoice.get("transaction_date")) and not normalise_date(invoice["transaction_date"]):
        issue("invoice.transaction_date", "Use the date format YYYY-MM-DD.")
    if not _blank(invoice.get("transaction_time")) and not re.fullmatch(r"\d{2}:\d{2}:\d{2}", str(invoice["transaction_time"])):
        issue("invoice.transaction_time", "Use the time format HH:MM:SS.")

    for party_name, party in (("supplier", supplier), ("buyer", buyer)):
        tin = party.get("tin")
        if not _blank(tin) and not _TIN.fullmatch(str(tin).upper()):
            issue(f"{party_name}.tin", "TIN format not recognised (e.g. C2584563222, IG12345678901, or a general TIN EI000000000x0).")
        id_type, id_value = party.get("id_type"), party.get("id_value")
        if not _blank(id_type) and id_type not in ID_TYPES:
            issue(f"{party_name}.id_type", "ID type must be BRN, NRIC, PASSPORT or ARMY.")
        elif not _blank(id_value) and str(id_value) != "NA" and id_type in ID_TYPES and len(str(id_value)) > ID_TYPES[id_type]:
            issue(f"{party_name}.id_value", f"{id_type} numbers are at most {ID_TYPES[id_type]} characters.")
        if id_type == "NRIC" and not _blank(id_value) and str(id_value) != "NA" and not re.fullmatch(r"\d{12}", str(id_value)):
            issue(f"{party_name}.id_value", "MyKad / MyPR / MyKAS numbers are 12 digits without dashes.")
        if id_type == "BRN" and not _blank(id_value) and re.fullmatch(r"\d+", str(id_value)) and len(str(id_value)) not in (12,):
            issue(f"{party_name}.id_value", "New-format SSM registration numbers have 12 digits; check the number.", "warning")
        sst = party.get("sst")
        if not _blank(sst) and sst != "NA" and (not re.fullmatch(r"[A-Za-z0-9;-]+", str(sst)) or str(sst).count(";") > 1):
            issue(f"{party_name}.sst", "SST number allows letters, digits, '-' and at most one ';'.")
        phone = party.get("phone")
        if not _blank(phone) and not (phone == "NA" and party_name == "buyer" and consolidated):
            if not re.fullmatch(r"\+?\d{8,15}", str(phone).replace("-", "")):
                issue(f"{party_name}.phone", "Contact number must be in E.164 form, e.g. +60123456789.")
        email = party.get("email")
        if not _blank(email) and not _EMAIL.fullmatch(str(email)):
            issue(f"{party_name}.email", "E-mail address is not valid.")
        address = party.get("address") or {}
        country = address.get("country")
        if not _blank(country) and not re.fullmatch(r"[A-Z]{3}", str(country)):
            issue(f"{party_name}.address.country", "Country must be an ISO 3166-1 alpha-3 code, e.g. MYS.")
        if country == "MYS" and not _blank(address.get("state")) and address.get("state") not in STATE_CODES:
            issue(f"{party_name}.address.state", "Malaysian addresses need a MyInvois state code.")
    if not _blank(supplier.get("msic")) and not re.fullmatch(r"\d{5}", str(supplier["msic"])):
        issue("supplier.msic", "MSIC code must be 5 digits.")
    ttx = supplier.get("ttx")
    if not _blank(ttx) and ttx != "NA" and not re.fullmatch(r"[A-Za-z0-9-]+", str(ttx)):
        issue("supplier.ttx", "Tourism tax number allows letters, digits and '-'.")
    if supplier.get("tin") and str(supplier.get("tin")).upper().startswith("EI") and str(supplier["tin"]).upper() != "EI00000000030":
        issue("supplier.tin", "A general TIN cannot be used for the supplier of a normal e-Invoice.", "warning")
    if consolidated:
        if buyer.get("tin") != GENERAL_PUBLIC_TIN or buyer.get("name") != "General Public":
            issue("buyer.tin", "Consolidated e-Invoices use buyer 'General Public' and TIN EI00000000010.")
    elif buyer.get("tin") == GENERAL_PUBLIC_TIN and _blank(buyer.get("id_value")):
        issue("buyer.id_value", "With the general public TIN, enter the buyer's MyKad / ID number.", "warning")

    lines = draft.get("lines") or []
    if not lines:
        issue("lines", "At least one invoice line is required.")
    for index, line in enumerate(lines):
        prefix = f"lines.{index}"
        for spec in LINE_SPEC:
            value = line.get(spec["key"])
            if spec["rule"] == "M" and _blank(value):
                issue(f"{prefix}.{spec['key']}", f"Line {index + 1}: {spec['label']} is mandatory.")
            elif not _blank(value) and spec.get("options") and str(value) not in spec["options"]:
                issue(f"{prefix}.{spec['key']}", f"Line {index + 1}: {spec['label']} must be a MyInvois code.")
            elif not _blank(value) and spec.get("max") and len(str(value)) > spec["max"]:
                issue(f"{prefix}.{spec['key']}", f"Line {index + 1}: {spec['label']} is too long.")
        for key in ("quantity", "unit_price", "discount", "subtotal", "total_excl_tax", "tax_rate", "tax_amount"):
            if not _blank(line.get(key)) and _dec(line[key]) is None:
                issue(f"{prefix}.{key}", f"Line {index + 1}: {key.replace('_', ' ')} must be a number.")
        if consolidated and line.get("classification") != "004":
            issue(f"{prefix}.classification", f"Line {index + 1}: consolidated e-Invoices use classification 004.")
        if line.get("tax_type") == "E" and _blank(line.get("exemption_reason")):
            issue(f"{prefix}.exemption_reason", f"Line {index + 1}: a tax exemption needs the exemption reason.")
        if line.get("tax_type") in {"01", "02", "03", "04", "05"} and _blank(line.get("tax_rate")):
            issue(f"{prefix}.tax_rate", f"Line {index + 1}: enter the tax rate for this tax type.")
        if line.get("tax_type") == "06" and (_dec(line.get("tax_amount")) or 0) != 0:
            issue(f"{prefix}.tax_amount", f"Line {index + 1}: tax type 'Not Applicable' cannot carry tax.")
        qty, price, sub = _dec(line.get("quantity")), _dec(line.get("unit_price")), _dec(line.get("subtotal"))
        if qty is not None and price is not None and sub is not None and abs(qty * price - sub) > Decimal("0.01"):
            issue(f"{prefix}.subtotal", f"Line {index + 1}: quantity x unit price ({_money(qty * price)}) differs from the subtotal.", "warning")
        excl, disc = _dec(line.get("total_excl_tax")), _dec(line.get("discount")) or Decimal(0)
        if sub is not None and excl is not None and abs(sub - disc - excl) > Decimal("0.01"):
            issue(f"{prefix}.total_excl_tax", f"Line {index + 1}: subtotal minus discount ({_money(sub - disc)}) differs from the line total.", "warning")

    for key in ("total_excl_tax", "tax_amount", "total_incl_tax", "discount", "charge", "rounding", "payable"):
        if not _blank(totals.get(key)) and _dec(totals[key]) is None:
            issue(f"totals.{key}", f"{key.replace('_', ' ').capitalize()} must be a number.")
    line_totals = [_dec(line.get("total_excl_tax")) for line in lines]
    excl, tax, incl, payable = (_dec(totals.get(k)) for k in ("total_excl_tax", "tax_amount", "total_incl_tax", "payable"))
    if lines and all(v is not None for v in line_totals) and excl is not None:
        expected = sum(line_totals, Decimal(0)) - (_dec(totals.get("discount")) or 0) + (_dec(totals.get("charge")) or 0)
        if abs(expected - excl) > Decimal("0.01"):
            issue("totals.total_excl_tax", f"Line totals give {_money(expected)} excluding tax, not {_money(excl)}.")
    if excl is not None and tax is not None and incl is not None and abs(excl + tax - incl) > Decimal("0.01"):
        issue("totals.total_incl_tax", f"Total excluding tax plus tax is {_money(excl + tax)}, not {_money(incl)}.")
    if incl is not None and payable is not None:
        expected_payable = incl + (_dec(totals.get("rounding")) or 0)
        if abs(expected_payable - payable) > Decimal("0.01"):
            issue("totals.payable", f"Total including tax plus rounding is {_money(expected_payable)}, but the receipt total is {_money(payable)}. Check line items, discount, tax and rounding.")
    line_taxes = [_dec(line.get("tax_amount")) for line in lines]
    if lines and all(t is not None for t in line_taxes) and tax is not None and abs(sum(line_taxes, Decimal(0)) - tax) > Decimal("0.01"):
        issue("totals.tax_amount", "Line tax amounts do not add up to the total tax.")

    mandatory_paths = [spec["path"] for spec in FIELD_SPEC if spec["rule"] == "M"]
    filled = sum(1 for path in mandatory_paths if not _blank(_get(draft, path)))
    line_mandatory = [spec["key"] for spec in LINE_SPEC if spec["rule"] == "M"]
    total_required = len(mandatory_paths) + len(line_mandatory) * max(1, len(lines))
    filled += sum(1 for line in lines for key in line_mandatory if not _blank(line.get(key)))
    errors = [item for item in issues if item["severity"] == "error"]
    needs_input = sorted({item["path"] for item in errors if item["message"].endswith("is mandatory.")})
    return {
        "ready": not errors,
        "issues": issues,
        "error_count": len(errors),
        "warning_count": len(issues) - len(errors),
        "mandatory_filled": filled,
        "mandatory_total": total_required,
        "needs_input": needs_input,
    }


# ---------------------------------------------------------------- UBL JSON
_SCHEME = [{"ID": [{"_": "OTH", "schemeID": "UN/ECE 5153", "schemeAgencyID": "6"}]}]


def _v(value: Any, **attributes: Any) -> list[dict[str, Any]]:
    node = {"_": value}
    node.update({key: val for key, val in attributes.items() if val is not None})
    return [node]


def _amount(value: Any, currency: str) -> list[dict[str, Any]]:
    return _v(float(_dec(value) or 0), currencyID=currency)


def _party(party: dict[str, Any], supplier: bool) -> dict[str, Any]:
    address = party.get("address") or {}
    lines = [address.get(key) for key in ("line0", "line1", "line2") if not _blank(address.get(key))]
    postal: dict[str, Any] = {"CityName": _v(address.get("city"))}
    if not _blank(address.get("postal")):
        postal["PostalZone"] = _v(address.get("postal"))
    postal["CountrySubentityCode"] = _v(address.get("state"))
    postal["AddressLine"] = [{"Line": _v(line)} for line in lines]
    postal["Country"] = [{"IdentificationCode": _v(address.get("country"), listID="ISO3166-1", listAgencyID="6")}]
    identification = [{"ID": _v(str(party.get("tin")).upper(), schemeID="TIN")},
                      {"ID": _v(party.get("id_value"), schemeID=party.get("id_type"))},
                      {"ID": _v(party.get("sst") or "NA", schemeID="SST")}]
    if supplier:
        identification.append({"ID": _v(party.get("ttx") or "NA", schemeID="TTX")})
    contact: dict[str, Any] = {"Telephone": _v(party.get("phone"))}
    if not _blank(party.get("email")):
        contact["ElectronicMail"] = _v(party.get("email"))
    body: dict[str, Any] = {}
    if supplier:
        body["IndustryClassificationCode"] = _v(party.get("msic"), name=party.get("activity"))
    body.update({"PartyIdentification": identification, "PostalAddress": [postal],
                 "PartyLegalEntity": [{"RegistrationName": _v(party.get("name"))}], "Contact": [contact]})
    return {"Party": [body]}


def build_ubl_json(draft: dict[str, Any], issued_at: datetime | None = None) -> dict[str, Any]:
    """Serialise a validated draft as an unsigned MyInvois Invoice v1.0 UBL JSON document."""
    issued_at = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    invoice, totals = draft["invoice"], draft["totals"]
    currency = invoice.get("currency") or "MYR"
    tax_groups: dict[str, dict[str, Decimal]] = {}
    lines_json = []
    for index, line in enumerate(draft["lines"], start=1):
        tax_type = line.get("tax_type") or "06"
        excl = _dec(line.get("total_excl_tax")) or Decimal(0)
        tax_amount = _dec(line.get("tax_amount")) or Decimal(0)
        group = tax_groups.setdefault(tax_type, {"taxable": Decimal(0), "tax": Decimal(0)})
        group["taxable"] += excl; group["tax"] += tax_amount
        category: dict[str, Any] = {"ID": _v(tax_type)}
        if tax_type == "E" and line.get("exemption_reason"):
            category["TaxExemptionReason"] = _v(line["exemption_reason"])
        category["TaxScheme"] = _SCHEME
        subtotal: dict[str, Any] = {"TaxableAmount": _amount(excl, currency), "TaxAmount": _amount(tax_amount, currency)}
        if not _blank(line.get("tax_rate")):
            subtotal["Percent"] = _v(float(_dec(line["tax_rate"]) or 0))
        subtotal["TaxCategory"] = [category]
        entry: dict[str, Any] = {"ID": _v(str(index))}
        if not _blank(line.get("quantity")):
            entry["InvoicedQuantity"] = _v(float(_dec(line["quantity"]) or 0), unitCode=line.get("unit_code") or None)
        entry["LineExtensionAmount"] = _amount(excl, currency)
        if not _blank(line.get("discount")) and (_dec(line["discount"]) or 0) != 0:
            entry["AllowanceCharge"] = [{"ChargeIndicator": _v(False), "AllowanceChargeReason": _v("Discount"),
                                         "Amount": _amount(line["discount"], currency)}]
        entry["TaxTotal"] = [{"TaxAmount": _amount(tax_amount, currency), "TaxSubtotal": [subtotal]}]
        entry["Item"] = [{"CommodityClassification": [{"ItemClassificationCode": _v(line.get("classification"), listID="CLASS")}],
                          "Description": _v(line.get("description"))}]
        entry["Price"] = [{"PriceAmount": _amount(line.get("unit_price"), currency)}]
        entry["ItemPriceExtension"] = [{"Amount": _amount(line.get("subtotal"), currency)}]
        lines_json.append(entry)

    document: dict[str, Any] = {
        "ID": _v(invoice.get("number")),
        "IssueDate": _v(issued_at.strftime("%Y-%m-%d")),
        "IssueTime": _v(issued_at.strftime("%H:%M:%SZ")),
        "InvoiceTypeCode": _v(invoice.get("type_code") or "01", listVersionID="1.0"),
        "DocumentCurrencyCode": _v(currency),
        "TaxCurrencyCode": _v("MYR"),
    }
    if currency != "MYR":
        document["TaxExchangeRate"] = [{"SourceCurrencyCode": _v(currency), "TargetCurrencyCode": _v("MYR"),
                                        "CalculationRate": _v(float(_dec(invoice.get("exchange_rate")) or 0))}]
    if invoice.get("transaction_date"):
        document["InvoicePeriod"] = [{"StartDate": _v(invoice["transaction_date"]), "EndDate": _v(invoice["transaction_date"]),
                                      "Description": _v("Not Applicable")}]
    if invoice.get("bill_reference"):
        document["BillingReference"] = [{"AdditionalDocumentReference": [{"ID": _v(invoice["bill_reference"])}]}]
    document["AccountingSupplierParty"] = [_party(draft["supplier"], True)]
    document["AccountingCustomerParty"] = [_party(draft["buyer"], False)]
    if invoice.get("payment_mode"):
        document["PaymentMeans"] = [{"PaymentMeansCode": _v(invoice["payment_mode"])}]
    if invoice.get("payment_terms"):
        document["PaymentTerms"] = [{"Note": _v(invoice["payment_terms"])}]
    allowances = []
    if not _blank(totals.get("discount")) and (_dec(totals["discount"]) or 0) != 0:
        allowances.append({"ChargeIndicator": _v(False), "AllowanceChargeReason": _v("Discount"), "Amount": _amount(totals["discount"], currency)})
    if not _blank(totals.get("charge")) and (_dec(totals["charge"]) or 0) != 0:
        allowances.append({"ChargeIndicator": _v(True), "AllowanceChargeReason": _v("Charge"), "Amount": _amount(totals["charge"], currency)})
    if allowances:
        document["AllowanceCharge"] = allowances
    document["TaxTotal"] = [{
        "TaxAmount": _amount(totals.get("tax_amount"), currency),
        "TaxSubtotal": [{"TaxableAmount": _amount(group["taxable"], currency), "TaxAmount": _amount(group["tax"], currency),
                         "TaxCategory": [{"ID": _v(code), "TaxScheme": _SCHEME}]} for code, group in tax_groups.items()],
    }]
    monetary: dict[str, Any] = {
        "LineExtensionAmount": _amount(sum(((_dec(l.get("total_excl_tax")) or Decimal(0)) for l in draft["lines"]), Decimal(0)), currency),
        "TaxExclusiveAmount": _amount(totals.get("total_excl_tax"), currency),
        "TaxInclusiveAmount": _amount(totals.get("total_incl_tax"), currency),
    }
    if not _blank(totals.get("discount")):
        monetary["AllowanceTotalAmount"] = _amount(totals["discount"], currency)
    if not _blank(totals.get("charge")):
        monetary["ChargeTotalAmount"] = _amount(totals["charge"], currency)
    if not _blank(totals.get("rounding")):
        monetary["PayableRoundingAmount"] = _amount(totals["rounding"], currency)
    monetary["PayableAmount"] = _amount(totals.get("payable"), currency)
    document["LegalMonetaryTotal"] = [monetary]
    document["InvoiceLine"] = lines_json
    return {
        "_D": "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2",
        "_A": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
        "_B": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
        "Invoice": [document],
    }


def form_schema() -> dict[str, Any]:
    """Field schema and code lists for the browser form."""
    return {"fields": FIELD_SPEC, "line_fields": LINE_SPEC,
            "codes": {"state": STATE_CODES, "payment": PAYMENT_MODES, "tax": TAX_TYPES, "classification": CLASSIFICATION_CODES}}


def supplier_profile_key(name: str | None, id_value: str | None = None) -> str | None:
    if not _blank(id_value) and str(id_value) != "NA":
        return "id:" + re.sub(r"[^A-Z0-9]", "", str(id_value).upper())
    if _blank(name):
        return None
    return "name:" + re.sub(r"[^a-z0-9]", "", str(name).lower())[:80]


def profile_from_draft(draft: dict[str, Any], party: str) -> dict[str, Any]:
    keys = ("name", "tin", "id_type", "id_value", "sst", "ttx", "msic", "activity", "phone", "email", "address")
    return {key: copy.deepcopy(draft[party].get(key)) for key in keys if key in draft[party]}


# ---------------------------------------------------------------- purchase receipts (buyer side)
# For a business that collects its purchase receipts for the accountant, the
# e-invoice is the SUPPLIER's obligation. The buyer submits nothing to
# MyInvois for an ordinary purchase: it keeps the receipt (LHDN accepts normal
# receipts as supporting documents until the legislation is amended), or asks
# the supplier for an individual e-invoice if it wants one. A self-billed
# e-invoice is only issued by the buyer in the specific self-billed cases.
INDIVIDUAL_EINVOICE_THRESHOLD = Decimal("10000")

_VALIDATED_HINTS = re.compile(r"(?i)myinvois|hasil\.gov\.my|\bUUID\b|validation\s*(?:link|date|id)|validated\s+e-?invoice|unique\s+identifier")
_REQUEST_HINTS = re.compile(r"(?i)e[\s-]?invoice|einvois|e[\s-]?invois")
_DEADLINE_DAY = re.compile(r"(?i)(?:up\s*to|by|within|before)\s*(?:the\s*)?(\d{1,2})\s*(?:st|nd|rd|th)?\s*(?:day\s*)?(?:of\s*)?(?:the\s*)?(?:fol+[oa]?wing|next)\s*month")
_DEADLINE_HOURS = re.compile(r"(?i)within\s*(\d{1,3})\s*(?:hours|hrs|jam)")
_DEADLINE_DAYS = re.compile(r"(?i)within\s*(\d{1,2})\s*days?\s*(?:from|after|of)\s*(?:the\s*)?(?:date\s*of\s*)?(?:purchase|transaction|receipt|invoice)")


def _ocr_text(receipt: dict[str, Any]) -> str:
    text = receipt.get("ocr_text") or ""
    if not text and isinstance(receipt.get("audit"), dict):
        text = "\n".join(str(line.get("text", "")) for line in receipt["audit"].get("ocr_lines", []))
    return text


def _request_deadline(text: str, transaction_date: str | None) -> tuple[str | None, str | None]:
    """(deadline ISO date or None, rule as printed) for requesting an e-invoice."""
    from datetime import date, timedelta
    try:
        when = date.fromisoformat(transaction_date) if transaction_date else None
    except ValueError:
        when = None
    match = _DEADLINE_DAY.search(text)
    if match:
        day = int(match.group(1))
        rule = f"by day {day} of the month after purchase"
        if when and 1 <= day <= 28:
            year, month = (when.year + 1, 1) if when.month == 12 else (when.year, when.month + 1)
            return date(year, month, day).isoformat(), rule
        return None, rule
    match = _DEADLINE_HOURS.search(text)
    if match:
        hours = int(match.group(1))
        rule = f"within {hours} hours of purchase"
        return ((when + timedelta(days=max(1, hours // 24))).isoformat() if when else None), rule
    match = _DEADLINE_DAYS.search(text)
    if match:
        days = int(match.group(1))
        return ((when + timedelta(days=days)).isoformat() if when else None), f"within {days} days of purchase"
    if when:
        # the supplier's consolidated e-invoice is due 7 days after month end;
        # once it is submitted the sale can no longer get its own e-invoice
        year, month = (when.year + 1, 1) if when.month == 12 else (when.year, when.month + 1)
        return date(year, month, 7).isoformat(), "before the supplier consolidates the month (usually by the 7th of the next month)"
    return None, None


def assess_purchase_receipt(receipt: dict[str, Any], draft: dict[str, Any] | None = None,
                            today: str | None = None) -> dict[str, Any]:
    """What the buyer needs to do about e-invoicing for one purchase receipt.

    Uses only what the receipt shows. Nothing here asks the user to type
    supplier TIN/MSIC: those are the supplier's e-invoice fields.
    """
    from datetime import date
    draft = draft or suggest_from_receipt(receipt)
    text = _ocr_text(receipt)
    supplier = draft.get("supplier", {})
    invoice = draft.get("invoice", {})
    totals = draft.get("totals", {})
    total = _dec(totals.get("payable") or totals.get("total_incl_tax") or receipt.get("total_amount"))
    tax = _dec(totals.get("tax_amount"))
    essentials = {
        "supplier_name": supplier.get("name") or receipt.get("merchant"),
        "supplier_registration": supplier.get("id_value"),
        "supplier_tin": supplier.get("tin"),
        "supplier_sst": None if supplier.get("sst") in (None, "NA") else supplier.get("sst"),
        "receipt_number": invoice.get("number") or receipt.get("receipt_number"),
        "transaction_date": invoice.get("transaction_date") or normalise_date(receipt.get("transaction_date")),
        "total": _money(total),
        "tax": _money(tax) if tax else None,
        "payment_mode": PAYMENT_MODES.get(invoice.get("payment_mode") or "", None),
        "currency": invoice.get("currency") or "MYR",
    }
    review = [label for key, label in (("supplier_name", "supplier name"), ("transaction_date", "date"), ("total", "total amount"))
              if _blank(essentials[key])]
    validated = bool(_VALIDATED_HINTS.search(text))
    printed_request = bool(_REQUEST_HINTS.search(text)) and not validated
    large = total is not None and total > INDIVIDUAL_EINVOICE_THRESHOLD
    deadline, rule = _request_deadline(text, essentials["transaction_date"])
    if validated:
        status, label = "validated_einvoice", "Validated e-invoice shown on receipt"
        action = "Keep this document; record the e-invoice reference (UUID / QR) for the accountant."
    elif large:
        status, label = "supplier_must_issue", "Supplier must issue an e-invoice"
        action = ("Above RM10,000 a sale cannot go into the supplier's consolidated e-invoice. "
                  "Ask the supplier for the validated e-invoice and keep it with this receipt.")
    else:
        status, label = "keep_receipt", "Keep as supporting document"
        action = ("Nothing to submit to MyInvois. The supplier reports this sale; keep the receipt as proof of expense. "
                  "If you want an individual e-invoice in your business name, request it from the supplier.")
    expired = bool(deadline and today and today > deadline)
    return {
        "role": "purchase",
        "status": status,
        "label": label,
        "action": action,
        "request": {
            "possible": status != "validated_einvoice" and not expired,
            "printed_on_receipt": printed_request,
            "deadline": deadline,
            "rule": rule,
            "expired": expired,
        },
        "essentials": essentials,
        "needs_review": review,
        "self_billed_note": ("Only issue a self-billed e-invoice yourself in LHDN's self-billed cases "
                             "(e.g. foreign suppliers, agents/dealers' commission, payments to individuals not in business)."),
    }
