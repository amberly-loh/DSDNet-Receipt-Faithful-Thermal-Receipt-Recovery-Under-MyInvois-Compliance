from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any


DATE_PATTERNS = (
    # OCR may concatenate a four-digit year and its time, for example
    # ``15/01/201911:0516AM``. Accept the date only when the following digits
    # form a plausible time; do not generally truncate longer numbers.
    re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)\d{2}(?=(?:(?:[01]\d|2[0-3])(?::?[0-5]\d))|\D|$)"),
    # Thermal POS OCR commonly joins the date and time: 01-09-2420:56.
    # Accept a two-digit year only when it ends the token or is followed by a
    # valid 24-hour time, preventing the joined text from becoming year 2420.
    re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-]\d{2}(?=(?:(?:[01]\d|2[0-3]):[0-5]\d)|\D|$)"),
    re.compile(r"\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])\b"),
    # Month names, common on fuel and card receipts: ``08 APR 2023``,
    # ``08-Apr-23``, ``08APR2023``.
    re.compile(r"(?i)(?<![0-9A-Z])(?:0?[1-9]|[12]\d|3[01])[\s./-]*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEPT?|OCT|NOV|DEC)[A-Z]*[\s./,-]*(?:(?:19|20)\d{2}|\d{2})(?![0-9])"),
)
_MONTHS = {name: index for index, name in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), start=1)}
TOTAL_PATTERN = re.compile(r"(?i)\b(?:net\s+amt|grand\s+total|total\s+rm|total\s+amt|total|amount)\b[^\d]{0,16}(?:RM\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)")
MONEY_PATTERN = re.compile(r"(?i)(?:RM\s*)?([0-9][0-9,]*\.\d{2})(?!\d)")
NUMBER_PATTERN = re.compile(r"(?i)\b(?:(?:receipt|invoice|document|bill)\s*)?(?:(?:no\.?|number)|#)(?:\s*[:#-]?)\s*([A-Z0-9][A-Z0-9/-]{2,})")

TAG_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fuel", ("petronas", "shell", "caltex", "bhpetrol", "petrol", "diesel", "fuel", "pump")),
    ("hardware", ("hardware", "d.i.y", "diy", "chop hock", "tool", "drill", "chainsaw", "screw", "nail", "engine oil", "trimmer")),
    ("furniture", ("furniture", "mattress", "sofa", "carpet")),
    ("grocery", ("food mart", "grocery", "supermarket", "minimart", "mini mart")),
    ("office", ("stationery", "office", "copier", "document bag", "paper")),
    ("medical", ("pharmacy", "clinic", "medical")),
)


@dataclass
class FieldRecord:
    name: str
    value: str | None
    ocr_confidence: float | None
    localization_valid: bool
    mandatory_rule_passed: bool
    support_status: str
    decision: str
    reason: str
    retained_pixels: int = 0
    rule_status: str = "not_evaluated"
    box: Any = None
    hint: str | None = None
    suggested: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matching_lines(lines: list[dict[str, Any]], tokens: list[str]) -> list[dict[str, Any]]:
    lowered = [token.lower() for token in tokens if token]
    return [line for line in lines if any(token in str(line.get("text", "")).lower() for token in lowered)]


def _mean_confidence(lines: list[dict[str, Any]], tokens: list[str]) -> float | None:
    selected = []
    for line in _matching_lines(lines, tokens):
        value = line.get("confidence")
        if value is not None: selected.append(float(value))
    return sum(selected) / len(selected) if selected else None


def _box_for(lines: list[dict[str, Any]], tokens: list[str]) -> Any:
    matches = _matching_lines(lines, tokens)
    return matches[0].get("box") if matches else None


def _box_pixels(box: Any) -> int:
    try:
        points = box if len(box) == 4 and isinstance(box[0], (list, tuple)) else [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        xs, ys = [float(point[0]) for point in points], [float(point[1]) for point in points]
        return max(0, int((max(xs) - min(xs)) * (max(ys) - min(ys))))
    except (TypeError, ValueError, IndexError):
        return 0


def _rule_passes(name: str, value: str | None) -> bool:
    if not value: return False
    try:
        if name == "transaction_date":
            for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d", "%Y-%m-%d"):
                try: datetime.strptime(value, pattern); return True
                except ValueError: continue
            return False
        if name == "total_amount": return Decimal(value) >= 0
        if name == "receipt_invoice_number": return 3 <= len(value) <= 64 and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/-]+", value))
        if name == "merchant": return 3 <= len(value) <= 80 and any(ch.isalpha() for ch in value)
    except (InvalidOperation, ValueError):
        return False
    return False


def extract_fields(text: str, lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    # An invalid date earlier in the OCR must not hide a later valid one.
    raw_date = next((match.group(0) for pattern in DATE_PATTERNS for match in pattern.finditer(text) if _normalise_date(match.group(0))), None)
    if raw_date is None:
        raw_date = _spatial_date(lines) or _repair_date_near_label(text)
    date = _normalise_date(raw_date)
    # A literal printed identifier (for example, ``No: 61569``) is stronger
    # evidence than a nearby company-registration code.  Spatial pairing is a
    # fallback for layouts where OCR separates the label and value.
    receipt_number = _explicit_invoice_number(text) or _spatial_invoice_number(lines)
    spatial_total = _spatial_total(lines)
    total_match = _text_total(text)
    if spatial_total:
        total = f"{Decimal(spatial_total):.2f}"
    elif total_match:
        total = total_match
    else:
        total = None
    if total is None:
        total = _single_ringgit_amount(text)
    date_hint = None if date else _partial_date(text)
    merchant = _merchant_candidate(lines)
    return {
        "merchant": {"value": merchant, "confidence": _mean_confidence(lines, [merchant or ""]), "box": _box_for(lines, [merchant or ""])},
        "transaction_date": {"value": date, "confidence": _mean_confidence(lines, [raw_date or "date"]), "box": _box_for(lines, [raw_date or "date"]), "hint": date_hint},
        "receipt_invoice_number": {"value": receipt_number, "confidence": _mean_confidence(lines, [receipt_number or "receipt", "invoice", "document"]), "box": _box_for(lines, [receipt_number or "receipt", "invoice", "document"])},
        "total_amount": {"value": total, "confidence": _mean_confidence(lines, [total or "total"]), "box": _box_for(lines, [total or "total"])},
    }


def _single_ringgit_amount(text: str) -> str | None:
    """Fallback total: the only RM-prefixed amount on the receipt (fuel and
    POS receipts print "RM 10.00" above a TOTAL label OCR may garble)."""
    values = {f"{Decimal(m.group(1).replace(',', '')):.2f}" for m in re.finditer(r"(?i)RM\s*([0-9][0-9,]*\.\d{2})(?!\d)", text or "")}
    values = {v for v in values if Decimal(v) > 0}
    return values.pop() if len(values) == 1 else None


def _partial_date(text: str) -> str | None:
    """Day and month that were read when the year was not (e.g. a stain over it)."""
    match = re.search(r"(?i)(?<!\d)(0?[1-9]|[12]\d|3[01])\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*(?!\s*[0-9]{2})", text or "")
    if match:
        return f"{int(match.group(1)):02d} {match.group(2).title()}"
    return None


def infer_receipt_tags(text: str, merchant: str | None = None) -> list[str]:
    """Infer conservative, user-editable tags from observed receipt text."""
    evidence = f"{merchant or ''}\n{text or ''}".lower()
    tags = [tag for tag, tokens in TAG_RULES if any(token in evidence for token in tokens)]
    # the kind of business recognised from brand, fuel grade or items bought
    from .receipt_context import recognise_receipt
    context = recognise_receipt(text or "", None, merchant)
    kind = (context.get("business_type") or "").lower()
    for word, tag in (("petrol", "fuel"), ("electronics", "electronics"), ("supermarket", "grocery"), ("grocery", "grocery"),
                      ("convenience", "convenience"), ("restaurant", "meals"), ("cafe", "meals"), ("pharmacy", "medical"),
                      ("clinic", "medical"), ("hardware", "hardware"), ("stationery", "office"), ("bookstore", "office"),
                      ("electricity", "utilities"), ("telco", "utilities"), ("water", "utilities"), ("toll", "transport"),
                      ("parking", "transport"), ("travel", "travel"), ("hotel", "travel"), ("furniture", "furniture")):
        if word in kind:
            tags.append(tag)
            break
    if re.search(r"\b(?:exchange|trade[\s-]*in|refund|return)\b", evidence):
        tags.append("exchange-refund")
    if re.search(r"\bcash\s*sale\b", evidence):
        tags.append("cash-sale")
    elif re.search(r"\b(?:tax\s+)?invoice\b", evidence):
        tags.append("invoice")
    form_terms = sum(term in evidence for term in ("particulars", "quantity", "unit price", "issued by"))
    if form_terms >= 2:
        tags.append("handwritten-form")
    return list(dict.fromkeys(tags))[:4]


def _plausible(day) -> str | None:
    """Receipts are not dated in the future or before 2000 (OCR misreads)."""
    from datetime import date, timedelta
    if day < date(2000, 1, 1) or day > date.today() + timedelta(days=1):
        return None
    return day.isoformat()


def _normalise_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().replace(".", "/")
    named = re.fullmatch(r"(?i)(\d{1,2})[\s/,-]*([A-Z]{3})[A-Z]*[\s/,-]*(\d{2}|\d{4})", value)
    if named:
        month = _MONTHS.get(named.group(2).upper()[:3])
        year = int(named.group(3)); year += 2000 if year < 100 else 0
        try:
            return _plausible(datetime(year, month, int(named.group(1))).date()) if month else None
        except ValueError:
            return None
    for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d", "%Y-%m-%d"):
        try: return _plausible(datetime.strptime(value, pattern).date())
        except ValueError: continue
    return None


def _repair_date_near_label(text: str) -> str | None:
    """Recover a narrowly defined OCR separator error beside a Date label.

    Handwritten slashes are sometimes read as ``1``.  For example, the source
    ``13/7/24`` can become ``Date: 13/7124``.  The repair is deliberately
    limited to a Date-labelled line, requires a valid calendar date, and does
    not guess when a month is absent (``19/24`` remains unresolved).
    """
    for line in text.splitlines():
        if not re.search(r"(?i)\bdate\b", line):
            continue
        cleaned = line.replace("|", "/").replace(".", "/")
        exact = next((match.group(0) for pattern in DATE_PATTERNS for match in pattern.finditer(cleaned) if _normalise_date(match.group(0))), None)
        if exact:
            return exact
        candidate = _repair_date_token(cleaned)
        if candidate:
            return candidate
    return None


def _repair_date_token(value: str) -> str | None:
    cleaned = value.replace("|", "/").replace(".", "/")
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(cleaned):
            if _normalise_date(match.group(0)):
                return match.group(0)
    for match in re.finditer(r"(?<!\d)([0-3]?\d)[/-]([0-9])([1Il])([0-9]{2})(?!\d)", cleaned):
        candidate = f"{match.group(1)}/{match.group(2)}/{match.group(4)}"
        if _normalise_date(candidate):
            return candidate
    return None


def _spatial_date(lines: list[dict[str, Any]]) -> str | None:
    labels = [line for line in lines if re.fullmatch(r"(?i)\s*date\s*:?-?\s*", str(line.get("text", "")))]
    candidates: list[tuple[float, str]] = []
    for label in labels:
        label_box = _bounds(label.get("box"))
        if not label_box:
            continue
        _, ly1, lx2, ly2 = label_box
        lcy = (ly1 + ly2) / 2
        for line in lines:
            value_box = _bounds(line.get("box"))
            candidate = _repair_date_token(str(line.get("text", "")))
            if not value_box or not candidate:
                continue
            vx1, vy1, _, vy2 = value_box
            vcy = (vy1 + vy2) / 2
            if abs(vcy - lcy) <= 80 and vx1 >= lx2 - 10:
                candidates.append((abs(vcy - lcy) + max(0, vx1 - lx2) / 30, candidate))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _explicit_invoice_number(text: str) -> str | None:
    for line in text.splitlines():
        # Company/tax registrations are not transaction identifiers. ROC is
        # a common Malaysian label and previously won over the real Invoice No.
        if re.search(r"(?i)\b(?:company|co\.?\s*reg|registration|reg\.?|roc|business|sst|gst|tax|phone|tel|fax)\b", line):
            continue
        # product serials, IMEIs and customer/staff/member ids identify goods
        # or people, not the transaction
        if re.search(r"(?i)\b(?:ser[il1]a[l1]|imei|s/?n|customer|member|staff|card|account|acc|model|part|batch|lot)\b", line):
            continue
        match = NUMBER_PATTERN.search(line)
        if match:
            value = match.group(1)
            if any(ch.isdigit() for ch in value) and not any(pattern.fullmatch(value) for pattern in DATE_PATTERNS):
                return value
    return None


def _bounds(box: Any) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _total_priority(text: str) -> int | None:
    if re.search(r"(?i)\b(?:sub[\s-]*total|total\s+(?:discount|saving|qty|quantity|items?|item\s+count|count|tax|gst)|tax\s+total)\b", text): return None
    if re.search(r"(?i)\b(?:rounded?\s+total|total\s+(?:payable|due|amount)|amount\s+(?:payable|due|to\s+pay)|payment\s+due|balance\s+due|net\s+amt|grand\s+total|net\s+total)\b", text): return 0
    if re.search(r"(?i)\b(?:total\s+(?:sales|incl|excl|includes)|takeout\s+total)\b", text): return 2
    if re.search(r"(?i)\b(?:total|net\s+amount)\b", text): return 1
    return None


def _text_total(text: str) -> str | None:
    candidates = []
    for index, line in enumerate(text.splitlines()):
        priority = _total_priority(line)
        if priority is None: continue
        match = re.search(r"(?i)\b(?:total|amt|due|payable)\b[^\d\n]{0,20}(?:RM\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)", line)
        if match and not re.match(r"\s*%", line[match.end():]):
            candidates.append((priority, -index, f"{Decimal(match.group(1).replace(',', '')):.2f}"))
    return min(candidates)[2] if candidates else None


def _spatial_total(lines: list[dict[str, Any]]) -> str | None:
    labels = [line for line in lines if _total_priority(str(line.get("text", ""))) is not None]
    candidates: list[tuple[float, str]] = []
    for label in labels:
        label_box = _bounds(label.get("box"))
        if not label_box:
            continue
        label_text = str(label.get("text", "")).lower()
        priority = _total_priority(label_text)
        inline = _text_total(label_text)
        if inline is not None:
            candidates.append((priority * 1000, inline))
        _, ly1, lx2, ly2 = label_box
        lcy = (ly1 + ly2) / 2
        for line in lines:
            value_box = _bounds(line.get("box"))
            match = re.fullmatch(r"\s*(?:RM\s*)?-?\s*([0-9][0-9,]*(?:\.\d{1,2})?)\s*", str(line.get("text", "")), re.I)
            if not value_box or not match:
                continue
            # Multi-view OCR may produce a truncated and a complete reading
            # for the exact same handwritten amount (for example 11 and 116).
            # Prefer the longer reading when its confidence is comparable.
            digits = re.sub(r"\D", "", match.group(1))
            confidence = float(line.get("confidence") or 0)
            shadowed = False
            for other in lines:
                other_box = _bounds(other.get("box"))
                other_match = re.fullmatch(r"\s*(?:RM\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)\s*", str(other.get("text", "")), re.I)
                if not other_box or not other_match or not _same_box_region(value_box, other_box):
                    continue
                other_digits = re.sub(r"\D", "", other_match.group(1))
                other_confidence = float(other.get("confidence") or 0)
                if len(other_digits) > len(digits) and other_confidence >= confidence - 0.12:
                    shadowed = True
                    break
            if shadowed:
                continue
            vx1, vy1, _, vy2 = value_box
            vcy = (vy1 + vy2) / 2
            tolerance = max(6, .65 * max(ly2 - ly1, vy2 - vy1))
            if abs(vcy - lcy) <= tolerance and vx1 >= lx2 - 5:
                value = Decimal(match.group(1).replace(",", ""))
                if value >= 0:
                    candidates.append((priority * 1000 + abs(vcy - lcy) + max(0, vx1 - lx2) / 40, f"{value:.2f}"))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _same_box_region(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = min(max((ax2 - ax1) * (ay2 - ay1), 1.0), max((bx2 - bx1) * (by2 - by1), 1.0))
    return intersection / smaller >= 0.8


def _spatial_invoice_number(lines: list[dict[str, Any]]) -> str | None:
    labels = [line for line in lines if re.fullmatch(r"(?i)\s*(?:(?:invoice|receipt|order|bill|trans(?:action)?|doc(?:ument)?)\s*)?(?:no|no\.|number|#)\s*:?\s*", str(line.get("text", "")))]
    candidates: list[tuple[float, str]] = []
    for label in labels:
        label_box = _bounds(label.get("box"))
        if not label_box:
            continue
        _, ly1, lx2, ly2 = label_box
        lcy = (ly1 + ly2) / 2
        for line in lines:
            value_box = _bounds(line.get("box"))
            if not value_box:
                continue
            vx1, vy1, _, vy2 = value_box
            vcy = (vy1 + vy2) / 2
            value = str(line.get("text", "")).strip(" :#-")
            if abs(vcy - lcy) <= 45 and vx1 >= lx2 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 /-]{2,}", value) and any(ch.isdigit() for ch in value):
                candidates.append((abs(vcy - lcy) + (vx1 - lx2) / 20, re.sub(r"\s+", "", value)))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    # Some faded receipts lose the literal "No." label while the nearby code
    # remains readable. Rank code-like strings in the upper document region,
    # rejecting field labels and ordinary dates/times.
    positioned = [(line, _bounds(line.get("box"))) for line in lines]
    bottoms = [bounds[3] for _, bounds in positioned if bounds]
    document_bottom = max(bottoms, default=0)
    fallbacks: list[tuple[float, str]] = []
    rejected = re.compile(r"(?i)\b(?:from|date|cash|term|page|total|amount|price|tel|fax|company|invoice|receipt|document)\b")
    for line, bounds in positioned:
        value = str(line.get("text", "")).strip(" :#")
        if not bounds or bounds[1] > document_bottom * 0.22 or rejected.search(value) or re.match(r"(?i)(?:TEL|FAX|COMPANY|CO\.?\s*REG|REG\.?|GST|SST)", value) or re.search(r"(?i)(?:PCS|PACK|UNIT|KG)$", value):
            continue
        compact = re.sub(r"\s+", "", value)
        digit_count = sum(ch.isdigit() for ch in compact)
        has_letter = any(ch.isalpha() for ch in compact)
        code_shape = has_letter and bool(re.search(r"[-./]", compact))
        if 5 <= len(compact) <= 32 and digit_count >= 2 and code_shape and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./-]+", compact):
            if not any(pattern.fullmatch(compact) for pattern in DATE_PATTERNS):
                confidence = float(line.get("confidence") or 0)
                punctuation_bonus = 0.25 if re.search(r"[-./]", compact) else 0.0
                fallbacks.append((confidence + punctuation_bonus, compact))
    return max(fallbacks, default=(0, None), key=lambda item: item[0])[1]


def _merchant_candidate(lines: list[dict[str, Any]]) -> str | None:
    reject = re.compile(r"(?i)\b(?:receipt|invoice|cash\s+sale|address|tel|date|item|qty|quantity|price|amount|total|tax|sst|thank)\b")
    ordered = sorted(lines, key=lambda item: _line_position(item.get("box")))
    candidates: list[tuple[float, str]] = []
    for index, line in enumerate(ordered[:18]):
        value = str(line.get("text", "")).strip(" -:|")
        letters = sum(ch.isalpha() for ch in value)
        if 3 <= len(value) <= 80 and letters >= 3 and not reject.search(value) and not re.fullmatch(r"[\d\W]+", value):
            confidence = float(line.get("confidence") or 0)
            company_bonus = 4.0 if re.search(r"(?i)\b(?:sdn\.?\s*bhd|enterprise|trading|company|store|mart|restaurant|kedai)\b", value) else 0.0
            noise_penalty = 3.0 if letters / max(len(value), 1) < 0.55 else 0.0
            candidates.append((company_bonus + confidence * 2.0 - index * 0.08 - noise_penalty, value))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def _line_position(box: Any) -> tuple[float, float]:
    try:
        return (min(float(point[1]) for point in box), min(float(point[0]) for point in box))
    except (TypeError, ValueError, IndexError):
        return (float("inf"), float("inf"))


def route_fields(
    extracted: dict[str, dict[str, Any]],
    support_available: bool,
    model_available: bool,
) -> list[FieldRecord]:
    records = []
    for name, payload in extracted.items():
        value = payload.get("value")
        confidence = payload.get("confidence")
        box = payload.get("box")
        retained_pixels = _box_pixels(box)
        localization_valid = bool(value) and retained_pixels >= 20
        rule_passed = _rule_passes(name, value)
        if not model_available:
            reason = "Trained DSDNet checkpoint unavailable."
        elif not support_available:
            reason = "Source-trace support unavailable for a naturally degraded receipt."
        elif not localization_valid:
            reason = "Required field was not localized."
        elif not rule_passed:
            reason = "Mandatory field rule failed."
        else:
            reason = "Formal calibrated score and validated thresholds are unavailable."
        records.append(
            FieldRecord(
                name=name,
                value=value,
                ocr_confidence=confidence,
                localization_valid=localization_valid,
                mandatory_rule_passed=rule_passed,
                support_status="available" if support_available else "unavailable",
                decision="Human decision required",
                reason=reason,
                retained_pixels=retained_pixels,
                rule_status="passed" if rule_passed else "failed",
                box=box,
                hint=payload.get("hint"),
                suggested=payload.get("suggested"),
            )
        )
    return records
