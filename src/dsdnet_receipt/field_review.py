"""A second look at the date, receipt number and total, used by the app only.

`fields.extract_fields` is the extractor measured in the thesis evaluation, so it
is left exactly as evaluated. This layer runs after it in the app and corrects
three things that real receipts showed it gets wrong:

* the total: receipts may print "Total" before service charge and tax and the
  amount actually paid under another label ("SubTotal", "Nett", "Amount due"),
  or several totals. The amount paid is preferred: the total that equals the
  payment line, else the total that equals another total plus the charges,
  else a grand-total label, else the largest total.
* the date: the scanner sometimes reads a "/" as "1" ("13/1012026"), and the
  printing time at the bottom ("Printing on 18/10/2026") is not the sale date.
  A date printed with a time on a non-printing line is preferred.
* the receipt number: a labelled number ("Receipt : MYINUS-1000-20003", with
  OCR spellings such as "Reciept" or "Reaiept") is stronger evidence than an
  unlabelled code picked up near the address.

Every change is recorded with the earlier value and the reason, so it can be
shown and audited. Nothing here is used by the evaluation scripts.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from .semantic import _best_physical_lines, _bounds, _center
from .fields import extract_fields as _evaluated_extract_fields  # bound now: the evaluated extractor itself

# ------------------------------------------------------------------ amounts
_AMOUNT = re.compile(r"(?i)(?:RM\s*)?(-?\d{1,6}(?:,\d{3})*\.\d{2})(?!\d)")
_GRAND = re.compile(r"(?i)grand\s*total|total\s*rounded|rounded\s*total|total\s*payable|amount\s*payable|amount\s*due|total\s*due|net+\s*(?:total|amount|amt)|"
                    r"total\s*amount|total\s*amt|jumlah\s*(?:besar|bayaran)?|total\s*\(?incl")
_SUBTOTAL = re.compile(r"(?i)sub\s*-?\s*total")
_TOTAL = re.compile(r"(?i)\btotal\b|total\s*:")
_NOT_A_TOTAL = re.compile(r"(?i)total\s*(?:qty|quantity|item|items|disc|discount|saving|savings|tax|sst|gst|excl|before)|taxable|"
                          r"total\s*rm\s*$|no\.?\s*of\s*items")
_TENDER = re.compile(r"(?i)\b(?:cash|tunai|duit\s*now|duitnow|card|visa|master\s*card|mastercard|debit|credit|amex|union\s*pay|"
                     r"t\s*n\s*g|touch\s*n\s*go|grab\s*pay|boost|e-?wallet|shopee\s*pay|mae|fpx|qr\s*pay|paid|payment)\b")
_CHANGE = re.compile(r"(?i)\b(?:change|balance|baki)\b")
_CHARGE = re.compile(r"(?i)service\s*charge|\bsst\b|\bgst\b|\btax\b|\bsvc\b|cukai")
_ROUNDING = re.compile(r"(?i)round")


def _money(text: str) -> Decimal | None:
    found = _AMOUNT.findall(text or "")
    if not found:
        return None
    try:
        return Decimal(found[-1].replace(",", ""))
    except InvalidOperation:
        return None


def _height(line: dict[str, Any]) -> float:
    bounds = _bounds(line.get("box"))
    return max(1.0, bounds[3] - bounds[1]) if bounds else 20.0


def _row_amount(label: dict[str, Any], lines: list[dict[str, Any]]) -> tuple[Decimal | None, dict[str, Any] | None]:
    """The amount printed on the same row as a label, to its right (or inside
    it), and the OCR line it was read from."""
    inline = _money(label.get("text", ""))
    if inline is not None:
        return inline, label
    bounds = _bounds(label.get("box"))
    if not bounds:
        return None, None
    _, label_y = _center(label)
    best, best_gap, best_line = None, None, None
    for other in lines:
        if other is label:
            continue
        other_bounds = _bounds(other.get("box"))
        amount = _money(other.get("text", ""))
        if amount is None or not other_bounds or other_bounds[0] < bounds[2] - 5:
            continue
        if re.search(r"[A-Za-z]{3,}", other.get("text", "")):
            continue
        gap = abs(_center(other)[1] - label_y)
        if gap <= 0.9 * max(_height(label), _height(other)) and (best_gap is None or gap < best_gap):
            best, best_gap, best_line = amount, gap, other
    return best, best_line


def _with_source(value: str, reason: str, source: dict[Decimal, dict[str, Any]]) -> tuple[str, str, dict[str, Any] | None]:
    return value, reason, source.get(Decimal(value))


def review_total(lines: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    value, why, _ = review_total_with_source(lines)
    return value, why


def review_total_with_source(lines: list[dict[str, Any]]) -> tuple[str | None, str | None, dict[str, Any] | None]:
    physical = _best_physical_lines(lines)
    source: dict[Decimal, dict[str, Any]] = {}
    totals: list[tuple[str, Decimal, float]] = []
    tenders: list[Decimal] = []
    payment_y: float | None = None
    charges: list[Decimal] = []
    for line in physical:
        text = str(line.get("text", ""))
        if _CHANGE.search(text):
            continue
        amount, amount_line = _row_amount(line, physical)
        if amount is None or amount <= 0:
            continue
        if _NOT_A_TOTAL.search(text):
            continue
        y = _center(line)[1]
        if not _TENDER.search(text) or _TOTAL.search(text) or _GRAND.search(text) or _SUBTOTAL.search(text):
            source.setdefault(amount, amount_line)
        if _GRAND.search(text):
            totals.append(("grand", amount, y))
        elif _SUBTOTAL.search(text):
            totals.append(("subtotal", amount, y))
        elif _TOTAL.search(text):
            totals.append(("total", amount, y))
        elif _TENDER.search(text):
            tenders.append(amount)
            payment_y = y if payment_y is None else min(payment_y, y)
        elif _CHARGE.search(text) and not _ROUNDING.search(text):
            charges.append(amount)
    if not totals:
        return None, None, None
    amounts = sorted({amount for _, amount, _ in totals})
    paid = [amount for amount in amounts if amount in tenders]
    if paid:
        return _with_source(f"{max(paid):.2f}", "matches the payment line", source)
    grand = [amount for kind, amount, _ in totals if kind == "grand"]
    if grand:
        return _with_source(f"{grand[-1]:.2f}", "labelled as the final amount payable", source)
    extra = sum(charges, Decimal("0"))
    if charges and len(amounts) > 1:
        for low in amounts:
            for high in amounts:
                if high > low and abs(low + extra - high) <= Decimal("0.05"):
                    return _with_source(f"{high:.2f}", "equals the earlier total plus service charge, tax and rounding", source)
    if len(amounts) == 1:
        return _with_source(f"{amounts[0]:.2f}", "the only total printed", source)
    # several totals and nothing else to go on: the last one printed before the
    # payment lines (a tax summary under the payment repeats the pre-tax total)
    above = [item for item in totals if payment_y is None or item[2] < payment_y] or totals
    final = [item for item in above if item[0] != "subtotal"] or above
    return _with_source(f"{max(final, key=lambda item: item[2])[1]:.2f}", "the last total printed before payment", source)


# -------------------------------------------------------------------- dates
_MONTH_NAMES = {name: index for index, name in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1)}
_YEAR_END = r"(?:(?!\d)|(?=\d{1,2}\s*:\s*\d{2}))"   # a year may run straight into the time: 20263:17
_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*((?:19|20)\d{2})" + _YEAR_END)
# two-digit years only with the same separator twice (12/11/24), never 1.4-11
_DMY_SHORT = re.compile(r"(?<!\d)(\d{1,2})\s*([/.\-])\s*(\d{1,2})\s*\2\s*(\d{2})(?!\d)")
# a "/" read as 1, l or I before the year: 13/1012026 -> 13/10/2026
_DMY_MISREAD = re.compile(r"(?<!\d)(\d{1,2})\s*[/.\-]\s*(\d{1,2})[1lI|]((?:20)\d{2})" + _YEAR_END)
_YMD = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?!\d)")
_DMONY = re.compile(r"(?i)(?<!\d)(\d{1,2})\s*[-/ ]?\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*[-/ ,]?\s*((?:19|20)\d{2}|\d{2})(?!\d)")
_TIME = re.compile(r"\d{1,2}\s*:\s*\d{2}")
_PRINTED = re.compile(r"(?i)print|reprint|copy\s*printed")


def _valid(year: int, month: int, day: int, today: date) -> date | None:
    if year < 100:
        year += 2000
    try:
        value = date(year, month, day)
    except ValueError:
        return None
    return value if date(2000, 1, 1) <= value <= today + timedelta(days=1) else None


def _dates_in(text: str, today: date) -> list[date]:
    found: list[date] = []
    for pattern in (_DMY, _DMY_MISREAD):
        for d, m, y in pattern.findall(text):
            value = _valid(int(y), int(m), int(d), today)
            if value:
                found.append(value)
    for d, _, m, y in _DMY_SHORT.findall(text):
        value = _valid(int(y), int(m), int(d), today)
        if value and value.year >= today.year - 10:
            found.append(value)
    for y, m, d in _YMD.findall(text):
        value = _valid(int(y), int(m), int(d), today)
        if value:
            found.append(value)
    for d, mon, y in _DMONY.findall(text):
        value = _valid(int(y), _MONTH_NAMES[mon.upper()[:3]], int(d), today)
        if value and (len(y) == 4 or value.year >= today.year - 10):
            found.append(value)
    return found


def review_date(lines: list[dict[str, Any]], today: date | None = None) -> tuple[str | None, bool, str | None]:
    """The sale date, whether it was printed with a time, and why it was chosen."""
    value, timed, why, _ = review_date_with_source(lines, today)
    return value, timed, why


def review_date_with_source(lines: list[dict[str, Any]], today: date | None = None) -> tuple[str | None, bool, str | None, dict[str, Any] | None]:
    today = today or date.today()
    scores: dict[date, float] = {}
    origin: dict[date, tuple[float, dict[str, Any]]] = {}
    timed: set[date] = set()
    heights = [_center(line)[1] for line in lines] or [0.0]
    top, bottom = min(heights), max(heights)
    for line in lines:
        text = str(line.get("text", ""))
        if _PRINTED.search(text):
            continue
        has_time = bool(_TIME.search(text))
        upper = (_center(line)[1] - top) <= 0.6 * max(1.0, bottom - top)
        for value in _dates_in(text, today):
            scores[value] = scores.get(value, 0.0) + 1.0 + (2.0 if has_time else 0.0) + (0.5 if upper else 0.0)
            if has_time:
                timed.add(value)
            rank = (2.0 if has_time else 0.0) + float(line.get("confidence") or 0)
            if value not in origin or rank > origin[value][0]:
                origin[value] = (rank, line)
    if not scores:
        return None, False, None, None
    best = max(scores, key=lambda value: (scores[value], value in timed))
    return (best.isoformat(), best in timed, ("printed with the sale time" if best in timed else "the only plausible sale date"),
            origin[best][1])


# ----------------------------------------------------------- receipt number
_NUMBER_LABEL = re.compile(
    r"(?i)(?<![A-Za-z])(?:r[ec][a-z]{0,3}pt|rcpt|invoice|inv|bill|doc(?:ument)?|order|trans(?:action)?|txn|ref(?:erence)?|"
    r"chk|check|slip|ticket|resit)\s*(?:no|num(?:ber)?|#)?\s*\.?\s*[:#.\-]?\s*([A-Z0-9][A-Z0-9\-/]{3,24})")


def _plausible_number(value: str) -> bool:
    digits = sum(ch.isdigit() for ch in value)
    if digits < 2 or re.fullmatch(r"\d{1,2}[/.\-]\d{1,2}([/.\-]\d{2,4})?", value):
        return False
    return not re.fullmatch(r"0\d{8,10}", value)  # a phone number


_LABEL_ONLY = re.compile(r"(?i)^\W*(?:r[ec][a-z]{0,3}pt|rcpt|invoice|inv|bill|doc(?:ument)?|order|trans(?:action)?|txn|ref(?:erence)?|"
                         r"chk|check|slip|ticket|resit)\s*(?:no|num(?:ber)?|#)?\s*\.?\s*[:#.\-]?\W*$")


def review_number(lines: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    value, why, _ = review_number_with_source(lines)
    return value, why


def review_number_with_source(lines: list[dict[str, Any]]) -> tuple[str | None, str | None, dict[str, Any] | None]:
    found: dict[str, float] = {}
    origin: dict[str, dict[str, Any]] = {}

    def keep(value: str, confidence: float, line: dict[str, Any]) -> None:
        value = value.strip(" :#.-/")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/]{3,24}", value) and _plausible_number(value):
            if confidence >= found.get(value, -1.0):
                origin[value] = line
            found[value] = max(found.get(value, 0.0), confidence)

    for line in lines:
        text = str(line.get("text", ""))
        for match in _NUMBER_LABEL.finditer(text):
            keep(match.group(1), float(line.get("confidence") or 0), line)
        if _LABEL_ONLY.match(text):
            # the label and its value were read as separate boxes on one row
            bounds = _bounds(line.get("box"))
            if not bounds:
                continue
            row_y = _center(line)[1]
            for other in lines:
                other_bounds = _bounds(other.get("box"))
                if other is line or not other_bounds or other_bounds[0] < bounds[2] - 15:
                    continue
                if other_bounds[0] - bounds[2] > 6 * _height(line):
                    continue
                if abs(_center(other)[1] - row_y) <= 0.7 * max(_height(line), _height(other)):
                    keep(str(other.get("text", "")), float(other.get("confidence") or 0), other)
    if not found:
        return None, None, None
    # different views of the same print read slightly differently: keep the
    # fullest reading (most characters), then the most confident
    best = max(found, key=lambda value: (len(value), found[value]))
    return best, "printed after a receipt or invoice label", origin.get(best)


# -------------------------------------------------------------------- apply
def review_key_fields(lines: list[dict[str, Any]], extracted: dict[str, dict[str, Any]], today: date | None = None) -> dict[str, Any]:
    """Correct `extracted` in place where the receipt shows stronger evidence.
    Returns a record of every change for the audit trail."""
    changes: dict[str, Any] = {}

    def change(name: str, value: str, reason: str, line: dict[str, Any] | None) -> None:
        field = extracted.setdefault(name, {})
        previous = field.get("value")
        if previous == value:
            return
        field["value"] = value
        # the confidence and location now describe the line the new value came from
        if line is not None:
            field["confidence"] = line.get("confidence")
            field["box"] = line.get("box")
        field["reviewed_by_app"] = reason
        changes[name] = {"from": previous, "to": value, "reason": reason}

    total, why, line = review_total_with_source(lines)
    if total:
        change("total_amount", total, why, line)

    sale_date, with_time, why, line = review_date_with_source(lines, today)
    current = (extracted.get("transaction_date") or {}).get("value")
    if sale_date and (not current or with_time):
        change("transaction_date", sale_date, why, line)

    number, why, line = review_number_with_source(lines)
    current = (extracted.get("receipt_invoice_number") or {}).get("value")
    if number and (not current or current not in number):
        change("receipt_invoice_number", number, why, line)

    return {"method": "app_field_review_v1", "changes": changes}


def extract_fields_reviewed(text: str, lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The evaluated extractor followed by this review: the shared field parser
    of the 12 November 2026 parser revision (every method uses it)."""
    # always the evaluated extractor captured at import, even after a run script points
    # fields.extract_fields at this function (otherwise it would call itself)
    extracted = _evaluated_extract_fields(text, lines)
    review_key_fields(lines, extracted)
    return extracted
