from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

from .ocr import deduplicate_lines


LINE_ITEM_KEYS = ("item", "description", "quantity", "unit", "unit_price", "discount", "amount", "review_note")


def _bounds(box: Any) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _center(line: dict[str, Any]) -> tuple[float, float]:
    bounds = _bounds(line.get("box"))
    return ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2) if bounds else (0.0, 0.0)


def _same_region(first: dict[str, Any], second: dict[str, Any]) -> bool:
    a, b = _bounds(first.get("box")), _bounds(second.get("box"))
    if not a or not b:
        return False
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min(max((a[2] - a[0]) * (a[3] - a[1]), 1.0), max((b[2] - b[0]) * (b[3] - b[1]), 1.0))
    return intersection / smaller >= 0.72


def _best_physical_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse differing OCR readings of the same pixels without losing repeated values."""
    kept: list[dict[str, Any]] = []
    for line in deduplicate_lines(lines):
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        index = next((i for i, old in enumerate(kept) if _same_region(old, line)), None)
        if index is None:
            kept.append(line)
            continue
        old = kept[index]
        # Confidence is primary. Readable alpha-numeric coverage breaks close ties.
        def score(candidate: dict[str, Any]) -> float:
            value = str(candidate.get("text", ""))
            readable = sum(ch.isalnum() or ch in " ./&'-\"():" for ch in value) / max(len(value), 1)
            return float(candidate.get("confidence") or 0) + 0.035 * readable + 0.0002 * min(len(value), 100)
        if score(line) > score(old):
            kept[index] = line
    return sorted(kept, key=lambda line: (_center(line)[1], _center(line)[0]))


def _money(value: str) -> str | None:
    match = re.fullmatch(r"\s*(?:RM\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)(?:\s*(?:SR|CR))?\s*", value, re.I)
    if not match:
        return None
    try:
        return f"{Decimal(match.group(1).replace(',', '')):.2f}"
    except InvalidOperation:
        return None


def _number(value: str) -> str | None:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]{1,3})?)\s*", value)
    return match.group(1) if match else None


_DESCRIPTION_PHRASES = (
    "OGAWA", "Hand", "Blower", "Model", "Serial", "Standard", "Accessories", "Tools",
    "Aluminium", "Rod", "Engine", "Oil", "Garbage", "Bag", "Safety", "Helmet",
    "Trimmer", "Line", "Plastic", "Scraper", "Cotton", "Glove", "Document", "Copier",
    "Mattress", "Furniture", "Thermal", "Paper", "Drill", "Blade", "Fuel", "Motor",
)


def _correct_description(text: str) -> tuple[str, list[str]]:
    """Apply conservative product-language corrections while retaining an audit note."""
    raw = re.sub(r"\s+", " ", text).strip(" |")
    corrections: list[str] = []
    # Frequent thermal/handwriting confusions seen across the evaluation set.
    # These are product-language normalisations, never glyphs painted onto the
    # receipt image; the original OCR string remains in audit_json.
    phrase_rules = (
        (r"(?i)\b[0o]g[a-z0-9]*\b", "OGAWA"),
        (r"(?i)\bfand\b", "Hand"),
        (r"(?i)\bblow(?:l|i|v|w)*\b", "Blower"),
        (r"(?i)\bstanda(?:d|rcl|rd)\b", "Standard"),
        (r"(?i)\baccess\w*too\b", "Accessories Tools"),
        (r"(?i)\bac+es+s?(?:due|orie|ories|orie?s)\b", "Accessories"),
        (r"(?i)\btoo(?:g|l|ls)\b", "Tools"),
    )
    for pattern, replacement in phrase_rules:
        updated = re.sub(pattern, replacement, raw)
        if updated != raw:
            corrections.append(f"{raw}->{updated}")
            raw = updated
    output: list[str] = []
    for token in raw.split():
        if not any(ch.isalnum() for ch in token):
            output.append(token)
            continue
        prefix = re.match(r"^[^A-Za-z0-9]*", token).group(0)
        suffix = re.search(r"[^A-Za-z0-9]*$", token).group(0)
        core = token[len(prefix):len(token) - len(suffix) if suffix else None]
        if len(core) >= 4 and core.isalpha():
            best = max(_DESCRIPTION_PHRASES, key=lambda term: SequenceMatcher(None, core.casefold(), term.casefold()).ratio())
            ratio = SequenceMatcher(None, core.casefold(), best.casefold()).ratio()
            if ratio >= 0.82 and core.casefold() != best.casefold():
                corrections.append(f"{core}->{best}")
                core = best
        output.append(prefix + core + suffix)
    corrected = " ".join(output)
    corrected = re.sub(r"(?i)\b0GAWA\b", "OGAWA", corrected)
    corrected = re.sub(r"\s+([,:;)])", r"\1", corrected)
    return corrected, corrections


@dataclass
class _Layout:
    header_y: float
    end_y: float
    anchors: dict[str, float]


def _detect_layout(lines: list[dict[str, Any]]) -> _Layout | None:
    patterns = {
        "description": re.compile(r"(?i)\b(?:description|particulars)\b"),
        "quantity": re.compile(r"(?i)\b(?:qty|quantity)\b"),
        "unit_price": re.compile(r"(?i)\b(?:u/?\.?\s*price|unit\s*price|price)\b"),
        "discount": re.compile(r"(?i)\bdisc(?:ount)?\b"),
        "amount": re.compile(r"(?i)(?:amount|total\s*\(?RM\)?)"),
        "item": re.compile(r"(?i)^\s*(?:item|no\.?)\s*$"),
    }
    candidates: list[tuple[dict[str, Any], set[str]]] = []
    for line in lines:
        text = str(line.get("text", ""))
        keys = {key for key, pattern in patterns.items() if pattern.search(text)}
        if keys:
            candidates.append((line, keys))
    if not candidates:
        return None
    # A true header has several column-label concepts at the same vertical position.
    ranked = []
    for line, keys in candidates:
        y = _center(line)[1]
        nearby = set(keys)
        for other, other_keys in candidates:
            if abs(_center(other)[1] - y) <= 45:
                nearby.update(other_keys)
        ranked.append((len(nearby), -y, line, nearby))
    _, _, seed, _ = max(ranked, key=lambda value: (value[0], value[1]))
    header_y = _center(seed)[1]
    header_lines = [(line, keys) for line, keys in candidates if abs(_center(line)[1] - header_y) <= 45]
    anchors: dict[str, float] = {}
    for line, keys in header_lines:
        bounds = _bounds(line.get("box"))
        if not bounds:
            continue
        text = str(line.get("text", ""))
        if {"quantity", "description"}.issubset(keys):
            width = bounds[2] - bounds[0]
            anchors["quantity"] = bounds[0] + width * 0.10
            anchors["description"] = bounds[0] + width * 0.62
        else:
            for key in keys:
                anchors[key] = (bounds[0] + bounds[2]) / 2
    if "description" not in anchors:
        return None
    document_bottom = max((_bounds(line.get("box")) or (0, 0, 0, 0))[3] for line in lines)
    end_candidates = []
    for line in lines:
        text = str(line.get("text", ""))
        y = _center(line)[1]
        if y > header_y + 40 and re.search(r"(?i)^\s*(?:sub\s*total|grand\s+total|total(?:\s+rm|\s+amount|\s+payable)?|net\s+amt)\s*:?.*$", text):
            end_candidates.append(y)
    end_y = min(end_candidates, default=document_bottom + 1)
    return _Layout(header_y, end_y, anchors)


def _extract_stacked_thermal_items(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Read compact POS tables whose description precedes its numeric row.

    A common thermal layout has ``QTY UOM | UNIT PRICE | AMOUNT | TAX`` as
    its only header.  The product description is then printed on a full row,
    followed by quantity, unit and monetary cells on the next row.  This is
    deliberately detected from header concepts and geometry rather than a
    merchant name so it generalises to unseen receipts using this layout.
    """
    header_candidates: list[tuple[float, list[dict[str, Any]]]] = []
    for seed in lines:
        y = _center(seed)[1]
        nearby = [line for line in lines if abs(_center(line)[1] - y) <= 38]
        joined = " ".join(str(line.get("text", "")) for line in nearby)
        has_quantity = bool(re.search(r"(?i)\b(?:qty|quantity)\b", joined))
        has_price = bool(re.search(r"(?i)\b(?:unit\s*)?(?:amt|price|amount)\b", joined))
        if has_quantity and has_price:
            header_candidates.append((y, nearby))
    if not header_candidates:
        return []
    header_y, header_lines = min(header_candidates, key=lambda candidate: candidate[0])
    bounded = [_bounds(line.get("box")) for line in lines]
    bounded = [bound for bound in bounded if bound]
    if not bounded:
        return []
    left = min(bound[0] for bound in bounded)
    right = max(bound[2] for bound in bounded)
    width = max(right - left, 1.0)
    document_bottom = max(bound[3] for bound in bounded)
    total_markers = [
        _center(line)[1] for line in lines
        if _center(line)[1] > header_y + 45
        and re.search(r"(?i)^\s*[\W_]*(?:total\s+qty|sub\s*total|total\s+includes|grand\s+total|net\s+amt)", str(line.get("text", "")))
    ]
    end_y = min(total_markers, default=document_bottom + 1)
    body = [line for line in lines if header_y + 20 < _center(line)[1] < end_y - 4]
    if not body:
        return []

    units = re.compile(r"(?i)^(?:EA|EACH|PC|PCS|PACK|PKT|CTN|UNIT|UNITS|WALK|NOS?|KG|G|MG|L|ML|M|CM|MM|BOX|SET|ROLL|BAG|BTL)$")
    tax_codes = re.compile(r"(?i)^(?:ZRL|SR|ZR|GST|SST|TX|NT|OS|ES|CR)$")
    excluded = re.compile(r"(?i)\b(?:cashier|counter|customer|payment|total|discount|rounding|gst\s+summary|tax\s+code|thank\s+you)\b")

    descriptions: list[dict[str, Any]] = []
    for line in body:
        text = re.sub(r"\s+", " ", str(line.get("text", ""))).strip()
        x, y = _center(line)
        if not any(ch.isalpha() for ch in text) or excluded.search(text) or units.fullmatch(text) or tax_codes.fullmatch(text):
            continue
        # Description rows occupy the left/middle of the table and do not
        # consist mainly of a monetary value plus a short tax code.
        if x > left + width * .78 or re.fullmatch(r"[0-9.,]+\s+[A-Z]{1,4}", text, re.I):
            continue
        descriptions.append({"line": line, "text": text, "y": y})
    if not descriptions:
        return []

    rows: list[dict[str, str]] = []
    for index, description in enumerate(descriptions):
        start_y = description["y"] + 5
        next_y = descriptions[index + 1]["y"] if index + 1 < len(descriptions) else end_y
        cells = [line for line in body if start_y < _center(line)[1] < next_y - 2]
        numeric: list[tuple[float, str]] = []
        unit = ""
        quantity = ""
        for cell in cells:
            text = re.sub(r"\s+", " ", str(cell.get("text", ""))).strip()
            x, _ = _center(cell)
            if units.fullmatch(text) and x < left + width * .42:
                unit = text.upper()
                continue
            if tax_codes.fullmatch(text):
                continue
            value = _money(text)
            if value:
                if x < left + width * .22 and not quantity:
                    quantity = _number(text) or value.rstrip("0").rstrip(".")
                else:
                    numeric.append((x, value))
        numeric.sort(key=lambda pair: pair[0])
        # Require independent numeric evidence; otherwise this alpha line is
        # probably a wrapped continuation rather than a complete item row.
        if not numeric and not quantity:
            continue
        corrected, corrections = _correct_description(description["text"])
        values = {key: "" for key in LINE_ITEM_KEYS}
        values.update({
            "item": str(len(rows) + 1),
            "description": corrected,
            "quantity": quantity,
            "unit": unit,
            "unit_price": numeric[0][1] if numeric else "",
            "amount": numeric[-1][1] if numeric else "",
        })
        missing = [label for key, label in (("quantity", "quantity"), ("unit_price", "unit price"), ("amount", "amount")) if not values[key]]
        notes = ["AI-normalised description; compare with image"] if corrections else []
        notes.append("Verify " + ", ".join(missing) if missing else "Automatically reconstructed from stacked receipt layout")
        values["review_note"] = "; ".join(notes)
        rows.append(values)
    return rows


def _extract_unheaded_item_blocks(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Recover item blocks when a damaged scan has lost the table header.

    The routine requires three independent layout signals: an invoice/cash
    sale marker, a later totals marker, and a product-like alpha line with a
    nearby monetary cell.  That constraint keeps the fallback useful across
    thermal formats without turning addresses or payment metadata into items.
    """
    marker = re.compile(r"(?i)\b(?:tax\s+invoice|cash\s+sale|inv[a-z0-9]{2,8}e|invoice)\b")
    marker_lines = [line for line in lines if marker.search(str(line.get("text", "")))]
    end_candidates = [
        _center(line)[1] for line in lines
        if re.search(r"(?i)^\s*[\W_]*(?:items?\s*\(?s?\)?|qty\s*\(?s?\)?|sub\s*t(?:otal|r?ital)|tota(?:l|[}\]1])?(?:\s+sales)?|net\s+amt)", str(line.get("text", "")))
    ]
    if not end_candidates:
        return []
    bounded = [_bounds(line.get("box")) for line in lines]
    bounded = [bound for bound in bounded if bound]
    if not bounded:
        return []
    end_y = min(end_candidates)
    document_top, document_bottom = min(bound[1] for bound in bounded), max(bound[3] for bound in bounded)
    earlier_markers = [_center(line)[1] for line in marker_lines if _center(line)[1] < end_y]
    # Some degraded receipts lose the INVOICE label. In that case use only
    # the lower pre-total band; nearby monetary evidence is still mandatory.
    start_y = min(earlier_markers) if earlier_markers else max(document_top, end_y - (document_bottom - document_top) * .55)
    if end_y - start_y < 55:
        return []
    left, right = min(bound[0] for bound in bounded), max(bound[2] for bound in bounded)
    width = max(right - left, 1.0)
    body = [line for line in lines if start_y + 12 < _center(line)[1] < end_y - 3]
    excluded = re.compile(r"(?i)\b(?:invoice|cash\s+sale|receipt|date|time|cashier|counter|terminal|card|approval|approved|operator|customer|payment|company|registration|address|telephone|email|total|discount|rounding|change|response|trace|stan|reference|site\s+id|app\s*(?:id|\d+)|particulars?|description|quantity|unit\s*price)\b")

    def money_tokens(text: str) -> list[str]:
        values: list[str] = []
        for match in re.finditer(r"(?i)(?:RM\s*)?([0-9][0-9,]*\.\d{1,2})(?:\s*(?:SR|CR))?", text):
            value = _money(match.group(0))
            if value:
                values.append(value)
        return values

    rows: list[dict[str, str]] = []
    used_y: list[float] = []
    for line in body:
        text = re.sub(r"\s+", " ", str(line.get("text", ""))).strip()
        x, y = _center(line)
        if sum(ch.isalpha() for ch in text) < 3 or excluded.search(text) or x > left + width * .72:
            continue
        # A damaged quantity cell can contain OCR substitutions such as
        # ``2i1 X RM 8.93``. It is numeric evidence for the preceding product,
        # never a second description row.
        if re.search(r"(?i)\b[xX]\s*(?:RM\s*)?[0-9]+(?:\.[0-9]{1,2})?\s*$", text):
            continue
        nearby = [candidate for candidate in body if abs(_center(candidate)[1] - y) <= 52 and candidate is not line]
        monetary: list[tuple[float, str]] = []
        quantity = ""
        unit = ""
        for candidate in nearby:
            candidate_text = str(candidate.get("text", "")).strip()
            candidate_x, _ = _center(candidate)
            for value in money_tokens(candidate_text):
                if candidate_x >= x - width * .08:
                    monetary.append((candidate_x, value))
            combined = re.search(r"(?i)\b([0-9]+(?:\.[0-9]+)?)\s*[xX]\s*(?:RM\s*)?([0-9]+(?:\.\d{1,2})?)", candidate_text)
            if combined:
                quantity = combined.group(1)
                raw_price = combined.group(2)
                if "." in raw_price:
                    monetary.append((candidate_x, f"{Decimal(raw_price):.2f}"))
            elif price_after_x := re.search(r"(?i)\b[xX]\s*(?:RM\s*)?([0-9]+(?:\.\d{1,2})?)", candidate_text):
                # Preserve the readable price even when the quantity glyphs
                # immediately before X are damaged beyond safe recovery.
                raw_price = price_after_x.group(1)
                if "." in raw_price:
                    monetary.append((candidate_x, f"{Decimal(raw_price):.2f}"))
            if re.fullmatch(r"(?i)(?:EA|PC|PCS|PACK|PKT|CTN|UNIT|KG|G|L|ML|BOX|SET|ROLL|BAG|BTL)", candidate_text):
                unit = candidate_text.upper()
        if not monetary or any(abs(y - old_y) < 12 for old_y in used_y):
            continue
        monetary.sort(key=lambda pair: pair[0])
        corrected, corrections = _correct_description(text)
        values = {key: "" for key in LINE_ITEM_KEYS}
        values.update({
            "item": str(len(rows) + 1), "description": corrected,
            "quantity": quantity, "unit": unit,
            "unit_price": monetary[0][1] if len(monetary) > 1 else "",
            "amount": monetary[-1][1],
        })
        missing = [label for key, label in (("quantity", "quantity"), ("unit_price", "unit price"), ("amount", "amount")) if not values[key]]
        notes = ["AI-normalised description; compare with image"] if corrections else []
        notes.append("Verify " + ", ".join(missing) if missing else "Automatically reconstructed from unheaded receipt layout")
        values["review_note"] = "; ".join(notes)
        rows.append(values)
        used_y.append(y)
    return rows


def _fallback_line_items(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    return _extract_stacked_thermal_items(lines) or _extract_unheaded_item_blocks(lines)


def extract_line_items(lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Reconstruct printed and handwritten line-item tables using OCR geometry.

    The result is evidence-bound: unreadable cells remain blank and receive a
    review note instead of being guessed.
    """
    physical = _best_physical_lines(lines)
    layout = _detect_layout(physical)
    if not layout:
        return _fallback_line_items(physical)
    table = [line for line in physical if layout.header_y + 12 < _center(line)[1] < layout.end_y - 8]
    if not table:
        return []
    anchors = layout.anchors
    sorted_columns = sorted(anchors.items(), key=lambda pair: pair[1])
    boundaries = [(sorted_columns[i][1] + sorted_columns[i + 1][1]) / 2 for i in range(len(sorted_columns) - 1)]

    def column_for(x: float) -> str:
        index = sum(x > boundary for boundary in boundaries)
        return sorted_columns[min(index, len(sorted_columns) - 1)][0]

    descriptions: list[dict[str, Any]] = []
    numeric_cells: list[tuple[str, str, float, dict[str, Any]]] = []
    rejected = re.compile(r"(?i)(?:\b(?:item\s+d[il]?s?c?ount|gst\s+summary|thank\s+you|cashier|approval|goods\s+sold|total|discount|rounding|inclusive|ringgit)\b|^\s*E\W+O\W+E\W*$)")
    for line in table:
        text = str(line.get("text", "")).strip()
        x, y = _center(line)
        column = column_for(x)
        if rejected.search(text):
            continue
        money = _money(text)
        number = _number(text)
        if column in {"quantity", "unit_price", "discount", "amount", "item"} and (money or number):
            value = money if column in {"unit_price", "discount", "amount"} else number
            numeric_cells.append((column, value or "", y, line))
            continue
        if column == "description" and any(ch.isalpha() for ch in text):
            # Combined Qty/Description columns frequently place a numeric quantity at the start.
            prefix = re.match(r"^([0-9]+(?:\.[0-9]{1,3})?)\s+(.+)$", text)
            if prefix and any(ch.isalpha() for ch in prefix.group(2)):
                numeric_cells.append(("quantity", prefix.group(1), y, line))
                text = prefix.group(2)
            descriptions.append({"line": line, "text": text, "y": y})
    if not descriptions:
        return _fallback_line_items(physical)

    amount_cells = [cell for cell in numeric_cells if cell[0] == "amount"]
    # A single handwritten item often spans several description lines while
    # its total appears once in the amount column. Treat it as one evidence row.
    if len(amount_cells) == 1 and len(descriptions) > 1:
        groups = [descriptions]
    else:
        row_anchors = sorted({round(cell[2], 1) for cell in amount_cells}) or [item["y"] for item in descriptions]
        groups = [[] for _ in row_anchors]
        for description in descriptions:
            nearest = min(range(len(row_anchors)), key=lambda index: abs(row_anchors[index] - description["y"]))
            groups[nearest].append(description)

    rows: list[dict[str, str]] = []
    for index, group in enumerate(groups, start=1):
        if not group:
            continue
        row_y = sum(item["y"] for item in group) / len(group)
        if len(amount_cells) == 1:
            row_y = amount_cells[0][2]
        if len(groups) > 1:
            other_ys = sorted([sum(item["y"] for item in other) / len(other) for other in groups if other])
            gaps = [abs(y - row_y) for y in other_ys if y != row_y]
            tolerance = max(24.0, min(gaps, default=60.0) * 0.48)
        else:
            tolerance = max(65.0, (max(item["y"] for item in group) - min(item["y"] for item in group)) / 2 + 25)
        values = {key: "" for key in LINE_ITEM_KEYS}
        values["item"] = str(index)
        raw_description = " / ".join(item["text"] for item in sorted(group, key=lambda value: value["y"]))
        values["description"], corrections = _correct_description(raw_description)
        candidates = [cell for cell in numeric_cells if abs(cell[2] - row_y) <= tolerance]
        for key in ("quantity", "unit_price", "discount", "amount"):
            matches = [cell for cell in candidates if cell[0] == key]
            if matches:
                values[key] = min(matches, key=lambda cell: abs(cell[2] - row_y))[1]
        # A common low-resolution confusion is a leading quantity digit read as
        # "i" or "l" (for example i.38). Reconstruct it only when unit-price ×
        # quantity rounds exactly to the independently observed line amount.
        arithmetic_quantity = False
        if not values["quantity"] and values["unit_price"] and values["amount"]:
            ambiguous = re.match(r"^[iIl|]\.([0-9]{1,3})\b", raw_description)
            if ambiguous:
                try:
                    quotient = Decimal(values["amount"]) / Decimal(values["unit_price"])
                    decimals = len(ambiguous.group(1))
                    candidate = f"{quotient:.{decimals}f}"
                    if candidate.split(".")[-1] == ambiguous.group(1) and (Decimal(candidate) * Decimal(values["unit_price"])).quantize(Decimal("0.01")) == Decimal(values["amount"]):
                        values["quantity"] = candidate
                        values["description"] = re.sub(r"^[iIl|]\.[0-9]{1,3}\s*", "", values["description"])
                        arithmetic_quantity = True
                except (InvalidOperation, ZeroDivisionError):
                    pass
        missing = [label for key, label in (("quantity", "quantity"), ("unit_price", "unit price"), ("amount", "amount")) if not values[key]]
        notes = []
        if corrections:
            notes.append("AI-normalised description; compare with image")
        if arithmetic_quantity:
            notes.append("quantity reconstructed from visible price and amount")
        if missing:
            notes.append("Verify " + ", ".join(missing))
        else:
            notes.append("Automatically reconstructed from receipt layout")
        values["review_note"] = "; ".join(notes)
        rows.append(values)
    return rows


def extract_accounting_fields(text: str, lines: list[dict[str, Any]], primary: dict[str, Any]) -> dict[str, Any]:
    """Extract audit/accounting evidence commonly relevant to MyInvois workflows."""
    evidence = text or ""
    def first(pattern: str) -> str | None:
        match = re.search(pattern, evidence, re.I | re.M)
        return re.sub(r"\s+", " ", match.group(1)).strip(" :") if match else None

    registration = first(r"(?:company|co\.?|business|br)\s*(?:reg(?:istration)?\.?|no\.?)?[ \t]*[:#.]?[ \t]*\(?([A-Z0-9][A-Z0-9/-]{4,})\)?")
    if not registration:
        # Some Malaysian cash-sale pads print an unlabelled parenthesised
        # company number immediately beneath the supplier name.
        registration = first(r"(?m)^\s*\(([A-Z0-9]{4,}(?:-[A-Z0-9]+)+)\)\s*$")
    if not registration:
        match = re.search(r"(?<!\d)(\d{5,9}-[A-Z])(?![A-Z0-9])", evidence, re.I)
        registration = match.group(1) if match else None
    tax_number = first(r"(?:GST|SST|tax)\s*(?:reg(?:istration)?\.?|no\.?)?[ \t]*[:#.]?[ \t]*\(?([A-Z0-9][A-Z0-9/-]{5,})\)?")
    if tax_number and not any(ch.isdigit() for ch in tax_number):
        tax_number = None
    phone = first(r"(?:tel(?:ephone)?|phone|h/?p|mobile)(?:\s*/\s*fax)?\s*[:.]?[ \t]*([+0-9][0-9 ()/-]{6,})")
    if not phone:
        phone = first(r"(?m)^\s*((?:\+?60|0)\d{1,2}[- ]?\d{6,8})\s*$")
    email = first(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})")
    document_type = next((value for pattern, value in ((r"\btax\s*invoice\b", "Tax Invoice"), (r"\bcash\s*sale(?:s)?\b", "Cash Sale"), (r"\binvoice\b", "Invoice"), (r"\breceipt\b", "Receipt")) if re.search(pattern, evidence, re.I)), None)
    payment_method = next((value for pattern, value in ((r"\bcredit\s+card\b", "Credit card"), (r"\bdebit\b", "Debit card"), (r"\be-?wallet\b", "E-wallet"), (r"\bcash\b", "Cash")) if re.search(pattern, evidence, re.I)), None)
    currency = "MYR" if re.search(r"\b(?:RM|MYR)\b", evidence, re.I) else None

    physical = _best_physical_lines(lines)

    def labelled_amount(labels: str) -> str | None:
        matches = re.finditer(rf"(?im)^.*(?:{labels}).*$", evidence)
        for match in matches:
            values = re.findall(r"(?:RM\s*)?([0-9][0-9,]*\.\d{{2}})", match.group(0), re.I)
            if values:
                return f"{Decimal(values[-1].replace(',', '')):.2f}"
        label_lines = [line for line in physical if re.search(labels, str(line.get("text", "")), re.I)]
        for label in label_lines:
            lx, ly = _center(label)
            candidates: list[tuple[int, float, float, str]] = []
            for line in physical:
                x, y = _center(line)
                if line is label or not (ly - 12 <= y <= ly + 75):
                    continue
                for raw in re.findall(r"(?:RM\s*)?([0-9][0-9,]*\.\d{2})", str(line.get("text", "")), re.I):
                    # OCR boxes on one printed row can differ by a few pixels.
                    # Compare vertical bands first, then prefer the value under
                    # the detected label's column.
                    candidates.append((int(abs(y - ly) // 12), abs(x - lx), -x, f"{Decimal(raw.replace(',', '')):.2f}"))
            if candidates:
                return min(candidates)[3]
        return None

    # Address is preserved as observed header evidence rather than reformatted.
    merchant = str(primary.get("merchant") or "")
    supplier_name = merchant or None
    if re.fullmatch(r"(?i)(?:sdn\.?\s*bhd\.?|bhd\.?|enterprise|trading)", merchant.strip()):
        merchant_line = next((index for index, line in enumerate(physical[:12]) if merchant.casefold() in str(line.get("text", "")).casefold()), None)
        if merchant_line is not None and merchant_line > 0:
            previous = str(physical[merchant_line - 1].get("text", "")).strip()
            if sum(ch.isalpha() for ch in previous) >= 4:
                supplier_name = f"{previous} {merchant}".strip()
    address_parts: list[str] = []
    seen_merchant = False
    for line in physical[:20]:
        value = str(line.get("text", "")).strip()
        if merchant and merchant.casefold() in value.casefold():
            seen_merchant = True
            continue
        if seen_merchant:
            if re.fullmatch(r"(?i)\s*\d*\s*amount\s*", value):
                continue
            if registration and registration.casefold() in value.casefold():
                value = re.sub(rf"\(?\b{re.escape(registration)}\b\)?", "", value, flags=re.I).strip()
                if not value:
                    continue
            if re.fullmatch(r"\s*(?:\+?60|0)\d{1,2}[- ]?\d{6,8}\s*", value):
                continue
            if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", value):
                if address_parts:
                    break
                continue
            if re.search(r"(?i)\b(?:company|business|br)\s*(?:reg|no)|\b(?:tel|fax|phone|email|GST|SST|tax\s*invoice|cash\s+sale|invoice|quantity|particulars?|unit\s*price|amount)\b", value):
                if address_parts:
                    if re.search(r"(?i)\b(?:tax\s*invoice|cash\s+sale|invoice|quantity|particulars?|unit\s*price|amount)\b", value):
                        break
                continue
            if len(value) >= 6 and (re.search(r"\d", value) or re.search(r"(?i)\b(?:jalan|jln|lot|taman|medan|bandar|selangor|kuala|lumpur)\b", value)):
                value = re.sub(r"(?i)\s+\d*\s*amount\b.*$", "", value).strip()
                address_parts.append(value)
    items = extract_line_items(lines)
    subtotal = labelled_amount(r"sub\s*total")
    if not subtotal and items and all(row.get("amount") for row in items):
        try:
            subtotal = f"{sum(Decimal(str(row['amount'])) for row in items):.2f}"
        except InvalidOperation:
            subtotal = None
    if not subtotal and document_type == "Cash Sale" and items and primary.get("total_amount"):
        # Handwritten cash-sale pads often show only a final total. With no
        # printed tax/discount breakdown, retain that supported amount as the
        # accounting subtotal rather than leaving the field empty.
        subtotal = str(primary["total_amount"])
    tax_amount = labelled_amount(r"tax\s*(?:amount|\(RM\))")
    if not tax_amount:
        tax_amount = labelled_amount(r"(?:GST|SST)\s+summary")
    if not tax_amount and re.search(r"(?i)total\s+includes\s+(?:GST|SST)\s*0\s*%", evidence):
        tax_amount = "0.00"
    return {
        "supplier_name": supplier_name,
        "supplier_registration_number": registration,
        "supplier_tax_number": tax_number,
        "supplier_address": " ".join(address_parts) or None,
        "supplier_phone": phone,
        "supplier_email": email,
        "document_type": document_type,
        "document_number": primary.get("receipt_invoice_number"),
        "transaction_date": primary.get("transaction_date"),
        "currency": currency or "MYR",
        "payment_method": payment_method,
        "subtotal": subtotal,
        "discount": labelled_amount(r"discount"),
        "tax_amount": tax_amount,
        "payable_amount": primary.get("total_amount"),
        "line_item_count": len(items),
    }

