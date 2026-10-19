from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any


DATE_PATTERNS = (
    re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)?\d{2}\b"),
    re.compile(r"\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])\b"),
)
TOTAL_PATTERN = re.compile(r"(?i)\b(?:net\s+amt|grand\s+total|total\s+rm|total\s+amt|total|amount)\b[^\d]{0,16}(?:RM\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)")
MONEY_PATTERN = re.compile(r"(?i)(?:RM\s*)?([0-9][0-9,]*\.\d{2})")
NUMBER_PATTERN = re.compile(r"(?i)\b(?:(?:receipt|invoice|document|bill)\s*)?(?:(?:no|number)\b|#)\.?(?:\s*[:#-]?)\s*([A-Z0-9][A-Z0-9/-]{2,})")


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
    raw_date = next((match.group(0) for pattern in DATE_PATTERNS if (match := pattern.search(text))), None)
    date = _normalise_date(raw_date)
    number_match = NUMBER_PATTERN.search(text)
    receipt_number = _spatial_invoice_number(lines) or (number_match.group(1) if number_match else None)
    spatial_total = _spatial_total(lines)
    total_match = TOTAL_PATTERN.search(text)
    if spatial_total:
        total = f"{Decimal(spatial_total):.2f}"
    elif total_match:
        total = f"{Decimal(total_match.group(1).replace(',', '')):.2f}"
    else:
        amounts = [match.group(1).replace(",", "") for match in MONEY_PATTERN.finditer(text)]
        total = f"{Decimal(amounts[-1]):.2f}" if amounts else None
    merchant = _merchant_candidate(lines)
    return {
        "merchant": {"value": merchant, "confidence": _mean_confidence(lines, [merchant or ""]), "box": _box_for(lines, [merchant or ""])},
        "transaction_date": {"value": date, "confidence": _mean_confidence(lines, [raw_date or "date"]), "box": _box_for(lines, [raw_date or "date"])},
        "receipt_invoice_number": {"value": receipt_number, "confidence": _mean_confidence(lines, [receipt_number or "receipt", "invoice", "document"]), "box": _box_for(lines, [receipt_number or "receipt", "invoice", "document"])},
        "total_amount": {"value": total, "confidence": _mean_confidence(lines, [total or "total"]), "box": _box_for(lines, [total or "total"])},
    }


def _normalise_date(value: str | None) -> str | None:
    if not value:
        return None
    for pattern in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d", "%Y-%m-%d"):
        try: return datetime.strptime(value, pattern).date().isoformat()
        except ValueError: continue
    return None


def _bounds(box: Any) -> tuple[float, float, float, float] | None:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        return min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError, IndexError):
        return None


def _spatial_total(lines: list[dict[str, Any]]) -> str | None:
    labels = [line for line in lines if re.search(r"(?i)\b(?:net\s+amt|grand\s+total|total\s+payable|total\s+amt|total)\b", str(line.get("text", "")))]
    candidates: list[tuple[float, str]] = []
    for label in labels:
        label_box = _bounds(label.get("box"))
        if not label_box:
            continue
        _, ly1, lx2, ly2 = label_box
        lcy = (ly1 + ly2) / 2
        for line in lines:
            value_box = _bounds(line.get("box"))
            match = re.fullmatch(r"\s*(?:RM\s*)?([0-9][0-9,]*\.\d{2})\s*", str(line.get("text", "")), re.I)
            if not value_box or not match:
                continue
            vx1, vy1, _, vy2 = value_box
            vcy = (vy1 + vy2) / 2
            if abs(vcy - lcy) <= 55 and vx1 >= lx2 - 20:
                candidates.append((abs(vcy - lcy) + max(0, vx1 - lx2) / 20, match.group(1).replace(",", "")))
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _spatial_invoice_number(lines: list[dict[str, Any]]) -> str | None:
    labels = [line for line in lines if re.fullmatch(r"(?i)\s*(?:invoice\s*)?(?:no|no\.|number|#)\s*", str(line.get("text", "")))]
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
    return min(candidates, default=(0, None), key=lambda item: item[0])[1]


def _merchant_candidate(lines: list[dict[str, Any]]) -> str | None:
    reject = re.compile(r"(?i)\b(?:receipt|invoice|cash\s+sale|address|tel|date|item|qty|quantity|price|amount|total|tax|sst|thank)\b")
    ordered = sorted(lines, key=lambda item: _line_position(item.get("box")))
    for line in ordered[:12]:
        value = str(line.get("text", "")).strip(" -:|")
        letters = sum(ch.isalpha() for ch in value)
        if 3 <= len(value) <= 80 and letters >= 3 and not reject.search(value) and not re.fullmatch(r"[\d\W]+", value):
            return value
    return None


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
            )
        )
    return records
