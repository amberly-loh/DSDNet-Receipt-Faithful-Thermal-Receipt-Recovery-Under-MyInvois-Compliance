from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


DATE_PATTERNS = (
    re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)?\d{2}\b"),
    re.compile(r"\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])\b"),
)
TOTAL_PATTERN = re.compile(r"(?i)\b(?:grand\s+total|total\s+amt|total|amount)\b[^\d]{0,12}(?:RM\s*)?([0-9][0-9,]*\.\d{2})")
MONEY_PATTERN = re.compile(r"(?i)(?:RM\s*)?([0-9][0-9,]*\.\d{2})")
NUMBER_PATTERN = re.compile(r"(?i)\b(?:receipt|invoice|document|bill)\s*(?:no|number|#|:)\.?(?:\s*[:#-]?)\s*([A-Z0-9][A-Z0-9/-]{3,})")


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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean_confidence(lines: list[dict[str, Any]], tokens: list[str]) -> float | None:
    selected = []
    lowered = [token.lower() for token in tokens if token]
    for line in lines:
        text = str(line.get("text", "")).lower()
        if any(token in text for token in lowered):
            value = line.get("confidence")
            if value is not None:
                selected.append(float(value))
    return sum(selected) / len(selected) if selected else None


def extract_fields(text: str, lines: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    date = next((match.group(0) for pattern in DATE_PATTERNS if (match := pattern.search(text))), None)
    number_match = NUMBER_PATTERN.search(text)
    receipt_number = number_match.group(1) if number_match else None
    total_match = TOTAL_PATTERN.search(text)
    if total_match:
        total = total_match.group(1).replace(",", "")
    else:
        amounts = [match.group(1).replace(",", "") for match in MONEY_PATTERN.finditer(text)]
        total = amounts[-1] if amounts else None
    return {
        "transaction_date": {"value": date, "confidence": _mean_confidence(lines, [date or "date"])},
        "receipt_invoice_number": {"value": receipt_number, "confidence": _mean_confidence(lines, [receipt_number or "receipt", "invoice", "document"])},
        "total_amount": {"value": total, "confidence": _mean_confidence(lines, [total or "total"])},
    }


def route_fields(
    extracted: dict[str, dict[str, Any]],
    support_available: bool,
    model_available: bool,
) -> list[FieldRecord]:
    records = []
    for name, payload in extracted.items():
        value = payload.get("value")
        confidence = payload.get("confidence")
        localization_valid = bool(value)
        rule_passed = bool(value)
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
            )
        )
    return records
