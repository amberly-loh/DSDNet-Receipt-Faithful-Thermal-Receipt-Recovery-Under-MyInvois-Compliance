from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


# Review vocabulary only. Merchant names, item descriptions and numeric values
# must never be guessed from this stable set of receipt/form labels.
RECEIPT_TERMS = (
    "invoice", "receipt", "description", "quantity", "unit", "price",
    "amount", "discount", "subtotal", "total", "payable", "payment",
    "cash", "change", "date", "customer", "supplier", "merchant",
    "cashier", "bill", "item",
    "address", "telephone", "email", "notes", "terms", "authorised",
    "authorized", "signature", "issued", "received", "registration",
    "currency", "tax", "service", "charge", "rounding", "balance",
)

_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _match_case(source: str, target: str) -> str:
    if source.isupper(): return target.upper()
    if source.istitle() or source[:1].isupper(): return target.title()
    return target


def suggest_receipt_text(text: str) -> dict[str, Any] | None:
    """Suggest conservative label corrections without changing OCR evidence."""
    changes: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        observed = match.group(0)
        lowered = observed.casefold()
        if len(lowered) < 4 or lowered in RECEIPT_TERMS: return observed
        similarity, candidate = max(
            (SequenceMatcher(None, lowered, term).ratio(), term) for term in RECEIPT_TERMS
        )
        threshold = 0.82 if len(lowered) >= 7 else 0.86
        if similarity < threshold or abs(len(lowered) - len(candidate)) > 2: return observed
        replacement = _match_case(observed, candidate)
        changes.append({"observed": observed, "suggested": replacement, "similarity": round(similarity, 3)})
        return replacement

    suggested = _TOKEN.sub(replace, str(text or ""))
    if not changes or suggested == text: return None
    return {
        "observed": text, "suggested": suggested, "changes": changes,
        "status": "unconfirmed_language_suggestion", "source_support": False,
        "instruction": "Verify against the original receipt before use.",
    }


def suggestions_for_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        suggestion = suggest_receipt_text(str(line.get("text", "")))
        if suggestion: suggestions.append({"line_index": index, **suggestion})
    return suggestions
