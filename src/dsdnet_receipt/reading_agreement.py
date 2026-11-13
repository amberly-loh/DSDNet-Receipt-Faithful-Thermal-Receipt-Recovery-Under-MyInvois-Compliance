"""Reading agreement: the key fields are read from several independent readings of the same
rectified receipt and the value that most readings agree on is kept (13 November 2026).

Evidence-aware in the sense of the thesis title: a value is accepted on the strength of
independent evidence for it, not on one reading alone. The readings are the application's own
fused multi-view reading (always listed first and kept whenever it is among the most agreed
values), the rectified image read directly, its Otsu and perceptually filtered versions, and the
DSDNet restoration read on its own. Every reading uses the same text reader and the same field
parser and review. Only merchant, date and total are voted on; the vote and every reading are
kept in the audit, so no value appears without the reading it came from.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

VOTED_FIELDS = ("merchant", "transaction_date", "total_amount")
EXTRA_READERS = ("A_direct", "B_otsu", "C_perceptual")


def _key(name: str, value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    if name == "merchant":
        # two readings of the same name differ only in spacing and punctuation
        return re.sub(r"[^A-Z0-9&]", "", str(value).upper()) or None
    return str(value).strip()


def _spacing(value: Any) -> tuple[int, int]:
    text = str(value)
    # the reader drops word gaps and dots far more often than it invents them
    return (len(text.split()), text.count("."))


def vote(name: str, readings: list[tuple[str, dict[str, Any]]]) -> tuple[str | None, dict[str, Any] | None, dict[str, Any]]:
    """`readings` is a list of (reading name, extracted fields); the first is the primary reading.
    Returns (value, field record it came from, audit record)."""
    values = [(label, (fields.get(name) or {}).get("value"), fields.get(name) or {}) for label, fields in readings]
    keys = [_key(name, value) for _, value, _ in values]
    counts = Counter(key for key in keys if key)
    record = {"readings": {label: value for label, value, _ in values}, "support": dict(counts)}
    if not counts:
        record["chosen_from"] = None
        return None, None, record
    best = max(counts.values())
    primary_key = keys[0]
    winner = primary_key if primary_key and counts[primary_key] == best else next(k for k in keys if k and counts[k] == best)
    candidates = [(label, value, field) for (label, value, field), key in zip(values, keys) if key == winner]
    label, value, field = max(candidates, key=lambda item: _spacing(item[1])) if name == "merchant" else candidates[0]
    record.update(chosen_from=label, agreeing=counts[winner], of=len(readings))
    return value, field, record


def agree(primary: dict[str, dict[str, Any]], others: list[tuple[str, dict[str, Any]]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return the voted values of the key fields and the audit record."""
    readings = [("fused_views", primary)] + list(others)
    chosen: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {"method": "reading_agreement_v1", "fields": {}}
    for name in VOTED_FIELDS:
        value, field, record = vote(name, readings)
        audit["fields"][name] = record
        if value is not None:
            chosen[name] = {"value": value, "confidence": (field or {}).get("confidence"), "box": (field or {}).get("box")}
    return chosen, audit


def read_alternatives(ocr_engine: Any, processed: Any, model_output: Any,
                      extract: Callable[[str, list], dict[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    from .baselines import BASELINES
    images = [(name, BASELINES[name](processed)) for name in EXTRA_READERS]
    if model_output is not None and getattr(model_output, "shape", None) is not None and model_output.shape[:2] == processed.shape[:2]:
        images.append(("dsdnet_restoration", model_output))
    out = []
    for name, image in images:
        try:
            result = ocr_engine.run(image)
            out.append((name, extract(result.text, result.lines)))
        except Exception:  # a reading that fails simply does not vote
            continue
    return out
