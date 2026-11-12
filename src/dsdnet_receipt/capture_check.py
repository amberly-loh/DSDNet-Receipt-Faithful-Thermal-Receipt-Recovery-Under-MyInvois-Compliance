"""After-scan check: does the photo look turned the wrong way or cut badly?

The automatic turn and cut are usually right, but when they are not the
reading fails quietly. This check looks only at what the scanner already
produced (the text lines and their positions) and, when the result looks
like a wrong turn or a bad cut, asks the person to try again. It never
changes any value; it only decides whether to show the "try again" prompt.
"""
from __future__ import annotations

from typing import Any

MESSAGES = {
    "rotation": "This receipt may be turned the wrong way, so most of the text could not be read. "
                "Scan it again with the top of the receipt facing up, or turn it on the phone before sending.",
    "crop": "Part of this receipt may have been cut off. Scan it again with the whole receipt in the frame, "
            "or move the crop corners to the paper's edges on the phone.",
    "unreadable": "Very little text could be read. Check the photo is sharp, the receipt is upright and the "
                  "whole receipt is in the frame, then scan it again.",
}


def _box_size(box) -> tuple[float, float, float, float, float, float] | None:
    try:
        xs = [float(p[0]) for p in box]; ys = [float(p[1]) for p in box]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys), max(xs) - min(xs), max(ys) - min(ys)


def check_capture(lines: list[dict[str, Any]], image_shape: tuple[int, ...] | None,
                  fields: dict[str, Any] | None = None, upright_check: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return {"status": "ok"|"retry", "issue": ..., "message": ..., "signals": {...}}."""
    fields = fields or {}
    good = [line for line in lines or []
            if float(line.get("confidence") or 0) >= 0.5 and len(str(line.get("text") or "").strip()) >= 2]
    chars = sum(len(str(line.get("text") or "").strip()) for line in good)
    mean_conf = (sum(float(line.get("confidence") or 0) for line in good) / len(good)) if good else 0.0
    boxes = [b for b in (_box_size(line.get("box")) for line in good) if b]
    tall = sum(1 for b in boxes if b[5] > 1.5 * b[4])           # text running up and down = turned sideways
    short = sum(1 for line in good if len(str(line.get("text")).strip()) <= 3)
    have = {key: bool(str((fields.get(key) or {}).get("value") if isinstance(fields.get(key), dict) else fields.get(key) or "").strip())
            for key in ("total_amount", "transaction_date", "merchant")}
    signals = {"lines": len(good), "characters": chars, "mean_confidence": round(mean_conf, 3),
               "vertical_lines": tall, "short_lines": short, **{f"has_{k}": v for k, v in have.items()}}

    def result(issue):
        return {"status": "retry" if issue else "ok", "issue": issue, "message": MESSAGES.get(issue, ""), "signals": signals}

    # 1. Sideways: most readable boxes run up and down, or the reading is
    #    almost all one-to-three-character fragments.
    if len(boxes) >= 4 and tall >= 0.5 * len(boxes):
        return result("rotation")
    if len(good) >= 6 and short >= 0.7 * len(good) and not have["total_amount"]:
        return result("rotation")
    # 2. Upside down and neither way read well.
    if upright_check and max(float(upright_check.get("upright_score") or 0), float(upright_check.get("upside_down_score") or 0)) < 40 \
            and chars < 60:
        return result("rotation")
    # 3. Nothing much was read at all.
    if len(good) < 4 or chars < 40:
        return result("unreadable")
    # 4. Cut off: text reaches the top or bottom edge and the total (usually
    #    near the bottom) or the date/merchant (near the top) is missing.
    if image_shape and boxes and not have["total_amount"]:
        height = float(image_shape[0])
        margin = 0.015 * height
        top_cut = any(b[1] <= margin for b in boxes)
        bottom_cut = any(b[3] >= height - margin for b in boxes)
        if bottom_cut or (top_cut and not (have["merchant"] or have["transaction_date"])):
            return result("crop")
    return result(None)
