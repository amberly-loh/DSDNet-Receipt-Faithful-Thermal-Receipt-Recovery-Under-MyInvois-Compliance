"""Per-user settings: which optional (slower) processing steps run, and display/export choices."""
from __future__ import annotations

from typing import Any

# key, group, label, description, default, kind ("bool" or a tuple of choices)
SETTINGS: list[dict[str, Any]] = [
    {"key": "extra_ocr_view", "group": "Processing", "label": "Third scanner pass (contrast-boosted copy)",
     "description": "Each scanner pass takes several seconds. Automatic runs it only for faded or blurred receipts.",
     "default": "auto", "choices": {"auto": "Automatic (faded or blurred receipts only)", "always": "Always", "never": "Never (fastest)"}, "slow": True},
    {"key": "second_reading", "group": "Processing", "label": "Re-read the cleaned image when the date, total or number is missing",
     "description": "Only runs when one of these fields was not found.", "default": True, "slow": True},
    {"key": "reading_agreement", "group": "Processing", "label": "Compare several readings of the merchant, date and total",
     "description": "Reads the receipt a few more ways and keeps the value most readings agree on. More accurate, but adds several seconds. Off in the Fast preset.",
     "default": True, "slow": True},
    {"key": "show_fields_first", "group": "Processing", "label": "Show the fields first, finish the cleaned image afterwards",
     "description": "Merchant, date and total appear as soon as they are read; the cleaned image follows a few seconds later. The fields are the same either way.",
     "default": True, "slow": False},
    {"key": "fast_restoration", "group": "Processing", "label": "Faster restoration (half precision on the GPU)",
     "description": "Runs the restoration network in half precision. Optional; it gave no measurable speed gain in testing.", "default": False, "slow": False},
    {"key": "fast_resolution", "group": "Processing", "label": "Faster processing at lower resolution",
     "description": "Processes the receipt at a long edge of at most 1280 pixels. Part of the Fast preset.", "default": False, "slow": False},
    {"key": "severe_cleanup", "group": "Restoration", "label": "Show-through and crease cleanup",
     "description": "Tries an extra cleaned version of creased or show-through receipts and keeps it only if it is better.",
     "default": True, "slow": True},
    {"key": "damage_separation", "group": "Restoration", "label": "Separate text, tables and logos from noise",
     "description": "Removes specks, crease shading and stain edges when this provably keeps all text.", "default": True},
    {"key": "stroke_joining", "group": "Restoration", "label": "Join broken strokes on faded receipts",
     "description": "Reconnects letters only along faint ink still in the photo.", "default": True},
    {"key": "default_view", "group": "Display", "label": "Image shown first",
     "description": "Which image opens when you view a receipt.", "default": "faithful",
     "choices": {"faithful": "Faithful recovery", "original": "Original"}},
    {"key": "show_myinvois", "group": "Display", "label": "Show the e-Invoice (MyInvois) panel", "description": "", "default": True},
    {"key": "open_reading_results", "group": "Display", "label": "Expand the scanner text by default", "description": "", "default": False},
    {"key": "pack_images", "group": "Export", "label": "Include receipt images in the accountant pack",
     "description": "Turn off for a smaller, faster download with the spreadsheet only.", "default": True},
]
PRESETS = {
    "fast": {"show_fields_first": True, "reading_agreement": False, "extra_ocr_view": "never", "second_reading": False, "severe_cleanup": False, "fast_restoration": False, "fast_resolution": True},
    "balanced": {"show_fields_first": True, "reading_agreement": True, "extra_ocr_view": "auto", "second_reading": True, "severe_cleanup": True, "fast_restoration": False, "fast_resolution": False},
    "thorough": {"show_fields_first": True, "reading_agreement": True, "extra_ocr_view": "always", "second_reading": True, "severe_cleanup": True, "damage_separation": True, "stroke_joining": True, "fast_restoration": False, "fast_resolution": False},
}
_BY_KEY = {item["key"]: item for item in SETTINGS}


def defaults() -> dict[str, Any]:
    return {item["key"]: item["default"] for item in SETTINGS}


def resolve(saved: dict[str, Any] | None) -> dict[str, Any]:
    values = defaults()
    for key, value in (saved or {}).items():
        spec = _BY_KEY.get(key)
        if not spec:
            continue
        if "choices" in spec:
            if value in spec["choices"]:
                values[key] = value
        elif isinstance(value, bool):
            values[key] = value
    return values


def clean(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    preset = payload.get("preset")
    values = dict(PRESETS.get(preset, {})) if isinstance(preset, str) else {}
    values.update({k: v for k, v in payload.items() if k != "preset"})
    return {k: v for k, v in resolve(values).items() if k in values or k in PRESETS.get(preset or "", {})}
