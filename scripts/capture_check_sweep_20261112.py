"""How often would the after-scan "try again" prompt appear on receipts that were scanned correctly?

Runs the capture check over every saved audit of the parser-revision run (synthetic degradations of
upright, fully framed receipts, so every prompt there is a false alarm) and over any receipts saved in
the app. Writes artifacts/capture_check_sweep.json. Reads only; changes nothing.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsdnet_receipt.capture_check import check_capture  # noqa: E402


def sweep(paths):
    counts, flagged = collections.Counter(), []
    for audit_path in paths:
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        image = cv2.imread(str(audit_path.parent / "rectified_input.png"), cv2.IMREAD_GRAYSCALE)
        fields = {f.get("name"): f.get("value") for f in audit.get("fields") or [] if isinstance(f, dict)}
        out = check_capture(audit.get("ocr_lines") or [], image.shape if image is not None else None, fields,
                            (audit.get("preprocessing") or {}).get("upright_check"))
        counts[out["issue"] or "ok"] += 1
        if out["status"] == "retry":
            flagged.append({"run": audit_path.parent.name, "issue": out["issue"], "signals": out["signals"]})
    total = sum(counts.values())
    return {"receipts": total, "counts": dict(counts), "prompt_rate": (total - counts["ok"]) / total if total else None,
            "flagged": flagged[:60]}


def main() -> None:
    evaluation = sorted((ROOT / "artifacts" / "parser_revision_20261112" / "attempt2").glob("*/sandbox/artifacts/user_runs/*/audit.json"))
    app = sorted((ROOT / "artifacts" / "user_runs").glob("*/audit.json"))
    report = {"evaluation_receipts_all_upright_and_framed": sweep(evaluation), "app_receipts": sweep(app)}
    (ROOT / "artifacts" / "capture_check_sweep.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("receipts", "counts", "prompt_rate")} for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
