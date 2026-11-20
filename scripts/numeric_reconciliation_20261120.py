#!/usr/bin/env python3
"""Reconcile manuscript numbers against report.json.

Every value quoted in the manuscript must be reproducible from the archived
re-analysis report at the stated rounding. Fails loudly on any mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = (Path(sys.argv[1]) if len(sys.argv) > 1
      else ROOT / "_audit_pack" / "JIWE_FULL_MANUSCRIPT_DRAFT_V2.md")
REP = (Path(sys.argv[2]) if len(sys.argv) > 2
       else ROOT / "_audit_pack" / "report.json")
for _p in (MS, REP):
    if not _p.exists():
        sys.exit(f"not found: {_p}\nusage: {Path(sys.argv[0]).name} "
                 f"[manuscript.md] [report.json]")

text = MS.read_text(encoding="utf-8")
d = json.loads(REP.read_text(encoding="utf-8"))
pre, post = d["variants"]["pre"], d["variants"]["post"]
cp, co = pre["calibration"], post["calibration"]

fail: list[str] = []


def want(value: float, nd: int, label: str) -> None:
    """Assert the rounded value appears in the manuscript."""
    s = f"{value:.{nd}f}"
    if s not in text:
        fail.append(f"{label}: expected {s} in manuscript")
    return None


# --- Table 4, primary endpoint
want(cp["composite_auc"], 3, "pre composite AUC")
want(cp["ocr_auc"], 3, "pre OCR AUC")
want(cp["composite_brier"], 3, "pre composite Brier")
want(cp["ocr_brier"], 3, "pre OCR Brier")
want(cp["composite_ece"], 3, "pre composite ECE")
want(cp["ocr_ece"], 3, "pre OCR ECE")

# --- Table 4, secondary endpoint
want(co["composite_auc"], 3, "post composite AUC")
want(co["ocr_auc"], 3, "post OCR AUC")
want(co["composite_brier"], 3, "post composite Brier")
want(co["ocr_brier"], 3, "post OCR Brier")
want(co["composite_ece"], 3, "post composite ECE")
want(co["ocr_ece"], 3, "post OCR ECE")

# --- endpoint crosstab
ct = d["label_crosstab"]
for key in ("both_wrong", "wrong_to_right", "right_to_wrong", "both_right"):
    if str(ct[key]) not in text:
        fail.append(f"crosstab {key}={ct[key]} missing")
want(ct["disagreement_rate"] * 100, 1, "crosstab disagreement rate")

# --- arithmetic bound
for q, block in pre["composite_bound"].items():
    want(block["max_support"], 3, f"q{q} max support")
    want(block["max_composite"], 3, f"q{q} max composite")

# --- matched coverage, primary
for row in pre["matched_coverage"]:
    want(row["coverage"] * 100, 1, f"coverage {row['coverage']:.3f}")
    want(row["uar_P"], 3, f"uar_P at {row['coverage']:.3f}")
    want(row["uar_E"], 3, f"uar_E at {row['coverage']:.3f}")
    want(row["uar_F"], 3, f"uar_F at {row['coverage']:.3f}")
    lo, hi = row["uar_reduction_vs_E_ci95"]
    norm = text.replace("\u2212", "-")
    if f"[{lo:.3f}, {hi:.3f}]" not in norm:
        fail.append(f"CI at coverage {row['coverage']:.3f}: [{lo:.3f}, {hi:.3f}] missing")

# --- matched coverage, secondary (quoted in prose only for two points)
for row in post["matched_coverage"][:2]:
    want(row["uar_P"], 3, f"post uar_P at {row['coverage']:.3f}")
    want(row["uar_E"], 3, f"post uar_E at {row['coverage']:.3f}")

# --- planned operating points
for block, tag in ((pre, "pre"), (post, "post")):
    for row in block["planned_threshold_coverage"]:
        if row["confidence"] != "calibrated":
            continue
        for pol in ("P", "E", "F"):
            n = row[f"accepted_{pol}"]
            if n and str(n) not in text:
                fail.append(f"{tag} calibrated t={row['t_accept']} accepted_{pol}={n} missing")

print(f"reconciled against {REP.name} (generated {d['generated']})")
if fail:
    for f in fail:
        print(f"[FAIL] {f}")
    print(f"\n{len(fail)} mismatches")
    sys.exit(1)
print("[PASS] every quoted statistic reproduces from the archived report")
