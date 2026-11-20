#!/usr/bin/env python3
"""Reproduce the RQ1 six-method comparison and the field-level support
distribution from the committed per-record outputs.

Every number this prints is a value the manuscript reports, so a reader can
check the Results section against the archive without re-running the pipeline.
No model inference and no image access is involved: the script only aggregates
`experiments/**/records.jsonl`.

Run from the repository root:

    python scripts/rq1_comparison_20261120.py

Definitions used here match the manuscript:

* The primary window is positions 41-95 of the locked test partition: 55 sources,
  165 images, 495 field instances.
* Field accuracy counts the three extracted fields (merchant, transaction date,
  total amount). High-risk accuracy counts the two fields that carry the
  acceptance decision (transaction date, total amount).
* The proposed pipeline is scored on its five-reading vote (`agreed_*`); the
  baselines are single-reading and use their own correctness fields.
* Page CER, WER and Word F1 are means over images, not pooled over characters.
* Support is the lower quantile selected on validation data (q10), not the mean
  over retained pixels. Support is undefined when a field box retains fewer than
  20 ink pixels, which is why the support counts below are smaller than the
  field-instance count.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
FIELDS = ("merchant", "transaction_date", "total_amount")
HIGH_RISK = ("transaction_date", "total_amount")
WINDOW = (41, 95)


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def in_window(rows: list[dict]) -> list[dict]:
    lo, hi = WINDOW
    return [r for r in rows if r.get("position") and lo <= r["position"] <= hi]


def tally(rows: list[dict], fields: tuple[str, ...], prefix: str) -> tuple[int, int]:
    hits = sum(1 for r in rows for f in fields if r[f"{prefix}{f}_correct"])
    return hits, len(rows) * len(fields)


def comparison_table() -> None:
    proposed = in_window([r for r in load(EXP / "agreement_20261113" / "records.jsonl")
                          if r["method"] == "P_full"])
    baseline = in_window(load(EXP / "parser_revision_20261112" / "attempt2" / "locked"
                              / "records.jsonl"))

    groups = [("Proposed (five-reading)", proposed, "agreed_")]
    for name in ("D_dsdnet_raw", "A_direct", "C_perceptual", "B_otsu", "B_sauvola"):
        groups.append((name, [r for r in baseline if r["method"] == name], ""))

    print("## RQ1 comparison, all six methods")
    print()
    print(f"Primary window, positions {WINDOW[0]}-{WINDOW[1]}: "
          f"{len({r['source_id'] for r in proposed})} sources, "
          f"{len(proposed)} images, {len(proposed) * len(FIELDS)} field instances.")
    print()
    print("| Method | Field acc. | High-risk acc. | Merchant | Date | Amount | Page CER | WER | Word F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, rows, prefix in groups:
        fa = tally(rows, FIELDS, prefix)
        hr = tally(rows, HIGH_RISK, prefix)
        cells = [label, f"{fa[0]}/{fa[1]} = {fa[0] / fa[1]:.4f}",
                 f"{hr[0]}/{hr[1]} = {hr[0] / hr[1]:.4f}"]
        for field in FIELDS:
            hit, total = tally(rows, (field,), prefix)
            cells.append(f"{hit}/{total} = {hit / total:.4f}")
        cells.append(f"{statistics.fmean(r['cer'] for r in rows):.4f}")
        cells.append(f"{statistics.fmean(r['wer'] for r in rows):.4f}")
        cells.append(f"{statistics.fmean(r['word_f1'] for r in rows):.4f}")
        print("| " + " | ".join(cells) + " |")
    print()


def support_distribution() -> None:
    rows = [r for r in load(EXP / "final_run_20261108" / "records.jsonl")
            if r["method"] == "D_rq3" and r["partition"] == "test"]

    instances = sum(1 for r in rows for f in HIGH_RISK)
    localisation_failures = sum(
        1 for r in rows for f in HIGH_RISK
        if not r["field_evidence"][f].get("box_valid", True))

    print("## Field-level support distribution")
    print()
    print(f"Source: `experiments/final_run_20261108/records.jsonl`, method `D_rq3`, "
          f"test partition, {len(rows)} images.")
    print()
    print("| Quantity | Value |")
    print("|---|---:|")
    print(f"| High-risk field instances | {instances} |")
    print(f"| Instances failing the localisation test | {localisation_failures} |")
    print()
    print("| Quantile | n | median | mean | minimum | maximum |")
    print("|---|---:|---:|---:|---:|---:|")
    for key, label in (("q10", "q10 (selected on validation)"),
                       ("q15", "q15"),
                       ("q20", "q20"),
                       ("support_mean", "Mean over retained pixels")):
        values = [r["field_evidence"][f][key] for r in rows for f in HIGH_RISK
                  if r["field_evidence"][f].get(key) is not None]
        print(f"| {label} | {len(values)} | {statistics.median(values):.4f} | "
              f"{statistics.fmean(values):.4f} | {min(values):.4f} | {max(values):.4f} |")
    print()
    print("The selection of q10 over q15 and q20 is recorded in "
          "`experiments/final_evaluation_20261107/final_evaluation_report.json` "
          "at `rq3_validation_selection.P.chosen.q`.")
    print()


def main() -> int:
    comparison_table()
    support_distribution()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
