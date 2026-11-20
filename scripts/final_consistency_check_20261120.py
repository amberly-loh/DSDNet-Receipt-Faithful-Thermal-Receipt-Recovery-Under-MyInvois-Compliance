#!/usr/bin/env python3
"""Final pre-submission consistency check for the JIWE manuscript.

Pure standard library. Reports PASS/FAIL for a fixed set of mechanical checks:
  1. withdrawn numeric claims must not appear as current results
  2. table / figure numbering must be contiguous and every object cited
  3. section numbering must be contiguous
  4. reference markers must be sequential from 1 and all cited
  5. length / abstract / keyword constraints
  6. required declaration blocks present
Exit code 0 only if no FAIL.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_DEFAULT = (Path(__file__).resolve().parents[1] / "_audit_pack"
            / "JIWE_FULL_MANUSCRIPT_DRAFT_V2.md")
MS = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
if not MS.exists():
    sys.exit(f"manuscript not found: {MS}\nusage: {Path(sys.argv[0]).name} [manuscript.md]")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))


text = MS.read_text(encoding="utf-8")
lines = text.splitlines()

# ---------------------------------------------------------------- 1. withdrawn
# Numbers that were withdrawn. Each entry: (pattern, allowed-context-regex).
# A hit is a FAIL unless the surrounding sentence marks it as withdrawn/superseded.
WITHDRAWN = [
    ("0.375", r"withdraw|retract|not reproducible|no longer|supersed|earlier version|previous draft|previously reported"),
    ("0.040", r"withdraw|retract|not reproducible|no longer|supersed|earlier version|previous draft|previously reported"),
    ("0.386", r"withdraw|retract|no longer|supersed|earlier version|previous draft|previously reported"),
    ("0.094", r"withdraw|retract|no longer|supersed|earlier version|previous draft|corrected|revised"),
    ("0.059", r"withdraw|retract|no longer|supersed|earlier version|previous draft|corrected|revised"),
    ("0.074", r"withdraw|retract|no longer|supersed|earlier version|previous draft|previously reported"),
    ("59.6", r"withdraw|retract|not reproducible|no longer|supersed|earlier version|previous draft"),
    ("0.762", r"withdraw|retract|no longer|supersed|earlier version|previous draft"),
    ("recalibrated threshold", r"withdraw|retract|no longer|supersed|not reported|planned operating"),
    ("re-selected threshold", r"withdraw|retract|no longer|supersed|not reproducible"),
    ("re-selection", r"withdraw|retract|no longer|supersed|not reproducible"),
]

# Manually reviewed occurrences where the number recurs legitimately, not as a
# reinstated claim. Each entry is (withdrawn token, exact line fragment, reason).
REVIEWED = [
    ("0.059", "| OCR-confidence ECE | 0.069 | 0.059 |",
     "secondary post-agreement OCR ECE, independently recomputed as 0.0590; "
     "coincides with the withdrawn published value but is a different quantity"),
]

bad: list[str] = []
for pat, allow in WITHDRAWN:
    for i, ln in enumerate(lines, 1):
        if pat.lower() in ln.lower():
            if any(p == pat and frag in ln for p, frag, _ in REVIEWED):
                continue
            # look at the sentence window (this line + previous line) for a withdrawal cue
            window = " ".join(lines[max(0, i - 2):i + 1]).lower()
            if not re.search(allow, window):
                bad.append(f"L{i}: '{pat}' without withdrawal cue -> {ln.strip()[:110]}")
check("withdrawn values not presented as current", not bad, "\n    ".join(bad) if bad else
      f"all withdrawn figures carry an explicit withdrawal cue ({len(REVIEWED)} reviewed exception)")

# ------------------------------------------------------------------ 2. tables
# main-text citations only: everything before the Supplementary Material block
main_text = text.partition("# Supplementary Material")[0]

tabs = sorted({int(m) for m in re.findall(r"\*\*Table (\d+)\.", text)})
tab_cites = sorted({int(m) for m in re.findall(r"Table (\d+)(?![\d.])", main_text)})
check("Table captions contiguous from 1",
      tabs == list(range(1, max(tabs) + 1)) if tabs else False, f"captions: {tabs}")
uncited_t = [t for t in tabs if t not in tab_cites]
check("every table cited in main text", not uncited_t,
      f"uncited: {uncited_t}" if uncited_t else f"{len(tabs)} tables all cited in main text")

figs = sorted({int(m) for m in re.findall(r"\*\*Figure (\d+)\.", text)})
fig_cites = sorted({int(m) for m in re.findall(r"Figure (\d+)(?![\d.])", main_text)})
check("Figure captions contiguous from 1",
      figs == list(range(1, max(figs) + 1)) if figs else False, f"captions: {figs}")
uncited_f = [f for f in figs if f not in fig_cites]
check("every figure cited in main text", not uncited_f,
      f"uncited: {uncited_f}" if uncited_f else f"{len(figs)} figures all cited in main text")

# ---------------------------------------------------------------- 3. sections
lvl1 = [int(m) for m in re.findall(r"^# (\d)\. [A-Z]", text, re.M)]
check("level-1 sections are 1..5", lvl1 == [1, 2, 3, 4, 5], f"found {lvl1}")

subs = re.findall(r"^## (\d)\.(\d+) ", text, re.M)
for major in "234":
    got = [int(b) for a, b in subs if a == major]
    check(f"section {major}.x contiguous",
          got == list(range(1, len(got) + 1)), f"{major}.1-{major}.{len(got)} -> {got}")

# --------------------------------------------------------------- 4. references
body, _, tail = text.partition("# References")
ref_block, _, _ = tail.partition("# Author Biographies")
ref_nums = [int(m) for m in re.findall(r"^\[(\d+)\]", ref_block, re.M)]
check("reference list is 1..N sequential", ref_nums == list(range(1, len(ref_nums) + 1)),
      f"{len(ref_nums)} entries, first={ref_nums[:1]}, last={ref_nums[-1:]}")

cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", body)})
missing = [c for c in cited if c not in ref_nums]
check("no citation to a non-existent reference", not missing, f"dangling: {missing}" if missing else "none")
never = [n for n in ref_nums if n not in cited]
check("no uncited reference", not never, f"uncited: {never}" if never else "all cited")

# ------------------------------------------------------------------- 5. length
abstract = re.search(r"## Abstract\n\n(.+?)\n\n", text, re.S)
aw = len(abstract.group(1).split()) if abstract else -1
check("abstract <= 300 words", 0 < aw <= 300, f"{aw} words")

kw_line = re.search(r"\*\*Keywords?[—:\-]*\*\*\s*(.+)", text)
kws = [k.strip() for k in re.split(r"[;,]", kw_line.group(1))] if kw_line else []
check("5-7 keywords", 5 <= len(kws) <= 7, f"{len(kws)}: {kws}")

title = lines[0].lstrip("# ").strip()
check("title <= 15 words", len(title.split()) <= 15, f"{len(title.split())} words: {title}")

plain = re.sub(r"```.*?```", "", text, flags=re.S)
plain = re.sub(r"\|.*?\|", "", plain)
plain = re.sub(r"[#*`>]", "", plain)
words = len(plain.split())
check("body within 6,000-12,000 words", 6000 <= words <= 12000, f"~{words} words")

# ------------------------------------------------------------- 6. declarations
for block in ["Acknowledgement", "Funding Statement", "Author Contributions",
              "Conflict of Interests", "Ethics Statement", "Data Availability", "Code Availability"]:
    check(f"declaration present: {block}", f"## {block}" in text, "")

decl_pos = text.find("# Declarations")
ref_pos = text.find("# References")
check("Declarations precede References", 0 < decl_pos < ref_pos, f"{decl_pos} < {ref_pos}")

# placeholder scan
ph = re.findall(r"(?i)\b(TBD|TODO|XXX|\[insert|placeholder|lorem)\b", text)
check("no unresolved placeholders", not ph, f"found: {set(ph)}" if ph else "none")

# formula integrity
check("field-score formula has full subscripts",
      "0.45 · S_support + 0.35 · S_ocr + 0.20 · S_rule" in text,
      "S_f = 0.45·S_support + 0.35·S_ocr + 0.20·S_rule")
check("no truncated subscript form", "0.45S_" not in text.replace("0.45 · S_", ""), "")

# ---------------------------------------------------------------------- report
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"file: {MS.name}\n")
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")
print(f"\n{n_pass}/{len(results)} checks passed")
sys.exit(0 if n_pass == len(results) else 1)
