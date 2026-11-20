# RQ2/RQ3 re-analysis, 20 November 2026

This directory holds the aggregate output of a re-analysis that replaces the
field-level calibration and selective-acceptance numbers in the manuscript.
It exists because the previously reported numbers could not be reproduced from
the archive: the support evidence and the correctness label came from two
different runs five days apart, and the join that crossed them was never
deposited.

## What was run

`scripts/rq2_rq3_reanalysis_20261120.py` reads only machine-readable records
from the private data repository:

| Role | Source | Size |
|---|---|---|
| Validation (selects `q`, fits isotonic maps) | `final_run_20261108`, method `D_rq3`, `partition = validation` | 40 sources, 120 images, 240 high-risk fields |
| Test (evaluated) | same run and method, `partition = test` | 95 sources, 285 images, 570 high-risk fields |
| Post-agreement labels (secondary variant only) | `agreement_20261113`, joined on `variant_key` | 570 fields |

The protocol matches `scripts/final_evaluation_20261107.py`: identical score
weights (0.45 support, 0.35 calibrated OCR confidence, 0.20 rule), identical
pool-adjacent-violators isotonic fit, identical validation selection rule,
identical matched-coverage construction, identical source-cluster bootstrap
(10,000 draws, seed 56).

The bootstrap draw count was 2,000 in the first version of this directory. It was
raised to 10,000 to match the protocol as the manuscript states it and as
`final_evaluation_20261107.py` implements it. The primary (pre-agreement)
intervals are unchanged to four decimal places, because resampling 95 source
clusters yields a discrete distribution whose 2.5th and 97.5th percentiles land on
the same realised values either way. Three secondary (post-agreement) interval
endpoints moved in the third decimal; none is quoted in the manuscript.

## Headline result

The direction of the calibration comparison depends on the correctness label,
and the single-configuration analysis reverses the previously reported one.

| Quantity | Previously reported | Pre-agreement (single config) | Post-agreement (cross-stage) |
|---|---:|---:|---:|
| Composite AUC | 0.870 | 0.884 | 0.781 |
| OCR-confidence AUC | 0.841 | 0.799 | 0.765 |
| Composite Brier | 0.138 | 0.128 | 0.190 |
| OCR-confidence Brier | 0.154 | 0.166 | 0.196 |
| Composite ECE | 0.094 | **0.054** | 0.079 |
| OCR-confidence ECE | 0.059 | **0.069** | 0.059 |

Under the internally consistent analysis the composite score is better
calibrated than OCR confidence; the previously reported worse-than-OCR ordering
appears only when pre-agreement scores are evaluated against post-agreement
labels.

## Second result: the zero-coverage outcome is arithmetic

The composite is `0.45 * support + 0.35 * c_ocr + 0.20 * rule`, so it is bounded
by `0.45 * max(support) + 0.55`. With the realised support maxima the bound is
0.777 (q10), 0.783 (q15) and 0.787 (q20), all below the smallest planned
acceptance threshold of 0.80. No field can be accepted at any planned operating
point, for any calibration map. This is a specification error, not a data
property.

## Regenerate

```bash
python scripts/rq2_rq3_reanalysis_20261120.py \
    --data-root /path/to/DSDNet-Receipt-Data-Private-main \
    --out artifacts/rq2_rq3_reanalysis_20261120/report.json

python scripts/rq2_rq3_figures_20261120.py \
    --report artifacts/rq2_rq3_reanalysis_20261120/report.json \
    --out artifacts/rq2_rq3_reanalysis_20261120/figures
```

The figure script writes `fig1_reliability.png` and `fig2_matched_coverage.png`.
Only aggregate outputs are committed here; no receipt images, no per-field rows,
no private paths.

## Figures

Both figures carry two panels. The single-panel versions withheld the evidence
each one argues from: the reliability curve was drawn through probability bins
holding between 19 and 261 fields without showing which, and the matched-coverage
figure plotted point estimates while the interval carrying the comparative claim
sat in a table.

| | Panel (a) | Panel (b) |
|---|---|---|
| `fig1_reliability` | Reliability curves, composite score against calibrated OCR confidence | Fields in each probability bin |
| `fig2_matched_coverage` | Unsupported acceptance rate against coverage, three policies | Paired reduction against Baseline E with its 95% source-cluster bootstrap interval |

Filled markers in `fig2` panel (b) mark the two coverage points where the interval
excludes zero, at 24.9% and 40.0%; open markers mark the three where it does not.

### Layout gates

`--qa` runs the panel-alignment gate, writes a PDF for the rendered-text and
collision audits, and writes `<stem>.alignment.json`. The gate module
`audit_panel_alignment.py` is not part of this repository. Set `FIGURE_QA_DIR` to
the directory holding it to run the gate; without it, `--qa` exits with an error
rather than skipping the check. The `.alignment.json` files committed here are the
gate's output, so the check can be confirmed without installing anything.

All three gates pass on the committed figures:

| Gate | Figure 3 | Figure 4 |
|---|---|---|
| Panel alignment | PASS | PASS |
| Minimum rendered text, floor 5 pt | PASS, 7.5 pt | PASS, 7.5 pt |
| Text-stroke collisions | PASS, 0 fail 0 warn | PASS, 0 fail 0 warn |

## Files

- `report.json` — full aggregate report, both label variants, reliability bins
- `summary.csv` — flat list of headline numbers
- `figures/` — generated by the figure script
  - `fig1_reliability.png`, `fig2_matched_coverage.png` — 600 dpi, drawn at 6.6 in
  - `fig1_reliability.pdf`, `fig2_matched_coverage.pdf` — QA only, not submission files
  - `fig1_reliability.alignment.json`, `fig2_matched_coverage.alignment.json` — gate output

The two PDFs are the one pair of files here that `artifacts/SHA256SUMS` does not
pin. Matplotlib writes a `CreationDate` into every PDF, so a regenerated file has
a different digest even when the drawing is identical, and a pinned digest would
make `sha256sum -c` fail on a clean rerun. They exist to feed the rendered-text
and collision audits; the PNGs are what the manuscript uses, and the PNGs are
pinned. Both text files that the scripts write (`report.json`, `summary.csv`) and
the gate's JSON are written with LF line endings, because `artifacts/**` is pinned
`-text` in `.gitattributes` and a CRLF write would produce a different digest on
Windows than on Linux.
