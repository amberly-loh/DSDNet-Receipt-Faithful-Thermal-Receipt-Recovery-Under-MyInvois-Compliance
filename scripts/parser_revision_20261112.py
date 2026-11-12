"""Parser revision: one-time test run of the shared field parser revision (12 November 2026).

Disclosed post-evaluation revision. Chapter 5 (error analysis, 5.6) found the total amount to be the
field that limits every method: the shared field rule picked the wrong one of several amounts
(subtotal, rounding, cash) or missed a total printed on a separate line from its label. A review
step for the date and the total was then written from receipts scanned in the application (not from
any evaluation receipt) and checked once on the 40 validation sources only
(artifacts/dev_validation_20261108; no test receipt involved):

    validation, 120 images      field accuracy        high-risk accuracy
    D (DSDNet alone)            0.289 -> 0.367        0.325 -> 0.442
    P (proposed system)         0.317 -> 0.406        0.383 -> 0.517

It was adopted on that basis and frozen (src/dsdnet_receipt/field_review.py). The review runs after the
evaluated extractor (fields.extract_fields, unchanged) and keeps the box and confidence of the line each
new value was read from. For this run it is the shared field parser of every method: A, B-Otsu,
B-Sauvola, C, D, P (inside the application) and the D_rq3 evidence for RQ2/RQ3.

This script repeats the locked evaluation loop (scripts/final_run_20261108.py) unchanged apart from the
parser: the 40 validation sources (RQ2/RQ3 calibration and selection), the 95 locked test sources
(primary 41-95) and the 53 replication sources. AC1, AC2/AC3, AC4 and NFR1 decision rules are
unchanged. Results are reported whatever they show, beside the locked, replication and 9 November
revision results, which are retained.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401  (loads the GPU text reader first when it is enabled)
import final_run_20261108 as fr  # noqa: E402

RUN_ROOT = fe.ROOT / "artifacts" / "parser_revision_20261112"
# Attempt 1 (00:04-00:24, 12 Nov) failed on every image with a RecursionError in the parser wrapper
# before any field was extracted or scored; its folders (locked/, replication/) are kept as they are.
# Attempt 2 runs the same frozen plan after the one-line fix (field_review.py binds the evaluated
# extractor at import) and a test that reproduces this script's swap.
OUT = RUN_ROOT / "attempt2"
REPLICATION_MANIFEST = fe.ROOT / "artifacts" / "replication_20261108" / "replication_manifest.jsonl"


def use_revised_parser() -> None:
    # The application binds its own extractor (and applies the review itself) when it is imported, so
    # import it first; then every other method's call to fields.extract_fields gets the revised parser.
    import dsdnet_receipt.web_app  # noqa: F401
    import dsdnet_receipt.fields as fields
    from dsdnet_receipt.field_review import extract_fields_reviewed
    fields.extract_fields = extract_fields_reviewed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = OUT / "plan.json"
    # fail fast: the revised parser must run once on a real receipt layout before the long loop starts
    use_revised_parser()
    from dsdnet_receipt.fields import extract_fields
    extract_fields("TOTAL 12.90", [{"text": "TOTAL 12.90", "confidence": 0.9, "box": [[0, 0], [100, 0], [100, 20], [0, 20]]}])
    if not plan.exists():
        plan.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "doc": __doc__,
                                    "parts": ["locked: 40 validation sources (D_rq3) + 95 test sources, all methods",
                                              "replication: 53 test sources, all methods"],
                                    "attempt": "2 (attempt 1 failed before any field was scored; see the note at the top of this script)",
                                    "decision_rules": fr.PLAN}, indent=1), encoding="utf-8")
    load_variants, sampled_variants = fe.load_variants, fe.sampled_variants

    fr.OUT = OUT / "locked"
    fr.main(None)

    variants = [json.loads(line) for line in REPLICATION_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    fr.OUT = OUT / "replication"
    fe.load_variants = lambda: variants
    fe.sampled_variants = lambda v, t, s: list(v)
    try:
        fr.main("test")
    finally:
        fe.load_variants, fe.sampled_variants = load_variants, sampled_variants
    (OUT / "DONE.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    finally:
        from dsdnet_receipt.ocr import shutdown_gpu_worker
        shutdown_gpu_worker()
