"""Second fresh test on real receipts, run once (13 November 2026), for the improved system (reading agreement).

The first fresh test (fresh_test_20261113) was used to design the reading agreement, so it is development data
now. This test uses every remaining eligible photo (same rules: never scanned in the app, not a duplicate, not a
SROIE dataset image, not a screenshot), none of which was looked at before it was selected. P records also hold
the single-reading values (the system before the change), so both versions are compared on the same receipts.

Original description of the first fresh test:

Every earlier test receipt had been used while the system was developed. This test uses real receipt photos
that played no part in any decision: photographed for this test and never scanned in the app before. The
plan is fixed before the run:

* Data: the photos in fresh_test_20261113/photos, labelled by a person from the paper receipts in
  fresh_test_20261113/labels.csv (merchant as printed, date, total payable). Labels are written before the run
  and not changed after it.
* Methods: A direct, B Otsu, B Sauvola, C perceptual and D DSDNet alone on the same preprocessed (cropped and
  turned) image, and P, the full application, on the original photo. Every method uses the same text reader
  and field parser.
* Measures: field accuracy (merchant, date, total) and high-risk accuracy (date, total) per receipt. No clean
  text exists for real receipts, so CER is not measured.
* Analysis: one-sided Wilcoxon signed-rank test per baseline on receipt-level field accuracy, Holm correction
  over the five baselines, 95% bootstrap interval of the difference (10,000 draws); McNemar on single fields.
* The run happens once. Results are reported whatever they show. A second run is refused.

Usage:  python scripts/fresh_test_20261113.py prepare   (makes the folder and labels.csv to fill in)
        python scripts/fresh_test_20261113.py run       (once, after labels.csv is complete)
"""
from __future__ import annotations

import csv
import io
import shutil
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402

ROOT = fe.ROOT / "fresh_test_2_20261113"
FIRST = fe.ROOT / "fresh_test_20261113"
PHOTOS = ROOT / "photos"
LABELS = ROOT / "labels.csv"
OUT = ROOT / "results"
COLUMNS = ["file", "merchant", "transaction_date", "total_amount", "notes"]
EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def prepare() -> None:
    PHOTOS.mkdir(parents=True, exist_ok=True)
    rows = {}
    if LABELS.exists():
        with LABELS.open(newline="", encoding="utf-8-sig") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
    for p in sorted(PHOTOS.iterdir()):
        if p.suffix.lower() in EXT and p.name not in rows:
            rows[p.name] = {c: "" for c in COLUMNS} | {"file": p.name}
    with LABELS.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows.values())
    print(f"{len(rows)} photos listed in {LABELS}")


def select(n: int = 0, seed: int = 0) -> None:
    """Copy every remaining eligible photo from the first fresh test's photos_not_used into this test."""
    import hashlib, sqlite3
    if (ROOT / "selection.json").exists():
        print("selection already made"); return
    scanned = set()
    for audit in (fe.ROOT / "artifacts" / "user_runs").glob("*/audit.json"):
        try:
            h = json.loads(audit.read_text(encoding="utf-8")).get("content_sha256")
            if h: scanned.add(h)
        except (OSError, ValueError):
            pass
    for db in (fe.ROOT / "artifacts" / "app_data").glob("*.sqlite3"):
        try:
            with sqlite3.connect(db) as con:
                scanned.update(r[0] for r in con.execute("SELECT content_sha256 FROM receipts WHERE content_sha256 IS NOT NULL"))
        except sqlite3.Error:
            pass
    seen = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (FIRST / "photos").iterdir() if p.is_file()}
    PHOTOS.mkdir(parents=True, exist_ok=True)
    chosen, excluded = [], {}
    for p in sorted((FIRST / "photos_not_used").iterdir()):
        if p.suffix.lower() not in EXT:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        why = ("duplicate_or_first_test" if h in seen else "scanned_in_app" if h in scanned else
               "sroie_dataset" if p.name.upper().startswith("X5") else "screenshot" if p.name.lower().startswith("screenshot") else None)
        if why:
            excluded[why] = excluded.get(why, 0) + 1; continue
        seen.add(h); chosen.append(p.name)
        shutil.copy2(p, PHOTOS / p.name)
    (ROOT / "selection.json").write_text(json.dumps({"chosen": chosen, "excluded": excluded, "app_hashes": len(scanned)}, indent=1), encoding="utf-8")
    prepare()
    print(f"copied {len(chosen)} photos; excluded {excluded}")


def run() -> None:
    from dsdnet_receipt.baselines import BASELINES
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.ocr import PaddleOCRAdapter
    from dsdnet_receipt.preprocessing import preprocess
    from dsdnet_receipt.restoration import RestorationEngine
    from dsdnet_receipt.settings import PRESETS
    from parser_revision_20261112 import use_revised_parser

    if (OUT / "DONE.txt").exists():
        print("The fresh test has already been run once; it is not run again."); return
    with LABELS.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("file")]
    # Pre-run exclusion rule (fixed before any output is seen): a photo is left out when its
    # notes start with EXCLUDE or it has no complete label (merchant, date and total).
    def usable(r):
        return (r["merchant"].strip() and r["transaction_date"].strip() and r["total_amount"].strip()
                and not r.get("notes", "").strip().upper().startswith("EXCLUDE"))
    labels = [r for r in rows if usable(r)]
    left_out = [{"file": r["file"], "notes": r.get("notes", "")} for r in rows if not usable(r)]
    print(f"{len(labels)} labelled receipts will be used; {len(left_out)} left out before the run:")
    for r in left_out:
        print(f"  - {r['file']}: {r['notes']}")
    if len(labels) < 20:
        print("Fewer than 20 usable receipts. Nothing was run.")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "doc": __doc__,
                                               "receipts": len(labels), "left_out": left_out}, indent=1), encoding="utf-8")
    (OUT / "labels_frozen.csv").write_bytes(LABELS.read_bytes())

    use_revised_parser()
    import dsdnet_receipt.fields as fields_module
    from dsdnet_receipt.web_app import create_app, decode_upright
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    ocr = PaddleOCRAdapter(config)
    engine = RestorationEngine(config)
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "fresh.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token})

    path = OUT / "records.jsonl"
    with path.open("a", encoding="utf-8") as sink:
        for n, row in enumerate(labels, start=1):
            refs = {"merchant": row["merchant"], "transaction_date": row["transaction_date"], "total_amount": row["total_amount"]}
            raw = (PHOTOS / row["file"]).read_bytes()

            def emit(method, values, seconds, error=None):
                rec = {"file": row["file"], "method": method, "seconds": seconds, "error": error,
                       **{f"{k}_value": values.get(k) for k in fe.FIELDS},
                       **{f"{k}_correct": bool(fe.field_correct(k, values.get(k), refs[k])) for k in fe.FIELDS}}
                sink.write(json.dumps(rec) + "\n"); sink.flush()

            try:
                image = decode_upright(raw)
                registered, _ = preprocess(image, config, preserve_color=True)
                gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY) if registered.ndim == 3 else registered
                for method in ("A_direct", "B_otsu", "B_sauvola", "C_perceptual"):
                    t0 = time.perf_counter()
                    result = ocr.run(BASELINES[method](gray))
                    found = fields_module.extract_fields(result.text, result.lines)
                    emit(method, {k: (found.get(k) or {}).get("value") for k in fe.FIELDS}, time.perf_counter() - t0)
                t0 = time.perf_counter()
                restoration = engine.run(gray)
                restored = restoration.model_output if restoration.model_output is not None else restoration.image
                result = ocr.run(restored)
                found = fields_module.extract_fields(result.text, result.lines)
                emit("D_dsdnet_raw", {k: (found.get(k) or {}).get("value") for k in fe.FIELDS}, time.perf_counter() - t0)
            except Exception as error:  # a failure is scored as a miss for the methods that did not finish
                emit("ERROR_BASELINES", {}, 0.0, repr(error))
            t0 = time.perf_counter()
            response = client.post("/api/analyze", data={"receipt": (io.BytesIO(raw), row["file"])},
                                   headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
            body = response.get_json(silent=True) or {}
            receipt = body.get("receipt") or {}
            seconds = time.perf_counter() - t0
            error = None if response.status_code == 200 else (body.get("error") or f"HTTP {response.status_code}")
            agreed = {k: receipt.get(k) for k in fe.FIELDS}
            single = dict(agreed); single.update(((body.get("audit") or {}).get("reading_agreement") or {}).get("single_reading") or {})
            emit("P_full", agreed, seconds, error)
            emit("P_single_reading", single, seconds, error)
            if receipt.get("public_id"):
                client.delete(f"/api/receipts/{receipt['public_id']}", headers={"X-CSRF-Token": token})
            print(f"[{n}/{len(labels)}] {row['file']}", flush=True)
    (OUT / "DONE.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        {"prepare": prepare, "run": run, "select": select}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
    finally:
        try:
            from dsdnet_receipt.ocr import shutdown_gpu_worker
            shutdown_gpu_worker()
        except Exception:
            pass
