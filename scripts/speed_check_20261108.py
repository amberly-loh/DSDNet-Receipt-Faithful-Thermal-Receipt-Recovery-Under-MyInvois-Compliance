"""Speed and no-harm check on validation receipts only (8 November 2026).

Runs the deployed application (balanced preset) on the 5 validation source receipts used by
the confirmatory no-harm check (15 images), under one processing configuration given by
environment variables (RECEIPT_TILE_BATCH, RECEIPT_OCR_GPU). Records end-to-end time,
stage times, text and fields so that a faster configuration is adopted only if it does not
reduce extraction quality. No test receipt is used.
"""
from __future__ import annotations

import argparse, io, json, os, sys, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401  (loads the GPU text reader first when it is enabled)

OUT = fe.OUT / "speed_20261108"


def main(label: str, validation_count: int) -> None:
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.metrics import cer, wer
    from dsdnet_receipt.receipt_context import reading_copy_lines
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    references = fe.load_references()
    variants = fe.load_variants()
    val_order = [v for v in fe.sampled_variants(variants, 0, 40) if v["project_partition"] == "validation"]
    val_sources = list(dict.fromkeys(v["source_id"] for v in val_order))[:validation_count]
    plan = [v for s in val_sources for v in val_order if v["source_id"] == s]
    sandbox = OUT / f"sandbox_{label}"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = sandbox / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "speed.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token})

    def analyse(image, name):
        ok, encoded = cv2.imencode(".png", image)
        t0 = time.perf_counter()
        response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), name)},
                               headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
        return response, time.perf_counter() - t0

    # warm-up on the first image so that model loading is not counted (the running app is already loaded)
    warm, warm_seconds = analyse(fe.read_gray(plan[0]["degraded_path"]), "warmup.png")
    path = OUT / f"{label}.jsonl"
    path.write_text("", encoding="utf-8")
    env = {k: os.environ.get(k) for k in ("RECEIPT_TILE_BATCH", "RECEIPT_OCR_GPU", "RECEIPT_MODEL_VIEW", "PYTHONPATH")}
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps({"label": label, "env": env, "warmup_seconds": warm_seconds, "warmup_status": warm.status_code}) + "\n")
        for v in plan:
            key = f"{v['source_id']}/{v['variant_id']}"
            reference = references[v["source_id"]]
            reference_text = fe.norm_text(reference["ocr_reference_text"])
            refs = {"merchant": reference["company"], "transaction_date": reference["transaction_date"], "total_amount": reference["total_amount"]}
            response, seconds = analyse(fe.read_gray(v["degraded_path"]), f"{v['source_id']}_{v['variant_id']}.png")
            body = response.get_json(silent=True) or {}
            row = {"variant_key": key, "source_id": v["source_id"], "severity": v["severity"], "seconds": seconds, "status": response.status_code}
            if response.status_code == 200:
                receipt = body.get("receipt") or {}
                audit = body.get("audit") or {}
                lines = audit.get("ocr_lines") or []
                hyp = fe.norm_text("\n".join(l["text"] for l in reading_copy_lines(lines, tidy=False)))
                row.update({"cer": float(cer(reference_text, hyp)), "wer": float(wer(reference_text, hyp)), **fe.word_bag_scores(reference_text, hyp),
                            "stage_seconds": audit.get("stage_seconds"), "views": audit.get("ocr_input_images"), "text": hyp})
                for n in fe.FIELDS:
                    row[f"{n}_value"] = receipt.get(n)
                    row[f"{n}_correct"] = bool(fe.field_correct(n, receipt.get(n), refs[n]))
                client.delete(f"/api/receipts/{receipt.get('public_id')}", headers={"X-CSRF-Token": token})
            else:
                row["error"] = body.get("error")
            sink.write(json.dumps(row) + "\n"); sink.flush()
            print(label, key, f"{seconds:.1f}s", flush=True)
    (OUT / f"{label}.done").write_text("done", encoding="utf-8")


def _finish() -> None:
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--validation-count", type=int, default=5)
    a = parser.parse_args()
    if a.label in ("current", "batch16_cpu_skip"):
        # superseded: the previous configuration's validation times and outputs are already
        # recorded by the confirmatory validation check (P_full on the same 5 sources)
        sys.exit(0)
    main(a.label, a.validation_count)
    _finish()
