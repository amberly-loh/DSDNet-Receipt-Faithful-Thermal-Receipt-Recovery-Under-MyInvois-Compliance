"""NFR1 clean re-time of the final system (12 November 2026).

The parser-revision run (artifacts/parser_revision_20261112/attempt2) was interrupted by a restart and two
sleeps, and the receipt app shared the GPU for part of it, so its per-image times are not a valid NFR1
measurement. This script times the final system (P_full, revised shared parser) again on the same 285 locked
test images, with nothing else running, under the same conditions as the locked run: balanced preset, GPU text
reader, both engines warmed before timing, one image at a time.

Nothing about the system is changed or chosen here; field values are recorded only to confirm they match the
parser-revision run. Output: artifacts/parser_revision_20261112/nfr1_retime/records.jsonl and summary.json.
"""
from __future__ import annotations

import io
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401  (loads the GPU text reader first when it is enabled)
from parser_revision_20261112 import use_revised_parser  # noqa: E402

import os
OUT = fe.ROOT / "artifacts" / "parser_revision_20261112" / os.environ.get("NFR1_DIR", "nfr1_retime")


def main(preset: str = "balanced", primary_only: bool = False) -> None:
    """preset 'balanced' is the default app setting; 'fast' is the Fast preset defined before the evaluation.
    primary_only limits the run to the 55 primary sources (positions 41-95 of the seeded order)."""
    global OUT
    if preset != "balanced":
        OUT = OUT.parent / (f"nfr1_retime_{preset}" if OUT.name == "nfr1_retime" else f"{OUT.name}_{preset}")
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app

    use_revised_parser()
    OUT.mkdir(parents=True, exist_ok=True)
    variants = fe.load_variants()
    test = [v for v in fe.sampled_variants(variants, 95, 0) if v["project_partition"] == "test"]
    order = list(dict.fromkeys(v["source_id"] for v in test))
    if primary_only:
        primary = set(order[40:])
        test = [v for v in test if v["source_id"] in primary]
    references = fe.load_references()
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "retime.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    if client.patch("/api/settings", json=dict(PRESETS[preset]), headers={"X-CSRF-Token": token}).status_code != 200:
        raise RuntimeError("settings preset failed")

    def analyse(gray, name):
        ok, encoded = cv2.imencode(".png", gray)
        return client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), name)},
                           headers={"X-CSRF-Token": token}, content_type="multipart/form-data")

    for _ in range(2):  # warm the restoration model and the text reader; not timed
        warm = analyse(fe.read_gray(test[0]["degraded_path"]), "warmup.png").get_json(silent=True) or {}
        if warm.get("receipt"):
            from dsdnet_receipt.web_app import wait_for_deferred_display
            wait_for_deferred_display()
            client.delete(f"/api/receipts/{warm['receipt']['public_id']}", headers={"X-CSRF-Token": token})

    path = OUT / "records.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["variant_key"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    with path.open("a", encoding="utf-8") as sink:
        for index, variant in enumerate(test, start=1):
            key = f"{variant['source_id']}/{variant['variant_id']}"
            if key in done:
                continue
            degraded = fe.read_gray(variant["degraded_path"])
            t0 = time.perf_counter()
            response = analyse(degraded, f"{variant['source_id']}_{variant['variant_id']}.png")
            seconds = time.perf_counter() - t0
            body = response.get_json(silent=True) or {}
            receipt = body.get("receipt") or {}
            row = {"variant_key": key, "severity": variant["severity"], "seconds": seconds, "status": response.status_code,
                   "stage_seconds": (body.get("audit") or {}).get("stage_seconds"),
                   "values": {n: receipt.get(n) for n in fe.FIELDS}, "error": body.get("error"),
                   "correct": {n: bool(fe.field_correct(n, receipt.get(n), {"merchant": references[variant["source_id"]]["company"],
                                                                             "transaction_date": references[variant["source_id"]]["transaction_date"],
                                                                             "total_amount": references[variant["source_id"]]["total_amount"]}[n]))
                               for n in fe.FIELDS}}
            sink.write(json.dumps(row) + "\n"); sink.flush()
            if receipt.get("public_id"):
                from dsdnet_receipt.web_app import wait_for_deferred_display
                wait_for_deferred_display()  # Fast preset: let the cleaned image finish before the next receipt (not timed)
                client.delete(f"/api/receipts/{receipt['public_id']}", headers={"X-CSRF-Token": token})
            print(f"[{index}/{len(test)}] {key} {seconds:.1f}s", flush=True)

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    times = [r["seconds"] for r in rows]
    summary = {"images": len(times), "mean_seconds": statistics.fmean(times), "median_seconds": statistics.median(times),
               "p95_seconds": sorted(times)[max(0, int(round(0.95 * len(times))) - 1)], "max_seconds": max(times),
               "met_nfr1_15s": statistics.fmean(times) <= 15.0, "preset": preset, "primary_only": primary_only,
               "field_accuracy_image_mean": statistics.fmean([statistics.fmean(r["correct"].values()) for r in rows if r.get("correct")] or [0]),
               "by_severity": {s: statistics.fmean([r["seconds"] for r in rows if r["severity"] == s])
                               for s in sorted({r["severity"] for r in rows})}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "DONE.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else "balanced", "--primary" in sys.argv)
    finally:
        from dsdnet_receipt.ocr import shutdown_gpu_worker
        shutdown_gpu_worker()
