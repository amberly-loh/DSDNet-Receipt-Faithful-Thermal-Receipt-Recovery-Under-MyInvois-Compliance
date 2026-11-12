"""Supplementary measurement (8 Nov 2026): the application's existing Fast preset on the primary test
receipts (seeded positions 41-95). The preset predates the evaluation; it is measured, not tuned, and its
accuracy is reported beside the default (balanced) preset so the speed/accuracy trade-off is visible."""
from __future__ import annotations
import io, json, sys, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401

OUT = fe.ROOT / "artifacts" / "final_run_20261108"


def main(label: str = "fast_preset"):
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.metrics import cer, wer
    from dsdnet_receipt.receipt_context import reading_copy_lines
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    refs_all = fe.load_references()
    test = [v for v in fe.sampled_variants(fe.load_variants(), 95, 0) if v["project_partition"] == "test"]
    order = list(dict.fromkeys(v["source_id"] for v in test)); pos = {s: i + 1 for i, s in enumerate(order)}
    work = [v for v in test if pos[v["source_id"]] > 40]
    sandbox = OUT / f"sandbox_{label}"; sandbox.mkdir(exist_ok=True)
    cfg = dict(config); cfg["artifacts_root"] = str(sandbox / "artifacts")
    (sandbox / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    app = create_app(sandbox / "cfg.json", testing=True, database_path=sandbox / "fast.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    assert client.patch("/api/settings", json=dict(PRESETS["fast"]), headers={"X-CSRF-Token": token}).status_code == 200
    ok, enc = cv2.imencode(".png", fe.read_gray(work[0]["degraded_path"]))
    client.post("/api/analyze", data={"receipt": (io.BytesIO(enc.tobytes()), "warm.png")}, headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
    path = OUT / f"{label}.jsonl"; done = set()
    if path.exists():
        done = {json.loads(l)["variant_key"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    with path.open("a", encoding="utf-8") as sink:
        for v in work:
            key = f"{v['source_id']}/{v['variant_id']}"
            if key in done:
                continue
            ref = refs_all[v["source_id"]]; rt = fe.norm_text(ref["ocr_reference_text"])
            refs = {"merchant": ref["company"], "transaction_date": ref["transaction_date"], "total_amount": ref["total_amount"]}
            ok, enc = cv2.imencode(".png", fe.read_gray(v["degraded_path"]))
            t = time.perf_counter()
            r = client.post("/api/analyze", data={"receipt": (io.BytesIO(enc.tobytes()), key.replace("/", "_") + ".png")}, headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
            secs = time.perf_counter() - t; body = r.get_json(silent=True) or {}
            row = {"variant_key": key, "source_id": v["source_id"], "severity": v["severity"], "position": pos[v["source_id"]], "method": "P_fast", "seconds": secs, "status": r.status_code}
            if r.status_code == 200:
                rec = body.get("receipt") or {}; aud = body.get("audit") or {}
                hyp = fe.norm_text("\n".join(l["text"] for l in reading_copy_lines(aud.get("ocr_lines") or [], tidy=False)))
                row.update({"cer": float(cer(rt, hyp)), "wer": float(wer(rt, hyp)), **fe.word_bag_scores(rt, hyp), "stage_seconds": aud.get("stage_seconds")})
                for n in fe.FIELDS:
                    row[f"{n}_value"] = rec.get(n); row[f"{n}_correct"] = bool(fe.field_correct(n, rec.get(n), refs[n]))
                client.delete(f"/api/receipts/{rec.get('public_id')}", headers={"X-CSRF-Token": token})
            else:
                row["error"] = body.get("error")
                hyp = ""
                row.update({"cer": float(cer(rt, hyp)), "wer": float(wer(rt, hyp)), **fe.word_bag_scores(rt, hyp)})
                for n in fe.FIELDS:
                    row[f"{n}_value"] = None; row[f"{n}_correct"] = False
            sink.write(json.dumps(row) + "\n"); sink.flush()
            print(key, f"{secs:.1f}s", flush=True)
    (OUT / f"DONE_{label}.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fast_preset")
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()
