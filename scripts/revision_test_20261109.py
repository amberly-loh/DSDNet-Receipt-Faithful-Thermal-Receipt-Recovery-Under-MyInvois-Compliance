"""RQ1 revision: one-time test run (9 November 2026).

Disclosed post-evaluation revision. After the locked and replication results were known, an error
analysis showed the full system losing to Baseline D on business names and on page CER for severely
degraded receipts. Two changes were then developed on the 40 validation sources only
(artifacts/dev_validation_20261108) and frozen before this script was written:
  1. when two views read the same letters at the same place, keep the reading with more word gaps,
     then the more confident one (ocr.py, _reading_preference);
  2. the reading copy keeps every line the reader accepted (cut-off 0.50, the reader's own floor,
     instead of 0.55) (receipt_context.py).
No test receipt was read while the changes were developed. This script runs the revised full system
(P_full, balanced preset) once on every test receipt: the locked set (95 sources; primary 41-95) and
the replication set (53 sources). Baselines A-D do not use either changed function and are reused from
the locked and replication runs. AC1 and NFR1 decision rules are unchanged; results are reported
whatever they show, beside the original locked results. The Fast preset is then timed again on the
primary sources for NFR1.
"""
from __future__ import annotations

import io, json, sys, time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401

OUT = fe.ROOT / "artifacts" / "revision_test_20261109"
REPLICATION = fe.ROOT / "artifacts" / "replication_20261108" / "replication_manifest.jsonl"


def main() -> None:
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.metrics import cer, wer
    from dsdnet_receipt.receipt_context import reading_copy_lines
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    locked = [v for v in fe.sampled_variants(fe.load_variants(), 95, 0) if v["project_partition"] == "test"]
    order = list(dict.fromkeys(v["source_id"] for v in locked))
    position = {s: i + 1 for i, s in enumerate(order)}
    replication = [json.loads(l) for l in REPLICATION.read_text(encoding="utf-8").splitlines() if l.strip()]
    work = [("locked", v) for v in locked if position[v["source_id"]] > 40]
    work += [("replication", v) for v in replication]
    work += [("locked", v) for v in locked if position[v["source_id"]] <= 40]
    if any(v["project_partition"] != "test" for _, v in work):
        raise RuntimeError("non-test receipt in the test run")
    plan = OUT / "plan.json"
    if not plan.exists():
        plan.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "doc": __doc__,
                                    "images": len(work), "locked_sources": len(order),
                                    "replication_sources": len({v["source_id"] for v in replication})}, indent=1), encoding="utf-8")

    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    references = fe.load_references()
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "revision.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    if client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token}).status_code != 200:
        raise RuntimeError("settings preset failed")
    ok, encoded = cv2.imencode(".png", fe.read_gray(work[0][1]["degraded_path"]))
    client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), "warmup.png")},
                headers={"X-CSRF-Token": token}, content_type="multipart/form-data")

    path = OUT / "records.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["variant_key"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as sink:
        for index, (subset, variant) in enumerate(work, start=1):
            key = f"{variant['source_id']}/{variant['variant_id']}"
            if key in done:
                continue
            reference = references[variant["source_id"]]
            reference_text = fe.norm_text(reference["ocr_reference_text"])
            refs = {"merchant": reference["company"], "transaction_date": reference["transaction_date"], "total_amount": reference["total_amount"]}
            row = {"variant_key": key, "source_id": variant["source_id"], "variant_id": variant["variant_id"], "partition": "test",
                   "subset": subset, "severity": variant["severity"], "position": position.get(variant["source_id"]), "method": "P_revised"}
            try:
                ok, encoded = cv2.imencode(".png", fe.read_gray(variant["degraded_path"]))
                t0 = time.perf_counter()
                response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), key.replace("/", "_") + ".png")},
                                       headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
                seconds = time.perf_counter() - t0
                body = response.get_json(silent=True) or {}
                receipt = body.get("receipt") or {}
                audit = body.get("audit") or {}
                text = "\n".join(l["text"] for l in reading_copy_lines(audit.get("ocr_lines") or [], tidy=False)) if response.status_code == 200 else ""
                if response.status_code != 200:
                    row["error"] = body.get("error") or f"HTTP {response.status_code}"
                hyp = fe.norm_text(text)
                row.update({"cer": float(cer(reference_text, hyp)), "wer": float(wer(reference_text, hyp)),
                            **fe.word_bag_scores(reference_text, hyp), "seconds": seconds, "text": hyp})
                for name in fe.FIELDS:
                    row[f"{name}_value"] = receipt.get(name)
                    row[f"{name}_correct"] = bool(fe.field_correct(name, receipt.get(name), refs[name]))
                if receipt.get("public_id"):
                    client.delete(f"/api/receipts/{receipt.get('public_id')}", headers={"X-CSRF-Token": token})
            except Exception as error:
                row["error"] = repr(error)
            sink.write(json.dumps(row) + "\n"); sink.flush()
            print(f"[{index}/{len(work)}] {subset} {key} {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    (OUT / "DONE_P.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main()
    import fast_preset_20261108 as fp
    fp.OUT = OUT
    fp.main("fast_revision")
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()
