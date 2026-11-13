"""Reading agreement: one-time run of the improved proposed system (13 November 2026).

Improvement (src/dsdnet_receipt/reading_agreement.py): merchant, date and total are each kept at the value
that most independent readings of the rectified receipt agree on (the application's fused reading, the
rectified image read directly, its Otsu and perceptual versions, and the DSDNet restoration read alone; the
fused reading wins ties). It was designed from the first fresh real-receipt test (fresh_test_20261113,
treated from now on as development data) and is checked here before any test image is processed.

Plan, fixed before the run:
1. Validation gate: the full system on the 120 validation images (40 validation sources). Every record holds
   the single-reading values (the system before this change) and the agreed values. The change is adopted
   only if, on validation, agreed field accuracy is higher than single-reading field accuracy and agreed
   high-risk accuracy is not lower. The decision is written to gate.json before any test image is read.
2. Test: the full system on the 285 locked test images (95 sources) and the 159 replication images (53
   sources), one image at a time with nothing else running (these times are the NFR1 measurement of the
   default preset). The baselines A-D are unchanged and their parser-revision records are reused.
3. The P_full values are the agreed values if the gate adopted the change, otherwise the single-reading values.
Results are reported whatever they show. A second run is refused.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401  (loads the GPU text reader first when it is enabled)
from parser_revision_20261112 import use_revised_parser, REPLICATION_MANIFEST  # noqa: E402

OUT = fe.ROOT / "artifacts" / "agreement_20261113"
VOTED = ("merchant", "transaction_date", "total_amount")


def gate_decision(rows: list[dict]) -> dict:
    n = len(rows)
    def acc(prefix, fields):
        return sum(r[f"{prefix}{f}_correct"] for r in rows for f in fields) / (n * len(fields)) if n else 0.0
    single_f, agreed_f = acc("single_", VOTED), acc("", VOTED)
    single_h, agreed_h = acc("single_", fe.HIGH_RISK), acc("", fe.HIGH_RISK)
    return {"images": n, "single_field_accuracy": single_f, "agreed_field_accuracy": agreed_f,
            "single_high_risk_accuracy": single_h, "agreed_high_risk_accuracy": agreed_h,
            "adopted": bool(agreed_f > single_f and agreed_h >= single_h),
            "rule": "adopt iff agreed field accuracy > single-reading field accuracy and agreed high-risk accuracy >= single-reading"}


def main() -> None:
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.metrics import cer, wer
    from dsdnet_receipt.receipt_context import reading_copy_lines
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "DONE.txt").exists():
        print("This run has already been completed once; it is not run again."); return
    use_revised_parser()
    variants = fe.load_variants()
    test_order = [v for v in fe.sampled_variants(variants, 95, 0) if v["project_partition"] == "test"]
    test_sources = list(dict.fromkeys(v["source_id"] for v in test_order))
    val_order = [v for v in fe.sampled_variants(variants, 0, 40) if v["project_partition"] == "validation"]
    val_sources = list(dict.fromkeys(v["source_id"] for v in val_order))[:40]
    position = {s: i + 1 for i, s in enumerate(test_sources)}
    replication = [json.loads(l) for l in REPLICATION_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    work = [("validation", "locked", v) for s in val_sources for v in val_order if v["source_id"] == s]
    work += [("test", "locked", v) for v in test_order if position[v["source_id"]] > 40]
    work += [("test", "locked", v) for v in test_order if position[v["source_id"]] <= 40]
    work += [("test", "replication", v) for v in replication]
    plan = OUT / "plan.json"
    if not plan.exists():
        plan.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "doc": __doc__,
                                    "images": {"validation": sum(w[0] == "validation" for w in work),
                                               "locked_test": sum(w[0] == "test" and w[1] == "locked" for w in work),
                                               "replication": sum(w[1] == "replication" for w in work)}}, indent=1), encoding="utf-8")

    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    references = fe.load_references()
    path = OUT / "records.jsonl"
    done = set()
    rows_by_part: dict[str, list[dict]] = {"validation": []}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done.add((row["set"], row["variant_key"]))
                if row["partition"] == "validation" and "error" not in row:
                    rows_by_part["validation"].append(row)
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "agreement.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    if client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token}).status_code != 200:
        raise RuntimeError("settings preset failed")
    warm = fe.read_gray(work[0][2]["degraded_path"])
    ok, encoded = cv2.imencode(".png", warm)
    for _ in range(2):  # warm every engine so that model loading is not timed
        client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), "warmup.png")},
                    headers={"X-CSRF-Token": token}, content_type="multipart/form-data")

    gate_path = OUT / "gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else None
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as sink:
        for index, (partition, set_name, variant) in enumerate(work, start=1):
            if partition == "test" and gate is None:
                gate = gate_decision(rows_by_part["validation"])
                gate["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
                gate_path.write_text(json.dumps(gate, indent=1), encoding="utf-8")
                print("GATE", json.dumps(gate), flush=True)
            key = f"{variant['source_id']}/{variant['variant_id']}"
            if (set_name, key) in done:
                continue
            reference = references[variant["source_id"]]
            reference_text = fe.norm_text(reference["ocr_reference_text"])
            refs = {"merchant": reference["company"], "transaction_date": reference["transaction_date"], "total_amount": reference["total_amount"]}
            base = {"set": set_name, "variant_key": key, "source_id": variant["source_id"], "variant_id": variant["variant_id"],
                    "partition": partition, "severity": variant["severity"], "position": position.get(variant["source_id"]), "method": "P_full"}
            degraded = fe.read_gray(variant["degraded_path"])
            ok, encoded = cv2.imencode(".png", degraded)
            t0 = time.perf_counter()
            response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), f"{variant['source_id']}_{variant['variant_id']}.png")},
                                   headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
            seconds = time.perf_counter() - t0
            body = response.get_json(silent=True) or {}
            receipt = body.get("receipt") or {}
            audit = body.get("audit") or {}
            lines = audit.get("ocr_lines") or []
            text = "\n".join(l["text"] for l in reading_copy_lines(lines, tidy=False)) if response.status_code == 200 else ""
            hyp = fe.norm_text(text)
            agreed = {n: receipt.get(n) for n in fe.FIELDS} if response.status_code == 200 else {}
            single = dict(agreed)
            single.update((audit.get("reading_agreement") or {}).get("single_reading") or {})
            use = agreed if (gate is None or gate["adopted"]) else single
            row = {**base, "cer": float(cer(reference_text, hyp)), "wer": float(wer(reference_text, hyp)),
                   **fe.word_bag_scores(reference_text, hyp), "seconds": seconds, "text": hyp,
                   "stage_seconds": audit.get("stage_seconds"), "agreement": (audit.get("reading_agreement") or {}).get("fields")}
            if response.status_code != 200:
                row["error"] = body.get("error") or f"HTTP {response.status_code}"
            for name in fe.FIELDS:
                row[f"{name}_value"] = use.get(name)
                row[f"{name}_correct"] = bool(fe.field_correct(name, use.get(name), refs[name]))
                row[f"agreed_{name}_value"] = agreed.get(name)
                row[f"agreed_{name}_correct"] = bool(fe.field_correct(name, agreed.get(name), refs[name]))
                row[f"single_{name}_value"] = single.get(name)
                row[f"single_{name}_correct"] = bool(fe.field_correct(name, single.get(name), refs[name]))
            if partition == "validation":
                # the gate compares agreed against single-reading values on validation only
                for name in fe.FIELDS:
                    row[f"{name}_correct"] = row[f"agreed_{name}_correct"]
                rows_by_part["validation"].append(row)
            sink.write(json.dumps(row) + "\n"); sink.flush()
            if receipt.get("public_id"):
                client.delete(f"/api/receipts/{receipt['public_id']}", headers={"X-CSRF-Token": token})
            print(f"[{index}/{len(work)}] {set_name} {partition} {key} {seconds:.1f}s {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    (OUT / "DONE.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    finally:
        from dsdnet_receipt.ocr import shutdown_gpu_worker
        shutdown_gpu_worker()
