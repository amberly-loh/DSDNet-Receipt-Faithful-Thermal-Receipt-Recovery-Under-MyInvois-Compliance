"""Confirmatory evaluation of the improved application (7 November 2026, evening).

Design, fixed before any of its outcomes were seen:
* Test: the next 30 source receipts of the seeded test order (positions 41-70), none of
  which was scored in the first evaluation; all three severities of each.
* Validation check: 5 validation sources, where the improved application is compared with
  the previous application (dsdnet_restoration reading view switched off) to show that the
  change does not harm restoration-based extraction.
* Every method receives the same information: Baseline D now runs DSDNet without the true
  blur level of the synthetic degradation, exactly as the application must on a real receipt.
* Text (page CER, WER, word F1) and fields are scored in one pass on the same images.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402

OUT = fe.OUT / "confirmatory"


def main(test_from: int, test_count: int, validation_count: int) -> None:
    from dsdnet_receipt.baselines import BASELINES
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.fields import extract_fields
    from dsdnet_receipt.metrics import cer, wer
    from dsdnet_receipt.ocr import PaddleOCRAdapter
    from dsdnet_receipt.restoration import RestorationEngine
    from dsdnet_receipt.receipt_context import reading_copy_lines
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    references = fe.load_references()
    variants = fe.load_variants()
    test_order = [v for v in fe.sampled_variants(variants, 95, 0) if v["project_partition"] == "test"]
    test_sources = list(dict.fromkeys(v["source_id"] for v in test_order))[test_from:test_from + test_count]
    first_run = set(list(dict.fromkeys(v["source_id"] for v in test_order))[:40])
    assert not first_run & set(test_sources), "confirmatory sources must be unseen"
    val_order = [v for v in fe.sampled_variants(variants, 0, 40) if v["project_partition"] == "validation"]
    val_sources = list(dict.fromkeys(v["source_id"] for v in val_order))[:validation_count]
    plan = [("validation", v) for s in val_sources for v in val_order if v["source_id"] == s] + \
           [("test", v) for s in test_sources for v in test_order if v["source_id"] == s]
    (OUT / "plan.json").write_text(json.dumps({"test_sources": test_sources, "validation_sources": val_sources,
                                               "unseen_check": "no overlap with the first 40 test sources"}, indent=1), encoding="utf-8")
    path = OUT / "records.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done.add((row["variant_key"], row["method"]))
    ocr = PaddleOCRAdapter(config)
    engine = RestorationEngine(config)
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "confirmatory.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    if client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token}).status_code != 200:
        raise RuntimeError("settings preset failed")
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as sink:
        for index, (partition, variant) in enumerate(plan, start=1):
            key = f"{variant['source_id']}/{variant['variant_id']}"
            reference = references[variant["source_id"]]
            reference_text = fe.norm_text(reference["ocr_reference_text"])
            refs = {"merchant": reference["company"], "transaction_date": reference["transaction_date"], "total_amount": reference["total_amount"]}
            methods = ["A_direct", "D_dsdnet_raw", "P_full", "P_previous"] if partition == "validation" else \
                      ["A_direct", "B_otsu", "B_sauvola", "C_perceptual", "D_dsdnet_raw", "P_full"]
            methods = [m for m in methods if (key, m) not in done]
            if not methods:
                continue
            degraded = fe.read_gray(variant["degraded_path"])

            def emit(method, text, values, seconds, extra=None):
                hyp = fe.norm_text(text)
                row = {"variant_key": key, "source_id": variant["source_id"], "variant_id": variant["variant_id"], "partition": partition,
                       "severity": variant["severity"], "method": method, "cer": float(cer(reference_text, hyp)), "wer": float(wer(reference_text, hyp)),
                       **fe.word_bag_scores(reference_text, hyp), "seconds": seconds, "text": hyp, **(extra or {})}
                for name in fe.FIELDS:
                    row[f"{name}_value"] = values.get(name)
                    row[f"{name}_correct"] = bool(fe.field_correct(name, values.get(name), refs[name]))
                sink.write(json.dumps(row) + "\n"); sink.flush()

            try:
                for method in methods:
                    t0 = time.perf_counter()
                    if method in BASELINES:
                        result = ocr.run(BASELINES[method](degraded))
                        fields = extract_fields(result.text, result.lines)
                        emit(method, result.text, {n: (fields.get(n) or {}).get("value") for n in fe.FIELDS}, time.perf_counter() - t0)
                    elif method == "D_dsdnet_raw":
                        restoration = engine.run(degraded)  # same information as the application: no true blur level
                        image = restoration.model_output if restoration.model_output is not None else restoration.image
                        result = ocr.run(image)
                        fields = extract_fields(result.text, result.lines)
                        emit(method, result.text, {n: (fields.get(n) or {}).get("value") for n in fe.FIELDS}, time.perf_counter() - t0)
                    else:
                        os.environ["RECEIPT_MODEL_VIEW"] = "0" if method == "P_previous" else "1"
                        ok, encoded = cv2.imencode(".png", degraded)
                        response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), f"{variant['source_id']}_{variant['variant_id']}_{method}.png")},
                                               headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
                        body = response.get_json(silent=True) or {}
                        if response.status_code != 200:
                            emit(method, "", {}, time.perf_counter() - t0, {"error": body.get("error") or f"HTTP {response.status_code}"})
                            continue
                        receipt = body.get("receipt") or {}
                        lines = (body.get("audit") or {}).get("ocr_lines") or []
                        text = "\n".join(l["text"] for l in reading_copy_lines(lines, tidy=False))
                        emit(method, text, {n: receipt.get(n) for n in fe.FIELDS}, time.perf_counter() - t0,
                             {"views": (body.get("audit") or {}).get("ocr_input_images")})
                        client.delete(f"/api/receipts/{receipt.get('public_id')}", headers={"X-CSRF-Token": token})
            except Exception as error:
                sink.write(json.dumps({"variant_key": key, "method": "ERROR", "error": repr(error)}) + "\n"); sink.flush()
            finally:
                os.environ["RECEIPT_MODEL_VIEW"] = "1"
            print(f"[{index}/{len(plan)}] {partition} {key} done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    (OUT / "DONE.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-from", type=int, default=40)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--validation-count", type=int, default=5)
    a = parser.parse_args()
    main(a.test_from, a.test_count, a.validation_count)
