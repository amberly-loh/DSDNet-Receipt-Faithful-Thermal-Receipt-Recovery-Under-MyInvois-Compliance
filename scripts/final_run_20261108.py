"""Final locked evaluation of the final configuration (8 November 2026).

The plan is written to final_run_20261108/plan.json before any outcome is scored and is
not changed afterwards:
* Final configuration: the deployed application (balanced preset) with the processing
  settings adopted on validation receipts only (see speed_20261108/), and the same text
  reader configuration for every method.
* Test receipts: all 95 controlled test sources in the seeded order (seed 20261107), all
  three severities. Primary analysis for RQ1: sources 41-95 of the seeded order, which no
  design decision used. Secondary: all 95 sources.
* Methods: A direct, B Otsu, B Sauvola, C perceptual, D DSDNet without support (identity
  kernel, the information the application has) and P the full system.
* RQ2/RQ3 evidence follows the first-evaluation protocol: DSDNet with the controlled blur
  level, pixel support against Gsup, and field evidence on the high-risk fields, on all
  test receipts and on the 40 validation sources used for calibration and selection.
"""
from __future__ import annotations

import argparse, hashlib, io, json, os, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401  (loads the GPU text reader first when it is enabled)

OUT = fe.ROOT / "artifacts" / "final_run_20261108"
PLAN = {
    "written": None, "seed": fe.SAMPLE_SEED,
    "test_sources": "all 95, seeded order", "primary_rq1": "seeded positions 41-95 (55 sources)",
    "secondary_rq1": "all 95 sources",
    "rq1_rule": "AC1: field accuracy and page CER both better than the baseline: one-sided Wilcoxon on source means, Holm over the five baselines per metric, p_holm < 0.05 and 95% bootstrap CI of the benefit above 0",
    "rq2_rule": "AC2/AC3 as in Chapter 3: pixel support AUC and Trace-F1 with source-cluster 95% CIs; Brier, ECE and reliability of the composite field score against OCR-only; human-versus-rule agreement from the independent perceptual audit (Cohen's kappa). Supported when the AUC CI lower bound exceeds 0.5, composite Brier and ECE are not worse than OCR-only, and kappa is at least 0.40 (moderate agreement)",
    "rq3_rule": "AC4: proposed (support-gated) UAR not greater than Baseline E at the matched coverage points 25, 40, 50, 60 and 75 percent, 50 percent highlighted; Chapter 3/4 policies and weights unchanged",
    "nfr1_rule": "mean end-to-end processing time of the proposed system per test image <= 15 s on the development laptop",
}


def main(only: str | None) -> None:
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
    plan_path = OUT / "plan.json"
    variants = fe.load_variants()
    test_order = [v for v in fe.sampled_variants(variants, 95, 0) if v["project_partition"] == "test"]
    test_sources = list(dict.fromkeys(v["source_id"] for v in test_order))
    val_order = [v for v in fe.sampled_variants(variants, 0, 40) if v["project_partition"] == "validation"]
    val_sources = list(dict.fromkeys(v["source_id"] for v in val_order))[:40]
    if not plan_path.exists():
        plan = dict(PLAN, written=time.strftime("%Y-%m-%d %H:%M:%S"), test_source_order=test_sources,
                    validation_sources=val_sources,
                    environment={k: os.environ.get(k) for k in ("RECEIPT_TILE_BATCH", "RECEIPT_OCR_GPU", "RECEIPT_MODEL_VIEW")})
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    position = {s: i + 1 for i, s in enumerate(test_sources)}
    work = [("validation", v) for s in val_sources for v in val_order if v["source_id"] == s]
    work += [("test", v) for v in test_order if position[v["source_id"]] > 40]
    work += [("test", v) for v in test_order if position[v["source_id"]] <= 40]
    if only:
        work = [w for w in work if w[0] == only]

    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    references = fe.load_references()
    path = OUT / "records.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); done.add((row["variant_key"], row["method"]))
    ocr = PaddleOCRAdapter(config)
    engine = RestorationEngine(config)
    calibration_path = Path(config["dsdnet"].get("evidence_support_calibration", ""))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {}
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "final.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    if client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token}).status_code != 200:
        raise RuntimeError("settings preset failed")
    # warm both engines so that model loading is not counted in any method's time
    warm = fe.read_gray(work[0][1]["degraded_path"])
    engine.run(warm); ocr.run(warm)
    ok, encoded = cv2.imencode(".png", warm)
    client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), "warmup.png")},
                headers={"X-CSRF-Token": token}, content_type="multipart/form-data")

    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as sink:
        for index, (partition, variant) in enumerate(work, start=1):
            key = f"{variant['source_id']}/{variant['variant_id']}"
            methods = ["D_rq3"] if partition == "validation" else \
                ["A_direct", "B_otsu", "B_sauvola", "C_perceptual", "D_dsdnet_raw", "P_full", "D_rq3"]
            methods = [m for m in methods if (key, m) not in done]
            if not methods:
                continue
            reference = references[variant["source_id"]]
            reference_text = fe.norm_text(reference["ocr_reference_text"])
            refs = {"merchant": reference["company"], "transaction_date": reference["transaction_date"], "total_amount": reference["total_amount"]}
            degraded = fe.read_gray(variant["degraded_path"])
            base = {"variant_key": key, "source_id": variant["source_id"], "variant_id": variant["variant_id"], "partition": partition,
                    "severity": variant["severity"], "position": position.get(variant["source_id"])}

            def emit(method, text, values, seconds, extra=None):
                hyp = fe.norm_text(text)
                row = {**base, "method": method, "cer": float(cer(reference_text, hyp)), "wer": float(wer(reference_text, hyp)),
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
                        restoration = engine.run(degraded)
                        image = restoration.model_output if restoration.model_output is not None else restoration.image
                        result = ocr.run(image)
                        fields = extract_fields(result.text, result.lines)
                        emit(method, result.text, {n: (fields.get(n) or {}).get("value") for n in fe.FIELDS}, time.perf_counter() - t0)
                    elif method == "P_full":
                        ok, encoded = cv2.imencode(".png", degraded)
                        response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), f"{variant['source_id']}_{variant['variant_id']}.png")},
                                               headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
                        body = response.get_json(silent=True) or {}
                        seconds = time.perf_counter() - t0
                        if response.status_code != 200:
                            emit(method, "", {}, seconds, {"error": body.get("error") or f"HTTP {response.status_code}"})
                            continue
                        receipt = body.get("receipt") or {}
                        audit = body.get("audit") or {}
                        lines = audit.get("ocr_lines") or []
                        text = "\n".join(l["text"] for l in reading_copy_lines(lines, tidy=False))
                        emit(method, text, {n: receipt.get(n) for n in fe.FIELDS}, seconds,
                             {"views": audit.get("ocr_input_images"), "stage_seconds": audit.get("stage_seconds")})
                        client.delete(f"/api/receipts/{receipt.get('public_id')}", headers={"X-CSRF-Token": token})
                    else:  # D_rq3: RQ2/RQ3 evidence under the first-evaluation protocol
                        target = fe.read_gray(variant["support_target"]) if variant.get("support_target") and Path(variant["support_target"]).is_file() else None
                        if target is not None and target.shape != degraded.shape:
                            target = cv2.resize(target, (degraded.shape[1], degraded.shape[0]), interpolation=cv2.INTER_NEAREST)
                        foreground = fe.foreground_for(variant, degraded.shape)
                        restoration = engine.run(degraded, float(variant["parameters"]["blur_sigma"]))
                        image = restoration.model_output if restoration.model_output is not None else restoration.image
                        support = restoration.diagnostic_map
                        result = ocr.run(image)
                        fields = extract_fields(result.text, result.lines)
                        extra = {}
                        if support is not None and target is not None:
                            region = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                            rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
                            pool = np.flatnonzero(region.ravel())
                            if pool.size:
                                pick = rng.choice(pool, size=min(4096, pool.size), replace=False)
                                extra["pixel_scores"] = (support.ravel()[pick].astype(np.float64) / 255.0).round(4).tolist()
                                extra["pixel_labels"] = (target.ravel()[pick] > 0).astype(int).tolist()
                        evidence = {}
                        for name in fe.HIGH_RISK:
                            record = fields.get(name) or {}
                            value = record.get("value")
                            evidence[name] = {"value": value, "correct": bool(fe.field_correct(name, value, refs[name])),
                                              "ocr_confidence": record.get("confidence"), "rule": fe.rule_factor(name, value),
                                              **fe.field_evidence(name, record, image, support, target, foreground)}
                        extra["field_evidence"] = evidence
                        extra["support_calibration"] = {k: calibration.get(k) for k in ("scales", "decision_threshold", "checkpoint_sha256")}
                        emit(method, result.text, {n: (fields.get(n) or {}).get("value") for n in fe.FIELDS}, time.perf_counter() - t0, extra)
            except Exception as error:
                sink.write(json.dumps({**base, "method": "ERROR", "error": repr(error)}) + "\n"); sink.flush()
            print(f"[{index}/{len(work)}] {partition} {key} {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    (OUT / ("DONE.txt" if not only else f"DONE_{only}.txt")).write_text("done", encoding="utf-8")


def _finish() -> None:
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["validation", "test"], default=None)
    main(parser.parse_args().only)
    _finish()
