"""Development data for the post-evaluation RQ1 revision (8 November 2026).

Validation receipts only (the 40 validation sources used for calibration, all three
severities). For every image this records, without changing the application:
* D: DSDNet without support, read once (exactly as Baseline D in the final run);
* P: the full application (balanced preset) with every text-reader call it makes, view by
  view, so that alternative merging and field rules can be developed offline.
No test receipt is read by this script.
"""
from __future__ import annotations

import io, json, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401

OUT = fe.ROOT / "artifacts" / "dev_validation_20261108"
CAPTURE: list = []


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def main() -> None:
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.ocr import PaddleOCRAdapter
    from dsdnet_receipt.restoration import RestorationEngine
    from dsdnet_receipt.receipt_context import reading_copy_lines
    from dsdnet_receipt.settings import PRESETS
    from dsdnet_receipt.web_app import create_app

    original = PaddleOCRAdapter.run_views

    def recording(self, views):
        import hashlib
        names = {hashlib.sha256(image.tobytes()).hexdigest(): name for name, image in views}
        per_view = []
        plain_run = self.run

        def run_and_record(image):
            result = plain_run(image)
            per_view.append({"view": names.get(hashlib.sha256(image.tobytes()).hexdigest(), "?"),
                             "shape": list(image.shape), "lines": result.lines})
            return result

        self.run = run_and_record
        try:
            return original(self, views)
        finally:
            del self.run
            CAPTURE.append(per_view)

    PaddleOCRAdapter.run_views = recording

    OUT.mkdir(parents=True, exist_ok=True)
    variants = fe.load_variants()
    val_order = [v for v in fe.sampled_variants(variants, 0, 40) if v["project_partition"] == "validation"]
    val_sources = list(dict.fromkeys(v["source_id"] for v in val_order))[:40]
    work = [v for s in val_sources for v in val_order if v["source_id"] == s]
    if any(v["project_partition"] != "validation" for v in work):
        raise RuntimeError("non-validation receipt in the development set")

    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    references = fe.load_references()
    ocr = PaddleOCRAdapter(config)
    engine = RestorationEngine(config)
    sandbox = OUT / "sandbox"; sandbox.mkdir(exist_ok=True)
    app_config = dict(config); app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg = OUT / "config_snapshot.json"; cfg.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg, testing=True, database_path=sandbox / "dev.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    if client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token}).status_code != 200:
        raise RuntimeError("settings preset failed")

    path = OUT / "records.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["variant_key"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as sink:
        for index, variant in enumerate(work, start=1):
            key = f"{variant['source_id']}/{variant['variant_id']}"
            if key in done:
                continue
            reference = references[variant["source_id"]]
            row = {"variant_key": key, "source_id": variant["source_id"], "severity": variant["severity"],
                   "partition": variant["project_partition"],
                   "refs": {"merchant": reference["company"], "transaction_date": reference["transaction_date"],
                            "total_amount": reference["total_amount"]},
                   "reference_text": fe.norm_text(reference["ocr_reference_text"])}
            try:
                degraded = fe.read_gray(variant["degraded_path"])
                restoration = engine.run(degraded)
                image = restoration.model_output if restoration.model_output is not None else restoration.image
                d = ocr.run(image)
                row["D"] = {"text": d.text, "lines": d.lines}
                CAPTURE.clear()
                ok, encoded = cv2.imencode(".png", degraded)
                response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), f"{key.replace('/', '_')}.png")},
                                       headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
                body = response.get_json(silent=True) or {}
                receipt = body.get("receipt") or {}
                audit = body.get("audit") or {}
                lines = audit.get("ocr_lines") or []
                row["P"] = {"status": response.status_code, "fields": {n: receipt.get(n) for n in fe.FIELDS},
                            "text": "\n".join(l["text"] for l in reading_copy_lines(lines, tidy=False)),
                            "ocr_lines": lines, "calls": list(CAPTURE)}
                if receipt.get("public_id"):
                    client.delete(f"/api/receipts/{receipt.get('public_id')}", headers={"X-CSRF-Token": token})
            except Exception as error:
                row["error"] = repr(error)
            sink.write(json.dumps(row, default=_plain) + "\n"); sink.flush()
            print(f"[{index}/{len(work)}] {key} {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    (OUT / "DONE.txt").write_text("done", encoding="utf-8")


if __name__ == "__main__":
    main()
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()
