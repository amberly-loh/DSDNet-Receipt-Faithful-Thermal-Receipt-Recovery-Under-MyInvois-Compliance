"""Diagnostic (validation receipts only): dump the OCR lines of direct OCR and of
the deployed application's multi-view reading, so the text-error gap between the
proposed path and direct OCR can be traced to its cause. Not part of the scored run."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import final_evaluation_20261107 as fe  # noqa: E402

OUT = ROOT / "artifacts" / "final_evaluation_20261107" / "diagnostic_reading_copy"


def main(sources: int = 4) -> None:
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.ocr import PaddleOCRAdapter
    from dsdnet_receipt.web_app import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(ROOT / "configs" / "default.json")
    references = fe.load_references()
    variants = [v for v in fe.load_variants() if v["project_partition"] == "validation"]
    order = fe.sampled_variants(variants, 0, sources)  # first validation sources of the seeded order
    chosen = [v for v in order if v["project_partition"] == "validation"][: sources * 3]
    ocr = PaddleOCRAdapter(config)
    sandbox = OUT / "sandbox"
    sandbox.mkdir(exist_ok=True)
    app_config = dict(config)
    app_config["artifacts_root"] = str(sandbox / "artifacts")
    cfg_path = OUT / "config_snapshot.json"
    cfg_path.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(cfg_path, testing=True, database_path=sandbox / "diag.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    from dsdnet_receipt.settings import PRESETS
    assert client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token}).status_code == 200
    with (OUT / "diagnostic_lines.jsonl").open("w", encoding="utf-8") as sink:
        for v in chosen:
            key = f"{v['source_id']}/{v['variant_id']}"
            t0 = time.perf_counter()
            degraded = fe.read_gray(v["degraded_path"])
            direct = ocr.run(degraded)
            ok, encoded = cv2.imencode(".png", degraded)
            data = {"receipt": (__import__("io").BytesIO(encoded.tobytes()), f"{v['source_id']}_{v['variant_id']}.png")}
            response = client.post("/api/analyze", data=data, content_type="multipart/form-data",
                                   headers={"X-CSRF-Token": token})
            body = response.get_json() or {}
            audit = body.get("audit") or {}
            receipt_id = (body.get("receipt") or {}).get("public_id")
            row = {"variant_key": key, "severity": v["severity"], "status": response.status_code,
                   "reference_text": references[v["source_id"]]["ocr_reference_text"],
                   "direct_lines": direct.lines, "app_lines": audit.get("ocr_lines") or [],
                   "app_text": body.get("ocr_text") or "", "audit_keys": sorted(audit.keys()),
                   "seconds": time.perf_counter() - t0}
            sink.write(json.dumps(row, default=str) + "\n")
            sink.flush()
            if receipt_id:
                client.delete(f"/api/receipts/{receipt_id}", headers={"X-CSRF-Token": token})
            print(key, response.status_code, round(row["seconds"], 1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
