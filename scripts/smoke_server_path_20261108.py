"""Smoke test of the deployed server path (isolated worker process + GPU text reader) on two validation images."""
from __future__ import annotations
import io, json, sys, tempfile, time
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402


def main():
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.web_app import create_app
    from dsdnet_receipt.settings import PRESETS
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    tmp = Path(tempfile.mkdtemp())
    cfg = dict(config); cfg["artifacts_root"] = str(tmp / "artifacts")
    (tmp / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    app = create_app(tmp / "cfg.json", testing=False, database_path=tmp / "smoke.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    client.patch("/api/settings", json=dict(PRESETS["balanced"]), headers={"X-CSRF-Token": token})
    val = [v for v in fe.sampled_variants(fe.load_variants(), 0, 40) if v["project_partition"] == "validation"][:3]
    out = []
    for v in val:
        ok, enc = cv2.imencode(".png", fe.read_gray(v["degraded_path"]))
        t = time.perf_counter()
        r = client.post("/api/analyze", data={"receipt": (io.BytesIO(enc.tobytes()), "smoke.png")}, headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
        body = r.get_json(silent=True) or {}
        out.append({"key": v["source_id"] + "/" + v["variant_id"], "status": r.status_code, "seconds": time.perf_counter() - t,
                    "lines": len((body.get("audit") or {}).get("ocr_lines") or []), "error": body.get("error"),
                    "total": (body.get("receipt") or {}).get("total_amount")})
        print(out[-1], flush=True)
    (fe.ROOT / "artifacts" / "smoke_server_path_20261108.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    worker = app.extensions.get("isolated_receipt_worker")
    if worker:
        worker.close()


if __name__ == "__main__":
    main()
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()
