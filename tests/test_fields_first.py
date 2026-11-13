"""Fast preset: fields are returned first and the cleaned image is finished afterwards."""
import io
import json

import cv2
import numpy as np

from dsdnet_receipt import ocr as ocr_module
from dsdnet_receipt import restoration as restoration_module
from dsdnet_receipt import web_app
from dsdnet_receipt.settings import PRESETS


def _fake_ocr(self, image, *args, **kwargs):
    lines = [{"text": "KOK FONG TOOLS SDN BHD", "confidence": 0.9, "box": [[10, 10], [300, 10], [300, 30], [10, 30]]},
             {"text": "DATE 12/11/2024", "confidence": 0.9, "box": [[10, 50], [300, 50], [300, 70], [10, 70]]},
             {"text": "TOTAL 12.90", "confidence": 0.9, "box": [[10, 90], [300, 90], [300, 110], [10, 110]]}]
    return ocr_module.OCRResult("\n".join(line["text"] for line in lines), lines, "fake", "ok", None)


def _fake_restoration(self, image, *args, **kwargs):
    extra = {name: None for name in restoration_module.RestorationResult.__dataclass_fields__ if name not in ("image", "model_available", "status")}
    return restoration_module.RestorationResult(image=image, model_available=True, status="ok", **extra)


def test_fast_preset_returns_fields_then_finishes_the_cleaned_image(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_module.PaddleOCRAdapter, "run", _fake_ocr)
    monkeypatch.setattr(restoration_module.RestorationEngine, "run", _fake_restoration)
    monkeypatch.setattr(restoration_module, "assess_receipt_degradation", lambda image: {"flags": []})
    from dsdnet_receipt.config import load_config
    config, _ = load_config("configs/default.json")
    config = dict(config); config["artifacts_root"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    app = web_app.create_app(config_path, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    assert client.patch("/api/settings", json=dict(PRESETS["fast"]), headers={"X-CSRF-Token": token}).status_code == 200
    image = np.full((400, 300), 255, np.uint8); cv2.putText(image, "TOTAL 12.90", (10, 200), 0, 1, 0, 2)
    ok, encoded = cv2.imencode(".png", image)
    response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), "a.png")},
                           headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
    assert response.status_code == 200
    body = response.get_json()
    assert body["receipt"]["total_amount"] == "12.90"
    assert body["audit"]["processing_settings"]["show_fields_first"] is True
    assert "faithful_view_and_cleanup" not in body["audit"]["stage_seconds"]
    assert web_app.wait_for_deferred_display(timeout=60)
    detail = client.get(f"/api/receipts/{body['receipt']['public_id']}").get_json()
    assert detail["audit"]["deferred_display"]["status"] == "done"
    assert detail["audit"]["stored_file_sha256"]["faithful_content"]
    assert client.get(f"/api/receipts/{body['receipt']['public_id']}/image/faithful").status_code == 200


def test_every_preset_shows_fields_first():
    assert all(PRESETS[name]["show_fields_first"] is True for name in ("fast", "balanced", "thorough"))


def test_default_preset_fields_are_the_same_with_and_without_fields_first(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_module.PaddleOCRAdapter, "run", _fake_ocr)
    monkeypatch.setattr(restoration_module.RestorationEngine, "run", _fake_restoration)
    monkeypatch.setattr(restoration_module, "assess_receipt_degradation", lambda image: {"flags": []})
    from dsdnet_receipt.config import load_config
    config, _ = load_config("configs/default.json")
    config = dict(config); config["artifacts_root"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    app = web_app.create_app(config_path, testing=True, database_path=tmp_path / "db.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    image = np.full((400, 300), 255, np.uint8); cv2.putText(image, "TOTAL 12.90", (10, 200), 0, 1, 0, 2)
    ok, encoded = cv2.imencode(".png", image)
    results = []
    for first in (True, False):
        assert client.patch("/api/settings", json=dict(PRESETS["balanced"], show_fields_first=first), headers={"X-CSRF-Token": token}).status_code == 200
        body = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), "a.png")},
                           headers={"X-CSRF-Token": token}, content_type="multipart/form-data").get_json()
        assert web_app.wait_for_deferred_display(timeout=60)
        results.append({k: body["receipt"][k] for k in ("merchant", "transaction_date", "total_amount", "receipt_number")})
    assert results[0] == results[1]
