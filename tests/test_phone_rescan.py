"""A phone rescan replaces the bad scan it was asked to redo, and nothing else."""
import io
import json

import cv2
import numpy as np

from dsdnet_receipt import ocr as ocr_module
from dsdnet_receipt import restoration as restoration_module
from dsdnet_receipt import web_app


def _fake_ocr(self, image, *args, **kwargs):
    lines = [{"text": "KOK FONG TOOLS SDN BHD", "confidence": 0.9, "box": [[10, 10], [300, 10], [300, 30], [10, 30]]},
             {"text": "DATE 12/11/2024", "confidence": 0.9, "box": [[10, 50], [300, 50], [300, 70], [10, 70]]},
             {"text": "TOTAL 12.90", "confidence": 0.9, "box": [[10, 90], [300, 90], [300, 110], [10, 110]]}]
    return ocr_module.OCRResult("\n".join(line["text"] for line in lines), lines, "fake", "ok", None)


def _fake_restoration(self, image, *args, **kwargs):
    extra = {name: None for name in restoration_module.RestorationResult.__dataclass_fields__ if name not in ("image", "model_available", "status")}
    return restoration_module.RestorationResult(image=image, model_available=True, status="ok", **extra)


def _photo():
    image = np.full((400, 300), 255, np.uint8); cv2.putText(image, "TOTAL 12.90", (10, 200), 0, 1, 0, 2)
    return cv2.imencode(".png", image)[1].tobytes()


def test_rescan_replaces_only_the_scan_this_phone_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_module.PaddleOCRAdapter, "run", _fake_ocr)
    monkeypatch.setattr(restoration_module.RestorationEngine, "run", _fake_restoration)
    monkeypatch.setattr(restoration_module, "assess_receipt_degradation", lambda image: {"flags": []})
    monkeypatch.setattr(web_app, "lan_addresses", lambda: ["192.168.1.20"])
    from dsdnet_receipt.config import load_config
    config, _ = load_config("configs/default.json")
    config = dict(config); config["artifacts_root"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "config.json"; config_path.write_text(json.dumps(config))
    app = web_app.create_app(config_path, testing=True, database_path=tmp_path / "db.sqlite3")
    desk = app.test_client()
    token = desk.get("/api/auth").get_json()["csrf_token"]
    token = desk.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    # a receipt uploaded on the computer itself
    own = desk.post("/api/analyze", data={"receipt": (io.BytesIO(_photo()), "desk.png")}, headers={"X-CSRF-Token": token},
                    content_type="multipart/form-data").get_json()["receipt"]["public_id"]
    code = desk.post("/api/phone/pair", headers={"X-CSRF-Token": token}).get_json()["url"].split("#", 1)[1]
    phone = app.test_client(); phone.environ_base["REMOTE_ADDR"] = "192.168.1.31"

    def send(replaces=""):
        data = {"receipt": (io.BytesIO(_photo()), "phone.png")}
        if replaces:
            data["replaces"] = replaces
        response = phone.post("/api/phone/upload", data=data, headers={"X-Phone-Token": code}, content_type="multipart/form-data")
        assert response.status_code == 200
        return response.get_json()

    first = send()["receipt"]["public_id"]
    second = send(replaces=first)
    assert second["replaced"] == first
    assert desk.get(f"/api/receipts/{first}").status_code == 404           # the bad scan is gone
    assert desk.get(f"/api/receipts/{second['receipt']['public_id']}").status_code == 200
    # a phone cannot remove a receipt it did not send
    third = send(replaces=own)
    assert third["replaced"] is None
    assert desk.get(f"/api/receipts/{own}").status_code == 200
    web_app.wait_for_deferred_display(timeout=60)
