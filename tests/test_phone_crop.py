from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import dsdnet_receipt.web_app as web_app
from dsdnet_receipt.preprocessing import suggest_outline
from dsdnet_receipt.web_app import (apply_exif_orientation, apply_manual_geometry, create_app, decode_upright,
                                    parse_manual_geometry)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"
ROOT = Path(__file__).resolve().parents[1]


def _desk_photo():
    """A tilted receipt photographed on a dark desk, plus its true corners."""
    h, w = 1600, 1200
    rng = np.random.RandomState(1)
    desk = cv2.add(np.full((h, w, 3), (40, 70, 110), np.uint8), rng.randint(0, 25, (h, w, 3), np.uint8))
    receipt = np.full((900, 420, 3), 245, np.uint8)
    for i, y in enumerate(range(60, 860, 38)):
        cv2.putText(receipt, f"ITEM {i:02d}   RM {i * 1.5:5.2f}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    true = np.float32([[380, 300], [790, 250], [880, 1150], [430, 1240]])
    transform = cv2.getPerspectiveTransform(np.float32([[0, 0], [419, 0], [419, 899], [0, 899]]), true)
    warped = cv2.warpPerspective(receipt, transform, (w, h))
    mask = cv2.warpPerspective(np.full((900, 420), 255, np.uint8), transform, (w, h))
    desk[mask > 0] = warped[mask > 0]
    return desk, true


def test_every_camera_orientation_tag_matches_pillow() -> None:
    base = np.random.RandomState(0).randint(0, 255, (30, 50, 3), np.uint8)
    for tag in range(1, 9):
        picture = Image.fromarray(base); exif = picture.getexif(); exif[0x0112] = tag
        buffer = io.BytesIO(); picture.save(buffer, "PNG", exif=exif.tobytes())
        expected = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(buffer.getvalue()))))
        assert np.array_equal(apply_exif_orientation(base, tag), expected), tag


def test_sideways_phone_photo_is_decoded_upright_once() -> None:
    picture = Image.new("RGB", (400, 200), "white"); exif = picture.getexif(); exif[0x0112] = 6
    buffer = io.BytesIO(); picture.save(buffer, "JPEG", exif=exif.tobytes())
    assert decode_upright(buffer.getvalue()).shape[:2] == (400, 200)


def test_outline_is_suggested_for_a_receipt_on_a_desk_and_sideways_needs_a_turn() -> None:
    photo, true = _desk_photo()
    found = suggest_outline(photo)
    assert found["found"] and found["suggested_rotation"] == 0
    corners = np.asarray(found["corners"]) * [photo.shape[1] - 1, photo.shape[0] - 1]
    assert np.abs(corners - true).max() < 40
    sideways = suggest_outline(cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE))
    assert sideways["found"] and sideways["suggested_rotation"] == 90
    assert suggest_outline(np.full((300, 200, 3), 255, np.uint8))["found"] is False


def test_confirmed_rotation_and_crop_are_validated_and_applied() -> None:
    photo, true = _desk_photo()
    corners = (true / [photo.shape[1] - 1, photo.shape[0] - 1]).tolist()
    geometry = parse_manual_geometry({"rotation": "0", "corners": json.dumps(corners)})
    cut = apply_manual_geometry(photo, geometry)
    assert 0.9 < cut.shape[0] / 900 < 1.2 and 0.9 < cut.shape[1] / 420 < 1.2
    assert float(cut.mean()) > 200  # mostly paper, no desk
    assert apply_manual_geometry(photo, parse_manual_geometry({"rotation": "90", "corners": None})).shape[:2] == (1200, 1600)
    whole = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert apply_manual_geometry(photo, parse_manual_geometry({"rotation": "180", "corners": json.dumps(whole)})).shape == photo.shape
    assert parse_manual_geometry({}) is None
    for bad in ({"rotation": "45"}, {"rotation": "x"}, {"rotation": "0", "corners": "[[0,0],[1,0]]"},
                {"rotation": "0", "corners": "[[0,0],[1.5,0],[1,1],[0,1]]"},
                {"rotation": "0", "corners": "[[0,0],[0.01,0],[0.01,0.01],[0,0.01]]"}):
        try:
            parse_manual_geometry(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_phone_detect_needs_the_pairing_code_and_bad_crops_are_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_app, "lan_addresses", lambda: ["192.168.1.20"])
    app = create_app(CONFIG, testing=True, database_path=tmp_path / "db.sqlite3")
    desk = app.test_client()
    token = desk.get("/api/auth").get_json()["csrf_token"]
    token = desk.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]
    code = desk.post("/api/phone/pair", headers={"X-CSRF-Token": token}).get_json()["url"].split("#", 1)[1]
    phone = app.test_client(); phone.environ_base["REMOTE_ADDR"] = "192.168.1.31"
    photo, _ = _desk_photo()
    preview = cv2.imencode(".jpg", cv2.resize(photo, (900, 1200)))[1].tobytes()
    def send(path, headers, data):
        return phone.post(path, data=data, headers=headers, content_type="multipart/form-data")
    assert send("/api/phone/detect", {"X-Phone-Token": "wrong"}, {"preview": (io.BytesIO(preview), "p.jpg")}).status_code == 401
    answer = send("/api/phone/detect", {"X-Phone-Token": code}, {"preview": (io.BytesIO(preview), "p.jpg")})
    assert answer.status_code == 200 and answer.get_json()["found"]
    assert send("/api/phone/detect", {"X-Phone-Token": code}, {"preview": (io.BytesIO(b"nope"), "p.jpg")}).status_code == 400
    refused = send("/api/phone/upload", {"X-Phone-Token": code},
                   {"receipt": (io.BytesIO(preview), "r.jpg"), "rotation": "45", "corners": "[]"})
    assert refused.status_code == 400 and "quarter turn" in refused.get_json()["error"]


def test_phone_page_has_the_check_step() -> None:
    html = (ROOT / "web" / "templates" / "phone.html").read_text(encoding="utf-8")
    for element in ("cropCanvas", "cropRotateLeft", "cropRotateRight", "cropAuto", "cropWhole", "cropRetake", "cropSend"):
        assert f'id="{element}"' in html
    script = (ROOT / "web" / "static" / "phone.js").read_text(encoding="utf-8")
    assert "/api/phone/detect" in script and "form.append('rotation'" in script and "form.append('corners'" in script


def test_an_untouched_turn_is_only_a_suggestion() -> None:
    corners = json.dumps([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert parse_manual_geometry({"rotation": "90", "corners": corners})["rotation_confirmed"] is False
    assert parse_manual_geometry({"rotation": "90", "rotation_confirmed": "1"})["rotation_confirmed"] is True
    script = (ROOT / "web" / "static" / "phone.js").read_text(encoding="utf-8")
    assert "form.append('rotation_confirmed'" in script and "if (entry.corners)" in script
    assert "could not be read. Retake it flat and upright" in script


def test_scan_page_prefers_the_phone_over_a_computer_webcam() -> None:
    html = (ROOT / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="cameraCard" class="camera-card touch-only"' in html and "webcam-link pointer-only" in html
    css = (ROOT / "web" / "static" / "styles.css").read_text(encoding="utf-8")
    assert "@media (hover:hover) and (pointer:fine){.touch-only{display:none!important}}" in css
