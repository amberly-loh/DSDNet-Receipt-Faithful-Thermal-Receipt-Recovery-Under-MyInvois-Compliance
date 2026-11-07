import cv2
import numpy as np

from dsdnet_receipt.preprocessing import _crop_photographed_paper


def _receipt(width=500, height=1100):
    paper = np.full((height, width, 3), (205, 222, 228), np.uint8)  # cream thermal paper
    for y in range(60, height - 40, 34):
        cv2.putText(paper, "ITEM 12.50  TOTAL", (30, y), cv2.FONT_HERSHEY_SIMPLEX, .9, (60, 60, 60), 2)
    return paper


def _photo(background, angle):
    paper = _receipt()
    canvas = np.zeros((1600, 1200, 3), np.uint8); canvas[:] = background
    canvas = np.clip(canvas.astype(int) + np.random.default_rng(3).normal(0, 10, canvas.shape), 0, 255).astype(np.uint8)
    matrix = cv2.getRotationMatrix2D((250, 550), angle, 1.0); matrix[:, 2] += [350, 250]
    warped = cv2.warpAffine(paper, matrix, (1200, 1600))
    mask = cv2.warpAffine(np.full(paper.shape[:2], 255, np.uint8), matrix, (1200, 1600))
    canvas[mask > 0] = warped[mask > 0]
    return canvas


def test_receipt_photographed_on_a_desk_is_cut_out_and_straightened():
    for background, angle in (((40, 40, 45), -8), ((40, 90, 150), 15), ((160, 90, 30), -25)):
        out, record = _crop_photographed_paper(_photo(background, angle))
        assert record["status"] == "rectified", (background, angle, record)
        height, width = record["output_shape"]
        assert abs(width - 500) < 60 and abs(height - 1100) < 120, record
        # no desk left in the result
        assert np.median(cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[:, :, 2]) > 180


def test_scanned_receipt_with_stains_is_not_cropped():
    scan = _receipt(620, 900)
    cv2.circle(scan, (300, 820), 140, (150, 160, 170), -1)  # a stain reaching the edge
    scan[:40] = (170, 175, 180)
    out, record = _crop_photographed_paper(scan)
    assert record["status"] != "rectified" and out is scan


def test_receipt_lying_across_beige_cloth_is_cut_out_and_stood_upright():
    rng = np.random.default_rng(7)
    canvas = np.zeros((1200, 1600, 3), np.uint8); canvas[:] = (150, 185, 205)  # beige cloth
    for k in range(14):  # coloured printed pattern on the cloth
        cv2.circle(canvas, (int(rng.integers(0, 1600)), int(rng.integers(0, 1200))), int(rng.integers(60, 160)),
                   [(70, 80, 170), (80, 150, 90), (60, 160, 200), (110, 110, 115)][k % 4], -1)
    canvas = np.clip(canvas.astype(int) + rng.normal(0, 8, canvas.shape), 0, 255).astype(np.uint8)
    paper = np.full((430, 1600, 3), (214, 214, 216), np.uint8)  # grey-white thermal paper, running off both sides
    for x in range(60, 1560, 34):
        cv2.putText(paper, "TOTAL 12.50", (x, 400), cv2.FONT_HERSHEY_SIMPLEX, .8, (50, 50, 50), 2)
    matrix = cv2.getRotationMatrix2D((800, 215), 1.5, 1.0); matrix[:, 2] += [0, 360]
    warped = cv2.warpAffine(paper, matrix, (1600, 1200))
    mask = cv2.warpAffine(np.full(paper.shape[:2], 255, np.uint8), matrix, (1600, 1200))
    canvas[mask > 0] = warped[mask > 0]
    out, record = _crop_photographed_paper(canvas)
    assert record["status"] == "rectified", record
    assert record.get("quarter_turn_degrees") == 90
    height, width = out.shape[:2]
    assert height > 2.5 * width, out.shape  # a tall strip, as receipts are
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    assert np.mean(hsv[:, :, 1] > 40) < 0.05  # almost no coloured cloth left


def test_upside_down_photo_is_turned_by_reading_both_ways():
    from types import SimpleNamespace
    from dsdnet_receipt.web_app import upright_by_reading
    page = np.full((400, 200, 3), 230, np.uint8); page[10:30, 10:60] = 0  # print starts at the top left when upright

    class Reader:  # reads well only when the dark header is at the top left
        def run(self, view):
            good = view[:40, :70].mean() < 200
            return SimpleNamespace(lines=[{"text": "CeX AEON Rawang", "confidence": 0.97 if good else 0.4}])

    out, record = upright_by_reading(np.ascontiguousarray(page[::-1, ::-1]), Reader())
    assert record["turned_180"] and out[:40, :70].mean() < 200
    out, record = upright_by_reading(page, Reader())
    assert not record["turned_180"] and out is page
