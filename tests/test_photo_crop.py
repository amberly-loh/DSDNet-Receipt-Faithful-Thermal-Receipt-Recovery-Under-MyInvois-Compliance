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
