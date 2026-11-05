import cv2
import numpy as np

from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import make_faithful_content_view


def test_registered_color_and_gray_have_identical_geometry():
    image = np.full((500, 320, 3), (205, 235, 210), np.uint8)
    for y in range(50, 460, 40):
        cv2.putText(image, 'TOTAL 30.00', (25, y), 0, .55, (35, 35, 35), 1)
    config = {'preprocessing': {'max_long_edge': 400, 'boundary_detection': False}}
    gray, _ = preprocess(image, config)
    color, _ = preprocess(image, config, preserve_color=True)
    assert color.shape[:2] == gray.shape
    # Resize and color conversion differ by at most rounding at glyph edges.
    assert np.max(np.abs(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(int) - gray.astype(int))) <= 1


def test_color_faithful_preserves_paper_ink_and_signature_when_model_erases_them():
    color = np.full((300, 300, 3), (205, 235, 210), np.uint8)
    cv2.rectangle(color, (20, 30), (65, 75), (120, 25, 15), -1)
    cv2.line(color, (40, 220), (140, 195), (180, 35, 20), 2)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    out = make_faithful_content_view(gray, np.full_like(gray, 255), [], source_color=color)
    assert out.shape == color.shape
    assert out[120, 120, 1] > out[120, 120, 0] + 20
    assert out[50, 50, 0] > out[50, 50, 2] + 65
    assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[208, 90] < 150


def test_table_route_keeps_all_grid_lines_without_ocr():
    source = np.full((500, 400), 230, np.uint8)
    for y in range(60, 440, 35):
        cv2.line(source, (20, y), (380, y), 90, 1)
    for x in (20, 160, 280, 380):
        cv2.line(source, (x, 60), (x, 410), 90, 1)
    cv2.circle(source, (80, 35), 12, 45, -1)
    cv2.line(source, (120, 455), (240, 470), 70, 2)
    out = make_faithful_content_view(source, np.full_like(source, 255), [])
    assert np.all(out[95, 25:375] < 140)
    assert np.all(out[65:405, 160] < 140)
    assert out[35, 80] < 100
    assert out[463, 180] < 140


def test_diagonal_gap_requires_candidate_evidence():
    from dsdnet_receipt.restoration import repair_evidence_constrained_gaps
    source = np.full((50, 50), 245, np.uint8)
    cv2.line(source, (10, 10), (35, 35), 80, 1)
    candidate = source.copy()
    source[22, 22] = 245
    line = {'confidence': .95, 'box': [[5,5],[40,5],[40,40],[5,40]]}
    output = repair_evidence_constrained_gaps(np.full_like(source, 255), source, candidate, [line])
    assert output[22, 22] < 220
    unsupported = repair_evidence_constrained_gaps(np.full_like(source, 255), source, source, [line])
    assert unsupported[22, 22] == 255


def test_neutral_paper_lift_retains_faint_ink_and_rules():
    from dsdnet_receipt.restoration import lighten_neutral_paper
    source = np.full((160, 260), 215, np.uint8)
    cv2.putText(source, 'TOTAL 12.00', (20, 65), 0, .6, 165, 1)
    cv2.line(source, (15, 100), (240, 100), 130, 1)
    output = lighten_neutral_paper(source, source)
    assert output[125, 125] > source[125, 125] + 5
    assert np.array_equal(output[source < 180], source[source < 180])


def test_native_thermal_contrast_does_not_depend_on_which_words_ocr_finds():
    source = np.full((350, 170), 235, np.uint8)
    for y in range(35, 320, 30):
        cv2.putText(source, 'TOTAL 12.00', (10, y), 0, .35, 145, 1)
    lines = [{'confidence': .99, 'box': [[8,15],[155,15],[155,38],[8,38]]}]
    without_ocr = make_faithful_content_view(source, source, [])
    with_partial_ocr = make_faithful_content_view(source, source, lines)
    assert np.array_equal(without_ocr, with_partial_ocr)


def test_clipped_midtone_page_can_be_rectified_without_closed_edge_contour():
    from dsdnet_receipt.preprocessing import _rectify_receipt
    image = np.zeros((700, 500), np.uint8)
    polygon = np.array([[0,80],[350,0],[499,699],[170,699]], np.int32)
    cv2.fillConvexPoly(image, polygon, 210)
    cv2.putText(image, 'INVOICE', (100,180), 0, .6, 50, 1)
    output, record = _rectify_receipt(image)
    assert record['status'] == 'rectified'
    assert output.shape[0] > output.shape[1]
    assert np.mean(output < 20) < .03
