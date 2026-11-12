from dsdnet_receipt.capture_check import check_capture


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def line(text, b, conf=0.95):
    return {"text": text, "confidence": conf, "box": b}


UPRIGHT = [line("KEDAI RUNCIT ABC SDN BHD", box(40, 40, 600, 80)), line("Date: 19/10/2026 10:21", box(40, 100, 520, 130)),
           line("Milo 2 x 4.50 9.00", box(40, 300, 600, 330)), line("Roti 1 x 3.20 3.20", box(40, 340, 600, 370)),
           line("TOTAL RM 12.20", box(40, 700, 600, 740)), line("CASH 20.00 CHANGE 7.80", box(40, 760, 600, 790))]


def test_a_normal_receipt_is_left_alone():
    fields = {"total_amount": "12.20", "transaction_date": "2026-10-19", "merchant": "KEDAI RUNCIT ABC"}
    assert check_capture(UPRIGHT, (1000, 700), fields)["status"] == "ok"


def test_text_running_up_and_down_means_turned_sideways():
    sideways = [line(t, box(40 + 50 * i, 40, 80 + 50 * i, 560)) for i, t in enumerate(["KEDAI", "TOTAL", "12.20", "Milo", "Roti", "CASH"])]
    out = check_capture(sideways, (700, 1000), {})
    assert out["status"] == "retry" and out["issue"] == "rotation" and "turned" in out["message"]


def test_almost_nothing_read_asks_again():
    out = check_capture([line("ab", box(1, 1, 30, 10)), line("1.0", box(1, 20, 30, 30))], (1000, 700), {})
    assert out["status"] == "retry"


def test_text_cut_at_the_bottom_with_no_total_means_bad_crop():
    cut = UPRIGHT[:4] + [line("TOT", box(40, 985, 300, 1000), 0.6)]
    out = check_capture(cut, (1000, 700), {"merchant": "KEDAI", "transaction_date": "2026-10-19"})
    assert out["status"] == "retry" and out["issue"] == "crop"


def test_check_never_changes_values():
    fields = {"total_amount": "12.20"}
    check_capture(UPRIGHT, (1000, 700), fields)
    assert fields == {"total_amount": "12.20"}
