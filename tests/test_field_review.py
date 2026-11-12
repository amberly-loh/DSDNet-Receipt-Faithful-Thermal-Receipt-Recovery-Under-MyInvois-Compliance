from __future__ import annotations

from datetime import date
from pathlib import Path

from dsdnet_receipt.field_review import review_date, review_key_fields, review_number, review_total
from dsdnet_receipt.semantic import extract_line_items

TODAY = date(2026, 11, 11)
ROOT = Path(__file__).resolve().parents[1]


def line(text, x, y, w=None, h=28, confidence=0.95):
    w = w or max(20, 14 * len(text))
    return {"text": text, "confidence": confidence, "box": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]}


def karaoke_receipt():
    """The Koshidaka layout: 'Total' before charges, the amount paid under 'SubTotal'."""
    return [
        line("Koshidaka Malaysia Sdn Bhd", 272, 102),
        line("m17hi-4", 499, 155, confidence=0.55),
        line("INVOICE", 414, 268),
        line("Reciept", 172, 367, w=100),
        line("MYINUS-100-20003", 300, 367, w=320, confidence=0.87),
        line("Pax:2", 706, 366),
        line("Cashier b1", 166, 403),
        line("13/1012026 3:17:03 PM", 513, 403, w=360),
        line("Cty", 162, 438, w=40), line("Code & Desc", 235, 440, w=150), line("Total RM", 679, 443, w=110),
        line("2", 181, 485, w=14), line("HH-WE-HEAD CHARGE", 231, 485, w=260), line("43.22", 710, 490, w=70),
        line("PACKAGE HOUR", 232, 520), line("2-Hour singing session", 229, 553), line("#1x ADULT", 224, 588),
        line("#1x STUDENT", 223, 625),
        line("2", 167, 667, w=14), line("Type", 234, 667, w=60), line("1", 347, 668, w=14),
        line("Total:", 147, 706, w=80), line("43.22", 714, 721, w=70),
        line("Service Charge 10%", 148, 746), line("4.32", 728, 758, w=60),
        line("* SST 8%", 143, 781), line("3.46", 728, 794, w=60),
        line("Rounding Adj", 144, 820), line("0.00", 726, 833, w=60),
        line("SubTotal:", 146, 866, w=160, h=40), line("51.00", 630, 879, w=130, h=40),
        line("DUITNOW", 132, 918, w=120), line("51.00", 694, 938, w=70),
        line("Printing on 18/10/2026 3:18:29 PM", 119, 1231, w=420),
    ]


def test_karaoke_receipt_fields_are_reviewed_from_what_was_printed() -> None:
    lines = karaoke_receipt()
    extracted = {"merchant": {"value": "Koshidaka Malaysia Sdn Bhd"}, "transaction_date": {"value": None},
                 "receipt_invoice_number": {"value": "m17hi-4"}, "total_amount": {"value": "43.22"}}
    record = review_key_fields(lines, extracted, today=TODAY)
    assert extracted["total_amount"]["value"] == "51.00"
    assert extracted["transaction_date"]["value"] == "2026-10-13"      # not the printing date
    assert extracted["receipt_invoice_number"]["value"] == "MYINUS-100-20003"
    assert record["changes"]["total_amount"] == {"from": "43.22", "to": "51.00", "reason": "matches the payment line"}


def test_karaoke_receipt_has_one_item_with_its_description_lines() -> None:
    items = extract_line_items(karaoke_receipt())
    assert len(items) == 1
    assert items[0]["quantity"] == "2" and items[0]["amount"] == "43.22"
    assert "PACKAGE HOUR" in items[0]["description"] and "STUDENT" in items[0]["description"]
    assert "Type" not in items[0]["description"]


def test_total_rounded_and_single_total_and_tax_summary_under_payment() -> None:
    rounded = [line("Total", 28, 678), line("RM 16.52", 466, 681), line("ROUNDING ADJUSTMENT", 25, 705),
               line("-RM 0.02", 468, 710), line("TOTAL ROUNDED", 26, 732), line("RM 16.50", 467, 737),
               line("CASH", 24, 760), line("RM 17.00", 466, 764), line("CHANGE", 23, 788), line("RM 0.50", 478, 792)]
    assert review_total(rounded)[0] == "16.50"
    assert review_total([line("TOTAL", 30, 500), line("12.90", 400, 502)]) == ("12.90", "the only total printed")
    with_summary = [line("SUB-TOTAL(EX)", 588, 870), line("25.80", 932, 870), line("TOTAL TAX", 590, 900), line("1.55", 944, 898),
                    line("TOTAL", 588, 958), line("27.35", 915, 958), line("CASH", 590, 986), line("27.35", 916, 986),
                    line("TOTAL :", 586, 1268), line("25.80", 826, 1268)]
    assert review_total(with_summary)[0] == "27.35"
    assert review_total([line("Thank you", 10, 10)]) == (None, None)


def test_dates_misread_separators_and_unsafe_short_years() -> None:
    assert review_date([line("13/10/20263:17:03PM", 500, 400)], TODAY)[:2] == ("2026-10-13", True)
    assert review_date([line("Date:1.4-11-y", 100, 500)], TODAY)[0] is None          # mixed separators: not a date
    assert review_date([line("Date: 14/11/24", 100, 500)], TODAY)[0] == "2024-11-14"
    assert review_date([line("Date: 01/02/2031", 100, 500)], TODAY)[0] is None       # in the future
    assert review_date([line("Printed 02/03/2026 10:00", 100, 900)], TODAY)[0] is None


def test_receipt_numbers_need_a_label_and_digits() -> None:
    assert review_number([line("RecieptMYINUS-200-20003", 170, 367)])[0] == "MYINUS-200-20003"
    assert review_number([line("Invoice No: CS2405-174", 10, 10)])[0] == "CS2405-174"
    assert review_number([line("To request e Invoice please scan", 10, 10)])[0] is None
    assert review_number([line("Bill: 0123456789", 10, 10)])[0] is None             # a phone number


def test_the_evaluated_extractor_is_not_changed_by_the_review_layer() -> None:
    source = (ROOT / "src" / "dsdnet_receipt" / "fields.py").read_text(encoding="utf-8")
    assert "field_review" not in source
    app = (ROOT / "src" / "dsdnet_receipt" / "web_app.py").read_text(encoding="utf-8")
    assert app.index("extracted = extract_fields(ocr.text, ocr.lines)") < app.index("field_review_record = review_key_fields(ocr.lines, extracted)")


def test_shared_parser_of_the_revision_is_the_evaluated_extractor_plus_the_review() -> None:
    from dsdnet_receipt.field_review import extract_fields_reviewed
    from dsdnet_receipt.fields import extract_fields
    lines = karaoke_receipt()
    text = "\n".join(item["text"] for item in lines)
    expected = extract_fields(text, lines)
    review_key_fields(lines, expected)
    assert extract_fields_reviewed(text, lines) == expected
    reviewed = extract_fields_reviewed(text, lines)
    assert reviewed["total_amount"]["value"] == "51.00" and reviewed["total_amount"]["box"] is not None


def test_revised_parser_survives_the_run_script_swap(monkeypatch) -> None:
    # parser_revision_20261112.py points fields.extract_fields at extract_fields_reviewed;
    # the first attempt then recursed forever. Reproduce that swap exactly.
    import dsdnet_receipt.fields as fields
    from dsdnet_receipt.field_review import extract_fields_reviewed
    lines = karaoke_receipt()
    text = "\n".join(item["text"] for item in lines)
    expected = extract_fields_reviewed(text, lines)
    monkeypatch.setattr(fields, "extract_fields", extract_fields_reviewed)
    from dsdnet_receipt.fields import extract_fields  # what the run loop imports inside main()
    assert extract_fields(text, lines) == expected
    assert expected["total_amount"]["value"] == "51.00"
