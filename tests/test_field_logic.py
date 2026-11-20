from datetime import date

from dsdnet_receipt.field_logic import apply_logic, check_total, repair_date, suggest_partial_date

TODAY = date(2026, 11, 14)


def lines(*texts):
    return [{"text": t, "confidence": 0.9} for t in texts]


def test_lost_decimal_point_on_fuel_receipt():
    # "RM 30 00" read as 3000; no amount on the receipt kept its decimal point
    ocr = lines("FS DEUSP3060", "3000", "2150R", "Total", "R3000", "3000", "Cash", "3000", "2023-043910:0435")
    assert check_total("3000.00", ocr)[0] == "30.00"


def test_handwritten_whole_ringgit_bill_is_kept():
    # no printed time: a handwritten bill, 340 is RM 340
    assert check_total("340.00", lines("OGAWA Hand Blower", "340", "Total 340"))[0] is None


def test_total_with_decimal_point_is_kept():
    assert check_total("3000.00", lines("TOTAL RM 3,000.00", "CASH 3000.00"))[0] is None
    assert check_total("1590.00", lines("TOTAL 1590.00", "ITEM 1590.00"))[0] is None


def test_lost_decimal_point_confirmed_by_other_line():
    assert check_total("2400.00", lines("HARDWARE 2400", "Net Amt : 24.00"))[0] == "24.00"


def test_only_rm_amount_when_total_missing():
    assert check_total(None, lines("TOTAL AMOUNT", "CASH.", "10", "RM-10", "RM"))[0] == "10.00"


def test_cash_minus_change_corrects_misread_total():
    ocr = lines("TOTAL 83.35", "CASH 100.00", "CHANGE 36.65", "SUBTOTAL 63.35")
    assert check_total("83.35", ocr)[0] == "63.35"


def test_exact_payment_with_dropped_digit_keeps_total():
    ocr = lines("TOTAL INCL GST 56.00", "PAY CASH 6.00", "CHANGE 0.00")
    assert check_total("56.00", ocr)[0] is None


def test_iso_date_with_impossible_day():
    value, why, _ = repair_date(lines("2023-043910:0435", "TerminalID"), TODAY)
    assert value == "2023-04-30" and "look-alike" in why


def test_missing_separator_before_year():
    assert repair_date(lines("DATE 25/122018B1339PM"), TODAY)[0] == "2018-12-25"


def test_misspelt_month_name():
    assert repair_date(lines("UNIHAKKA 21MR 2018 7:59"), TODAY)[0] == "2018-03-21"


def test_amount_followed_by_year_is_not_a_date():
    assert repair_date(lines("AMOUNT 13.87 2000 30.00"), TODAY)[0] is None


def test_unreadable_year_is_suggested_not_filled():
    ocr = lines("Date", "29/02453645PM")
    assert suggest_partial_date(ocr, TODAY) == ("29/02", "2024-02-29")
    extracted = {"transaction_date": {"value": None}, "total_amount": {"value": "24.00"}}
    record = apply_logic(ocr, extracted, TODAY)
    assert extracted["transaction_date"]["value"] is None
    assert extracted["transaction_date"]["suggested"] == "2024-02-29"
    assert "transaction_date_suggestion" in record["changes"]


def test_every_change_is_recorded():
    extracted = {"total_amount": {"value": "3000.00"}, "transaction_date": {"value": None}}
    record = apply_logic(lines("Total R3000", "Cash 3000", "2023-043910:0435"), extracted, TODAY)  # printed time 10:04
    assert record["changes"]["total_amount"]["from"] == "3000.00"
    assert record["changes"]["total_amount"]["to"] == "30.00"
    assert extracted["transaction_date"]["value"] == "2023-04-30"


def test_unsure_date_is_estimated_and_other_values_offered():
    from dsdnet_receipt.field_logic import add_options
    ocr = lines("SKL SELAYANG BARU DIY ENTERPRISE", "Date", "29/02453645PM", "Net Amt : 24.00")
    extracted = {"merchant": {"value": "SKLSEUAYANG BARUDIY ENTERPRISE"}, "transaction_date": {"value": None}, "total_amount": {"value": "24.00"}}
    add_options(ocr, extracted, TODAY)
    assert extracted["transaction_date"]["value"] == "2024-02-29"
    assert extracted["transaction_date"]["estimated"].startswith("Estimated:")
    assert extracted["merchant"]["options"][0]["value"] == "SKL SELAYANG BARU DIY ENTERPRISE"


def test_read_values_are_not_marked_estimated():
    from dsdnet_receipt.field_logic import add_options
    extracted = {"merchant": {"value": "NTME TRADING"}, "transaction_date": {"value": "2024-08-12"}, "total_amount": {"value": "12.00"}}
    add_options(lines("NTME TRADING (M) SDN BHD", "Date 12/08/2024", "Total Payable RM 12.00"), extracted, TODAY)
    assert not any(extracted[n].get("estimated") for n in extracted)
    assert extracted["transaction_date"]["options"] == []
