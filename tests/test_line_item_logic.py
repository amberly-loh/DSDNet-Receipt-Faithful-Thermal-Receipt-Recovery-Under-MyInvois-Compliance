from dsdnet_receipt.line_item_logic import refine_line_items


def row(description="", quantity="", unit_price="", amount="", unit=""):
    return {"item": "1", "description": description, "quantity": quantity, "unit": unit,
            "unit_price": unit_price, "discount": "", "amount": amount, "review_note": ""}


def lines(*texts):
    return [{"text": t} for t in texts]


def test_quantity_and_price_inside_description():
    out = refine_line_items([row("1 P08 BM6A PLUG 2.00 PCS 6.00", amount="12.00")], lines("P08 BM6A PLUG", "PCS", "6.00", "12.00"), "12.00")
    assert out[0]["description"] == "P08 BM6A PLUG"
    assert (out[0]["quantity"], out[0]["unit"], out[0]["unit_price"], out[0]["amount"]) == ("2", "PCS", "6.00", "12.00")


def test_printed_quantity_completes_rows_and_model_line_is_folded():
    items = [row("PCC CASTLE (40BAGS/PLT)", unit_price="22.00"), row("FOSROC AP CONPLAST 5L", unit_price="42.00"),
             row("ARROW PIONEER BOX", unit_price="14.80"), row('MODELM-39816")', unit_price="14.80")]
    text = lines("20.000 BAG", "22.00", "440.00", "42.00", "1.000 UNIT", "42.00", "1.000 PCS", "14.80", "14.80", "TOTAL 496.80")
    out = refine_line_items(items, text, "496.80")
    assert [r["amount"] for r in out] == ["440.00", "42.00", "14.80"]
    assert out[0]["quantity"] == "20" and out[0]["unit"] == "BAG"
    assert "MODELM" in out[2]["description"]


def test_repeated_amount_becomes_the_remainder_of_the_total():
    out = refine_line_items([row("TOYS", amount="19.00"), row("HARDWARE", amount="19.00")], lines("TOYS 19.00", "Net Amt : 24.00"), "24.00")
    assert [r["amount"] for r in out] == ["19.00", "5.00"]
    assert "total less the other items" in out[1]["review_note"]


def test_pump_codes_are_not_items_and_fuel_row_is_built():
    items = [row("POS1", unit_price="46.82", amount="96.00"), row("PBL#", unit_price="46.82", amount="96.00"),
             row("Blaze 95", unit_price="46.82", amount="96.00")]
    text = lines("PETRON", "Pump 24", "46.828L2.050", "46.82", "96.00", "2.050", "Total RM: 96.00")
    out = refine_line_items(items, text, "96.00", "RON95 petrol")
    assert len(out) == 1
    assert (out[0]["description"], out[0]["quantity"], out[0]["unit"], out[0]["unit_price"]) == ("RON95 petrol", "46.82", "L", "2.050")


def test_fuel_row_from_digits_without_decimal_points():
    text = lines("FS DEUSP3060", "1396", "2150R", "hhtre", "Total", "R3000", "Cash 3000")
    out = refine_line_items([], text, "30.00")
    assert (out[0]["description"], out[0]["quantity"], out[0]["unit_price"], out[0]["amount"]) == ("Diesel", "13.96", "2.150", "30.00")


def test_product_name_under_the_price():
    items = [row("Type Descriplion Atir Size Qty Price Ext Price Sale 77922616/3C7X375 ..0 0 2279723780027 1 RM35.00", amount="35.00")]
    text = lines("Items", "Qty", "2279723780027 1", "RM35.00", "Glitter Heart Swim Ring", "1 Item Sold", "Receipt Total", "RM35.00")
    out = refine_line_items(items, text, "35.00")
    assert out[0]["description"] == "Glitter Heart Swim Ring"
    assert (out[0]["quantity"], out[0]["unit_price"]) == ("1", "35.00")


def test_quantity_from_amount_and_price():
    out = refine_line_items([row("GARBAGE BAG", unit_price="6.90", amount="13.80")], lines("6.90", "13.80"), "13.80")
    assert out[0]["quantity"] == "2"


def test_nothing_invented_without_arithmetic_support():
    items = [row("OGAWA Hand Blower")]
    out = refine_line_items(items, lines("OGAWA Hand Blower", "340", "340"), "340.00")
    assert out[0]["amount"] == "" and "Verify" in out[0]["review_note"]
