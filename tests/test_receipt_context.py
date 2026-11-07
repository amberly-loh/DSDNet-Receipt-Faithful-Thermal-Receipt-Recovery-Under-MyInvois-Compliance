from dsdnet_receipt.receipt_context import clean_address_line, recognise_receipt, tidy_text
from dsdnet_receipt.semantic import extract_accounting_fields


def test_glued_and_misread_words_are_repaired_only_from_the_lexicon():
    assert tidy_text("tapan Pulth Tujuh Erterprise") == "Lapan Puluh Tujuh Enterprise"
    assert tidy_text("PQSA-PS 8atu7JalanlpohBatu6").endswith("P/S Batu 7 Jalan Ipoh Batu 6")
    assert tidy_text("Taman Tapan Jaya") == "Taman Tapan Jaya"  # ambiguous: left as read
    assert clean_address_line("LotPT170MukimBaru KualaLun)pur DATE06FEB VNO150") == "Lot PT 170 Mukim Baru Kuala Lumpur"


def test_fuel_receipt_is_recognised_and_dealer_kept_as_supplier():
    lines = [{"text": t, "confidence": 0.9, "box": [[60, y], [260, y], [260, y + 12], [60, y + 12]]} for t, y in (
        ("PETRONAS", 70), ("tapan Pulth Tujuh Erterprise", 92), ("PSBatu7Jatontpoh(Batu6)", 115),
        ("LotPT170MukimBaru KualaLun)pur", 136), ("INALRECEIPTTCOPYY", 162), ("Pnmax95RM10.00", 182),
        ("DATE06FEB", 196), ("Pump7", 218), ("PQSA-PS 8atu7JalanlpohBatu6", 265), ("Grand Total 10.00", 377))]
    fields = extract_accounting_fields("\n".join(l["text"] for l in lines), lines, {"merchant": "PETRONAS", "total_amount": "10.00"})
    assert fields["supplier_name"] == "Lapan Puluh Tujuh Enterprise"
    assert fields["business_type"] == "Petrol station" and fields["suggested_expense_type"] == "fuel"
    assert fields["purchase_summary"] == "Fuel (RON95 petrol)"
    assert fields["document_type"] == "Receipt"
    assert "DATE" not in fields["supplier_address"] and "Kuala Lumpur" in fields["supplier_address"]


def test_grocery_items_give_the_purpose_without_a_known_brand():
    context = recognise_receipt("KEDAI RUNCIT AH SENG\nBERAS 5KG 18.90\nGULA 1KG 2.85\nTELUR GRED A 13.50\nTOTAL 35.25")
    assert context["purchase_summary"] == "Groceries" and context["expense_type"] == "inventory"
    assert recognise_receipt("ABC SDN BHD\nTOTAL 10.00")["business_type"] is None


def test_reading_copy_keeps_one_reading_per_printed_place():
    from dsdnet_receipt.receipt_context import reading_copy_lines
    box = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    lines = [
        {"text": "tapan Pulth Tujuh Erterprise", "confidence": 0.85, "box": box(75, 93, 249, 108)},
        {"text": "arerpse", "confidence": 0.71, "box": box(74, 92, 249, 109)},
        {"text": "aprTrerprise", "confidence": 0.71, "box": box(75, 92, 249, 109)},
        {"text": "Pomax95RM10.00", "confidence": 0.89, "box": box(53, 181, 184, 195)},
        {"text": "100", "confidence": 0.89, "box": box(121, 181, 182, 192)},
        {"text": "OA", "confidence": 0.66, "box": box(140, 199, 168, 209)},
        {"text": "Rounagg", "confidence": 0.68, "box": box(180, 353, 234, 369)},
    ]
    assert [line["text"] for line in reading_copy_lines(lines)] == [
        "Lapan Puluh Tujuh Enterprise", "Primax 95 RM 10.00", "Rounding"]


def test_one_line_items_are_read_from_small_till_receipts():
    from dsdnet_receipt.semantic import extract_line_items
    box = lambda y: [[10, y], [200, y], [200, y + 12], [10, y + 12]]
    lines = [{"text": t, "confidence": .9, "box": box(y)} for t, y in (
        ("KEDAI KOPI ABC", 10), ("2 NASI LEMAK 11.00", 40), ("TEH TARIK 2.80", 60), ("SST 6% 0.83", 80),
        ("TOTAL 14.63", 100), ("CASH 20.00", 120))]
    rows = [(r["quantity"], r["description"], r["amount"]) for r in extract_line_items(lines)]
    assert rows == [("2", "NASI LEMAK", "11.00"), ("", "TEH TARIK", "2.80")]
    fuel = [{"text": "Pomax95RM10.00", "confidence": .89, "box": box(180)}, {"text": "Grand Total", "confidence": .93, "box": box(377)}]
    assert [(r["description"], r["amount"]) for r in extract_line_items(fuel)] == [("Primax 95", "10.00")]


def test_multi_line_items_close_at_their_price_and_keep_serials_with_them():
    from dsdnet_receipt.semantic import extract_line_items
    box = lambda x0, y, x1: [[x0, y], [x1, y], [x1, y + 12], [x0, y + 12]]
    rows = [("Exchange Goods Details", 100, 60, 260), ("Nintendo Switch Black Pro Controller", 120, 10, 230),
            ("+ USB C Cable", 136, 14, 90), ("(Switch Accessories, MY)", 152, 10, 150), ("1 @", 152, 200, 222),
            ("RM126.00", 152, 240, 300), ("Serial Number : PDGN3AA", 168, 10, 160), ("Fire Emblem Engage", 196, 10, 120),
            ("(Switch Games R2 ENG, MY)", 212, 10, 160), ("1 @", 212, 200, 222), ("RM106.00", 212, 240, 300),
            ("Total Item Count: 2", 240, 80, 220), ("Payment Due", 270, 10, 90), ("RM-232.00", 270, 240, 300)]
    lines = [{"text": t, "confidence": .95, "box": box(x0, y, x1)} for t, y, x0, x1 in rows]
    items = extract_line_items(lines)
    assert [(i["quantity"], i["amount"]) for i in items] == [("1", "126.00"), ("1", "106.00")]
    assert "Serial Number" in items[0]["description"] and "Fire Emblem" in items[1]["description"]
    assert "Serial" not in items[1]["description"]


def test_tags_follow_the_recognised_business():
    from dsdnet_receipt.fields import infer_receipt_tags
    assert "electronics" in infer_receipt_tags("CeX AEON Rawang\nExchange Goods Details", "CeX AEON Rawang")
    assert infer_receipt_tags("PETRONAS\nPrimax 95 RM10.00", "PETRONAS")[0] == "fuel"
