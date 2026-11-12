from dsdnet_receipt.semantic import extract_accounting_fields, extract_line_items


def line(text, x1, y1, x2, y2, confidence=.95):
    return {"text": text, "box": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "confidence": confidence}


def test_combined_header_reconstructs_item_and_ambiguous_quantity():
    lines = [
        line("Qty Description", 50,100,300,130), line("Price", 500,100,600,130), line("Total(RM)",650,100,800,130),
        line('i.38 3/4"ALUMINIUM ROD', 90,160,390,190), line("5.80",530,160,590,190), line("19.60 SR",680,160,780,190),
        line("Total",300,240,400,270), line("19.60",680,240,760,270),
    ]
    rows = extract_line_items(lines)
    assert len(rows) == 1
    assert rows[0]["description"] == '3/4"ALUMINIUM ROD'
    assert rows[0]["quantity"] == "3.38"
    assert rows[0]["unit_price"] == "5.80"
    assert rows[0]["amount"] == "19.60"


def test_handwritten_multiline_description_is_one_evidence_row():
    lines = [
        line("Quantity",50,100,150,130), line("PARTICULARS",300,100,600,130), line("Unit Price",700,100,850,130), line("Amount RM",900,100,1050,130),
        line("0GAwn-fand BlowLv",250,160,600,190), line("Modwi OHB J6oo7",250,205,600,235),
        line("Standard AccessoneaToo",250,250,650,280), line("340",930,250,1000,280),
        line("TOTAL RM",750,700,900,730), line("340",930,700,1000,730),
    ]
    rows = extract_line_items(lines)
    assert len(rows) == 1
    assert "OGAWA-Hand Blower" in rows[0]["description"]
    assert "Standard Accessories Tools" in rows[0]["description"]
    assert rows[0]["amount"] == "340.00"


def test_compact_stacked_thermal_layout_reconstructs_all_items():
    lines = [
        line("QTY UOM", 85,637,214,665), line("UNIT Amt Exc. Amt Inc. GST",279,637,657,667),
        line("Price",279,672,343,697), line("Tax Code",541,672,657,700),
        line("A&M SARSAP 330ML",86,718,372,747), line("1",91,757,116,778),
        line("WALK",139,753,229,783), line("1.20",284,753,347,783),
        line("1.20",405,753,466,785), line("1.20",526,755,599,785), line("ZRL",597,757,655,787),
        line("100PLUS REGULAR 500ML",88,803,456,832), line("1",93,842,116,863),
        line("WALK",144,840,229,867), line("1.90",285,838,347,868),
        line("1.90",406,838,466,868), line("1.90",526,838,599,868), line("ZRL",594,840,655,870),
        line("Total Qty: 2.00",81,1065,294,1104), line("3.10",574,1072,657,1102),
    ]
    rows = extract_line_items(lines)
    assert [row["description"] for row in rows] == ["A&M SARSAP 330ML", "100PLUS REGULAR 500ML"]
    assert [row["quantity"] for row in rows] == ["1", "1"]
    assert [row["unit"] for row in rows] == ["WALK", "WALK"]
    assert [row["unit_price"] for row in rows] == ["1.20", "1.90"]
    assert [row["amount"] for row in rows] == ["1.20", "1.90"]


def test_damaged_header_falls_back_to_product_blocks_before_total():
    lines = [
        line("MR D.I.Y. SDN BHD",180,100,500,130), line("-INVOICE-",260,180,410,210),
        line("PAPERONE COPIER A4 75G 500S",70,260,430,290), line("1 X 13.50",390,292,500,322),
        line("13.50",540,292,610,322), line("DOCUMENT BAG TS-117F",70,340,360,370),
        line("2 X 1.51",390,372,490,402), line("3.02",540,372,610,402),
        line("Item(s): 2",70,440,190,470), line("Total",70,500,150,530), line("RM 16.52",520,500,630,530),
    ]
    rows = extract_line_items(lines)
    assert [row["description"] for row in rows] == ["PAPERONE COPIER A4 75G 500S", "DOCUMENT BAG TS-117F"]
    assert [row["quantity"] for row in rows] == ["1", "2"]
    assert [row["unit_price"] for row in rows] == ["13.50", "1.51"]
    assert [row["amount"] for row in rows] == ["13.50", "3.02"]


def test_fuzzy_total_and_damaged_quantity_row_do_not_create_a_fake_item():
    lines = [
        line("INV8ICE NO : 18291/102/T0380", 200, 500, 600, 530),
        line("SS93 LEA & PERRINS 50S WORC", 60, 700, 445, 730),
        line("RM98.89", 500, 700, 635, 730),
        line("2i1 X RM 8.93", 85, 745, 275, 780),
        line("Tota} Sales (Inclusive GST) RM", 80, 835, 480, 865),
        line("98.87", 525, 835, 610, 865),
    ]
    rows = extract_line_items(lines)
    assert len(rows) == 1
    assert rows[0]["description"] == "SS93 LEA & PERRINS 50S WORC"
    assert rows[0]["unit_price"] == "8.93"
    assert rows[0]["amount"] == "98.89"


def test_accounting_fields_use_header_context_and_item_arithmetic():
    lines = [
        line("PASARAYA BORONG PINTAR", 100, 100, 620, 130),
        line("SDN BHD", 280, 140, 460, 170),
        line("BR No.: (124525-H)", 250, 180, 500, 210),
        line("NO 19-G JALAN TASIK UTAMA 4", 90, 220, 620, 250),
        line("MEDAN NIAGA TASIK DAMAI", 200, 260, 540, 290),
        line("016-5498845", 290, 300, 450, 330),
        line("TAXINVOICE", 220, 400, 520, 440),
        line("QTY UOM", 85, 500, 214, 530), line("UNIT Amt Inc. GST", 279, 500, 657, 530),
        line("A&M SARSAP 330ML", 86, 550, 372, 580), line("1", 91, 590, 116, 615),
        line("WALK", 139, 590, 229, 620), line("1.20", 284, 590, 347, 620), line("1.20", 526, 590, 599, 620),
        line("Total Qty: 1.00", 82, 680, 294, 710), line("1.20", 574, 680, 657, 710),
        line("Total Includes GST 0%", 83, 725, 416, 755),
        line("Cash", 85, 780, 159, 810),
    ]
    text = "\n".join(item["text"] for item in lines)
    result = extract_accounting_fields(text, lines, {
        "merchant": "SDN BHD", "transaction_date": "2018-03-09",
        "receipt_invoice_number": "19-G", "total_amount": "1.20",
    })
    assert result["supplier_name"] == "PASARAYA BORONG PINTAR SDN BHD"
    assert result["supplier_registration_number"] == "124525-H"
    assert result["supplier_tax_number"] is None
    assert result["supplier_address"] == "NO 19-G JALAN TASIK UTAMA 4 MEDAN NIAGA TASIK DAMAI"
    assert result["supplier_phone"] == "016-5498845"
    assert result["document_type"] == "Tax Invoice"
    assert result["currency"] == "MYR"
    assert result["payment_method"] == "Cash"
    assert result["subtotal"] == "1.20"
    assert result["tax_amount"] == "0.00"


def test_accounting_fields_support_unlabelled_malaysian_cash_sale_header():
    text = """CHOP HOCK CHUAN HIN
(206259-X)
No. 2, Jalan Jati Utama 2, Bandar Baru Batang Kali,
44300 Batang Kali, Hulu Selangor.
Tel / Fax: 03-6057 3046
CASH SALE
Quantity PARTICULARS Unit Price Amount RM sen
OGAWA Hand Blower
TOTAL RM 340.00
"""
    lines = [
        line(value, 10, y * 30, 500, y * 30 + 18)
        for y, value in enumerate(text.splitlines(), start=1)
    ]
    result = extract_accounting_fields(text, lines, {
        "merchant": "CHOP HOCK CHUAN HIN", "receipt_invoice_number": "61036",
        "transaction_date": "2024-03-19", "total_amount": "340.00",
    })
    assert result["supplier_registration_number"] == "206259-X"
    assert result["supplier_phone"] == "03-6057 3046"
    assert result["supplier_address"] == "No. 2, Jalan Jati Utama 2, Bandar Baru Batang Kali, 44300 Batang Kali, Hulu Selangor."
    assert result["document_type"] == "Cash Sale"
    assert result["subtotal"] == "340.00"
