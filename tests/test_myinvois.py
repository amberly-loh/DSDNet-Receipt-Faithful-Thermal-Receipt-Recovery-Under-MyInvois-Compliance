from __future__ import annotations

import json
from pathlib import Path

from dsdnet_receipt import myinvois as mi

NTME_OCR = [
    {"text": "NTME TRADING (M) SDN BHD 199301013761 (268499-X)", "box": [[340, 110], [1040, 110], [1040, 145], [340, 145]], "confidence": .97},
    {"text": "NO 29, JALAN 3/108C, TAMAN SUNGAI BESI, 57100 KUALA LUMPUR", "box": [[370, 147], [860, 147], [860, 168], [370, 168]], "confidence": .93},
    {"text": "All cheques should be crossed and made payable to NTME TRADING (M) SDN BHD", "box": [[30, 1270], [630, 1270], [630, 1290], [30, 1290]], "confidence": .9},
    {"text": "1", "box": [[90, 550], [100, 550], [100, 570], [90, 570]], "confidence": .9},
    {"text": "CMR6A PLUG", "box": [[140, 548], [260, 548], [260, 572], [140, 572]], "confidence": .95},
    {"text": "2.00", "box": [[690, 552], [730, 552], [730, 572], [690, 572]], "confidence": .95},
    {"text": "PCS", "box": [[770, 552], [800, 552], [800, 572], [770, 572]], "confidence": .95},
    {"text": "20.00", "box": [[860, 552], [910, 552], [910, 572], [860, 572]], "confidence": .95},
    {"text": "40.00", "box": [[1110, 552], [1170, 552], [1170, 572], [1110, 572]], "confidence": .95},
]


def ntme_receipt() -> dict:
    return {
        "merchant": "NTME TRADING(M) SDN BHD 199301013761(268499-x)",
        "transaction_date": "21/05/2024", "receipt_number": "CS 2405-171", "total_amount": "40.00",
        "accounting_fields": {
            "supplier_name": "NTME TRADING(M) SDN BHD 199301013761(268499-x)",
            "supplier_registration_number": "268499-x",
            "supplier_address": "NO 29, JALAN 3/108C, TAMAN SUNGAI BESI, 57100 KUALA LUMPUR, TEL 03-79806311",
            "supplier_phone": "03-7980631103-79806175", "supplier_email": "EMAILntme6663@gmail.com",
            "payment_method": "Cash", "currency": "MYR", "tax_amount": None, "discount": None,
        },
        # the receipt reader put the quantity in the unit-price column
        "line_items": [{"item": "1", "description": "CMR6A PLUG", "quantity": "", "unit": "", "unit_price": "2.00", "discount": "", "amount": "40.00"}],
        "ocr_text": "\n".join(line["text"] for line in NTME_OCR),
        "audit": {"ocr_lines": NTME_OCR},
    }


def complete(draft: dict) -> dict:
    draft["supplier"].update({"tin": "C2584563222", "msic": "46900", "activity": "Wholesale of a variety of goods"})
    draft["buyer"].update({"name": "Example Sdn Bhd", "tin": "C1234567890", "id_type": "BRN", "id_value": "202001234567",
                           "sst": "NA", "phone": "+60312345678"})
    draft["buyer"]["address"].update({"line0": "Lot 66", "city": "Petaling Jaya", "state": "10", "country": "MYS"})
    return draft


def test_receipt_values_are_cleaned_and_reconciled():
    draft = mi.suggest_from_receipt(ntme_receipt())
    supplier = draft["supplier"]
    assert supplier["name"] == "NTME TRADING(M) SDN BHD"
    assert (supplier["id_type"], supplier["id_value"]) == ("BRN", "199301013761")
    assert supplier["phone"] == "+60379806311"
    assert supplier["email"] == "ntme6663@gmail.com"
    assert supplier["address"]["postal"] == "57100" and supplier["address"]["state"] == "14"
    assert supplier["address"]["city"] == "Kuala Lumpur"
    assert draft["invoice"]["transaction_date"] == "2024-05-21"
    assert draft["invoice"]["payment_mode"] == "01"          # not "cheque" from the printed note
    line = draft["lines"][0]
    assert (line["quantity"], line["unit_price"], line["total_excl_tax"]) == ("2", "20.00", "40.00")
    assert draft["totals"]["payable"] == "40.00" and draft["totals"]["total_incl_tax"] == "40.00"
    assert draft["provenance"]["supplier.id_value"] == "receipt"
    assert draft["provenance"]["lines.0.tax_type"] == "default"


def test_validation_lists_missing_fields_then_passes_when_complete():
    draft = mi.suggest_from_receipt(ntme_receipt())
    first = mi.validate_draft(draft)
    assert not first["ready"]
    assert {"supplier.tin", "supplier.msic", "buyer.tin"} <= set(first["needs_input"])
    result = mi.validate_draft(complete(draft))
    assert result["ready"], result["issues"]
    document = mi.build_ubl_json(draft)
    invoice = document["Invoice"][0]
    assert document["_D"].endswith("Invoice-2")
    assert invoice["InvoiceTypeCode"][0] == {"_": "01", "listVersionID": "1.0"}
    ids = {item["ID"][0]["schemeID"]: item["ID"][0]["_"] for item in invoice["AccountingSupplierParty"][0]["Party"][0]["PartyIdentification"]}
    assert ids == {"TIN": "C2584563222", "BRN": "199301013761", "SST": "NA", "TTX": "NA"}
    assert invoice["LegalMonetaryTotal"][0]["PayableAmount"][0] == {"_": 40.0, "currencyID": "MYR"}
    assert invoice["InvoiceLine"][0]["Item"][0]["CommodityClassification"][0]["ItemClassificationCode"][0]["_"] == "022"
    json.dumps(document)


def test_format_and_arithmetic_rules():
    draft = complete(mi.suggest_from_receipt(ntme_receipt()))
    draft["supplier"]["tin"] = "12345"
    draft["supplier"]["msic"] = "4690"
    draft["supplier"]["phone"] = "03-7980"
    draft["totals"]["payable"] = "45.00"
    paths = {issue["path"] for issue in mi.validate_draft(draft)["issues"] if issue["severity"] == "error"}
    assert {"supplier.tin", "supplier.msic", "supplier.phone", "totals.payable"} <= paths


def test_consolidated_mode_uses_general_public_rules():
    base = mi.suggest_from_receipt(ntme_receipt())
    draft = mi.merge_draft(base, {"mode": "consolidated"})
    assert draft["buyer"]["tin"] == mi.GENERAL_PUBLIC_TIN and draft["buyer"]["name"] == "General Public"
    assert draft["buyer"]["address"]["state"] == "17"
    assert all(line["classification"] == "004" for line in draft["lines"])
    draft["supplier"].update({"tin": "C2584563222", "msic": "46900", "activity": "Wholesale"})
    assert mi.validate_draft(draft)["ready"]


def test_profiles_fill_gaps_but_user_values_win():
    base = mi.suggest_from_receipt(ntme_receipt())
    profile = {"tin": "C2584563222", "msic": "46900", "activity": "Wholesale", "name": "Old name"}
    merged = mi.merge_draft(base, {"supplier": {"msic": "47520"}, "provenance": {"supplier.msic": "user"}}, ("supplier", profile))
    assert merged["supplier"]["tin"] == "C2584563222" and merged["provenance"]["supplier.tin"] == "profile"
    assert merged["supplier"]["msic"] == "47520"
    assert merged["supplier"]["name"] == "NTME TRADING(M) SDN BHD"      # receipt value is kept
    # a blank saved value does not hide a later profile value unless the user cleared it
    merged = mi.merge_draft(base, {"supplier": {"tin": None}}, ("supplier", profile))
    assert merged["supplier"]["tin"] == "C2584563222"


def test_myinvois_endpoints_save_validate_and_export(tmp_path: Path) -> None:
    from dsdnet_receipt.storage import utc_now
    from dsdnet_receipt.web_app import create_app

    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "receipts.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    response = client.post("/api/register", json={"name": "Test User", "email": "mi@valid-domain.my", "password": "long-test-password-1",
                                                  "password_confirmation": "long-test-password-1", "policy_accepted": True},
                           headers={"X-CSRF-Token": token})
    token, user = response.get_json()["csrf_token"], response.get_json()["user"]
    receipt = ntme_receipt()
    now = utc_now()
    for public_id in ("USER-MI-1", "USER-MI-2"):
        app.extensions["receipt_store"].create_receipt({
            "public_id": public_id, "user_id": user["id"], "source_name": "ntme.png", "created_at": now, "updated_at": now,
            "status": "needs_review", "merchant": receipt["merchant"], "transaction_date": receipt["transaction_date"],
            "receipt_number": receipt["receipt_number"], "total_amount": receipt["total_amount"], "tags": "[]", "notes": "",
            "ocr_text": receipt["ocr_text"], "audit_json": json.dumps({"ocr_lines": NTME_OCR}),
            "input_path": str(tmp_path / "a.png"), "restored_path": str(tmp_path / "b.png"), "runtime_seconds": .1,
            "restoration_status": "ok", "support_status": "unavailable",
            "line_items_json": json.dumps(receipt["line_items"]), "accounting_fields_json": json.dumps(receipt["accounting_fields"]),
        })
    state = client.get("/api/receipts/USER-MI-1/myinvois").get_json()
    assert not state["validation"]["ready"] and state["schema"]["fields"]
    assert client.get("/api/receipts/USER-MI-1/myinvois.json").status_code == 422
    draft = complete(state["draft"])
    saved = client.patch("/api/receipts/USER-MI-1/myinvois", json={"draft": draft, "remember_supplier": True, "remember_buyer": True},
                         headers={"X-CSRF-Token": token})
    assert saved.status_code == 200 and saved.get_json()["validation"]["ready"]
    export = client.get("/api/receipts/USER-MI-1/myinvois.json")
    assert export.status_code == 200 and json.loads(export.data)["Invoice"][0]["ID"][0]["_"] == "CS 2405-171"
    # the second receipt from the same supplier is completed from the saved profiles
    second = client.get("/api/receipts/USER-MI-2/myinvois").get_json()
    assert second["draft"]["supplier"]["tin"] == "C2584563222"
    assert second["draft"]["buyer"]["tin"] == "C1234567890"
    assert second["validation"]["ready"]
    assert client.patch("/api/receipts/USER-MI-1/myinvois", json={"draft": "x"}, headers={"X-CSRF-Token": token}).status_code == 400
    # buyer side: an ordinary purchase needs nothing typed in
    assert state["purchase"]["status"] == "keep_receipt" and state["purchase"]["essentials"]["total"] == "40.00"
    # one workbook for the accountant covering every receipt
    from openpyxl import load_workbook
    import io
    pack = client.get("/api/accountant-pack.xlsx")
    assert pack.status_code == 200
    book = load_workbook(io.BytesIO(pack.data))
    rows = list(book["Expenses"].iter_rows(min_row=2, max_row=3, values_only=True))
    assert len(rows) == 2 and rows[0][1].startswith("NTME") and rows[0][3] == 40.0
    assert rows[0][7] == "Keep as supporting document"
    # zip: workbook + recovered and original image per receipt
    import zipfile
    (tmp_path / "a.png").write_bytes(b"png"); (tmp_path / "b.png").write_bytes(b"png")
    archive = zipfile.ZipFile(io.BytesIO(client.get("/api/accountant-pack.zip").data))
    names = archive.namelist()
    assert "accountant-pack.xlsx" in names
    assert sum(name.startswith("receipts/") for name in names) >= 2
    assert client.get("/api/accountant-pack.zip?month=2030-01").status_code == 200
    assert client.get("/api/accountant-pack.zip?from=bad").status_code == 400
    assert book["By supplier"].cell(2, 3).value == 2


def test_purchase_assessment_buyer_side():
    request_text = ("REQUEST FOR E-INVOICE ARE PERMITTED ONLY UP TO 3rd DAY OF THE FOLLOWING MONTH "
                    "AND MUST BE MADE AT THE ORIGINAL STORE OF PURCHASE")
    small = {"merchant": "MR D.I.Y (KUCHAI) SDN BHD", "transaction_date": "2025-01-01", "total_amount": "2.60", "ocr_text": request_text}
    result = mi.assess_purchase_receipt(small, today="2025-01-15")
    assert result["status"] == "keep_receipt"
    assert result["request"]["printed_on_receipt"] and result["request"]["deadline"] == "2025-02-03"
    assert result["request"]["possible"] and not result["needs_review"]
    assert mi.assess_purchase_receipt(small, today="2025-03-01")["request"]["expired"]
    # December purchase rolls into January
    december = dict(small, transaction_date="2024-12-20")
    assert mi.assess_purchase_receipt(december, today="2024-12-21")["request"]["deadline"] == "2025-01-03"
    # above RM10,000 the supplier cannot consolidate the sale
    large = {"merchant": "Supplier", "transaction_date": "2026-03-02", "total_amount": "12500.00", "ocr_text": "TOTAL 12500.00"}
    assert mi.assess_purchase_receipt(large, today="2026-03-03")["status"] == "supplier_must_issue"
    # a validated e-invoice printed with its MyInvois link
    validated = {"merchant": "Shop", "transaction_date": "2026-03-02", "total_amount": "50.00",
                 "ocr_text": "Validated e-Invoice https://myinvois.hasil.gov.my/ABC UUID 123"}
    assert mi.assess_purchase_receipt(validated)["status"] == "validated_einvoice"
    # unreadable essentials are flagged, not demanded field by field
    blank = {"ocr_text": ""}
    assert set(mi.assess_purchase_receipt(blank)["needs_review"]) == {"supplier name", "date", "total amount"}


def test_second_reading_fills_only_missing_fields():
    import numpy as np
    from types import SimpleNamespace
    from dsdnet_receipt.web_app import second_reading

    class FakeOCR:
        calls = 0
        def run_views(self, views):
            FakeOCR.calls += 1
            text = "Bill No POS/7/2064\nDate 29/02/24 5:36:45 PM\nNet Amt 24.00"
            return SimpleNamespace(text=text, lines=[{"text": t, "box": [[0, 10 * i], [80, 10 * i], [80, 10 * i + 8], [0, 10 * i + 8]], "confidence": .9}
                                                     for i, t in enumerate(text.splitlines())])
    extracted = {"transaction_date": {"value": None}, "total_amount": {"value": "24.00"}, "receipt_invoice_number": {"value": "X1"}}
    record = second_reading(FakeOCR(), extracted, np.full((400, 300), 255, np.uint8), (200, 150))
    assert record["run"] and record["missing_before"] == ["transaction_date"]
    assert extracted["transaction_date"]["value"] == "2024-02-29"
    assert extracted["total_amount"]["value"] == "24.00" and extracted["receipt_invoice_number"]["value"] == "X1"
    complete = {"transaction_date": {"value": "2024-02-29"}, "total_amount": {"value": "1"}, "receipt_invoice_number": {"value": "2"}}
    FakeOCR.calls = 0
    assert not second_reading(FakeOCR(), complete, np.zeros((10, 10), np.uint8), (10, 10))["run"] and FakeOCR.calls == 0


def test_duplicates_category_and_named_dates():
    from dsdnet_receipt.web_app import possible_duplicate, expense_category
    from dsdnet_receipt.fields import extract_fields
    a = {"merchant": "SKISELAYANGBARU DIY ENTERPRISE", "transaction_date": "2024-02-29", "total_amount": "24.00", "receipt_number": None}
    b = {"merchant": "SKLSELAYANG BARUDIY ENTERPRISE", "transaction_date": "2024-02-29", "total_amount": "24.0", "receipt_number": None}
    assert possible_duplicate(a, b)                                   # OCR variants of the same shop
    assert not possible_duplicate(a, dict(b, total_amount="25.00"))
    assert not possible_duplicate(a, dict(b, merchant="PETRONAS PS BATU"))
    assert possible_duplicate({"receipt_number": "POS/7/2064", "total_amount": "24.00"}, {"receipt_number": "POS72064", "total_amount": "24.00"})
    assert possible_duplicate({"content_sha256": "x"}, {"content_sha256": "x"})
    assert expense_category({"merchant": "PETRONAS", "tags": []}) == "Fuel & transport"
    assert expense_category({"merchant": "Zz", "tags": []}) == "Other expense"
    assert extract_fields("DATE : 08 APR 2023", [])["transaction_date"]["value"] == "2023-04-08"
    assert extract_fields("Date 06FEB2023", [])["transaction_date"]["value"] == "2023-02-06"
    assert extract_fields("Date 12/12/2099", [])["transaction_date"]["value"] is None   # not in the future


def test_fuel_receipt_total_and_partial_date():
    from dsdnet_receipt.fields import extract_fields
    fields = extract_fields("PETRONAS\nDATE06FEB\nPnmax95RM10.00\nRM2.050/L\nCrand Total", [])
    assert fields["total_amount"]["value"] == "10.00"          # the only RM amount
    assert fields["transaction_date"]["value"] is None          # year not readable: never guessed
    assert fields["transaction_date"]["hint"] == "06 Feb"
    assert extract_fields("RM 5.00\nRM 7.00", [])["total_amount"]["value"] is None   # ambiguous: left blank


def test_settings_api_and_presets(tmp_path):
    from dsdnet_receipt.web_app import create_app
    config = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    app = create_app(config, testing=True, database_path=tmp_path / "s.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    response = client.post("/api/register", json={"name": "S User", "email": "s@valid-domain.my", "password": "long-test-password-1",
                                                  "password_confirmation": "long-test-password-1", "policy_accepted": True},
                           headers={"X-CSRF-Token": token})
    token = response.get_json()["csrf_token"]
    values = client.get("/api/settings").get_json()["values"]
    assert values["extra_ocr_view"] == "auto" and values["stroke_joining"] is True
    saved = client.patch("/api/settings", json={"preset": "fast", "show_myinvois": False, "bogus": 1, "default_view": "nope"},
                         headers={"X-CSRF-Token": token}).get_json()["values"]
    assert saved["extra_ocr_view"] == "never" and saved["second_reading"] is False and saved["severe_cleanup"] is False
    assert saved["show_myinvois"] is False and saved["default_view"] == "faithful" and "bogus" not in saved
    assert client.get("/api/settings").get_json()["values"]["show_myinvois"] is False


def test_date_strip_reading_completes_year_only_when_read():
    import numpy as np
    from types import SimpleNamespace
    from dsdnet_receipt.web_app import date_year_reading, year_from_strip_text
    assert year_from_strip_text("DATE 06FEB 2O23", "06 Feb") == "2023-02-06"
    assert year_from_strip_text("06 FEB 202", "06 Feb") is None
    assert year_from_strip_text("06 MAR 2023", "06 Feb") is None
    class OCR:
        def __init__(self, text): self.text = text
        def run(self, image): return SimpleNamespace(lines=[{"text": self.text}])
    source = np.full((60, 300), 230, np.uint8)
    box = [[100, 20], [160, 20], [160, 34], [100, 34]]
    extracted = {"transaction_date": {"value": None, "hint": "06 Feb", "box": box}}
    record = date_year_reading(OCR("DATE 06 FEB 2023"), extracted, source)
    assert record["filled"] == "2023-02-06" and extracted["transaction_date"]["value"] == "2023-02-06"
    extracted = {"transaction_date": {"value": None, "hint": "06 Feb", "box": box}}
    assert "filled" not in date_year_reading(OCR("06 FEB 2"), extracted, source)
    assert extracted["transaction_date"]["value"] is None


def test_partly_hidden_year_is_only_suggested_when_one_year_fits():
    from dsdnet_receipt.web_app import _partly_read_year
    readings = ["after_month: 5FED223", "after_month: SFEB22", "after_month_detect: 3FE820"]
    assert _partly_read_year(readings, "06 Feb") == "2023-02-06"
    assert _partly_read_year(readings + ["after_month: FEB 2022"], "06 Feb") is None
    assert _partly_read_year(["after_month: FEB 229"], "06 Feb") is None  # future year
    assert _partly_read_year(["line: FEB 223"], "06 Feb") is None
