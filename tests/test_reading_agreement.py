from dsdnet_receipt.reading_agreement import agree, vote


def fields(merchant=None, date=None, total=None):
    out = {}
    if merchant: out["merchant"] = {"value": merchant, "confidence": 0.9, "box": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    if date: out["transaction_date"] = {"value": date}
    if total: out["total_amount"] = {"value": total}
    return out


def test_majority_overrides_single_primary_reading():
    primary = fields("KOKFONG TOOLS SDN BHD", "2024-11-12", "4.00")
    others = [("A_direct", fields("KOK FONG TOOLS SDN BHD", "2024-11-12", "14.00")),
              ("B_otsu", fields(None, None, "14.00")),
              ("C_perceptual", fields("KOK FONG TOOLS SDN. BHD.", None, "14.00"))]
    chosen, audit = agree(primary, others)
    assert chosen["total_amount"]["value"] == "14.00"
    assert chosen["transaction_date"]["value"] == "2024-11-12"
    # same letters: the reading with the most word gaps is kept
    assert chosen["merchant"]["value"] == "KOK FONG TOOLS SDN. BHD."
    assert audit["fields"]["total_amount"]["chosen_from"] in {"A_direct", "B_otsu", "C_perceptual"}


def test_primary_wins_ties_and_missing_values_are_filled():
    primary = fields("SHOP", None, "10.00")
    others = [("A_direct", fields("SHOP", "2024-01-02", "12.00"))]
    chosen, audit = agree(primary, others)
    assert chosen["total_amount"]["value"] == "10.00"
    assert chosen["transaction_date"]["value"] == "2024-01-02"


def test_no_readings_no_value():
    value, field, record = vote("total_amount", [("fused_views", {}), ("A_direct", {})])
    assert value is None and record["chosen_from"] is None


def test_fast_preset_turns_reading_agreement_off():
    from dsdnet_receipt.settings import PRESETS, defaults
    assert defaults()["reading_agreement"] is True
    assert PRESETS["fast"]["reading_agreement"] is False
    assert PRESETS["balanced"]["reading_agreement"] is True
