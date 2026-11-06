from __future__ import annotations

import numpy as np
import cv2

from dsdnet_receipt.baselines import otsu, perceptual_filter, sauvola
from dsdnet_receipt.degradation import degrade, variant_seed
from dsdnet_receipt.metrics import cer, psnr, ssim, wer
from dsdnet_receipt.fields import extract_fields, infer_receipt_tags, route_fields
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.ocr import OCRResult, PaddleOCRAdapter, deduplicate_lines
from dsdnet_receipt.language_assistance import suggest_receipt_text, suggestions_for_lines
from dsdnet_receipt.restoration import (
    make_evidence_preserving_view,
    select_faithful_display_candidate,
)


def test_seed_is_stable() -> None:
    assert variant_seed(56, "SRC-0001", "V01", "light") == variant_seed(
        56, "SRC-0001", "V01", "light"
    )


def test_degradation_is_reproducible() -> None:
    image = np.full((128, 128), 255, dtype=np.uint8)
    image[30:90, 50:55] = 0
    config = {
        "degradation": {
            "severities": {
                "light": {
                    "blur_sigma": [0.3, 0.6],
                    "contrast": [0.8, 1.0],
                    "noise_sigma": [0.0, 0.02],
                    "foreground_dropout": [0.0, 0.05],
                    "illumination_amplitude": [0.0, 0.1],
                }
            },
            "abrasion_area": [0.0, 0.05],
            "visibility_threshold": 0.05,
        }
    }
    first, first_masks, _ = degrade(image, "light", config, 123)
    second, second_masks, _ = degrade(image, "light", config, 123)
    assert np.array_equal(first, second)
    assert all(np.array_equal(first_masks[key], second_masks[key]) for key in first_masks)


def test_baselines_preserve_shape() -> None:
    image = np.arange(4096, dtype=np.uint8).reshape(64, 64)
    assert otsu(image).shape == image.shape
    assert sauvola(image).shape == image.shape
    assert perceptual_filter(image).shape == image.shape


def test_metrics_identity() -> None:
    image = np.full((32, 32), 127, dtype=np.uint8)
    assert psnr(image, image) == float("inf")
    assert abs(ssim(image, image) - 1.0) < 1e-9
    assert cer("abc", "abc") == 0.0
    assert wer("a b", "a b") == 0.0


def test_target_field_extraction() -> None:
    text = "DOCUMENT NO: TD01167104\nDATE: 25/12/2018\nTOTAL: RM 9.00"
    fields = extract_fields(text, [{"text": text, "confidence": 0.91}])
    assert fields["transaction_date"]["value"] == "2018-12-25"
    assert fields["receipt_invoice_number"]["value"] == "TD01167104"
    assert fields["total_amount"]["value"] == "9.00"


def test_natural_receipt_never_auto_accepts_without_support() -> None:
    extracted = {
        "transaction_date": {"value": "25/12/2018", "confidence": 0.99},
        "receipt_invoice_number": {"value": "TD01167104", "confidence": 0.99},
        "total_amount": {"value": "9.00", "confidence": 0.99},
    }
    decisions = route_fields(extracted, support_available=False, model_available=True)
    assert all(item.decision == "Human decision required" for item in decisions)
    assert all(item.support_status == "unavailable" for item in decisions)


def test_preprocessing_corrects_quarter_turn_receipt() -> None:
    receipt = np.full((600, 300), 245, dtype=np.uint8)
    for y in range(80, 540, 35):
        cv2.line(receipt, (35, y), (265, y), 20, 3)
    sideways = cv2.rotate(receipt, cv2.ROTATE_90_CLOCKWISE)
    processed, record = preprocess(sideways, {"preprocessing": {"max_long_edge": 1600, "boundary_detection": False, "orientation_correction": True}})
    assert record["orientation_correction"]["coarse_rotation_degrees"] == 90
    assert processed.shape[0] > processed.shape[1]


def test_preprocessing_does_not_turn_creased_portrait_receipt_sideways() -> None:
    receipt = np.full((700, 420), 238, dtype=np.uint8)
    for y in range(90, 640, 42):
        cv2.putText(
            receipt, "TOTAL RM 12.00", (35, y),
            cv2.FONT_HERSHEY_SIMPLEX, .62, 55, 1,
        )
    cv2.line(receipt, (10, 80), (410, 610), 150, 8)
    cv2.line(receipt, (15, 500), (405, 470), 165, 6)
    processed, record = preprocess(
        receipt,
        {"preprocessing": {
            "max_long_edge": 1600,
            "boundary_detection": False,
            "orientation_correction": True,
        }},
    )
    assert record["orientation_correction"]["coarse_rotation_degrees"] == 0
    assert processed.shape[0] > processed.shape[1]


def test_preprocessing_uses_parallel_text_rows_not_single_crease_for_deskew() -> None:
    from dsdnet_receipt.preprocessing import _rotate_bound

    receipt = np.full((800, 400), 245, dtype=np.uint8)
    for y in range(100, 720, 50):
        cv2.putText(
            receipt, "TOTAL RM 12.00", (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, .65, 30, 2,
        )
    cv2.line(receipt, (0, 360), (399, 330), 110, 4)  # conflicting fold
    tilted = _rotate_bound(receipt, 4.0)
    corrected, record = preprocess(
        tilted,
        {"preprocessing": {
            "max_long_edge": 1600,
            "boundary_detection": False,
            "orientation_correction": True,
        }},
    )
    orientation = record["orientation_correction"]
    assert orientation["text_row_count"] >= 3
    assert -5.5 <= orientation["deskew_degrees"] <= -2.5
    from dsdnet_receipt.preprocessing import _orient_and_deskew
    _, residual = _orient_and_deskew(corrected)
    assert abs(residual["text_row_skew_degrees"]) < 1.0


def test_spatial_total_does_not_take_number_from_following_notes() -> None:
    lines = [
        {"text": "Total Payable", "confidence": .98, "box": [[100,100],[220,100],[220,120],[100,120]]},
        {"text": "RM", "confidence": .99, "box": [[250,100],[280,100],[280,120],[250,120]]},
        {"text": "102.50", "confidence": .99, "box": [[320,100],[390,100],[390,120],[320,120]]},
        {"text": "1. All cheques must be crossed", "confidence": .95, "box": [[10,140],[250,140],[250,160],[10,160]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["total_amount"]["value"] == "102.50"


def test_spatial_invoice_number_uses_value_to_right_of_label() -> None:
    lines = [
        {"text": "INVOICE", "confidence": .99, "box": [[100,10],[200,10],[200,30],[100,30]]},
        {"text": "NO", "confidence": .99, "box": [[240,10],[280,10],[280,30],[240,30]]},
        {"text": ": CS 2404-211", "confidence": .99, "box": [[320,10],[450,10],[450,30],[320,30]]},
        {"text": "CDO NO", "confidence": .95, "box": [[240,60],[310,60],[310,80],[240,80]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["receipt_invoice_number"]["value"] == "CS2404-211"


def test_field_extraction_prefers_payable_total_and_business_name() -> None:
    lines = [
        {"text": "#FI%ARZ", "confidence": 0.56, "box": [[5, 5], [70, 5], [70, 20], [5, 20]]},
        {"text": "Sin Lianhap Sdn Bhd", "confidence": 0.99, "box": [[80, 25], [280, 25], [280, 45], [80, 45]]},
        {"text": "TOTAL DISCOUNT", "confidence": 0.99, "box": [[400, 400], [520, 400], [520, 420], [400, 420]]},
        {"text": "0.00", "confidence": 1.0, "box": [[600, 400], [650, 400], [650, 420], [600, 420]]},
        {"text": "TOTAL PAYABLE", "confidence": 0.99, "box": [[400, 450], [520, 450], [520, 470], [400, 470]]},
        {"text": "94.50", "confidence": 1.0, "box": [[600, 450], [650, 450], [650, 470], [600, 470]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["merchant"]["value"] == "Sin Lianhap Sdn Bhd"
    assert fields["total_amount"]["value"] == "94.50"


def test_invoice_fallback_rejects_item_codes_and_units() -> None:
    lines = [
        {"text": "CS.H9oS0", "confidence": 0.75, "box": [[500, 30], [600, 30], [600, 50], [500, 50]]},
        {"text": "SP-RNL2435", "confidence": 0.99, "box": [[30, 300], [160, 300], [160, 320], [30, 320]]},
        {"text": "1.00 PCS", "confidence": 1.0, "box": [[500, 300], [580, 300], [580, 320], [500, 320]]},
        {"text": "TOTAL", "confidence": 1.0, "box": [[30, 700], [100, 700], [100, 720], [30, 720]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["receipt_invoice_number"]["value"] == "CS.H9oS0"


def test_invalid_calendar_date_is_not_extracted() -> None:
    fields = extract_fields("DATE 31/02/2025\nTOTAL RM 10", [])
    assert fields["transaction_date"]["value"] is None
    assert fields["total_amount"]["value"] == "10.00"


def test_date_is_extracted_when_thermal_ocr_joins_date_and_time() -> None:
    lines = [{"text": "01-09-2420:56SH02B046", "confidence": .93,
              "box": [[30, 651], [346, 651], [346, 667], [30, 667]]}]
    fields = extract_fields(lines[0]["text"], lines)
    assert fields["transaction_date"]["value"] == "2024-09-01"
    assert fields["transaction_date"]["confidence"] == .93


def test_total_prefers_payable_and_does_not_use_cash_or_change_as_total() -> None:
    text = 'TOTAL DISCOUNT 0.00\nTOTAL PAYABLE RM 13.80\nCASH 20.00\nCHANGE 6.20'
    assert extract_fields(text, [])['total_amount']['value'] == '13.80'
    assert extract_fields('ITEM 13.80\nCASH 20.00\nCHANGE 6.20', [])['total_amount']['value'] is None


def test_invalid_date_does_not_hide_a_later_valid_date() -> None:
    assert extract_fields('DATE 31/02/2025\nDATE 01/09/2024', [])['transaction_date']['value'] == '2024-09-01'


def test_registration_and_payment_method_are_not_receipt_numbers() -> None:
    assert extract_fields('CO.REG NO 750441-W\nPayment No CASH\nTOTAL RM 13.80', [])['receipt_invoice_number']['value'] is None
    assert extract_fields('CO.REG NO 750441-W\nInvoice No POS/197695', [])['receipt_invoice_number']['value'] == 'POS/197695'


def test_explicit_receipt_number_beats_nearby_registration_number() -> None:
    lines = [
        {"text": "No: 61569", "confidence": .99, "box": [[955, 76], [1203, 80], [1202, 122], [955, 118]]},
        {"text": "(206259-X)", "confidence": .99, "box": [[560, 240], [720, 240], [720, 270], [560, 270]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["receipt_invoice_number"]["value"] == "61569"


def test_compact_timestamp_and_roc_number_do_not_hide_invoice_evidence() -> None:
    text = """OJCMARKETING SDN BHD
ROC NO:538358-H
TAX INVOICE
Invoice NoPEGIV-1030765
Date
15/01/201911:0516AM
TOTAL: 193.00"""
    fields = extract_fields(text, [])
    assert fields["transaction_date"]["value"] == "2019-01-15"
    assert fields["receipt_invoice_number"]["value"] == "PEGIV-1030765"


def test_number_label_without_space_is_supported() -> None:
    assert extract_fields("Cash Sale\nNo001624", [])["receipt_invoice_number"]["value"] == "001624"


def test_dot_date_and_labelled_slash_as_one_repair() -> None:
    assert extract_fields("Date: 30.11.2024", [])["transaction_date"]["value"] == "2024-11-30"
    assert extract_fields("Date: 13/7l24", [])["transaction_date"]["value"] == "2024-07-13"
    assert extract_fields("Date: 13/7124", [])["transaction_date"]["value"] == "2024-07-13"
    assert extract_fields("Date: 19/24", [])["transaction_date"]["value"] is None


def test_date_label_and_handwritten_value_can_be_separate_ocr_lines() -> None:
    lines = [
        {"text": "Date:", "confidence": .99, "box": [[869,461],[934,466],[931,496],[867,491]]},
        {"text": "13/7124", "confidence": .72, "box": [[950,445],[1120,445],[1120,500],[950,500]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["transaction_date"]["value"] == "2024-07-13"


def test_tags_are_inferred_from_observed_merchant_and_form_text() -> None:
    tags = infer_receipt_tags(
        "CASH SALE\nPARTICULARS\nQuantity\nUnit Price\nIssued by",
        "CHOP HOCK CHUAN HIN",
    )
    assert tags == ["hardware", "cash-sale", "handwritten-form"]


def test_handwritten_total_below_label_is_spatially_extracted() -> None:
    lines = [
        {"text": "TOTAL RM", "confidence": .98, "box": [[893,1372],[1028,1372],[1028,1395],[893,1395]]},
        {"text": "116", "confidence": .67, "box": [[1109,1327],[1281,1327],[1281,1392],[1109,1392]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["total_amount"]["value"] == "116.00"


def test_complete_multiview_amount_beats_truncated_reading_from_same_box() -> None:
    lines = [
        {"text": "TOTAL RM", "confidence": .98, "box": [[893,1372],[1028,1372],[1028,1395],[893,1395]]},
        {"text": "11", "confidence": .56, "box": [[1109,1327],[1281,1327],[1281,1392],[1109,1392]]},
        {"text": "116", "confidence": .56, "box": [[1109,1327],[1281,1327],[1281,1392],[1109,1392]]},
    ]
    fields = extract_fields("\n".join(line["text"] for line in lines), lines)
    assert fields["total_amount"]["value"] == "116.00"


def test_tax_percentage_is_not_a_total() -> None:
    text = 'TOTAL INCL. GST@6%\nTOTAL RM 37.10\nCASH RM 40.00'
    assert extract_fields(text, [])['total_amount']['value'] == '37.10'


def test_empty_and_unsupported_images_are_rejected() -> None:
    config = {"preprocessing": {"max_long_edge": 1600, "boundary_detection": False, "orientation_correction": False}}
    for image in (np.array([], dtype=np.uint8), np.zeros((2, 2, 2, 2), dtype=np.uint8)):
        try: preprocess(image, config)
        except ValueError: pass
        else: raise AssertionError("Unsupported image should be rejected")


def test_multiview_ocr_merges_shifted_duplicate_but_keeps_repeated_value(monkeypatch) -> None:
    adapter = object.__new__(PaddleOCRAdapter)
    adapter.version = "2.7.0.3"
    outputs = iter([
        OCRResult("44300 HULU YAM LAMA\n130.00", [
            {"text":"44300 HULU YAM LAMA", "confidence":.95, "box":[[356,125],[650,125],[650,145],[356,145]]},
            {"text":"130.00", "confidence":.98, "box":[[1000,648],[1080,648],[1080,668],[1000,668]]},
        ], adapter.version, "ok"),
        OCRResult("44300 HULU YAM LAMA\n130.00", [
            {"text":"44300 HULU YAM LAMA", "confidence":.99, "box":[[356,127],[652,127],[652,147],[356,147]]},
            {"text":"130.00", "confidence":.96, "box":[[1002,1262],[1082,1262],[1082,1282],[1002,1282]]},
        ], adapter.version, "ok"),
    ])
    monkeypatch.setattr(adapter, "run", lambda image: next(outputs))
    result = adapter.run_views([("original", np.zeros((2,2),np.uint8)), ("enhanced", np.ones((2,2),np.uint8))])
    assert [line["text"] for line in result.lines].count("44300 HULU YAM LAMA") == 1
    assert [line["text"] for line in result.lines].count("130.00") == 2
    assert next(line for line in result.lines if line["text"] == "44300 HULU YAM LAMA")["confidence"] == .99


def test_persisted_ocr_deduplication_handles_rounding_boundary() -> None:
    lines = [
        {"text":"44300 HULU YAM LAMA, SELANGOR DARUL EHSAN.","confidence":.952,"box":[[356,125],[900,125],[900,145],[356,145]]},
        {"text":"44300 HULU YAM LAMA, SELANGOR DARUL EHSAN.","confidence":.995,"box":[[356,127],[902,127],[902,147],[356,147]]},
    ]
    result = deduplicate_lines(lines)
    assert len(result) == 1 and result[0]["confidence"] == .995


def test_reading_view_suppresses_fold_but_preserves_unrecognised_ink() -> None:
    readable = np.full((80, 160), 255, np.uint8)
    readable[9:13, 10:150] = 180   # broad fold outside OCR localisation
    readable[30:34, 20:80] = 105   # unrecognised handwriting/signature
    readable[50:54, 25:130] = 165  # faint OCR-localised printed text
    lines = [{"text":"TOTAL 102.50", "box":[[20,46],[135,46],[135,58],[20,58]], "confidence":.95}]
    output = make_evidence_preserving_view(readable, lines)
    assert np.count_nonzero(output[9:13, 10:150] < 245) <= 20
    assert np.count_nonzero(output[30:34, 20:80] < 180) >= 220
    assert np.count_nonzero(output[50:54, 25:130] < 220) >= 400


def test_clean_view_keeps_continuous_faint_document_rules() -> None:
    readable = np.full((120, 220), 238, np.uint8)
    readable[30, 15:205] = 190
    readable[20:105, 40] = 188
    output = make_evidence_preserving_view(readable, [])
    assert np.count_nonzero(output[30, 15:205] < 230) >= 180
    assert np.count_nonzero(output[20:105, 40] < 230) >= 75


def test_clean_view_suppresses_texture_while_retaining_faint_text() -> None:
    yy, xx = np.indices((180, 320))
    texture = 232 + 7 * np.sin(xx / 8.0) + 6 * np.cos(yy / 11.0)
    readable = np.clip(texture, 0, 255).astype(np.uint8)
    cv2.putText(readable, "TOTAL 102.50", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 155, 1, cv2.LINE_AA)
    lines = [{"text": "TOTAL 102.50", "box": [[30, 60], [205, 60], [205, 92], [30, 92]]}]
    output = make_evidence_preserving_view(readable, lines)
    text_region = output[58:95, 28:210]
    background = output[110:165, 25:295]
    assert np.count_nonzero(text_region < 235) >= 100
    assert np.count_nonzero(background < 235) / background.size < 0.02


def test_adaptive_view_preserves_faint_thermal_text_without_binarising() -> None:
    from dsdnet_receipt.restoration import make_adaptive_document_view
    receipt = np.full((180, 320), 247, np.uint8)
    cv2.putText(receipt, "Bill No Date Cashier", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, 222, 1, cv2.LINE_AA)
    cv2.putText(receipt, "HARDWARE 1 10.00", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .55, 218, 1, cv2.LINE_AA)
    cv2.putText(receipt, "Net Amt 12.00", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, .7, 175, 2, cv2.LINE_AA)
    output, mode = make_adaptive_document_view(receipt, [])
    assert mode == "faint_thermal_source_enhanced"
    assert np.count_nonzero(output[35:110] < 245) >= 200
    assert len(np.unique(output)) > 16


def test_faithful_content_view_suppresses_paper_shading_without_drawing_text() -> None:
    from dsdnet_receipt.restoration import make_faithful_content_view

    source = np.full((80, 120), 232, dtype=np.uint8)
    source[:, 54:66] = 205  # broad fold/crease shading
    source[30:34, 18:46] = 80  # measured printed stroke
    output = make_faithful_content_view(source, source)

    assert float(output[:, 54:66].mean()) > float(source[:, 54:66].mean())
    assert float(output[30:34, 18:46].mean()) < 180
    assert output.shape == source.shape


def test_source_stroke_enhancement_never_typesets_missing_text() -> None:
    from dsdnet_receipt.restoration import enhance_supported_source_strokes
    source = np.full((80, 240), 245, np.uint8)
    cv2.line(source, (40, 35), (120, 35), 220, 1)
    base = np.full_like(source, 255)
    lines = [{"text":"A completely different OCR reading", "confidence":.99,
              "box":[[30,20],[210,20],[210,50],[30,50]]}]
    output = enhance_supported_source_strokes(base, source, lines)
    # Only measured source pixels can become dark; blank areas stay blank.
    assert np.count_nonzero(output[34:37, 39:122] < 250) > 20
    assert np.count_nonzero(output[5:15, 20:220] < 250) == 0


def test_source_stroke_enhancement_does_not_bridge_pixel_gaps() -> None:
    from dsdnet_receipt.restoration import enhance_supported_source_strokes
    source = np.full((70, 180), 245, np.uint8)
    # Two measured pieces separated by a one-pixel dropout.
    source[34, 60:85] = 205
    source[34, 86:112] = 205
    base = np.full_like(source, 255)
    lines = [{"text":"supported", "confidence":.99,
              "box":[[50,20],[125,20],[125,50],[50,50]]}]
    output = enhance_supported_source_strokes(base, source, lines)
    assert output[34, 85] >= 250
    # No distant pixels are created merely because OCR supplied a word.
    assert np.count_nonzero(output[5:15, 20:160] < 250) == 0


def test_faithful_view_does_not_repair_missing_source_pixels() -> None:
    from dsdnet_receipt.restoration import make_faithful_content_view

    source = np.full((80, 220), 242, np.uint8)
    source[38, 55:85] = 80
    source[38, 86:118] = 80
    lines = [{"text": "supported", "confidence": .96,
              "box": [[45, 24], [130, 24], [130, 54], [45, 54]]}]
    output = make_faithful_content_view(source, source, lines)
    assert output[38, 85] >= 230


def test_gap_repair_uses_endpoints_and_independent_candidate_support() -> None:
    from dsdnet_receipt.restoration import repair_evidence_constrained_gaps

    source = np.full((80, 220), 242, np.uint8)
    source[38, 55:85] = 80
    source[38, 86:118] = 80
    candidate = source.copy()
    candidate[38, 85] = 85
    base = np.full_like(source, 255)
    lines = [{"text": "", "confidence": .96,
              "box": [[45, 24], [130, 24], [130, 54], [45, 54]]}]
    output = repair_evidence_constrained_gaps(base, source, candidate, lines)
    assert output[38, 85] < 200
    assert np.count_nonzero(output[5:15, 20:160] < 250) == 0


def test_faithfulness_gate_rejects_unsupported_ink_and_returns_source() -> None:
    from dsdnet_receipt.restoration import select_faithful_recovery
    source = np.full((100, 160), 245, np.uint8)
    cv2.putText(source, "TOTAL", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .7, 90, 1)
    candidate = source.copy()
    candidate[60:90, 20:140] = 30
    decision = select_faithful_recovery(source, candidate)
    assert not decision.accepted
    assert np.array_equal(decision.image, source)


def test_faithfulness_gate_accepts_small_source_supported_change() -> None:
    from dsdnet_receipt.restoration import select_faithful_recovery
    source = np.full((100, 160), 245, np.uint8)
    cv2.putText(source, "TOTAL", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .7, 100, 2)
    candidate = source.copy()
    candidate[source < 180] = np.maximum(candidate[source < 180] - 12, 0)
    decision = select_faithful_recovery(source, candidate)
    assert decision.accepted
    assert np.array_equal(decision.image, candidate)


def test_faithful_view_does_not_enhance_unprotected_crease() -> None:
    from dsdnet_receipt.restoration import make_faithful_content_view
    source = np.full((120, 240), 245, np.uint8)
    cv2.putText(source, "TOTAL 13.80", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, .7, 95, 2)
    # A fold shadow is a broad illumination field. A crisp two-pixel line is
    # also a valid printed divider and must not be labelled an artifact solely
    # because OCR did not recognize it.
    crease = np.zeros_like(source)
    cv2.line(crease, (5,20),(230,28),70,2)
    crease = cv2.GaussianBlur(crease,(0,0),3)
    source = np.clip(source.astype(np.int16)-crease,0,255).astype(np.uint8)
    lines = [{"text": "TOTAL 13.80", "confidence": .95,
              "box": [[25, 53], [190, 53], [190, 82], [25, 82]]}]
    output = make_faithful_content_view(source, source, lines)
    assert float(output[18:31].mean()) > 242
    assert np.count_nonzero(output[55:82, 25:195] < 220) > 30


def test_region_adaptive_fusion_keeps_unsupported_candidate_marks_out() -> None:
    from dsdnet_receipt.restoration import region_adaptive_source_fusion
    source = np.full((96, 96), 245, np.uint8)
    cv2.putText(source, "12.00", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, 155, 1)
    candidate = source.copy()
    candidate[source < 220] = 90
    candidate[70:85, 70:85] = 20
    output, record = region_adaptive_source_fusion(source, candidate)
    assert np.array_equal(output, source)
    assert record["selected_tiles"] == 0


def test_region_adaptive_fusion_accepts_supported_contrast_only() -> None:
    from dsdnet_receipt.restoration import region_adaptive_source_fusion
    source = np.full((96, 96), 245, np.uint8)
    cv2.putText(source, "12.00", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, 190, 1)
    candidate = source.copy()
    candidate[source < 220] = 145
    output, record = region_adaptive_source_fusion(source, candidate)
    assert record["selected_tiles"] == 1
    assert np.all(output[source >= 220] == source[source >= 220])
    assert float(output[source < 220].mean()) < float(source[source < 220].mean())


def test_gap_repair_requires_source_endpoints_and_dsdnet_path() -> None:
    from dsdnet_receipt.restoration import repair_evidence_constrained_gaps
    source = np.full((50, 100), 245, np.uint8)
    source[25, 30:45] = 120
    source[25, 47:62] = 120
    candidate = source.copy()
    candidate[25, 45:47] = 150
    line = {"text": "broken", "confidence": .95,
            "box": [[20, 15], [75, 15], [75, 35], [20, 35]]}
    output = repair_evidence_constrained_gaps(np.full_like(source, 255), source, candidate, [line])
    assert np.all(output[25, 45:47] < 250)


def test_gap_repair_refuses_unsupported_missing_pixels() -> None:
    from dsdnet_receipt.restoration import repair_evidence_constrained_gaps
    source = np.full((50, 100), 245, np.uint8)
    source[25, 30:45] = 120
    source[25, 47:62] = 120
    candidate = source.copy()  # candidate contains no evidence in the gap
    line = {"text": "broken", "confidence": .95,
            "box": [[20, 15], [75, 15], [75, 35], [20, 35]]}
    output = repair_evidence_constrained_gaps(np.full_like(source, 255), source, candidate, [line])
    assert np.all(output[25, 45:47] == 255)


def test_line_contrast_normalisation_never_fades_readable_source_text() -> None:
    from dsdnet_receipt.restoration import normalize_localized_source_stroke_contrast
    source = np.full((60, 180), 245, np.uint8)
    cv2.putText(source, "FAINT SOLID", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .55, 210, 1)
    candidate = np.full_like(source, 252)
    line = {"text": "FAINT SOLID", "confidence": .96,
            "box": [[5, 18], [170, 18], [170, 45], [5, 45]]}
    output = normalize_localized_source_stroke_contrast(candidate, source, [line])
    measured = source < 220
    assert np.all(output[measured] <= source[measured])
    assert float(output[measured].mean()) < float(source[measured].mean())


def test_line_contrast_normalisation_does_not_draw_ocr_word() -> None:
    from dsdnet_receipt.restoration import normalize_localized_source_stroke_contrast
    source = np.full((60, 180), 245, np.uint8)
    line = {"text": "TEXT THAT IS NOT PRESENT", "confidence": .99,
            "box": [[5, 18], [170, 18], [170, 45], [5, 45]]}
    output = normalize_localized_source_stroke_contrast(np.full_like(source, 255), source, [line])
    assert np.count_nonzero(output < 245) == 0


def test_line_contrast_normalisation_reduces_ink_tone_variation() -> None:
    from dsdnet_receipt.restoration import normalize_localized_source_stroke_contrast
    source = np.full((36, 100), 245, dtype=np.uint8)
    source[12:25, 22] = 205
    source[12:25, 52] = 150
    line = {"confidence": .95, "box": [[15, 8], [60, 8], [60, 29], [15, 29]]}
    output = normalize_localized_source_stroke_contrast(
        np.full_like(source, 255), source, [line], target_darkness=58.0
    )
    faint_darkness = 245 - float(np.mean(output[14:23, 22]))
    strong_darkness = 245 - float(np.mean(output[14:23, 52]))
    source_gap = (245 - 150) - (245 - 205)
    assert faint_darkness > (245 - 205)
    assert strong_darkness - faint_darkness < source_gap
    assert np.all(output[source == 245] == 255)


def test_source_text_tone_balancing_darkens_faint_glyphs_without_darkening_crease() -> None:
    from dsdnet_receipt.restoration import balance_source_text_tone
    source = np.full((120, 300), 245, np.uint8)
    cv2.putText(source, "FAINT", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, .7, 230, 1)
    cv2.putText(source, "DARK", (150, 48), cv2.FONT_HERSHEY_SIMPLEX, .7, 120, 1)
    cv2.line(source, (5, 88), (295, 95), 205, 2)
    output = balance_source_text_tone(source, source)
    faint = (source < 240) & (source > 200) & (np.indices(source.shape)[0] < 65)
    crease = np.indices(source.shape)[0] > 80
    assert float(np.mean(output[faint])) < float(np.mean(source[faint]))
    assert np.array_equal(output[crease], source[crease])


def test_faint_supported_strokes_survive_cleaning_without_drawing_new_pixels() -> None:
    from dsdnet_receipt.restoration import preserve_faint_source_supported_strokes
    source = np.full((60, 180), 245, np.uint8)
    cv2.putText(source, "FAINT", (12, 39), cv2.FONT_HERSHEY_SIMPLEX, .65, 239, 1)
    candidate = source.copy()
    candidate[source < 244] = 232
    cleaned = np.full_like(source, 252)
    line = {"text": "FAINT", "confidence": .91,
            "box": [[8, 15], [105, 15], [105, 46], [8, 46]]}
    output = preserve_faint_source_supported_strokes(cleaned, source, candidate, [line])
    measured = source < 244
    assert np.count_nonzero(output[measured] < cleaned[measured]) >= int(np.count_nonzero(measured) * .90)
    assert np.count_nonzero(output[(source == 245)] < cleaned[(source == 245)]) == 0


def test_faint_stroke_preservation_rejects_unprotected_crease() -> None:
    from dsdnet_receipt.restoration import preserve_faint_source_supported_strokes
    source = np.full((80, 200), 245, np.uint8)
    cv2.line(source, (5, 64), (195, 70), 225, 2)
    candidate = source.copy()
    cleaned = np.full_like(source, 252)
    line = {"text": "TOTAL", "confidence": .95,
            "box": [[20, 15], [120, 15], [120, 42], [20, 42]]}
    output = preserve_faint_source_supported_strokes(cleaned, source, candidate, [line])
    assert np.array_equal(output[60:74], cleaned[60:74])


def test_source_text_detector_preserves_faint_text_missed_by_ocr() -> None:
    from dsdnet_receipt.restoration import (
        detect_source_text_mask,
        preserve_faint_source_supported_strokes,
    )
    source = np.full((100, 260), 245, np.uint8)
    cv2.putText(source, "DATE 01-09-24", (16, 55), cv2.FONT_HERSHEY_SIMPLEX, .58, 239, 1)
    candidate = np.full_like(source, 245)  # model did not recover the line
    mask = detect_source_text_mask(source)
    assert np.count_nonzero(mask & (source < 244)) > 15
    cleaned = np.full_like(source, 252)
    output = preserve_faint_source_supported_strokes(cleaned, source, candidate, [])
    assert np.count_nonzero(output[source < 244] < cleaned[source < 244]) > 15


def test_source_text_detector_rejects_long_fold_and_isolated_speck() -> None:
    from dsdnet_receipt.restoration import detect_source_text_mask
    source = np.full((120, 280), 245, np.uint8)
    cv2.line(source, (5, 70), (275, 76), 225, 2)
    source[18:20, 40:42] = 200
    mask = detect_source_text_mask(source)
    assert np.count_nonzero(mask) == 0


def test_damaged_candidate_causes_whole_line_source_fallback() -> None:
    from dsdnet_receipt.restoration import apply_line_level_source_fallback
    source = np.full((60, 200), 245, np.uint8)
    cv2.putText(source, "TOTAL 13.80", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .6, 135, 1)
    candidate = np.full_like(source, 245)
    candidate[22:34, 120:150] = 30  # unsupported blotch, source text lost
    display = candidate.copy()
    line = {"text": "TOTAL 13.80", "confidence": .97,
            "box": [[5, 16], [185, 16], [185, 45], [5, 45]]}
    output = apply_line_level_source_fallback(display, source, candidate, [line])
    source_strokes = source < 200
    assert np.count_nonzero(output[source_strokes] < 220) >= int(np.count_nonzero(source_strokes) * .95)
    assert float(output[22:34, 120:150].mean()) > float(candidate[22:34, 120:150].mean())


def test_safe_candidate_line_is_not_replaced() -> None:
    from dsdnet_receipt.restoration import apply_line_level_source_fallback
    source = np.full((60, 200), 245, np.uint8)
    cv2.putText(source, "TOTAL 13.80", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, .6, 170, 1)
    candidate = source.copy()
    candidate[source < 220] = 140
    line = {"text": "TOTAL 13.80", "confidence": .97,
            "box": [[5, 16], [185, 16], [185, 45], [5, 45]]}
    output = apply_line_level_source_fallback(candidate, source, candidate, [line])
    assert np.array_equal(output, candidate)


def test_degradation_router_distinguishes_clear_from_faded_blurred() -> None:
    from dsdnet_receipt.restoration import assess_receipt_degradation
    clear = np.full((300, 200), 245, np.uint8)
    faint = clear.copy()
    for y in range(40, 260, 35):
        cv2.putText(clear, "TOTAL 12.00", (10, y), cv2.FONT_HERSHEY_SIMPLEX, .5, 40, 1)
        cv2.putText(faint, "TOTAL 12.00", (10, y), cv2.FONT_HERSHEY_SIMPLEX, .5, 220, 1)
    faint = cv2.GaussianBlur(faint, (7, 7), 2)
    clear_profile = assess_receipt_degradation(clear)
    faint_profile = assess_receipt_degradation(faint)
    assert "faded" not in clear_profile["flags"]
    assert "blurred" not in clear_profile["flags"]
    assert {"faded", "blurred"}.issubset(faint_profile["flags"])
    assert faint_profile["policy"]["target_darkness"] > clear_profile["policy"]["target_darkness"]
    assert faint_profile["policy"]["sharpen_strength"] > 0
    assert faint_profile["policy"]["paper_cleaning"] == "gentle"
    assert faint_profile["policy"]["text_protection"] == "wide"


def test_gentle_paper_cleaning_retains_more_faint_source_tone_than_strong() -> None:
    from dsdnet_receipt.restoration import suppress_unprotected_paper_texture
    source = np.full((80, 180), 235, np.uint8)
    source[35:37, 20:160] = 225
    image = source.copy()
    gentle = suppress_unprotected_paper_texture(image, source, [], cleaning="gentle")
    strong = suppress_unprotected_paper_texture(image, source, [], cleaning="strong")
    assert float(gentle[35:37, 20:160].mean()) < float(strong[35:37, 20:160].mean())


def test_ocr_rectangle_does_not_protect_crossing_crease_as_text() -> None:
    from dsdnet_receipt.restoration import suppress_unprotected_paper_texture
    source = np.full((100, 260), 245, np.uint8)
    cv2.putText(source, "TOTAL 12.00", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, .6, 120, 1)
    cv2.line(source, (5, 70), (255, 75), 210, 2)
    line = {"confidence": .96, "box": [[2, 30], [258, 30], [258, 82], [2, 82]]}
    output = suppress_unprotected_paper_texture(source, source, [line], cleaning="strong")
    # The long fold crosses the OCR rectangle but is still cleaned.
    assert float(output[72:75, 5:25].mean()) > float(source[72:75, 5:25].mean())
    # Measured character pixels remain protected.
    text = source < 180
    assert np.count_nonzero(output[text] <= source[text]) >= int(np.count_nonzero(text) * .85)


def test_paper_cleaning_is_stronger_in_locally_degraded_corner() -> None:
    from dsdnet_receipt.restoration import suppress_unprotected_paper_texture
    source = np.full((160, 240), 245, np.uint8)
    # Broad fold/shadow confined to the lower-right corner.
    source[95:150, 145:225] = 218
    source = cv2.GaussianBlur(source, (0, 0), 4)
    output = suppress_unprotected_paper_texture(source, source, [], cleaning="gentle")
    clean_change = float(np.mean(output[10:70, 10:90].astype(float) - source[10:70, 10:90]))
    damaged_change = float(np.mean(output[105:140, 160:215].astype(float) - source[105:140, 160:215]))
    assert damaged_change > clean_change + 5.0


def test_wide_source_text_protection_finds_blurred_faint_characters() -> None:
    from dsdnet_receipt.restoration import detect_source_text_mask
    source = np.full((100, 260), 245, np.uint8)
    cv2.putText(source, "ADDRESS LINE", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, .55, 239, 1)
    source = cv2.GaussianBlur(source, (5, 5), 1.2)
    standard = detect_source_text_mask(source, protection="standard")
    wide = detect_source_text_mask(source, protection="wide")
    assert np.count_nonzero(wide) >= np.count_nonzero(standard)
    assert np.count_nonzero(wide) > 0


def test_structural_coherence_fallback_returns_source_after_text_deletion() -> None:
    from dsdnet_receipt.restoration import apply_structural_coherence_fallback
    source = np.full((140, 300), 245, np.uint8)
    for y in (35, 62, 89, 116):
        cv2.putText(source, "TOTAL RM 12.00", (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, 150, 1)
    destroyed = np.full_like(source, 255)
    output = apply_structural_coherence_fallback(destroyed, source)
    assert np.array_equal(output, source)


def test_structural_coherence_gate_keeps_coherent_enhancement() -> None:
    from dsdnet_receipt.restoration import apply_structural_coherence_fallback
    source = np.full((140, 300), 245, np.uint8)
    for y in (35, 62, 89, 116):
        cv2.putText(source, "TOTAL RM 12.00", (12, y), cv2.FONT_HERSHEY_SIMPLEX, .55, 180, 1)
    enhanced = source.copy()
    enhanced[source < 230] = 120
    output = apply_structural_coherence_fallback(enhanced, source)
    assert np.array_equal(output, enhanced)


def test_local_quality_fallback_repairs_only_damaged_text_region() -> None:
    from dsdnet_receipt.restoration import apply_local_quality_fallback
    source = np.full((180, 320), 245, np.uint8)
    cv2.putText(source, "GOOD TOTAL 12.00", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, .65, 130, 1)
    cv2.putText(source, "FADED DATE 01-09", (15, 145), cv2.FONT_HERSHEY_SIMPLEX, .65, 190, 1)
    candidate = source.copy()
    candidate[source < 180] = 90  # valid stronger recovery in the upper tile
    candidate[115:160] = 255      # lower line was destroyed
    output = apply_local_quality_fallback(candidate, source, tile_size=96, overlap=32)
    lower_text = (source < 230) & (np.indices(source.shape)[0] > 110)
    upper_text = (source < 180) & (np.indices(source.shape)[0] < 80)
    assert np.count_nonzero(output[lower_text] < 245) > np.count_nonzero(candidate[lower_text] < 245)
    assert float(np.mean(output[upper_text])) < float(np.mean(source[upper_text]))


def test_local_quality_fallback_does_not_create_tile_seams_on_clean_input() -> None:
    from dsdnet_receipt.restoration import apply_local_quality_fallback
    source = np.full((190, 330), 245, np.uint8)
    for y in (40, 90, 140):
        cv2.putText(source, "TOTAL RM 12.00", (15, y), cv2.FONT_HERSHEY_SIMPLEX, .6, 150, 1)
    output = apply_local_quality_fallback(source, source, tile_size=96, overlap=32)
    assert np.array_equal(output, source)


def test_barcode_region_is_restored_from_original_source() -> None:
    from dsdnet_receipt.restoration import detect_machine_readable_mask, protect_machine_readable_regions
    source = np.full((120, 260), 245, np.uint8)
    for x in range(55, 205, 5):
        source[35:85, x:x + (2 if x % 10 else 3)] = 35
    cleaned = np.full_like(source, 255)
    mask = detect_machine_readable_mask(source)
    assert np.count_nonzero(mask[30:90, 45:215]) > 500
    output = protect_machine_readable_regions(cleaned, source)
    bars = source < 100
    assert np.array_equal(output[bars], source[bars])
    assert np.all(output[mask & ~bars] == 255)


def test_ordinary_text_line_is_not_detected_as_barcode() -> None:
    from dsdnet_receipt.restoration import detect_machine_readable_mask
    source = np.full((100, 320), 245, np.uint8)
    cv2.putText(source, "TOTAL RM 123.45", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .8, 40, 2)
    assert np.count_nonzero(detect_machine_readable_mask(source)) == 0


def test_stroke_fidelity_metrics_separate_deletion_from_false_ink() -> None:
    from dsdnet_receipt.validation import stroke_fidelity_metrics

    clean = np.full((20, 30), 255, np.uint8)
    clean[5:10, 5:15] = 80
    foreground = clean < 128
    recovered = clean.copy()
    recovered[5:10, 5:10] = 250  # deleted half of a known stroke
    recovered[14:17, 20:24] = 100  # invented mark on clean paper
    metrics = stroke_fidelity_metrics(clean, recovered, foreground)

    assert 0.45 <= metrics["stroke_recall"] <= 0.55
    assert 0.45 <= metrics["deleted_stroke_rate"] <= 0.55
    assert metrics["false_stroke_pixels"] == 12


def test_thermal_overlay_renders_only_supported_ocr_text() -> None:
    from dsdnet_receipt.restoration import overlay_supported_ocr_text
    image = np.full((100, 260), 250, np.uint8)
    lines = [
        {"text":"Payment Cash", "confidence":.97, "box":[[20,20],[170,20],[170,42],[20,42]]},
        {"text":"uncertain", "confidence":.42, "box":[[20,60],[170,60],[170,82],[20,82]]},
    ]
    output, count = overlay_supported_ocr_text(image, lines)
    assert count == 1
    assert np.count_nonzero(output[18:45] < 100) > 20
    assert np.count_nonzero(output[58:85] < 245) == 0


def test_thermal_overlay_deduplicates_same_region_with_different_readings() -> None:
    from dsdnet_receipt.restoration import overlay_supported_ocr_text
    image = np.full((100, 300), 250, np.uint8)
    lines = [
        {"text":"Gross Amt", "confidence":.96, "box":[[80,20],[190,20],[190,42],[80,42]]},
        {"text":"Gross Arnt", "confidence":.91, "box":[[84,21],[184,21],[184,43],[84,43]]},
        {"text":"12.00", "confidence":.98, "box":[[220,20],[280,20],[280,42],[220,42]]},
    ]
    _, count = overlay_supported_ocr_text(image, lines)
    assert count == 2


def test_thermal_overlay_preserves_detected_box_centre() -> None:
    from dsdnet_receipt.restoration import overlay_supported_ocr_text
    image = np.full((90, 320), 250, np.uint8)
    box = [[100,25],[260,25],[260,55],[100,55]]
    output, count = overlay_supported_ocr_text(
        image, [{"text":"Cash", "confidence":.99, "box":box}]
    )
    ys, xs = np.where(output < 100)
    assert count == 1
    assert abs(float(xs.mean()) - 180.0) < 12.0
    assert abs(float(ys.mean()) - 40.0) < 8.0


def test_faithful_text_strengthening_uses_evidence_not_ocr_characters() -> None:
    from dsdnet_receipt.restoration import regularize_continuous_supported_text

    source = np.full((80, 180), 245, np.uint8)
    source[34:49, 40:43] = 237
    source[34:49, 55:58] = 237
    source[40:43, 43:49] = 237
    source[40:43, 49:51] = 242  # a real but very faint two-pixel ink trace
    source[40:43, 51:55] = 237
    candidate = source.copy()
    candidate[34:49, 40:43] = 222
    candidate[34:49, 55:58] = 222
    candidate[40:43, 40:58] = 222
    display = np.full_like(source, 250)
    geometry = [[24, 30], [100, 30], [100, 48], [24, 48]]
    first = regularize_continuous_supported_text(
        display, source, candidate,
        [{"text": "CMR6A PLUG", "confidence": .96, "box": geometry}],
        target_darkness=180.0,
    )
    second = regularize_continuous_supported_text(
        display, source, candidate,
        [{"text": "DIFFERENT OCR", "confidence": .96, "box": geometry}],
        target_darkness=180.0,
    )
    assert np.array_equal(first, second)
    measured = np.zeros_like(source, dtype=bool)
    measured[34:49, 40:43] = True
    measured[34:49, 55:58] = True
    measured[40:43, 43:55] = True
    assert float(first[measured].mean()) < 180.0
    assert float(first[40:43, 49:51].mean()) < 205.0
    assert np.all(first[5:15, 5:20] == 250)


def test_faithful_text_strengthening_does_not_draw_on_blank_source() -> None:
    from dsdnet_receipt.restoration import regularize_continuous_supported_text

    blank = np.full((60, 160), 248, np.uint8)
    output = regularize_continuous_supported_text(
        blank,
        blank,
        blank,
        [{"text": "OCR HALLUCINATION", "confidence": 1.0,
          "box": [[10, 15], [150, 15], [150, 40], [10, 40]]}],
        target_darkness=190.0,
    )
    assert np.array_equal(output, blank)


def test_evidence_support_rewards_agreement_and_stability() -> None:
    import torch
    from dsdnet_receipt.evidence_support import SupportScales, support_components

    observed = torch.full((1, 1, 64, 64), 0.9)
    observed[:, :, 28:36, 12:52] = 0.2
    restored = observed.clone()
    stable = restored.clone()
    unstable = restored.clone()
    unstable[:, :, 28:36, 12:52] = 0.8
    quiet = torch.zeros((1, 1, 64, 64))
    noisy = quiet.clone()
    noisy[:, :, 28:36, 12:52] = 1.0

    def identity_blur(value, kernel, padding):
        return value

    maps = support_components(
        observed,
        torch.ones((1, 1, 1)),
        {
            "restored": restored,
            "stage_states": [unstable, stable],
            "diagnostic_summaries": [
                {"prior_disagreement": noisy, "data_disagreement": noisy}
            ],
        },
        identity_blur,
        "replicate",
        SupportScales(residual=.1, stage=.1, uncertainty=.1),
    )
    assert float(maps["combined"][0, 0, 0, 0]) < .05
    assert float(maps["combined"][0, 0, 31, 20]) > .3
    assert all(value.shape == observed.shape for value in maps.values())


def test_receipt_language_assistance_is_conservative_and_auditable() -> None:
    suggestion = suggest_receipt_text("Total PayabIE")
    assert suggestion is not None
    assert suggestion["suggested"] == "Total Payable"
    assert suggestion["source_support"] is False
    assert suggestion["status"] == "unconfirmed_language_suggestion"
    assert suggest_receipt_text("NTME TRADING SDN BHD") is None
    assert suggest_receipt_text("YASEI S-P500 RECOIL ST") is None
    lines = [{"text": "INVOlCE"}, {"text": "75.00"}]
    assert suggestions_for_lines(lines)[0]["line_index"] == 0


def test_faithful_selector_keeps_clean_source_supported_candidate() -> None:
    source = np.full((360, 260), 238, np.uint8)
    candidate = np.full_like(source, 248)
    for y in range(45, 320, 34):
        cv2.putText(source, "TOTAL 12.00", (24, y), cv2.FONT_HERSHEY_SIMPLEX, .55, 75, 1)
        cv2.putText(candidate, "TOTAL 12.00", (24, y), cv2.FONT_HERSHEY_SIMPLEX, .55, 45, 1)
    # A broad fold exists only in the source and should not force fallback.
    cv2.line(source, (0, 170), (259, 205), 175, 9)
    selected = select_faithful_display_candidate(source, candidate, [])
    # Neutral paper may now be lifted, but clean ink must stay unchanged and
    # no fold from the source may be introduced as darker pixels.
    assert np.array_equal(selected[candidate < 200], candidate[candidate < 200])
    assert np.all(selected >= candidate)


def test_faithful_selector_falls_back_when_most_text_is_deleted() -> None:
    source = np.full((360, 260), 238, np.uint8)
    for y in range(45, 320, 34):
        cv2.putText(source, "TOTAL 12.00", (24, y), cv2.FONT_HERSHEY_SIMPLEX, .55, 70, 1)
    candidate = np.full_like(source, 248)
    cv2.putText(candidate, "TOTAL", (24, 45), cv2.FONT_HERSHEY_SIMPLEX, .55, 45, 1)
    selected = select_faithful_display_candidate(source, candidate, [])
    # Source-normalized fallback must recover evidence from lower text rows.
    assert np.count_nonzero(selected[200:330] < 238) > 30
