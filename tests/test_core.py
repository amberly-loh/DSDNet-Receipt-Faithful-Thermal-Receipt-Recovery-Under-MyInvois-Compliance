from __future__ import annotations

import numpy as np
import cv2

from dsdnet_receipt.baselines import otsu, perceptual_filter, sauvola
from dsdnet_receipt.degradation import degrade, variant_seed
from dsdnet_receipt.metrics import cer, psnr, ssim, wer
from dsdnet_receipt.fields import extract_fields, route_fields
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.ocr import OCRResult, PaddleOCRAdapter, deduplicate_lines
from dsdnet_receipt.language_assistance import suggest_receipt_text, suggestions_for_lines
from dsdnet_receipt.restoration import make_evidence_preserving_view


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
    result = adapter.run_views([("original", np.zeros((2,2),np.uint8)), ("clean", np.zeros((2,2),np.uint8))])
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
    assert mode == "faint_thermal_grayscale"
    assert np.count_nonzero(output[35:110] < 245) >= 200
    assert len(np.unique(output)) > 16


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
