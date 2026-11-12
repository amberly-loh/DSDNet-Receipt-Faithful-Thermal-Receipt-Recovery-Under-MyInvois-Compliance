"""RQ1 revision (9 Nov 2026): choices made on validation receipts only."""
import numpy as np

from dsdnet_receipt.ocr import OCRResult, PaddleOCRAdapter, deduplicate_lines
from dsdnet_receipt.fields import extract_fields
from dsdnet_receipt.receipt_context import reading_copy_lines

BOX_A = [[80, 25], [380, 25], [380, 45], [80, 45]]
BOX_B = [[82, 26], [382, 26], [382, 46], [82, 46]]


def _adapter(monkeypatch, *results):
    adapter = object.__new__(PaddleOCRAdapter)
    adapter.version = "2.7.0.3"
    outputs = iter(results)
    monkeypatch.setattr(adapter, "run", lambda image: next(outputs))
    return adapter


def test_merge_keeps_the_spaced_reading_of_the_same_letters(monkeypatch) -> None:
    joined = {"text": "GARDENIABAKERIES(KL)SDNBHD", "confidence": .98, "box": BOX_A}
    spaced = {"text": "GARDENIA BAKERIES(KL)SDN BHD", "confidence": .91, "box": BOX_B}
    adapter = _adapter(monkeypatch, OCRResult(joined["text"], [joined], "2.7.0.3", "ok"),
                       OCRResult(spaced["text"], [spaced], "2.7.0.3", "ok"))
    result = adapter.run_views([("a", np.zeros((2, 2), np.uint8)), ("b", np.ones((2, 2), np.uint8))])
    assert [line["text"] for line in result.lines] == ["GARDENIA BAKERIES(KL)SDN BHD"]
    assert extract_fields(result.text, result.lines)["merchant"]["value"] == "GARDENIA BAKERIES(KL)SDN BHD"


def test_merge_still_prefers_confidence_when_spacing_is_equal(monkeypatch) -> None:
    low = {"text": "HON HWA HARDWARE", "confidence": .80, "box": BOX_A}
    high = {"text": "HON HWA HARDWARE", "confidence": .95, "box": BOX_B}
    adapter = _adapter(monkeypatch, OCRResult(low["text"], [low], "2.7.0.3", "ok"),
                       OCRResult(high["text"], [high], "2.7.0.3", "ok"))
    result = adapter.run_views([("a", np.zeros((2, 2), np.uint8)), ("b", np.ones((2, 2), np.uint8))])
    assert len(result.lines) == 1 and result.lines[0]["confidence"] == .95


def test_merge_does_not_join_different_letters(monkeypatch) -> None:
    first = {"text": "Golden Arches Sdn Bhd", "confidence": .95, "box": BOX_A}
    second = {"text": "Golden Arohes Sdn Bhd", "confidence": .97, "box": BOX_B}
    adapter = _adapter(monkeypatch, OCRResult(first["text"], [first], "2.7.0.3", "ok"),
                       OCRResult(second["text"], [second], "2.7.0.3", "ok"))
    result = adapter.run_views([("a", np.zeros((2, 2), np.uint8)), ("b", np.ones((2, 2), np.uint8))])
    assert len(result.lines) == 2


def test_persisted_deduplication_uses_the_same_preference() -> None:
    lines = [{"text": "YONGCEN ENTERPRISE", "confidence": .97, "box": BOX_A},
             {"text": "YONG CEN ENTERPRISE", "confidence": .90, "box": BOX_B}]
    assert [line["text"] for line in deduplicate_lines(lines)] == ["YONG CEN ENTERPRISE"]


def test_reading_copy_keeps_faint_words_the_reader_accepted() -> None:
    lines = [{"text": "Cash", "confidence": .52, "box": [[10, 10], [60, 10], [60, 30], [10, 30]]},
             {"text": "10.00", "confidence": .99, "box": [[200, 10], [260, 10], [260, 30], [200, 30]]},
             {"text": "Change", "confidence": .49, "box": [[10, 50], [80, 50], [80, 70], [10, 70]]}]
    assert [item["text"] for item in reading_copy_lines(lines, tidy=False)] == ["Cash", "10.00"]
