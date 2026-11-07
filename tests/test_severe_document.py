import cv2
import numpy as np

from dsdnet_receipt.restoration import make_faithful_content_view
from dsdnet_receipt.severe_document import make_severe_document_view, needs_severe_document_recovery


def _faded_creased_invoice(seed: int = 4):
    """Faint grey print, a diagonal fold, a soft fold shadow and speckle."""
    rng = np.random.default_rng(seed)
    height, width = 900, 1100
    clean = np.full((height, width), 255, np.uint8)
    for row in range(10):
        y = 120 + row * 55
        cv2.putText(clean, f"INV {row} CS 2405-171 TOTAL 40.00", (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    cv2.line(clean, (40, 700), (1060, 700), 0, 2)
    text_mask = clean < 128
    paper = np.full((height, width), 196.0, np.float32)
    paper += cv2.GaussianBlur(rng.normal(0, 14, (height, width)).astype(np.float32), (0, 0), 12)
    image = paper - (255.0 - clean) * 0.28            # faded ink: ~70 levels of contrast
    fold = np.zeros((height, width), np.uint8)
    cv2.line(fold, (0, 60), (width, 860), 255, 2)
    image[fold > 0] -= 24                             # thin fold edge
    shadow = cv2.GaussianBlur((fold > 0).astype(np.float32), (0, 0), 18) * 900
    image -= shadow                                   # broad fold shading
    speckle = np.zeros((height, width), bool)
    ys = rng.integers(760, 880, 60)
    xs = rng.integers(20, 1080, 60)
    speckle[ys, xs] = True
    speckle = cv2.dilate(speckle.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    image[speckle] -= 45
    image += rng.normal(0, 3, (height, width))
    return np.clip(image, 0, 255).astype(np.uint8), text_mask, fold > 0, speckle


def test_severe_view_darkens_faint_print_and_returns_noise_to_paper():
    source, text, fold, speckle = _faded_creased_invoice()
    view = make_severe_document_view(source)
    assert view.shape == source.shape and view.dtype == np.uint8
    # 1. faint print becomes dark ink (source median ink ~125)
    assert float(np.median(source[text])) > 110
    assert float(np.median(view[text])) < 70
    # 2. speckle in an empty paper band is removed
    assert float(np.mean(view[speckle] > 200)) > 0.9
    # 3. fold away from text is not promoted to ink
    away = fold & (cv2.dilate(text.astype(np.uint8), np.ones((25, 25), np.uint8)) == 0)
    assert float(np.mean(view[away] < 128)) < 0.15
    # 4. paper is clean white
    paper = ~cv2.dilate((text | fold | speckle).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert float(np.mean(view[paper] > 240)) > 0.98


def test_severe_view_keeps_glyph_shapes_intact():
    source, text, _, _ = _faded_creased_invoice()
    view = make_severe_document_view(source)
    recovered = view < 128
    # Most true stroke pixels are recovered and few new ink pixels appear.
    recall = float(np.mean(recovered[text]))
    halo = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    spurious = float(np.mean(recovered[~halo]))
    assert recall > 0.85
    assert spurious < 0.002


def test_severe_view_is_selected_only_for_faded_broad_documents():
    source, _, _, _ = _faded_creased_invoice()
    assert needs_severe_document_recovery(source, {"flags": ["faded", "creased"]})
    assert not needs_severe_document_recovery(source, {"flags": ["creased"]})
    lines = [{"box": [[60, 95], [700, 95], [700, 130], [60, 130]], "confidence": 0.9, "text": "x"}]
    record = {}
    faithful = make_faithful_content_view(
        source, source, lines, None, 0.95,
        {"flags": ["faded", "creased"], "metrics": {}, "policy": {}},
        refinement_audit=record,
    )
    assert record.get("severe_document_view", {}).get("method") == "source_anchored_severe_document_view_v1"
    severe = make_severe_document_view(source)
    # the finishing stage returns speckle/fold marks to paper; the only ink it
    # adds is stroke joining along faint source traces, which stays small
    assert record.get("degradation_finishing", {}).get("applied") is True
    assert float(np.mean(faithful < severe.astype(int) - 40)) < 0.01


def _show_through_receipt():
    import cv2
    page = np.full((900, 520), 228, np.uint8)
    ghost = np.full_like(page, 255)
    cv2.putText(ghost, "SHOP ONLINE", (20, 260), cv2.FONT_HERSHEY_DUPLEX, 2.0, 0, 9)
    ghost = cv2.GaussianBlur(cv2.flip(ghost, 1), (0, 0), 3.0)
    page = np.minimum(page, (150 + ghost.astype(np.float32) * (78 / 255)).astype(np.uint8))   # reverse side
    lines = []
    for row in range(12):
        y = 120 + row * 55
        text = f"ITEM {row:02d}  1 X 2.60   2.60"
        cv2.putText(page, text, (30, y), cv2.FONT_HERSHEY_PLAIN, 1.6, 70, 2)
        lines.append({"box": [[26, y - 22], [470, y - 22], [470, y + 6], [26, y + 6]], "confidence": 0.9, "text": text})
    cv2.line(page, (0, 430), (519, 470), 175, 2)                                                  # crease
    rng = np.random.default_rng(3)
    page = np.clip(page.astype(np.float32) + rng.normal(0, 3, page.shape), 0, 255).astype(np.uint8)
    return page, lines


def test_severe_receipt_view_drops_show_through_and_keeps_print():
    from dsdnet_receipt.receipt_cleanup import make_severe_receipt_view
    page, lines = _show_through_receipt()
    view, record = make_severe_receipt_view(page, lines, return_record=True)
    scale = view.shape[0] / page.shape[0]
    import cv2
    view = cv2.resize(view, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_AREA)
    text_band = np.zeros(page.shape, bool)
    for line in lines:
        (x0, y0), (x1, y1) = line["box"][0], line["box"][2]
        text_band[y0:y1, x0:x1] = True
    printed = (page < 120) & text_band
    assert (view[printed] < 150).mean() > 0.8                     # the print survives
    ghost_only = ~cv2.dilate(text_band.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    ghost_only[180:300, :] &= True
    assert (view[180:300][ghost_only[180:300]] < 150).mean() < 0.01   # the mirror image is gone
    assert record["text_lines"] == len(lines) and scale >= 1


def test_finishing_selects_severe_view_only_when_it_clears_clutter():
    from dsdnet_receipt.receipt_cleanup import finish_faithful_view
    page, lines = _show_through_receipt()
    cluttered = np.where(page < 205, 0, 255).astype(np.uint8)       # a view that kept the show-through
    record = {}
    finish_faithful_view(cluttered, page, lines, {"flags": ["creased"]}, record=record)
    severe = record["degradation_finishing"]["severe_receipt_view"]
    assert severe["selected"] and severe["off_text_ink_ratio"] <= 0.5
    clean = np.where(page < 120, 0, 255).astype(np.uint8)           # already clean: keep it
    record = {}
    out = finish_faithful_view(clean, page, lines, {"flags": ["creased"]}, record=record)
    assert not record["degradation_finishing"]["severe_receipt_view"]["selected"]
    assert np.array_equal(out, clean)


def test_stroke_joining_only_follows_faint_source_ink():
    import cv2
    from dsdnet_receipt.receipt_cleanup import join_broken_strokes
    source = np.full((120, 400), 235, np.uint8)
    for x in range(20, 380, 30):
        cv2.putText(source, "H", (x, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 120, 2)
    view = np.where(source < 200, 0, 255).astype(np.uint8)
    broken = view.copy(); broken[52:56, :] = 255            # a horizontal break through every letter
    joined = join_broken_strokes(broken, source, [])
    # the break is filled where the source still has ink, never on bare paper
    assert np.mean(joined[52:56][source[52:56] < 200] < 200) > 0.7
    assert np.all(joined[source >= 230] >= 200)
