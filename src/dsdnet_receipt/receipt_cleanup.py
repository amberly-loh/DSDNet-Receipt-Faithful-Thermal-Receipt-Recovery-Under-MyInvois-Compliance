"""Degradation-aware finishing of the faithful receipt view.

Applied after the existing display-view selection so that every kind of
upload is handled by one final, source-anchored stage:

* tiny or very low-resolution captures (screenshots, distant photos) are
  enlarged and shown as a contrast-normalised grey view, because any
  black/white decision at 6-8 px character height discards the print;
* torn or punched holes (scanner backing showing through) are shown as flat
  grey "missing paper" rather than as black ink;
* outside text lines, speckle, crease webs, fold traces and water-stain
  tide lines are returned to paper, while logos, signatures, halftone
  pictures, table grids, dotted rules and punctuation are kept;
* inside OCR-localised text lines, faint print is normalised to that line's
  own strongest source strokes so pale lines reach the same ink tone.

Text lines are localised from OCR boxes and from rows of character-sized
marks; recognised strings are never rendered, and every ink pixel is backed
by measured source darkness.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_LIFT = 225.0          # ink tone reached by the strongest stroke of a faint line
_SOURCE_ROW = 0.22     # source-row detection threshold relative to strongest print
_FAINT_SW_K = 3.4      # faint view: black-hat width in measured stroke widths
_FAINT_NOISE_P = 85    # faint view: paper-texture percentile treated as zero ink
_TINY_MIN_SIDE = 450   # captures smaller than this get the enlarged faint view
_SEVERE_CLUTTER_MAX = 0.5  # severe view must remove at least half of the off-text marks
_SEPARATE_MIN_COVERAGE = 0.75   # share of text rows the OCR also found
_SEPARATE_MAX_LOST = 5           # character-shaped dark marks the separation may drop
_SEPARATE_MAX_LOST_INK = 0.10    # print-dark ink it may drop, relative to what it keeps
_LOW_RES_CHAR_HEIGHT = 9.0  # print this small (px) is anti-aliased: keep grey levels
_SEVERE_CLUTTER_PRESENT = 0.004  # ...and only when the current view has off-text marks to clear
_SEVERE_TEXT_MIN = 0.5     # ...while keeping at least half of the ink on OCR text lines


# ---------------------------------------------------------------- helpers
def _gray(image: np.ndarray) -> np.ndarray:
    import cv2
    image = image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _noise_sigma(image: np.ndarray) -> float:
    import cv2
    lap = cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)
    return float(np.median(np.abs(lap - np.median(lap)))) * 1.4826 / 2.45


def _character_height(binary: np.ndarray, default: float = 14.0) -> float:
    import cv2
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    heights = [int(s[3]) for s in stats[1:count] if 5 <= s[3] <= 80 and s[4] >= 10 and s[2] <= 4 * s[3]]
    return float(np.median(heights)) if len(heights) >= 8 else default


def _source_evidence(source: np.ndarray, char_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Stroke-scale darkness of the source (black-hat on paper-normalised image)."""
    import cv2
    paper = cv2.morphologyEx(source, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    paper = cv2.medianBlur(paper, 21).astype(np.float32)
    norm = np.clip(source.astype(np.float32) / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (0, 0), 1.0)
    kernel = int(np.clip(int(char_height * 1.2) | 1, 7, 31))
    evidence = cv2.morphologyEx(norm, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))).astype(np.float32)
    return evidence, paper


def _ocr_zone(shape: tuple[int, int], ocr_lines: list[dict[str, Any]] | None, scale: float) -> np.ndarray:
    import cv2
    zone = np.zeros(shape, np.uint8)
    boxes = []
    for line in ocr_lines or []:
        if float(line.get("confidence") or 0) < 0.30:
            continue
        points = np.asarray(line.get("box", []), np.float32)
        if points.shape == (4, 2):
            boxes.append(points * scale)
    if not boxes:
        return zone > 0
    median_height = float(np.median([np.ptp(b[:, 1]) for b in boxes]))
    for points in boxes:
        height = float(np.ptp(points[:, 1]))
        if median_height and height > 3 * median_height:
            continue  # large reverse-side or watermark lettering is not a text line
        centre = points.mean(axis=0)
        grown = points + np.sign(points - centre) * 0.18 * height
        cv2.fillConvexPoly(zone, np.round(grown).astype(np.int32), 1)
    return zone > 0


def _text_rows(ink: np.ndarray, char_height: float) -> np.ndarray:
    """Rows of three or more character-sized marks."""
    import cv2
    ch = char_height
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    charlike = np.zeros(count, bool)
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        charlike[index] = (0.35 * ch <= h <= 1.8 * ch) and w <= 3 * ch and area >= 0.05 * ch * ch
    chars = charlike[labels].astype(np.uint8)
    rows = cv2.morphologyEx(chars, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(1.2 * ch)), 1)))
    count2, labels2, stats2, _ = cv2.connectedComponentsWithStats(rows, 8)
    char_labels = np.where(chars > 0, labels, 0)
    zone = np.zeros(ink.shape, np.uint8)
    for index in range(1, count2):
        x, y, w, h, _ = stats2[index]
        if h > 2.2 * ch or w < 1.2 * ch:
            continue
        members = char_labels[y:y + h, x:x + w][labels2[y:y + h, x:x + w] == index]
        if len(np.unique(members[members > 0])) < 3:
            continue
        pad = int(0.3 * ch)
        cv2.rectangle(zone, (x - pad, y - pad), (x + w + pad, y + h + pad), 1, -1)
    return zone > 0


def detect_holes(source: np.ndarray, paper: np.ndarray, char_height: float,
                 text_zone: np.ndarray | None = None) -> np.ndarray:
    """Torn or punched holes: near-black, solid, thicker than any glyph, off text lines."""
    import cv2
    ch = char_height
    dark = ((source.astype(np.float32) < 0.30 * paper) & (source < 70)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    holes = np.zeros(source.shape, bool)
    # A hole shows the scanner backing, which is far darker than the receipt's
    # own print; a solid printed logo or dot has the print's own tone.
    print_pixels = (source.astype(np.float32) < 0.75 * paper) & (dark == 0)
    print_level = float(np.percentile(source[print_pixels], 5)) if np.count_nonzero(print_pixels) > 50 else 0.0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 2.0 * ch * ch or area / float(w * h) < 0.45 or min(w, h) < 1.5 * ch:
            continue
        component = labels == index
        if float(np.mean(source[component])) >= 0.45 * print_level:
            continue
        if text_zone is not None and float(text_zone[component].mean()) > 0.3:
            continue
        inner = cv2.distanceTransform(component[y:y + h, x:x + w].astype(np.uint8), cv2.DIST_L2, 3)
        if float(inner.max()) < 0.55 * ch:
            continue
        holes |= component
    return cv2.dilate(holes.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


_STAIN_CACHE: dict[tuple, np.ndarray] = {}
_STAIN_WORK_SIDE = 520   # stains are broad: detect them on a reduced copy


def detect_stain_bands(source: np.ndarray) -> np.ndarray:
    """Long, broad, non-straight dark bands: water-stain tide lines and fold shadows.

    Works on a reduced copy (stains are much larger than print) and caches
    per image, so repeated calls in one run cost almost nothing.
    """
    import cv2, hashlib
    source = _gray(source)
    key = (source.shape, hashlib.blake2b(source.tobytes(), digest_size=16).digest())
    cached = _STAIN_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    height, width = source.shape
    factor = max(height, width) / float(_STAIN_WORK_SIDE)
    if factor > 1.25:
        size = (max(1, round(width / factor)), max(1, round(height / factor)))
        small = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        # printed rules are thin, so find them at full resolution
        paper = cv2.medianBlur(cv2.resize(cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), (width, height)), 5).astype(np.float32)
        ruled = _thin_rule_pixels(source.astype(np.float32) < 0.75 * paper)
        ruled = cv2.resize(cv2.dilate(ruled.astype(np.uint8), np.ones((7, 7), np.uint8)), size, interpolation=cv2.INTER_AREA) > 0
        band = _detect_stain_bands_full(small, unit=1.0 / factor, ruled=ruled, full_shape=(height, width))
        band = cv2.resize(band.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    else:
        band = _detect_stain_bands_full(source)
    if len(_STAIN_CACHE) > 16:
        _STAIN_CACHE.clear()
    _STAIN_CACHE[key] = band
    return band.copy()


def _detect_stain_bands_full(source: np.ndarray, unit: float = 1.0, ruled: np.ndarray | None = None,
                             full_shape: tuple[int, int] | None = None) -> np.ndarray:
    """`unit` is the size of one full-resolution pixel in this image (1/factor
    on a reduced copy), so every pixel constant keeps its original meaning."""
    import cv2
    height, width = source.shape
    full_min = min(full_shape) if full_shape else min(height, width)
    kernel = int(max(61 * unit, full_min * unit / 6)) | 1
    paper = cv2.morphologyEx(source, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    paper = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), kernel / 6)
    darkness = np.clip(255 - source.astype(np.float32) / np.maximum(paper, 1) * 255, 0, 255)
    broad = cv2.GaussianBlur(darkness, (0, 0), max(2.0 * unit, full_min * unit / 160))
    threshold = max(float(np.percentile(broad, 96)), float(np.median(broad)) + 12)
    candidate = (broad >= threshold).astype(np.uint8)
    if ruled is None:
        ruled = cv2.dilate(_thin_rule_pixels(source.astype(np.float32) < 0.75 * paper).astype(np.uint8),
                           np.ones((7, 7), np.uint8)) > 0
    # a tide line fades in and out; link pieces that continue one another
    link = max(3, int(max(5 * unit, full_min * unit / 20))) | 1
    linked = cv2.dilate(candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link, link)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(linked, 8)
    band = np.zeros(source.shape, bool)
    longest = max(height, width)
    for index in range(1, count):
        x, y, w, h, _ = stats[index]
        major, minor = max(w, h), max(1, min(w, h))
        if major < 0.3 * longest or minor <= max(6 * unit, 0.03 * major):
            continue
        member = (labels == index) & (candidate > 0)
        fill = float(np.count_nonzero(member)) / float(w * h)
        if fill > 0.5:
            continue  # filled block, e.g. a logo or shaded panel
        if float(np.mean(ruled[member])) > 0.5:
            continue  # table grids and ruled forms are print, not stains
        band |= member
    grow = max(3, int(round(5 * unit)) | 1)
    return cv2.dilate(band.astype(np.uint8), np.ones((grow, grow), np.uint8)) > 0


def _thin_rule_pixels(ink: np.ndarray) -> np.ndarray:
    """Long straight runs that are also thin (printed rules, not broad bands)."""
    import cv2
    height, width = ink.shape
    mask = ink.astype(np.uint8)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 25), 1)))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 25))))
    thick_h = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    thick_v = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    return ((horizontal & ~thick_h) | (vertical & ~thick_v)) > 0


def _rule_pixels(ink: np.ndarray) -> np.ndarray:
    import cv2
    height, width = ink.shape
    mask = ink.astype(np.uint8)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 25), 1)))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 25))))
    return (horizontal | vertical) > 0


# ---------------------------------------------------------------- faint view
def make_faint_document_view(source: np.ndarray, *, target_char_height: float = 20.0,
                             return_record: bool = False):
    """Enlarged, contrast-normalised grey view for tiny or extremely faint captures.

    No binarisation: at 6-8 px character height a black/white decision
    discards the print. The black-hat width follows the measured stroke
    width so bands wider than a stroke (tide lines, fold shadows) drop out.
    """
    import cv2
    src = _gray(source)
    paper0 = cv2.medianBlur(cv2.morphologyEx(src, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), 21).astype(np.float32)
    probe0 = 255.0 - np.clip(src.astype(np.float32) / np.maximum(paper0, 1) * 255, 0, 255)
    char_height = _character_height(probe0 >= np.percentile(probe0, 96), default=8.0)
    scale = float(np.clip(target_char_height / max(char_height, 1.0), 1.0, 4.0))
    big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else src
    close = int(31 * scale) | 1
    paper = cv2.morphologyEx(big, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    paper = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), close / 4)
    norm = np.clip(big.astype(np.float32) / np.maximum(paper, 1) * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(norm, None, h=float(np.clip(1.2 * _noise_sigma(norm), 3, 10)),
                                        templateWindowSize=7, searchWindowSize=21)
    wide = int(target_char_height * 1.3) | 1
    probe = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide, wide))).astype(np.float32)
    strokes = (probe >= np.percentile(probe, 97)).astype(np.uint8)
    distance = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    ridge = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (strokes > 0)
    stroke_width = float(2 * np.median(distance[ridge])) if ridge.any() else 4.0
    kernel = int(np.clip(int(_FAINT_SW_K * stroke_width) | 1, 7, wide))
    evidence = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))).astype(np.float32)
    top = float(np.percentile(evidence, 99.5))
    noise = float(np.percentile(evidence, _FAINT_NOISE_P))
    t = np.clip((evidence - noise) / max(top - noise, 1.0), 0, 1)
    output = np.clip(255 - 235 * np.power(t, 0.75), 0, 255).astype(np.uint8)
    if not return_record:
        return output
    return output, {"method": "enlarged_faint_document_view_v1", "scale": scale,
                    "stroke_width": stroke_width, "blackhat_kernel": kernel}


# ---------------------------------------------------------------- cleanup
def suppress_receipt_artifacts(view: np.ndarray, source: np.ndarray,
                               ocr_lines: list[dict[str, Any]] | None = None, *,
                               lift_faint_lines: bool = True,
                               record: dict[str, Any] | None = None) -> np.ndarray:
    import cv2
    view = _gray(view)
    source = _gray(source)
    scale = view.shape[1] / source.shape[1]
    if view.shape != source.shape:
        source = cv2.resize(source, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_CUBIC)
    height, width = view.shape
    ink = view < 150
    ch = _character_height(ink, default=max(10.0, height / 60))
    evidence, paper = _source_evidence(source, ch)
    ocr_zone = _ocr_zone(view.shape, ocr_lines, scale)
    holes = detect_holes(source, paper, ch, ocr_zone)
    valid = ~holes
    floor = float(np.percentile(evidence[valid], 60)) + 2.0
    strongest = float(np.percentile(evidence[valid], 99.5))
    source_rows = _text_rows((evidence >= max(floor + 3, _SOURCE_ROW * strongest)) & valid, ch)
    zone = ocr_zone | _text_rows(ink & valid, ch) | source_rows
    zone = cv2.dilate(zone.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    in_zone_ink = ink & zone & valid
    reference = float(np.median(evidence[in_zone_ink])) if np.count_nonzero(in_zone_ink) > 50 \
        else float(np.percentile(evidence, 99))
    output = view.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats((ink & valid).astype(np.uint8), 8)
    area = np.maximum(stats[:, 4], 1)
    inside = np.bincount(labels.ravel(), weights=zone.ravel(), minlength=count) / area
    source_dark = np.bincount(labels.ravel(), weights=evidence.ravel(), minlength=count) / area
    rule_fraction = np.bincount(labels.ravel(), weights=_rule_pixels(ink & valid).ravel(), minlength=count) / area
    small = np.zeros(count, bool)
    small[1:] = stats[1:, 4] < 0.3 * ch * ch
    # dense clusters of small dark marks (halftone pictures) and dotted rules stay
    grouped = cv2.dilate(small[labels].astype(np.uint8), np.ones((int(0.6 * ch) | 1,) * 2, np.uint8))
    group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    component_group = np.zeros(count, np.int32)
    for index in np.nonzero(small)[0]:
        x, y, w, h, _ = stats[index]
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == index)
        component_group[index] = group_labels[y + ys[0], x + xs[0]]
    members = np.bincount(component_group[small], minlength=group_count)
    # halftone pictures are 2-D patches; a one-dot-wide column is a scan streak
    patch = np.minimum(group_stats[:, 2], group_stats[:, 3]) >= 1.2 * ch
    clustered = np.zeros(count, bool)
    clustered[small] = (members[component_group[small]] >= 12) & patch[component_group[small]]
    rowed = cv2.morphologyEx(small[labels].astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.9 * ch) | 1, 1)))
    dot_count, dot_labels, dot_stats, _ = cv2.connectedComponentsWithStats(rowed, 8)
    dotted = np.zeros(dot_count, bool)
    dotted[1:] = (dot_stats[1:, 2] >= 5 * ch) & (dot_stats[1:, 3] <= 0.6 * ch)
    on_dotted = np.bincount(labels.ravel(), weights=dotted[dot_labels].ravel(), minlength=count) > 0

    removed = np.zeros(count, bool)
    long_trace = np.zeros(count, bool)
    reasons: dict[str, int] = {}
    longest = max(height, width)

    def drop(index: int, reason: str) -> None:
        removed[index] = True
        reasons[reason] = reasons.get(reason, 0) + 1

    for index in range(1, count):
        x, y, w, h, a = stats[index]
        major, minor = max(w, h), max(1, min(w, h))
        fill = a / float(w * h)
        if minor <= max(5, int(0.012 * major)) and major >= 40 and fill > 0.5:
            continue                                             # straight printed rule
        if major > 0.25 * longest and fill < 0.3 and rule_fraction[index] < 0.25:
            long_trace[index] = True
            drop(index, "long_trace")                            # tide line / fold trace
            continue
        if on_dotted[index]:
            continue
        if inside[index] >= 0.5:
            if a < 0.03 * ch * ch and source_dark[index] < 0.6 * reference:
                drop(index, "speck_in_text_line")
            continue
        punctuation = fill >= 0.4 and major <= 2.5 * minor and source_dark[index] >= 0.6 * reference and a >= 4
        if a < 0.3 * ch * ch and not punctuation and (source_dark[index] < 0.8 * reference or not clustered[index]):
            drop(index, "speckle")
        elif source_dark[index] < 0.45 * reference:
            drop(index, "weak_in_source")
        elif major > 2.5 * ch and fill < 0.15 and rule_fraction[index] < 0.25:
            drop(index, "crease_web")
    removed_pixels = removed[labels]
    output[removed_pixels] = 255
    # a tide line crossing a text line keeps the strongly measured part there
    keep = long_trace[labels] & zone & (evidence >= 0.6 * reference)
    output[keep] = view[keep]
    fringe = (cv2.dilate(removed_pixels.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) \
        & ~removed_pixels & (view >= 150) & ~zone
    output[fringe] = 255

    # small, isolated, source-weak grey flecks on plain paper
    grey = ((view >= 150) & (view < 235) & ~zone).astype(np.uint8)
    grey_count, grey_labels, grey_stats, _ = cv2.connectedComponentsWithStats(grey, 8)
    if grey_count > 1:
        grey_area = np.maximum(grey_stats[:, 4], 1)
        grey_mean = np.bincount(grey_labels.ravel(), weights=evidence.ravel(), minlength=grey_count) / grey_area
        density = cv2.blur((view < 235).astype(np.float32), (int(2 * ch) | 1,) * 2)
        grey_density = np.bincount(grey_labels.ravel(), weights=density.ravel(), minlength=grey_count) / grey_area
        dark_near = cv2.dilate((view < 150).astype(np.uint8), np.ones((int(0.5 * ch) | 1,) * 2, np.uint8))
        grey_near = np.bincount(grey_labels.ravel(), weights=dark_near.ravel().astype(np.float32), minlength=grey_count) > 0
        flecks = (grey_stats[:, 4] < 0.3 * ch * ch) & (grey_mean < 0.45 * reference) & (grey_density < 0.25) & ~grey_near
        flecks[0] = False
        output[flecks[grey_labels]] = 255

    band = detect_stain_bands(source)
    # inside a stain, keep marks that still carry stroke-scale ink
    output[band & ~zone & (output < 245) & (evidence < 0.6 * reference)] = 255

    lifted = 0
    if lift_faint_lines:
        # Only OCR-confirmed text lines are lifted: crease rows found from
        # geometry alone must not be darkened.
        lift_zone = cv2.dilate(ocr_zone.astype(np.uint8), np.ones((3, 3), np.uint8))
        zone_count, zone_labels, zone_stats, _ = cv2.connectedComponentsWithStats(lift_zone, 8)
        for index in range(1, zone_count):
            x, y, w, h, _ = zone_stats[index]
            region = zone_labels[y:y + h, x:x + w] == index
            local = evidence[y:y + h, x:x + w]
            candidates = local[region & (local > floor)]
            if candidates.size < 20:
                continue
            line_top = float(np.percentile(candidates, 95))
            if line_top < floor + 6:
                continue
            low, high = max(floor, 0.32 * line_top), max(floor + 6, 0.75 * line_top)
            t = np.clip((local - low) / (high - low), 0, 1)
            target = 255 - _LIFT * t * t * (3 - 2 * t)
            current = output[y:y + h, x:x + w].astype(np.float32)
            updated = np.where(region & valid[y:y + h, x:x + w], np.minimum(current, target), current)
            lifted += int(np.count_nonzero(updated < current - 30))
            output[y:y + h, x:x + w] = updated.round().astype(np.uint8)
        # undo lifting on marks that are not character-shaped (crease pieces)
        changed = output.astype(np.int16) < view.astype(np.int16) - 30
        if changed.any():
            lc, ll, ls, _ = cv2.connectedComponentsWithStats((output < 150).astype(np.uint8), 8)
            changed_fraction = np.bincount(ll.ravel(), weights=changed.ravel(), minlength=lc) / np.maximum(ls[:, 4], 1)
            revert = np.zeros(lc, bool)
            for index in range(1, lc):
                _, _, w, h, a = ls[index]
                if changed_fraction[index] < 0.3:
                    continue
                charlike = (0.45 * ch <= h <= 1.6 * ch) and w <= 2.5 * ch and w <= 4 * h and a / float(w * h) >= 0.18
                revert[index] = not charlike
            back = revert[ll] & changed
            output[back] = view[back]

    output[holes] = 228   # physically missing paper: flat grey, never ink
    if record is not None:
        record.update({
            "method": "source_anchored_artifact_suppression_v1",
            "character_height": ch, "removed_components": int(removed.sum()), "removal_reasons": reasons,
            "lifted_pixels": lifted, "hole_fraction": float(holes.mean()),
            "stain_fraction": float(band.mean()), "text_zone_fraction": float(zone.mean()),
            "ocr_role": "localisation only; recognised strings are not rendered",
        })
    return output



# ---------------------------------------------------------------- severe receipt view
def _off_paper(source: np.ndarray, char_height: float) -> np.ndarray:
    """Scanner bed or table showing past a torn/folded corner (large, near-black, at the border)."""
    import cv2
    height, width = source.shape
    level = float(np.percentile(source, 75))
    dark = ((source.astype(np.float32) < 0.45 * level) & (source < 100)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    off = np.zeros(source.shape, bool)
    mx, my = max(3, int(0.02 * width)), max(3, int(0.02 * height))
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 0.003 * height * width:
            continue
        if not (x <= mx or y <= my or x + w >= width - mx or y + h >= height - my):
            continue
        component = labels == index
        if cv2.distanceTransform(component[y:y + h, x:x + w].astype(np.uint8), cv2.DIST_L2, 3).max() < 0.8 * char_height:
            continue
        off |= component
    return off


def _line_boxes(ocr_lines: list[dict[str, Any]] | None, scale: float) -> list[np.ndarray]:
    """Grown OCR line boxes; lettering much taller than the body text (reverse-side
    show-through, watermarks) is left out."""
    boxes = []
    for line in ocr_lines or []:
        if float(line.get("confidence") or 0) < 0.30:
            continue
        points = np.asarray(line.get("box", []), np.float32)
        if points.shape == (4, 2):
            boxes.append(points * scale)
    if not boxes:
        return []
    median_height = float(np.median([np.ptp(b[:, 1]) for b in boxes]))
    grown = []
    for points in boxes:
        height = float(np.ptp(points[:, 1]))
        if median_height and height > 1.6 * median_height:
            continue
        centre = points.mean(axis=0)
        grown.append(points + np.sign(points - centre) * 0.18 * height)
    return grown


def _smoothstep(x: np.ndarray, low, high) -> np.ndarray:
    t = np.clip((x - low) / np.maximum(high - low, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def make_severe_receipt_view(source: np.ndarray, ocr_lines: list[dict[str, Any]] | None = None, *,
                             target_char_height: float = 22.0, return_record: bool = False):
    """Source-anchored view for receipts with show-through, crease webs and stains.

    The page is flattened at character scale and measured with a stroke-scale
    black-hat: sharp thermal/ink strokes respond fully, while the blurred
    mirror image of the reverse side, crease shading and broad stain bands
    do not. Faint OCR lines get thresholds from their own strength, thick
    strokes (pen, bold print) are filled from the flattened darkness, and
    tide lines inside a detected stain band are dropped off text lines.
    """
    import cv2
    src = _gray(source)
    paper0 = cv2.medianBlur(cv2.morphologyEx(src, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), 21).astype(np.float32)
    probe = 255 - np.clip(src / np.maximum(paper0, 1) * 255, 0, 255)
    ch0 = _character_height(probe >= np.percentile(probe, 96), default=12.0)
    band0 = detect_stain_bands(src)
    off0 = _off_paper(src, ch0)
    scale = float(np.clip(target_char_height / max(ch0, 1.0), 1.0, 3.0))

    def resize(image, interpolation=cv2.INTER_CUBIC):
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation) if scale > 1.01 else image

    big = resize(src)
    band = resize(band0.astype(np.uint8), cv2.INTER_NEAREST) > 0
    off = cv2.dilate(resize(off0.astype(np.uint8), cv2.INTER_NEAREST), np.ones((5, 5), np.uint8)) > 0
    ch = ch0 * scale
    big = np.where(off, 255, big).astype(np.uint8)
    kernel = int(2.2 * ch) | 1
    background = cv2.morphologyEx(big, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), ch / 3)
    norm = np.clip(big.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(norm, None, h=float(np.clip(1.1 * _noise_sigma(norm), 3, 10)),
                                        templateWindowSize=7, searchWindowSize=21)
    depth = (255 - denoised.astype(np.float32)) / 255.0
    boxes = _line_boxes(ocr_lines, scale)
    text_mask = np.zeros(depth.shape, np.uint8)
    for points in boxes:
        cv2.fillConvexPoly(text_mask, np.round(points).astype(np.int32), 1)
    probe_area = depth[text_mask > 0] if np.count_nonzero(text_mask) > 500 else depth.ravel()
    strokes = ((depth >= 0.5 * np.percentile(probe_area, 99)) & ((text_mask > 0) if np.count_nonzero(text_mask) > 500 else True)).astype(np.uint8)
    distance = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    ridge = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (strokes > 0)
    stroke_width = float(2 * np.median(distance[ridge])) if ridge.any() else 3.0
    bh_kernel = int(np.clip(int(_FAINT_SW_K * stroke_width) | 1, 7, int(1.3 * ch) | 1))
    evidence = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_kernel, bh_kernel))).astype(np.float32) / 255.0

    zone = np.zeros(evidence.shape, np.uint8)
    line_ref = np.full(evidence.shape, np.inf, np.float32)
    refs = []
    for points in boxes:
        mask = np.zeros(evidence.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(points).astype(np.int32), 1)
        mask = mask > 0
        values = evidence[mask]
        if values.size < 50:
            continue
        r = float(np.percentile(values, 97))
        refs.append(r)
        line_ref[mask] = np.minimum(line_ref[mask], r)
        zone[mask] = 1
    zone = zone > 0
    line_ref[~np.isfinite(line_ref)] = 0
    reference = float(np.median(refs)) if len(refs) >= 3 else float(np.percentile(evidence, 99.3))
    noise = float(np.percentile(evidence[~off], 70)) if (~off).any() else 0.0
    # an OCR line fainter than the receipt gets thresholds from its own
    # strength; never stricter than the receipt-wide ones
    strength = np.where(zone, np.clip(line_ref, 0.35 * reference, reference), reference)
    low = np.maximum(noise + 0.02, 0.28 * strength)
    core = np.maximum(low + 0.03, 0.55 * strength)
    weak = ((evidence >= low) & ~off).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(weak, 8)
    has_core = np.zeros(count, bool)
    has_core[np.unique(labels[(evidence >= core) & (weak > 0)])] = True
    has_core[0] = False
    keep = has_core[labels]
    fill_kernel = int(2 * stroke_width) | 1
    closed = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel, fill_kernel))) > 0
    interior = closed & ~keep & (depth >= core) & ~band & ~off
    tone = np.where(interior, _smoothstep(depth, low, 0.8 * strength), _smoothstep(evidence, low, 0.8 * strength))
    output = np.where(keep | interior, 255 - 240 * tone, 255).astype(np.uint8)

    removed_tide = 0
    marks = (output < 200).astype(np.uint8)
    mc, ml, ms, _ = cv2.connectedComponentsWithStats(marks, 8)
    if mc > 1 and band.any():
        area = np.maximum(ms[:, 4], 1)
        in_band = np.bincount(ml.ravel(), weights=band.ravel().astype(np.float32), minlength=mc) / area
        in_zone = np.bincount(ml.ravel(), weights=zone.ravel().astype(np.float32), minlength=mc) / area
        w, h = ms[:, 2], ms[:, 3]
        fill = area / np.maximum(w * h, 1)
        charlike = (h <= 1.8 * ch) & (w <= 3 * ch)
        # compact marks next to a stain (a logo) stay; long or ragged ones go
        tide = (in_band > 0.5) & (((in_zone < 0.3) & (np.maximum(w, h) >= 3.5 * ch)) | ((in_zone < 0.7) & ~charlike & (fill < 0.3)))
        tide[0] = False
        removed_tide = int(tide.sum())
        grow = cv2.dilate(tide[ml].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        output[grow & ~zone] = 255
    if not return_record:
        return output
    return output, {
        "method": "show_through_suppressing_receipt_view_v1", "scale": scale, "character_height": ch,
        "stroke_width": stroke_width, "blackhat_kernel": bh_kernel, "text_lines": len(refs),
        "stain_fraction": float(band.mean()), "off_paper_fraction": float(off.mean()),
        "removed_tide_marks": removed_tide,
        "ocr_role": "localisation only; recognised strings are not rendered",
    }


def remove_scan_streaks(view: np.ndarray, record: dict[str, Any] | None = None) -> np.ndarray:
    """Faint dotted lines running nearly the full page height or width: scanner
    or screen-capture streaks. Dark ink they cross is kept."""
    import cv2
    view = _gray(view)
    height, width = view.shape
    marks = (view < 235).astype(np.uint8)
    ch = _character_height(view < 150, default=max(10.0, height / 60))
    output = view.copy()
    removed = 0
    for vertical in (True, False):
        length = height if vertical else width
        link = (1, max(3, int(1.2 * ch)) | 1) if vertical else (max(3, int(1.2 * ch)) | 1, 1)
        joined = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, link))
        run = (1, int(0.4 * length)) if vertical else (int(0.4 * length), 1)
        lines = cv2.morphologyEx(joined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, run))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
        for index in range(1, count):
            x, y, w, h, _ = stats[index]
            thickness = w if vertical else h
            if thickness > max(3, 0.25 * ch):
                continue
            component = labels == index
            on = component & (marks > 0)
            # a printed rule is solid and dark; a streak is broken and grey
            if on.sum() > 0.6 * component.sum() and float(np.median(view[on])) < 150:
                continue
            band = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            clear = band & (view >= 120)
            output[clear] = 255
            removed += 1
    if record is not None:
        record["scan_streaks_removed"] = removed
    return output


def _print_height(src: np.ndarray) -> float:
    import cv2
    paper = cv2.medianBlur(cv2.morphologyEx(src, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), 21).astype(np.float32)
    probe = 255 - np.clip(src / np.maximum(paper, 1) * 255, 0, 255)
    return _character_height(probe >= np.percentile(probe, 96), default=12.0)


def make_enhanced_original(source: np.ndarray, ocr_lines: list[dict[str, Any]] | None = None, *,
                           target_char_height: float = 24.0, return_record: bool = False):
    """The original receipt with paper shading flattened and the print darkened.

    Nothing is removed or redrawn: every grey level of the source maps
    monotonically to the output, so faint print, logos and anti-aliased
    edges keep their shape. Used for small low-resolution captures, where
    any binarisation breaks strokes, and as the "Enhanced original" view.
    """
    import cv2
    src = _gray(source)
    paper0 = cv2.medianBlur(cv2.morphologyEx(src, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), 21).astype(np.float32)
    probe = 255 - np.clip(src / np.maximum(paper0, 1) * 255, 0, 255)
    ch0 = _character_height(probe >= np.percentile(probe, 96), default=12.0)
    scale = float(np.clip(target_char_height / max(ch0, 1.0), 1.0, 2.5))
    big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.01 else src.copy()
    off = _off_paper(src, ch0)
    if off.any():
        big[cv2.resize(off.astype(np.uint8), (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 255
    ch = ch0 * scale
    kernel = int(1.6 * ch) | 1
    background = cv2.morphologyEx(big, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), ch / 4)
    norm = np.clip(big.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(norm, None, h=float(np.clip(_noise_sigma(norm), 3, 10)),
                                        templateWindowSize=7, searchWindowSize=21)
    darkness = (255 - denoised.astype(np.float32)) / 255.0
    zone = np.zeros(darkness.shape, np.uint8)
    for points in _line_boxes(ocr_lines, scale):
        cv2.fillConvexPoly(zone, np.round(points).astype(np.int32), 1)
    zone = zone > 0
    top = float(np.percentile(darkness[zone], 97)) if zone.sum() > 500 else float(np.percentile(darkness, 99.5))
    floor = float(np.percentile(darkness, 60)) + 0.02
    tone = np.clip((darkness - floor) / max(0.8 * top - floor, 0.05), 0, 1) ** 0.7
    output = (255 - 240 * tone).astype(np.uint8)
    if not return_record:
        return output
    return output, {"method": "flattened_contrast_stretch_v1", "scale": scale, "character_height": ch,
                    "print_level": top, "paper_level": floor, "binarised": False}


def make_soft_receipt_view(source: np.ndarray, ocr_lines: list[dict[str, Any]] | None = None, *,
                           target_char_height: float = 24.0, return_record: bool = False):
    """Clean grey view for small, low-resolution captures.

    Print is measured with a stroke-scale black-hat on the flattened page, so
    crease shading and paper texture (wider than a stroke) drop out, while
    the tone stays continuous so anti-aliased letters keep their joins.
    Solid print (reverse bars, filled logo parts) is kept from the flattened
    darkness. Crease ridges, tide lines and isolated specks off the text
    lines are removed; straight printed boxes and rules are kept.
    """
    import cv2
    src = _gray(source)
    ch0 = _print_height(src)
    scale = float(np.clip(target_char_height / max(ch0, 1.0), 1.0, 3.0))
    big = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.01 else src.copy()
    off = _off_paper(src, ch0)
    if off.any():
        big[cv2.resize(off.astype(np.uint8), (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 255
    ch = ch0 * scale
    kernel = int(2.2 * ch) | 1
    background = cv2.GaussianBlur(cv2.morphologyEx(big, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))).astype(np.float32), (0, 0), ch / 3)
    norm = np.clip(big.astype(np.float32) / np.maximum(background, 1) * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(norm, None, h=float(np.clip(1.1 * _noise_sigma(norm), 3, 12)),
                                        templateWindowSize=7, searchWindowSize=21)
    depth = (255 - denoised.astype(np.float32)) / 255.0
    zone = np.zeros(depth.shape, np.uint8)
    for points in _line_boxes(ocr_lines, scale):
        cv2.fillConvexPoly(zone, np.round(points).astype(np.int32), 1)
    zone = zone > 0
    has_zone = zone.sum() > 500
    probe = depth[zone] if has_zone else depth.ravel()
    strokes = ((depth >= 0.5 * np.percentile(probe, 99)) & (zone if has_zone else True)).astype(np.uint8)
    distance = cv2.distanceTransform(strokes, cv2.DIST_L2, 3)
    ridge = (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8))) & (strokes > 0)
    stroke_width = float(2 * np.median(distance[ridge])) if ridge.any() else 3.0
    bh_kernel = int(np.clip(int(_FAINT_SW_K * stroke_width) | 1, 7, int(1.3 * ch) | 1))
    evidence = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_kernel, bh_kernel))).astype(np.float32) / 255.0
    top = float(np.percentile(evidence[zone], 97)) if has_zone else float(np.percentile(evidence, 99.5))
    floor = float(np.percentile(evidence, 75)) + 0.015
    tone = np.clip((evidence - floor) / max(0.75 * top - floor, 0.04), 0, 1)
    marks = (tone > 0.35).astype(np.uint8)
    closed = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * stroke_width) | 1,) * 2)) > 0
    depth_tone = np.clip((depth - floor) / max(0.75 * top - floor, 0.04), 0, 1)
    tone = np.where(closed & (marks == 0), np.maximum(tone, depth_tone), tone)
    band = cv2.resize(detect_stain_bands(src).astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    depth_top = float(np.percentile(depth[zone], 97)) if has_zone else float(np.percentile(depth, 99.5))
    solid = ((depth >= 0.5 * depth_top) & (depth > floor + 0.05)).astype(np.uint8)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.5 * ch) | 1,) * 2))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if count > 1:
        area = np.maximum(stats[:, 4], 1).astype(np.float32)
        in_zone = np.bincount(labels.ravel(), weights=zone.ravel().astype(np.float32), minlength=count) / area
        in_band = np.bincount(labels.ravel(), weights=band.ravel().astype(np.float32), minlength=count) / area
        box_fill = area / np.maximum(stats[:, 2] * stats[:, 3], 1)
        printed_bar = (box_fill > 0.8) & (stats[:, 2] > 3 * stats[:, 3])   # reverse-printed heading bar
        # printed solids have sharp edges; crease shadows fade out softly
        ring = cv2.dilate(solid, np.ones((5, 5), np.uint8)) > cv2.erode(solid, np.ones((5, 5), np.uint8))
        edge = np.zeros(count, np.float32); edge_n = np.zeros(count, np.float32)
        ring_labels = cv2.dilate(labels.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)
        gradient = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float32) / 255.0
        np.add.at(edge, ring_labels[ring], gradient[ring]); np.add.at(edge_n, ring_labels[ring], 1)
        sharp = (edge / np.maximum(edge_n, 1)) >= 0.35 * depth_top
        keep = ((in_zone > 0.5) | (area < 15 * ch * ch) | printed_bar) & (in_band < 0.3) & (sharp | (in_zone > 0.5) | printed_bar)
        keep[0] = False
        grow = cv2.dilate(keep[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        tone = np.where(grow, np.maximum(tone, np.clip((depth - floor) / max(0.75 * depth_top - floor, 0.04), 0, 1)), tone)
    # reverse-printed heading bars (white text on a dark band): find the band
    # at heading scale and take its tone from the flattened darkness, which
    # keeps the white letters white
    band_rows = (depth >= 0.25).astype(np.uint8)
    band_rows = cv2.morphologyEx(band_rows, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.8 * ch) | 1, 3)))
    band_rows = cv2.morphologyEx(band_rows, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * ch) | 1, int(0.4 * ch) | 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(band_rows, 8)
    bars = 0
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if w >= 6 * ch and 0.4 * ch <= h <= 2.2 * ch and area >= 0.75 * w * h:
            region = (slice(y, y + h), slice(x, x + w))
            tone[region] = np.maximum(tone[region], np.clip((depth[region] - 0.08) / max(0.75 * depth_top - 0.08, 0.05), 0, 1))
            zone[region] = True
            bars += 1
    output = (255 - 240 * tone ** 1.1).astype(np.uint8)
    near_text = cv2.dilate(zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.6 * ch) | 1, int(0.3 * ch) | 1))) > 0
    mask = (output < 200).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    removed = {"tide": 0, "crease": 0, "speck": 0}
    if count > 1:
        area = np.maximum(stats[:, 4], 1).astype(np.float32)
        in_band = np.bincount(labels.ravel(), weights=band.ravel().astype(np.float32), minlength=count) / area
        in_text = np.bincount(labels.ravel(), weights=near_text.ravel().astype(np.float32), minlength=count) / area
        width, height = stats[:, 2], stats[:, 3]
        major = np.maximum(width, height); fill = area / np.maximum(width * height, 1)
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * ch), 1)))
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(3 * ch))))
        straight = np.bincount(labels.ravel(), weights=((horizontal | vertical) > 0).ravel().astype(np.float32), minlength=count) / area
        large = area >= 0.3 * ch * ch
        near_large = cv2.dilate((large[labels] & (labels > 0)).astype(np.uint8), np.ones((int(0.8 * ch) | 1,) * 2, np.uint8)) > 0
        beside = np.bincount(labels.ravel(), weights=near_large.ravel().astype(np.float32), minlength=count) > 0
        tide = (in_band > 0.4) & (in_text < 0.5) & (major >= 1.5 * ch) & (straight < 0.4)
        crease = (in_text < 0.3) & (major >= 2.5 * ch) & (fill < 0.18) & (straight < 0.4)
        speck = ~large & (in_text < 0.3) & ~beside
        # crease shadows are grey even at their darkest point; print is not
        peak = np.zeros(count, np.float32)
        np.maximum.at(peak, labels.ravel(), depth.ravel())
        shadow = (in_text < 0.3) & (peak < 0.55 * depth_top) & (straight < 0.4)
        drop = tide | crease | speck | shadow
        drop[0] = False
        removed = {"tide": int(tide.sum()), "crease": int(crease.sum()), "speck": int(speck.sum()), "shadow": int(shadow.sum())}
        grow = cv2.dilate(drop[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        output[(grow & (output >= 60)) | drop[labels]] = 255
        kept = cv2.dilate((output < 170).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        output[(output >= 170) & ~near_text & ~kept] = 255   # crease shading off the text
    if not return_record:
        return output
    return output, {"method": "soft_low_resolution_view_v1", "scale": scale, "character_height": ch,
                    "stroke_width": stroke_width, "blackhat_kernel": bh_kernel, "removed": removed, "heading_bars": bars,
                    "binarised": False, "ocr_role": "localisation only; recognised strings are not rendered"}


def separate_content_from_damage(view: np.ndarray, source: np.ndarray, ocr_lines: list[dict[str, Any]] | None,
                                 rec: dict[str, Any] | None = None) -> np.ndarray:
    """Classify every mark as text, table rule, logo, signature/handwriting,
    picture or damage, and keep only the content.

    Damage is what print is not: grey crease shading, specks, tide lines
    inside a stain band, ragged marks wider than the print's strokes.
    """
    import cv2
    ocr = ocr_lines
    view=_gray(view); src=_gray(source)
    H,W=view.shape; scale=W/src.shape[1]
    ink=(view<200).astype(np.uint8)
    ch=_character_height(view<150,default=max(10.0,H/60))
    # text lines: every OCR box (any size) + rows of character-sized marks
    ocr_zone=np.zeros(view.shape,np.uint8)
    for line in ocr or []:
        if float(line.get('confidence') or 0)<0.3: continue
        p=np.asarray(line.get('box',[]),np.float32)
        if p.shape!=(4,2): continue
        p=p*scale; c=p.mean(0); hh=float(np.ptp(p[:,1]))
        cv2.fillConvexPoly(ocr_zone,np.round(p+np.sign(p-c)*0.18*hh).astype(np.int32),1)
    ocr_zone=ocr_zone>0
    rows=_text_rows(view<150,ch)
    zone=ocr_zone|rows
    n,l,st,_=cv2.connectedComponentsWithStats(ink,8)
    area=np.maximum(st[:,4],1).astype(np.float32)
    def per(img): return np.bincount(l.ravel(),weights=img.ravel().astype(np.float32),minlength=n)/area
    inz=per(zone); dark=per(view)
    w,h=st[:,2],st[:,3]; major=np.maximum(w,h); minor=np.maximum(np.minimum(w,h),1); fill=area/np.maximum(w*h,1)
    border=(st[:,0]<=1)|(st[:,1]<=1)|(st[:,0]+w>=W-1)|(st[:,1]+h>=H-1)
    band=cv2.resize(detect_stain_bands(src).astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
    inb=per(band)
    # stroke thickness per mark vs the print's own stroke width
    dist=cv2.distanceTransform(ink,cv2.DIST_L2,3)
    ridge=(dist>=cv2.dilate(dist,np.ones((3,3),np.uint8)))&(ink>0)
    thick=np.zeros(n,np.float32); cnt=np.zeros(n,np.float32)
    np.add.at(thick,l[ridge],dist[ridge]); np.add.at(cnt,l[ridge],1); thick=2*thick/np.maximum(cnt,1)
    txt=(inz>=0.5); txt[0]=False
    sw=float(np.median(thick[txt])) if txt.any() else 3.0
    ref=float(np.median(dark[txt])) if txt.any() else 110.0
    inky=dark<=ref+30                                   # as dark as the print
    # table rules / box edges: long straight runs
    hk=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(int(3*ch),1)))
    vk=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,int(3*ch))))
    straight=per((hk|vk)>0)
    cls=np.full(n,'noise',object); cls[0]='bg'
    text=(inz>=0.5)&(h<=5*ch)&((inb<0.5)|inky)&~((w>6*ch)&(fill<0.12)&(straight<0.3))
    table=(straight>=0.45)&(dark<=ref+50)
    logo=(major>=1.2*ch)&(major<=10*ch)&(major<=2.2*minor)&(fill>=0.15)&(area>=0.6*ch*ch)&inky&~border&(inb<0.3)
    # handwriting / signature / pictures: ink-dark strokes of print-like width, outside the stain band
    stroke=(area>=1.2*ch*ch)&(major>=1.5*ch)&inky&(thick<=2.2*sw)&(inb<0.5)&~border
    small_dark=(area>=0.08*ch*ch)&(area<0.3*ch*ch)&(dark<=ref+10)   # punctuation / dots near content
    # pictures (product photos, halftone art): dense patches of marks
    marks=((ink>0)&(view<215)).astype(np.uint8)
    dens=cv2.blur(marks.astype(np.float32),(int(2*ch)|1,)*2)
    patch=(dens>=0.18).astype(np.uint8)
    patch=cv2.morphologyEx(patch,cv2.MORPH_OPEN,np.ones((int(ch)|1,)*2,np.uint8))
    pn,pl,ps,_=cv2.connectedComponentsWithStats(patch,8)
    okp=np.zeros(pn,bool)
    for i in range(1,pn):
        x,y,ww,hh,aa=ps[i]
        reg=pl==i
        if aa<4*ch*ch or ww>0.6*W or float(band[reg].mean())>0.3 or float(zone[reg].mean())>0.5: continue
        pc=np.unique(l[reg&(ink>0)]); 
        if len(pc)>=8 and float(view[reg&(ink>0)].mean())<=ref+70: okp[i]=True
    pic_mask=okp[pl]
    picture=(per(pic_mask)>=0.6)&(area>=1)
    for name,m in [('picture',picture),('stroke',stroke),('logo',logo),('table',table),('text',text)]:
        cls[m&(np.arange(n)>0)]=name
    keep=(cls!='noise')&(cls!='bg')
    # dots kept only next to other content
    km=keep[l]
    near=cv2.dilate(km.astype(np.uint8),np.ones((int(0.8*ch)|1,)*2,np.uint8))>0
    nb=per(near)>0
    dots=small_dark&nb&~keep; dots[0]=False
    keep|=dots
    # a damage mark running through a text line: keep only its in-line, off-band part
    partial=(~keep)&(inz>=0.2); partial[0]=False
    out=np.full(view.shape,255,np.uint8)
    km=keep[l]; out[km]=view[km]
    pm=partial[l]&zone&~band; out[pm]=view[pm]
    # safety numbers
    charlike=(h>=0.5*ch)&(h<=1.6*ch)&(w<=2.0*ch)&(dark<=ref+15)&(area>=0.15*ch*ch)&(inb<0.5)
    lost=int((charlike&~keep&~partial)[1:].sum())
    lost_ink=float(area[(~keep)&(dark<=ref+15)&(np.arange(n)>0)].sum())/max(float(area[keep].sum()),1)
    cov=float(ocr_zone[rows].mean()) if rows.any() else 0.0
    counts={k:int((cls==k).sum()) for k in ('text','table','logo','stroke','picture','noise')}
    if rec is not None: rec.update(classes=counts,ocr_coverage=round(cov,3),lost_charlike=lost,lost_ink=round(lost_ink,3),stroke_width=round(sw,2))
    return out

def join_broken_strokes(view: np.ndarray, source: np.ndarray, ocr_lines: list[dict[str, Any]] | None,
                        rec: dict[str, Any] | None = None) -> np.ndarray:
    """Reconnect broken strokes inside OCR text lines, only along faint ink
    still present in the original (hysteresis from confirmed print).
    Gaps with no trace in the source stay open; nothing is drawn from the
    recognised text."""
    import cv2
    ocr = ocr_lines
    view=_gray(view); src=_gray(source)
    H,W=view.shape; scale=W/src.shape[1]
    ch=_print_height(src)*scale
    big=cv2.resize(src,(W,H),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    k=int(1.6*ch)|1
    bg=cv2.GaussianBlur(cv2.morphologyEx(big,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))),(0,0),ch/5)
    norm=cv2.GaussianBlur(np.clip(big/np.maximum(bg,1)*255,0,255).astype(np.uint8),(0,0),max(0.8,scale*0.5))
    # text rows from the image itself (not from which words OCR found)
    zone=cv2.dilate(_text_rows(view<150,ch).astype(np.uint8),np.ones((3,3),np.uint8))>0
    dark=view<150
    dist=cv2.distanceTransform(dark.astype(np.uint8),cv2.DIST_L2,3)
    rid=(dist>=cv2.dilate(dist,np.ones((3,3),np.uint8)))&dark&zone
    sw=float(2*np.median(dist[rid])) if rid.any() else 3.0
    kk=int(max(5,3.2*sw))|1
    ev=cv2.morphologyEx(norm,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kk,kk))).astype(np.float32)
    strong=float(np.percentile(ev[dark&zone],50)) if (dark&zone).any() else float(np.percentile(ev,99))
    noise=float(np.percentile(ev[zone&(view>=250)],85)) if (zone&(view>=250)).any() else 0
    weak=zone&(ev>=max(noise+1.5,0.33*strong))
    # hysteresis: weak trace counts only when connected to confirmed print
    near=cv2.dilate(dark.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(max(3,0.25*ch))|1,)*2))>0
    allowed=(weak&near)|dark
    n,l=cv2.connectedComponents(allowed.astype(np.uint8),connectivity=8)
    touch=np.zeros(n,bool); touch[np.unique(l[dark])]=True; touch[0]=False
    stroke=touch[l]&zone
    # keep letter counters open: only take pixels darker than the trace's own local midpoint
    loc=cv2.dilate(ev,np.ones((int(sw*2)|1,)*2,np.uint8))
    stroke&=(ev>=0.45*loc)|dark
    t=np.clip((ev-noise)/max(0.75*strong-noise,1),0,1)
    tone=(255-235*np.maximum(t,0.65)).astype(np.uint8)
    out=view.copy()
    m=stroke&~dark
    out[m]=np.minimum(out[m],tone[m])
    # existing grey stroke pixels inside text become full ink
    g=stroke&(view<230)
    out[g]=np.minimum(out[g],np.maximum(20,(view[g].astype(np.int32)-90)).astype(np.uint8))
    if rec is not None: rec.update(added=int(m.sum()),ch=round(ch,1),sw=round(sw,2))
    return out

def _ink_change(candidate: np.ndarray, current: np.ndarray, ocr_lines, scale: float) -> tuple[float, float, float]:
    """(text ink kept, off-text ink kept) of candidate relative to current, and
    the share of the off-text area the current view marks."""
    zone = _ocr_zone(current.shape, ocr_lines, scale)
    a, b = candidate < 150, current < 150
    text = float((a & zone).sum()) / max(float((b & zone).sum()), 1.0)
    clutter = float((a & ~zone).sum()) / max(float((b & ~zone).sum()), 1.0)
    present = float((b & ~zone).sum()) / max(float((~zone).sum()), 1.0)
    return text, clutter, present


# ---------------------------------------------------------------- routing
def finish_faithful_view(view: np.ndarray, source: np.ndarray,
                         ocr_lines: list[dict[str, Any]] | None,
                         degradation_profile: dict[str, Any] | None,
                         *, neutral_source: bool = True,
                         record: dict[str, Any] | None = None,
                         options: dict[str, Any] | None = None) -> np.ndarray:
    """Final degradation-aware stage applied to every faithful view.

    Clean, colour or ordinary receipts are returned unchanged. Degraded
    neutral receipts (faded/blurred print, holes, stains, tiny captures) get
    the matching treatment.
    """
    import cv2
    if not neutral_source:
        return view
    options = options or {}
    allow_severe = options.get("severe_cleanup", True)
    allow_separation = options.get("damage_separation", True)
    allow_joining = options.get("stroke_joining", True)
    original_view = view
    view = _gray(view)
    src = _gray(source)
    flags = set((degradation_profile or {}).get("flags", []))
    stage: dict[str, Any] = {"flags": sorted(flags)}
    base = view
    # small AND degraded: low-resolution photos/screenshots of faint receipts
    small = min(src.shape) < _TINY_MIN_SIDE
    tiny = small and bool({"faded", "blurred"} & flags)
    low_resolution = small and _print_height(src) <= _LOW_RES_CHAR_HEIGHT
    if low_resolution and flags and not tiny:
        # a small but legible capture (screenshot, low-resolution photo):
        # binarising its anti-aliased print breaks strokes, so keep a
        # continuous tone and drop only crease, stain and speck marks
        enhanced, enhanced_record = make_soft_receipt_view(src, ocr_lines, return_record=True)
        stage.update(soft_view=enhanced_record, applied=True)
        try:
            if not allow_separation:
                raise LookupError("disabled")
            info = {}
            separated = separate_content_from_damage(enhanced, src, ocr_lines, rec=info)
            safe = (info["ocr_coverage"] >= _SEPARATE_MIN_COVERAGE and info["lost_charlike"] <= _SEPARATE_MAX_LOST
                    and info["lost_ink"] <= _SEPARATE_MAX_LOST_INK)
            info["selected"] = bool(safe)
            stage["content_separation"] = info
            if safe:
                enhanced = separated
        except LookupError:
            pass
        except Exception as error:
            stage["content_separation"] = {"error": type(error).__name__}
        if record is not None:
            record["degradation_finishing"] = stage
            record["applied"] = True
        return enhanced
    if tiny:
        base, faint_record = make_faint_document_view(src, return_record=True)
        stage["faint_view"] = faint_record
    paper = cv2.medianBlur(cv2.morphologyEx(src, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))), 21).astype(np.float32)
    probe_height = _character_height(src < 0.75 * paper, default=14.0)
    has_holes = bool(detect_holes(src, paper, probe_height).any())
    has_stain = bool(detect_stain_bands(src).any())
    # Only the degradation assessment decides whether finishing runs: scanner
    # borders and ruled forms can look like holes/stains on clean receipts.
    degraded = tiny or bool({"faded", "blurred"} & flags)
    stage.update(holes_detected=has_holes, stain_detected=has_stain, applied=degraded)
    if degraded:
        cleanup: dict[str, Any] = {}
        base = suppress_receipt_artifacts(
            base, src, ocr_lines,
            lift_faint_lines=("faded" in flags) and not tiny,
            record=cleanup,
        )
        stage["cleanup"] = cleanup
        base = remove_scan_streaks(base, record=cleanup)
    result = base if degraded else original_view
    # Show-through, crease webs and tide lines: a second, source-anchored
    # candidate replaces the view only when it removes most of the off-text
    # marks while keeping the text lines (bold forms and pictures fail this).
    if flags and not small and allow_severe:
        current = _gray(result)
        # cheap pre-check: the candidate can only be chosen when the current
        # view has marks off the text lines, so skip the costly build otherwise
        _, _, clutter_now = _ink_change(current, current, ocr_lines, current.shape[1] / src.shape[1])
        if clutter_now < _SEVERE_CLUTTER_PRESENT:
            stage["severe_receipt_view"] = {"skipped": "no off-text marks to clean", "selected": False, "off_text_ink_share": round(clutter_now, 4)}
            allow_severe = False
    if flags and not small and allow_severe:
        current = _gray(result)
        try:
            candidate, severe_record = make_severe_receipt_view(src, ocr_lines, return_record=True)
            candidate = suppress_receipt_artifacts(candidate, src, ocr_lines, lift_faint_lines=False)
            candidate = cv2.resize(candidate, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)
            text_kept, clutter_kept, clutter_present = _ink_change(candidate, current, ocr_lines, current.shape[1] / src.shape[1])
            chosen = (clutter_present >= _SEVERE_CLUTTER_PRESENT and clutter_kept <= _SEVERE_CLUTTER_MAX
                      and text_kept >= _SEVERE_TEXT_MIN)
            severe_record.update(text_ink_ratio=round(text_kept, 3), off_text_ink_ratio=round(clutter_kept, 3),
                                 off_text_ink_share=round(clutter_present, 4), selected=bool(chosen))
            stage["severe_receipt_view"] = severe_record
            if chosen:
                result = candidate
                stage["applied"] = True
        except Exception as error:  # the finishing stage must never break a run
            stage["severe_receipt_view"] = {"error": type(error).__name__}
    # last step for damaged receipts: keep text, tables, logos, signatures and
    # pictures, drop marks classified as damage, but only when that provably
    # loses no character-like ink (otherwise the current view is kept)
    if flags and allow_separation:
        try:
            info: dict[str, Any] = {}
            separated = separate_content_from_damage(_gray(result), src, ocr_lines, rec=info)
            safe = (info["ocr_coverage"] >= _SEPARATE_MIN_COVERAGE and info["lost_charlike"] <= _SEPARATE_MAX_LOST
                    and info["lost_ink"] <= _SEPARATE_MAX_LOST_INK)
            info["selected"] = bool(safe)
            stage["content_separation"] = info
            if safe:
                result = separated
                stage["applied"] = True
        except Exception as error:
            stage["content_separation"] = {"error": type(error).__name__}
    # faded or blurred print: join broken strokes along faint source ink
    if {"faded", "blurred"} & flags and allow_joining:
        try:
            info = {}
            result = join_broken_strokes(_gray(result), src, ocr_lines, rec=info)
            stage["stroke_joining"] = info
            stage["applied"] = True
        except Exception as error:
            stage["stroke_joining"] = {"error": type(error).__name__}
    if record is not None:
        record["degradation_finishing"] = stage
        if result.shape[:2] != original_view.shape[:2]:
            # lets the app store a size-matched original for side-by-side compare
            record["applied"] = True
    return result
