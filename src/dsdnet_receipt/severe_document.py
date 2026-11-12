"""Source-anchored recovery view for severely degraded printed documents.

Used for faded, creased A4/A5 invoices and cash bills where the learned
recovery and the continuous tone transfer leave three visible problems:

* faint print stays grey (e.g. Chinese company names, small address lines);
* fold edges and paper grain survive as speckle and short dark dashes;
* glyphs are broken where a fold crosses them, because the stroke-repair
  stages copy the model's fold response into the displayed image.

The view below is derived from the rectified source only. OCR text is never
rendered, and no glyph is drawn from a template:

1. paper normalisation (ink-free closing + median) and non-local-means
   denoising;
2. stroke-scale evidence by morphological black-hat, which keeps narrow
   print and drops broad fold shading;
3. hysteresis: confident ink cores seed a bounded geodesic growth through
   weaker evidence, which reconnects faint parts of the same glyph without
   following folds across the page;
4. component tests that return fold ridges, blurry fold blobs, isolated
   paper speckle and wavy fold traces to paper while keeping punctuation,
   straight printed rules, logos and signatures;
5. a locally normalised tone curve so faint regions reach the same dark
   ink tone as strong regions, followed by sub-pixel contour smoothing;
6. weak stroke-scale evidence next to accepted text is shown as grey rather
   than dropped, so uncertain parts stay visible instead of being guessed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Tuned on natural NTME/Heng/Kok Fong cash bills (development receipts only).
_BLACKHAT_SCALE = 1.2      # black-hat kernel relative to character height
_WEAK_FRACTION = 0.42      # weak-evidence threshold relative to core threshold
_FAINT_GAIN = 3.0          # gain for grey display of weak evidence near text
_FAINT_MAX = 150.0         # weakest evidence never darker than mid grey
_ISOLATED_SHARPNESS = 0.85 # isolated groups must have print-like edges ...
_INK_REFERENCE = 0.60      # local darkness reaching full ink tone
_CONTOUR_LOW, _CONTOUR_HIGH = 0.20, 0.60
_INK_TONE = 20.0


def _smoothstep(values: np.ndarray, low: Any, high: Any) -> np.ndarray:
    t = np.clip((values - low) / np.maximum(high - low, 1e-3), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normalised_darkness(source: np.ndarray) -> np.ndarray:
    import cv2

    paper = cv2.morphologyEx(
        source, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    paper = cv2.medianBlur(paper, 21).astype(np.float32)
    paper = np.maximum(paper, float(np.percentile(source, 50)) * 0.8)
    normalised = np.clip(source.astype(np.float32) / np.maximum(paper, 1.0) * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoising(normalised, None, h=9, templateWindowSize=7, searchWindowSize=21)
    return 255.0 - denoised.astype(np.float32)


def _character_height(binary: np.ndarray, default: float = 14.0) -> float:
    import cv2

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    heights = [int(s[3]) for s in stats[1:count] if 5 <= s[3] <= 60 and s[4] >= 12]
    return float(np.median(heights)) if len(heights) >= 10 else default


def _straight_rules(evidence: np.ndarray, low: float) -> np.ndarray:
    """Printed form rules: long and confined to one row/column band."""
    import cv2

    height, width = evidence.shape
    weak = (evidence >= low).astype(np.uint8)
    rules = np.zeros_like(weak)
    for horizontal in (True, False):
        length = max(40, (width // 12) if horizontal else (height // 14))
        kernel = (length, 1) if horizontal else (1, length)
        opened = cv2.morphologyEx(weak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        for index in range(1, count):
            _, _, w, h, area = stats[index]
            span, thickness = (w, h) if horizontal else (h, w)
            if thickness <= max(5, int(0.008 * span)) and area >= 0.6 * span:
                rules[labels == index] = 1
    return rules


def _fold_lines(mask: np.ndarray, char_height: float) -> np.ndarray:
    """Long straight diagonal traces.

    Printed rules and text rows are axis aligned, so a long straight trace
    between 8 and 82 degrees is a paper fold. Near-horizontal folds cannot be
    separated from text rows by line geometry and are left to the component
    tests above.
    """
    import cv2

    lines = cv2.HoughLinesP(
        (mask > 0).astype(np.uint8) * 255, 1, np.pi / 180, threshold=int(4 * char_height),
        minLineLength=int(8 * char_height), maxLineGap=int(max(3, char_height * 0.3)),
    )
    folds = np.zeros(mask.shape, np.uint8)
    if lines is None:
        return folds.astype(bool)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = abs(float(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))) % 180.0
        angle = min(angle, 180.0 - angle)
        if 8.0 <= angle <= 82.0:
            cv2.line(folds, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=5)
    return folds.astype(bool)


def severe_document_ink_strength(source: np.ndarray) -> float:
    """Robust upper ink darkness after paper normalisation (0-255)."""
    darkness = _normalised_darkness(source.astype(np.uint8))
    inner = darkness[darkness < 200]
    return float(np.percentile(inner, 99.5)) if inner.size else 255.0


def needs_severe_document_recovery(source: np.ndarray, degradation_profile: dict[str, Any] | None) -> bool:
    """Select the view for faded print on broad documents.

    The degradation assessment already flags faded print (weak stroke
    strength or low normalised contrast). Clean, strongly printed invoices
    keep the existing continuous view, which preserves their stroke weight.
    """
    flags = set((degradation_profile or {}).get("flags", []))
    return "faded" in flags


def make_severe_document_view(source: np.ndarray, *, return_record: bool = False):
    import cv2

    src = source.astype(np.uint8)
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height, width = src.shape
    base = _normalised_darkness(src)

    # 1. stroke-scale evidence
    provisional_height = _character_height(base >= np.percentile(base, 97))
    kernel_size = int(np.clip(int(provisional_height * _BLACKHAT_SCALE) | 1, 7, 25))
    blackhat = cv2.morphologyEx(
        np.clip(255 - base, 0, 255).astype(np.uint8), cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    ).astype(np.float32)
    # Solid marks wider than the kernel (logos, filled boxes) keep full darkness.
    inner_base = base[base < 200]
    solid_level = float(np.percentile(inner_base, 99.7)) if inner_base.size else 160.0
    evidence = np.where(base >= solid_level, np.maximum(blackhat, base), blackhat)

    # 2. thresholds from the image's own statistics
    median = float(np.median(evidence))
    spread = float(np.median(np.abs(evidence - median))) * 1.4826 + 1.0
    inner = evidence[evidence < 200]
    top = float(np.percentile(inner, 99.5)) if inner.size else 120.0
    core_level = float(np.clip(max(median + 0.45 * (top - median), median + 4.0 * spread), 30.0, 80.0))
    weak_level = float(max(median + 2.0 * spread, _WEAK_FRACTION * core_level))

    gx = cv2.Sobel(evidence, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(evidence, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)

    # 3. confident ink cores
    core = (evidence >= core_level).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    char_height = _character_height(core)
    ring = cv2.dilate(labels.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    ring[core > 0] = 0
    edge = np.bincount(ring.ravel(), weights=gradient.ravel(), minlength=count) / np.maximum(
        np.bincount(ring.ravel(), minlength=count), 1)
    mean = np.bincount(labels.ravel(), weights=evidence.ravel(), minlength=count) / np.maximum(stats[:, 4], 1)
    relative_edge = edge / np.maximum(mean, 1.0)
    typical = (stats[1:, 4] >= 12) & (stats[1:, 4] <= char_height * char_height * 3)
    ink_sharpness = float(np.median(relative_edge[1:][typical])) if np.any(typical) else 1.0
    tiny = max(3, int(0.02 * char_height * char_height))
    keep = np.zeros(count, bool)
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        if area < tiny:
            continue
        if max(w, h) > 2.5 * char_height and area / float(w * h) < 0.2 and mean[index] < core_level + 25:
            continue  # sparse fold ridge
        if area >= 0.3 * char_height * char_height and relative_edge[index] < 0.55 * ink_sharpness:
            continue  # blurry fold shadow
        keep[index] = True
    grown = keep[labels].astype(np.uint8)

    # 4. bounded geodesic growth through weak evidence
    weak = (evidence >= weak_level).astype(np.uint8)
    for _ in range(max(2, int(round(char_height * 0.45)))):
        grown = cv2.dilate(grown, np.ones((3, 3), np.uint8)) & weak

    # 5. speckle: small marks survive only near text, or as compact dark punctuation
    small = int(0.15 * char_height * char_height)
    count2, labels2, stats2, _ = cv2.connectedComponentsWithStats(grown, 8)
    large = np.zeros(count2, bool)
    large[1:] = stats2[1:, 4] >= small
    reach = cv2.dilate(large[labels2].astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(1.6 * char_height) | 1, int(0.9 * char_height) | 1)))
    near_text = np.bincount(labels2.ravel(), weights=reach.ravel(), minlength=count2) > 0
    peak = np.zeros(count2)
    np.maximum.at(peak, labels2.ravel(), evidence.ravel())
    w2, h2 = stats2[:, 2], stats2[:, 3]
    fill = stats2[:, 4] / np.maximum(w2 * h2, 1)
    compact = ((fill >= 0.4) & (np.maximum(w2, h2) <= 2.5 * np.maximum(np.minimum(w2, h2), 1))
               & (peak >= core_level + 15) & (stats2[:, 4] >= tiny))
    accepted = large | near_text | compact
    accepted[0] = False
    mask = accepted[labels2].astype(np.uint8)

    # 6. groups of marks must be dark or sharp-edged like print
    groups = cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(2.2 * char_height) | 1, int(1.2 * char_height) | 1)))
    count4, labels4, _, _ = cv2.connectedComponentsWithStats(groups, 8)
    if count4 > 1:
        inside = np.where(mask > 0, labels4, 0)
        border = np.where((cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & (mask == 0), labels4, 0)
        group_mean = np.bincount(inside.ravel(), weights=evidence.ravel(), minlength=count4) / np.maximum(
            np.bincount(inside.ravel(), minlength=count4), 1)
        group_edge = np.bincount(border.ravel(), weights=gradient.ravel(), minlength=count4) / np.maximum(
            np.bincount(border.ravel(), minlength=count4), 1)
        drop = (group_mean < core_level) & (group_edge / np.maximum(group_mean, 1) < _ISOLATED_SHARPNESS * ink_sharpness)
        drop[0] = False
        mask[drop[labels4] & (mask > 0)] = 0

    # 7. long thin wavy traces without character stems are folds
    count5, labels5, stats5, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count5 > 1:
        mean5 = np.bincount(labels5.ravel(), weights=evidence.ravel(), minlength=count5) / np.maximum(stats5[:, 4], 1)
        major = np.maximum(stats5[:, 2], stats5[:, 3])
        minor = np.maximum(np.minimum(stats5[:, 2], stats5[:, 3]), 1)
        fold = ((major > 4 * char_height) & (minor <= 1.3 * char_height) & (major / minor >= 5)
                & (major / minor < 35) & (mean5 < 1.6 * core_level))
        fold[0] = False
        for index in np.nonzero(fold)[0]:
            x, y, w, h, _ = stats5[index]
            component = labels5[y:y + h, x:x + w] == index
            horizontal = w >= h
            column_counts = component.sum(axis=0 if horizontal else 1)
            extent = h if horizontal else w
            occupied = column_counts[column_counts > 0]
            if occupied.size and float(np.mean(occupied >= 0.65 * extent)) > 0.04:
                fold[index] = False  # glyph stems present: this is text
        mask[fold[labels5]] = 0
    # 8. long straight traces that are not printed rules (diagonal, or tilted
    #    and thicker than the page's rules) are folds. Remove them outside
    #    character neighbourhoods only, so a fold crossing a glyph never cuts it.
    fold_lines = _fold_lines(mask, char_height)
    if fold_lines.any():
        remainder = (mask > 0) & ~fold_lines
        count6, labels6, stats6, _ = cv2.connectedComponentsWithStats(remainder.astype(np.uint8), 8)
        glyphs = np.zeros(count6, bool)
        glyphs[1:] = stats6[1:, 4] >= small
        glyph_zone = cv2.dilate(glyphs[labels6].astype(np.uint8), np.ones((3, 3), np.uint8),
                                iterations=4) > 0
        mask[fold_lines & ~glyph_zone] = 0
        # Fragments lying mostly along a detected fold, without character
        # stems, are pieces of that fold rather than print.
        fold_band = cv2.dilate(fold_lines.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        count8, labels8, stats8, _ = cv2.connectedComponentsWithStats(mask, 8)
        on_fold = np.bincount(labels8.ravel(), weights=fold_band.ravel(), minlength=count8) / np.maximum(stats8[:, 4], 1)
        for index in np.nonzero(on_fold >= 0.6)[0]:
            if index == 0:
                continue
            x, y, w, h, _ = stats8[index]
            component = labels8[y:y + h, x:x + w] == index
            horizontal = w >= h
            counts = component.sum(axis=0 if horizontal else 1)
            extent = h if horizontal else w
            occupied = counts[counts > 0]
            has_stems = extent >= 0.5 * char_height and occupied.size and float(np.mean(occupied >= 0.65 * extent)) > 0.04
            if not has_stems:
                mask[labels8 == index] = 0
    rules = _straight_rules(evidence, weak_level)
    if fold_lines.any():
        # Rule fragments inside a fold band are kept only when they belong to
        # a page-spanning printed rule.
        count7, labels7, stats7, _ = cv2.connectedComponentsWithStats(rules, 8)
        span = np.maximum(stats7[:, 2], stats7[:, 3])
        long_rule = span >= 0.3 * max(width, height) * 0.77
        long_rule[0] = False
        rules = (rules > 0) & (~fold_lines | long_rule[labels7])
    mask |= rules.astype(np.uint8)

    # 8. locally normalised ink tone and contour smoothing
    window = int(2.5 * char_height) | 1
    local_max = cv2.dilate(np.where(mask > 0, evidence, 0).astype(np.float32),
                           cv2.getStructuringElement(cv2.MORPH_RECT, (window, window)))
    local_max = cv2.GaussianBlur(local_max, (0, 0), char_height * 0.5)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    full = np.maximum(weak_level + 10.0, _INK_REFERENCE * local_max)
    alpha = np.where(near, _smoothstep(evidence, weak_level * 0.8, full), 0.0).astype(np.float32)
    upsampled = cv2.resize(alpha, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    upsampled = _smoothstep(cv2.GaussianBlur(upsampled, (0, 0), 0.7), _CONTOUR_LOW, _CONTOUR_HIGH)
    alpha = cv2.resize(upsampled, (width, height), interpolation=cv2.INTER_AREA)
    output = 255.0 - alpha * (255.0 - _INK_TONE)

    # 9. uncertain strokes next to text stay visible as grey
    zone_size = max(3, int(char_height * 0.5)) | 1
    zone = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zone_size, zone_size))) > 0
    faint = cv2.GaussianBlur(np.clip((evidence - weak_level * 0.45) * _FAINT_GAIN, 0, _FAINT_MAX), (0, 0), 0.6)
    if fold_lines.any():
        zone &= ~(fold_lines & (mask == 0))
    output = np.where(zone, np.minimum(output, 255.0 - faint), output)
    output = np.clip(output, 0, 255).round().astype(np.uint8)
    if not return_record:
        return output
    return output, {
        "method": "source_anchored_severe_document_view_v1",
        "blackhat_kernel": kernel_size,
        "character_height": char_height,
        "core_level": core_level,
        "weak_level": weak_level,
        "ink_pixels_fraction": float(np.mean(mask > 0)),
    }
