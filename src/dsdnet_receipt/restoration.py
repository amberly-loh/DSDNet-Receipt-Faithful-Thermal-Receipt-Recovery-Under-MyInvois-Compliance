from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass
class RestorationResult:
    image: np.ndarray
    model_available: bool
    status: str
    error: str | None
    checkpoint: str | None
    stage_count: int = 0
    diagnostic_map: np.ndarray | None = None
    diagnostics: dict[str, Any] | None = None
    model_output: np.ndarray | None = None


class RestorationEngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._model = None
        self._device = None
        self._loaded_checkpoint: Path | None = None

    def run(self, image: np.ndarray, blur_sigma: float | None = None) -> RestorationResult:
        checkpoint_value = self.config["dsdnet"].get("checkpoint")
        if not checkpoint_value:
            return RestorationResult(image.copy(), False, "checkpoint_unavailable", None, None)
        checkpoint = Path(checkpoint_value)
        if not checkpoint.exists():
            return RestorationResult(image.copy(), False, "checkpoint_unavailable", f"Checkpoint not found: {checkpoint}", str(checkpoint))
        try:
            import torch

            from .dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel

            if self._model is None or self._loaded_checkpoint != checkpoint:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._model = DSDNetReceiptAdapter(
                    self.config["dsdnet"]["source_path"],
                    stages=int(self.config["dsdnet"]["stages"]),
                    branches=int(self.config["dsdnet"]["maxout_branches"]),
                    prior_channels=int(self.config["dsdnet"].get("prior_channels", 16)),
                    data_channels=int(self.config["dsdnet"].get("data_channels", 16)),
                ).to(self._device)
                payload = torch.load(checkpoint, map_location=self._device)
                state = payload.get("model_state_dict", payload.get("state_dict", payload.get("model", payload))) if isinstance(payload, dict) else payload
                self._model.load_state_dict(state, strict=True)
                self._model.eval()
                self._loaded_checkpoint = checkpoint
            model = self._model
            device = self._device
            if blur_sigma is None:
                # The pinned FFT convolution expects non-zero spatial padding.
                # A centred 3x3 impulse is the exact identity operator while
                # keeping that interface valid for naturally degraded receipts.
                kernel = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
                kernel[:, 1, 1] = 1.0
            else:
                kernel = gaussian_kernel(float(blur_sigma)).to(device)
            tile_size = int(self.config["dsdnet"].get("tile_size", 128))
            overlap = int(self.config["dsdnet"].get("tile_overlap", 32))
            if not 0 <= overlap < tile_size:
                raise ValueError("tile_overlap must be non-negative and smaller than tile_size")
            source = image.astype(np.float32) / 255.0
            height, width = source.shape
            pad_bottom = max(0, tile_size - height)
            pad_right = max(0, tile_size - width)
            padded = np.pad(source, ((0, pad_bottom), (0, pad_right)), mode="edge")
            padded_height, padded_width = padded.shape
            stride = tile_size - overlap
            tops = list(range(0, max(padded_height - tile_size, 0) + 1, stride))
            lefts = list(range(0, max(padded_width - tile_size, 0) + 1, stride))
            if tops[-1] != padded_height - tile_size:
                tops.append(padded_height - tile_size)
            if lefts[-1] != padded_width - tile_size:
                lefts.append(padded_width - tile_size)
            window_1d = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
            window = np.outer(window_1d, window_1d)
            accumulated = np.zeros_like(padded, dtype=np.float32)
            weights = np.zeros_like(padded, dtype=np.float32)
            diagnostic_accumulated = np.zeros_like(padded, dtype=np.float32)
            positions = [(top, left) for top in tops for left in lefts]
            batch_size = max(1, int(self.config["dsdnet"].get("tile_batch_size", 1)))
            with torch.inference_mode():
                for start in range(0, len(positions), batch_size):
                    batch_positions = positions[start : start + batch_size]
                    tiles = np.stack([
                        padded[top : top + tile_size, left : left + tile_size]
                        for top, left in batch_positions
                    ])
                    observed = torch.from_numpy(tiles).unsqueeze(1).to(device)
                    batch_kernel = kernel.expand(len(batch_positions), -1, -1)
                    output = model(observed, batch_kernel, retain_diagnostics=True, summarize_diagnostics=True)
                    restored_batch = output["restored"]
                    stages = output["stage_states"]
                    summaries = output["diagnostic_summaries"]
                    stage_change = (stages[-1] - stages[-2]).abs().squeeze(1)
                    branch_disagreement = torch.stack([
                        item["prior_disagreement"].squeeze(1) + item["data_disagreement"].squeeze(1)
                        for item in summaries
                    ]).mean(dim=0)
                    diagnostic_batch = torch.exp(-4.0 * stage_change - 2.0 * branch_disagreement)
                    restored_batch = restored_batch.detach().clamp(0, 1).squeeze(1).cpu().numpy()
                    diagnostic_batch = diagnostic_batch.detach().clamp(0, 1).cpu().numpy()
                    for restored_tile, diagnostic_tile, (top, left) in zip(restored_batch, diagnostic_batch, batch_positions):
                        accumulated[top : top + tile_size, left : left + tile_size] += restored_tile * window
                        diagnostic_accumulated[top : top + tile_size, left : left + tile_size] += diagnostic_tile * window
                        weights[top : top + tile_size, left : left + tile_size] += window
            restored = (accumulated / np.maximum(weights, 1e-8))[:height, :width]
            blend_alpha = float(self.config["dsdnet"].get("restoration_blend_alpha", 1.0))
            if not 0.0 < blend_alpha <= 1.0:
                raise ValueError("restoration_blend_alpha must be in (0, 1]")
            # Validation-selected source-preserving residual fusion.  Keeping
            # this explicit in configuration makes the deployed inference
            # path auditable and prevents an unrecorded post-test adjustment.
            model_output = (restored * 255.0).round().astype(np.uint8)
            restored = blend_alpha * restored + (1.0 - blend_alpha) * source
            diagnostic = (diagnostic_accumulated / np.maximum(weights, 1e-8))[:height, :width]
            diagnostic_u8 = (diagnostic * 255.0).round().astype(np.uint8)
            return RestorationResult((restored * 255.0).round().astype(np.uint8), True, "ok", None, str(checkpoint), int(self.config["dsdnet"]["stages"]), diagnostic_u8, {"mode": "stage-stability-and-branch-agreement", "formal_source_support": False, "mean": float(diagnostic.mean()), "display_output": "raw_x8", "ocr_input": f"source_preserving_fusion_alpha_{blend_alpha:.2f}"}, model_output)
        except Exception as error:
            return RestorationResult(image.copy(), False, "restoration_failed", f"{type(error).__name__}: {error}", str(checkpoint))


def make_readability_view(source: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """Create a conservative, non-generative user-facing reading view."""
    import cv2

    base = restored.astype(np.uint8)
    # Estimate slowly varying paper illumination and divide it away. This
    # whitens empty paper without inventing strokes or replacing OCR text.
    background = cv2.GaussianBlur(base, (0, 0), 31)
    flattened = cv2.divide(base, np.maximum(background, 1), scale=245)
    readable = cv2.fastNlMeansDenoising(flattened, None, h=5, templateWindowSize=7, searchWindowSize=21)
    # Thresholding happens only after text and document structures have been
    # localised. Applying one global threshold here breaks faint characters
    # and continuous table rules into disconnected fragments.
    return readable


def make_faithful_content_view(source: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """Suppress paper texture while retaining only image-derived receipt content.

    This is a presentation transform, not OCR reconstruction: it neither draws
    glyphs nor changes logos.  Broad, low-frequency paper shading is removed
    from the recovery while dark content from the source/recovery is retained.
    """
    import cv2

    source_u8 = source.astype(np.uint8)
    restored_u8 = restored.astype(np.uint8)
    background = cv2.GaussianBlur(restored_u8, (0, 0), 31)
    flattened = cv2.divide(restored_u8, np.maximum(background, 1), scale=248)
    flattened = cv2.fastNlMeansDenoising(
        flattened, None, h=4, templateWindowSize=7, searchWindowSize=21
    )

    # Preserve foreground that is measurably darker than its immediate paper
    # neighbourhood. This protects printed text, rules, signatures and solid
    # logos without copying broad fold shadows into the clean background.
    source_background = cv2.GaussianBlur(source_u8, (0, 0), 15)
    source_darkness = np.maximum(
        source_background.astype(np.int16) - source_u8.astype(np.int16), 0
    )
    recovery_background = cv2.GaussianBlur(restored_u8, (0, 0), 15)
    recovery_darkness = np.maximum(
        recovery_background.astype(np.int16) - restored_u8.astype(np.int16), 0
    )
    content_darkness = np.maximum(source_darkness, recovery_darkness).astype(np.float32)
    content = np.clip(255.0 - content_darkness * 1.45, 0, 255).astype(np.uint8)
    content_mask = content_darkness >= 3.0
    faithful = flattened.copy()
    faithful[content_mask] = np.minimum(faithful[content_mask], content[content_mask])
    return faithful


def make_faint_thermal_view(readable: np.ndarray) -> np.ndarray:
    """Preserve faint thermal strokes using continuous, non-binary contrast."""
    import cv2

    source = readable.astype(np.uint8)
    background = cv2.GaussianBlur(source, (0, 0), 31)
    normalised = cv2.divide(source, np.maximum(background, 1), scale=245)
    darkness = np.maximum(245 - normalised.astype(np.int16), 0).astype(np.float32)
    # A soft floor removes minor scanner noise while retaining every measured
    # shade above it. Unlike binarisation, this does not delete weak character
    # fragments or force strong totals and faint labels to the same thickness.
    return np.clip(255.0 - np.maximum(darkness - 2.0, 0.0) * 2.0, 0, 255).astype(np.uint8)


def enhance_supported_source_strokes(
    image: np.ndarray,
    evidence_source: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    minimum_confidence: float = 0.70,
) -> np.ndarray:
    """Strengthen measured ink inside OCR regions without drawing new glyphs."""
    import cv2

    output = image.copy()
    source = evidence_source.astype(np.uint8)
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < minimum_confidence:
            continue
        points = np.asarray(line.get("box", []), dtype=np.float32)
        if points.shape != (4, 2):
            continue
        x, y, width, height = cv2.boundingRect(points.astype(np.int32))
        if width < 4 or height < 4:
            continue
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2, y2 = min(source.shape[1], x + width + 1), min(source.shape[0], y + height + 1)
        roi = source[y1:y2, x1:x2]
        # A local low-frequency estimate separates measured ink strokes from
        # paper tone. The soft response retains anti-aliasing and the receipt's
        # original typeface, spacing, baseline and imperfections.
        local_background = cv2.GaussianBlur(roi, (0, 0), max(2.0, height / 3.0))
        measured_darkness = np.maximum(
            local_background.astype(np.int16) - roi.astype(np.int16), 0
        ).astype(np.float32)
        stroke_response = np.maximum(measured_darkness - 1.0, 0.0)
        # Join only one-pixel discontinuities already bounded by measured ink.
        # Closing cannot invent a word or substitute a glyph; it merely bridges
        # tiny scanner/dropout gaps between neighbouring source-supported pixels.
        closed_horizontal = cv2.morphologyEx(
            stroke_response, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8)
        )
        closed_vertical = cv2.morphologyEx(
            stroke_response, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)
        )
        repaired_response = np.maximum(closed_horizontal, closed_vertical)
        # Keep the repair conservative: a closed pixel may be no stronger than
        # the strongest measured response in its immediate 3x3 neighbourhood.
        local_support = cv2.dilate(stroke_response, np.ones((3, 3), np.uint8))
        repaired_response = np.minimum(repaired_response, local_support)
        boosted = np.clip(255.0 - repaired_response * 3.0, 0, 255).astype(np.uint8)
        output[y1:y2, x1:x2] = np.minimum(output[y1:y2, x1:x2], boosted)
    return output


def overlay_supported_ocr_text(image: np.ndarray, ocr_lines: list[dict[str, Any]], minimum_confidence: float = 0.85) -> tuple[np.ndarray, int]:
    """Render only high-confidence OCR readings over their measured regions."""
    import cv2
    from .language_assistance import suggest_receipt_text

    def bounds(line):
        points = np.asarray(line.get("box", []), dtype=np.float32)
        if points.shape != (4, 2): return None
        return float(points[:,0].min()), float(points[:,1].min()), float(points[:,0].max()), float(points[:,1].max())

    def same_line(first, second):
        a, b = bounds(first), bounds(second)
        if a is None or b is None: return False
        aw, ah = max(a[2]-a[0],1), max(a[3]-a[1],1); bw, bh = max(b[2]-b[0],1), max(b[3]-b[1],1)
        vertical_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        horizontal_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        same_band = vertical_overlap / min(ah, bh) >= 0.55
        same_span = horizontal_overlap / min(aw, bw) >= 0.35
        centres_close = abs((a[0]+a[2]-b[0]-b[2])/2) <= max(12,.28*max(aw,bw))
        return same_band and (same_span or centres_close)

    # OCR views often disagree by one character. Group by geometry first so
    # only one reading is typeset at each physical line.
    groups: list[list[dict[str, Any]]] = []
    for line in ocr_lines:
        group = next((item for item in groups if same_line(item[0], line)), None)
        if group is None:
            groups.append([line])
        else:
            group.append(line)
    selected = []
    for group in groups:
        selected.append(max(group, key=lambda item: float(item.get("confidence") or 0.0)))

    output = image.copy()
    valid_heights = [max(1.0, (box[3] - box[1])) for item in selected if (box := bounds(item)) is not None]
    median_height = float(np.median(valid_heights)) if valid_heights else 18.0
    rendered = 0
    occupied: list[tuple[int, int, int, int]] = []
    for line in sorted(selected, key=lambda item: (bounds(item) or (0,0,0,0))[1]):
        text = str(line.get("text", "")).strip()
        confidence = float(line.get("confidence") or 0.0)
        points = np.asarray(line.get("box", []), dtype=np.float32)
        if confidence < minimum_confidence or points.shape != (4, 2) or not text:
            continue
        suggestion = suggest_receipt_text(text)
        if suggestion:
            text = suggestion["suggested"]
        # Do not typeset isolated uncertain punctuation or extremely small
        # fragments; these are more likely to be paper damage than content.
        if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 2:
            continue
        x, y, width, height = cv2.boundingRect(points.astype(np.int32))
        if width < 12 or height < 8:
            continue
        # Thermal OCR commonly drops the decimal point in monetary columns.
        # Apply cents formatting only to pure numeric readings located in the
        # right half and lower half of the receipt; IDs and dates are excluded.
        if re.fullmatch(r"\d{3,7}", text) and x >= int(output.shape[1] * .45) and y >= int(output.shape[0] * .45):
            text = text[:-2] + "." + text[-2:]
        pad_x, pad_y = 2, 2
        x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
        x2, y2 = min(output.shape[1], x+width+pad_x), min(output.shape[0], y+height+pad_y)
        # Preserve nearby rules by clearing only the central text band.
        output[y1:y2, x1:x2] = 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        # A noisy OCR box can span two thermal rows. Cap glyph height using the
        # document median so one bad box cannot typeset oversized, overlapping
        # text (for example Gross Amt / Svr Charge / Discount).
        target_height = max(8, min(int(height * 0.62), int(median_height * 0.78)))
        scale = max(0.25, target_height / 22.0)
        thickness = 1 if target_height < 22 else 2
        measured = cv2.getTextSize(text, font, scale, thickness)[0]
        if measured[0] > max(width, 1):
            scale *= width / measured[0]
            measured = cv2.getTextSize(text, font, scale, thickness)[0]
        # Preserve the detected line centre instead of left-anchoring a
        # narrower substitute font. This keeps columns and labels aligned with
        # their positions in the evidence image.
        draw_x = max(0, min(output.shape[1] - measured[0], x + (width - measured[0]) // 2))
        draw_top = max(0, y + (height - measured[1]) // 2)
        baseline_y = min(output.shape[0]-2, draw_top + measured[1])
        text_rect = (draw_x, max(0, baseline_y-measured[1]), min(output.shape[1], draw_x+measured[0]), min(output.shape[0], baseline_y+3))
        collision = any(
            max(0, min(text_rect[2], old[2]) - max(text_rect[0], old[0]))
            * max(0, min(text_rect[3], old[3]) - max(text_rect[1], old[1]))
            > 0.20 * max(1, (text_rect[2]-text_rect[0]) * (text_rect[3]-text_rect[1]))
            for old in occupied
        )
        if collision:
            continue
        cv2.putText(output, text, (draw_x, baseline_y), font, scale, 25, thickness, cv2.LINE_AA)
        occupied.append(text_rect)
        rendered += 1
    return output, rendered


def make_adaptive_document_view(readable: np.ndarray, ocr_lines: list[dict[str, Any]], evidence_source: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Choose clean-document or faint-thermal rendering from measured contrast."""
    import cv2

    source = readable.astype(np.uint8)
    detection_source = (evidence_source if evidence_source is not None else source).astype(np.uint8)
    background = cv2.GaussianBlur(detection_source, (0, 0), 31)
    normalised = cv2.divide(detection_source, np.maximum(background, 1), scale=245)
    faint_thermal = float(normalised.std()) < 25.0 and float(np.mean(normalised < 200)) < 0.03
    if faint_thermal:
        # Start from the rectified evidence for faint thermal paper. Model-led
        # cleanup can amplify fold edges on this material; authentic source
        # strokes are strengthened conservatively below instead.
        thermal = make_faint_thermal_view(evidence_source if evidence_source is not None else source)
        if evidence_source is not None and ocr_lines:
            thermal = enhance_supported_source_strokes(thermal, evidence_source, ocr_lines)
        return thermal, "faint_thermal_source_enhanced"
    return make_evidence_preserving_view(source, ocr_lines), "component_aware_document"


def make_evidence_preserving_view(readable: np.ndarray, ocr_lines: list[dict[str, Any]]) -> np.ndarray:
    """Create an edge-supported, background-normalised reading copy."""
    import cv2
    from .baselines import sauvola

    source = readable.astype(np.uint8)
    denoised = cv2.bilateralFilter(source, 7, 24, 24)
    background = cv2.morphologyEx(
        denoised, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    )
    normalised = cv2.divide(denoised, np.maximum(background, 1), scale=245)
    local_foreground = sauvola(normalised, window=41, k=0.17) == 0
    edges = cv2.Canny(normalised, 22, 70, L2gradient=True)
    edge_support = cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0
    candidates = local_foreground & edge_support

    # OCR is used only to protect known text-bearing regions from broad-mark
    # rejection; it never creates foreground on its own.
    localised = np.zeros_like(source, dtype=np.uint8)
    weak_mark_regions = np.zeros_like(source, dtype=np.uint8)
    ocr_foreground = np.zeros_like(source, dtype=bool)
    for line in ocr_lines:
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape == (4, 2):
            cv2.fillPoly(localised, [points], 255)
            x, y, width, height = cv2.boundingRect(points)
            x2, y2 = min(source.shape[1], x + width), min(source.shape[0], y + height)
            x, y = max(0, x), max(0, y)
            if x2 > x and y2 > y:
                region = normalised[y:y2, x:x2]
                polygon = np.zeros_like(region, dtype=np.uint8)
                shifted = points - np.array([x, y], dtype=np.int32)
                cv2.fillPoly(polygon, [shifted], 255)
                values = region[polygon > 0]
                if values.size:
                    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    threshold = min(float(threshold), 205.0)
                    region_mask = (region <= threshold) & (polygon > 0)
                    # Closing helps genuinely broken large glyphs, but it
                    # thickens small totals/footer text. Apply it only where
                    # the measured text region is tall enough to tolerate it.
                    if height >= 32:
                        region_mask = cv2.morphologyEx(region_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
                    ocr_foreground[y:y2, x:x2] |= region_mask
                line_text = str(line.get("text", ""))
                confidence = float(line.get("confidence") or 0.0)
                weak_mark_hint = confidence < 0.90 or bool(re.search(r"(?i)stamp|chop|signature|authori[sz]ed|received|issued|paid", line_text))
                if weak_mark_hint:
                    cv2.fillPoly(weak_mark_regions, [points], 255)
    candidates |= ocr_foreground

    # Recover coherent weak marks that OCR does not localise, such as stamps
    # and signatures. Component filtering keeps this path local: small paper
    # grain and page-spanning folds are not promoted to document ink.
    weak_binary = cv2.adaptiveThreshold(normalised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 11)
    # Expand only around OCR-indicated weak marks. This recovers surrounding
    # seal rings and signature strokes that are not themselves recognised as
    # text, without lowering the threshold across the whole page.
    weak_mark_regions = cv2.dilate(weak_mark_regions, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))) > 0
    candidates |= (weak_binary > 0) & weak_mark_regions
    # Detect a coherent circular seal independently of OCR. Requiring a
    # lower-page circle of realistic size and measured circumference support
    # avoids mistaking rectangular totals tables or random texture for stamps.
    seal_regions = np.zeros_like(source, dtype=np.uint8)
    circle_source = cv2.medianBlur(source, 5)
    circles = cv2.HoughCircles(
        circle_source, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
        param1=80, param2=30, minRadius=35, maxRadius=80,
    )
    if circles is not None:
        edge_map = cv2.Canny(circle_source, 40, 100)
        yy, xx = np.ogrid[:source.shape[0], :source.shape[1]]
        accepted = 0
        for circle_x, circle_y, radius in np.round(circles[0]).astype(int):
            if circle_y < int(source.shape[0] * 0.55):
                continue
            distance = np.sqrt((xx - circle_x) ** 2 + (yy - circle_y) ** 2)
            annulus = np.abs(distance - radius) <= 2
            support = float(np.count_nonzero(edge_map[annulus])) / max(int(np.count_nonzero(annulus)), 1)
            if support >= 0.24:
                cv2.circle(seal_regions, (circle_x, circle_y), int(radius * 1.18), 255, thickness=cv2.FILLED)
                accepted += 1
                if accepted >= 2:
                    break
    candidates |= (weak_binary > 0) & (seal_regions > 0)
    localised[seal_regions > 0] = 255
    weak_count, weak_labels, weak_stats, _ = cv2.connectedComponentsWithStats(weak_binary, 8)
    weak_structures = np.zeros_like(candidates)
    for weak_index in range(1, weak_count):
        x, y, width, height, area = [int(value) for value in weak_stats[weak_index]]
        if area < 24 or area > int(source.size * 0.02):
            continue
        if width > int(source.shape[1] * 0.55) or height > int(source.shape[0] * 0.20):
            continue
        component = weak_labels == weak_index
        median_tone = float(np.median(normalised[component]))
        if median_tone < 165.0 and x > 1 and y > 1 and x + width < source.shape[1] - 1 and y + height < source.shape[0] - 1:
            weak_structures[component] = True
    candidates |= weak_structures
    localised = cv2.dilate(localised, np.ones((15, 15), np.uint8)) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), 8)
    foreground = np.zeros_like(candidates)
    image_width = source.shape[1]
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        component = labels == index
        protected = bool(np.any(localised & component))
        # Only suppress page-spanning, thick unlabelled bands. A lower width
        # cutoff wrongly rejects connected words and nearby field labels.
        median_tone = float(np.median(normalised[component]))
        broad_unlabelled_mark = (
            width > int(0.60 * image_width)
            and height > 2
            and median_tone > 155.0
            and not protected
        )
        # Outside OCR-supported regions, require a materially dark and
        # substantial component.  This retains handwriting, stamps and logos
        # while rejecting the pale fragments produced by paper grain/folds.
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        touches_page_edge = x <= 1 or y <= 1 or x + width >= source.shape[1] - 1 or y + height >= source.shape[0] - 1
        independently_supported = area >= 12 and median_tone < 135.0 and not touches_page_edge
        if (protected or independently_supported) and not broad_unlabelled_mark:
            foreground[component] = True

    # Recover genuinely thin form rules that local thresholding can miss,
    # while excluding multi-row/column crease bands.
    horizontal_response = cv2.morphologyEx(
        255 - normalised, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    ) > 35
    vertical_response = cv2.morphologyEx(
        255 - normalised, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
    ) > 35
    # A printed rule remains confined to one or two rows/columns. A fold may
    # contain locally straight segments, but its complete connected response
    # wanders across several rows/columns. Classify the complete structure so
    # those fragments cannot be accidentally painted back into the clean view.
    thin_rules = np.zeros_like(foreground)
    for response, horizontal in ((horizontal_response, True), (vertical_response, False)):
        response_count, response_labels, response_stats, _ = cv2.connectedComponentsWithStats(
            response.astype(np.uint8), 8
        )
        for response_index in range(1, response_count):
            response_width = int(response_stats[response_index, cv2.CC_STAT_WIDTH])
            response_height = int(response_stats[response_index, cv2.CC_STAT_HEIGHT])
            is_uniformly_thin = response_height <= 2 if horizontal else response_width <= 2
            is_long_enough = response_width >= 21 if horizontal else response_height >= 21
            if is_uniformly_thin and is_long_enough:
                thin_rules[response_labels == response_index] = True
    foreground |= thin_rules

    # Preserve genuinely solid source graphics (for example merchant logos)
    # as graphics rather than treating their interior as paper texture. A
    # small closing operation repairs pinholes, while the minimum-area gate
    # prevents background speckles from becoming new foreground.
    solid_seed = (source < 145).astype(np.uint8)
    solid_seed = cv2.morphologyEx(
        solid_seed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    solid_count, solid_labels, solid_stats, _ = cv2.connectedComponentsWithStats(solid_seed, 8)
    minimum_graphic_area = max(200, int(source.size * 0.0010))
    solid_graphics = np.zeros_like(source, dtype=np.uint8)
    for solid_index in range(1, solid_count):
        area = int(solid_stats[solid_index, cv2.CC_STAT_AREA])
        width = int(solid_stats[solid_index, cv2.CC_STAT_WIDTH])
        height = int(solid_stats[solid_index, cv2.CC_STAT_HEIGHT])
        fill_ratio = area / max(width * height, 1)
        component = solid_labels == solid_index
        ocr_overlap = float(np.count_nonzero(localised & component)) / max(area, 1)
        if area >= minimum_graphic_area and width >= 20 and height >= 20 and fill_ratio >= 0.35 and ocr_overlap < 0.05:
            component_u8 = component.astype(np.uint8)
            contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(solid_graphics, contours, -1, 1, thickness=cv2.FILLED)
    foreground |= solid_graphics > 0

    output = np.full_like(source, 255)
    # Use measured grayscale stroke strength so small text is not thickened by
    # the same rendering rule as solid graphics. Logos alone are filled solid.
    ink_tones = np.clip((normalised.astype(np.float32) - 105.0) * 2.35, 0, 225).astype(np.uint8)
    output[foreground] = ink_tones[foreground]
    output[thin_rules] = np.minimum(output[thin_rules], 35)
    output[solid_graphics > 0] = 0
    return output
