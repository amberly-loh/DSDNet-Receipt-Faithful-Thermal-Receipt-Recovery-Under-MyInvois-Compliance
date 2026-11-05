from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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


@dataclass
class FaithfulnessDecision:
    accepted: bool
    image: np.ndarray
    reason: str
    metrics: dict[str, float]


def region_adaptive_source_fusion(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    tile_size: int = 96,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select DSDNet only in local regions where it beats the source safely.

    The selector is deliberately conservative. Readable source regions remain
    untouched; candidate tiles must improve local contrast while retaining
    source edges and avoiding unsupported ink. No OCR text is used or drawn.
    """
    import cv2

    source_u8 = source.astype(np.uint8)
    candidate_u8 = candidate.astype(np.uint8)
    if source_u8.shape != candidate_u8.shape:
        return source_u8.copy(), {"selected_tiles": 0, "total_tiles": 0, "reason": "geometry_mismatch"}
    height, width = source_u8.shape
    output = source_u8.copy()
    selected = 0
    total = 0
    decisions: list[dict[str, Any]] = []
    for top in range(0, height, tile_size):
        for left in range(0, width, tile_size):
            bottom, right = min(top + tile_size, height), min(left + tile_size, width)
            src = source_u8[top:bottom, left:right]
            cand = candidate_u8[top:bottom, left:right]
            total += 1
            src_bg = cv2.GaussianBlur(src, (0, 0), 9)
            cand_bg = cv2.GaussianBlur(cand, (0, 0), 9)
            src_darkness = np.maximum(src_bg.astype(np.int16) - src.astype(np.int16), 0)
            cand_darkness = np.maximum(cand_bg.astype(np.int16) - cand.astype(np.int16), 0)
            src_ink = src_darkness >= 8
            cand_ink = cand_darkness >= 8
            support = cv2.dilate(src_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            unsupported = float(np.mean(cand_ink & ~support))
            src_edges = cv2.Canny(src, 40, 110) > 0
            cand_edges = cv2.dilate((cv2.Canny(cand, 40, 110) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            edge_retention = float(np.count_nonzero(src_edges & cand_edges) / max(np.count_nonzero(src_edges), 1))
            src_strength = float(np.mean(src_darkness[src_ink])) if src_ink.any() else 0.0
            cand_strength = float(np.mean(cand_darkness[src_ink])) if src_ink.any() else 0.0
            dark_ratio = float(cand_ink.sum() / max(int(src_ink.sum()), 1))
            has_content = int(src_ink.sum()) >= max(8, int(src.size * 0.002))
            improves = cand_strength >= src_strength * 1.04 + 0.5
            safe = unsupported <= 0.006 and 0.72 <= dark_ratio <= 1.32 and edge_retention >= 0.84
            use_candidate = bool(has_content and improves and safe)
            if use_candidate:
                # Change only source-supported neighbourhoods; paper remains
                # source-identical even inside an accepted tile.
                region = output[top:bottom, left:right]
                region[support] = cand[support]
                selected += 1
            decisions.append({
                "top": top, "left": left, "selected": use_candidate,
                "unsupported_dark_fraction": unsupported,
                "edge_retention": edge_retention,
                "dark_area_ratio": dark_ratio,
                "source_strength": src_strength,
                "candidate_strength": cand_strength,
            })
    return output, {
        "mode": "source_constrained_region_adaptive_selection",
        "tile_size": tile_size,
        "selected_tiles": selected,
        "total_tiles": total,
        "selected_fraction": float(selected / max(total, 1)),
        "decisions": decisions,
    }


def select_faithful_recovery(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    maximum_changed_fraction: float = 0.35,
    maximum_unsupported_dark_fraction: float = 0.012,
    maximum_dark_area_ratio: float = 1.45,
    minimum_source_edge_retention: float = 0.72,
) -> FaithfulnessDecision:
    """Accept recovery only when its changes remain supported by the source."""
    import cv2

    source_u8 = source.astype(np.uint8)
    candidate_u8 = candidate.astype(np.uint8)
    if source_u8.shape != candidate_u8.shape:
        return FaithfulnessDecision(False, source_u8.copy(), "geometry_mismatch", {})
    delta = np.abs(candidate_u8.astype(np.int16) - source_u8.astype(np.int16))
    changed_fraction = float(np.mean(delta >= 24))
    source_background = cv2.GaussianBlur(source_u8, (0, 0), 15)
    candidate_background = cv2.GaussianBlur(candidate_u8, (0, 0), 15)
    source_dark = (source_background.astype(np.int16) - source_u8.astype(np.int16)) >= 10
    candidate_dark = (candidate_background.astype(np.int16) - candidate_u8.astype(np.int16)) >= 10
    source_support = cv2.dilate(source_dark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    unsupported_dark_fraction = float(np.mean(candidate_dark & ~source_support))
    dark_area_ratio = float(candidate_dark.sum() / max(int(source_dark.sum()), 1))
    source_edges = cv2.Canny(source_u8, 45, 120) > 0
    candidate_edges = cv2.dilate(
        (cv2.Canny(candidate_u8, 45, 120) > 0).astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ) > 0
    source_edge_retention = float(
        np.count_nonzero(source_edges & candidate_edges) / max(np.count_nonzero(source_edges), 1)
    )
    metrics = {
        "changed_fraction": changed_fraction,
        "unsupported_dark_fraction": unsupported_dark_fraction,
        "dark_area_ratio": dark_area_ratio,
        "source_edge_retention": source_edge_retention,
    }
    failures = []
    if changed_fraction > maximum_changed_fraction:
        failures.append("excessive_pixel_change")
    if unsupported_dark_fraction > maximum_unsupported_dark_fraction:
        failures.append("unsupported_dark_strokes")
    if dark_area_ratio > maximum_dark_area_ratio:
        failures.append("stroke_area_inflation")
    if source_edge_retention < minimum_source_edge_retention:
        failures.append("source_edge_loss")
    if failures:
        return FaithfulnessDecision(False, source_u8.copy(), ",".join(failures), metrics)
    return FaithfulnessDecision(True, candidate_u8.copy(), "accepted", metrics)


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
            from .evidence_support import SupportScales, evidence_support_from_output

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
            calibration_path = Path(self.config["dsdnet"].get("evidence_support_calibration", ""))
            if calibration_path.is_file():
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                scales = SupportScales(**calibration["scales"])
                support_threshold = float(calibration["decision_threshold"])
            else:
                scales = SupportScales(.32, .32, .32)
                support_threshold = None
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
                    diagnostic_batch = evidence_support_from_output(
                        model, observed, batch_kernel, output, scales
                    )["combined"].squeeze(1)
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
            return RestorationResult((restored * 255.0).round().astype(np.uint8), True, "ok", None, str(checkpoint), int(self.config["dsdnet"]["stages"]), diagnostic_u8, {"mode": "calibrated-residual-stage-branch-evidence", "formal_source_support": False, "natural_source_gate": "human_review_required", "validation_threshold_reference": support_threshold, "mean": float(diagnostic.mean()), "display_output": "raw_x8", "ocr_input": f"source_preserving_fusion_alpha_{blend_alpha:.2f}"}, model_output)
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


def assess_receipt_degradation(image: np.ndarray) -> dict[str, Any]:
    """Classify natural receipt degradation using source-only measurements."""
    import cv2

    source = image.astype(np.uint8)
    paper = cv2.GaussianBlur(source, (0, 0), 31)
    normalised = cv2.divide(source, np.maximum(paper, 1), scale=245)
    darkness = np.maximum(paper.astype(np.int16) - source.astype(np.int16), 0)
    ink = darkness >= 5
    contrast_std = float(normalised.std())
    ink_fraction = float(np.mean(ink))
    laplacian_variance = float(cv2.Laplacian(normalised, cv2.CV_32F).var())
    strong_ink = darkness >= 18
    protected_ink = cv2.dilate(strong_ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    broad_paper = cv2.GaussianBlur(source, (0, 0), 61)
    low_frequency_residual = cv2.absdiff(paper, broad_paper)
    background_pixels = ~protected_ink
    texture_fraction = float(
        np.mean(low_frequency_residual[background_pixels] >= 5)
    ) if background_pixels.any() else 0.0
    stroke_strength = float(np.percentile(darkness[ink], 75)) if ink.any() else 0.0
    edges = cv2.Canny(source, 35, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=35,
        minLineLength=max(30, min(source.shape) // 5), maxLineGap=8,
    )
    crease_lines = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            if angle <= 12 or angle >= 78:
                crease_lines += 1
    faded = contrast_std < 18.0 or stroke_strength < 22.0
    blurred = laplacian_variance < 310.0
    creased = crease_lines >= 3 or texture_fraction >= 0.08
    flags = [name for name, active in (("faded", faded), ("blurred", blurred), ("creased", creased)) if active]
    profile = "mixed" if len(flags) >= 2 else flags[0] if flags else "clear"
    return {
        "profile": profile,
        "flags": flags,
        "metrics": {
            "normalised_contrast_std": contrast_std,
            "ink_fraction": ink_fraction,
            "stroke_strength_p75": stroke_strength,
            "laplacian_variance": laplacian_variance,
            "texture_fraction": texture_fraction,
            "long_crease_line_count": crease_lines,
        },
        "policy": {
            "target_darkness": 72.0 if faded else (66.0 if blurred else 58.0),
            "sharpen_strength": 0.55 if blurred else 0.0,
            # Blur means weak character edges are fragile even when global
            # contrast looks adequate. Preserve them before attacking folds.
            "paper_cleaning": "gentle" if blurred else ("strong" if creased else "standard"),
            "text_protection": "wide" if blurred or faded else "standard",
        },
    }


def complete_document_preservation_mode(source: np.ndarray) -> str | None:
    import cv2
    if source.ndim == 3:
        chroma = np.max(source, axis=2).astype(np.int16) - np.min(source, axis=2)
        if float(np.mean(chroma > 12)) > 0.005:
            return "registered_color_preservation"
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    else:
        gray = source
    if gray.shape[1] / gray.shape[0] >= 0.70:
        return "broad_document_preservation"
    # Work at a fixed detection scale: a slight slope across a high-resolution
    # scan otherwise breaks long one-pixel morphology kernels.
    detection_scale = min(1.0, 600.0 / max(gray.shape))
    if detection_scale < 1.0:
        gray = cv2.resize(gray, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    rules = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    horizontal = cv2.morphologyEx(rules, cv2.MORPH_OPEN, np.ones((1, max(25, gray.shape[1] // 8)), np.uint8))
    vertical = cv2.morphologyEx(rules, cv2.MORPH_OPEN, np.ones((max(25, gray.shape[0] // 8), 1), np.uint8))
    crossings = cv2.dilate(horizontal, np.ones((3, 3), np.uint8)) & cv2.dilate(vertical, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(crossings, 8)
    if sum(int(area) >= 2 for area in stats[1:, cv2.CC_STAT_AREA]) >= 6:
        return "table_content_preservation"
    return None


def make_faithful_content_view(
    source: np.ndarray,
    restored: np.ndarray,
    ocr_lines: list[dict[str, Any]] | None = None,
    artifact_mask_checkpoint: str | Path | None = None,
    artifact_mask_threshold: float = 0.95,
    degradation_profile: dict[str, Any] | None = None,
    source_color: np.ndarray | None = None,
    native_refinement: dict[str, Any] | None = None,
    refinement_audit: dict[str, Any] | None = None,
) -> np.ndarray:
    """Suppress paper texture while retaining only image-derived receipt content.

    This is a presentation transform, not OCR reconstruction: it neither draws
    glyphs nor changes logos.  Broad, low-frequency paper shading is removed
    from the recovery while dark content from the source/recovery is retained.
    """
    import cv2

    source_u8 = source.astype(np.uint8)
    if source_color is not None:
        if source_color.shape[:2] != source_u8.shape:
            raise ValueError("Faithful color source must share the model input geometry")
    preservation_source = source_color if source_color is not None else source_u8
    source_detail = np.maximum(cv2.GaussianBlur(source_u8,(0,0),3).astype(np.float32)-source_u8,0)
    neutral_source = source_color is None or float(np.mean(np.ptp(source_color.astype(np.int16),axis=2)>12))<=.005
    if neutral_source and float(np.median(source_u8))<170 and float(np.percentile(source_detail,95))<6:
        # Underexposed, extremely weak print cannot support aggressive paper
        # subtraction. Preserve its complete measured ordering and shapes via
        # a single exposure mapping instead of showing washed-out fragments.
        paper_pixels=source_u8[source_u8<245]
        reference=float(np.percentile(paper_pixels,90)) if paper_pixels.size else 245.0
        return np.clip(250+(source_u8.astype(np.float32)-reference)*1.8,0,255).round().astype(np.uint8)
    preservation_mode = complete_document_preservation_mode(preservation_source)
    if preservation_mode:
        small_broad_print = preservation_mode == 'broad_document_preservation' and source_u8.shape[1] < 500
        complete = make_complete_document_view(preservation_source, thermal=small_broad_print, ocr_lines=ocr_lines)
        # Structured invoices used to return before source-supported stroke
        # repair.  Repair only tiny gaps bracketed by measured endpoints and
        # independently supported by the model; OCR supplies geometry, never
        # replacement characters.
        if complete.ndim == 2:
            complete = repair_evidence_constrained_gaps(
                complete, source_u8, restored.astype(np.uint8), ocr_lines or []
            )
            complete = normalize_localized_source_stroke_contrast(
                complete, source_u8, ocr_lines or [],
                target_darkness=72.0 if small_broad_print else 64.0,
                sharpen_strength=0.65,
            )
            complete = sharpen_measured_print_edges(complete)
        return complete
    if source_u8.shape[1] / source_u8.shape[0] < 0.70:
        # Use one continuous tone transfer for thermal print. Repeated
        # component-wise darkness floors made only selected words bold.
        continuous = make_complete_document_view(source_u8, thermal=True, ocr_lines=ocr_lines)
        continuous = repair_evidence_constrained_gaps(
            continuous, source_u8, restored.astype(np.uint8), ocr_lines or []
        )
        continuous = protect_machine_readable_regions(continuous, source_u8)
        if native_refinement:
            from .native_refinement import refine_native_print
            continuous, record = refine_native_print(source_u8, continuous, native_refinement)
            if refinement_audit is not None:
                refinement_audit.update(record)
        return continuous
    restored_u8 = restored.astype(np.uint8)
    degradation_profile = degradation_profile or assess_receipt_degradation(source_u8)
    profile_policy = degradation_profile.get("policy", {})
    target_darkness = float(profile_policy.get("target_darkness", 58.0))
    sharpen_strength = float(profile_policy.get("sharpen_strength", 0.0))
    paper_cleaning = str(profile_policy.get("paper_cleaning", "standard"))
    text_protection = str(profile_policy.get("text_protection", "standard"))
    degradation_flags = set(degradation_profile.get("flags", []))
    degradation_metrics = degradation_profile.get("metrics", {})
    severe_texture = (
        float(degradation_metrics.get("texture_fraction", 0.0)) >= 0.15
        and int(degradation_metrics.get("long_crease_line_count", 0)) >= 10
    )
    mixed_faint_fold = (
        {"faded", "blurred", "creased"}.issubset(degradation_flags)
        and float(degradation_metrics.get("texture_fraction", 0.0)) >= 0.10
        and int(degradation_metrics.get("long_crease_line_count", 0)) >= 10
    )
    if severe_texture or mixed_faint_fold:
        conservative = make_conservative_creased_view(source_u8, restored_u8, ocr_lines or [])
        return select_faithful_display_candidate(source_u8, conservative, ocr_lines or [])
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

    # On faint thermal paper, long fold edges can be darker than the printed
    # ink and therefore survive ordinary background flattening. Identify only
    # long, near-horizontal bands and attenuate them outside OCR-localised text
    # regions. Intersections with text remain untouched: filling those pixels
    # would be reconstruction rather than faithful enhancement.
    normalised_source = cv2.divide(
        source_u8, np.maximum(cv2.GaussianBlur(source_u8, (0, 0), 31), 1), scale=245
    )
    faint_thermal = (
        float(normalised_source.std()) < 25.0
        and float(np.mean(normalised_source < 200)) < 0.03
    )
    # Strongly creased receipts must not use the model-derived content map:
    # folds and reverse-side print can receive a stronger response than ink.
    # Route them through source-supported character geometry even when their
    # global contrast prevents classification as faint thermal paper.
    if faint_thermal or "creased" in degradation_flags:
        # Do not feed the learned recovery's fold response into the displayed
        # faithful view. Start from the source-normalised paper instead, then
        # strengthen only measured strokes inside confident OCR regions. This
        # reduces crease emphasis without painting across hidden characters.
        source_flattened = cv2.divide(
            source_u8,
            np.maximum(cv2.GaussianBlur(source_u8, (0, 0), 31), 1),
            scale=248,
        )
        source_flattened = cv2.fastNlMeansDenoising(
            source_flattened, None,
            h=2 if "blurred" in degradation_flags else 4,
            templateWindowSize=7, searchWindowSize=21,
        )
        source_supported = enhance_supported_source_strokes(
            source_flattened, source_u8, ocr_lines or [], minimum_confidence=0.70
        )
        if artifact_mask_checkpoint and Path(artifact_mask_checkpoint).is_file():
            source_supported = suppress_learned_paper_artifacts(
                source_supported,
                source_u8,
                ocr_lines or [],
                artifact_mask_checkpoint,
                artifact_mask_threshold,
            )
        source_supported = repair_evidence_constrained_gaps(
            source_supported, source_u8, restored_u8, ocr_lines or []
        )
        source_supported = suppress_unprotected_paper_texture(
            source_supported, source_u8, ocr_lines or [], cleaning=paper_cleaning,
        )
        source_supported = preserve_faint_source_supported_strokes(
            source_supported, source_u8, restored_u8, ocr_lines or [],
            target_darkness=target_darkness, protection=text_protection,
        )
        source_supported = normalize_localized_source_stroke_contrast(
            source_supported, source_u8, ocr_lines or [], target_darkness=target_darkness,
            sharpen_strength=sharpen_strength,
        )
        final = apply_line_level_source_fallback(
            source_supported, source_u8, restored_u8, ocr_lines or [],
            target_darkness=target_darkness,
        )
        final = apply_local_quality_fallback(final, source_u8)
        # The local safety fallback can blend a faint source tone back over a
        # correctly localised character. Re-apply bounded source-supported ink
        # contrast last so paper decisions never wash out readable strokes.
        final = preserve_faint_source_supported_strokes(
            final, source_u8, restored_u8, ocr_lines or [],
            target_darkness=min(target_darkness + 8.0, 78.0),
            protection="wide",
        )
        final = normalize_localized_source_stroke_contrast(
            final, source_u8, ocr_lines or [],
            target_darkness=min(target_darkness + 6.0, 76.0),
            sharpen_strength=sharpen_strength,
        )
        final = balance_source_text_tone(
            final, source_u8, ocr_lines or [], target_darkness=160.0
        )
        final = protect_machine_readable_regions(final, source_u8)
        final = apply_structural_coherence_fallback(final, source_u8)
        return select_faithful_display_candidate(source_u8, final, ocr_lines or [])
    # Character-gap repair must not be limited to the faint-thermal branch.
    # Ordinary receipts can also contain one-pixel stroke dropouts even when
    # their global contrast is high. Use the darker measured response from the
    # source/model only inside OCR-localised regions; OCR text itself is never
    # rendered into this faithful view.
    evidence = np.minimum(source_u8, restored_u8)
    faithful = enhance_supported_source_strokes(
        faithful, evidence, ocr_lines or [], minimum_confidence=0.70,
        maximum_gap=0,
    )
    faithful = repair_evidence_constrained_gaps(
        faithful, source_u8, restored_u8, ocr_lines or []
    )
    faithful = suppress_unprotected_paper_texture(
        faithful, source_u8, ocr_lines or [], cleaning=paper_cleaning,
    )
    faithful = preserve_faint_source_supported_strokes(
        faithful, source_u8, restored_u8, ocr_lines or [],
        target_darkness=target_darkness, protection=text_protection,
    )
    faithful = normalize_localized_source_stroke_contrast(
        faithful, source_u8, ocr_lines or [], target_darkness=target_darkness,
        sharpen_strength=sharpen_strength,
    )
    final = apply_line_level_source_fallback(
        faithful, source_u8, restored_u8, ocr_lines or [],
        target_darkness=target_darkness,
    )
    final = apply_local_quality_fallback(final, source_u8)
    final = preserve_faint_source_supported_strokes(
        final, source_u8, restored_u8, ocr_lines or [],
        target_darkness=min(target_darkness + 8.0, 78.0),
        protection="wide",
    )
    final = normalize_localized_source_stroke_contrast(
        final, source_u8, ocr_lines or [],
        target_darkness=min(target_darkness + 6.0, 76.0),
        sharpen_strength=sharpen_strength,
    )
    final = balance_source_text_tone(
        final, source_u8, ocr_lines or [], target_darkness=160.0
    )
    final = protect_machine_readable_regions(final, source_u8)
    final = apply_structural_coherence_fallback(final, source_u8)
    return select_faithful_display_candidate(source_u8, final, ocr_lines or [])


def make_complete_document_view(source: np.ndarray, *, thermal: bool = False, ocr_lines: list[dict[str, Any]] | None = None) -> np.ndarray:
    """Continuous luminance enhancement for colored or ruled documents.

    Every source pixel remains represented; no OCR or component whitelist
    determines whether a logo, signature, rule or script is allowed to exist.
    """
    import cv2

    color = source.ndim == 3
    lab = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB) if color else None
    luminance = lab[:, :, 0] * 2.55 if color else source.astype(np.float32)
    raw_luminance=luminance.copy()
    native_small_print = (thermal and min(luminance.shape) < 250
                          and float(np.percentile(luminance, 85) - np.percentile(luminance, 5)) < 90.0)
    # At native small-print resolution a stroke can be only one pixel wide.
    # Patch denoising averages those faint strokes together with the paper.
    if not native_small_print:
        luminance = cv2.fastNlMeansDenoising(luminance.round().astype(np.uint8), None, h=3, templateWindowSize=7, searchWindowSize=17).astype(np.float32)
    # Estimate paper from local upper intensities, rather than from the mean
    # of ink and paper. Mean-field normalization leaves mottled grey islands
    # around dense handwriting and table intersections.
    kernel_size = max(31, min(101, int(min(luminance.shape) / 8) | 1))
    if thermal:
        kernel_size = 31
    if native_small_print:
        # A 31px paper estimate spans several native text rows and leaves
        # crease shadows behind. Five pixels span a small-print stroke while
        # following the local paper illumination between adjacent characters.
        kernel_size = 5
    elif not color or float(np.mean(np.ptp(source.astype(np.int16), axis=2) > 12)) <= .005:
        # Resolve paper at the stroke neighbourhood scale. A page-wide closing
        # kernel retained fold shadows and reverse-print clouds as dark ink.
        kernel_size = max(9,min(31,(int(min(luminance.shape)/80)|1)))
    paper = cv2.morphologyEx(luminance, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    neutral_paper = not color or float(np.mean(np.ptp(source.astype(np.int16),axis=2)>12))<=.005
    paper = cv2.GaussianBlur(paper, (0, 0), 0.5 if native_small_print else (1.0 if neutral_paper else 5.0))
    # Solid logos must never become their own paper estimate.
    # White scanner margins/rotation padding must not set the paper floor for
    # an underexposed grey page. A robust central percentile retains solid
    # marks without forcing the entire dark page below a white-canvas floor.
    paper = np.maximum(paper, float(np.percentile(luminance, 60)) * 0.85)
    tinted = color and float(np.mean(np.ptp(source.astype(np.int16), axis=2) > 12)) > 0.005
    anchor = float(np.percentile(paper, 80)) if tinted else 250.0
    normalized = np.clip(luminance / np.maximum(paper, 1.0) * anchor, 0, 255)
    darkness = np.maximum(anchor - normalized, 0.0)
    # Continuous noise knee: no binary foreground/paper boundary, no glyph
    # dilation, and no component deletion. Strong ink receives bounded gain.
    knee = 4.0 if tinted else (12.0 if native_small_print else (6.0 if thermal else 32.0))
    darkness = np.maximum(darkness - knee * (1.0 - np.exp(-darkness / knee)), 0)
    gain = 4.0 if native_small_print else (1.65 if thermal else 1.15)
    if native_small_print:
        native_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if color else source
        print_pixels = detect_source_text_mask(native_gray, protection="wide") & (darkness > 2)
        if np.count_nonzero(print_pixels) >= 20:
            # Stronger originals must not inherit the gain needed for faded
            # print: clipping many dark pixels together creates heavy smears.
            gain = float(np.clip(180.0 / max(float(np.percentile(darkness[print_pixels], 90)), 1), 1.0, 4.0))
    enhanced = np.clip(anchor - darkness * gain, 0, 255)
    gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if color else source
    text = detect_source_text_mask(gray_source, protection="wide")
    fine = cv2.GaussianBlur(raw_luminance, (0, 0), 1.5) - raw_luminance
    measured_text = text & (fine >= 3.0) & (darkness >= 8.0)
    if thermal and np.count_nonzero(text) >= 20:
        local_ink = np.maximum(cv2.GaussianBlur(gray_source, (0, 0), 5).astype(np.float32) - gray_source, 0)
        if float(np.percentile(local_ink[text], 70)) < 6.0:
            # Very faint print needs extra gain only at measured character
            # pixels. Amplifying the whole page also amplifies crease shadows.
            locations = np.zeros_like(gray_source, np.uint8)
            for line in ocr_lines or []:
                points = np.asarray(line.get('box', []), np.int32)
                if points.shape == (4, 2) and float(line.get('confidence') or 0) >= .35:
                    cv2.fillConvexPoly(locations, points, 1)
            faint_text = (text | (locations > 0)) & (local_ink >= 1.5) & (fine >= 1.0)
            enhanced[faint_text] = np.minimum(enhanced[faint_text],
                                             np.clip(anchor - np.minimum(local_ink[faint_text] * 16.0, 170.0), 0, 255))
    # Increase only measured glyph darkness; keep the paper tone and original
    # silhouettes. Bounded gain avoids thickening, dilation and edge halos.
    if not thermal:
        enhanced[measured_text] = np.maximum(
            enhanced[measured_text] - np.minimum(darkness[measured_text] * 0.45, 40.0), 0.0
        )
    if not native_small_print and not tinted:
        # Paper suppression must not compress faint measured character edges
        # into near-white. Reintroduce only their fine-scale source response,
        # above a robust image-noise floor and within source text groups.
        # The response starts at zero continuously; there is no black glyph
        # floor, dilation of output ink, or OCR wording reconstruction.
        # Estimate noise locally: white margins must not set the threshold for
        # a textured or shadowed corner elsewhere on the page. A small median
        # rejects isolated stroke edges while measuring surrounding grain.
        local_noise=cv2.medianBlur(np.abs(fine).astype(np.float32),5)
        noise_floor=np.maximum(1.5,2.0*local_noise)
        signal=np.maximum(fine-noise_floor,0)
        # Non-text marks (pencil signatures, Chinese details, thin rules) must
        # retain measured fine contrast even when no text group/OCR box exists.
        enhanced=np.minimum(enhanced,anchor-np.minimum(signal*2.5,40.0))
        protected_text=np.zeros_like(text)
        box_heights=[float(np.ptp(np.asarray(line.get('box',[]))[:,1])) for line in ocr_lines or []
                     if np.asarray(line.get('box',[])).shape==(4,2)]
        typical_height=float(np.median(box_heights)) if box_heights else 0
        for line in ocr_lines or []:
            points=np.asarray(line.get('box',[]),np.int32)
            if points.shape==(4,2) and float(line.get('confidence') or 0)>=.35:
                # Large weak reverse-side lettering is not body print. Keep
                # its source contrast, but do not promote it into black text.
                if typical_height and np.ptp(points[:,1])>2.5*typical_height:
                    continue
                locations=np.zeros_like(gray_source,np.uint8)
                cv2.fillConvexPoly(locations,points,1)
                protected_text |= locations>0
        supported=protected_text & (signal>0)
        ink_strength=160.0*(-np.expm1(-signal/4.0))
        enhanced[supported]=np.minimum(enhanced[supported],anchor-ink_strength[supported])
    if not tinted:
        enhanced = lighten_neutral_paper(enhanced.round().astype(np.uint8), gray_source).astype(np.float32)
    if thermal:
        enhanced = sharpen_measured_print_edges(enhanced.round().astype(np.uint8))
    if not color:
        return enhanced.round().astype(np.uint8)
    lab[:, :, 0] = enhanced / 2.55
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) * 255.0, 0, 255).round().astype(np.uint8)


def sharpen_measured_print_edges(image: np.ndarray) -> np.ndarray:
    """Bound edge contrast to existing local extrema at the glyph scale.

    The scale comes from connected print-sized marks, independent of merchant,
    OCR recognition or filename. Flat paper receives no sharpening. No erosion,
    dilation of the output, thresholding or synthetic glyphs are used.
    """
    import cv2

    pixels = image.astype(np.float32)
    count, _, stats, _ = cv2.connectedComponentsWithStats((image < 170).astype(np.uint8), 8)
    heights = [h for x, y, w, h, area in stats[1:count]
               if 3 <= h <= 50 and 1 <= w <= h * 3 and 3 <= area <= 600]
    if len(heights) < 8:
        return image.copy()
    sigma = float(np.clip(np.median(heights) / 12.0, .65, 1.1))
    local_min = cv2.erode(pixels, np.ones((3, 3), np.uint8))
    local_max = cv2.dilate(pixels, np.ones((3, 3), np.uint8))
    weight = np.clip((local_max - local_min - 20.0) / 50.0, 0, 1)
    detail = pixels - cv2.GaussianBlur(pixels, (0, 0), sigma)
    result = np.clip(pixels + .8 * detail * weight, local_min, local_max)
    return np.clip(result, 0, 255).round().astype(np.uint8)


def lighten_neutral_paper(image: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Monotone paper-tone shoulder: lift lights, retain dark ink exactly."""
    output = image.astype(np.float32)
    if image.shape != source.shape:
        raise ValueError("Neutral paper source and output geometry must agree")
    # A global monotone shoulder cannot introduce local bright/dark islands.
    # Pixelwise foreground/background switches previously made grain patchy.
    lift = 0.80 * np.square(np.clip((output - 190.0) / 60.0, 0.0, 1.0))
    output += (255.0 - output) * lift
    return np.clip(output, 0, 255).round().astype(np.uint8)


def make_source_normalized_faithful_view(source: np.ndarray) -> np.ndarray:
    """Reduce paper illumination without deleting measured document marks."""
    import cv2

    source_u8 = source.astype(np.uint8)
    # The background scale is much wider than receipt strokes and table rules.
    # Division suppresses gradual shading and broad fold illumination while
    # preserving the exact source geometry of text, logos and handwriting.
    background = cv2.GaussianBlur(source_u8, (0, 0), 23.0)
    normalized = cv2.divide(source_u8, np.maximum(background, 1), scale=244)
    normalized = cv2.fastNlMeansDenoising(
        normalized, None, h=2, templateWindowSize=7, searchWindowSize=17
    )
    # Keep strong original marks dark while lifting only already-light paper.
    source_local = np.maximum(
        cv2.GaussianBlur(source_u8, (0, 0), 5.0).astype(np.int16)
        - source_u8.astype(np.int16),
        0,
    )
    # A hard pixel switch between lifted paper and copied source creates
    # mottled grey islands at the threshold. Use one continuous tone curve.
    darkness = np.maximum(248.0 - normalized.astype(np.float32), 0.0)
    return np.clip(248.0 - darkness * 1.25, 0, 255).astype(np.uint8)


def _thin_document_structure_mask(image: np.ndarray) -> np.ndarray:
    """Return source-measured thin rules while excluding broad fold bands."""
    import cv2

    image_u8 = image.astype(np.uint8)
    height, width = image_u8.shape
    local = cv2.GaussianBlur(image_u8, (0, 0), 2.2)
    darkness = np.maximum(local.astype(np.int16) - image_u8.astype(np.int16), 0)
    binary = (darkness >= 1.5).astype(np.uint8)
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, width // 22), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, height // 30))),
    )
    structure = np.zeros_like(binary)
    for raw, horizontal_axis in ((horizontal, True), (vertical, False)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        for label in range(1, count):
            x, y, component_width, component_height, _area = map(int, stats[label])
            long_side = component_width if horizontal_axis else component_height
            short_side = component_height if horizontal_axis else component_width
            required = max(18, (width // 14) if horizontal_axis else (height // 18))
            if long_side < required or short_side > max(3, int(min(height, width) * 0.006)):
                continue
            if x <= 1 or y <= 1 or x + component_width >= width - 1 or y + component_height >= height - 1:
                continue
            structure[labels == label] = 1
    return structure.astype(bool)


def select_faithful_display_candidate(
    source: np.ndarray,
    candidate: np.ndarray,
    ocr_lines: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Keep aggressive recovery only when source content survives locally."""
    import cv2

    source_u8 = source.astype(np.uint8)
    candidate_u8 = candidate.astype(np.uint8)
    if source_u8.shape != candidate_u8.shape:
        return make_source_normalized_faithful_view(source_u8)

    source_text = detect_source_text_mask(source_u8, protection="wide")
    source_structure = _thin_document_structure_mask(source_u8)
    candidate_background = cv2.GaussianBlur(candidate_u8, (0, 0), 5.0)
    candidate_darkness = np.maximum(
        candidate_background.astype(np.int16) - candidate_u8.astype(np.int16), 0
    )
    candidate_ink = candidate_darkness >= 1.5
    text_pixels = int(np.count_nonzero(source_text))
    structure_pixels = int(np.count_nonzero(source_structure))
    text_retention = float(
        np.count_nonzero(candidate_ink & source_text) / max(text_pixels, 1)
    )
    structure_retention = float(
        np.count_nonzero(candidate_ink & source_structure) / max(structure_pixels, 1)
    )

    source_background = cv2.GaussianBlur(source_u8, (0, 0), 7.0)
    source_dark = (
        source_background.astype(np.int16) - source_u8.astype(np.int16)
    ) >= 2
    support = cv2.dilate(source_dark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    unsupported_fraction = float(np.mean(candidate_ink & ~support))

    # Text retention is the primary gate.  A permissive structure threshold is
    # intentional: folds can resemble long rules, especially on thermal paper,
    # and must not force an otherwise clean restoration back to the creased
    # source.  Structure fallback is reserved for documents with substantial
    # measured table geometry and near-total deletion.
    fails_text = text_pixels >= 40 and text_retention < 0.52
    substantial_structure = structure_pixels >= max(5000, int(source_u8.size * 0.003))
    fails_structure = substantial_structure and structure_retention < 0.22
    # The support metric is conservative and can label anti-aliased edges as
    # unsupported.  Only a gross, page-wide introduction of dark ink should
    # override a candidate here; local blob suppression happens upstream.
    creates_ink = unsupported_fraction > 0.075
    if fails_text or fails_structure or creates_ink:
        normalized = make_source_normalized_faithful_view(source_u8)
        # Preserve the cleaner candidate only where it is already supported by
        # the source; elsewhere use normalized source evidence. This prevents
        # both missing structures and newly introduced blobs.
        supported_candidate = candidate_ink & support
        normalized[supported_candidate] = np.minimum(
            normalized[supported_candidate], candidate_u8[supported_candidate]
        )
        return lighten_neutral_paper(normalized, source_u8)
    return lighten_neutral_paper(candidate_u8, source_u8)


def finalize_measured_content(source: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Bound added ink by measured source detail and retain non-OCR marks.

    Uses pixel evidence rather than recognised vocabulary, so Chinese print,
    signatures, rules and logos remain eligible for preservation.
    """
    import cv2

    src = source.astype(np.uint8)
    output = candidate.astype(np.uint8).copy()
    background = cv2.GaussianBlur(src, (0, 0), 11.0)
    fine_background = cv2.GaussianBlur(src, (0, 0), 2.0)
    detail = np.maximum(background.astype(np.float32) - src, 0)
    fine = np.maximum(fine_background.astype(np.float32) - src, 0)
    # No candidate may amplify unsupported scanner specks into dark islands.
    normalized = cv2.divide(src, np.maximum(background, 1), scale=248)
    unsupported = (fine < 1.0) & (detail < 2.0) & (output < 235)
    output[unsupported] = np.maximum(output[unsupported], normalized[unsupported])
    text = detect_source_text_mask(src, protection="wide")
    rules = _thin_document_structure_mask(src)
    # Retain strong measured marks even when OCR cannot name them. Avoid a
    # component-size whitelist that would delete logos or connected script.
    measured = (fine >= 10.0) & (detail >= 25.0)
    compact = np.zeros_like(measured)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(measured.astype(np.uint8), 8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if (area >= 12 and max(width, height) / max(min(width, height), 1) < 12
                and x > 1 and y > 1 and x + width < src.shape[1] - 1
                and y + height < src.shape[0] - 1):
            compact[labels == label] = True
    # Only a crossing-rule network establishes table geometry; isolated long
    # lines may be folds. Keep those decisions with the upstream cleaner.
    horizontal = cv2.morphologyEx(rules.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 18), np.uint8))
    vertical = cv2.morphologyEx(rules.astype(np.uint8), cv2.MORPH_OPEN, np.ones((18, 1), np.uint8))
    crossings = cv2.dilate(horizontal, np.ones((3, 3), np.uint8)) & cv2.dilate(vertical, np.ones((3, 3), np.uint8))
    preserve = compact | ((rules & (fine >= 1.5)) if np.count_nonzero(crossings) >= 4 else False)
    output[preserve] = np.minimum(output[preserve], normalized[preserve])
    # Equalise only eligible ink pixels; do not thicken glyph silhouettes.
    eligible = text & (fine >= 3.0) & (detail >= 6.0)
    target = np.clip(248.0 - 125.0 * (1.0 - np.exp(-fine / 9.0)), 0, 255).astype(np.uint8)
    output[eligible] = np.minimum(output[eligible], target[eligible])
    return output


def make_conservative_creased_view(
    source: np.ndarray,
    candidate: np.ndarray | None = None,
    ocr_lines: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Render source-supported receipt content on a clean paper field.

    Highly crumpled paper is not used as the visual base: doing so inevitably
    retains folds that cross words.  Only compact, line-supported source ink
    is rendered.  Elongated fold geometry is rejected by the text detector,
    so a crease cannot be reintroduced merely because it intersects a word.
    """
    import cv2

    source_u8 = source.astype(np.uint8)
    output = np.full_like(source_u8, 248)
    height, width = source_u8.shape
    target_height = min(height, 1000)
    scale = target_height / float(height)
    normalized = cv2.resize(
        source_u8,
        (max(1, int(round(width * scale))), target_height),
        interpolation=cv2.INTER_AREA,
    )
    normalized_text = detect_source_text_mask(normalized, protection="wide")
    normalized_text = cv2.dilate(
        normalized_text.astype(np.uint8), np.ones((2, 2), np.uint8)
    ) > 0
    text = cv2.resize(
        normalized_text.astype(np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    source_background = cv2.GaussianBlur(source_u8, (0, 0), 15)
    source_darkness = np.maximum(
        source_background.astype(np.int16) - source_u8.astype(np.int16), 0
    ).astype(np.float32)
    fine_background = cv2.GaussianBlur(source_u8, (0, 0), 2.2)
    fine_darkness = np.maximum(
        fine_background.astype(np.int16) - source_u8.astype(np.int16), 0
    ).astype(np.float32)
    localised = np.zeros_like(source_u8, dtype=np.uint8)
    for line in ocr_lines or []:
        if float(line.get("confidence", 0.0)) < 0.55:
            continue
        polygon = line.get("polygon") or line.get("box")
        if not polygon or len(polygon) < 3:
            continue
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(localised, [points], 1)
    # A broad fold can be dark relative to the page but has little response at
    # character scale. Require both page-scale and fine-scale source evidence
    # inside OCR rectangles so folds crossing words are not labelled as ink.
    text |= (
        (localised > 0)
        & (source_darkness >= 2.0)
        & (fine_darkness >= 0.75)
    )

    # Preserve thin separators and form/table rules independently of OCR.
    # Extract them from fine-scale source ink so wide crease bands do not pass.
    fine_binary = (fine_darkness >= 1.5).astype(np.uint8)
    horizontal_raw = cv2.morphologyEx(
        fine_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(16, width // 22), 1)),
    )
    vertical_raw = cv2.morphologyEx(
        fine_binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(16, height // 30))),
    )
    horizontal = np.zeros_like(horizontal_raw)
    horizontal_count, horizontal_labels, horizontal_stats, _ = cv2.connectedComponentsWithStats(
        horizontal_raw, 8
    )
    for label in range(1, horizontal_count):
        x, y, line_width, line_height, _area = map(int, horizontal_stats[label])
        if line_width < max(18, width // 12):
            continue
        if line_height > max(3, int(height * 0.0045)):
            continue
        if y <= int(height * 0.02) or y + line_height >= int(height * 0.98):
            continue
        horizontal[horizontal_labels == label] = 1

    # Vertical rules are retained only when the page has repeated horizontal
    # and vertical structure (an invoice/form). Thermal receipts otherwise
    # turn long creases and photographed page edges into false borders.
    horizontal_fraction = float(np.mean(horizontal > 0))
    vertical_fraction = float(np.mean(vertical_raw > 0))
    structured_form = horizontal_fraction >= 0.0020 and vertical_fraction >= 0.0012
    vertical = np.zeros_like(vertical_raw)
    if structured_form:
        vertical_count, vertical_labels, vertical_stats, _ = cv2.connectedComponentsWithStats(
            vertical_raw, 8
        )
        for label in range(1, vertical_count):
            x, y, line_width, line_height, _area = map(int, vertical_stats[label])
            if line_height < max(18, height // 16):
                continue
            if line_width > max(3, int(width * 0.006)):
                continue
            if x <= int(width * 0.02) or x + line_width >= int(width * 0.98):
                continue
            vertical[vertical_labels == label] = 1
    structure = ((horizontal > 0) | (vertical > 0)) & (source_darkness >= 2.0)
    text |= structure
    text = cv2.dilate(text.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    if text.any():
        background = cv2.GaussianBlur(source_u8, (0, 0), 7)
        darkness = np.maximum(
            background.astype(np.int16) - source_u8.astype(np.int16), 0
        ).astype(np.float32)
        # Character-scale response controls tone; broad background shading no
        # longer makes one word darker than another.
        darkness = np.minimum(darkness, np.maximum(fine_darkness * 1.8, 0.0))
        values = darkness[text]
        reference = float(np.percentile(values, 70)) if values.size else 1.0
        relative = np.clip(darkness / max(reference, 1.0), 0.0, 1.8)
        desired = np.maximum(darkness * 1.10, 92.0 * np.power(relative, 0.72))
        core_floor = 86.0 * np.clip((darkness - 0.5) / 3.5, 0.0, 1.0)
        desired = np.maximum(desired, core_floor)
        desired = np.minimum(desired, 154.0)
        rendered = np.clip(248.0 - desired, 0, 255).astype(np.uint8)
        output[text] = np.minimum(output[text], rendered[text])
    return protect_machine_readable_regions(output, source_u8)


def balance_source_text_tone(
    image: np.ndarray,
    source: np.ndarray,
    ocr_lines: list[dict[str, Any]] | None = None,
    *,
    target_darkness: float = 64.0,
) -> np.ndarray:
    """Give source-detected characters a consistent tone without darkening folds.

    Character geometry comes from compact source components. Long crease and
    paper structures are rejected by ``detect_source_text_mask`` before this
    function applies a bounded darkness floor. OCR text is never consulted.
    """
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    text = detect_source_text_mask(source_u8, protection="wide")
    # Tone balancing is limited to recognised line geometry. Compact paper
    # damage elsewhere can resemble a glyph component, which previously made
    # grey blobs darker alongside the actual receipt text.
    if ocr_lines:
        regions = np.zeros_like(source_u8, dtype=np.uint8)
        for line in ocr_lines:
            if float(line.get("confidence") or 0.0) < 0.35:
                continue
            points = np.asarray(line.get("box") or line.get("polygon") or [], dtype=np.int32)
            if points.shape == (4, 2):
                cv2.fillConvexPoly(regions, points, 1)
        regions = cv2.dilate(regions, np.ones((5, 5), np.uint8))
        text &= regions > 0
    if np.count_nonzero(text) < 4:
        return output
    background = cv2.GaussianBlur(source_u8, (0, 0), 7)
    darkness = np.maximum(
        background.astype(np.int16) - source_u8.astype(np.int16), 0
    ).astype(np.float32)
    values = darkness[text]
    reference = float(np.percentile(values, 70)) if values.size else 1.0
    relative = np.clip(darkness / max(reference, 1.0), 0.0, 1.8)
    desired = target_darkness * np.power(relative, 0.72)
    # A measured character core receives a common minimum tone; weaker edge
    # pixels remain proportional so glyphs do not become artificially thick.
    core_floor = target_darkness * np.clip((darkness - 1.0) / 5.0, 0.0, 0.78)
    desired = np.maximum(desired, core_floor)
    # Every accepted text-mask pixel is measured source ink. Give it a common
    # readable floor so weak thermal traces do not remain pale grey beside
    # darker characters from the same line.
    desired = np.maximum(desired, target_darkness * 0.72)
    desired = np.maximum(desired, darkness)
    desired = np.minimum(desired, target_darkness * 1.25)
    rendered = np.clip(background.astype(np.float32) - desired, 0, 255).astype(np.uint8)
    output[text] = np.minimum(output[text], rendered[text])
    return output


def apply_local_quality_fallback(
    image: np.ndarray,
    source: np.ndarray,
    *,
    tile_size: int = 128,
    overlap: int = 48,
) -> np.ndarray:
    """Replace only locally unsafe recovery with conservative source contrast.

    Receipts commonly contain several degradation types at once. Overlapping
    tiles are scored for source-stroke retention, unsupported new darkness and
    edge loss. Unsafe areas smoothly fall back to illumination-normalised
    source pixels; good areas keep the stronger recovery.
    """
    import cv2

    output = image.astype(np.uint8)
    source_u8 = source.astype(np.uint8)
    height, width = source_u8.shape
    source_background = cv2.GaussianBlur(source_u8, (0, 0), 31)
    normalized_source = cv2.divide(source_u8, np.maximum(source_background, 1), scale=248)
    normalized_source = cv2.fastNlMeansDenoising(
        normalized_source, None, h=1, templateWindowSize=7, searchWindowSize=21
    )
    text = detect_source_text_mask(source_u8, protection="wide")
    # Safest fallback for severely textured paper: preserve the paper exactly
    # and strengthen only character-shaped pixels measured in the source.
    source_safe = source_u8.copy()
    safe_background = cv2.GaussianBlur(source_u8, (0, 0), 7)
    safe_darkness = np.maximum(
        safe_background.astype(np.int16) - source_u8.astype(np.int16), 0
    ).astype(np.float32)
    text_values = safe_darkness[text]
    safe_reference = float(np.percentile(text_values, 70)) if text_values.size else 1.0
    relative = np.clip(safe_darkness / max(safe_reference, 1.0), 0.0, 1.8)
    equalized = 58.0 * np.power(relative, 0.78)
    equalized = np.maximum(equalized, 58.0 * np.minimum(relative * 2.0, 0.82))
    equalized = np.maximum(equalized, safe_darkness)
    equalized = np.minimum(equalized, 78.0)
    safe_render = np.clip(
        safe_background.astype(np.float32) - equalized, 0, 255
    ).astype(np.uint8)
    source_safe[text] = np.minimum(source_safe[text], safe_render[text])
    # In locally wrinkled/printed-through areas, division can amplify the very
    # paper structure we are trying to suppress. Interpolate back toward the
    # untouched source there; use normalized source only in locally stable
    # paper regions.
    local_paper = cv2.GaussianBlur(source_u8, (0, 0), 15).astype(np.float32)
    broad_paper = cv2.GaussianBlur(source_u8, (0, 0), 71).astype(np.float32)
    texture = cv2.GaussianBlur(np.abs(local_paper - broad_paper), (0, 0), 9.0)
    source_weight = np.clip((texture - 1.0) / 6.0, 0.0, 1.0)
    conservative = (
        normalized_source.astype(np.float32) * (1.0 - source_weight)
        + source_safe.astype(np.float32) * source_weight
    ).round().astype(np.uint8)
    text_protection = cv2.dilate(text.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    src_local = cv2.GaussianBlur(source_u8, (0, 0), 7)
    out_local = cv2.GaussianBlur(output, (0, 0), 7)
    src_dark = np.maximum(src_local.astype(np.int16) - source_u8.astype(np.int16), 0).astype(np.float32)
    out_dark = np.maximum(out_local.astype(np.int16) - output.astype(np.int16), 0).astype(np.float32)
    src_edge = cv2.magnitude(
        cv2.Sobel(source_u8, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(source_u8, cv2.CV_32F, 0, 1, ksize=3),
    )
    out_edge = cv2.magnitude(
        cv2.Sobel(output, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(output, cv2.CV_32F, 0, 1, ksize=3),
    )
    fallback = np.zeros((height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    step = max(16, tile_size - overlap)
    y_starts = list(range(0, max(height - tile_size, 0) + 1, step))
    x_starts = list(range(0, max(width - tile_size, 0) + 1, step))
    if not y_starts or y_starts[-1] != max(height - tile_size, 0):
        y_starts.append(max(height - tile_size, 0))
    if not x_starts or x_starts[-1] != max(width - tile_size, 0):
        x_starts.append(max(width - tile_size, 0))
    for y in y_starts:
        for x in x_starts:
            y2, x2 = min(y + tile_size, height), min(x + tile_size, width)
            tile_text = text[y:y2, x:x2]
            tile_out_dark = out_dark[y:y2, x:x2]
            tile_src_dark = src_dark[y:y2, x:x2]
            text_count = int(np.count_nonzero(tile_text))
            retention = float(np.mean(tile_out_dark[tile_text] >= 1.5)) if text_count >= 12 else 1.0
            tile_protection = text_protection[y:y2, x:x2]
            paper = ~cv2.dilate(tile_protection.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            unsupported = float(np.mean(
                paper & (tile_out_dark > tile_src_dark + 6.0) & (tile_out_dark >= 8.0)
            ))
            source_edge_mean = float(np.mean(src_edge[y:y2, x:x2][tile_text])) if text_count >= 12 else 1.0
            edge_ratio = (
                float(np.mean(out_edge[y:y2, x:x2][tile_text])) / max(source_edge_mean, 1e-6)
                if text_count >= 12 else 1.0
            )
            severity = max(
                np.clip((0.68 - retention) / 0.38, 0.0, 1.0),
                np.clip((unsupported - 0.025) / 0.075, 0.0, 1.0),
                np.clip((0.78 - edge_ratio) / 0.28, 0.0, 1.0),
                np.clip((float(np.mean(source_weight[y:y2, x:x2])) - 0.30) / 0.35, 0.0, 1.0),
            )
            if severity <= 0:
                continue
            window_y = np.hanning(max(y2 - y, 3))
            window_x = np.hanning(max(x2 - x, 3))
            window = np.maximum(np.outer(window_y, window_x), 0.08).astype(np.float32)
            fallback[y:y2, x:x2] += window * float(severity)
            weights[y:y2, x:x2] += window
    blend = np.divide(fallback, np.maximum(weights, 1e-6))
    blend = cv2.GaussianBlur(blend, (0, 0), 7.0)
    blend = np.clip(blend, 0.0, 1.0)
    result = output.astype(np.float32) * (1.0 - blend) + conservative.astype(np.float32) * blend
    return np.clip(result, 0, 255).round().astype(np.uint8)


def apply_structural_coherence_fallback(
    image: np.ndarray,
    source: np.ndarray,
    *,
    minimum_retention: float = 0.18,
) -> np.ndarray:
    """Return the original when cleaning destroys coherent source text.

    The comparison uses only source-derived compact text components, not all
    paper texture. A low retention score therefore indicates that character
    structure was lost rather than that folds were successfully removed.
    """
    import cv2

    source_u8 = source.astype(np.uint8)
    output = image.astype(np.uint8)
    text_mask = detect_source_text_mask(source_u8, protection="wide")
    if np.count_nonzero(text_mask) < 40:
        return output
    output_background = cv2.GaussianBlur(output, (0, 0), 5.0)
    output_darkness = np.maximum(
        output_background.astype(np.int16) - output.astype(np.int16), 0
    ).astype(np.float32)
    retention = float(np.count_nonzero((output_darkness >= 1.5) & text_mask) / np.count_nonzero(text_mask))
    if retention < minimum_retention:
        return source_u8.copy()
    return output


def detect_machine_readable_mask(source: np.ndarray) -> np.ndarray:
    """Detect QR/barcode-like regions without interpreting their payload."""
    import cv2

    source_u8 = source.astype(np.uint8)
    height, width = source_u8.shape
    protected = np.zeros_like(source_u8, dtype=np.uint8)

    # QR geometry, including undecodable but detectable symbols.
    try:
        detected, points = cv2.QRCodeDetector().detect(source_u8)
        if detected and points is not None:
            polygon = np.asarray(points, dtype=np.int32).reshape(-1, 2)
            cv2.fillConvexPoly(protected, polygon, 1)
    except cv2.error:
        pass

    # Dense vertical transitions identify ordinary 1-D receipt barcodes.
    background = cv2.GaussianBlur(source_u8, (0, 0), 9)
    darkness = np.maximum(background.astype(np.int16) - source_u8.astype(np.int16), 0).astype(np.uint8)
    dark = (darkness >= 6).astype(np.uint8)

    vertical = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    grouped = cv2.morphologyEx(
        vertical, cv2.MORPH_CLOSE, np.ones((3, max(9, width // 45)), np.uint8)
    )
    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, region_width, region_height = cv2.boundingRect(contour)
        area = region_width * region_height
        if area < max(350, int(source_u8.size * 0.0008)):
            continue
        aspect = float(region_width / max(region_height, 1))
        if not (2.5 <= aspect <= 10.0 and region_height >= 12):
            continue
        region = dark[y:y + region_height, x:x + region_width]
        column_activity = np.mean(np.any(region > 0, axis=0))
        if column_activity < 0.45:
            continue
        # Real 1-D bars are vertically coherent. Ordinary text can have many
        # active columns but disappears under a long vertical opening.
        vertical_core = cv2.morphologyEx(
            region, cv2.MORPH_OPEN,
            np.ones((max(5, region_height // 2), 1), np.uint8),
        )
        coherence = float(np.count_nonzero(vertical_core) / max(np.count_nonzero(region), 1))
        if coherence < 0.80:
            continue
        cv2.rectangle(protected, (x, y), (x + region_width, y + region_height), 1, -1)

    if protected.any():
        protected = cv2.dilate(protected, np.ones((7, 7), np.uint8), iterations=1)
    return protected.astype(bool)


def protect_machine_readable_regions(image: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Keep measured QR/barcode ink out of text and paper transforms.

    Only dark module/bar pixels are copied. Copying the full detection box
    pastes grey rectangular paper patches into an otherwise normalised view.
    """
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    protected = detect_machine_readable_mask(source_u8)
    local_background = cv2.GaussianBlur(source_u8, (0, 0), 9)
    darkness = np.maximum(
        local_background.astype(np.int16) - source_u8.astype(np.int16), 0
    )
    measured_modules = protected & (darkness >= 3)
    output[measured_modules] = np.minimum(output[measured_modules], source_u8[measured_modules])
    return output


def preserve_faint_source_supported_strokes(
    image: np.ndarray,
    source: np.ndarray,
    dsdnet_candidate: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    *,
    target_darkness: float = 58.0,
    minimum_confidence: float = 0.35,
    protection: str = "standard",
) -> np.ndarray:
    """Restore weak measured stroke pixels without promoting paper texture.

    A pixel must be locally darker in the source and either supported by the
    independently produced DSDNet candidate or belong to a source-derived
    text component. OCR characters are never used, and no previously white
    source pixel is painted by this function.
    """
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    candidate_u8 = dsdnet_candidate.astype(np.uint8)
    source_text_mask = detect_source_text_mask(source_u8, protection=protection)
    processed = np.zeros(source_u8.shape, bool)
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < minimum_confidence:
            continue
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape != (4, 2):
            continue
        x, y, width, height = cv2.boundingRect(points)
        if width < 4 or height < 4:
            continue
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2, y2 = min(source_u8.shape[1], x + width + 1), min(source_u8.shape[0], y + height + 1)
        src = source_u8[y1:y2, x1:x2]
        cand = candidate_u8[y1:y2, x1:x2]
        src_bg = cv2.GaussianBlur(src, (0, 0), max(2.0, height / 3.0))
        cand_bg = cv2.GaussianBlur(cand, (0, 0), max(2.0, height / 3.0))
        src_darkness = np.maximum(src_bg.astype(np.int16) - src.astype(np.int16), 0).astype(np.float32)
        cand_darkness = np.maximum(cand_bg.astype(np.int16) - cand.astype(np.int16), 0).astype(np.float32)

        # The low source threshold deliberately keeps anti-aliased/faded edge
        # pixels that the old 2.5-level cutoff discarded. Candidate agreement
        # prevents an isolated paper fluctuation from being promoted.
        candidate_support = cv2.dilate(
            (cand_darkness >= 2.0).astype(np.uint8), np.ones((3, 3), np.uint8)
        ) > 0
        source_component = source_text_mask[y1:y2, x1:x2]
        measured = (src_darkness >= 1.25) & (candidate_support | source_component)
        if np.count_nonzero(measured) < 4:
            continue
        reference = float(np.percentile(src_darkness[measured], 70))
        gain = float(np.clip(target_darkness / max(reference, 1.0), 1.0, 1.70))
        rendered = np.clip(
            src_bg.astype(np.float32) - src_darkness * gain, 0, 255
        ).astype(np.uint8)
        region = output[y1:y2, x1:x2]
        region[measured] = np.minimum(region[measured], src[measured])
        region[measured] = np.minimum(region[measured], rendered[measured])
        processed[y1:y2, x1:x2] |= measured

    # Recover source-derived text missed by OCR. Only compact components that
    # participate in a multi-component text band are eligible; isolated marks
    # and elongated fold geometry were rejected by detect_source_text_mask.
    remaining = source_text_mask & ~processed
    if remaining.any():
        source_background = cv2.GaussianBlur(source_u8, (0, 0), 5.0)
        darkness = np.maximum(
            source_background.astype(np.int16) - source_u8.astype(np.int16), 0
        ).astype(np.float32)
        values = darkness[remaining]
        reference = float(np.percentile(values, 70)) if values.size else 1.0
        gain = float(np.clip(target_darkness / max(reference, 1.0), 1.0, 1.65))
        rendered = np.clip(
            source_background.astype(np.float32) - darkness * gain, 0, 255
        ).astype(np.uint8)
        output[remaining] = np.minimum(output[remaining], source_u8[remaining])
        output[remaining] = np.minimum(output[remaining], rendered[remaining])
    return output


def detect_source_text_mask(source: np.ndarray, *, protection: str = "standard") -> np.ndarray:
    """Locate faint character-like source components without using OCR text.

    Weak local-darkness components are filtered by glyph-scale geometry and
    must occur in a horizontal group. This protects missed characters while
    excluding long fold edges, broad shadows and isolated scanner specks.
    """
    import cv2

    source_u8 = source.astype(np.uint8)
    height, width = source_u8.shape
    background = cv2.GaussianBlur(source_u8, (0, 0), 5.0)
    darkness = np.maximum(
        background.astype(np.int16) - source_u8.astype(np.int16), 0
    ).astype(np.float32)
    wide = protection == "wide"
    weak = (darkness >= (0.75 if wide else 1.25)).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(weak, 8)
    glyphs: list[tuple[int, int, int, int, int]] = []
    glyph_mask = np.zeros_like(weak)
    glyph_labels = np.zeros(count,np.uint8)
    # Wide protection also admits a short connected word: faint thermal
    # antialiasing often joins neighbouring glyphs into one component.
    max_glyph_width = max(18 if wide else 10, int(width * (0.25 if wide else 0.055)))
    # Small mobile captures can still contain 12--22 px high receipt glyphs;
    # do not tie the lower allowance too tightly to the full image height.
    max_glyph_height = max(24, int(height * 0.045))
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        box_area = max(component_width * component_height, 1)
        density = float(area / box_area)
        aspect = float(max(component_width, component_height) / max(min(component_width, component_height), 1))
        if not (2 <= area <= 500):
            continue
        if component_width > max_glyph_width or component_height > max_glyph_height:
            continue
        if density < 0.08 or aspect > 8.0:
            continue
        glyph_labels[label] = 1
        glyphs.append((label, x, y, component_width, component_height))

    if len(glyphs) < 2:
        return np.zeros_like(source_u8, dtype=bool)
    glyph_mask = glyph_labels[labels]

    # Group neighbouring glyph candidates into horizontal text bands. Requiring
    # several components prevents a lone stain or pinhole from becoming text.
    join_width = max(7 if wide else 5, int(width * (0.026 if wide else 0.018)))
    joined = cv2.dilate(
        glyph_mask, np.ones((3, join_width), np.uint8), iterations=1
    )
    group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    accepted = np.zeros_like(weak)
    accepted_labels = np.zeros(count,np.uint8)
    candidate_groups: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for group in range(1, group_count):
        x, y, group_width, group_height, _area = map(int, group_stats[group])
        if group_width < 8 or group_height > max(30, int(height * 0.06)):
            continue
        group_region = group_labels[y:y+group_height,x:x+group_width] == group
        local_labels=labels[y:y+group_height,x:x+group_width]
        local_glyphs=glyph_mask[y:y+group_height,x:x+group_width]
        member_labels = np.unique(local_labels[group_region & (local_glyphs > 0)])
        member_labels = member_labels[member_labels > 0]
        candidate_groups.append((group, x, y, group_width, group_height, member_labels))
    for group, x, y, group_width, group_height, member_labels in candidate_groups:
        enough_members = len(member_labels) >= (3 if wide else 2)
        # A faint connected word may be one component. Accept it only when a
        # separate text group exists on the same horizontal line; isolated
        # stains and fold fragments therefore remain rejected.
        same_line_peer = False
        if wide and len(member_labels) < 3:
            centre_y = y + group_height / 2.0
            same_line_peer = any(
                other_group != group
                and abs((other_y + other_height / 2.0) - centre_y) <= max(6.0, group_height * 0.8)
                and len(other_members) >= 1
                for other_group, _ox, other_y, _ow, other_height, other_members in candidate_groups
            )
        if not enough_members and not same_line_peer:
            continue
        for member in member_labels:
            accepted_labels[member] = 1
    return accepted_labels[labels].astype(bool)


def apply_line_level_source_fallback(
    image: np.ndarray,
    source: np.ndarray,
    dsdnet_candidate: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    *,
    target_darkness: float = 58.0,
) -> np.ndarray:
    """Use one coherent source-derived rendering when a whole line degrades."""
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    candidate_u8 = dsdnet_candidate.astype(np.uint8)
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < 0.35:
            continue
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape != (4, 2):
            continue
        x, y, width, height = cv2.boundingRect(points)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2, y2 = min(source_u8.shape[1], x + width + 1), min(source_u8.shape[0], y + height + 1)
        src = source_u8[y1:y2, x1:x2]
        cand = candidate_u8[y1:y2, x1:x2]
        src_bg = cv2.GaussianBlur(src, (0, 0), max(2.0, height / 3.0))
        cand_bg = cv2.GaussianBlur(cand, (0, 0), max(2.0, height / 3.0))
        src_darkness = np.maximum(src_bg.astype(np.int16) - src.astype(np.int16), 0).astype(np.float32)
        cand_darkness = np.maximum(cand_bg.astype(np.int16) - cand.astype(np.int16), 0).astype(np.float32)
        source_ink = src_darkness >= 3.0
        candidate_ink = cand_darkness >= 3.0
        if np.count_nonzero(source_ink) < 4:
            continue
        candidate_support = cv2.dilate(candidate_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        source_support = cv2.dilate(source_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        retention = float(np.count_nonzero(source_ink & candidate_support) / np.count_nonzero(source_ink))
        false_ink = float(np.count_nonzero(candidate_ink & ~source_support) / max(np.count_nonzero(candidate_ink), 1))
        dark_ratio = float(np.count_nonzero(candidate_ink) / np.count_nonzero(source_ink))
        # A line is atomic: never combine a damaged candidate word with source
        # characters from the same line.
        accept_line = retention >= 0.94 and false_ink <= 0.025 and 0.78 <= dark_ratio <= 1.28
        if accept_line:
            continue

        region = output[y1:y2, x1:x2]
        paper = np.clip(0.10 * src_bg.astype(np.float32) + 0.90 * 255.0, 248, 255).astype(np.uint8)
        region[:] = np.maximum(region, paper)
        measured = src_darkness >= 2.5
        reference = float(np.percentile(src_darkness[measured], 75))
        gain = float(np.clip(target_darkness / max(reference, 1.0), 1.0, 1.85))
        source_render = np.clip(
            src_bg.astype(np.float32) - src_darkness * gain, 0, 255
        ).astype(np.uint8)
        region[measured] = np.minimum(region[measured], source_render[measured])
        region[measured] = np.minimum(region[measured], src[measured])
    return output


def normalize_localized_source_stroke_contrast(
    image: np.ndarray,
    source: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    *,
    target_darkness: float = 58.0,
    sharpen_strength: float = 0.0,
) -> np.ndarray:
    """Prevent patchy light/dark text while preserving source geometry.

    A single bounded gain is calculated per OCR-localised line and applied
    only to pixels already darker than their local source-paper estimate.
    OCR supplies the rectangle only; its recognised characters are never used.
    """
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < 0.35:
            continue
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape != (4, 2):
            continue
        x, y, width, height = cv2.boundingRect(points)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2, y2 = min(source_u8.shape[1], x + width + 1), min(source_u8.shape[0], y + height + 1)
        roi = source_u8[y1:y2, x1:x2]
        if sharpen_strength > 0:
            detail_blur = cv2.GaussianBlur(roi, (0, 0), 1.0)
            detail_source = cv2.addWeighted(
                roi, 1.0 + sharpen_strength, detail_blur, -sharpen_strength, 0
            )
        else:
            detail_source = roi
        background = cv2.GaussianBlur(roi, (0, 0), max(2.0, height / 3.0))
        darkness = np.maximum(
            background.astype(np.int16) - detail_source.astype(np.int16), 0
        ).astype(np.float32)
        measured = darkness >= 2.5
        if np.count_nonzero(measured) < 4:
            continue
        reference = float(np.percentile(darkness[measured], 75))
        gain = float(np.clip(target_darkness / max(reference, 1.0), 1.0, 1.85))
        # Linear gain alone leaves weak-but-real thermal strokes grey while
        # already dark strokes become black. Compress that range gently, only
        # at source-measured pixels, so ink tone is more consistent without
        # drawing OCR characters or promoting blank-paper texture.
        relative = np.clip(darkness / max(reference, 1.0), 0.0, 1.8)
        equalized_darkness = target_darkness * np.power(relative, 0.78)
        equalized_darkness = np.maximum(equalized_darkness, darkness * gain)
        # Bring weak stroke cores toward the line tone while leaving very weak
        # anti-aliased boundary pixels proportional to their evidence.
        core_floor = target_darkness * np.minimum(relative * 2.0, 0.82)
        equalized_darkness = np.maximum(equalized_darkness, core_floor)
        equalized_darkness = np.minimum(equalized_darkness, target_darkness * 1.35)
        # Gaussian background estimates dip around dense/bold glyphs. Anchor
        # rendering to the line's measured paper tone so that identical stroke
        # evidence does not receive different grey levels merely due to nearby
        # ink density.
        paper_level = float(np.percentile(roi, 80))
        render_background = np.maximum(background.astype(np.float32), paper_level)
        consistent = np.full_like(roi, 255)
        consistent[measured] = np.clip(
            render_background[measured] - equalized_darkness[measured], 0, 255
        ).astype(np.uint8)
        region = output[y1:y2, x1:x2]
        # Never make a measured source stroke lighter than the original.
        region[measured] = np.minimum(region[measured], roi[measured])
        region[measured] = np.minimum(region[measured], consistent[measured])
    return output


def repair_evidence_constrained_gaps(
    image: np.ndarray,
    source: np.ndarray,
    dsdnet_candidate: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    *,
    maximum_gap: int = 2,
    minimum_confidence: float = 0.35,
) -> np.ndarray:
    """Reconnect tiny strokes only with source endpoints and DSDNet support.

    Unlike morphological closing, this cannot bridge arbitrary dark regions:
    both source endpoints must exist, every missing pixel must be dark in the
    independently produced DSDNet candidate, and repair is restricted to a
    localised text box. The maximum supported gap is two pixels.
    """
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    candidate_u8 = dsdnet_candidate.astype(np.uint8)
    regions: list[tuple[np.ndarray, bool]] = []
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < minimum_confidence:
            continue
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape != (4, 2):
            continue
        regions.append((points, True))

    # Crusty or broken print is precisely where OCR localization may fail.
    # Add source-derived text groups that are not already covered. These boxes
    # contain no recognised wording; they only allow the same endpoint/model
    # evidence test to run on missed lines.
    source_text = detect_source_text_mask(source_u8, protection="wide")
    grouped = cv2.dilate(source_text.astype(np.uint8), np.ones((3, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    for gx, gy, gw, gh, area in stats[1:count]:
        if area < 12 or gw < 4 or gh < 2 or gw > source_u8.shape[1] * .95:
            continue
        box = np.asarray([[gx, gy], [gx + gw, gy], [gx + gw, gy + gh], [gx, gy + gh]], np.int32)
        regions.append((box, False))

    for points, ocr_localised in regions:
        x, y, width, height = cv2.boundingRect(points)
        x1, y1 = max(0, x - 1), max(0, y - 1)
        x2, y2 = min(source_u8.shape[1], x + width + 1), min(source_u8.shape[0], y + height + 1)
        src = source_u8[y1:y2, x1:x2]
        cand = candidate_u8[y1:y2, x1:x2]
        src_bg = cv2.GaussianBlur(src, (0, 0), max(2.0, height / 3.0))
        cand_bg = cv2.GaussianBlur(cand, (0, 0), max(2.0, height / 3.0))
        src_darkness = np.maximum(src_bg.astype(np.int16) - src.astype(np.int16), 0)
        cand_darkness = np.maximum(cand_bg.astype(np.int16) - cand.astype(np.int16), 0)
        endpoints = src_darkness >= 5
        candidate_support = cand_darkness >= 5
        missing = src_darkness < 3
        if not ocr_localised:
            local_text = source_text[y1:y2, x1:x2]
            endpoints &= local_text
            candidate_support &= cv2.dilate(local_text.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        repair = np.zeros(src.shape, bool)
        for gap in range(1, min(maximum_gap, 2) + 1):
            span = gap + 1
            if src.shape[1] > span:
                path = np.ones((src.shape[0], src.shape[1] - span), bool)
                for offset in range(1, span):
                    path &= candidate_support[:, offset:src.shape[1] - span + offset]
                    path &= missing[:, offset:src.shape[1] - span + offset]
                bounded = endpoints[:, :-span] & endpoints[:, span:] & path
                for offset in range(1, span):
                    repair[:, offset:src.shape[1] - span + offset] |= bounded
            if src.shape[0] > span:
                path = np.ones((src.shape[0] - span, src.shape[1]), bool)
                for offset in range(1, span):
                    path &= candidate_support[offset:src.shape[0] - span + offset, :]
                    path &= missing[offset:src.shape[0] - span + offset, :]
                bounded = endpoints[:-span, :] & endpoints[span:, :] & path
                for offset in range(1, span):
                    repair[offset:src.shape[0] - span + offset, :] |= bounded
            # Diagonal character strokes need the same bounded evidence test
            # as horizontal/vertical strokes (e.g. A, M, N and handwriting).
            if src.shape[0] > span and src.shape[1] > span:
                for mirrored in (False, True):
                    ep = endpoints[:, ::-1] if mirrored else endpoints
                    cs = candidate_support[:, ::-1] if mirrored else candidate_support
                    ms = missing[:, ::-1] if mirrored else missing
                    path = ep[:-span, :-span] & ep[span:, span:]
                    for offset in range(1, span):
                        path &= cs[offset:src.shape[0] - span + offset, offset:src.shape[1] - span + offset]
                        path &= ms[offset:src.shape[0] - span + offset, offset:src.shape[1] - span + offset]
                    diagonal = np.zeros(src.shape, bool)
                    for offset in range(1, span):
                        diagonal[offset:src.shape[0] - span + offset, offset:src.shape[1] - span + offset] |= path
                    repair |= diagonal[:, ::-1] if mirrored else diagonal
        if not repair.any():
            continue
        # Candidate controls shape and darkness; repairs cannot be stronger
        # than nearby measured source strokes.
        endpoint_cap = cv2.dilate(src_darkness.astype(np.uint8), np.ones((3, 3), np.uint8))
        repaired_darkness = np.minimum(cand_darkness, endpoint_cap)
        repaired_pixels = np.clip(255.0 - repaired_darkness * 1.45, 0, 255).astype(np.uint8)
        region = output[y1:y2, x1:x2]
        region[repair] = np.minimum(region[repair], repaired_pixels[repair])
    return output


def suppress_unprotected_paper_texture(
    image: np.ndarray,
    source: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    *,
    cleaning: str = "standard",
) -> np.ndarray:
    """Whiten paper texture outside source-located receipt content.

    OCR contributes geometry only. It never contributes characters. Compact,
    strongly measured components protect logos and marks missed by OCR, while
    elongated low-density fold/crease components are deliberately excluded.
    """
    import cv2

    output = image.astype(np.uint8).copy()
    source_u8 = source.astype(np.uint8)
    ocr_regions = np.zeros(source_u8.shape, np.uint8)
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < 0.35:
            continue
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape == (4, 2):
            cv2.fillConvexPoly(ocr_regions, points, 1)

    # OCR provides localisation, not blanket protection. Protecting every
    # pixel in a line rectangle also protects any fold crossing that line and
    # was the main cause of enhanced creases. Retain only source-derived,
    # glyph-shaped components inside those rectangles.
    source_text = detect_source_text_mask(
        source_u8, protection="wide" if cleaning == "gentle" else "standard"
    )
    protected = ((ocr_regions > 0) & source_text).astype(np.uint8)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))

    local_background = cv2.GaussianBlur(source_u8, (0, 0), 15)
    darkness = np.maximum(
        local_background.astype(np.int16) - source_u8.astype(np.int16), 0
    ).astype(np.uint8)
    strong = (darkness >= 32).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong, 8)
    for label in range(1, count):
        _x, _y, width, height, area = stats[label]
        box_area = max(int(width * height), 1)
        density = float(area / box_area)
        aspect = float(max(width, height) / max(min(width, height), 1))
        # Preserve compact printed marks/logos; reject long sparse creases.
        if 10 <= area <= 6000 and density >= 0.18 and aspect <= 8.0:
            protected[labels == label] = 1

    unprotected = protected == 0
    # Build a smooth spatial degradation map. A receipt can be clean in one
    # corner and folded, noisy or faded in another, so a single page-wide
    # cleaning strength is inappropriate. Low-frequency disagreement between
    # local and broad paper estimates highlights folds/shading without using
    # OCR content or generating pixels.
    paper = cv2.GaussianBlur(source_u8, (0, 0), 31).astype(np.float32)
    broad_paper = cv2.GaussianBlur(source_u8, (0, 0), 71).astype(np.float32)
    local_artifact = cv2.GaussianBlur(np.abs(paper - broad_paper), (0, 0), 9.0)
    artifact_strength = np.clip((local_artifact - 1.5) / 8.5, 0.0, 1.0)
    base_white = {"gentle": 0.42, "standard": 0.62, "strong": 0.76}.get(cleaning, 0.62)
    white_weight = np.clip(base_white + (1.0 - base_white) * artifact_strength, 0.0, 0.96)
    clean_paper = np.clip(
        (1.0 - white_weight) * paper + white_weight * 255.0, 238, 255
    ).astype(np.uint8)
    output[unprotected] = np.maximum(output[unprotected], clean_paper[unprotected])
    return output


def suppress_learned_paper_artifacts(
    image: np.ndarray,
    evidence_source: np.ndarray,
    ocr_lines: list[dict[str, Any]],
    checkpoint: str | Path,
    threshold: float = 0.95,
) -> np.ndarray:
    """Lighten learned crease pixels only outside protected content regions."""
    import cv2

    from .artifact_mask import predict_artifact_mask

    artifact = predict_artifact_mask(evidence_source, checkpoint, threshold)
    ocr_regions = np.zeros(image.shape, dtype=np.uint8)
    for line in ocr_lines:
        if float(line.get("confidence") or 0.0) < 0.50:
            continue
        points = np.asarray(line.get("box", []), dtype=np.int32)
        if points.shape == (4, 2):
            cv2.fillConvexPoly(ocr_regions, points, 1)
    source_text = detect_source_text_mask(evidence_source, protection="wide")
    protected = ((ocr_regions > 0) & source_text).astype(np.uint8)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))

    # Protect source-supported solid graphics such as merchant logos and seals.
    local_background = cv2.GaussianBlur(evidence_source, (0, 0), 15)
    darkness = np.maximum(
        local_background.astype(np.int16) - evidence_source.astype(np.int16), 0
    ).astype(np.uint8)
    solid = (darkness >= 35).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= 24 and width >= 3 and height >= 3:
            protected[labels == label] = 1

    # Protect straight table/rule geometry. Curved or irregular folds are not
    # promoted to content merely because they are long.
    edges = cv2.Canny(evidence_source, 40, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=45,
        minLineLength=max(30, min(image.shape) // 7), maxLineGap=4,
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= 2.0 or angle >= 88.0:
                cv2.line(protected, (x1, y1), (x2, y2), 1, 3)

    removable = artifact & (protected == 0)
    if not removable.any():
        return image
    paper_estimate = cv2.GaussianBlur(image, (0, 0), 9)
    output = image.copy()
    output[removable] = np.maximum(output[removable], paper_estimate[removable])
    return output


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
    maximum_gap: int = 0,
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
        # The official faithful view never closes gaps or paints connecting
        # pixels. It only increases contrast at already measured source pixels.
        measured = stroke_response > 0
        boosted = np.full_like(roi, 255)
        boosted[measured] = np.clip(
            255.0 - stroke_response[measured] * 1.45, 0, 255
        ).astype(np.uint8)
        output[y1:y2, x1:x2] = np.minimum(output[y1:y2, x1:x2], boosted)
    return output


def overlay_supported_ocr_text(image: np.ndarray, ocr_lines: list[dict[str, Any]], minimum_confidence: float = 0.85) -> tuple[np.ndarray, int]:
    """Render a clean, prominent reading copy from high-confidence OCR boxes."""
    import cv2
    from .language_assistance import suggest_receipt_text

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover - OpenCV remains a safe fallback
        Image = ImageDraw = ImageFont = None

    def font_path(text: str) -> Path | None:
        candidates = (
            [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simsun.ttc")]
            if any(ord(character) > 127 for character in text)
            else [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/calibri.ttf")]
        )
        return next((candidate for candidate in candidates if candidate.is_file()), None)

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
        if sum(character.isalnum() for character in text) < 2:
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
        # A noisy OCR box can span two thermal rows. Cap glyph height using the
        # document median so one bad box cannot typeset oversized, overlapping
        # text (for example Gross Amt / Svr Charge / Discount).
        target_height = max(10, min(int(height * 0.82), int(median_height * 0.92)))
        selected_font_path = font_path(text)
        pil_font = None
        pil_bbox = None
        if ImageFont is not None and selected_font_path is not None:
            font_size = target_height
            while font_size >= 7:
                candidate_font = ImageFont.truetype(str(selected_font_path), font_size)
                candidate_bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=candidate_font)
                if candidate_bbox[2] - candidate_bbox[0] <= max(width, 1) or font_size == 7:
                    pil_font, pil_bbox = candidate_font, candidate_bbox
                    break
                font_size -= 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, target_height / 22.0)
        thickness = 1 if target_height < 16 else 2
        measured = ((pil_bbox[2] - pil_bbox[0], pil_bbox[3] - pil_bbox[1])
                    if pil_bbox is not None else cv2.getTextSize(text, font, scale, thickness)[0])
        if pil_bbox is None and measured[0] > max(width, 1):
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
        # Clear only after collision checks; skipped boxes must never erase
        # genuine rules or marks. The reading copy intentionally uses near-black
        # glyphs so faint receipts remain legible at normal viewing size.
        output[y1:y2, x1:x2] = 255
        if pil_font is not None and pil_bbox is not None:
            pil_image = Image.fromarray(output)
            draw = ImageDraw.Draw(pil_image)
            draw_y = y + (height - measured[1]) // 2 - pil_bbox[1]
            draw.text((draw_x, draw_y), text, font=pil_font, fill=8)
            output = np.asarray(pil_image).copy()
        else:
            cv2.putText(output, text, (draw_x, baseline_y), font, scale, 8, thickness, cv2.LINE_AA)
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
