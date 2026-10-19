from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    readable = cv2.fastNlMeansDenoising(flattened, None, h=7, templateWindowSize=7, searchWindowSize=21)
    # Light paper grain and fold shading are mapped to clean paper. Dark ink,
    # including handwriting and signatures, is retained whether OCR finds it
    # or not.
    readable[readable > 190] = 255
    return readable
