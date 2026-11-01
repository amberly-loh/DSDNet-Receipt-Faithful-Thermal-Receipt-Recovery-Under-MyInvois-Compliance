from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SupportScales:
    """Validation-selected scales for the three support components."""

    residual: float
    stage: float
    uncertainty: float

    def __post_init__(self) -> None:
        if min(self.residual, self.stage, self.uncertainty) <= 0:
            raise ValueError("Support scales must be positive")


def _match_size(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape[-2:] == reference.shape[-2:]:
        return value
    return F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)


def _confidence(error: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.exp(-error.clamp_min(0.0) / max(float(scale), 1e-8)).clamp(0.0, 1.0)


def _ink_likelihood(restored: torch.Tensor) -> torch.Tensor:
    """Estimate dark receipt strokes relative to their local paper tone."""

    local_paper = F.avg_pool2d(restored, kernel_size=31, stride=1, padding=15)
    local_contrast = (local_paper - restored).clamp_min(0.0)
    return (1.0 - torch.exp(-local_contrast / 0.05)).clamp(0.0, 1.0)


def support_components(
    observed: torch.Tensor,
    kernel: torch.Tensor,
    model_output: dict[str, Any],
    blur_operator: Any,
    padding_mode: str,
    scales: SupportScales,
) -> dict[str, torch.Tensor]:
    """Compute source-supported confidence without using clean references.

    The maps correspond to the thesis' A_res, A_stage and A_unc terms.  A_res
    measures forward-model agreement with the observed receipt, A_stage measures
    convergence of the last two DSDNet stages, and A_unc measures disagreement
    between the learned maxout branches.  Clean images and support targets are
    deliberately absent from this runtime function.
    """

    terms = support_evidence_terms(
        observed, kernel, model_output, blur_operator, padding_mode
    )
    ink = terms["ink"]
    residual_error = terms["residual_error"]
    stage_error = terms["stage_error"]
    branch_error = terms["uncertainty_error"]

    # Support is support for recovered content, not confidence that blank paper
    # is blank. Requiring locally dark restored evidence prevents the large
    # background area from dominating calibration and false acceptance.
    residual = ink * _confidence(residual_error, scales.residual)
    stage = ink * _confidence(stage_error, scales.stage)
    uncertainty = ink * _confidence(branch_error, scales.uncertainty)
    combined = torch.stack((residual, stage, uncertainty), dim=0).mean(dim=0)
    return {
        "residual": residual,
        "stage": stage,
        "uncertainty": uncertainty,
        "combined": combined,
    }


def support_evidence_terms(
    observed: torch.Tensor,
    kernel: torch.Tensor,
    model_output: dict[str, Any],
    blur_operator: Any,
    padding_mode: str,
) -> dict[str, torch.Tensor]:
    """Return uncalibrated evidence terms for validation-only scale fitting."""

    restored = model_output["restored"]
    stages = model_output.get("stage_states", [])
    diagnostics = model_output.get("diagnostic_summaries", [])
    if len(stages) < 2:
        raise ValueError("At least two retained DSDNet stages are required")
    if not diagnostics:
        raise ValueError("Summarised branch diagnostics are required")

    projected = blur_operator(restored, kernel, padding_mode)
    residual_error = (observed - projected).abs()
    stage_error = (stages[-1] - stages[-2]).abs()

    latest = diagnostics[-1]
    branch_error = 0.5 * (
        latest["prior_disagreement"] + latest["data_disagreement"]
    )
    branch_error = _match_size(branch_error, restored)

    ink = _ink_likelihood(restored)
    return {
        "ink": ink,
        "residual_error": residual_error,
        "stage_error": stage_error,
        "uncertainty_error": branch_error,
    }


def evidence_support_from_output(
    adapter: Any,
    observed: torch.Tensor,
    kernel: torch.Tensor,
    model_output: dict[str, Any],
    scales: SupportScales,
) -> dict[str, torch.Tensor]:
    """Adapter convenience wrapper kept separate for straightforward testing."""

    return support_components(
        observed=observed,
        kernel=kernel,
        model_output=model_output,
        blur_operator=adapter.arch.bconv,
        padding_mode=adapter.network.d[0].padding_mode,
        scales=scales,
    )
