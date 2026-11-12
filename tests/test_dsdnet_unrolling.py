"""Conformance checks for the adapter's unrolled forward pass."""
from pathlib import Path
import torch

from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel


def test_eight_stage_adapter_matches_pinned_forward_and_exposes_branches():
    torch.set_num_threads(1)
    torch.manual_seed(56)
    vendor = Path(__file__).resolve().parents[2] / 'vendor' / 'DSDNet'
    model = DSDNetReceiptAdapter(vendor, stages=8, branches=4, prior_channels=16, data_channels=16).eval()
    image = torch.rand(1, 1, 32, 32)
    kernel = gaussian_kernel(0.6)
    with torch.inference_mode():
        result = model(image, kernel, retain_diagnostics=True)
        reference = model.network(image, kernel)
    assert len(result['stage_states']) == 8
    assert len(result['branch_responses']) == 8
    assert result['restored'].shape == image.shape
    assert torch.isfinite(result['restored']).all()
    assert torch.allclose(result['restored'], reference, rtol=1e-5, atol=1e-6)
    for stage in result['branch_responses']:
        for key in ('prior', 'data'):
            assert len(stage[key]) == 2
            assert all(branch.shape[1] == 64 for branch in stage[key])
