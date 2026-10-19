from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def load_pinned_architecture(source_root: str | Path) -> ModuleType:
    path = Path(source_root) / "basicsr" / "models" / "archs" / "DSDNet_arch.py"
    spec = importlib.util.spec_from_file_location("pinned_dsdnet_arch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pinned DSDNet architecture: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gaussian_kernel(sigma: float, size: int | None = None) -> torch.Tensor:
    if sigma <= 0:
        return torch.ones((1, 1, 1), dtype=torch.float32)
    width = size or max(3, int(2 * np.ceil(3 * sigma) + 1))
    if width % 2 == 0:
        width += 1
    coordinates = torch.arange(width, dtype=torch.float32) - width // 2
    vector = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = torch.outer(vector, vector)
    kernel /= kernel.sum()
    return kernel.unsqueeze(0)


class DSDNetReceiptAdapter(nn.Module):
    """Thin receipt adapter around the pinned DSDNet implementation.

    The inherited PriorNet, DataNet, Maxout, CGUNet, padding, convolution and
    activation implementations are not replaced. The adapter fixes grayscale
    input, eight iterations and four maxout branches, while exposing stage
    states and the tensors entering each maxout pooling operation.
    """

    def __init__(
        self,
        source_root: str | Path,
        stages: int = 8,
        branches: int = 4,
        prior_channels: int = 16,
        data_channels: int = 16,
    ) -> None:
        super().__init__()
        self.arch = load_pinned_architecture(source_root)
        self.network = self.arch.DSDNet(
            n_iter=stages,
            in_nc=1,
            out_nc=1,
            nf_p=prior_channels,
            nf_d=data_channels,
            max_nc=branches,
            dropout_rate=0.0,
            padding_mode="replicate",
        )
        self.stages = stages
        self.branches = branches
        self.prior_channels = prior_channels
        self.data_channels = data_channels

    def forward(
        self,
        observed: torch.Tensor,
        kernel: torch.Tensor,
        retain_diagnostics: bool = True,
        gradient_checkpointing: bool = False,
        summarize_diagnostics: bool = False,
    ) -> dict[str, Any]:
        net = self.network
        x = self.arch.fft_deconv(observed, kernel)
        stage_states = []
        branch_responses = []
        diagnostic_summaries = []
        ui = torch.zeros_like(net.p[0].conv(x))
        initial_residual = observed - self.arch.bconv(x, kernel, net.d[0].padding_mode)
        uj = torch.zeros_like(net.d[0].conv(initial_residual))
        for index in range(net.n):
            def run_stage(
                x_state: torch.Tensor,
                ui_state: torch.Tensor,
                uj_state: torch.Tensor,
                stage_index: int = index,
            ):
                w = torch.cat((x_state, torch.ones_like(x_state)), dim=1)
                mn = net.hn[stage_index](x_state)
                mp = net.hp[stage_index](w)
                md = net.hd[stage_index](w)
                prior_input = net.p[stage_index].conv(x_state)
                data_residual = observed - self.arch.bconv(x_state, kernel, net.d[stage_index].padding_mode)
                data_input = net.d[stage_index].conv(data_residual)
                prior_max_input = prior_input if stage_index == 0 else prior_input + ui_state
                data_max_input = data_input if stage_index == 0 else data_input + uj_state
                prior_dropped = net.p[stage_index].m.dropout(prior_max_input)
                data_dropped = net.d[stage_index].m.dropout(data_max_input)
                prior_branch_1 = net.p[stage_index].m.conv1(prior_dropped)
                prior_branch_2 = net.p[stage_index].m.conv2(prior_dropped)
                data_branch_1 = net.d[stage_index].m.conv1(data_dropped)
                data_branch_2 = net.d[stage_index].m.conv2(data_dropped)
                v = net.p[stage_index].m.ml(prior_branch_1) - net.p[stage_index].m.ml(prior_branch_2)
                z = net.d[stage_index].m.ml(data_branch_1) - net.d[stage_index].m.ml(data_branch_2)
                if stage_index == 0:
                    next_x = net.i[stage_index](x_state, observed, v, z, kernel, mp, md, mn)
                    next_ui = prior_input - v
                    next_uj = data_input - z
                else:
                    next_x = net.i[stage_index](x_state, observed, v, z, kernel, mp, md, mn, ui_state, uj_state)
                    next_ui = ui_state + prior_input - v
                    next_uj = uj_state + data_input - z
                if retain_diagnostics:
                    return next_x, next_ui, next_uj, prior_branch_1, prior_branch_2, data_branch_1, data_branch_2
                return next_x, next_ui, next_uj

            if gradient_checkpointing and self.training and not retain_diagnostics:
                x, ui, uj = checkpoint(run_stage, x, ui, uj, use_reentrant=False)
            else:
                stage_output = run_stage(x, ui, uj)
                x, ui, uj = stage_output[:3]
            if retain_diagnostics:
                stage_states.append(x)
                if summarize_diagnostics:
                    diagnostic_summaries.append({
                        "prior_disagreement": (stage_output[3] - stage_output[4]).abs().mean(dim=1, keepdim=True),
                        "data_disagreement": (stage_output[5] - stage_output[6]).abs().mean(dim=1, keepdim=True),
                    })
                else:
                    branch_responses.append({"prior": stage_output[3:5], "data": stage_output[5:7]})
        return {"restored": x, "stage_states": stage_states, "branch_responses": branch_responses, "diagnostic_summaries": diagnostic_summaries}
