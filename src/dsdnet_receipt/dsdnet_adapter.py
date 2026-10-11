from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import nn


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

    def __init__(self, source_root: str | Path, stages: int = 8, branches: int = 4) -> None:
        super().__init__()
        self.arch = load_pinned_architecture(source_root)
        self.network = self.arch.DSDNet(
            n_iter=stages,
            in_nc=1,
            out_nc=1,
            nf_p=49,
            nf_d=48,
            max_nc=branches,
            dropout_rate=0.0,
            padding_mode="replicate",
        )
        self.stages = stages
        self.branches = branches

    def forward(self, observed: torch.Tensor, kernel: torch.Tensor, retain_diagnostics: bool = True) -> dict[str, Any]:
        net = self.network
        x = self.arch.fft_deconv(observed, kernel)
        stage_states = []
        branch_responses = []
        ui = uj = None
        for index in range(net.n):
            w = torch.cat((x, torch.ones_like(x)), dim=1)
            mn = net.hn[index](x)
            mp = net.hp[index](w)
            md = net.hd[index](w)
            prior_input = net.p[index].conv(x)
            data_residual = observed - self.arch.bconv(x, kernel, net.d[index].padding_mode)
            data_input = net.d[index].conv(data_residual)
            prior_max_input = prior_input if ui is None else prior_input + ui
            data_max_input = data_input if uj is None else data_input + uj
            prior_dropped = net.p[index].m.dropout(prior_max_input)
            data_dropped = net.d[index].m.dropout(data_max_input)
            prior_branches = (net.p[index].m.conv1(prior_dropped), net.p[index].m.conv2(prior_dropped))
            data_branches = (net.d[index].m.conv1(data_dropped), net.d[index].m.conv2(data_dropped))
            v = net.p[index].m.ml(prior_branches[0]) - net.p[index].m.ml(prior_branches[1])
            z = net.d[index].m.ml(data_branches[0]) - net.d[index].m.ml(data_branches[1])
            if index == 0:
                x = net.i[index](x, observed, v, z, kernel, mp, md, mn)
                ui = prior_input - v
                uj = data_input - z
            else:
                x = net.i[index](x, observed, v, z, kernel, mp, md, mn, ui, uj)
                ui = ui + prior_input - v
                uj = uj + data_input - z
            if retain_diagnostics:
                stage_states.append(x)
                branch_responses.append({"prior": prior_branches, "data": data_branches})
        return {"restored": x, "stage_states": stage_states, "branch_responses": branch_responses}
