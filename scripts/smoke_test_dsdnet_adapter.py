from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel  # noqa: E402

def main() -> None:
    config, _ = load_config(REPO / "configs" / "default.json")
    torch.manual_seed(config["base_seed"])
    model = DSDNetReceiptAdapter(config["dsdnet"]["source_path"], stages=8, branches=4).eval()
    observed = torch.rand((1, 1, 16, 16), dtype=torch.float32)
    kernel = gaussian_kernel(1.0, 7)
    with torch.no_grad():
        output = model(observed, kernel)
    report = {
        "status": "PASS",
        "source_commit": config["dsdnet"]["source_commit"],
        "input_shape": list(observed.shape),
        "output_shape": list(output["restored"].shape),
        "stage_states": len(output["stage_states"]),
        "branch_stages": len(output["branch_responses"]),
        "prior_branch_channels_stage_1": list(output["branch_responses"][0]["prior"][0].shape),
        "data_branch_channels_stage_1": list(output["branch_responses"][0]["data"][0].shape),
        "device": str(observed.device),
        "interpretation": "Reduced-size CPU structural smoke test only; random weights are not restoration evidence.",
    }
    if report["output_shape"] != [1, 1, 16, 16] or report["stage_states"] != 8:
        report["status"] = "FAIL"
    target = REPO / "artifacts" / "readiness" / "dsdnet_adapter_smoke_test.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
