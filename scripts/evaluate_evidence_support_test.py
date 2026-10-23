from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_evidence_support_validation import centre_crop, f1, roc_auc  # noqa: E402
from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel  # noqa: E402
from dsdnet_receipt.evidence_support import SupportScales, evidence_support_from_output  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass locked test evaluation of evidence support")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pixels-per-variant", type=int, default=4096)
    args = parser.parse_args()
    output_path = ROOT / "artifacts/formal_run/formal_test_support.json"
    if output_path.exists():
        raise SystemExit("Formal support test already exists; refusing an accidental second pass")

    calibration = json.loads((ROOT / "artifacts/formal_run/support_calibration.json").read_text(encoding="utf-8"))
    scales = SupportScales(**calibration["scales"])
    threshold = float(calibration["decision_threshold"])
    training = json.loads((ROOT / "configs/dsdnet_receipt_train.json").read_text(encoding="utf-8"))
    protocol, _ = load_config(ROOT / "configs/default.json")
    manifest = ROOT / "artifacts/protocol_lock/generation_manifest_611_sources.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row["project_partition"] == "test"]

    network = training["network"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DSDNetReceiptAdapter(
        protocol["dsdnet"]["source_path"], stages=int(network["stages"]),
        branches=int(network["maxout_branches"]), prior_channels=int(network["prior_channels"]),
        data_channels=int(network["data_channels"]),
    ).to(device)
    payload = torch.load(args.checkpoint.resolve(), map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    size = int(training["training"]["crop_height"])
    rng = np.random.default_rng(56)
    by_source: defaultdict[str, list[dict[str, float]]] = defaultdict(list)

    for index, row in enumerate(rows, start=1):
        degraded = centre_crop(read_gray(row["degraded_path"]), size)
        target = centre_crop(read_gray(row["support_target"]), size).ravel() > 127
        observed = torch.from_numpy(degraded).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        kernel = gaussian_kernel(float(row["parameters"]["blur_sigma"])).to(device)
        with torch.inference_mode():
            model_output = model(observed, kernel, retain_diagnostics=True, summarize_diagnostics=True)
            maps = evidence_support_from_output(model, observed, kernel, model_output, scales)
        count = min(args.pixels_per_variant, target.size)
        chosen = rng.choice(target.size, size=count, replace=False)
        values = {name: maps[name].squeeze().cpu().numpy().ravel()[chosen] for name in maps}
        truth = target[chosen]
        by_source[str(row["source_id"])].append({
            "combined_roc_auc": roc_auc(truth, values["combined"]),
            "combined_trace_f1": f1(truth, values["combined"], threshold),
            "residual_roc_auc": roc_auc(truth, values["residual"]),
            "stage_roc_auc": roc_auc(truth, values["stage"]),
            "uncertainty_roc_auc": roc_auc(truth, values["uncertainty"]),
        })
        print(f"Support test {index}/{len(rows)}", flush=True)

    metrics = {}
    for key in next(iter(by_source.values()))[0]:
        source_values = []
        for variants in by_source.values():
            finite = [row[key] for row in variants if np.isfinite(row[key])]
            if finite:
                source_values.append(float(np.mean(finite)))
        metrics[key] = float(np.mean(source_values))
    report = {
        "partition": "test", "formal_test_evidence": True,
        "locked_scales": calibration["scales"], "locked_threshold": threshold,
        "source_groups": len(by_source), "variants": len(rows),
        "sampled_pixels_per_variant": args.pixels_per_variant,
        "metrics": metrics,
        "warning": "No post-test tuning is permitted.",
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
