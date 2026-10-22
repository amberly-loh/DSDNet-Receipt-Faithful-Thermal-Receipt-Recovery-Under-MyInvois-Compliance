from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.artifact_mask import ReceiptArtifactMaskNet  # noqa: E402
from train_artifact_mask import make_batch  # noqa: E402


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only artifact-mask threshold selection")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=64)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu")
    config = payload["config"]
    manifest = (REPO / "configs" / config["generation_manifest"]).resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    records = [row for row in records if row["project_partition"] == "validation"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReceiptArtifactMaskNet(int(config["training"]["width"])).to(device)
    model.load_state_dict(payload["model"]); model.eval()
    accumulators = {threshold: [0, 0, 0, 0] for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)}
    for index in range(args.batches):
        observed, target = make_batch(
            records,
            config["natural_receipt_augmentation"],
            int(config["training"]["crop_size"]),
            2,
            int(config["seed"]) + 90_000 + index,
        )
        probability = torch.sigmoid(model(observed.to(device)))
        truth = target.to(device) >= 0.5
        for threshold, values in accumulators.items():
            prediction = probability >= threshold
            values[0] += int((prediction & truth).sum())
            values[1] += int((prediction | truth).sum())
            values[2] += int((prediction & ~truth).sum())
            values[3] += int((~truth).sum())
    grid = []
    for threshold, (intersection, union, false_positive, background) in accumulators.items():
        grid.append({
            "threshold": threshold,
            "crease_iou": intersection / max(union, 1),
            "false_positive_rate": false_positive / max(background, 1),
        })
    limit = float(config["selection"]["maximum_false_positive_rate"])
    eligible = [row for row in grid if row["false_positive_rate"] <= limit]
    selected = max(eligible, key=lambda row: row["crease_iou"], default=None)
    summary = {
        "partition": "validation only",
        "formal_test_evidence": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "threshold_grid": grid,
        "selected": selected,
        "deployment_enabled": bool(selected and selected["crease_iou"] >= float(config["selection"]["minimum_crease_iou"])),
    }
    output = args.checkpoint.parent / "threshold_selection.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
