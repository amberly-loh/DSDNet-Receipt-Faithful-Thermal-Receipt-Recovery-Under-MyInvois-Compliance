from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.baselines import BASELINES  # noqa: E402
from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402
from dsdnet_receipt.metrics import psnr, ssim  # noqa: E402
from dsdnet_receipt.validation import stroke_fidelity_metrics  # noqa: E402


def centre_crop_pair(clean: np.ndarray, degraded: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = clean.shape
    if height < size or width < size:
        import cv2
        scale = max(size / height, size / width)
        clean = cv2.resize(clean, (int(np.ceil(width * scale)), int(np.ceil(height * scale))), interpolation=cv2.INTER_CUBIC)
        degraded = cv2.resize(degraded, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_CUBIC)
        height, width = clean.shape
    top = (height - size) // 2
    left = (width - size) // 2
    return clean[top : top + size, left : left + size], degraded[top : top + size, left : left + size]


def source_aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    by_source: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    source_psnr = [float(np.mean([float(row["psnr"]) for row in group])) for group in by_source.values()]
    source_ssim = [float(np.mean([float(row["ssim"]) for row in group])) for group in by_source.values()]
    source_l1 = [float(np.mean([float(row["l1"]) for row in group])) for group in by_source.values()]
    mean_ssim = float(np.mean(source_ssim))
    mean_l1 = float(np.mean(source_l1))
    stroke_recall = [float(np.mean([float(row["stroke_recall"]) for row in group])) for group in by_source.values()]
    deleted_rate = [float(np.mean([float(row["deleted_stroke_rate"]) for row in group])) for group in by_source.values()]
    false_rate = [float(np.mean([float(row["false_stroke_rate"]) for row in group])) for group in by_source.values()]
    return {
        "sources": len(by_source),
        "variants": len(rows),
        "mean_psnr": float(np.mean(source_psnr)),
        "mean_ssim": mean_ssim,
        "mean_l1": mean_l1,
        "mean_validation_objective": mean_l1 + 0.2 * (1.0 - mean_ssim),
        "mean_stroke_recall": float(np.mean(stroke_recall)),
        "mean_deleted_stroke_rate": float(np.mean(deleted_rate)),
        "mean_false_stroke_rate": float(np.mean(false_rate)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only restoration comparison for checkpoint selection")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts" / "protocol_lock" / "generation_manifest_611_sources.jsonl")
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()
    training = json.loads((REPO / "configs" / "dsdnet_receipt_train.json").read_text(encoding="utf-8"))
    protocol, _ = load_config(REPO / "configs" / "default.json")
    manifest = args.manifest.resolve()
    variants = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    variants = [row for row in variants if row["project_partition"] == "validation"]
    source_ids = sorted({row["source_id"] for row in variants})
    if args.max_sources:
        source_ids = source_ids[: args.max_sources]
        variants = [row for row in variants if row["source_id"] in source_ids]
    if not variants:
        raise SystemExit("No validation variants are available")

    network = training["network"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DSDNetReceiptAdapter(
        protocol["dsdnet"]["source_path"],
        stages=int(network["stages"]),
        branches=int(network["maxout_branches"]),
        prior_channels=int(network["prior_channels"]),
        data_channels=int(network["data_channels"]),
    ).to(device)
    payload = torch.load(args.checkpoint.resolve(), map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    crop_size = int(training["training"]["crop_height"])
    rows: list[dict[str, object]] = []
    for index, variant in enumerate(variants, start=1):
        clean, degraded = centre_crop_pair(
            read_gray(variant["clean_reference"]),
            read_gray(variant["degraded_path"]),
            crop_size,
        )
        foreground_path = variant.get("clean_foreground") or str(
            Path(variant["degraded_path"]).with_name("clean_foreground.png")
        )
        foreground, _ = centre_crop_pair(
            read_gray(foreground_path),
            read_gray(foreground_path),
            crop_size,
        )
        foreground = foreground > 127
        for method_id, method in BASELINES.items():
            started = time.perf_counter()
            restored = method(degraded)
            fidelity = stroke_fidelity_metrics(clean, restored, foreground)
            rows.append({
                "source_id": variant["source_id"], "variant_id": variant["variant_id"],
                "severity": variant["severity"], "method": method_id,
                "psnr": psnr(clean, restored), "ssim": ssim(clean, restored),
                "l1": float(np.mean(np.abs(clean.astype(np.float32) - restored.astype(np.float32))) / 255.0),
                "seconds": time.perf_counter() - started,
                **fidelity,
            })
        observed = torch.from_numpy(degraded).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        kernel = gaussian_kernel(float(variant["parameters"]["blur_sigma"])).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            restored_tensor = model(observed, kernel, retain_diagnostics=False)["restored"].clamp(0, 1)
        restored = (restored_tensor.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)
        fidelity = stroke_fidelity_metrics(clean, restored, foreground)
        rows.append({
            "source_id": variant["source_id"], "variant_id": variant["variant_id"],
            "severity": variant["severity"], "method": "Proposed_DSDNet_Receipt",
            "psnr": psnr(clean, restored), "ssim": ssim(clean, restored),
            "l1": float(F.l1_loss(restored_tensor, torch.from_numpy(clean).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)).cpu()),
            "seconds": time.perf_counter() - started,
            **fidelity,
        })
        print(f"Validated {index}/{len(variants)} variants", flush=True)

    # Include the run directory so checkpoints with the same iteration number
    # from different experiments never overwrite one another's evidence.
    output = REPO / "artifacts" / "validation_restoration" / f"{args.checkpoint.parent.name}_{args.checkpoint.stem}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "records.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summary = {
        "partition": "validation only",
        "formal_test_evidence": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iteration": int(payload.get("iteration", -1)),
        "manifest": str(manifest),
        "crop_size": crop_size,
        "methods": {method: source_aggregate(method_rows) for method, method_rows in grouped.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
