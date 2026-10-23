from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.dsdnet_adapter import DSDNetReceiptAdapter, gaussian_kernel  # noqa: E402
from dsdnet_receipt.evidence_support import support_evidence_terms  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402


def centre_crop(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape
    if height < size or width < size:
        scale = max(size / height, size / width)
        image = cv2.resize(
            image, (int(np.ceil(width * scale)), int(np.ceil(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        height, width = image.shape
    top, left = (height - size) // 2, (width - size) // 2
    return image[top : top + size, left : left + size]


def roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(bool)
    positives, negatives = int(target.sum()), int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    sorted_scores = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(score)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
    return float((ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def f1(target: np.ndarray, score: np.ndarray, threshold: float) -> float:
    predicted = score >= threshold
    tp = int(np.count_nonzero(predicted & target))
    fp = int(np.count_nonzero(predicted & ~target))
    fn = int(np.count_nonzero(~predicted & target))
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def macro_metric(samples: list[dict[str, object]], key: str, threshold: float | None = None) -> float:
    by_source: defaultdict[str, list[float]] = defaultdict(list)
    for sample in samples:
        value = roc_auc(sample["target"], sample[key]) if threshold is None else f1(sample["target"], sample[key], threshold)
        if np.isfinite(value):
            by_source[str(sample["source_id"])].append(float(value))
    return float(np.mean([np.mean(values) for values in by_source.values()]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only calibration of thesis evidence support")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/protocol_lock/generation_manifest_611_sources.jsonl")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--pixels-per-variant", type=int, default=4096)
    args = parser.parse_args()

    training = json.loads((ROOT / "configs/dsdnet_receipt_train.json").read_text(encoding="utf-8"))
    protocol, _ = load_config(ROOT / "configs/default.json")
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if row["project_partition"] == "validation"]
    source_ids = sorted({str(row["source_id"]) for row in rows})
    if args.max_sources:
        allowed = set(source_ids[: args.max_sources])
        rows = [row for row in rows if str(row["source_id"]) in allowed]
    if not rows:
        raise SystemExit("No validation rows found; test data must not be substituted")

    network = training["network"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DSDNetReceiptAdapter(
        protocol["dsdnet"]["source_path"], stages=int(network["stages"]),
        branches=int(network["maxout_branches"]), prior_channels=int(network["prior_channels"]),
        data_channels=int(network["data_channels"]),
    ).to(device)
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    size = int(training["training"]["crop_height"])
    rng = np.random.default_rng(56)
    samples: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        degraded = centre_crop(read_gray(row["degraded_path"]), size)
        target = centre_crop(read_gray(row["support_target"]), size) > 127
        observed = torch.from_numpy(degraded).float().div_(255).unsqueeze(0).unsqueeze(0).to(device)
        kernel = gaussian_kernel(float(row["parameters"]["blur_sigma"])).to(device)
        with torch.inference_mode():
            output = model(observed, kernel, retain_diagnostics=True, summarize_diagnostics=True)
            terms = support_evidence_terms(
                observed, kernel, output, model.arch.bconv,
                model.network.d[0].padding_mode,
            )
        count = min(int(args.pixels_per_variant), target.size)
        chosen = rng.choice(target.size, size=count, replace=False)
        sample: dict[str, object] = {"source_id": row["source_id"], "target": target.ravel()[chosen]}
        sample["ink"] = terms["ink"].squeeze().cpu().numpy().ravel()[chosen]
        for name in ("residual", "stage", "uncertainty"):
            sample[f"error_{name}"] = terms[f"{name}_error"].squeeze().cpu().numpy().ravel()[chosen]
        samples.append(sample)
        print(f"Evidence {index}/{len(rows)}", flush=True)

    scale_grid = (.01, .02, .04, .08, .16, .32)
    # Equal per-variant tuning subsamples keep the exhaustive grid tractable
    # and prevent large receipts from dominating scale selection. Final metrics
    # below still use every sampled validation pixel and source-macro averaging.
    tuning = {
        key: np.concatenate([np.asarray(sample[key])[:128] for sample in samples])
        for key in ("target", "ink", "error_residual", "error_stage", "error_uncertainty")
    }
    best: tuple[float, tuple[float, float, float]] | None = None
    for residual in scale_grid:
        for stage in scale_grid:
            for uncertainty in scale_grid:
                combined = tuning["ink"] * (
                    np.exp(-tuning["error_residual"] / residual)
                    + np.exp(-tuning["error_stage"] / stage)
                    + np.exp(-tuning["error_uncertainty"] / uncertainty)
                ) / 3.0
                score = roc_auc(tuning["target"], combined)
                if best is None or score > best[0]:
                    best = (score, (residual, stage, uncertainty))
    assert best is not None
    residual, stage, uncertainty = best[1]
    for sample in samples:
        sample["residual"] = sample["ink"] * np.exp(-sample["error_residual"] / residual)
        sample["stage"] = sample["ink"] * np.exp(-sample["error_stage"] / stage)
        sample["uncertainty"] = sample["ink"] * np.exp(-sample["error_uncertainty"] / uncertainty)
        sample["combined"] = (sample["residual"] + sample["stage"] + sample["uncertainty"]) / 3.0
    threshold_grid = np.linspace(.05, .95, 37)
    threshold, validation_f1 = max(
        ((float(value), macro_metric(samples, "combined", float(value))) for value in threshold_grid),
        key=lambda item: item[1],
    )
    report = {
        "schema_version": 1,
        "partition": "validation only",
        "formal_test_evidence": False,
        "selection_rule": "maximise equal-per-variant tuning-subsample ROC AUC; then maximise source-macro Trace-F1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "manifest": str(args.manifest.resolve()),
        "source_groups": len({str(row["source_id"]) for row in rows}),
        "variants": len(rows),
        "sampled_pixels_per_variant": int(args.pixels_per_variant),
        "scales": {"residual": residual, "stage": stage, "uncertainty": uncertainty},
        "decision_threshold": threshold,
        "validation_metrics": {
            "tuning_subsample_roc_auc": best[0],
            "combined_roc_auc": macro_metric(samples, "combined"),
            "combined_trace_f1": validation_f1,
            "residual_roc_auc": macro_metric(samples, "residual"),
            "stage_roc_auc": macro_metric(samples, "stage"),
            "uncertainty_roc_auc": macro_metric(samples, "uncertainty"),
        },
        "warning": "These are model-selection results, not held-out test claims.",
    }
    output = ROOT / "artifacts/formal_run/support_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
