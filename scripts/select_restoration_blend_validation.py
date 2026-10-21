from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compare_fixed_ocr_validation import DATASET, CORPUS, field_scores, source_summary  # noqa: E402
from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402
from dsdnet_receipt.metrics import cer, psnr, ssim, wer  # noqa: E402
from dsdnet_receipt.ocr import PaddleOCRAdapter  # noqa: E402
from dsdnet_receipt.restoration import RestorationEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only source-preserving blend selection")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts" / "protocol_lock" / "generation_manifest_611_sources.jsonl")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()
    if any(not 0.0 < alpha <= 1.0 for alpha in args.alphas):
        raise SystemExit("Each alpha must be in (0, 1]")
    config, _ = load_config(REPO / "configs" / "default.json")
    config["dsdnet"]["checkpoint"] = str(args.checkpoint.resolve())
    with (DATASET / "sroie_protocol_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        references = {row["source_id"]: row for row in csv.DictReader(stream)}
    variants = [json.loads(line) for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines() if line]
    source_ids = sorted({row["source_id"] for row in variants if row["project_partition"] == "validation"})[: args.sources]
    selected = [row for row in variants if row["source_id"] in source_ids]
    restoration = RestorationEngine(config)
    ocr = PaddleOCRAdapter(config)
    checkpoint = args.checkpoint.resolve()
    checkpoint_stat = checkpoint.stat()
    checkpoint_identity = hashlib.sha256(
        f"{checkpoint}|{checkpoint_stat.st_size}|{checkpoint_stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:12]
    output = REPO / "artifacts" / "validation_blend_selection" / f"{checkpoint.stem}_{checkpoint_identity}"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "records.jsonl"
    rows: list[dict[str, object]] = []
    if progress_path.exists():
        rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line]
    completed = {(str(row["source_id"]), str(row["variant_id"]), float(row["alpha"])) for row in rows}
    for index, variant in enumerate(selected, start=1):
        if all((str(variant["source_id"]), str(variant["variant_id"]), alpha) in completed for alpha in args.alphas):
            print(f"Reused blend grid for {index}/{len(selected)} validation variants", flush=True)
            continue
        degraded = read_gray(variant["degraded_path"])
        clean = read_gray(variant["clean_reference"])
        proposed = restoration.run(degraded, float(variant["parameters"]["blur_sigma"]))
        if not proposed.model_available:
            raise RuntimeError(proposed.error or proposed.status)
        for alpha in args.alphas:
            if (str(variant["source_id"]), str(variant["variant_id"]), alpha) in completed:
                continue
            blended = np.clip(
                alpha * proposed.image.astype(np.float32) + (1.0 - alpha) * degraded.astype(np.float32),
                0, 255,
            ).round().astype(np.uint8)
            prediction = ocr.run(blended)
            correct, eligible, accuracy = field_scores(references[variant["source_id"]], prediction.text)
            row = {
                "source_id": variant["source_id"], "variant_id": variant["variant_id"], "severity": variant["severity"],
                "method": f"Proposed_blend_{alpha:.2f}", "alpha": alpha,
                "psnr": psnr(clean, blended), "ssim": ssim(clean, blended),
                "cer": cer(references[variant["source_id"]]["ocr_reference_text"], prediction.text),
                "wer": wer(references[variant["source_id"]]["ocr_reference_text"], prediction.text),
                "correct_fields": correct, "eligible_fields": eligible, "field_accuracy": accuracy,
                "processing_seconds": 0.0, "ocr_seconds": 0.0,
                "ocr_status": prediction.status, "ocr_error": prediction.error or "",
            }
            rows.append(row)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        print(f"Evaluated blend grid for {index}/{len(selected)} validation variants", flush=True)
    with (output / "records.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries = {method: source_summary(method_rows) for method, method_rows in grouped.items()}
    ranked = sorted(
        summaries,
        key=lambda method: (-float(summaries[method]["field_accuracy"] or 0.0), float(summaries[method]["mean_cer"]), method),
    )
    payload = {
        "partition": "validation only", "formal_test_evidence": False,
        "selection_rule": "maximum field accuracy, then minimum CER, then method identifier",
        "selected": ranked[0], "methods": summaries,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
