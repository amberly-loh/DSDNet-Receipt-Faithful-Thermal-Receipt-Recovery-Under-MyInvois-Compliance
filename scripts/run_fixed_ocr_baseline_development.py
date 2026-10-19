from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.baselines import BASELINES  # noqa: E402
from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.images import read_gray  # noqa: E402
from dsdnet_receipt.metrics import cer, psnr, ssim, wer  # noqa: E402
from dsdnet_receipt.ocr import PaddleOCRAdapter  # noqa: E402

DATASET = REPO.parent / "FYP datasets" / "External" / "SROIE-Malaysia"
CORPUS = REPO / "artifacts" / "controlled_degradation_sroie"

def normalized_amount(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)

def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-OCR development gate for Baselines A-C.")
    parser.add_argument("--sources", type=int, default=10)
    args = parser.parse_args()
    config, _ = load_config(REPO / "configs" / "default.json")
    with (DATASET / "sroie_protocol_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        references = {row["source_id"]: row for row in csv.DictReader(stream)}
    variants = [json.loads(line) for line in (CORPUS / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    train_sources = sorted({row["source_id"] for row in variants if row["project_partition"] == "train"})[: args.sources]
    selected = [row for row in variants if row["source_id"] in train_sources]
    ocr = PaddleOCRAdapter(config)
    results = []
    for index, variant in enumerate(selected, start=1):
        degraded = read_gray(variant["degraded_path"])
        clean = read_gray(variant["clean_reference"])
        reference = references[variant["source_id"]]
        for baseline_id, method in BASELINES.items():
            started = time.perf_counter()
            output = method(degraded)
            process_seconds = time.perf_counter() - started
            ocr_started = time.perf_counter()
            prediction = ocr.run(output)
            ocr_seconds = time.perf_counter() - ocr_started
            expected_total = normalized_amount(reference["total_amount"])
            predicted_amounts = {normalized_amount(line["text"]) for line in prediction.lines}
            total_eligible = bool(expected_total)
            results.append({
                "source_id": variant["source_id"],
                "variant_id": variant["variant_id"],
                "severity": variant["severity"],
                "baseline": baseline_id,
                "psnr": psnr(clean, output),
                "ssim": ssim(clean, output),
                "cer": cer(reference["ocr_reference_text"], prediction.text),
                "wer": wer(reference["ocr_reference_text"], prediction.text),
                "total_field_eligible": total_eligible,
                "total_field_exact": bool(total_eligible and expected_total in predicted_amounts),
                "processing_seconds": process_seconds,
                "ocr_seconds": ocr_seconds,
                "ocr_status": prediction.status,
                "ocr_lines": len(prediction.lines),
                "ocr_error": prediction.error or "",
            })
        print(f"Evaluated {index}/{len(selected)} variants", flush=True)
    output_dir = REPO / "artifacts" / "baseline_fixed_ocr_development"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "records.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[str(row["baseline"])].append(row)
    summary = {}
    for baseline, rows in grouped.items():
        eligible = [row for row in rows if row["total_field_eligible"]]
        summary[baseline] = {
            "records": len(rows),
            "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
            "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
            "mean_cer": float(np.mean([row["cer"] for row in rows])),
            "mean_wer": float(np.mean([row["wer"] for row in rows])),
            "total_field_accuracy": float(np.mean([row["total_field_exact"] for row in eligible])) if eligible else None,
            "failed_ocr_count": sum(row["ocr_status"] != "ok" for row in rows),
            "mean_processing_seconds": float(np.mean([row["processing_seconds"] for row in rows])),
            "mean_ocr_seconds": float(np.mean([row["ocr_seconds"] for row in rows])),
        }
    payload = {
        "status": "PASS" if all(value["failed_ocr_count"] == 0 for value in summary.values()) else "FAIL",
        "scope": f"development gate only: first {args.sources} project-training sources, all three severities",
        "formal_test_evidence": False,
        "field_accuracy_scope": "total_amount only; date and receipt/invoice number await manual annotation",
        "methods": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
