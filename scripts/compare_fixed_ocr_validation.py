from __future__ import annotations

import argparse
import csv
import hashlib
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
from dsdnet_receipt.restoration import RestorationEngine  # noqa: E402

DATASET = REPO.parent / "FYP datasets" / "External" / "SROIE-Malaysia"
CORPUS = REPO / "artifacts" / "controlled_degradation_sroie"


def digits(value: str) -> str:
    return re.sub(r"[^0-9]", "", value)


def alphanumeric(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def field_scores(reference: dict[str, str], prediction: str) -> tuple[int, int, float]:
    checks = []
    if reference["transaction_date"].strip():
        checks.append(digits(reference["transaction_date"]) in digits(prediction))
    if reference["total_amount"].strip():
        checks.append(digits(reference["total_amount"]) in digits(prediction))
    if reference["receipt_invoice_number"].strip():
        checks.append(alphanumeric(reference["receipt_invoice_number"]) in alphanumeric(prediction))
    correct = sum(checks)
    return correct, len(checks), correct / len(checks) if checks else float("nan")


def source_summary(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_id"])].append(row)
    def aggregate(key: str) -> float:
        return float(np.mean([np.mean([float(row[key]) for row in group]) for group in groups.values()]))
    eligible = [row for row in rows if int(row["eligible_fields"]) > 0]
    return {
        "sources": len(groups), "variants": len(rows),
        "mean_psnr": aggregate("psnr"), "mean_ssim": aggregate("ssim"),
        "mean_cer": aggregate("cer"), "mean_wer": aggregate("wer"),
        "field_accuracy": float(np.mean([float(row["field_accuracy"]) for row in eligible])) if eligible else None,
        "failed_ocr_count": sum(row["ocr_status"] != "ok" for row in rows),
        "mean_processing_seconds": aggregate("processing_seconds"),
        "mean_ocr_seconds": aggregate("ocr_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-OCR comparison of baselines and proposed DSDNet-Receipt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sources", type=int)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts" / "protocol_lock" / "generation_manifest_611_sources.jsonl")
    args = parser.parse_args()
    config, _ = load_config(REPO / "configs" / "default.json")
    config["dsdnet"]["checkpoint"] = str(args.checkpoint.resolve())
    if args.partition == "test":
        calibration = json.loads((REPO / "artifacts/formal_run/support_calibration.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest()
        if calibration.get("checkpoint_sha256") != digest:
            raise SystemExit("Checkpoint differs from validation-locked checkpoint")
    with (DATASET / "sroie_protocol_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        references = {row["source_id"]: row for row in csv.DictReader(stream)}
    variants = [json.loads(line) for line in args.manifest.resolve().read_text(encoding="utf-8").splitlines() if line]
    source_ids = sorted({row["source_id"] for row in variants if row["project_partition"] == args.partition})
    if args.sources:
        source_ids = source_ids[: args.sources]
    selected = [row for row in variants if row["source_id"] in source_ids]
    ocr = PaddleOCRAdapter(config)
    restoration = RestorationEngine(config)
    rows: list[dict[str, object]] = []
    for index, variant in enumerate(selected, start=1):
        degraded = read_gray(variant["degraded_path"])
        clean = read_gray(variant["clean_reference"])
        outputs: dict[str, np.ndarray] = {}
        process_times: dict[str, float] = {}
        for method_id, method in BASELINES.items():
            process_started = time.perf_counter()
            outputs[method_id] = method(degraded)
            process_times[method_id] = time.perf_counter() - process_started
        proposed_started = time.perf_counter()
        proposed = restoration.run(degraded, float(variant["parameters"]["blur_sigma"]))
        proposed_seconds = time.perf_counter() - proposed_started
        if not proposed.model_available:
            raise RuntimeError(proposed.error or proposed.status)
        outputs["Proposed_DSDNet_Receipt"] = proposed.image
        process_times["Proposed_DSDNet_Receipt"] = proposed_seconds
        for method_id, output in outputs.items():
            ocr_started = time.perf_counter()
            prediction = ocr.run(output)
            ocr_seconds = time.perf_counter() - ocr_started
            correct, eligible, accuracy = field_scores(references[variant["source_id"]], prediction.text)
            rows.append({
                "source_id": variant["source_id"], "variant_id": variant["variant_id"], "severity": variant["severity"],
                "method": method_id, "psnr": psnr(clean, output), "ssim": ssim(clean, output),
                "cer": cer(references[variant["source_id"]]["ocr_reference_text"], prediction.text),
                "wer": wer(references[variant["source_id"]]["ocr_reference_text"], prediction.text),
                "correct_fields": correct, "eligible_fields": eligible, "field_accuracy": accuracy,
                "processing_seconds": process_times[method_id], "ocr_seconds": ocr_seconds,
                "ocr_status": prediction.status, "ocr_error": prediction.error or "",
            })
        print(f"Evaluated {index}/{len(selected)} {args.partition} variants", flush=True)
    output_root = "formal_test_fixed_ocr" if args.partition == "test" else "validation_fixed_ocr"
    output = REPO / "artifacts" / output_root / args.checkpoint.stem
    if args.partition == "test" and (output / "summary.json").exists():
        raise SystemExit("Formal OCR test already exists; refusing an accidental second pass")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "records.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    payload = {
        "partition": args.partition, "formal_test_evidence": args.partition == "test",
        "checkpoint": str(args.checkpoint.resolve()), "sources": len(source_ids),
        "manifest": str(args.manifest.resolve()),
        "field_accuracy_scope": "transaction date, total amount, and receipt/invoice number when annotated",
        "methods": {method: source_summary(method_rows) for method, method_rows in grouped.items()},
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
