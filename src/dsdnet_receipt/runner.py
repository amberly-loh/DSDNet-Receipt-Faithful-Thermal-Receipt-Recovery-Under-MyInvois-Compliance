from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .baselines import BASELINES
from .config import write_json
from .degradation import generate_variant
from .images import read_gray, sha256_file, write_image
from .manifest import clean_candidates, read_manifest
from .metrics import psnr, ssim
from .ocr import PaddleOCRAdapter
from .preprocessing import preprocess


def run_pilot(config: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    records = read_manifest(manifest_path)
    candidates = clean_candidates(records, 10)
    dataset_root = Path(config["dataset_root"])
    output_root = Path(config["artifacts_root"]) / "pilot_degradation"
    generated: list[dict[str, Any]] = []
    for candidate in candidates:
        source = dataset_root / candidate["relative_path"]
        for sequence, severity in enumerate(config["degradation"]["severities"], start=1):
            generated.append(
                generate_variant(
                    source,
                    candidate["source_id"],
                    f"V{sequence:02d}",
                    severity,
                    output_root,
                    config,
                )
            )
    write_json(output_root / "generation_manifest.json", generated)
    write_json(output_root / "candidate_clean_sources.json", candidates)
    return generated


def verify_reproducibility(config: dict[str, Any], generated: list[dict[str, Any]]) -> dict[str, Any]:
    from tempfile import TemporaryDirectory

    checks = []
    with TemporaryDirectory() as temporary:
        for record in generated:
            regenerated = generate_variant(
                record["source_path"],
                record["source_id"],
                record["variant_id"],
                record["severity"],
                temporary,
                config,
            )
            checks.append(
                {
                    "source_id": record["source_id"],
                    "variant_id": record["variant_id"],
                    "severity": record["severity"],
                    "identical": sha256_file(record["output_path"]) == sha256_file(regenerated["output_path"]),
                }
            )
    return {"all_identical": all(item["identical"] for item in checks), "checks": checks}


def run_baselines(config: dict[str, Any], generated: list[dict[str, Any]], use_ocr: bool = False) -> list[dict[str, Any]]:
    output_root = Path(config["artifacts_root"]) / "baseline_outputs"
    ocr = PaddleOCRAdapter(config) if use_ocr else None
    results: list[dict[str, Any]] = []
    for item in generated:
        degraded = read_gray(item["output_path"])
        clean = read_gray(item["source_path"])
        processed, preprocessing_record = preprocess(degraded, config)
        if clean.shape != processed.shape:
            clean = __import__("cv2").resize(clean, (processed.shape[1], processed.shape[0]))
        for baseline_id, method in BASELINES.items():
            started = time.perf_counter()
            output = method(processed)
            elapsed = time.perf_counter() - started
            target = output_root / item["source_id"] / f"{item['variant_id']}_{item['severity']}" / f"{baseline_id}.png"
            write_image(target, output)
            ocr_result = ocr.run(output) if ocr else None
            results.append(
                {
                    "source_id": item["source_id"],
                    "variant_id": item["variant_id"],
                    "severity": item["severity"],
                    "baseline": baseline_id,
                    "psnr": psnr(clean, output),
                    "ssim": ssim(clean, output),
                    "runtime_seconds": elapsed,
                    "ocr_status": ocr_result.status if ocr_result else "not_run",
                    "ocr_text": ocr_result.text if ocr_result else "",
                    "ocr_error": ocr_result.error if ocr_result else "",
                    "output_path": str(target.resolve()),
                    "preprocessing": json.dumps(preprocessing_record, sort_keys=True),
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "pilot_baseline_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)
    return results

