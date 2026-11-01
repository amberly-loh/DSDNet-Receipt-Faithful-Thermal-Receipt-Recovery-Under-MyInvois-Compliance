from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config, write_json  # noqa: E402
from dsdnet_receipt.degradation import degrade, variant_seed  # noqa: E402
from dsdnet_receipt.images import read_gray, sha256_file, write_image  # noqa: E402

DATASET = REPO.parent / "FYP datasets" / "External" / "SROIE-Malaysia"

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the controlled three-severity Malaysian SROIE corpus.")
    parser.add_argument("--config", default=str(REPO / "configs" / "default.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    with (DATASET / "sroie_protocol_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        sources = list(csv.DictReader(stream))
    if args.limit is not None:
        sources = sources[: args.limit]
    output = Path(config["artifacts_root"]) / "controlled_degradation_sroie"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation_manifest.jsonl"
    existing: set[tuple[str, str]] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            existing.add((record["source_id"], record["severity"]))
    maximum = int(config["preprocessing"]["max_long_edge"])
    generated = 0
    with manifest_path.open("a", encoding="utf-8") as log:
        for position, source in enumerate(sources, start=1):
            source_id = source["source_id"]
            clean = read_gray(DATASET / source["relative_path"])
            scale = min(1.0, maximum / max(clean.shape))
            if scale < 1.0:
                clean = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            source_dir = output / source["project_partition"] / source_id
            clean_path = source_dir / "clean_reference.png"
            if not clean_path.exists():
                write_image(clean_path, clean)
            for number, severity in enumerate(config["degradation"]["severities"], start=1):
                if (source_id, severity) in existing:
                    continue
                variant_id = f"V{number:02d}"
                seed = variant_seed(config["base_seed"], source_id, variant_id, severity)
                degraded, masks, parameters = degrade(clean, severity, config, seed)
                variant_dir = source_dir / f"{variant_id}_{severity}"
                degraded_path = variant_dir / "degraded.png"
                write_image(degraded_path, degraded)
                for name, mask in masks.items():
                    write_image(variant_dir / f"{name}.png", mask.astype(np.uint8) * 255)
                write_json(variant_dir / "parameters.json", parameters)
                record = {
                    "source_id": source_id,
                    "source_group": source["source_id"],
                    "project_partition": source["project_partition"],
                    "variant_id": variant_id,
                    "severity": severity,
                    "seed": seed,
                    "clean_reference": str(clean_path.resolve()),
                    "degraded_path": str(degraded_path.resolve()),
                    "degraded_sha256": sha256_file(degraded_path),
                    "support_target": str((variant_dir / "support_target.png").resolve()),
                    "parameters": parameters,
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                generated += 1
            if position % 25 == 0:
                print(f"Processed {position}/{len(sources)} sources; added {generated} variants", flush=True)
    summary = {
        "sources_requested": len(sources),
        "variants_expected": len(sources) * len(config["degradation"]["severities"]),
        "variants_added_this_run": generated,
        "partition_rule": "Chapter 3 deterministic 70:15:15 source partition; variants inherit source partition",
        "formal_status": "controlled Malaysian corpus generated; formal use still requires the exact Chapter 4 OCR environment and a trained checkpoint",
    }
    write_json(output / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
