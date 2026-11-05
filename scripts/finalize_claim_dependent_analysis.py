from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from run_claim_dependent_analysis import (  # noqa: E402
    KERNEL_FACTORS,
    STAGES,
    error_analysis,
    paired_tests,
    save_plots,
    severity_analysis,
    summary_by,
    write_csv,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a completed claim-dependent inference run")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "artifacts" / "protocol_lock" / "generation_manifest_611_sources.jsonl")
    parser.add_argument("--validation-records", type=Path, default=REPO / "artifacts" / "validation_restoration" / "checkpoint_43000_locked_validation_20261102" / "records.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    manifest = args.manifest.resolve()
    validation_records = args.validation_records.resolve()
    training_path = REPO / "configs" / "dsdnet_receipt_train.json"
    default_path = REPO / "configs" / "default.json"
    stage_records = output / "stage_ablation_records.csv"
    kernel_records = output / "kernel_sensitivity_records.csv"
    if not stage_records.is_file() or not kernel_records.is_file():
        raise SystemExit("Completed stage and kernel records are required")

    stage_frame = pd.read_csv(stage_records)
    kernel_frame = pd.read_csv(kernel_records)
    stage_summary_rows = summary_by(stage_frame, "stage")
    kernel_summary_rows = summary_by(kernel_frame, "kernel_factor")
    write_csv(output / "table_stage_ablation.csv", stage_summary_rows)
    write_csv(output / "table_kernel_sensitivity.csv", kernel_summary_rows)
    stage_tests = paired_tests(stage_frame, "stage", 8, [1, 2, 4, 6], "stage_depth")
    kernel_tests = paired_tests(kernel_frame, "kernel_factor", 1.0, [0.5, 0.75, 1.25, 1.5], "kernel_factor")
    write_csv(output / "stage_ablation_paired_tests.csv", stage_tests)
    write_csv(output / "kernel_sensitivity_paired_tests.csv", kernel_tests)

    severity_summary, severity_tests, severity_summary_rows = severity_analysis(validation_records)
    severity_summary.to_csv(output / "table_severity_robustness.csv", index=False)
    write_csv(output / "severity_statistical_tests.csv", severity_tests)
    error_frame, error_report = error_analysis(validation_records)
    error_frame.to_csv(output / "source_error_analysis.csv", index=False)
    (output / "error_failure_report.json").write_text(json.dumps(error_report, indent=2), encoding="utf-8")

    stage_summary = pd.DataFrame(stage_summary_rows).sort_values("stage")
    kernel_summary = pd.DataFrame(kernel_summary_rows).sort_values("kernel_factor")
    save_plots(output, stage_summary, kernel_summary, severity_summary, error_frame)

    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        git_head = "unavailable"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    source_ids = set(stage_frame.source_id.astype(str))
    summary = {
        "analysis_scope": "claim-dependent development validation; no fresh independent test",
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(checkpoint_payload.get("iteration", -1)),
        "validation_sources": len(source_ids),
        "validation_variants": int(stage_frame[["source_id", "variant_id"]].drop_duplicates().shape[0]),
        "crop_size": 128,
        "stage_depths": list(STAGES),
        "kernel_factors": list(KERNEL_FACTORS),
        "stage_ablation": stage_summary_rows,
        "kernel_sensitivity": kernel_summary_rows,
        "stage_tests": stage_tests,
        "kernel_tests": kernel_tests,
        "severity_robustness": severity_summary_rows,
        "severity_tests": severity_tests,
        "error_failure_analysis": error_report,
        "interpretation_boundaries": [
            "Intermediate-stage evaluation isolates inference depth but does not retrain a smaller architecture.",
            "Kernel sensitivity perturbs the supplied blur estimate and does not model every natural degradation.",
            "Synthetic severity results are development validation, not an independent natural-receipt test.",
            "Negative and mixed findings are retained, including false-stroke trade-offs and source-level failures.",
        ],
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "git_head": git_head,
            "checkpoint_sha256": sha256(checkpoint),
            "manifest_sha256": sha256(manifest),
            "validation_records_sha256": sha256(validation_records),
            "stage_records_sha256": sha256(stage_records),
            "kernel_records_sha256": sha256(kernel_records),
            "training_config_sha256": sha256(training_path),
            "runtime_config_sha256": sha256(default_path),
            "inference_script_sha256": sha256(REPO / "scripts" / "run_claim_dependent_analysis.py"),
            "finalization_script_sha256": sha256(Path(__file__).resolve()),
        },
    }
    (output / "claim_dependent_analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "sources": len(source_ids), "stage_rows": len(stage_frame), "kernel_rows": len(kernel_frame)}, indent=2))


if __name__ == "__main__":
    main()
