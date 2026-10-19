from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsdnet_receipt.config import load_config


def sha256(path: Path) -> str | None:
    if not path.is_file(): return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    config, config_path = load_config(ROOT / "configs" / "default.json")
    artifacts = Path(config["artifacts_root"])
    required = {
        "dataset_manifest": artifacts / "protocol_lock" / "generation_manifest_611_sources.jsonl",
        "dependency_record": ROOT / "CHAPTER4_ENVIRONMENT.md",
        "training_provenance": artifacts / "training_development" / "training_summary.json",
        "algorithm_tests": artifacts / "formal_run" / "algorithm1_tests.json",
        "protocol_manifest": artifacts / "formal_run" / "chapter3_4_protocol_manifest.json",
        "support_calibration": artifacts / "formal_run" / "support_calibration.json",
        "decision_thresholds": artifacts / "formal_run" / "decision_thresholds.json",
        "model_checkpoint": Path(config["dsdnet"]["checkpoint"]),
    }
    checks = []
    for name, path in required.items():
        checks.append({"name": name, "path": str(path), "status": "PASS" if path.is_file() else "FAIL", "sha256": sha256(path)})
    exact = {
        "dsdnet.stages": (config["dsdnet"].get("stages"), 8),
        "dsdnet.maxout_branches": (config["dsdnet"].get("maxout_branches"), 4),
        "ocr.required_version": (config["ocr"].get("required_version"), "2.7.0.3"),
        "ocr.ocr_version": (config["ocr"].get("ocr_version"), "PP-OCRv3"),
    }
    for name, (actual, expected) in exact.items():
        checks.append({"name": name, "actual": actual, "expected": expected, "status": "PASS" if actual == expected else "FAIL"})
    status = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {"status": status, "formal_evidence_allowed": status == "PASS", "configuration": str(config_path), "checks": checks,
              "reasons": [check["name"] for check in checks if check["status"] == "FAIL"]}
    output = artifacts / "formal_run" / "conformance_report.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
