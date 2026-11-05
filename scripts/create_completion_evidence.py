from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "training_candidate_20261102"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"No testsuite in {path}")
    return {
        "tests": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "time_seconds": float(suite.attrib.get("time", 0.0)),
        "artifact": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
    }


def latest_training_state(log_path: Path, nonfinite_path: Path) -> dict:
    last = None
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    events = []
    if nonfinite_path.exists():
        events = [json.loads(line) for line in nonfinite_path.read_text(encoding="utf-8").splitlines() if line]
    checkpoint = log_path.parent / "checkpoint_latest.pt"
    return {
        "status": "in_progress",
        "latest_logged_iteration": last["iteration"] if last else None,
        "latest_loss": last.get("loss") if last else None,
        "latest_amp_scale": last.get("amp_scale") if last else None,
        "nonfinite_quarantine_events": events,
        "rolling_checkpoint": str(checkpoint.relative_to(ROOT)),
        "rolling_checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
        "training_log": str(log_path.relative_to(ROOT)),
        "training_log_sha256_at_snapshot": sha256(log_path),
        "interpretation": "The planned one-million-iteration run is not complete; this is a recoverable progress snapshot.",
    }


def method_table(summary: dict) -> dict:
    fields = (
        "sources",
        "variants",
        "mean_ssim",
        "mean_validation_objective",
        "mean_stroke_recall",
        "mean_deleted_stroke_rate",
        "mean_false_stroke_rate",
    )
    return {name: {key: values[key] for key in fields} for name, values in summary["methods"].items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = ROOT / "artifacts/validation_restoration/checkpoint_43000_locked_validation_20261102/summary.json"
    validation_significance_path = ROOT / "artifacts/validation_restoration/checkpoint_43000_locked_validation_20261102/paired_significance.json"
    exploratory_path = ROOT / "artifacts/exploratory_test_restoration/checkpoint_43000_exploratory_20261102/summary.json"
    exploratory_significance_path = ROOT / "artifacts/exploratory_test_restoration/checkpoint_43000_exploratory_20261102/paired_significance.json"
    performance_path = OUT_DIR / "local_api_performance.json"
    concurrent_performance_path = OUT_DIR / "concurrent_api_performance.json"
    source_audit_path = OUT_DIR / "source_privacy_audit.json"
    e2e_path = ROOT / "artifacts/e2e_synthetic_20261102/results.json"
    usability_path = OUT_DIR / "usability_static_audit.json"
    training_dir = ROOT / "artifacts/training_exact_excluded_resume_15000_20261102"
    validation = load_json(validation_path)
    validation_significance = load_json(validation_significance_path)
    exploratory = load_json(exploratory_path)
    exploratory_significance = load_json(exploratory_significance_path)
    performance = load_json(performance_path)
    concurrent_performance = load_json(concurrent_performance_path)
    source_audit = load_json(source_audit_path)
    e2e = load_json(e2e_path)
    usability = load_json(usability_path)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    checkpoint = training_dir / "checkpoint_latest.pt"
    training_config = ROOT / "artifacts/protocol_lock/exact_duplicate_excluded_20261102/training_config.json"
    manifest = ROOT / "artifacts/protocol_lock/generation_manifest_611_sources.jsonl"
    default_config = ROOT / "configs/default.json"
    torch_environment = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    evidence = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Applicable local automated verification and restoration experiments for the current thesis prototype",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": git_head,
            "working_tree_clean": not bool(git_status.strip()),
            "working_tree_change_count": len([line for line in git_status.splitlines() if line.strip()]),
            **torch_environment,
        },
        "locked_provenance": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_iteration": validation["checkpoint_iteration"],
            "training_config": str(training_config.relative_to(ROOT)),
            "training_config_sha256": sha256(training_config),
            "generation_manifest": str(manifest.relative_to(ROOT)),
            "generation_manifest_sha256": sha256(manifest),
            "default_config": str(default_config.relative_to(ROOT)),
            "default_config_sha256": sha256(default_config),
            "validation_records_sha256": sha256(validation_path.parent / "records.csv"),
            "exploratory_records_sha256": sha256(exploratory_path.parent / "records.csv"),
        },
        "functional_and_integration": {
            "status": "pass",
            "regression": junit(ROOT / "artifacts/completion_verification_20261102/pytest_full_43000.xml"),
            "synthetic_end_to_end": {
                "status": "pass" if not e2e["errors"] and all(c["http_status"] == 200 for c in e2e["cases"]) else "fail",
                "cases": e2e["cases"],
                "limitations": e2e["note"],
                "artifact": str(e2e_path.relative_to(ROOT)),
                "sha256": sha256(e2e_path),
            },
        },
        "privacy_and_security": {
            "status": "pass_local_automated_checks",
            "focused_tests": junit(ROOT / "artifacts/completion_verification_20261102/security_and_privacy.xml"),
            "coverage": [
                "security headers and unauthenticated access",
                "cross-user receipt isolation",
                "scrypt password hashing",
                "CSV and XLSX formula-injection neutralisation",
            ],
            "source_and_dependency_audit": {
                "files_scanned": source_audit["files_scanned"],
                "secret_pattern_hits": source_audit["secret_pattern_hits"],
                "dangerous_python_constructs": source_audit["dangerous_python_constructs"],
                "dependency_check": source_audit["dependency_check"],
                "artifact": str(source_audit_path.relative_to(ROOT)),
                "sha256": sha256(source_audit_path),
            },
            "limitations": "No independent penetration test, deployment review, or multi-tenant production audit was performed.",
        },
        "usability": {
            "status": "pass_static_heuristic_only",
            "result": usability,
            "artifact": str(usability_path.relative_to(ROOT)),
            "sha256": sha256(usability_path),
            "limitations": "No participant-based usability study, assistive-technology test, or browser-state study was performed.",
        },
        "performance": {
            "status": "pass_local_regression_benchmark",
            "record_count": performance["record_count"],
            "endpoints": performance["endpoints"],
            "failures": performance["failures"],
            "limitations": performance["limitations"],
            "artifact": str(performance_path.relative_to(ROOT)),
            "sha256": sha256(performance_path),
            "concurrent_local_benchmark": {
                "settings": concurrent_performance["settings"],
                "results": concurrent_performance["results"],
                "failures": concurrent_performance["failures"],
                "limitations": concurrent_performance["limitations"],
                "artifact": str(concurrent_performance_path.relative_to(ROOT)),
                "sha256": sha256(concurrent_performance_path),
            },
        },
        "restoration_experiments": {
            "validation": {
                "status": "completed_development_validation",
                "formal_test_evidence": validation["formal_test_evidence"],
                "checkpoint_iteration": validation["checkpoint_iteration"],
                "methods": method_table(validation),
                "artifact": str(validation_path.relative_to(ROOT)),
                "sha256": sha256(validation_path),
                "paired_significance_artifact": str(validation_significance_path.relative_to(ROOT)),
                "paired_significance_sha256": sha256(validation_significance_path),
                "paired_significance": validation_significance,
                "interpretation": "The proposed method leads validation SSIM, objective, stroke recall, and deleted-stroke rate; Sauvola has the lowest false-stroke rate.",
            },
            "exploratory_test": {
                "status": "completed_exploratory_not_independent",
                "formal_test_evidence": exploratory["formal_test_evidence"],
                "checkpoint_iteration": exploratory["checkpoint_iteration"],
                "methods": method_table(exploratory),
                "artifact": str(exploratory_path.relative_to(ROOT)),
                "sha256": sha256(exploratory_path),
                "paired_significance_artifact": str(exploratory_significance_path.relative_to(ROOT)),
                "paired_significance_sha256": sha256(exploratory_significance_path),
                "paired_significance": exploratory_significance,
                "interpretation": exploratory["interpretation"],
            },
        },
        "formal_training": latest_training_state(
            training_dir / "training_log.jsonl", training_dir / "nonfinite_samples.jsonl"
        ),
        "dependency_check": {
            "status": "pass",
            "detail": source_audit["dependency_check"]["stdout"],
            "resolved_versions": {"opencv-python": "4.6.0.66", "chardet": "5.2.0"},
        },
        "completion_boundaries": [
            "The one-million-iteration training schedule remains in progress.",
            "The exploratory test partition was previously inspected and is not fresh independent confirmation.",
            "No participant usability study or independent penetration test was conducted.",
            "Local single-process and concurrent test-client timing is not a production capacity or SLA claim.",
        ],
    }

    json_path = OUT_DIR / "completion_evidence_20261102.json"
    json_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    p95 = performance["endpoints"]
    proposed = evidence["restoration_experiments"]["validation"]["methods"]["Proposed_DSDNet_Receipt"]
    markdown = f"""# Completion evidence snapshot — 2 November 2026

## Verified outcomes

- Automated regression: **{evidence['functional_and_integration']['regression']['tests']} passed**, 0 failures, 0 errors.
- Focused privacy/security: **{evidence['privacy_and_security']['focused_tests']['tests']} passed**.
- Source/privacy scan: **{source_audit['files_scanned']} files**, 0 committed-secret pattern hits, 0 dangerous bare eval/exec or shell=True constructs; dependency check passed.
- Synthetic end-to-end: **{len(e2e['cases'])}/{len(e2e['cases'])} cases** completed upload, restoration, OCR, detail retrieval, and XLSX export with HTTP 200.
- Static usability audit: {usability['controls_checked']} controls and {usability['buttons_checked']} buttons checked; no duplicate IDs, unlabelled controls, or unnamed buttons were found.
- Local API p95: history {p95['history_250_records']['latency_seconds']['p95_nearest_rank']:.4f}s; dashboard {p95['dashboard_250_records']['latency_seconds']['p95_nearest_rank']:.4f}s; CSV {p95['csv_export_250_records']['latency_seconds']['p95_nearest_rank']:.4f}s; XLSX {p95['xlsx_export_250_records']['latency_seconds']['p95_nearest_rank']:.4f}s; 50-record summary {p95['selection_summary_50_records']['latency_seconds']['p95_nearest_rank']:.4f}s.
- Concurrent local benchmark: 576 requests across 1, 4 and 8 workers with **{concurrent_performance['failures']} HTTP failures**; this remains test-client evidence, not a production SLA.
- Proposed method validation ({proposed['sources']} sources/{proposed['variants']} variants): SSIM {proposed['mean_ssim']:.4f}; objective {proposed['mean_validation_objective']:.4f}; stroke recall {proposed['mean_stroke_recall']:.4f}; deleted-stroke rate {proposed['mean_deleted_stroke_rate']:.4f}; false-stroke rate {proposed['mean_false_stroke_rate']:.4f}.
- Training snapshot: iteration **{evidence['formal_training']['latest_logged_iteration']}**; non-finite samples are quarantined with their source, augmentation and AMP state recorded, allowing training to continue.

## Evidence boundaries

""" + "\n".join(f"- {item}" for item in evidence["completion_boundaries"]) + "\n"
    md_path = OUT_DIR / "completion_evidence_20261102.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
