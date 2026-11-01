from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, write_json
from .environment import collect_environment
from .manifest import build_manifest, write_manifest
from .runner import run_baselines, run_pilot, verify_reproducibility
from .verification import leakage_report


def main() -> None:
    parser = argparse.ArgumentParser(description="DSDNet-Receipt research pipeline")
    parser.add_argument("command", choices=["readiness", "pilot", "baselines", "verify"])
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--ocr", action="store_true", help="Run PaddleOCR during baseline evaluation")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    artifacts = Path(config["artifacts_root"])
    manifest_path = artifacts / "readiness" / "dataset_manifest.csv"
    generation_path = artifacts / "pilot_degradation" / "generation_manifest.json"

    if args.command == "readiness":
        records = build_manifest(config)
        write_manifest(manifest_path, records)
        write_json(artifacts / "readiness" / "environment.json", collect_environment(config, config_path))
        write_json(artifacts / "readiness" / "leakage_report.json", leakage_report(manifest_path))
        print(f"Wrote {len(records)} image records to {manifest_path}")
        return

    if not manifest_path.exists():
        raise SystemExit("Run the readiness command first.")

    if args.command == "pilot":
        generated = run_pilot(config, manifest_path)
        report = verify_reproducibility(config, generated)
        write_json(artifacts / "pilot_degradation" / "reproducibility_report.json", report)
        print(f"Generated {len(generated)} pilot variants; reproducible={report['all_identical']}")
        return

    if args.command == "baselines":
        if not generation_path.exists():
            raise SystemExit("Run the pilot command first.")
        import json

        generated = json.loads(generation_path.read_text(encoding="utf-8"))
        results = run_baselines(config, generated, use_ocr=args.ocr)
        print(f"Produced {len(results)} baseline records")
        return

    report = leakage_report(manifest_path)
    write_json(artifacts / "readiness" / "leakage_report.json", report)
    print(report["status"])


if __name__ == "__main__":
    main()

