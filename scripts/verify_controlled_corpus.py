from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO.parent / "FYP datasets" / "External" / "CORD-v2"
CORPUS = REPO / "artifacts" / "controlled_degradation"

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main() -> None:
    with (DATASET / "cord_v2_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        sources = {row["source_id"]: row for row in csv.DictReader(stream)}
    records = [json.loads(line) for line in (CORPUS / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    partitions: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        key = (record["source_id"], record["severity"])
        if key in seen:
            errors.append(f"duplicate variant: {key}")
        seen.add(key)
        source = sources.get(record["source_id"])
        if source is None:
            errors.append(f"unknown source: {record['source_id']}")
            continue
        if record["project_partition"] != source["project_partition"]:
            errors.append(f"partition mismatch: {record['source_id']}")
        partitions[record["source_id"]].add(record["project_partition"])
        degraded = Path(record["degraded_path"])
        support = Path(record["support_target"])
        if not degraded.is_file() or not support.is_file():
            errors.append(f"missing controlled output: {key}")
        elif digest(degraded) != record["degraded_sha256"]:
            errors.append(f"hash mismatch: {key}")
    leakage = [source_id for source_id, values in partitions.items() if len(values) != 1]
    severity_counts = Counter(record["severity"] for record in records)
    partition_variant_counts = Counter(record["project_partition"] for record in records)
    expected_severities = {"light": 1000, "moderate": 1000, "severe": 1000}
    expected_partitions = {"train": 2100, "validation": 450, "test": 450}
    if dict(severity_counts) != expected_severities:
        errors.append(f"severity counts: {dict(severity_counts)}")
    if dict(partition_variant_counts) != expected_partitions:
        errors.append(f"partition variant counts: {dict(partition_variant_counts)}")
    if leakage:
        errors.append(f"source partition leakage: {leakage[:20]}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "independent_sources": len(partitions),
        "variants": len(records),
        "severity_counts": dict(severity_counts),
        "partition_variant_counts": dict(partition_variant_counts),
        "hashes_checked": len(records),
        "source_partition_leakage": leakage,
        "errors": errors,
    }
    target = REPO / "artifacts" / "readiness" / "controlled_corpus_verification.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
