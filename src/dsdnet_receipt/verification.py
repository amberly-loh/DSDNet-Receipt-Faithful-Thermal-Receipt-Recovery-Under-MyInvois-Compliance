from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import write_json
from .manifest import read_manifest


def leakage_report(manifest_path: str | Path) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    partitions: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        partitions[record["source_id"]].add(record["assigned_partition"])
        hashes[record["sha256"]].add(record["assigned_partition"])
    group_violations = {key: sorted(value) for key, value in partitions.items() if len(value) > 1}
    exact_duplicate_violations = {key: sorted(value) for key, value in hashes.items() if len(value) > 1}
    counts: dict[str, int] = defaultdict(int)
    for source_id, values in partitions.items():
        counts[next(iter(values))] += 1
    return {
        "status": "PASS" if not group_violations and not exact_duplicate_violations else "FAIL",
        "independent_source_groups": len(partitions),
        "source_groups_by_partition": dict(sorted(counts.items())),
        "source_group_partition_violations": group_violations,
        "exact_duplicate_partition_violations": exact_duplicate_violations,
        "note": "Source grouping and consent remain provisional until manual confirmation.",
    }

