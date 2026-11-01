from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .images import difference_hash, hamming_distance, image_quality, iter_images, read_gray, sha256_file


MANIFEST_COLUMNS = [
    "source_id",
    "file_name",
    "relative_path",
    "receipt_source",
    "source_type",
    "merchant_template_family",
    "language_script",
    "clean_reference_available",
    "available_fields",
    "high_risk_fields",
    "consent_usage_permission",
    "assigned_partition",
    "exclusion_status",
    "exclusion_reason",
    "sha256",
    "perceptual_hash",
    "width",
    "height",
    "quality_score",
    "annotation_status",
]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def partition_for(group_key: str, split: dict[str, float]) -> str:
    unit = int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:16], 16) / 2**64
    if unit < split["train"]:
        return "train"
    if unit < split["train"] + split["validation"]:
        return "validation"
    return "test"


def build_manifest(config: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["dataset_root"])
    paths = iter_images(root)
    records: list[dict[str, Any]] = []
    hashes: list[int] = []
    uf = UnionFind(len(paths))

    for path in paths:
        image = read_gray(path)
        quality = image_quality(image)
        p_hash = difference_hash(image)
        hashes.append(p_hash)
        records.append(
            {
                "file_name": path.name,
                "relative_path": str(path.relative_to(root)),
                "receipt_source": "self-collected local dataset",
                "source_type": config["manifest"]["default_source_type"],
                "merchant_template_family": "requires confirmation",
                "language_script": config["manifest"]["default_language_script"],
                "clean_reference_available": "requires confirmation",
                "available_fields": "requires annotation",
                "high_risk_fields": ";".join(config["manifest"]["high_risk_fields"]),
                "consent_usage_permission": config["manifest"]["default_consent_status"],
                "exclusion_status": "pending review",
                "exclusion_reason": "",
                "sha256": sha256_file(path),
                "perceptual_hash": f"{p_hash:016x}",
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "quality_score": round(quality["quality_score"], 6),
                "annotation_status": "not annotated",
            }
        )

    exact: dict[str, int] = {}
    for i, record in enumerate(records):
        prior = exact.get(record["sha256"])
        if prior is None:
            exact[record["sha256"]] = i
        else:
            uf.union(prior, i)

    threshold = int(config["manifest"]["near_duplicate_hamming"])
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming_distance(hashes[i], hashes[j]) <= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(records)):
        groups.setdefault(uf.find(index), []).append(index)

    for sequence, root_index in enumerate(sorted(groups), start=1):
        source_id = f"SRC-{sequence:04d}"
        members = groups[root_index]
        partition = partition_for(source_id, config["partition"])
        for index in members:
            records[index]["source_id"] = source_id
            records[index]["assigned_partition"] = partition

    return records


def write_manifest(path: str | Path, records: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def clean_candidates(records: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: float(row["quality_score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in ordered:
        if record["source_id"] in seen:
            continue
        selected.append(record)
        seen.add(record["source_id"])
        if len(selected) == count:
            break
    return selected

