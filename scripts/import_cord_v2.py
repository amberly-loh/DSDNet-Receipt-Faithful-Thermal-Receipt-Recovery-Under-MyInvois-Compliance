from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


DATASET_REVISION = "refs/convert/parquet"
FILES = [
    ("test", "0000", "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet", 234202795),
    ("train", "0000", "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", 490224630),
    ("train", "0001", "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0001.parquet", 441418432),
    ("train", "0002", "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0002.parquet", 443802181),
    ("train", "0003", "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/train/0003.parquet", 455555434),
    ("validation", "0000", "https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet", 242080800),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected_size: int) -> None:
    if target.exists() and target.stat().st_size == expected_size:
        return
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DSDNet-Receipt/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"Download size mismatch for {url}")
    partial.replace(target)


def extension_for(payload: bytes, declared_path: str | None) -> str:
    if declared_path:
        suffix = Path(declared_path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png"}:
            return ".jpg" if suffix == ".jpeg" else suffix
    if payload.startswith(b"\x89PNG"):
        return ".png"
    return ".jpg"


def import_file(parquet_path: Path, split: str, shard: str, root: Path, start: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    image_dir = root / "images" / split
    label_dir = root / "ground_truth" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    index = start
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=8, columns=["image", "ground_truth"]):
        for record in batch.to_pylist():
            image_record = record["image"]
            payload = image_record.get("bytes")
            if payload is None:
                raise RuntimeError("CORD parquet row does not contain embedded image bytes")
            suffix = extension_for(payload, image_record.get("path"))
            source_id = f"CORD-{split.upper()}-{index:04d}"
            image_path = image_dir / f"{source_id}{suffix}"
            label_path = label_dir / f"{source_id}.json"
            image_path.write_bytes(payload)
            parsed = json.loads(record["ground_truth"])
            label_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            rows.append(
                {
                    "source_id": source_id,
                    "dataset": "CORD-v2",
                    "official_split": split,
                    "source_group": source_id,
                    "image_path": str(image_path.relative_to(root)),
                    "annotation_path": str(label_path.relative_to(root)),
                    "width": width,
                    "height": height,
                    "sha256": sha256(image_path),
                    "license": "CC-BY-4.0",
                    "language_domain": "Indonesian receipts; Latin script",
                    "study_role": "external controlled development and benchmark corpus",
                    "formal_malaysian_evidence": "no",
                    "parquet_shard": shard,
                }
            )
            index += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the official CORD-v2 receipt dataset with provenance.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    temp = root / ".download"
    temp.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []
    counters = {"train": 0, "validation": 0, "test": 0}
    downloads: list[dict[str, str | int]] = []

    for split, shard, url, expected_size in FILES:
        parquet_path = temp / f"{split}-{shard}.parquet"
        print(f"Downloading {split}/{shard} ({expected_size / 1_000_000:.1f} MB)", flush=True)
        download(url, parquet_path, expected_size)
        parquet_hash = sha256(parquet_path)
        shard_rows = import_file(parquet_path, split, shard, root, counters[split])
        counters[split] += len(shard_rows)
        records.extend(shard_rows)
        downloads.append({"split": split, "shard": shard, "url": url, "bytes": expected_size, "sha256": parquet_hash, "rows": len(shard_rows)})
        parquet_path.unlink()

    expected = {"train": 800, "validation": 100, "test": 100}
    if counters != expected or len(records) != 1000:
        raise RuntimeError(f"Expected 1,000 rows with {expected}; imported {counters}")

    source_ids = [str(record["source_id"]) for record in records]
    random.Random(56).shuffle(source_ids)
    project_partition = {
        source_id: "train" if index < 700 else "validation" if index < 850 else "test"
        for index, source_id in enumerate(source_ids)
    }
    for record in records:
        record["project_partition"] = project_partition[str(record["source_id"])]

    manifest = root / "cord_v2_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    provenance = {
        "dataset": "CORD-v2",
        "publisher": "NAVER CLOVA",
        "official_repository": "https://github.com/clovaai/cord",
        "official_dataset": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
        "license": "CC-BY-4.0",
        "revision": DATASET_REVISION,
        "counts": counters,
        "project_partition_counts": {"train": 700, "validation": 150, "test": 150},
        "project_partition_seed": 56,
        "total": len(records),
        "role_boundary": "External controlled development data. It is not Malaysian natural-domain evidence and does not replace the self-collected robustness set. The publisher split is retained as provenance; the Chapter 3 project partition is 70:15:15.",
        "download_files": downloads,
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    try:
        temp.rmdir()
    except OSError:
        pass
    print(f"Imported {len(records)} independent receipt sources to {root}")


if __name__ == "__main__":
    main()
