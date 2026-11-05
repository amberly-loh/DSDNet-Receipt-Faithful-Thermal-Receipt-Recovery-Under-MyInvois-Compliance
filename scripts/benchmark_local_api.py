"""Reproducible local API/export performance benchmark using synthetic records.

This measures Flask test-client and SQLite behaviour only.  It deliberately
does not claim network, multi-host, OCR, restoration, or production capacity.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.storage import utc_now  # noqa: E402
from dsdnet_receipt.web_app import create_app  # noqa: E402


RECORD_COUNT = 250
REQUEST_REPEATS = 30
OUT = ROOT / "artifacts" / "training_candidate_20261102" / "local_api_performance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timed_requests(client, method: str, path: str, repeats: int, **kwargs):
    durations = []
    statuses = []
    sizes = []
    for _ in range(repeats):
        started = time.perf_counter()
        response = client.open(path, method=method, **kwargs)
        durations.append(time.perf_counter() - started)
        statuses.append(response.status_code)
        sizes.append(len(response.data))
    ordered = sorted(durations)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return {
        "requests": repeats,
        "statuses": sorted(set(statuses)),
        "response_bytes_min": min(sizes),
        "response_bytes_max": max(sizes),
        "latency_seconds": {
            "mean": statistics.fmean(durations),
            "median": statistics.median(durations),
            "p95_nearest_rank": ordered[p95_index],
            "minimum": min(durations),
            "maximum": max(durations),
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="receipt-api-benchmark-") as directory:
        temp = Path(directory)
        database = temp / "benchmark.sqlite3"
        source = temp / "synthetic.png"
        source.write_bytes(b"synthetic benchmark placeholder")
        app = create_app(ROOT / "configs" / "default.json", testing=True, database_path=database)
        client = app.test_client()
        token = client.get("/api/auth").get_json()["csrf_token"]
        session = client.post("/api/guest", headers={"X-CSRF-Token": token})
        if session.status_code != 201:
            raise RuntimeError(f"Guest session failed with HTTP {session.status_code}")
        user = session.get_json()["user"]
        token = session.get_json()["csrf_token"]
        store = app.extensions["receipt_store"]
        now = utc_now()
        for index in range(RECORD_COUNT):
            store.create_receipt(
                {
                    "public_id": f"BENCH-{index:05d}",
                    "user_id": user["id"],
                    "source_name": f"synthetic-{index:05d}.png",
                    "created_at": now,
                    "updated_at": now,
                    "status": "verified" if index % 3 else "needs_review",
                    "merchant": f"Synthetic Merchant {index % 20:02d}",
                    "transaction_date": f"2026-10-{(index % 28) + 1:02d}",
                    "receipt_number": f"R-{index:05d}",
                    "total_amount": f"{(index % 500) + 0.25:.2f}",
                    "tags": "[]",
                    "notes": "synthetic local performance record",
                    "ocr_text": "synthetic benchmark OCR text",
                    "audit_json": "{}",
                    "input_path": str(source),
                    "restored_path": str(source),
                    "runtime_seconds": 0.0,
                    "restoration_status": "ok",
                    "support_status": "unavailable",
                }
            )

        endpoints = {
            "history_250_records": timed_requests(client, "GET", "/api/receipts", REQUEST_REPEATS),
            "dashboard_250_records": timed_requests(client, "GET", "/api/dashboard", REQUEST_REPEATS),
            "csv_export_250_records": timed_requests(client, "GET", "/api/export.csv", 10),
            "xlsx_export_250_records": timed_requests(client, "GET", "/api/export.xlsx", 10),
            "selection_summary_50_records": timed_requests(
                client,
                "POST",
                "/api/receipts/selection-summary",
                REQUEST_REPEATS,
                json={"ids": [f"BENCH-{index:05d}" for index in range(50)]},
                headers={"X-CSRF-Token": token},
            ),
        }
        failures = {
            name: result["statuses"]
            for name, result in endpoints.items()
            if result["statuses"] != [200]
        }
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "isolated local Flask test-client and SQLite benchmark with synthetic records",
            "record_count": RECORD_COUNT,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
            "settings": {
                "request_repeats": REQUEST_REPEATS,
                "csv_xlsx_repeats": 10,
                "selection_size": 50,
                "database": "temporary isolated SQLite database",
                "client": "single-process Flask test client",
            },
            "endpoints": endpoints,
            "failures": failures,
            "provenance": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": sha256(Path(__file__).resolve()),
                "application_sha256": sha256(ROOT / "src" / "dsdnet_receipt" / "web_app.py"),
                "configuration_sha256": sha256(ROOT / "configs" / "default.json"),
            },
            "limitations": [
                "No network transport, browser rendering, OCR, restoration, or concurrent users are measured.",
                "Synthetic records do not represent a production data distribution.",
                "Results establish local regression performance only and are not a capacity or SLA claim.",
            ],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
