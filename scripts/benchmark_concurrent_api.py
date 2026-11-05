"""Concurrent local API regression benchmark with isolated synthetic data.

This is a development benchmark, not a production load test or SLA claim.
Each worker uses its own authenticated Flask test client against one temporary
SQLite database so concurrent request handling and export-free read paths are
exercised without retaining personal data.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.storage import utc_now  # noqa: E402
from dsdnet_receipt.web_app import create_app  # noqa: E402


OUT = ROOT / "artifacts" / "training_candidate_20261102" / "concurrent_api_performance.json"
RECORD_COUNT = 250
REQUESTS_PER_LEVEL = 64
CONCURRENCY_LEVELS = (1, 4, 8)
EMAIL = "benchmark@valid-domain.my"
PASSWORD = "synthetic-benchmark-password-2026"
_local = threading.local()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def authenticated_client(app):
    client = getattr(_local, "client", None)
    token = getattr(_local, "csrf", None)
    if client is None:
        client = app.test_client()
        token = client.get("/api/auth").get_json()["csrf_token"]
        response = client.post(
            "/api/login",
            json={"email": EMAIL, "password": PASSWORD},
            headers={"X-CSRF-Token": token},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Worker login failed with HTTP {response.status_code}")
        token = response.get_json()["csrf_token"]
        _local.client = client
        _local.csrf = token
    return client, token


def run_request(app, endpoint: str) -> tuple[float, int, int]:
    client, token = authenticated_client(app)
    started = time.perf_counter()
    if endpoint == "selection_summary_50_records":
        response = client.post(
            "/api/receipts/selection-summary",
            json={"ids": [f"CBENCH-{index:05d}" for index in range(50)]},
            headers={"X-CSRF-Token": token},
        )
    else:
        path = {"history_250_records": "/api/receipts", "dashboard_250_records": "/api/dashboard"}[endpoint]
        response = client.get(path)
    return time.perf_counter() - started, response.status_code, len(response.data)


def benchmark(app, endpoint: str, workers: int) -> dict:
    started = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_request, app, endpoint) for _ in range(REQUESTS_PER_LEVEL)]
        for future in as_completed(futures):
            rows.append(future.result())
    wall = time.perf_counter() - started
    latencies = [row[0] for row in rows]
    statuses = sorted({row[1] for row in rows})
    return {
        "workers": workers,
        "requests": len(rows),
        "statuses": statuses,
        "failures": sum(row[1] != 200 for row in rows),
        "wall_seconds": wall,
        "throughput_requests_per_second": len(rows) / wall,
        "response_bytes": {"minimum": min(row[2] for row in rows), "maximum": max(row[2] for row in rows)},
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95_nearest_rank": percentile(latencies, 0.95),
            "maximum": max(latencies),
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="receipt-concurrent-benchmark-") as directory:
        temp = Path(directory)
        database = temp / "benchmark.sqlite3"
        source = temp / "synthetic.png"
        source.write_bytes(b"synthetic benchmark placeholder")
        app = create_app(ROOT / "configs" / "default.json", testing=True, database_path=database)
        owner = app.test_client()
        token = owner.get("/api/auth").get_json()["csrf_token"]
        registration = owner.post(
            "/api/register",
            json={
                "name": "Concurrent Benchmark",
                "email": EMAIL,
                "password": PASSWORD,
                "password_confirmation": PASSWORD,
                "policy_accepted": True,
            },
            headers={"X-CSRF-Token": token},
        )
        if registration.status_code != 201:
            raise RuntimeError(f"Registration failed with HTTP {registration.status_code}")
        user = registration.get_json()["user"]
        store = app.extensions["receipt_store"]
        now = utc_now()
        for index in range(RECORD_COUNT):
            store.create_receipt(
                {
                    "public_id": f"CBENCH-{index:05d}", "user_id": user["id"],
                    "source_name": f"synthetic-{index:05d}.png", "created_at": now, "updated_at": now,
                    "status": "verified" if index % 3 else "needs_review",
                    "merchant": f"Synthetic Merchant {index % 20:02d}",
                    "transaction_date": f"2026-10-{(index % 28) + 1:02d}",
                    "receipt_number": f"R-{index:05d}", "total_amount": f"{(index % 500) + 0.25:.2f}",
                    "tags": "[]", "notes": "synthetic concurrent benchmark record",
                    "ocr_text": "synthetic benchmark OCR text", "audit_json": "{}",
                    "input_path": str(source), "restored_path": str(source), "runtime_seconds": 0.0,
                    "restoration_status": "ok", "support_status": "unavailable",
                }
            )

        results = {
            endpoint: [benchmark(app, endpoint, workers) for workers in CONCURRENCY_LEVELS]
            for endpoint in ("history_250_records", "dashboard_250_records", "selection_summary_50_records")
        }
        failures = sum(row["failures"] for rows in results.values() for row in rows)
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "concurrent local Flask test-client and SQLite regression benchmark with synthetic records",
            "environment": {"python": sys.version, "platform": platform.platform()},
            "settings": {
                "record_count": RECORD_COUNT,
                "requests_per_level": REQUESTS_PER_LEVEL,
                "concurrency_levels": list(CONCURRENCY_LEVELS),
                "selection_size": 50,
                "database": "temporary isolated SQLite database",
                "authentication": "one synthetic registered account; one client session per worker thread",
            },
            "results": results,
            "failures": failures,
            "provenance": {
                "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
                "application_sha256": sha256(ROOT / "src" / "dsdnet_receipt" / "web_app.py"),
                "configuration_sha256": sha256(ROOT / "configs" / "default.json"),
            },
            "limitations": [
                "This does not include network transport, a browser, OCR, restoration, or multiple hosts.",
                "Synthetic data and a test client do not establish production capacity or an SLA.",
                "Human usability and accessibility remain separate participant-based evaluations.",
            ],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
