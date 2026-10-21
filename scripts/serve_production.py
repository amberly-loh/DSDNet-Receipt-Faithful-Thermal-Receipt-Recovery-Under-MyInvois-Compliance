from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waitress import serve

from dsdnet_receipt.web_app import create_app


def main() -> None:
    host = os.getenv("RECEIPT_HOST", "127.0.0.1")
    port = int(os.getenv("RECEIPT_PORT", "5055"))
    threads = max(2, int(os.getenv("RECEIPT_THREADS", "4")))
    app = create_app(ROOT / "configs" / "default.json")
    serve(app, host=host, port=port, threads=threads, url_scheme="https" if os.getenv("RECEIPT_HTTPS") == "1" else "http")


if __name__ == "__main__":
    main()
