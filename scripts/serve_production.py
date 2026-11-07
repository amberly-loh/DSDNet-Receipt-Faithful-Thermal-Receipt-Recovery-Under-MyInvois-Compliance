from __future__ import annotations

import os
import sys
import faulthandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

faulthandler.enable()
# Configure native runtimes before importing OCR/model libraries.
# The text reader runs ~6x faster with MKL-DNN and a few CPU threads
# (measured: 12.8 s -> 2.0 s per pass, identical text). Set RECEIPT_FAST_OCR=0
# to go back to the single-thread setting.
_fast = os.getenv("RECEIPT_FAST_OCR", "1") != "0"
_threads = str(max(1, min(6, (os.cpu_count() or 2) // 2))) if _fast else "1"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, _threads)
os.environ.setdefault("FLAGS_use_mkldnn", "1" if _fast else "0")

import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

from waitress import serve

from dsdnet_receipt.web_app import create_app


def main() -> None:
    # Phone scanning: also listen on this computer's Wi-Fi address so a paired
    # phone can send photos. Everything except the phone page and paired
    # uploads still needs this computer itself or a signed-in account.
    # Set RECEIPT_PHONE=0 to listen on this computer only.
    os.environ.setdefault("RECEIPT_PHONE", "1")
    host = os.getenv("RECEIPT_LISTEN") or ("0.0.0.0" if os.environ["RECEIPT_PHONE"] == "1" else os.getenv("RECEIPT_HOST", "127.0.0.1"))
    port = int(os.getenv("RECEIPT_PORT", "5055"))
    threads = max(2, int(os.getenv("RECEIPT_THREADS", "4")))
    app = create_app(ROOT / "configs" / "default.json")
    serve(app, host=host, port=port, threads=threads, url_scheme="https" if os.getenv("RECEIPT_HTTPS") == "1" else "http")


if __name__ == "__main__":
    main()
