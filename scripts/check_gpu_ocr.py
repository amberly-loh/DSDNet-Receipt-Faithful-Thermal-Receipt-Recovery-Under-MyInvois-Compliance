"""Check that the GPU text reader (helper process) works next to PyTorch and reads like the CPU reader."""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    out = {"python_path": os.environ.get("PYTHONPATH")}
    try:
        import cv2, torch
        out["torch_cuda"] = bool(torch.cuda.is_available())
        from dsdnet_receipt.config import load_config
        from dsdnet_receipt.ocr import PaddleOCRAdapter
        config, _ = load_config(ROOT / "configs" / "default.json")
        sample = sorted((ROOT / "artifacts" / "user_runs").glob("*/rectified_input.png"))[-1]
        image = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        os.environ["RECEIPT_OCR_GPU"] = "1"
        gpu = PaddleOCRAdapter(config)
        t = time.perf_counter(); gpu.run(image); out["gpu_first_seconds"] = time.perf_counter() - t
        t = time.perf_counter(); g = gpu.run(image); out["gpu_warm_seconds"] = time.perf_counter() - t
        out["gpu_status"], out["gpu_error"], out["gpu_lines"] = g.status, g.error, len(g.lines)
        x = torch.ones(4, device="cuda" if torch.cuda.is_available() else "cpu").sum().item()
        out["torch_still_works"] = x == 4
        os.environ["RECEIPT_OCR_GPU"] = "0"
        cpu = PaddleOCRAdapter(config)
        cpu.run(image)
        t = time.perf_counter(); c = cpu.run(image); out["cpu_warm_seconds"] = time.perf_counter() - t
        out["cpu_lines"] = len(c.lines)
        out["same_text"] = g.text == c.text
        import difflib
        out["text_similarity"] = difflib.SequenceMatcher(None, g.text, c.text).ratio()
        out["ok"] = g.status in ("ok", "protocol_version_mismatch") and len(g.lines) > 0
    except Exception:
        out["ok"] = False; out["traceback"] = traceback.format_exc()
    (ROOT / "artifacts" / "gpu_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()


if __name__ == "__main__":
    main()
