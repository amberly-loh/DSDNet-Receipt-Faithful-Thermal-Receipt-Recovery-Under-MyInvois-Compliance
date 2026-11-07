"""Compare text-reader speed/output across engine settings. Writes artifacts/profile_ocr.json."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "profile_ocr.json"
VARIANTS = [
    {"name": "current_1thread", "cpu_threads": 1, "enable_mkldnn": False, "use_gpu": False},
    {"name": "6threads", "cpu_threads": 6, "enable_mkldnn": False, "use_gpu": False},
    {"name": "6threads_mkldnn", "cpu_threads": 6, "enable_mkldnn": True, "use_gpu": False},
    {"name": "gpu", "cpu_threads": 6, "enable_mkldnn": False, "use_gpu": True},
]

def child(variant: dict, image_path: str) -> dict:
    os.environ["FLAGS_use_mkldnn"] = "1" if variant["enable_mkldnn"] else "0"
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(variant["cpu_threads"])
    import cv2, paddle
    from paddleocr import PaddleOCR
    info = {"name": variant["name"], "paddle": paddle.__version__, "cuda_build": bool(paddle.device.is_compiled_with_cuda())}
    if variant["use_gpu"] and not info["cuda_build"]:
        info["skipped"] = "paddle has no CUDA build"; return info
    t = time.perf_counter()
    engine = PaddleOCR(lang="en", ocr_version="PP-OCRv3", use_angle_cls=False, show_log=False, use_gpu=variant["use_gpu"],
                       enable_mkldnn=variant["enable_mkldnn"], cpu_threads=variant["cpu_threads"])
    info["load"] = round(time.perf_counter() - t, 2)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    engine.ocr(image, cls=False)  # warm-up
    times, texts = [], []
    for _ in range(2):
        t = time.perf_counter(); pred = engine.ocr(image, cls=False); times.append(round(time.perf_counter() - t, 2))
        texts = [item[1][0] for page in pred or [] for item in page or []]
    info.update(seconds_per_pass=times, lines=len(texts), text=texts)
    return info

def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        print("RESULT" + json.dumps(child(json.loads(sys.argv[2]), sys.argv[3]))); return
    runs = sorted((ROOT / "artifacts" / "user_runs").glob("USER-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    sample = next(str(p / "rectified_input.png") for p in runs if (p / "rectified_input.png").is_file() and (p / "rectified_input.png").stat().st_size > 200_000)
    report = {"sample": sample, "runs": []}
    for variant in VARIANTS:
        try:
            out = subprocess.run([sys.executable, __file__, "--child", json.dumps(variant), sample], capture_output=True, text=True, timeout=900)
            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
            report["runs"].append(json.loads(line[6:]) if line else {"name": variant["name"], "error": out.stderr[-1500:], "returncode": out.returncode})
        except Exception as error:
            report["runs"].append({"name": variant["name"], "error": repr(error)})
        OUT.write_text(json.dumps(report, indent=2))
    base = next((r.get("text") for r in report["runs"] if r.get("name") == "current_1thread"), None)
    for r in report["runs"]:
        if base is not None and r.get("text") is not None:
            r["same_text_as_current"] = r["text"] == base
            r["text_lines_differing"] = sum(1 for a, b in zip(r["text"], base) if a != b) + abs(len(r["text"]) - len(base))
        r.pop("text", None)
    report["done"] = True
    OUT.write_text(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
