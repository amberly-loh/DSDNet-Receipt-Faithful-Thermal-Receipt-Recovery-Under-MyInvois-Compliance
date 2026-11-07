"""Time each processing stage on one stored receipt, at 1 and N CPU threads.
Writes artifacts/profile_speed.json. Run with scripts/profile_speed.cmd."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "profile_speed.json"


def child(threads: int, image_path: str) -> dict:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["FLAGS_use_mkldnn"] = "1" if threads > 1 else "0"
    sys.path.insert(0, str(ROOT / "src"))
    import cv2
    cv2.setNumThreads(threads); cv2.ocl.setUseOpenCL(False)
    try:
        import torch; torch.set_num_threads(threads)
    except Exception:
        pass
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.preprocessing import preprocess
    from dsdnet_receipt.restoration import (RestorationEngine, assess_receipt_degradation, make_faithful_content_view,
                                            make_readability_view, region_adaptive_source_fusion, select_faithful_recovery)
    from dsdnet_receipt.ocr import PaddleOCRAdapter
    config, _ = load_config(ROOT / "configs" / "default.json")
    config["ocr"]["enable_mkldnn"] = threads > 1; config["ocr"]["cpu_threads"] = threads
    t = {}
    def mark(name, start): t[name] = round(time.perf_counter() - start, 2)
    s = time.perf_counter(); engine = RestorationEngine(config); ocr = PaddleOCRAdapter(config); mark("load_engines", s)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    total = time.perf_counter()
    s = time.perf_counter(); reg, _ = preprocess(image, config, preserve_color=True); mark("preprocess", s)
    g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
    s = time.perf_counter(); res = engine.run(g); mark("dsdnet_restoration", s)
    s = time.perf_counter(); prof = assess_receipt_degradation(g); fused, _ = region_adaptive_source_fusion(g, res.image)
    sel = select_faithful_recovery(g, fused).image; read = make_readability_view(g, sel); mark("fusion_and_views", s)
    s = time.perf_counter(); r1 = ocr.run(g); mark("ocr_view_1_original", s)
    s = time.perf_counter(); r2 = ocr.run(sel); mark("ocr_view_2_recovered", s)
    s = time.perf_counter(); r3 = ocr.run(read); mark("ocr_view_3_readability_only_if_faded", s)
    lines = r1.lines + r2.lines
    mask = config.get("artifact_mask", {}); rec = {}
    s = time.perf_counter()
    make_faithful_content_view(g, sel, lines, mask.get("checkpoint") if mask.get("enabled") else None, float(mask.get("threshold", .95)),
                               prof, source_color=reg if reg.ndim == 3 else None, native_refinement=config.get("native_refinement"), refinement_audit=rec)
    mark("faithful_view_incl_finishing", s)
    t["total_processing"] = round(time.perf_counter() - total, 2)
    t["image_shape"] = list(g.shape); t["flags"] = prof.get("flags"); t["threads"] = threads
    t["cuda"] = bool(getattr(engine, "_device", None) and str(engine._device) == "cuda")
    return t


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        print("RESULT" + json.dumps(child(int(sys.argv[2]), sys.argv[3]))); return
    runs = sorted((ROOT / "artifacts" / "user_runs").glob("USER-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    sample = next(str(p / "input.png") for p in runs if (p / "input.png").is_file() and (p / "input.png").stat().st_size > 500_000)
    cpus = os.cpu_count() or 1
    report = {"cpu_count": cpus, "sample": sample, "runs": []}
    for threads in (max(2, min(6, cpus // 2)),):
        try:
            out = subprocess.run([sys.executable, __file__, "--child", str(threads), sample], capture_output=True, text=True, timeout=1200)
            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
            report["runs"].append(json.loads(line[6:]) if line else {"threads": threads, "error": out.stderr[-2000:], "returncode": out.returncode})
        except Exception as error:
            report["runs"].append({"threads": threads, "error": repr(error)})
        OUT.write_text(json.dumps(report, indent=2))
    report["done"] = True
    OUT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
