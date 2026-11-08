"""Profile the faithful-view cleanup on the slowest test receipts (timing only; outputs saved for identity checks)."""
from __future__ import annotations
import cProfile, io, json, os, pstats, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401

KEYS = sys.argv[1].split(",")
OUT = fe.ROOT / "artifacts" / "profile_faithful_20261108"


def main():
    import cv2
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.ocr import PaddleOCRAdapter, shutdown_gpu_worker
    from dsdnet_receipt.preprocessing import preprocess
    from dsdnet_receipt.restoration import (RestorationEngine, assess_receipt_degradation, make_faithful_content_view,
                                            make_readability_view, region_adaptive_source_fusion, select_faithful_recovery)
    from dsdnet_receipt.settings import PRESETS, defaults
    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    engine = RestorationEngine(config); ocr = PaddleOCRAdapter(config)
    opts = defaults(); opts.update(PRESETS["balanced"])
    variants = {f"{v['source_id']}/{v['variant_id']}": v for v in fe.load_variants()}
    report = []
    for key in KEYS:
        v = variants[key]
        image = cv2.imread(v["degraded_path"], cv2.IMREAD_COLOR)
        reg, _ = preprocess(image, config, preserve_color=True)
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY) if reg.ndim == 3 else reg
        res = engine.run(g)
        prof = assess_receipt_degradation(g)
        fused, _ = region_adaptive_source_fusion(g, res.image)
        sel = select_faithful_recovery(g, fused).image
        views = [("rectified_input", g), ("source_preserving_recovery", sel)]
        if res.model_output is not None and res.model_output.shape[:2] == g.shape[:2]:
            views.append(("dsdnet_restoration", res.model_output))
        if {"faded", "blurred"} & set(prof.get("flags", [])):
            views.append(("readability_view", make_readability_view(g, sel)))
        lines = ocr.run_views(views).lines
        name = key.replace("/", "_")
        np.save(OUT / f"{name}_source.npy", g); np.save(OUT / f"{name}_selected.npy", sel)
        (OUT / f"{name}_lines.json").write_text(json.dumps(lines, default=float), encoding="utf-8")
        (OUT / f"{name}_profile.json").write_text(json.dumps(prof, default=float), encoding="utf-8")
        if reg.ndim == 3:
            np.save(OUT / f"{name}_color.npy", reg)
        mask = config.get("artifact_mask", {})
        pr = cProfile.Profile(); t = time.perf_counter(); pr.enable()
        fv = make_faithful_content_view(g, sel, lines, mask.get("checkpoint") if mask.get("enabled") else None, float(mask.get("threshold", .95)),
                                        prof, source_color=reg if reg.ndim == 3 else None, native_refinement=config.get("native_refinement"),
                                        refinement_audit={}, finishing_options=opts)
        pr.disable(); dt = time.perf_counter() - t
        np.save(OUT / f"{name}_faithful.npy", fv)
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(35)
        (OUT / f"{name}_cprofile.txt").write_text(s.getvalue(), encoding="utf-8")
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
        (OUT / f"{name}_tottime.txt").write_text(s.getvalue(), encoding="utf-8")
        report.append({"key": key, "shape": list(g.shape), "flags": prof.get("flags"), "seconds": dt, "lines": len(lines)})
        print(report[-1], flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    shutdown_gpu_worker()


if __name__ == "__main__":
    main()
