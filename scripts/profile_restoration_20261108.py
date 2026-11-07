"""Time DSDNet restoration settings on validation receipts and check the outputs stay the same."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402

def main():
    import torch
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.restoration import RestorationEngine
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    variants = fe.load_variants()
    val = [v for v in fe.sampled_variants(variants, 0, 40) if v["project_partition"] == "validation"][:6]
    images = [fe.read_gray(v["degraded_path"]) for v in val]
    engine = RestorationEngine(config)
    out = {"shapes": [list(i.shape) for i in images], "runs": []}
    ref = {}
    for label, batch, bench in (("b16", 16, False), ("b16_again", 16, False), ("b32", 32, False), ("b64", 64, False), ("b64_bench", 64, True)):
        os.environ["RECEIPT_TILE_BATCH"] = str(batch)
        torch.backends.cudnn.benchmark = bench
        engine.run(images[0])  # warm
        torch.cuda.synchronize()
        times, diffs, diag = [], [], []
        for k, img in enumerate(images):
            t = time.perf_counter(); r = engine.run(img); torch.cuda.synchronize(); times.append(time.perf_counter() - t)
            mo = r.model_output.astype(int)
            if label == "b16":
                ref[k] = (mo, r.diagnostic_map.astype(int), r.image.astype(int))
            else:
                diffs.append(int(np.abs(mo - ref[k][0]).max())); diag.append(int(np.abs(r.diagnostic_map.astype(int) - ref[k][1]).max()))
                diffs.append(int(np.abs(r.image.astype(int) - ref[k][2]).max()))
        out["runs"].append({"label": label, "mean_seconds": float(np.mean(times)), "times": times,
                            "max_pixel_diff_vs_b16": max(diffs) if diffs else 0, "max_support_diff_vs_b16": max(diag) if diag else 0})
        print(out["runs"][-1], flush=True)
    (fe.ROOT / "artifacts" / "profile_restoration_20261108.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

if __name__ == "__main__":
    main()
