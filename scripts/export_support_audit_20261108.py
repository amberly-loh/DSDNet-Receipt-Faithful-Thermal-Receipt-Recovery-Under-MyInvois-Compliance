"""Independent human perceptual audit of the support rule (Chapter 3, AC2; 8 November 2026).

Exports 100 seeded audit items from the controlled test receipts. Each item shows the same
40x40-pixel neighbourhood twice: on the left the damaged input, on the right the DSDNet
restoration, with a red ring at one restored stroke pixel. A person answers whether the
stroke under the ring is visible in the damaged input. The rule label (Gsup at that pixel)
is written to a separate file and is not shown to the person. 50 items have Gsup = 1 and
50 have Gsup = 0, drawn only from pixels where the restoration shows ink.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402

OUT = fe.ROOT / "artifacts" / "support_audit_20261108"


def main() -> None:
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.restoration import RestorationEngine
    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    engine = RestorationEngine(config)
    variants = fe.load_variants()
    test = [v for v in fe.sampled_variants(variants, 95, 0) if v["project_partition"] == "test"]
    rng = np.random.default_rng(20261108)
    picks = [test[i] for i in rng.permutation(len(test))[:34]]
    items = {1: [], 0: []}
    for v in picks:
        if len(items[1]) >= 50 and len(items[0]) >= 50:
            break
        if not (v.get("support_target") and Path(v["support_target"]).is_file()):
            continue
        degraded = fe.read_gray(v["degraded_path"])
        target = fe.read_gray(v["support_target"])
        if target.shape != degraded.shape:
            target = cv2.resize(target, (degraded.shape[1], degraded.shape[0]), interpolation=cv2.INTER_NEAREST)
        restoration = engine.run(degraded, float(v["parameters"]["blur_sigma"]))
        restored = restoration.model_output if restoration.model_output is not None else restoration.image
        foreground = fe.foreground_for(v, degraded.shape)
        region = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ink = restored < 128
        h, w = degraded.shape
        border = np.zeros_like(ink); border[20:h - 20, 20:w - 20] = True
        for label in (1, 0):
            pool = np.flatnonzero((region & ink & border & ((target > 0) == bool(label))).ravel())
            if pool.size and len(items[label]) < 50:
                for idx in rng.choice(pool, size=min(3, pool.size, 50 - len(items[label])), replace=False):
                    y, x = divmod(int(idx), w)
                    items[label].append((v, degraded, restored, x, y))
    chosen = items[1][:50] + items[0][:50]
    order = rng.permutation(len(chosen))
    labels = []
    for n, i in enumerate(order, start=1):
        v, degraded, restored, x, y = chosen[i]
        crops = []
        for image in (degraded, restored):
            crop = image[y - 20:y + 20, x - 20:x + 20]
            big = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            cv2.circle(big, (120, 120), 13, (0, 0, 255), 2)
            crops.append(big)
        gap = np.full((240, 16, 3), 255, np.uint8)
        cv2.imwrite(str(OUT / f"item_{n:03d}.png"), np.hstack([crops[0], gap, crops[1]]))
        labels.append({"item": n, "variant_key": f"{v['source_id']}/{v['variant_id']}", "severity": v["severity"],
                       "x": x, "y": y, "rule_label_gsup": int(i < len(items[1][:50]))})
    (OUT / "rule_labels_hidden.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    (OUT / "DONE.txt").write_text(f"{len(labels)} items", encoding="utf-8")


if __name__ == "__main__":
    main()
