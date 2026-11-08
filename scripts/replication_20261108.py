"""Independent replication on the 53 test-partition SROIE receipts that were never generated, used or
scored (8 Nov 2026). They were left out of the 611-source generation manifest only because generation
stopped after the first 611 rows of the protocol manifest.

The plan is written to artifacts/replication_20261108/plan.json before any image is generated or scored:
* same degradation generator and seeds; three severities per receipt;
* same final system (balanced preset, GPU text reader) and the same baselines A-D, text reader and references;
* RQ2/RQ3 evidence (D_rq3) under the first-evaluation protocol; calibration and selection stay on the
  40 validation sources of the final run;
* AC1, AC2/AC3, AC4 and NFR1 decision rules unchanged. Results are reported on the replication set alone and
  on the union with the locked primary set (positions 41-95), 108 receipts never used for any design
  decision. The extension was decided after the locked run's verdicts were known, to gain statistical power;
  the locked run's results are retained and reported beside it.
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_evaluation_20261107 as fe  # noqa: E402
import dsdnet_receipt  # noqa: E402,F401
import final_run_20261108 as fr  # noqa: E402

OUT = fe.ROOT / "artifacts" / "replication_20261108"
DATASET = fe.ROOT.parent / "FYP datasets" / "External" / "SROIE-Malaysia"


def generate():
    import cv2
    from dsdnet_receipt.config import load_config, write_json
    from dsdnet_receipt.degradation import degrade, variant_seed
    from dsdnet_receipt.images import read_gray, sha256_file, write_image
    config, _ = load_config(fe.ROOT / "configs" / "default.json")
    with (DATASET / "sroie_protocol_manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    locked = {json.loads(l)["source_id"] for l in fe.MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()}
    sources = [r for r in rows if r["project_partition"] == "test" and r["source_id"] not in locked]
    plan = OUT / "plan.json"
    if not plan.exists():
        plan.write_text(json.dumps({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "sources": [r["source_id"] for r in sources],
                                    "count": len(sources), "doc": __doc__}, indent=1), encoding="utf-8")
    out = OUT / "degraded"; maximum = int(config["preprocessing"]["max_long_edge"])
    manifest = OUT / "replication_manifest.jsonl"
    if manifest.exists():
        return
    records = []
    for source in sources:
        sid = source["source_id"]
        clean = read_gray(DATASET / source["relative_path"])
        scale = min(1.0, maximum / max(clean.shape))
        if scale < 1.0:
            clean = cv2.resize(clean, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sdir = out / sid; cpath = sdir / "clean_reference.png"; write_image(cpath, clean)
        for number, severity in enumerate(config["degradation"]["severities"], start=1):
            vid = f"V{number:02d}"; seed = variant_seed(config["base_seed"], sid, vid, severity)
            degraded, masks, parameters = degrade(clean, severity, config, seed)
            vdir = sdir / f"{vid}_{severity}"; dpath = vdir / "degraded.png"; write_image(dpath, degraded)
            for name, mask in masks.items():
                write_image(vdir / f"{name}.png", mask.astype(np.uint8) * 255)
            write_json(vdir / "parameters.json", parameters)
            records.append({"source_id": sid, "source_group": sid, "project_partition": "test", "variant_id": vid, "severity": severity,
                            "seed": seed, "clean_reference": str(cpath.resolve()), "degraded_path": str(dpath.resolve()),
                            "degraded_sha256": sha256_file(dpath), "support_target": str((vdir / "support_target.png").resolve()), "parameters": parameters})
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    generate()
    variants = [json.loads(l) for l in (OUT / "replication_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # reuse the locked run's evaluation loop on the replication variants
    fr.OUT = OUT
    fe.load_variants = lambda: variants
    fe.sampled_variants = lambda v, t, s: list(v)
    fr.main("test")
    from dsdnet_receipt.ocr import shutdown_gpu_worker
    shutdown_gpu_worker()
