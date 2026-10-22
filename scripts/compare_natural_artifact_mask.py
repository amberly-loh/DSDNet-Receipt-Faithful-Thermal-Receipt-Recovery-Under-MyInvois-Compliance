from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.images import read_gray, write_image  # noqa: E402
from dsdnet_receipt.restoration import make_faithful_content_view  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current and learned-mask faithful views on a natural receipt")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config, _ = load_config(REPO / "configs" / "default.json")
    source = read_gray(run_dir / "rectified_input.png")
    recovered = read_gray(run_dir / "ocr_input.png")
    audit = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    lines = audit.get("ocr_lines", [])
    baseline = make_faithful_content_view(source, recovered, lines)
    mask_config = config["artifact_mask"]
    candidate = make_faithful_content_view(
        source, recovered, lines, mask_config["checkpoint"], mask_config["threshold"]
    )
    output = REPO / "artifacts" / "candidate_natural_checks" / run_dir.name
    write_image(output / "faithful_before_mask.png", baseline)
    write_image(output / "faithful_learned_mask.png", candidate)
    divider = np.full((source.shape[0], 4), 160, np.uint8)
    write_image(output / "side_by_side.png", np.hstack((baseline, divider, candidate)))
    changed = np.abs(candidate.astype(np.int16) - baseline.astype(np.int16))
    summary = {
        "run": run_dir.name,
        "changed_pixel_fraction": float(np.mean(changed > 0)),
        "mean_absolute_change": float(changed.mean()),
        "maximum_change": int(changed.max()),
        "left": "current faithful",
        "right": "learned-mask candidate",
        "formal_test_evidence": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
