from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsdnet_receipt.config import load_config, write_json
from dsdnet_receipt.ocr import PaddleOCRAdapter, deduplicate_lines
from dsdnet_receipt.preprocessing import preprocess
from dsdnet_receipt.restoration import RestorationEngine, make_adaptive_document_view, make_readability_view


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural-receipt candidate check; not formal test evidence")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config, _ = load_config(REPO / "configs" / "default.json")
    config["dsdnet"]["checkpoint"] = str(args.checkpoint.resolve())
    config["dsdnet"]["restoration_blend_alpha"] = args.alpha
    source = cv2.imread(str(run / "input.png"), cv2.IMREAD_GRAYSCALE)
    if source is None: raise SystemExit("Input image not found")
    processed, preprocessing = preprocess(source, config)
    restoration = RestorationEngine(config).run(processed)
    readability = make_readability_view(processed, restoration.image)
    ocr = PaddleOCRAdapter(config).run_views([
        ("rectified_input", processed),
        ("source_preserving_recovery", restoration.image),
        ("readability_view", readability),
    ])
    view, display_mode = make_adaptive_document_view(readability, ocr.lines, processed)
    output = REPO / "artifacts" / "candidate_natural_checks" / f"{args.checkpoint.parent.name}_{args.checkpoint.stem}_{run.name}"
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "original.png"), processed)
    cv2.imwrite(str(output / "model_recovery.png"), restoration.image)
    cv2.imwrite(str(output / "document_view.png"), view)
    lines = deduplicate_lines(ocr.lines)
    summary = {
        "scope": "single natural receipt development check; not formal test evidence",
        "source_run": run.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "alpha": args.alpha,
        "display_mode": display_mode,
        "ocr_status": ocr.status,
        "ocr_line_count": len(lines),
        "mean_ocr_confidence": sum(float(line.get("confidence") or 0) for line in lines) / max(len(lines), 1),
        "recognised_text": [line.get("text", "") for line in lines],
        "preprocessing": preprocessing,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
