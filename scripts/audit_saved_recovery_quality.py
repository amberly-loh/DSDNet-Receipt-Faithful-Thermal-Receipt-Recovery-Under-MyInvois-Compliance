from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsdnet_receipt.config import load_config  # noqa: E402
from dsdnet_receipt.restoration import (  # noqa: E402
    RestorationEngine,
    assess_receipt_degradation,
    detect_source_text_mask,
    make_faithful_content_view,
)


def _read(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _digest(image: np.ndarray) -> str:
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(small.tobytes()).hexdigest()


def _quality(source: np.ndarray, recovered: np.ndarray) -> dict[str, float]:
    text = detect_source_text_mask(source, protection="wide")
    text = cv2.dilate(text.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    src_bg = cv2.GaussianBlur(source, (0, 0), 7)
    out_bg = cv2.GaussianBlur(recovered, (0, 0), 7)
    src_dark = np.maximum(src_bg.astype(np.int16) - source.astype(np.int16), 0).astype(np.float32)
    out_dark = np.maximum(out_bg.astype(np.int16) - recovered.astype(np.int16), 0).astype(np.float32)
    text_count = max(int(np.count_nonzero(text)), 1)
    retention = float(np.count_nonzero((out_dark >= 1.5) & text) / text_count)
    paper = ~cv2.dilate(text.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    unsupported = paper & (out_dark > src_dark + 6.0) & (out_dark >= 8.0)
    unsupported_rate = float(np.count_nonzero(unsupported) / max(np.count_nonzero(paper), 1))
    src_edges = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3) ** 2
    out_edges = cv2.Sobel(recovered, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(recovered, cv2.CV_32F, 0, 1, ksize=3) ** 2
    source_edge = float(np.mean(np.sqrt(src_edges[text]))) if text.any() else 0.0
    output_edge = float(np.mean(np.sqrt(out_edges[text]))) if text.any() else 0.0
    edge_ratio = output_edge / max(source_edge, 1e-6)
    ink = out_dark[text & (out_dark >= 2.0)]
    tone_cv = float(np.std(ink) / max(np.mean(ink), 1e-6)) if ink.size else 9.0
    changed = float(np.mean(np.abs(recovered.astype(np.int16) - source.astype(np.int16)) >= 5))
    # Higher is worse. Weight content loss most heavily, then invented dark
    # structure and blur. Tone CV is capped because mixed font weights are real.
    failure = (
        3.0 * max(0.0, 0.82 - retention)
        + 8.0 * unsupported_rate
        + 1.5 * max(0.0, 0.90 - edge_ratio)
        + 0.35 * max(0.0, min(tone_cv, 2.0) - 0.85)
    )
    return {
        "text_retention": retention,
        "unsupported_darkening_rate": unsupported_rate,
        "text_edge_ratio": edge_ratio,
        "ink_tone_cv": tone_cv,
        "changed_pixel_fraction": changed,
        "failure_score": float(failure),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch audit saved natural receipt recoveries")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-worst", type=int, default=16)
    parser.add_argument("--run-list", type=Path, help="Optional JSON array of run names to re-audit")
    parser.add_argument("--run", action="append", help="Specific saved run name; repeatable")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "batch_recovery_audit")
    parser.add_argument("--checkpoint", type=Path, help="Rerun every source through this DSDNet checkpoint")
    args = parser.parse_args()
    config, _ = load_config(ROOT / "configs" / "default.json")
    if args.checkpoint:
        config["dsdnet"]["checkpoint"] = str(args.checkpoint.resolve())
    engine = RestorationEngine(config) if args.checkpoint else None
    mask = config.get("artifact_mask", {})
    runs_root = Path(config["artifacts_root"]) / "user_runs"
    rows: list[dict[str, object]] = []
    candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    seen: set[str] = set()
    selected_runs = sorted(runs_root.iterdir(), reverse=True)
    if args.run_list:
        selection = json.loads(args.run_list.read_text(encoding="utf-8"))
        selected_names = set(selection.get("worst_runs", []) if isinstance(selection, dict) else selection)
        selected_runs = [run for run in selected_runs if run.name in selected_names]
    if args.run:
        selected_names = set(args.run)
        selected_runs = [run for run in selected_runs if run.name in selected_names]
    for run in selected_runs:
        source_path = run / "rectified_input.png"
        if not source_path.is_file():
            source_path = run / "input.png"
        recovery_path = run / "ocr_input.png"
        if not recovery_path.is_file():
            recovery_path = run / "restored.png"
        audit_path = run / "audit.json"
        source, model = _read(source_path), _read(recovery_path)
        if source is None or (model is None and engine is None) or not audit_path.is_file():
            continue
        digest = _digest(source)
        if digest in seen:
            continue
        seen.add(digest)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        profile = assess_receipt_degradation(source)
        if engine is not None:
            model_result = engine.run(source)
            if not model_result.model_available:
                raise RuntimeError(f"Checkpoint failed for {run.name}: {model_result.error}")
            model = model_result.image
        recovered = make_faithful_content_view(
            source, model, audit.get("ocr_lines", []),
            mask.get("checkpoint") if mask.get("enabled") else None,
            float(mask.get("threshold", 0.95)), profile,
        )
        metrics = _quality(source, recovered)
        rows.append({
            "run": run.name,
            "source_name": audit.get("source", ""),
            "pixel_sha256": digest,
            "profile": profile["profile"],
            "flags": "+".join(profile["flags"]),
            **metrics,
        })
        candidates.append((float(metrics["failure_score"]), run.name, source, recovered))
        if args.limit and len(rows) >= args.limit:
            break
        print(f"Audited {len(rows)} unique receipts", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: float(row["failure_score"]), reverse=True)
    if rows:
        with (args.output / "records.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    worst = args.output / "worst"
    worst.mkdir(exist_ok=True)
    for rank, (score, run_name, source, recovered) in enumerate(sorted(candidates, reverse=True)[: args.save_worst], 1):
        height = max(source.shape[0], recovered.shape[0])
        width = max(source.shape[1], recovered.shape[1])
        left = cv2.copyMakeBorder(source, 0, height-source.shape[0], 0, width-source.shape[1], cv2.BORDER_CONSTANT, value=230)
        right = cv2.copyMakeBorder(recovered, 0, height-recovered.shape[0], 0, width-recovered.shape[1], cv2.BORDER_CONSTANT, value=230)
        divider = np.full((height, 5), 128, np.uint8)
        cv2.imwrite(str(worst / f"{rank:02d}_{run_name}_{score:.3f}.png"), np.hstack((left, divider, right)))
    summary = {
        "unique_receipts": len(rows),
        "profiles": {p: sum(row["profile"] == p for row in rows) for p in sorted({str(row["profile"]) for row in rows})},
        "mean": {key: float(np.mean([float(row[key]) for row in rows])) for key in (
            "text_retention", "unsupported_darkening_rate", "text_edge_ratio", "ink_tone_cv", "changed_pixel_fraction", "failure_score"
        )} if rows else {},
        "worst_runs": [row["run"] for row in rows[: args.save_worst]],
        "scope": "natural-source regression audit; not clean-reference restoration evidence",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else "saved model output",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
