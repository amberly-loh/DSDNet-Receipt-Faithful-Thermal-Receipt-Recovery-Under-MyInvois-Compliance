"""Final end-to-end evaluation of the deployed DSDNet-Receipt system (7 November 2026).

Answers the three research questions on the controlled SROIE-Malaysia corpus
with the checkpoint and code that the application actually runs:

RQ1  Text and field extraction. Every method receives the same degraded
     full receipt, the same fixed PaddleOCR configuration and the same field
     parser. Methods: A direct, B Otsu, B Sauvola, C perceptual, D raw DSDNet
     (deployed checkpoint, single OCR view), the proposed system with its
     optional steps switched off ("fast" settings), and the full proposed
     system (the application's own upload path).
RQ2  Pixel trace support (support map against the controlled target G_sup)
     and field-level calibration of the composite decision score.
RQ3  Selective acceptance of the high-risk fields (date, total): proposed
     evidence gate against Baseline E (restore and extract, 0.70 OCR + 0.30
     rule) and Baseline F (calibrated OCR confidence only), with q and the
     thresholds chosen on validation only.

Selection (q, thresholds, isotonic maps) uses the validation partition (86
source groups). Reported results use the test partition (95 source groups),
which neither the deployed checkpoint nor the selection step has used.

Usage:  python scripts/final_evaluation_20261107.py collect   (resumable)
        python scripts/final_evaluation_20261107.py analyse
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts" / "final_evaluation_20261107"
MANIFEST = ROOT / "artifacts" / "protocol_lock" / "generation_manifest_611_sources.jsonl"
REFERENCES = ROOT.parent / "FYP datasets" / "External" / "SROIE-Malaysia" / "sroie_protocol_manifest.csv"
BASELINE_ORDER = ["A_direct", "B_otsu", "B_sauvola", "C_perceptual", "D_dsdnet_raw", "P_fast", "P_full"]
LABELS = {
    "A_direct": "A Direct extraction", "B_otsu": "B Otsu", "B_sauvola": "B Sauvola",
    "C_perceptual": "C Perceptual", "D_dsdnet_raw": "D DSDNet raw", "P_fast": "Proposed (fast)",
    "P_full": "Proposed (full)",
}
FIELDS = ("merchant", "transaction_date", "total_amount")
HIGH_RISK = ("transaction_date", "total_amount")


# ---------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unreadable image {path}")
    return image


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").upper()).strip()


def norm_merchant(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9& ]", " ", str(value or "").upper())).strip()


def norm_date(value: str) -> str | None:
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    for pattern in (r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2})\b"):
        match = re.search(pattern, text)
        if match:
            day, month, year = match.groups()
            year = int(year) + (2000 if len(year) == 2 else 0)
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return None
    return None


def norm_amount(value: str) -> str | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d{1,2})?", str(value or ""))
    if not match:
        return None
    try:
        return f"{abs(float(match.group(0).replace(',', ''))):.2f}"
    except ValueError:
        return None


def field_correct(name: str, predicted: str | None, reference: str) -> bool:
    if not predicted or not str(reference).strip():
        return False
    if name == "merchant":
        return norm_merchant(predicted) == norm_merchant(reference)
    if name == "transaction_date":
        return norm_date(predicted) is not None and norm_date(predicted) == norm_date(reference)
    return norm_amount(predicted) is not None and norm_amount(predicted) == norm_amount(reference)


def load_variants() -> list[dict]:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["project_partition"] in {"validation", "test"}]


def load_references() -> dict[str, dict]:
    with REFERENCES.open("r", newline="", encoding="utf-8-sig") as stream:
        return {row["source_id"]: row for row in csv.DictReader(stream)}


def foreground_for(variant: dict, shape: tuple[int, int]) -> np.ndarray:
    path = variant.get("clean_foreground") or str(Path(variant["degraded_path"]).with_name("clean_foreground.png"))
    if not Path(path).is_file():
        path = str(Path(variant["clean_reference"]).with_name("clean_foreground.png"))
    if Path(path).is_file():
        mask = read_gray(path) > 127
    else:  # derive from the clean reference when the stored mask is absent
        clean = read_gray(variant["clean_reference"])
        mask = clean < 160
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def bounds(box) -> tuple[int, int, int, int] | None:
    try:
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
    except (TypeError, ValueError, IndexError):
        return None
    return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


# ---------------------------------------------------------------- collection
def field_evidence(name: str, record: dict, restored: np.ndarray, support: np.ndarray | None,
                   support_target: np.ndarray | None, foreground: np.ndarray | None) -> dict:
    """Support aggregation for one extracted field, following Chapter 4:
    E_f is the support over retained foreground-aligned pixels inside the
    field's OCR box; fewer than 20 retained pixels means support undefined."""
    out = {"box_valid": False, "retained_pixels": 0, "q10": None, "q15": None, "q20": None,
           "support_mean": None, "supported": None}
    box = bounds(record.get("box")) if record else None
    if not box or support is None:
        return out
    x0, y0, x1, y1 = box
    h, w = restored.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return out
    region = restored[y0:y1, x0:x1]
    threshold, _ = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retained = region <= threshold  # ink pixels of the restored field
    values = support[y0:y1, x0:x1][retained].astype(np.float64) / 255.0
    out.update({"box_valid": True, "retained_pixels": int(retained.sum())})
    if values.size >= 20:
        out.update({"q10": float(np.quantile(values, .10)), "q15": float(np.quantile(values, .15)),
                    "q20": float(np.quantile(values, .20)), "support_mean": float(values.mean())})
    if support_target is not None and foreground is not None:
        fg = foreground[y0:y1, x0:x1]
        if fg.sum() >= 10:
            out["supported"] = bool((support_target[y0:y1, x0:x1][fg] > 0).mean() >= 0.5)
    return out


def rule_factor(name: str, value: str | None) -> float:
    """Deterministic rule factor P_rule: mandatory format checks for the
    high-risk fields (a failure is 0, the review floor)."""
    if not value:
        return 0.0
    if name == "transaction_date":
        parsed = norm_date(value)
        if not parsed:
            return 0.0
        year = int(parsed[:4])
        return 1.0 if 2000 <= year <= datetime.now().year else 0.8
    if name == "total_amount":
        amount = norm_amount(value)
        if not amount:
            return 0.0
        return 1.0 if 0 < float(amount) < 1_000_000 else 0.8
    return 1.0


def collect(limit: int | None) -> None:
    from dsdnet_receipt.baselines import BASELINES
    from dsdnet_receipt.config import load_config
    from dsdnet_receipt.fields import extract_fields
    from dsdnet_receipt.metrics import cer, psnr, ssim, wer
    from dsdnet_receipt.ocr import PaddleOCRAdapter
    from dsdnet_receipt.restoration import RestorationEngine
    from dsdnet_receipt.validation import stroke_fidelity_metrics
    from dsdnet_receipt.web_app import create_app

    OUT.mkdir(parents=True, exist_ok=True)
    config, _ = load_config(ROOT / "configs" / "default.json")
    references = load_references()
    variants = load_variants()
    if limit:
        keep = sorted({v["source_id"] for v in variants if v["project_partition"] == "test"})[:limit] + \
            sorted({v["source_id"] for v in variants if v["project_partition"] == "validation"})[:limit]
        variants = [v for v in variants if v["source_id"] in keep]
    records_path = OUT / "records.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["variant_key"], row["method"]))
    ocr = PaddleOCRAdapter(config)
    engine = RestorationEngine(config)
    calibration_path = Path(config["dsdnet"].get("evidence_support_calibration", ""))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.is_file() else {}

    # the deployed application, in-process, for the proposed system
    sandbox = OUT / "sandbox"
    sandbox.mkdir(exist_ok=True)
    app_config = dict(config)
    app_config["artifacts_root"] = str(sandbox / "artifacts")
    app_config_path = OUT / "config_snapshot.json"
    app_config_path.write_text(json.dumps(app_config, indent=2), encoding="utf-8")
    app = create_app(app_config_path, testing=True, database_path=sandbox / "evaluation.sqlite3")
    client = app.test_client()
    token = client.get("/api/auth").get_json()["csrf_token"]
    token = client.post("/api/workspace", headers={"X-CSRF-Token": token}).get_json()["csrf_token"]

    def set_preset(name: str) -> None:
        from dsdnet_receipt.settings import PRESETS
        response = client.patch("/api/settings", json=dict(PRESETS[name]), headers={"X-CSRF-Token": token})
        if response.status_code != 200:
            raise RuntimeError(f"settings preset {name} failed: {response.status_code}")

    started_all = time.perf_counter()
    with records_path.open("a", encoding="utf-8") as sink:
        for index, variant in enumerate(variants, start=1):
            key = f"{variant['source_id']}/{variant['variant_id']}"
            reference = references[variant["source_id"]]
            needed = [m for m in BASELINE_ORDER if (key, m) not in done]
            if variant["project_partition"] == "validation":
                needed = [m for m in needed if m == "D_dsdnet_raw"]  # selection data only
            if not needed:
                continue
            degraded = read_gray(variant["degraded_path"])
            clean = read_gray(variant["clean_reference"])
            if clean.shape != degraded.shape:
                clean = cv2.resize(clean, (degraded.shape[1], degraded.shape[0]), interpolation=cv2.INTER_AREA)
            foreground = foreground_for(variant, degraded.shape)
            target = read_gray(variant["support_target"]) if variant.get("support_target") and Path(variant["support_target"]).is_file() else None
            if target is not None and target.shape != degraded.shape:
                target = cv2.resize(target, (degraded.shape[1], degraded.shape[0]), interpolation=cv2.INTER_NEAREST)
            reference_text = norm_text(reference["ocr_reference_text"])

            def base_row(method: str) -> dict:
                return {"variant_key": key, "source_id": variant["source_id"], "variant_id": variant["variant_id"],
                        "partition": variant["project_partition"], "severity": variant["severity"], "method": method}

            def score_text(row: dict, text: str, values: dict, image: np.ndarray | None) -> None:
                hypothesis = norm_text(text)
                row["cer"] = cer(reference_text, hypothesis)
                row["wer"] = wer(reference_text, hypothesis)
                refs = {"merchant": reference["company"], "transaction_date": reference["transaction_date"],
                        "total_amount": reference["total_amount"]}
                for name in FIELDS:
                    row[f"{name}_value"] = values.get(name)
                    row[f"{name}_correct"] = bool(field_correct(name, values.get(name), refs[name]))
                if image is not None and image.shape == clean.shape:
                    row["psnr"] = float(psnr(clean, image))
                    row["ssim"] = float(ssim(clean, image))
                    row.update({k: v for k, v in stroke_fidelity_metrics(clean, image, foreground).items()
                                if k in ("stroke_recall", "deleted_stroke_rate", "false_stroke_rate")})

            try:
                for method in [m for m in needed if m in BASELINES]:
                    row = base_row(method)
                    t0 = time.perf_counter()
                    image = BASELINES[method](degraded)
                    t1 = time.perf_counter()
                    result = ocr.run(image)
                    t2 = time.perf_counter()
                    fields = extract_fields(result.text, result.lines)
                    score_text(row, result.text, {n: (fields.get(n) or {}).get("value") for n in FIELDS}, image)
                    row.update({"processing_seconds": t1 - t0, "ocr_seconds": t2 - t1, "total_seconds": t2 - t0,
                                "ocr_status": result.status})
                    sink.write(json.dumps(row) + "\n"); sink.flush()
                if "D_dsdnet_raw" in needed:
                    row = base_row("D_dsdnet_raw")
                    t0 = time.perf_counter()
                    restoration = engine.run(degraded, float(variant["parameters"]["blur_sigma"]))
                    if not restoration.model_available:
                        raise RuntimeError(restoration.error or restoration.status)
                    image = restoration.model_output if restoration.model_output is not None else restoration.image
                    support = restoration.diagnostic_map
                    t1 = time.perf_counter()
                    result = ocr.run(image)
                    t2 = time.perf_counter()
                    fields = extract_fields(result.text, result.lines)
                    score_text(row, result.text, {n: (fields.get(n) or {}).get("value") for n in FIELDS}, image)
                    row.update({"processing_seconds": t1 - t0, "ocr_seconds": t2 - t1, "total_seconds": t2 - t0,
                                "ocr_status": result.status})
                    # RQ2 pixel support against the controlled target, inside the text neighbourhood
                    if support is not None and target is not None:
                        region = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                        rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
                        index_pool = np.flatnonzero(region.ravel())
                        if index_pool.size:
                            pick = rng.choice(index_pool, size=min(4096, index_pool.size), replace=False)
                            row["pixel_scores"] = (support.ravel()[pick].astype(np.float64) / 255.0).round(4).tolist()
                            row["pixel_labels"] = (target.ravel()[pick] > 0).astype(int).tolist()
                    # RQ2/RQ3 field evidence on the high-risk fields
                    refs = {"transaction_date": reference["transaction_date"], "total_amount": reference["total_amount"]}
                    evidence = {}
                    for name in HIGH_RISK:
                        record = fields.get(name) or {}
                        value = record.get("value")
                        evidence[name] = {
                            "value": value, "correct": bool(field_correct(name, value, refs[name])),
                            "ocr_confidence": record.get("confidence"), "rule": rule_factor(name, value),
                            **field_evidence(name, record, image, support, target, foreground),
                        }
                    row["field_evidence"] = evidence
                    row["support_calibration"] = {k: calibration.get(k) for k in ("scales", "decision_threshold", "checkpoint_sha256")}
                    sink.write(json.dumps(row) + "\n"); sink.flush()
                for method, preset in (("P_fast", "fast"), ("P_full", "balanced")):
                    if method not in needed:
                        continue
                    set_preset(preset)
                    ok, encoded = cv2.imencode(".png", degraded)
                    t0 = time.perf_counter()
                    response = client.post("/api/analyze", data={"receipt": (io.BytesIO(encoded.tobytes()), f"{variant['source_id']}_{variant['variant_id']}.png")},
                                           headers={"X-CSRF-Token": token}, content_type="multipart/form-data")
                    elapsed = time.perf_counter() - t0
                    body = response.get_json(silent=True) or {}
                    row = base_row(method)
                    if response.status_code != 200:
                        row.update({"error": body.get("error") or f"HTTP {response.status_code}", "total_seconds": elapsed})
                        for name in FIELDS:
                            row[f"{name}_correct"] = False
                        row["cer"] = 1.0; row["wer"] = 1.0
                    else:
                        receipt = body.get("receipt") or {}
                        values = {"merchant": receipt.get("merchant"), "transaction_date": receipt.get("transaction_date"),
                                  "total_amount": receipt.get("total_amount")}
                        score_text(row, body.get("ocr_text") or "", values, None)
                        audit = body.get("audit") or {}
                        row.update({"total_seconds": elapsed, "stage_seconds": audit.get("stage_seconds"),
                                    "line_items": len(receipt.get("line_items") or []),
                                    "processing_settings": audit.get("processing_settings")})
                        client.delete(f"/api/receipts/{receipt.get('public_id')}", headers={"X-CSRF-Token": token})
                    sink.write(json.dumps(row) + "\n"); sink.flush()
            except Exception as error:  # record and continue: failures are part of the result
                failure = base_row("ERROR")
                failure.update({"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-2000:]})
                sink.write(json.dumps(failure) + "\n"); sink.flush()
            elapsed_all = time.perf_counter() - started_all
            print(f"[{index}/{len(variants)}] {key} {variant['project_partition']} done in {elapsed_all/60:.1f} min", flush=True)
            (OUT / "progress.json").write_text(json.dumps({"index": index, "total": len(variants), "minutes": elapsed_all / 60,
                                                           "updated": datetime.now().isoformat()}), encoding="utf-8")
    write_provenance(config)


def write_provenance(config: dict) -> None:
    import torch
    checkpoint = Path(config["dsdnet"]["checkpoint"])
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = len([l for l in subprocess.check_output(["git", "status", "--porcelain", "--", "src", "web", "tests", "scripts", "configs"], cwd=ROOT, text=True).splitlines() if l.strip()])
    except Exception:
        head, branch, dirty = "unavailable", "unavailable", None
    import importlib.metadata as md
    provenance = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "paddleocr": md.version("paddleocr"), "paddlepaddle": md.version("paddlepaddle"), "opencv": cv2.__version__,
        "git_head": head, "git_branch": branch, "uncommitted_code_files": dirty,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "references": str(REFERENCES), "references_sha256": sha256(REFERENCES),
        "default_config_sha256": sha256(ROOT / "configs" / "default.json"),
        "ocr_settings": config["ocr"], "dsdnet_settings": {k: v for k, v in config["dsdnet"].items() if k != "source_path"},
        "preprocessing": config["preprocessing"],
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- statistics
def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def bootstrap_mean_ci(values: np.ndarray, draws: int = 10000, seed: int = 56) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return float("nan"), float("nan")
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata
    labels = labels.astype(bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def isotonic_fit(scores: np.ndarray, outcomes: np.ndarray):
    """Pool-adjacent-violators, non-decreasing, clipped to [0, 1]; returns a
    piecewise-linear predictor (Chapter 3 calibration rule)."""
    order = np.argsort(scores, kind="mergesort")
    x, y = scores[order].astype(float), outcomes[order].astype(float)
    blocks = []  # [sum_y, count, x_min, x_max]
    for xi, yi in zip(x, y):
        blocks.append([yi, 1.0, xi, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, lo, hi = blocks.pop()
            blocks[-1][0] += s; blocks[-1][1] += c; blocks[-1][3] = hi
    knots_x, knots_y = [], []
    for s, c, lo, hi in blocks:
        knots_x += [lo, hi]; knots_y += [s / c, s / c]
    knots_x, knots_y = np.array(knots_x), np.clip(np.array(knots_y), 0, 1)
    return lambda values: np.interp(np.asarray(values, float), knots_x, knots_y)


def ece(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & ((probabilities < high) if high < 1 else (probabilities <= high))
        if mask.any():
            gap = abs(probabilities[mask].mean() - outcomes[mask].mean())
            total += mask.mean() * gap
            table.append({"bin": f"{low:.1f}-{high:.1f}", "n": int(mask.sum()), "mean_score": float(probabilities[mask].mean()),
                          "observed": float(outcomes[mask].mean())})
    return float(total), table


# ---------------------------------------------------------------- analysis
def analyse() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.stats import friedmanchisquare, wilcoxon

    rows = [json.loads(l) for l in (OUT / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    errors = [r for r in rows if r["method"] == "ERROR"]
    rows = [r for r in rows if r["method"] != "ERROR"]
    # keep the last record per (variant, method) in case of a resumed run
    latest = {}
    for row in rows:
        latest[(row["variant_key"], row["method"])] = row
    frame = pd.DataFrame(list(latest.values()))
    test = frame[frame.partition == "test"].copy()
    test["field_accuracy"] = test[[f"{n}_correct" for n in FIELDS]].mean(axis=1)
    test["high_risk_accuracy"] = test[[f"{n}_correct" for n in HIGH_RISK]].mean(axis=1)
    methods = [m for m in BASELINE_ORDER if m in set(test.method)]
    complete_keys = set.intersection(*[set(test[test.method == m].variant_key) for m in methods])
    test = test[test.variant_key.isin(complete_keys)]
    report: dict = {"generated": datetime.now(timezone.utc).isoformat(), "test_variants": len(complete_keys),
                    "test_sources": int(test.source_id.nunique()), "errors_recorded": len(errors),
                    "error_examples": [e.get("error") for e in errors[:10]]}

    # ---- RQ1 summary (source-level means, then mean over sources)
    metrics = ["cer", "wer", "field_accuracy", "high_risk_accuracy"] + [f"{n}_correct" for n in FIELDS]
    source = test.groupby(["method", "source_id"], as_index=False)[metrics + ["total_seconds"]].mean()
    summary = []
    for method in methods:
        part = source[source.method == method]
        entry = {"method": method, "label": LABELS[method]}
        for metric in metrics + ["total_seconds"]:
            values = part[metric].astype(float).to_numpy()
            low, high = bootstrap_mean_ci(values)
            entry[metric] = float(values.mean()); entry[f"{metric}_ci"] = [low, high]
        summary.append(entry)
    report["rq1_summary"] = summary

    # paired tests: proposed full against every other method, at source level
    tests = []
    for metric, better in (("cer", "lower"), ("wer", "lower"), ("field_accuracy", "higher"), ("high_risk_accuracy", "higher")):
        pvals, partial = [], []
        proposed = source[source.method == "P_full"].set_index("source_id")[metric]
        for method in methods:
            if method == "P_full":
                continue
            other = source[source.method == method].set_index("source_id")[metric]
            joined = pd.concat([proposed, other], axis=1, keys=["p", "o"]).dropna()
            diff = (joined.o - joined.p) if better == "lower" else (joined.p - joined.o)
            nonzero = diff[diff != 0]
            p = float(wilcoxon(nonzero, alternative="greater").pvalue) if len(nonzero) >= 6 else 1.0
            low, high = bootstrap_mean_ci(diff.to_numpy())
            partial.append({"metric": metric, "comparator": method, "mean_benefit": float(diff.mean()), "ci95": [low, high],
                            "sources_improved": int((diff > 0).sum()), "sources_worse": int((diff < 0).sum()),
                            "sources_tied": int((diff == 0).sum()), "p_one_sided": p})
            pvals.append(p)
        for item, adjusted in zip(partial, holm(pvals)):
            item["p_holm"] = adjusted
            item["significant"] = adjusted < 0.05 and item["ci95"][0] > 0
        tests += partial
    report["rq1_paired_tests"] = tests

    # severity breakdown and Friedman across methods
    test_sev = test.groupby(["method", "severity", "source_id"], as_index=False)[["cer", "field_accuracy"]].mean()
    severity = test_sev.groupby(["method", "severity"], as_index=False)[["cer", "field_accuracy"]].mean()
    report["rq1_severity"] = severity.to_dict(orient="records")
    pivot = source.pivot(index="source_id", columns="method", values="cer").dropna()
    report["friedman_cer"] = float(friedmanchisquare(*[pivot[m] for m in methods]).pvalue)
    pivot_f = source.pivot(index="source_id", columns="method", values="field_accuracy").dropna()
    report["friedman_field_accuracy"] = float(friedmanchisquare(*[pivot_f[m] for m in methods]).pvalue)

    # image fidelity on full receipts (methods that keep the receipt geometry)
    fidelity = []
    for method in [m for m in methods if m in ("A_direct", "B_otsu", "B_sauvola", "C_perceptual", "D_dsdnet_raw")]:
        part = test[test.method == method].groupby("source_id")[["psnr", "ssim", "stroke_recall", "deleted_stroke_rate", "false_stroke_rate"]].mean()
        fidelity.append({"method": method, **{k: float(v) for k, v in part.mean().items()}})
    report["image_fidelity_full_receipt"] = fidelity

    # ---- RQ2 pixel support (test) and field calibration (fit on validation)
    dtest = frame[(frame.method == "D_dsdnet_raw") & (frame.partition == "test")]
    dval = frame[(frame.method == "D_dsdnet_raw") & (frame.partition == "validation")]
    threshold = None
    for row in dtest.itertuples():
        if isinstance(getattr(row, "support_calibration", None), dict):
            threshold = row.support_calibration.get("decision_threshold")
            break
    threshold = float(threshold) if threshold is not None else 0.15
    pixel = []
    for row in dtest.itertuples():
        scores, labels = getattr(row, "pixel_scores", None), getattr(row, "pixel_labels", None)
        if isinstance(scores, list) and scores:
            s, y = np.array(scores), np.array(labels)
            predicted = s >= threshold
            tp, fp, fn = (predicted & (y == 1)).sum(), (predicted & (y == 0)).sum(), (~predicted & (y == 1)).sum()
            pixel.append({"source_id": row.source_id, "auc": auc(s, y),
                          "f1": float(2 * tp / max(2 * tp + fp + fn, 1))})
    pix = pd.DataFrame(pixel).groupby("source_id").mean()
    report["rq2_pixel"] = {"sources": len(pix), "threshold": threshold,
                           "auc": float(pix.auc.mean()), "auc_ci": list(bootstrap_mean_ci(pix.auc.dropna().to_numpy())),
                           "trace_f1": float(pix.f1.mean()), "trace_f1_ci": list(bootstrap_mean_ci(pix.f1.to_numpy()))}

    def field_table(part):
        out = []
        for row in part.itertuples():
            for name, ev in (row.field_evidence or {}).items():
                out.append({"source_id": row.source_id, "variant_key": row.variant_key, "field": name, **ev})
        return pd.DataFrame(out)

    fval, ftest = field_table(dval), field_table(dtest)
    for table in (fval, ftest):
        table["ocr_confidence"] = table["ocr_confidence"].astype(float).fillna(0.0)
        table["correct"] = table["correct"].astype(bool)
    ocr_map = isotonic_fit(fval.ocr_confidence.to_numpy(), fval.correct.to_numpy())
    for table in (fval, ftest):
        table["c_ocr"] = ocr_map(table.ocr_confidence.to_numpy())

    def s_field(table, q):
        support = table[f"q{q}"].astype(float)
        return (0.45 * support.fillna(0.0) + 0.35 * table.c_ocr + 0.20 * table.rule).clip(0, 1)

    # ---- RQ3: validation selection of q and thresholds (Chapter 3/4 rules)
    pairs = [(0.80, 0.60), (0.85, 0.70), (0.90, 0.75)]

    def decide(table, q, t_accept, t_review, kind):
        if kind == "P":
            support = table[f"q{q}"].astype(float)
            score = s_field(table, q)
            accept = (score >= t_accept) & support.notna() & (support >= t_review) & table.box_valid & (table.rule > 0)
        elif kind == "E":
            score = (0.70 * table.c_ocr + 0.30 * table.rule).clip(0, 1)
            accept = (score >= t_accept) & (table.rule > 0)
        else:  # F: calibrated OCR confidence only
            score = table.c_ocr
            accept = score >= t_accept
        return accept.to_numpy(bool), score.to_numpy(float)

    def outcome(table, accept):
        unsupported = accept & (~table.correct.to_numpy() | ~(table.supported.fillna(False).astype(bool).to_numpy()))
        accepted = int(accept.sum())
        return {"accepted": accepted, "coverage": accepted / max(len(table), 1),
                "accepted_accuracy": float(table.correct.to_numpy()[accept].mean()) if accepted else None,
                "uar": float(unsupported.sum() / accepted) if accepted else None,
                "unsupported_acceptances": int(unsupported.sum()), "fields": len(table)}

    selection = {}
    for kind in ("P", "E", "F"):
        candidates = []
        for q in ((10, 15, 20) if kind == "P" else (None,)):
            for t_accept, t_review in pairs:
                accept, _ = decide(fval, q, t_accept, t_review, kind)
                result = outcome(fval, accept)
                candidates.append({"q": q, "t_accept": t_accept, "t_review": t_review, **result})
        feasible = [c for c in candidates if c["uar"] is not None and c["coverage"] >= 0.5]
        if feasible:
            best = sorted(feasible, key=lambda c: (c["uar"], -(c["accepted_accuracy"] or 0), -c["coverage"], c["t_accept"], c["q"] or 0))[0]
            status = "feasible"
        else:
            best = sorted([c for c in candidates if c["uar"] is not None] or candidates, key=lambda c: (-c["coverage"], c["uar"] or 1))[0]
            status = "coverage constraint not met; highest-coverage candidate reported"
        selection[kind] = {"status": status, "chosen": best, "candidates": candidates}
    report["rq3_validation_selection"] = selection

    evaluation = {}
    for kind in ("P", "E", "F"):
        chosen = selection[kind]["chosen"]
        accept, score = decide(ftest, chosen["q"], chosen["t_accept"], chosen["t_review"], kind)
        evaluation[kind] = {**outcome(ftest, accept), "q": chosen["q"], "t_accept": chosen["t_accept"], "t_review": chosen["t_review"]}
        ftest[f"accept_{kind}"] = accept
        ftest[f"score_{kind}"] = score
    report["rq3_test_operating_points"] = evaluation

    # matched-coverage risk curves: rank fields by each policy's score
    # (proposed: fields failing the support gate rank last)
    curves, matched = {}, []
    correct = ftest.correct.to_numpy()
    supported = ftest.supported.fillna(False).astype(bool).to_numpy()
    q_sel = selection["P"]["chosen"]["q"]
    gate = ftest[f"q{q_sel}"].astype(float).notna().to_numpy() & ftest.box_valid.to_numpy(bool) & (ftest.rule.to_numpy() > 0)
    ranks = {"P": np.where(gate, ftest.score_P, ftest.score_P - 2.0), "E": ftest.score_E.to_numpy(), "F": ftest.score_F.to_numpy()}
    n = len(ftest)
    for kind, values in ranks.items():
        order = np.lexsort((np.arange(n), -values))
        unsafe = (~correct | ~supported)[order]
        curves[kind] = [{"coverage": k / n, "uar": float(unsafe[:k].mean())} for k in range(1, n + 1)]
    for target in (0.25, 0.40, 0.50, 0.60, 0.75):
        k = int(round(target * n))
        entry = {"coverage": k / n}
        for kind in ranks:
            entry[f"uar_{kind}"] = curves[kind][k - 1]["uar"]
        # cluster bootstrap of the P-E difference at this coverage
        sources = ftest.source_id.to_numpy()
        unique = np.unique(sources)
        rng = np.random.default_rng(56)
        diffs = []
        for _ in range(2000):
            pick = rng.choice(unique, size=unique.size, replace=True)
            idx = np.concatenate([np.flatnonzero(sources == s) for s in pick])
            sub_n = idx.size
            kk = max(1, int(round(target * sub_n)))
            def uar_for(kind):
                vals = ranks[kind][idx]
                order = np.lexsort((np.arange(sub_n), -vals))
                return float((~correct[idx] | ~supported[idx])[order][:kk].mean())
            diffs.append(uar_for("E") - uar_for("P"))
        entry["uar_reduction_vs_E_ci95"] = [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))]
        matched.append(entry)
    report["rq3_matched_coverage"] = matched

    # calibration of the composite score (isotonic on validation, ECE on test)
    s_val = s_field(fval, q_sel).to_numpy(); s_test = s_field(ftest, q_sel).to_numpy()
    field_map = isotonic_fit(s_val, fval.correct.to_numpy())
    c_test = field_map(s_test)
    ece_value, bins = ece(c_test, ftest.correct.to_numpy().astype(float))
    ece_ocr, _ = ece(ftest.c_ocr.to_numpy(), ftest.correct.to_numpy().astype(float))
    report["rq2_field_calibration"] = {
        "fields_validation": len(fval), "fields_test": len(ftest), "q": q_sel,
        "ece_composite": ece_value, "brier_composite": float(np.mean((c_test - ftest.correct.to_numpy()) ** 2)),
        "ece_ocr_only": ece_ocr, "brier_ocr_only": float(np.mean((ftest.c_ocr.to_numpy() - ftest.correct.to_numpy()) ** 2)),
        "auc_composite_for_correct": auc(s_test, ftest.correct.to_numpy()),
        "auc_ocr_for_correct": auc(ftest.ocr_confidence.to_numpy(), ftest.correct.to_numpy()),
        "auc_support_for_supported": auc(ftest[f"q{q_sel}"].astype(float).fillna(0).to_numpy(), supported),
        "reliability_bins": bins,
        "support_undefined_fields": int(ftest[f"q{q_sel}"].isna().sum()),
    }

    # ---- performance
    perf = test.groupby("method").total_seconds.describe(percentiles=[.5, .95]).reset_index()
    report["performance_seconds"] = perf.to_dict(orient="records")

    (OUT / "final_evaluation_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(summary).to_csv(OUT / "table_rq1_summary.csv", index=False)
    pd.DataFrame(tests).to_csv(OUT / "table_rq1_paired_tests.csv", index=False)
    severity.to_csv(OUT / "table_rq1_severity.csv", index=False)
    ftest.to_csv(OUT / "rq3_test_fields.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = [LABELS[m] for m in methods]
    ax[0].bar(names, [s["cer"] for s in summary], color=["#9aa5a2"] * (len(methods) - 1) + ["#1b6b5a"])
    ax[0].set_title("Character error rate (lower is better)"); ax[0].tick_params(axis="x", rotation=40)
    ax[1].bar(names, [s["field_accuracy"] for s in summary], color=["#9aa5a2"] * (len(methods) - 1) + ["#1b6b5a"])
    ax[1].set_title("Field accuracy (merchant, date, total)"); ax[1].tick_params(axis="x", rotation=40)
    fig.tight_layout(); fig.savefig(OUT / "figure_rq1_methods.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, label in (("P", "Proposed evidence gate"), ("E", "Baseline E restore+extract"), ("F", "Baseline F OCR confidence")):
        cov = [p["coverage"] for p in curves[kind]]; risk = [p["uar"] for p in curves[kind]]
        ax.plot(cov, risk, label=label)
    ax.set_xlabel("Coverage (share of high-risk fields accepted)"); ax.set_ylabel("Unsupported acceptance rate")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "figure_rq3_risk_coverage.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.plot([b["mean_score"] for b in bins], [b["observed"] for b in bins], "o-", color="#1b6b5a")
    ax.set_xlabel("Calibrated composite score"); ax.set_ylabel("Observed field correctness")
    fig.tight_layout(); fig.savefig(OUT / "figure_rq2_reliability.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in methods:
        part = severity[severity.method == method].set_index("severity").reindex(["light", "moderate", "severe"])
        ax.plot(["light", "moderate", "severe"], part.field_accuracy, marker="o", label=LABELS[method])
    ax.set_ylabel("Field accuracy"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "figure_rq1_severity.png", dpi=180); plt.close(fig)
    print(json.dumps({k: report[k] for k in ("test_variants", "test_sources", "rq1_summary")}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("collect", "analyse"))
    parser.add_argument("--limit", type=int, default=None, help="sources per partition (smoke run)")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.limit)
    else:
        analyse()
