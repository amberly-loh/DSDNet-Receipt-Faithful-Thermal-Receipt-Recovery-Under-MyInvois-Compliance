"""RQ2/RQ3 re-analysis from archived records only (20 November 2026).

Purpose
-------
The published field-level calibration and selective-acceptance numbers could not
be reproduced from the archive: the support evidence and the correctness label
came from two different runs five days apart, and the join that crossed them was
never deposited. This script re-runs both analyses inside a single archived
configuration, using the same score definition, the same validation-selection
rule and the same matched-coverage construction as
``scripts/final_evaluation_20261107.py``.

Nothing is re-run at the image level. Every input is a machine-readable record
that already exists in the experiment archive:

  validation fields  experiment_outputs/final_evaluation_20261107/records.jsonl
                     (partition == "validation", method == "P_full", 40 sources)
  test fields        experiment_outputs/final_run_20261108/records.jsonl
                     (method == "D_rq3", 95 sources)
  post-agreement     experiment_outputs/agreement_20261113/records.jsonl
  labels             (agreed_*_correct, joined on variant_key)

Two label variants are reported side by side, because the two archived runs
disagree on 125 of the 570 high-risk fields:

  pre   the correctness recorded with the field evidence itself (single
        configuration: evidence, prediction and label all from the same run)
  post  the correctness of the reading-agreement vote, joined across runs
        (secondary cross-stage analysis)

Usage
-----
    python scripts/rq2_rq3_reanalysis_20261120.py --data-root <path to private repo>
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FIELDS = ("merchant", "transaction_date", "total_amount")
HIGH_RISK = ("transaction_date", "total_amount")
PAIRS = ((0.80, 0.60), (0.85, 0.70), (0.90, 0.75))
COVERAGE_POINTS = (0.25, 0.40, 0.50, 0.60, 0.75)


# ---------------------------------------------------------------- primitives
def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with average ranks for ties (equivalent to scipy rankdata)."""
    labels = labels.astype(bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def isotonic_fit(scores: np.ndarray, outcomes: np.ndarray):
    """Pool-adjacent-violators, non-decreasing, clipped to [0, 1]."""
    order = np.argsort(scores, kind="mergesort")
    x, y = scores[order].astype(float), outcomes[order].astype(float)
    blocks: list[list[float]] = []
    for xi, yi in zip(x, y):
        blocks.append([yi, 1.0, xi, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, _lo, hi = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
            blocks[-1][3] = hi
    kx, ky = [], []
    for s, c, lo, hi in blocks:
        kx += [lo, hi]
        ky += [s / c, s / c]
    kx, ky = np.array(kx), np.clip(np.array(ky), 0, 1)
    return lambda values: np.interp(np.asarray(values, float), kx, ky)


def ece(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & ((probabilities < high) if high < 1 else (probabilities <= high))
        if mask.any():
            gap = abs(probabilities[mask].mean() - outcomes[mask].mean())
            total += mask.mean() * gap
            table.append({
                "bin": f"{low:.1f}-{high:.1f}",
                "n": int(mask.sum()),
                "mean_score": float(probabilities[mask].mean()),
                "observed": float(outcomes[mask].mean()),
            })
    return float(total), table


def brier(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


# ---------------------------------------------------------------- loading
def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def field_rows(records: list[dict]) -> list[dict]:
    out = []
    for row in records:
        for name, evidence in (row.get("field_evidence") or {}).items():
            if name not in HIGH_RISK:
                continue
            out.append({
                "source_id": row["source_id"],
                "variant_key": row["variant_key"],
                "field": name,
                **{k: v for k, v in evidence.items() if k not in ("value",)},
                "value": evidence.get("value"),
            })
    return out


def agreed_labels(path: Path) -> dict[tuple[str, str], bool]:
    table = {}
    for row in load_records(path):
        for name in HIGH_RISK:
            key = (row["variant_key"], name)
            if f"agreed_{name}_correct" in row:
                table[key] = bool(row[f"agreed_{name}_correct"])
    return table


def to_arrays(rows: list[dict], label: str, labels: dict[tuple[str, str], bool] | None):
    """Return dict of numpy arrays. label: 'pre' uses the recorded correctness."""
    support = {}
    for q in (10, 15, 20):
        support[q] = np.array([np.nan if r.get(f"q{q}") is None else float(r[f"q{q}"]) for r in rows])
    conf = np.array([0.0 if r.get("ocr_confidence") is None else float(r["ocr_confidence"]) for r in rows])
    rule = np.array([0.0 if r.get("rule") is None else float(r["rule"]) for r in rows])
    box = np.array([bool(r.get("box_valid")) for r in rows])
    supported = np.array([bool(r.get("supported")) if r.get("supported") is not None else False for r in rows])
    if label == "pre":
        correct = np.array([bool(r.get("correct")) for r in rows])
    else:
        assert labels is not None
        correct = np.array([labels.get((r["variant_key"], r["field"]), bool(r.get("correct"))) for r in rows])
    return {
        "q": support, "ocr_confidence": conf, "rule": rule, "box_valid": box,
        "supported": supported, "correct": correct,
        "source_id": np.array([r["source_id"] for r in rows]),
        "variant_key": np.array([r["variant_key"] for r in rows]),
        "field": np.array([r["field"] for r in rows]),
    }


# ---------------------------------------------------------------- analysis
def s_field(table: dict, q: int) -> np.ndarray:
    support = table["q"][q]
    support = np.where(np.isnan(support), 0.0, support)
    return np.clip(0.45 * support + 0.35 * table["c_ocr"] + 0.20 * table["rule"], 0, 1)


def decide(table: dict, q: int | None, t_accept: float, t_review: float, kind: str):
    if kind == "P":
        support = table["q"][q]
        score = s_field(table, q)
        accept = (score >= t_accept) & ~np.isnan(support) & (support >= t_review) & table["box_valid"] & (table["rule"] > 0)
    elif kind == "E":
        score = np.clip(0.70 * table["c_ocr"] + 0.30 * table["rule"], 0, 1)
        accept = (score >= t_accept) & (table["rule"] > 0)
    else:
        score = table["c_ocr"]
        accept = score >= t_accept
    return accept, score


def outcome(table: dict, accept: np.ndarray) -> dict:
    correct = table["correct"]
    unsupported = accept & (~correct | ~table["supported"])
    accepted = int(accept.sum())
    return {
        "fields": int(len(correct)),
        "accepted": accepted,
        "coverage": accepted / max(len(correct), 1),
        "accepted_accuracy": float(correct[accept].mean()) if accepted else None,
        "uar": float(unsupported.sum() / accepted) if accepted else None,
        "unsupported_acceptances": int(unsupported.sum()),
    }


def select_on_validation(val: dict) -> dict:
    selection = {}
    for kind in ("P", "E", "F"):
        candidates = []
        for q in ((10, 15, 20) if kind == "P" else (None,)):
            for t_accept, t_review in PAIRS:
                accept, _ = decide(val, q, t_accept, t_review, kind)
                candidates.append({"q": q, "t_accept": t_accept, "t_review": t_review, **outcome(val, accept)})
        feasible = [c for c in candidates if c["uar"] is not None and c["coverage"] >= 0.5]
        if feasible:
            best = sorted(feasible, key=lambda c: (c["uar"], -(c["accepted_accuracy"] or 0),
                                                  -c["coverage"], c["t_accept"], c["q"] or 0))[0]
            status = "feasible"
        else:
            pool = [c for c in candidates if c["uar"] is not None] or candidates
            best = sorted(pool, key=lambda c: (-c["coverage"], c["uar"] if c["uar"] is not None else 1))[0]
            status = "coverage constraint not met; highest-coverage candidate reported"
        selection[kind] = {"status": status, "chosen": best, "candidates": candidates}
    return selection


def matched_coverage(test: dict, q_sel: int, seed: int = 56, draws: int = 2000) -> list[dict]:
    correct = test["correct"]
    supported = test["supported"]
    gate = (~np.isnan(test["q"][q_sel])) & test["box_valid"] & (test["rule"] > 0)
    ranks = {
        "P": np.where(gate, s_field(test, q_sel), s_field(test, q_sel) - 2.0),
        "E": np.clip(0.70 * test["c_ocr"] + 0.30 * test["rule"], 0, 1),
        "F": test["c_ocr"],
    }
    n = len(correct)
    curves = {}
    for kind, values in ranks.items():
        order = np.lexsort((np.arange(n), -values))
        unsafe = (~correct | ~supported)[order]
        curves[kind] = [float(unsafe[:k].mean()) for k in range(1, n + 1)]
    sources = test["source_id"]
    unique = np.unique(sources)
    rng = np.random.default_rng(seed)
    out = []
    for target in COVERAGE_POINTS:
        k = int(round(target * n))
        entry = {"coverage": k / n}
        for kind in ranks:
            entry[f"uar_{kind}"] = curves[kind][k - 1]
        diffs = []
        for _ in range(draws):
            pick = rng.choice(unique, size=unique.size, replace=True)
            idx = np.concatenate([np.flatnonzero(sources == s) for s in pick])
            sub_n = idx.size
            kk = max(1, int(round(target * sub_n)))
            sub = {}
            for kind in ranks:
                vals = ranks[kind][idx]
                order = np.lexsort((np.arange(sub_n), -vals))
                sub[kind] = float((~correct[idx] | ~supported[idx])[order][:kk].mean())
            diffs.append(sub["E"] - sub["P"])
        entry["uar_reduction_vs_E_ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
        out.append(entry)
    return out


def analyse_variant(val: dict, test: dict, label: str) -> dict:
    # calibrate OCR confidence on validation, apply to both
    ocr_map = isotonic_fit(val["ocr_confidence"], val["correct"].astype(float))
    for table in (val, test):
        table["c_ocr"] = ocr_map(table["ocr_confidence"])

    selection = select_on_validation(val)
    q_sel = selection["P"]["chosen"]["q"]

    evaluation = {}
    for kind in ("P", "E", "F"):
        chosen = selection[kind]["chosen"]
        accept, _score = decide(test, chosen["q"], chosen["t_accept"], chosen["t_review"], kind)
        evaluation[kind] = {**outcome(test, accept), "q": chosen["q"],
                            "t_accept": chosen["t_accept"], "t_review": chosen["t_review"]}

    # calibration of the composite, isotonic fitted on validation
    s_val = s_field(val, q_sel)
    s_test = s_field(test, q_sel)
    field_map = isotonic_fit(s_val, val["correct"].astype(float))
    c_test = field_map(s_test)

    y_test = test["correct"].astype(float)
    metrics = {
        "label": label,
        "fields": int(len(y_test)),
        "sources": int(len(np.unique(test["source_id"]))),
        "composite_auc": rank_auc(s_test, test["correct"]),
        "ocr_auc": rank_auc(test["c_ocr"], test["correct"]),
        "composite_brier": brier(c_test, y_test),
        "ocr_brier": brier(test["c_ocr"], y_test),
    }
    metrics["composite_ece"], bins_c = ece(c_test, y_test)
    metrics["ocr_ece"], bins_o = ece(test["c_ocr"], y_test)
    metrics["reliability_composite"] = bins_c
    metrics["reliability_ocr"] = bins_o

    # ---- planned-threshold diagnostics -------------------------------------
    # (a) an arithmetic bound: the composite cannot exceed
    #     0.45*max(q) + 0.35*1 + 0.20*1, so with a support scale that peaks
    #     below 0.53 the smallest planned acceptance threshold of 0.80 is
    #     unreachable for every calibration map.
    bounds = {}
    for q in (10, 15, 20):
        qmax = float(np.nanmax(test["q"][q])) if np.isfinite(test["q"][q]).any() else float("nan")
        bounds[q] = {"max_support": qmax, "max_composite": 0.45 * qmax + 0.35 + 0.20}
    # (b) coverage at the planned pairs, under calibrated and raw confidence,
    #     because the archived code uses the calibrated term while the thesis
    #     coverage figures imply the raw one.
    planned = []
    for conf_name, conf in (("calibrated", test["c_ocr"]), ("raw", test["ocr_confidence"])):
        for t_accept, t_review in PAIRS:
            score_p = np.clip(0.45 * np.where(np.isnan(test["q"][q_sel]), 0.0, test["q"][q_sel])
                              + 0.35 * conf + 0.20 * test["rule"], 0, 1)
            accept_p = ((score_p >= t_accept) & ~np.isnan(test["q"][q_sel])
                        & (test["q"][q_sel] >= t_review) & test["box_valid"] & (test["rule"] > 0))
            score_e = np.clip(0.70 * conf + 0.30 * test["rule"], 0, 1)
            accept_e = (score_e >= t_accept) & (test["rule"] > 0)
            accept_f = conf >= t_accept
            planned.append({
                "confidence": conf_name, "t_accept": t_accept, "t_review": t_review,
                "accepted_P": int(accept_p.sum()), "accepted_E": int(accept_e.sum()),
                "accepted_F": int(accept_f.sum()), "fields": int(len(conf)),
            })

    return {
        "selected_q": q_sel,
        "validation_selection": selection,
        "test_operating_points": evaluation,
        "matched_coverage": matched_coverage(test, q_sel),
        "calibration": metrics,
        "composite_bound": bounds,
        "planned_threshold_coverage": planned,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="path to DSDNet-Receipt-Data-Private-main")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.data_root)
    # Both partitions come from the same run and the same configuration, which
    # is the whole point of this re-analysis: the published numbers joined a
    # support score from one run to a label from another. Here the validation
    # partition (40 sources) selects q and the thresholds, and the test
    # partition (95 sources) is evaluated, both from D_rq3 of the 8 November run.
    rq3 = [r for r in load_records(root / "experiment_outputs/final_run_20261108/records.jsonl")
           if r.get("method") == "D_rq3" and r.get("field_evidence")]
    val_records = [r for r in rq3 if r.get("partition") == "validation"]
    test_records = [r for r in rq3 if r.get("partition") == "test"]
    labels = agreed_labels(root / "experiment_outputs/agreement_20261113/records.jsonl")

    val_rows, test_rows = field_rows(val_records), field_rows(test_records)
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "protocol": "single archived configuration; validation selection; matched coverage",
        "inputs": {
            "validation": "final_run_20261108 (D_rq3, partition=validation, 40 sources)",
            "test": "final_run_20261108 (D_rq3, partition=test, 95 sources)",
            "agreed_labels": "agreement_20261113 (joined on variant_key)",
        },
        "counts": {
            "validation_images": len(val_records), "validation_fields": len(val_rows),
            "validation_sources": len({r["source_id"] for r in val_rows}),
            "test_images": len(test_records), "test_fields": len(test_rows),
            "test_sources": len({r["source_id"] for r in test_rows}),
            "agreed_labels": len(labels),
        },
        "variants": {},
    }
    for label in ("pre", "post"):
        val = to_arrays(val_rows, label, labels)
        test = to_arrays(test_rows, label, labels)
        report["variants"][label] = analyse_variant(val, test, label)

    # label cross-tabulation, the evidence for the endpoint finding
    pre = to_arrays(test_rows, "pre", None)["correct"]
    post = to_arrays(test_rows, "post", labels)["correct"]
    report["label_crosstab"] = {
        "n": int(len(pre)),
        "both_wrong": int(((~pre) & (~post)).sum()),
        "wrong_to_right": int(((~pre) & post).sum()),
        "right_to_wrong": int((pre & (~post)).sum()),
        "both_right": int((pre & post).sum()),
        "disagreement_rate": float((pre != post).mean()),
    }

    out = Path(args.out) if args.out else root / "experiment_outputs/rq2_rq3_reanalysis_20261120/report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # flat CSV of the headline numbers for the manuscript
    csv_path = out.with_name("summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "quantity", "value"])
        for label, block in report["variants"].items():
            cal = block["calibration"]
            for key in ("composite_auc", "ocr_auc", "composite_brier", "ocr_brier", "composite_ece", "ocr_ece"):
                writer.writerow([label, key, f"{cal[key]:.4f}"])
            for kind, op in block["test_operating_points"].items():
                writer.writerow([label, f"coverage_{kind}", f"{op['coverage']:.4f}"])
                writer.writerow([label, f"accepted_accuracy_{kind}",
                                 "" if op["accepted_accuracy"] is None else f"{op['accepted_accuracy']:.4f}"])
                writer.writerow([label, f"uar_{kind}", "" if op["uar"] is None else f"{op['uar']:.4f}"])
            for point in block["matched_coverage"]:
                for kind in ("P", "E", "F"):
                    writer.writerow([label, f"uar_{kind}_at_{point['coverage']:.3f}", f"{point[f'uar_{kind}']:.4f}"])

    print(json.dumps({
        "counts": report["counts"],
        "label_crosstab": report["label_crosstab"],
        "pre": {k: v for k, v in report["variants"]["pre"]["calibration"].items()
                if k.startswith(("composite_", "ocr_")) and "reliability" not in k},
        "post": {k: v for k, v in report["variants"]["post"]["calibration"].items()
                 if k.startswith(("composite_", "ocr_")) and "reliability" not in k},
        "bound": report["variants"]["pre"]["composite_bound"],
        "planned": report["variants"]["pre"]["planned_threshold_coverage"],
        "written": str(out),
    }, indent=1))


if __name__ == "__main__":
    main()
