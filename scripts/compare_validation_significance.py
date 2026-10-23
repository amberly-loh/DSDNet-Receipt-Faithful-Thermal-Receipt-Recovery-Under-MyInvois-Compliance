from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
PROPOSED = "Proposed_DSDNet_Receipt"
METRICS = {
    "ssim": 1.0,
    "l1": -1.0,
    "stroke_recall": 1.0,
    "deleted_stroke_rate": -1.0,
    "false_stroke_rate": -1.0,
}


def source_means(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["source_id"])].append(row)
    result: defaultdict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (method, source), group in grouped.items():
        result[method][source] = {
            metric: float(np.mean([float(row[metric]) for row in group])) for metric in METRICS
        }
    return dict(result)


def bootstrap_ci(differences: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> list[float]:
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [.025, .975])]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired source-level validation significance report")
    parser.add_argument(
        "--records", type=Path,
        default=ROOT / "artifacts/validation_restoration/training_foreground_dev_v4_checkpoint_0000500/records.csv",
    )
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    with args.records.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = source_means(rows)
    if PROPOSED not in values:
        raise SystemExit(f"Missing {PROPOSED} records")

    rng = np.random.default_rng(56)
    comparisons = []
    for baseline in sorted(method for method in values if method != PROPOSED):
        sources = sorted(set(values[PROPOSED]) & set(values[baseline]))
        for metric, direction in METRICS.items():
            proposed = np.asarray([values[PROPOSED][source][metric] for source in sources])
            reference = np.asarray([values[baseline][source][metric] for source in sources])
            benefit = direction * (proposed - reference)
            statistic, p_value = wilcoxon(
                benefit, alternative="greater", zero_method="wilcox", method="approx"
            )
            comparisons.append({
                "baseline": baseline,
                "metric": metric,
                "direction": "higher" if direction > 0 else "lower",
                "source_groups": len(sources),
                "proposed_mean": float(proposed.mean()),
                "baseline_mean": float(reference.mean()),
                "mean_benefit": float(benefit.mean()),
                "benefit_95pct_bootstrap_ci": bootstrap_ci(benefit, rng),
                "wilcoxon_statistic": float(statistic),
                "p_value_one_sided": float(p_value),
            })
    adjusted = holm_adjust([row["p_value_one_sided"] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row["holm_adjusted_p"] = value
        row["significantly_better"] = bool(
            value < .05 and row["benefit_95pct_bootstrap_ci"][0] > 0
        )
    report = {
        "partition": args.partition,
        "formal_test_evidence": args.partition == "test",
        "unit_of_analysis": "source group (three degradation variants averaged per source)",
        "test": "one-sided paired Wilcoxon signed-rank with Holm family-wise correction",
        "interval": "10,000-draw paired source bootstrap CI for mean benefit",
        "records": str(args.records.resolve()),
        "all_baseline_metric_comparisons_significant": all(
            row["significantly_better"] for row in comparisons
        ),
        "comparisons": comparisons,
        "warning": "No post-test tuning is permitted." if args.partition == "test" else "Validation evidence supports selection only; confirm once on the locked held-out test.",
    }
    filename = "formal_test_significance.json" if args.partition == "test" else "validation_significance.json"
    output = ROOT / "artifacts/formal_run" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
