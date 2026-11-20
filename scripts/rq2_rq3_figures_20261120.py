"""Figures for the RQ2/RQ3 re-analysis (20 November 2026).

Reads the aggregate report written by ``rq2_rq3_reanalysis_20261120.py`` and
produces two figures:

  fig1_reliability.png   reliability diagram, composite score against calibrated
                         OCR confidence, primary (pre-agreement) endpoint
  fig2_matched_coverage.png   unsupported acceptance rate against coverage for
                         the proposed policy, Baseline E and Baseline F

Usage:
    python scripts/rq2_rq3_figures_20261120.py --report <report.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1b1b"
GRID = "#d8d8d8"
C_PRO = "#1f5fa8"
C_E = "#b3392b"
C_F = "#6b6b6b"


def reliability(ax, bins, label, colour, marker):
    xs = [b["mean_score"] for b in bins]
    ys = [b["observed"] for b in bins]
    ns = [b["n"] for b in bins]
    ax.plot([0, 1], [0, 1], linestyle=":", color="#9a9a9a", linewidth=1.0, zorder=1)
    ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.8, markersize=5,
            label=f"{label} (n={sum(ns)})", zorder=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.8, "figure.dpi": 300,
    })

    # ---- figure 1: reliability
    pre = report["variants"]["pre"]["calibration"]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    reliability(ax, pre["reliability_composite"], "Composite score", C_PRO, "o")
    reliability(ax, pre["reliability_ocr"], "OCR confidence", C_E, "s")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed accuracy")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.set_title(f"ECE composite {pre['composite_ece']:.3f}  |  OCR {pre['ocr_ece']:.3f}",
                 fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_reliability.png", bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: matched coverage
    points = report["variants"]["pre"]["matched_coverage"]
    xs = [p["coverage"] * 100 for p in points]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    # The legend carries the method name used in the manuscript. A generic
    # "Proposed" label forces the reader to map figure to method, so the label
    # is the pipeline name itself.
    for key, label, colour, marker in (("P", "DSDNet-Receipt", C_PRO, "o"),
                                       ("E", "Baseline E", C_E, "s"),
                                       ("F", "Baseline F", C_F, "^")):
        ys = [p[f"uar_{key}"] * 100 for p in points]
        ax.plot(xs, ys, marker=marker, color=colour, linewidth=1.8, markersize=5, label=label)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Unsupported acceptance rate (%)")
    ax.set_xlim(20, 80)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "fig2_matched_coverage.png", bbox_inches="tight")
    plt.close(fig)

    print("wrote", out / "fig1_reliability.png")
    print("wrote", out / "fig2_matched_coverage.png")


if __name__ == "__main__":
    main()
