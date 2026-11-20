"""Figures for the RQ2/RQ3 re-analysis (20 November 2026).

Reads the aggregate report written by ``rq2_rq3_reanalysis_20261120.py`` and
produces two two-panel figures:

  fig1_reliability.png        (a) reliability diagram, composite score against
                              calibrated OCR confidence, primary (pre-agreement)
                              endpoint; (b) the number of fields in each
                              probability bin
  fig2_matched_coverage.png   (a) unsupported acceptance rate against coverage for
                              the proposed policy, Baseline E and Baseline F;
                              (b) the paired reduction against Baseline E with its
                              95% source-cluster bootstrap interval

Both figures answer a question the earlier single-panel versions left open. The
reliability curve was drawn through bins that hold 19 to 261 fields, so a reader
could not see that most of the mass sits in two bins. The matched-coverage figure
plotted point estimates only, so the interval that carries the paper's comparative
claim was not visible at all.

Usage:
    python scripts/rq2_rq3_figures_20261120.py --report <report.json> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ------------------------------------------------------------------ style
INK = "#272727"
GRID = "#DCDCDC"
NEUTRAL = "#767676"
C_PROPOSED = "#0F4D92"
C_BASELINE_E = "#B64342"
C_BASELINE_F = "#767676"
C_OCR = "#B64342"
C_COMPOSITE = "#0F4D92"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
})

WIDTH = 6.6  # inches; the manuscript places landscape figures at this width,
# so a 7 pt label here prints at 7 pt rather than being scaled up


def panel_label(ax, text):
    ax.text(-0.16, 1.03, text, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left", color=INK)


def parse_bin(label):
    """'0.0-0.1' -> (0.0, 0.1)."""
    lo, hi = label.split("-")
    return float(lo), float(hi)


def save(fig, out, stem, qa=False):
    """Export the figure, running the layout gates first when they are available.

    The manuscript takes the PNG. The PDF is written only on the QA path, where
    it feeds the rendered-text and collision audits; it is not a submission file.

    The alignment gate lives outside this repository. Point ``FIGURE_QA_DIR`` at
    the directory holding ``audit_panel_alignment.py`` to run it, or drop that
    file next to this script. The gate writes ``<stem>.alignment.json``; the
    copies shipped in this folder are its output, so a reader can confirm the
    gate ran without installing anything.
    """
    if qa:
        qa_dir = os.environ.get("FIGURE_QA_DIR")
        if qa_dir:
            sys.path.insert(0, qa_dir)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ImportError:
            raise SystemExit(
                "the alignment gate was requested but audit_panel_alignment.py "
                "could not be imported; set FIGURE_QA_DIR to its directory, or "
                "rerun without --qa to export the PNG and PDF only"
            )
        require_matplotlib_panel_alignment(
            fig, json_out=out / f"{stem}.alignment.json",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
        # The gate writes its JSON with the platform line ending. artifacts/**
        # is pinned -text in .gitattributes, so a CRLF file would carry a
        # different digest on Windows than on Linux. Normalise the bytes only;
        # the gate's content is untouched.
        gate = out / f"{stem}.alignment.json"
        gate.write_bytes(gate.read_bytes().replace(b"\r\n", b"\n"))
        fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=600)
    plt.close(fig)


def reliability(report, out, qa=False):
    pre = report["variants"]["pre"]["calibration"]
    comp = pre["reliability_composite"]
    ocr = pre["reliability_ocr"]

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(WIDTH, 4.15), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.0], "hspace": 0.12},
        layout="constrained",
    )

    # ---- panel a: reliability curves
    ax.plot([0, 1], [0, 1], linestyle=(0, (1.5, 2)), color=NEUTRAL,
            linewidth=0.9, zorder=1)
    ax.plot([b["mean_score"] for b in comp], [b["observed"] for b in comp],
            marker="o", color=C_COMPOSITE, linewidth=1.5, markersize=4.2,
            label="Composite field score", zorder=3)
    ax.plot([b["mean_score"] for b in ocr], [b["observed"] for b in ocr],
            marker="s", color=C_OCR, linewidth=1.5, markersize=4.0,
            label="Calibrated OCR confidence", zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Observed accuracy")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    # The legend sits above the axes, not inside them: with no frame it would
    # otherwise be crossed by the gridlines, which fails the collision audit.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=2,
              handlelength=1.3, borderaxespad=0.0, columnspacing=1.4)
    panel_label(ax, "a")

    # ---- panel b: bin occupancy
    for series, colour, offset, name in (
        (comp, C_COMPOSITE, -0.018, "composite"),
        (ocr, C_OCR, +0.018, "OCR"),
    ):
        centres = [(parse_bin(b["bin"])[0] + parse_bin(b["bin"])[1]) / 2 for b in series]
        counts = [b["n"] for b in series]
        axn.bar([c + offset for c in centres], counts, width=0.034,
                color=colour, linewidth=0, alpha=0.9, label=name, zorder=2)

    axn.set_ylim(0, 330)
    axn.set_yticks([0, 150, 300])
    axn.set_xlabel("Mean predicted probability")
    axn.set_ylabel("Fields per bin")
    axn.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
    panel_label(axn, "b")

    save(fig, out, "fig1_reliability", qa=qa)


def matched_coverage(report, out, qa=False):
    pts = report["variants"]["pre"]["matched_coverage"]
    x = [p["coverage"] * 100 for p in pts]
    p_rate = [p["uar_P"] * 100 for p in pts]
    e_rate = [p["uar_E"] * 100 for p in pts]
    f_rate = [p["uar_F"] * 100 for p in pts]

    # paired reduction against Baseline E, and its interval, in percentage points
    est = [e - p for e, p in zip(e_rate, p_rate)]
    lo = [p["uar_reduction_vs_E_ci95"][0] * 100 for p in pts]
    hi = [p["uar_reduction_vs_E_ci95"][1] * 100 for p in pts]
    excludes_zero = [(l > 0) or (h < 0) for l, h in zip(lo, hi)]

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(WIDTH, 4.25), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.25], "hspace": 0.12},
        layout="constrained",
    )

    # ---- panel a: absolute rates
    ax.plot(x, p_rate, marker="o", color=C_PROPOSED, linewidth=1.5, markersize=4.2,
            label="DSDNet-Receipt", zorder=4)
    ax.plot(x, e_rate, marker="s", color=C_BASELINE_E, linewidth=1.5, markersize=4.0,
            label="Baseline E", zorder=3)
    ax.plot(x, f_rate, marker="^", color=C_BASELINE_F, linewidth=1.5, markersize=4.2,
            label="Baseline F", zorder=3)

    ax.set_xlim(20, 80)
    ax.set_ylim(28, 68)
    ax.set_ylabel("Unsupported acceptance rate (%)")
    ax.set_yticks([30, 40, 50, 60])
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    # The legend sits above the axes, not inside them: with no frame it would
    # otherwise be crossed by the gridlines, which fails the collision audit.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=3,
              handlelength=1.3, borderaxespad=0.0, columnspacing=1.4)
    panel_label(ax, "a")

    # ---- panel b: paired reduction with its interval
    axd.axhline(0, color=NEUTRAL, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    for xi, ei, li, hi_, sig in zip(x, est, lo, hi, excludes_zero):
        axd.plot([xi, xi], [li, hi_], color=NEUTRAL, linewidth=1.0, zorder=2,
                 solid_capstyle="butt")
        axd.plot([xi - 0.6, xi + 0.6], [li, li], color=NEUTRAL, linewidth=0.8, zorder=2)
        axd.plot([xi - 0.6, xi + 0.6], [hi_, hi_], color=NEUTRAL, linewidth=0.8, zorder=2)
        axd.plot(xi, ei, marker="o", markersize=4.2, zorder=3,
                 color=C_PROPOSED if sig else "white",
                 markeredgecolor=C_PROPOSED, markeredgewidth=0.9)

    axd.set_ylim(-4, 29)
    axd.set_yticks([0, 10, 20])
    axd.set_ylabel("Reduction vs\nBaseline E (pp)")
    axd.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
    panel_label(axd, "b")

    axd.set_xticks(x)
    axd.set_xticklabels([f"{v:.1f}" for v in x])
    axd.set_xlabel("Coverage (%)")

    save(fig, out, "fig2_matched_coverage", qa=qa)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qa", action="store_true",
                        help="run the layout gate and also write PDF for auditing")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    reliability(report, out, qa=args.qa)
    matched_coverage(report, out, qa=args.qa)

    print("wrote", out / "fig1_reliability.png")
    print("wrote", out / "fig2_matched_coverage.png")


if __name__ == "__main__":
    main()
