#!/usr/bin/env python3
"""Regenerate Figure 3 and Figure 4 of the IJRISS manuscript.

Both figures read the aggregate report written by the deposited re-analysis
script, so every plotted value traces to an archived number.

Figure 3  reliability diagram, composite field score against calibrated OCR
          confidence, primary (pre-agreement) endpoint. Panel (a) curves,
          panel (b) field count per probability bin.
Figure 4  unsupported acceptance rate against coverage. Panel (a) absolute
          rates on a full 0 to 100 percent axis, panel (b) the paired
          reduction against Baseline E with its source-cluster interval.
          Coverage above the support-gate ceiling of 380/570 = 66.7 percent
          is shaded and labelled, because a gated policy cannot reach it.

Usage:
    python scripts/ijriss_rq_figures_20261120.py \
        --report <report.json> --out <figure directory>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

INK = "#1b1b1b"
GRID = "#d8d8d8"
C_PRO = "#1f5fa8"
C_E = "#b3392b"
C_F = "#6b6b6b"
SHADE = "#e9e9e9"

# The gate can accept at most the fields that carry a support value.
GATE_CEILING = 380 / 570 * 100.0


def _centre(label: str) -> float:
    lo, hi = label.split("-")
    return (float(lo) + float(hi)) / 2.0


def reliability(report: dict, out: Path) -> None:
    pre = report["variants"]["pre"]["calibration"]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(6.6, 4.9), sharex=True,
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12})

    for key, label, colour, marker, offset in (
            ("reliability_composite", "Composite field score", C_PRO, "o", -0.021),
            ("reliability_ocr", "Calibrated OCR confidence", C_E, "s", 0.021)):
        rows = pre[key]
        ax.plot([0, 1], [0, 1], linestyle=":", color="#9a9a9a", linewidth=1.1, zorder=1)
        ax.plot([r["mean_score"] for r in rows], [r["observed"] for r in rows],
                marker=marker, color=colour, linewidth=1.9, markersize=5.5,
                label=f"{label} (n={sum(r['n'] for r in rows)})", zorder=3)

        bx.bar([_centre(r["bin"]) + offset for r in rows], [r["n"] for r in rows],
               width=0.040, color=colour, label=label, zorder=3)

    ax.set_ylabel("Observed accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, handlelength=1.6)
    ax.text(0.985, 0.03, f"ECE composite {pre['composite_ece']:.3f}"
            f"   OCR {pre['ocr_ece']:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=INK)

    bx.set_ylabel("Fields per bin")
    bx.set_xlabel("Mean predicted probability")
    bx.set_ylim(0, 340)
    bx.grid(True, axis="y", color=GRID, linewidth=0.6)
    bx.set_axisbelow(True)

    for a in (ax, bx):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=INK, labelsize=8.5)

    ax.text(-0.10, 1.03, "a", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left", color=INK)
    bx.text(-0.10, 1.03, "b", transform=bx.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left", color=INK)

    fig.savefig(out / "fig_reliability.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def matched_coverage(report: dict, out: Path) -> None:
    points = report["variants"]["pre"]["matched_coverage"]
    xs = [p["coverage"] * 100 for p in points]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(6.6, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.12})

    for a in (ax, bx):
        a.axvspan(GATE_CEILING, 80, color=SHADE, zorder=0, linewidth=0)
        a.axvline(GATE_CEILING, color="#8a8a8a", linewidth=0.9,
                  linestyle=(0, (4, 3)), zorder=1)

    for key, label, colour, marker in (("P", "DSDNet-Receipt", C_PRO, "o"),
                                       ("E", "Baseline E", C_E, "s"),
                                       ("F", "Baseline F", C_F, "^")):
        ax.plot(xs, [p[f"uar_{key}"] * 100 for p in points], marker=marker,
                color=colour, linewidth=1.9, markersize=6, label=label, zorder=3)

    ax.set_ylabel("Unsupported acceptance rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, handlelength=1.6)
    ax.text((GATE_CEILING + 80) / 2, 86, "outside the\nsupport gate",
            ha="center", va="center", fontsize=8.0, color="#4a4a4a")

    lo = [p["uar_reduction_vs_E_ci95"][0] * 100 for p in points]
    hi = [p["uar_reduction_vs_E_ci95"][1] * 100 for p in points]
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    for x, m, l, h in zip(xs, mid, lo, hi):
        bx.plot([x, x], [l, h], color=C_PRO, linewidth=1.1, zorder=2)
        bx.plot([x - 0.55, x + 0.55], [l, l], color=C_PRO, linewidth=1.1, zorder=2)
        bx.plot([x - 0.55, x + 0.55], [h, h], color=C_PRO, linewidth=1.1, zorder=2)
    filled = [p["uar_reduction_vs_E_ci95"][0] > 0 for p in points]
    bx.plot(xs, mid, linestyle="none")
    bx.plot([x for x, f in zip(xs, filled) if f], [m for m, f in zip(mid, filled) if f],
            "o", color=C_PRO, markersize=6, zorder=4)
    bx.plot([x for x, f in zip(xs, filled) if not f], [m for m, f in zip(mid, filled) if not f],
            "o", markerfacecolor="white", markeredgecolor=C_PRO, markeredgewidth=1.3,
            markersize=6, zorder=4)
    bx.axhline(0, color="#8a8a8a", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)

    bx.set_ylabel("Reduction vs\nBaseline E (pp)")
    bx.set_xlabel("Coverage (%)")
    bx.set_xlim(20, 80)
    bx.set_ylim(-8, 30)
    bx.yaxis.set_major_locator(MultipleLocator(10))

    for a in (ax, bx):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        a.tick_params(colors=INK, labelsize=8.5)

    ax.text(-0.10, 1.03, "a", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left", color=INK)
    bx.text(-0.10, 1.03, "b", transform=bx.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left", color=INK)

    fig.savefig(out / "fig_matched_coverage.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.8,
        "figure.dpi": 300,
    })

    reliability(report, out)
    matched_coverage(report, out)
    print(f"wrote {out / 'fig_reliability.png'}")
    print(f"wrote {out / 'fig_matched_coverage.png'}")
    print(f"gate ceiling drawn at {GATE_CEILING:.1f}% coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
