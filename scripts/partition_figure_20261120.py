#!/usr/bin/env python3
"""Figure 1 of the IJRISS manuscript: source-group partitioning.

The diagram is drawn at the printed width (6.6 in) so that every label is set
at its final point size and no scaling shrinks the type. Content is fixed by
the manuscript's partition table: a 70/15/15 split by source group, a
development panel whose labels are available to the builder, and a read-only
held-out branch that no design decision touches.

Usage:
    python scripts/ijriss_partition_figure_20261120.py --out <figure directory>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle  # noqa: E402

INK = "#1b1b1b"
LINE = "#333333"
PANEL = "#ededed"
FILL = "#ffffff"
GREY_FILL = "#e4e4e4"
FS_BOX = 7.0
FS_SMALL = 6.6
FS_HEAD = 7.2

FIG_W, FIG_H = 6.6, 4.05


def box(ax, x, y, w, h, lines, weight="bold", fill=FILL, lw=1.0, fs=FS_BOX):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=LINE,
                           linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center",
            fontsize=fs, color=INK, zorder=3, linespacing=1.35,
            fontweight=weight)


def arrow(ax, start, end, style="-|>", dashed=False, lw=1.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=LINE, zorder=4,
        linestyle=(0, (3, 2.2)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS_BOX})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 63)
    ax.axis("off")

    # ---- development panel
    ax.add_patch(Rectangle((22.0, 24.0), 76.5, 38.0, facecolor=PANEL,
                           edgecolor="#8f8f8f", linewidth=0.9, zorder=1))
    ax.text(23.6, 59.9, "Development partitions (labels available to development)",
            ha="left", va="center", fontsize=FS_HEAD, fontweight="bold", color=INK)

    # ---- grouped source pool
    ax.add_patch(Rectangle((1.5, 33.0), 13.5, 15.0, facecolor=GREY_FILL,
                           edgecolor=LINE, linewidth=1.0, zorder=2))
    ax.add_patch(Ellipse((8.25, 48.0), 13.5, 3.4, facecolor=GREY_FILL,
                         edgecolor=LINE, linewidth=1.0, zorder=3))
    ax.text(8.25, 40.6, "Grouped\nreceipt sources\n(source-group IDs)",
            ha="center", va="center", fontsize=FS_SMALL, color=INK,
            linespacing=1.4, zorder=4)

    # ---- split table
    tx, ty, tw, th, rows = 26.5, 31.0, 11.5, 20.5, 6
    ax.add_patch(Rectangle((tx, ty), tw, th, facecolor=FILL, edgecolor=LINE,
                           linewidth=1.0, zorder=2))
    for k in range(1, rows):
        yy = ty + th * k / rows
        ax.plot([tx, tx + tw], [yy, yy], color=LINE, linewidth=0.7, zorder=3)
    for k in (1, 2):
        xx = tx + tw * k / 3
        ax.plot([xx, xx], [ty, ty + th], color=LINE, linewidth=0.7, zorder=3)
    ax.text(tx + tw / 2, ty - 2.0, "70 : 15 : 15 split\nby source group",
            ha="center", va="top", fontsize=FS_SMALL, color="#3a3a3a",
            linespacing=1.35)

    arrow(ax, (15.0, 40.5), (tx, 40.5))

    # ---- development boxes
    box(ax, 41.5, 46.0, 15.0, 9.0, ["Training set", "(70%)"])
    box(ax, 41.5, 31.0, 15.0, 9.0, ["Validation set", "(15%)"])
    box(ax, 60.5, 45.5, 17.0, 11.5, ["DSDNet training", "(model builder)"], lw=2.0)
    box(ax, 60.5, 29.5, 17.0, 13.0,
        ["Checkpoint,", "calibration and", "threshold selection"], weight="normal")
    box(ax, 81.5, 38.0, 15.0, 12.0,
        ["Validation", "scores", "(AUC, ECE,", "UAR)"], weight="normal")

    arrow(ax, (38.0, 50.5), (41.5, 50.5))
    arrow(ax, (38.0, 35.5), (41.5, 35.5))
    arrow(ax, (56.5, 50.5), (60.5, 50.5))
    arrow(ax, (56.5, 35.5), (60.5, 35.5))
    arrow(ax, (69.0, 45.5), (69.0, 42.5))
    ax.text(70.0, 43.6, "candidate\ncheckpoints", ha="left", va="top",
            fontsize=FS_SMALL, color="#3a3a3a", style="italic", linespacing=1.3)
    arrow(ax, (77.5, 35.5), (81.5, 41.0))
    arrow(ax, (89.0, 50.0), (77.5, 51.0), dashed=True)
    ax.text(91.0, 53.2, "evaluate\nand select", ha="center", va="bottom",
            fontsize=FS_SMALL, color="#3a3a3a", style="italic", linespacing=1.3)

    # ---- held-out branch
    box(ax, 41.5, 8.5, 15.5, 9.0, ["Held-out test set", "(15%, read-only)"], lw=2.0)
    box(ax, 60.5, 7.5, 17.0, 10.0, ["Locked final model", "and configuration"], lw=2.0)
    box(ax, 82.0, 8.5, 14.0, 9.0, ["Final locked", "evaluation", "(run once)"])

    ax.plot([8.25, 8.25, 41.5], [33.0, 13.0, 13.0], color=LINE, linewidth=1.0, zorder=3)
    arrow(ax, (8.25, 13.0), (41.5, 13.0))
    arrow(ax, (69.0, 29.5), (69.0, 17.5))
    ax.text(69.8, 23.5, "frozen\nbefore testing", ha="left", va="center",
            fontsize=FS_SMALL, color="#3a3a3a", style="italic", linespacing=1.3)
    arrow(ax, (57.0, 13.0), (60.5, 13.0))
    arrow(ax, (77.5, 12.5), (82.0, 12.5))

    ax.text(50.0, 2.4, "Test labels never enter training, validation, calibration "
            "or threshold selection.", ha="center", va="center",
            fontsize=FS_SMALL, color="#3a3a3a", style="italic")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(out / "fig_partitioning.png", dpi=600)
    plt.close(fig)
    print(f"wrote {out / 'fig_partitioning.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
