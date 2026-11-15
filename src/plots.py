"""Figures for the README. All saved as PNG into figures/."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette (validated categorical slots + neutral inks)
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")


def _save(fig, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def tdt_plot(results: pd.DataFrame, threshold: float, out: pathlib.Path) -> None:
    y = -np.log10(results["p_value"])
    sig = results["significant_bonferroni"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.vlines(results["position"], 0, y, color=BLUE_LIGHT, linewidth=1)
    ax.scatter(results["position"][~sig], y[~sig], s=22, color=BLUE_LIGHT, zorder=3,
               edgecolor=SURFACE, linewidth=1)
    ax.scatter(results["position"][sig], y[sig], s=46, color=BLUE, zorder=4,
               edgecolor=SURFACE, linewidth=1.5)
    t = -np.log10(threshold)
    ax.axhline(t, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1, t + 0.12, "Bonferroni  p < 0.05 / 103", ha="left", va="bottom",
            color=INK_2, fontsize=8.5)
    top = results.sort_values("p_value").head(3).sort_values("position")
    placements = [((-8, 2), "right"), ((0, 9), "center"), ((8, 2), "left")]
    for (_, r), (offset, ha) in zip(top.iterrows(), placements):
        ax.annotate(f"{r.snp}  OR {r.transmission_or:.1f}", (r.position, -np.log10(r.p_value)),
                    xytext=offset, textcoords="offset points", ha=ha, fontsize=8.5, color=INK)
    ax.set_xlim(0, 104)
    ax.set_ylim(0, max(y.max() + 0.8, t + 1))
    ax.set_xlabel("SNP, in order along the IBD5 region (chromosome 5q31)")
    ax.set_ylabel("−log10(p)")
    ax.set_title("Transmission test finds an association signal in IBD5", pad=22)
    _subtitle(ax, f"{int(sig.sum())} of 103 SNPs pass the multiple-testing threshold "
                  "(transmission disequilibrium test, 129 trios)")
    _save(fig, out)


def roc_plot(model: dict, out: pathlib.Path) -> None:
    fpr, tpr = model["_roc"]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2)
    ax.text(0.55, 0.2, f"AUC {model['auc_mean']:.2f} ± {model['auc_sd']:.2f}",
            fontsize=10, color=INK, fontweight="bold")
    ax.text(0.62, 0.55, "chance", fontsize=8.5, color=INK_2, rotation=40)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Case vs pseudo-control", pad=22)
    _subtitle(ax, "Out-of-fold predictions, families kept together")
    _save(fig, out)
