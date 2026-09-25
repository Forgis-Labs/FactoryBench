"""Render Figure 2(a) from an explicit matrix.

Why this exists
---------------
`generate_figures.py` builds the heatmap from `output/replies/`. Those per-item
replies no longer exist, so the published `fig_main_heatmap.pdf` could not be
regenerated after the chance-correction fix -- and it had been drawn by the old
script, which floored the correction at zero and hardcoded E=0.25 for every
single-select item. The result was a figure 2-3 points above the body text on
every L1-L3 cell, with no negative values at all while the text reported three.

This module takes the matrix as data so the figure can be rebuilt from the
values the paper actually reports, without the replies. Every number below is
quoted verbatim from Section 5.2; the L4 column is uncorrected and already
agreed with the old figure.

Two cells are NOT recoverable: Qwen3-4B on L2 and L3. The paper never states
them, and a clipped aggregate cannot be converted to a signed one without the
per-item chance levels that lived in the replies. Interpolating from the other
rows' offsets would be wrong in a predictable direction: clipping only bites on
below-chance items, Qwen3-4B has the most of them, so its correction is larger
than the rows we can measure. They are carried over from the old figure and
marked with a dagger, not silently patched.

matplotlib only -- the original used seaborn, which is not a dependency of this
repo's environment.

Usage:
    python scripts/render_main_heatmap.py --out output/figures/fig_main_heatmap.pdf
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

TIGER = "#ff5a00"
FLICKER = "#DC4B07"
GUNMETAL = "#122128"
BRAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "brand", ["#FFFFFF", "#FFD6B8", TIGER, FLICKER, GUNMETAL])

MODELS = [
    "Claude Sonnet 4.6", "GPT-5.1", "Mistral Large 3", "Qwen3-235B",
    "DeepSeek V3.2", "Simple Baseline", "Qwen3-4B",
]
LEVELS = [("Level 1", 572), ("Level 2", 1430), ("Level 3", 712), ("Level 4", 286)]
NAN = float("nan")

# Signed chance-corrected (%), verbatim from Section 5.2. L4 is uncorrected.
VALUES = {
    "Claude Sonnet 4.6": [11.8, 14.6, 45.7, 15.9],
    "GPT-5.1":           [3.2,   6.9, 27.5, 21.5],
    "Mistral Large 3":   [5.9,   5.3, 19.7, 15.6],
    "Qwen3-235B":        [3.6,   4.9, 20.6, 19.9],
    "DeepSeek V3.2":     [-0.3,  5.8, 17.1, 17.8],
    "Simple Baseline":   [4.3,  -0.2,  6.2,  NAN],
    "Qwen3-4B":          [-1.8,  2.3,  9.6,  5.6],
}
CARRIED_OVER = {("Qwen3-4B", 1), ("Qwen3-4B", 2)}

VMIN, VMAX = -25.0, 100.0


def set_style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.linewidth": 0.8, "axes.edgecolor": GUNMETAL,
        "axes.labelcolor": GUNMETAL, "xtick.color": GUNMETAL,
        "ytick.color": GUNMETAL, "text.color": GUNMETAL,
        "axes.grid": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
    })


def render(out: pathlib.Path) -> None:
    set_style()
    matrix = np.array([VALUES[m] for m in MODELS], dtype=float)
    masked = np.ma.masked_invalid(matrix)
    cmap = BRAND_CMAP.copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(masked, cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

    # white gridlines between cells, matching the original
    ax.set_xticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
    for i, m in enumerate(MODELS):
        for j in range(len(LEVELS)):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            label = "%.1f%%" % v
            if (m, j) in CARRIED_OVER:
                label += "†"
            r, g, b, _ = cmap(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if lum < 0.5 else GUNMETAL)

    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(["%s\n(N=%s)" % (n, format(c, ",")) for n, c in LEVELS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    ax.set_title("FactoryBench Results: Model × Level (Signed Chance-Corrected)",
                 fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Chance-Corrected Accuracy, signed (%)")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)
    print("cells carried over from the pre-fix figure (marked †): %d"
          % len(CARRIED_OVER))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("output/figures/fig_main_heatmap.pdf"))
    render(ap.parse_args().out)
