"""GPT-5.1 L4 diagnostic: score distribution + per-template heatmap.

Two-panel figure:
  (a) Stacked bar of {0.0, 0.5, 1.0} fractions for L4.1 Troubleshooting and
      L4.2 Optimization.
  (b) Heatmap of mean GPT-5.1 score per (template, dataset) cell.

Output: output/figures/fig_l4_gpt51.pdf
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
from collections import Counter, defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

TIGER    = "#ff5a00"
FLICKER  = "#DC4B07"
STEEL    = "#878f92"
GUNMETAL = "#122128"

BRAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "brand_l4", ["#FFFFFF", "#FFE2CC", "#FFB888", TIGER, FLICKER])

DATASET_DISPLAY = {
    "aursad":      "AURSAD",
    "vorausad":    "VORAUSAD",
    "factorywave": "FactoryWave",
}
DATASET_ORDER = ["aursad", "vorausad", "factorywave"]

TEMPLATE_DISPLAY = {
    ("troubleshooting", 1): "L4.1 Troubleshooting",
    ("optimization",    2): "L4.2 Optimization",
}
TEMPLATE_ORDER = [("troubleshooting", 1), ("optimization", 2)]


def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.linewidth":    0.8,
        "axes.edgecolor":    GUNMETAL,
        "axes.labelcolor":   GUNMETAL,
        "xtick.color":       GUNMETAL,
        "ytick.color":       GUNMETAL,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def load_records(repo: pathlib.Path):
    qmap = {}
    for f in glob.glob(str(repo / "output/questions/level4/*.json")):
        d = json.load(open(f))
        qmap[d["id"]] = {
            "template_id":   d["template_id"],
            "template_type": d["template_type"],
            "dataset":       d["provenance"]["dataset"],
        }

    records = []
    replies = sorted(glob.glob(
        str(repo / "output/replies/level4/gpt-5_1-1/level4_*_answer.json")))
    for f in replies:
        a = json.load(open(f))
        pf = os.path.basename(a["prompt_file"])
        p = json.load(open(repo / "output/prompts/level4" / pf))
        qid = p["metadata"]["qa_pair_id"]
        if qid not in qmap:
            continue
        m = qmap[qid]
        records.append({
            **m,
            "judge_score": a["llm_judge_score"],
        })
    return records


def panel_a(ax, records):
    by_tt = defaultdict(list)
    for r in records:
        by_tt[(r["template_type"], r["template_id"])].append(r["judge_score"])

    score_vals = [0.0, 0.5, 1.0]
    colors     = [GUNMETAL, STEEL, TIGER]
    labels     = ["0.0  Wrong", "0.5  Partial", "1.0  Correct"]

    keys = [k for k in TEMPLATE_ORDER if k in by_tt]
    y = np.arange(len(keys))
    left = np.zeros(len(keys))

    for sv, color, label in zip(score_vals, colors, labels):
        vals = np.array([
            sum(s == sv for s in by_tt[k]) / len(by_tt[k]) * 100 for k in keys
        ])
        ax.barh(y, vals, left=left, color=color, label=label,
                height=0.55, edgecolor="white", linewidth=0.6)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 5:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{TEMPLATE_DISPLAY[k]}\n(n={len(by_tt[k])})" for k in keys
    ])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Proportion of items (%)")
    ax.set_title("(a) GPT-5.1 score distribution", fontweight="bold",
                 loc="left", color=GUNMETAL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color=STEEL)
    ax.set_axisbelow(True)


def panel_b(ax, records):
    cell_scores = defaultdict(list)
    for r in records:
        cell_scores[((r["template_type"], r["template_id"]),
                     r["dataset"])].append(r["judge_score"])

    rows = TEMPLATE_ORDER
    cols = DATASET_ORDER
    mat = np.full((len(rows), len(cols)), np.nan)
    counts = np.zeros((len(rows), len(cols)), dtype=int)
    for i, rk in enumerate(rows):
        for j, ck in enumerate(cols):
            scores = cell_scores.get((rk, ck), [])
            if scores:
                mat[i, j] = float(np.mean(scores))
                counts[i, j] = len(scores)

    im = ax.imshow(mat, cmap=BRAND_CMAP, vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([DATASET_DISPLAY[c] for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TEMPLATE_DISPLAY[r] for r in rows])
    ax.set_title("(b) Mean score by template × dataset",
                 fontweight="bold", loc="left", color=GUNMETAL)

    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(mat[i, j]):
                ax.text(j, i, ", ", ha="center", va="center",
                        color=STEEL, fontsize=11)
            else:
                col = "white" if mat[i, j] >= 0.45 else GUNMETAL
                ax.text(j, i, f"{mat[i, j]:.2f}\nn={counts[i, j]}",
                        ha="center", va="center",
                        color=col, fontsize=9, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8)
    cbar.set_label("Mean GPT-5.1 score", fontsize=9)


def main():
    repo = pathlib.Path(__file__).resolve().parents[1]
    set_style()
    records = load_records(repo)
    print(f"Loaded {len(records)} GPT-5.1 L4 records")
    print("By (template, dataset):", Counter(
        (r["template_type"], r["dataset"]) for r in records).most_common())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    panel_a(axes[0], records)
    panel_b(axes[1], records)
    fig.tight_layout()

    out = repo / "output/figures/fig_l4_gpt51.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")

    out_paper = repo / "docs/neurips_tex/figures/fig_l4_gpt51.pdf"
    fig.savefig(out_paper, bbox_inches="tight")
    print(f"saved {out_paper}")


if __name__ == "__main__":
    main()
