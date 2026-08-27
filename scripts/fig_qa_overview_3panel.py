"""3-panel overview of the released FactoryBench Q&A dataset.

Loads the released Q&A files (level_{1..4}) from
the public HuggingFace dataset and emits a single PDF with:

  (a) Dataset size per level.
  (b) Answer-format mix per level (four canonical buckets).
  (c) Sub-series length distribution per level.

Output: docs/neurips_tex/figures/fig_qa_overview_3panel.pdf
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(r"./.env",
            override=True)

REPO = pathlib.Path(__file__).resolve().parents[1]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so `src.evaluation` resolves when run from anywhere
HF_REPO = "FactoryBench/FactoryBench"
LEVELS = (1, 2, 3, 4)
SPLITS = ("train", "validation", "test")

OUT = REPO / "docs/neurips_tex/figures/fig_qa_overview_3panel.pdf"

TIGER    = "#ff5a00"
FLICKER  = "#DC4B07"
STEEL    = "#878f92"
GUNMETAL = "#122128"
LIGHT    = "#FFE2CC"

SPLIT_COLORS = {
    "train":      GUNMETAL,
    "validation": STEEL,
    "test":       TIGER,
}
LEVEL_COLORS = {
    1: "#3b82f6",
    2: TIGER,
    3: "#10b981",
    4: FLICKER,
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   11,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.linewidth":    0.9,
        "axes.edgecolor":    GUNMETAL,
        "axes.labelcolor":   GUNMETAL,
        "xtick.color":       GUNMETAL,
        "ytick.color":       GUNMETAL,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _classify_answer_style(rec: dict) -> str:
    """Answer format for a Q&A record, as the scorer sees it.

    Delegates to ``infer_answer_format`` (the same function
    ``score_prediction`` dispatches on) so the figure can never disagree with
    how items are actually graded. The previous local heuristic had two bugs:
    ranking answers (4-letter permutations like "BDAC") fell through to the
    single-select branch, hiding 3,192 ranking items, and tensor answers stored
    as strings were reported as free-form, putting free-form mass in L2 and L3
    where no free-form template exists.
    """
    from src.evaluation.run_foundry_eval import infer_answer_format
    return infer_answer_format(rec)


def load_all() -> pd.DataFrame:
    token = os.getenv("HF_TOKEN")
    rows = []
    for level in LEVELS:
        # one file per level; the release carries no train/validation/test split
        local = hf_hub_download(
            repo_id=HF_REPO, filename=f"factorybench_qa/level_{level}.jsonl",
            repo_type="dataset", token=token, force_download=False,
        )
        with open(local, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                ts = rec.get("context", {}).get("time_series", [])
                rows.append({
                    "level":        level,
                    "id":           rec.get("id"),
                    "answer_style": _classify_answer_style(rec),
                    "ts_rows":      len(ts) if isinstance(ts, list) else 0,
                })
    return pd.DataFrame(rows)


def main():
    set_style()
    df = load_all()
    print(f"loaded {len(df):,} Q&A items")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12.5))
    plt.subplots_adjust(hspace=0.45)

    # (a) size per level. The release is a single undivided pool, so this is
    # one bar per level rather than a stack.
    ax = axes[0]
    sizes = df.groupby("level")["id"].count()
    ax.bar(sizes.index, sizes.values, color=GUNMETAL,
           edgecolor="white", linewidth=0.6)
    for lvl, v in sizes.items():
        ax.text(lvl, v / 2, f"{int(v):,}", ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.set_xticks(list(sizes.index))
    ax.set_xticklabels([f"L{lvl}" for lvl in sizes.index])
    ax.set_ylabel("Samples")
    ax.set_title(f"(a) Dataset size per level  ({sizes.sum():,} samples total)")

    # (b) answer-format mix per level (four canonical buckets)
    STYLE_BUCKET = {
        "numerical":                     "Numerical",
        "tensor":                        "Tensor",
        "multiple_choice_single_select":  "MC single-select",
        "multiple_choice_multi_select":   "MC multi-select",
        "ranking":                       "Ranking",
        "free_form":                     "Free-form",
    }
    BUCKET_ORDER = [
        "Numerical",
        "Tensor",
        "MC single-select",
        "MC multi-select",
        "Ranking",
        "Free-form",
    ]
    BUCKET_COLOR = {
        "Numerical":        TIGER,
        "Tensor":           "#f0a94b",
        "MC single-select": "#3b82f6",
        "MC multi-select":  FLICKER,
        "Ranking":          "#7c5cbf",
        "Free-form":        GUNMETAL,
    }

    ax = axes[1]
    mix = df.copy()
    mix["bucket"] = mix["answer_style"].map(STYLE_BUCKET).fillna("Other")
    afmt = (
        mix.pivot_table(index="bucket", columns="level",
                        values="id", aggfunc="count")
        .fillna(0).astype(int)
    )
    afmt = afmt.reindex(index=[b for b in BUCKET_ORDER if b in afmt.index])
    proportions = afmt.div(afmt.sum(axis=0), axis=1).fillna(0)
    bottom = np.zeros(proportions.shape[1])
    for bucket in proportions.index:
        vals = proportions.loc[bucket].values
        color = BUCKET_COLOR.get(bucket, STEEL)
        ax.bar([f"L{lvl}" for lvl in proportions.columns], vals, bottom=bottom,
               color=color, label=bucket, edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            if v >= 0.06:
                ax.text(j, bottom[j] + v / 2, f"{int(round(v*100))}%",
                        ha="center", va="center", color="white",
                        fontsize=11, fontweight="bold")
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(p*100)}%" for p in np.linspace(0, 1, 6)])
    ax.legend(title="Answer format", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.set_title("(b) Answer-format mix")

    # (c) sub-series length distribution
    ax = axes[2]
    for lvl in LEVELS:
        sub = df[df.level == lvl]["ts_rows"]
        sub = sub[sub > 0]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=30, alpha=0.55, color=LEVEL_COLORS[lvl],
                label=f"L{lvl}  median={int(sub.median())}",
                edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Time-series rows per sample")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False)
    ax.set_title("(c) Sub-series length distribution")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
