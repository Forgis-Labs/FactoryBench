"""Distribution figures for the released FactoryBench Q&A dataset on HuggingFace.

Reads the 12 split files (level_{1..4}/{train,validation,test}.jsonl) from
FactoryBench/FactoryBench, then renders two complementary figures:

  (1) fig_qa_source_per_level.pdf  --  stacked bars of source-dataset mix
                                       across levels, plus the train/val/test
                                       split share inside each level.
  (2) fig_qa_fault_distribution.pdf -- distribution of injected faults
                                       (root_cause field) across the released
                                       Q&A items.

Output: docs/neurips_tex/figures/.
"""
from __future__ import annotations

import json
import os
import pathlib
from collections import Counter, defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(r"./.env",
            override=True)

REPO = "FactoryBench/FactoryBench"
LEVELS = (1, 2, 3, 4)
SPLITS = ("train", "validation", "test")
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs/neurips_tex/figures"

TIGER    = "#ff5a00"
FLICKER  = "#DC4B07"
STEEL    = "#878f92"
GUNMETAL = "#122128"


def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
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


def load_records():
    token = os.getenv("HF_TOKEN")
    records = []
    for level in LEVELS:
        for split in SPLITS:
            path = f"factorybench_qa/level_{level}/{split}.jsonl"
            local = hf_hub_download(
                repo_id=REPO, filename=path, repo_type="dataset",
                token=token, force_download=False,
            )
            with open(local, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    records.append({
                        "level":          level,
                        "split":          split,
                        "template_id":    rec.get("template_id"),
                        "template_type":  rec.get("template_type"),
                        "root_cause":     rec.get("root_cause"),
                        "dataset":        (rec.get("provenance") or {}).get("dataset"),
                    })
    return records


# ── Figure 1: per-level dataset origin × split ─────────────────────────
def fig_source_per_level(records, out_dir: pathlib.Path) -> None:
    DS_DISPLAY = {
        "factorywave":      "FactoryWave (UR3)",
        "factorywave_kuka": "FactoryWave (KUKA)",
        "aursad":           "AURSAD",
        "vorausad":         "voraus-AD",
        "":                 "(unspecified)",
        None:               "(unspecified)",
    }
    DS_ORDER = ["factorywave", "factorywave_kuka", "aursad", "vorausad"]
    DS_COLOR = {
        "factorywave":      TIGER,
        "factorywave_kuka": FLICKER,
        "aursad":           "#1f77b4",
        "vorausad":         "#2ca02c",
    }

    by_level_ds = defaultdict(Counter)
    by_level_split = defaultdict(Counter)
    for r in records:
        ds = r["dataset"] or ""
        by_level_ds[r["level"]][ds] += 1
        by_level_split[r["level"]][r["split"]] += 1

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.25},
    )

    levels = list(LEVELS)
    x = np.arange(len(levels))

    # ── Panel (a): source-dataset mix per level (absolute counts) ──────
    bottoms = np.zeros(len(levels))
    for ds in DS_ORDER:
        vals = np.array([by_level_ds[lv].get(ds, 0) for lv in levels])
        if vals.sum() == 0:
            continue
        ax_a.bar(x, vals, bottom=bottoms, color=DS_COLOR[ds],
                 label=DS_DISPLAY[ds], width=0.66, edgecolor="white", linewidth=0.6)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 600:
                ax_a.text(xi, b + v / 2, f"{v:,}", ha="center", va="center",
                          fontsize=8.5, color="white", fontweight="bold")
        bottoms += vals

    totals = bottoms.astype(int)
    for xi, t in zip(x, totals):
        ax_a.text(xi, t + max(totals) * 0.015, f"{t:,}", ha="center", va="bottom",
                  fontsize=9, color=GUNMETAL, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"L{lv}" for lv in levels])
    ax_a.set_ylabel("Q&A items")
    ax_a.set_title("(a) Dataset source per level", fontweight="bold",
                   loc="left", color=GUNMETAL)
    ax_a.legend(loc="upper right", frameon=False)
    ax_a.yaxis.grid(True, linestyle="--", alpha=0.35, color=STEEL)
    ax_a.set_axisbelow(True)
    ax_a.set_ylim(0, max(totals) * 1.10)

    # ── Panel (b): train/val/test share per level (percentages) ────────
    SPLIT_COLOR = {"train": GUNMETAL, "validation": STEEL, "test": TIGER}
    SPLIT_LABEL = {"train": "Train", "validation": "Validation", "test": "Test"}
    bottoms = np.zeros(len(levels))
    totals_b = np.array([sum(by_level_split[lv].values()) for lv in levels],
                        dtype=float)
    for sp in SPLITS:
        vals = np.array(
            [by_level_split[lv].get(sp, 0) / totals_b[i] * 100
             for i, lv in enumerate(levels)]
        )
        ax_b.bar(x, vals, bottom=bottoms, color=SPLIT_COLOR[sp],
                 label=SPLIT_LABEL[sp], width=0.66, edgecolor="white", linewidth=0.6)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 4:
                ax_b.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                          fontsize=9, color="white", fontweight="bold")
        bottoms += vals

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"L{lv}" for lv in levels])
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("Share (%)")
    ax_b.set_title("(b) Train/validation/test split per level",
                   fontweight="bold", loc="left", color=GUNMETAL)
    ax_b.legend(loc="lower right", frameon=False)
    ax_b.yaxis.grid(True, linestyle="--", alpha=0.35, color=STEEL)
    ax_b.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / "fig_qa_source_per_level.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ── Figure 2: fault / root-cause distribution ──────────────────────────
def fig_fault_distribution(records, out_dir: pathlib.Path) -> None:
    rc_counts = Counter()
    for r in records:
        rc = r.get("root_cause")
        if rc is None or rc == "":
            continue
        rc_counts[rc] += 1

    if not rc_counts:
        print("no root_cause field populated; skipping fault figure")
        return

    items = rc_counts.most_common()
    nominal = next((v for k, v in items if k == "normal"), 0)
    fault_items = [(k, v) for k, v in items if k != "normal"]
    fault_items.sort(key=lambda kv: -kv[1])

    labels = [k.replace("_", " ") for k, _ in fault_items]
    values = [v for _, v in fault_items]

    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.32 * len(fault_items) + 1)))
    y = np.arange(len(fault_items))
    ax.barh(y, values, color=TIGER, height=0.7, edgecolor="white", linewidth=0.6)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.005, yi, f"{v:,}", va="center", ha="left",
                fontsize=9, color=GUNMETAL)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Q&A items")
    ax.set_title(
        f"Fault-type distribution across released Q&A items "
        f"(nominal/no-fault items not shown: {nominal:,})",
        fontweight="bold", loc="left", color=GUNMETAL,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color=STEEL)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.10)

    fig.tight_layout()
    out = out_dir / "fig_qa_fault_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print(f"loaded {len(records):,} Q&A items from {REPO}")

    fig_source_per_level(records, OUT_DIR)
    fig_fault_distribution(records, OUT_DIR)


if __name__ == "__main__":
    main()
