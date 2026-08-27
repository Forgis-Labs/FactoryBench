#!/usr/bin/env python3
"""
Analyze Q&A pair distributions for FactoryBench and save publication figures.

Outputs (saved to --figures-dir, default: figures/):
  f2_level_dist.png, Question count by level
  f2_template_dist.png, Template type distribution per level
  f2_answer_balance.png, Per-position T/F balance across levels
  f2_provenance_dist.png, Dataset provenance distribution
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

LEVEL_NAMES = {
    1: "L1: State",
    2: "L2: Intervention",
    3: "L3: Counterfactual",
    4: "L4: Decision Making",
}

# Neutral, publication-friendly palette
LEVEL_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
GRAY = "#888888"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Data loading ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(root: Path) -> List[Dict[str, Any]]:
    questions = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and "level" in payload and "answer" in payload:
            questions.append(payload)
    return questions

# ── Shared data helper ────────────────────────────────────────────────────────

def _build_level_provenance(questions: List[Dict]):
    by_level: Dict[int, Counter] = defaultdict(Counter)
    for q in questions:
        prov = q.get("provenance") or {}
        ds = prov.get("dataset", "unknown") if isinstance(prov, dict) else "unknown"
        by_level[q.get("level")][ds] += 1
    levels = sorted(by_level)
    all_datasets = sorted({ds for c in by_level.values() for ds in c})
    # Consistent dataset colour palette
    cmap = plt.cm.get_cmap("tab10", max(len(all_datasets), 1))
    ds_colors = {ds: cmap(i) for i, ds in enumerate(all_datasets)}
    total = sum(sum(c.values()) for c in by_level.values())
    return by_level, levels, all_datasets, ds_colors, total


# ── Option A: Stacked horizontal bar ─────────────────────────────────────────

def fig_option_a(questions: List[Dict], out: Path) -> None:
    """Stacked horizontal bars, one row per level, segments coloured by dataset."""
    by_level, levels, all_datasets, ds_colors, total = _build_level_provenance(questions)
    levels_rev = levels[::-1]
    level_labels = [LEVEL_NAMES.get(l, f"L{l}") for l in levels_rev]

    fig, ax = plt.subplots(figsize=(6.5, 0.9 * len(levels) + 1.4))
    lefts = np.zeros(len(levels))
    for ds in all_datasets:
        vals = np.array([by_level[l].get(ds, 0) for l in levels_rev], dtype=float)
        bars = ax.barh(level_labels, vals, left=lefts, color=ds_colors[ds],
                       height=0.52, label=ds, edgecolor="white", linewidth=0.6)
        for bar, val, left in zip(bars, vals, lefts):
            if val > total * 0.03:
                ax.text(left + val / 2, bar.get_y() + bar.get_height() / 2,
                        f"{int(val):,}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        lefts += vals

    for i, level in enumerate(levels_rev):
        n = sum(by_level[level].values())
        ax.text(lefts[i] + total * 0.01, i,
                f"n={n:,}  ({100*n/total:.1f}%)",
                va="center", fontsize=8, color="#444444")

    ax.set_xlabel("Q&A pairs")
    ax.set_title("Distribution by level and source dataset", pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, max(lefts) * 1.32)
    ax.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False, fontsize=8, title_fontsize=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")



# ── Option C: Bubble / dot matrix ────────────────────────────────────────────

def fig_option_c(questions: List[Dict], out: Path) -> None:
    """Bubble matrix, rows = datasets, cols = levels, bubble area ∝ count."""
    by_level, levels, all_datasets, ds_colors, total = _build_level_provenance(questions)

    fig, ax = plt.subplots(figsize=(max(5, len(levels) * 1.5), max(3, len(all_datasets) * 1.1) + 1))

    max_count = max(by_level[l].get(ds, 0) for l in levels for ds in all_datasets)
    MAX_AREA = 2800

    for xi, level in enumerate(levels):
        for yi, ds in enumerate(all_datasets):
            n = by_level[level].get(ds, 0)
            if n == 0:
                continue
            size = MAX_AREA * (n / max_count)
            ax.scatter(xi, yi, s=size, color=ds_colors[ds], alpha=0.82, zorder=3,
                       edgecolors="white", linewidths=0.8)
            ax.text(xi, yi, f"{n:,}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([LEVEL_NAMES.get(l, f"L{l}") for l in levels], fontsize=9)
    ax.set_yticks(range(len(all_datasets)))
    ax.set_yticklabels(all_datasets, fontsize=9)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(-0.6, len(all_datasets) - 0.4)
    ax.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("Q&A pairs by level and source dataset", pad=10)

    handles = [
        plt.scatter([], [], s=60, color=ds_colors[ds], alpha=0.82,
                    edgecolors="white", linewidths=0.8, label=ds)
        for ds in all_datasets
    ]
    ax.legend(handles=handles, title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")



# ── Figure 3: Per-position T/F balance ────────────────────────────────────────

def fig_answer_balance(questions: List[Dict], out: Path) -> None:
    by_level: Dict[int, List[str]] = defaultdict(list)
    for q in questions:
        ans = q.get("answer")
        if isinstance(ans, str):
            by_level[q.get("level")].append(ans)

    levels = sorted(by_level)
    fig, axes = plt.subplots(1, len(levels), figsize=(3 * len(levels), 3), sharey=True)
    if len(levels) == 1:
        axes = [axes]

    for ax, level, color in zip(axes, levels, LEVEL_COLORS):
        answers = by_level[level]
        modal_len = Counter(len(a) for a in answers).most_common(1)[0][0]
        fixed = [a for a in answers if len(a) == modal_len]
        positions = [chr(65 + i) for i in range(modal_len)]
        t_pcts = [sum(a[i] == "T" for a in fixed) / len(fixed) * 100 for i in range(modal_len)]
        f_pcts = [100 - p for p in t_pcts]

        x = np.arange(modal_len)
        ax.bar(x, t_pcts, label="True", color=color, alpha=0.85, width=0.5)
        ax.bar(x, f_pcts, bottom=t_pcts, label="False", color=GRAY, alpha=0.5, width=0.5)
        ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.set_title(LEVEL_NAMES.get(level, f"L{level}"), fontsize=10)
        ax.set_xlabel("Option")
        if ax is axes[0]:
            ax.set_ylabel("% of answers")
        ax.set_ylim(0, 100)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LEVEL_COLORS[0], alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, color=GRAY, alpha=0.5),
    ]
    fig.legend(handles, ["True", "False"], loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.suptitle("Per-option T/F balance by level", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Text summary ───────────────────────────────────────────────────────────────

def print_summary(questions: List[Dict]) -> None:
    total = len(questions)
    level_counts: Counter = Counter(q.get("level") for q in questions)
    print(f"\n  Total Q&A pairs: {total:,}")
    for level in sorted(level_counts):
        n = level_counts[level]
        print(f"  {LEVEL_NAMES.get(level, f'L{level}'):<22}: {n:>5}  ({100*n/total:.1f}%)")

# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distribution figures for FactoryBench Q&A pairs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/questions"),
        help="Root directory containing Q&A JSON files (searched recursively)")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    root = args.input.resolve()
    figures_dir = args.figures_dir.resolve()

    if not root.exists():
        print(f"ERROR: Input directory not found: {root}")
        return

    questions = load_questions(root)
    if not questions:
        print(f"No valid Q&A files found in {root}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nFactoryBench, Q&A Distribution Analysis")
    print(f"Source : {root}  ({len(questions):,} pairs)")
    print(f"Figures: {figures_dir}\n")

    print_summary(questions)
    print()

    print("Generating level × provenance options:")
    fig_option_a(questions, figures_dir / "f2_option_a_stacked_bar.png")
    fig_option_c(questions, figures_dir / "f2_option_c_bubble.png")
    fig_answer_balance(questions, figures_dir / "f2_answer_balance.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
