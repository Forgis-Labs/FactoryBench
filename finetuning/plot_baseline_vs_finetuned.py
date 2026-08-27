"""Bar charts comparing baseline (pretrained Shrike/bearing) to FactoryBench
DoRA-finetuned, in the project palette used in the paper.

Reads the *_summary.json artifacts that ``finetuning/score_predictions.py``
writes alongside each scored JSONL. Defaults to the two folders we built
during the L1-L3 comparison, but you can pass any pair.

Outputs PNG (for slides) + PDF (for paper) to papers/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# the project palette
PALETTE = {
    "orange":    "#FF5A00",   # tiger / brand_orange
    "fire":      "#FF4D00",
    "flicker":   "#DC4B07",
    "navy":      "#122128",   # dark_navy / gunmetal
    "panel":     "#202E35",
    "steel":     "#878F92",
    "panelbg":   "#F0F4F8",
    "blue":      "#3A7BD5",
}

BASELINE_COLOR  = PALETTE["navy"]      # pretrained Shrike checkpoint
FINETUNED_COLOR = PALETTE["orange"]    # DoRA-finetuned on FactoryBench


def _apply_style() -> None:
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     PALETTE["panel"],
        "axes.labelcolor":    PALETTE["navy"],
        "xtick.color":        PALETTE["panel"],
        "ytick.color":        PALETTE["panel"],
        "axes.grid":          True,
        "grid.color":         PALETTE["steel"],
        "grid.linestyle":     ":",
        "grid.linewidth":     0.6,
        "grid.alpha":         0.35,
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
    })


def _load_summaries(folder: Path) -> dict[int, dict]:
    """Return {level_int: summary_dict} from *_summary.json files in folder."""
    out: dict[int, dict] = {}
    for p in sorted(folder.glob("level_*_predictions_summary.json")):
        # level_1_predictions_summary.json -> 1
        lvl = int(p.stem.split("_")[1])
        out[lvl] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _agg_per_format(summaries: dict[int, dict]) -> dict[str, tuple[int, float]]:
    """Pool across levels: {format -> (total_n, weighted_mean)}."""
    pool: dict[str, list[tuple[int, float]]] = {}
    for s in summaries.values():
        for fmt, stats in s.get("by_answer_format", {}).items():
            pool.setdefault(fmt, []).append((stats["n"], stats["mean"]))
    out: dict[str, tuple[int, float]] = {}
    for fmt, entries in pool.items():
        total_n = sum(n for n, _ in entries)
        wm = (sum(n * m for n, m in entries) / total_n) if total_n else 0.0
        out[fmt] = (total_n, wm)
    return out


def plot_per_level(base: dict[int, dict], fine: dict[int, dict],
                   out_stem: Path, title_suffix: str = "") -> None:
    levels = sorted(set(base) | set(fine))
    x = np.arange(len(levels))
    w = 0.36

    base_means = [base.get(l, {}).get("mean_score") or 0.0 for l in levels]
    fine_means = [fine.get(l, {}).get("mean_score") or 0.0 for l in levels]
    ns = [base.get(l, fine.get(l, {})).get("n", 0) for l in levels]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars_b = ax.bar(x - w / 2, base_means, w, label="Pretrained (Shrike/bearing)",
                    color=BASELINE_COLOR, edgecolor="none")
    bars_f = ax.bar(x + w / 2, fine_means, w, label="DoRA-finetuned on FactoryBench",
                    color=FINETUNED_COLOR, edgecolor="none")

    for bars, vals in [(bars_b, base_means), (bars_f, fine_means)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, color=PALETTE["navy"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}\n(n={n})" for l, n in zip(levels, ns)])
    ax.set_ylabel("Cascade-scored mean")
    ax.set_ylim(0, max(max(base_means + fine_means) * 1.25, 0.05))
    ax.set_title("FactoryBench performance by level" + title_suffix,
                 color=PALETTE["navy"], pad=10)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote  {out_stem.with_suffix('.png')}")
    print(f"  wrote  {out_stem.with_suffix('.pdf')}")


def plot_per_format(base: dict[int, dict], fine: dict[int, dict],
                    out_stem: Path, title_suffix: str = "") -> None:
    bf = _agg_per_format(base)
    ff = _agg_per_format(fine)
    fmts = sorted(set(bf) | set(ff))
    # Render in a friendlier order
    order = ["numerical", "multiple_choice_multi_select",
             "multiple_choice_single_select", "ranking", "tensor", "free_form"]
    fmts = sorted(fmts, key=lambda f: order.index(f) if f in order else len(order))

    labels = {
        "numerical": "numerical\n(tolerance)",
        "multiple_choice_multi_select": "T/F multi\n(per-bit)",
        "multiple_choice_single_select": "A-D single\n(MC)",
        "ranking": "ranking\n(Kendall-τ)",
        "tensor": "tensor\n(6-vec)",
        "free_form": "free-form\n(LLM judge)",
    }

    x = np.arange(len(fmts))
    w = 0.36
    base_means = [bf.get(f, (0, 0.0))[1] for f in fmts]
    fine_means = [ff.get(f, (0, 0.0))[1] for f in fmts]
    ns = [max(bf.get(f, (0, 0))[0], ff.get(f, (0, 0))[0]) for f in fmts]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bars_b = ax.bar(x - w / 2, base_means, w, label="Pretrained (Shrike/bearing)",
                    color=BASELINE_COLOR, edgecolor="none")
    bars_f = ax.bar(x + w / 2, fine_means, w, label="DoRA-finetuned on FactoryBench",
                    color=FINETUNED_COLOR, edgecolor="none")

    for bars, vals in [(bars_b, base_means), (bars_f, fine_means)]:
        for bar, v in zip(bars, vals):
            if v > 1e-4:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9, color=PALETTE["navy"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{labels.get(f, f)}\nn={n}" for f, n in zip(fmts, ns)])
    ax.set_ylabel("Cascade-scored mean")
    ymax = max(base_means + fine_means)
    ax.set_ylim(0, max(ymax * 1.25, 0.05))
    ax.set_title("FactoryBench performance by answer format" + title_suffix,
                 color=PALETTE["navy"], pad=10)
    # Legend ABOVE the plot, ranking bar can be tall enough to clash with
    # an in-plot legend at the right or top corners.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote  {out_stem.with_suffix('.png')}")
    print(f"  wrote  {out_stem.with_suffix('.pdf')}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="baseline_l123",
                   help="Folder with baseline level_*_predictions_summary.json")
    p.add_argument("--finetuned", default="finetuned_l123",
                   help="Folder with finetuned level_*_predictions_summary.json")
    p.add_argument("--out-dir", default="papers/figures",
                   help="Where to write the PNG+PDF outputs")
    p.add_argument("--suffix", default=", bearing_mixed_r32, L1–L3",
                   help="Appended to plot titles")
    args = p.parse_args()

    _apply_style()

    base_dir = Path(args.baseline)
    fine_dir = Path(args.finetuned)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _load_summaries(base_dir)
    fine = _load_summaries(fine_dir)
    if not base:
        raise SystemExit(f"No *_summary.json in {base_dir}, run score_predictions.py first")
    if not fine:
        raise SystemExit(f"No *_summary.json in {fine_dir}, run score_predictions.py first")

    print(f"Baseline levels:  {sorted(base.keys())}")
    print(f"Finetuned levels: {sorted(fine.keys())}")

    plot_per_level(base, fine, out_dir / "fbench_baseline_vs_finetuned_by_level",
                   title_suffix=args.suffix)
    plot_per_format(base, fine, out_dir / "fbench_baseline_vs_finetuned_by_format",
                    title_suffix=args.suffix)


if __name__ == "__main__":
    main()
