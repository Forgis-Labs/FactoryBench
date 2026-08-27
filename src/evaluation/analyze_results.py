"""Analyze FactoryBench evaluation results and generate publication-quality figures.

Generates:
  1. Per-model bar charts, accuracy by answer format
  2. Multi-model comparison heatmap, models (rows) x answer formats (columns)

Style: NeurIPS paper format with project colours.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.chance_correct import chance_correct

# ── project palette ──────────────────────────────────────────────
TIGER = "#ff5a00"
FIRE = "#FF4D00"
FLICKER = "#DC4B07"
STEEL = "#878f92"
GUNMETAL = "#122128"
PANEL = "#202e35"

# One colour per answer-format slot (cycled if more formats appear)
FORMAT_PALETTE = [TIGER, FIRE, FLICKER, STEEL, GUNMETAL, PANEL]

# Custom colour-map for heatmaps: white → orange → dark
BRAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "brand", ["#FFFFFF", "#FFD6B8", TIGER, FLICKER, GUNMETAL])

# ── Human-readable labels & canonical ordering ────────────────────────
FORMAT_LABELS = {
    "multiple_choice_single_select": "MCQ",
    "multiple_choice_multi_select": "MCQ (Multi)",
    "numerical": "Numerical",
    "tensor": "Tensor",
    "ranking": "Ranking",
    "free_form": "Free Form",
}

FORMAT_ORDER = ["MCQ", "MCQ (Multi)", "Numerical", "Tensor", "Ranking", "Free Form"]


# ── NeurIPS-style matplotlib defaults ─────────────────────────────────
def set_neurips_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "axes.edgecolor": GUNMETAL,
        "axes.labelcolor": GUNMETAL,
        "xtick.color": GUNMETAL,
        "ytick.color": GUNMETAL,
        "text.color": GUNMETAL,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ── Helpers ───────────────────────────────────────────────────────────
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_question_map(questions_dir: Path | None) -> Dict[str, str]:
    """Map question-file stem -> human-readable format label (fallback source)."""
    qmap: Dict[str, str] = {}
    if questions_dir is None or not questions_dir.is_dir():
        return qmap
    for q_path in questions_dir.rglob("*.json"):
        try:
            q = load_json(q_path)
            af = q.get("answer_format")
            if isinstance(af, dict):
                fmt = af.get("type", "unknown")
            elif isinstance(af, str):
                fmt = af
            else:
                fmt = "unknown"
            qmap[q_path.stem] = FORMAT_LABELS.get(fmt, fmt)
        except Exception:
            pass
    return qmap


def build_question_payload_map(questions_dir: Path | None) -> Dict[str, Dict[str, Any]]:
    """Map question-file stem -> full question payload (used by chance correction)."""
    qmap: Dict[str, Dict[str, Any]] = {}
    if questions_dir is None or not questions_dir.is_dir():
        return qmap
    for q_path in questions_dir.rglob("*.json"):
        try:
            q = load_json(q_path)
            if isinstance(q, dict):
                qmap[q_path.stem] = q
        except Exception:
            pass
    return qmap


def resolve_format_label(reply: Dict[str, Any], question_map: Dict[str, str]) -> str:
    """Derive a human-readable answer-format label from a reply JSON."""
    af = reply.get("answer_format")
    if isinstance(af, str) and af not in ("unknown", ""):
        return FORMAT_LABELS.get(af, af)

    prompt_file = Path(reply.get("prompt_file", ""))
    return question_map.get(prompt_file.stem, "unknown")


def load_replies(replies_dir: Path) -> List[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for r_path in replies_dir.rglob("*_answer.json"):
        try:
            results.append(load_json(r_path))
        except Exception:
            continue
    return results


def aggregate(
    results: List[Dict[str, Any]],
    question_map: Dict[str, str],
    payload_map: Dict[str, Dict[str, Any]] | None = None,
    apply_chance_correct: bool = False) -> Dict[str, Dict[str, List[float]]]:
    """Return model -> format_label -> [scores].

    When ``apply_chance_correct`` is True, raw scores are mapped through
    ``chance_correct`` using the matching question payload (so per-item
    option counts and permutation lengths are honoured). Free-form scores
    pass through unchanged.
    """
    agg: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for res in results:
        model = res.get("model", "unknown")
        score = res.get("score")
        if score is None:
            continue
        fmt = resolve_format_label(res, question_map)
        if apply_chance_correct:
            stem = Path(res.get("prompt_file", "")).stem
            q = (payload_map or {}).get(stem) or {}
            raw_af = res.get("answer_format")
            if not isinstance(raw_af, str) or raw_af in ("", "unknown"):
                qaf = q.get("answer_format")
                raw_af = qaf.get("type") if isinstance(qaf, dict) else (qaf or "unknown")
            corrected = chance_correct(float(score), raw_af, q)
            if corrected is None:
                continue
            agg[model][fmt].append(float(corrected))
        else:
            agg[model][fmt].append(float(score))
    return agg


def _ordered_formats(format_scores: Dict[str, List[float]]) -> List[str]:
    """Return formats in canonical order, followed by any extras."""
    present = [f for f in FORMAT_ORDER if f in format_scores]
    extra = sorted(set(format_scores.keys()) - set(FORMAT_ORDER))
    return present + extra


# ── Figure 1: per-model bar chart ─────────────────────────────────────
def plot_bar_chart(
    model: str,
    format_scores: Dict[str, List[float]],
    figures_dir: Path) -> None:
    formats = _ordered_formats(format_scores)
    if not formats:
        return

    accs = [np.mean(format_scores[f]) * 100 for f in formats]
    counts = [len(format_scores[f]) for f in formats]
    labels = [f"{f}\n(n={c})" for f, c in zip(formats, counts)]
    colors = [FORMAT_PALETTE[i % len(FORMAT_PALETTE)] for i in range(len(formats))]

    fig, ax = plt.subplots(figsize=(max(5, len(formats) * 1.4), 4))
    bars = ax.bar(
        range(len(formats)), accs,
        color=colors, width=0.65, edgecolor="white", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.1f}%",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=GUNMETAL)

    ax.set_xticks(range(len(formats)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, min(110, max(accs) + 15))
    ax.set_title(f"Accuracy by Answer Format \u2014 {model}", fontweight="bold", pad=12)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=STEEL)
    ax.set_axisbelow(True)

    all_scores = [s for scores in format_scores.values() for s in scores]
    overall = np.mean(all_scores) * 100 if all_scores else 0
    ax.text(
        0.98, 0.95,
        f"Overall: {overall:.1f}%  (N={len(all_scores)})",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color=STEEL, style="italic")

    fig.tight_layout()
    safe = model.replace("/", "_").replace(":", "_")
    out = figures_dir / f"bar_{safe}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart saved: {out}")


# ── Figure 2: multi-model comparison heatmap ──────────────────────────
def plot_comparison_heatmap(
    agg: Dict[str, Dict[str, List[float]]],
    figures_dir: Path) -> None:
    if not agg:
        return

    all_fmts: set[str] = set()
    for fs in agg.values():
        all_fmts.update(fs.keys())
    formats = [f for f in FORMAT_ORDER if f in all_fmts] + sorted(all_fmts - set(FORMAT_ORDER))
    models = sorted(agg.keys())

    matrix = np.full((len(models), len(formats)), np.nan)
    annot = [[""] * len(formats) for _ in models]

    for i, model in enumerate(models):
        for j, fmt in enumerate(formats):
            scores = agg[model].get(fmt, [])
            if scores:
                acc = np.mean(scores) * 100
                matrix[i, j] = acc
                annot[i][j] = f"{acc:.1f}\n(n={len(scores)})"

    col_labels = []
    for fmt in formats:
        total = sum(len(agg[m].get(fmt, [])) for m in models)
        col_labels.append(f"{fmt}\n(N={total})")

    fig, ax = plt.subplots(
        figsize=(max(6, len(formats) * 1.8), max(3, len(models) * 1.2 + 1)))

    sns.heatmap(
        matrix,
        annot=np.array(annot),
        fmt="",
        cmap=BRAND_CMAP,
        vmin=0, vmax=100,
        linewidths=1.5, linecolor="white",
        cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        ax=ax,
        mask=np.isnan(matrix))

    ax.set_yticklabels(models, rotation=0, fontweight="bold")
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_title(
        "Model Comparison \u2014 Accuracy by Answer Format",
        fontweight="bold", pad=14)

    fig.tight_layout()
    out = figures_dir / f"heatmap_comparison_{model}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison heatmap saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze FactoryBench results and generate figures.")
    parser.add_argument("--input", type=Path, required=True, help="Directory with LLM reply JSONs")
    parser.add_argument("--questions", type=Path, default=None, help="Question JSONs (fallback for format lookup)")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Output directory for figures")
    parser.add_argument("--chance-correct", action="store_true",
                        help="Apply per-format chance correction (max(0, (s-E)/(1-E))) before aggregating.")
    args = parser.parse_args()

    set_neurips_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    question_map = build_question_map(args.questions)
    payload_map = build_question_payload_map(args.questions) if args.chance_correct else {}
    results = load_replies(args.input)

    if not results:
        print(f"No result files found in {args.input}")
        return

    agg = aggregate(results, question_map, payload_map, apply_chance_correct=args.chance_correct)
    print(f"Loaded {len(results)} replies across {len(agg)} model(s).\n")

    for model, format_scores in agg.items():
        plot_bar_chart(model, format_scores, args.figures_dir)

    plot_comparison_heatmap(agg, args.figures_dir)

    print(f"\nAll figures saved to {args.figures_dir}")


if __name__ == "__main__":
    main()
