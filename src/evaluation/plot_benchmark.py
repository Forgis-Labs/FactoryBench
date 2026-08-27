"""Cross-model benchmark plots from local reply JSONs.

Walks output/replies/level{1..4}/<model>/*_answer.json, joins with question
payloads for answer_format, and produces:
  1. model_x_level.png, grouped bar: mean score per (model, level)
  2. model_x_format.png, grouped bar: mean score per (model, answer_format)
  3. heatmap.png, model × level mean score heatmap
  4. summary.csv, full pivot table

Usage:
    python -m src.evaluation.plot_benchmark
    python -m src.evaluation.plot_benchmark --replies-root output/replies --questions-root output/questions --output output/plots
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.run_foundry_eval import build_question_index, infer_answer_format

logger = logging.getLogger(__name__)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
LEVEL_RE = re.compile(r"level(\d+)")


def _level_from_stem(stem: str) -> Optional[int]:
    m = LEVEL_RE.match(stem)
    return int(m.group(1)) if m else None


def _question_stem_from_reply_stem(reply_stem: str) -> str:
    s = reply_stem[:-len("_answer")] if reply_stem.endswith("_answer") else reply_stem
    parts = s.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return s


def load_records(
    replies_root: Path,
    questions_root: Path) -> List[Dict[str, Any]]:
    q_indices: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for lvl_dir in sorted(questions_root.glob("level*")):
        m = LEVEL_RE.match(lvl_dir.name)
        if not m:
            continue
        q_indices[int(m.group(1))] = build_question_index(lvl_dir)

    records: List[Dict[str, Any]] = []
    for level_dir in sorted(replies_root.glob("level*")):
        m = LEVEL_RE.match(level_dir.name)
        if not m:
            continue
        level = int(m.group(1))
        q_index = q_indices.get(level, {})
        for model_dir in sorted(level_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for reply_path in model_dir.glob("*_answer.json"):
                try:
                    r = json.loads(reply_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(r, dict):
                    continue
                score = r.get("score")
                if score is None:
                    continue
                q_stem = _question_stem_from_reply_stem(reply_path.stem)
                q = q_index.get(q_stem) or {}
                fmt = r.get("answer_format") or infer_answer_format(q)
                if fmt == "unknown":
                    fmt = infer_answer_format(q)
                records.append({
                    "model": model,
                    "level": level,
                    "answer_format": fmt,
                    "template_type": q.get("template_type") or r.get("question_type") or "unknown",
                    "score": float(score),
                })
    return records


def _group_mean(records, key_fn):
    buckets: Dict[Any, List[float]] = {}
    for r in records:
        buckets.setdefault(key_fn(r), []).append(r["score"])
    return {k: (float(np.mean(v)), len(v)) for k, v in buckets.items()}


def plot_model_x_level(records, output: Path) -> None:
    models = sorted({r["model"] for r in records})
    levels = sorted({r["level"] for r in records})
    data = _group_mean(records, lambda r: (r["model"], r["level"]))

    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(levels))
    for i, m in enumerate(models):
        means = [data.get((m, lv), (0.0, 0))[0] for lv in levels]
        ax.bar(x + i * width, means, width, label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f"Level {lv}" for lv in levels])
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model accuracy by level")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


LEVEL_NAMES = {
    1: "Level 1\nState",
    2: "Level 2\nIntervention",
    3: "Level 3\nCounterfactual",
    4: "Level 4\nDecision",
}

MODEL_DISPLAY = {
    "gpt-5.1": "GPT-5.1",
    "gpt-5_1": "GPT-5.1",
    "gpt-5.1-1": "GPT-5.1",
    "gpt-5_1-1": "GPT-5.1",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "DeepSeek-V3_1": "DeepSeek V3.1",
    "DeepSeek-V3.1": "DeepSeek V3.1",
    "Mistral-Large-3": "Mistral Large 3",
}


def _configure_paper_rc() -> None:
    """Matplotlib rc settings for paper-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Nimbus Roman", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,   # TrueType, editable in Illustrator
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_heatmap(records, output: Path) -> None:
    """Paper-quality headline heatmap: Levels (cols) × Models (rows)."""
    models = sorted({r["model"] for r in records})
    levels = sorted({r["level"] for r in records})
    data = _group_mean(records, lambda r: (r["model"], r["level"]))
    mat = np.array([[data.get((m, lv), (np.nan, 0))[0] for lv in levels] for m in models])
    counts = np.array([[data.get((m, lv), (np.nan, 0))[1] for lv in levels] for m in models])

    fig_w = 1.6 + 1.6 * len(levels)
    fig_h = 1.2 + 0.7 * len(models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("viridis")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    # Axis ticks & labels
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([LEVEL_NAMES.get(lv, f"Level {lv}") for lv in levels])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in models])
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Minor ticks for thin white cell separators
    ax.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cell annotations: score (bold) + n below
    for i in range(len(models)):
        for j in range(len(levels)):
            val = mat[i, j]
            n = int(counts[i, j]) if not np.isnan(counts[i, j]) else 0
            if n == 0 or np.isnan(val):
                ax.text(j, i, ", ", ha="center", va="center", color="#999999", fontsize=11)
                continue
            text_color = "white" if val < 0.55 else "black"
            ax.text(j, i - 0.12, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=13, fontweight="bold")
            ax.text(j, i + 0.22, f"n={n}", ha="center", va="center",
                    color=text_color, fontsize=8, alpha=0.85)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean accuracy", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    # Save PNG + PDF
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def plot_score_vs_level(records, output: Path) -> None:
    """Line plot: mean score per level, one line per model, with ±1 SE shaded band."""
    models = sorted({r["model"] for r in records})
    levels = sorted({r["level"] for r in records})

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        means, ses = [], []
        for lv in levels:
            scores = [r["score"] for r in records if r["model"] == m and r["level"] == lv]
            if scores:
                means.append(float(np.mean(scores)))
                ses.append(float(np.std(scores) / np.sqrt(len(scores))))
            else:
                means.append(np.nan)
                ses.append(0.0)
        means_arr = np.array(means)
        ses_arr = np.array(ses)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(levels, means_arr, marker="o", linewidth=2, label=m, color=color)
        ax.fill_between(levels, means_arr - ses_arr, means_arr + ses_arr, color=color, alpha=0.15)

    ax.set_xticks(levels)
    ax.set_xlabel("Level (1=State, 2=Intervention, 3=Counterfactual, 4=Decision)")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Reasoning progression: accuracy vs level")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_score_box_by_level(records, output: Path) -> None:
    """Box plots: score distribution per level, grouped by model."""
    models = sorted({r["model"] for r in records})
    levels = sorted({r["level"] for r in records})

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        positions = [lv + (i - (len(models) - 1) / 2) * width for lv in levels]
        data = [
            [r["score"] for r in records if r["model"] == m and r["level"] == lv]
            or [0.0]
            for lv in levels
        ]
        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.9, patch_artist=True,
            medianprops={"color": "black"}, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.7)
        # legend proxy
        ax.plot([], [], color=PALETTE[i % len(PALETTE)], linewidth=8, label=m, alpha=0.7)

    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{lv}" for lv in levels])
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Score distribution by level")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_format_across_levels(records, output: Path) -> None:
    """Small-multiples: one subplot per answer_format, lines show model score vs level."""
    formats = sorted({r["answer_format"] for r in records})
    models = sorted({r["model"] for r in records})
    levels = sorted({r["level"] for r in records})
    n = len(formats)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.2), sharey=True)
    axes = np.array(axes).reshape(-1)
    for k, fmt in enumerate(formats):
        ax = axes[k]
        for i, m in enumerate(models):
            means = []
            for lv in levels:
                scores = [
                    r["score"] for r in records
                    if r["model"] == m and r["level"] == lv and r["answer_format"] == fmt
                ]
                means.append(float(np.mean(scores)) if scores else np.nan)
            ax.plot(levels, means, marker="o", linewidth=1.8,
                    label=m, color=PALETTE[i % len(PALETTE)])
        ax.set_xticks(levels)
        ax.set_ylim(0, 1.0)
        ax.set_title(fmt, fontsize=10)
        ax.grid(True, alpha=0.3)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    # legend on first subplot
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Accuracy per answer-format, across levels")
    fig.supxlabel("Level")
    fig.supylabel("Mean score")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_template_type_heatmap(records, output: Path) -> None:
    """Heatmap: model × template_type across all levels."""
    types = sorted({r["template_type"] for r in records if r.get("template_type") and r["template_type"] != "unknown"})
    if not types:
        return
    models = sorted({r["model"] for r in records})
    data = _group_mean(records, lambda r: (r["model"], r["template_type"]))
    mat = np.array([[data.get((m, t), (np.nan, 0))[0] for t in types] for m in models])

    fig, ax = plt.subplots(figsize=(1 + 1.1 * len(types), 1 + 0.6 * len(models)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(types)):
            val = mat[i, j]
            n = data.get((models[i], types[j]), (np.nan, 0))[1]
            if n == 0 or np.isnan(val):
                cell, color = ", ", "gray"
            else:
                cell = f"{val:.2f}\nn={n}"
                color = "white" if val < 0.5 else "black"
            ax.text(j, i, cell, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean score")
    ax.set_title("Model × Template type (all levels)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_level_composition(records, output: Path) -> None:
    """Stacked bar: question count composition per level by answer_format."""
    levels = sorted({r["level"] for r in records})
    formats = sorted({r["answer_format"] for r in records})
    # Count unique (level, format, question), collapse models so each question
    # is counted once per level (use any single model to avoid inflating).
    # Simple approach: divide count by number of models.
    models = sorted({r["model"] for r in records})
    n_models = max(1, len(models))
    counts = {(lv, f): 0 for lv in levels for f in formats}
    for r in records:
        counts[(r["level"], r["answer_format"])] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(levels))
    for i, f in enumerate(formats):
        vals = np.array([counts[(lv, f)] / n_models for lv in levels])
        ax.bar(levels, vals, bottom=bottoms, label=f, color=PALETTE[i % len(PALETTE)])
        bottoms += vals
    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{lv}" for lv in levels])
    ax.set_ylabel("Questions (per model)")
    ax.set_title("Answer-format composition per level")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def plot_heatmap_per_level(records, output_dir: Path) -> None:
    """One heatmap per level: models × answer_format."""
    levels = sorted({r["level"] for r in records})
    for level in levels:
        subset = [r for r in records if r["level"] == level]
        if not subset:
            continue
        models = sorted({r["model"] for r in subset})
        formats = sorted({r["answer_format"] for r in subset})
        data = _group_mean(subset, lambda r: (r["model"], r["answer_format"]))
        mat = np.array([[data.get((m, f), (np.nan, 0))[0] for f in formats] for m in models])

        fig, ax = plt.subplots(figsize=(1 + 1.3 * len(formats), 1 + 0.6 * len(models)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(formats)))
        ax.set_xticklabels(formats, rotation=25, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(formats)):
                val = mat[i, j]
                n = data.get((models[i], formats[j]), (np.nan, 0))[1]
                if n == 0 or np.isnan(val):
                    cell = ", "
                    color = "gray"
                else:
                    cell = f"{val:.2f}\n(n={n})"
                    color = "white" if val < 0.5 else "black"
                ax.text(j, i, cell, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, label="Mean score")
        ax.set_title(f"Level {level}: Model × Answer Format")
        fig.tight_layout()
        out = output_dir / f"heatmap_level{level}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_heatmap_per_model(records, output_dir: Path) -> None:
    """One heatmap per model: levels × answer_format."""
    models = sorted({r["model"] for r in records})
    for model in models:
        subset = [r for r in records if r["model"] == model]
        if not subset:
            continue
        levels = sorted({r["level"] for r in subset})
        formats = sorted({r["answer_format"] for r in subset})
        data = _group_mean(subset, lambda r: (r["level"], r["answer_format"]))
        mat = np.array([[data.get((lv, f), (np.nan, 0))[0] for f in formats] for lv in levels])

        fig, ax = plt.subplots(figsize=(1 + 1.3 * len(formats), 1 + 0.6 * len(levels)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(formats)))
        ax.set_xticklabels(formats, rotation=25, ha="right")
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels([f"L{lv}" for lv in levels])
        for i in range(len(levels)):
            for j in range(len(formats)):
                val = mat[i, j]
                n = data.get((levels[i], formats[j]), (np.nan, 0))[1]
                if n == 0 or np.isnan(val):
                    cell = ", "
                    color = "gray"
                else:
                    cell = f"{val:.2f}\n(n={n})"
                    color = "white" if val < 0.5 else "black"
                ax.text(j, i, cell, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, label="Mean score")
        ax.set_title(f"{model}: Level × Answer Format")
        fig.tight_layout()
        safe = model.replace("/", "_")
        out = output_dir / f"heatmap_{safe}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_model_x_format(records, output: Path) -> None:
    models = sorted({r["model"] for r in records})
    formats = sorted({r["answer_format"] for r in records})
    data = _group_mean(records, lambda r: (r["model"], r["answer_format"]))

    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(formats))
    for i, m in enumerate(models):
        means = [data.get((m, f), (0.0, 0))[0] for f in formats]
        ax.bar(x + i * width, means, width, label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(formats, rotation=25, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model accuracy by answer format")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output}")


def write_summary_csv(records, output: Path) -> None:
    per_ml = _group_mean(records, lambda r: (r["model"], r["level"]))
    per_mf = _group_mean(records, lambda r: (r["model"], r["answer_format"]))
    per_m = _group_mean(records, lambda r: r["model"])
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["axis", "group", "key", "mean_score", "n"])
        for (m, lv), (mean, n) in sorted(per_ml.items()):
            w.writerow(["model_x_level", m, f"level_{lv}", f"{mean:.4f}", n])
        for (m, fmt), (mean, n) in sorted(per_mf.items()):
            w.writerow(["model_x_format", m, fmt, f"{mean:.4f}", n])
        for m, (mean, n) in sorted(per_m.items()):
            w.writerow(["model_overall", m, "overall", f"{mean:.4f}", n])
    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model benchmark plots from reply JSONs.")
    parser.add_argument("--replies-root", type=Path, default=Path("output/replies"))
    parser.add_argument("--questions-root", type=Path, default=Path("output/questions"))
    parser.add_argument("--output", type=Path, default=Path("output/plots"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    records = load_records(args.replies_root, args.questions_root)
    if not records:
        parser.error(f"No scored replies found under {args.replies_root}")
    logger.info(f"Loaded {len(records)} scored replies")

    _configure_paper_rc()
    args.output.mkdir(parents=True, exist_ok=True)

    # Headline paper figure: Models × Levels heatmap (PNG + PDF).
    plot_heatmap(records, args.output / "heatmap_models_x_levels")

    # Supporting figures
    plot_model_x_level(records, args.output / "model_x_level.png")
    plot_model_x_format(records, args.output / "model_x_format.png")
    # Inter-level plots
    plot_score_vs_level(records, args.output / "progression.png")
    plot_score_box_by_level(records, args.output / "distribution_by_level.png")
    plot_format_across_levels(records, args.output / "format_across_levels.png")
    plot_template_type_heatmap(records, args.output / "template_type_heatmap.png")
    plot_level_composition(records, args.output / "level_composition.png")
    # Per-level / per-model drilldowns
    plot_heatmap_per_level(records, args.output)
    plot_heatmap_per_model(records, args.output)
    write_summary_csv(records, args.output / "summary.csv")


if __name__ == "__main__":
    main()
