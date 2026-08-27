"""
Generate evaluation graphs from level2/level3 reply files.

Usage:
    python -m src.evaluation.plot_results --replies output/replies/level2 --questions output/questions/level2 --output output/plots/level2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_replies(replies_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(replies_dir.glob("*_answer.json")):
        try:
            records.append(load_json(path))
        except Exception:
            pass
    return records


def build_question_index(questions_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in questions_dir.rglob("*.json"):
        try:
            q = load_json(path)
            if isinstance(q, dict):
                index[path.stem] = q
        except Exception:
            pass
    return index


def _infer_answer_type(answer: Any) -> str:
    s = str(answer).strip().upper()
    if s and all(c in "TF" for c in s) and len(s) > 1:
        return "multi_select"
    if s.startswith("[") and s.endswith("]"):
        return "tensor"
    if s and all(c in "ABCD" for c in s) and len(s) > 1:
        return "ranking"
    return "numerical"


def enrich(records: List[Dict[str, Any]], question_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for r in records:
        stem = r.get("custom_id", "").rsplit("_", 1)[0]  # strip trailing _N
        q = question_index.get(stem, {})
        r = dict(r)
        r["difficulty"]     = q.get("difficulty", "unknown")
        r["template_id"]    = q.get("template_id")
        r["template_type"]  = q.get("template_type", "unknown")
        declared_type       = (q.get("answer_format") or {}).get("type")
        r["answer_type"]    = declared_type or _infer_answer_type(q.get("answer", r.get("ground_truth", "")))
        r["dataset"]        = (q.get("provenance") or {}).get("dataset", "unknown")
        usage = r.get("usage") or {}
        r["prompt_tokens"]     = usage.get("prompt_tokens", 0)
        r["completion_tokens"] = usage.get("completion_tokens", 0)
        r["reasoning_tokens"]  = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        r["total_tokens"]      = usage.get("total_tokens", 0)
        enriched.append(r)
    return enriched


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_distribution(records: List[Dict], output_dir: Path) -> None:
    scores = [r["score"] for r in records if r.get("score") is not None]
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 12)
    ax.hist(scores, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.8)
    ax.axvline(np.mean(scores), color=PALETTE[1], linewidth=2, linestyle="--", label=f"Mean = {np.mean(scores):.2f}")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()
    _save(fig, output_dir, "score_distribution.png")


def plot_score_by_difficulty(records: List[Dict], output_dir: Path) -> None:
    order = ["easy", "medium", "hard"]
    by_diff: Dict[str, List[float]] = {}
    for r in records:
        if r.get("score") is None:
            continue
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(r["score"])

    labels = [d for d in order if d in by_diff]
    means  = [np.mean(by_diff[d]) for d in labels]
    stds   = [np.std(by_diff[d])  for d in labels]
    counts = [len(by_diff[d])     for d in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means, yerr=stds, color=PALETTE[:len(labels)],
                  edgecolor="white", linewidth=0.8, capsize=5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Score")
    ax.set_title("Score by Difficulty")
    _save(fig, output_dir, "score_by_difficulty.png")


def plot_score_by_answer_type(records: List[Dict], output_dir: Path) -> None:
    by_type: Dict[str, List[float]] = {}
    for r in records:
        if r.get("score") is None:
            continue
        t = r.get("answer_type", "unknown")
        by_type.setdefault(t, []).append(r["score"])

    if not by_type:
        return

    labels = sorted(by_type)
    means  = [np.mean(by_type[l]) for l in labels]
    counts = [len(by_type[l])     for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, color=PALETTE[:len(labels)], edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Score")
    ax.set_title("Score by Answer Type")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, output_dir, "score_by_answer_type.png")


def plot_score_by_dataset(records: List[Dict], output_dir: Path) -> None:
    by_ds: Dict[str, List[float]] = {}
    for r in records:
        if r.get("score") is None:
            continue
        ds = r.get("dataset", "unknown")
        by_ds.setdefault(ds, []).append(r["score"])

    if len(by_ds) < 2:
        return  # only interesting with multiple datasets

    labels = sorted(by_ds)
    means  = [np.mean(by_ds[l]) for l in labels]
    counts = [len(by_ds[l])     for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, color=PALETTE[:len(labels)], edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={count}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Score")
    ax.set_title("Score by Dataset")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, output_dir, "score_by_dataset.png")


def plot_token_usage(records: List[Dict], output_dir: Path) -> None:
    prompt     = [r["prompt_tokens"]     for r in records if r.get("prompt_tokens")]
    completion = [r["completion_tokens"] for r in records if r.get("completion_tokens")]
    reasoning  = [r["reasoning_tokens"]  for r in records if r.get("reasoning_tokens")]

    if not prompt:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    data_labels = [("Prompt tokens", prompt, PALETTE[0]),
                   ("Completion tokens", completion, PALETTE[1]),
                   ("Reasoning tokens", reasoning, PALETTE[2])]

    for ax, (label, data, color) in zip(axes, data_labels):
        if data:
            ax.hist(data, bins=15, color=color, edgecolor="white", linewidth=0.8)
            ax.axvline(np.mean(data), color="black", linewidth=1.5, linestyle="--",
                       label=f"μ={np.mean(data):.0f}")
            ax.legend(fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Count")

    fig.suptitle("Token Usage Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, "token_usage.png")


def plot_reasoning_vs_score(records: List[Dict], output_dir: Path) -> None:
    pts = [(r["reasoning_tokens"], r["score"])
           for r in records
           if r.get("score") is not None and r.get("reasoning_tokens", 0) > 0]
    if len(pts) < 5:
        return

    x, y = zip(*pts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color=PALETTE[0], alpha=0.7, edgecolors="white", linewidth=0.5)

    # Trend line
    z = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 100)
    ax.plot(xs, np.poly1d(z)(xs), color=PALETTE[1], linewidth=2, linestyle="--", label="Trend")

    ax.set_xlabel("Reasoning tokens")
    ax.set_ylabel("Score")
    ax.set_title("Reasoning Tokens vs Score")
    ax.legend()
    _save(fig, output_dir, "reasoning_vs_score.png")


def plot_score_heatmap(records: List[Dict], output_dir: Path) -> None:
    difficulties = ["easy", "medium", "hard"]
    answer_types = sorted({r.get("answer_type", "unknown") for r in records})

    matrix = np.full((len(difficulties), len(answer_types)), np.nan)
    for i, diff in enumerate(difficulties):
        for j, atype in enumerate(answer_types):
            vals = [r["score"] for r in records
                    if r.get("difficulty") == diff and r.get("answer_type") == atype
                    and r.get("score") is not None]
            if vals:
                matrix[i, j] = np.mean(vals)

    if np.all(np.isnan(matrix)):
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Score")

    ax.set_xticks(range(len(answer_types)))
    ax.set_xticklabels(answer_types, rotation=20, ha="right")
    ax.set_yticks(range(len(difficulties)))
    ax.set_yticklabels(difficulties)
    ax.set_title("Mean Score: Difficulty × Answer Type")

    for i in range(len(difficulties)):
        for j in range(len(answer_types)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color="black" if 0.3 < val < 0.8 else "white")

    _save(fig, output_dir, "score_heatmap.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation results from reply JSON files.")
    parser.add_argument("--replies",   type=Path, required=True, help="Directory of *_answer.json files")
    parser.add_argument("--questions", type=Path, required=True, help="Directory of question JSON files")
    parser.add_argument("--output",    type=Path, required=True, help="Output directory for plots")
    args = parser.parse_args()

    print("Loading replies...")
    records = load_replies(args.replies)
    print(f"  {len(records)} replies loaded")

    print("Loading questions...")
    q_index = build_question_index(args.questions)
    records = enrich(records, q_index)

    scored = [r for r in records if r.get("score") is not None]
    overall = np.mean([r["score"] for r in scored]) if scored else 0
    print(f"  {len(scored)} scored | overall mean score = {overall:.3f}")

    print("Generating plots...")
    plot_score_distribution(records, args.output)
    plot_score_by_difficulty(records, args.output)
    plot_score_by_answer_type(records, args.output)
    plot_score_by_dataset(records, args.output)
    plot_token_usage(records, args.output)
    plot_reasoning_vs_score(records, args.output)
    plot_score_heatmap(records, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
