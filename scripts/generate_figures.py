"""Generate publication-quality figures for the FactoryBench paper.

Five figures:
  1. fig_main_heatmap, Model × Level chance-corrected accuracy
  2. fig_difficulty_curve, Accuracy vs level per model (line plot)
  3. fig_qtype_heatmap, Model × question-type per level (2×2 grid)
  4. fig_l4_score_dist, L4 free-form score distribution {0, 0.5, 1} per model
  5. fig_judge_agreement, L4 pairwise judge agreement + score distributions

Usage::

    python -m scripts.generate_figures
    python -m scripts.generate_figures --replies-root output/test_eval/replies --out-dir output/figures
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Brand colours ──────────────────────────────────────────────────────
TIGER    = "#ff5a00"
FLICKER  = "#DC4B07"
STEEL    = "#878f92"
GUNMETAL = "#122128"

BRAND_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "brand", ["#FFFFFF", "#FFD6B8", TIGER, FLICKER, GUNMETAL])

MODEL_DISPLAY: Dict[str, str] = {
    "gpt-5_1-1":         "GPT-5.1",
    "claude-sonnet-4_6": "Claude Sonnet 4.6",
    "deepseek-v3_2":     "DeepSeek V3.2",
    "mistral-large-3":   "Mistral Large 3",
    "qwen-3-235b":       "Qwen3-235B",
    "qwen-3-4b":         "Qwen3-4B",
    "baseline_simple":   "Simple Baseline",
}

MODEL_LINE_COLORS: Dict[str, str] = {
    "gpt-5_1-1":         TIGER,
    "claude-sonnet-4_6": "#1f77b4",
    "deepseek-v3_2":     "#2ca02c",
    "mistral-large-3":   "#9467bd",
    "qwen-3-235b":       "#8c564b",
    "qwen-3-4b":         "#e377c2",
    "baseline_simple":   "#aaaaaa",
}

QTYPE_DISPLAY: Dict[str, str] = {
    "predictive":                     "Predictive",
    "identification":                 "Identification",
    "comparative":                    "Comparative",
    "anomaly detection":              "Anomaly Det.",
    "anomaly_detection":              "Anomaly Det.",
    "trajectory_outcome_multiselect": "Trajectory",
    "intervention_outcome":           "Intervention",
    "troubleshooting":                "Troubleshooting",
    "optimization":                   "Optimization",
    "unknown":                        "Unknown",
}

LEVEL_LABELS = {1: "L1\nNumerical", 2: "L2\nMCQ", 3: "L3\nMulti-MCQ", 4: "L4\nFree-Form"}

# Conservative default chance scores per format
CHANCE_E: Dict[str, float] = {
    "numerical":                     0.25,
    "multiple_choice_single_select": 0.25,
    "multiple_choice_multi_select":  0.50,
    "ranking":                       0.25,
    "free_form":                     0.00,
}


# ── Style ──────────────────────────────────────────────────────────────
def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         10,
        "axes.titlesize":    12,
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
        "text.color":        GUNMETAL,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
    })


# ── Data loading ───────────────────────────────────────────────────────
def _effective_score(reply: dict) -> Optional[float]:
    """llm_judge_score for free_form (3-judge ensemble), else deterministic score."""
    af = reply.get("answer_format", "")
    s = reply.get("llm_judge_score") if af == "free_form" else reply.get("score")
    return None if s is None else float(s)


def _chance_correct(score: float, af: str) -> float:
    E = CHANCE_E.get(af, 0.0)
    if E >= 1.0:
        return 1.0 if score >= 1.0 else 0.0
    return max(0.0, min(1.0, (score - E) / (1.0 - E)))


def load_data(replies_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    level_dirs = sorted(
        d for d in replies_root.iterdir()
        if d.is_dir() and d.name.startswith("level")
    )
    for level_dir in level_dirs:
        level = int(level_dir.name.replace("level", ""))
        for model_dir in sorted(level_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name not in MODEL_DISPLAY:
                continue
            model_slug = model_dir.name
            for f in model_dir.glob("*_answer.json"):
                try:
                    r = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                af = r.get("answer_format", "unknown")
                qt = QTYPE_DISPLAY.get(r.get("question_type") or "unknown", "Unknown")
                s = _effective_score(r)
                votes = r.get("llm_judge_votes") or {}
                rows.append({
                    "level":         level,
                    "model":         model_slug,
                    "answer_format": af,
                    "question_type": qt,
                    "score":         s,
                    "score_cc":      _chance_correct(s, af) if s is not None else None,
                    "judge_gpt":     (votes.get("gpt-5.1-1") or {}).get("score"),
                    "judge_sonnet":  (votes.get("claude-sonnet-4.6") or {}).get("score"),
                    "judge_deepseek":(votes.get("deepseek-v3.2") or {}).get("score"),
                })
    return pd.DataFrame(rows)


def _model_order(df: pd.DataFrame) -> List[str]:
    """Model display names sorted by overall chance-corrected accuracy (desc)."""
    sub = df[df["score_cc"].notna()]
    overall = sub.groupby("model")["score_cc"].mean()
    return [MODEL_DISPLAY[m] for m in overall.sort_values(ascending=False).index
            if m in MODEL_DISPLAY]


# ── Figure 1: Main results heatmap ─────────────────────────────────────
def fig_main_heatmap(df: pd.DataFrame, model_order: List[str], out_dir: pathlib.Path) -> None:
    sub = df[df["score_cc"].notna()].copy()
    sub["model_name"] = sub["model"].map(MODEL_DISPLAY)
    agg = sub.groupby(["model_name", "level"])["score_cc"].agg(["mean", "count"]).reset_index()

    levels = sorted(sub["level"].unique())
    matrix = np.full((len(model_order), len(levels)), np.nan)
    annot  = [[""] * len(levels) for _ in model_order]

    for i, m in enumerate(model_order):
        for j, lv in enumerate(levels):
            row = agg[(agg["model_name"] == m) & (agg["level"] == lv)]
            if not row.empty:
                v = row.iloc[0]["mean"] * 100
                matrix[i, j] = v
                annot[i][j] = f"{v:.1f}%"

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        matrix, annot=np.array(annot), fmt="",
        cmap=BRAND_CMAP, vmin=0, vmax=100,
        linewidths=1.5, linecolor="white",
        cbar_kws={"label": "Chance-Corrected Accuracy (%)", "shrink": 0.8},
        ax=ax, mask=np.isnan(matrix))
    level_qa = agg.groupby("level")["count"].max()
    ax.set_yticklabels(model_order, rotation=0)
    ax.set_xticklabels([f"Level {l}\n(N={level_qa.get(l, 0):,})" for l in levels], rotation=0)
    ax.set_title("FactoryBench Results: Model × Level (Chance-Corrected)", fontweight="bold", pad=12)
    fig.tight_layout()
    _save(fig, out_dir / "fig_main_heatmap.pdf")


# ── Figure 2: Difficulty curve ─────────────────────────────────────────
def fig_difficulty_curve(df: pd.DataFrame, model_order: List[str], out_dir: pathlib.Path) -> None:
    sub = df[df["score_cc"].notna()].copy()
    sub["model_name"] = sub["model"].map(MODEL_DISPLAY)
    agg = sub.groupby(["model", "level"])["score_cc"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    for m_slug, grp in agg.groupby("model"):
        if m_slug not in MODEL_DISPLAY:
            continue
        grp  = grp.sort_values("level")
        name = MODEL_DISPLAY[m_slug]
        # rank in model_order → dashed for lower-ranked models
        rank = model_order.index(name) if name in model_order else 99
        ax.plot(
            grp["level"], grp["score_cc"] * 100,
            marker="o", linewidth=1.8, markersize=6,
            color=MODEL_LINE_COLORS.get(m_slug, STEEL),
            linestyle="-" if rank < 3 else "--",
            label=name)

    ax.axhline(0, color=STEEL, linestyle=":", linewidth=1.0, alpha=0.6, label="Random baseline")
    ax.set_xlabel("Benchmark Level")
    ax.set_ylabel("Chance-Corrected Accuracy (%)")
    ax.set_title("Performance Across Levels", fontweight="bold")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([LEVEL_LABELS[l] for l in [1, 2, 3, 4]])
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=STEEL)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="none", ncol=2)
    fig.tight_layout()
    _save(fig, out_dir / "fig_difficulty_curve.pdf")


# ── Figure 3: Question-type breakdown ─────────────────────────────────
def fig_qtype_heatmap(df: pd.DataFrame, model_order: List[str], out_dir: pathlib.Path) -> None:
    sub = df[df["score_cc"].notna()].copy()
    sub["model_name"] = sub["model"].map(MODEL_DISPLAY)
    levels = sorted(sub["level"].unique())

    fig, axes = plt.subplots(1, len(levels), figsize=(4.5 * len(levels), 5))
    if len(levels) == 1:
        axes = [axes]

    for ax, lv in zip(axes, levels):
        lsub   = sub[sub["level"] == lv]
        qtypes = sorted(lsub["question_type"].unique())
        mnames = [m for m in model_order if m in lsub["model_name"].values]

        matrix = np.full((len(mnames), len(qtypes)), np.nan)
        for i, m in enumerate(mnames):
            for j, qt in enumerate(qtypes):
                vals = lsub[(lsub["model_name"] == m) & (lsub["question_type"] == qt)]["score_cc"]
                if len(vals):
                    matrix[i, j] = vals.mean() * 100

        annot = [[f"{v:.0f}" if not np.isnan(v) else "" for v in row] for row in matrix]
        is_last = (lv == levels[-1])
        sns.heatmap(
            matrix, annot=np.array(annot), fmt="",
            cmap=BRAND_CMAP, vmin=0, vmax=100,
            linewidths=1.0, linecolor="white",
            cbar=is_last,
            cbar_kws={"label": "Acc. (%)", "shrink": 0.8} if is_last else {},
            ax=ax, mask=np.isnan(matrix),
            xticklabels=qtypes, yticklabels=mnames if lv == levels[0] else [])
        ax.set_title(f"Level {lv}", fontweight="bold")
        ax.set_xticklabels(qtypes, rotation=35, ha="right")

    fig.suptitle("Accuracy by Question Type (Chance-Corrected)", fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig_qtype_heatmap.pdf")


# ── Figure 4: L4 score distribution ───────────────────────────────────
def fig_l4_score_dist(df: pd.DataFrame, model_order: List[str], out_dir: pathlib.Path) -> None:
    sub = df[
        (df["level"] == 4) &
        (df["answer_format"] == "free_form") &
        df["score"].notna()
    ].copy()
    sub["model_name"] = sub["model"].map(MODEL_DISPLAY)

    # Sort by % score=1.0 ascending (worst at top → best at bottom for horizontal bars)
    pct_correct = (
        sub[sub["score"] == 1.0].groupby("model_name").size() /
        sub.groupby("model_name").size()
    ).reindex([m for m in reversed(model_order) if m in sub["model_name"].values]).fillna(0)
    mnames = list(pct_correct.index)

    score_vals = [0.0, 0.5, 1.0]
    colors     = [GUNMETAL, STEEL, TIGER]
    labels     = ["0.0, Wrong", "0.5, Partial", "1.0, Correct"]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(mnames) * 0.75)))
    y    = np.arange(len(mnames))
    left = np.zeros(len(mnames))

    for sv, color, label in zip(score_vals, colors, labels):
        vals = np.array([
            (sub[sub["model_name"] == m]["score"] == sv).sum() / max(1, (sub["model_name"] == m).sum()) * 100
            for m in mnames
        ])
        ax.barh(y, vals, left=left, color=color, label=label, height=0.6, edgecolor="white", linewidth=0.4)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(mnames)
    ax.set_xlabel("Proportion of L4 Free-Form Items (%)")
    ax.set_title("L4 Free-Form Score Distribution (Troubleshooting + Optimization)", fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=STEEL)
    ax.set_axisbelow(True)
    n_total = len(sub) // max(1, len(mnames))
    ax.set_title(
        f"L4 Free-Form Score Distribution  (≈{n_total:,} items/model)",
        fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir / "fig_l4_score_dist.pdf")


# ── Figure 5: Judge agreement ──────────────────────────────────────────
def fig_judge_agreement(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    base = df[(df["level"] == 4) & (df["answer_format"] == "free_form")].copy()

    all_judge_cols  = ["judge_gpt", "judge_sonnet", "judge_deepseek"]
    all_judge_names = ["GPT-5.1", "Sonnet 4.6", "DeepSeek V3.2"]

    # Keep only judges that have at least one valid score
    active = [
        (col, name) for col, name in zip(all_judge_cols, all_judge_names)
        if base[col].notna().any()
    ]
    if len(active) < 2:
        print("  fig_judge_agreement: fewer than 2 judges have valid scores, skipping.")
        return

    judge_cols  = [c for c, _ in active]
    judge_names = [n for _, n in active]
    n_judges    = len(judge_names)

    # Pairwise agreement computed over items where BOTH judges have a valid score
    agree      = np.full((n_judges, n_judges), np.nan)
    agree_n    = np.zeros((n_judges, n_judges), dtype=int)
    for i, ci in enumerate(judge_cols):
        for j, cj in enumerate(judge_cols):
            mask = base[ci].notna() & base[cj].notna()
            n    = mask.sum()
            agree_n[i, j] = n
            if n > 0:
                agree[i, j] = (base.loc[mask, ci].values == base.loc[mask, cj].values).mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(4.5 * n_judges, 4.5))

    # Left: pairwise agreement heatmap
    ax = axes[0]
    annot = [
        [f"{agree[i,j]:.1f}%\n(n={agree_n[i,j]:,})" if not np.isnan(agree[i,j]) else ""
         for j in range(n_judges)]
        for i in range(n_judges)
    ]
    sns.heatmap(
        agree, annot=np.array(annot), fmt="",
        cmap=BRAND_CMAP, vmin=50, vmax=100,
        linewidths=1.5, linecolor="white",
        cbar_kws={"label": "Exact Agreement (%)", "shrink": 0.8},
        ax=ax, xticklabels=judge_names, yticklabels=judge_names,
        mask=np.isnan(agree))
    ax.set_title("Pairwise Judge Agreement", fontweight="bold")

    # Right: score distribution per judge (over items with a valid score for that judge)
    ax = axes[1]
    score_vals   = [0.0, 0.5, 1.0]
    score_colors = {0.0: GUNMETAL, 0.5: STEEL, 1.0: TIGER}
    x     = np.arange(n_judges)
    width = 0.22

    for k, sv in enumerate(score_vals):
        vals = []
        ns   = []
        for col in judge_cols:
            valid = base[col].dropna()
            vals.append((valid == sv).mean() * 100 if len(valid) else 0.0)
            ns.append(len(valid))
        offset = (k - 1) * width
        bars   = ax.bar(x + offset, vals, width=width,
                        color=score_colors[sv], label=f"Score {sv}",
                        edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    xlabels = [f"{n}\n(n={ns[i]:,})" for i, n in enumerate(judge_names)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("% of Scored L4 Items")
    ax.set_title("Score Distribution per Judge", fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=STEEL)
    ax.set_axisbelow(True)

    fig.suptitle("L4 Judge Ensemble Reliability", fontweight="bold", fontsize=13)
    fig.tight_layout()
    _save(fig, out_dir / "fig_judge_agreement.pdf")


# ── Helpers ────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, path: pathlib.Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Entry point ────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--replies-root", type=pathlib.Path,
                        default=pathlib.Path("output/test_eval/replies"))
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("output/figures"))
    args = parser.parse_args()

    set_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reply data (this may take ~30s for large datasets)...")
    df = load_data(args.replies_root)
    print(f"  {len(df):,} records | models: {sorted(df['model'].unique())} | levels: {sorted(df['level'].unique())}")
    print()

    morder = _model_order(df)
    print(f"Model order (best→worst): {morder}\n")

    print("Generating figures...")
    fig_main_heatmap(df, morder, args.out_dir)
    fig_difficulty_curve(df, morder, args.out_dir)
    fig_qtype_heatmap(df, morder, args.out_dir)
    fig_l4_score_dist(df, morder, args.out_dir)
    fig_judge_agreement(df, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
