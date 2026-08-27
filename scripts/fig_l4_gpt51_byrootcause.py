"""GPT-5.1 L4 score distribution per fault (Troubleshooting) and per parameter
(Optimization).

Two-panel figure with stacked horizontal bars. Each row is a specific
ground-truth fault (L4.1) or misconfigured parameter (L4.2); the stack shows
the {0, 0.5, 1} score distribution. n is annotated for each bar.

Output: docs/neurips_tex/figures/fig_l4_gpt51_byrootcause.pdf
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

REPO = pathlib.Path(__file__).resolve().parents[1]

TIGER    = "#ff5a00"
FLICKER  = "#DC4B07"
STEEL    = "#878f92"
GUNMETAL = "#122128"

# Display labels
ROOT_CAUSE_DISPLAY = {
    "normal":                       "Nominal (no fault)",
    "loosening_phase":              "Loosening phase",
    "missing_screw":                "Missing screw",
    "extra_assembly_component":     "Extra component",
    "gripper_activation_failure":   "Gripper act. failure",
    "additional_axis_payload":      "Add'l axis payload",
    "hole_obstruction":             "Hole obstruction",
    "damaged_screw_thread":         "Damaged screw thread",
    "external_arm_disturbance":     "Arm disturbance",
    "missing_peg":                  "Missing peg",
    "collision_cardboard_object":   "Collision (cardboard)",
    "unstable_mounting_platform":   "Unstable mounting",
    "collision_rigid_object":       "Collision (rigid)",
    "peg_surface_contamination":    "Peg surface contam.",
    "gripper_release_during_motion": "Gripper release",
    "damaged_plate_thread":         "Damaged plate thread",
    "peg_insertion_misalignment":   "Peg misalignment",
}
PARAMETER_DISPLAY = {
    "payload_misconfiguration":     "Payload mass",
    "payload_cog_misconfiguration": "Payload CoG",
    "tcp_frame_misconfiguration":   "TCP frame",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.labelsize":    13,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   13,
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


def load_records(questions_dir=None, replies_dir=None):
    """Join GPT-5.1's Level-4 replies to their questions.

    Replies are matched to questions on the file stem rather than through the
    prompt file's ``qa_pair_id``: the Lite runs name a reply after the question
    it came from (``q_00007_7_answer.json`` for ``q_00007.json``), so the join
    needs no prompt directory and no id lookup.
    """
    qdir = pathlib.Path(questions_dir) if questions_dir else REPO / "output/test_eval/questions/level4"
    rdir = pathlib.Path(replies_dir) if replies_dir else REPO / "output/test_eval/replies/level4/gpt-5_1-1"

    qmap = {}
    for f in glob.glob(str(qdir / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        qmap[pathlib.Path(f).stem] = d

    records = []
    for f in sorted(glob.glob(str(rdir / "*_answer.json"))):
        a = json.load(open(f, encoding="utf-8"))
        if a.get("llm_judge_score") is None:
            continue
        stem = "_".join(pathlib.Path(f).name.split("_")[:2])
        q = qmap.get(stem)
        if q is None:
            continue
        records.append({
            "template_type": q["template_type"],
            "root_cause":    q.get("root_cause"),
            "judge_score":   a["llm_judge_score"],
        })
    return records


def stacked_bar(ax, buckets, key_order, display_map, title, min_n=2,
                other_label="Other (low n)"):
    """Render a horizontal stacked bar of {0, 0.5, 1} fractions per key.

    Keys with count < min_n are aggregated into a single "Other" bar.
    """
    main_keys = [k for k in key_order if len(buckets[k]) >= min_n]
    other_keys = [k for k in key_order if 0 < len(buckets[k]) < min_n]

    rows = []
    for k in main_keys:
        scores = buckets[k]
        if np.mean(scores) == 0.0:
            continue  # drop bars that are 100% wrong
        rows.append((display_map.get(k, k), scores))
    if other_keys:
        merged = []
        for k in other_keys:
            merged.extend(buckets[k])
        if merged and np.mean(merged) > 0.0:
            rows.append((f"{other_label}\n({len(other_keys)} types)", merged))

    rows = sorted(rows, key=lambda r: -np.mean(r[1]))

    labels = [r[0] + f"\n(n={len(r[1])})" for r in rows]
    means = [float(np.mean(r[1])) for r in rows]
    score_vals = [0.0, 0.5, 1.0]
    colors     = [GUNMETAL, STEEL, TIGER]
    series_lab = ["0.0  Wrong", "0.5  Partial", "1.0  Correct"]

    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for sv, color, slab in zip(score_vals, colors, series_lab):
        vals = np.array([
            sum(s == sv for s in r[1]) / len(r[1]) * 100 for r in rows
        ])
        ax.barh(y, vals, left=left, color=color, label=slab,
                height=0.6, edgecolor="white", linewidth=0.6)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 7:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=11, color="white", fontweight="bold")
        left += vals

    # Annotate overall mean accuracy at the right edge of each bar.
    for i, m in enumerate(means):
        ax.text(106.5, i, f"{m * 100:.1f}%", ha="center", va="center",
                fontsize=12, color=TIGER, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 113)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Proportion of items (%)")
    # Header above the right-edge per-category mean column
    ax.text(106.5, -0.7, "Score", ha="center", va="bottom",
            fontsize=12, color=TIGER, fontweight="bold")
    ax.set_title(title, fontweight="bold", loc="left", color=GUNMETAL, pad=18)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color=STEEL)
    ax.set_axisbelow(True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions-dir")
    ap.add_argument("--replies-dir")
    args = ap.parse_args()
    set_style()
    records = load_records(args.questions_dir, args.replies_dir)
    print(f"loaded {len(records)} GPT-5.1 L4 records")

    troubleshooting = defaultdict(list)
    optimization = defaultdict(list)
    for r in records:
        if r["template_type"] == "troubleshooting":
            troubleshooting[r["root_cause"]].append(r["judge_score"])
        elif r["template_type"] == "optimization":
            optimization[r["root_cause"]].append(r["judge_score"])

    print("troubleshooting buckets:",
          [(k, len(v), round(np.mean(v), 3)) for k, v in troubleshooting.items()])
    print("optimization buckets:",
          [(k, len(v), round(np.mean(v), 3)) for k, v in optimization.items()])

    ts_all = [s for vs in troubleshooting.values() for s in vs]
    op_all = [s for vs in optimization.values() for s in vs]
    ts_mean = float(np.mean(ts_all)) * 100 if ts_all else 0.0
    op_mean = float(np.mean(op_all)) * 100 if op_all else 0.0

    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.35},
    )
    stacked_bar(
        axes[0], troubleshooting,
        list(ROOT_CAUSE_DISPLAY.keys()), ROOT_CAUSE_DISPLAY,
        f"(a) L4.1 Troubleshooting   (mean {ts_mean:.1f}%)",
        min_n=2,
    )
    stacked_bar(
        axes[1], optimization,
        list(PARAMETER_DISPLAY.keys()), PARAMETER_DISPLAY,
        f"(b) L4.2 Optimization   (mean {op_mean:.1f}%)",
        min_n=1,
    )
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_paper = REPO / "docs/neurips_tex/figures/fig_l4_gpt51_byrootcause.pdf"
    fig.savefig(out_paper, bbox_inches="tight")
    print(f"saved {out_paper}")


if __name__ == "__main__":
    main()
