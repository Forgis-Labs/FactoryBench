#!/usr/bin/env python
"""Figures for the FactoryBench probing experiment.

Reads results.json produced by run_probing.py and emits:
  fig_probe_vs_answer.pdf   - per-concept best linear-probe acc vs behavioural acc
  fig_layerwise.pdf         - per-layer linear-probe accuracy + selectivity curves
  fig_linearity.pdf         - linear vs MLP probe (best layer), per concept
  summary_table.csv         - machine-readable summary

Usage:
    python plot_probes.py --results output/probing/qwen3_4b/results.json \
                          --out     output/probing/qwen3_4b
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORANGE = "#FF5A00"
NAVY = "#122128"
STEEL = "#878F92"
BLUE = "#3A7BD5"
GREEN = "#2ECC71"


def _best(concept, readout="last"):
    """Best layer by grouped-CV score (never test accuracy, which would be a
    layer-selection peek). Falls back to linear_acc only if cv_acc is absent."""
    arr = concept["layers"][readout]
    return max(arr, key=lambda e: e.get("cv_acc", e["linear_acc"]))


def fig_probe_vs_answer(results, out):
    concepts = results["concepts"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "--", color=STEEL, lw=1, zorder=1)
    for key, c in concepts.items():
        b = _best(c)
        p = b["linear_acc"]
        beh = c.get("behavioural")
        a = beh["acc"] if beh else None
        chance = c["chance_majority"]
        if a is None:
            # no behavioural: place on x-axis marker at chance for reference
            ax.scatter(chance, p, s=90, color=BLUE, edgecolor=NAVY, zorder=3, marker="s")
            ax.annotate(f"{key}\n(no beh.)", (chance, p), fontsize=7,
                        xytext=(5, 4), textcoords="offset points")
            continue
        ax.scatter(a, p, s=110, color=ORANGE, edgecolor=NAVY, zorder=3)
        ax.annotate(key, (a, p), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel("Behavioural accuracy $a$ (model answers the question)")
    ax.set_ylabel("Best linear-probe accuracy $p$ (concept decodable)")
    ax.set_title("Representation vs read-out\n(above diagonal = concept present but unused)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_probe_vs_answer.pdf")
    plt.close(fig)


def fig_layerwise(results, out):
    concepts = results["concepts"]
    n = len(concepts)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), squeeze=False)
    for j, (key, c) in enumerate(concepts.items()):
        ax = axes[0][j]
        arr = c["layers"]["last"]
        L = [e["layer"] for e in arr]
        lin = [e["linear_acc"] for e in arr]
        sel = [e["selectivity"] for e in arr]
        ax.plot(L, lin, "-o", ms=3, color=ORANGE, label="linear probe")
        ax.plot(L, sel, "-", color=BLUE, label="selectivity")
        if "randinit_acc" in arr[0]:
            ax.plot(L, [e["randinit_acc"] for e in arr], ":", color=STEEL,
                    label="random-init")
        ax.axhline(c["chance_majority"], ls="--", lw=0.8, color=NAVY, label="chance")
        beh = c.get("behavioural")
        if beh:
            ax.axhline(beh["acc"], ls="-.", lw=1, color=GREEN, label="behavioural $a$")
        ax.set_title(f"{key}", fontsize=10)
        ax.set_xlabel("layer"); ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel("accuracy")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Per-layer linear decodability", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_layerwise.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_linearity(results, out):
    concepts = results["concepts"]
    keys = list(concepts)
    lin, mlp = [], []
    for k in keys:
        b = _best(concepts[k])
        lin.append(b["linear_acc"])
        mlp.append(b.get("mlp_acc", b["linear_acc"]))
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(1.4 * len(keys) + 2, 3.6))
    ax.bar(x - 0.2, lin, 0.4, color=ORANGE, label="linear")
    ax.bar(x + 0.2, mlp, 0.4, color=NAVY, label="MLP")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("best-layer accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Linearity: linear vs non-linear probe")
    ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out / "fig_linearity.pdf")
    plt.close(fig)


def summary_csv(results, out):
    rows = []
    for key, c in results["concepts"].items():
        b = _best(c)
        beh = c.get("behavioural")
        rows.append({
            "concept": key, "level": c["level"], "n_items": c["n_items"],
            "n_classes": c["n_classes"], "chance": round(c["chance_majority"], 3),
            "best_layer": b["layer"],
            "linear_acc": round(b["linear_acc"], 3),
            "ci95_lo": round(b["linear_ci95"][0], 3),
            "ci95_hi": round(b["linear_ci95"][1], 3),
            "selectivity": round(b["selectivity"], 3),
            "mlp_acc": round(b.get("mlp_acc", float("nan")), 3),
            "randinit_acc": round(b.get("randinit_acc", float("nan")), 3),
            "raw_input_acc": round(c.get("raw_input_acc", float("nan")), 3),
            "behavioural_acc": round(beh["acc"], 3) if beh else "",
            "gap_p_minus_a": round(b["linear_acc"] - beh["acc"], 3) if beh else "",
        })
    with open(out / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--results-random", type=Path, default=None,
                    help="results.json from a --randomize-weights control run; "
                         "its linear_acc is merged in as randinit_acc")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = json.load(open(args.results))
    # Drop concepts that failed to probe (e.g. carry an "error" instead of layers)
    results["concepts"] = {k: c for k, c in results["concepts"].items()
                           if "layers" in c}
    if args.results_random and args.results_random.exists():
        rnd = json.load(open(args.results_random))
        for key, c in results["concepts"].items():
            rc = rnd["concepts"].get(key)
            if not rc:
                continue
            for ro in ("last", "mean"):
                for e, re_ in zip(c["layers"][ro], rc["layers"][ro]):
                    e["randinit_acc"] = re_["linear_acc"]
    fig_probe_vs_answer(results, args.out)
    fig_layerwise(results, args.out)
    fig_linearity(results, args.out)
    rows = summary_csv(results, args.out)
    print("wrote figures + summary_table.csv to", args.out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
