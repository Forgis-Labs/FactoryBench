#!/usr/bin/env python
"""Publication figures for the FactoryBench linear-probing experiment.

Reads output/probing/qwen3_4b_rerun/results.json and emits two vector PDFs to
output/figures/:

  fig_probe_vs_readout.pdf  -- headline. Per-concept linear-probe accuracy p vs the
                               model's own behavioural accuracy a (and the raw-signal
                               baseline + chance). Anomaly: decodable (p~0.83) but
                               unusable (a~0.37, always answers "yes").
  fig_probe_layerwise.pdf   -- per-layer probe accuracy across depth (0..36, `last`
                               read-out) for anomaly and fault_family, with the
                               behavioural floor a=0.37 drawn across every depth;
                               inset panel contrasts task_phase last (1.0, leakage)
                               vs mean-over-time-series (~0.59).

Reproducible:
    python scripts/probing/make_paper_figures.py

Numbers are the CV-selected best-layer accuracies (test-reported).

Colours follow a fixed, colorblind-safe palette:
  warm oranges  -> representation / linear probe (the "hero")
  gunmetal      -> ink (axes, text) and the fault_family / deep-structure series
  steel grey    -> neutral references (chance, raw-signal baseline)
  blue          -> behavioural accuracy a (what the model actually answers)
  green         -> positive control ("good" signal), used sparingly
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

# --- Fixed colorblind-safe palette (roles) -----------------------------------
ORANGE  = "#FF5A00"   # linear probe p (representation, hero)
FIRE    = "#FF4D00"   # fire            -- warm emphasis
FLICKER = "#DC4B07"   # flicker         -- secondary warm tone
GUNMETAL = "#122128"  # gunmetal        -- ink (axes/text) + fault_family series
STEEL   = "#878F92"   # steel grey      -- chance / raw-signal baseline / neutral
BLUE    = "#3A7BD5"   # blue            -- behavioural accuracy a
GREEN   = "#2ECC71"   # green           -- positive control ("good" signal)
PANEL_BG = "#F0F4F8"  # panelbg         -- subtle surface fill (unused by default)
AMBER   = "#FFF3CD"   # warm_amber      -- subtle highlight band
INK = GUNMETAL

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "output" / "probing" / "qwen3_4b_rerun2" / "results.json"
RANDINIT = REPO / "output" / "probing" / "qwen3_4b_rerun2_randinit_results.json"
OUTDIR = REPO / "output" / "figures"


def _chance(c):
    """Chance = uniform random guessing, 1/k (k = number of classes), matching the
    main paper's chance-correction convention (E = 1/k). This is a genuine
    per-sample random-guess baseline, not the majority-class no-information rate."""
    return 1.0 / c["n_classes"]


def _rand_best(name, ro="last"):
    """Random-init control: best CV-selected probe accuracy for a concept, or None."""
    try:
        rc = json.load(open(RANDINIT))["concepts"][name]
        arr = rc["layers"][ro]
        return max(arr, key=lambda e: e.get("cv_acc", e["linear_acc"]))["linear_acc"]
    except Exception:
        return None


def _rand_curve(name, ro="last"):
    """Random-init per-layer accuracy curve (aligned to layer index)."""
    try:
        arr = json.load(open(RANDINIT))["concepts"][name]["layers"][ro]
        return (np.array([e["layer"] for e in arr]),
                np.array([e["linear_acc"] for e in arr]))
    except Exception:
        return None, None


def rcparams():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 8,
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def load():
    d = json.load(open(RESULTS))
    return d["concepts"]


# =============================================================================
# Figure 1 -- headline: represented but not used
# =============================================================================
def fig_probe_vs_readout(C, out):
    order = ["anomaly", "fault_family"]
    def _ntest(name):
        c = C[name]
        return c.get("n_test") or (c.get("behavioural") or {}).get("n") or c["n_items"]
    labels = [f"fault presence\n(healthy vs. fault)\n$n={_ntest('anomaly')}$ test items",
              f"fault type\n(6-way)\n$n={_ntest('fault_family')}$ test items"]

    def probe(name):  # CV-selected best-layer test acc + its 95% CI
        bl = C[name]["best_linear"]["last"]
        return bl["linear_acc"], bl["ci95"]

    p = {n: probe(n) for n in order}
    raw = {n: C[n]["raw_input_acc"] for n in order}
    rnd = {n: _rand_best(n) for n in order}          # random-init control
    chance = {n: _chance(C[n]) for n in order}
    beh = {n: C[n]["behavioural"]["acc"] for n in order if C[n].get("behavioural")}

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.985, bottom=0.135)

    x = np.arange(len(order))
    YMAX = 1.22
    ax.axvspan(-0.6, 0.6, color=AMBER, alpha=0.55, zorder=0, lw=0)

    def bars_for(n):
        # (kind, value, facecolor, hatch, ci, label-colour)
        L = [("raw", raw[n], STEEL, None, None, STEEL)]
        if rnd[n] is not None:
            L.append(("rnd", rnd[n], "none", "////", None, STEEL))
        acc, ci = p[n]
        L.append(("probe", acc, ORANGE, None, ci, FLICKER))
        if n in beh:
            L.append(("beh", beh[n], BLUE, None, C[n]["behavioural"].get("ci95"), BLUE))
            nc = C[n]["behavioural"].get("negated_control")
            if nc:  # show BOTH framings so we don't foreground the worse one
                L.append(("beh2", nc["acc"], "none", "\\\\\\\\", nc.get("ci95"), BLUE))
        return L

    def vlab(xx, yy, s, color, dy=0.015):
        ax.text(xx, yy + dy, s, ha="center", va="bottom", fontsize=5.6, color=color,
                zorder=6, bbox=dict(boxstyle="round,pad=0.08", fc="white",
                                    ec="none", alpha=0.9))

    for i, n in enumerate(order):
        B = bars_for(n); m = len(B); w = 0.86 / m
        for j, (kind, val, fc, hatch, ci, lc) in enumerate(B):
            xx = x[i] + (j - (m - 1) / 2.0) * w
            ax.bar(xx, val, w * 0.9, color=(fc if fc != "none" else "none"),
                   edgecolor=(lc if fc == "none" else "white"), hatch=hatch,
                   linewidth=(0.9 if fc == "none" else 0.6), zorder=3)
            if ci:
                ax.errorbar(xx, val, yerr=[[val - ci[0]], [ci[1] - val]], fmt="none",
                            ecolor=INK, elinewidth=0.7, capsize=2, capthick=0.7, zorder=5)
            vlab(xx, ci[1] if ci else val, f"{val:.2f}", lc)
        ax.plot([x[i] - 0.46, x[i] + 0.46], [chance[n]] * 2, ls=(0, (4, 2)),
                color=INK, lw=0.9, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, YMAX)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlim(-0.62, len(order) - 0.38)
    ax.grid(axis="y", color=STEEL, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor=STEEL, label="raw-signal baseline (no model)"),
        Patch(facecolor="none", edgecolor=STEEL, hatch="////", label="random-init model (control)"),
        Patch(facecolor=ORANGE, label="trained probe $p$ (decodable?)"),
        Patch(facecolor=BLUE, label="behavioural $a$ (benchmark framing)"),
        Patch(facecolor="none", edgecolor=BLUE, hatch="\\\\\\\\", label="behavioural, re-framed (healthy?)"),
        Line2D([0], [0], color=INK, ls=(0, (4, 2)), lw=0.9, label="chance $= 1/k$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.995),
              ncol=1, fontsize=5.9, handlelength=1.4, labelspacing=0.38,
              borderaxespad=0.2)

    fig.savefig(out / "fig_probe_vs_readout.pdf")
    plt.close(fig)


# =============================================================================
# Figure 2 -- per-layer decodability across depth
# =============================================================================
def fig_probe_layerwise(C, out):
    fig = plt.figure(figsize=(5.5, 2.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    def curve(name, ro="last"):
        arr = C[name]["layers"][ro]
        L = np.array([e["layer"] for e in arr])
        acc = np.array([e["linear_acc"] for e in arr])
        lo = np.array([e["linear_ci95"][0] for e in arr])
        hi = np.array([e["linear_ci95"][1] for e in arr])
        return L, acc, lo, hi

    # ---- Panel A: anomaly & fault_family (last read-out) -----------------
    La, aa, alo, ahi = curve("anomaly")
    Lf, af, flo, fhi = curve("fault_family")
    beh = C["anomaly"]["behavioural"]["acc"]
    ch_a = _chance(C["anomaly"])
    ch_f = _chance(C["fault_family"])

    # random-init control curves (same architecture, untrained weights)
    Lar, aar = _rand_curve("anomaly")
    Lfr, afr = _rand_curve("fault_family")

    axA.fill_between(La, alo, ahi, color=ORANGE, alpha=0.13, lw=0, zorder=1)
    axA.fill_between(Lf, flo, fhi, color=GUNMETAL, alpha=0.09, lw=0, zorder=1)
    axA.plot(La, aa, "-o", color=ORANGE, ms=2.7, lw=1.4, zorder=5,
             label="fault presence (trained)")
    axA.plot(Lf, af, "-s", color=GUNMETAL, ms=2.6, lw=1.3, zorder=4,
             label="fault type (trained)")
    if aar is not None:
        axA.plot(Lar, aar, ls=(0, (3, 2)), color=ORANGE, lw=1.0, alpha=0.75,
                 zorder=3, label="fault presence (random-init)")
    if afr is not None:
        axA.plot(Lfr, afr, ls=(0, (3, 2)), color=GUNMETAL, lw=1.0, alpha=0.75,
                 zorder=3, label="fault type (random-init)")

    # Only the behavioural floor is drawn here (per-concept chance = 1/k is shown
    # in Fig.~1); two extra chance hlines would collide with the fault curves.
    axA.axhline(beh, ls="--", color=BLUE, lw=1.1, zorder=3)
    axA.text(35.5, beh - 0.006, f"fault-presence behavioural $a={beh:.2f}$", va="top",
             ha="right", fontsize=5.9, color=BLUE)

    # Honest message: fault_family's trained curve tracks its random-init curve,
    # i.e. no learned gain; the CV-selected best is shallow, not deep.
    fbl = C["fault_family"]["best_linear"]["last"]
    axA.annotate("trained probe tracks\nrandom-init: no learned gain",
                 xy=(24, af[24] if len(af) > 24 else af[-1]), xytext=(11, 0.30),
                 fontsize=5.9, color=GUNMETAL, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=GUNMETAL, lw=0.5))

    axA.set_xlabel("Layer  (0 = embedding $\\ldots$ 36)")
    axA.set_ylabel("Linear-probe accuracy")
    axA.set_xlim(-1, 37)
    axA.set_ylim(0.20, 1.0)
    axA.set_xticks([0, 6, 12, 18, 24, 30, 36])
    axA.grid(color=STEEL, alpha=0.20, lw=0.5)
    axA.set_axisbelow(True)
    axA.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=1,
               fontsize=5.4, handlelength=1.6, labelspacing=0.28)
    axA.set_title("(a)  depth trajectory: trained vs. random-init",
                  fontsize=7.2, fontweight="bold", loc="left", pad=3)

    # ---- Panel B: task_phase leakage -------------------------------------
    Ll, la, *_ = curve("task_phase", "last")
    Lm, ma, mlo, mhi = curve("task_phase", "mean")
    ch_t = _chance(C["task_phase"])
    tp_last = C["task_phase"]["best_linear"]["last"]["linear_acc"]   # CV-selected
    tp_mean = C["task_phase"]["best_linear"]["mean"]["linear_acc"]

    axB.fill_between(Lm, mlo, mhi, color=GUNMETAL, alpha=0.09, lw=0, zorder=1)
    axB.plot(Ll, la, "-o", color=ORANGE, ms=2.5, lw=1.3, zorder=5)
    axB.plot(Lm, ma, "-s", color=GUNMETAL, ms=2.3, lw=1.2, zorder=4)
    axB.axhline(ch_t, ls=":", color=STEEL, lw=1.0, zorder=2)
    axB.text(35.5, ch_t + 0.007, "chance", va="bottom", ha="right", fontsize=5.6,
             color=STEEL)

    # direct labels (only two series -> no legend box)
    axB.text(19, 0.955, "last token\n(prompt text)", color=ORANGE, fontsize=6.0,
             ha="center", va="top")
    axB.text(30, 0.66, "mean over\ntime-series", color=GUNMETAL, fontsize=6.0,
             ha="center", va="bottom")
    axB.text(1.5, 0.885, f"${tp_last:.2f}$ from wording,\n${tp_mean:.2f}$ from "
             "signal:\nlexical leakage,\nexcluded", fontsize=6.0, color=INK,
             ha="left", va="top")

    axB.set_xlabel("Layer")
    axB.set_ylabel("Linear-probe accuracy")
    axB.set_xlim(-1, 37)
    axB.set_ylim(0.20, 1.05)
    axB.set_xticks([0, 12, 24, 36])
    axB.grid(color=STEEL, alpha=0.20, lw=0.5)
    axB.set_axisbelow(True)
    axB.set_title("(b)  task phase (leakage control)", fontsize=7.5,
                  fontweight="bold", loc="left", pad=3)

    fig.savefig(out / "fig_probe_layerwise.pdf")
    plt.close(fig)


def main():
    rcparams()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    C = load()
    fig_probe_vs_readout(C, OUTDIR)
    fig_probe_layerwise(C, OUTDIR)
    for f in ["fig_probe_vs_readout.pdf", "fig_probe_layerwise.pdf"]:
        pth = OUTDIR / f
        print(f"wrote {pth}  ({pth.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
