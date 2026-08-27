"""Compute analysis artifacts from a TSFM results file.

Defaults reproduce the original Chronos-Bolt behavior when called with no
arguments. Pass --results / --prefix to analyze a different results file
(e.g. TimesFM) and write parallel outputs.

Outputs (named with the configured prefix; default ""):
  - {prefix}analysis_summary.json: per-template aggregates, bootstrap 95% CIs,
    chance-corrected scores, exact-match L1.7 variant, signal-type breakdown,
    horizon breakdown, timing.
  - {prefix}fig_per_template.png
  - {prefix}fig_distribution.png
  - {prefix}fig_signal_breakdown.png
  - {prefix}fig_horizon_breakdown.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

CHANCE = 0.25
RNG_SEED = 12345


def chance_correct(raw: float) -> float:
    return max(0.0, (raw - CHANCE) / (1.0 - CHANCE))


def bootstrap_ci(scores, n_boot: int = 1000, alpha: float = 0.05):
    """Bootstrap percentile CI for the mean of `scores`."""
    rng = np.random.default_rng(RNG_SEED)
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return (None, None)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(arr[idx].mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def signal_category(channel: str) -> str:
    """Map a raw channel name to a signal-type bucket."""
    c = channel.lower()
    if "feedback_pos" in c or "setpoint_pos" in c:
        return "position"
    if "feedback_speed" in c or "setpoint_speed" in c:
        return "speed"
    if "torque" in c or "effort" in c:
        return "torque/effort"
    if "contact_force" in c:
        return "contact force"
    return "other"


def primary_signal(r: dict) -> str:
    chs = r.get("target_channels") or []
    if not chs:
        return "?"
    # Strip joint index for tensor: "feedback_pos_0" -> "feedback_pos"
    if r.get("answer_format") == "tensor":
        # 6 channels share a base; use base
        base = re.sub(r"_\d+$", "", chs[0])
        return base
    return chs[0]


def exact_match_l17(results):
    """Re-score L1.7 numerical items under exact-match (|pred-gt|<1e-4)."""
    rescored = []
    for r in results:
        if r.get("level") != 1 or r.get("template_id") != 7:
            continue
        if "score" not in r:
            continue
        pred = r.get("prediction")
        gt = r.get("ground_truth")
        if pred is None or gt is None:
            continue
        s = float(abs(float(pred) - float(gt)) < 1e-4)
        rescored.append(s)
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=HERE / "results_full.json",
        help="Path to the per-item results JSON to analyze.",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix prepended to each output artifact "
             "(empty => match the original Chronos artifact names).",
    )
    args = parser.parse_args()

    summary_path = HERE / f"{args.prefix}analysis_summary.json"
    fig_per_template_path = HERE / f"{args.prefix}fig_per_template.png"
    fig_distribution_path = HERE / f"{args.prefix}fig_distribution.png"
    fig_signal_path = HERE / f"{args.prefix}fig_signal_breakdown.png"
    fig_horizon_path = HERE / f"{args.prefix}fig_horizon_breakdown.png"

    data = json.loads(args.results.read_text())
    results = data["results"]

    summary = {
        "model": data["aggregate"].get("model"),
        "device": data["aggregate"].get("device"),
        "n_total": len(results),
        "total_inference_time_s": data["aggregate"].get("total_inference_time_s"),
        "templates": {},
        "signal_breakdown": {},
        "horizon_breakdown": {},
    }

    # ---------- Per-template ----------
    template_keys = [(1, 7, "L1.7"), (2, 4, "L2.4"), (2, 5, "L2.5")]
    for level, tid, tag in template_keys:
        rows = [r for r in results if r.get("level") == level and r.get("template_id") == tid]
        scores = [r["score"] for r in rows if "score" in r]
        if not scores:
            continue
        raw_mean = float(np.mean(scores))
        raw_std = float(np.std(scores))
        cc = chance_correct(raw_mean)
        lo, hi = bootstrap_ci(scores)
        cc_lo = chance_correct(lo) if lo is not None else None
        cc_hi = chance_correct(hi) if hi is not None else None
        summary["templates"][tag] = {
            "n": len(rows),
            "raw_mean": round(raw_mean, 4),
            "raw_std": round(raw_std, 4),
            "raw_ci_lo": round(lo, 4),
            "raw_ci_hi": round(hi, 4),
            "cc_mean": round(cc, 4),
            "cc_ci_lo": round(cc_lo, 4),
            "cc_ci_hi": round(cc_hi, 4),
            "scores": scores,
        }

    # L1.7 exact-match variant
    em = exact_match_l17(results)
    if em:
        em_mean = float(np.mean(em))
        em_lo, em_hi = bootstrap_ci(em)
        summary["L1.7_exact_match"] = {
            "n": len(em),
            "raw_mean": round(em_mean, 4),
            "raw_ci_lo": round(em_lo, 4),
            "raw_ci_hi": round(em_hi, 4),
            "cc_mean": round(chance_correct(em_mean), 4),
            "note": "Strict exact-match (|pred-gt|<1e-4), matching the LLM L1.7 fallback when no margin field exists.",
        }

    # ---------- Signal-type breakdown ----------
    by_cat: dict[str, list[float]] = {}
    for r in results:
        if "score" not in r:
            continue
        chs = r.get("target_channels") or []
        if not chs:
            continue
        cat = signal_category(chs[0])
        by_cat.setdefault(cat, []).append(r["score"])
    for cat, sc in by_cat.items():
        lo, hi = bootstrap_ci(sc)
        summary["signal_breakdown"][cat] = {
            "n": len(sc),
            "raw_mean": round(float(np.mean(sc)), 4),
            "raw_ci_lo": round(lo, 4),
            "raw_ci_hi": round(hi, 4),
        }

    # ---------- Horizon breakdown ----------
    by_h: dict[int, list[float]] = {}
    for r in results:
        if "score" not in r:
            continue
        h = r.get("horizon_steps")
        if h is None:
            continue
        by_h.setdefault(int(h), []).append(r["score"])
    for h in sorted(by_h):
        sc = by_h[h]
        lo, hi = bootstrap_ci(sc)
        summary["horizon_breakdown"][str(h)] = {
            "n": len(sc),
            "raw_mean": round(float(np.mean(sc)), 4),
            "raw_ci_lo": round(lo, 4),
            "raw_ci_hi": round(hi, 4),
        }

    # ---------- Timing ----------
    times = [r.get("inference_time_s", 0.0) for r in results if "inference_time_s" in r]
    summary["timing"] = {
        "mean_per_item_s": round(float(np.mean(times)), 4) if times else None,
        "median_per_item_s": round(float(np.median(times)), 4) if times else None,
        "max_per_item_s": round(float(np.max(times)), 4) if times else None,
    }

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")

    # ---------- Plots ----------
    plt.style.use("default")

    # Per-template raw + CC with CI bars
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    tags = ["L1.7", "L2.4", "L2.5"]
    colors = ["#4A6FA5", "#C9A227", "#FF5A00"]

    raw_means = [summary["templates"][t]["raw_mean"] for t in tags]
    raw_los = [summary["templates"][t]["raw_ci_lo"] for t in tags]
    raw_his = [summary["templates"][t]["raw_ci_hi"] for t in tags]
    cc_means = [summary["templates"][t]["cc_mean"] for t in tags]
    cc_los = [summary["templates"][t]["cc_ci_lo"] for t in tags]
    cc_his = [summary["templates"][t]["cc_ci_hi"] for t in tags]
    ns = [summary["templates"][t]["n"] for t in tags]

    raw_err = [
        [m - lo for m, lo in zip(raw_means, raw_los)],
        [hi - m for m, hi in zip(raw_means, raw_his)],
    ]
    cc_err = [
        [m - lo for m, lo in zip(cc_means, cc_los)],
        [hi - m for m, hi in zip(cc_means, cc_his)],
    ]

    bars1 = axes[0].bar(tags, raw_means, yerr=raw_err, color=colors, edgecolor="black",
                        linewidth=0.5, capsize=4)
    axes[0].axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance (E=0.25)")
    axes[0].set_ylabel("Raw Score")
    axes[0].set_title("Raw Accuracy (95% bootstrap CI)")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="upper right")
    for bar, n in zip(bars1, ns):
        axes[0].text(bar.get_x() + bar.get_width() / 2, 0.02,
                     f"n={n}", ha="center", fontsize=8, color="white", fontweight="bold")

    bars2 = axes[1].bar(tags, cc_means, yerr=cc_err, color=colors, edgecolor="black",
                        linewidth=0.5, capsize=4)
    axes[1].set_ylabel("Chance-Corrected Score")
    axes[1].set_title("Chance-Corrected Accuracy")
    axes[1].set_ylim(0, 1.05)
    for bar, n in zip(bars2, ns):
        axes[1].text(bar.get_x() + bar.get_width() / 2, 0.02,
                     f"n={n}", ha="center", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_per_template_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Score distribution (histogram)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (level, tid, tag) in zip(axes, template_keys):
        rows = [r for r in results if r.get("level") == level and r.get("template_id") == tid]
        scores = [r["score"] for r in rows if "score" in r]
        ax.hist(scores, bins=np.linspace(0, 1, 11), edgecolor="black",
                color="#FF5A00", alpha=0.75)
        ax.set_title(f"{tag} (n={len(scores)})", fontsize=10)
        ax.set_xlabel("Per-item score")
        ax.set_ylabel("Count")
        ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(fig_distribution_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Signal-type breakdown
    cats = sorted(summary["signal_breakdown"].keys())
    means = [summary["signal_breakdown"][c]["raw_mean"] for c in cats]
    nss = [summary["signal_breakdown"][c]["n"] for c in cats]
    los = [summary["signal_breakdown"][c]["raw_ci_lo"] for c in cats]
    his = [summary["signal_breakdown"][c]["raw_ci_hi"] for c in cats]
    err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cats, means, yerr=err, color="#4A6FA5", edgecolor="black", capsize=4)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance (E=0.25)")
    ax.set_ylabel("Mean raw score")
    ax.set_title("Score by signal category (95% CI)")
    ax.set_ylim(0, 1.05)
    for bar, n in zip(bars, nss):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f"n={n}", ha="center",
                fontsize=8, color="white", fontweight="bold")
    ax.legend(fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_signal_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Horizon breakdown
    horizons = sorted(int(k) for k in summary["horizon_breakdown"].keys())
    hmeans = [summary["horizon_breakdown"][str(h)]["raw_mean"] for h in horizons]
    hns = [summary["horizon_breakdown"][str(h)]["n"] for h in horizons]
    hlos = [summary["horizon_breakdown"][str(h)]["raw_ci_lo"] for h in horizons]
    hhis = [summary["horizon_breakdown"][str(h)]["raw_ci_hi"] for h in horizons]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(horizons, hmeans, yerr=[
        [m - lo for m, lo in zip(hmeans, hlos)],
        [hi - m for m, hi in zip(hmeans, hhis)],
    ], fmt="o-", color="#FF5A00", capsize=3, markersize=6)
    for h, m, n in zip(horizons, hmeans, hns):
        ax.annotate(f"n={n}", (h, m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="gray")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance (E=0.25)")
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel("Mean raw score")
    ax.set_title("Score vs forecast horizon (95% CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_horizon_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figures: {fig_per_template_path.name}, "
          f"{fig_distribution_path.name}, {fig_signal_path.name}, "
          f"{fig_horizon_path.name}")

    # Print a compact summary
    print("\n=== SUMMARY ===")
    for tag in tags:
        t = summary["templates"][tag]
        print(f"{tag}: n={t['n']}  raw={t['raw_mean']:.3f} "
              f"[{t['raw_ci_lo']:.3f},{t['raw_ci_hi']:.3f}]  "
              f"cc={t['cc_mean']:.3f}")
    if "L1.7_exact_match" in summary:
        e = summary["L1.7_exact_match"]
        print(f"L1.7 (exact-match): n={e['n']}  raw={e['raw_mean']:.3f} "
              f"[{e['raw_ci_lo']:.3f},{e['raw_ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
