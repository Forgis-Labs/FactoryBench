#!/usr/bin/env python3
"""Fit level 3's multi-select thresholds to the data, per source.

L3.2 and L3.3 ("Given the counterfactual scenario where {event}, select all
statements that would apply") judge four statements whose truth turns on
whether a measured quantity crosses a sampled threshold. Level 2 fitted those
thresholds to the corpus some time ago and keeps a per-robot table; level 3
never did, and drew every threshold from hand-picked defaults. Measured on the
release, the consequences compound:

  * three statement families are constant. "at least one joint sweeps more than
    N degrees" and "the most active joint accumulates more than N degrees" are
    False in 100% of items because level 3's evaluator had no handler for them
    at all and the builder scores an unevaluable statement as a confident
    False. "at least one joint speed drops sharply" is True in 100%.
  * the per-position rates sit near 0.30 rather than 0.50, so answering all-F
    scores 0.700 against a chance rate of 0.500.
  * a solver that recognises the statement family and ignores the signal
    entirely scores 86.4% on held-out slots.

This script sweeps each threshold over the windows the generator actually
samples and picks the value whose true-rate is closest to 50%, evaluating
through ``evaluate_mc_statement`` rather than reimplementing the statistic so
the calibration cannot drift from the scorer.

Windows are built the way level 3 builds them, which is not the way level 2
does: the context comes from the *baseline* episode around the event onset, and
the rows judged against it come from the *counterfactual* episode from the
onset onward. Calibrating on level 2's single-episode windows would fit the
wrong distribution.

It also reports the undeterminable rate per statement. That is the other half
of the imbalance and thresholds cannot fix it: a statement about a channel the
episode does not record evaluates to None, which the builder turns into False.
Those need excluding from the option pool for that episode, which is what
``utils.mc_availability`` now does at generation time.

Usage:
    python scripts/calibrate_l3_tmpl3_thresholds.py --episodes 300
    python scripts/calibrate_l3_tmpl3_thresholds.py --episodes 300 --write
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.question_generation.level3.level3 import (  # noqa: E402
    CONTEXT_MAX,
    CONTEXT_MIN,
    VALID_DATASETS,
    _first_timestamp_ms,
    _legacy_mc_option_id,
    discover_cf_episode_pairs,
    find_event_onset_index,
    normalize_timestamps,
    sample_window_around_index,
)
from src.question_generation.level3.mc_truth import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    evaluate_mc_statement,
)
from src.question_generation.level2.mc_truth import robot_key  # noqa: E402
from src.question_generation.utils.io import load_json  # noqa: E402

# Threshold key -> the statement whose truth it gates. Multi-knob statements
# list their magnitude knob only; the second knob (coverage / axis count)
# controls robustness to noise rather than how hard the statement is to
# satisfy, so moving it changes what the statement means.
STATEMENT_KEYS: Dict[str, str] = {
    "mc_003": "speed_drop_ratio",
    "mc_005": "stall_current_increase",
    "mc_006": "force_low_increase",
    "mc_007": "force_spike_increase",
    "mc_008": "tracking_abs_error",
    "mc_010": "vibration_spike",
    "mc_011": "vibration_nominal_band",
    "mc_012": "current_relax_drop",
    "mc_014": "robot_current_abs",
    "mc_016": "tcp_tracking_abs_error",
    "mc_017": "temp_rise_slope",
    "mc_018": "temp_stable_slope",
    "mc_019": "no_effect_agg_increase",
    "mc_030": "joint_excursion_deg",
    "mc_031": "joint_path_deg",
    "mc_032": "torque_p2p_nm",
}

# As multiples of the current default, wide enough that a badly-centred default
# can still be pulled to the median.
SWEEP_MULTIPLIERS = [
    0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0,
    1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0,
]
TARGET_TRUE_RATE = 0.50
MIN_DETERMINABLE = 20


def load_meta(ep_path: Path) -> Dict[str, Any]:
    meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
    if not meta_path.is_file():
        return {}
    try:
        meta = load_json(meta_path)
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


Window = Tuple[List[dict], List[dict], dict, Optional[str]]


def _rows(path: Path, split: str, fmt: str) -> Any:
    """The generator's loader, which is nested inside its own function."""
    raw = load_json(path)
    if isinstance(raw, dict):
        return raw.get("baseline" if fmt != "combined" else split, [])
    return raw


def collect_windows(datasets_dir: Path, n_windows: int, seed: int) -> List[Window]:
    """(baseline context, counterfactual post-event rows, metadata, source).

    Built the way level 3 builds them: pairs come from the same discovery the
    generator uses, the context is a window of the baseline run around the
    event onset, and the rows judged against it run from the onset onward in
    the counterfactual.
    """
    rng = random.Random(seed)
    pairs = discover_cf_episode_pairs(datasets_dir, list(VALID_DATASETS))
    rng.shuffle(pairs)

    windows: List[Window] = []
    for pair in pairs:
        if len(windows) >= n_windows:
            break
        fmt = pair.get("format", "flat")
        try:
            normal_rows = _rows(pair["non_alt_path"], "baseline", fmt)
            alt_rows = _rows(pair["alt_path"], "counterfactual", fmt)
        except Exception:
            continue
        if not isinstance(normal_rows, list) or not isinstance(alt_rows, list):
            continue
        if len(normal_rows) < CONTEXT_MIN or len(alt_rows) < CONTEXT_MIN:
            continue
        onset = find_event_onset_index(alt_rows)
        if onset is None:
            continue
        sampled = sample_window_around_index(
            normal_rows, center_index=onset,
            min_len=CONTEXT_MIN, max_len=CONTEXT_MAX, margin=5,
        )
        if sampled is None:
            continue
        subseries, _start, length = sampled
        if not subseries or onset - length + 1 < 0:
            continue
        post_event_rows = alt_rows[onset:]
        if not post_event_rows:
            continue
        base = _first_timestamp_ms(subseries)
        subseries = normalize_timestamps(subseries, base)
        post_event_rows = normalize_timestamps(post_event_rows, base)
        meta = load_meta(pair["alt_path"])
        windows.append((subseries, post_event_rows, meta, robot_key(meta)))
    return windows


def rate_at(mc_id: str, key: str, value: float, windows: List[Window],
            base: Dict[str, float]) -> Tuple[float, float, int]:
    """(true rate over determinable, undeterminable rate, n determinable)."""
    thresholds = dict(base)
    thresholds[key] = value
    true_n = det_n = none_n = 0
    for subseries, post_event_rows, meta, _src in windows:
        try:
            verdict = evaluate_mc_statement(
                _legacy_mc_option_id(mc_id),
                subseries=subseries,
                post_event_rows=post_event_rows,
                thresholds=thresholds,
                episode_metadata=meta,
            )
        except Exception:
            verdict = None
        if verdict is None:
            none_n += 1
            continue
        det_n += 1
        true_n += bool(verdict)
    total = det_n + none_n
    return (
        (true_n / det_n) if det_n else 0.0,
        (none_n / total) if total else 1.0,
        det_n,
    )


def fit(windows: List[Window], label: str) -> Dict[str, float]:
    """Best threshold per statement over one population of windows."""
    fitted: Dict[str, float] = {}
    print(f"\n=== {label}  ({len(windows)} windows) ===")
    print(f"  {'stmt':7s} {'key':28s} {'default':>10} {'fitted':>10} "
          f"{'rate@def':>9} {'rate@fit':>9} {'undet':>7} {'n':>5}")
    for mc_id, key in STATEMENT_KEYS.items():
        if key not in DEFAULT_THRESHOLDS:
            continue
        default = float(DEFAULT_THRESHOLDS[key])
        base_rate, undet, n_det = rate_at(mc_id, key, default, windows, DEFAULT_THRESHOLDS)
        if n_det < MIN_DETERMINABLE:
            print(f"  {mc_id:7s} {key:28s} {default:10.4g} {'skip':>10} "
                  f"{base_rate:9.2f} {'-':>9} {undet:7.0%} {n_det:5d}")
            continue
        best_value, best_rate, best_gap = default, base_rate, abs(base_rate - TARGET_TRUE_RATE)
        for mult in SWEEP_MULTIPLIERS:
            value = default * mult
            rate, _u, nd = rate_at(mc_id, key, value, windows, DEFAULT_THRESHOLDS)
            if nd < MIN_DETERMINABLE:
                continue
            gap = abs(rate - TARGET_TRUE_RATE)
            if gap < best_gap:
                best_value, best_rate, best_gap = value, rate, gap
        fitted[key] = best_value
        print(f"  {mc_id:7s} {key:28s} {default:10.4g} {best_value:10.4g} "
              f"{base_rate:9.2f} {best_rate:9.2f} {undet:7.0%} {n_det:5d}")
    return fitted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-dir", type=Path, default=Path("data"))
    ap.add_argument("--episodes", type=int, default=300,
                    help="Windows to collect (per source split afterwards).")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--min-per-source", type=int, default=40,
                    help="Sources with fewer windows fall back to the corpus fit.")
    ap.add_argument("--write", action="store_true",
                    help="Write PER_ROBOT_THRESHOLDS into level3/mc_truth.py.")
    args = ap.parse_args()

    windows = collect_windows(args.datasets_dir, args.episodes, args.seed)
    if not windows:
        raise SystemExit(f"no usable counterfactual windows under {args.datasets_dir}")

    by_source: Dict[str, List[Window]] = collections.defaultdict(list)
    for w in windows:
        if w[3]:
            by_source[w[3]].append(w)
    print(f"collected {len(windows)} windows; sources: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_source.items())))

    corpus = fit(windows, "corpus-wide")
    per_source: Dict[str, Dict[str, float]] = {}
    for source, group in sorted(by_source.items()):
        if len(group) < args.min_per_source:
            print(f"\n=== {source}: {len(group)} windows, below "
                  f"--min-per-source={args.min_per_source}, using the corpus fit ===")
            continue
        per_source[source] = fit(group, source)

    if not args.write:
        print("\ndry run. Re-run with --write to update level3/mc_truth.py.")
        print(json.dumps({"corpus": corpus, "per_source": per_source}, indent=2))
        return 0

    path = Path("src/question_generation/level3/mc_truth.py")
    text = path.read_text(encoding="utf-8")
    # corpus fit becomes the default centre, per-source overlays on top
    for key, value in corpus.items():
        pattern = rf'(^    "{re.escape(key)}": )[-+0-9.eE]+(,)$'
        text, n = re.subn(pattern, rf'\g<1>{value:.6g}\g<2>', text, count=1, flags=re.M)
        if not n:
            print(f"  warning: could not update default for {key}")
    body = "\n".join(
        f'    "{src}": {{\n'
        + "".join(f'        "{k}": {v:.6g},\n' for k, v in sorted(vals.items()))
        + "    },"
        for src, vals in sorted(per_source.items())
    )
    replacement = (
        "PER_ROBOT_THRESHOLDS: Dict[str, Dict[str, float]] = {\n" + body + "\n}"
        if per_source else
        "PER_ROBOT_THRESHOLDS: Dict[str, Dict[str, float]] = {}"
    )
    text, n = re.subn(
        r"PER_ROBOT_THRESHOLDS: Dict\[str, Dict\[str, float\]\] = \{.*?\n\}|"
        r"PER_ROBOT_THRESHOLDS: Dict\[str, Dict\[str, float\]\] = \{\}",
        replacement, text, count=1, flags=re.S,
    )
    if not n:
        raise SystemExit("could not locate PER_ROBOT_THRESHOLDS to replace")
    path.write_text(text, encoding="utf-8")
    print(f"\nwrote {len(corpus)} corpus centres and "
          f"{len(per_source)} per-source tables to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
