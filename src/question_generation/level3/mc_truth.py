from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

EPS = 1e-9
MC_IDS = [f"mc_{i:03d}" for i in range(1, 20)]


def _canonical_statement_id(statement_id: str) -> str:
    text = str(statement_id).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"l2_mc_{int(match.group(1)):03d}"

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "speed_drop_ratio": 0.525,
    "stall_current_increase": 0.025,
    "stall_speed_frac": 0.10,
    "force_low_increase": 0.20,
    "force_low_coverage": 0.80,
    "force_spike_increase": 0.40,
    "vibration_spike": 0.50,
    "vibration_nominal_band": 0.15,
    "vibration_nominal_coverage": 0.90,
    "current_relax_drop": 0.8,
    "temp_rise_slope": 2e-06,
    "temp_rise_min_axes": 2.0,
    "temp_stable_slope": 1.6e-08,
    "temp_stable_axes_ratio": 0.80,
    "no_effect_agg_increase": 8,
    # Keys the excursion statements need. They were only ever defined on the
    # level 2 side, so level 2 evaluated mc_030/031/032 against fitted centres
    # while level 3 could not evaluate them at all. Centres below are the level
    # 2 corpus-wide values; PER_ROBOT_THRESHOLDS overrides them per source.
    # Absolute-magnitude centre for the tracking statements, replacing the
    # ratio-to-baseline form. Fitted by the calibration script.
    "robot_current_abs": 0.75,
    "tcp_tracking_abs_error": 0.0025,
    "tracking_abs_error": 0.14175,
    "joint_excursion_deg": 67,
    "joint_path_deg": 114.84,
    "torque_p2p_nm": 106.6,
}

# Per-source centres fitted by scripts/calibrate_l3_tmpl3_thresholds.py against
# the windows level 3 actually samples. Level 2 has carried an equivalent table
# for some time; level 3 shipped with the unfitted defaults, which is most of
# why a solver that recognised the statement and ignored the signal scored
# 86.4% on held-out L3.3 slots against a 50% chance rate. Sources absent here
# fall back to DEFAULT_THRESHOLDS.
PER_ROBOT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "factorywave_kuka": {
        "current_relax_drop": 0.8,
        "joint_excursion_deg": 60.3,
        "joint_path_deg": 103.356,
        "temp_rise_slope": 2e-06,
        "temp_stable_slope": 8e-07,
        "torque_p2p_nm": 106.6,
        "tracking_abs_error": 0.063,
    },
    "factorywave_ur3": {
        "current_relax_drop": 0.8,
        "joint_excursion_deg": 67,
        "joint_path_deg": 114.84,
        "no_effect_agg_increase": 8,
        "robot_current_abs": 0.75,
        "speed_drop_ratio": 0.525,
        "stall_current_increase": 0.025,
        "tcp_tracking_abs_error": 0.0025,
        "temp_rise_slope": 2e-06,
        "temp_stable_slope": 1.6e-08,
        "tracking_abs_error": 0.1575,
    },
}


# Statements that are exact negations of one another. Each pair now shares a
# single threshold, so if both land in one item the answer at those two
# positions is always exactly one T and the model can score one of them for
# free by negating the other. The option builder must never draw both.
COMPLEMENT_PAIRS: tuple = (
    ("mc_003", "mc_004"),   # joint speed drops sharply / no joint does
    ("mc_008", "mc_009"),   # mean tracking error exceeds / stays below
    ("mc_013", "mc_014"),   # mean robot current below / exceeds
    ("mc_015", "mc_016"),   # TCP motion stays aligned / becomes misaligned
)

# Statements the threshold calibration could not bring inside [0.15, 0.85] on
# this corpus, measured over 600 sampled windows:
#   mc_005            stall never fires (0.00)
#   mc_006, mc_007    contact force: factorywave records no such channel
#   mc_010, mc_011    vibration: likewise absent
#   mc_017, mc_018    joint temperature is flat, median in-window range is 0
#   mc_032            commanded torque undeterminable on 88% of windows
# Keeping them would ship options whose truth value is fixed regardless of the
# episode, which is the defect the calibration exists to remove.
UNCALIBRATABLE_MC_IDS: frozenset = frozenset({
    "mc_005", "mc_006", "mc_007", "mc_010", "mc_011", "mc_017", "mc_018", "mc_032",
})


def complement_of(option_id: str) -> str | None:
    """The statement that is this one's exact negation, if any."""
    sid = str(option_id)
    for a, b in COMPLEMENT_PAIRS:
        if sid == a:
            return b
        if sid == b:
            return a
    return None


def thresholds_for(source: Optional[str]) -> Dict[str, float]:
    """DEFAULT_THRESHOLDS overlaid with this source's fitted centres."""
    merged = dict(DEFAULT_THRESHOLDS)
    if source:
        merged.update(PER_ROBOT_THRESHOLDS.get(str(source), {}))
    return merged


def _get_threshold(thresholds: Optional[Dict[str, float]], key: str) -> float:
    if thresholds and key in thresholds:
        try:
            return float(thresholds[key])
        except (TypeError, ValueError):
            pass
    return float(DEFAULT_THRESHOLDS[key])


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _baseline_row(subseries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not subseries:
        return None
    row = subseries[-1]
    return row if isinstance(row, dict) else None


def _series(post_event_rows: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in post_event_rows:
        if not isinstance(row, dict):
            continue
        value = _to_float(row.get(key))
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def _indexed_keys(prefix: str, rows: Iterable[Dict[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if isinstance(key, str) and key.startswith(prefix + "_"):
                tail = key[len(prefix) + 1 :]
                if tail.isdigit():
                    keys.add(key)
    return sorted(keys, key=lambda x: int(x.rsplit("_", 1)[1]))


def _safe_ratio(value: float, baseline: float) -> float:
    return value / max(EPS, abs(baseline))


def _vector_norm(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(math.sqrt(sum(v * v for v in values)))


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _tracking_error_row(row: Dict[str, Any], tcp: bool = False) -> Optional[float]:
    pos_prefix = "setpoint_tcp" if tcp else "setpoint_pos"
    fb_pos_prefix = "feedback_tcp" if tcp else "feedback_pos"
    spd_prefix = "setpoint_tcp_speed" if tcp else "setpoint_speed"
    fb_spd_prefix = "feedback_tcp_speed" if tcp else "feedback_speed"

    pos_keys = _indexed_keys(pos_prefix, [row])
    fb_pos_keys = _indexed_keys(fb_pos_prefix, [row])
    common_axes = []
    for key in pos_keys:
        axis = key.rsplit("_", 1)[1]
        fb_key = f"{fb_pos_prefix}_{axis}"
        if fb_key in fb_pos_keys:
            common_axes.append(axis)

    if not common_axes:
        return None

    errs: List[float] = []
    for axis in common_axes:
        sp = _to_float(row.get(f"{pos_prefix}_{axis}"))
        fb = _to_float(row.get(f"{fb_pos_prefix}_{axis}"))
        if sp is None or fb is None:
            continue
        err = abs(sp - fb)
        spd_sp = _to_float(row.get(f"{spd_prefix}_{axis}"))
        spd_fb = _to_float(row.get(f"{fb_spd_prefix}_{axis}"))
        if spd_sp is not None and spd_fb is not None:
            err += abs(spd_sp - spd_fb)
        errs.append(err)

    if not errs:
        return None
    return _mean(errs)


def _tracking_errors(rows: List[Dict[str, Any]], tcp: bool = False) -> List[float]:
    values: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _tracking_error_row(row, tcp=tcp)
        if value is not None:
            values.append(value)
    return values


def _contact_norm_row(row: Dict[str, Any]) -> Optional[float]:
    est_keys = _indexed_keys("est_contact_force", [row])
    true_keys = _indexed_keys("true_force", [row])
    use_keys = est_keys if est_keys else true_keys
    if not use_keys:
        return None

    vals: List[float] = []
    for key in use_keys:
        value = _to_float(row.get(key))
        if value is None:
            return None
        vals.append(value)
    return _vector_norm(vals)


def _contact_norms(rows: List[Dict[str, Any]]) -> List[float]:
    norms: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _contact_norm_row(row)
        if value is not None:
            norms.append(value)
    return norms


def _aggregate_error(rows: List[Dict[str, Any]], baseline_row: Optional[Dict[str, Any]]) -> Optional[float]:
    tr = _mean(_tracking_errors(rows, tcp=False))
    tcp_tr = _mean(_tracking_errors(rows, tcp=True))
    force_n = _mean(_contact_norms(rows))

    robot_term: Optional[float] = None
    if baseline_row is not None:
        pre_robot = _to_float(baseline_row.get("robot_current"))
        post_robot = _mean(_series(rows, "robot_current"))
        if pre_robot is not None and post_robot is not None:
            robot_term = abs(post_robot - pre_robot) / max(EPS, abs(pre_robot))

    terms = [x for x in [tr, tcp_tr, force_n, robot_term] if x is not None]
    if not terms:
        return None
    return float(sum(terms) / len(terms))


def evaluate_mc_statement(
    statement_id: str,
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
    episode_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    sid = _canonical_statement_id(statement_id)

    if sid == "l2_mc_020":
        if episode_metadata is None:
            return None
        success = episode_metadata.get("task_success")
        if success is None:
            return None
        return bool(success)

    baseline = _baseline_row(subseries)
    if baseline is None or not post_event_rows:
        return None

    if sid in {"l2_mc_003", "l2_mc_004", "l2_mc_005", "l2_mc_012"}:
        speed_keys = _indexed_keys("feedback_speed", [baseline])
        current_keys = _indexed_keys("effort_current", [baseline])

    if sid in {"l2_mc_003", "l2_mc_004"}:
        # Ratio of window means, not a single baseline row against the
        # post-event minimum. Speeds oscillate through zero during ordinary
        # motion, so min(post) sits at or below zero on almost every window and
        # the old form was True 100% of the time whatever the threshold. The
        # mean-to-mean ratio has real spread (p10 0.08, median 0.57, p90 1.70),
        # so a threshold separates the population. The two statements share it
        # and are exact complements, the shape level 2 gave the tracking pair.
        drop_ratio = _get_threshold(thresholds, "speed_drop_ratio")
        ratios = []
        for speed_key in speed_keys:
            pre = _mean([abs(v) for v in _series(subseries, speed_key)])
            post = _mean([abs(v) for v in _series(post_event_rows, speed_key)])
            if not pre or post is None:
                continue
            ratios.append(post / pre)
        # No speed channel carried data (KUKA records none), so whether a speed
        # drop occurred is unknown. Returning False would assert "no drop" on
        # every such episode regardless of the threshold.
        if not ratios:
            return None
        if sid == "l2_mc_003":
            return min(ratios) <= drop_ratio
        return min(ratios) > drop_ratio

    if sid == "l2_mc_005":
        current_increase = _get_threshold(thresholds, "stall_current_increase")
        speed_frac = _get_threshold(thresholds, "stall_speed_frac")
        evaluated = 0
        for current_key in current_keys:
            axis = current_key.rsplit("_", 1)[1]
            speed_key = f"feedback_speed_{axis}"
            pre_current = _to_float(baseline.get(current_key))
            pre_speed = _to_float(baseline.get(speed_key))
            post_current = _mean(_series(post_event_rows, current_key))
            post_speed_abs = _mean([abs(v) for v in _series(post_event_rows, speed_key)])
            if (
                pre_current is None
                or pre_speed is None
                or post_current is None
                or post_speed_abs is None
            ):
                continue
            evaluated += 1
            current_up = post_current >= (1.0 + current_increase) * pre_current
            speed_low = post_speed_abs <= speed_frac * max(EPS, abs(pre_speed))
            if current_up and speed_low:
                return True
        if evaluated == 0:
            return None
        return False

    if sid == "l2_mc_006":
        low_increase = _get_threshold(thresholds, "force_low_increase")
        low_coverage = _get_threshold(thresholds, "force_low_coverage")
        pre_norm = _contact_norm_row(baseline)
        post_norms = _contact_norms(post_event_rows)
        if pre_norm is None or not post_norms:
            return None
        ok = sum(1 for n in post_norms if n <= (1.0 + low_increase) * pre_norm)
        return (ok / len(post_norms)) >= low_coverage

    if sid == "l2_mc_007":
        spike_increase = _get_threshold(thresholds, "force_spike_increase")
        pre_norm = _contact_norm_row(baseline)
        post_norms = _contact_norms(post_event_rows)
        if pre_norm is None or not post_norms:
            return None
        return any(n >= (1.0 + spike_increase) * pre_norm for n in post_norms)

    if sid in {"l2_mc_008", "l2_mc_009"}:
        # Absolute post-event error, not a ratio to the pre-event value.
        # Level 2 moved to this form after finding the ratio collapses on this
        # corpus: post/pre tracking error is so concentrated that almost no
        # window flips at any threshold, which left these two at 0.07 and 0.95
        # on level 3. The absolute error has real spread and its median splits
        # the population. The two statements are exact complements.
        abs_error = _get_threshold(thresholds, "tracking_abs_error")
        post_err = _mean(_tracking_errors(post_event_rows, tcp=False))
        if post_err is None:
            return None
        if sid == "l2_mc_008":
            return post_err >= abs_error
        return post_err < abs_error

    if sid == "l2_mc_012":
        # How far the peak relaxes, as a fraction of the peak itself. The old
        # form needed the peak to clear (1+r) x a single baseline sample and
        # the tail to fall (1-d) below that peak; the baseline sample makes the
        # first leg near-automatic and the statement sat at 0.92. The relax
        # fraction is self-normalising (p10 0.71, median 0.80, p90 0.93).
        relax_drop = _get_threshold(thresholds, "current_relax_drop")
        fractions = []
        for current_key in current_keys:
            series = _series(post_event_rows, current_key)
            if len(series) < 3:
                continue
            peak = max(abs(v) for v in series)
            if peak <= EPS:
                continue
            fractions.append((peak - abs(series[-1])) / peak)
        if not fractions:
            return None
        return max(fractions) >= relax_drop

    if sid in {"l2_mc_013", "l2_mc_014"}:
        # Absolute mean current, not a spread or a rise measured against one
        # baseline sample. Both old forms keyed off that sample and pinned at
        # 0.82 and 0.07; the post-event mean splits cleanly (p10 0.69, median
        # 0.75, p90 0.79). Shared threshold, exact complements.
        abs_current = _get_threshold(thresholds, "robot_current_abs")
        mean_series = _mean(_series(post_event_rows, "robot_current"))
        if mean_series is None:
            return None
        if sid == "l2_mc_014":
            return mean_series >= abs_current
        return mean_series < abs_current

    if sid in {"l2_mc_015", "l2_mc_016"}:
        # The TCP analogue of the tracking pair level 2 already converted.
        # Ratio to the baseline row left these at 0.94 and 0.07; the absolute
        # post-event error spreads over an order of magnitude (p10 0.0011,
        # median 0.0025, p90 0.0131). Shared threshold, exact complements.
        abs_error = _get_threshold(thresholds, "tcp_tracking_abs_error")
        post_err = _mean(_tracking_errors(post_event_rows, tcp=True))
        if post_err is None:
            return None
        if sid == "l2_mc_016":
            return post_err >= abs_error
        return post_err < abs_error

    if sid in {"l2_mc_017", "l2_mc_018"}:
        temp_rise_slope = _get_threshold(thresholds, "temp_rise_slope")
        temp_rise_min_axes = int(round(_get_threshold(thresholds, "temp_rise_min_axes")))
        temp_stable_slope = _get_threshold(thresholds, "temp_stable_slope")
        temp_stable_axes_ratio = _get_threshold(thresholds, "temp_stable_axes_ratio")
        temp_keys = _indexed_keys("joint_temp", [baseline])
        if not temp_keys:
            return None
        rises = 0
        stable = 0
        valid = 0
        for key in temp_keys:
            pre = _to_float(baseline.get(key))
            series = _series(post_event_rows, key)
            if pre is None or len(series) < 2:
                continue
            valid += 1
            slope = (series[-1] - series[0]) / max(EPS, abs(pre))
            if slope > temp_rise_slope:
                rises += 1
            if abs(slope) <= temp_stable_slope:
                stable += 1
        if valid == 0:
            return None
        if sid == "l2_mc_017":
            return rises >= temp_rise_min_axes
        return (stable / valid) >= temp_stable_axes_ratio

    if sid in {"l2_mc_030", "l2_mc_031", "l2_mc_032"}:
        # Ported from the level 2 evaluator, which has carried these three
        # since the excursion statements were added. Level 3 never had them,
        # so they fell through to None and the option builder turned that into
        # a confident False: "at least one joint sweeps more than N degrees"
        # and "the most active joint accumulates more than N degrees" were
        # False in 100% of released L3.3 items, whatever the episode did.
        if sid == "l2_mc_032":
            family, key = "effort_target_torque", "torque_p2p_nm"
        else:
            family, key = "feedback_pos", ("joint_excursion_deg" if sid == "l2_mc_030" else "joint_path_deg")
        limit = _get_threshold(thresholds, key)
        best: Optional[float] = None
        for signal_key in _indexed_keys(family, [baseline]):
            series = _series(post_event_rows, signal_key)
            if len(series) < 2:
                continue
            if sid == "l2_mc_031":
                # total path travelled, so a joint that oscillates back to
                # where it started still registers the motion it did
                value = sum(abs(series[i + 1] - series[i]) for i in range(len(series) - 1))
            else:
                # peak-to-peak span, for both excursion and torque variation
                value = max(series) - min(series)
            best = value if best is None else max(best, value)
        if best is None:
            return None
        return best > limit

    if sid == "l2_mc_019":
        agg_increase = _get_threshold(thresholds, "no_effect_agg_increase")
        safety_values = [_to_float(row.get("safety_mode")) for row in post_event_rows]
        observed = [v for v in safety_values if v is not None]
        # Distinguish "safety_mode says something abnormal" (a real False) from
        # "this robot does not report safety_mode at all" (unknown). Folding the
        # second into False made the statement a guaranteed F on those robots.
        if not observed:
            return None
        if any(int(v) != 1 for v in observed):
            return False

        pre_agg = _aggregate_error([baseline], baseline)
        post_agg = _aggregate_error(post_event_rows, baseline)
        if pre_agg is None or post_agg is None:
            return None
        return post_agg <= (1.0 + agg_increase) * pre_agg

    return None


def evaluate_all_mc_statements(
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    statement_ids: Optional[List[str]] = None,
    thresholds_by_id: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Optional[bool]]:
    ids = statement_ids or MC_IDS
    return {
        sid: evaluate_mc_statement(
            sid,
            subseries=subseries,
            post_event_rows=post_event_rows,
            thresholds=(thresholds_by_id or {}).get(sid),
        )
        for sid in ids
    }
