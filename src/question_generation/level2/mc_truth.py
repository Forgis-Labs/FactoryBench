from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

EPS = 1e-9
MC_IDS = [f"mc_{i:03d}" for i in range(1, 20)] + ["mc_030", "mc_031", "mc_032"]


def _canonical_statement_id(statement_id: str) -> str:
    text = str(statement_id).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"l2_mc_{int(match.group(1)):03d}"

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "speed_drop_ratio": 0.30,
    "speed_stable_tol": 0.10,
    "stall_current_increase": 0.25,
    "stall_speed_frac": 0.10,
    "force_low_increase": 0.20,
    "force_low_coverage": 0.80,
    "force_spike_increase": 0.40,
    "tracking_increase": 0.25,
    "tracking_stable_increase": 0.10,
    "vibration_spike": 0.50,
    "vibration_nominal_band": 0.15,
    "vibration_nominal_coverage": 0.90,
    "current_peak_increase": 2.50,
    "current_relax_drop": 0.10,
    "robot_current_stable_range": 0.10,
    "robot_current_increase": 0.20,
    "tcp_tracking_stable_increase": 0.10,
    "tcp_tracking_increase": 0.25,
    "temp_rise_slope": 0.005,
    "temp_rise_min_axes": 2.0,
    "temp_stable_slope": 0.002,
    "temp_stable_axes_ratio": 0.80,
    "no_effect_agg_increase": 0.10,
    # Absolute-magnitude threshold for the tracking statements. The
    # ratio-to-baseline form collapses on this corpus: post/pre tracking error
    # is so concentrated that only 20 of 78 windows flip at ANY threshold, so
    # no constant can split the population. The absolute post-event error has
    # real spread and its median splits exactly 50/50. Centre fitted by
    # scripts/calibrate_l2_tmpl3_thresholds.py.
    "tracking_abs_error": 0.0630,
    # All centres below are fitted so the true-rate averages 50% over the
    # THRESHOLD DISTRIBUTION the generator samples (Gaussian, 20% relative sd),
    # not at a single fixed point. Calibrating at a point leaves the marginal
    # off by 5-7 points once sampling spread and distribution skew are folded
    # in, so each item still varies while the aggregate stays balanced.
    #
    # Motion / effort magnitude statements (mc_030..mc_032). Built on channels
    # every robot in the corpus records, and on absolute magnitudes rather than
    # ratios to a pre-event baseline, which is what keeps their distributions
    # wide enough for a constant to split them. Centres are population medians
    # measured over KUKA event windows; see the cross-robot caveat in
    # scripts/calibrate_l2_tmpl3_thresholds.py.
    "joint_excursion_deg": 67.0,
    "joint_path_deg": 104.4,
    "torque_p2p_nm": 106.6,
}


# ---------------------------------------------------------------------------
# Per-robot threshold centres
#
# A single global constant cannot serve this corpus. The same statement's
# statistic differs by two orders of magnitude between robots: the tracking
# error centre that splits aursad 50/50 is 0.00064, while UR needs 0.125, a
# 195x gap. Fitting one pooled constant produced 0% true on aursad and 95% on
# UR, averaging to a 50% that described no actual robot and would have shipped
# looking calibrated.
#
# Centres below are fitted per dataset so the MARGINAL true-rate over the
# sampled threshold distribution is 50% on that robot. Fitted by
# scripts/calibrate_l2_tmpl3_thresholds.py over windows pooled from every
# dataset. Robots absent from this table fall back to DEFAULT_THRESHOLDS.
PER_ROBOT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "aursad": {
        "tracking_abs_error": 0.000642,
        "joint_excursion_deg": 2.324,
        "joint_path_deg": 2.326,
        "torque_p2p_nm": 5.199,
        "force_spike_increase": 0.075190,
        "force_low_increase": 0.002001,
        "robot_current_increase": 0.126,
        "tcp_tracking_increase": 9.514,
    },
    "factorywave_kuka": {
        "tracking_abs_error": 0.062870,
        "current_peak_increase": 2.617,
        "joint_excursion_deg": 64.98,
        "joint_path_deg": 104.5,
        "torque_p2p_nm": 106.2,
    },
    "factorywave_ur": {
        "tracking_abs_error": 0.125100,
        "joint_excursion_deg": 59.68,
        "joint_path_deg": 215.5,
        "speed_stable_tol": 16.60,
        "force_spike_increase": 1.793,
        "force_low_increase": 0.272,
        "tcp_tracking_increase": 0.047640,
    },
}

# Statements that will not calibrate on a given robot even with a per-robot
# centre, and are therefore withheld from its option pool. Leaving them in
# hands the solver a near-constant letter.
#
#   mc_012  two-knob AND (peak AND relax); the peak knob alone cannot be tuned.
#           85% on aursad, 100% on UR at any centre. Only KUKA calibrates.
#   mc_003  runs to the grid edge: 95% aursad, 100% UR.
#   mc_004  23% on aursad; fine on UR.
#   mc_014  27% on UR; fine on aursad.
#   mc_005  stall (current AND speed) never co-occurs: 0% aursad, 18% UR.
#   mc_013  robot_current range: 0% on both robots that record the channel.
#   mc_015  7% on aursad.
#   mc_019  "nothing extraordinary" is true on 95% of aursad windows.
ROBOT_DISABLED_MC: Dict[str, set] = {
    "aursad": {"mc_003", "mc_004", "mc_005", "mc_012", "mc_013", "mc_015", "mc_019"},
    "factorywave_kuka": {"mc_003"},
    "factorywave_ur": {"mc_003", "mc_005", "mc_012", "mc_013", "mc_014"},
}


def robot_key(episode_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    """Identify which robot an episode came from, for threshold lookup.

    Prefers an explicit ``dataset`` tag, otherwise composes the normalized
    metadata's ``data_source`` and ``robot_type`` (e.g. factorywave + kuka).
    """
    if not isinstance(episode_metadata, dict):
        return None
    explicit = episode_metadata.get("dataset")
    if explicit:
        return str(explicit)
    source = episode_metadata.get("data_source")
    robot = episode_metadata.get("robot_type")
    if source and robot:
        return f"{source}_{robot}"
    return str(source) if source else None


# ``robot_key`` composes data_source + robot_type and so yields
# "factorywave_ur3", while both tables below were written as "factorywave_ur".
# Nothing raised on the mismatch: the lookups just missed and every UR3 episode
# silently took the corpus defaults. That covered 84% of the L2 corpus, so
# neither the fitted centres nor the disable list ever applied to it, which is
# why mc_003 sat at 1.00 and mc_012 at 0.89 despite both being listed as
# disabled for this robot.
_ROBOT_ALIASES: Dict[str, str] = {
    "factorywave_ur3": "factorywave_ur",
    "factorywave_ur3e": "factorywave_ur",
    "factorywave_ur5": "factorywave_ur",
}


def _canonical_robot(robot: Optional[str]) -> Optional[str]:
    if not robot:
        return None
    key = str(robot)
    return _ROBOT_ALIASES.get(key, key)


def thresholds_for_robot(robot: Optional[str]) -> Dict[str, float]:
    """DEFAULT_THRESHOLDS overlaid with this robot's fitted centres."""
    merged = dict(DEFAULT_THRESHOLDS)
    canonical = _canonical_robot(robot)
    if canonical:
        merged.update(PER_ROBOT_THRESHOLDS.get(canonical, {}))
    return merged


def is_statement_enabled(option_id: str, robot: Optional[str]) -> bool:
    """False when the statement cannot be calibrated on this robot."""
    if not robot:
        return True
    return _canonical_statement_id(option_id).replace("l2_", "") not in ROBOT_DISABLED_MC.get(
        _canonical_robot(robot), set())


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

    if sid == "l2_mc_003":
        drop_ratio = _get_threshold(thresholds, "speed_drop_ratio")
        evaluated = 0
        for speed_key in speed_keys:
            pre = _to_float(baseline.get(speed_key))
            series = _series(post_event_rows, speed_key)
            if pre is None or not series:
                continue
            evaluated += 1
            if min(series) <= (1.0 - drop_ratio) * pre:
                return True
        # No speed channel carried data (KUKA records none), so whether a
        # speed drop occurred is unknown. Returning False here would assert
        # "no drop" on every such episode regardless of the threshold.
        if evaluated == 0:
            return None
        return False

    if sid == "l2_mc_004":
        stable_tol = _get_threshold(thresholds, "speed_stable_tol")
        evaluated = 0
        for speed_key in speed_keys:
            pre = _to_float(baseline.get(speed_key))
            series = _series(post_event_rows, speed_key)
            if pre is None or not series:
                continue
            evaluated += 1
            max_abs_diff = max(abs(v - pre) for v in series)
            if max_abs_diff > stable_tol * max(EPS, abs(pre)):
                return False
        if evaluated == 0:
            return None
        return True

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
        abs_error = _get_threshold(thresholds, "tracking_abs_error")
        post_err = _mean(_tracking_errors(post_event_rows, tcp=False))
        if post_err is None:
            return None
        if sid == "l2_mc_008":
            return post_err >= abs_error
        return post_err < abs_error

    if sid == "l2_mc_012":
        peak_increase = _get_threshold(thresholds, "current_peak_increase")
        relax_drop = _get_threshold(thresholds, "current_relax_drop")
        split = max(1, int(0.4 * len(post_event_rows)))
        early = post_event_rows[:split]
        evaluated = 0
        for current_key in current_keys:
            pre = _to_float(baseline.get(current_key))
            early_series = _series(early, current_key)
            post_series = _series(post_event_rows, current_key)
            if pre is None or not early_series or not post_series:
                continue
            evaluated += 1
            peak = max(early_series)
            final = post_series[-1]
            if peak >= (1.0 + peak_increase) * pre and final <= (1.0 - relax_drop) * peak:
                return True
        if evaluated == 0:
            return None
        return False

    if sid in {"l2_mc_013", "l2_mc_014"}:
        stable_range = _get_threshold(thresholds, "robot_current_stable_range")
        current_increase = _get_threshold(thresholds, "robot_current_increase")
        pre = _to_float(baseline.get("robot_current"))
        series = _series(post_event_rows, "robot_current")
        if pre is None or not series:
            return None
        mean_series = _mean(series)
        if mean_series is None:
            return None
        if sid == "l2_mc_013":
            return (max(series) - min(series)) <= stable_range * max(EPS, abs(pre))
        return mean_series >= (1.0 + current_increase) * pre

    if sid in {"l2_mc_015", "l2_mc_016"}:
        tcp_stable = _get_threshold(thresholds, "tcp_tracking_stable_increase")
        tcp_increase = _get_threshold(thresholds, "tcp_tracking_increase")
        pre_err = _tracking_error_row(baseline, tcp=True)
        post_err = _mean(_tracking_errors(post_event_rows, tcp=True))
        if pre_err is None or post_err is None:
            return None
        if sid == "l2_mc_015":
            return post_err <= (1.0 + tcp_stable) * pre_err
        return post_err >= (1.0 + tcp_increase) * pre_err

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
