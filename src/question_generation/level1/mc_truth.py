from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def interpolate_value(t_target: float, t1: float, v1: float, t2: float, v2: float) -> float:
    """Linear interpolation between two values."""
    if t2 == t1:
        return v1
    alpha = (t_target - t1) / (t2 - t1)
    return (1 - alpha) * v1 + alpha * v2


_MACHINE_METADATA_CACHE: List[Dict[str, Any]] = None


def load_machine_metadata() -> List[Dict[str, Any]]:
    """Load machines.json from the data directory."""
    global _MACHINE_METADATA_CACHE
    if _MACHINE_METADATA_CACHE is not None:
        return _MACHINE_METADATA_CACHE

    machines_path = Path("data/labelling/machines.json")
    try:
        with open(machines_path, "r", encoding="utf-8") as f:
            _MACHINE_METADATA_CACHE = json.load(f)
    except Exception:
        _MACHINE_METADATA_CACHE = []
    return _MACHINE_METADATA_CACHE


def get_machine_by_id(machine_id: int) -> Optional[Dict[str, Any]]:
    metadata = load_machine_metadata()
    for m in metadata:
        if m.get("machine_id") == machine_id:
            return m
    return None


def find_bracketing_indices(rows: List[Dict[str, Any]], t_ms: float) -> Tuple[Optional[int], Optional[int]]:
    """Find indices of samples bracketing time t_ms. Returns (before_idx, after_idx)."""
    timestamps = [(i, r.get("timestamp_ms")) for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    
    if not timestamps:
        return None, None
    
    # Find exact match
    for i, ts in timestamps:
        if ts == t_ms:
            return i, i
    
    # Find bracketing
    before_idx, after_idx = None, None
    for i, ts in timestamps:
        if ts <= t_ms:
            before_idx = i
        if ts >= t_ms and after_idx is None:
            after_idx = i
            break
            
    # if it's past the end, snap to last
    if before_idx is not None and after_idx is None:
        after_idx = before_idx
    # if it's before the start, snap to first
    if before_idx is None and after_idx is not None:
        before_idx = after_idx
        
    return before_idx, after_idx


def interpolate_signal_at_time(
    rows: List[Dict[str, Any]],
    key: str,
    t_ms: float,
) -> Tuple[Optional[float], Optional[str]]:
    """Interpolate a signal at a given time. Returns (value, mode) or (None, None)."""
    before_idx, after_idx = find_bracketing_indices(rows, t_ms)
    if before_idx is None or after_idx is None:
        return None, None
    try:
        if before_idx == after_idx:
            return float(rows[before_idx][key]), "exact"
        v1 = float(rows[before_idx][key])
        v2 = float(rows[after_idx][key])
        t1 = float(rows[before_idx]["timestamp_ms"])
        t2 = float(rows[after_idx]["timestamp_ms"])
        return interpolate_value(t_ms, t1, v1, t2, v2), "interpolated"
    except (KeyError, TypeError, ValueError):
        return None, None


def get_wrench_components_at_time(
    rows: List[Dict[str, Any]],
    t_ms: float,
    indices: List[int],
) -> Tuple[Optional[List[float]], Optional[str], Optional[str]]:
    """Get wrench components (force/torque) at time, preferring external sensor then estimate."""
    # Try external sensor first
    values = []
    modes = []
    for idx in indices:
        val, mode = interpolate_signal_at_time(rows, f"true_force_{idx}", t_ms)
        if val is None:
            values = []
            break
        values.append(val)
        modes.append(mode)
    if values:
        interp_mode = "exact" if all(m == "exact" for m in modes) else "interpolated"
        return values, "external sensor", interp_mode

    # Fallback to controller estimate
    values = []
    modes = []
    for idx in indices:
        val, mode = interpolate_signal_at_time(rows, f"est_contact_force_{idx}", t_ms)
        if val is None:
            return None, None, None
        values.append(val)
        modes.append(mode)
    interp_mode = "exact" if all(m == "exact" for m in modes) else "interpolated"
    return values, "controller estimate", interp_mode


def answer_q1_position_check(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_1: float
) -> Dict[str, Any]:
    pos_key = f"feedback_pos_{axis}"
    val_t1, mode_t1 = interpolate_signal_at_time(rows, pos_key, t1_ms)
    val_t2, mode_t2 = interpolate_signal_at_time(rows, pos_key, t2_ms)
    
    if val_t1 is None or val_t2 is None:
        return {"answer": "Unknown", "reasoning": "Missing signal values"}

    delta_q = abs(val_t2 - val_t1)
    moved = delta_q > eps_1
    is_same = not moved
    return {
        "answer": "Yes" if is_same else "No", 
        "reasoning": f"Δq={delta_q:.6f} > {eps_1} -> moved={moved}",
        "raw_value": delta_q,
        "is_true": is_same
    }


def answer_q2_friction_increase(
    rows: List[Dict[str, Any]],
    t1_ms: float,
    t2_ms: float,
    axis: int,
    eps_2: float,
    delta_1_ms: int = 500,
) -> Dict[str, Any]:
    speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
    current_key = f"effort_current_{axis}"
    
    timestamps = [r.get("timestamp_ms") for r in rows if r.get("timestamp_ms") is not None]
    if not timestamps:
        return {"answer": "Unknown", "reasoning": "No valid timestamps"}
        
    def compute_friction_proxy(w_start: float, w_end: float) -> Optional[float]:
        window_rows = [r for r in rows if r.get("timestamp_ms") is not None and w_start <= r["timestamp_ms"] <= w_end]
        if len(window_rows) < 3:
            return None
            
        ratios = []
        for r in window_rows:
            try:
                v = float(r[speed_key])
                i = float(r[current_key])
                if abs(v) > 1e-6:
                    ratios.append(abs(i) / abs(v))
            except (KeyError, TypeError, ValueError):
                continue
                
        if not ratios:
            return None
        return float(np.median(ratios))

    f1 = compute_friction_proxy(t1_ms - delta_1_ms, t1_ms + delta_1_ms)
    f2 = compute_friction_proxy(t2_ms - delta_1_ms, t2_ms + delta_1_ms)
    
    if f1 is None or f2 is None:
        return {"answer": "Unknown", "reasoning": "Missing samples for proxy calculation"}

    ratio = f2 / f1 if f1 != 0 else float('inf')
    threshold_ratio = 1.0 + (eps_2 / 100.0)
    increased = ratio > threshold_ratio
    return {
        "answer": "Yes" if increased else "No", 
        "reasoning": f"f2/f1={ratio:.4f} > {threshold_ratio:.4f}",
        "raw_value": ratio,
        "f1": f1,
        "f2": f2,
        "is_true": increased
    }



def answer_q3_end_effector_accel(rows: List[Dict[str, Any]], t_ms: float, threshold: float = 1.0) -> Dict[str, Any]:
    vib0, _ = interpolate_signal_at_time(rows, "vibration_0", t_ms)
    vib1, _ = interpolate_signal_at_time(rows, "vibration_1", t_ms)
    vib2, _ = interpolate_signal_at_time(rows, "vibration_2", t_ms)
    
    if vib0 is None or vib1 is None or vib2 is None:
        return {"answer": "Unknown", "reasoning": "Missing vibration data"}
        
    G = 9.81
    a_x, a_y, a_z = vib0 * G, vib1 * G, vib2 * G
    magnitude = math.sqrt(a_x**2 + a_y**2 + a_z**2)
    
    axis_vals = [abs(a_x), abs(a_y), abs(a_z)]
    highest_idx = int(np.argmax(axis_vals))
    highest_axis = ["X", "Y", "Z"][highest_idx]

    return {
        "answer": f"{a_x:.4f}_{a_y:.4f}_{a_z:.4f}",
        "reasoning": f"Magnitude {magnitude:.4f} m/s^2. Axis max: {highest_axis}",
        "raw_value": [a_x, a_y, a_z],
        "magnitude": magnitude,
        "highest_axis": highest_axis,
        "is_above_threshold": magnitude > threshold
    }


def answer_q4_external_force(rows: List[Dict[str, Any]], t_ms: float, eps_3: float) -> Dict[str, Any]:
    values, source, _ = get_wrench_components_at_time(rows, t_ms, [0, 1, 2])
    if values is None:
        return {"answer": "Unknown", "reasoning": "Missing external force data"}

    fx, fy, fz = values
    magnitude = math.sqrt(fx**2 + fy**2 + fz**2)
    detected = magnitude >= eps_3
    return {
        "answer": "Yes" if detected else "No", 
        "reasoning": f"Magnitude {magnitude:.4f} >= {eps_3}",
        "raw_value": magnitude,
        "is_true": detected
    }


def answer_q5_joint_jerk(rows: List[Dict[str, Any]], t_ms: float, axis: int, threshold: float = 5.0) -> Dict[str, Any]:
    valid = [(i, r.get("timestamp_ms")) for i, r in enumerate(rows) if r.get("timestamp_ms") is not None]
    if len(valid) < 5:
        return {"answer": "Unknown", "reasoning": "Insufficient elements"}

    closest_pos = min(range(len(valid)), key=lambda p: abs(valid[p][1] - t_ms))
    if closest_pos < 2 or closest_pos + 2 >= len(valid):
        return {"answer": "Unknown", "reasoning": "Boundary limits"}

    idx_km2, t_km2 = valid[closest_pos - 2]
    idx_km1, t_km1 = valid[closest_pos - 1]
    idx_k, t_k = valid[closest_pos]
    idx_kp1, t_kp1 = valid[closest_pos + 1]
    idx_kp2, t_kp2 = valid[closest_pos + 2]

    speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
    try:
        v_km2 = float(rows[idx_km2][speed_key])
        v_k = float(rows[idx_k][speed_key])
        v_kp2 = float(rows[idx_kp2][speed_key])

        dt_k_km2 = (t_k - t_km2) / 1000.0
        dt_kp2_k = (t_kp2 - t_k) / 1000.0
        if dt_k_km2 <= 0 or dt_kp2_k <= 0:
            return {"answer": "Unknown", "reasoning": "Invalid timestamps"}

        accel_km1 = (v_k - v_km2) / dt_k_km2
        accel_kp1 = (v_kp2 - v_k) / dt_kp2_k

        dt_kp1_km1 = (t_kp1 - t_km1) / 1000.0
        if dt_kp1_km1 <= 0:
            return {"answer": "Unknown", "reasoning": "Invalid timestamp"}

        jerk = (accel_kp1 - accel_km1) / dt_kp1_km1
    except (KeyError, TypeError, ValueError):
        return {"answer": "Unknown", "reasoning": "Missing features"}

    jerk_mag = abs(jerk)
    if jerk_mag < 1.0:
        jerk_range = "Low"
    elif jerk_mag < 5.0:
        jerk_range = "Medium"
    else:
        jerk_range = "High"

    return {
        "answer": str(round(jerk, 4)), 
        "reasoning": f"Calculated central difference: {jerk:.4f}. Range: {jerk_range}",
        "raw_value": jerk,
        "jerk_range": jerk_range,
        "is_above_threshold": jerk_mag > threshold
    }


def answer_q6_torque_magnitude(rows: List[Dict[str, Any]], t_ms: float, axis_label: str, threshold: float = 2.0) -> Dict[str, Any]:
    axis_map = {"x": 3, "y": 4, "z": 5}
    if axis_label not in axis_map:
        return {"answer": "Unknown", "reasoning": "Invalid axis"}

    idx = axis_map[axis_label]
    values, _, _ = get_wrench_components_at_time(rows, t_ms, [idx])
    if values is None:
        return {"answer": "Unknown", "reasoning": "Missing data"}

    torque_mag = abs(values[0])
    return {
        "answer": str(round(torque_mag, 4)), 
        "reasoning": f"Calculated absolute torque: {torque_mag:.4f}",
        "raw_value": torque_mag,
        "is_above_threshold": torque_mag > threshold
    }




def get_num_joints(rows: List[Dict[str, Any]], signal_prefix: str = "feedback_pos_") -> int:
    """Detect the number of joints by inspecting columns in the first row."""
    if not rows:
        return 0
    first_row = rows[0]
    
    # Handle aliases
    prefixes = [signal_prefix]
    if signal_prefix == "feedback_speed_":
        prefixes.append("feedback_vel_")
    
    for pref in prefixes:
        count = 0
        while f"{pref}{count}" in first_row:
            count += 1
        if count > 0:
            return count
    return 0


def get_signal_key(row: Dict[str, Any], prefix: str, axis: int) -> str:
    """Get the key for a signal, handling aliases."""
    key = f"{prefix}{axis}"
    if key in row:
        return key
    
    # Aliases
    if prefix == "feedback_speed_":
        alt_key = f"feedback_vel_{axis}"
        if alt_key in row:
            return alt_key
    elif prefix == "vibration_":
        alt_key = f"auxiliary_accel_tool_{axis}"
        if alt_key in row:
            return alt_key
            
    return key


def answer_joint_speed(rows: List[Dict[str, Any]], t_ms: float, axis: int, threshold: float = 0.5) -> Dict[str, Any]:
    """Cat 8: Evaluate joint speed against a threshold."""
    speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
    val, _ = interpolate_signal_at_time(rows, speed_key, t_ms)
    if val is None:
        return {"answer": "Unknown", "reasoning": "Missing speed data"}

    abs_speed = abs(val)
    below = abs_speed < threshold
    return {
        "answer": str(round(val, 4)),
        "reasoning": f"Speed: {val:.4f} rad/s (abs: {abs_speed:.4f}). Threshold: {threshold}",
        "raw_value": val,
        "is_below_threshold": below,
        "is_true": below # For TF
    }


def answer_motor_current(rows: List[Dict[str, Any]], t_ms: float, axis: int, threshold: float = 1.0) -> Dict[str, Any]:
    """Cat 9: Evaluate motor current against a threshold."""
    current_key = f"effort_current_{axis}"
    val, _ = interpolate_signal_at_time(rows, current_key, t_ms)
    if val is None:
        return {"answer": "Unknown", "reasoning": "Missing current data"}

    abs_current = abs(val)
    exceeds = abs_current > threshold
    return {
        "answer": str(round(val, 4)),
        "reasoning": f"Current: {val:.4f} A (abs: {abs_current:.4f}). Threshold: {threshold}",
        "raw_value": val,
        "is_above_threshold": exceeds,
        "is_true": exceeds # For TF
    }



def answer_tracking_error(rows: List[Dict[str, Any]], t_ms: float, axis: int, threshold: float = 0.01, t2_ms: Optional[float] = None) -> Dict[str, Any]:
    """Cat 7: Evaluate tracking error (setpoint vs feedback). Supports one or two timestamps."""
    def get_error(t):
        sp_key = f"setpoint_pos_{axis}"
        fb_key = f"feedback_pos_{axis}"
        val_sp, _ = interpolate_signal_at_time(rows, sp_key, t)
        val_fb, _ = interpolate_signal_at_time(rows, fb_key, t)
        if val_sp is None or val_fb is None:
            return None
        return abs(val_sp - val_fb)

    err1 = get_error(t_ms)
    if err1 is None:
        return {"answer": "Unknown", "reasoning": "Missing position data"}

    if t2_ms is not None:
        err2 = get_error(t2_ms)
        if err2 is None:
            return {"answer": "Unknown", "reasoning": "Missing position data at t2"}
        
        # Comparison for MC
        diff = err2 - err1
        eps = 1e-4 # small stability threshold
        if diff > eps:
            trend = "increased"
        elif diff < -eps:
            trend = "decreased"
        else:
            trend = "stable"
        
        return {
            "answer": trend,
            "reasoning": f"Error at {t_ms}ms: {err1:.4f}, at {t2_ms}ms: {err2:.4f}. Trend: {trend}",
            "trend": trend,
            "err1": err1,
            "err2": err2
        }

    above = err1 > threshold
    return {
        "answer": str(round(err1, 4)),
        "reasoning": f"Tracking Error: {err1:.4f}. Threshold: {threshold}",
        "raw_value": err1,
        "is_above_threshold": above,
        "is_true": above # For TF
    }


def answer_joint_comparison(rows: List[Dict[str, Any]], t_ms: float, signal_prefix: str) -> Dict[str, Any]:
    """Generic MC logic for 'Which joint has the highest absolute X at t?'."""
    num_joints = get_num_joints(rows, signal_prefix)
    if num_joints == 0:
        return {"answer": "Unknown", "reasoning": "No joints detected"}
    
    values = []
    for axis in range(num_joints):
        key = get_signal_key(rows[0], signal_prefix, axis)
        val, _ = interpolate_signal_at_time(rows, key, t_ms)
        if val is None:
            return {"answer": "Unknown", "reasoning": f"Missing data for joint {axis} (key: {key})"}
        values.append((axis, abs(val)))
    
    # Sort descending by absolute value
    values.sort(key=lambda x: x[1], reverse=True)
    
    highest_joint = values[0][0]
    return {
        "answer": str(highest_joint),
        "reasoning": f"Highest absolute value found at joint {highest_joint}. Values: {values}",
        "highest_joint": highest_joint,
        "sorted_values": values
    }


def answer_q8_joint_speed_check(rows: List[Dict[str, Any]], t_ms: float, machine_id: int, joints_list: List[int]) -> Dict[str, Any]:
    """Cat 8 (Semantic): Multi-select speed limit check for 4 joints."""
    machine = get_machine_by_id(machine_id)
    if not machine or "joint_speed_limits" not in machine:
        return {"answer": "Unknown", "reasoning": f"No speed limit metadata for machine {machine_id}"}

    limits = machine["joint_speed_limits"]
    results = []
    reasoning_parts = []
    
    for axis in joints_list:
        speed_key = get_signal_key(rows[0], "feedback_speed_", axis)
        val, _ = interpolate_signal_at_time(rows, speed_key, t_ms)
        if val is None:
            return {"answer": "Unknown", "reasoning": f"Missing speed data for joint {axis}"}
        
        limit = limits[axis] if axis < len(limits) else 3.14
        is_within = abs(val) <= limit
        results.append("T" if is_within else "F")
        reasoning_parts.append(f"J{axis}: |{val:.2f}| <= {limit}")

    answer = "".join(results)
    return {
        "answer": answer,
        "reasoning": "; ".join(reasoning_parts),
        "raw_results": results
    }


def answer_q9_joint_current_check(rows: List[Dict[str, Any]], t_ms: float, axis: int, machine_id: int) -> Dict[str, Any]:
    """Cat 9 (Semantic): Single-select current limit check."""
    machine = get_machine_by_id(machine_id)
    if not machine or "rated_current_per_joint" not in machine:
        return {"answer": "Unknown", "reasoning": f"No current metadata for machine {machine_id}"}

    limits = machine["rated_current_per_joint"]
    limit = limits[axis] if axis < len(limits) else 2.0
    
    current_key = f"effort_current_{axis}"
    val, _ = interpolate_signal_at_time(rows, current_key, t_ms)
    if val is None:
        return {"answer": "Unknown", "reasoning": "Missing current data"}

    is_within = abs(val) <= limit
    return {
        "answer": "Yes" if is_within else "No",
        "reasoning": f"Current |{val:.2f}A| <= {limit}A for joint {axis}.",
        "is_true": is_within
    }


def answer_q10_signal_description(rows: List[Dict[str, Any]], t1_ms: float, t2_ms: float, axis: int, signal_name: str) -> Dict[str, Any]:
    """Cat 10 (Semantic/Rule-based): Free-form description of signal behavior."""
    # Deterministic characterizer
    prefix_map = {
        "effort_current": "effort_current_",
        "feedback_speed": "feedback_speed_",
        "feedback_pos": "feedback_pos_"
    }
    key = get_signal_key(rows[0], prefix_map.get(signal_name, "feedback_pos_"), axis)
    
    window_rows = [r for r in rows if r.get("timestamp_ms") is not None and t1_ms <= r["timestamp_ms"] <= t2_ms]
    if len(window_rows) < 5:
        return {"answer": "Unknown", "reasoning": "Too few samples in window"}
        
    vals = [float(r[key]) for r in window_rows if key in r]
    if not vals:
        return {"answer": "Unknown", "reasoning": "No valid data for signal"}
        
    v_start, v_end = vals[0], vals[-1]
    v_min, v_max = min(vals), max(vals)
    delta = v_end - v_start
    range_val = v_max - v_min
    
    # Generic description logic
    label = signal_name.replace("_", " ")
    if abs(delta) < 0.01 * (abs(v_min) + 1e-6) or range_val < 1e-4:
        desc = f"The {label} for joint {axis} remains stable around {v_start:.2f} units throughout the interval."
    elif delta > 0:
        desc = f"The {label} for joint {axis} shows an increasing trend, rising from {v_start:.2f} to {v_end:.2f} units."
    else:
        desc = f"The {label} for joint {axis} decreases from {v_start:.2f} down to {v_end:.2f} units across the window."
        
    if range_val > 5 * abs(delta) and range_val > 0.1:
        desc += f" It exhibits significant fluctuations with a peak value of {v_max:.2f}."

    return {
        "answer": desc,
        "reasoning": "Rule-based characterization of window statistics.",
        "desc": desc
    }
