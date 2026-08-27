"""
Per-template feature category filtering for token reduction.

Defines feature categories by prefix and a retention policy per Level 1
template type.  Features belonging to categories NOT in the retention
list are dropped from the rows before encoding, reducing prompt size
by 50-70% while keeping distractor categories so the benchmark remains
valid (the model still has to identify which signals matter).
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Feature categories – each maps to a set of column-name prefixes
# ---------------------------------------------------------------------------

FEATURE_CATEGORY_PREFIXES: Dict[str, List[str]] = {
    "position": ["feedback_pos_", "setpoint_pos_"],
    "speed": ["feedback_speed_", "feedback_vel_", "setpoint_speed_"],
    "acceleration": ["setpoint_acc_"],
    "current": ["effort_current_", "effort_target_current_"],
    "torque": ["effort_target_torque_"],
    "force": ["est_contact_force_", "true_force_"],
    "vibration": ["vibration_", "auxiliary_accel_tool_"],
    "tcp": [
        "setpoint_tcp_",
        "feedback_tcp_",
        "setpoint_tcp_speed_",
        "feedback_tcp_speed_",
    ],
    "control": ["control_output_"],
    "metadata": [
        "joint_temp_",
        "joint_mode_",
        "robot_mode",
        "safety_mode",
        "execution_time",
        "tool_momentum",
        "main_voltage",
        "robot_voltage",
        "robot_current",
        "joint_voltage_",
        "speed_scaling",
        "target_speed_fraction",
        "digital_input_bits",
        "digital_output_bits",
        "runtime_state",
    ],
}

# ---------------------------------------------------------------------------
# Per-template retention policy  (required + distractor categories)
# ---------------------------------------------------------------------------

TEMPLATE_CATEGORIES: Dict[str, List[str]] = {
    "state_joint_moved": ["position", "speed"],
    "state_friction_increase": ["current", "speed", "position"],
    "state_acceleration": ["vibration", "force", "speed"],
    "state_external_force_detected": ["force", "vibration"],
    "state_jerk": ["speed", "position"],
    "state_torque_magnitude": ["force", "current"],
    "state_joint_speed_ranking": ["speed", "current"],
    "state_joint_within_rated_speed": ["speed", "current"],
    "state_current_within_rated": ["current", "speed"],
    "state_signal_description": ["current", "speed", "position"],
    "state_safety_mode": ["metadata", "speed"],
    "state_robot_mode": ["metadata", "current"],
    "state_signal_prediction": ["current", "speed", "position"],
    "state_signal_anomaly": ["current", "speed", "position"],
}

# Columns that are always preserved regardless of category filtering.
_ALWAYS_KEEP = {"timestamp_ms", "fault_label", "event"}


def _categorize_feature(name: str) -> str | None:
    """Return the category a feature belongs to, or None if uncategorized."""
    for category, prefixes in FEATURE_CATEGORY_PREFIXES.items():
        for prefix in prefixes:
            if name.startswith(prefix) or name == prefix.rstrip("_"):
                return category
    return None


def _allowed_features(template_type: str) -> Set[str] | None:
    """Return the set of allowed categories for *template_type*, or None
    if no policy is defined (meaning keep everything)."""
    categories = TEMPLATE_CATEGORIES.get(template_type)
    if categories is None:
        return None
    return set(categories)


def filter_features_for_template(
    rows: List[Dict[str, Any]],
    template_type: str,
) -> List[Dict[str, Any]]:
    """Remove feature columns not relevant to *template_type*.

    Keeps:
    - Features in the required + distractor categories for this template
    - Uncategorized features (natural distractors)
    - Protected columns (timestamp_ms, fault_label, event)
    """
    allowed = _allowed_features(template_type)
    if allowed is None:
        return rows  # no policy → keep everything

    if not rows:
        return rows

    # Pre-compute which keys to keep from the first row (all rows share
    # the same schema).
    keys_to_keep: Set[str] = set()
    for key in rows[0].keys():
        if key in _ALWAYS_KEEP:
            keys_to_keep.add(key)
            continue
        category = _categorize_feature(key)
        if category is None or category in allowed:
            keys_to_keep.add(key)

    return [{k: v for k, v in row.items() if k in keys_to_keep} for row in rows]
