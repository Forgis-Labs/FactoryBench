"""
Transient Spike (event 2) and Transient Dip (event 3) applicators.

These inject sensor-level perturbations: they modify the logged sensor
data dict *after* the true physics values have been read but *before*
the row is written to CSV.  This simulates a faulty sensor reading
without affecting the actual physics simulation.

The perturbation is additive:
  - Spike: sensor_value += delta   (delta > 0)
  - Dip:   sensor_value -= delta   (delta > 0)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

# -----------------------------------------------------------------------
# Mapping from events.json feature names → CSV column names
# -----------------------------------------------------------------------
# events.json uses names like "effort_current_0" while the CSV logger
# uses "joint_torque_nm_shoulder_pan".  This map bridges the two.

_JOINT_INDEX_TO_NAME = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]


def _feature_to_csv_column(feature: str) -> str | None:
    """Map an events.json sensor feature name to the CSV column name.

    Returns None if the feature has no direct CSV mapping (the caller
    should skip it gracefully).
    """
    # effort_current_0..5  →  joint_torque_nm_<name>
    if feature.startswith("effort_current_"):
        idx = int(feature.split("_")[-1])
        return f"joint_torque_nm_{_JOINT_INDEX_TO_NAME[idx]}"

    # est_contact_force_0..5  →  contact_force_{x,y,z}_n (only 3 axes logged)
    if feature.startswith("est_contact_force_"):
        idx = int(feature.split("_")[-1])
        axes = ["x", "y", "z"]
        if idx < 3:
            return f"contact_force_{axes[idx]}_n"
        return None  # indices 3-5 not directly logged as individual columns

    # effort_target_torque_0..5  →  joint_cmd_torque_nm_<name>
    if feature.startswith("effort_target_torque_"):
        idx = int(feature.split("_")[-1])
        return f"joint_cmd_torque_nm_{_JOINT_INDEX_TO_NAME[idx]}"

    # feedback_speed_0..5  →  joint_vel_radps_<name>
    if feature.startswith("feedback_speed_"):
        idx = int(feature.split("_")[-1])
        return f"joint_vel_radps_{_JOINT_INDEX_TO_NAME[idx]}"

    # vibration_0..4  →  no direct CSV column (would need a vibration sensor)
    if feature.startswith("vibration_"):
        return None

    # acoustic_0  →  no direct CSV column
    if feature == "acoustic_0":
        return None

    # robot_current  →  no direct CSV column (aggregate)
    if feature == "robot_current":
        return None

    # tool_momentum  →  no direct CSV column
    if feature == "tool_momentum":
        return None

    # robot_voltage / main_voltage  →  no direct CSV column
    if feature in ("robot_voltage", "main_voltage"):
        return None

    return None


# -----------------------------------------------------------------------
# Default parameter ranges
# -----------------------------------------------------------------------

# Duration in sim steps (at 60Hz: 30 steps = 0.5s, 300 = 5s)
_DEFAULT_DURATION_RANGE = (30, 200)

# Delta magnitude ranges per feature family (in physical units)
_DEFAULT_DELTA_RANGES: Dict[str, tuple] = {
    "effort_current":       (0.5, 5.0),     # Nm
    "est_contact_force":    (1.0, 20.0),    # N
    "effort_target_torque": (0.5, 5.0),     # Nm
    "feedback_speed":       (0.05, 0.5),    # rad/s
    "vibration":            (0.01, 0.1),    # arbitrary units
    "acoustic":             (0.01, 0.1),    # arbitrary units
    "robot_current":        (0.1, 2.0),     # A
    "tool_momentum":        (0.01, 0.5),    # kg·m/s
    "robot_voltage":        (0.5, 5.0),     # V
    "main_voltage":         (0.5, 5.0),     # V
}


def _get_feature_family(feature: str) -> str:
    """Strip the trailing index to get the family name."""
    parts = feature.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return feature


# -----------------------------------------------------------------------
# Applicators
# -----------------------------------------------------------------------

class TransientSpikeApplicator(BaseApplicator):
    """Event 2: Transient Spike, adds a positive delta to a sensor feature."""

    def __init__(
        self,
        duration_range: tuple = _DEFAULT_DURATION_RANGE,
        delta_ranges: Dict[str, tuple] | None = None):
        self._dur_range = duration_range
        self._delta_ranges = delta_ranges or _DEFAULT_DELTA_RANGES

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        constraints = event_def.get("variable_constraints", {})
        features = constraints.get("feature_i", [])
        if not features:
            features = ["effort_current_0"]

        feature = features[int(rng.integers(0, len(features)))]
        family = _get_feature_family(feature)
        lo, hi = self._delta_ranges.get(family, (0.5, 5.0))
        delta = float(rng.uniform(lo, hi))
        duration = int(rng.integers(self._dur_range[0], self._dur_range[1] + 1))

        return {
            "feature_i": feature,
            "delta": delta,
            "_duration": duration,
        }

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        col = _feature_to_csv_column(params["feature_i"])
        print(f"[TransientSpike] START feature={params['feature_i']} "
              f"col={col} delta=+{params['delta']:.4f}")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        col = _feature_to_csv_column(params["feature_i"])
        if col is None or col not in ctx.sensor_data:
            return
        try:
            original = float(ctx.sensor_data[col])
            ctx.sensor_data[col] = f"{original + params['delta']:.6f}"
        except (ValueError, TypeError):
            pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        print(f"[TransientSpike] END feature={params['feature_i']}")


class TransientDipApplicator(BaseApplicator):
    """Event 3: Transient Dip, subtracts a positive delta from a sensor feature."""

    def __init__(
        self,
        duration_range: tuple = _DEFAULT_DURATION_RANGE,
        delta_ranges: Dict[str, tuple] | None = None):
        self._dur_range = duration_range
        self._delta_ranges = delta_ranges or _DEFAULT_DELTA_RANGES

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        constraints = event_def.get("variable_constraints", {})
        features = constraints.get("feature_i", [])
        if not features:
            features = ["effort_current_0"]

        feature = features[int(rng.integers(0, len(features)))]
        family = _get_feature_family(feature)
        lo, hi = self._delta_ranges.get(family, (0.5, 5.0))
        delta = float(rng.uniform(lo, hi))
        duration = int(rng.integers(self._dur_range[0], self._dur_range[1] + 1))

        return {
            "feature_i": feature,
            "delta": delta,
            "_duration": duration,
        }

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        col = _feature_to_csv_column(params["feature_i"])
        print(f"[TransientDip] START feature={params['feature_i']} "
              f"col={col} delta=-{params['delta']:.4f}")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        col = _feature_to_csv_column(params["feature_i"])
        if col is None or col not in ctx.sensor_data:
            return
        try:
            original = float(ctx.sensor_data[col])
            ctx.sensor_data[col] = f"{original - params['delta']:.6f}"
        except (ValueError, TypeError):
            pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        print(f"[TransientDip] END feature={params['feature_i']}")
