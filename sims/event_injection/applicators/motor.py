"""
Motor Miscommutation Injection (event 12) applicator.

Simulates a commutation phase offset on a motor by perturbing the
joint position target that the controller has already set for the
current step.

The ripple is driven by the joint's actual electrical angle
(joint_position * pole_pairs), not by wall-clock time, so it is
tied to rotor position as in a real BLDC miscommutation.  The
perturbation amplitude also scales with joint velocity: at near-zero
speed only a small static error remains, while at higher speeds the
oscillating ripple dominates.

Sensor data (logged torque and position) is also corrupted to reflect
the perturbation.

This is a mixed-level injection: it applies a real physics perturbation
(position-dependent target offset on the affected joint) AND corrupts
the corresponding sensor readings.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

_DEFAULT_PHASE_OFFSET_RANGE = (7.0, 12.0)   # degrees, targets ~25% CF rate

_JOINT_INDEX_TO_NAME = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]

_MOTOR_CHOICES = [f"motor_{i}" for i in range(6)]

# Position-target offset in radians per degree of phase offset.
# This controls the amplitude of the sinusoidal wobble injected
# into the joint's position target each step.
#
#   5° offset → peak ±0.015 rad (±0.86°), subtle tracking error
#  15° offset → peak ±0.045 rad (±2.6°), noticeable, some failures
#  30° offset → peak ±0.090 rad (±5.2°), significant, frequent drops
_RAD_PER_DEGREE = 0.003


class MotorMiscommutationApplicator(BaseApplicator):
    """Event 12: Motor Miscommutation, injects position-target ripple."""

    valid_phases = [0, 1, 2, 3, 4, 5, 6]  # any phase with arm motion

    def __init__(
        self,
        phase_offset_range: tuple = _DEFAULT_PHASE_OFFSET_RANGE):
        self._phase_range = phase_offset_range
        self._step_count = 0

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        motor = _MOTOR_CHOICES[int(rng.integers(0, len(_MOTOR_CHOICES)))]
        phase_offset = float(rng.uniform(self._phase_range[0], self._phase_range[1]))
        return {
            "motor": motor,
            "phase_offset_deg": phase_offset,
            "_duration": 999999,  # persistent, lasts the entire episode
        }

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        self._step_count = 0
        motor_idx = int(params["motor"].split("_")[-1])
        joint_name = _JOINT_INDEX_TO_NAME[motor_idx]
        print(f"[MotorMiscommutation] START {params['motor']} "
              f"({joint_name}) offset={params['phase_offset_deg']:.1f}°")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        self._step_count += 1
        motor_idx = int(params["motor"].split("_")[-1])
        joint_name = _JOINT_INDEX_TO_NAME[motor_idx]

        phase_deg = params["phase_offset_deg"]
        phase_rad = np.radians(phase_deg)

        # Read actual joint position and velocity
        robot = ctx.extra.get("robot") if ctx.extra else None
        action = ctx.extra.get("action") if ctx.extra else None

        joint_pos = 0.0
        joint_vel = 0.0
        if robot is not None:
            try:
                positions = robot.get_joint_positions()
                joint_pos = float(positions[motor_idx])
            except Exception:
                pass
            try:
                velocities = robot.get_joint_velocities()
                joint_vel = float(velocities[motor_idx])
            except Exception:
                pass

        # Ripple is a function of rotor electrical angle, not time.
        # UR5 motors have ~4-8 pole pairs depending on joint; we use 6
        # as a representative value.  The electrical angle is
        # joint_pos * pole_pairs, and the miscommutation shifts it.
        pole_pairs = 6
        electrical_angle = joint_pos * pole_pairs
        ripple = np.sin(electrical_angle + phase_rad) - np.sin(electrical_angle)

        # Velocity scaling: at low speed the ripple has little dynamic
        # effect (mostly a static torque reduction); at higher speed
        # the oscillating error becomes dominant.
        vel_scale = min(abs(joint_vel) / 0.5, 1.0)  # ramps 0→1 over 0–0.5 rad/s
        vel_factor = 0.15 + 0.85 * vel_scale         # floor of 15% even at rest

        # ── Physics perturbation: offset the position target ──────────
        if robot is not None and action is not None:
            try:
                offset_rad = phase_deg * _RAD_PER_DEGREE * ripple * vel_factor
                # action.joint_positions may be None, or individual
                # elements may be None during gripper-only phases
                jp = action.joint_positions
                if jp is not None and jp[motor_idx] is not None:
                    ctrl_target = float(jp[motor_idx])
                else:
                    ctrl_target = joint_pos
                new_target = ctrl_target + offset_rad
                robot._articulation_view.set_joint_position_targets(
                    np.array([[new_target]]),
                    joint_indices=np.array([motor_idx]))
            except Exception as e:
                print(f"[MotorMiscommutation] target perturbation failed: {e}")

        # ── Sensor corruption: logged torque and position ─────────────
        vel_col = f"joint_vel_radps_{joint_name}"
        try:
            vel = float(ctx.sensor_data.get(vel_col, 0.0))
        except (ValueError, TypeError):
            vel = 0.0

        sensor_ripple = phase_rad * max(abs(vel), 0.1) * ripple * vel_factor

        torque_col = f"joint_torque_nm_{joint_name}"
        if torque_col in ctx.sensor_data:
            try:
                original = float(ctx.sensor_data[torque_col])
                ctx.sensor_data[torque_col] = f"{original + sensor_ripple:.6f}"
            except (ValueError, TypeError):
                pass

        pos_col = f"joint_pos_rad_{joint_name}"
        if pos_col in ctx.sensor_data:
            try:
                original = float(ctx.sensor_data[pos_col])
                pos_error = phase_rad * 0.01 * ripple
                ctx.sensor_data[pos_col] = f"{original + pos_error:.6f}"
            except (ValueError, TypeError):
                pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        motor_idx = int(params["motor"].split("_")[-1])
        joint_name = _JOINT_INDEX_TO_NAME[motor_idx]
        print(f"[MotorMiscommutation] END {params['motor']} ({joint_name})")
