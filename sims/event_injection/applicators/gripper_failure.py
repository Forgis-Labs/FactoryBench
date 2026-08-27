"""
Gripper Activation Failure Injection (event 13) applicator.

Simulates a gripper that fails to activate (close) when commanded.
During the event, the finger_joint drive target is forced to 0 (open)
and the maxForce is reduced to near-zero, preventing the gripper from
gripping.  Original values are restored on end.

This is a physics-level injection: it directly prevents the gripper
from closing by overriding the drive parameters.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

_DEFAULT_DURATION_RANGE = (80, 400)


class GripperActivationFailureApplicator(BaseApplicator):
    """Event 13: Gripper Activation Failure, gripper cannot close."""

    valid_phases = [2, 3]  # settle, close, when gripper should be closing

    def __init__(
        self,
        duration_range: tuple = _DEFAULT_DURATION_RANGE):
        self._dur_range = duration_range
        self._original_max_force: float | None = None
        self._original_stiffness: float | None = None
        self._fj_path: str | None = None

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        duration = int(rng.integers(self._dur_range[0], self._dur_range[1] + 1))
        return {
            "gripper": "robotiq_2f_85",
            "_duration": duration,
        }

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if ctx.stage is None:
            return
        self._fj_path = ctx.robot_prim_path + "/robotiq/joints/finger_joint"
        try:
            from pxr import UsdPhysics
            fj_prim = ctx.stage.GetPrimAtPath(self._fj_path)
            if not fj_prim.IsValid():
                print(f"[GripperFailure] finger_joint not found: {self._fj_path}")
                self._fj_path = None
                return
            drive = UsdPhysics.DriveAPI(fj_prim, "angular")
            self._original_max_force = drive.GetMaxForceAttr().Get()
            self._original_stiffness = drive.GetStiffnessAttr().Get()
            # Kill the drive, gripper can't close
            drive.GetMaxForceAttr().Set(0.01)
            drive.GetStiffnessAttr().Set(0.01)
            print(f"[GripperFailure] START, finger_joint drive disabled "
                  f"(was maxForce={self._original_max_force}, "
                  f"kp={self._original_stiffness})")
        except Exception as e:
            print(f"[GripperFailure] on_start failed: {e}")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        # Keep the drive killed every step in case something resets it
        if self._fj_path is None or ctx.stage is None:
            return
        try:
            from pxr import UsdPhysics
            fj_prim = ctx.stage.GetPrimAtPath(self._fj_path)
            if fj_prim.IsValid():
                drive = UsdPhysics.DriveAPI(fj_prim, "angular")
                drive.GetMaxForceAttr().Set(0.01)
                drive.GetStiffnessAttr().Set(0.01)
        except Exception:
            pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if self._fj_path is None or self._original_max_force is None:
            return
        try:
            from pxr import UsdPhysics
            fj_prim = ctx.stage.GetPrimAtPath(self._fj_path)
            if fj_prim.IsValid():
                drive = UsdPhysics.DriveAPI(fj_prim, "angular")
                drive.GetMaxForceAttr().Set(self._original_max_force)
                drive.GetStiffnessAttr().Set(self._original_stiffness)
                print(f"[GripperFailure] END, finger_joint drive restored "
                      f"(maxForce={self._original_max_force}, "
                      f"kp={self._original_stiffness})")
        except Exception as e:
            print(f"[GripperFailure] on_end failed: {e}")
        finally:
            self._original_max_force = None
            self._original_stiffness = None
            self._fj_path = None
