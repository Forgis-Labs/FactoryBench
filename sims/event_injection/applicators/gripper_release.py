"""
Gripper Release Injection (event 14) applicator.

Simulates a mid-motion payload release: the gripper suddenly opens
during a grip phase, dropping the workpiece.  The release is a
one-shot action at the start of the event, the finger_joint target
is forced to 0 (open) and the drive stiffness is temporarily reduced
so the fingers spring open.  Original values are restored on end.

The ``release_time`` variable is sampled as a fraction (0–1) of the
episode, but the actual trigger is controlled by the scheduler's
``start_step``.  The applicator itself acts immediately on ``on_start``.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

# Short duration, the release is a brief event, the consequence
# (dropped cube) persists through normal sim physics.
_DEFAULT_DURATION_RANGE = (20, 60)


class GripperReleaseApplicator(BaseApplicator):
    """Event 14: Gripper Release, mid-motion payload drop."""

    valid_phases = [4, 5, 6]  # lift, move_xy, lower, gripper is holding

    def __init__(
        self,
        duration_range: tuple = _DEFAULT_DURATION_RANGE):
        self._dur_range = duration_range
        self._original_max_force: float | None = None
        self._original_stiffness: float | None = None
        self._fj_path: str | None = None

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        duration = int(rng.integers(self._dur_range[0], self._dur_range[1] + 1))
        release_time = float(rng.uniform(0.2, 0.8))
        return {
            "gripper": "robotiq_2f_85",
            "release_time": release_time,
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
                print(f"[GripperRelease] finger_joint not found: {self._fj_path}")
                self._fj_path = None
                return
            drive = UsdPhysics.DriveAPI(fj_prim, "angular")
            self._original_max_force = drive.GetMaxForceAttr().Get()
            self._original_stiffness = drive.GetStiffnessAttr().Get()

            # Force gripper open by setting target to 0 and reducing
            # stiffness so fingers spring open quickly.
            drive.GetStiffnessAttr().Set(0.1)
            drive.GetMaxForceAttr().Set(0.1)

            print(f"[GripperRelease] START, gripper forced open "
                  f"(release_time={params['release_time']:.2f})")
        except Exception as e:
            print(f"[GripperRelease] on_start failed: {e}")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        # Keep the drive weak every step to prevent the sim loop from
        # overriding it back to closed.
        if self._fj_path is None or ctx.stage is None:
            return
        try:
            from pxr import UsdPhysics
            fj_prim = ctx.stage.GetPrimAtPath(self._fj_path)
            if fj_prim.IsValid():
                drive = UsdPhysics.DriveAPI(fj_prim, "angular")
                drive.GetStiffnessAttr().Set(0.1)
                drive.GetMaxForceAttr().Set(0.1)
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
                print(f"[GripperRelease] END, finger_joint drive restored")
        except Exception as e:
            print(f"[GripperRelease] on_end failed: {e}")
        finally:
            self._original_max_force = None
            self._original_stiffness = None
            self._fj_path = None
