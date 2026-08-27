"""
Payload Addition (event 6) applicator.

Increases the workpiece mass by *x* kg at the trigger step,
then restores the original mass when the event ends.

This is a *physics-level* injection: it changes the actual USD
MassAPI attribute, so the robot controller feels the extra load
through joint torques, contact forces, etc.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

# Default ranges
_DEFAULT_MASS_DELTA_RANGE = (0.35, 0.70)  # kg to add, targets ~25% CF rate


class PayloadAdditionApplicator(BaseApplicator):
    """Event 6: Payload Addition, temporarily increases workpiece mass."""

    valid_phases = [4, 5, 6]  # lift, move_xy, lower, robot is carrying

    def __init__(
        self,
        mass_delta_range: tuple = _DEFAULT_MASS_DELTA_RANGE):
        self._mass_range = mass_delta_range
        self._original_mass: float | None = None

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        x = float(rng.uniform(self._mass_range[0], self._mass_range[1]))
        return {
            "x": x,
            "_duration": 999999,  # persistent, lasts the entire episode
        }

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if ctx.stage is None or not ctx.cube_prim_path:
            return
        try:
            from pxr import UsdPhysics
            prim = ctx.stage.GetPrimAtPath(ctx.cube_prim_path)
            if not prim.IsValid():
                return
            mass_api = UsdPhysics.MassAPI(prim)
            self._original_mass = mass_api.GetMassAttr().Get()
            new_mass = self._original_mass + params["x"]
            mass_api.GetMassAttr().Set(new_mass)
            print(f"[PayloadAddition] mass {self._original_mass:.3f} → "
                  f"{new_mass:.3f} kg (+{params['x']:.3f})")
        except Exception as e:
            print(f"[PayloadAddition] on_start failed: {e}")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        # The mass change persists via USD, nothing to do per step.
        pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if self._original_mass is None:
            return
        if ctx.stage is None or not ctx.cube_prim_path:
            self._original_mass = None
            return
        try:
            from pxr import UsdPhysics
            prim = ctx.stage.GetPrimAtPath(ctx.cube_prim_path)
            if not prim.IsValid():
                return
            mass_api = UsdPhysics.MassAPI(prim)
            mass_api.GetMassAttr().Set(self._original_mass)
            print(f"[PayloadAddition] mass restored to {self._original_mass:.3f} kg")
        except Exception as e:
            print(f"[PayloadAddition] on_end failed: {e}")
        finally:
            self._original_mass = None
