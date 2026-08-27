"""
Friction Decrease Injection (event 11) applicator.

Temporarily *decreases* the friction coefficient on **both** the workpiece
(cube) and the gripper finger pads simultaneously, making the entire
grip interface more slippery.

The reduction is sampled as a relative fraction (e.g. 0.30 = 30% drop)
of the current friction, so the applicator adapts to whatever friction
values the simulation is using without hardcoded constants.

This is a physics-level injection: it modifies the USD PhysicsMaterial
on the cube **and** both finger pads at the same time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

_DEFAULT_FRACTION_RANGE = (0.45, 0.60)  # targets ~25% CF rate

# Minimum friction after reduction, never go below this
_FRICTION_FLOOR = 0.05

# All surfaces that form the grip contact, modified together.
_GRIP_COMPONENTS = [
    "cube",
    "finger_pad_left",
    "finger_pad_right",
]


class FrictionDecreaseApplicator(BaseApplicator):
    """Event 11: Friction Decrease, lowers friction on both cube and gripper."""

    valid_phases = [3, 4, 5, 6]  # close through lower, gripper is on the box

    def __init__(
        self,
        fraction_range: tuple = _DEFAULT_FRACTION_RANGE):
        self._frac_range = fraction_range
        # Store originals for each component: list of (mat_path, orig_static, orig_dynamic)
        self._originals: List[Tuple[str, float, float]] = []

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        fraction = float(rng.uniform(self._frac_range[0], self._frac_range[1]))
        return {
            "delta": fraction,              # stored as fraction for the event log
            "reduction_pct": fraction * 100, # human-readable percentage for HUD
            "_duration": 999999,             # persistent, lasts the entire episode
        }

    @staticmethod
    def _resolve_mat_path(component: str, ctx: SimContext) -> str | None:
        """Map a component name to its USD PhysicsMaterial path.

        Returns None if the component doesn't exist in this sim setup.
        """
        if component == "cube":
            for suffix in ["/PhysMat", "/CubePhysicsMaterial", "/PhysicsMaterial"]:
                candidate = ctx.cube_prim_path + suffix
                if ctx.stage and ctx.stage.GetPrimAtPath(candidate).IsValid():
                    return candidate
            # Fall back: look up the bound physics material
            try:
                from pxr import UsdShade
                cube_prim = ctx.stage.GetPrimAtPath(ctx.cube_prim_path)
                if cube_prim.IsValid():
                    binding_api = UsdShade.MaterialBindingAPI(cube_prim)
                    bound = binding_api.GetDirectBinding("physics")
                    if bound and bound.GetMaterial():
                        mp = bound.GetMaterial().GetPrim()
                        if mp.IsValid():
                            return str(mp.GetPath())
            except Exception:
                pass
            return None
        robotiq_base = ctx.robot_prim_path + "/robotiq"
        if component == "finger_pad_left":
            return robotiq_base + "/left_inner_finger_pad/GripMaterial"
        if component == "finger_pad_right":
            return robotiq_base + "/right_inner_finger_pad/GripMaterial"
        return None

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if ctx.stage is None:
            return
        self._originals = []
        frac = params["delta"]

        from pxr import UsdPhysics

        for comp in _GRIP_COMPONENTS:
            mat_path = self._resolve_mat_path(comp, ctx)
            if mat_path is None:
                continue
            try:
                mat_prim = ctx.stage.GetPrimAtPath(mat_path)
                if not mat_prim.IsValid():
                    continue
                phys_mat = UsdPhysics.MaterialAPI(mat_prim)
                orig_static = phys_mat.GetStaticFrictionAttr().Get()
                orig_dynamic = phys_mat.GetDynamicFrictionAttr().Get()

                new_static = max(orig_static * (1.0 - frac), _FRICTION_FLOOR)
                new_dynamic = max(orig_dynamic * (1.0 - frac), _FRICTION_FLOOR)
                phys_mat.GetStaticFrictionAttr().Set(new_static)
                phys_mat.GetDynamicFrictionAttr().Set(new_dynamic)

                self._originals.append((mat_path, orig_static, orig_dynamic))
                print(f"[FrictionDecrease] {comp} "
                      f"static {orig_static:.3f} → {new_static:.3f}  "
                      f"dynamic {orig_dynamic:.3f} → {new_dynamic:.3f}  "
                      f"(reduction {frac*100:.0f}%)")
            except Exception as e:
                print(f"[FrictionDecrease] {comp} failed: {e}")

        if not self._originals:
            print("[FrictionDecrease] WARNING: no materials were modified")

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if not self._originals:
            return
        try:
            from pxr import UsdPhysics
            for mat_path, orig_static, orig_dynamic in self._originals:
                mat_prim = ctx.stage.GetPrimAtPath(mat_path)
                if mat_prim.IsValid():
                    phys_mat = UsdPhysics.MaterialAPI(mat_prim)
                    phys_mat.GetStaticFrictionAttr().Set(orig_static)
                    phys_mat.GetDynamicFrictionAttr().Set(orig_dynamic)
            print(f"[FrictionDecrease] friction restored on {len(self._originals)} component(s)")
        except Exception as e:
            print(f"[FrictionDecrease] on_end failed: {e}")
        finally:
            self._originals = []
