"""
Collision Injection (event 16) applicator.

Simulates an unexpected collision by spawning a visible projectile
that physically flies into the robot arm.  The projectile is built
from USD geometric primitives (no external asset references).

The projectile spawns at a random position around the robot (random
angle, radius, and height) and targets a randomly chosen robot link,
so collisions can arrive from any direction and hit any part of the
arm.

The prim is spawned as **visual-only** (no RigidBodyAPI, no
CollisionAPI).  During flight it is moved along the ballistic arc
by updating its USD translate op each step, PhysX never sees it,
so it cannot interfere with other bodies.

One step before impact the physics APIs (RigidBodyAPI, CollisionAPI,
MassAPI, CCD) are added together with the flight velocity.  PhysX
discovers a brand-new *dynamic* body and the projectile naturally
collides with the robot on the next step, a real, mass-based
collision with realistic force.

Lifecycle:
  - on_episode_setup: spawns a visual-only object at a random position.
  - on_start: computes the ballistic trajectory toward a random link.
  - on_step: moves the visual along the arc; one step before impact
    adds physics APIs so the body collides naturally.
  - on_end / on_episode_cleanup: removes the prim.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from event_injection.applicators.base import BaseApplicator
from event_injection.context import SimContext

_DEFAULT_IMPULSE_RANGE = (6.0, 12.0)  # target impact momentum in kg·m/s, targets ~25% CF rate
# Duration must cover the full flight (up to 0.50 s) plus post-impact settling.
_DEFAULT_DURATION_RANGE = (30, 50)
# Flight time is clamped so objects neither teleport nor float.
_T_FLIGHT_MIN = 0.10   # seconds
_T_FLIGHT_MAX = 0.50
# Assumed collision contact duration for sensor-force estimate.
_CONTACT_DT = 0.016667  # one physics step at 60 Hz

_OBJECT_CHOICES = [
    "unknown_debris",
    "adjacent_part",
    "tool",
    "bolt",
    "pipe_section",
    "cardboard_box",
    "metal_plate",
    "gear",
    "bottle",
    "wood_block",
]

_PROJECTILE_PRIM = "/World/collision_projectile"
_PROJECTILE_MAT_SCOPE = "/World/Looks/collision_projectile"

# Object type → (mass_kg, color_rgb, metallic, roughness)
_OBJECT_PROPS = {
    "unknown_debris": (0.15, (0.45, 0.35, 0.25), 0.0,  0.75),
    "adjacent_part":  (0.40, (0.55, 0.56, 0.58), 0.80, 0.30),
    "tool":           (0.60, (0.38, 0.38, 0.40), 0.85, 0.25),
    "bolt":           (0.25, (0.62, 0.62, 0.65), 0.90, 0.20),
    "pipe_section":   (0.70, (0.50, 0.50, 0.52), 0.85, 0.30),
    "cardboard_box":  (0.20, (0.72, 0.58, 0.38), 0.0,  0.90),
    "metal_plate":    (0.90, (0.60, 0.60, 0.62), 0.85, 0.25),
    "gear":           (0.55, (0.48, 0.48, 0.50), 0.90, 0.20),
    "bottle":         (0.30, (0.20, 0.55, 0.20), 0.05, 0.40),
    "wood_block":     (0.35, (0.65, 0.45, 0.25), 0.0,  0.85),
}

# Spawn position is sampled in polar coordinates around the robot base.
_SPAWN_RADIUS_RANGE = (0.45, 0.75)   # metres from robot base XY
_SPAWN_Z_RANGE = (0.10, 0.50)        # height above ground
_ROBOT_BASE_XY = np.array([0.0, 0.0])

# All targetable UR5 links (randomly chosen per episode).
_TARGET_LINKS = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]

_SCALE_RANGE = (0.5, 2.0)  # uniform scale factor applied to projectile geometry
_GRAVITY = 9.81


class CollisionApplicator(BaseApplicator):
    """Event 16: Collision, spawns a projectile that hits the robot arm."""

    valid_phases = [4, 5, 6]

    def __init__(
        self,
        impulse_range: tuple = _DEFAULT_IMPULSE_RANGE,
        duration_range: tuple = _DEFAULT_DURATION_RANGE,
        force_object: str | None = None):
        self._impulse_range = impulse_range
        self._dur_range = duration_range
        self._force_object = force_object
        self._direction: np.ndarray | None = None
        self._projectile_spawned = False
        self._launch_pos: np.ndarray | None = None
        self._launch_vel: np.ndarray | None = None
        self._t_flight: float = 0.0
        self._flight_elapsed: float = 0.0
        self._in_flight: bool = False
        self._obj_type: str = ""
        self._scale: float = 1.0
        self._impact_force_est: float = 0.0

    def sample_params(self, event_def: dict, rng: np.random.Generator) -> Dict[str, Any]:
        obj = self._force_object or _OBJECT_CHOICES[int(rng.integers(0, len(_OBJECT_CHOICES)))]
        impulse = float(rng.uniform(self._impulse_range[0], self._impulse_range[1]))
        traj_frac = float(rng.uniform(0.1, 0.9))
        duration = int(rng.integers(self._dur_range[0], self._dur_range[1] + 1))

        # Random spawn position around the robot (full 360°).
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = float(rng.uniform(*_SPAWN_RADIUS_RANGE))
        spawn_origin = [
            float(_ROBOT_BASE_XY[0] + radius * np.cos(angle)),
            float(_ROBOT_BASE_XY[1] + radius * np.sin(angle)),
            float(rng.uniform(*_SPAWN_Z_RANGE)),
        ]

        # Random target link on the robot arm.
        target_link = _TARGET_LINKS[int(rng.integers(0, len(_TARGET_LINKS)))]

        # Random volume scale (mass scales with volume ∝ scale³).
        scale = float(rng.uniform(*_SCALE_RANGE))

        return {
            "object": obj,
            "impact_impulse": impulse,
            "trajectory_fraction": traj_frac,
            "_duration": duration,
            "_spawn_origin": spawn_origin,
            "_target_link": target_link,
            "_scale": scale,
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _clear_projectile(self, stage):
        from pxr import Sdf
        old = stage.GetPrimAtPath(_PROJECTILE_PRIM)
        if old.IsValid():
            stage.RemovePrim(Sdf.Path(_PROJECTILE_PRIM))
        old_mat = stage.GetPrimAtPath(_PROJECTILE_MAT_SCOPE)
        if old_mat.IsValid():
            stage.RemovePrim(Sdf.Path(_PROJECTILE_MAT_SCOPE))
        self._projectile_spawned = False

    def _make_material(self, stage, name, color, metallic=0.0, roughness=0.5):
        from pxr import UsdShade, Sdf, Gf
        path = f"{_PROJECTILE_MAT_SCOPE}/{name}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _bind_material(self, prim, mat):
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(mat)

    def _set_prim_position(self, stage, pos):
        from pxr import UsdGeom, Gf
        prim = stage.GetPrimAtPath(_PROJECTILE_PRIM)
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3f(*[float(v) for v in pos]))
                return
        xf.AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in pos]))

    def _make_physics_body(self, stage, scale: float = 1.0):
        """Add RigidBodyAPI + CollisionAPI + MassAPI + CCD to the
        projectile prim, turning it from a visual into a dynamic
        physics body.  Called once at handoff time."""
        from pxr import UsdPhysics, PhysxSchema

        prim = stage.GetPrimAtPath(_PROJECTILE_PRIM)
        if not prim.IsValid():
            return

        # RigidBody on parent (dynamic, not kinematic)
        UsdPhysics.RigidBodyAPI.Apply(prim)

        # Mass, scales with volume (scale³)
        base_mass = _OBJECT_PROPS.get(self._obj_type, (0.1))[0]
        mass = base_mass * (scale ** 3)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(mass)

        # CCD (safe, body is dynamic)
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rb.GetEnableCCDAttr().Set(True)

        # CollisionAPI on every child geometry prim
        for child in prim.GetChildren():
            UsdPhysics.CollisionAPI.Apply(child)

    # ── Spawn helpers (VISUAL ONLY, no physics APIs) ─────────────────

    def _make_parent(self, stage, position, scale: float = 1.0):
        from pxr import UsdGeom, Gf
        parent = UsdGeom.Xform.Define(stage, _PROJECTILE_PRIM)
        xf = UsdGeom.Xformable(parent.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3f(*[float(v) for v in position]))
        if scale != 1.0:
            s = float(scale)
            xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
        return parent

    def _spawn_debris(self, position, ctx, scale=1.0):
        """Irregular rock/chunk."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["unknown_debris"]
        mat = self._make_material(stage, "debris", color, metallic, roughness)
        self._make_parent(stage, position, scale)

        rock = UsdGeom.Sphere.Define(stage, _PROJECTILE_PRIM + "/rock")
        rock.GetRadiusAttr().Set(0.055)
        self._bind_material(rock.GetPrim(), mat)

    def _spawn_adjacent_part(self, position, ctx, scale=1.0):
        """L-shaped metal bracket."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["adjacent_part"]
        mat = self._make_material(stage, "bracket", color, metallic, roughness)
        self._make_parent(stage, position, scale)

        arm = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/arm")
        arm_xf = UsdGeom.Xformable(arm)
        arm_xf.AddScaleOp().Set(Gf.Vec3f(0.10, 0.03, 0.02))
        self._bind_material(arm.GetPrim(), mat)

        leg = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/leg")
        leg_xf = UsdGeom.Xformable(leg)
        leg_xf.AddTranslateOp().Set(Gf.Vec3f(-0.085, 0.04, 0.0))
        leg_xf.AddScaleOp().Set(Gf.Vec3f(0.02, 0.05, 0.02))
        self._bind_material(leg.GetPrim(), mat)

    def _spawn_tool(self, position, ctx, scale=1.0):
        """Wrench-like hand tool."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["tool"]
        bright = (min(color[0] * 1.3, 1.0),
                  min(color[1] * 1.3, 1.0),
                  min(color[2] * 1.3, 1.0))
        mat_handle = self._make_material(stage, "wrench_handle", color, metallic, roughness)
        mat_head = self._make_material(stage, "wrench_head", bright, metallic, roughness)
        self._make_parent(stage, position, scale)

        shaft = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/handle")
        shaft.GetRadiusAttr().Set(0.012)
        shaft.GetHeightAttr().Set(0.30)
        shaft.GetAxisAttr().Set("X")
        self._bind_material(shaft.GetPrim(), mat_handle)

        head = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/head")
        head_xf = UsdGeom.Xformable(head)
        head_xf.AddTranslateOp().Set(Gf.Vec3f(0.17, 0.0, 0.0))
        head_xf.AddScaleOp().Set(Gf.Vec3f(0.035, 0.03, 0.01))
        self._bind_material(head.GetPrim(), mat_head)

    def _spawn_bolt(self, position, ctx, scale=1.0):
        """Hex bolt, cylinder shaft + hexagonal head (approximated as a
        short, wide cylinder)."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["bolt"]
        mat = self._make_material(stage, "bolt", color, metallic, roughness)
        self._make_parent(stage, position, scale)

        shaft = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/shaft")
        shaft.GetRadiusAttr().Set(0.008)
        shaft.GetHeightAttr().Set(0.10)
        shaft.GetAxisAttr().Set("Z")
        self._bind_material(shaft.GetPrim(), mat)

        head = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/head")
        head.GetRadiusAttr().Set(0.018)
        head.GetHeightAttr().Set(0.015)
        head.GetAxisAttr().Set("Z")
        head_xf = UsdGeom.Xformable(head)
        head_xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.058))
        self._bind_material(head.GetPrim(), mat)

    def _spawn_pipe_section(self, position, ctx, scale=1.0):
        """Short steel pipe segment, outer cylinder with a thinner inner
        cylinder subtracted visually (two concentric cylinders)."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["pipe_section"]
        mat_outer = self._make_material(stage, "pipe_outer", color, metallic, roughness)
        darker = tuple(c * 0.7 for c in color)
        mat_inner = self._make_material(stage, "pipe_inner", darker, metallic, roughness)
        self._make_parent(stage, position, scale)

        outer = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/outer")
        outer.GetRadiusAttr().Set(0.03)
        outer.GetHeightAttr().Set(0.18)
        outer.GetAxisAttr().Set("X")
        self._bind_material(outer.GetPrim(), mat_outer)

        # Visual inner bore
        inner = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/inner")
        inner.GetRadiusAttr().Set(0.022)
        inner.GetHeightAttr().Set(0.185)
        inner.GetAxisAttr().Set("X")
        self._bind_material(inner.GetPrim(), mat_inner)

    def _spawn_cardboard_box(self, position, ctx, scale=1.0):
        """Small cardboard shipping box."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["cardboard_box"]
        mat = self._make_material(stage, "cardboard", color, metallic, roughness)
        tape_color = (0.75, 0.72, 0.55)
        mat_tape = self._make_material(stage, "tape", tape_color, 0.0, 0.6)
        self._make_parent(stage, position, scale)

        box = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/box")
        box_xf = UsdGeom.Xformable(box)
        box_xf.AddScaleOp().Set(Gf.Vec3f(0.10, 0.07, 0.06))
        self._bind_material(box.GetPrim(), mat)

        # Tape strip across top
        tape = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/tape")
        tape_xf = UsdGeom.Xformable(tape)
        tape_xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.061))
        tape_xf.AddScaleOp().Set(Gf.Vec3f(0.10, 0.015, 0.002))
        self._bind_material(tape.GetPrim(), mat_tape)

    def _spawn_metal_plate(self, position, ctx, scale=1.0):
        """Flat rectangular steel plate."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["metal_plate"]
        mat = self._make_material(stage, "plate", color, metallic, roughness)
        self._make_parent(stage, position, scale)

        plate = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/plate")
        plate_xf = UsdGeom.Xformable(plate)
        plate_xf.AddScaleOp().Set(Gf.Vec3f(0.14, 0.09, 0.008))
        self._bind_material(plate.GetPrim(), mat)

    def _spawn_gear(self, position, ctx, scale=1.0):
        """Spur gear, thick toothed disc approximated as a large cylinder
        (body) with smaller cylinders for the hub and a central bore."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["gear"]
        mat = self._make_material(stage, "gear_body", color, metallic, roughness)
        hub_color = tuple(min(c * 1.2, 1.0) for c in color)
        mat_hub = self._make_material(stage, "gear_hub", hub_color, metallic, roughness)
        self._make_parent(stage, position, scale)

        # Outer disc (tooth ring)
        disc = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/disc")
        disc.GetRadiusAttr().Set(0.05)
        disc.GetHeightAttr().Set(0.018)
        disc.GetAxisAttr().Set("Z")
        self._bind_material(disc.GetPrim(), mat)

        # Hub
        hub = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/hub")
        hub.GetRadiusAttr().Set(0.025)
        hub.GetHeightAttr().Set(0.022)
        hub.GetAxisAttr().Set("Z")
        self._bind_material(hub.GetPrim(), mat_hub)

        # Central bore
        bore = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/bore")
        bore.GetRadiusAttr().Set(0.010)
        bore.GetHeightAttr().Set(0.025)
        bore.GetAxisAttr().Set("Z")
        bore_color = tuple(c * 0.5 for c in color)
        mat_bore = self._make_material(stage, "gear_bore", bore_color, metallic, roughness)
        self._bind_material(bore.GetPrim(), mat_bore)

    def _spawn_bottle(self, position, ctx, scale=1.0):
        """Plastic coolant / lubricant bottle, cylinder body + smaller
        cylinder neck + sphere cap."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["bottle"]
        mat_body = self._make_material(stage, "bottle_body", color, metallic, roughness)
        cap_color = (0.15, 0.15, 0.15)
        mat_cap = self._make_material(stage, "bottle_cap", cap_color, 0.1, 0.6)
        self._make_parent(stage, position, scale)

        body = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/body")
        body.GetRadiusAttr().Set(0.035)
        body.GetHeightAttr().Set(0.14)
        body.GetAxisAttr().Set("Z")
        self._bind_material(body.GetPrim(), mat_body)

        neck = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/neck")
        neck.GetRadiusAttr().Set(0.015)
        neck.GetHeightAttr().Set(0.04)
        neck.GetAxisAttr().Set("Z")
        neck_xf = UsdGeom.Xformable(neck)
        neck_xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.09))
        self._bind_material(neck.GetPrim(), mat_body)

        cap = UsdGeom.Cylinder.Define(stage, _PROJECTILE_PRIM + "/cap")
        cap.GetRadiusAttr().Set(0.017)
        cap.GetHeightAttr().Set(0.02)
        cap.GetAxisAttr().Set("Z")
        cap_xf = UsdGeom.Xformable(cap)
        cap_xf.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.12))
        self._bind_material(cap.GetPrim(), mat_cap)

    def _spawn_wood_block(self, position, ctx, scale=1.0):
        """Wooden pallet offcut / block."""
        from pxr import UsdGeom, Gf
        stage = ctx.stage
        _, color, metallic, roughness = _OBJECT_PROPS["wood_block"]
        mat = self._make_material(stage, "wood", color, metallic, roughness)
        # Darker end-grain faces
        end_color = tuple(c * 0.75 for c in color)
        mat_end = self._make_material(stage, "wood_end", end_color, metallic, roughness)
        self._make_parent(stage, position, scale)

        main = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/block")
        main_xf = UsdGeom.Xformable(main)
        main_xf.AddScaleOp().Set(Gf.Vec3f(0.12, 0.06, 0.04))
        self._bind_material(main.GetPrim(), mat)

        # End cap (visual detail)
        end = UsdGeom.Cube.Define(stage, _PROJECTILE_PRIM + "/end")
        end_xf = UsdGeom.Xformable(end)
        end_xf.AddTranslateOp().Set(Gf.Vec3f(0.121, 0.0, 0.0))
        end_xf.AddScaleOp().Set(Gf.Vec3f(0.003, 0.06, 0.04))
        self._bind_material(end.GetPrim(), mat_end)

    # Dispatch table for spawn helpers.
    _SPAWN_FN = {
        "unknown_debris": _spawn_debris,
        "adjacent_part":  _spawn_adjacent_part,
        "tool":           _spawn_tool,
        "bolt":           _spawn_bolt,
        "pipe_section":   _spawn_pipe_section,
        "cardboard_box":  _spawn_cardboard_box,
        "metal_plate":    _spawn_metal_plate,
        "gear":           _spawn_gear,
        "bottle":         _spawn_bottle,
        "wood_block":     _spawn_wood_block,
    }

    # ── Targeting ──────────────────────────────────────────────────────

    @staticmethod
    def _get_link_world_pos(stage, path):
        from pxr import UsdGeom
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return None
        tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        return np.array([tf.GetRow(3)[0], tf.GetRow(3)[1], tf.GetRow(3)[2]])

    def _get_target_pos(self, ctx, target_link: str) -> np.ndarray | None:
        """Return the world position of *target_link* on the robot."""
        stage = ctx.stage
        if ctx.robot_prim_path:
            pos = self._get_link_world_pos(
                stage, ctx.robot_prim_path + "/" + target_link
            )
            if pos is not None:
                return pos
        # Fallback: try any available link.
        if ctx.robot_prim_path:
            for link in _TARGET_LINKS:
                pos = self._get_link_world_pos(
                    stage, ctx.robot_prim_path + "/" + link
                )
                if pos is not None:
                    return pos
        # Last resort: the pick cube.
        if ctx.cube_prim_path:
            pos = self._get_link_world_pos(stage, ctx.cube_prim_path)
            if pos is not None:
                return pos
        return None

    # ── Lifecycle hooks ───────────────────────────────────────────────

    def on_episode_setup(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if ctx.stage is None:
            return
        try:
            self._clear_projectile(ctx.stage)
            obj_type = params["object"]
            spawn_pos = np.array(params["_spawn_origin"])
            self._scale = params.get("_scale", 1.0)
            spawn_fn = self._SPAWN_FN.get(obj_type, self._spawn_debris)
            spawn_fn(self, spawn_pos, ctx, self._scale)
            self._projectile_spawned = True
            self._in_flight = False
            self._launch_pos = None
            self._launch_vel = None
            self._flight_elapsed = 0.0
            print(f"[Collision] Placed {obj_type} at {spawn_pos} scale={self._scale:.2f} (visual only)")
        except Exception as e:
            print(f"[Collision] on_episode_setup failed: {e}")
            import traceback
            traceback.print_exc()

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        """Compute the ballistic trajectory and begin visual flight."""
        if ctx.stage is None:
            return
        try:
            target_link = params.get("_target_link", "forearm_link")
            target_pos = self._get_target_pos(ctx, target_link)
            if target_pos is None:
                print("[Collision] on_start: no target found")
                return

            spawn = params.get("_spawn_origin")
            self._launch_pos = np.array(spawn, dtype=float) if spawn else np.array([0.45, 0.40, 0.15])
            delta = target_pos - self._launch_pos
            dist = np.linalg.norm(delta)
            if dist < 1e-4:
                return
            self._direction = delta / dist

            # Derive impact speed from desired impulse and actual mass.
            impulse = params["impact_impulse"]
            scale = params.get("_scale", 1.0)
            self._obj_type = params["object"]
            base_mass = _OBJECT_PROPS.get(self._obj_type, (0.1))[0]
            mass = base_mass * (scale ** 3)
            impact_speed = impulse / mass  # m/s needed at impact

            # Derive flight time from distance and desired impact speed,
            # clamped to keep the visual trajectory readable.
            self._t_flight = float(np.clip(
                dist / max(impact_speed, 1e-3),
                _T_FLIGHT_MIN, _T_FLIGHT_MAX))

            vx = delta[0] / self._t_flight
            vy = delta[1] / self._t_flight
            vz = delta[2] / self._t_flight + 0.5 * _GRAVITY * self._t_flight

            self._launch_vel = np.array([vx, vy, vz])
            self._flight_elapsed = 0.0
            self._in_flight = True
            self._impact_force_est = impulse / _CONTACT_DT  # for sensor corruption

            print(f"[Collision] START object={self._obj_type} "
                  f"target={target_link} "
                  f"impulse={impulse:.2f}kg·m/s "
                  f"mass={mass:.3f}kg "
                  f"speed={np.linalg.norm(self._launch_vel):.2f}m/s "
                  f"t_flight={self._t_flight:.2f}s "
                  f"dist={dist:.2f}m")
        except Exception as e:
            print(f"[Collision] on_start failed: {e}")
            import traceback
            traceback.print_exc()

    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        # ── Advance visual flight ────────────────────────────────────
        if self._in_flight and self._launch_vel is not None:
            self._flight_elapsed += ctx.sim_dt

            if self._flight_elapsed < self._t_flight - ctx.sim_dt:
                t = self._flight_elapsed
                pos = (self._launch_pos + self._launch_vel * t).copy()
                pos[2] -= 0.5 * _GRAVITY * t * t
                self._set_prim_position(ctx.stage, pos)
            else:
                # ── Handoff: add physics one step before target ──────
                # The prim has been visual-only until now (no physics
                # APIs at all).  We now add RigidBodyAPI + CollisionAPI
                # + MassAPI + CCD and bake the current flight velocity.
                # PhysX discovers a brand-new *dynamic* body and the
                # projectile naturally collides with the robot on the
                # next step.  Because the body has realistic mass
                # (0.05–0.3 kg), the collision force scales properly.
                from pxr import UsdPhysics, Gf

                t = self._flight_elapsed
                cur_vel = self._launch_vel.copy()
                cur_vel[2] -= _GRAVITY * t

                self._make_physics_body(ctx.stage, self._scale)

                prim = ctx.stage.GetPrimAtPath(_PROJECTILE_PRIM)
                if prim.IsValid():
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    rb.GetVelocityAttr().Set(
                        Gf.Vec3f(*[float(v) for v in cur_vel])
                    )

                self._in_flight = False
                print(f"[Collision] Handoff, dynamic body created, "
                      f"speed={np.linalg.norm(cur_vel):.2f}m/s")

        # ── Corrupt sensor data ──────────────────────────────────────
        if self._direction is None:
            return
        # Estimated contact force = impulse / contact_duration
        force_est = getattr(self, "_impact_force_est", 0.0)
        for i, axis in enumerate(["x", "y", "z"]):
            col = f"contact_force_{axis}_n"
            if col in ctx.sensor_data:
                try:
                    original = float(ctx.sensor_data[col])
                    ctx.sensor_data[col] = f"{original + force_est * self._direction[i]:.6f}"
                except (ValueError, TypeError):
                    pass

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        print(f"[Collision] END object={params['object']}")
        self._direction = None
        self._in_flight = False
        self._launch_vel = None
        self._impact_force_est = 0.0

    def on_episode_cleanup(self, params: Dict[str, Any], ctx: SimContext) -> None:
        if not self._projectile_spawned or ctx.stage is None:
            return
        try:
            self._clear_projectile(ctx.stage)
        except Exception:
            pass
