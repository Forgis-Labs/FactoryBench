"""
FactoryBench / pick_and_place / run.py
UR3 pick-and-place simulation for Isaac Sim 5.x.

Uses RMPFlow for reactive motion control with a fixed waypoint trajectory.
The robot follows the same nominal path every episode, adapting in real-time
to perception noise and physics randomization via the RMPFlow controller.

Run:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur3/pick_and_place/run.py [--headless] [--seed SEED] [--episodes N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the script directory is on sys.path so local imports work
_SCRIPT_DIR = str(Path(__file__).parent.resolve())
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
from isaacsim import SimulationApp

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument("--headless", action="store_true")
_parser.add_argument("--seed",     type=int, default=0)
_parser.add_argument("--episodes", type=int, default=0, help="0 = run forever")
_args = _parser.parse_args()

simulation_app = SimulationApp({
    "width": 1280, "height": 720,
    "headless": _args.headless,
})

# ---------------------------------------------------------------------------
# Deferred imports
# ---------------------------------------------------------------------------

import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.storage.native import get_assets_root_path
import isaacsim.robot_motion.motion_generation as mg
import yaml

from episode_logger import EpisodeLogger, collect_sensors, get_task_phase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_TASK_DIR = Path(__file__).parent.resolve()

def load_config() -> dict:
    with open(_TASK_DIR / "config" / "task.yaml") as f:
        return yaml.safe_load(f)

CFG = load_config()

ROBOT_PRIM       = CFG["robot"]["prim"]
EEF_PRIM         = CFG["robot"]["eef_prim"]
HOME_JOINTS      = np.array(CFG["robot"]["home_joints"])

CUBE_PRIM        = CFG["scene"]["cube_prim"]
CONVEYOR_NOMINAL = np.array(CFG["scene"]["conveyor_nominal"])
CONVEYOR_XY_RANGE= float(CFG["scene"]["conveyor_xy_range"])
BIN_POSITION     = np.array(CFG["scene"]["bin_position"])

SIM_DT = float(CFG["physics"]["sim_dt"])

CUBE_MASS_RANGE        = tuple(CFG["domain_randomization"]["cube_mass_range"])
CUBE_FRICTION_RANGE    = tuple(CFG["domain_randomization"]["cube_friction_range"])
CUBE_RESTITUTION_RANGE = tuple(CFG["domain_randomization"]["cube_restitution_range"])
CUBE_DIM_RANGE         = tuple(CFG["domain_randomization"]["cube_dim_w_range"])
CUBE_HEIGHT_RANGE      = tuple(CFG["domain_randomization"]["cube_dim_h_range"])

PERCEPTION_NOISE_XY_STD = float(CFG["noise"]["perception_xy_std"])
PERCEPTION_NOISE_Z_STD  = float(CFG["noise"]["perception_z_std"])

SUCTION_RADIUS = 0.10  # 100mm, generous to account for RMPFlow convergence error
LOG_DIR = str(_TASK_DIR / CFG["logging"]["log_dir"])

UR3_USD = "/Isaac/Robots/UniversalRobots/ur3/ur3.usd"

# Conveyor top surface z (center_z + half_height)
CONVEYOR_TOP_Z = CONVEYOR_NOMINAL[2] + 0.02


# ---------------------------------------------------------------------------
# Workpiece helpers
# ---------------------------------------------------------------------------

def sample_workpiece_dims(rng) -> np.ndarray:
    w = float(rng.uniform(*CUBE_DIM_RANGE))
    d = float(rng.uniform(*CUBE_DIM_RANGE))
    h = float(rng.uniform(*CUBE_HEIGHT_RANGE))
    return np.array([w, d, h])


def randomize_cube_pose(rng) -> np.ndarray:
    offset = rng.uniform(-CONVEYOR_XY_RANGE, CONVEYOR_XY_RANGE, size=2)
    pos = CONVEYOR_NOMINAL.copy()
    pos[0] += offset[0]
    pos[1] += offset[1]
    return pos


def randomize_cube_physics(cube, rng):
    mass        = float(rng.uniform(*CUBE_MASS_RANGE))
    friction    = float(rng.uniform(*CUBE_FRICTION_RANGE))
    restitution = float(rng.uniform(*CUBE_RESTITUTION_RANGE))
    randomize_cube_physics._last_restitution = restitution
    try:
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(CUBE_PRIM)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(mass)
        mat_prim = stage.GetPrimAtPath(CUBE_PRIM + "/PhysicsMaterial")
        if mat_prim.IsValid():
            UsdPhysics.MaterialAPI(mat_prim).GetStaticFrictionAttr().Set(friction)
            UsdPhysics.MaterialAPI(mat_prim).GetDynamicFrictionAttr().Set(friction * 0.85)
            UsdPhysics.MaterialAPI(mat_prim).GetRestitutionAttr().Set(restitution)
    except Exception:
        pass
    print(f"[Randomize] mass={mass:.3f}kg  friction={friction:.3f}  "
          f"restitution={restitution:.3f}")
    return mass, friction


def apply_perception_noise(pos: np.ndarray, rng) -> np.ndarray:
    noisy = pos.copy()
    noisy[0] += rng.normal(0, PERCEPTION_NOISE_XY_STD)
    noisy[1] += rng.normal(0, PERCEPTION_NOISE_XY_STD)
    noisy[2] += rng.normal(0, PERCEPTION_NOISE_Z_STD)
    return noisy


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_workcell(stage):
    """Build a realistic industrial manufacturing workcell with PBR materials."""

    # ---- Material helper ----
    def _mat(name, color, metallic=0.0, roughness=0.5):
        path = f"/World/Looks/{name}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    # ---- Geometry helpers ----
    def _box(path, pos, size, mat=None, collision=True):
        c = UsdGeom.Cube.Define(stage, path)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3f(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(size[0]/2, size[1]/2, size[2]/2))
        if collision:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if mat:
            UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat)
        return c

    def _cyl(path, pos, radius, height, mat=None, collision=False, axis="Z"):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.GetRadiusAttr().Set(radius)
        c.GetHeightAttr().Set(height)
        c.GetAxisAttr().Set(axis)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3f(*pos))
        if collision:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if mat:
            UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat)
        return c

    # ===== PBR Materials =====
    m_concrete   = _mat("concrete",       (0.48, 0.46, 0.42), metallic=0.0,  roughness=0.85)
    m_steel_dk   = _mat("steel_dark",     (0.22, 0.23, 0.25), metallic=0.85, roughness=0.35)
    m_steel_lt   = _mat("steel_brushed",  (0.58, 0.59, 0.60), metallic=0.80, roughness=0.28)
    m_aluminum   = _mat("aluminum",       (0.72, 0.73, 0.74), metallic=0.90, roughness=0.22)
    m_belt       = _mat("conveyor_belt",  (0.10, 0.10, 0.12), metallic=0.0,  roughness=0.75)
    m_conv_frame = _mat("conveyor_frame", (0.15, 0.28, 0.52), metallic=0.70, roughness=0.40)
    m_yellow     = _mat("safety_yellow",  (0.95, 0.78, 0.05), metallic=0.0,  roughness=0.55)
    m_bin        = _mat("bin_metal",      (0.32, 0.35, 0.37), metallic=0.75, roughness=0.42)
    m_cabinet    = _mat("cabinet",        (0.76, 0.76, 0.74), metallic=0.65, roughness=0.35)
    m_rubber     = _mat("rubber",         (0.06, 0.06, 0.07), metallic=0.0,  roughness=0.92)
    m_red        = _mat("estop_red",      (0.82, 0.08, 0.08), metallic=0.0,  roughness=0.50)
    m_green      = _mat("status_green",   (0.08, 0.75, 0.12), metallic=0.2,  roughness=0.35)
    m_fence      = _mat("fence_post",     (0.62, 0.63, 0.60), metallic=0.70, roughness=0.40)
    m_mesh       = _mat("wire_mesh",      (0.18, 0.18, 0.20), metallic=0.60, roughness=0.55)
    m_cable      = _mat("cable_tray",     (0.12, 0.12, 0.14), metallic=0.50, roughness=0.60)

    # ===== Concrete floor (visual overlay on default ground plane) =====
    _box("/World/cell/floor", [0, 0, -0.005], [3.0, 3.0, 0.01], m_concrete, collision=False)

    # Anti-fatigue rubber mat (operator standing area)
    _box("/World/cell/rubber_mat", [0.0, 0.55, 0.002], [1.0, 0.6, 0.004], m_rubber, collision=False)

    # ===== Industrial workbench (T-slot aluminum top on steel frame) =====
    # Table pushed out from robot base (robot at origin, base radius ~60mm)
    TABLE_H = 0.10
    TABLE_CX = 0.45   # table centre X -- cleared from robot base
    _box("/World/cell/table_top", [TABLE_CX, 0.0, TABLE_H - 0.006],
         [0.60, 0.50, 0.012], m_aluminum, collision=False)  # visual only

    # Steel frame legs (40mm square tube)
    for i, (lx, ly) in enumerate([(TABLE_CX - 0.25, -0.22), (TABLE_CX - 0.25, 0.22),
                                   (TABLE_CX + 0.25, -0.22), (TABLE_CX + 0.25, 0.22)]):
        _box(f"/World/cell/table_leg_{i}", [lx, ly, TABLE_H/2 - 0.006],
             [0.04, 0.04, TABLE_H - 0.012], m_steel_dk, collision=False)

    # Cross braces
    _box("/World/cell/brace_f", [TABLE_CX, -0.22, 0.03], [0.46, 0.03, 0.03], m_steel_dk, collision=False)
    _box("/World/cell/brace_b", [TABLE_CX,  0.22, 0.03], [0.46, 0.03, 0.03], m_steel_dk, collision=False)
    _box("/World/cell/brace_l", [TABLE_CX - 0.25, 0.00, 0.03], [0.03, 0.40, 0.03], m_steel_dk, collision=False)
    _box("/World/cell/brace_r", [TABLE_CX + 0.25, 0.00, 0.03], [0.03, 0.40, 0.03], m_steel_dk, collision=False)

    # ===== Robot mounting plate (machined aluminum) =====
    _box("/World/cell/mount_plate", [0.0, 0.0, -0.005],
         [0.18, 0.18, 0.01], m_aluminum, collision=False)

    # ===== Conveyor belt assembly =====
    # cx must keep belt near edge (x≥0.20) away from robot base at origin
    cx, cy = 0.45, 0.00
    cl, cw = 0.45, 0.24

    # Belt surface
    _box("/World/cell/conv_belt", [cx, cy, TABLE_H + 0.002],
         [cl, cw, 0.004], m_belt, collision=True)

    # Side rails (industrial blue)
    _box("/World/cell/conv_rail_l", [cx, cy + cw/2 + 0.015, TABLE_H + 0.015],
         [cl, 0.025, 0.03], m_conv_frame, collision=True)
    _box("/World/cell/conv_rail_r", [cx, cy - cw/2 - 0.015, TABLE_H + 0.015],
         [cl, 0.025, 0.03], m_conv_frame, collision=True)

    # End caps
    _box("/World/cell/conv_end_near", [cx - cl/2, cy, TABLE_H + 0.015],
         [0.02, cw + 0.06, 0.03], m_conv_frame, collision=False)
    _box("/World/cell/conv_end_far",  [cx + cl/2, cy, TABLE_H + 0.015],
         [0.02, cw + 0.06, 0.03], m_conv_frame, collision=False)

    # Rollers (visual detail)
    for i in range(7):
        rx = cx - cl/2 + 0.05 + i * (cl - 0.10) / 6
        _cyl(f"/World/cell/conv_roller_{i}", [rx, cy, TABLE_H - 0.005],
             radius=0.012, height=cw, mat=m_steel_lt, axis="Y")

    # ===== Parts collection bin (sheet metal) =====
    bp = BIN_POSITION
    bw, bd, bh = 0.22, 0.22, 0.10
    wt = 0.003  # 3mm sheet metal

    # Bin floor
    _box("/World/cell/bin_floor", [bp[0], bp[1], bp[2] - bh/2 + wt/2],
         [bw, bd, wt], m_bin, collision=True)

    # Bin walls
    for i, (ox, oy, sx, sy) in enumerate([
        (0, -bd/2 + wt/2, bw, wt), (0, bd/2 - wt/2, bw, wt),
        (-bw/2 + wt/2, 0, wt, bd), (bw/2 - wt/2, 0, wt, bd),
    ]):
        _box(f"/World/cell/bin_wall_{i}", [bp[0]+ox, bp[1]+oy, bp[2]],
             [sx, sy, bh], m_bin, collision=True)

    # Reinforcing rim
    for i, (ox, oy, sx, sy) in enumerate([
        (0, -bd/2, bw+0.01, 0.012), (0, bd/2, bw+0.01, 0.012),
        (-bw/2, 0, 0.012, bd+0.01), (bw/2, 0, 0.012, bd+0.01),
    ]):
        _box(f"/World/cell/bin_rim_{i}", [bp[0]+ox, bp[1]+oy, bp[2]+bh/2+0.004],
             [sx, sy, 0.008], m_bin, collision=False)

    # Bin stand (small steel table)
    stand_top_z = bp[2] - bh/2 - 0.005
    _box("/World/cell/bin_stand_top", [bp[0], bp[1], stand_top_z],
         [bw+0.04, bd+0.04, 0.006], m_steel_dk, collision=True)
    for i, (ox, oy) in enumerate([(-0.10, -0.10), (0.10, -0.10),
                                   (-0.10, 0.10), (0.10, 0.10)]):
        _box(f"/World/cell/bin_leg_{i}", [bp[0]+ox, bp[1]+oy, stand_top_z/2],
             [0.025, 0.025, stand_top_z], m_steel_dk, collision=False)

    # ===== Safety perimeter fence =====
    fh = 1.0  # fence height
    fz = fh / 2

    # Posts (aluminum extrusion)
    for i, (fx, fy) in enumerate([
        (-0.65, -0.55), (0.08, -0.55), (0.80, -0.55),
        (0.80, -0.05), (0.80, 0.45), (-0.65, 0.45),
    ]):
        _cyl(f"/World/cell/fence_post_{i}", [fx, fy, fz],
             radius=0.02, height=fh, mat=m_fence)

    # Mesh panels (thin boxes)
    pt = 0.005
    # Back wall (y=-0.55)
    _box("/World/cell/fence_back_l", [-0.285, -0.55, fz],
         [0.69, pt, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/fence_back_r", [0.44, -0.55, fz],
         [0.68, pt, fh-0.10], m_mesh, collision=False)
    # Right wall (x=0.80)
    _box("/World/cell/fence_right_l", [0.80, -0.30, fz],
         [pt, 0.46, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/fence_right_r", [0.80, 0.20, fz],
         [pt, 0.46, fh-0.10], m_mesh, collision=False)
    # Left wall partial (leave gap for operator access)
    _box("/World/cell/fence_left", [-0.65, -0.225, fz],
         [pt, 0.61, fh-0.10], m_mesh, collision=False)

    # Yellow safety top rails
    _box("/World/cell/rail_back",  [0.075, -0.55, fh+0.01],
         [1.49, 0.03, 0.02], m_yellow, collision=False)
    _box("/World/cell/rail_right", [0.80, -0.05, fh+0.01],
         [0.03, 1.04, 0.02], m_yellow, collision=False)
    _box("/World/cell/rail_left",  [-0.65, -0.225, fh+0.01],
         [0.03, 0.69, 0.02], m_yellow, collision=False)

    # ===== Control cabinet (robot controller, RAL 7035 light gray) =====
    cab_x, cab_y = -0.55, -0.40
    _box("/World/cell/cabinet_body", [cab_x, cab_y, 0.35],
         [0.30, 0.25, 0.70], m_cabinet, collision=False)
    _box("/World/cell/cab_handle", [cab_x+0.155, cab_y, 0.45],
         [0.01, 0.06, 0.005], m_steel_dk, collision=False)
    _box("/World/cell/cab_led", [cab_x+0.12, cab_y-0.10, 0.65],
         [0.015, 0.015, 0.015], m_green, collision=False)
    _box("/World/cell/cab_vent", [cab_x, cab_y, 0.08],
         [0.26, 0.21, 0.08], m_steel_dk, collision=False)

    # ===== Cable tray (floor channel from cabinet to robot) =====
    _box("/World/cell/cable_tray", [cab_x/2, cab_y/2, 0.012],
         [abs(cab_x)-0.10, 0.06, 0.02], m_cable, collision=False)

    # ===== Emergency stop post =====
    ex, ey = -0.50, 0.30
    _cyl("/World/cell/estop_post", [ex, ey, 0.45],
         radius=0.015, height=0.90, mat=m_yellow)
    _box("/World/cell/estop_plate", [ex, ey, 0.88],
         [0.06, 0.06, 0.01], m_yellow, collision=False)
    _cyl("/World/cell/estop_button", [ex, ey, 0.91],
         radius=0.025, height=0.03, mat=m_red)

    # ===== Safety floor markings (yellow tape) =====
    tt = 0.003
    tw = 0.05
    _box("/World/cell/tape_front", [0.0, -0.40, tt/2],
         [0.90, tw, tt], m_yellow, collision=False)
    _box("/World/cell/tape_right", [0.50, 0.0, tt/2],
         [tw, 0.85, tt], m_yellow, collision=False)
    _box("/World/cell/tape_left",  [-0.50, -0.10, tt/2],
         [tw, 0.65, tt], m_yellow, collision=False)

    # ===== Industrial overhead LED lighting =====
    for i, (lx, ly) in enumerate([(0.20, 0.0), (-0.20, -0.20), (0.50, -0.30)]):
        light = UsdLux.RectLight.Define(stage, f"/World/cell/led_panel_{i}")
        xf = UsdGeom.Xformable(light)
        xf.AddTranslateOp().Set(Gf.Vec3f(lx, ly, 1.80))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))
        light.GetWidthAttr().Set(0.30)
        light.GetHeightAttr().Set(0.15)
        light.GetIntensityAttr().Set(25000)
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.97, 0.92))  # cool white

        # Fixture housing (visual)
        _box(f"/World/cell/light_housing_{i}", [lx, ly, 1.79],
             [0.32, 0.17, 0.04], m_steel_lt, collision=False)

    # Ambient fill (subtle, prevents pitch-black shadows)
    dome = UsdLux.DomeLight.Define(stage, "/World/cell/ambient")
    dome.GetIntensityAttr().Set(200)
    dome.GetColorAttr().Set(Gf.Vec3f(0.85, 0.90, 1.0))

    print("[Scene] Manufacturing workcell built.")


def spawn_workpiece(world, rng, spawn_pos, dims, existing_cube=None) -> DynamicCuboid:
    color = np.array(rng.uniform(0.2, 0.9, 3))
    if existing_cube is None:
        cube = world.scene.add(DynamicCuboid(
            prim_path=CUBE_PRIM,
            name="pick_cube",
            position=spawn_pos,
            scale=dims,
            color=color,
            mass=0.5))
        return cube
    # For existing cubes: use Isaac Sim API to update scale (avoids corrupting
    # PhysX tensor views that raw USD xform-op writes cause).
    existing_cube.set_local_scale(dims)
    return existing_cube


# ---------------------------------------------------------------------------
# Suction gripper (physics joint based)
# ---------------------------------------------------------------------------

class SuctionGripper:
    """Kinematic suction gripper: tracks EE position each step while attached."""

    def __init__(self, flange_prim_path):
        self._flange   = flange_prim_path
        self._cube_obj = None
        self._offset   = np.zeros(3)  # ee_pos - cube_pos at attach time
        self.attached  = False

    def attach(self, cube_prim_path, ee_pos, cube_pos, cube_obj=None) -> bool:
        xy_err = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
        if xy_err > SUCTION_RADIUS:
            print(f"[Suction] Miss (XY err={xy_err*1000:.1f}mm > {SUCTION_RADIUS*1000:.0f}mm)")
            return False

        self._cube_obj = cube_obj
        self._cube_prim_path = cube_prim_path
        self._offset   = ee_pos - cube_pos  # remember relative offset
        self.attached  = True

        # Disable cube dynamics (make it kinematic) and disable its collision
        # to prevent PhysX impulse forces from destabilizing the robot arm
        if cube_obj is not None:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(cube_prim_path)
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.GetKinematicEnabledAttr().Set(True)
            self._set_cube_collision(prim, False)

        print(f"[Suction] Attached (XY err={xy_err*1000:.1f}mm)")
        return True

    @staticmethod
    def _set_cube_collision(prim, enabled: bool):
        """Enable/disable collision on the cube and all its child prims."""
        from pxr import Usd
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(enabled)

    def track(self, ee_pos):
        """Call every step while attached to move cube with EE."""
        if self.attached and self._cube_obj is not None:
            cube_target = ee_pos - self._offset
            self._cube_obj.set_world_pose(
                position=cube_target,
                orientation=np.array([1, 0, 0, 0]))

    def detach(self):
        if self._cube_obj is not None and self.attached:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(CUBE_PRIM)
            if prim.IsValid():
                rb = UsdPhysics.RigidBodyAPI(prim)
                rb.GetKinematicEnabledAttr().Set(False)
                self._set_cube_collision(prim, True)
        self._cube_obj = None
        self.attached  = False


# ---------------------------------------------------------------------------
# RMPFlow controller for UR3
# ---------------------------------------------------------------------------

class UR3RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt=SIM_DT):
        rmp_config = mg.interface_config_loader.load_supported_motion_policy_config(
            "UR3", "RMPflow"
        )
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**rmp_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt
        )
        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp
        )
        self._default_position, self._default_orientation = \
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)
        print("[RMPFlow] UR3 controller initialized.")

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_eef_pos(stage) -> np.ndarray:
    for p in (
        ROBOT_PRIM + "/wrist_3_link/tool0",
        ROBOT_PRIM + "/tool0",
        EEF_PRIM):
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            t = mat.ExtractTranslation()
            return np.array([t[0], t[1], t[2]])
    raise RuntimeError("EEF prim not found on stage")


# ---------------------------------------------------------------------------
# Trajectory waypoint generator
# ---------------------------------------------------------------------------

def build_waypoints(cube_pos, cube_dims, bin_pos):
    """
    Build the Cartesian waypoint sequence for one pick-and-place cycle.
    All positions are in world frame (meters).
    Returns list of (name, target_pos, target_orient_quat).

    A 'neutral' transit waypoint bridges the pick zone and bin zone so that
    RMPFlow does not get stuck in a configuration-space local minimum when
    the arm swings from one side of the workspace to the other.
    """
    cube_top_z = cube_pos[2] + cube_dims[2] / 2.0
    approach_height = 0.10  # clearance above cube top
    grasp_height = cube_top_z + 0.005  # just above cube top for suction contact
    lift_height = cube_top_z + approach_height

    bin_above_z = 0.25   # safe height above bin
    bin_place_z = 0.15   # release height (above table level)

    # Mid-workspace transit point, avoids RMPFlow local minima
    neutral_pos = np.array([0.20, -0.17, 0.28])

    # Orientation: tool pointing straight down, 180° rotation about X so tool0 Z
    # aligns with -world Z.  Quaternion (w,x,y,z) = (0,1,0,0).
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])

    waypoints = [
        ("above_cube",    np.array([cube_pos[0], cube_pos[1], cube_top_z + approach_height]), down_quat),
        ("descend",       np.array([cube_pos[0], cube_pos[1], grasp_height]),                 down_quat),
        # GRASP happens here (grasp_wp_idx = 2)
        ("lift",          np.array([cube_pos[0], cube_pos[1], lift_height]),                   down_quat),
        ("neutral_to_bin", neutral_pos,                                                        down_quat),
        ("above_bin",     np.array([bin_pos[0],  bin_pos[1],  bin_above_z]),                   down_quat),
        ("place_descend", np.array([bin_pos[0],  bin_pos[1],  bin_place_z]),                   down_quat),
        # RELEASE happens here (release_wp_idx = 6)
        ("retract_bin",   np.array([bin_pos[0],  bin_pos[1],  bin_above_z]),                   down_quat),
        ("neutral_to_home", neutral_pos,                                                       down_quat),
        ("home",          np.array([0.20, 0.00, 0.25]),                                       down_quat),
    ]
    return waypoints


def enforce_wrist_down(action):
    """Post-process RMPFlow action to guarantee the tool always points straight down.

    Uses the UR3 kinematic identity:
      wrist_2 = -pi/2
      wrist_1 = -pi/2 - shoulder_lift - elbow
    This ensures the cumulative rotation of the wrist compensates for the arm
    pose so the flange Z axis stays aligned with -world Z at all times.
    """
    pos = action.joint_positions
    if pos is not None:
        pos = np.array(pos, dtype=float)
        pos[4] = -np.pi / 2                             # wrist_2 fixed
        pos[3] = -np.pi / 2 - pos[1] - pos[2]          # wrist_1 compensates
        action.joint_positions = pos
    vel = action.joint_velocities
    if vel is not None:
        vel = np.array(vel, dtype=float)
        vel[3] = -(vel[1] + vel[2])  # derivative of the wrist_1 constraint
        vel[4] = 0.0
        action.joint_velocities = vel
    return action


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class S:
    INIT           = "INIT"
    MOVE_TO_WP     = "MOVE_TO_WP"
    WAIT_SETTLE    = "WAIT_SETTLE"
    ATTACH         = "ATTACH"
    DETACH         = "DETACH"
    DONE           = "DONE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.RandomState(_args.seed)

    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)

    # -- Step 1: GPU Physics Pipeline --
    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_broadphase_type("GPU")
    physics_context.enable_fabric(True)
    physics_context.enable_ccd(False)
    print("[GPU] GPU dynamics, broadphase, and fabric enabled.")

    # -- Step 5: Suppress GPU→CPU readback (fabric handles USD reads) --
    carb.settings.get_settings().set_bool("/physics/suppressReadback", True)
    print("[GPU] Physics readback suppressed.")

    stage = omni.usd.get_context().get_stage()

    # Ground plane (physics collider)
    world.scene.add_default_ground_plane()

    # Build realistic manufacturing environment
    TABLE_HEIGHT = 0.10
    build_workcell(stage)

    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + UR3_USD, ROBOT_PRIM)

    robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="ur3"))
    # Set default joints BEFORE reset (like the UR10 example)
    robot.set_joints_default_state(positions=HOME_JOINTS)

    cur_dims  = sample_workpiece_dims(rng)
    cur_spawn = randomize_cube_pose(rng)
    # Place cube on the work table surface
    cur_spawn[2] = TABLE_HEIGHT + cur_dims[2] / 2.0 + 0.005
    cube = spawn_workpiece(world, rng, cur_spawn, cur_dims)

    world.reset()

    # Disable gravity on robot to prevent arm instability (standard practice
    # for UR robots in Isaac Sim, their URDF has zero joint damping/friction).
    robot.disable_gravity()
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)

    # Explicitly set joints and step a few frames to stabilize physics
    robot.set_joint_positions(HOME_JOINTS)
    robot.set_joint_velocities(np.zeros(6))
    for _ in range(5):
        world.step(render=True)

    # Initialize RMPFlow controller
    rmp_controller = UR3RMPFlowController(
        name="ur3_rmpflow", robot_articulation=robot, physics_dt=SIM_DT
    )
    articulation_controller = robot.get_articulation_controller()

    # Validate EEF prim
    if not stage.GetPrimAtPath(EEF_PRIM).IsValid():
        carb.log_error(f"EEF prim {EEF_PRIM} not found. Aborting.")
        simulation_app.close()
        return

    logger  = EpisodeLogger(LOG_DIR)
    suction = SuctionGripper(flange_prim_path=EEF_PRIM)

    cur_mass, cur_friction = randomize_cube_physics(cube, rng)
    logger.init_sensors(world, robot, cube, stage,
                        flange_prim_path=EEF_PRIM, sim_dt=SIM_DT)

    # State machine variables
    state          = S.INIT
    step           = 0
    wait_timer     = 0
    episode        = 0
    sim_time       = 0.0
    plan_attempts  = 0
    plan_time_last = 0.0
    pick_attempts  = 0

    waypoints      = []
    wp_idx         = 0
    wp_step_count  = 0     # steps spent on current waypoint
    wp_timeout     = 600   # max steps per waypoint (~10s at 60Hz)
    grasp_wp_idx   = 2     # index after which to attach (after "descend")
    release_wp_idx = 6     # index after which to detach (after "place_descend")
    pos_threshold  = 0.055 # 55mm position error to consider waypoint reached
    cur_wp_name    = ""    # current waypoint name for task phase logging

    def transition(new_state):
        nonlocal state
        print(f"[{step:6d}] {state:20s} -> {new_state}")
        state = new_state

    def reset_episode(success: bool, reason: str = ""):
        nonlocal state, wait_timer, episode, cur_mass, cur_friction
        nonlocal cur_spawn, cur_dims, cube
        nonlocal sim_time, plan_attempts, plan_time_last, pick_attempts
        nonlocal waypoints, wp_idx, cur_wp_name

        logger.end_episode(
            episode, success=success, reason=reason,
            cube_mass=cur_mass, cube_friction=cur_friction,
            cube_restitution=getattr(randomize_cube_physics, "_last_restitution", 0.0),
            cube_dims=cur_dims, cube_spawn=cur_spawn,
            sim_time=sim_time,
            plan_attempts=plan_attempts,
            pick_attempts=pick_attempts)
        episode       += 1
        sim_time       = 0.0
        plan_attempts  = 0
        plan_time_last = 0.0
        pick_attempts  = 0

        if _args.episodes > 0 and episode >= _args.episodes:
            print(f"[FactoryBench] Reached {_args.episodes} episodes. Stopping.")
            logger.close()
            simulation_app.close()
            sys.exit(0)

        cur_dims  = sample_workpiece_dims(rng)
        cur_spawn = randomize_cube_pose(rng)
        cur_spawn[2] = 0.10 + cur_dims[2] / 2.0 + 0.005  # TABLE_HEIGHT + half cube + margin
        suction.detach()

        # Re-enable dynamics on cube before any changes
        stage_r = omni.usd.get_context().get_stage()
        cube_prim = stage_r.GetPrimAtPath(CUBE_PRIM)
        if cube_prim.IsValid():
            rb = UsdPhysics.RigidBodyAPI(cube_prim)
            rb.GetKinematicEnabledAttr().Set(False)

        # Update cube scale & physics (may invalidate tensor views)
        cube = spawn_workpiece(world, rng, cur_spawn, cur_dims, existing_cube=cube)
        cur_mass, cur_friction = randomize_cube_physics(cube, rng)
        print(f"[UR3] Episode {episode} -- dims={np.round(cur_dims*1000).astype(int)}mm  "
              f"spawn={np.round(cur_spawn,3)}  mass={cur_mass:.3f}kg")

        # Set default states BEFORE world.reset() so reset applies them
        robot.set_joints_default_state(positions=HOME_JOINTS)
        cube.set_default_state(position=cur_spawn,
                               orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        # world.reset() re-creates all tensor views (needed because scale
        # change invalidates them) and applies the default states set above.
        world.reset()

        # Must re-apply after each world.reset()
        robot.disable_gravity()
        robot.set_solver_position_iteration_count(64)
        robot.set_solver_velocity_iteration_count(64)

        rmp_controller.reset()
        robot.set_joint_positions(HOME_JOINTS)
        robot.set_joint_velocities(np.zeros(6))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))

        # Step a few frames to stabilize physics
        for _ in range(5):
            world.step(render=True)

        # Verify cube pose is sane
        _cpos, _ = cube.get_world_pose()
        if np.any(np.abs(_cpos) > 10.0):
            print(f"[WARN] Cube pos after reset is garbage: {_cpos}")

        logger.reset_diff_state()
        wait_timer = 0
        waypoints = []
        wp_idx = 0
        cur_wp_name = ""
        state = S.INIT

    print(f"[UR3] Episode {episode} -- dims={np.round(cur_dims*1000).astype(int)}mm  "
          f"spawn={np.round(cur_spawn,3)}  mass={cur_mass:.3f}kg")

    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        step     += 1
        sim_time += SIM_DT

        cube_pos, _ = cube.get_world_pose()
        cube_pos    = np.asarray(cube_pos)
        ee_pos      = get_eef_pos(stage)

        # Sanity check: if cube position is garbage, use the expected spawn
        if np.any(np.abs(cube_pos) > 10.0):
            if step % 60 == 0:
                print(f"[WARN] Cube pos garbage: {cube_pos}, using spawn pos")
            cube_pos = cur_spawn.copy()

        # Track cube with EE while attached (kinematic gripper)
        if suction.attached:
            suction.track(ee_pos)

        # Drop detection
        if cube_pos[2] < -0.15 and not suction.attached:
            print(f"[{step:6d}] Cube dropped. Resetting.")
            reset_episode(success=False, reason="dropped")
            continue

        # Timeout detection (45 seconds)
        if sim_time > 45.0:
            print(f"[{step:6d}] Episode timeout.")
            reset_episode(success=False, reason="timeout")
            continue

        # ------------------------------------------------------------------
        # State machine (sets _step_action and _ee_target for logging)
        # ------------------------------------------------------------------
        _step_action = None
        _ee_target = None
        _ee_target_quat = None
        if state == S.INIT:
            # Wait a few frames for physics to settle, then build waypoints
            wait_timer += 1
            if wait_timer >= 10:
                print(f"[{step:6d}] Cube pos: {np.round(cube_pos, 4)}  EE pos: {np.round(ee_pos, 4)}")
                perceived_pos = apply_perception_noise(cube_pos, rng)
                waypoints = build_waypoints(perceived_pos, cur_dims, BIN_POSITION)
                for i, (wn, wp, _) in enumerate(waypoints):
                    print(f"  wp[{i}] '{wn}': {np.round(wp, 4)}")
                wp_idx = 0
                wp_step_count = 0
                transition(S.MOVE_TO_WP)

        elif state == S.MOVE_TO_WP:
            if wp_idx >= len(waypoints):
                transition(S.DONE)
                continue

            wp_name, wp_pos, wp_quat = waypoints[wp_idx]
            cur_wp_name = wp_name
            wp_step_count += 1

            # Command RMPFlow toward this waypoint
            action = rmp_controller.forward(
                target_end_effector_position=wp_pos,
                target_end_effector_orientation=wp_quat)
            action = enforce_wrist_down(action)
            articulation_controller.apply_action(action)
            _step_action = action
            _ee_target = wp_pos
            _ee_target_quat = wp_quat

            # Debug: print progress every 120 steps (~2s)
            pos_err = float(np.linalg.norm(ee_pos - wp_pos))
            if wp_step_count % 120 == 0:
                print(f"[{step:6d}] wp='{wp_name}' ee={np.round(ee_pos,3)} "
                      f"tgt={np.round(wp_pos,3)} err={pos_err*1000:.1f}mm")

            # Check if we've reached the waypoint or timed out
            reached = pos_err < pos_threshold
            timed_out = wp_step_count >= wp_timeout

            if reached or timed_out:
                if timed_out:
                    print(f"[{step:6d}] Waypoint '{wp_name}' timeout (err={pos_err*1000:.1f}mm), advancing")
                else:
                    print(f"[{step:6d}] Reached waypoint '{wp_name}' (err={pos_err*1000:.1f}mm)")
                wp_idx += 1
                wp_step_count = 0

                # Check for grasp/release transitions
                if wp_idx == grasp_wp_idx:
                    pick_attempts += 1
                    transition(S.ATTACH)
                elif wp_idx == release_wp_idx:
                    transition(S.DETACH)

        elif state == S.ATTACH:
            # Keep commanding descend position to hold the robot steady
            _, hold_pos, hold_quat = waypoints[wp_idx - 1]
            action = rmp_controller.forward(
                target_end_effector_position=hold_pos,
                target_end_effector_orientation=hold_quat)
            action = enforce_wrist_down(action)
            articulation_controller.apply_action(action)
            _step_action = action
            _ee_target = hold_pos
            _ee_target_quat = hold_quat

            wait_timer += 1
            if wait_timer >= 20:
                attached = suction.attach(CUBE_PRIM, ee_pos, cube_pos, cube_obj=cube)
                wait_timer = 0
                if attached:
                    # Continue to next waypoint (lift)
                    transition(S.MOVE_TO_WP)
                else:
                    # Retry: back up to above_cube then descend again
                    wp_idx = 0  # restart from above_cube with SAME waypoints
                    transition(S.MOVE_TO_WP)

        elif state == S.DETACH:
            # Keep commanding place position to hold the robot steady
            _, hold_pos, hold_quat = waypoints[wp_idx - 1]
            action = rmp_controller.forward(
                target_end_effector_position=hold_pos,
                target_end_effector_orientation=hold_quat)
            action = enforce_wrist_down(action)
            articulation_controller.apply_action(action)
            _step_action = action
            _ee_target = hold_pos
            _ee_target_quat = hold_quat

            wait_timer += 1
            if wait_timer >= 20:
                suction.detach()
                wait_timer = 0
                # Continue to retract waypoints
                transition(S.MOVE_TO_WP)

        # Sensor logging, record everything (after state machine so action is available)
        logger.step(collect_sensors(
            logger, episode, step, sim_time, state,
            robot, cube, stage, suction,
            cur_mass, cur_friction,
            cur_restitution=getattr(randomize_cube_physics, "_last_restitution", 0.0),
            cur_dims=cur_dims,
            plan_attempts=plan_attempts,
            plan_time_last=plan_time_last,
            pick_attempts=pick_attempts,
            planned_action=_step_action,
            task_phase=get_task_phase(state, cur_wp_name),
            ee_target_pos=_ee_target,
            ee_target_quat=_ee_target_quat))

        if state == S.DONE:
            wait_timer += 1
            if wait_timer >= 60:
                print(f"[{step:6d}] Cycle complete.")
                reset_episode(success=True, reason="placed")
                wait_timer = 0

    logger.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
