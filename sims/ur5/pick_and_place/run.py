"""
FactoryBench / ur5 / pick_and_place / run.py
UR5 + Robotiq 2F-85 pick-and-place simulation for Isaac Sim 5.x.

Uses RMPFlow for reactive motion control with a fixed waypoint trajectory.
The UR5 arm is loaded from the standard Isaac Sim asset; the Robotiq 2F-85
gripper is loaded separately and attached via a physics FixedJoint at the wrist.

Run:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur5/pick_and_place/run.py [--headless] [--seed SEED] [--episodes N] [--events]
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
# CLI, only parse when run directly (not when imported)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--headless", action="store_true")
    _parser.add_argument("--seed",     type=int, default=0)
    _parser.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    _parser.add_argument("--events",   action="store_true", help="Enable random event injection")
    _parser.add_argument("--max_events_per_episode", type=int, default=2,
                         help="Max events to inject per episode (min is always 0)")
    _parser.add_argument("--event_i", type=int, default=None,
                         help="Debug: force event with this ID every episode")
    _parser.add_argument("--run_type", type=str, default=None,
                         help="Tag for logging (e.g. 'baseline' or 'counterfactual')")
    _parser.add_argument("--log_dir", type=str, default=None,
                         help="Override log output directory")
    _parser.add_argument("--start_episode", type=int, default=0,
                         help="Skip this many episodes (fast-forward RNG, no logging)")
    _args = _parser.parse_args()

    # --event_i implies --events
    if _args.event_i is not None:
        _args.events = True

    simulation_app = SimulationApp({
        "width": 1280, "height": 720,
        "headless": _args.headless,
    })
else:
    # Imported as a module, provide dummy _args so constants can be loaded
    class _args:
        headless = True
        seed = 0
        episodes = 0
        events = False
        max_events_per_episode = 2
        event_i = None

# ---------------------------------------------------------------------------
# Deferred imports, these require SimulationApp to exist.
# When imported, the caller must have created SimulationApp first.
# ---------------------------------------------------------------------------

import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.controllers import PickPlaceController
import isaacsim.robot_motion.motion_generation as mg
import yaml

from episode_logger import EpisodeLogger, collect_sensors, get_task_phase

# Event injection (optional)
_FACTORYBENCH_DIR = str(Path(__file__).parent.parent.parent.resolve())
if _FACTORYBENCH_DIR not in sys.path:
    sys.path.insert(0, _FACTORYBENCH_DIR)

if _args.events:
    from event_injection import EventScheduler, SimContext, BUILTIN_APPLICATORS

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
CUBE_DIM_W_RANGE       = tuple(CFG["domain_randomization"]["cube_dim_w_range"])
CUBE_DIM_D_RANGE       = tuple(CFG["domain_randomization"]["cube_dim_d_range"])
CUBE_HEIGHT_RANGE      = tuple(CFG["domain_randomization"]["cube_dim_h_range"])

PERCEPTION_NOISE_XY_STD = float(CFG["noise"]["perception_xy_std"])
PERCEPTION_NOISE_Z_STD  = float(CFG["noise"]["perception_z_std"])
JOINT_NOISE_STD         = float(CFG["noise"]["joint_std"])

GRASP_XY_THRESHOLD = 0.12  # 120mm, max XY error for grasp attempt
LOG_DIR = str(_TASK_DIR / CFG["logging"]["log_dir"])

TABLE_HEIGHT = 0.10
EEF_INITIAL_HEIGHT = 0.42
EVENTS_DT = [0.008, 0.005, 0.04, 0.008, 0.008, 0.004, 0.008, 0.10, 0.016, 0.012]

UR5_USD = "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
ROBOTIQ_USD = str(Path(__file__).parent.parent.parent / "assets" / "robotiq_2f_85" / "robotiq_2f_85.usd")
ROBOTIQ_PRIM = "/World/ur5/robotiq"
ROBOTIQ_BASE = ROBOTIQ_PRIM  # URDF-imported gripper has links directly under root

# Conveyor top surface z (center_z + half_height)
CONVEYOR_TOP_Z = CONVEYOR_NOMINAL[2] + 0.02

# Gripper finger_joint base close angle (degrees).
# Actual close target is increased linearly with cube mass to maintain
# grip force, heavier objects get a larger overshoot past contact.
GRIPPER_CLOSE_DEG_BASE = 56.0
GRIPPER_CLOSE_DEG_MAX  = 68.0   # for heaviest cubes

# Distance from flange to the Robotiq 2F-85 finger pad grasp centre (metres)
GRIPPER_TCP_Z = 0.150


# ---------------------------------------------------------------------------
# Workpiece helpers
# ---------------------------------------------------------------------------

def sample_workpiece_dims(rng) -> np.ndarray:
    a = float(rng.uniform(*CUBE_DIM_W_RANGE))  # narrow side
    b = float(rng.uniform(*CUBE_DIM_D_RANGE))  # wide side
    h = float(rng.uniform(*CUBE_HEIGHT_RANGE))
    # dims[1] is always the narrowest horizontal side so the gripper
    # (which closes along Y) grabs the thin dimension.
    narrow, wide = min(a, b), max(a, b)
    return np.array([wide, narrow, h])


def randomize_cube_pose(rng):
    """Returns (position, yaw_angle_rad)."""
    offset = rng.uniform(-CONVEYOR_XY_RANGE, CONVEYOR_XY_RANGE, size=2)
    pos = CONVEYOR_NOMINAL.copy()
    pos[0] += offset[0]
    pos[1] += offset[1]
    yaw = float(rng.uniform(0, 2 * np.pi))
    return pos, yaw


def _apply_physics_material(stage, prim_path, static_friction, dynamic_friction,
                            restitution=0.0):
    """Create or update a physics material on a prim, ensuring the attributes
    exist and PhysX friction combine mode is set to 'max' so the higher
    friction of the two contacting surfaces dominates.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    # Find existing bound physics material
    mat_prim = None
    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound = binding_api.GetDirectBinding("physics")
    if bound and bound.GetMaterial():
        mp = bound.GetMaterial().GetPrim()
        if mp.IsValid():
            mat_prim = mp

    # Create material if none found
    if mat_prim is None:
        mat_path = prim_path + "/PhysMat"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            mat_prim = mat.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.weakerThanDescendants,
            "physics")

    # Apply UsdPhysics.MaterialAPI and PhysxSchema.PhysxMaterialAPI
    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat = UsdPhysics.MaterialAPI(mat_prim)

    # Use Create (not Get) to guarantee attributes exist
    phys_mat.CreateStaticFrictionAttr().Set(static_friction)
    phys_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_mat.CreateRestitutionAttr().Set(restitution)

    # Set PhysX friction combine mode to "max", the higher friction
    # of the two contacting surfaces is used.  Without this, PhysX
    # defaults to "average", which halves friction when one surface
    # has no material (friction=0).
    if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("average")
    physx_mat.CreateRestitutionCombineModeAttr().Set("average")

    return mat_prim


def _ensure_surface_friction(stage, surface_prim_path, friction=0.8):
    """Apply a physics material to a collision surface (e.g. conveyor belt)
    if one doesn't already exist.  Called once during scene setup."""
    prim = stage.GetPrimAtPath(surface_prim_path)
    if not prim.IsValid():
        return
    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound = binding_api.GetDirectBinding("physics")
    if bound and bound.GetMaterial():
        return  # already has a physics material
    _apply_physics_material(stage, surface_prim_path, friction, friction * 0.85)
    print(f"[Scene] Applied physics material to {surface_prim_path} (μ={friction})")


def randomize_cube_physics(cube, rng):
    mass        = float(rng.uniform(*CUBE_MASS_RANGE))
    friction    = float(rng.uniform(*CUBE_FRICTION_RANGE))
    restitution = float(rng.uniform(*CUBE_RESTITUTION_RANGE))
    randomize_cube_physics._last_restitution = restitution
    try:
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(CUBE_PRIM)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        UsdPhysics.MassAPI(prim).CreateMassAttr().Set(mass)
        _apply_physics_material(stage, CUBE_PRIM, friction, friction * 0.85,
                                restitution)
    except Exception as e:
        print(f"[Randomize] WARNING: failed to set physics: {e}")
    print(f"[Randomize] mass={mass:.3f}kg  friction={friction:.3f}  "
          f"restitution={restitution:.3f}")
    return mass, friction


# ---------------------------------------------------------------------------
# HUD overlay, on-screen display of cube mass & friction
# ---------------------------------------------------------------------------
_hud_window = None
_hud_mass_label = None
_hud_friction_label = None


def setup_hud():
    """Create a small floating window showing cube mass & friction.

    Uses explicit screen position (top-left), NOT docked into Viewport,
    which can silently fail in headless-first configurations.
    The window reference is kept in a global to prevent garbage collection.
    """
    global _hud_window, _hud_mass_label, _hud_friction_label
    if _args.headless:
        return
    try:
        import omni.ui as ui
        _hud_window = ui.Window(
            "Cube Info", width=220, height=80,
            position_x=20, position_y=20,
            flags=(ui.WINDOW_FLAGS_NO_RESIZE
                   | ui.WINDOW_FLAGS_NO_SCROLLBAR
                   | ui.WINDOW_FLAGS_NO_COLLAPSE
                   | ui.WINDOW_FLAGS_NO_MOVE))
        with _hud_window.frame:
            with ui.VStack(spacing=4, height=0):
                _hud_mass_label = ui.Label(
                    "mass: -- kg",
                    style={"font_size": 22, "color": 0xFFFFFFFF})
                _hud_friction_label = ui.Label(
                    "friction: --",
                    style={"font_size": 22, "color": 0xFFFFFFFF})
        print("[HUD] Cube info overlay created")
    except Exception as e:
        print(f"[HUD] Could not create overlay: {e}")


def update_hud(mass: float, friction: float):
    """Update the HUD text.  Safe to call even if the HUD failed to init."""
    if _hud_mass_label is not None:
        _hud_mass_label.text = f"mass: {mass:.3f} kg"
    if _hud_friction_label is not None:
        _hud_friction_label.text = f"friction: {friction:.2f}"


# ---------------------------------------------------------------------------
# Event indicator overlay
# ---------------------------------------------------------------------------
_evt_window = None
_evt_label = None
_evt_params_label = None
_evt_rect = None


def setup_event_indicator():
    """Create a small window that shows active event info and variables."""
    global _evt_window, _evt_label, _evt_params_label, _evt_rect
    if _args.headless:
        return
    try:
        import omni.ui as ui
        _evt_window = ui.Window(
            "Event", width=360, height=70,
            position_x=20, position_y=110,
            flags=(ui.WINDOW_FLAGS_NO_RESIZE
                   | ui.WINDOW_FLAGS_NO_SCROLLBAR
                   | ui.WINDOW_FLAGS_NO_COLLAPSE
                   | ui.WINDOW_FLAGS_NO_MOVE
                   | ui.WINDOW_FLAGS_NO_TITLE_BAR))
        with _evt_window.frame:
            with ui.ZStack(height=0):
                _evt_rect = ui.Rectangle(
                    style={"background_color": 0xFF333333,
                           "border_radius": 4})
                with ui.VStack(spacing=2):
                    _evt_label = ui.Label(
                        "  No active event",
                        style={"font_size": 18, "color": 0xFF888888},
                        alignment=ui.Alignment.LEFT_CENTER)
                    _evt_params_label = ui.Label(
                        "",
                        style={"font_size": 14, "color": 0xFFCCCCCC},
                        alignment=ui.Alignment.LEFT_CENTER)
        print("[HUD] Event indicator created")
    except Exception as e:
        print(f"[HUD] Event indicator failed: {e}")


def _format_params(params: dict) -> str:
    """Format event params as a compact string, skipping internal keys."""
    parts = []
    for k, v in params.items():
        if k.startswith("_"):
            continue
        if k == "delta":
            # Show friction reduction as a percentage
            parts.append(f"reduction={v*100:.0f}%")
        elif k == "reduction_pct":
            continue  # already shown via delta
        elif k == "impact_impulse":
            parts.append(f"impulse={v:.2f}kg·m/s")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    return "  " + "  ".join(parts) if parts else ""


def update_event_indicator(active_events):
    """Update the event indicator with the current active event(s)."""
    if _evt_label is None:
        return
    if not active_events:
        _evt_label.text = "  No active event"
        _evt_label.set_style({"font_size": 18, "color": 0xFF888888})
        if _evt_params_label is not None:
            _evt_params_label.text = ""
        if _evt_rect is not None:
            _evt_rect.set_style({"background_color": 0xFF333333,
                                 "border_radius": 4})
        return

    evt = active_events[0]
    _evt_label.text = f"  EVENT: {evt.event_name} (id={evt.event_id})"
    _evt_label.set_style({"font_size": 18, "color": 0xFFFFFFFF})
    if _evt_params_label is not None:
        _evt_params_label.text = _format_params(evt.params)
    if _evt_rect is not None:
        _evt_rect.set_style({"background_color": 0xFF0000CC,
                             "border_radius": 4})


def yaw_to_quat(yaw_rad):
    """Convert a Z-axis yaw angle to a quaternion [w, x, y, z]."""
    c, s = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
    return np.array([c, 0.0, 0.0, s])


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
    """Build a realistic industrial manufacturing workcell with PBR materials.

    Scaled for the UR5's ~850mm reach: larger table, conveyor further out,
    bin further to the side, wider safety perimeter.
    """

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

    # ===== Concrete floor =====
    _box("/World/cell/floor", [0, 0, -0.005], [4.0, 4.0, 0.01], m_concrete, collision=False)
    _box("/World/cell/rubber_mat", [0.0, 0.70, 0.002], [1.2, 0.7, 0.004], m_rubber, collision=False)

    # ===== Industrial workbench =====
    TABLE_H = 0.10
    TABLE_CX = 0.55
    _box("/World/cell/table_top", [TABLE_CX, 0.0, TABLE_H - 0.006],
         [0.70, 0.60, 0.012], m_aluminum, collision=False)
    for i, (lx, ly) in enumerate([(TABLE_CX - 0.30, -0.27), (TABLE_CX - 0.30, 0.27),
                                   (TABLE_CX + 0.30, -0.27), (TABLE_CX + 0.30, 0.27)]):
        _box(f"/World/cell/table_leg_{i}", [lx, ly, TABLE_H/2 - 0.006],
             [0.05, 0.05, TABLE_H - 0.012], m_steel_dk, collision=False)
    _box("/World/cell/brace_f", [TABLE_CX, -0.27, 0.03], [0.56, 0.04, 0.04], m_steel_dk, collision=False)
    _box("/World/cell/brace_b", [TABLE_CX,  0.27, 0.03], [0.56, 0.04, 0.04], m_steel_dk, collision=False)
    _box("/World/cell/brace_l", [TABLE_CX - 0.30, 0.00, 0.03], [0.04, 0.50, 0.04], m_steel_dk, collision=False)
    _box("/World/cell/brace_r", [TABLE_CX + 0.30, 0.00, 0.03], [0.04, 0.50, 0.04], m_steel_dk, collision=False)

    # Robot mounting plate
    _box("/World/cell/mount_plate", [0.0, 0.0, -0.005], [0.22, 0.22, 0.01], m_aluminum, collision=False)

    # ===== Conveyor belt assembly =====
    cx, cy = 0.55, 0.00
    cl, cw = 0.60, 0.28
    # Flat belt surface, thick enough to be smooth, collision-enabled
    _box("/World/cell/conv_belt", [cx, cy, TABLE_H + 0.003], [cl, cw, 0.006], m_belt, collision=True)
    _box("/World/cell/conv_rail_l", [cx, cy + cw/2 + 0.018, TABLE_H + 0.018],
         [cl, 0.030, 0.036], m_conv_frame, collision=True)
    _box("/World/cell/conv_rail_r", [cx, cy - cw/2 - 0.018, TABLE_H + 0.018],
         [cl, 0.030, 0.036], m_conv_frame, collision=True)
    _box("/World/cell/conv_end_near", [cx - cl/2, cy, TABLE_H + 0.018],
         [0.025, cw + 0.07, 0.036], m_conv_frame, collision=False)
    _box("/World/cell/conv_end_far",  [cx + cl/2, cy, TABLE_H + 0.018],
         [0.025, cw + 0.07, 0.036], m_conv_frame, collision=False)
    # Rollers sit well below the belt, visible through the sides but
    # don't protrude above the belt surface.  No collision.
    for i in range(9):
        rx = cx - cl/2 + 0.05 + i * (cl - 0.10) / 8
        _cyl(f"/World/cell/conv_roller_{i}", [rx, cy, TABLE_H - 0.012],
             radius=0.010, height=cw, mat=m_steel_lt, axis="Y", collision=False)

    # ===== Parts collection bin =====
    bp = BIN_POSITION
    bw, bd, bh = 0.28, 0.28, 0.12
    wt = 0.003
    _box("/World/cell/bin_floor", [bp[0], bp[1], bp[2] - bh/2 + wt/2], [bw, bd, wt], m_bin, collision=True)
    for i, (ox, oy, sx, sy) in enumerate([
        (0, -bd/2 + wt/2, bw, wt), (0, bd/2 - wt/2, bw, wt),
        (-bw/2 + wt/2, 0, wt, bd), (bw/2 - wt/2, 0, wt, bd),
    ]):
        _box(f"/World/cell/bin_wall_{i}", [bp[0]+ox, bp[1]+oy, bp[2]], [sx, sy, bh], m_bin, collision=True)
    for i, (ox, oy, sx, sy) in enumerate([
        (0, -bd/2, bw+0.01, 0.014), (0, bd/2, bw+0.01, 0.014),
        (-bw/2, 0, 0.014, bd+0.01), (bw/2, 0, 0.014, bd+0.01),
    ]):
        _box(f"/World/cell/bin_rim_{i}", [bp[0]+ox, bp[1]+oy, bp[2]+bh/2+0.004],
             [sx, sy, 0.008], m_bin, collision=False)
    stand_top_z = bp[2] - bh/2 - 0.005
    _box("/World/cell/bin_stand_top", [bp[0], bp[1], stand_top_z],
         [bw+0.06, bd+0.06, 0.008], m_steel_dk, collision=True)
    for i, (ox, oy) in enumerate([(-0.13, -0.13), (0.13, -0.13), (-0.13, 0.13), (0.13, 0.13)]):
        _box(f"/World/cell/bin_leg_{i}", [bp[0]+ox, bp[1]+oy, stand_top_z/2],
             [0.03, 0.03, stand_top_z], m_steel_dk, collision=False)

    # ===== Safety perimeter fence =====
    fh = 1.2
    fz = fh / 2
    for i, (fx, fy) in enumerate([
        (-0.80, -0.70), (0.10, -0.70), (1.00, -0.70),
        (1.00, -0.10), (1.00, 0.55), (-0.80, 0.55),
    ]):
        _cyl(f"/World/cell/fence_post_{i}", [fx, fy, fz], radius=0.022, height=fh, mat=m_fence)
    pt = 0.005
    _box("/World/cell/fence_back_l", [-0.35, -0.70, fz], [0.86, pt, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/fence_back_r", [0.55, -0.70, fz], [0.86, pt, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/fence_right_l", [1.00, -0.40, fz], [pt, 0.56, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/fence_right_r", [1.00, 0.225, fz], [pt, 0.61, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/fence_left", [-0.80, -0.30, fz], [pt, 0.76, fh-0.10], m_mesh, collision=False)
    _box("/World/cell/rail_back",  [0.10, -0.70, fh+0.01], [1.84, 0.035, 0.025], m_yellow, collision=False)
    _box("/World/cell/rail_right", [1.00, -0.075, fh+0.01], [0.035, 1.29, 0.025], m_yellow, collision=False)
    _box("/World/cell/rail_left",  [-0.80, -0.30, fh+0.01], [0.035, 0.84, 0.025], m_yellow, collision=False)

    # ===== Control cabinet =====
    cab_x, cab_y = -0.70, -0.50
    _box("/World/cell/cabinet_body", [cab_x, cab_y, 0.40], [0.35, 0.30, 0.80], m_cabinet, collision=False)
    _box("/World/cell/cab_handle", [cab_x+0.18, cab_y, 0.50], [0.012, 0.07, 0.006], m_steel_dk, collision=False)
    _box("/World/cell/cab_led", [cab_x+0.14, cab_y-0.12, 0.75], [0.018, 0.018, 0.018], m_green, collision=False)
    _box("/World/cell/cab_vent", [cab_x, cab_y, 0.08], [0.30, 0.25, 0.10], m_steel_dk, collision=False)
    _box("/World/cell/cable_tray", [cab_x/2, cab_y/2, 0.014],
         [abs(cab_x)-0.10, 0.07, 0.024], m_cable, collision=False)

    # ===== Emergency stop =====
    ex, ey = -0.65, 0.38
    _cyl("/World/cell/estop_post", [ex, ey, 0.50], radius=0.018, height=1.00, mat=m_yellow)
    _box("/World/cell/estop_plate", [ex, ey, 0.98], [0.07, 0.07, 0.012], m_yellow, collision=False)
    _cyl("/World/cell/estop_button", [ex, ey, 1.01], radius=0.028, height=0.035, mat=m_red)

    # ===== Safety floor markings =====
    tt, tw = 0.003, 0.05
    _box("/World/cell/tape_front", [0.0, -0.52, tt/2], [1.10, tw, tt], m_yellow, collision=False)
    _box("/World/cell/tape_right", [0.62, 0.0, tt/2], [tw, 1.10, tt], m_yellow, collision=False)
    _box("/World/cell/tape_left",  [-0.62, -0.12, tt/2], [tw, 0.85, tt], m_yellow, collision=False)

    # ===== Overhead LED lighting =====
    for i, (lx, ly) in enumerate([(0.25, 0.0), (-0.25, -0.25), (0.60, -0.40), (0.0, 0.30)]):
        light = UsdLux.RectLight.Define(stage, f"/World/cell/led_panel_{i}")
        xf = UsdGeom.Xformable(light)
        xf.AddTranslateOp().Set(Gf.Vec3f(lx, ly, 2.00))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))
        light.GetWidthAttr().Set(0.35)
        light.GetHeightAttr().Set(0.18)
        light.GetIntensityAttr().Set(28000)
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.97, 0.92))
        _box(f"/World/cell/light_housing_{i}", [lx, ly, 1.99],
             [0.37, 0.20, 0.04], m_steel_lt, collision=False)
    dome = UsdLux.DomeLight.Define(stage, "/World/cell/ambient")
    dome.GetIntensityAttr().Set(200)
    dome.GetColorAttr().Set(Gf.Vec3f(0.85, 0.90, 1.0))

    print("[Scene] Manufacturing workcell built (UR5 layout).")


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
    existing_cube.set_local_scale(dims)
    # Update visual color each episode
    try:
        from pxr import Gf
        stage = omni.usd.get_context().get_stage()
        color_attr = stage.GetPrimAtPath(CUBE_PRIM).GetAttribute("primvars:displayColor")
        if color_attr.IsValid():
            color_attr.Set([Gf.Vec3f(*color)])
    except Exception:
        pass
    return existing_cube


# ---------------------------------------------------------------------------
# Robotiq gripper attachment and control
# ---------------------------------------------------------------------------

def setup_gripper(stage):
    """Attach the Robotiq base_link to the UR5 wrist via a FixedJoint.

    The gripper's ArticulationRootAPI is removed so it merges into the
    UR5 articulation tree (one unified articulation, stable physics).
    The joint connects to wrist_3_link (a rigid body); the flange
    offset is encoded in localPos0.
    """
    WRIST_BODY = ROBOT_PRIM + "/wrist_3_link"
    base_path = ROBOTIQ_BASE + "/robotiq_arg2f_base_link"
    base_prim = stage.GetPrimAtPath(base_path)

    # Ensure base_link is a dynamic rigid body (NOT kinematic)
    if base_prim.IsValid() and base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(base_prim)
        rb.GetKinematicEnabledAttr().Set(False)

    # Remove ArticulationRootAPI from gripper prims so the gripper
    # does not form its own articulation.  The FixedJoint will merge
    # it into the UR5 articulation instead.
    for check_path in [ROBOTIQ_PRIM, base_path]:
        p = stage.GetPrimAtPath(check_path)
        if p.IsValid():
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[Gripper] Removed ArticulationRootAPI from {check_path}")
            if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                print(f"[Gripper] Removed PhysxArticulationAPI from {check_path}")

    # Clear any stale Xform on the gripper root, the articulation
    # solver computes body positions from joint angles, so a static
    # parent transform would only confuse the renderer.
    grip_xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOTIQ_PRIM))
    grip_xf.ClearXformOpOrder()

    # Compute flange position offset relative to wrist_3_link.
    # Only the translation is used, the flange's local rotation would
    # rotate the gripper to the side.  The gripper base_link Z axis
    # should align with wrist_3_link Z (both point along the tool axis).
    flange_prim = stage.GetPrimAtPath(EEF_PRIM)
    if flange_prim.IsValid():
        offset_xf = UsdGeom.Xformable(flange_prim).GetLocalTransformation(
            Usd.TimeCode.Default()
        )
        offset_pos = offset_xf.ExtractTranslation()
    else:
        offset_pos = Gf.Vec3d(0, 0, 0)

    # Create FixedJoint, becomes part of the UR5 articulation tree
    joint_path = ROBOTIQ_BASE + "/wrist_fixed_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.GetBody0Rel().SetTargets([Sdf.Path(WRIST_BODY)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(base_path)])
        joint.GetLocalPos0Attr().Set(
            Gf.Vec3f(float(offset_pos[0]), float(offset_pos[1]),
                      float(offset_pos[2]))
        )
        joint.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))  # identity
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        print(f"[Gripper] FixedJoint: {WRIST_BODY} -> {base_path}  "
              f"offset=({offset_pos[0]:.4f}, {offset_pos[1]:.4f}, "
              f"{offset_pos[2]:.4f})  rot=identity")
    else:
        print("[Gripper] FixedJoint already exists")

    # Add PhysxMimicJointAPI to each mimic joint so all fingers follow
    # finger_joint.  The URDF import doesn't create these constraints.
    # Add PhysxMimicJointAPI, detect each joint's physics:axis to use
    # the correct mimic instance name (rotX, rotY, or rotZ).
    finger_joint_path = Sdf.Path(ROBOTIQ_BASE + "/joints/finger_joint")
    _MIMIC_JOINTS = [
        ("right_outer_knuckle_joint", -1.0),
        ("left_inner_knuckle_joint",  -1.0),
        ("right_inner_knuckle_joint", -1.0),
        ("left_inner_finger_joint",    1.0),
        ("right_inner_finger_joint",   1.0),
    ]

    def _set_or_create(prim, name, val_type, val):
        attr = prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(val)
        else:
            prim.CreateAttribute(name, val_type).Set(val)

    for jname, gearing in _MIMIC_JOINTS:
        jpath = ROBOTIQ_BASE + "/joints/" + jname
        jprim = stage.GetPrimAtPath(jpath)
        if not jprim.IsValid():
            continue
        # Detect the joint's rotation axis
        axis_attr = jprim.GetAttribute("physics:axis")
        axis = axis_attr.Get() if axis_attr.IsValid() else "X"
        rot_instance = "rot" + str(axis)  # rotX, rotY, or rotZ
        # Apply the typed API schema
        if not jprim.HasAPI(PhysxSchema.PhysxMimicJointAPI, rot_instance):
            PhysxSchema.PhysxMimicJointAPI.Apply(jprim, rot_instance)
        prefix = f"physxMimicJoint:{rot_instance}"
        _set_or_create(jprim, f"{prefix}:gearing",
                       Sdf.ValueTypeNames.Float, gearing)
        _set_or_create(jprim, f"{prefix}:naturalFrequency",
                       Sdf.ValueTypeNames.Float, 0.0)
        _set_or_create(jprim, f"{prefix}:dampingRatio",
                       Sdf.ValueTypeNames.Float, 0.0)
        rel = jprim.GetRelationship(f"{prefix}:referenceJoint")
        if not rel:
            rel = jprim.CreateRelationship(f"{prefix}:referenceJoint")
        rel.SetTargets([finger_joint_path])
        # Note: mimic joint drive stiffness/damping are left as-is from the
        # USD asset. The PhysxMimicJointAPI handles coordination.
        print(f"[Gripper]   {jname}: axis={axis} mimic={rot_instance} gearing={gearing}")
    print(f"[Gripper] PhysxMimicJointAPI added to {len(_MIMIC_JOINTS)} joints")

    # finger_joint DriveAPI is left as-is from the USD asset.
    # Runtime control goes through set_joint_position_targets in the sim loop.

    # Apply rubber-like friction material to finger pads.
    # Randomized per episode via update_gripper_friction().
    _DEFAULT_PAD_FRICTION = 1.2
    for pad_name in ["left_inner_finger_pad", "right_inner_finger_pad",
                     "left_inner_finger", "right_inner_finger",
                     "left_outer_finger", "right_outer_finger"]:
        pad_path = ROBOTIQ_BASE + "/" + pad_name
        pad_prim = stage.GetPrimAtPath(pad_path)
        if not pad_prim.IsValid():
            continue
        # Create or reuse a physics material on the pad
        mat_path = pad_path + "/GripMaterial"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            mat = UsdShade.Material.Define(stage, mat_path)
            UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mat_prim = stage.GetPrimAtPath(mat_path)
        phys_mat = UsdPhysics.MaterialAPI(mat_prim)
        phys_mat.CreateStaticFrictionAttr().Set(_DEFAULT_PAD_FRICTION)
        phys_mat.CreateDynamicFrictionAttr().Set(_DEFAULT_PAD_FRICTION)
        phys_mat.CreateRestitutionAttr().Set(0.0)
        # Bind material to the pad body
        UsdShade.MaterialBindingAPI.Apply(pad_prim)
        UsdShade.MaterialBindingAPI(pad_prim).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.weakerThanDescendants,
            "physics"
        )
    print(f"[Gripper] Finger pad friction (default μ={_DEFAULT_PAD_FRICTION}) applied")


def _grip_strength_for_mass(cube_mass):
    """Scale grip PD gains and close angle with cube mass.
    Returns (kp, kd, close_rad)."""
    mass_lo, mass_hi = CUBE_MASS_RANGE
    t = (cube_mass - mass_lo) / max(mass_hi - mass_lo, 1e-6)
    t = max(0.0, min(1.0, t))
    # PD gains
    kp_lo, kp_hi = 5000.0, 12000.0
    kp = kp_lo + t * (kp_hi - kp_lo)
    kd = kp * 0.08
    # Close angle, heavier cubes get more overshoot for stronger squeeze
    close_deg = GRIPPER_CLOSE_DEG_BASE + t * (GRIPPER_CLOSE_DEG_MAX - GRIPPER_CLOSE_DEG_BASE)
    close_rad = np.radians(close_deg)
    return kp, kd, close_rad


def update_gripper_for_episode(stage, rng, robot, cube_mass, cube_friction):
    """Set finger pad friction and grip gains scaled to cube mass.
    Returns the pad friction used."""
    pad_friction = 1.2

    for pad_name in ["left_inner_finger_pad", "right_inner_finger_pad",
                     "left_inner_finger", "right_inner_finger",
                     "left_outer_finger", "right_outer_finger"]:
        mat_path = ROBOTIQ_BASE + "/" + pad_name + "/GripMaterial"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if mat_prim.IsValid():
            phys_mat = UsdPhysics.MaterialAPI(mat_prim)
            phys_mat.GetStaticFrictionAttr().Set(pad_friction)
            phys_mat.GetDynamicFrictionAttr().Set(pad_friction)

    kp, kd, close_rad = _grip_strength_for_mass(cube_mass)
    _GRIP_INDICES = np.array([6, 7, 8, 9, 10, 11])
    robot._articulation_view.set_gains(
        kps=np.array([[kp] * 6]), kds=np.array([[kd] * 6]),
        joint_indices=_GRIP_INDICES
    )

    print(f"[Gripper] pad_μ={pad_friction:.2f}  cube_μ={cube_friction:.2f}  "
          f"cube_mass={cube_mass:.3f}kg  kp={kp:.0f}  kd={kd:.0f}  "
          f"close={np.degrees(close_rad):.1f}deg")
    return pad_friction, close_rad



# ---------------------------------------------------------------------------
# RMPFlow controller for UR5
# ---------------------------------------------------------------------------

class UR5RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name, robot_articulation, physics_dt=SIM_DT):
        rmp_config = mg.interface_config_loader.load_supported_motion_policy_config(
            "UR5", "RMPflow"
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
        print("[RMPFlow] UR5 controller initialized.")

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    rng = np.random.RandomState(_args.seed)

    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Ground plane (physics collider)
    world.scene.add_default_ground_plane()

    # Build realistic manufacturing environment (UR5 scale)
    build_workcell(stage)

    # Apply physics friction to collision surfaces that cubes interact with
    _ensure_surface_friction(stage, "/World/cell/conv_belt", friction=0.6)
    _ensure_surface_friction(stage, "/World/cell/bin_floor", friction=0.8)
    _ensure_surface_friction(stage, "/World/cell/bin_stand_top", friction=0.6)

    # Load UR5 arm
    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + UR5_USD, ROBOT_PRIM)
    robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="ur5"))

    # Load Robotiq 2F-85 gripper (local USD from URDF import)
    add_reference_to_stage(ROBOTIQ_USD, ROBOTIQ_PRIM)

    # Attach gripper to UR5 wrist via FixedJoint BEFORE world.reset()
    setup_gripper(stage)

    cur_dims  = sample_workpiece_dims(rng)
    cur_spawn, cur_yaw = randomize_cube_pose(rng)
    cur_spawn[2] = TABLE_HEIGHT + cur_dims[2] / 2.0 + 0.005
    cube = spawn_workpiece(world, rng, cur_spawn, cur_dims)

    world.reset()

    n_dof = robot.num_dof
    print(f"[UR5] num_dof = {n_dof}  (6 arm + {n_dof - 6} gripper)")

    home_full = np.zeros(n_dof)
    home_full[:len(HOME_JOINTS)] = HOME_JOINTS
    robot.set_joints_default_state(positions=home_full)

    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    robot.set_joint_positions(home_full)
    robot.set_joint_velocities(np.zeros(n_dof))

    for _ in range(10):
        world.step(render=True)

    print(f"[UR5] Joint positions after init: {np.round(robot.get_joint_positions(), 4)}")

    # --- ParallelGripper (Isaac Sim built-in) ---
    # Only drives finger_joint; PhysX mimic constraints handle the rest.
    _CLOSE_RAD = np.radians(GRIPPER_CLOSE_DEG_MAX)
    gripper = ParallelGripper(
        end_effector_prim_path=ROBOTIQ_BASE + "/robotiq_arg2f_base_link",
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),        # radians, fully open
        joint_closed_positions=np.array([_CLOSE_RAD]),  # radians, fully closed
        action_deltas=None,
        use_mimic_joints=True)
    gripper.initialize(
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names)
    fj_idx = gripper.joint_dof_indicies[0]
    print(f"[Gripper] ParallelGripper initialized  "
          f"open=0.0rad  close={_CLOSE_RAD:.3f}rad ({GRIPPER_CLOSE_DEG_MAX}deg max)  "
          f"finger_idx={fj_idx}")

    # Initial gripper gains (overridden per-episode by update_gripper_for_episode).
    _GRIP_INDICES = np.array([6, 7, 8, 9, 10, 11])
    _init_kp, _init_kd, _ = _grip_strength_for_mass(np.mean(CUBE_MASS_RANGE))
    robot._articulation_view.set_gains(
        kps=np.array([[_init_kp] * 6]), kds=np.array([[_init_kd] * 6]),
        joint_indices=_GRIP_INDICES
    )
    gripper.open()
    for _ in range(20):
        world.step(render=True)

    # (Diagnostic moved after update_gripper_for_episode)

    # --- RMPFlow + PickPlaceController ---
    rmp_controller = UR5RMPFlowController(
        name="ur5_rmpflow", robot_articulation=robot, physics_dt=SIM_DT
    )

    # PickPlaceController: 10 phases with smooth sinusoidal interpolation.
    #  0: above pick   1: descend   2: settle   3: close gripper
    #  4: lift         5: move XY   6: lower    7: open gripper
    #  8: retract      9: return
    # events_dt controls speed of each phase (smaller = slower/smoother).
    pick_place = PickPlaceController(
        name="ur5_pick_place",
        cspace_controller=rmp_controller,
        gripper=gripper,
        end_effector_initial_height=EEF_INITIAL_HEIGHT,
        events_dt=EVENTS_DT)

    # Validate EEF prim
    if not stage.GetPrimAtPath(EEF_PRIM).IsValid():
        carb.log_error(f"EEF prim {EEF_PRIM} not found. Aborting.")
        simulation_app.close()
        return

    _log_dir = _args.log_dir if _args.log_dir else LOG_DIR
    if _args.run_type:
        _log_dir = os.path.join(_log_dir, _args.run_type)
    logger = EpisodeLogger(_log_dir)

    # Compatibility shim: collect_sensors expects gripper.attached / .is_closed
    class _GripperProxy:
        def __init__(self, pg):
            self._pg = pg
        @property
        def attached(self):
            jp = self._pg.get_joint_positions()
            return jp is not None and float(jp[0]) > _CLOSE_RAD * 0.3
        @property
        def is_closed(self):
            return self.attached
    gripper_proxy = _GripperProxy(gripper)

    # Compute phase step boundaries from events_dt for event scheduling.
    # boundaries[i] = first step of phase i; boundaries[-1] = total steps.
    _events_dt = EVENTS_DT
    _phase_boundaries = [0]
    for dt in _events_dt:
        _phase_boundaries.append(_phase_boundaries[-1] + int(round(1.0 / dt)))

    # Event injection scheduler (optional)
    event_scheduler = None
    if _args.events:
        events_json = Path(__file__).parent.parent.parent / "events.json"
        event_scheduler = EventScheduler(
            events_json_path=str(events_json),
            task_name="pick_and_place",
            applicators=BUILTIN_APPLICATORS,
            rng_seed=_args.seed + 10000,
            num_events_range=(0, _args.max_events_per_episode),
            force_event_id=_args.event_i)
        event_scheduler.set_phase_boundaries(_phase_boundaries)

    cur_mass, cur_friction = randomize_cube_physics(cube, rng)
    _, _cur_close_rad = update_gripper_for_episode(stage, rng, robot, cur_mass, cur_friction)

    # Diagnostic: print all joint params after full setup including set_max_efforts
    _all_gains = robot._articulation_view.get_gains()
    _all_efforts = robot._articulation_view.get_max_efforts()
    print(f"[Diag] DOF names: {robot.dof_names}")
    print(f"[Diag] kp:  {np.round(_all_gains[0][0], 1)}")
    print(f"[Diag] kd:  {np.round(_all_gains[1][0], 1)}")
    print(f"[Diag] max_efforts: {np.round(_all_efforts[0], 1)}")

    setup_hud()
    update_hud(cur_mass, cur_friction)
    setup_event_indicator()

    # Ensure gripper is fully open before the first episode starts.
    # update_gripper_for_episode changes drive gains, so we must re-open
    # and give the physics enough steps to settle.
    gripper.open()
    for _ in range(30):
        world.step(render=True)

    logger.init_sensors(world, robot, cube, stage,
                        flange_prim_path=EEF_PRIM, sim_dt=SIM_DT)

    step           = 0
    episode        = 0
    sim_time       = 0.0
    ep_step        = 0
    plan_attempts  = 0
    plan_time_last = 0.0
    pick_attempts  = 0

    # Orientation: tool pointing down (default, updated per episode)
    ee_orient = euler_angles_to_quat(np.array([0, np.pi, 0]))

    PHASE_NAMES = ["above_pick", "descend", "settle", "close",
                   "lift", "move_xy", "lower", "open",
                   "retract", "return"]

    # Slip detection state, detects the exact frame the cube starts moving
    # relative to the EEF by comparing current offset to the grip-time offset.
    _slip_detected = False
    _slip_prev_cube_pos = None
    _slip_prev_eef_pos = None
    _slip_rel_vel_window = []
    _SLIP_WINDOW = 40
    _SLIP_VEL_THRESHOLD = 0.03

    def compute_targets():
        """Compute pick/place positions and gripper orientation for the
        current episode.  The gripper yaw is aligned with the cube's
        narrow axis so the fingers always grab the thin side."""
        nonlocal ee_orient
        perceived = apply_perception_noise(cur_spawn.copy(), rng)
        pick = perceived.copy()
        pick[2] = perceived[2] + GRIPPER_TCP_Z  # finger pads at cube centre height
        place = BIN_POSITION.copy()
        place[2] = BIN_POSITION[2] + GRIPPER_TCP_Z + 0.04
        # Rotate gripper to match cube yaw, the gripper closes along
        # its local Y axis, and the cube's narrow side (dims[1]) is
        # along the cube's local Y.  Matching the yaw aligns them.
        ee_orient = euler_angles_to_quat(np.array([0, np.pi, cur_yaw]))
        return pick, place

    def reset_episode(success: bool, reason: str = ""):
        nonlocal episode, cur_mass, cur_friction
        nonlocal cur_spawn, cur_dims, cube, cur_yaw
        nonlocal sim_time, plan_attempts, plan_time_last, pick_attempts
        nonlocal ep_step, pick_pos, place_pos, ee_orient
        nonlocal _slip_detected, _slip_rel_vel_window
        nonlocal _slip_prev_cube_pos, _slip_prev_eef_pos

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
        pick_attempts  = 1
        _slip_detected = False
        _slip_rel_vel_window = []
        _slip_prev_cube_pos = None
        _slip_prev_eef_pos = None

        if _args.episodes > 0 and episode >= _args.episodes:
            print(f"[FactoryBench] Reached {_args.episodes} episodes. Stopping.")
            logger.close()
            simulation_app.close()
            sys.exit(0)

        cur_dims  = sample_workpiece_dims(rng)
        cur_spawn, cur_yaw = randomize_cube_pose(rng)
        cur_spawn[2] = 0.10 + cur_dims[2] / 2.0 + 0.005

        cube = spawn_workpiece(world, rng, cur_spawn, cur_dims, existing_cube=cube)

        n_dof = robot.num_dof
        home_full = np.zeros(n_dof)
        home_full[:len(HOME_JOINTS)] = HOME_JOINTS
        robot.set_joints_default_state(positions=home_full)
        cube.set_default_state(position=cur_spawn,
                               orientation=yaw_to_quat(cur_yaw))

        setup_gripper(stage)
        world.reset()

        # Set mass/friction AFTER world.reset() so the physics engine
        # picks up the new values on the next world.step().
        cur_mass, cur_friction = randomize_cube_physics(cube, rng)
        print(f"[UR5] Episode {episode} -- dims={np.round(cur_dims*1000).astype(int)}mm  "
              f"spawn={np.round(cur_spawn,3)}  mass={cur_mass:.3f}kg")

        robot.set_solver_position_iteration_count(64)
        robot.set_solver_velocity_iteration_count(64)
        rmp_controller.reset()
        pick_place.reset(end_effector_initial_height=EEF_INITIAL_HEIGHT)
        robot.set_joint_positions(home_full)
        robot.set_joint_velocities(np.zeros(n_dof))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        gripper.post_reset()

        _, _cur_close_rad = update_gripper_for_episode(stage, rng, robot, cur_mass, cur_friction)
        update_hud(cur_mass, cur_friction)
        gripper.open()

        for _ in range(30):
            world.step(render=True)

        logger.reset_diff_state()
        ep_step = 0
        pick_pos, place_pos = compute_targets()
        print(f"  pick={np.round(pick_pos, 4)}  place={np.round(place_pos, 4)}")

        if event_scheduler is not None:
            _setup_ctx = SimContext(
                stage=stage,
                cube_prim_path=CUBE_PRIM,
                robot_prim_path=ROBOT_PRIM,
                sim_dt=SIM_DT,
                extra={"robot": robot})
            event_scheduler.reset(_setup_ctx)
            event_scheduler.schedule_episode(max_episode_steps=1500)
            event_scheduler.setup_episode(_setup_ctx)

    # --- Fast-forward RNG if --start_episode is set ---
    if _args.start_episode > 0:
        # Episode 0 already consumed its RNG draws during init above.
        # Fast-forward through episodes 1..(start_episode-1) by consuming
        # the same RNG calls each episode would make, without simulating.
        _skip_to = _args.start_episode
        print(f"[UR5] Fast-forwarding RNG from episode 0 to {_skip_to}...")
        # Episode 0 init consumed dims+pose+color+physics (12 draws) but
        # NOT perception noise (3 draws), consume them now.
        rng.normal(0, PERCEPTION_NOISE_XY_STD)
        rng.normal(0, PERCEPTION_NOISE_XY_STD)
        rng.normal(0, PERCEPTION_NOISE_Z_STD)
        for _sk in range(1, _skip_to):
            sample_workpiece_dims(rng)        # 3 draws
            randomize_cube_pose(rng)           # 3 draws
            rng.uniform(0.2, 0.9, 3)          # 3 draws (spawn_workpiece color)
            rng.uniform(*CUBE_MASS_RANGE)      # 1 draw
            rng.uniform(*CUBE_FRICTION_RANGE)  # 1 draw
            rng.uniform(*CUBE_RESTITUTION_RANGE)  # 1 draw
            rng.normal(0, PERCEPTION_NOISE_XY_STD)  # 1 draw
            rng.normal(0, PERCEPTION_NOISE_XY_STD)  # 1 draw
            rng.normal(0, PERCEPTION_NOISE_Z_STD)   # 1 draw
        # Now set up the actual start episode
        episode = _skip_to
        cur_dims = sample_workpiece_dims(rng)
        cur_spawn, cur_yaw = randomize_cube_pose(rng)
        cur_spawn[2] = 0.10 + cur_dims[2] / 2.0 + 0.005
        cube = spawn_workpiece(world, rng, cur_spawn, cur_dims, existing_cube=cube)
        setup_gripper(stage)
        world.reset()
        cur_mass, cur_friction = randomize_cube_physics(cube, rng)
        robot.set_solver_position_iteration_count(64)
        robot.set_solver_velocity_iteration_count(64)
        rmp_controller.reset()
        pick_place.reset(end_effector_initial_height=EEF_INITIAL_HEIGHT)
        n_dof = robot.num_dof
        home_full = np.zeros(n_dof)
        home_full[:len(HOME_JOINTS)] = HOME_JOINTS
        robot.set_joint_positions(home_full)
        robot.set_joint_velocities(np.zeros(n_dof))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        gripper.post_reset()
        _, _cur_close_rad = update_gripper_for_episode(stage, rng, robot, cur_mass, cur_friction)
        update_hud(cur_mass, cur_friction)
        gripper.open()
        for _ in range(30):
            world.step(render=True)
        logger.reset_diff_state()
        ep_step = 0
        print(f"[UR5] Fast-forwarded to episode {_skip_to}")

    # --- First episode ---
    print(f"[UR5] Episode {episode} -- dims={np.round(cur_dims*1000).astype(int)}mm  "
          f"spawn={np.round(cur_spawn,3)}  mass={cur_mass:.3f}kg")
    pick_pos, place_pos = compute_targets()
    print(f"  pick={np.round(pick_pos, 4)}  place={np.round(place_pos, 4)}")
    pick_attempts = 1

    if event_scheduler is not None:
        _setup_ctx = SimContext(
            stage=stage,
            cube_prim_path=CUBE_PRIM,
            robot_prim_path=ROBOT_PRIM,
            sim_dt=SIM_DT,
            extra={"robot": robot})
        event_scheduler.schedule_episode(max_episode_steps=1500)
        event_scheduler.setup_episode(_setup_ctx)

    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        step     += 1
        ep_step  += 1
        sim_time += SIM_DT

        cube_pos, _ = cube.get_world_pose()
        cube_pos    = np.asarray(cube_pos)
        if np.any(np.abs(cube_pos) > 10.0):
            cube_pos = cur_spawn.copy()

        # Drop detection
        if cube_pos[2] < -0.15:
            print(f"[{step:6d}] Cube dropped. Resetting.")
            reset_episode(success=False, reason="dropped")
            continue

        if sim_time > 60.0:
            print(f"[{step:6d}] Episode timeout.")
            reset_episode(success=False, reason="timeout")
            continue

        # --- PickPlaceController: one call handles everything ---
        current_joints = robot.get_joint_positions()
        action = pick_place.forward(
            picking_position=pick_pos,
            placing_position=place_pos,
            current_joint_positions=current_joints,
            end_effector_orientation=ee_orient)

        robot.apply_action(action)

        # Reinforce gripper target every step, PickPlaceController only
        # sends the close/open command during phases 3 and 7, but
        # apply_action during arm phases can clear the gripper target.
        phase = min(pick_place.get_current_event(), 9)
        fj_idx = gripper.joint_dof_indicies[0]
        if phase >= 3 and phase < 7:
            grip_target = _cur_close_rad
        elif phase <= 2:
            grip_target = np.radians(-3.0)
        else:
            grip_target = 0.0
        robot._articulation_view.set_joint_position_targets(
            np.array([[grip_target]]), joint_indices=np.array([fj_idx])
        )
        phase_name = PHASE_NAMES[phase]

        # Sensor logging, record everything
        # Determine current Cartesian target based on controller phase
        if phase < 4:
            _ee_target = pick_pos
        else:
            _ee_target = place_pos

        # Read actual gripper finger position
        _all_jpos = robot.get_joint_positions()
        _gripper_actual = float(_all_jpos[fj_idx]) if _all_jpos is not None and len(_all_jpos) > fj_idx else None

        sensor_row = collect_sensors(
            logger, episode, step, sim_time, phase_name,
            robot, cube, stage, gripper_proxy,
            cur_mass, cur_friction,
            cur_restitution=getattr(randomize_cube_physics, "_last_restitution", 0.0),
            cur_dims=cur_dims,
            plan_attempts=plan_attempts,
            plan_time_last=plan_time_last,
            pick_attempts=pick_attempts,
            planned_action=action,
            task_phase=phase_name,
            joint_noise_std=JOINT_NOISE_STD,
            ee_target_pos=_ee_target,
            ee_target_quat=ee_orient,
            gripper_cmd_rad=grip_target,
            gripper_pos_rad=_gripper_actual,
            controller_phase=phase)

        # Slip detection, compare cube and EEF velocities via finite
        # difference so both have identical lag characteristics.
        # A real slip produces sustained negative relative Z velocity.
        _slip_just_detected = False
        if phase >= 3 and phase < 7 and not _slip_detected:
            from pxr import UsdGeom
            flange_xf = UsdGeom.Xformable(stage.GetPrimAtPath(EEF_PRIM))
            eef_world = flange_xf.ComputeLocalToWorldTransform(0)
            eef_pos = np.array([eef_world.ExtractTranslation()[i] for i in range(3)])

            if _slip_prev_cube_pos is not None and _slip_prev_eef_pos is not None:
                cube_vel_z = (cube_pos[2] - _slip_prev_cube_pos[2]) / SIM_DT
                eef_vel_z = (eef_pos[2] - _slip_prev_eef_pos[2]) / SIM_DT
                rel_vel_z = cube_vel_z - eef_vel_z

                _slip_rel_vel_window.append(rel_vel_z)
                if len(_slip_rel_vel_window) > _SLIP_WINDOW:
                    _slip_rel_vel_window.pop(0)

                if len(_slip_rel_vel_window) == _SLIP_WINDOW:
                    avg_rel_vel = float(np.mean(_slip_rel_vel_window))
                    if avg_rel_vel < -_SLIP_VEL_THRESHOLD:
                        sensor_row["event_id"] = 1
                        print(f"[{step:6d}] SLIP detected: avg relative Z vel "
                              f"= {avg_rel_vel:.4f} m/s over {_SLIP_WINDOW} frames")
                        _slip_detected = True
                        _slip_just_detected = True

            _slip_prev_cube_pos = cube_pos.copy()
            _slip_prev_eef_pos = eef_pos.copy()
        else:
            _slip_rel_vel_window.clear()
            _slip_prev_cube_pos = None
            _slip_prev_eef_pos = None

        # Event injection
        if event_scheduler is not None:
            from episode_logger import JOINT_NAMES as _JNAMES
            evt_ctx = SimContext(
                stage=stage,
                sensor_data=sensor_row,
                cube_prim_path=CUBE_PRIM,
                robot_prim_path=ROBOT_PRIM,
                joint_names=_JNAMES,
                sim_dt=SIM_DT,
                episode_step=ep_step,
                state_machine=phase_name,
                extra={"robot": robot, "action": action})
            active_events = event_scheduler.step(ep_step, evt_ctx)
            if active_events:
                evt = active_events[0]
                sensor_row["event_id"] = evt.event_id
                # Log all public (non-internal) random variable values.
                pstr = ";".join(
                    f"{k}={v}" for k, v in evt.params.items()
                    if not k.startswith("_")
                )
                sensor_row["event_params"] = pstr
            update_event_indicator(active_events)

        # Show slip detection on the HUD, persists once detected
        if _slip_just_detected or _slip_detected:
            if _evt_label is not None:
                _evt_label.text = "  EVENT: Grip Slip (id=1)"
                _evt_label.set_style({"font_size": 18, "color": 0xFFFFFFFF})
            if _evt_params_label is not None:
                _evt_params_label.text = "  cube slipped from gripper"
            if _evt_rect is not None:
                _evt_rect.set_style({"background_color": 0xFFCC6600,
                                     "border_radius": 4})
        else:
            if event_scheduler is None:
                update_event_indicator([])

        logger.step(sensor_row)

        # Episode complete?
        if pick_place.is_done():
            cube_final, _ = cube.get_world_pose()
            cube_final = np.asarray(cube_final)
            bp = BIN_POSITION
            in_bin = (abs(cube_final[0] - bp[0]) < 0.14 and
                      abs(cube_final[1] - bp[1]) < 0.14 and
                      cube_final[2] > bp[2] - 0.12 and
                      cube_final[2] < bp[2] + 0.12)
            if in_bin:
                print(f"[{step:6d}] Cycle complete, cube in bin.")
                reset_episode(success=True, reason="placed")
            else:
                print(f"[{step:6d}] Cycle complete, cube NOT in bin "
                      f"(pos={np.round(cube_final, 3)}).")
                reset_episode(success=False, reason="missed_bin")

    logger.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
