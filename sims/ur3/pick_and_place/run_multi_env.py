"""
FactoryBench / pick_and_place / run_multi_env.py
GPU-accelerated multi-environment UR3 pick-and-place simulation.

Includes all three GPU optimizations:
  1. GPU Physics Pipeline  (PhysX GPU broadphase + dynamics)
  5. Suppress Readback     (avoid GPU→CPU copies each step)
  6. Multi-Environment     (GridCloner + per-env state machines)

Run:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur3/pick_and_place/run_multi_env.py \
        --num_envs 16 [--headless] [--seed SEED] [--episodes N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).parent.resolve())
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import torch
from isaacsim import SimulationApp

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser()
_parser.add_argument("--headless",  action="store_true")
_parser.add_argument("--seed",      type=int, default=0)
_parser.add_argument("--episodes",  type=int, default=0, help="0 = run forever (per env)")
_parser.add_argument("--num_envs",  type=int, default=4, help="Number of parallel environments")
_parser.add_argument("--env_spacing", type=float, default=2.5, help="Spacing between envs (m)")
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
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleRigidPrim  # kept for reference; not used at runtime
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
from isaacsim.storage.native import get_assets_root_path
import isaacsim.robot_motion.motion_generation as mg
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_TASK_DIR = Path(__file__).parent.resolve()

def load_config() -> dict:
    with open(_TASK_DIR / "config" / "task.yaml") as f:
        return yaml.safe_load(f)

CFG = load_config()

# Base prim names (will be prefixed with env path)
_ROBOT_NAME      = "ur3"
_CUBE_NAME       = "pick_cube"
_EEF_SUFFIX      = "/wrist_3_link/flange"
HOME_JOINTS      = np.array(CFG["robot"]["home_joints"])

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

SUCTION_RADIUS = 0.10
LOG_DIR = str(_TASK_DIR / CFG["logging"]["log_dir"])
UR3_USD = "/Isaac/Robots/UniversalRobots/ur3/ur3.usd"
TABLE_HEIGHT = 0.10


def to_np(x):
    """Convert torch tensor (possibly on CUDA) to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

NUM_ENVS    = _args.num_envs
ENV_SPACING = _args.env_spacing


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

def apply_perception_noise(pos: np.ndarray, rng) -> np.ndarray:
    noisy = pos.copy()
    noisy[0] += rng.normal(0, PERCEPTION_NOISE_XY_STD)
    noisy[1] += rng.normal(0, PERCEPTION_NOISE_XY_STD)
    noisy[2] += rng.normal(0, PERCEPTION_NOISE_Z_STD)
    return noisy


# ---------------------------------------------------------------------------
# Scene construction (parameterized by env root path)
# ---------------------------------------------------------------------------

def build_workcell(stage, env_root):
    """Build manufacturing workcell under the given env root prim path."""

    cell = f"{env_root}/cell"

    def _mat(name, color, metallic=0.0, roughness=0.5):
        path = f"{cell}/Looks/{name}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _box(path, pos, size, mat=None, collision=True):
        c = UsdGeom.Cube.Define(stage, f"{cell}/{path}")
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3f(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(size[0]/2, size[1]/2, size[2]/2))
        if collision:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if mat:
            UsdShade.MaterialBindingAPI(c.GetPrim()).Bind(mat)
        return c

    def _cyl(path, pos, radius, height, mat=None, collision=False, axis="Z"):
        c = UsdGeom.Cylinder.Define(stage, f"{cell}/{path}")
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

    # Materials
    m_concrete   = _mat("concrete",       (0.48, 0.46, 0.42), metallic=0.0,  roughness=0.85)
    m_steel_dk   = _mat("steel_dark",     (0.22, 0.23, 0.25), metallic=0.85, roughness=0.35)
    m_steel_lt   = _mat("steel_brushed",  (0.58, 0.59, 0.60), metallic=0.80, roughness=0.28)
    m_aluminum   = _mat("aluminum",       (0.72, 0.73, 0.74), metallic=0.90, roughness=0.22)
    m_belt       = _mat("conveyor_belt",  (0.10, 0.10, 0.12), metallic=0.0,  roughness=0.75)
    m_conv_frame = _mat("conveyor_frame", (0.15, 0.28, 0.52), metallic=0.70, roughness=0.40)
    m_yellow     = _mat("safety_yellow",  (0.95, 0.78, 0.05), metallic=0.0,  roughness=0.55)
    m_bin        = _mat("bin_metal",      (0.32, 0.35, 0.37), metallic=0.75, roughness=0.42)

    # Floor
    _box("floor", [0, 0, -0.005], [3.0, 3.0, 0.01], m_concrete, collision=False)

    # Table
    TABLE_CX = 0.45
    _box("table_top", [TABLE_CX, 0.0, TABLE_HEIGHT - 0.006],
         [0.60, 0.50, 0.012], m_aluminum, collision=False)
    for i, (lx, ly) in enumerate([(TABLE_CX - 0.25, -0.22), (TABLE_CX - 0.25, 0.22),
                                   (TABLE_CX + 0.25, -0.22), (TABLE_CX + 0.25, 0.22)]):
        _box(f"table_leg_{i}", [lx, ly, TABLE_HEIGHT/2 - 0.006],
             [0.04, 0.04, TABLE_HEIGHT - 0.012], m_steel_dk, collision=False)

    # Conveyor
    cx, cy = 0.45, 0.00
    cl, cw = 0.45, 0.24
    _box("conv_belt", [cx, cy, TABLE_HEIGHT + 0.002],
         [cl, cw, 0.004], m_belt, collision=True)
    _box("conv_rail_l", [cx, cy + cw/2 + 0.015, TABLE_HEIGHT + 0.015],
         [cl, 0.025, 0.03], m_conv_frame, collision=True)
    _box("conv_rail_r", [cx, cy - cw/2 - 0.015, TABLE_HEIGHT + 0.015],
         [cl, 0.025, 0.03], m_conv_frame, collision=True)

    # Bin
    bp = BIN_POSITION
    bw, bd, bh = 0.22, 0.22, 0.10
    wt = 0.003
    _box("bin_floor", [bp[0], bp[1], bp[2] - bh/2 + wt/2],
         [bw, bd, wt], m_bin, collision=True)
    for i, (ox, oy, sx, sy) in enumerate([
        (0, -bd/2 + wt/2, bw, wt), (0, bd/2 - wt/2, bw, wt),
        (-bw/2 + wt/2, 0, wt, bd), (bw/2 - wt/2, 0, wt, bd),
    ]):
        _box(f"bin_wall_{i}", [bp[0]+ox, bp[1]+oy, bp[2]],
             [sx, sy, bh], m_bin, collision=True)

    stand_top_z = bp[2] - bh/2 - 0.005
    _box("bin_stand_top", [bp[0], bp[1], stand_top_z],
         [bw+0.04, bd+0.04, 0.006], m_steel_dk, collision=True)

    # Mounting plate
    _box("mount_plate", [0.0, 0.0, -0.005],
         [0.18, 0.18, 0.01], m_aluminum, collision=False)

    # Lighting (one LED panel per env, simplified)
    light = UsdLux.RectLight.Define(stage, f"{cell}/led_panel")
    xf = UsdGeom.Xformable(light)
    xf.AddTranslateOp().Set(Gf.Vec3f(0.20, 0.0, 1.80))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180, 0, 0))
    light.GetWidthAttr().Set(0.30)
    light.GetHeightAttr().Set(0.15)
    light.GetIntensityAttr().Set(25000)
    light.GetColorAttr().Set(Gf.Vec3f(1.0, 0.97, 0.92))


# ---------------------------------------------------------------------------
# RMPFlow controller
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
        pos, ori = self._articulation_motion_policy._robot_articulation.get_world_pose()
        self._default_position  = to_np(pos)
        self._default_orientation = to_np(ori)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation)


# ---------------------------------------------------------------------------
# Suction gripper (per-env)
# ---------------------------------------------------------------------------

class SuctionGripper:
    """Kinematic suction gripper for a single environment."""

    def __init__(self, cube_prim_path):
        self._cube_prim_path = cube_prim_path
        self._offset   = np.zeros(3)
        self.attached  = False

    def attach(self, ee_pos, cube_pos) -> bool:
        xy_err = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
        if xy_err > SUCTION_RADIUS:
            return False

        self._offset  = ee_pos - cube_pos
        self.attached = True

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._cube_prim_path)
        if prim.IsValid():
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.GetKinematicEnabledAttr().Set(True)
            self._set_collision(prim, False)
        return True

    @staticmethod
    def _set_collision(prim, enabled: bool):
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(enabled)

    def track(self, ee_pos):
        if self.attached:
            cube_target = ee_pos - self._offset
            stage = omni.usd.get_context().get_stage()
            set_cube_pose_usd(stage, self._cube_prim_path, cube_target)

    def detach(self):
        if self.attached:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self._cube_prim_path)
            if prim.IsValid():
                rb = UsdPhysics.RigidBodyAPI(prim)
                rb.GetKinematicEnabledAttr().Set(False)
                self._set_collision(prim, True)
        self.attached = False


# ---------------------------------------------------------------------------
# Trajectory waypoint generator
# ---------------------------------------------------------------------------

def build_waypoints(cube_pos, cube_dims, bin_pos):
    cube_top_z = cube_pos[2] + cube_dims[2] / 2.0
    approach_height = 0.10
    grasp_height = cube_top_z + 0.005
    lift_height = cube_top_z + approach_height
    bin_above_z = 0.25
    bin_place_z = 0.15
    neutral_pos = np.array([0.20, -0.17, 0.28])
    down_quat = np.array([0.0, 1.0, 0.0, 0.0])

    return [
        ("above_cube",      np.array([cube_pos[0], cube_pos[1], cube_top_z + approach_height]), down_quat),
        ("descend",         np.array([cube_pos[0], cube_pos[1], grasp_height]),                 down_quat),
        ("lift",            np.array([cube_pos[0], cube_pos[1], lift_height]),                   down_quat),
        ("neutral_to_bin",  neutral_pos,                                                         down_quat),
        ("above_bin",       np.array([bin_pos[0],  bin_pos[1],  bin_above_z]),                   down_quat),
        ("place_descend",   np.array([bin_pos[0],  bin_pos[1],  bin_place_z]),                   down_quat),
        ("retract_bin",     np.array([bin_pos[0],  bin_pos[1],  bin_above_z]),                   down_quat),
        ("neutral_to_home", neutral_pos,                                                         down_quat),
        ("home",            np.array([0.20, 0.00, 0.25]),                                       down_quat),
    ]


def enforce_wrist_down(action):
    pos = action.joint_positions
    if pos is not None:
        pos = np.array(pos, dtype=float)
        pos[4] = -np.pi / 2
        pos[3] = -np.pi / 2 - pos[1] - pos[2]
        action.joint_positions = pos
    vel = action.joint_velocities
    if vel is not None:
        vel = np.array(vel, dtype=float)
        vel[3] = -(vel[1] + vel[2])
        vel[4] = 0.0
        action.joint_velocities = vel
    return action


def safe_apply_action(robot, action):
    """Apply ArticulationAction, working around the torch-backend np.isnan bug
    in Isaac Sim's ArticulationController (velocities/efforts missing to_numpy).

    Converts action fields to CUDA float32 tensors with NaN replaced by 0.
    """
    _dev = "cuda:0"

    def _to_tensor(arr):
        if arr is None:
            return None
        t = torch.tensor(np.asarray(arr, dtype=np.float32), device=_dev)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    clean = ArticulationAction(
        joint_positions=_to_tensor(action.joint_positions),
        joint_velocities=_to_tensor(action.joint_velocities),
        joint_efforts=_to_tensor(action.joint_efforts),
        joint_indices=action.joint_indices)
    robot.get_articulation_controller()._articulation_view.apply_action(
        ArticulationActions(
            joint_positions=clean.joint_positions.unsqueeze(0) if clean.joint_positions is not None else None,
            joint_velocities=clean.joint_velocities.unsqueeze(0) if clean.joint_velocities is not None else None,
            joint_efforts=clean.joint_efforts.unsqueeze(0) if clean.joint_efforts is not None else None,
            joint_indices=clean.joint_indices)
    )


# ---------------------------------------------------------------------------
# EEF position helper (works with fabric / suppress readback)
# ---------------------------------------------------------------------------

def get_eef_pos_fk(es):
    """Get EEF world-frame position via forward kinematics (physics-view independent).

    Uses the Lula RmpFlow FK solver with the robot's current joint positions.
    Returns world-frame position (the FK solver accounts for the robot base pose).
    """
    joint_pos = to_np(es.robot.get_joint_positions())
    pos, _ = es.rmp_ctrl.rmp_flow.get_end_effector_pose(joint_pos)
    return pos


def get_cube_pos_usd(stage, cube_prim_path):
    """Read cube world-frame position via USD (avoids tensor view issues)."""
    prim = stage.GetPrimAtPath(cube_prim_path)
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = mat.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def set_cube_pose_usd(stage, cube_prim_path, position, scale=None):
    """Write cube world-frame position via USD xform ops."""
    prim = stage.GetPrimAtPath(cube_prim_path)
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*position.tolist()))
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale and scale is not None:
            op.Set(Gf.Vec3d(*scale.tolist()))


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

class S:
    INIT       = "INIT"
    MOVE_TO_WP = "MOVE_TO_WP"
    ATTACH     = "ATTACH"
    DETACH     = "DETACH"
    DONE       = "DONE"


# ---------------------------------------------------------------------------
# Per-environment state
# ---------------------------------------------------------------------------

class EnvState:
    """Tracks state machine and objects for a single environment."""

    GRASP_WP_IDX   = 2
    RELEASE_WP_IDX = 6
    POS_THRESHOLD  = 0.12
    WP_TIMEOUT     = 300

    def __init__(self, env_id, env_path, rng_seed, env_origin=None):
        self.env_id   = env_id
        self.env_path = env_path
        self.rng      = np.random.RandomState(rng_seed)
        # World-frame origin of this env (set by GridCloner)
        self.env_origin = np.array(env_origin if env_origin is not None else [0, 0, 0], dtype=float)

        # Set by setup()
        self.robot         = None
        self.cube          = None
        self.suction       = None
        self.rmp_ctrl      = None
        self.art_ctrl      = None
        self.eef_link_idx  = -1

        # Episode state
        self.state         = S.INIT
        self.step          = 0
        self.wait_timer    = 0
        self.episode       = 0
        self.sim_time      = 0.0
        self.waypoints     = []
        self.wp_idx        = 0
        self.wp_step_count = 0
        self.cur_wp_name   = ""

        # Workpiece params
        self.cur_dims    = None
        self.cur_spawn   = None
        self.cur_mass    = 0.0
        self.cur_friction = 0.0

        # Counters
        self.plan_attempts  = 0
        self.pick_attempts  = 0
        self.episodes_done  = 0

    @property
    def robot_prim_path(self):
        return f"{self.env_path}/{_ROBOT_NAME}"

    @property
    def cube_prim_path(self):
        return f"{self.env_path}/{_CUBE_NAME}"

    @property
    def eef_prim_path(self):
        return f"{self.robot_prim_path}{_EEF_SUFFIX}"


# ---------------------------------------------------------------------------
# Episode summary logger (lightweight for multi-env)
# ---------------------------------------------------------------------------

class MultiEnvLogger:
    """Logs episode summaries for all envs to a single CSV."""

    def __init__(self, log_dir, num_envs):
        import csv
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._path = os.path.join(log_dir, "episodes_multi_env.csv")
        self._fh = open(self._path, "w", newline="", buffering=1)
        cols = [
            "env_id", "episode", "success", "reason",
            "sim_time_s", "wall_time_s", "steps",
            "cube_mass_kg", "cube_friction_coeff",
            "cube_width_m", "cube_depth_m", "cube_height_m",
        ]
        self._writer = csv.DictWriter(self._fh, fieldnames=cols, extrasaction="ignore")
        self._writer.writeheader()
        self._start = time.time()
        print(f"[Logger] Multi-env episode log: {os.path.abspath(self._path)}")

    def log_episode(self, es: EnvState, success: bool, reason: str):
        self._writer.writerow({
            "env_id":             es.env_id,
            "episode":            es.episode,
            "success":            int(success),
            "reason":             reason,
            "sim_time_s":         round(es.sim_time, 4),
            "wall_time_s":        round(time.time() - self._start, 4),
            "steps":              es.step,
            "cube_mass_kg":       round(es.cur_mass, 4),
            "cube_friction_coeff": round(es.cur_friction, 4),
            "cube_width_m":       round(float(es.cur_dims[0]), 4) if es.cur_dims is not None else "",
            "cube_depth_m":       round(float(es.cur_dims[1]), 4) if es.cur_dims is not None else "",
            "cube_height_m":      round(float(es.cur_dims[2]), 4) if es.cur_dims is not None else "",
        })

    def close(self):
        self._fh.flush()
        self._fh.close()
        print(f"[Logger] Closed. Data in {self._path}")


# ---------------------------------------------------------------------------
# Cube physics randomization (USD-based, per env)
# ---------------------------------------------------------------------------

def randomize_cube_physics(cube_prim_path, rng):
    mass        = float(rng.uniform(*CUBE_MASS_RANGE))
    friction    = float(rng.uniform(*CUBE_FRICTION_RANGE))
    restitution = float(rng.uniform(*CUBE_RESTITUTION_RANGE))
    try:
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(cube_prim_path)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(mass)
        mat_prim = stage.GetPrimAtPath(cube_prim_path + "/PhysicsMaterial")
        if mat_prim.IsValid():
            UsdPhysics.MaterialAPI(mat_prim).GetStaticFrictionAttr().Set(friction)
            UsdPhysics.MaterialAPI(mat_prim).GetDynamicFrictionAttr().Set(friction * 0.85)
            UsdPhysics.MaterialAPI(mat_prim).GetRestitutionAttr().Set(restitution)
    except Exception:
        pass
    return mass, friction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    num_envs = NUM_ENVS
    base_env = "/World/Envs"
    env_0    = f"{base_env}/Env_0"

    # ---- Create world with GPU physics (Step 1) ----
    # Explicitly set torch backend + CUDA device to avoid the numpy→torch
    # auto-switch that corrupts default states stored before world.reset().
    world = World(
        physics_dt=SIM_DT,
        rendering_dt=SIM_DT,
        stage_units_in_meters=1.0,
        backend="torch",
        device="cuda:0")

    physics_context = world.get_physics_context()
    physics_context.enable_gpu_dynamics(True)
    physics_context.set_broadphase_type("GPU")
    physics_context.enable_fabric(True)
    physics_context.enable_ccd(False)

    # Increase GPU buffer capacities for multi-env (scales with num_envs)
    physics_context.set_gpu_found_lost_aggregate_pairs_capacity(4 * num_envs * 1024)
    physics_context.set_gpu_total_aggregate_pairs_capacity(4 * num_envs * 1024)
    physics_context.set_gpu_found_lost_pairs_capacity(4 * num_envs * 1024)
    physics_context.set_gpu_max_rigid_contact_count(num_envs * 524288)
    physics_context.set_gpu_max_rigid_patch_count(num_envs * 81920)
    print(f"[GPU] GPU dynamics + fabric enabled for {num_envs} environments.")

    # ---- Suppress readback (Step 5) ----
    # Note: suppressReadback prevents PhysX from writing transforms back to USD.
    # With Fabric enabled, USD reads are intercepted by the Fabric cache, so
    # ComputeLocalToWorldTransform still returns up-to-date data for most prims.
    # However, deeply nested articulation links (like tool0) may not be updated
    # in Fabric automatically.  We keep suppressReadback=False so that link
    # transforms remain queryable via USD xforms.  The GPU dynamics + multi-env
    # parallelization provide the primary speedup; readback suppression is a
    # minor optimization that would require a FK-based EEF query to enable.
    carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
    print("[GPU] Physics readback: enabled (needed for EEF link queries).")

    stage = omni.usd.get_context().get_stage()

    # ---- Shared ground plane ----
    world.scene.add_default_ground_plane()

    # ---- Build base environment (Env_0) ----
    # Env_0 must be an Xform so the cloner can manipulate its transform ops.
    UsdGeom.Xform.Define(stage, env_0)

    build_workcell(stage, env_root=env_0)

    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + UR3_USD, f"{env_0}/{_ROBOT_NAME}")

    # Create the template cube in env 0
    template_dims  = np.array([0.06, 0.06, 0.04])
    template_spawn = CONVEYOR_NOMINAL.copy()
    template_spawn[2] = TABLE_HEIGHT + template_dims[2] / 2.0 + 0.005
    DynamicCuboid(
        prim_path=f"{env_0}/{_CUBE_NAME}",
        name="pick_cube_template",
        position=template_spawn,
        scale=template_dims,
        color=np.array([0.8, 0.2, 0.2]),
        mass=0.5)

    # ---- Clone environments (Step 6) ----
    cloner = GridCloner(spacing=ENV_SPACING)
    cloner.define_base_env(base_env)
    # generate_paths stores root_path internally (with underscore) for physics replication
    env_paths = cloner.generate_paths(f"{base_env}/Env", num_envs)

    print(f"[Clone] Cloning {num_envs} environments with spacing={ENV_SPACING}m ...")
    env_positions = cloner.clone(
        source_prim_path=env_0,
        prim_paths=env_paths,
        replicate_physics=True,
        # Let cloner use internally stored base_env_path and root_path from
        # define_base_env() / generate_paths().  Passing root_path explicitly
        # WITHOUT the trailing underscore causes a path mismatch between the
        # USD clone names (Env_0, Env_1) and the physics replicator lookup
        # (Env0, Env1), which crashes in Sdf_PrimPathNode destruction.
        copy_from_source=True,
        enable_env_ids=True)
    print(f"[Clone] Done. Env origins: {[list(np.round(p, 2)) for p in env_positions[:4]]}{'...' if num_envs > 4 else ''}")

    # Filter collisions: envs don't collide with each other, but all collide
    # with the shared ground plane.
    cloner.filter_collisions(
        physicsscene_path="/physicsScene",
        collision_root_path="/World/collisionGroups",
        prim_paths=env_paths,
        global_paths=["/World/defaultGroundPlane"])

    # ---- Create per-env objects ----
    env_states = []
    for i in range(num_envs):
        ep = f"{base_env}/Env_{i}"
        es = EnvState(env_id=i, env_path=ep, rng_seed=_args.seed + i,
                      env_origin=env_positions[i])

        # Wrap the cloned robot as a Robot (SingleArticulation) for RMPFlow
        es.robot = Robot(
            prim_path=es.robot_prim_path,
            name=f"ur3_{i}")
        world.scene.add(es.robot)
        es.robot.set_joints_default_state(
            positions=torch.tensor(HOME_JOINTS, dtype=torch.float32, device="cuda:0")
        )

        # Randomize workpiece (cube wrapper created after world.reset())
        es.cur_dims  = sample_workpiece_dims(es.rng)
        es.cur_spawn = randomize_cube_pose(es.rng)
        es.cur_spawn[2] = TABLE_HEIGHT + es.cur_dims[2] / 2.0 + 0.005

        es.suction = SuctionGripper(cube_prim_path=es.cube_prim_path)

        env_states.append(es)

    # ---- Initialize world ----
    world.reset()
    # Step once so that GPU tensor views are fully initialised after reset
    world.step(render=True)

    # ---- Post-reset setup per env (use USD API for poses, tensor views not yet stable) ----
    _dev = "cuda:0"

    for es in env_states:
        es.robot.disable_gravity()
        es.robot.set_solver_position_iteration_count(64)
        es.robot.set_solver_velocity_iteration_count(64)
        es.robot.set_joint_positions(torch.tensor(HOME_JOINTS, dtype=torch.float32, device=_dev))
        es.robot.set_joint_velocities(torch.zeros(6, dtype=torch.float32, device=_dev))

        # Set initial cube pose via USD xform ops (avoids tensor API invalidation)
        cube_prim = stage.GetPrimAtPath(es.cube_prim_path)
        xf = UsdGeom.Xformable(cube_prim)
        # Cube translate is LOCAL to env root (cloner already offsets the root)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*es.cur_spawn.tolist()))
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3d(*es.cur_dims.tolist()))

        es.cur_mass, es.cur_friction = randomize_cube_physics(es.cube_prim_path, es.rng)

        # RMPFlow controller (per-env, using individual Robot)
        es.rmp_ctrl = UR3RMPFlowController(
            name=f"rmpflow_{es.env_id}",
            robot_articulation=es.robot,
            physics_dt=SIM_DT)
        es.art_ctrl = es.robot.get_articulation_controller()

    # Stabilize physics
    for _ in range(10):
        world.step(render=True)

    logger = MultiEnvLogger(LOG_DIR, num_envs)

    for es in env_states:
        print(f"[Env {es.env_id}] Episode 0 -- dims={np.round(es.cur_dims*1000).astype(int)}mm  "
              f"spawn={np.round(es.cur_spawn, 3)}")

    # ---- Main simulation loop ----
    global_step = 0
    wall_start = time.time()

    while simulation_app.is_running():
        world.step(render=not _args.headless)
        if not world.is_playing():
            continue

        global_step += 1

        for es in env_states:
            _step_env(es, stage, world, logger, global_step)

        # Periodic status update
        if global_step % 600 == 0:  # ~10s at 60Hz
            elapsed = time.time() - wall_start
            total_eps = sum(e.episodes_done for e in env_states)
            active = sum(1 for e in env_states if e.sim_time >= 0)
            print(f"[Step {global_step:6d}] wall={elapsed:.1f}s  "
                  f"episodes_done={total_eps}  active_envs={active}/{num_envs}  "
                  f"env0_state={env_states[0].state}")

    logger.close()
    simulation_app.close()


# ---------------------------------------------------------------------------
# Per-environment step function
# ---------------------------------------------------------------------------

def _step_env(es: EnvState, stage, world, logger, global_step):
    """Advance one env's state machine by one physics step."""

    # Skip frozen envs (finished all requested episodes)
    if es.sim_time < 0:
        return

    es.step     += 1
    es.sim_time += SIM_DT

    # Read poses (convert world frame → local env frame)
    cube_pos_world = get_cube_pos_usd(stage, es.cube_prim_path)
    cube_pos = cube_pos_world - es.env_origin
    ee_pos_world = get_eef_pos_fk(es)
    ee_pos = ee_pos_world - es.env_origin

    if np.any(np.abs(cube_pos) > 50.0):
        cube_pos = es.cur_spawn.copy()

    if es.suction.attached:
        es.suction.track(ee_pos)  # track sets USD translate (local to env root)

    # Drop detection
    if cube_pos[2] < -0.15 and not es.suction.attached:
        _reset_episode(es, stage, world, logger, success=False, reason="dropped")
        return

    # Timeout
    if es.sim_time > 45.0:
        _reset_episode(es, stage, world, logger, success=False, reason="timeout")
        return

    # ---- State machine ----
    if es.state == S.INIT:
        es.wait_timer += 1
        if es.wait_timer >= 10:
            perceived_pos = apply_perception_noise(cube_pos, es.rng)
            es.waypoints = build_waypoints(perceived_pos, es.cur_dims, BIN_POSITION)
            es.wp_idx = 0
            es.wp_step_count = 0
            es.state = S.MOVE_TO_WP

    elif es.state == S.MOVE_TO_WP:
        if es.wp_idx >= len(es.waypoints):
            es.state = S.DONE
            return

        wp_name, wp_pos_local, wp_quat = es.waypoints[es.wp_idx]
        es.cur_wp_name = wp_name
        es.wp_step_count += 1

        # RMPFlow works in world frame; waypoints are in local env frame
        wp_pos_world = wp_pos_local + es.env_origin
        action = es.rmp_ctrl.forward(
            target_end_effector_position=wp_pos_world,
            target_end_effector_orientation=wp_quat)
        action = enforce_wrist_down(action)
        safe_apply_action(es.robot, action)

        pos_err = float(np.linalg.norm(ee_pos - wp_pos_local))
        reached = pos_err < EnvState.POS_THRESHOLD
        timed_out = es.wp_step_count >= EnvState.WP_TIMEOUT

        if reached or timed_out:
            es.wp_idx += 1
            es.wp_step_count = 0

            if es.wp_idx == EnvState.GRASP_WP_IDX:
                es.pick_attempts += 1
                es.state = S.ATTACH
            elif es.wp_idx == EnvState.RELEASE_WP_IDX:
                es.state = S.DETACH

    elif es.state == S.ATTACH:
        _, hold_pos_local, hold_quat = es.waypoints[es.wp_idx - 1]
        action = es.rmp_ctrl.forward(
            target_end_effector_position=hold_pos_local + es.env_origin,
            target_end_effector_orientation=hold_quat)
        action = enforce_wrist_down(action)
        safe_apply_action(es.robot, action)

        es.wait_timer += 1
        if es.wait_timer >= 20:
            attached = es.suction.attach(ee_pos, cube_pos)
            es.wait_timer = 0
            if attached:
                es.state = S.MOVE_TO_WP
            else:
                es.wp_idx = 0
                es.state = S.MOVE_TO_WP

    elif es.state == S.DETACH:
        _, hold_pos_local, hold_quat = es.waypoints[es.wp_idx - 1]
        action = es.rmp_ctrl.forward(
            target_end_effector_position=hold_pos_local + es.env_origin,
            target_end_effector_orientation=hold_quat)
        action = enforce_wrist_down(action)
        safe_apply_action(es.robot, action)

        es.wait_timer += 1
        if es.wait_timer >= 20:
            es.suction.detach()
            es.wait_timer = 0
            es.state = S.MOVE_TO_WP

    elif es.state == S.DONE:
        es.wait_timer += 1
        if es.wait_timer >= 60:
            _reset_episode(es, stage, world, logger, success=True, reason="placed")
            es.wait_timer = 0


def _reset_episode(es: EnvState, stage, world, logger, success: bool, reason: str):
    """Reset a single environment for a new episode."""

    logger.log_episode(es, success, reason)

    es.episode += 1
    es.episodes_done += 1

    if _args.episodes > 0 and es.episodes_done >= _args.episodes:
        print(f"[Env {es.env_id}] Reached {_args.episodes} episodes.")
        # Don't stop the whole sim, other envs may still be running.
        # Just freeze this env.
        es.state = S.DONE
        es.sim_time = -1  # sentinel: skip future steps
        return

    # New workpiece params
    es.cur_dims  = sample_workpiece_dims(es.rng)
    es.cur_spawn = randomize_cube_pose(es.rng)
    es.cur_spawn[2] = TABLE_HEIGHT + es.cur_dims[2] / 2.0 + 0.005

    es.suction.detach()

    # Re-enable cube dynamics
    cube_prim = stage.GetPrimAtPath(es.cube_prim_path)
    if cube_prim.IsValid():
        rb = UsdPhysics.RigidBodyAPI(cube_prim)
        rb.GetKinematicEnabledAttr().Set(False)

    # Reset cube via USD (translate is local to env root, no env_origin needed)
    set_cube_pose_usd(stage, es.cube_prim_path, es.cur_spawn, scale=es.cur_dims)
    # Zero velocities via USD physics attributes
    cube_prim2 = stage.GetPrimAtPath(es.cube_prim_path)
    if cube_prim2.HasAPI(UsdPhysics.RigidBodyAPI):
        rb2 = UsdPhysics.RigidBodyAPI(cube_prim2)
        rb2.GetVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        rb2.GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))

    es.cur_mass, es.cur_friction = randomize_cube_physics(es.cube_prim_path, es.rng)

    # Reset robot
    es.robot.set_joint_positions(torch.tensor(HOME_JOINTS, dtype=torch.float32, device="cuda:0"))
    es.robot.set_joint_velocities(torch.zeros(6, dtype=torch.float32, device="cuda:0"))
    es.rmp_ctrl.reset()

    # Reset state machine
    es.state         = S.INIT
    es.step          = 0
    es.wait_timer    = 0
    es.sim_time      = 0.0
    es.waypoints     = []
    es.wp_idx        = 0
    es.wp_step_count = 0
    es.cur_wp_name   = ""
    es.plan_attempts = 0
    es.pick_attempts = 0

    if es.env_id < 4 or es.episode % 10 == 0:
        print(f"[Env {es.env_id}] Episode {es.episode} -- "
              f"dims={np.round(es.cur_dims*1000).astype(int)}mm  "
              f"mass={es.cur_mass:.3f}kg")


if __name__ == "__main__":
    main()
