"""
FactoryBench / pick_and_place / base_runner.py
Shared pick-and-place simulation for any UR robot + Robotiq 2F-85 gripper.

All robot-specific values are loaded from a task_shared.yaml config file.
This module is never run directly, use the per-robot thin wrappers instead.

Logic, phase ordering, success checks, event injection, and slip detection
are identical to the original FactoryBench/ur5/pick_and_place/run.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_FACTORYBENCH_DIR = str(Path(__file__).parent.parent.resolve())


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class RC:
    """Runtime config populated from the YAML dict."""
    pass


def init_from_config(cfg: dict, task_dir: str) -> RC:
    """Parse a config dict into an RC object with all derived constants."""
    rc = RC()
    rc.task_dir = Path(task_dir)

    # Robot
    r = cfg["robot"]
    rc.robot_name     = r["name"]
    rc.robot_prim     = r["prim"]
    rc.eef_prim       = r["eef_prim"]
    rc.home_joints    = np.array(r["home_joints"])
    _robot_usd = r["robot_usd"]
    # Resolve local URDF paths relative to FactoryBench dir
    if _robot_usd.endswith(".urdf") and not os.path.isabs(_robot_usd):
        rc.robot_usd = str(Path(_FACTORYBENCH_DIR) / _robot_usd)
    else:
        rc.robot_usd = _robot_usd
    rc.rmpflow_name   = r["rmpflow_name"]
    rc.wrist_link     = r.get("wrist_link", "wrist_3_link")

    # Gripper
    g = r["gripper"]
    rc.gripper_builtin   = g.get("builtin", False)
    rc.tcp_z             = float(g["tcp_z"])
    rc.grip_joint_indices = np.array(g["grip_joint_indices"])
    rc.grip_kp           = float(g["grip_kp"])
    rc.grip_kd           = float(g["grip_kd"])
    rc.max_effort        = float(g["max_effort"])

    # ParallelGripper config
    rc.gripper_joint_names     = g.get("joint_prim_names", ["finger_joint"])
    rc.gripper_open_positions  = np.array(g.get("joint_opened_positions", [0.0]))
    rc.gripper_close_positions = np.array(g.get("joint_closed_positions", [40.0]))
    rc.gripper_use_mimic       = g.get("use_mimic_joints", True)
    # EEF prim for ParallelGripper (where the gripper attaches)
    rc.gripper_eef_prim        = g.get("eef_prim", None)

    if rc.gripper_builtin:
        # Built-in gripper (e.g. Franka Panda), no external USD to load
        rc.gripper_usd       = None
        rc.gripper_prim      = None
        rc.gripper_base_link = None
        rc.gripper_base      = None
        rc.close_deg_base    = None
        rc.close_deg_max     = None
        rc.pad_friction      = 0.0
        rc.mimic_joints      = []
        rc.finger_pad_links  = []
        rc.grip_kp_range     = (5000.0, 12000.0)
        # Grip targets are the close positions directly (metres for prismatic)
        rc.grip_close_targets = rc.gripper_close_positions
        rc.grip_open_targets  = rc.gripper_open_positions
    else:
        # External gripper (Robotiq), load from USD and attach via FixedJoint
        rc.gripper_usd       = str(Path(_FACTORYBENCH_DIR) / g["usd"])
        rc.gripper_prim      = g["prim"]
        rc.gripper_base_link = g["base_link"]
        rc.gripper_base      = rc.gripper_prim
        rc.close_deg_base    = float(g.get("close_deg_base", g.get("close_deg", 56.0)))
        rc.close_deg_max     = float(g.get("close_deg_max", g.get("close_deg", 68.0)))
        rc.pad_friction      = float(g.get("pad_friction", 1.2))
        rc.mimic_joints      = g.get("mimic_joints", [])
        rc.finger_pad_links  = g.get("finger_pad_links", [])
        rc.grip_kp_range     = tuple(g.get("grip_kp_range", [5000.0, 12000.0]))
        # Grip targets in radians, use close_deg_max as the ParallelGripper
        # close limit (runtime close angle is scaled per-episode via
        # _grip_strength_for_mass).
        rc.grip_close_targets = np.array([np.radians(rc.close_deg_max)])
        rc.grip_open_targets  = np.array([0.0])

    # Scene
    s = cfg["scene"]
    rc.cube_prim         = s["cube_prim"]
    rc.conveyor_nominal  = np.array(s["conveyor_nominal"])
    rc.conveyor_xy_range = float(s["conveyor_xy_range"])
    rc.bin_position      = np.array(s["bin_position"])
    rc.table_height      = float(s["table_height"])
    rc.workcell          = s["workcell"]

    # Controller
    c = s["controller"]
    rc.eef_initial_height = float(c["end_effector_initial_height"])
    rc.events_dt          = c["events_dt"]

    # Physics
    rc.sim_dt = float(cfg["physics"]["sim_dt"])

    # Domain randomisation
    dr = cfg["domain_randomization"]
    rc.cube_mass_range        = tuple(dr["cube_mass_range"])
    rc.cube_friction_range    = tuple(dr["cube_friction_range"])
    rc.cube_restitution_range = tuple(dr["cube_restitution_range"])
    rc.cube_dim_w_range       = tuple(dr["cube_dim_w_range"])
    rc.cube_dim_d_range       = tuple(dr["cube_dim_d_range"])
    rc.cube_dim_h_range       = tuple(dr["cube_dim_h_range"])

    # Noise
    n = cfg["noise"]
    rc.perception_xy_std = float(n["perception_xy_std"])
    rc.perception_z_std  = float(n["perception_z_std"])
    rc.joint_noise_std   = float(n["joint_std"])

    # Control
    rc.grasp_xy_threshold = float(cfg["control"]["grasp_xy_threshold"])

    # Logging
    rc.log_dir = str(rc.task_dir / cfg["logging"]["log_dir"])

    # Slip detection
    rc.slip_drift_threshold = float(cfg["slip_detection"].get("drift_threshold", 0.005))

    # Episode
    ep = cfg["episode"]
    rc.timeout_s        = float(ep["timeout_s"])
    rc.drop_z_threshold = float(ep["drop_z_threshold"])
    rc.bin_xy_tolerance = float(ep["bin_xy_tolerance"])
    rc.bin_z_tolerance  = float(ep["bin_z_tolerance"])
    rc.settle_steps     = int(ep["settle_steps"])
    rc.init_steps       = int(ep["init_steps"])

    # Derived
    rc.conveyor_top_z = rc.conveyor_nominal[2] + 0.02

    return rc


# ---------------------------------------------------------------------------
# Workpiece helpers
# ---------------------------------------------------------------------------

def sample_workpiece_dims(rng, rc: RC) -> np.ndarray:
    a = float(rng.uniform(*rc.cube_dim_w_range))
    b = float(rng.uniform(*rc.cube_dim_d_range))
    h = float(rng.uniform(*rc.cube_dim_h_range))
    narrow, wide = min(a, b), max(a, b)
    return np.array([wide, narrow, h])


def randomize_cube_pose(rng, rc: RC):
    offset = rng.uniform(-rc.conveyor_xy_range, rc.conveyor_xy_range, size=2)
    pos = rc.conveyor_nominal.copy()
    pos[0] += offset[0]
    pos[1] += offset[1]
    yaw = float(rng.uniform(0, 2 * np.pi))
    return pos, yaw


def _apply_physics_material(stage, prim_path, static_friction, dynamic_friction,
                            restitution=0.0):
    """Create or update a physics material on a prim, ensuring attributes
    exist and PhysX friction combine mode is set to 'max'."""
    from pxr import UsdShade, PhysxSchema
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None

    mat_prim = None
    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound = binding_api.GetDirectBinding("physics")
    if bound and bound.GetMaterial():
        mp = bound.GetMaterial().GetPrim()
        if mp.IsValid():
            mat_prim = mp

    if mat_prim is None:
        mat_path = prim_path + "/PhysMat"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            from pxr import UsdPhysics as _UP
            mat = UsdShade.Material.Define(stage, mat_path)
            _UP.MaterialAPI.Apply(mat.GetPrim())
            mat_prim = mat.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.weakerThanDescendants,
            "physics")

    from pxr import UsdPhysics as _UP
    if not mat_prim.HasAPI(_UP.MaterialAPI):
        _UP.MaterialAPI.Apply(mat_prim)
    phys_mat = _UP.MaterialAPI(mat_prim)
    phys_mat.CreateStaticFrictionAttr().Set(static_friction)
    phys_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    phys_mat.CreateRestitutionAttr().Set(restitution)

    if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    physx_mat = PhysxSchema.PhysxMaterialAPI(mat_prim)
    physx_mat.CreateFrictionCombineModeAttr().Set("average")
    physx_mat.CreateRestitutionCombineModeAttr().Set("average")

    return mat_prim


def _ensure_surface_friction(stage, surface_prim_path, friction=0.8):
    """Apply a physics material to a collision surface if one doesn't exist."""
    from pxr import UsdShade
    prim = stage.GetPrimAtPath(surface_prim_path)
    if not prim.IsValid():
        return
    binding_api = UsdShade.MaterialBindingAPI(prim)
    bound = binding_api.GetDirectBinding("physics")
    if bound and bound.GetMaterial():
        return
    _apply_physics_material(stage, surface_prim_path, friction, friction * 0.85)
    print(f"[Scene] Applied physics material to {surface_prim_path} (μ={friction})")


def randomize_cube_physics(cube, rng, rc: RC, stage, UsdPhysics):
    mass        = float(rng.uniform(*rc.cube_mass_range))
    friction    = float(rng.uniform(*rc.cube_friction_range))
    restitution = float(rng.uniform(*rc.cube_restitution_range))
    randomize_cube_physics._last_restitution = restitution
    try:
        prim = stage.GetPrimAtPath(rc.cube_prim)
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        UsdPhysics.MassAPI(prim).CreateMassAttr().Set(mass)
        _apply_physics_material(stage, rc.cube_prim, friction, friction * 0.85,
                                restitution)
    except Exception as e:
        print(f"[Randomize] WARNING: failed to set physics: {e}")
    print(f"[Randomize] mass={mass:.3f}kg  friction={friction:.3f}  "
          f"restitution={restitution:.3f}")
    return mass, friction


def apply_perception_noise(pos: np.ndarray, rng, rc: RC) -> np.ndarray:
    noisy = pos.copy()
    noisy[0] += rng.normal(0, rc.perception_xy_std)
    noisy[1] += rng.normal(0, rc.perception_xy_std)
    noisy[2] += rng.normal(0, rc.perception_z_std)
    return noisy


def yaw_to_quat(yaw_rad):
    c, s = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
    return np.array([c, 0.0, 0.0, s])


def check_in_bin(cube_pos, rc: RC) -> bool:
    bp = rc.bin_position
    return (abs(cube_pos[0] - bp[0]) < rc.bin_xy_tolerance and
            abs(cube_pos[1] - bp[1]) < rc.bin_xy_tolerance and
            cube_pos[2] > bp[2] - rc.bin_z_tolerance and
            cube_pos[2] < bp[2] + rc.bin_z_tolerance)


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
_hud_window = None
_hud_mass_label = None
_hud_friction_label = None


def setup_hud(headless):
    global _hud_window, _hud_mass_label, _hud_friction_label
    if headless:
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


def setup_event_indicator(headless):
    global _evt_window, _evt_label, _evt_params_label, _evt_rect
    if headless:
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


# ---------------------------------------------------------------------------
# Scene construction (parameterized by rc.workcell)
# ---------------------------------------------------------------------------

def build_workcell(stage, rc: RC):
    """Build a realistic industrial manufacturing workcell with PBR materials.
    All positions and sizes come from rc.workcell config."""
    from pxr import UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf

    wc = rc.workcell

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

    TABLE_H = rc.table_height
    fs = wc["floor_size"]

    # ===== Concrete floor =====
    _box("/World/cell/floor", [0, 0, -0.005], [fs[0], fs[1], 0.01], m_concrete, collision=False)
    rmp = wc["rubber_mat_pos"]
    rms = wc["rubber_mat_size"]
    _box("/World/cell/rubber_mat", rmp, rms, m_rubber, collision=False)

    # ===== Industrial workbench =====
    t = wc["table"]
    TABLE_CX = t["center_x"]
    ts = t["top_size"]
    ls = t["leg_spread"]
    _box("/World/cell/table_top", [TABLE_CX, 0.0, TABLE_H - 0.006],
         ts, m_aluminum, collision=False)
    for i, (lx, ly) in enumerate([
        (TABLE_CX - ls[0], -ls[1]), (TABLE_CX - ls[0], ls[1]),
        (TABLE_CX + ls[0], -ls[1]), (TABLE_CX + ls[0], ls[1]),
    ]):
        _box(f"/World/cell/table_leg_{i}", [lx, ly, TABLE_H/2 - 0.006],
             [0.05, 0.05, TABLE_H - 0.012], m_steel_dk, collision=False)
    _box("/World/cell/brace_f", [TABLE_CX, -ls[1], 0.03],
         [ls[0]*2 - 0.04, 0.04, 0.04], m_steel_dk, collision=False)
    _box("/World/cell/brace_b", [TABLE_CX, ls[1], 0.03],
         [ls[0]*2 - 0.04, 0.04, 0.04], m_steel_dk, collision=False)
    _box("/World/cell/brace_l", [TABLE_CX - ls[0], 0.00, 0.03],
         [0.04, ls[1]*2 - 0.04, 0.04], m_steel_dk, collision=False)
    _box("/World/cell/brace_r", [TABLE_CX + ls[0], 0.00, 0.03],
         [0.04, ls[1]*2 - 0.04, 0.04], m_steel_dk, collision=False)

    # Robot mounting plate
    mps = t["mount_plate_size"]
    _box("/World/cell/mount_plate", [0.0, 0.0, -0.005], mps, m_aluminum, collision=False)

    # ===== Conveyor belt assembly =====
    conv = wc["conveyor"]
    cx, cy = conv["center"]
    cl = conv["length"]
    cw = conv["width"]
    nr = conv["num_rollers"]
    _box("/World/cell/conv_belt", [cx, cy, TABLE_H + 0.003], [cl, cw, 0.006], m_belt, collision=True)
    _box("/World/cell/conv_rail_l", [cx, cy + cw/2 + 0.018, TABLE_H + 0.018],
         [cl, 0.030, 0.036], m_conv_frame, collision=True)
    _box("/World/cell/conv_rail_r", [cx, cy - cw/2 - 0.018, TABLE_H + 0.018],
         [cl, 0.030, 0.036], m_conv_frame, collision=True)
    _box("/World/cell/conv_end_near", [cx - cl/2, cy, TABLE_H + 0.018],
         [0.025, cw + 0.07, 0.036], m_conv_frame, collision=False)
    _box("/World/cell/conv_end_far",  [cx + cl/2, cy, TABLE_H + 0.018],
         [0.025, cw + 0.07, 0.036], m_conv_frame, collision=False)
    for i in range(nr):
        rx = cx - cl/2 + 0.05 + i * (cl - 0.10) / max(nr - 1, 1)
        _cyl(f"/World/cell/conv_roller_{i}", [rx, cy, TABLE_H - 0.012],
             radius=0.010, height=cw, mat=m_steel_lt, axis="Y", collision=False)

    # ===== Parts collection bin =====
    bp = rc.bin_position
    bn = wc["bin"]
    bw, bd, bh = bn["width"], bn["depth"], bn["height"]
    wt = bn["wall_thickness"]
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
    sls = bn["stand_leg_spread"]
    for i, (ox, oy) in enumerate([(-sls, -sls), (sls, -sls), (-sls, sls), (sls, sls)]):
        _box(f"/World/cell/bin_leg_{i}", [bp[0]+ox, bp[1]+oy, stand_top_z/2],
             [0.03, 0.03, stand_top_z], m_steel_dk, collision=False)

    # ===== Safety perimeter fence =====
    fc = wc["fence"]
    fh = fc["height"]
    fz = fh / 2
    for i, post in enumerate(fc["posts"]):
        _cyl(f"/World/cell/fence_post_{i}", [post[0], post[1], fz],
             radius=0.022, height=fh, mat=m_fence)
    pt = 0.005
    for i, p in enumerate(fc["panels_back"]):
        _box(f"/World/cell/fence_back_{i}", [p["pos"][0], p["pos"][1], fz],
             [p["width"], pt, fh-0.10], m_mesh, collision=False)
    for i, p in enumerate(fc["panels_right"]):
        _box(f"/World/cell/fence_right_{i}", [p["pos"][0], p["pos"][1], fz],
             [pt, p["depth"], fh-0.10], m_mesh, collision=False)
    for i, p in enumerate(fc["panels_left"]):
        _box(f"/World/cell/fence_left_{i}", [p["pos"][0], p["pos"][1], fz],
             [pt, p["depth"], fh-0.10], m_mesh, collision=False)
    rb = fc["rail_back"]
    _box("/World/cell/rail_back", [rb["pos"][0], rb["pos"][1], fh+0.01],
         [rb["width"], 0.035, 0.025], m_yellow, collision=False)
    rr = fc["rail_right"]
    _box("/World/cell/rail_right", [rr["pos"][0], rr["pos"][1], fh+0.01],
         [0.035, rr["depth"], 0.025], m_yellow, collision=False)
    rl = fc["rail_left"]
    _box("/World/cell/rail_left", [rl["pos"][0], rl["pos"][1], fh+0.01],
         [0.035, rl["depth"], 0.025], m_yellow, collision=False)

    # ===== Control cabinet =====
    cab = wc["cabinet"]["position"]
    _box("/World/cell/cabinet_body", [cab[0], cab[1], 0.40], [0.35, 0.30, 0.80], m_cabinet, collision=False)
    _box("/World/cell/cab_handle", [cab[0]+0.18, cab[1], 0.50], [0.012, 0.07, 0.006], m_steel_dk, collision=False)
    _box("/World/cell/cab_led", [cab[0]+0.14, cab[1]-0.12, 0.75], [0.018, 0.018, 0.018], m_green, collision=False)
    _box("/World/cell/cab_vent", [cab[0], cab[1], 0.08], [0.30, 0.25, 0.10], m_steel_dk, collision=False)
    _box("/World/cell/cable_tray", [cab[0]/2, cab[1]/2, 0.014],
         [abs(cab[0])-0.10, 0.07, 0.024], m_cable, collision=False)

    # ===== Emergency stop =====
    es = wc["estop"]["position"]
    _cyl("/World/cell/estop_post", [es[0], es[1], 0.50], radius=0.018, height=1.00, mat=m_yellow)
    _box("/World/cell/estop_plate", [es[0], es[1], 0.98], [0.07, 0.07, 0.012], m_yellow, collision=False)
    _cyl("/World/cell/estop_button", [es[0], es[1], 1.01], radius=0.028, height=0.035, mat=m_red)

    # ===== Safety floor markings =====
    tt, tw = 0.003, 0.05
    tm = wc["tape_markings"]
    _box("/World/cell/tape_front", [tm["front"]["pos"][0], tm["front"]["pos"][1], tt/2],
         [tm["front"]["width"], tw, tt], m_yellow, collision=False)
    _box("/World/cell/tape_right", [tm["right"]["pos"][0], tm["right"]["pos"][1], tt/2],
         [tw, tm["right"]["depth"], tt], m_yellow, collision=False)
    _box("/World/cell/tape_left",  [tm["left"]["pos"][0], tm["left"]["pos"][1], tt/2],
         [tw, tm["left"]["depth"], tt], m_yellow, collision=False)

    # ===== Overhead LED lighting =====
    for i, (lx, ly) in enumerate(wc["lights"]):
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

    print(f"[Scene] Manufacturing workcell built ({rc.robot_name} layout).")


def spawn_workpiece(world, rng, spawn_pos, dims, rc: RC, existing_cube=None):
    from isaacsim.core.api.objects import DynamicCuboid
    color = np.array(rng.uniform(0.2, 0.9, 3))
    if existing_cube is None:
        cube = world.scene.add(DynamicCuboid(
            prim_path=rc.cube_prim,
            name="pick_cube",
            position=spawn_pos,
            scale=dims,
            color=color,
            mass=0.5))
        return cube
    existing_cube.set_local_scale(dims)
    # Update visual color each episode
    try:
        import omni.usd
        from pxr import Gf
        _stage = omni.usd.get_context().get_stage()
        color_attr = _stage.GetPrimAtPath(rc.cube_prim).GetAttribute("primvars:displayColor")
        if color_attr.IsValid():
            color_attr.Set([Gf.Vec3f(*color)])
    except Exception:
        pass
    return existing_cube


# ---------------------------------------------------------------------------
# Robotiq gripper attachment and control
# ---------------------------------------------------------------------------

def setup_gripper(stage, rc: RC):
    """Attach the Robotiq base_link to the robot wrist via a FixedJoint.

    The gripper's ArticulationRootAPI is removed so it merges into the
    robot articulation tree (one unified articulation, stable physics).
    The joint connects to wrist_link (a rigid body); the flange
    offset is encoded in localPos0.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema

    WRIST_BODY = rc.robot_prim + "/" + rc.wrist_link
    GRIPPER_ROOT = rc.gripper_base  # e.g. "/World/ur5/robotiq"
    base_path = GRIPPER_ROOT + "/" + rc.gripper_base_link
    base_prim = stage.GetPrimAtPath(base_path)

    # Ensure base_link is a dynamic rigid body (NOT kinematic)
    if base_prim.IsValid() and base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(base_prim)
        rb.GetKinematicEnabledAttr().Set(False)

    # Remove ArticulationRootAPI from gripper prims so the gripper
    # does not form its own articulation.
    for check_path in [GRIPPER_ROOT, base_path]:
        p = stage.GetPrimAtPath(check_path)
        if p.IsValid():
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                print(f"[Gripper] Removed ArticulationRootAPI from {check_path}")
            if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                print(f"[Gripper] Removed PhysxArticulationAPI from {check_path}")

    # Clear any stale Xform on the gripper root
    grip_xf = UsdGeom.Xformable(stage.GetPrimAtPath(GRIPPER_ROOT))
    grip_xf.ClearXformOpOrder()

    # Compute flange position offset relative to wrist_link.
    flange_prim = stage.GetPrimAtPath(rc.eef_prim)
    if flange_prim.IsValid():
        offset_xf = UsdGeom.Xformable(flange_prim).GetLocalTransformation(
            Usd.TimeCode.Default()
        )
        offset_pos = offset_xf.ExtractTranslation()
    else:
        offset_pos = Gf.Vec3d(0, 0, 0)

    # Create FixedJoint, under the gripper root, not the base_link
    joint_path = GRIPPER_ROOT + "/wrist_fixed_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.GetBody0Rel().SetTargets([Sdf.Path(WRIST_BODY)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(base_path)])
        joint.GetLocalPos0Attr().Set(
            Gf.Vec3f(float(offset_pos[0]), float(offset_pos[1]),
                      float(offset_pos[2]))
        )
        joint.GetLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        print(f"[Gripper] FixedJoint: {WRIST_BODY} -> {base_path}  "
              f"offset=({offset_pos[0]:.4f}, {offset_pos[1]:.4f}, "
              f"{offset_pos[2]:.4f})  rot=identity")
    else:
        print("[Gripper] FixedJoint already exists")

    # Add PhysxMimicJointAPI to each mimic joint so all fingers follow
    # finger_joint.  Paths are relative to the gripper root (not base_link).
    finger_joint_path = Sdf.Path(GRIPPER_ROOT + "/joints/finger_joint")

    def _set_or_create(prim, name, val_type, val):
        attr = prim.GetAttribute(name)
        if attr.IsValid():
            attr.Set(val)
        else:
            prim.CreateAttribute(name, val_type).Set(val)

    for mj in rc.mimic_joints:
        jname, gearing = mj["name"], mj["gearing"]
        jpath = GRIPPER_ROOT + "/joints/" + jname
        jprim = stage.GetPrimAtPath(jpath)
        if not jprim.IsValid():
            continue
        # Detect the joint's rotation axis
        axis_attr = jprim.GetAttribute("physics:axis")
        axis = axis_attr.Get() if axis_attr.IsValid() else "X"
        rot_instance = "rot" + str(axis)
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
        print(f"[Gripper]   {jname}: axis={axis} mimic={rot_instance} gearing={gearing}")
    print(f"[Gripper] PhysxMimicJointAPI added to {len(rc.mimic_joints)} joints")

    # finger_joint DriveAPI is left as-is from the USD asset.
    # Runtime control goes through set_joint_position_targets in the sim loop.

    # Apply rubber-like friction material to finger pads.
    _DEFAULT_PAD_FRICTION = rc.pad_friction
    for pad_name in rc.finger_pad_links:
        pad_path = GRIPPER_ROOT + "/" + pad_name
        pad_prim = stage.GetPrimAtPath(pad_path)
        if not pad_prim.IsValid():
            continue
        mat_path = pad_path + "/GripMaterial"
        if not stage.GetPrimAtPath(mat_path).IsValid():
            from pxr import UsdShade
            mat = UsdShade.Material.Define(stage, mat_path)
            UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mat_prim = stage.GetPrimAtPath(mat_path)
        phys_mat = UsdPhysics.MaterialAPI(mat_prim)
        phys_mat.CreateStaticFrictionAttr().Set(rc.pad_friction)
        phys_mat.CreateDynamicFrictionAttr().Set(rc.pad_friction)
        phys_mat.CreateRestitutionAttr().Set(0.0)
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(pad_prim)
        UsdShade.MaterialBindingAPI(pad_prim).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.weakerThanDescendants,
            "physics"
        )
    print(f"[Gripper] Finger pad friction (default μ={_DEFAULT_PAD_FRICTION}) applied")


def _grip_strength_for_mass(cube_mass, rc: RC):
    """Scale grip PD gains and close angle with cube mass.
    Returns (kp, kd, close_rad)."""
    mass_lo, mass_hi = rc.cube_mass_range
    t = (cube_mass - mass_lo) / max(mass_hi - mass_lo, 1e-6)
    t = max(0.0, min(1.0, t))
    # PD gains
    kp_lo, kp_hi = rc.grip_kp_range
    kp = kp_lo + t * (kp_hi - kp_lo)
    kd = kp * 0.08
    # Close angle, heavier cubes get more overshoot for stronger squeeze
    close_deg = rc.close_deg_base + t * (rc.close_deg_max - rc.close_deg_base)
    close_rad = np.radians(close_deg)
    return kp, kd, close_rad


def update_gripper_for_episode(stage, rng, robot, cube_mass, cube_friction, rc: RC):
    """Set finger pad friction and grip gains scaled to cube mass.
    Returns (pad_friction, close_rad)."""
    from pxr import UsdPhysics
    pad_friction = rc.pad_friction

    for pad_name in rc.finger_pad_links:
        mat_path = rc.gripper_base + "/" + pad_name + "/GripMaterial"
        mat_prim = stage.GetPrimAtPath(mat_path)
        if mat_prim.IsValid():
            phys_mat = UsdPhysics.MaterialAPI(mat_prim)
            phys_mat.GetStaticFrictionAttr().Set(pad_friction)
            phys_mat.GetDynamicFrictionAttr().Set(pad_friction)

    kp, kd, close_rad = _grip_strength_for_mass(cube_mass, rc)
    robot._articulation_view.set_gains(
        kps=np.array([[kp] * len(rc.grip_joint_indices)]),
        kds=np.array([[kd] * len(rc.grip_joint_indices)]),
        joint_indices=rc.grip_joint_indices)

    print(f"[Gripper] pad_μ={pad_friction:.2f}  cube_μ={cube_friction:.2f}  "
          f"cube_mass={cube_mass:.3f}kg  kp={kp:.0f}  kd={kd:.0f}  "
          f"close={np.degrees(close_rad):.1f}deg")
    return pad_friction, close_rad


# ---------------------------------------------------------------------------
# RMPFlow controller
# ---------------------------------------------------------------------------

def create_rmpflow_controller(name, robot_articulation, rmpflow_name, physics_dt):
    """Create an RMPFlow controller for the given robot."""
    import isaacsim.robot_motion.motion_generation as mg

    class _URRMPFlowController(mg.MotionPolicyController):
        def __init__(self):
            rmp_config = mg.interface_config_loader.load_supported_motion_policy_config(
                rmpflow_name, "RMPflow"
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
            print(f"[RMPFlow] {rmpflow_name} controller initialized.")

        def reset(self):
            mg.MotionPolicyController.reset(self)
            self._motion_policy.set_robot_base_pose(
                robot_position=self._default_position,
                robot_orientation=self._default_orientation)

    return _URRMPFlowController()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    parser.add_argument("--events",   action="store_true", help="Enable random event injection")
    parser.add_argument("--max_events_per_episode", type=int, default=2)
    parser.add_argument("--event_i", type=int, default=None,
                        help="Debug: force event with this ID every episode")
    parser.add_argument("--run_type", type=str, default=None,
                        help="Tag for logging (e.g. 'baseline' or 'counterfactual')")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Override log output directory")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run(config_path: str, cli_args=None):
    """Main entry point. Loads config from config_path, runs simulation."""
    cfg = load_config(config_path)
    # task_dir is the pick_and_place directory (parent of config/)
    task_dir = str(Path(config_path).parent.parent.resolve())
    rc = init_from_config(cfg, task_dir)

    if cli_args is None:
        cli_args = parse_args()

    if cli_args.event_i is not None:
        cli_args.events = True

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "width": 1280, "height": 720,
        "headless": cli_args.headless,
    })

    # Deferred imports
    import carb
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.robot.manipulators.controllers import PickPlaceController

    # Shared episode_logger from FactoryBench/pick_and_place/
    _shared_dir = str(Path(__file__).parent.resolve())
    if _shared_dir not in sys.path:
        sys.path.insert(0, _shared_dir)
    from episode_logger import EpisodeLogger, collect_sensors, get_task_phase

    # Event injection (optional)
    if cli_args.events:
        if _FACTORYBENCH_DIR not in sys.path:
            sys.path.insert(0, _FACTORYBENCH_DIR)
        from event_injection import EventScheduler, SimContext, BUILTIN_APPLICATORS

    rng = np.random.RandomState(cli_args.seed)

    world = World(physics_dt=rc.sim_dt, rendering_dt=rc.sim_dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    world.scene.add_default_ground_plane()
    build_workcell(stage, rc)

    # Apply physics friction to collision surfaces
    _ensure_surface_friction(stage, "/World/cell/conv_belt", friction=0.6)
    _ensure_surface_friction(stage, "/World/cell/bin_floor", friction=0.8)
    _ensure_surface_friction(stage, "/World/cell/bin_stand_top", friction=0.6)

    # Load robot, USD from Isaac Sim assets, or URDF from local file
    if rc.robot_usd.endswith(".urdf"):
        from isaacsim.asset.importer.urdf import import_robot
        result = import_robot(
            asset_path=rc.robot_usd,
            prim_path=rc.robot_prim)
        print(f"[Robot] Imported URDF: {rc.robot_usd} -> {rc.robot_prim}")
    else:
        assets_root = get_assets_root_path()
        add_reference_to_stage(assets_root + rc.robot_usd, rc.robot_prim)
    robot = world.scene.add(Robot(prim_path=rc.robot_prim, name=rc.robot_name))

    # Load gripper
    if not rc.gripper_builtin:
        # External gripper (Robotiq), load USD and attach via FixedJoint
        add_reference_to_stage(rc.gripper_usd, rc.gripper_prim)
        setup_gripper(stage, rc)

    cur_dims  = sample_workpiece_dims(rng, rc)
    cur_spawn, cur_yaw = randomize_cube_pose(rng, rc)
    cur_spawn[2] = rc.table_height + cur_dims[2] / 2.0 + 0.005
    cube = spawn_workpiece(world, rng, cur_spawn, cur_dims, rc)

    world.reset()

    n_dof = robot.num_dof
    n_arm = len(rc.home_joints)
    print(f"[{rc.robot_name.upper()}] num_dof = {n_dof}  ({n_arm} arm + {n_dof - n_arm} gripper)")

    home_full = np.zeros(n_dof)
    home_full[:n_arm] = rc.home_joints
    robot.set_joints_default_state(positions=home_full)

    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    robot.set_joint_positions(home_full)
    robot.set_joint_velocities(np.zeros(n_dof))

    for _ in range(rc.init_steps):
        world.step(render=True)

    # Set up ParallelGripper
    if rc.gripper_builtin:
        gripper_eef_path = rc.gripper_eef_prim or (rc.robot_prim + "/panda_rightfinger")
    else:
        gripper_eef_path = rc.gripper_base + "/" + rc.gripper_base_link
    # For Robotiq (revolute): use grip_close_targets (radians from close_deg_max)
    # For built-in (prismatic): use gripper_close_positions as-is
    _pg_close = rc.grip_close_targets if not rc.gripper_builtin else rc.gripper_close_positions
    _pg_open  = rc.grip_open_targets  if not rc.gripper_builtin else rc.gripper_open_positions
    gripper = ParallelGripper(
        end_effector_prim_path=gripper_eef_path,
        joint_prim_names=rc.gripper_joint_names,
        joint_opened_positions=_pg_open,
        joint_closed_positions=_pg_close,
        action_deltas=None,
        use_mimic_joints=rc.gripper_use_mimic)
    gripper.initialize(
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names)
    fj_idx = gripper.joint_dof_indicies[0]

    # Gripper drive gains left at USD asset's original values.
    gripper.open()
    for _ in range(20):
        world.step(render=True)

    rmp_controller = create_rmpflow_controller(
        name=f"{rc.robot_name}_rmpflow",
        robot_articulation=robot,
        rmpflow_name=rc.rmpflow_name,
        physics_dt=rc.sim_dt)

    pick_place = PickPlaceController(
        name=f"{rc.robot_name}_pick_place",
        cspace_controller=rmp_controller,
        gripper=gripper,
        end_effector_initial_height=rc.eef_initial_height,
        events_dt=rc.events_dt)

    if not stage.GetPrimAtPath(rc.eef_prim).IsValid():
        carb.log_error(f"EEF prim {rc.eef_prim} not found. Aborting.")
        simulation_app.close()
        return

    _log_dir = cli_args.log_dir if cli_args.log_dir else rc.log_dir
    if cli_args.run_type:
        _log_dir = os.path.join(_log_dir, cli_args.run_type)
    logger = EpisodeLogger(_log_dir)

    _CLOSE_RAD = rc.grip_close_targets[0]  # max close in radians

    class _GripperProxy:
        def __init__(self, pg):
            self._pg = pg
        @property
        def attached(self):
            jp = self._pg.get_joint_positions()
            if jp is None:
                return False
            if rc.gripper_builtin:
                # Prismatic: fingers close toward 0, "attached" when below midpoint
                mid = float(rc.gripper_open_positions[0]) * 0.5
                return float(jp[0]) < mid
            else:
                # Revolute (Robotiq): "attached" when above 30% of close_deg_max
                return float(jp[0]) > _CLOSE_RAD * 0.3
        @property
        def is_closed(self):
            return self.attached
    gripper_proxy = _GripperProxy(gripper)

    # Phase step boundaries for event scheduling
    _phase_boundaries = [0]
    for dt in rc.events_dt:
        _phase_boundaries.append(_phase_boundaries[-1] + int(round(1.0 / dt)))

    event_scheduler = None
    if cli_args.events:
        events_json = Path(_FACTORYBENCH_DIR) / "events.json"
        event_scheduler = EventScheduler(
            events_json_path=str(events_json),
            task_name="pick_and_place",
            applicators=BUILTIN_APPLICATORS,
            rng_seed=cli_args.seed + 10000,
            num_events_range=(0, cli_args.max_events_per_episode),
            force_event_id=cli_args.event_i)
        event_scheduler.set_phase_boundaries(_phase_boundaries)

    cur_mass, cur_friction = randomize_cube_physics(cube, rng, rc, stage, UsdPhysics)
    if not rc.gripper_builtin:
        # Initial gains from mean mass (overridden per-episode)
        _init_kp, _init_kd, _ = _grip_strength_for_mass(np.mean(rc.cube_mass_range), rc)
        robot._articulation_view.set_gains(
            kps=np.array([[_init_kp] * len(rc.grip_joint_indices)]),
            kds=np.array([[_init_kd] * len(rc.grip_joint_indices)]),
            joint_indices=rc.grip_joint_indices)
        _, _cur_close_rad = update_gripper_for_episode(stage, rng, robot, cur_mass, cur_friction, rc)
    else:
        _cur_close_rad = _CLOSE_RAD

    # Diagnostic: print all joint params after full setup
    _all_gains = robot._articulation_view.get_gains()
    _all_efforts = robot._articulation_view.get_max_efforts()
    print(f"[Diag] DOF names: {robot.dof_names}")
    print(f"[Diag] kp:  {np.round(_all_gains[0][0], 1)}")
    print(f"[Diag] kd:  {np.round(_all_gains[1][0], 1)}")
    print(f"[Diag] max_efforts: {np.round(_all_efforts[0], 1)}")

    setup_hud(cli_args.headless)
    update_hud(cur_mass, cur_friction)
    setup_event_indicator(cli_args.headless)

    gripper.open()
    for _ in range(rc.settle_steps):
        world.step(render=True)

    logger.init_sensors(world, robot, cube, stage,
                        flange_prim_path=rc.eef_prim, sim_dt=rc.sim_dt)

    step           = 0
    episode        = 0
    sim_time       = 0.0
    ep_step        = 0
    plan_attempts  = 0
    plan_time_last = 0.0
    pick_attempts  = 0

    ee_orient = euler_angles_to_quat(np.array([0, np.pi, 0]))

    PHASE_NAMES = ["above_pick", "descend", "settle", "close",
                   "lift", "move_xy", "lower", "open",
                   "retract", "return"]

    _slip_detected = False
    _slip_prev_cube_pos = None
    _slip_prev_eef_pos = None
    _slip_rel_vel_window = []

    def compute_targets():
        nonlocal ee_orient
        perceived = apply_perception_noise(cur_spawn.copy(), rng, rc)
        pick = perceived.copy()
        pick[2] = perceived[2] + rc.tcp_z
        place = rc.bin_position.copy()
        place[2] = rc.bin_position[2] + rc.tcp_z + 0.04
        ee_orient = euler_angles_to_quat(np.array([0, np.pi, cur_yaw]))
        return pick, place

    def reset_episode(success: bool, reason: str = ""):
        nonlocal episode, cur_mass, cur_friction
        nonlocal cur_spawn, cur_dims, cube, cur_yaw
        nonlocal sim_time, plan_attempts, plan_time_last, pick_attempts
        nonlocal ep_step, pick_pos, place_pos, ee_orient
        nonlocal _slip_detected, _slip_rel_vel_window
        nonlocal _slip_prev_cube_pos, _slip_prev_eef_pos
        nonlocal _cur_close_rad

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

        if cli_args.episodes > 0 and episode >= cli_args.episodes:
            print(f"[FactoryBench] Reached {cli_args.episodes} episodes. Stopping.")
            logger.close()
            simulation_app.close()
            sys.exit(0)

        cur_dims  = sample_workpiece_dims(rng, rc)
        cur_spawn, cur_yaw = randomize_cube_pose(rng, rc)
        cur_spawn[2] = rc.table_height + cur_dims[2] / 2.0 + 0.005

        cube = spawn_workpiece(world, rng, cur_spawn, cur_dims, rc, existing_cube=cube)

        n_dof_reset = robot.num_dof
        home_reset = np.zeros(n_dof_reset)
        home_reset[:len(rc.home_joints)] = rc.home_joints
        robot.set_joints_default_state(positions=home_reset)
        cube.set_default_state(position=cur_spawn,
                               orientation=yaw_to_quat(cur_yaw))

        if not rc.gripper_builtin:
            setup_gripper(stage, rc)
        world.reset()

        # Set mass/friction AFTER world.reset() so physics engine picks them up
        cur_mass, cur_friction = randomize_cube_physics(cube, rng, rc, stage, UsdPhysics)
        print(f"[{rc.robot_name.upper()}] Episode {episode} -- "
              f"dims={np.round(cur_dims*1000).astype(int)}mm  "
              f"spawn={np.round(cur_spawn,3)}  mass={cur_mass:.3f}kg")

        robot.set_solver_position_iteration_count(64)
        robot.set_solver_velocity_iteration_count(64)
        rmp_controller.reset()
        pick_place.reset(end_effector_initial_height=rc.eef_initial_height)
        robot.set_joint_positions(home_reset)
        robot.set_joint_velocities(np.zeros(n_dof_reset))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        gripper.post_reset()

        if not rc.gripper_builtin:
            _, _cur_close_rad = update_gripper_for_episode(stage, rng, robot, cur_mass, cur_friction, rc)
        update_hud(cur_mass, cur_friction)

        gripper.open()

        for _ in range(rc.settle_steps):
            world.step(render=True)

        logger.reset_diff_state()
        ep_step = 0
        pick_pos, place_pos = compute_targets()
        print(f"  pick={np.round(pick_pos, 4)}  place={np.round(place_pos, 4)}")

        if event_scheduler is not None:
            _setup_ctx = SimContext(
                stage=stage,
                cube_prim_path=rc.cube_prim,
                robot_prim_path=rc.robot_prim,
                sim_dt=rc.sim_dt,
                extra={"robot": robot})
            event_scheduler.reset(_setup_ctx)
            event_scheduler.schedule_episode(max_episode_steps=1500)
            event_scheduler.setup_episode(_setup_ctx)

    # --- First episode ---
    print(f"[{rc.robot_name.upper()}] Episode {episode} -- "
          f"dims={np.round(cur_dims*1000).astype(int)}mm  "
          f"spawn={np.round(cur_spawn,3)}  mass={cur_mass:.3f}kg")
    pick_pos, place_pos = compute_targets()
    print(f"  pick={np.round(pick_pos, 4)}  place={np.round(place_pos, 4)}")
    pick_attempts = 1

    if event_scheduler is not None:
        _setup_ctx = SimContext(
            stage=stage,
            cube_prim_path=rc.cube_prim,
            robot_prim_path=rc.robot_prim,
            sim_dt=rc.sim_dt,
            extra={"robot": robot})
        event_scheduler.schedule_episode(max_episode_steps=1500)
        event_scheduler.setup_episode(_setup_ctx)

    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        step     += 1
        ep_step  += 1
        sim_time += rc.sim_dt

        cube_pos, _ = cube.get_world_pose()
        cube_pos    = np.asarray(cube_pos)
        if np.any(np.abs(cube_pos) > 10.0):
            cube_pos = cur_spawn.copy()

        if cube_pos[2] < rc.drop_z_threshold:
            print(f"[{step:6d}] Cube dropped. Resetting.")
            reset_episode(success=False, reason="dropped")
            continue

        if sim_time > rc.timeout_s:
            print(f"[{step:6d}] Episode timeout.")
            reset_episode(success=False, reason="timeout")
            continue

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
        if not rc.gripper_builtin:
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

        # For builtin grippers, grip_target may not be set in the else branch above
        _grip_cmd = grip_target if not rc.gripper_builtin else None

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
            joint_noise_std=rc.joint_noise_std,
            ee_target_pos=_ee_target,
            ee_target_quat=ee_orient,
            gripper_cmd_rad=_grip_cmd,
            gripper_pos_rad=_gripper_actual,
            controller_phase=phase)

        # Slip detection, compare cube and EEF velocities via finite
        # difference so both have identical lag characteristics.
        # A real slip produces sustained negative relative Z velocity.
        _slip_just_detected = False
        _SLIP_WINDOW = 40
        _SLIP_VEL_THRESHOLD = 0.03
        if phase >= 3 and phase < 7 and not _slip_detected:
            flange_xf = UsdGeom.Xformable(stage.GetPrimAtPath(rc.eef_prim))
            eef_world = flange_xf.ComputeLocalToWorldTransform(0)
            eef_pos = np.array([eef_world.ExtractTranslation()[i] for i in range(3)])

            if _slip_prev_cube_pos is not None and _slip_prev_eef_pos is not None:
                cube_vel_z = (cube_pos[2] - _slip_prev_cube_pos[2]) / rc.sim_dt
                eef_vel_z = (eef_pos[2] - _slip_prev_eef_pos[2]) / rc.sim_dt
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
                cube_prim_path=rc.cube_prim,
                robot_prim_path=rc.robot_prim,
                joint_names=_JNAMES,
                sim_dt=rc.sim_dt,
                episode_step=ep_step,
                state_machine=phase_name,
                extra={"robot": robot, "action": action})
            active_events = event_scheduler.step(ep_step, evt_ctx)
            if active_events:
                evt = active_events[0]
                sensor_row["event_id"] = evt.event_id
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
            in_bin = check_in_bin(cube_final, rc)
            if in_bin:
                print(f"[{step:6d}] Cycle complete, cube in bin.")
                reset_episode(success=True, reason="placed")
            else:
                print(f"[{step:6d}] Cycle complete, cube NOT in bin "
                      f"(pos={np.round(cube_final, 3)}).")
                reset_episode(success=False, reason="missed_bin")

    logger.close()
    simulation_app.close()


def main(config_path: str):
    """Load config and run. Called by per-robot thin wrappers."""
    cli_args = parse_args()
    run(config_path, cli_args)
