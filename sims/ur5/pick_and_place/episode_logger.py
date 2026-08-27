"""
episode_logger.py -- comprehensive per-step CSV sensor logger for the
UR pick-and-place sim.

Columns recorded
----------------
Bookkeeping:
  episode, global_step, episode_step, sim_time_s, wall_time_s,
  state_machine, task_phase

Per joint (6x each, suffix = joint name):
  joint_pos_rad_*          actual joint position, rad
  joint_vel_radps_*        joint velocity, rad/s
  joint_accel_radps2_*     joint acceleration (finite-diff of vel), rad/s^2
  joint_torque_nm_*        measured joint effort / torque, Nm
  joint_cmd_pos_rad_*      commanded position, rad
  joint_cmd_vel_radps_*    commanded velocity, rad/s
  joint_cmd_torque_nm_*    commanded effort, Nm
  joint_pos_error_rad_*    position tracking error (cmd - actual), rad

EEF setpoint (Cartesian target the controller is tracking):
  ee_cmd_pos_{x,y,z}_m              target TCP position, m
  ee_cmd_quat_{w,x,y,z}             target TCP orientation quaternion

EEF (end-effector / flange):
  ee_pos_{x,y,z}_m                  position, m
  ee_quat_{w,x,y,z}                 orientation quaternion (world frame)
  ee_euler_{roll,pitch,yaw}_rad     orientation Euler angles ZYX, rad
  ee_tool_z_dir_{x,y,z}             tool z-axis direction in world frame
  ee_linvel_{x,y,z}_mps             linear velocity, m/s
  ee_angvel_{x,y,z}_radps           angular velocity, rad/s
  ee_linacc_{x,y,z}_mps2            linear acceleration (finite-diff), m/s^2
  ee_angacc_{x,y,z}_radps2          angular acceleration (finite-diff), rad/s^2

Workpiece (cube):
  cube_pos_{x,y,z}_m                position, m
  cube_quat_{w,x,y,z}               orientation quaternion
  cube_euler_{roll,pitch,yaw}_rad   Euler angles ZYX, rad
  cube_linvel_{x,y,z}_mps           linear velocity, m/s
  cube_angvel_{x,y,z}_radps         angular velocity, rad/s
  cube_linacc_{x,y,z}_mps2          linear acceleration (finite-diff), m/s^2
  cube_angacc_{x,y,z}_radps2        angular acceleration (finite-diff), rad/s^2

Relative geometry:
  ee_cube_offset_{x,y,z}_m     ee_pos - cube_pos, m
  ee_cube_distance_m            Euclidean distance EEF to cube, m
  ee_cube_relvel_{x,y,z}_mps   relative velocity EEF - cube, m/s

Gripper setpoint / state:
  gripper_cmd_rad                    gripper finger_joint target, rad
  gripper_pos_rad                    gripper finger_joint actual position, rad
  controller_phase                   PickPlaceController phase index (0-9)

Gripper / contact:
  gripper_attached                       0 or 1
  contact_force_{x,y,z}_n               contact force vector at flange, N
  contact_force_mag_n                    contact force magnitude, N
  contact_torque_{x,y,z}_nm             contact torque at flange, Nm
  contact_torque_mag_nm                  contact torque magnitude, Nm

Domain randomization / episode params:
  cube_mass_kg, cube_friction_coeff, cube_restitution_coeff
  cube_width_m, cube_depth_m, cube_height_m

Planning / control:
  planning_attempts       RMPFlow planning attempts this episode
  planning_time_s         wall time of last planning call, s
  pick_attempts           suction attach attempts this episode

Standalone analysis:
  python episode_logger.py logs/run/steps.csv
"""

import csv
import math
import os
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]


def _build_columns():
    cols = [
        "episode", "global_step", "episode_step",
        "sim_time_s", "wall_time_s",
        "state_machine", "task_phase",
    ]

    for n in JOINT_NAMES:
        cols += [
            f"joint_pos_rad_{n}",
            f"joint_vel_radps_{n}",
            f"joint_accel_radps2_{n}",
            f"joint_torque_nm_{n}",
            f"joint_cmd_pos_rad_{n}",
            f"joint_cmd_vel_radps_{n}",
            f"joint_cmd_torque_nm_{n}",
            f"joint_pos_error_rad_{n}",
        ]

    # EEF setpoint (Cartesian target the controller is tracking)
    cols += [
        "ee_cmd_pos_x_m", "ee_cmd_pos_y_m", "ee_cmd_pos_z_m",
        "ee_cmd_quat_w", "ee_cmd_quat_x", "ee_cmd_quat_y", "ee_cmd_quat_z",
    ]

    cols += [
        "ee_pos_x_m", "ee_pos_y_m", "ee_pos_z_m",
        "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z",
        "ee_euler_roll_rad", "ee_euler_pitch_rad", "ee_euler_yaw_rad",
        "ee_tool_z_dir_x", "ee_tool_z_dir_y", "ee_tool_z_dir_z",
        "ee_linvel_x_mps", "ee_linvel_y_mps", "ee_linvel_z_mps",
        "ee_angvel_x_radps", "ee_angvel_y_radps", "ee_angvel_z_radps",
        "ee_linacc_x_mps2", "ee_linacc_y_mps2", "ee_linacc_z_mps2",
        "ee_angacc_x_radps2", "ee_angacc_y_radps2", "ee_angacc_z_radps2",
    ]

    cols += [
        "cube_pos_x_m", "cube_pos_y_m", "cube_pos_z_m",
        "cube_quat_w", "cube_quat_x", "cube_quat_y", "cube_quat_z",
        "cube_euler_roll_rad", "cube_euler_pitch_rad", "cube_euler_yaw_rad",
        "cube_linvel_x_mps", "cube_linvel_y_mps", "cube_linvel_z_mps",
        "cube_angvel_x_radps", "cube_angvel_y_radps", "cube_angvel_z_radps",
        "cube_linacc_x_mps2", "cube_linacc_y_mps2", "cube_linacc_z_mps2",
        "cube_angacc_x_radps2", "cube_angacc_y_radps2", "cube_angacc_z_radps2",
    ]

    cols += [
        "ee_cube_offset_x_m", "ee_cube_offset_y_m", "ee_cube_offset_z_m",
        "ee_cube_distance_m",
        "ee_cube_relvel_x_mps", "ee_cube_relvel_y_mps", "ee_cube_relvel_z_mps",
    ]

    # Gripper setpoint / feedback
    cols += [
        "gripper_cmd_rad",
        "gripper_pos_rad",
    ]

    # Controller phase
    cols += [
        "controller_phase",
    ]

    cols += [
        "gripper_attached",
        "contact_force_x_n", "contact_force_y_n", "contact_force_z_n",
        "contact_force_mag_n",
        "contact_torque_x_nm", "contact_torque_y_nm", "contact_torque_z_nm",
        "contact_torque_mag_nm",
    ]

    cols += [
        "cube_mass_kg", "cube_friction_coeff", "cube_restitution_coeff",
        "cube_width_m", "cube_depth_m", "cube_height_m",
    ]

    cols += [
        "planning_attempts",
        "planning_time_s",
        "pick_attempts",
    ]

    cols += ["event_id", "event_params"]

    return cols


COLUMNS = _build_columns()


# ---------------------------------------------------------------------------
# EpisodeLogger
# ---------------------------------------------------------------------------

class EpisodeLogger:

    def __init__(self, log_dir: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._dir       = log_dir
        self._run_start = time.time()
        self._ep_steps  = 0

        steps_path = os.path.join(log_dir, "steps.csv")
        self._steps_fh     = open(steps_path, "w", newline="", buffering=1)
        self._steps_writer = csv.DictWriter(
            self._steps_fh, fieldnames=COLUMNS, extrasaction="ignore"
        )
        self._steps_writer.writeheader()

        summary_path = os.path.join(log_dir, "episodes.csv")
        self._summary_fh = open(summary_path, "w", newline="", buffering=1)
        _summary_cols = [
            "episode", "success", "reason", "episode_steps",
            "wall_time_s", "sim_time_s",
            "cube_mass_kg", "cube_friction_coeff", "cube_restitution_coeff",
            "cube_width_m", "cube_depth_m", "cube_height_m",
            "cube_spawn_x_m", "cube_spawn_y_m", "cube_spawn_z_m",
            "planning_attempts", "pick_attempts",
        ]
        self._summary_writer = csv.DictWriter(
            self._summary_fh, fieldnames=_summary_cols, extrasaction="ignore"
        )
        self._summary_writer.writeheader()

        self._contact_sensor = None
        self._flange_path    = None

        # Finite-difference state (previous step values)
        self._prev_jvel    = None   # (6,) rad/s
        self._prev_ee_vel  = None   # (3,) m/s
        self._prev_ee_avel = None   # (3,) rad/s
        self._prev_cu_vel  = None   # (3,) m/s
        self._prev_cu_avel = None   # (3,) rad/s
        self._sim_dt       = 1.0 / 60.0

        print(f"[Logger] Writing to {os.path.abspath(log_dir)}/  "
              f"({len(COLUMNS)} columns)")

    def init_sensors(self, world, robot, cube, stage, flange_prim_path: str,
                     sim_dt: float = 1.0 / 60.0):
        self._flange_path = flange_prim_path
        self._sim_dt      = sim_dt
        self._robot       = robot

        # Enable joint effort (torque) reporting -- off by default in Isaac Sim.
        try:
            _ = robot.get_measured_joint_efforts()
            self._effort_api = True
            print("[Logger] Joint effort sensing enabled.")
        except Exception as e:
            self._effort_api = False
            print(f"[Logger] Joint effort sensing unavailable ({e}).")

        # Contact sensor on the flange -- records full force+torque vector
        try:
            from isaacsim.sensors.physics import ContactSensor
            self._contact_sensor = ContactSensor(
                prim_path=flange_prim_path + "/contact_sensor",
                name="flange_contact",
                min_threshold=0,
                max_threshold=1e6,
                radius=-1,
            )
            world.scene.add(self._contact_sensor)
            print("[Logger] ContactSensor attached to flange.")
        except Exception as e:
            print(f"[Logger] ContactSensor unavailable ({e}), contact fields=0.")

    def reset_diff_state(self):
        """Call at the start of each episode to avoid stale finite-diff values."""
        self._prev_jvel    = None
        self._prev_ee_vel  = None
        self._prev_ee_avel = None
        self._prev_cu_vel  = None
        self._prev_cu_avel = None

    def step(self, data: dict):
        self._ep_steps   += 1
        data["episode_step"] = self._ep_steps
        data["wall_time_s"]  = round(time.time() - self._run_start, 4)
        self._steps_writer.writerow(data)

    def end_episode(self, episode: int, success: bool, reason: str = "",
                    cube_mass: float = 0.0, cube_friction: float = 0.0,
                    cube_restitution: float = 0.0,
                    cube_dims: np.ndarray = None,
                    cube_spawn: np.ndarray = None,
                    sim_time: float = 0.0,
                    plan_attempts: int = 0,
                    pick_attempts: int = 0):
        row = {
            "episode":                episode,
            "success":                int(success),
            "reason":                 reason,
            "episode_steps":          self._ep_steps,
            "wall_time_s":            round(time.time() - self._run_start, 4),
            "sim_time_s":             round(sim_time, 4),
            "cube_mass_kg":           round(cube_mass,        4),
            "cube_friction_coeff":    round(cube_friction,    4),
            "cube_restitution_coeff": round(cube_restitution, 4),
            "cube_width_m":           round(float(cube_dims[0]), 4) if cube_dims is not None else "",
            "cube_depth_m":           round(float(cube_dims[1]), 4) if cube_dims is not None else "",
            "cube_height_m":          round(float(cube_dims[2]), 4) if cube_dims is not None else "",
            "cube_spawn_x_m":        round(float(cube_spawn[0]), 4) if cube_spawn is not None else "",
            "cube_spawn_y_m":        round(float(cube_spawn[1]), 4) if cube_spawn is not None else "",
            "cube_spawn_z_m":        round(float(cube_spawn[2]), 4) if cube_spawn is not None else "",
            "planning_attempts":     plan_attempts,
            "pick_attempts":         pick_attempts,
        }
        self._summary_writer.writerow(row)
        self._summary_fh.flush()
        self._steps_fh.flush()
        self._ep_steps = 0
        self.reset_diff_state()

    def close(self):
        self._steps_fh.flush();   self._steps_fh.close()
        self._summary_fh.flush(); self._summary_fh.close()
        print(f"[Logger] Closed. Data in {os.path.abspath(self._dir)}/")

    def get_contact_data(self):
        """
        Returns (force_vec, torque_vec) as (3,) arrays in Newtons / Nm.
        Falls back to zeros if sensor is unavailable or returns unexpected shape.
        """
        def _to3(val):
            arr = np.asarray(val, dtype=float).flatten()
            out = np.zeros(3)
            out[:min(3, len(arr))] = arr[:3]
            return out

        zero = np.zeros(3)
        if self._contact_sensor is None:
            return zero, zero
        try:
            frame  = self._contact_sensor.get_current_frame()
            force  = _to3(frame.get("force",  [0, 0, 0]))
            torque = _to3(frame.get("torque", [0, 0, 0]))
            return force, torque
        except Exception:
            return zero, zero

    # ------------------------------------------------------------------
    # Finite-difference helpers (called from collect_sensors)
    # ------------------------------------------------------------------

    def _diff(self, current, prev_attr):
        """Compute (current - prev) / dt, store current as new prev."""
        prev = getattr(self, prev_attr)
        result = (current - prev) / self._sim_dt if prev is not None else np.zeros_like(current)
        setattr(self, prev_attr, current.copy())
        return result


# ---------------------------------------------------------------------------
# Task phase mapping
# ---------------------------------------------------------------------------

# Maps waypoint names to human-readable task phase descriptions.
WAYPOINT_PHASE_MAP = {
    "go_home":         "moving_to_home",
    "above_cube":      "approaching_cube",
    "descend":         "descending_to_grasp",
    "lift":            "lifting_cube",
    "neutral_to_bin":  "transiting_to_bin",
    "above_bin":       "approaching_bin",
    "place_descend":   "lowering_to_place",
    "retract_bin":     "retracting_from_bin",
    "neutral_to_home": "transiting_to_home",
    "home":            "returning_home",
}

# Maps state machine states to task phases (used when not in MOVE_TO_WP).
STATE_PHASE_MAP = {
    "INIT":           "initializing",
    "ATTACH":         "grasping",
    "CLOSE_GRIPPER":  "grasping",
    "DETACH":         "releasing",
    "OPEN_GRIPPER":   "releasing",
    "DONE":           "cycle_complete",
    "WAIT_SETTLE":    "settling",
}


def get_task_phase(state: str, waypoint_name: str = "") -> str:
    """Derive a descriptive task phase from the state machine state and
    the current waypoint name (if in MOVE_TO_WP)."""
    if state == "MOVE_TO_WP" and waypoint_name:
        return WAYPOINT_PHASE_MAP.get(waypoint_name, f"moving_to_{waypoint_name}")
    return STATE_PHASE_MAP.get(state, state.lower())


# ---------------------------------------------------------------------------
# collect_sensors  (called every sim step)
# ---------------------------------------------------------------------------

def collect_sensors(
    logger,
    episode:         int,
    sim_step:        int,
    sim_time:        float,
    state:           str,
    robot,
    cube,
    stage,
    suction,
    cur_mass:        float,
    cur_friction:    float,
    cur_restitution: float,
    cur_dims:        np.ndarray,
    plan_attempts:   int   = 0,
    plan_time_last:  float = 0.0,
    pick_attempts:   int   = 0,
    planned_action         = None,
    task_phase:      str   = "",
    joint_noise_std: float = 0.0,
    ee_target_pos:   np.ndarray = None,
    ee_target_quat:  np.ndarray = None,
    gripper_cmd_rad: float = None,
    gripper_pos_rad: float = None,
    controller_phase: int  = None,
) -> dict:

    row = {c: "" for c in COLUMNS}
    row["episode"]            = episode
    row["global_step"]        = sim_step
    row["sim_time_s"]         = _f(sim_time)
    row["state_machine"]      = state
    row["task_phase"]         = task_phase
    row["planning_attempts"]  = plan_attempts
    row["planning_time_s"]    = _f(plan_time_last)
    row["pick_attempts"]      = pick_attempts
    row["event_id"]           = 0
    row["event_params"]       = ""
    row["cube_mass_kg"]           = _f(cur_mass)
    row["cube_friction_coeff"]    = _f(cur_friction)
    row["cube_restitution_coeff"] = _f(cur_restitution)
    if cur_dims is not None and len(cur_dims) >= 3:
        row["cube_width_m"]  = _f(cur_dims[0])
        row["cube_depth_m"]  = _f(cur_dims[1])
        row["cube_height_m"] = _f(cur_dims[2])

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------
    jpos = np.zeros(6)
    jvel = np.zeros(6)
    jeff = np.zeros(6)
    try:
        js   = robot.get_joints_state()
        jpos = _flat(js.positions,  6)
        jvel = _flat(js.velocities, 6)
        if getattr(logger, "_effort_api", False):
            try:
                jeff = _flat(logger._robot.get_measured_joint_efforts(), 6)
            except Exception:
                jeff = _flat(js.efforts, 6)
        else:
            jeff = _flat(js.efforts, 6)
    except Exception:
        pass

    # Apply encoder noise to joint readings (simulates real sensor noise)
    if joint_noise_std > 0:
        jpos = jpos + np.random.normal(0, joint_noise_std, jpos.shape)
        jvel = jvel + np.random.normal(0, joint_noise_std * 10, jvel.shape)
        jeff = jeff + np.random.normal(0, joint_noise_std * 50, jeff.shape)

    jacc = logger._diff(jvel, "_prev_jvel")

    jcmd_pos = np.full(6, np.nan)
    jcmd_vel = np.full(6, np.nan)
    jcmd_eff = np.full(6, np.nan)
    if planned_action is not None:
        try:
            if planned_action.joint_positions is not None:
                jcmd_pos = _flat(planned_action.joint_positions, 6)
            if planned_action.joint_velocities is not None:
                jcmd_vel = _flat(planned_action.joint_velocities, 6)
            if planned_action.joint_efforts is not None:
                jcmd_eff = _flat(planned_action.joint_efforts, 6)
        except Exception:
            pass

    for i, n in enumerate(JOINT_NAMES):
        row[f"joint_pos_rad_{n}"]       = _f(jpos[i])
        row[f"joint_vel_radps_{n}"]     = _f(jvel[i])
        row[f"joint_accel_radps2_{n}"]  = _f(jacc[i])
        row[f"joint_torque_nm_{n}"]     = _f(jeff[i])
        row[f"joint_cmd_pos_rad_{n}"]   = _fn(jcmd_pos[i])
        row[f"joint_cmd_vel_radps_{n}"] = _fn(jcmd_vel[i])
        row[f"joint_cmd_torque_nm_{n}"] = _fn(jcmd_eff[i])
        if not np.isnan(jcmd_pos[i]):
            row[f"joint_pos_error_rad_{n}"] = _f(jcmd_pos[i] - jpos[i])

    # ------------------------------------------------------------------
    # EEF setpoint (Cartesian target)
    # ------------------------------------------------------------------
    if ee_target_pos is not None:
        row["ee_cmd_pos_x_m"] = _f(ee_target_pos[0])
        row["ee_cmd_pos_y_m"] = _f(ee_target_pos[1])
        row["ee_cmd_pos_z_m"] = _f(ee_target_pos[2])
    if ee_target_quat is not None:
        row["ee_cmd_quat_w"] = _f(ee_target_quat[0])
        row["ee_cmd_quat_x"] = _f(ee_target_quat[1])
        row["ee_cmd_quat_y"] = _f(ee_target_quat[2])
        row["ee_cmd_quat_z"] = _f(ee_target_quat[3])

    # ------------------------------------------------------------------
    # Gripper setpoint / feedback
    # ------------------------------------------------------------------
    if gripper_cmd_rad is not None:
        row["gripper_cmd_rad"] = _f(gripper_cmd_rad)
    if gripper_pos_rad is not None:
        row["gripper_pos_rad"] = _f(gripper_pos_rad)

    # ------------------------------------------------------------------
    # Controller phase
    # ------------------------------------------------------------------
    if controller_phase is not None:
        row["controller_phase"] = controller_phase

    # ------------------------------------------------------------------
    # EEF pose, velocity, acceleration
    # ------------------------------------------------------------------
    ee_pos  = np.zeros(3)
    ee_vel  = np.zeros(3)
    ee_avel = np.zeros(3)
    try:
        from pxr import UsdGeom
        _tool0_candidates = [
            "/World/ur3/wrist_3_link/tool0",
            "/World/ur5/wrist_3_link/tool0",
            "/World/ur3/tool0",
            "/World/ur5/tool0",
            logger._flange_path or "",
        ]
        fp = next((p for p in _tool0_candidates if p and stage.GetPrimAtPath(p).IsValid()),
                  logger._flange_path)
        prim = stage.GetPrimAtPath(fp)
        mat  = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            __import__("pxr").Usd.TimeCode.Default()
        )
        t    = mat.ExtractTranslation()
        ee_pos = np.array([t[0], t[1], t[2]])
        row["ee_pos_x_m"] = _f(t[0])
        row["ee_pos_y_m"] = _f(t[1])
        row["ee_pos_z_m"] = _f(t[2])

        R = mat.ExtractRotationMatrix()
        qw, qx, qy, qz = _rotmat_to_quat(R)
        row["ee_quat_w"] = _f(qw); row["ee_quat_x"] = _f(qx)
        row["ee_quat_y"] = _f(qy); row["ee_quat_z"] = _f(qz)

        roll, pitch, yaw = _quat_to_euler(qw, qx, qy, qz)
        row["ee_euler_roll_rad"]  = _f(roll)
        row["ee_euler_pitch_rad"] = _f(pitch)
        row["ee_euler_yaw_rad"]   = _f(yaw)

        row["ee_tool_z_dir_x"] = _f(R[2][0])
        row["ee_tool_z_dir_y"] = _f(R[2][1])
        row["ee_tool_z_dir_z"] = _f(R[2][2])

    except Exception:
        pass

    try:
        from pxr import UsdPhysics as _UP
        candidate_paths = [
            "/World/ur3/wrist_3_link",
            "/World/ur5/wrist_3_link",
            logger._flange_path or "",
        ]
        lv, av = None, None
        for cp in candidate_paths:
            if not cp:
                continue
            prim = stage.GetPrimAtPath(cp)
            if not prim.IsValid():
                continue
            rb = _UP.RigidBodyAPI(prim)
            lv = rb.GetVelocityAttr().Get()
            av = rb.GetAngularVelocityAttr().Get()
            if lv is not None:
                break
        if lv is not None:
            ee_vel = np.array([lv[0], lv[1], lv[2]])
            row["ee_linvel_x_mps"] = _f(lv[0])
            row["ee_linvel_y_mps"] = _f(lv[1])
            row["ee_linvel_z_mps"] = _f(lv[2])
        if av is not None:
            ee_avel = np.array([av[0], av[1], av[2]])
            row["ee_angvel_x_radps"] = _f(av[0])
            row["ee_angvel_y_radps"] = _f(av[1])
            row["ee_angvel_z_radps"] = _f(av[2])
    except Exception:
        pass

    ee_acc  = logger._diff(ee_vel,  "_prev_ee_vel")
    ee_aacc = logger._diff(ee_avel, "_prev_ee_avel")
    row["ee_linacc_x_mps2"] = _f(ee_acc[0])
    row["ee_linacc_y_mps2"] = _f(ee_acc[1])
    row["ee_linacc_z_mps2"] = _f(ee_acc[2])
    row["ee_angacc_x_radps2"] = _f(ee_aacc[0])
    row["ee_angacc_y_radps2"] = _f(ee_aacc[1])
    row["ee_angacc_z_radps2"] = _f(ee_aacc[2])

    # ------------------------------------------------------------------
    # Workpiece pose, velocity, acceleration
    # ------------------------------------------------------------------
    cube_pos  = np.zeros(3)
    cube_vel  = np.zeros(3)
    cube_avel = np.zeros(3)
    try:
        cube_pos, cube_q = cube.get_world_pose()
        cube_pos = np.asarray(cube_pos)
        row["cube_pos_x_m"] = _f(cube_pos[0])
        row["cube_pos_y_m"] = _f(cube_pos[1])
        row["cube_pos_z_m"] = _f(cube_pos[2])
        row["cube_quat_w"] = _f(cube_q[0]); row["cube_quat_x"] = _f(cube_q[1])
        row["cube_quat_y"] = _f(cube_q[2]); row["cube_quat_z"] = _f(cube_q[3])
        r, p, y = _quat_to_euler(cube_q[0], cube_q[1], cube_q[2], cube_q[3])
        row["cube_euler_roll_rad"]  = _f(r)
        row["cube_euler_pitch_rad"] = _f(p)
        row["cube_euler_yaw_rad"]   = _f(y)
    except Exception:
        pass

    try:
        lv = _flat(cube.get_linear_velocity(),  3)
        av = _flat(cube.get_angular_velocity(), 3)
        cube_vel  = lv
        cube_avel = av
        row["cube_linvel_x_mps"]  = _f(lv[0])
        row["cube_linvel_y_mps"]  = _f(lv[1])
        row["cube_linvel_z_mps"]  = _f(lv[2])
        row["cube_angvel_x_radps"] = _f(av[0])
        row["cube_angvel_y_radps"] = _f(av[1])
        row["cube_angvel_z_radps"] = _f(av[2])
    except Exception:
        pass

    cu_acc  = logger._diff(cube_vel,  "_prev_cu_vel")
    cu_aacc = logger._diff(cube_avel, "_prev_cu_avel")
    row["cube_linacc_x_mps2"]  = _f(cu_acc[0])
    row["cube_linacc_y_mps2"]  = _f(cu_acc[1])
    row["cube_linacc_z_mps2"]  = _f(cu_acc[2])
    row["cube_angacc_x_radps2"] = _f(cu_aacc[0])
    row["cube_angacc_y_radps2"] = _f(cu_aacc[1])
    row["cube_angacc_z_radps2"] = _f(cu_aacc[2])

    # ------------------------------------------------------------------
    # Relative geometry
    # ------------------------------------------------------------------
    try:
        rel    = ee_pos - cube_pos
        rel_v  = ee_vel - cube_vel
        row["ee_cube_offset_x_m"] = _f(rel[0])
        row["ee_cube_offset_y_m"] = _f(rel[1])
        row["ee_cube_offset_z_m"] = _f(rel[2])
        row["ee_cube_distance_m"] = _f(float(np.linalg.norm(rel)))
        row["ee_cube_relvel_x_mps"] = _f(rel_v[0])
        row["ee_cube_relvel_y_mps"] = _f(rel_v[1])
        row["ee_cube_relvel_z_mps"] = _f(rel_v[2])
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Contact sensor
    # ------------------------------------------------------------------
    row["gripper_attached"] = int(suction.attached)
    force_vec, torque_vec   = logger.get_contact_data()
    row["contact_force_x_n"]   = _f(force_vec[0])
    row["contact_force_y_n"]   = _f(force_vec[1])
    row["contact_force_z_n"]   = _f(force_vec[2])
    row["contact_force_mag_n"] = _f(float(np.linalg.norm(force_vec)))
    row["contact_torque_x_nm"]   = _f(torque_vec[0])
    row["contact_torque_y_nm"]   = _f(torque_vec[1])
    row["contact_torque_z_nm"]   = _f(torque_vec[2])
    row["contact_torque_mag_nm"] = _f(float(np.linalg.norm(torque_vec)))

    return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> str:
    try:
        return f"{float(v):.6f}"
    except Exception:
        return ""

def _fn(v) -> str:
    """Like _f but returns empty string for NaN."""
    try:
        f = float(v)
        return "" if math.isnan(f) else f"{f:.6f}"
    except Exception:
        return ""

def _flat(x, n: int = None) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float).flatten()
        if n is not None:
            out = np.zeros(n)
            out[:min(n, len(arr))] = arr[:n]
            return out
        return arr
    except Exception:
        return np.zeros(n) if n else np.array([])

def _rotmat_to_quat(r):
    m = [[r[j][i] for j in range(3)] for i in range(3)]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return 0.25/s, (m[2][1]-m[1][2])*s, (m[0][2]-m[2][0])*s, (m[1][0]-m[0][1])*s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        return (m[2][1]-m[1][2])/s, 0.25*s, (m[0][1]+m[1][0])/s, (m[0][2]+m[2][0])/s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        return (m[0][2]-m[2][0])/s, (m[0][1]+m[1][0])/s, 0.25*s, (m[1][2]+m[2][1])/s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        return (m[1][0]-m[0][1])/s, (m[0][2]+m[2][0])/s, (m[1][2]+m[2][1])/s, 0.25*s

def _quat_to_euler(qw, qx, qy, qz):
    """ZYX Euler angles (roll, pitch, yaw) in radians."""
    sinr = 2.0 * (qw*qx + qy*qz)
    cosr = 1.0 - 2.0 * (qx*qx + qy*qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (qw*qy - qz*qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2.0 * (qw*qz + qx*qy)
    cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
    yaw  = math.atan2(siny, cosy)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Standalone analysis
# ---------------------------------------------------------------------------

def _analyze(path: str):
    import sys
    from collections import Counter

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print("No rows."); return

    def col(name):
        vals = []
        for r in rows:
            try: vals.append(float(r[name]))
            except Exception: pass
        return np.array(vals) if vals else np.array([])

    def sec(title):
        print(f"\n{'='*60}\n{title}\n{'='*60}")

    print(f"\nLog  : {path}")
    print(f"Rows : {len(rows)}")
    ep = col("episode")
    if ep.size: print(f"Eps  : {int(ep.max())}")
    st = col("sim_time_s")
    if st.size: print(f"Sim time range: {st.min():.2f} - {st.max():.2f} s")

    sec("Task phase distribution")
    pc = Counter(r.get("task_phase", "") for r in rows)
    for p, c in sorted(pc.items(), key=lambda x: -x[1]):
        print(f"  {p:30s}: {c:6d} steps")

    sec("Joint positions (rad)")
    for n in JOINT_NAMES:
        v = col(f"joint_pos_rad_{n}")
        if v.size: print(f"  {n:15s}: [{v.min():.3f}, {v.max():.3f}]  std={v.std():.3f}")

    sec("Joint velocities (rad/s)")
    for n in JOINT_NAMES:
        v = col(f"joint_vel_radps_{n}")
        if v.size: print(f"  {n:15s}: [{v.min():.3f}, {v.max():.3f}]  std={v.std():.3f}")

    sec("Joint accelerations (rad/s^2)")
    for n in JOINT_NAMES:
        v = col(f"joint_accel_radps2_{n}")
        if v.size and v.std() > 1e-9:
            print(f"  {n:15s}: [{v.min():.2f}, {v.max():.2f}]  std={v.std():.2f}")

    sec("Joint tracking error (cmd - actual, rad)")
    for n in JOINT_NAMES:
        v = col(f"joint_pos_error_rad_{n}")
        if v.size and v.std() > 1e-9:
            print(f"  {n:15s}: mean={v.mean():.4f}  std={v.std():.4f}  max_abs={np.abs(v).max():.4f}")

    sec("Joint torques (Nm)")
    for n in JOINT_NAMES:
        v = col(f"joint_torque_nm_{n}")
        if v.size and v.std() > 1e-9:
            print(f"  {n:15s}: [{v.min():.2f}, {v.max():.2f}]  std={v.std():.2f}")

    sec("EEF position (m)")
    for ax in "xyz":
        v = col(f"ee_pos_{ax}_m")
        if v.size: print(f"  {ax}: [{v.min():.3f}, {v.max():.3f}]")

    sec("EEF orientation Euler (rad)")
    for ax in ["roll", "pitch", "yaw"]:
        v = col(f"ee_euler_{ax}_rad")
        if v.size: print(f"  {ax:5s}: [{v.min():.3f}, {v.max():.3f}]")

    sec("EEF tool_z direction (should be ~[0,0,-1])")
    for ax in "xyz":
        v = col(f"ee_tool_z_dir_{ax}")
        if v.size: print(f"  z_{ax}: mean={v.mean():.3f}  std={v.std():.4f}")

    sec("EEF linear velocity (m/s)")
    for ax in "xyz":
        v = col(f"ee_linvel_{ax}_mps")
        if v.size and v.std() > 1e-9: print(f"  v{ax}: [{v.min():.3f}, {v.max():.3f}]")

    sec("EEF linear acceleration (m/s^2)")
    for ax in "xyz":
        v = col(f"ee_linacc_{ax}_mps2")
        if v.size and v.std() > 1e-9: print(f"  a{ax}: [{v.min():.2f}, {v.max():.2f}]")

    sec("Cube position (m)")
    for ax in "xyz":
        v = col(f"cube_pos_{ax}_m")
        if v.size: print(f"  {ax}: [{v.min():.3f}, {v.max():.3f}]")

    sec("Cube velocity (m/s)")
    for ax in "xyz":
        v = col(f"cube_linvel_{ax}_mps")
        if v.size and v.std() > 1e-9: print(f"  v{ax}: [{v.min():.3f}, {v.max():.3f}]")

    sec("Relative EEF-cube geometry")
    d = col("ee_cube_distance_m")
    if d.size: print(f"  dist: mean={d.mean():.3f}  min={d.min():.4f}  max={d.max():.3f}")

    sec("Contact forces (N)")
    fm = col("contact_force_mag_n")
    if fm.size and fm.max() > 0:
        active = fm[fm > 0]
        print(f"  force mag: mean={active.mean():.2f}  max={fm.max():.2f}  "
              f"steps with contact={len(active)}/{len(fm)}")
    tm = col("contact_torque_mag_nm")
    if tm.size and tm.max() > 0:
        print(f"  torque mag: mean={tm[tm>0].mean():.3f}  max={tm.max():.3f}")

    sec("Domain randomization")
    for name in ["cube_mass_kg", "cube_friction_coeff", "cube_restitution_coeff",
                 "cube_width_m", "cube_depth_m", "cube_height_m"]:
        v = col(name)
        if v.size and v.std() > 1e-9:
            print(f"  {name:25s}: [{v.min():.3f}, {v.max():.3f}]")

    sec("State machine distribution")
    sc = Counter(r.get("state_machine", "") for r in rows)
    for s, c in sorted(sc.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {c}")

    sec("Planning stats")
    pa = col("planning_attempts")
    pt = col("planning_time_s")
    pk = col("pick_attempts")
    if pa.size: print(f"  planning_attempts: total={int(pa.sum())}  max_per_step={int(pa.max())}")
    if pt.size and pt.max() > 0:
        active = pt[pt > 0]
        print(f"  planning_time_s  : mean={active.mean():.2f}s  max={pt.max():.2f}s")
    if pk.size: print(f"  pick_attempts    : total={int(pk.sum())}  max_per_step={int(pk.max())}")
    print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python episode_logger.py <path/to/steps.csv>")
        sys.exit(1)
    _analyze(sys.argv[1])
