"""
canonical_trajectory.py -- fixed joint-space trajectory for the UR3 pick-and-place cell.

The trajectory is solved once at startup via cuRobo IK and cached to a JSON
file. On all subsequent runs the cached solution is loaded directly, so planning
is bypassed entirely and every episode follows the identical path.

Waypoint sequence
-----------------
HOME  ->  OVER_CONV  ->  APPROACH  ->  [per-episode DESCEND]
->  LIFT  ->  NEUTRAL  ->  PLACE_ABOVE  ->  [per-episode PLACE_DESCEND]
->  PLACE_ABOVE  ->  NEUTRAL  ->  HOME

The per-episode waypoints (DESCEND, LIFT, PLACE_DESCEND) are computed at
runtime from cube_z + fixed offsets, but use the same joint-space interpolation.
All other waypoints are identical across episodes.

Usage
-----
    from canonical_trajectory import CanonicalTrajectory
    traj = CanonicalTrajectory(planner, sim_app)
    traj.solve_and_cache()          # once at startup (no-op if cache exists)
    segments = traj.get_segments(cube_pos)   # list of (name, np.ndarray [N,6])
"""

import json
import math
import os
import threading
from pathlib import Path

import numpy as np

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_traj_cache.json")

# ---------------------------------------------------------------------------
# Fixed Cartesian targets (world frame, robot base at table level)
# ---------------------------------------------------------------------------

# Heights are conservative: APPROACH is above the tallest possible workpiece (80mm)
# plus a 70mm clearance margin.
WAYPOINTS_CARTESIAN = {
    "home":        np.array([ 0.40,  0.00,  0.25]),
    "over_conv":   np.array([ 0.40,  0.00,  0.22]),   # directly above conveyor centre
    "approach":    np.array([ 0.40,  0.00,  0.18]),   # 180mm -- clears 80mm box + margin
    "neutral":     np.array([ 0.20, -0.17,  0.28]),   # mid-arc, avoids all obstacles
    "place_above": np.array([ 0.00, -0.35,  0.22]),   # above bin
}

# Per-episode z offsets (relative to cube_z / bin_z)
PRE_GRASP_Z_OFFSET  =  0.15   # approach height above cube top
ATTACH_Z_OFFSET     =  0.03   # distance above cube centre for suction contact
LIFT_Z_OFFSET       =  0.15   # lift height above cube top after grasp
PLACE_Z_OFFSET      =  0.15   # descent target above bin floor
BIN_Z               =  0.10   # bin floor world z

DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])   # tool-pointing-down quaternion

UR3_VEL_LIMITS = np.array([3.14, 3.14, 3.14, 6.28, 6.28, 6.28])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enforce_wrist_down(traj: np.ndarray) -> np.ndarray:
    out = traj.copy()
    out[:, 4] = -math.pi / 2
    out[:, 3] = -math.pi / 2 - out[:, 1] - out[:, 2]
    return out


def _velocity_limit_traj(traj: np.ndarray, dt: float = 0.02) -> np.ndarray:
    out = [traj[0]]
    for i in range(1, len(traj)):
        prev, cur = out[-1], traj[i]
        delta = cur - prev
        steps = int(np.ceil(np.max(np.abs(delta) / (UR3_VEL_LIMITS * dt))))
        steps = max(steps, 1)
        for k in range(1, steps + 1):
            out.append(prev + delta * k / steps)
    return np.array(out)


def _interpolate_joint(q_start: np.ndarray, q_end: np.ndarray,
                       n: int = 50) -> np.ndarray:
    """Linear joint-space interpolation with wrist-down constraint and velocity limiting."""
    raw = np.stack([q_start + (q_end - q_start) * t / (n - 1) for t in range(n)])
    raw = _enforce_wrist_down(raw)
    return _velocity_limit_traj(raw)


# ---------------------------------------------------------------------------
# CanonicalTrajectory
# ---------------------------------------------------------------------------

class CanonicalTrajectory:
    """
    Manages the fixed joint-space trajectory.

    Parameters
    ----------
    planner : CuroboPlanner
        The cuRobo planner from ur3_pick_and_place.py. Used only during
        the one-time solve step.
    sim_app : SimulationApp
        Passed through to the planner's render keepalive thread.
    cache_file : str
        Path to the JSON cache. Defaults to canonical_traj_cache.json next
        to this file.
    """

    def __init__(self, planner, sim_app, cache_file: str = CACHE_FILE,
                 home_joints: np.ndarray = None):
        self._planner    = planner
        self._sim_app    = sim_app
        self._cache_file = cache_file
        self._waypoints  = {}   # name -> np.ndarray(6,)
        self._home_joints = (home_joints.copy() if home_joints is not None
                             else np.array([-0.284, -1.191, 1.410, -1.790, -1.5708, 0.0]))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_and_cache(self) -> bool:
        """
        Solve IK for all fixed waypoints via cuRobo and write the result to
        cache. If the cache already exists and is valid, load it instead.

        Returns True if the cache is ready to use.
        """
        if self._load_cache():
            return True
        print("[CanonicalTraj] No valid cache -- solving IK for canonical waypoints...")
        return self._solve_all()

    def get_segments(self, cube_pos: np.ndarray, cube_z_top: float = None) -> list:
        """
        Build the full episode trajectory as a list of (segment_name, joint_array).

        Each joint_array has shape [N, 6] and is ready to feed directly to
        ArticulationAction joint_positions.

        Parameters
        ----------
        cube_pos : np.ndarray (3,)
            World position of the workpiece centre (with perception noise applied).
        cube_z_top : float, optional
            Top surface z of the workpiece. If None, estimated from cube_pos[2].
        """
        if not self._waypoints:
            raise RuntimeError("Call solve_and_cache() before get_segments().")

        if cube_z_top is None:
            cube_z_top = cube_pos[2]   # assume pos is at centre, half-height above table

        q = self._waypoints   # shorthand

        # Per-episode Cartesian targets
        pick_approach_pos  = np.array([cube_pos[0], cube_pos[1],
                                       cube_z_top + PRE_GRASP_Z_OFFSET])
        pick_descend_pos   = np.array([cube_pos[0], cube_pos[1],
                                       cube_pos[2] + ATTACH_Z_OFFSET])
        lift_pos           = np.array([cube_pos[0], cube_pos[1],
                                       cube_z_top + LIFT_Z_OFFSET])
        place_descend_pos  = np.array([ 0.00, -0.35, BIN_Z + PLACE_Z_OFFSET])

        # Solve per-episode IK
        q_pick_app  = self._ik_once(pick_approach_pos,  seed=q["over_conv"])
        q_pick_desc = self._ik_once(pick_descend_pos,   seed=q_pick_app if q_pick_app is not None else q["over_conv"])
        q_lift      = self._ik_once(lift_pos,           seed=q_pick_desc if q_pick_desc is not None else q["over_conv"])
        q_place_desc= self._ik_once(place_descend_pos,  seed=q["place_above"])

        if any(x is None for x in [q_pick_app, q_pick_desc, q_lift, q_place_desc]):
            print("[CanonicalTraj] Per-episode IK failed -- falling back to cuRobo planner.")
            return None

        segments = [
            ("home_to_over_conv",   _interpolate_joint(q["home"],        q["over_conv"])),
            ("over_conv_to_approach", _interpolate_joint(q["over_conv"], q_pick_app)),
            ("approach_to_descend", _interpolate_joint(q_pick_app,       q_pick_desc)),
            # grasp happens here -- caller handles ATTACH dwell
            ("lift",                _interpolate_joint(q_pick_desc,      q_lift)),
            ("lift_to_neutral",     _interpolate_joint(q_lift,           q["neutral"])),
            ("neutral_to_place",    _interpolate_joint(q["neutral"],     q["place_above"])),
            ("place_descend",       _interpolate_joint(q["place_above"], q_place_desc)),
            # release happens here -- caller handles DETACH dwell
            ("retract_to_place_above", _interpolate_joint(q_place_desc,  q["place_above"])),
            ("place_to_neutral",    _interpolate_joint(q["place_above"], q["neutral"])),
            ("neutral_to_home",     _interpolate_joint(q["neutral"],     q["home"])),
        ]
        return segments

    def home_joints(self) -> np.ndarray:
        return self._waypoints.get("home", None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ik_once(self, target_pos: np.ndarray, seed: np.ndarray) -> np.ndarray | None:
        """
        Solve IK for a single Cartesian target via cuRobo, seeded from `seed`.
        Returns the joint config or None on failure.
        """
        import torch
        from curobo.types.math import Pose
        from curobo.types.state import JointState

        ta = self._planner.tensor_args

        # Sanitize seed: if cuRobo considers it in world collision, fall back to home.
        seed_clean = seed.copy()
        try:
            js_check = JointState.from_position(
                ta.to_device(torch.tensor(seed_clean, dtype=torch.float32).unsqueeze(0))
            )
            if self._planner.motion_gen.check_start_state(js_check):
                print("[CanonicalTraj] Seed in collision, falling back to home_joints.")
                seed_clean = self._home_joints.copy()
        except Exception:
            pass

        js = JointState.from_position(
            ta.to_device(torch.tensor(seed_clean, dtype=torch.float32).unsqueeze(0))
        )
        goal_pose = Pose(
            position=ta.to_device(
                torch.tensor(target_pos, dtype=torch.float32).unsqueeze(0)),
            quaternion=ta.to_device(
                torch.tensor(DOWN_QUAT, dtype=torch.float32).unsqueeze(0)),
        )
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        cfg = MotionGenPlanConfig(max_attempts=24, enable_graph=True,
                                  enable_opt=True, partial_ik_opt=True)
        # Also try with start-state-free in case minor penetration is reported
        cfg_free = MotionGenPlanConfig(max_attempts=24, enable_graph=True,
                                       enable_opt=True, partial_ik_opt=True)

        result_box, exc_box = [None], [None]
        def _plan():
            try:
                result_box[0] = self._planner.motion_gen.plan_single(js, goal_pose, cfg)
            except Exception as e:
                exc_box[0] = e
        t = threading.Thread(target=_plan, daemon=True)
        t.start()
        while t.is_alive():
            if self._sim_app:
                self._sim_app.update()
            t.join(timeout=0.033)
        if exc_box[0]:
            print(f"[CanonicalTraj] IK exception: {exc_box[0]}")
            return None

        r = result_box[0]
        if not r.success.item():
            # Retry: sometimes warmup wasn't enough; try once more from home seed
            if "WORLD_COLLISION" in str(r.status) and not np.allclose(seed_clean, self._home_joints):
                print(f"[CanonicalTraj] Retrying from home seed due to {r.status}...")
                js2 = JointState.from_position(
                    ta.to_device(torch.tensor(self._home_joints, dtype=torch.float32).unsqueeze(0))
                )
                result_box2, exc_box2 = [None], [None]
                def _plan2():
                    try: result_box2[0] = self._planner.motion_gen.plan_single(js2, goal_pose, cfg)
                    except Exception as e: exc_box2[0] = e
                t2 = threading.Thread(target=_plan2, daemon=True)
                t2.start()
                while t2.is_alive():
                    if self._sim_app: self._sim_app.update()
                    t2.join(timeout=0.033)
                if not exc_box2[0] and result_box2[0] and result_box2[0].success.item():
                    r = result_box2[0]
                else:
                    print(f"[CanonicalTraj] IK failed for target {np.round(target_pos, 3)}: {r.status}")
                    return None
            else:
                print(f"[CanonicalTraj] IK failed for target {np.round(target_pos, 3)}: {r.status}")
                return None

        traj_np = r.get_interpolated_plan().position.cpu().numpy()
        q = traj_np[-1].copy()   # terminal joint config = IK solution
        q[4] = -math.pi / 2
        q[3] = -math.pi / 2 - q[1] - q[2]
        return q

    def _solve_all(self) -> bool:
        """Solve IK for all fixed waypoints and cache to disk."""
        from curobo.types.state import JointState
        import torch

        home_j = self._home_joints.copy()

        # Warmup resets cuRobo's internal CUDA graphs and collision cache.
        # Without this the first plan call often fails with START_STATE_WORLD_COLLISION.
        try:
            print("[CanonicalTraj] Warming up MotionGen...")
            self._planner.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        except Exception as e:
            print(f"[CanonicalTraj] Warmup warning (non-fatal): {e}")

        self._waypoints["home"] = home_j
        seed = home_j

        order = ["over_conv", "approach", "neutral", "place_above"]
        for name in order:
            pos = WAYPOINTS_CARTESIAN[name]
            q = self._ik_once(pos, seed=seed)
            if q is None:
                # Retry from RETRACT_JOINTS -- the chained seed may be in collision
                print(f"[CanonicalTraj] Retrying '{name}' from home seed...")
                q = self._ik_once(pos, seed=self._home_joints)
            if q is None:
                print(f"[CanonicalTraj] FAILED to solve IK for '{name}'. Cache aborted.")
                self._waypoints = {}
                return False
            self._waypoints[name] = q
            seed = q
            T_fk = self._verify_fk(q)
            print(f"[CanonicalTraj] '{name}': j={np.round(q,4)}  FK={np.round(T_fk,4)}")

        self._write_cache()
        print(f"[CanonicalTraj] Cache written to {self._cache_file}")
        return True

    def _verify_fk(self, q: np.ndarray) -> np.ndarray:
        """Use cuRobo's FK to get world position of the solved config."""
        try:
            import torch
            from curobo.types.state import JointState
            js = JointState.from_position(
                self._planner.tensor_args.to_device(
                    torch.tensor(q, dtype=torch.float32).unsqueeze(0)
                )
            )
            fk = self._planner.motion_gen.compute_kinematics(js)
            pos = fk.ee_position.cpu().numpy().flatten()
            return pos
        except Exception:
            return np.zeros(3)

    def _write_cache(self):
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        data = {k: v.tolist() for k, v in self._waypoints.items()}
        with open(self._cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_cache(self) -> bool:
        if not os.path.exists(self._cache_file):
            return False
        try:
            with open(self._cache_file) as f:
                data = json.load(f)
            required = {"home", "over_conv", "approach", "neutral", "place_above"}
            if not required.issubset(data.keys()):
                print("[CanonicalTraj] Cache incomplete, re-solving.")
                return False
            self._waypoints = {k: np.array(v) for k, v in data.items()}
            print(f"[CanonicalTraj] Loaded cached waypoints from {self._cache_file}")
            return True
        except Exception as e:
            print(f"[CanonicalTraj] Cache load failed ({e}), re-solving.")
            return False
