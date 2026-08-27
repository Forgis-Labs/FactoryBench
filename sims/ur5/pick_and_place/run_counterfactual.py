"""
Counterfactual data generation for UR5 pick-and-place.

For each episode:
  1. Runs a baseline (no events)
  2. Replays the same episode with a random event injected

Output structure::

    counterfactual_data/
        episode_0000/
            baseline/steps.csv, episodes.csv
            counterfactual/steps.csv, episodes.csv, event.json
        episode_0001/
            ...

Usage:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur5/pick_and_place/run_counterfactual.py \\
        --episodes 100 --seed 0 [--headless]
"""

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).parent.resolve())
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_FACTORYBENCH_DIR = str(Path(__file__).parent.parent.parent.resolve())
if _FACTORYBENCH_DIR not in sys.path:
    sys.path.insert(0, _FACTORYBENCH_DIR)

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Counterfactual data generation")
_parser.add_argument("--headless", action="store_true")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--episodes", type=int, default=10)
_parser.add_argument("--output", type=str, default="counterfactual_data")
_parser.add_argument("--max_steps", type=int, default=1500)
_args = _parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "width": 1280, "height": 720,
    "headless": _args.headless,
})

# ---------------------------------------------------------------------------
# Deferred imports (after SimulationApp), import shared functions from run.py.
# SimulationApp must exist before this import because run.py's deferred
# imports (omni.usd, pxr, etc.) require the Omniverse runtime.
# ---------------------------------------------------------------------------
import omni.usd
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.controllers import PickPlaceController

from episode_logger import EpisodeLogger, collect_sensors
from event_injection import EventScheduler, SimContext, BUILTIN_APPLICATORS
from counterfactual import CounterfactualRunner
from counterfactual.episode_state import EpisodeState

# Import shared code from run.py (safe now, SimulationApp already exists,
# and the CLI/SimulationApp block is guarded by __name__ == "__main__")
from run import (
    CFG, SIM_DT, ROBOT_PRIM, EEF_PRIM, HOME_JOINTS,
    CUBE_PRIM, BIN_POSITION, GRIPPER_CLOSE_DEG_BASE, GRIPPER_CLOSE_DEG_MAX, GRIPPER_TCP_Z,
    ROBOTIQ_PRIM, ROBOTIQ_BASE, UR5_USD, ROBOTIQ_USD,
    CUBE_MASS_RANGE, _grip_strength_for_mass,
    build_workcell, setup_gripper, spawn_workpiece,
    sample_workpiece_dims, randomize_cube_pose, randomize_cube_physics,
    update_gripper_for_episode, apply_perception_noise, yaw_to_quat,
    UR5RMPFlowController,
    setup_hud, update_hud,
    setup_event_indicator, update_event_indicator)

TABLE_HEIGHT = 0.10
_TASK_DIR = Path(__file__).parent.resolve()

# Phase boundaries (same as run.py)
_EVENTS_DT = [0.008, 0.005, 0.04, 0.008, 0.008, 0.004, 0.008, 0.10, 0.016, 0.012]
_PHASE_BOUNDS = [0]
for dt in _EVENTS_DT:
    _PHASE_BOUNDS.append(_PHASE_BOUNDS[-1] + int(round(1.0 / dt)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    rng = np.random.RandomState(_args.seed)

    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane()
    build_workcell(stage)

    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + UR5_USD, ROBOT_PRIM)
    robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="ur5"))
    add_reference_to_stage(ROBOTIQ_USD, ROBOTIQ_PRIM)
    setup_gripper(stage)

    cur_dims = sample_workpiece_dims(rng)
    cur_spawn, cur_yaw = randomize_cube_pose(rng)
    cur_spawn[2] = TABLE_HEIGHT + cur_dims[2] / 2.0 + 0.005
    cube = spawn_workpiece(world, rng, cur_spawn, cur_dims)

    world.reset()

    n_dof = robot.num_dof
    home_full = np.zeros(n_dof)
    home_full[:len(HOME_JOINTS)] = HOME_JOINTS
    robot.set_joints_default_state(positions=home_full)

    gripper = ParallelGripper(
        end_effector_prim_path=ROBOTIQ_BASE + "/robotiq_arg2f_base_link",
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([np.radians(GRIPPER_CLOSE_DEG_MAX)]),
        action_deltas=None,
        use_mimic_joints=True)
    gripper.initialize(
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names)
    fj_idx = gripper.joint_dof_indicies[0]

    _GRIP_INDICES = np.array([6, 7, 8, 9, 10, 11])
    _init_kp, _init_kd, _ = _grip_strength_for_mass(np.mean(CUBE_MASS_RANGE))
    robot._articulation_view.set_gains(
        kps=np.array([[_init_kp] * 6]), kds=np.array([[_init_kd] * 6]),
        joint_indices=_GRIP_INDICES
    )
    gripper.open()
    for _ in range(20):
        world.step(render=True)

    rmp_controller = UR5RMPFlowController(
        name="ur5_rmpflow", robot_articulation=robot, physics_dt=SIM_DT
    )
    pick_place = PickPlaceController(
        name="ur5_pick_place",
        cspace_controller=rmp_controller,
        gripper=gripper,
        end_effector_initial_height=0.42,
        events_dt=_EVENTS_DT)

    # Gripper proxy for collect_sensors
    class _GripperProxy:
        def __init__(self, pg):
            self._pg = pg
        @property
        def attached(self):
            jp = self._pg.get_joint_positions()
            return jp is not None and float(jp[0]) > np.radians(GRIPPER_CLOSE_DEG_MAX) * 0.3
        @property
        def is_closed(self):
            return self.attached
    gripper_proxy = _GripperProxy(gripper)

    # HUD + event indicator, override run.py's _args.headless so the
    # HUD functions know whether to create windows.
    import run as _run_module
    _run_module._args.headless = _args.headless
    setup_hud()
    setup_event_indicator()

    logger = EpisodeLogger(str(_TASK_DIR / "logs" / "counterfactual_tmp"))
    logger.init_sensors(world, robot, cube, stage,
                        flange_prim_path=EEF_PRIM, sim_dt=SIM_DT)

    # Shared mutable state
    state = {
        "cur_mass": 0.0, "cur_friction": 0.0, "cur_dims": cur_dims,
        "cur_spawn": cur_spawn, "cur_yaw": cur_yaw,
        "pick_pos": None, "place_pos": None, "ee_orient": None,
        "ep_step": 0, "sim_time": 0.0, "episode": 0,
        "phase": 0, "done": False, "success": False, "reason": "",
    }

    PHASE_NAMES = ["above_pick", "descend", "settle", "close",
                   "lift", "move_xy", "lower", "open",
                   "retract", "return"]

    def compute_targets():
        perceived = apply_perception_noise(state["cur_spawn"].copy(), rng)
        pick = perceived.copy()
        pick[2] = perceived[2] + GRIPPER_TCP_Z
        place = BIN_POSITION.copy()
        place[2] = BIN_POSITION[2] + GRIPPER_TCP_Z + 0.04
        state["ee_orient"] = euler_angles_to_quat(
            np.array([0, np.pi, state["cur_yaw"]]))
        state["pick_pos"] = pick
        state["place_pos"] = place

    def reset_fn(ep_state: EpisodeState | None, inject_events: bool):
        """Reset episode.  If ep_state is None, sample fresh randomness."""
        nonlocal cube

        if ep_state is not None:
            # Replay: restore RNG and use saved state
            rng.set_state(ep_state.rng_state)
            state["cur_dims"] = ep_state.cube_dims.copy()
            state["cur_spawn"] = ep_state.cube_spawn.copy()
            state["cur_yaw"] = ep_state.cube_yaw
        else:
            # Fresh: sample new randomness (save state BEFORE sampling)
            state["cur_dims"] = sample_workpiece_dims(rng)
            state["cur_spawn"], state["cur_yaw"] = randomize_cube_pose(rng)
            state["cur_spawn"][2] = TABLE_HEIGHT + state["cur_dims"][2] / 2.0 + 0.005

        cube = spawn_workpiece(world, rng, state["cur_spawn"],
                               state["cur_dims"], existing_cube=cube)

        if ep_state is not None:
            # For replay, set mass/friction from saved state
            try:
                mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath(CUBE_PRIM))
                mass_api.GetMassAttr().Set(ep_state.cube_mass)
                mat_prim = stage.GetPrimAtPath(CUBE_PRIM + "/PhysicsMaterial")
                if mat_prim.IsValid():
                    UsdPhysics.MaterialAPI(mat_prim).GetStaticFrictionAttr().Set(ep_state.cube_friction)
                    UsdPhysics.MaterialAPI(mat_prim).GetDynamicFrictionAttr().Set(ep_state.cube_friction * 0.85)
                    UsdPhysics.MaterialAPI(mat_prim).GetRestitutionAttr().Set(ep_state.cube_restitution)
            except Exception:
                pass
            state["cur_mass"] = ep_state.cube_mass
            state["cur_friction"] = ep_state.cube_friction
        else:
            state["cur_mass"], state["cur_friction"] = randomize_cube_physics(cube, rng)

        home = np.zeros(n_dof)
        home[:len(HOME_JOINTS)] = HOME_JOINTS
        robot.set_joints_default_state(positions=home)
        cube.set_default_state(position=state["cur_spawn"],
                               orientation=yaw_to_quat(state["cur_yaw"]))

        setup_gripper(stage)
        world.reset()

        robot.set_solver_position_iteration_count(64)
        robot.set_solver_velocity_iteration_count(64)
        rmp_controller.reset()
        pick_place.reset(end_effector_initial_height=0.42)
        robot.set_joint_positions(home)
        robot.set_joint_velocities(np.zeros(n_dof))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        gripper.post_reset()

        _, _cur_close_rad = update_gripper_for_episode(
            stage, rng, robot, state["cur_mass"], state["cur_friction"])
        update_hud(state["cur_mass"], state["cur_friction"])
        gripper.open()
        for _ in range(30):
            world.step(render=True)

        logger.reset_diff_state()
        state["ep_step"] = 0
        state["sim_time"] = 0.0
        state["done"] = False
        state["success"] = False
        state["reason"] = ""
        compute_targets()

    def step_fn() -> dict:
        """Advance one sim step, return sensor row."""
        world.step(render=True)
        state["ep_step"] += 1
        state["sim_time"] += SIM_DT

        current_joints = robot.get_joint_positions()
        action = pick_place.forward(
            picking_position=state["pick_pos"],
            placing_position=state["place_pos"],
            current_joint_positions=current_joints,
            end_effector_orientation=state["ee_orient"])
        robot.apply_action(action)
        state["action"] = action

        phase = min(pick_place.get_current_event(), 9)
        state["phase"] = phase
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

        # Determine current Cartesian target based on controller phase
        if phase < 4:
            _ee_target = state["pick_pos"]
        else:
            _ee_target = state["place_pos"]

        # Read actual gripper finger position
        _all_jpos = robot.get_joint_positions()
        _gripper_actual = float(_all_jpos[fj_idx]) if _all_jpos is not None and len(_all_jpos) > fj_idx else None

        sensor_row = collect_sensors(
            logger, state["episode"], state["ep_step"], state["sim_time"],
            phase_name, robot, cube, stage, gripper_proxy,
            state["cur_mass"], state["cur_friction"],
            cur_restitution=0.0,
            cur_dims=state["cur_dims"],
            plan_attempts=0, plan_time_last=0.0, pick_attempts=1,
            planned_action=action, task_phase=phase_name,
            joint_noise_std=float(CFG["noise"]["joint_std"]),
            ee_target_pos=_ee_target,
            ee_target_quat=state["ee_orient"],
            gripper_cmd_rad=grip_target,
            gripper_pos_rad=_gripper_actual,
            controller_phase=phase)
        return sensor_row

    def is_done_fn() -> tuple:
        """Check if episode is finished."""
        cube_pos, _ = cube.get_world_pose()
        cube_pos = np.asarray(cube_pos)

        if cube_pos[2] < -0.15:
            state["done"] = True
            state["success"] = False
            state["reason"] = "dropped"
        elif state["sim_time"] > 60.0:
            state["done"] = True
            state["success"] = False
            state["reason"] = "timeout"
        elif pick_place.is_done():
            state["done"] = True
            state["success"] = True
            state["reason"] = "placed"

        return (state["done"], state["success"], state["reason"])

    def get_state_fn() -> EpisodeState:
        """Capture current episode state for replay."""
        return EpisodeState(
            rng_state=rng.get_state(),
            cube_dims=state["cur_dims"].copy(),
            cube_spawn=state["cur_spawn"].copy(),
            cube_yaw=state["cur_yaw"],
            cube_mass=state["cur_mass"],
            cube_friction=state["cur_friction"],
            cube_restitution=getattr(randomize_cube_physics, "_last_restitution", 0.0))

    def event_step_fn(ep_step: int, sensor_row: dict):
        """Run event scheduler for one step."""
        from episode_logger import JOINT_NAMES as _JNAMES
        ctx = SimContext(
            stage=stage,
            sensor_data=sensor_row,
            cube_prim_path=CUBE_PRIM,
            robot_prim_path=ROBOT_PRIM,
            joint_names=_JNAMES,
            sim_dt=SIM_DT,
            episode_step=ep_step,
            state_machine=PHASE_NAMES[state["phase"]],
            extra={"robot": robot, "action": state.get("action")})
        active = runner._scheduler.step(ep_step, ctx)
        update_event_indicator(active if active else None)
        return active

    # Create the runner
    output_dir = _TASK_DIR / _args.output
    runner = CounterfactualRunner(
        output_dir=output_dir,
        events_json_path=str(Path(__file__).parent.parent.parent / "events.json"),
        task_name="pick_and_place",
        applicators=BUILTIN_APPLICATORS,
        seed=_args.seed,
        phase_boundaries=_PHASE_BOUNDS)

    # Capture state BEFORE first reset for the runner
    print(f"[Counterfactual] Starting {_args.episodes} episode pairs → {output_dir}")
    runner.run(
        num_episodes=_args.episodes,
        reset_fn=reset_fn,
        step_fn=step_fn,
        is_done_fn=is_done_fn,
        get_state_fn=get_state_fn,
        event_step_fn=event_step_fn,
        max_steps=_args.max_steps)

    simulation_app.close()


if __name__ == "__main__":
    main()
