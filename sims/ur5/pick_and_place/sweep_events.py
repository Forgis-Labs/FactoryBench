#!/usr/bin/env python3
"""
sweep_events.py, sweep event variable ranges via counterfactual runs.

For each event type and parameter range, runs paired baseline/counterfactual
episodes to measure the counterfactual failure rate: what percentage of
baseline successes become failures after event injection.

Only one event type is injected per cell, the sweep isolates each event's
impact independently.

With --workers N, launches N parallel Isaac Sim processes.

Usage:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur5/pick_and_place/sweep_events.py \
        [--episodes 50] [--seed 42] [--workers 4]
"""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).parent.resolve())
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_FACTORYBENCH_DIR = str(Path(__file__).parent.parent.parent.resolve())
if _FACTORYBENCH_DIR not in sys.path:
    sys.path.insert(0, _FACTORYBENCH_DIR)

import numpy as np

# ── Per-episode log columns ───────────────────────────────────────────
_EP_LOG_COLS = [
    "cell_idx", "episode", "run_type",
    "event_id", "event_name", "param_name", "param_lo", "param_hi",
    "cube_mass_kg", "cube_friction_coeff", "cube_restitution_coeff",
    "cube_width_m", "cube_depth_m", "cube_height_m",
    "cube_spawn_x_m", "cube_spawn_y_m", "cube_spawn_z_m",
    "cube_final_x_m", "cube_final_y_m", "cube_final_z_m",
    "sim_time_s", "success", "reason",
]

# ── Sweep grid definition ────────────────────────────────────────────────
# Each entry: (event_id, event_name, param_name, param_ranges, extra_kwargs)
# param_ranges is a list of [lo, hi] for the swept variable.
# For events with no continuous variable, param_ranges has one [None, None].
# extra_kwargs is a dict of additional parameters (e.g. {"object": ...} for
# collision entries that lock the colliding object).
_COLLISION_OBJECTS = [
    "unknown_debris", "adjacent_part", "tool",
    "bolt", "pipe_section", "cardboard_box",
    "metal_plate", "gear", "bottle", "wood_block",
]
# Impact impulse ranges (kg·m/s), weakest to strongest.
_COLLISION_IMPULSE_RANGES = [
    [6.0, 12.0],    # ~25% CF rate range
]

SWEEP_GRID = [
    (6, "Payload Addition", "x", [
        [0.35, 0.70],
    ], {}),
    (11, "Friction Decrease", "delta", [
        [0.45, 0.60],
    ], {}),
    (12, "Motor Miscommutation", "phase_offset_deg", [
        [7.0, 12.0],
    ], {}),
    (13, "Gripper Activation Failure", None, [
        [None, None],
    ], {}),
    (14, "Gripper Release", None, [
        [None, None],
    ], {}),
]

# Flatten into indexed cells.
# Collision entries come first, sweeping all impulse ranges for each object
# before moving to the next object (weak → strong per object).
FULL_GRID = []
for obj in _COLLISION_OBJECTS:
    for r in _COLLISION_IMPULSE_RANGES:
        FULL_GRID.append((len(FULL_GRID), 16, f"Collision ({obj})",
                          "impact_impulse", r[0], r[1], {"object": obj}))

for event_id, event_name, param_name, ranges, extra in SWEEP_GRID:
    for r in ranges:
        FULL_GRID.append((len(FULL_GRID), event_id, event_name, param_name, r[0], r[1], extra))

TASK_DIR = Path(__file__).parent.resolve()
SWEEP_DIR = TASK_DIR / "sweeps"
PYTHON_SH = Path("/opt/IsaacSim/python.sh").resolve()

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sweep event variable ranges")
parser.add_argument("--episodes", type=int, default=50,
                    help="Counterfactual episode pairs per grid cell (default: 50)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed")
parser.add_argument("--workers", type=int, default=1,
                    help="Parallel Isaac Sim processes (default: 1)")
parser.add_argument("--max_steps", type=int, default=1500,
                    help="Max sim steps per episode (default: 1500)")
parser.add_argument("--event", type=int, default=None,
                    help="Only sweep this event ID (e.g. --event 11 for Friction Decrease)")
parser.add_argument("--headless", action="store_true", default=True,
                    help="Run headless (default: True)")
parser.add_argument("--visual", action="store_true",
                    help="Run with rendering (forces --workers 1)")
# Internal worker args
parser.add_argument("--_worker_cells", type=str, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("--_worker_outfile", type=str, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("--_worker_logfile", type=str, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("--_worker_headless", action="store_true", default=False,
                    help=argparse.SUPPRESS)
args = parser.parse_args()

if args.visual:
    args.headless = False
    args.workers = 1

# Filter grid to a single event if --event is specified
if args.event is not None:
    FULL_GRID = [c for c in FULL_GRID if c[1] == args.event]
    if not FULL_GRID:
        print(f"Error: no grid cells match --event {args.event}")
        print(f"Available event IDs: {sorted(set(e[0] for e in SWEEP_GRID))}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCHER MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_launcher():
    n_cells = len(FULL_GRID)
    n_workers = min(args.workers, n_cells)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = SWEEP_DIR / f"events_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Print only the events that will actually be swept
    active_event_ids = set(c[1] for c in FULL_GRID)
    print(f"Sweep grid: {n_cells} cells, {args.episodes} pairs each")
    print(f"Output: {run_dir}")
    for event_id, event_name, param_name, ranges, _extra in SWEEP_GRID:
        if event_id not in active_event_ids:
            continue
        if param_name:
            range_str = ", ".join(f"[{r[0]}-{r[1]}]" for r in ranges)
            print(f"  Event {event_id} ({event_name}): {param_name} = {range_str}")
        else:
            print(f"  Event {event_id} ({event_name}): no continuous param")
    print(f"Launching {n_workers} worker(s)...\n", flush=True)

    # Split cells across workers
    chunks = [[] for _ in range(n_workers)]
    for i in range(n_cells):
        chunks[i % n_workers].append(i)

    # Launch workers
    procs = []
    outfiles = []
    logfiles = []
    for w in range(n_workers):
        outfile = tempfile.NamedTemporaryFile(
            suffix=".json", prefix=f"sweep_w{w}_", delete=False)
        outfile.close()
        outfiles.append(outfile.name)

        logfile = tempfile.NamedTemporaryFile(
            suffix=".csv", prefix=f"sweep_log_w{w}_", delete=False)
        logfile.close()
        logfiles.append(logfile.name)

        cmd = [
            str(PYTHON_SH), __file__,
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--max_steps", str(args.max_steps),
            "--_worker_cells", json.dumps(chunks[w]),
            "--_worker_outfile", outfile.name,
            "--_worker_logfile", logfile.name,
        ]
        if args.event is not None:
            cmd += ["--event", str(args.event)]
        if args.headless:
            cmd.append("--_worker_headless")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        procs.append(proc)

    # Stream worker output
    import threading
    import re

    _keep = re.compile(
        r"^Starting |^Ready |^\[Cell |^  baseline|^  counterfactual|^  => "
    )

    def _stream(proc, prefix):
        if proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text and _keep.search(text):
                print(f"  {prefix} {text}", flush=True)

    threads = []
    for w, proc in enumerate(procs):
        t = threading.Thread(target=_stream, args=(proc, f"[w{w}]"), daemon=True)
        t.start()
        threads.append(t)

    t0 = time.time()
    for proc in procs:
        proc.wait()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    # Collect results
    all_results = [None] * n_cells
    for outfile in outfiles:
        try:
            with open(outfile) as f:
                worker_results = json.load(f)
            for r in worker_results:
                all_results[r["cell_idx"]] = r
        except Exception:
            pass
        os.unlink(outfile)

    # Merge per-worker episode logs into a single CSV
    episodes_csv = run_dir / "sweep_episodes.csv"
    with open(episodes_csv, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=_EP_LOG_COLS)
        writer.writeheader()
        for logfile in logfiles:
            try:
                with open(logfile, newline="") as lf:
                    reader = csv.DictReader(lf)
                    for row in reader:
                        writer.writerow(row)
            except Exception:
                pass
            os.unlink(logfile)
    print(f"Episode log saved to {episodes_csv}")

    results = [r for r in all_results if r is not None]

    # Summary table
    print(f"\nCompleted in {elapsed:.1f}s total\n")
    print("=" * 100)
    print("COUNTERFACTUAL SWEEP RESULTS")
    print("=" * 100)

    header = (f"{'Event':>30} {'Param Range':>18} "
              f"{'BL Succ':>10} {'CF Fail':>10} {'CF Rate':>10} "
              f"{'Pairs':>8}")
    print(header)
    print("-" * 100)

    for r in results:
        event_str = f"[{r['event_id']}] {r['event_name']}"
        if r["param_name"]:
            param_str = f"[{r['param_lo']:.2f}-{r['param_hi']:.2f}]"
        else:
            param_str = ", "
        print(f"{event_str:>30} {param_str:>18} "
              f"{r['baseline_successes']:>10d} {r['counterfactual_failures']:>10d} "
              f"{r['counterfactual_failure_rate']:>9.1%} "
              f"{r['total_pairs']:>8d}")

    print("-" * 100)
    print(f"Total cells evaluated: {len(results)}")

    # Write CSV
    results_csv = run_dir / "sweep_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "event_name", "param_name", "param_lo", "param_hi",
            "total_pairs", "baseline_successes", "baseline_failures",
            "counterfactual_successes", "counterfactual_failures",
            "counterfactual_failure_rate",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\nResults saved to {results_csv}")


# ═══════════════════════════════════════════════════════════════════════════
# WORKER MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_worker():
    cell_indices = json.loads(args._worker_cells)
    outfile = args._worker_outfile
    _headless = args._worker_headless

    _real_stdout = sys.stdout
    _real_stderr = sys.stderr

    def _print(*a, **kw):
        kw["file"] = _real_stdout
        kw["flush"] = True
        print(*a, **kw)

    _print(f"Starting ({len(cell_indices)} cells)...")

    # Suppress Isaac Sim startup logs via environment before SimulationApp
    os.environ["CARB_LOG_LEVEL"] = "error"

    # Redirect OS-level fds to suppress C++ output during init
    _saved_fd1 = os.dup(1)
    _saved_fd2 = os.dup(2)
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 1)
    os.dup2(_devnull_fd, 2)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "width": 1280, "height": 720, "headless": _headless,
        "log_level": "error",
    })

    import carb
    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/log/fileLogLevel", "error")
    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    import logging
    logging.getLogger("omni").setLevel(logging.ERROR)

    # Restore OS-level fds so the window/display works
    os.dup2(_saved_fd1, 1)
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd1)
    os.close(_saved_fd2)
    os.close(_devnull_fd)

    if _headless:
        # Keep Python-level stdout/stderr suppressed for headless
        _devnull_f = open(os.devnull, "w")
        sys.stdout = _devnull_f
        sys.stderr = _devnull_f

    import omni.usd
    from pxr import UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.robot.manipulators.controllers import PickPlaceController

    from event_injection import EventScheduler, SimContext
    from event_injection.applicators.payload import PayloadAdditionApplicator
    from event_injection.applicators.friction import FrictionDecreaseApplicator
    from event_injection.applicators.motor import MotorMiscommutationApplicator
    from event_injection.applicators.gripper_failure import GripperActivationFailureApplicator
    from event_injection.applicators.gripper_release import GripperReleaseApplicator
    from event_injection.applicators.collision import CollisionApplicator
    from event_injection.applicators.brownout import JointPowerBrownoutApplicator

    import run as _run_module

    # Override run.py's dummy _args.headless so HUD and other
    # visual features work when the sweep runs in visual mode.
    _run_module._args.headless = _headless

    ROBOT_PRIM = _run_module.ROBOT_PRIM
    EEF_PRIM = _run_module.EEF_PRIM
    HOME_JOINTS = _run_module.HOME_JOINTS
    CUBE_PRIM = _run_module.CUBE_PRIM
    BIN_POSITION = _run_module.BIN_POSITION
    SIM_DT = _run_module.SIM_DT
    GRIPPER_TCP_Z = _run_module.GRIPPER_TCP_Z
    GRIPPER_CLOSE_DEG_BASE = _run_module.GRIPPER_CLOSE_DEG_BASE
    GRIPPER_CLOSE_DEG_MAX = _run_module.GRIPPER_CLOSE_DEG_MAX
    UR5_USD = _run_module.UR5_USD
    ROBOTIQ_USD = _run_module.ROBOTIQ_USD
    ROBOTIQ_PRIM = _run_module.ROBOTIQ_PRIM
    ROBOTIQ_BASE = _run_module.ROBOTIQ_BASE
    TABLE_HEIGHT = _run_module.TABLE_HEIGHT
    EEF_INITIAL_HEIGHT = _run_module.EEF_INITIAL_HEIGHT

    _EVENTS_DT = _run_module.EVENTS_DT
    _PHASE_BOUNDS = [0]
    for dt in _EVENTS_DT:
        _PHASE_BOUNDS.append(_PHASE_BOUNDS[-1] + int(round(1.0 / dt)))

    PHASE_NAMES = ["above_pick", "descend", "settle", "close",
                   "lift", "move_xy", "lower", "open",
                   "retract", "return"]

    t_init = time.time()

    rng = np.random.RandomState(args.seed)
    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    world.scene.add_default_ground_plane()
    _run_module.build_workcell(stage)
    _run_module._ensure_surface_friction(stage, "/World/cell/conv_belt", friction=0.6)
    _run_module._ensure_surface_friction(stage, "/World/cell/bin_floor", friction=0.8)
    _run_module._ensure_surface_friction(stage, "/World/cell/bin_stand_top", friction=0.6)

    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + UR5_USD, ROBOT_PRIM)
    robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="ur5"))

    add_reference_to_stage(ROBOTIQ_USD, ROBOTIQ_PRIM)
    _run_module.setup_gripper(stage)

    cur_dims = _run_module.sample_workpiece_dims(rng)
    cur_spawn, cur_yaw = _run_module.randomize_cube_pose(rng)
    cur_spawn[2] = TABLE_HEIGHT + cur_dims[2] / 2.0 + 0.005
    cube = _run_module.spawn_workpiece(world, rng, cur_spawn, cur_dims)

    world.reset()

    n_dof = robot.num_dof
    home_full = np.zeros(n_dof)
    home_full[:len(HOME_JOINTS)] = HOME_JOINTS
    robot.set_joints_default_state(positions=home_full)
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    robot.set_joint_positions(home_full)
    robot.set_joint_velocities(np.zeros(n_dof))

    for _ in range(10):
        world.step(render=not _headless)

    _CLOSE_RAD = np.radians(GRIPPER_CLOSE_DEG_MAX)
    gripper = ParallelGripper(
        end_effector_prim_path=ROBOTIQ_BASE + "/robotiq_arg2f_base_link",
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([_CLOSE_RAD]),
        action_deltas=None,
        use_mimic_joints=True)
    gripper.initialize(
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names)
    fj_idx = gripper.joint_dof_indicies[0]

    _GRIP_INDICES = np.array([6, 7, 8, 9, 10, 11])
    _init_kp, _init_kd, _ = _run_module._grip_strength_for_mass(
        np.mean(_run_module.CUBE_MASS_RANGE))
    robot._articulation_view.set_gains(
        kps=np.array([[_init_kp] * 6]), kds=np.array([[_init_kd] * 6]),
        joint_indices=_GRIP_INDICES
    )
    gripper.open()
    for _ in range(20):
        world.step(render=not _headless)

    rmp_controller = _run_module.UR5RMPFlowController(
        name="ur5_rmpflow", robot_articulation=robot, physics_dt=SIM_DT
    )

    # HUD overlays (visual mode only)
    if not _headless:
        _run_module.setup_hud()
        _run_module.setup_event_indicator()

    _print(f"Ready ({time.time() - t_init:.1f}s)")

    # ── Helper: build applicator with specific param range ────────────────
    def make_applicator(event_id, param_lo, param_hi, extra=None):
        if extra is None:
            extra = {}
        if event_id == 6:
            return PayloadAdditionApplicator(mass_delta_range=(param_lo, param_hi))
        elif event_id == 11:
            return FrictionDecreaseApplicator(fraction_range=(param_lo, param_hi))
        elif event_id == 12:
            return MotorMiscommutationApplicator(phase_offset_range=(param_lo, param_hi))
        elif event_id == 13:
            return GripperActivationFailureApplicator()
        elif event_id == 14:
            return GripperReleaseApplicator()
        elif event_id == 16:
            return CollisionApplicator(impulse_range=(param_lo, param_hi),
                                       force_object=extra.get("object"))
        elif event_id == 17:
            return JointPowerBrownoutApplicator(gain_fraction_range=(param_lo, param_hi))
        else:
            raise ValueError(f"Unknown event_id: {event_id}")

    # ── Helper: run one episode ───────────────────────────────────────────
    def run_episode(ep_seed, inject_event, scheduler):
        """Run a single episode. Returns (success, reason, cube_final, ep_meta).

        Uses ep_seed to deterministically initialise the episode so that
        baseline and counterfactual runs share identical initial conditions.
        """
        nonlocal cube

        ep_rng = np.random.RandomState(ep_seed)
        cur_dims = _run_module.sample_workpiece_dims(ep_rng)
        cur_spawn, cur_yaw = _run_module.randomize_cube_pose(ep_rng)
        cur_spawn[2] = TABLE_HEIGHT + cur_dims[2] / 2.0 + 0.005

        cube = _run_module.spawn_workpiece(world, ep_rng, cur_spawn,
                                           cur_dims, existing_cube=cube)

        home = np.zeros(n_dof)
        home[:len(HOME_JOINTS)] = HOME_JOINTS
        robot.set_joints_default_state(positions=home)
        cube.set_default_state(position=cur_spawn,
                               orientation=_run_module.yaw_to_quat(cur_yaw))

        _run_module.setup_gripper(stage)
        world.reset()

        # Set mass/friction AFTER world.reset()
        cur_mass, cur_friction = _run_module.randomize_cube_physics(cube, ep_rng)
        cur_restitution = getattr(_run_module.randomize_cube_physics, "_last_restitution", 0.0)

        ep_meta = {
            "cube_mass_kg": cur_mass, "cube_friction_coeff": cur_friction,
            "cube_restitution_coeff": cur_restitution,
            "cube_width_m": cur_dims[0], "cube_depth_m": cur_dims[1],
            "cube_height_m": cur_dims[2],
            "cube_spawn_x_m": cur_spawn[0], "cube_spawn_y_m": cur_spawn[1],
            "cube_spawn_z_m": cur_spawn[2],
        }

        if not _headless:
            _run_module.update_hud(cur_mass, cur_friction)

        robot.set_solver_position_iteration_count(64)
        robot.set_solver_velocity_iteration_count(64)
        rmp_controller.reset()

        pick_place = PickPlaceController(
            name="pp_sweep",
            cspace_controller=rmp_controller,
            gripper=gripper,
            end_effector_initial_height=EEF_INITIAL_HEIGHT,
            events_dt=_EVENTS_DT)

        robot.set_joint_positions(home)
        robot.set_joint_velocities(np.zeros(n_dof))
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))
        gripper.post_reset()

        _, _cur_close_rad = _run_module.update_gripper_for_episode(
            stage, ep_rng, robot, cur_mass, cur_friction)
        gripper.open()
        for _ in range(30):
            world.step(render=not _headless)

        perceived = _run_module.apply_perception_noise(cur_spawn.copy(), ep_rng)
        pick_pos = perceived.copy()
        pick_pos[2] = perceived[2] + GRIPPER_TCP_Z
        place_pos = BIN_POSITION.copy()
        place_pos[2] = BIN_POSITION[2] + GRIPPER_TCP_Z + 0.04
        ee_orient = euler_angles_to_quat(np.array([0, np.pi, cur_yaw]))

        if inject_event and scheduler is not None:
            _setup_ctx = SimContext(
                stage=stage,
                cube_prim_path=CUBE_PRIM,
                robot_prim_path=ROBOT_PRIM,
                sim_dt=SIM_DT,
                extra={"robot": robot})
            scheduler.reset(_setup_ctx)
            scheduler.schedule_episode(args.max_steps)
            scheduler.setup_episode(_setup_ctx)

        sim_time = 0.0
        ep_step = 0
        _slip_detected = False
        _slip_prev_cube_pos = None
        _slip_prev_eef_pos = None
        _slip_rel_vel_window = []
        _SLIP_WINDOW = 40
        _SLIP_VEL_THRESHOLD = 0.03    # m/s, sustained relative Z velocity = slip

        while simulation_app.is_running():
            world.step(render=not _headless)
            if not world.is_playing():
                continue
            sim_time += SIM_DT
            ep_step += 1

            cube_pos, _ = cube.get_world_pose()
            cube_pos = np.asarray(cube_pos)
            if np.any(np.abs(cube_pos) > 10.0):
                cube_pos = cur_spawn.copy()

            # Drop detection, immediate, same as run.py
            if cube_pos[2] < -0.15:
                if inject_event and scheduler is not None:
                    scheduler.reset(_setup_ctx)
                cube_final = np.asarray(cube.get_world_pose()[0])
                return (False, "dropped", cube_final, ep_meta)

            if sim_time > 60.0:
                if inject_event and scheduler is not None:
                    scheduler.reset(_setup_ctx)
                cube_final = np.asarray(cube.get_world_pose()[0])
                return (False, "timeout", cube_final, ep_meta)

            # PickPlaceController, identical to run.py
            current_joints = robot.get_joint_positions()
            action = pick_place.forward(
                picking_position=pick_pos,
                placing_position=place_pos,
                current_joint_positions=current_joints,
                end_effector_orientation=ee_orient)
            robot.apply_action(action)

            phase = min(pick_place.get_current_event(), 9)
            if phase >= 3 and phase < 7:
                grip_target = _cur_close_rad
            elif phase <= 2:
                grip_target = np.radians(-3.0)
            else:
                grip_target = 0.0
            robot._articulation_view.set_joint_position_targets(
                np.array([[grip_target]]), joint_indices=np.array([fj_idx])
            )

            # Slip detection, compare cube and EEF velocities via finite
            # difference so both have identical lag characteristics.
            # A real slip produces sustained negative relative Z velocity.
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
                        if ep_step % 20 == 0:
                            print(f"[{ep_step:6d}] SLIP monitor: avg_rel_vel_z={avg_rel_vel:.4f} m/s  phase={phase}")
                        if avg_rel_vel < -_SLIP_VEL_THRESHOLD:
                            _slip_detected = True
                            print(f"[{ep_step:6d}] SLIP detected: avg relative Z vel "
                                  f"= {avg_rel_vel:.4f} m/s over {_SLIP_WINDOW} frames")

                _slip_prev_cube_pos = cube_pos.copy()
                _slip_prev_eef_pos = eef_pos.copy()
            else:
                _slip_rel_vel_window.clear()
                _slip_prev_cube_pos = None
                _slip_prev_eef_pos = None

            # Event injection
            active_events = []
            if inject_event and scheduler is not None:
                from episode_logger import JOINT_NAMES as _JNAMES
                ctx = SimContext(
                    stage=stage,
                    sensor_data={},
                    cube_prim_path=CUBE_PRIM,
                    robot_prim_path=ROBOT_PRIM,
                    joint_names=_JNAMES,
                    sim_dt=SIM_DT,
                    episode_step=ep_step,
                    state_machine=PHASE_NAMES[phase],
                    extra={"robot": robot, "action": action})
                active_events = scheduler.step(ep_step, ctx) or []
                if not _headless:
                    _run_module.update_event_indicator(active_events)
            elif not _headless:
                _run_module.update_event_indicator([])

            # Show slip on HUD
            if not _headless and _slip_detected:
                if hasattr(_run_module, '_evt_label') and _run_module._evt_label is not None:
                    _run_module._evt_label.text = "  EVENT: Grip Slip (id=1)"
                    _run_module._evt_label.set_style({"font_size": 18, "color": 0xFFFFFFFF})
                if hasattr(_run_module, '_evt_params_label') and _run_module._evt_params_label is not None:
                    _run_module._evt_params_label.text = "  cube slipped from gripper"
                if hasattr(_run_module, '_evt_rect') and _run_module._evt_rect is not None:
                    _run_module._evt_rect.set_style({"background_color": 0xFFCC6600,
                                                     "border_radius": 4})

            # Success check, identical to run.py
            if pick_place.is_done():
                if inject_event and scheduler is not None:
                    scheduler.reset(_setup_ctx)
                cube_final = np.asarray(cube.get_world_pose()[0])
                bp = BIN_POSITION
                in_bin = (abs(cube_final[0] - bp[0]) < 0.14 and
                          abs(cube_final[1] - bp[1]) < 0.14 and
                          cube_final[2] > bp[2] - 0.12 and
                          cube_final[2] < bp[2] + 0.12)
                if in_bin:
                    return (True, "placed", cube_final, ep_meta)
                else:
                    reason = "missed_bin_slip" if _slip_detected else "missed_bin"
                    return (False, reason, cube_final, ep_meta)

        if inject_event and scheduler is not None:
            scheduler.reset()
        cube_final = np.asarray(cube.get_world_pose()[0])
        return (False, "timeout", cube_final, ep_meta)

    # ── Per-episode CSV log ───────────────────────────────────────────────
    _log_fh = open(args._worker_logfile, "w", newline="", buffering=1)
    _log_writer = csv.DictWriter(_log_fh, fieldnames=_EP_LOG_COLS)
    _log_writer.writeheader()

    def _write_ep_log(ci, ep, run_type, event_id, event_name, param_name,
                      param_lo, param_hi, success, reason, cube_final, ep_meta):
        _log_writer.writerow({
            "cell_idx": ci, "episode": ep, "run_type": run_type,
            "event_id": event_id, "event_name": event_name,
            "param_name": param_name or "",
            "param_lo": f"{param_lo:.4f}" if param_lo is not None else "",
            "param_hi": f"{param_hi:.4f}" if param_hi is not None else "",
            "cube_mass_kg": f"{ep_meta['cube_mass_kg']:.4f}",
            "cube_friction_coeff": f"{ep_meta['cube_friction_coeff']:.4f}",
            "cube_restitution_coeff": f"{ep_meta['cube_restitution_coeff']:.4f}",
            "cube_width_m": f"{ep_meta['cube_width_m']:.4f}",
            "cube_depth_m": f"{ep_meta['cube_depth_m']:.4f}",
            "cube_height_m": f"{ep_meta['cube_height_m']:.4f}",
            "cube_spawn_x_m": f"{ep_meta['cube_spawn_x_m']:.4f}",
            "cube_spawn_y_m": f"{ep_meta['cube_spawn_y_m']:.4f}",
            "cube_spawn_z_m": f"{ep_meta['cube_spawn_z_m']:.4f}",
            "cube_final_x_m": f"{cube_final[0]:.4f}",
            "cube_final_y_m": f"{cube_final[1]:.4f}",
            "cube_final_z_m": f"{cube_final[2]:.4f}",
            "sim_time_s": "",
            "success": int(success),
            "reason": reason,
        })

    # ── Run assigned cells ────────────────────────────────────────────────
    results = []
    events_json_path = str(Path(__file__).parent.parent.parent / "events.json")

    for ci in cell_indices:
        cell_idx, event_id, event_name, param_name, param_lo, param_hi, extra = FULL_GRID[ci]
        cell_seed = args.seed + ci * 1000

        if param_name:
            _print(f"[Cell {ci+1}/{len(FULL_GRID)}] event={event_id} ({event_name}) "
                   f"{param_name}=[{param_lo:.2f}-{param_hi:.2f}]")
        else:
            _print(f"[Cell {ci+1}/{len(FULL_GRID)}] event={event_id} ({event_name})")

        applicator = make_applicator(event_id,
                                     param_lo if param_lo is not None else 0,
                                     param_hi if param_hi is not None else 0,
                                     extra=extra)

        scheduler = EventScheduler(
            events_json_path=events_json_path,
            task_name="pick_and_place",
            applicators={event_id: applicator},
            rng_seed=cell_seed + 99999,
            num_events_range=(1, 1),
            force_event_id=event_id)
        scheduler.set_phase_boundaries(_PHASE_BOUNDS)

        baseline_successes = 0
        baseline_failures = 0
        cf_successes = 0
        cf_failures = 0
        t0 = time.time()

        for ep in range(args.episodes):
            ep_seed = cell_seed + ep

            # Phase 1: Baseline (no events)
            success_bl, reason_bl, final_bl, meta_bl = run_episode(ep_seed, False, None)
            _write_ep_log(ci, ep, "baseline", event_id, event_name, param_name,
                          param_lo, param_hi, success_bl, reason_bl, final_bl, meta_bl)

            if not success_bl:
                baseline_failures += 1
                _print(f"  baseline ep {ep+1:3d}/{args.episodes} -> FAIL ({reason_bl}), skip CF")
                continue

            baseline_successes += 1

            # Phase 2: Counterfactual (same seed, with event injection)
            success_cf, reason_cf, final_cf, meta_cf = run_episode(ep_seed, True, scheduler)
            _write_ep_log(ci, ep, "counterfactual", event_id, event_name, param_name,
                          param_lo, param_hi, success_cf, reason_cf, final_cf, meta_cf)

            if success_cf:
                cf_successes += 1
                _print(f"  counterfactual ep {ep+1:3d}/{args.episodes} -> OK (survived)")
            else:
                cf_failures += 1
                _print(f"  counterfactual ep {ep+1:3d}/{args.episodes} -> FAIL ({reason_cf})")

        elapsed = time.time() - t0
        cf_rate = cf_failures / baseline_successes if baseline_successes > 0 else 0.0

        _print(f"  => baseline_succ={baseline_successes}/{args.episodes}  "
               f"cf_failures={cf_failures}/{baseline_successes}  "
               f"cf_rate={cf_rate:.1%}  ({elapsed:.1f}s)\n")

        results.append({
            "cell_idx": ci,
            "event_id": event_id,
            "event_name": event_name,
            "param_name": param_name or "",
            "param_lo": param_lo if param_lo is not None else "",
            "param_hi": param_hi if param_hi is not None else "",
            "total_pairs": args.episodes,
            "baseline_successes": baseline_successes,
            "baseline_failures": baseline_failures,
            "counterfactual_successes": cf_successes,
            "counterfactual_failures": cf_failures,
            "counterfactual_failure_rate": cf_rate,
        })

    _log_fh.flush()
    _log_fh.close()

    with open(outfile, "w") as f:
        json.dump(results, f)

    simulation_app.close()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_visual():
    """Visual mode: run all cells in-process (no subprocess) so the
    Isaac Sim window renders properly."""
    import json as _json

    # Set up as if we're a worker with all cells
    args._worker_cells = _json.dumps(list(range(len(FULL_GRID))))

    import tempfile
    _out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    _out.close()
    args._worker_outfile = _out.name

    _log = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    _log.close()
    args._worker_logfile = _log.name

    args._worker_headless = False

    run_worker()

    # Collect and print results
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = SWEEP_DIR / f"events_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.move(args._worker_logfile, str(run_dir / "sweep_episodes.csv"))

    with open(args._worker_outfile) as f:
        results = _json.load(f)
    os.unlink(args._worker_outfile)

    results_csv = run_dir / "sweep_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_id", "event_name", "param_name", "param_lo", "param_hi",
            "total_pairs", "baseline_successes", "baseline_failures",
            "counterfactual_successes", "counterfactual_failures",
            "counterfactual_failure_rate",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nResults saved to {run_dir}")
    for r in results:
        event_str = f"[{r['event_id']}] {r['event_name']}"
        if r["param_name"]:
            param_str = f"[{r['param_lo']:.2f}-{r['param_hi']:.2f}]"
        else:
            param_str = ", "
        print(f"  {event_str:>30} {param_str:>18}  "
              f"BL={r['baseline_successes']}  CF_fail={r['counterfactual_failures']}  "
              f"rate={r['counterfactual_failure_rate']:.1%}")


if __name__ == "__main__":
    if args._worker_cells is not None:
        try:
            run_worker()
        except Exception:
            import traceback
            with open("/tmp/sweep_events_crash.txt", "w") as _cf:
                traceback.print_exc(file=_cf)
            traceback.print_exc(file=sys.__stderr__)
            sys.exit(1)
    elif args.visual:
        run_visual()
    else:
        run_launcher()
