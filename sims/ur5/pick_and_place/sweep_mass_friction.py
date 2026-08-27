#!/usr/bin/env python3
"""
sweep_mass_friction.py, grid-search over cube mass & friction ranges.

Uses the same simulation code as run.py. With --workers N, launches N
parallel Isaac Sim processes to split the grid (each pays startup once).

Usage:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/ur5/pick_and_place/sweep_mass_friction.py \
        [--episodes 100] [--seed 42] [--workers 4]
"""

import argparse
import csv
import io
import itertools
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

import numpy as np

# ── Per-episode log columns ───────────────────────────────────────────
_EP_LOG_COLS = [
    "cell_idx", "episode",
    "mass_lo", "mass_hi", "friction_lo", "friction_hi",
    "cube_mass_kg", "cube_friction_coeff", "cube_restitution_coeff",
    "cube_width_m", "cube_depth_m", "cube_height_m",
    "cube_spawn_x_m", "cube_spawn_y_m", "cube_spawn_z_m",
    "cube_final_x_m", "cube_final_y_m", "cube_final_z_m",
    "sim_time_s", "success", "reason",
]

# ── Grid definition ──────────────────────────────────────────────────────
MASS_RANGES = [
    [0.3, 0.5],
    [0.5, 0.7],
    [0.7, 0.9],
]

FRICTION_RANGES = [
    [0.3, 0.5],
    [0.5, 0.8],
    [0.8, 1.0],
    [1.0, 1.5],
    [1.5, 2.0],
    [2.0, 3.0],
]

TASK_DIR = Path(__file__).parent.resolve()
SWEEP_DIR = TASK_DIR / "sweeps"
PYTHON_SH = Path("/opt/IsaacSim/python.sh").resolve()
FULL_GRID = list(itertools.product(MASS_RANGES, FRICTION_RANGES))

# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sweep mass & friction ranges")
parser.add_argument("--episodes", type=int, default=100,
                    help="Episodes per grid cell (default: 100)")
parser.add_argument("--seed", type=int, default=42,
                    help="Base random seed")
parser.add_argument("--workers", type=int, default=1,
                    help="Parallel Isaac Sim processes (default: 1)")
parser.add_argument("--headless", action="store_true", default=True,
                    help="Run headless (default: True)")
parser.add_argument("--visual", action="store_true",
                    help="Run with rendering (forces --workers 1)")
# Internal: worker mode, runs a slice of the grid and writes results to a file
parser.add_argument("--_worker_cells", type=str, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("--_worker_outfile", type=str, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("--_worker_logfile", type=str, default=None,
                    help=argparse.SUPPRESS)
parser.add_argument("--_worker_headless", action="store_true", default=False,
                    help=argparse.SUPPRESS)
args = parser.parse_args()

# --visual overrides --headless and caps workers to 1
if args.visual:
    args.headless = False
    args.workers = 1


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCHER MODE: split grid across N workers
# ═══════════════════════════════════════════════════════════════════════════

def run_launcher():
    n_workers = min(args.workers, len(FULL_GRID))
    n_cells = len(FULL_GRID)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = SWEEP_DIR / f"mass_friction_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {len(MASS_RANGES)} mass x {len(FRICTION_RANGES)} friction "
          f"= {n_cells} cells, {args.episodes} episodes each")
    print(f"Output: {run_dir}")
    print(f"Launching {n_workers} parallel Isaac Sim workers...\n", flush=True)

    # Split cells across workers
    chunks = [[] for _ in range(n_workers)]
    for i, cell in enumerate(FULL_GRID):
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
            "--_worker_cells", json.dumps(chunks[w]),
            "--_worker_outfile", outfile.name,
            "--_worker_logfile", logfile.name,
        ]
        if args.headless:
            cmd.append("--_worker_headless")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        procs.append(proc)

    # Stream worker output with prefixes
    import threading

    import re
    _keep = re.compile(
        r"^Starting \(\d+ cells\)|^Ready |^\[\d+/\d+\]|^  ep |^  => "
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
    print("=" * 72)
    print("SWEEP RESULTS")
    print("=" * 72)

    header = (f"{'Mass Range (kg)':>18} {'Friction Range':>18} "
              f"{'Success':>9} {'Total':>7} {'Rate':>8}")
    print(header)
    print("-" * 72)

    for r in results:
        mass_str = f"[{r['mass_lo']:.2f}-{r['mass_hi']:.2f}]"
        fric_str = f"[{r['friction_lo']:.2f}-{r['friction_hi']:.2f}]"
        print(f"{mass_str:>18} {fric_str:>18} "
              f"{r['success']:>9d} {r['total']:>7d} "
              f"{r['rate']:>7.1%}")

    print("-" * 72)
    print(f"Total cells: {len(results)}")

    # Write CSV
    results_csv = run_dir / "sweep_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mass_lo", "mass_hi", "friction_lo", "friction_hi",
            "total", "success", "rate"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in [
                "mass_lo", "mass_hi", "friction_lo", "friction_hi",
                "total", "success", "rate"]})
    print(f"\nResults saved to {results_csv}")


# ═══════════════════════════════════════════════════════════════════════════
# WORKER MODE: run assigned cells in a single Isaac Sim session
# ═══════════════════════════════════════════════════════════════════════════

def run_worker():
    cell_indices = json.loads(args._worker_cells)
    outfile = args._worker_outfile

    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    _headless = args._worker_headless

    def _print(*a, **kw):
        kw["file"] = _real_stdout
        kw["flush"] = True
        print(*a, **kw)

    _print(f"Starting ({len(cell_indices)} cells)...")

    # Suppress Isaac Sim startup logs
    os.environ["CARB_LOG_LEVEL"] = "error"

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

    os.dup2(_saved_fd1, 1)
    os.dup2(_saved_fd2, 2)
    os.close(_saved_fd1)
    os.close(_saved_fd2)
    os.close(_devnull_fd)

    if _headless:
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

    import run as _run_module
    _run_module._args.headless = _headless

    # All constants from run.py, run.py loads these from task.yaml,
    # so any config change is automatically picked up by the sweep.
    ROBOT_PRIM         = _run_module.ROBOT_PRIM
    EEF_PRIM           = _run_module.EEF_PRIM
    HOME_JOINTS        = _run_module.HOME_JOINTS
    CUBE_PRIM          = _run_module.CUBE_PRIM
    BIN_POSITION       = _run_module.BIN_POSITION
    SIM_DT             = _run_module.SIM_DT
    GRIPPER_TCP_Z      = _run_module.GRIPPER_TCP_Z
    GRIPPER_CLOSE_DEG_BASE = _run_module.GRIPPER_CLOSE_DEG_BASE
    GRIPPER_CLOSE_DEG_MAX  = _run_module.GRIPPER_CLOSE_DEG_MAX
    UR5_USD            = _run_module.UR5_USD
    ROBOTIQ_USD        = _run_module.ROBOTIQ_USD
    ROBOTIQ_PRIM       = _run_module.ROBOTIQ_PRIM
    TABLE_HEIGHT       = _run_module.TABLE_HEIGHT
    EEF_INITIAL_HEIGHT = _run_module.EEF_INITIAL_HEIGHT
    EVENTS_DT          = _run_module.EVENTS_DT

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

    gripper = ParallelGripper(
        end_effector_prim_path=ROBOTIQ_PRIM + "/robotiq_arg2f_base_link",
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

    # ── Per-episode CSV log ───────────────────────────────────────────────
    _log_fh = open(args._worker_logfile, "w", newline="", buffering=1)
    _log_writer = csv.DictWriter(_log_fh, fieldnames=_EP_LOG_COLS)
    _log_writer.writeheader()

    # ── Run assigned cells ────────────────────────────────────────────────
    results = []

    for ci in cell_indices:
        mass_r, fric_r = FULL_GRID[ci]
        cell_seed = args.seed + ci * 1000
        _print(f"[{ci+1}/{len(FULL_GRID)}] mass=[{mass_r[0]:.2f}-{mass_r[1]:.2f}]  "
               f"fric=[{fric_r[0]:.2f}-{fric_r[1]:.2f}]")

        ep_rng = np.random.RandomState(cell_seed)
        success_count = 0
        total_count = 0
        t0 = time.time()

        for ep in range(args.episodes):
            cur_dims = _run_module.sample_workpiece_dims(ep_rng)
            cur_spawn, cur_yaw = _run_module.randomize_cube_pose(ep_rng)
            cur_spawn[2] = TABLE_HEIGHT + cur_dims[2] / 2.0 + 0.005

            cube = _run_module.spawn_workpiece(world, ep_rng, cur_spawn, cur_dims,
                                               existing_cube=cube)

            mass = float(ep_rng.uniform(*mass_r))
            friction = float(ep_rng.uniform(*fric_r))
            restitution = float(ep_rng.uniform(*_run_module.CUBE_RESTITUTION_RANGE))

            home_full = np.zeros(n_dof)
            home_full[:len(HOME_JOINTS)] = HOME_JOINTS
            robot.set_joints_default_state(positions=home_full)
            cube.set_default_state(position=cur_spawn,
                                   orientation=_run_module.yaw_to_quat(cur_yaw))

            _run_module.setup_gripper(stage)
            world.reset()

            # Set mass/friction AFTER world.reset() so the physics engine
            # picks up the new values on the next world.step().
            try:
                prim = stage.GetPrimAtPath(CUBE_PRIM)
                if not prim.HasAPI(UsdPhysics.MassAPI):
                    UsdPhysics.MassAPI.Apply(prim)
                UsdPhysics.MassAPI(prim).CreateMassAttr().Set(mass)
                _run_module._apply_physics_material(
                    stage, CUBE_PRIM, friction, friction * 0.85, restitution)
            except Exception as e:
                _print(f"  WARNING: failed to set physics: {e}")

            if not _headless:
                _run_module.update_hud(mass, friction)

            robot.set_solver_position_iteration_count(64)
            robot.set_solver_velocity_iteration_count(64)
            rmp_controller.reset()

            pick_place = PickPlaceController(
                name=f"pp_{ci}_{ep}",
                cspace_controller=rmp_controller,
                gripper=gripper,
                end_effector_initial_height=EEF_INITIAL_HEIGHT,
                events_dt=EVENTS_DT)

            robot.set_joint_positions(home_full)
            robot.set_joint_velocities(np.zeros(n_dof))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            gripper.post_reset()

            _, _cur_close_rad = _run_module.update_gripper_for_episode(stage, ep_rng, robot, mass, friction)
            gripper.open()
            for _ in range(30):
                world.step(render=not _headless)

            perceived = _run_module.apply_perception_noise(cur_spawn.copy(), ep_rng)
            pick_pos = perceived.copy()
            pick_pos[2] = perceived[2] + GRIPPER_TCP_Z
            place_pos = BIN_POSITION.copy()
            place_pos[2] = BIN_POSITION[2] + GRIPPER_TCP_Z + 0.04
            ee_orient = euler_angles_to_quat(np.array([0, np.pi, cur_yaw]))

            sim_time = 0.0
            ep_success = False
            reason = ""
            fj_idx = gripper.joint_dof_indicies[0]

            while simulation_app.is_running():
                world.step(render=not _headless)
                if not world.is_playing():
                    continue
                sim_time += SIM_DT

                cube_pos, _ = cube.get_world_pose()
                cube_pos = np.asarray(cube_pos)
                if np.any(np.abs(cube_pos) > 10.0):
                    cube_pos = cur_spawn.copy()

                # Drop detection, immediate, same as run.py
                if cube_pos[2] < -0.15:
                    reason = "dropped"
                    break

                if sim_time > 60.0:
                    reason = "timeout"
                    break

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

                # Success check, identical to run.py
                if pick_place.is_done():
                    cube_final, _ = cube.get_world_pose()
                    cube_final = np.asarray(cube_final)
                    bp = BIN_POSITION
                    in_bin = (abs(cube_final[0] - bp[0]) < 0.14 and
                              abs(cube_final[1] - bp[1]) < 0.14 and
                              cube_final[2] > bp[2] - 0.12 and
                              cube_final[2] < bp[2] + 0.12)
                    ep_success = in_bin
                    reason = "placed" if in_bin else "missed_bin"
                    break

            # Get cube final position (for all outcomes)
            try:
                cube_final, _ = cube.get_world_pose()
                cube_final = np.asarray(cube_final)
            except Exception:
                cube_final = np.full(3, np.nan)

            total_count += 1
            if ep_success:
                success_count += 1

            _log_writer.writerow({
                "cell_idx": ci, "episode": ep,
                "mass_lo": mass_r[0], "mass_hi": mass_r[1],
                "friction_lo": fric_r[0], "friction_hi": fric_r[1],
                "cube_mass_kg": f"{mass:.4f}",
                "cube_friction_coeff": f"{friction:.4f}",
                "cube_restitution_coeff": f"{restitution:.4f}",
                "cube_width_m": f"{cur_dims[0]:.4f}",
                "cube_depth_m": f"{cur_dims[1]:.4f}",
                "cube_height_m": f"{cur_dims[2]:.4f}",
                "cube_spawn_x_m": f"{cur_spawn[0]:.4f}",
                "cube_spawn_y_m": f"{cur_spawn[1]:.4f}",
                "cube_spawn_z_m": f"{cur_spawn[2]:.4f}",
                "cube_final_x_m": f"{cube_final[0]:.4f}",
                "cube_final_y_m": f"{cube_final[1]:.4f}",
                "cube_final_z_m": f"{cube_final[2]:.4f}",
                "sim_time_s": f"{sim_time:.4f}",
                "success": int(ep_success),
                "reason": reason,
            })

            status = "OK" if ep_success else reason
            _print(f"  ep {ep+1:3d}/{args.episodes}  m={mass:.2f} f={friction:.2f} -> {status}")

        elapsed = time.time() - t0
        rate = success_count / total_count if total_count > 0 else 0.0
        _print(f"  => {success_count}/{total_count} = {rate:.1%}  ({elapsed:.1f}s)\n")

        results.append({
            "cell_idx": ci,
            "mass_lo": mass_r[0], "mass_hi": mass_r[1],
            "friction_lo": fric_r[0], "friction_hi": fric_r[1],
            "total": total_count, "success": success_count, "rate": rate,
        })

    _log_fh.flush()
    _log_fh.close()

    # Write results to temp file for launcher to collect
    with open(outfile, "w") as f:
        json.dump(results, f)

    simulation_app.close()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_visual():
    """Visual mode: run all cells in-process (no subprocess) so the
    Isaac Sim window renders properly."""

    args._worker_cells = json.dumps(list(range(len(FULL_GRID))))

    import tempfile
    _out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    _out.close()
    args._worker_outfile = _out.name

    _log = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    _log.close()
    args._worker_logfile = _log.name

    args._worker_headless = False

    run_worker()

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = SWEEP_DIR / f"mass_friction_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    shutil.move(args._worker_logfile, str(run_dir / "sweep_episodes.csv"))

    with open(args._worker_outfile) as f:
        results = json.load(f)
    os.unlink(args._worker_outfile)

    results_csv = run_dir / "sweep_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mass_lo", "mass_hi", "friction_lo", "friction_hi",
            "total", "success", "rate"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"\nResults saved to {run_dir}")
    for r in results:
        mass_str = f"[{r['mass_lo']:.2f}-{r['mass_hi']:.2f}]"
        fric_str = f"[{r['friction_lo']:.2f}-{r['friction_hi']:.2f}]"
        print(f"  {mass_str:>18} {fric_str:>18}  "
              f"{r['success']}/{r['total']}  {r['rate']:.1%}")


if __name__ == "__main__":
    if args._worker_cells is not None:
        run_worker()
    elif args.visual:
        run_visual()
    else:
        run_launcher()
