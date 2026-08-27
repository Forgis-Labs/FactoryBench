#!/usr/bin/env python3
"""
FactoryBench / pick_and_place / sweep_mass_friction.py
Shared grid-search over cube mass & friction ranges for any UR robot.

Usage:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/pick_and_place/sweep_mass_friction.py \
        --robot ur5 [--episodes 100] [--seed 42] [--workers 4]
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

import numpy as np

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

FACTORYBENCH_DIR = Path(__file__).parent.parent.resolve()
PYTHON_SH = Path("/opt/IsaacSim/python.sh").resolve()
FULL_GRID = list(itertools.product(MASS_RANGES, FRICTION_RANGES))

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


def _get_config_path(robot: str) -> str:
    return str(FACTORYBENCH_DIR / robot / "pick_and_place" / "config" / "task_shared.yaml")


def _get_sweep_dir(robot: str) -> Path:
    return FACTORYBENCH_DIR / robot / "pick_and_place" / "sweeps"


# ── CLI ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Sweep mass & friction ranges")
parser.add_argument("--robot", choices=["ur3", "ur5", "kuka_kr10", "franka", "abb_irb2600"], required=True)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--workers", type=int, default=1)
parser.add_argument("--_worker_cells", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--_worker_outfile", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--_worker_logfile", type=str, default=None, help=argparse.SUPPRESS)
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCHER MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_launcher():
    n_workers = min(args.workers, len(FULL_GRID))
    n_cells = len(FULL_GRID)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _get_sweep_dir(args.robot) / f"mass_friction_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Robot: {args.robot}")
    print(f"Grid: {len(MASS_RANGES)} mass x {len(FRICTION_RANGES)} friction "
          f"= {n_cells} cells, {args.episodes} episodes each")
    print(f"Output: {run_dir}")
    print(f"Launching {n_workers} parallel Isaac Sim workers...\n", flush=True)

    chunks = [[] for _ in range(n_workers)]
    for i, cell in enumerate(FULL_GRID):
        chunks[i % n_workers].append(i)

    procs = []
    outfiles = []
    logfiles = []
    for w in range(n_workers):
        outfile = tempfile.NamedTemporaryFile(suffix=".json", prefix=f"sweep_w{w}_", delete=False)
        outfile.close()
        outfiles.append(outfile.name)

        logfile = tempfile.NamedTemporaryFile(suffix=".csv", prefix=f"sweep_log_w{w}_", delete=False)
        logfile.close()
        logfiles.append(logfile.name)

        cmd = [
            str(PYTHON_SH), __file__,
            "--robot", args.robot,
            "--episodes", str(args.episodes),
            "--seed", str(args.seed),
            "--_worker_cells", json.dumps(chunks[w]),
            "--_worker_outfile", outfile.name,
            "--_worker_logfile", logfile.name,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(proc)

    import threading
    import re
    _keep = re.compile(r"^Starting \(\d+ cells\)|^Ready |^\[\d+/\d+\]|^  ep |^  => ")

    def _stream(proc, prefix):
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

    # Merge per-worker episode logs
    episodes_csv = run_dir / "sweep_episodes.csv"
    with open(episodes_csv, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=_EP_LOG_COLS)
        writer.writeheader()
        for logfile in logfiles:
            try:
                with open(logfile, newline="") as lf:
                    for row in csv.DictReader(lf):
                        writer.writerow(row)
            except Exception:
                pass
            os.unlink(logfile)
    print(f"Episode log saved to {episodes_csv}")

    results = [r for r in all_results if r is not None]

    # Summary table
    print(f"\nCompleted in {elapsed:.1f}s total\n")
    print("=" * 72)
    print(f"SWEEP RESULTS ({args.robot.upper()})")
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

    results_csv = run_dir / "sweep_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mass_lo", "mass_hi", "friction_lo", "friction_hi",
            "total", "success", "rate"])
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

    _real_stdout = sys.stdout
    sys.stdout = io.StringIO()

    def _print(*a, **kw):
        kw["file"] = _real_stdout
        kw["flush"] = True
        print(*a, **kw)

    _print(f"Starting ({len(cell_indices)} cells)...")

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"width": 1280, "height": 720, "headless": True})

    import carb
    carb.settings.get_settings().set("/log/level", "error")
    carb.settings.get_settings().set("/log/fileLogLevel", "error")
    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    import logging
    logging.getLogger("omni").setLevel(logging.ERROR)

    import omni.usd
    from pxr import UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.robot.manipulators.controllers import PickPlaceController

    _shared_dir = str(Path(__file__).parent.resolve())
    if _shared_dir not in sys.path:
        sys.path.insert(0, _shared_dir)

    import base_runner as _br

    config_path = _get_config_path(args.robot)
    cfg = _br.load_config(config_path)
    task_dir = str(Path(config_path).parent.parent.resolve())
    rc = _br.init_from_config(cfg, task_dir)

    t_init = time.time()

    rng = np.random.RandomState(args.seed)
    world = World(physics_dt=rc.sim_dt, rendering_dt=rc.sim_dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    world.scene.add_default_ground_plane()
    _br.build_workcell(stage, rc)
    _br._ensure_surface_friction(stage, "/World/cell/conv_belt", friction=0.6)
    _br._ensure_surface_friction(stage, "/World/cell/bin_floor", friction=0.8)
    _br._ensure_surface_friction(stage, "/World/cell/bin_stand_top", friction=0.6)

    assets_root = get_assets_root_path()
    add_reference_to_stage(assets_root + rc.robot_usd, rc.robot_prim)
    robot = world.scene.add(Robot(prim_path=rc.robot_prim, name=rc.robot_name))

    add_reference_to_stage(rc.gripper_usd, rc.gripper_prim)
    _br.setup_gripper(stage, rc)

    cur_dims = _br.sample_workpiece_dims(rng, rc)
    cur_spawn, cur_yaw = _br.randomize_cube_pose(rng, rc)
    cur_spawn[2] = rc.table_height + cur_dims[2] / 2.0 + 0.005
    cube = _br.spawn_workpiece(world, rng, cur_spawn, cur_dims, rc)

    world.reset()

    n_dof = robot.num_dof
    home_full = np.zeros(n_dof)
    home_full[:len(rc.home_joints)] = rc.home_joints
    robot.set_joints_default_state(positions=home_full)
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    robot.set_joint_positions(home_full)
    robot.set_joint_velocities(np.zeros(n_dof))

    for _ in range(rc.init_steps):
        world.step(render=False)

    gripper_base_link = rc.gripper_base + "/" + rc.gripper_base_link
    gripper = ParallelGripper(
        end_effector_prim_path=gripper_base_link,
        joint_prim_names=["finger_joint"],
        joint_opened_positions=np.array([0.0]),
        joint_closed_positions=np.array([rc.close_deg]),
        action_deltas=None,
        use_mimic_joints=True,
    )
    gripper.initialize(
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    robot._articulation_view.set_gains(
        kps=np.array([[rc.grip_kp] * len(rc.grip_joint_indices)]),
        kds=np.array([[rc.grip_kd] * len(rc.grip_joint_indices)]),
        joint_indices=rc.grip_joint_indices
    )
    gripper.open()
    for _ in range(20):
        world.step(render=False)

    rmp_controller = _br.create_rmpflow_controller(
        name=f"{rc.robot_name}_rmpflow",
        robot_articulation=robot,
        rmpflow_name=rc.rmpflow_name,
        physics_dt=rc.sim_dt,
    )

    sys.stderr = io.StringIO()
    _print(f"Ready ({time.time() - t_init:.1f}s)")

    # Per-episode CSV log
    _log_fh = open(args._worker_logfile, "w", newline="", buffering=1)
    _log_writer = csv.DictWriter(_log_fh, fieldnames=_EP_LOG_COLS)
    _log_writer.writeheader()

    BIN_POSITION = rc.bin_position

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
            cur_dims = _br.sample_workpiece_dims(ep_rng, rc)
            cur_spawn, cur_yaw = _br.randomize_cube_pose(ep_rng, rc)
            cur_spawn[2] = rc.table_height + cur_dims[2] / 2.0 + 0.005

            cube = _br.spawn_workpiece(world, ep_rng, cur_spawn, cur_dims, rc,
                                       existing_cube=cube)

            mass = float(ep_rng.uniform(*mass_r))
            friction = float(ep_rng.uniform(*fric_r))
            restitution = float(ep_rng.uniform(0.0, 0.2))

            home_full = np.zeros(n_dof)
            home_full[:len(rc.home_joints)] = rc.home_joints
            robot.set_joints_default_state(positions=home_full)
            cube.set_default_state(position=cur_spawn,
                                   orientation=_br.yaw_to_quat(cur_yaw))

            _br.setup_gripper(stage, rc)
            world.reset()

            # Set mass/friction AFTER world.reset()
            try:
                prim = stage.GetPrimAtPath(rc.cube_prim)
                if not prim.HasAPI(UsdPhysics.MassAPI):
                    UsdPhysics.MassAPI.Apply(prim)
                UsdPhysics.MassAPI(prim).CreateMassAttr().Set(mass)
                _br._apply_physics_material(
                    stage, rc.cube_prim, friction, friction * 0.85, restitution)
            except Exception as e:
                _print(f"  WARNING: failed to set physics: {e}")

            robot.set_solver_position_iteration_count(64)
            robot.set_solver_velocity_iteration_count(64)
            rmp_controller.reset()

            pick_place = PickPlaceController(
                name=f"pp_{ci}_{ep}",
                cspace_controller=rmp_controller,
                gripper=gripper,
                end_effector_initial_height=rc.eef_initial_height,
                events_dt=rc.events_dt,
            )

            robot.set_joint_positions(home_full)
            robot.set_joint_velocities(np.zeros(n_dof))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
            gripper.post_reset()

            _br.update_gripper_for_episode(stage, ep_rng, robot, mass, friction, rc)
            robot._articulation_view.set_gains(
                kps=np.array([[rc.grip_kp] * len(rc.grip_joint_indices)]),
                kds=np.array([[rc.grip_kd] * len(rc.grip_joint_indices)]),
                joint_indices=rc.grip_joint_indices
            )
            gripper.open()
            for _ in range(rc.settle_steps):
                world.step(render=False)

            perceived = _br.apply_perception_noise(cur_spawn.copy(), ep_rng, rc)
            pick_pos = perceived.copy()
            pick_pos[2] = perceived[2] + rc.tcp_z
            place_pos = BIN_POSITION.copy()
            place_pos[2] = BIN_POSITION[2] + rc.tcp_z + 0.04
            ee_orient = euler_angles_to_quat(np.array([0, np.pi, cur_yaw]))

            sim_time = 0.0
            ep_success = False
            reason = ""
            drop_time = None

            while simulation_app.is_running():
                world.step(render=False)
                if not world.is_playing():
                    continue
                sim_time += rc.sim_dt

                cube_pos, _ = cube.get_world_pose()
                cube_pos = np.asarray(cube_pos)
                if np.any(np.abs(cube_pos) > 10.0):
                    cube_pos = cur_spawn.copy()

                if cube_pos[2] < rc.drop_z_threshold and drop_time is None:
                    drop_time = sim_time
                if drop_time is not None and sim_time - drop_time >= 2.0:
                    reason = "dropped"
                    break
                if sim_time > rc.timeout_s:
                    reason = "timeout"
                    break

                current_joints = robot.get_joint_positions()
                action = pick_place.forward(
                    picking_position=pick_pos,
                    placing_position=place_pos,
                    current_joint_positions=current_joints,
                    end_effector_orientation=ee_orient,
                )
                if action.joint_positions is not None:
                    robot.apply_action(action)

                if pick_place.is_done():
                    cube_final, _ = cube.get_world_pose()
                    cube_final = np.asarray(cube_final)
                    in_bin = _br.check_in_bin(cube_final, rc)
                    ep_success = in_bin
                    reason = "placed" if in_bin else "missed_bin"
                    break

            # Get cube final position for logging
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

    with open(outfile, "w") as f:
        json.dump(results, f)

    simulation_app.close()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if args._worker_cells is not None:
        run_worker()
    else:
        run_launcher()
