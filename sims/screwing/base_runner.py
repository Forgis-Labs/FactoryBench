"""
FactoryBench / screwing / base_runner.py
Nut-bolt threading task adapted from the Factory environment (IsaacLab).

Uses Isaac Sim's SimulationApp directly (not IsaacLab's AppLauncher) to
avoid version-mismatch issues.  Loads the Factory M16 nut + bolt USD
assets, sets up the Franka Panda, and runs nut-threading episodes with
the same physics settings as the Factory paper.

Run via the thin wrapper:
    cd /opt/IsaacSim
    ./python.sh FactoryBench/franka/screwing/run_shared.py --headless --episodes 5
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import yaml

_FACTORYBENCH_DIR = str(Path(__file__).parent.parent.resolve())


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="FactoryBench NutThread")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0, help="0 = run forever")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--run_type", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser.parse_args()


def run(config_path: str, cli_args=None):
    cfg = load_config(config_path)
    task_dir = str(Path(config_path).parent.parent.resolve())

    if cli_args is None:
        cli_args = parse_args()

    # -----------------------------------------------------------------------
    # Launch Isaac Sim
    # -----------------------------------------------------------------------
    import numpy as np
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        "width": 1280, "height": 720,
        "headless": cli_args.headless,
    })

    # -----------------------------------------------------------------------
    # Deferred imports
    # -----------------------------------------------------------------------
    import carb
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf, PhysxSchema
    from isaacsim.core.api import World
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.api.objects import DynamicCylinder
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.robot.manipulators.grippers import ParallelGripper

    # -----------------------------------------------------------------------
    # Physics settings matching Factory paper
    # -----------------------------------------------------------------------
    phys = cfg.get("physics", {})
    sim_dt = float(phys.get("sim_dt", 1.0 / 120.0))

    world = World(
        physics_dt=sim_dt,
        rendering_dt=sim_dt,
        stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # Configure PhysX scene for contact-rich simulation
    physx_scene_path = "/physicsScene"
    physx_scene_prim = stage.GetPrimAtPath(physx_scene_path)
    if not physx_scene_prim.IsValid():
        from pxr import UsdPhysics as UP
        UP.Scene.Define(stage, physx_scene_path)
        physx_scene_prim = stage.GetPrimAtPath(physx_scene_path)

    if physx_scene_prim.IsValid():
        if not physx_scene_prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            PhysxSchema.PhysxSceneAPI.Apply(physx_scene_prim)
        px = PhysxSchema.PhysxSceneAPI(physx_scene_prim)
        px.CreateSolverTypeAttr().Set("TGS")
        px.CreateFrictionOffsetThresholdAttr().Set(
            float(phys.get("friction_offset_threshold", 0.01)))
        px.CreateFrictionCorrelationDistanceAttr().Set(
            float(phys.get("friction_correlation_distance", 0.00625)))
        px.CreateBounceThresholdAttr().Set(
            float(phys.get("bounce_threshold_velocity", 0.2)))
        px.CreateGpuMaxRigidContactCountAttr().Set(
            int(phys.get("gpu_max_rigid_contact_count", 2**23)))
        px.CreateGpuCollisionStackSizeAttr().Set(
            int(phys.get("gpu_collision_stack_size", 2**28)))
        px.CreateGpuMaxNumPartitionsAttr().Set(1)
        print(f"[Physics] PhysX scene configured (TGS solver, "
              f"contact_count={phys.get('gpu_max_rigid_contact_count', 2**23)})")

    # Ground plane
    world.scene.add_default_ground_plane()

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/ambient")
    dome.GetIntensityAttr().Set(2000)
    dome.GetColorAttr().Set(Gf.Vec3f(0.75, 0.75, 0.75))

    # -----------------------------------------------------------------------
    # Load Franka Panda
    # -----------------------------------------------------------------------
    rcfg = cfg["robot"]
    robot_usd = rcfg["robot_usd"]
    if not os.path.isabs(robot_usd):
        robot_usd = str(Path(_FACTORYBENCH_DIR) / robot_usd)

    ROBOT_PRIM = rcfg["prim"]
    add_reference_to_stage(robot_usd, ROBOT_PRIM)
    robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name=rcfg["name"]))

    # Configure solver iterations on robot articulation
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM)
    if robot_prim.IsValid():
        if not robot_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
        art_api = PhysxSchema.PhysxArticulationAPI(robot_prim)
        art_api.CreateSolverPositionIterationCountAttr().Set(
            int(phys.get("solver_position_iterations", 192)))
        art_api.CreateSolverVelocityIterationCountAttr().Set(
            int(phys.get("solver_velocity_iterations", 1)))

    print(f"[Robot] {rcfg['name']} loaded from {robot_usd}")

    # -----------------------------------------------------------------------
    # Load Factory M16 bolt (fixed on table)
    # -----------------------------------------------------------------------
    scfg = cfg["scene"]
    bolt_usd = scfg["bolt_usd"]
    if not os.path.isabs(bolt_usd):
        bolt_usd = str(Path(_FACTORYBENCH_DIR) / bolt_usd)
    BOLT_PRIM = scfg["bolt_prim"]
    bolt_pos = np.array(scfg["bolt_position"])

    add_reference_to_stage(bolt_usd, BOLT_PRIM)
    bolt_prim = stage.GetPrimAtPath(BOLT_PRIM)

    # The Factory bolt USD has its own internal rigid body child.
    # Don't add another RigidBodyAPI on the parent, just position it.
    bolt_xf = UsdGeom.Xformable(bolt_prim)
    bolt_xf.ClearXformOpOrder()
    bolt_xf.AddTranslateOp().Set(Gf.Vec3f(float(bolt_pos[0]), float(bolt_pos[1]), float(bolt_pos[2])))

    print(f"[Scene] M16 bolt at {bolt_pos}")

    # -----------------------------------------------------------------------
    # Load Factory M16 nut (held by gripper initially)
    # -----------------------------------------------------------------------
    nut_usd = scfg["nut_usd"]
    if not os.path.isabs(nut_usd):
        nut_usd = str(Path(_FACTORYBENCH_DIR) / nut_usd)
    NUT_PRIM = scfg["nut_prim"]

    add_reference_to_stage(nut_usd, NUT_PRIM)
    nut_prim = stage.GetPrimAtPath(NUT_PRIM)

    # The Factory nut USD has its own internal rigid body child.
    # Don't add another RigidBodyAPI, just position it.
    nut_init_pos = bolt_pos.copy()
    nut_init_pos[2] += 0.05  # above bolt
    nut_xf = UsdGeom.Xformable(nut_prim)
    nut_xf.ClearXformOpOrder()
    nut_xf.AddTranslateOp().Set(Gf.Vec3f(float(nut_init_pos[0]),
                                           float(nut_init_pos[1]),
                                           float(nut_init_pos[2])))

    print(f"[Scene] M16 nut at {nut_init_pos}")

    # -----------------------------------------------------------------------
    # World reset and robot init
    # -----------------------------------------------------------------------
    world.reset()

    n_dof = robot.num_dof
    home_joints = np.array(rcfg["home_joints"])
    home_full = np.zeros(n_dof)
    home_full[:len(home_joints)] = home_joints
    # Open gripper
    gcfg = rcfg["gripper"]
    open_pos = gcfg["joint_opened_positions"]
    for i, idx in enumerate(gcfg["grip_joint_indices"]):
        if idx < n_dof:
            home_full[idx] = open_pos[i] if i < len(open_pos) else open_pos[0]

    robot.set_joints_default_state(positions=home_full)
    robot.set_joint_positions(home_full)
    robot.set_joint_velocities(np.zeros(n_dof))

    robot.set_solver_position_iteration_count(192)
    robot.set_solver_velocity_iteration_count(1)

    # RMPFlow will control arm joints (0-6), leave their USD drives at
    # zero stiffness (as the Factory USD has them).  Only set PD gains
    # on the gripper joints so they can hold closed/open.
    n_arm = len(home_joints)
    grip_indices = np.array(gcfg["grip_joint_indices"])
    grip_kp_val = float(gcfg["grip_kp"])
    grip_kd_val = float(gcfg["grip_kd"])
    grip_effort = float(gcfg["max_effort"])

    # Set gains: zero for arm (RMPFlow drives via position targets),
    # high for gripper
    all_kp = np.zeros((1, n_dof))
    all_kd = np.zeros((1, n_dof))
    all_efforts = np.full((1, n_dof), 87.0)
    for idx in grip_indices:
        if idx < n_dof:
            all_kp[0, idx] = grip_kp_val
            all_kd[0, idx] = grip_kd_val
            all_efforts[0, idx] = grip_effort
    # Arm joints need damping to avoid oscillation with RMPFlow
    for i in range(n_arm):
        all_kd[0, i] = 40.0  # light damping for smooth motion
    robot._articulation_view.set_gains(kps=all_kp, kds=all_kd)
    robot._articulation_view.set_max_efforts(all_efforts)

    print(f"[Robot] num_dof={n_dof}  home_joints={home_joints}")
    print(f"[Robot] Arm: kp=0 (RMPFlow), kd=40. Gripper: kp={grip_kp_val}, kd={grip_kd_val}")

    # Settle
    for _ in range(60):
        world.step(render=True)

    print(f"[Robot] Settled. joints={np.round(robot.get_joint_positions(), 3)}")

    # -----------------------------------------------------------------------
    # RMPFlow controller for EEF motion
    # -----------------------------------------------------------------------
    import isaacsim.robot_motion.motion_generation as mg

    rmp_config = mg.interface_config_loader.load_supported_motion_policy_config(
        "Franka", "RMPflow")
    rmp_flow = mg.lula.motion_policies.RmpFlow(**rmp_config)
    articulation_rmp = mg.ArticulationMotionPolicy(robot, rmp_flow, sim_dt)
    rmp_controller = mg.MotionPolicyController(
        name="franka_rmpflow",
        articulation_motion_policy=articulation_rmp)

    # Set robot base pose for RMPFlow
    base_pos, base_ori = robot.get_world_pose()
    rmp_flow.set_robot_base_pose(
        robot_position=base_pos, robot_orientation=base_ori)
    print("[RMPFlow] Franka controller initialized")

    # Bolt tip position, where the nut threads onto
    bolt_height = float(scfg.get("bolt_height", 0.025))
    bolt_base_height = float(scfg.get("bolt_base_height", 0.01))
    bolt_tip = bolt_pos.copy()
    bolt_tip[2] += bolt_base_height + bolt_height
    tcp_z = float(gcfg.get("tcp_z", 0.1034))

    # EEF target: fingertip center at bolt tip (nut touches bolt top)
    eef_target = bolt_tip.copy()
    eef_target[2] += tcp_z  # offset from fingertip to EEF frame

    # Tool-down orientation (gripper pointing straight down)
    ee_orient = euler_angles_to_quat(np.array([0, np.pi, 0]))

    print(f"[Task] Bolt tip z={bolt_tip[2]:.4f}, EEF target z={eef_target[2]:.4f}")

    # -----------------------------------------------------------------------
    # Move to bolt before starting episodes
    # -----------------------------------------------------------------------
    print("[Task] Moving to bolt position...")
    for i in range(300):
        action = rmp_controller.forward(
            target_end_effector_position=eef_target,
            target_end_effector_orientation=ee_orient)
        robot.apply_action(action)
        world.step(render=True)

    print(f"[Task] EEF positioned. joints={np.round(robot.get_joint_positions()[:7], 3)}")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_dir = cli_args.log_dir or os.path.join(
        task_dir, cfg.get("logging", {}).get("log_dir", "logs"))
    if cli_args.run_type:
        log_dir = os.path.join(log_dir, cli_args.run_type)
    os.makedirs(log_dir, exist_ok=True)

    ep_log_path = os.path.join(log_dir, "episodes.csv")
    ep_log_fh = open(ep_log_path, "w", newline="", buffering=1)
    ep_cols = ["episode", "success", "reason", "steps", "sim_time_s", "wall_time_s"]
    ep_writer = csv.DictWriter(ep_log_fh, fieldnames=ep_cols)
    ep_writer.writeheader()
    print(f"[Logger] Writing to {os.path.abspath(log_dir)}/")

    # -----------------------------------------------------------------------
    # Main loop, the robot starts above the bolt and attempts threading
    # -----------------------------------------------------------------------
    step = 0
    episode = 0
    sim_time = 0.0
    ep_start = time.time()
    max_episodes = cli_args.episodes if cli_args.episodes > 0 else float("inf")
    timeout_s = float(cfg.get("episode", {}).get("timeout_s", 30.0))

    # Tool-down orientation
    ee_orient = euler_angles_to_quat(np.array([0, np.pi, 0]))

    print(f"[FactoryBench] Starting screwing episodes (max={cli_args.episodes or 'inf'})...")
    print(f"  sim_dt={sim_dt:.6f}  timeout={timeout_s}s")
    print(f"  bolt={scfg['bolt_usd']}  nut={scfg['nut_usd']}")

    while simulation_app.is_running() and episode < max_episodes:
        world.step(render=True)
        if not world.is_playing():
            continue

        step += 1
        sim_time += sim_dt

        # Timeout check
        if sim_time > timeout_s:
            wall_time = time.time() - ep_start
            print(f"[{step:6d}] Episode {episode} timeout ({sim_time:.1f}s)")
            ep_writer.writerow({
                "episode": episode, "success": 0, "reason": "timeout",
                "steps": step, "sim_time_s": f"{sim_time:.2f}",
                "wall_time_s": f"{wall_time:.2f}",
            })
            episode += 1
            sim_time = 0.0
            ep_start = time.time()
            # Reset robot and nut positions
            robot.set_joint_positions(home_full)
            robot.set_joint_velocities(np.zeros(n_dof))
            continue

        # RMPFlow holds EEF at bolt position; we add wrist rotation for threading.
        action = rmp_controller.forward(
            target_end_effector_position=eef_target,
            target_end_effector_orientation=ee_orient)
        if action.joint_positions is not None:
            arm_pos = np.array(action.joint_positions).flatten()
            full_pos = np.zeros(n_dof)
            full_pos[:len(arm_pos)] = arm_pos
            # Slowly rotate wrist (joint 6 = panda_joint7) for threading
            screw_vel = float(cfg.get("control", {}).get("screw_angular_vel", 1.5))
            full_pos[6] = home_joints[6] - screw_vel * sim_time  # CW rotation
            # Gripper closed
            close_pos = gcfg["joint_closed_positions"]
            for i, idx in enumerate(gcfg["grip_joint_indices"]):
                if idx < n_dof:
                    full_pos[idx] = close_pos[i] if i < len(close_pos) else close_pos[0]
            robot.apply_action(ArticulationAction(joint_positions=full_pos))

    # Cleanup
    ep_log_fh.close()
    simulation_app.close()
    print(f"[FactoryBench] Done. {episode} episodes.")


def _set_friction(stage, prim_path, friction):
    """Apply physics material with given friction to a prim."""
    from pxr import UsdShade, UsdPhysics, PhysxSchema
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    mat_path = prim_path + "/PhysMat"
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.IsValid():
        mat = UsdShade.Material.Define(stage, mat_path)
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mat_prim = mat.GetPrim()
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat = UsdPhysics.MaterialAPI(mat_prim)
    phys_mat.CreateStaticFrictionAttr().Set(friction)
    phys_mat.CreateDynamicFrictionAttr().Set(friction)


def main(config_path: str):
    cli_args = parse_args()
    run(config_path, cli_args)
