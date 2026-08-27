"""
SimContext, a lightweight container passed to every applicator on each step.

This is the *only* coupling between the injection framework and the
simulation.  Applicators never import Isaac Sim directly; they read/write
through SimContext so the same applicator works with any sim backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SimContext:
    """Snapshot of simulation state exposed to event applicators.

    Attributes
    ----------
    stage : Any
        The USD stage handle (``omni.usd.get_context().get_stage()``).
        Applicators that mutate physics properties use this.
    sensor_data : dict
        The mutable sensor row dict produced by ``collect_sensors()``.
        Sensor-corruption applicators (transient spike/dip) modify values
        in-place *before* the row is written to the CSV.
    cube_prim_path : str
        USD path of the workpiece prim (e.g. ``"/World/pick_cube"``).
    robot_prim_path : str
        USD path of the robot articulation root.
    joint_names : list[str]
        Ordered joint names (e.g. ``["shoulder_pan"..., "wrist_3"]``).
    sim_dt : float
        Physics timestep in seconds.
    episode_step : int
        Current step within the episode (0-based).
    state_machine : str
        Current state-machine state (e.g. ``"MOVE_TO_WP"``, ``"ATTACH"``).
    extra : dict
        Arbitrary task-specific data applicators may need.
    """

    stage: Any = None
    sensor_data: dict = field(default_factory=dict)
    cube_prim_path: str = ""
    robot_prim_path: str = ""
    joint_names: list = field(default_factory=list)
    sim_dt: float = 1.0 / 60.0
    episode_step: int = 0
    state_machine: str = ""
    extra: dict = field(default_factory=dict)
