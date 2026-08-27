"""
EpisodeState, captures everything needed to reproduce an episode.

This is the bridge between the simulation and the counterfactual runner.
The simulation fills in an EpisodeState at the start of each episode,
and the runner uses it to replay the episode with identical initial
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class EpisodeState:
    """Snapshot of all randomised quantities for one episode.

    The simulation must populate this before stepping.  The counterfactual
    runner passes it back to the simulation's reset callback to reproduce
    the exact same initial conditions.
    """

    # RNG state at the START of the episode (before any per-episode sampling).
    # For np.random.RandomState: call ``rng.get_state()`` to capture.
    rng_state: Any = None

    # Workpiece properties
    cube_dims: Optional[np.ndarray] = None      # [width, depth, height] in metres
    cube_spawn: Optional[np.ndarray] = None     # [x, y, z] in metres
    cube_yaw: float = 0.0                       # radians
    cube_mass: float = 0.0                      # kg
    cube_friction: float = 0.0
    cube_restitution: float = 0.0
    cube_color: Optional[np.ndarray] = None     # [r, g, b] 0-1

    # Gripper pad friction for this episode
    pad_friction: float = 1.5

    # Arbitrary extra state the simulation may need
    extra: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "EpisodeState":
        """Deep-enough copy for replay."""
        return EpisodeState(
            rng_state=self.rng_state,  # tuple, immutable
            cube_dims=self.cube_dims.copy() if self.cube_dims is not None else None,
            cube_spawn=self.cube_spawn.copy() if self.cube_spawn is not None else None,
            cube_yaw=self.cube_yaw,
            cube_mass=self.cube_mass,
            cube_friction=self.cube_friction,
            cube_restitution=self.cube_restitution,
            cube_color=self.cube_color.copy() if self.cube_color is not None else None,
            pad_friction=self.pad_friction,
            extra={k: v for k, v in self.extra.items()})
