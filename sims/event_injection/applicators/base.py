"""
BaseApplicator, abstract base class for all event applicators.

Subclass this to add new event types. Each applicator must implement:
  - sample_params(): randomly sample the event's variables
  - on_step(): apply the event effect for one sim step

Optionally override on_start() / on_end() for setup / teardown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from event_injection.context import SimContext


class BaseApplicator(ABC):
    """Interface that every event applicator must implement."""

    # Override in subclasses to restrict when the event can trigger.
    # Phase indices follow the PickPlaceController convention:
    #   0=above_pick  1=descend  2=settle  3=close
    #   4=lift        5=move_xy  6=lower   7=open
    #   8=retract     9=return
    # None means "any phase" (no restriction).
    valid_phases: list[int] | None = None

    @abstractmethod
    def sample_params(
        self,
        event_def: dict,
        rng: np.random.Generator) -> Dict[str, Any]:
        """Sample concrete parameter values for one event instance.

        Parameters
        ----------
        event_def : dict
            The raw event definition from events.json (includes
            ``variable_constraints``, ``variables``, etc.).
        rng : numpy.random.Generator
            Seeded RNG, use this for all randomness.

        Returns
        -------
        dict
            Sampled parameter values.  May include a special key
            ``_duration`` (int) which the scheduler extracts to set
            the event's active window.  All other keys are passed
            through to ``on_step`` / ``on_start`` / ``on_end``.
        """
        ...

    def on_start(self, params: Dict[str, Any], ctx: SimContext) -> None:
        """Called once when the event first becomes active.

        Use for one-time setup (e.g. storing original physics values
        before mutation).  Default is a no-op.
        """

    @abstractmethod
    def on_step(self, params: Dict[str, Any], ctx: SimContext) -> None:
        """Called every sim step while the event is active.

        This is where the actual injection happens, mutate
        ``ctx.sensor_data``, apply USD changes, etc.
        """
        ...

    def on_end(self, params: Dict[str, Any], ctx: SimContext) -> None:
        """Called once after the event's last active step.

        Use to restore original values (e.g. reset mass back to
        pre-injection value).  Default is a no-op.
        """

    def on_episode_setup(self, params: Dict[str, Any], ctx: SimContext) -> None:
        """Called at episode start after scheduling, before simulation.

        Use for pre-placing visible objects that will be used later
        when the event triggers.  Default is a no-op.
        """

    def on_episode_cleanup(self, params: Dict[str, Any], ctx: SimContext) -> None:
        """Called at episode end during scheduler reset.

        Use for removing prims that were spawned during the episode.
        Default is a no-op.
        """
