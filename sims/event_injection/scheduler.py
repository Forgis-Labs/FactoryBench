"""
EventScheduler, loads event definitions, schedules random events per episode,
and dispatches active events each step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from event_injection.context import SimContext
from event_injection.applicators.base import BaseApplicator


# -----------------------------------------------------------------------
# Scheduled event record
# -----------------------------------------------------------------------

@dataclass
class ScheduledEvent:
    """One concrete event instance scheduled for a specific step range."""

    event_id: int
    event_name: str
    trigger_step: int
    duration: int                    # number of steps the event is active
    params: Dict[str, Any]          # sampled variable values
    applicator: BaseApplicator
    _started: bool = field(default=False, repr=False)
    _ended: bool = field(default=False, repr=False)

    @property
    def end_step(self) -> int:
        return self.trigger_step + self.duration - 1

    def is_active(self, step: int) -> bool:
        return self.trigger_step <= step <= self.end_step

    def as_log_dict(self) -> dict:
        """Flat dict suitable for CSV / JSON logging."""
        return {
            "event_id": self.event_id,
            "event_name": self.event_name,
            "trigger_step": self.trigger_step,
            "duration": self.duration,
            **{f"event_param_{k}": v for k, v in self.params.items()},
        }


# -----------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------

class EventScheduler:
    """Loads events.json, schedules randomised events, dispatches per step.

    Parameters
    ----------
    events_json_path : str | Path
        Path to the events definition JSON file.
    task_name : str
        Only events whose ``tasks`` list includes this name are eligible.
    applicators : dict[int, BaseApplicator]
        Mapping from event id → applicator instance.  Events without a
        registered applicator are silently skipped during scheduling.
    rng_seed : int
        Seed for the internal NumPy RNG.
    num_events_range : tuple[int, int]
        Min/max number of events to schedule per episode (inclusive).
    margin_steps : int
        Minimum margin from episode start/end when placing trigger steps.
    """

    def __init__(
        self,
        events_json_path: str | Path,
        task_name: str,
        applicators: Dict[int, BaseApplicator],
        rng_seed: int = 0,
        num_events_range: Tuple[int, int] = (0, 2),
        margin_steps: int = 30,
        force_event_id: Optional[int] = None):
        self._rng = np.random.default_rng(rng_seed)
        self._task = task_name
        self._applicators = applicators
        self._num_range = num_events_range
        self._margin = margin_steps
        self._force_event_id = force_event_id

        # Load and filter event definitions
        with open(events_json_path) as f:
            all_events = json.load(f)
        self._eligible: List[dict] = [
            e for e in all_events
            if task_name in e.get("tasks", []) and e["id"] in applicators
        ]
        eligible_names = [e["name"] for e in self._eligible]
        print(f"[EventScheduler] task={task_name}  "
              f"eligible events ({len(self._eligible)}): {eligible_names}")
        if self._force_event_id is not None:
            print(f"[EventScheduler] DEBUG: forcing event id={self._force_event_id} every episode")

        # Per-episode state
        self._scheduled: List[ScheduledEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_phase_boundaries(self, boundaries: List[int]) -> None:
        """Set the cumulative step boundaries for each phase.

        Parameters
        ----------
        boundaries : list of int
            ``boundaries[i]`` is the first step of phase *i*.
            Must have len == num_phases + 1, where the last entry is the
            total episode length.  Example for 10 phases::

                [0, 125, 325, 350, 475, 600, 850, 975, 985, 1048, 1131]
        """
        self._phase_bounds = boundaries

    def _phase_step_range(self, phases: List[int]) -> Tuple[int, int]:
        """Return (earliest_step, latest_step) covering the given phases."""
        if not hasattr(self, "_phase_bounds") or self._phase_bounds is None:
            return (self._margin, 10000)
        lo = self._phase_bounds[min(phases)]
        hi = self._phase_bounds[max(phases) + 1] if max(phases) + 1 < len(self._phase_bounds) \
            else self._phase_bounds[-1]
        return (lo, hi)

    def schedule_episode(self, max_episode_steps: int) -> List[ScheduledEvent]:
        """Roll the dice for one episode.  Call at episode start.

        If ``force_event_id`` was set in the constructor, exactly one
        instance of that event is scheduled every episode (debug mode).

        Trigger steps are constrained to the applicator's ``valid_phases``
        if phase boundaries have been set via ``set_phase_boundaries()``.
        """
        self._scheduled.clear()

        if not self._eligible or max_episode_steps < 2 * self._margin:
            return self._scheduled

        # Debug mode: force a specific event
        if self._force_event_id is not None:
            forced = [e for e in self._eligible if e["id"] == self._force_event_id]
            if not forced:
                print(f"[EventScheduler] WARNING: forced event id={self._force_event_id} "
                      f"not found in eligible events")
                return self._scheduled
            defn = forced[0]
            applicator = self._applicators[defn["id"]]
            params = applicator.sample_params(defn, self._rng)
            duration = max(1, int(params.pop("_duration", 1)))
            lo, hi = self._get_trigger_range(applicator, duration, max_episode_steps)
            # For persistent events (very large duration), don't shrink the
            # trigger range, pick any step within the valid phases.
            trigger_hi = max(lo + 1, hi - duration + 1) if duration < hi - lo else hi
            trigger = int(self._rng.integers(lo, max(lo + 1, trigger_hi)))
            se = ScheduledEvent(
                event_id=defn["id"],
                event_name=defn["name"],
                trigger_step=trigger,
                duration=duration,
                params=params,
                applicator=applicator)
            self._scheduled.append(se)
            print(f"[EventScheduler] FORCED: {se.event_name}@step{se.trigger_step}"
                  f"(dur={se.duration})")
            return self._scheduled

        n_events = int(self._rng.integers(
            self._num_range[0], self._num_range[1] + 1
        ))

        # Track the earliest allowed start for the next event so they
        # never overlap.  Includes a small gap between events.
        _GAP = 5  # minimum steps between consecutive events
        next_available = self._margin

        for _ in range(n_events):
            defn = self._eligible[
                int(self._rng.integers(0, len(self._eligible)))
            ]
            applicator = self._applicators[defn["id"]]

            # Sample event-specific parameters
            params = applicator.sample_params(defn, self._rng)

            # Determine duration (some events are instantaneous)
            duration = max(1, int(params.pop("_duration", 1)))

            # Constrain trigger step to valid phases
            lo, hi = self._get_trigger_range(applicator, duration, max_episode_steps)
            earliest_start = max(next_available, lo)
            if duration >= hi - lo:
                # Persistent event, can start anywhere in the valid range
                latest_start = hi - 1
            else:
                latest_start = min(hi - duration, max_episode_steps - self._margin - duration)
            if earliest_start > latest_start:
                # No room left for this event, skip it
                continue
            trigger = int(self._rng.integers(earliest_start, latest_start + 1))

            se = ScheduledEvent(
                event_id=defn["id"],
                event_name=defn["name"],
                trigger_step=trigger,
                duration=duration,
                params=params,
                applicator=applicator)
            self._scheduled.append(se)

            # Next event can only start after this one ends
            next_available = trigger + duration + _GAP

        if self._scheduled:
            summary = ", ".join(
                f"{s.event_name}@step{s.trigger_step}(dur={s.duration})"
                for s in self._scheduled
            )
            print(f"[EventScheduler] Scheduled {len(self._scheduled)} events: {summary}")

        return self._scheduled

    def _get_trigger_range(
        self, applicator: BaseApplicator, duration: int, max_steps: int) -> Tuple[int, int]:
        """Return the (earliest, latest) step range for an event trigger."""
        if applicator.valid_phases is not None and hasattr(self, "_phase_bounds") \
                and self._phase_bounds is not None:
            lo, hi = self._phase_step_range(applicator.valid_phases)
        else:
            lo, hi = self._margin, max_steps - self._margin
        return (lo, hi)

    def setup_episode(self, ctx: SimContext) -> None:
        """Call after schedule_episode() to let applicators pre-place
        objects or do other visual setup before simulation begins."""
        for se in self._scheduled:
            try:
                se.applicator.on_episode_setup(se.params, ctx)
            except Exception as e:
                print(f"[EventScheduler] on_episode_setup failed for "
                      f"{se.event_name}: {e}")

    def step(self, current_step: int, ctx: SimContext) -> List[ScheduledEvent]:
        """Call every sim step.  Returns list of events that were active."""
        active = []
        for se in self._scheduled:
            if se._ended:
                continue
            if se.is_active(current_step):
                if not se._started:
                    se.applicator.on_start(se.params, ctx)
                    se._started = True
                se.applicator.on_step(se.params, ctx)
                active.append(se)
            elif se._started and current_step > se.end_step:
                se.applicator.on_end(se.params, ctx)
                se._ended = True
        return active

    def reset(self, ctx: SimContext | None = None):
        """Call at episode end to clean up applicator state."""
        _ctx = ctx or SimContext()
        for se in self._scheduled:
            if se._started and not se._ended:
                # Force cleanup of any still-active events
                try:
                    se.applicator.on_end(se.params, _ctx)
                except Exception:
                    pass
            try:
                se.applicator.on_episode_cleanup(se.params, _ctx)
            except Exception:
                pass
        self._scheduled.clear()

    def get_scheduled_events(self) -> List[ScheduledEvent]:
        """Return the current episode's scheduled events (for logging)."""
        return list(self._scheduled)

    def get_active_events_log(self, current_step: int) -> str:
        """Return a JSON-encodable string of active events at this step."""
        active = [
            se.as_log_dict()
            for se in self._scheduled
            if se.is_active(current_step)
        ]
        if not active:
            return ""
        import json as _json
        return _json.dumps(active)
