"""
CounterfactualRunner, orchestrates paired baseline/counterfactual episodes.

This is the core of the counterfactual data generation pipeline.  It is
simulation-agnostic: the caller provides callback functions that handle
the actual simulation logic.

Usage sketch (from a task-specific script)::

    runner = CounterfactualRunner(
        output_dir="counterfactual_data",
        events_json_path="events.json",
        task_name="pick_and_place",
        applicators=BUILTIN_APPLICATORS,
        seed=42)

    # The runner calls these callbacks:
    runner.run(
        num_episodes=100,
        reset_fn=my_reset,        # (EpisodeState, inject_events: bool) → None
        step_fn=my_step,          # () → dict (sensor_row)
        is_done_fn=my_is_done,    # () → (bool, bool, str)  (done, success, reason)
        get_state_fn=my_state,    # () → EpisodeState
        max_steps=1500)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from event_injection.scheduler import EventScheduler
from event_injection.applicators.base import BaseApplicator
from counterfactual.episode_state import EpisodeState


class CounterfactualRunner:
    """Runs paired baseline + counterfactual episodes and logs both.

    Parameters
    ----------
    output_dir : str | Path
        Root directory for output.  Each episode creates a subdirectory.
    events_json_path : str | Path
        Path to events.json definitions.
    task_name : str
        Task name for filtering eligible events.
    applicators : dict[int, BaseApplicator]
        Registered event applicators.
    seed : int
        Base RNG seed.
    phase_boundaries : list[int] | None
        Step boundaries for each phase (passed to scheduler).
    """

    def __init__(
        self,
        output_dir: str | Path,
        events_json_path: str | Path,
        task_name: str,
        applicators: Dict[int, BaseApplicator],
        seed: int = 0,
        phase_boundaries: Optional[List[int]] = None):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._events_json = str(events_json_path)
        self._task_name = task_name
        self._applicators = applicators
        self._seed = seed
        self._phase_boundaries = phase_boundaries

        # Scheduler for the counterfactual runs (baseline has no scheduler)
        self._scheduler = EventScheduler(
            events_json_path=self._events_json,
            task_name=self._task_name,
            applicators=self._applicators,
            rng_seed=seed + 99999,
            num_events_range=(1, 1),  # exactly 1 event per counterfactual
        )
        if self._phase_boundaries:
            self._scheduler.set_phase_boundaries(self._phase_boundaries)

    def run(
        self,
        num_episodes: int,
        reset_fn: Callable[[EpisodeState, bool], None],
        step_fn: Callable[[], dict],
        is_done_fn: Callable[[], Tuple[bool, bool, str]],
        get_state_fn: Callable[[], EpisodeState],
        event_step_fn: Optional[Callable[[int, dict], Optional[List]]] = None,
        max_steps: int = 1500):
        """Run the counterfactual generation loop.

        Parameters
        ----------
        num_episodes : int
            Number of episode pairs to generate.
        reset_fn : callable(EpisodeState, inject_events: bool) → None
            Reset the simulation to the given state.  If ``inject_events``
            is False, run a clean baseline.  If True, the runner will
            handle event injection via ``event_step_fn``.
        step_fn : callable() → dict
            Advance the simulation by one step and return the sensor row
            dict (same format as ``collect_sensors`` output).
        is_done_fn : callable() → (done: bool, success: bool, reason: str)
            Check if the current episode is finished.
        get_state_fn : callable() → EpisodeState
            Capture the current episode's initial state after reset.
        event_step_fn : callable(ep_step, sensor_row) → list | None
            Called each step during counterfactual runs.  Should call
            the event scheduler's step() and return active events.
            If None, the runner manages the scheduler directly.
        max_steps : int
            Maximum steps per episode before forced termination.
        """
        import csv

        for ep_idx in range(num_episodes):
            ep_dir = self._output_dir / f"episode_{ep_idx:04d}"

            # ----------------------------------------------------------
            # Phase 1: Baseline (no events)
            # ----------------------------------------------------------
            print(f"\n[Counterfactual] Episode {ep_idx}, BASELINE")
            reset_fn(None, False)  # None = fresh random state
            state = get_state_fn()

            baseline_dir = ep_dir / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline_rows = []

            for s in range(max_steps):
                sensor_row = step_fn()
                baseline_rows.append(sensor_row)
                done, success, reason = is_done_fn()
                if done:
                    break

            self._write_csv(baseline_dir / "steps.csv", baseline_rows)
            self._write_summary(baseline_dir / "episodes.csv",
                                ep_idx, success, reason, len(baseline_rows), state)

            # ----------------------------------------------------------
            # Phase 2: Counterfactual (replay with event injection)
            # ----------------------------------------------------------
            print(f"[Counterfactual] Episode {ep_idx}, COUNTERFACTUAL")
            reset_fn(state, True)  # replay same initial conditions

            self._scheduler.reset()
            scheduled = self._scheduler.schedule_episode(max_steps)

            cf_dir = ep_dir / "counterfactual"
            cf_dir.mkdir(parents=True, exist_ok=True)
            cf_rows = []

            from event_injection.context import SimContext

            for s in range(max_steps):
                sensor_row = step_fn()

                # Run event injection
                if event_step_fn is not None:
                    active = event_step_fn(s, sensor_row)
                else:
                    # Default: caller didn't provide event_step_fn,
                    # use scheduler directly with a minimal context
                    ctx = SimContext(sensor_data=sensor_row, episode_step=s)
                    active = self._scheduler.step(s, ctx)
                if active:
                    evt = active[0]
                    sensor_row["event_id"] = evt.event_id
                    pstr = ";".join(
                        f"{k}={v}" for k, v in evt.params.items()
                        if not k.startswith("_")
                    )
                    sensor_row["event_params"] = pstr

                cf_rows.append(sensor_row)
                done, success, reason = is_done_fn()
                if done:
                    break

            self._scheduler.reset()

            self._write_csv(cf_dir / "steps.csv", cf_rows)
            self._write_summary(cf_dir / "episodes.csv",
                                ep_idx, success, reason, len(cf_rows), state)

            # Write event metadata
            event_meta = []
            for se in scheduled:
                event_meta.append({
                    "event_id": se.event_id,
                    "event_name": se.event_name,
                    "trigger_step": se.trigger_step,
                    "duration": se.duration,
                    "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                               for k, v in se.params.items()
                               if not k.startswith("_")},
                })
            with open(cf_dir / "event.json", "w") as f:
                json.dump(event_meta, f, indent=2)

            print(f"[Counterfactual] Episode {ep_idx} done, "
                  f"baseline: {len(baseline_rows)} steps, "
                  f"counterfactual: {len(cf_rows)} steps")

        print(f"\n[Counterfactual] Finished {num_episodes} episode pairs → {self._output_dir}")

    @staticmethod
    def _write_csv(path: Path, rows: List[dict]):
        """Write a list of row dicts to a CSV file."""
        import csv
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_summary(path: Path, episode: int, success: bool, reason: str,
                       num_steps: int, state: EpisodeState):
        """Write a single-row episode summary CSV."""
        import csv
        row = {
            "episode": episode,
            "success": int(success),
            "reason": reason,
            "episode_steps": num_steps,
            "cube_mass_kg": f"{state.cube_mass:.6f}" if state.cube_mass else "",
            "cube_friction_coeff": f"{state.cube_friction:.6f}" if state.cube_friction else "",
            "cube_restitution_coeff": f"{state.cube_restitution:.6f}",
        }
        if state.cube_dims is not None:
            row["cube_width_m"] = f"{state.cube_dims[0]:.6f}"
            row["cube_depth_m"] = f"{state.cube_dims[1]:.6f}"
            row["cube_height_m"] = f"{state.cube_dims[2]:.6f}"
        if state.cube_spawn is not None:
            row["cube_spawn_x_m"] = f"{state.cube_spawn[0]:.6f}"
            row["cube_spawn_y_m"] = f"{state.cube_spawn[1]:.6f}"
            row["cube_spawn_z_m"] = f"{state.cube_spawn[2]:.6f}"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
