"""
FactoryBench event injection framework.

Provides a modular system for injecting randomized disruption events
into Isaac Sim simulations.  Task- and robot-agnostic: applicators
register themselves by event ID and the scheduler picks from events
that match the current task.

Quick-start
-----------
    from event_injection import EventScheduler, SimContext, BUILTIN_APPLICATORS

    scheduler = EventScheduler(
        events_json_path="FactoryBench/events.json",
        task_name="pick_and_place",
        applicators=BUILTIN_APPLICATORS,
        rng_seed=42,
    )
    # At episode start:
    scheduler.schedule_episode(max_episode_steps=3600)
    # Every sim step:
    ctx = SimContext(...)
    scheduler.step(current_step, ctx)
    # At episode end:
    scheduler.reset()
"""

from event_injection.context import SimContext
from event_injection.scheduler import EventScheduler
from event_injection.applicators import BUILTIN_APPLICATORS

__all__ = ["EventScheduler", "SimContext", "BUILTIN_APPLICATORS"]
