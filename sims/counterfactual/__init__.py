"""
Counterfactual data generation pipeline.

Runs each episode twice:
  1. **Baseline**, no events injected, normal execution.
  2. **Counterfactual**, same initial conditions (cube dims, mass, friction,
     spawn position, yaw), but with a random event injected.

Data is logged into per-episode subdirectories::

    output_dir/
        episode_000/
            baseline/
                steps.csv
                episodes.csv
            counterfactual/
                steps.csv
                episodes.csv
                event.json      # which event was injected + params
        episode_001/
            ...

The pipeline is simulation-agnostic: it receives callback functions for
episode setup, stepping, and teardown, so it can be reused across
different tasks (pick_and_place, screwing, etc.).
"""

from counterfactual.runner import CounterfactualRunner

__all__ = ["CounterfactualRunner"]
