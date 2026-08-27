"""
FactoryBench / screwing / episode_logger.py
Shared per-step CSV sensor logger for screw-driving tasks.

Extends the pick_and_place episode_logger with screwing-specific columns:
  screw_depth_m        bolt insertion depth into fixture, m
  screw_angle_rad      cumulative wrist rotation during screwing, rad

All base columns (joint state, EEF, workpiece, gripper, contacts, etc.)
are inherited.  Workpiece columns use "cube_" prefixes for schema
compatibility but contain bolt data.
"""

import csv
import os
import sys
import time
from pathlib import Path

# Import the base pick_and_place episode_logger via importlib to avoid
# circular import (this file is also named episode_logger.py).
import importlib.util as _ilu

_PP_LOGGER_PATH = str(Path(__file__).parent.parent / "pick_and_place" / "episode_logger.py")
_spec = _ilu.spec_from_file_location("_pp_episode_logger", _PP_LOGGER_PATH)
_pp_logger = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pp_logger)

JOINT_NAMES       = _pp_logger.JOINT_NAMES
_BASE_COLUMNS     = _pp_logger.COLUMNS
collect_sensors   = _pp_logger.collect_sensors
get_task_phase    = _pp_logger.get_task_phase
_BaseEpisodeLogger = _pp_logger.EpisodeLogger

# ---------------------------------------------------------------------------
# Extended columns for screwing
# ---------------------------------------------------------------------------

_SCREW_COLUMNS = [
    "screw_depth_m",
    "screw_angle_rad",
]

COLUMNS = _BASE_COLUMNS + _SCREW_COLUMNS


class EpisodeLogger(_BaseEpisodeLogger):
    """Extends the base EpisodeLogger with screwing-specific columns."""

    def __init__(self, log_dir: str):
        # Let base init set up everything (correct attr names, summary CSV, etc.)
        super().__init__(log_dir)
        # Re-create the steps CSV writer with extended columns so the
        # screwing-specific fields (screw_depth_m, screw_angle_rad) are written.
        self._steps_fh.close()
        steps_path = os.path.join(log_dir, "steps.csv")
        self._steps_fh = open(steps_path, "w", newline="", buffering=1)
        self._steps_writer = csv.DictWriter(
            self._steps_fh, fieldnames=COLUMNS, extrasaction="ignore"
        )
        self._steps_writer.writeheader()
