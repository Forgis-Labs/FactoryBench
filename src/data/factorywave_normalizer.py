"""
Normalize FactoryWave (ANON-ORG/FactoryWave) dataset to UR3e schema format.

Reads ur_signals parquet files (pick-and-place and peg-in-hole episodes recorded
from a live UR3 robot) and the episode metadata table, then outputs per-episode
JSON files in the standardized schema used by the FactoryBench pipeline.

For counterfactual episodes:
  - Groups CFs by their baseline episode
  - Computes KL divergence between each CF's pre-fault segment and the baseline
  - Keeps only the CF with the lowest KL divergence (best match to baseline)
  - Outputs as {"baseline": [...], "counterfactual": [...]} like simulations

For non-counterfactual episodes (normal, fault, trajectory-opt):
  - Outputs as a flat list of rows

Input:
    data/factorywave/data/ur_signals.parquet (or ur_signals_10hz/ + ur_signals_125hz/)
    data/factorywave/data/episode.parquet

Output:
    <output>/factorywave/<episode_id>.json
    <output>/factorywave/<episode_id>_metadata.json

Usage:
    python -m src.data.factorywave_normalizer \\
        --input data/factorywave/data \\
        --output data/normalized_episodes
"""

import ast
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pandas and pyarrow are required. Install with: pip install pandas pyarrow")

from src.data._decimation import decimate_dataframe

logger = logging.getLogger(__name__)


class _NaNSafeEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


# FactoryWave ur_signals column -> UR3e schema column
def _build_column_mapping() -> Dict[str, Optional[str]]:
    m: Dict[str, Optional[str]] = {}

    for i in range(6):
        # INTENT, joint commands
        m[f"setpoint_pos_{i}"] = f"target_joint_{i}"
        m[f"setpoint_speed_{i}"] = f"target_joint_vel_{i}"
        m[f"setpoint_acc_{i}"] = f"target_joint_accel_{i}"

        # OUTCOME, joint feedback
        m[f"feedback_pos_{i}"] = f"joint_{i}"
        m[f"feedback_speed_{i}"] = f"joint_vel_{i}"

        # OUTCOME, effort / current
        m[f"effort_current_{i}"] = f"joint_current_{i}"
        m[f"effort_target_current_{i}"] = f"target_joint_current_{i}"
        m[f"effort_target_torque_{i}"] = None

        # OUTCOME, controller output
        m[f"control_output_{i}"] = f"joint_control_output_{i}"

        # CONTEXT, per-joint
        m[f"joint_temp_{i}"] = f"joint_temp_{i}"
        m[f"joint_mode_{i}"] = f"joint_mode_{i}"
        m[f"joint_voltage_{i}"] = None

    # INTENT, TCP commands
    for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        m[f"setpoint_tcp_{i}"] = f"target_tcp_{axis}"
        m[f"setpoint_tcp_speed_{i}"] = f"target_tcp_speed_{axis}"

    # OUTCOME, TCP feedback
    for i, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        m[f"feedback_tcp_{i}"] = f"tcp_{axis}"
        m[f"feedback_tcp_speed_{i}"] = f"tcp_speed_{axis}"

    # OUTCOME, forces/torques
    for i, axis in enumerate(["x", "y", "z"]):
        m[f"true_force_{i}"] = f"tcp_force_{axis}"
        m[f"true_force_{i + 3}"] = f"tcp_torque_{axis}"
    for i in range(6):
        m[f"est_contact_force_{i}"] = None

    # OUTCOME, vibration (tool accelerometer)
    for i, axis in enumerate(["x", "y", "z"]):
        m[f"vibration_{i}"] = f"tool_accel_{axis}"

    m["acoustic_0"] = None
    m["protective_stop_state"] = None

    # INTENT, gripper
    m["gripper_command"] = "force"  # gripper force as proxy for grip command

    # CONTEXT, system-level
    m["robot_mode"] = "robot_mode"
    m["safety_mode"] = "safety_mode"
    m["digital_input_bits"] = "digital_inputs"
    m["digital_output_bits"] = "digital_outputs"
    m["runtime_state"] = "runtime_state"
    m["main_voltage"] = "main_voltage"
    m["robot_voltage"] = "robot_voltage"
    m["robot_current"] = "robot_current"
    m["speed_scaling"] = "speed_scaling"
    m["target_speed_fraction"] = "target_speed_fraction"
    m["tool_momentum"] = "momentum"

    return m


COLUMN_MAPPING = _build_column_mapping()

# Integer-typed schema columns
_INT_COLS = {
    "joint_mode_0", "joint_mode_1", "joint_mode_2",
    "joint_mode_3", "joint_mode_4", "joint_mode_5",
    "robot_mode", "safety_mode", "runtime_state",
    "digital_input_bits", "digital_output_bits",
}

# Columns that should not be anti-alias filtered during decimation
_NON_CONTINUOUS = {
    "time", "created_at", "updated_at",
    "episode_id", "fault", "task_phase", "skill_index",
    "status", "busy", "connected", "streaming",
    "robot_mode", "safety_mode", "robot_status", "safety_status",
    "runtime_state", "digital_inputs", "digital_outputs",
    "frame_count", "operation_counter", "grip_detected",
    "joint_mode_0", "joint_mode_1", "joint_mode_2",
    "joint_mode_3", "joint_mode_4", "joint_mode_5",
}

TARGET_HZ = 10  # Target sampling rate for normalized output

# Continuous signal columns used for KL divergence computation
_KL_SIGNAL_COLS = [
    "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
    "joint_vel_0", "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4", "joint_vel_5",
    "joint_current_0", "joint_current_1", "joint_current_2", "joint_current_3", "joint_current_4", "joint_current_5",
    "tcp_x", "tcp_y", "tcp_z",
    "tcp_force_x", "tcp_force_y", "tcp_force_z",
]


def compute_kl_divergence(baseline_df: pd.DataFrame, cf_df: pd.DataFrame, n_rows: int) -> float:
    """Compute KL divergence between baseline and CF pre-fault segments.

    Uses a histogram-based approximation of KL(baseline || cf) averaged across
    signal columns. Only compares the first n_rows of each (the pre-fault segment).
    Returns float('inf') if comparison is not possible.
    """
    import numpy as np

    cols = [c for c in _KL_SIGNAL_COLS if c in baseline_df.columns and c in cf_df.columns]
    if not cols:
        return float("inf")

    bl = baseline_df.iloc[:n_rows]
    cf = cf_df.iloc[:n_rows]

    if len(bl) < 5 or len(cf) < 5:
        return float("inf")

    kl_sum = 0.0
    valid_cols = 0

    for col in cols:
        bl_vals = pd.to_numeric(bl[col], errors="coerce").dropna().values
        cf_vals = pd.to_numeric(cf[col], errors="coerce").dropna().values

        if len(bl_vals) < 5 or len(cf_vals) < 5:
            continue

        # Shared bin edges from combined range
        all_vals = np.concatenate([bl_vals, cf_vals])
        n_bins = min(30, max(5, len(bl_vals) // 5))
        edges = np.linspace(all_vals.min() - 1e-9, all_vals.max() + 1e-9, n_bins + 1)

        # Compute histograms as probability distributions
        p, _ = np.histogram(bl_vals, bins=edges, density=True)
        q, _ = np.histogram(cf_vals, bins=edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()

        # KL(P || Q)
        kl = np.sum(p * np.log(p / q))
        kl_sum += kl
        valid_cols += 1

    if valid_cols == 0:
        return float("inf")
    return kl_sum / valid_cols


# Fault labels that were not flipped during recording and must be synthesized
# at the start of the peg-in-hole insert phase during normalization.
_FAULT_FLIP_INJECTION_IDS = {32, 34}
_PEG_IN_HOLE_INSERT_PHASE_LABEL = "1"

# Fault labels (joint-limit, self-collision) where the protective stop is the
# ground-truth onset and the fault_label flip should be re-aligned to the
# safety_mode NORMAL(1) -> PROTECTIVE_STOP(3) transition. Anything else in the
# raw `fault` column is overridden when such a transition is present.
_FAULT_FLIP_AT_PROTECTIVE_STOP_IDS = {19, 37}
_SAFETY_MODE_NORMAL = 1
_SAFETY_MODE_PROTECTIVE_STOP = 3


_PEG_IN_HOLE_FAULTS = {32, 33, 34, 35, 36, 39}
_SCREWING_FAULTS = {1, 2, 3, 4, 5}


def _infer_task_from_phases(rows: List[Dict[str, Any]], fault_id: Optional[int] = None) -> str:
    """Infer the task type from the number of distinct task_phase values."""
    phases = set()
    for r in rows:
        p = r.get("task_phase")
        if p is not None:
            phases.add(str(p))
    max_phase = max((int(p) for p in phases if p.isdigit()), default=-1)
    if max_phase >= 9:
        return "pick_and_place"
    # 9 phases (0-8) is ambiguous between screwing and peg_in_hole;
    # use fault_id to disambiguate if available.
    if fault_id is not None:
        if fault_id in _PEG_IN_HOLE_FAULTS:
            return "peg_in_hole"
        if fault_id in _SCREWING_FAULTS:
            return "screwing"
    return "unknown"


def _inject_fault_flip_if_missing(
    rows: List[Dict[str, Any]],
    fault_id: Optional[int]) -> bool:
    """For fault 32/34 peg-in-hole episodes the raw `fault` column stays 0.
    If no flip is present, mark every row from the first `insert` phase row
    (task_phase == "1") onward with the metadata fault_id. Mutates `rows`.
    Returns True if an injection was performed.
    """
    if fault_id not in _FAULT_FLIP_INJECTION_IDS:
        return False
    if any((r.get("fault_label") or 0) != 0 for r in rows):
        return False
    for i, r in enumerate(rows):
        if r.get("task_phase") == _PEG_IN_HOLE_INSERT_PHASE_LABEL:
            for r2 in rows[i:]:
                r2["fault_label"] = fault_id
            return True
    return False


def _align_fault_flip_to_protective_stop(
    rows: List[Dict[str, Any]],
    fault_id: Optional[int]) -> bool:
    """For fault 19 (joint-limit) and 37 (self-collision) episodes, force
    fault_label/event to flip from 0 to fault_id at the first safety_mode
    NORMAL(1) -> PROTECTIVE_STOP(3) transition. Mutates `rows`. Returns True
    if a transition was found and the rewrite was applied; otherwise leaves
    rows untouched (e.g. when the protective stop is outside the recorded
    window and safety_mode never reaches 3).
    """
    if fault_id not in _FAULT_FLIP_AT_PROTECTIVE_STOP_IDS:
        return False
    flip_idx: Optional[int] = None
    prev = None
    for i, r in enumerate(rows):
        sm = r.get("safety_mode")
        if prev == _SAFETY_MODE_NORMAL and sm == _SAFETY_MODE_PROTECTIVE_STOP:
            flip_idx = i
            break
        prev = sm
    if flip_idx is None:
        return False
    for r in rows[:flip_idx]:
        r["fault_label"] = 0
        r["event"] = 0
    for r in rows[flip_idx:]:
        r["fault_label"] = fault_id
        r["event"] = fault_id
    return True


def _to_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_episode_df(
    ep_df: pd.DataFrame,
    first_timestamp_us: int,
    metadata_fault_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Convert a single episode DataFrame to list of UR3e schema dicts.

    Args:
        metadata_fault_id: If provided, used as fault_label for all rows when
            the signal's fault column is 0 throughout (physical/config faults
            that aren't encoded in the signal data).
    """
    rows: List[Dict[str, Any]] = []

    # Check if signal fault column has any non-zero values
    signal_faults = pd.to_numeric(ep_df["fault"], errors="coerce").fillna(0)
    signal_has_fault = (signal_faults != 0).any()

    for _, row in ep_df.iterrows():
        result: Dict[str, Any] = {}

        # Timestamp → ms since episode start
        t = row.get("time")
        if pd.notna(t):
            t_us = int(pd.Timestamp(t).value // 1000)  # ns -> us
            result["timestamp_ms"] = (t_us - first_timestamp_us) // 1000
        else:
            result["timestamp_ms"] = None

        # Map columns
        for schema_col, src_col in COLUMN_MAPPING.items():
            if src_col is None or src_col not in row.index:
                result[schema_col] = None
            else:
                val = row[src_col]
                if schema_col in _INT_COLS:
                    result[schema_col] = _to_int(val)
                else:
                    result[schema_col] = _to_float(val)

        # Fault label: use signal if it has transitions, otherwise use metadata
        signal_fault = _to_int(row.get("fault")) or 0
        if signal_has_fault:
            result["fault_label"] = signal_fault
        else:
            result["fault_label"] = metadata_fault_id if metadata_fault_id else signal_fault

        result["task_phase"] = str(row.get("task_phase")) if pd.notna(row.get("task_phase")) else None
        result["event"] = result["fault_label"]

        rows.append(result)

    return rows


def load_episode_metadata(ep_row: pd.Series) -> Dict[str, Any]:
    """Parse episode metadata from the episode table row."""
    meta_str = ep_row.get("episode_metadata")
    if isinstance(meta_str, str):
        try:
            meta = ast.literal_eval(meta_str)
        except (ValueError, SyntaxError):
            meta = {}
    elif isinstance(meta_str, dict):
        meta = meta_str
    else:
        meta = {}

    # Fallback: get fault_id from fault_metadata if not in episode_metadata
    fault_id = meta.get("fault_id")
    if fault_id is None:
        fm_str = ep_row.get("fault_metadata")
        if isinstance(fm_str, str):
            try:
                fm = ast.literal_eval(fm_str)
                fault_id = fm.get("fault_id") if isinstance(fm, dict) else None
            except (ValueError, SyntaxError):
                pass
        elif isinstance(fm_str, dict):
            fault_id = fm_str.get("fault_id")

    # Infer condition from fault_id if not in metadata
    condition = meta.get("condition", "unknown")
    if condition == "unknown" and fault_id is not None:
        condition = "fault"
    elif condition == "unknown" and fault_id is None:
        condition = "normal"

    return {
        "episode_id": str(ep_row.get("id", "")),
        "robot_type": meta.get("robot_model", "ur3"),
        "data_source": "factorywave",
        "task": meta.get("task", "unknown"),
        "condition": condition,
        "fault_id": fault_id,
        "weight_of_box": meta.get("weight_of_box"),
        "shape_of_box": meta.get("shape_of_box"),
        "position_of_box": meta.get("position_of_box"),
        "gripper_model": meta.get("gripper_model"),
        "payload_mass_configured": meta.get("payload_mass_configured"),
        "correct_payload_mass": meta.get("correct_payload_mass"),
        "payload_cog_configured": meta.get("payload_cog_configured"),
        "correct_payload_cog": meta.get("correct_payload_cog"),
        "tcp_offset_configured": meta.get("tcp_offset_configured"),
        "correct_tcp_offset": meta.get("correct_tcp_offset"),
        "tcp_orientation_configured": meta.get("tcp_orientation_configured"),
        "offset": meta.get("offset"),
        "link_affected": meta.get("link_affected"),
        "object_description": meta.get("object_description"),
        "damage_description": meta.get("damage_description"),
        "material_description": meta.get("material_description"),
        "description_of_experiment": meta.get("description_of_experiment"),
        "counterfactual": bool(ep_row.get("counterfactual", False)),
        "cf_fault_id": _to_int(ep_row.get("cf_fault_id")),
        "cf_injection_timestep": _to_int(ep_row.get("cf_injection_timestep")),
        "cf_injection_time_s": _to_float(ep_row.get("cf_injection_time_s")),
        "cf_baseline_episode_id": str(ep_row.get("cf_baseline_episode_id"))
        if pd.notna(ep_row.get("cf_baseline_episode_id")) else None,
    }


def _decimate_episode(ep_df: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    """Decimate an episode DataFrame to the target Hz.

    Preserves the exact fault onset row: if the 0→non-zero transition in the
    fault column falls between kept samples, the nearest decimated row is
    replaced with the original onset row so the transition boundary is exact.
    """
    ep_df = ep_df.reset_index(drop=True)
    times = ep_df["time"].values.astype("int64")
    if len(times) <= 5:
        return ep_df

    diffs = pd.Series(times).diff().dropna()
    pos_diffs = diffs[diffs > 0]
    if len(pos_diffs) <= 3:
        return ep_df

    med_dt_us = pos_diffs.median()
    hz = 1e6 / med_dt_us
    q = max(1, int(round(hz / target_hz)))
    if q <= 1:
        return ep_df

    # Find fault onset BEFORE decimation
    onset_idx = _find_fault_onset(ep_df)

    # Decimate
    continuous = set(ep_df.columns) - _NON_CONTINUOUS
    decimated = decimate_dataframe(ep_df, q=q, continuous_cols=continuous)

    # Splice in the original onset row if it wasn't kept
    if onset_idx is not None:
        onset_row = ep_df.iloc[[onset_idx]]
        # Find where it should go in the decimated frame
        decimated_onset = onset_idx // q
        if decimated_onset >= len(decimated):
            decimated_onset = len(decimated) - 1

        # Check if the decimated row at that position already has the transition
        dec_faults = pd.to_numeric(decimated["fault"], errors="coerce").fillna(0).values
        already_has_transition = (
            decimated_onset > 0
            and dec_faults[decimated_onset] != 0
            and dec_faults[decimated_onset - 1] == 0
        )

        if not already_has_transition and decimated_onset < len(decimated):
            # Replace the nearest row with the original onset row
            decimated = pd.concat([
                decimated.iloc[:decimated_onset],
                onset_row.reset_index(drop=True),
                decimated.iloc[decimated_onset + 1:],
            ], ignore_index=True)

    return decimated


def _find_fault_onset(ep_df: pd.DataFrame) -> Optional[int]:
    """Find the row index where the fault column becomes non-zero.

    Returns None if fault is never non-zero or is constant throughout.
    """
    faults = pd.to_numeric(ep_df["fault"], errors="coerce").fillna(0).values
    if faults[0] != 0:
        # Fault present from the start, check if it was absent at any point
        # (this means no pre-fault segment exists)
        return None
    for i in range(1, len(faults)):
        if faults[i] != 0:
            return i
    return None


def normalize_dataset(
    input_dir: Path,
    output_dir: Path,
    target_hz: int = TARGET_HZ,
    limit: Optional[int] = None,
    cf_limit: Optional[int] = None,
    tasks: Optional[List[str]] = None) -> None:
    """Normalize FactoryWave ur_signals to per-episode UR3e schema JSON files.

    For counterfactual episodes: selects the best CF per baseline group using
    KL divergence on the pre-fault segment, then outputs as
    {"baseline": [...], "counterfactual": [...]}.

    For other episodes: outputs as a flat list of rows.
    """
    import numpy as np

    # Load episode table
    episode_path = input_dir / "episode.parquet"
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode table not found: {episode_path}")

    logger.info("Loading episode metadata...")
    ep_table = pd.read_parquet(episode_path)

    # Parse metadata to filter by task
    ep_table["_meta"] = ep_table["episode_metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or {})
    )
    ep_table["_task"] = ep_table["_meta"].apply(lambda m: m.get("task") if isinstance(m, dict) else None)
    ep_table["_condition"] = ep_table["_meta"].apply(lambda m: m.get("condition") if isinstance(m, dict) else None)

    if tasks:
        ep_table = ep_table[ep_table["_task"].isin(tasks)]
        logger.info(f"Filtered to tasks {tasks}: {len(ep_table)} episodes")

    ep_lookup = {str(row["id"]): row for _, row in ep_table.iterrows()}

    # Build counterfactual groups: baseline_id -> [cf_episode_ids]
    cf_groups: Dict[str, List[str]] = {}
    cf_episode_ids: set = set()
    for _, row in ep_table.iterrows():
        if row["_condition"] == "counterfactual" and pd.notna(row.get("cf_baseline_episode_id")):
            bl_id = str(row["cf_baseline_episode_id"])
            cf_id = str(row["id"])
            cf_groups.setdefault(bl_id, []).append(cf_id)
            cf_episode_ids.add(cf_id)

    baseline_ids = set(cf_groups.keys())
    logger.info(f"Episode metadata loaded: {len(ep_lookup)} episodes, "
                f"{len(cf_groups)} CF groups ({len(cf_episode_ids)} CF episodes)")

    # Find signal parquet files, prefer pre-downsampled 10hz files
    signal_paths = []
    for subdir in ["ur_signals_10hz", "ur_signals_10hz_sub", "ur_screwdriver_10hz"]:
        p = input_dir / subdir / "data.parquet"
        if p.exists():
            signal_paths.append(p)
    # Fallback to raw files if pre-downsampled not available
    if not signal_paths:
        for subdir in ["ur_signals_125hz", "ur_screwdriver_signals.parquet"]:
            p = input_dir / subdir if subdir.endswith(".parquet") else input_dir / subdir / "data.parquet"
            if p.exists():
                signal_paths.append(p)
    if not signal_paths:
        single = input_dir / "ur_signals.parquet"
        if single.exists():
            signal_paths.append(single)
    if not signal_paths:
        raise FileNotFoundError(f"No signal parquet files found in {input_dir}")

    logger.info(f"Signal files: {[str(p) for p in signal_paths]}")

    out_dir = output_dir / "factorywave"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Build the set of episode IDs we actually need to load
    # ---------------------------------------------------------------
    needed_ids: set = set()

    # CF groups: need baselines + their CF episodes
    cf_groups_to_process = list(cf_groups.items())
    if cf_limit is not None:
        cf_groups_to_process = cf_groups_to_process[:cf_limit + 50] if cf_limit > 0 else []
    for bl_id, cids in cf_groups_to_process:
        needed_ids.add(bl_id)
        needed_ids.update(cids)

    # Regular episodes (cap by limit)
    if limit is None or limit > 0:
        regular_count = 0
        for ep_id_str in ep_lookup:
            if ep_id_str not in cf_episode_ids and ep_id_str not in baseline_ids:
                needed_ids.add(ep_id_str)
                regular_count += 1
                if limit is not None and regular_count >= limit:
                    break

    logger.info(f"  Need {len(needed_ids)} episodes")

    # ---------------------------------------------------------------
    # Load precomputed CF selection cache (if available)
    # ---------------------------------------------------------------
    cf_cache_path = input_dir.parent / "cf_selection.json"
    if not cf_cache_path.exists():
        cf_cache_path = input_dir / "cf_selection.json"
    cf_cache: Dict[str, Any] = {}
    if cf_cache_path.exists():
        with open(cf_cache_path, encoding="utf-8") as f:
            cf_cache = json.load(f)
        logger.info(f"Loaded CF selection cache: {len(cf_cache)} groups")
    else:
        logger.info("No CF selection cache found, will compute KL on the fly")

    # ---------------------------------------------------------------
    # Narrow needed_ids using cache (only load baseline + best CF)
    # ---------------------------------------------------------------
    if cf_cache:
        cached_cf_ids: set = set()
        cf_groups_to_process_final = list(cf_groups.items())
        if cf_limit is not None:
            cf_groups_to_process_final = cf_groups_to_process_final[:cf_limit + 50] if cf_limit > 0 else []
        needed_ids = set()
        for bl_id, cids in cf_groups_to_process_final:
            if bl_id in cf_cache:
                best_cf_id = cf_cache[bl_id]["best_cf_id"]
                needed_ids.add(bl_id)
                needed_ids.add(best_cf_id)
                cached_cf_ids.add(best_cf_id)
            else:
                # No cache entry, load all candidates for on-the-fly KL
                needed_ids.add(bl_id)
                needed_ids.update(cids)
        # Regular episodes (cap by limit)
        if limit is None or limit > 0:
            regular_count = 0
            for ep_id_str in ep_lookup:
                if ep_id_str not in cf_episode_ids and ep_id_str not in baseline_ids:
                    needed_ids.add(ep_id_str)
                    regular_count += 1
                    if limit is not None and regular_count >= limit:
                        break

    # Skip loading signal data for episodes whose JSON already exists,
    # but still rewrite metadata for all episodes.
    already_done = {f.stem for f in out_dir.glob("*.json") if "_metadata" not in f.name}
    needed_before = len(needed_ids)
    needed_ids -= already_done
    skipped_signal_load = needed_before - len(needed_ids)
    if skipped_signal_load:
        logger.info(f"  Skipping signal load for {skipped_signal_load} already-normalized episodes (metadata will be refreshed)")

    # ---------------------------------------------------------------
    # Pass 1: Load needed episodes, sort by time
    # ---------------------------------------------------------------
    logger.info("Pass 1: Loading episodes...")
    episode_dfs: Dict[str, pd.DataFrame] = {}

    for sig_idx, signal_path in enumerate(signal_paths):
        pf = pq.ParquetFile(signal_path)
        n_rgs = pf.metadata.num_row_groups
        logger.info(f"  Reading {signal_path.name} ({pf.metadata.num_rows:,} rows, {n_rgs} row groups)...")
        for rg_idx in range(n_rgs):
            if n_rgs > 5 and (rg_idx + 1) % max(1, n_rgs // 5) == 0:
                logger.info(f"    Row group {rg_idx + 1}/{n_rgs} ({len(episode_dfs)} episodes loaded)...")
            df = pf.read_row_group(rg_idx).to_pandas(types_mapper=lambda t: None)
            for ep_id, ep_df in df.groupby("episode_id", sort=False):
                ep_id_str = str(ep_id)
                if ep_id_str not in needed_ids:
                    continue
                if ep_id_str in episode_dfs:
                    episode_dfs[ep_id_str] = pd.concat(
                        [episode_dfs[ep_id_str], ep_df], ignore_index=True
                    )
                else:
                    episode_dfs[ep_id_str] = ep_df.reset_index(drop=True)

    # Sort every episode by timestamp
    for ep_id_str in episode_dfs:
        episode_dfs[ep_id_str] = episode_dfs[ep_id_str].sort_values("time").reset_index(drop=True)

    logger.info(f"  Loaded and sorted {len(episode_dfs)} episodes")

    # Decimate episodes that are still above target Hz
    decimated_count = 0
    for ep_id_str, ep_df in episode_dfs.items():
        result = _decimate_episode(ep_df, target_hz)
        if len(result) < len(ep_df):
            decimated_count += 1
        episode_dfs[ep_id_str] = result

    if decimated_count:
        logger.info(f"  Decimated {decimated_count} episodes to {target_hz} Hz")

    # ---------------------------------------------------------------
    # Pass 2: Process counterfactual groups
    # ---------------------------------------------------------------
    total_cf_groups = len(cf_groups)
    logger.info(f"Pass 2: Processing {total_cf_groups} counterfactual groups...")
    cf_processed = 0
    cf_skipped = 0

    for baseline_id, cf_ids in cf_groups.items():
        if cf_limit is not None and cf_processed >= cf_limit:
            break

        if baseline_id not in episode_dfs:
            cf_skipped += 1
            continue

        out_file = out_dir / f"{baseline_id}.json"
        if out_file.exists():
            cf_processed += 1
            continue

        baseline_df = episode_dfs[baseline_id]

        # Use cache if available
        if baseline_id in cf_cache:
            entry = cf_cache[baseline_id]
            best_cf_id = entry["best_cf_id"]
            best_kl = entry["kl_divergence"]
            best_onset = entry["fault_onset_index"]
            n_candidates = entry["candidates_evaluated"]
            if best_cf_id not in episode_dfs:
                cf_skipped += 1
                continue
        else:
            # On-the-fly KL selection (fallback)
            candidates: List[Tuple[str, float, int]] = []
            for cf_id in cf_ids:
                if cf_id not in episode_dfs:
                    continue
                cf_df = episode_dfs[cf_id]
                cf_row = ep_lookup.get(cf_id)
                if cf_row is None:
                    continue

                onset = _find_fault_onset(cf_df)
                if onset is None:
                    raw_timestep = cf_row.get("cf_injection_timestep")
                    if pd.notna(raw_timestep) and raw_timestep is not None:
                        onset = max(1, int(float(raw_timestep) / (500.0 / target_hz)))

                if onset is not None and onset >= 5 and onset < len(cf_df):
                    n_rows = min(onset, len(baseline_df))
                    kl = compute_kl_divergence(baseline_df, cf_df, n_rows)
                    candidates.append((cf_id, kl, onset))
                else:
                    n_rows = min(len(baseline_df), len(cf_df))
                    if n_rows >= 5:
                        kl = compute_kl_divergence(baseline_df, cf_df, n_rows)
                        candidates.append((cf_id, kl, 0))

            if not candidates:
                cf_skipped += 1
                continue

            best_cf_id, best_kl, best_onset = min(candidates, key=lambda x: x[1])
            n_candidates = len(candidates)

        best_cf_df = episode_dfs[best_cf_id]
        cf_meta_row = ep_lookup.get(best_cf_id)

        # Normalize both (already sorted by time)
        bl_first_t = int(pd.Timestamp(baseline_df["time"].values[0]).value // 1000)
        cf_first_t = int(pd.Timestamp(best_cf_df["time"].values[0]).value // 1000)

        cf_fault_id = _to_int(cf_meta_row.get("cf_fault_id")) if cf_meta_row is not None else None
        baseline_rows = normalize_episode_df(baseline_df, bl_first_t)
        cf_rows = normalize_episode_df(best_cf_df, cf_first_t, metadata_fault_id=cf_fault_id)

        if not baseline_rows or not cf_rows:
            cf_skipped += 1
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"baseline": baseline_rows, "counterfactual": cf_rows}, f, indent=2, cls=_NaNSafeEncoder)

        bl_row = ep_lookup[baseline_id]
        cf_meta_row = ep_lookup.get(best_cf_id)
        metadata = load_episode_metadata(bl_row)
        # Override CF fields from the CF episode row
        if cf_meta_row is not None:
            metadata["cf_fault_id"] = _to_int(cf_meta_row.get("cf_fault_id"))
            metadata["cf_injection_timestep"] = _to_int(cf_meta_row.get("cf_injection_timestep"))
            metadata["cf_injection_time_s"] = _to_float(cf_meta_row.get("cf_injection_time_s"))
            metadata["cf_baseline_episode_id"] = baseline_id
            metadata["counterfactual"] = True
        metadata["baseline"] = {
            "num_samples": len(baseline_rows),
            "last_timestamp_ms": baseline_rows[-1]["timestamp_ms"],
        }
        metadata["counterfactual"] = {
            "episode_id": best_cf_id,
            "num_samples": len(cf_rows),
            "last_timestamp_ms": cf_rows[-1]["timestamp_ms"],
            "fault_onset_index": best_onset,
            "kl_divergence": round(best_kl, 6),
            "cf_fault_id": _to_int(cf_meta_row.get("cf_fault_id")) if cf_meta_row is not None else None,
            "cf_injection_timestep": _to_int(cf_meta_row.get("cf_injection_timestep")) if cf_meta_row is not None else None,
            "candidates_evaluated": n_candidates,
        }
        metadata["schema_version"] = "1.0"

        meta_file = out_dir / f"{baseline_id}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        cf_processed += 1
        if cf_processed % 50 == 0:
            logger.info(f"  {cf_processed} CF groups processed...")

    logger.info(f"  CF groups: {cf_processed} processed, {cf_skipped} skipped")

    # ---------------------------------------------------------------
    # Pass 3: Process non-counterfactual episodes (normal, fault, trajectory-opt)
    # ---------------------------------------------------------------
    total_regular = sum(1 for eid in episode_dfs if eid not in cf_episode_ids and eid not in baseline_ids and eid in ep_lookup)
    logger.info(f"Pass 3: Normalizing {total_regular} non-counterfactual episodes...")
    regular_processed = 0
    regular_skipped = 0

    for ep_id_str, ep_df in episode_dfs.items():
        # Skip CFs and baselines (already handled)
        if ep_id_str in cf_episode_ids or ep_id_str in baseline_ids:
            continue

        if ep_id_str not in ep_lookup:
            continue

        if limit and regular_processed >= limit:
            break

        out_file = out_dir / f"{ep_id_str}.json"
        if out_file.exists():
            regular_processed += 1
            continue

        # Normalize, use metadata fault_id for episodes where signal fault column is 0
        ep_row = ep_lookup[ep_id_str]
        ep_meta = ep_row.get("_meta", {})
        meta_fault_id = _to_int(ep_meta.get("fault_id")) if isinstance(ep_meta, dict) else None
        # Fallback to fault_metadata
        if meta_fault_id is None:
            fm_str = ep_row.get("fault_metadata")
            if isinstance(fm_str, str):
                try:
                    fm = ast.literal_eval(fm_str)
                    meta_fault_id = _to_int(fm.get("fault_id")) if isinstance(fm, dict) else None
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(fm_str, dict):
                meta_fault_id = _to_int(fm_str.get("fault_id"))
        first_t = int(pd.Timestamp(ep_df["time"].values[0]).value // 1000)
        normalized_rows = normalize_episode_df(ep_df, first_t, metadata_fault_id=meta_fault_id)

        if not normalized_rows:
            regular_skipped += 1
            continue

        # Metadata
        ep_row = ep_lookup[ep_id_str]
        metadata = load_episode_metadata(ep_row)

        # Infer task from phase count if not set
        if metadata.get("task") in (None, "unknown", ""):
            metadata["task"] = _infer_task_from_phases(normalized_rows, metadata.get("fault_id"))

        # Synthesize the missing fault flip for peg-in-hole fault 32/34 episodes.
        _inject_fault_flip_if_missing(normalized_rows, metadata.get("fault_id"))

        # Re-align fault 19 / 37 flips to the safety_mode 1->3 protective-stop
        # transition (when present in the recorded window).
        _align_fault_flip_to_protective_stop(normalized_rows, metadata.get("fault_id"))

        # Write as flat list
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(normalized_rows, f, indent=2, cls=_NaNSafeEncoder)
        metadata["num_samples"] = len(normalized_rows)
        metadata["schema_version"] = "1.0"
        metadata["timestamp_info"] = {
            "format": "milliseconds since episode start",
            "last_timestamp_ms": normalized_rows[-1]["timestamp_ms"],
        }

        meta_file = out_dir / f"{ep_id_str}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        regular_processed += 1
        if regular_processed % 100 == 0:
            logger.info(f"  {regular_processed} episodes normalized...")

    logger.info(f"  Regular episodes: {regular_processed} processed, {regular_skipped} skipped")

    # ---------------------------------------------------------------
    # Pass 4: Refresh metadata for already-normalized episodes
    # ---------------------------------------------------------------
    if already_done:
        logger.info(f"Pass 4: Refreshing metadata for {len(already_done)} already-normalized episodes...")
        meta_refreshed = 0
        for ep_id_str in already_done:
            if ep_id_str not in ep_lookup:
                continue
            ep_row = ep_lookup[ep_id_str]
            metadata = load_episode_metadata(ep_row)

            # Read existing JSON to get num_samples and last_timestamp
            data_file = out_dir / f"{ep_id_str}.json"
            if data_file.exists():
                try:
                    with open(data_file, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        metadata["num_samples"] = len(data)
                        if data:
                            metadata["timestamp_info"] = {
                                "format": "milliseconds since episode start",
                                "last_timestamp_ms": data[-1].get("timestamp_ms"),
                            }
                    elif isinstance(data, dict) and "baseline" in data:
                        bl = data["baseline"]
                        cf = data.get("counterfactual", [])
                        metadata["baseline"] = {
                            "num_samples": len(bl),
                            "last_timestamp_ms": bl[-1].get("timestamp_ms") if bl else None,
                        }
                        metadata["counterfactual"] = {
                            "num_samples": len(cf),
                            "last_timestamp_ms": cf[-1].get("timestamp_ms") if cf else None,
                        }
                except Exception:
                    pass

            metadata["schema_version"] = "1.0"
            meta_file = out_dir / f"{ep_id_str}_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            meta_refreshed += 1
            if meta_refreshed % 500 == 0:
                logger.info(f"  {meta_refreshed} metadata files refreshed...")

        logger.info(f"  Refreshed {meta_refreshed} metadata files")

    logger.info(f"Done. Output: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize FactoryWave dataset to UR3e schema JSON."
    )
    repo_root = Path(__file__).resolve().parents[2]

    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "factorywave" / "data",
        help="Input directory containing episode.parquet and ur_signals parquet files")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "normalized_episodes",
        help="Output directory for normalized JSON files")
    parser.add_argument(
        "--target-hz",
        type=int,
        default=TARGET_HZ,
        help=f"Target sampling rate after decimation (default: {TARGET_HZ})")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of regular (non-CF) episodes to normalize")
    parser.add_argument(
        "--cf-limit",
        type=int,
        default=None,
        help="Limit number of counterfactual groups to normalize")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Filter by task (e.g. pick_and_place peg_in_hole)")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    try:
        normalize_dataset(
            input_dir=args.input,
            output_dir=args.output,
            target_hz=args.target_hz,
            limit=args.limit,
            cf_limit=args.cf_limit,
            tasks=args.tasks)
        return 0
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
