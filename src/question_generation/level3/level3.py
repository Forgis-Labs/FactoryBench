"""
Level 3 question generator: Counterfactual Reasoning.

Reads normalized episode JSON files from paired counterfactual datasets,
samples random sub-series, and fills Level 3 question templates.
Answers are generated when determinable from episode readings.

Output: datasets/questions/level3/level3_{NNNN}.json

Usage:
    python -m src.questions.level3.level3 -n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from src.question_generation.utils.hf_streaming import (
    HfStreamUploader,
    add_streaming_args,
    make_uploader_from_args)
from src.question_generation.utils.mc_availability import filter_lookup_by_availability
from src.question_generation.utils.io import load_events, load_json, load_root_causes, load_templates
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill,
    fill_event_description,
    get_last_timestamp,
    pick_constrained_signal,
    pick_scalar_signal)
from src.question_generation.utils.time_series import (
    parse_event_id,
    pick_fault_label)
from src.question_generation.level3.mc_truth import (
    DEFAULT_THRESHOLDS,
    UNCALIBRATABLE_MC_IDS,
    complement_of,
    evaluate_mc_statement,
    thresholds_for)
from src.question_generation.level2.mc_truth import robot_key


logger = logging.getLogger(__name__)

VALID_DATASETS = ["factorywave", "factorywave_kuka"]
STEPS_AHEAD_RANGE = (1, 10)
CONTEXT_MIN = 32
CONTEXT_MAX = 64



EXCLUDED_JOINT_SIGNALS = {"joint_voltage", "joint_temp", "joint_mode"}
JOINT_INDEX_RANGE = set(range(6))
MIN_POST_EVENT_TIMESTAMPS_AFTER = 5
CF_DATASET_FOLDERS = ["factorywave"]  # full list; filtered at runtime via --datasets

# MC option IDs that require signals absent from simulation data
NON_SIMULATION_EXCLUDED_MC_IDS = {
    "mc_020",  # task_success, only available in simulation metadata
}

SIMULATION_EXCLUDED_MC_IDS = {
    "mc_005",  # effort_current
    "mc_013",  # robot_current
    "mc_014",  # robot_current
    "mc_017",  # joint_temp
    "mc_019",  # safety_mode
}

# MC option IDs that use mode signals (safety_mode, joint_mode, robot_mode),
# excluded from predictive templates where mode state is not being predicted.
PREDICTIVE_EXCLUDED_MC_IDS = {
    "mc_019",  # safety_mode
}

# Collision event IDs, excluded from predictive templates because collision
# effects are sharp discontinuities not predictable from the pre-event trajectory.
COLLISION_EVENT_IDS = {16, 17, 18, 19}


def pick_joint_indexed_signal_base(subseries: List[Dict[str, Any]], allowed_bases: Optional[set] = None) -> Optional[str]:
    """
    Pick a base signal name that has indexed variants for all joints 0..5.

    Example valid base: "setpoint_pos" (requires setpoint_pos_0 ... setpoint_pos_5).
    Excludes: joint_voltage, joint_temp, joint_mode.
    If allowed_bases is provided, only bases in that set are considered.
    """
    base_to_indices: Dict[str, set[int]] = {}

    for row in subseries:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            match = re.match(r"^(.*)_(\d+)$", str(key))
            if not match:
                continue

            base = match.group(1)
            idx = int(match.group(2))
            if idx not in JOINT_INDEX_RANGE:
                continue

            if base not in base_to_indices:
                base_to_indices[base] = set()
            base_to_indices[base].add(idx)

    candidates = [
        base
        for base, indices in base_to_indices.items()
        if indices == JOINT_INDEX_RANGE and base not in EXCLUDED_JOINT_SIGNALS
        and (allowed_bases is None or base in allowed_bases)
    ]

    if not candidates:
        return None

    return random.choice(sorted(candidates))


def sample_subsequent_chunks(
    rows: List[Dict[str, Any]],
    n_chunks: int = 4,
    min_chunk: int = 5,
    max_chunk: int = 7) -> List[List[Dict[str, Any]]]:
    """
    Sample n_chunks contiguous, subsequent chunks from rows.

    Chunks are back-to-back in time (no gaps / no overlap), all with the same
    randomly chosen chunk length in [min_chunk, max_chunk].
    """
    available = len(rows)
    if available < n_chunks * min_chunk:
        return []

    max_feasible_chunk = min(max_chunk, available // n_chunks)
    if max_feasible_chunk < min_chunk:
        return []

    chunk_len = random.randint(min_chunk, max_feasible_chunk)
    total_len = n_chunks * chunk_len
    start = random.randint(0, available - total_len)

    chunks: List[List[Dict[str, Any]]] = []
    for i in range(n_chunks):
        left = start + i * chunk_len
        right = left + chunk_len
        chunks.append(rows[left:right])
    return chunks


def rows_strictly_after_context(
    rows: List[Dict[str, Any]],
    subseries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows whose timestamp is strictly greater than the last context timestamp.

    The ranking template asks the model to order segments by when they appear
    as the event manifests. That is only a question about the future if the
    segments lie outside the window the model was already shown. Sampling from
    ``post_event_rows`` directly does not guarantee this: the context window is
    centred on the event onset and extends past it, and (at L3) the baseline
    and counterfactual episodes are identical until they diverge, so early
    post-onset chunks reproduce rows the model can already see and the ordering
    can be recovered by matching values instead of reasoning about propagation.
    Measured on the released benchmark, 13% of option segments appeared
    verbatim inside their own context.
    """
    cutoff = get_last_timestamp(subseries)
    out: List[Dict[str, Any]] = []
    for row in rows:
        ts = row.get("timestamp_ms")
        try:
            if ts is not None and int(float(ts)) > cutoff:
                out.append(row)
        except (TypeError, ValueError):
            continue
    return out


def filter_rows_to_template_features(
    rows: List[Dict[str, Any]],
    template: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Restrict rows to the template's ``important_features``.

    ``build_context`` applies this filter to the context but the option
    segments were encoded from the full row, so a ranking item showed a context
    defined over ~19 channels while asking the model to order segments carrying
    ~98, of which 79 appeared in no acronym mapping anywhere in the item. The
    segments and the context have to describe the same signals for the question
    to be answerable as posed.
    """
    keep = template.get("important_features")
    if not keep:
        return rows
    allowed = set(keep) | {"timestamp_ms"}
    return [{k: v for k, v in row.items() if k in allowed} for row in rows]


def encode_chunk_without_timestamps(rows: List[Dict[str, Any]]) -> str:
    """Encode a chunk after removing timestamp_ms from each row."""
    stripped_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        stripped_rows.append({k: v for k, v in row.items() if k != "timestamp_ms"})
    return encode_chunk(stripped_rows)


def get_row_at_or_after_timestamp(
    rows: List[Dict[str, Any]],
    target_timestamp_ms: int) -> Optional[Dict[str, Any]]:
    """Return the first row whose timestamp_ms is >= target_timestamp_ms."""
    for row in rows:
        ts = row.get("timestamp_ms")
        try:
            if ts is not None and int(float(ts)) >= target_timestamp_ms:
                return row
        except (TypeError, ValueError):
            continue
    return None


def _first_timestamp_ms(rows: List[Dict[str, Any]]) -> int:
    """Return first valid timestamp_ms in rows, or 0 if missing."""
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = row.get("timestamp_ms")
        try:
            if ts is not None:
                return int(float(ts))
        except (TypeError, ValueError):
            continue
    return 0


def normalize_timestamps(
    rows: List[Dict[str, Any]],
    base_timestamp_ms: int) -> List[Dict[str, Any]]:
    """Return a copy of rows with timestamp_ms shifted by base_timestamp_ms."""
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out_row = dict(row)
        ts = out_row.get("timestamp_ms")
        try:
            if ts is not None:
                out_row["timestamp_ms"] = int(float(ts)) - int(base_timestamp_ms)
        except (TypeError, ValueError):
            pass
        normalized.append(out_row)
    return normalized


def split_event_segment(
    post_event_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split rows starting at event onset into:
    - full contiguous event segment (same non-zero event token)
    - tail rows after the event segment
    """
    if not post_event_rows:
        return [], []

    onset_raw = post_event_rows[0].get("event", 0)
    onset_id = parse_event_id(onset_raw)
    if onset_id == 0:
        return [], post_event_rows

    event_segment: List[Dict[str, Any]] = []
    end_idx = 0
    onset_token = str(onset_raw)

    for i, row in enumerate(post_event_rows):
        row_token = str(row.get("event", 0))
        row_id = parse_event_id(row.get("event", 0))
        if row_id != onset_id or row_token != onset_token:
            end_idx = i
            break
        event_segment.append(row)
    else:
        end_idx = len(post_event_rows)

    return event_segment, post_event_rows[end_idx:]


def is_escalated_by_safety_mode(rows: List[Dict[str, Any]]) -> bool:
    """
    Escalation rule for template 3:
    If any timestep in the remaining sample has safety_mode != 1, it escalated.
    """
    for row in rows:
        value = row.get("safety_mode")
        try:
            if value is not None and int(float(value)) != 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def normalize_mc_option_id(value: str) -> str:
    """Normalize IDs like mc_19/l2_mc_19 -> mc_019."""
    text = str(value).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"mc_{int(match.group(1)):03d}"


def _legacy_mc_option_id(value: str) -> str:
    """Convert normalized mc_* IDs to legacy l2_mc_* IDs used by current rules."""
    normalized = normalize_mc_option_id(value)
    match = re.match(r"^mc_(\d+)$", normalized, re.IGNORECASE)
    if not match:
        return normalized
    return f"l2_mc_{int(match.group(1)):03d}"


def load_mc_option_lookup(path: Path, level: int) -> Dict[str, str]:
    """
    Load MC options as a map: id -> statement.
    Returns empty map if file is missing/invalid.
    """
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load MC options from {path}: {exc}")
        return {}

    if not isinstance(raw, list):
        logger.warning(f"MC options file has unexpected format (expected list): {path}")
        return {}

    lookup: Dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue

        allowed_levels = item.get("usable_levels")
        if isinstance(allowed_levels, list):
            parsed_levels: List[int] = []
            for value in allowed_levels:
                try:
                    parsed_levels.append(int(value))
                except (TypeError, ValueError):
                    continue
            if parsed_levels and level not in parsed_levels:
                continue
        else:
            legacy_level = item.get("applicable_level")
            if legacy_level is not None:
                try:
                    if int(legacy_level) != level:
                        continue
                except (TypeError, ValueError):
                    continue

        option_id = item.get("id")
        statement = item.get("statement")
        if isinstance(option_id, str) and isinstance(statement, str) and statement.strip():
            lookup[normalize_mc_option_id(option_id)] = statement.strip()
    return lookup


def discover_cf_episode_pairs(
    datasets_dir: Path,
    cf_dataset_folders: List[str]) -> List[Dict[str, Any]]:
    """
    Discover paired episode files inside cf dataset folders.

    For each cf folder (e.g., cf_aursad), expects one alt subfolder and one
    non-alt subfolder. Sampling uses non-alt as context source and alt as
    the counterfactual source containing the event.
    """
    normalized_root = datasets_dir / "normalized_episodes"
    pairs: List[Dict[str, Any]] = []

    for cf_name in cf_dataset_folders:
        cf_root = normalized_root / cf_name
        if not cf_root.exists() or not cf_root.is_dir():
            continue

        # Combined format: each episode file contains both baseline and counterfactual
        combined_files = [
            p for p in sorted(cf_root.glob("*.json"))
            if not p.name.endswith("_metadata.json")
        ]
        if combined_files:
            for p in combined_files:
                pairs.append({
                    "cf_dataset": cf_name,
                    "format": "combined",
                    "non_alt_path": p,
                    "alt_path": p,
                    "episode": p.stem,
                })
            continue

        # Legacy format: separate alt / non-alt subfolders
        subfolders = sorted([p for p in cf_root.iterdir() if p.is_dir()])
        if len(subfolders) < 2:
            continue

        alt_candidates = [p for p in subfolders if p.name.lower().startswith("alt")]
        non_alt_candidates = [p for p in subfolders if p not in alt_candidates]
        if not alt_candidates or not non_alt_candidates:
            continue

        alt_folder = alt_candidates[0]
        preferred_normal_name = cf_name[3:] if cf_name.lower().startswith("cf_") else cf_name
        preferred_normal = next((p for p in non_alt_candidates if p.name == preferred_normal_name), None)
        non_alt_folder = preferred_normal or non_alt_candidates[0]

        def _episode_map(folder: Path) -> Dict[str, Path]:
            episode_files = [
                p for p in folder.glob("*.json")
                if not p.name.endswith("_metadata.json")
            ]
            return {p.stem: p for p in episode_files}

        alt_map = _episode_map(alt_folder)
        non_alt_map = _episode_map(non_alt_folder)
        common = sorted(set(alt_map.keys()) & set(non_alt_map.keys()))
        if not common:
            continue

        for stem in common:
            pairs.append(
                {
                    "cf_dataset": cf_name,
                    "format": "split",
                    "non_alt_subfolder": non_alt_folder.name,
                    "alt_subfolder": alt_folder.name,
                    "non_alt_path": non_alt_map[stem],
                    "alt_path": alt_map[stem],
                    "episode": stem,
                }
            )

    return pairs


def find_event_onset_index(rows: List[Dict[str, Any]]) -> Optional[int]:
    """Return the first index where an event starts (event id becomes non-zero)."""
    prev_event_id = 0
    for idx, row in enumerate(rows):
        current_event_id = parse_event_id(row.get("event", 0))
        if current_event_id != 0 and prev_event_id == 0:
            return idx
        prev_event_id = current_event_id
    return None


def sample_window_around_index(
    rows: List[Dict[str, Any]],
    center_index: int,
    min_len: int,
    max_len: int,
    margin: int = 5) -> Optional[Tuple[List[Dict[str, Any]], int, int]]:
    """
    Sample one contiguous subseries containing center_index with at least
    `margin` timesteps from the subseries borders.
    Returns (subseries, start_index, length) or None when impossible.
    """
    n_rows = len(rows)
    if n_rows <= 0 or center_index < 0 or center_index >= n_rows:
        return None

    min_required_len = max(min_len, 2 * margin + 1)
    max_allowed_len = min(max_len, n_rows)
    if max_allowed_len < min_required_len:
        return None

    possible_lengths: List[int] = []
    for length in range(min_required_len, max_allowed_len + 1):
        start_low = max(0, center_index + margin - (length - 1))
        start_high = min(center_index - margin, n_rows - length)
        if start_low <= start_high:
            possible_lengths.append(length)

    if not possible_lengths:
        return None

    chosen_len = random.choice(possible_lengths)
    start_low = max(0, center_index + margin - (chosen_len - 1))
    start_high = min(center_index - margin, n_rows - chosen_len)
    if start_low > start_high:
        return None

    start_idx = random.randint(start_low, start_high)
    return rows[start_idx : start_idx + chosen_len], start_idx, chosen_len


def resolve_fixed_option(
    token: Any,
    mc_option_lookup: Dict[str, str]) -> Tuple[Optional[str], str]:
    """
    Resolve a fixed option token to (canonical_option_id, rendered_statement).
    If token is not a known MC ID, returns (None, str(token)).
    """
    if isinstance(token, str):
        canonical_id = normalize_mc_option_id(token)
        statement = mc_option_lookup.get(canonical_id)
        if statement:
            return canonical_id, statement
        return None, token
    return None, str(token)


def _sample_ratio(mean: float, rel_std: float = 0.20, min_value: float = 0.0,
                  max_value: Optional[float] = None) -> float:
    """Draw a threshold around ``mean``.

    ``max_value`` defaults to no upper bound. Level 2 removed the old 0.99
    default after finding it silently clipped any centre above 1 back to 0.99
    and pinned the statement's truth value; level 3 kept the old default, so
    the same clipping is still live here. Callers that genuinely need a sub-1
    cap, the coverage and axis-ratio knobs, pass max_value explicitly.
    """
    sampled = random.gauss(mean, max(1e-6, abs(mean) * rel_std))
    sampled = max(sampled, min_value)
    if max_value is not None:
        sampled = min(sampled, max_value)
    return float(sampled)


def sample_thresholds_for_statement(
    statement_id: str,
    source: Optional[str] = None) -> Dict[str, float]:
    """Draw this statement's thresholds around the centre fitted for ``source``.

    Thresholds still vary per item; only the centre they revolve around is
    source-specific. Passing source=None keeps the corpus-wide defaults, which
    is what unlisted sources get. Level 2 has worked this way for a while;
    level 3 drew every threshold from the unfitted defaults, which is what left
    its statement families near-deterministic.
    """
    centres = thresholds_for(source)
    sid = _legacy_mc_option_id(str(statement_id))
    if sid in {"l2_mc_003", "l2_mc_004"}:
        # Complements on one shared ratio-of-means threshold.
        return {"speed_drop_ratio": _sample_ratio(centres["speed_drop_ratio"])}
    if sid == "l2_mc_005":
        return {
            "stall_current_increase": _sample_ratio(centres["stall_current_increase"]),
            "stall_speed_frac": _sample_ratio(centres["stall_speed_frac"]),
        }
    if sid == "l2_mc_006":
        return {
            "force_low_increase": _sample_ratio(centres["force_low_increase"]),
            "force_low_coverage": _sample_ratio(centres["force_low_coverage"], rel_std=0.08, min_value=0.50, max_value=0.99),
        }
    if sid == "l2_mc_007":
        return {"force_spike_increase": _sample_ratio(centres["force_spike_increase"])}
    if sid in {"l2_mc_008", "l2_mc_009"}:
        # Both gate on the same absolute tracking error, so they share one
        # sampled threshold and are exact complements of each other.
        return {"tracking_abs_error": _sample_ratio(centres["tracking_abs_error"])}
    if sid == "l2_mc_010":
        return {"vibration_spike": _sample_ratio(centres["vibration_spike"])}
    if sid == "l2_mc_011":
        return {
            "vibration_nominal_band": _sample_ratio(centres["vibration_nominal_band"]),
            "vibration_nominal_coverage": _sample_ratio(centres["vibration_nominal_coverage"], rel_std=0.06, min_value=0.60, max_value=0.99),
        }
    if sid == "l2_mc_012":
        return {"current_relax_drop": _sample_ratio(centres["current_relax_drop"], max_value=0.99)}
    if sid in {"l2_mc_013", "l2_mc_014"}:
        return {"robot_current_abs": _sample_ratio(centres["robot_current_abs"])}
    if sid in {"l2_mc_015", "l2_mc_016"}:
        return {"tcp_tracking_abs_error": _sample_ratio(centres["tcp_tracking_abs_error"])}
    if sid == "l2_mc_017":
        min_axes = int(round(random.gauss(centres["temp_rise_min_axes"], 0.4)))
        min_axes = min(max(min_axes, 1), 6)
        return {
            "temp_rise_slope": _sample_ratio(centres["temp_rise_slope"], rel_std=0.25, min_value=0.0005, max_value=0.02),
            "temp_rise_min_axes": float(min_axes),
        }
    if sid == "l2_mc_018":
        return {
            "temp_stable_slope": _sample_ratio(centres["temp_stable_slope"], rel_std=0.25, min_value=0.0002, max_value=0.01),
            "temp_stable_axes_ratio": _sample_ratio(centres["temp_stable_axes_ratio"], rel_std=0.08, min_value=0.50, max_value=0.99),
        }
    if sid == "l2_mc_019":
        return {"no_effect_agg_increase": _sample_ratio(centres["no_effect_agg_increase"])}
    if sid == "l2_mc_030":
        return {"joint_excursion_deg": _sample_ratio(centres["joint_excursion_deg"])}
    if sid == "l2_mc_031":
        return {"joint_path_deg": _sample_ratio(centres["joint_path_deg"])}
    if sid == "l2_mc_032":
        return {"torque_p2p_nm": _sample_ratio(centres["torque_p2p_nm"])}
    return {}


def _fmt_pct(value: float) -> str:
    return str(int(round(100.0 * value)))


def render_statement_with_thresholds(
    statement_id: str,
    default_statement: str,
    thresholds: Dict[str, float]) -> str:
    sid = _legacy_mc_option_id(str(statement_id))
    if sid == "l2_mc_003":
        return (
            "Following the event, at least one joint speed drops sharply "
            f"(mean speed magnitude falls to <={_fmt_pct(thresholds['speed_drop_ratio'])}% of its pre-event mean)."
        )
    if sid == "l2_mc_004":
        return (
            "Following the event, no joint's mean speed magnitude drops sharply "
            f"(none falls to <={_fmt_pct(thresholds['speed_drop_ratio'])}% of its pre-event mean)."
        )
    if sid == "l2_mc_005":
        return (
            "Following the event, motor current rises by "
            f">={_fmt_pct(thresholds['stall_current_increase'])}% while speed magnitude stays "
            f"<={_fmt_pct(thresholds['stall_speed_frac'])}% of pre-event baseline (stall-like behavior)."
        )
    if sid == "l2_mc_006":
        return (
            "Following the event, contact-force magnitude remains low "
            f"(<={_fmt_pct(thresholds['force_low_increase'])}% above pre-event baseline for at least "
            f"{_fmt_pct(thresholds['force_low_coverage'])}% of timesteps)."
        )
    if sid == "l2_mc_007":
        return (
            "Following the event, contact force shows a significant spike "
            f"(>={_fmt_pct(thresholds['force_spike_increase'])}% above baseline norm)."
        )
    if sid == "l2_mc_008":
        return (
            "Following the event, mean joint tracking error exceeds "
            f"{thresholds['tracking_abs_error']:.4f} (absolute, commanded vs measured position)."
        )
    if sid == "l2_mc_009":
        return (
            "Following the event, mean joint tracking error stays below "
            f"{thresholds['tracking_abs_error']:.4f} (absolute, commanded vs measured position)."
        )
    if sid == "l2_mc_010":
        return (
            "Following the event, vibration exhibits a transient burst "
            f"(>={_fmt_pct(thresholds['vibration_spike'])}% above pre-event baseline on at least one axis)."
        )
    if sid == "l2_mc_011":
        return (
            "Following the event, vibration remains nominal "
            f"(within ±{_fmt_pct(thresholds['vibration_nominal_band'])}% for at least "
            f"{_fmt_pct(thresholds['vibration_nominal_coverage'])}% of timesteps)."
        )
    if sid == "l2_mc_012":
        return (
            "Following the event, at least one joint current peaks then relaxes "
            f"(final value >={_fmt_pct(thresholds['current_relax_drop'])}% below that joint's peak)."
        )
    if sid == "l2_mc_013":
        return (
            "Following the event, mean robot current stays below "
            f"{thresholds['robot_current_abs']:.3f} (absolute)."
        )
    if sid == "l2_mc_014":
        return (
            "Following the event, mean robot current exceeds "
            f"{thresholds['robot_current_abs']:.3f} (absolute)."
        )
    if sid == "l2_mc_015":
        return (
            "Following the event, command and measured TCP motion remain aligned "
            f"(mean TCP tracking error stays below {thresholds['tcp_tracking_abs_error']:.4f})."
        )
    if sid == "l2_mc_016":
        return (
            "Following the event, command and measured TCP motion become misaligned "
            f"(mean TCP tracking error exceeds {thresholds['tcp_tracking_abs_error']:.4f})."
        )
    if sid == "l2_mc_017":
        return (
            "Following the event, temperatures rise across multiple joints "
            f"(>={int(round(thresholds['temp_rise_min_axes']))} joints with slope >"
            f"{100.0 * thresholds['temp_rise_slope']:.2f}% of baseline per horizon)."
        )
    if sid == "l2_mc_018":
        return (
            "Following the event, temperatures remain stable "
            f"(slope magnitude <={100.0 * thresholds['temp_stable_slope']:.2f}% of baseline per horizon "
            f"for at least {_fmt_pct(thresholds['temp_stable_axes_ratio'])}% of joints)."
        )
    if sid == "l2_mc_019":
        return (
            "No significant effect: safety stays normal and aggregate error "
            "(mean of tracking, TCP-tracking, force, and robot-current deviation terms) "
            f"increases by <={_fmt_pct(thresholds['no_effect_agg_increase'])}%."
        )
    return default_statement


def build_multiselect_options_and_answer(
    answer_format: Dict[str, Any],
    baseline_subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    mc_option_lookup: Dict[str, str],
    episode_metadata: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None) -> Tuple[Dict[str, str], str]:
    """
    Build exactly 4 multi-select options:
    1) Keep fixed options from template (if resolvable IDs).
    2) Sample random additional statements (excluding already selected IDs).
    3) Evaluate each option truth value.
    4) Shuffle and emit A-D options plus T/F answer string.
    """
    def _canon(option_id: str) -> str:
        """mc_008, l2_mc_008 and mc_8 all name the same statement."""
        return _legacy_mc_option_id(str(option_id)).replace("l2_", "")

    selected_ids: List[str] = []
    used_canon: set = set()

    def _admit(option_id: str) -> bool:
        """Take this statement unless it is dead or negates one already taken."""
        canon = _canon(option_id)
        if canon in UNCALIBRATABLE_MC_IDS:
            return False
        if canon in used_canon:
            return False
        opposite = complement_of(canon)
        if opposite and opposite in used_canon:
            # Two options that are exact negations hand the model one of them
            # for free: the pairs share a threshold, so the answer at those
            # positions is always exactly one T whatever the episode did.
            return False
        selected_ids.append(option_id)
        used_canon.add(canon)
        return True

    fixed_tokens = answer_format.get("fixed_options") or answer_format.get("fixed_statements") or []
    for token in fixed_tokens:
        option_id, _statement = resolve_fixed_option(token, mc_option_lookup)
        if option_id:
            _admit(option_id)

    remaining_ids = [opt_id for opt_id in sorted(mc_option_lookup.keys())
                     if _canon(opt_id) not in used_canon]
    random.shuffle(remaining_ids)
    while len(selected_ids) < 4 and remaining_ids:
        _admit(remaining_ids.pop())

    if len(selected_ids) < 4:
        # Fewer than four usable statements for this episode. Emitting a
        # short item would change the answer-string length the scorer expects,
        # so the caller skips the item instead.
        raise ValueError(
            f"only {len(selected_ids)} calibratable, non-complementary options "
            f"available for this episode"
        )
    selected_ids = selected_ids[:4]

    options_data: List[Tuple[str, str, Optional[bool]]] = []
    for opt_id in selected_ids:
        sampled_thresholds = sample_thresholds_for_statement(
            opt_id, source or robot_key(episode_metadata))
        statement = render_statement_with_thresholds(
            opt_id,
            mc_option_lookup.get(opt_id, opt_id),
            sampled_thresholds)
        truth = evaluate_mc_statement(
            opt_id,
            subseries=baseline_subseries,
            post_event_rows=post_event_rows,
            thresholds=sampled_thresholds,
            episode_metadata=episode_metadata)
        options_data.append((opt_id, statement, truth))

    random.shuffle(options_data)

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    answer_chars: List[str] = []
    for idx, (_opt_id, statement, truth) in enumerate(options_data):
        if idx >= len(labels):
            break
        label = labels[idx]
        options[label] = statement
        answer_chars.append("T" if truth is True else "F")

    return options, "".join(answer_chars)


# ---------------------------------------------------------------------------
# Template filling  (level 3 specific)
# ---------------------------------------------------------------------------


def fill_template(
    template: Dict[str, Any],
    subseries: List[Dict[str, Any]],
    post_event_rows: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    mc_option_lookup: Dict[str, str],
    t2_ms: Optional[int] = None,
    answer_subseries: Optional[List[Dict[str, Any]]] = None,
    steps_ahead: Optional[int] = None,
    episode_metadata: Optional[Dict[str, Any]] = None,
    dataset_name: Optional[str] = None,
    event_description: str = "") -> Optional[Dict[str, Any]]:
    """
    Fill a Level 3 question template.

    post_event_rows: rows from the episode after the subseries end
                     (used for ranking chunks).

    Returns dict with question/options/answer_format/event_id/answer, or None on failure.

    Template IDs:
      1 - signal_segment_ranking        : options = {A/B/C/D: encoded chunk}
      4 - signal_value_prediction       : numerical
      5 - signal_value_prediction       : tensor
      6 - counterfactual_spatial        : what if the target were at an offset? (MC A-D)
    """
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]
    t = get_last_timestamp(subseries)
    baseline_rows = answer_subseries if answer_subseries is not None else subseries

    onset_id = parse_event_id(post_event_rows[0].get("event", 0)) if post_event_rows else 0
    event_obj = next((e for e in events if e["id"] == onset_id), random.choice(events))
    event_desc = fill_event_description(event_obj, subseries, t, post_event_rows)
    event_time = t if t2_ms is None else int(t2_ms)

    options: Dict[str, Any] = {}
    answer = None
    acceptance_bounds = None

    if tid == 1:
        ranking_pool = rows_strictly_after_context(post_event_rows, subseries)
        ranking_pool = filter_rows_to_template_features(ranking_pool, template)
        chunks = sample_subsequent_chunks(ranking_pool, n_chunks=4, min_chunk=5, max_chunk=7)
        if len(chunks) < 4:
            return None

        ordered_chunks = list(chunks)
        random.shuffle(chunks)
        labels = ["A", "B", "C", "D"]
        options = {
            label: encode_chunk_without_timestamps(chunks[i])
            for i, label in enumerate(labels)
        }

        chunk_to_label = {id(chunks[i]): labels[i] for i in range(len(chunks))}
        answer = "".join(chunk_to_label[id(chunk)] for chunk in ordered_chunks)

        question = fill(tmpl_text, event=event_description)

    elif tid == 2:
        try:
            options, answer = build_multiselect_options_and_answer(
                answer_format=answer_format,
                baseline_subseries=baseline_rows,
                post_event_rows=post_event_rows,
                mc_option_lookup=mc_option_lookup,
                episode_metadata=episode_metadata,
                source=robot_key(episode_metadata))
        except ValueError:
            # Not enough usable statements for this episode; skip the item
            # rather than emit one with fewer than four options.
            return None
        question = fill(tmpl_text, event=event_description)

    elif tid == 3:
        try:
            options, answer = build_multiselect_options_and_answer(
                answer_format=answer_format,
                baseline_subseries=baseline_rows,
                post_event_rows=post_event_rows,
                mc_option_lookup=mc_option_lookup,
                episode_metadata=episode_metadata,
                source=robot_key(episode_metadata))
        except ValueError:
            # Not enough usable statements for this episode; skip the item
            # rather than emit one with fewer than four options.
            return None
        question = fill(tmpl_text, event=event_description)

    elif tid == 6:
        task_target_names: Dict[str, str] = template.get("task_target_names", {})

        task_type: Optional[str] = None
        if episode_metadata:
            task_type = episode_metadata.get("task_type")
        if task_type is None and dataset_name:
            if "aursad" in dataset_name.lower():
                task_type = "screwing"
            elif "vorausad" in dataset_name.lower():
                task_type = "pick_and_place"
        target = task_target_names.get(task_type or "pick_and_place", "target location")

        offset_magnitudes = [10, 20, 30, 50, 75, 100, 150]
        dx = random.choice([-1, 0, 1]) * random.choice(offset_magnitudes)
        dy = random.choice([-1, 0, 1]) * random.choice(offset_magnitudes)
        dz = random.choice([-1, 0, 1]) * random.choice(offset_magnitudes)
        if dx == 0 and dy == 0 and dz == 0:
            dx = random.choice([20, 30, 50])
        offset_str = f"[Δx: {dx:+d} mm, Δy: {dy:+d} mm, Δz: {dz:+d} mm]"

        outcome = episode_metadata.get("spatial_outcome") if episode_metadata else None
        if outcome is None:
            return None
        outcome_map = {
            "joint_limit": "A",
            "singularity": "B",
            "success": "C",
            "self_collision": "D",
        }
        answer = outcome_map.get(str(outcome).lower())
        if answer is None:
            return None

        question = fill(tmpl_text, target=target, offset=offset_str)

    elif tid == 4:
        important_features = template.get("important_features")
        signal = pick_constrained_signal(subseries, important_features) if important_features else pick_scalar_signal(subseries)
        if signal is None:
            return None
        if steps_ahead is None or steps_ahead >= len(post_event_rows):
            return None
        target_row = post_event_rows[steps_ahead]
        t_event_ref = _first_timestamp_ms(post_event_rows)
        n_ms = int(float(target_row.get("timestamp_ms", t_event_ref))) - t_event_ref
        signal_value = target_row.get(signal)
        if not isinstance(signal_value, (int, float, np.floating)):
            return None
        answer = round(float(signal_value), 6)
        from src.question_generation.level1.level1 import _signal_display_name
        question = fill(tmpl_text, event=event_description, signal=_signal_display_name(signal), n=n_ms)
        vals = [float(r[signal]) for r in subseries if isinstance(r.get(signal), (int, float, np.floating))]
        std = float(np.std(vals)) if vals else 0.0
        acceptance_bounds = {"signal": signal, "std": round(std, 6), "margin": round(std * 0.75, 6)}

    elif tid == 5:
        important_features = template.get("important_features")
        allowed_bases: Optional[set] = None
        if important_features:
            imp_set = set(important_features)
            allowed_bases = set()
            for feat in imp_set:
                m = re.match(r"^(.*)_(\d+)$", feat)
                if m and int(m.group(2)) in JOINT_INDEX_RANGE:
                    base = m.group(1)
                    if all(f"{base}_{i}" in imp_set for i in range(6)):
                        allowed_bases.add(base)
        joint_signal = pick_joint_indexed_signal_base(subseries, allowed_bases=allowed_bases or None)
        if joint_signal is None:
            return None
        if steps_ahead is None or steps_ahead >= len(post_event_rows):
            return None
        target_row = post_event_rows[steps_ahead]
        t_event_ref = _first_timestamp_ms(post_event_rows)
        n_ms = int(float(target_row.get("timestamp_ms", t_event_ref))) - t_event_ref

        tensor_values: List[float] = []
        tensor_stds: List[float] = []
        for joint_idx in range(6):
            key = f"{joint_signal}_{joint_idx}"
            value = target_row.get(key)
            if not isinstance(value, (int, float, np.floating)):
                return None
            tensor_values.append(round(float(value), 6))
            vals = [float(r[key]) for r in subseries if isinstance(r.get(key), (int, float, np.floating))]
            tensor_stds.append(round(float(np.std(vals)) if vals else 0.0, 6))

        answer = "[" + ",".join(str(v) for v in tensor_values) + "]"
        _JOINT_SIGNAL_DISPLAY = {
            "setpoint_pos": "commanded joint positions",
            "setpoint_speed": "commanded joint velocities",
            "setpoint_acc": "commanded joint accelerations",
            "feedback_pos": "joint positions",
            "feedback_speed": "joint velocities",
            "effort_current": "joint motor currents",
            "effort_target_current": "target joint currents",
            "effort_target_torque": "joint motor torques",
            "control_output": "joint control outputs",
        }
        joint_signal_display = _JOINT_SIGNAL_DISPLAY.get(joint_signal, joint_signal.replace("_", " "))
        question = fill(tmpl_text, event=event_description, joint_signal=joint_signal_display, n=n_ms)
        acceptance_bounds = {"signal": joint_signal, "std": tensor_stds, "margin": [round(s * 0.75, 6) for s in tensor_stds]}

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "event_id": event_obj["id"],
        "answer": answer,
        "acceptance_bounds": acceptance_bounds,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_level3_questions(
    datasets_dir: Path,
    output_dir: Path,
    templates: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    events: List[Dict[str, Any]],
    n: int = 100,
    seed: Optional[int] = None,
    mc_option_lookup: Optional[Dict[str, str]] = None,
    datasets: Optional[List[str]] = None,
    enumerate_mode: bool = False,
    uploader: Optional[HfStreamUploader] = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if mc_option_lookup is None:
        mc_option_lookup = {}

    # Full option records (statement + required_features), needed to check an
    # option's evidence against the episode. The id -> statement lookup above
    # deliberately drops that metadata, so load the catalogue separately.
    try:
        mc_catalogue = load_json(datasets_dir / "mc_options" / "mc_options.json")
        if not isinstance(mc_catalogue, list):
            mc_catalogue = []
    except Exception as exc:
        logger.warning(f"Could not load MC option catalogue for availability filtering: {exc}")
        mc_catalogue = []

    output_dir.mkdir(parents=True, exist_ok=True)

    allowed = datasets if datasets else CF_DATASET_FOLDERS
    cf_pairs = discover_cf_episode_pairs(datasets_dir, allowed)
    if not cf_pairs:
        raise FileNotFoundError(
            f"No paired episodes found under {datasets_dir / 'normalized_episodes'} "
            f"for cf datasets: {allowed}"
        )

    pairs_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for pair in cf_pairs:
        dataset_name = str(pair.get("cf_dataset", ""))
        if not dataset_name:
            continue
        pairs_by_dataset.setdefault(dataset_name, []).append(pair)

    available_cf_datasets = [ds for ds, plist in pairs_by_dataset.items() if plist]
    if not available_cf_datasets:
        raise FileNotFoundError(
            f"No usable cf dataset pairs found under {datasets_dir / 'normalized_episodes'}"
        )

    # LRU-bounded so enumerate-mode runs (which touch every episode) don't OOM.
    episode_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    _EPISODE_CACHE_MAX = 50

    def load_episode(path: Path, split: str = "flat") -> List[Dict[str, Any]]:
        cache_key = f"{path}::{split}"
        if cache_key in episode_cache:
            episode_cache.move_to_end(cache_key)
            return episode_cache[cache_key]
        raw = load_json(path)
        if isinstance(raw, dict):
            episode_cache[cache_key] = raw.get(split, [])
        else:
            episode_cache[cache_key] = raw
        while len(episode_cache) > _EPISODE_CACHE_MAX:
            episode_cache.popitem(last=False)
        return episode_cache[cache_key]

    enum_iter = None
    if enumerate_mode:
        enum_combos = [
            (t, ds, pair)
            for t in templates
            for ds in available_cf_datasets
            for pair in pairs_by_dataset[ds]
        ]
        enum_iter = iter(enum_combos)
        max_total_attempts = len(enum_combos)
        logger.info(f"[enumerate] {len(enum_combos)} (template, cf_pair) combos to attempt; -n={n} caps output")
    else:
        max_total_attempts = n * 20

    generated = 0
    attempts = 0

    while generated < n and attempts < max_total_attempts:
        attempts += 1

        if enum_iter is not None:
            try:
                template, sampled_dataset, pair = next(enum_iter)
            except StopIteration:
                logger.info(f"[enumerate] all combos exhausted at {generated} questions")
                break
        else:
            template = random.choice(templates)
            sampled_dataset = random.choice(available_cf_datasets)
            pair = random.choice(pairs_by_dataset[sampled_dataset])
        non_alt_path = cast(Path, pair["non_alt_path"])
        alt_path = cast(Path, pair["alt_path"])

        fmt = pair.get("format", "flat")
        normal_rows = load_episode(non_alt_path, "baseline" if fmt == "combined" else "flat")
        alt_rows = load_episode(alt_path, "counterfactual" if fmt == "combined" else "flat")

        if not isinstance(normal_rows, list) or not isinstance(alt_rows, list):
            continue
        if not normal_rows or not alt_rows:
            continue

        event_onset_idx = find_event_onset_index(alt_rows)
        if event_onset_idx is None:
            continue

        steps_ahead = random.randint(*STEPS_AHEAD_RANGE)

        sampled_window = sample_window_around_index(
            normal_rows,
            center_index=event_onset_idx,
            min_len=CONTEXT_MIN,
            max_len=CONTEXT_MAX,
            margin=5)
        if sampled_window is None:
            continue

        subseries, subseries_start_index, subseries_length = sampled_window
        if not subseries:
            continue

        alt_start_idx = event_onset_idx - subseries_length + 1
        if alt_start_idx < 0:
            continue
        alt_answer_subseries = alt_rows[alt_start_idx : event_onset_idx + 1]
        if len(alt_answer_subseries) != subseries_length:
            continue

        post_event_rows = alt_rows[event_onset_idx:]
        if not post_event_rows:
            continue

        base_timestamp_ms = _first_timestamp_ms(subseries)
        subseries = normalize_timestamps(subseries, base_timestamp_ms)
        alt_answer_subseries = normalize_timestamps(alt_answer_subseries, base_timestamp_ms)
        post_event_rows = normalize_timestamps(post_event_rows, base_timestamp_ms)

        event_segment_rows, _post_after_event_rows = split_event_segment(post_event_rows)
        if not event_segment_rows:
            continue

        event_time_ms = get_last_timestamp(post_event_rows[:1])

        # Determine the event ID for this episode
        episode_event_id = parse_event_id(post_event_rows[0].get("event", 0))

        # Skip predictive templates for collision events
        if template.get("type") == "predictive" and episode_event_id in COLLISION_EVENT_IDS:
            continue

        if sampled_dataset == "simulations":
            effective_mc_lookup = {k: v for k, v in mc_option_lookup.items() if k not in SIMULATION_EXCLUDED_MC_IDS}
        else:
            effective_mc_lookup = {k: v for k, v in mc_option_lookup.items() if k not in NON_SIMULATION_EXCLUDED_MC_IDS}

        if template.get("type") == "predictive":
            effective_mc_lookup = {k: v for k, v in effective_mc_lookup.items() if k not in PREDICTIVE_EXCLUDED_MC_IDS}
        # Drop options whose evidence this episode does not carry. Without this
        # an item can ask about a channel the context never shows; on the
        # released benchmark that affected 46-74% of options on these
        # templates. Falls back to the unfiltered pool if too few survive.
        # Check the option against what the *context* will show, not the whole
        # episode. The context is trimmed to the template's important_features
        # before it is written, so an episode can carry a channel the model
        # never sees. That is how every TCP-tracking option on L2.2 and L2.3
        # got through: those templates list 30 features and none of them is a
        # TCP channel, yet the episode has all 24.
        _avail_rows = list(subseries) + list(post_event_rows)
        _imp_avail = template.get("important_features")
        if _imp_avail:
            _keep_avail = set(_imp_avail) | {"timestamp_ms"}
            _avail_rows = [
                {k: v for k, v in r.items() if k in _keep_avail} for r in _avail_rows
            ]
        effective_mc_lookup = filter_lookup_by_availability(
            effective_mc_lookup,
            mc_catalogue,
            _avail_rows,
            level=3)


        ep_metadata: Optional[Dict[str, Any]] = None
        ep_fault_id = None
        meta_path = alt_path.parent / f"{alt_path.stem}_metadata.json"
        if meta_path.exists():
            try:
                with meta_path.open(encoding="utf-8") as _f:
                    raw_meta = json.load(_f)
                if sampled_dataset == "simulations":
                    ep_metadata = raw_meta.get("counterfactual", {})
                else:
                    ep_metadata = raw_meta
                ep_fault_id = raw_meta.get("cf_fault_id") or raw_meta.get("fault_id")
                if ep_fault_id is not None:
                    ep_fault_id = int(float(ep_fault_id))
            except Exception:
                pass

        # Build event description from the event ID in the time series
        # Use the event_id to look up root cause; fall back to fault_id from metadata
        _effective_fault_id = episode_event_id if episode_event_id else ep_fault_id
        _rc = root_causes.get(_effective_fault_id, {}) if _effective_fault_id else {}
        _rc_name = _rc.get("root_cause", "")
        if _rc_name:
            _rc_display = _rc_name.replace("_", " ")
            _article = "an" if _rc_display[0] in "aeiou" else "a"
            _event_label = f"{_article} {_rc_display}"
        else:
            _event_label = "an unspecified fault"

        # Include injection timestep unless the event spans the whole episode
        # (event onset at index 0 = episode-wide fault).
        # The timestep must be quoted on the same clock as the context the model
        # sees, which was shifted to start at t=0 by normalize_timestamps above.
        # event_time_ms is the first timestamp of the normalized post_event_rows,
        # i.e. the onset on that shifted clock. Reading timestamp_ms straight off
        # alt_rows here would quote the raw episode clock and point at a timestep
        # that does not exist in the context window.
        _event_onset_idx = find_event_onset_index(alt_rows)
        if _event_onset_idx is not None and _event_onset_idx > 0:
            _event_desc = f"{_event_label} occurs at timestep {event_time_ms} ms"
        else:
            _event_desc = f"{_event_label} occurs"

        filled = fill_template(
            template,
            subseries,
            post_event_rows,
            events,
            effective_mc_lookup,
            t2_ms=event_time_ms,
            answer_subseries=alt_answer_subseries,
            steps_ahead=steps_ahead,
            episode_metadata=ep_metadata,
            dataset_name=sampled_dataset,
            event_description=_event_desc)
        if filled is None:
            continue

        important_features = template.get("important_features")
        context_subseries = subseries
        if important_features:
            keep = set(important_features) | {"timestamp_ms"}
            context_subseries = [{k: v for k, v in row.items() if k in keep} for row in subseries]
        context = build_context(context_subseries)

        item = {
            "id": str(uuid.uuid4()),
            "level": 3,
            "template_id": template["id"],
            "template_type": template["type"],
            "hides": template.get("hides", []),
            "question": filled["question"],
            "options": filled["options"],
            "answer": filled["answer"],
            "acceptance_bounds": filled.get("acceptance_bounds"),
            "provenance": {
                "dataset": pair["cf_dataset"],
                "sampled_subfolder": pair.get("non_alt_subfolder", non_alt_path.stem),
                "counterpart_subfolder": pair.get("alt_subfolder", alt_path.stem),
                "episode": non_alt_path.stem,
                "subseries_start_index": subseries_start_index,
                "subseries_length": subseries_length,
                "event_index_alt": event_onset_idx,
                "event_time_ms": event_time_ms,
            },
            "context": context,
        }

        out_path = output_dir / f"level3_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        logger.info(
            f"✓ [{generated + 1}/{n}] {out_path.name} "
            f"(template {template['id']}, {pair['cf_dataset']}/{pair.get('non_alt_subfolder', non_alt_path.stem)})"
        )
        generated += 1
        if uploader is not None:
            uploader.maybe_flush(generated)

    if uploader is not None:
        uploader.flush_remaining()

    if generated < n:
        logger.warning(f"Only generated {generated}/{n} questions after {attempts} attempts")
    else:
        logger.info(f"Done: {generated} questions written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Level 3 (Counterfactual Reasoning) Q&A pairs."
    )
    repo_root = Path(__file__).resolve().parents[3]

    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=repo_root / "data",
        help="Root data directory (default: <repo>/data)")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "output" / "questions" / "level3",
        help="Output directory (default: <repo>/output/questions/level3)")
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate (cap; in --enumerate mode this is an upper bound, not a target)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to sample from (default: all). Choices: {CF_DATASET_FOLDERS}")
    parser.add_argument(
        "--enumerate",
        dest="enumerate_mode",
        action="store_true",
        help="Walk every (template x cf_pair) combination deterministically instead "
             "of random sampling. -n becomes an upper cap. Combinations whose "
             "pair does not satisfy the template's preconditions are skipped.")
    add_streaming_args(parser)
    parser.add_argument(
        "--template-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict generation to these template ids (default: all). Useful "
             "for regenerating a single template without rerunning the rest.")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    templates = load_templates(Path(__file__).with_name("question_template.json"))
    if getattr(args, "template_ids", None):
        wanted = set(args.template_ids)
        templates = [t for t in templates if t["id"] in wanted]
        if not templates:
            parser.error(f"No templates match --template-ids {sorted(wanted)}")

    root_causes = load_root_causes(args.datasets_dir / "labelling" / "rca" / "root_causes.json")
    events = load_events(args.datasets_dir / "labelling" / "events.json")
    mc_option_lookup = load_mc_option_lookup(
        args.datasets_dir / "mc_options" / "mc_options.json",
        level=3)

    args.output.mkdir(parents=True, exist_ok=True)
    uploader = make_uploader_from_args(args, level=3, output_dir=args.output)

    generate_level3_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        events=events,
        mc_option_lookup=mc_option_lookup,
        enumerate_mode=args.enumerate_mode,
        uploader=uploader,
        n=args.n,
        seed=args.seed,
        datasets=args.datasets)


if __name__ == "__main__":
    main()
