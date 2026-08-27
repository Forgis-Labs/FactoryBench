"""
Level 1 question generator: State Understanding.

Reads normalized episode JSON files from aursad or vorausad datasets,
samples random sub-series, and fills Level 1 question templates.
Ground truth is derived directly from episode labels (fault_label) and metadata.

Output: output/questions/level1/level1_{NNNN}.json

Usage:
    python -m src.question_generation.level1.level1 -n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import uuid
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data._decimation import decimate_dataframe
from src.question_generation.utils.hf_streaming import (
    HfStreamUploader,
    add_streaming_args,
    make_uploader_from_args)
from src.question_generation.utils.io import load_json, load_root_causes, load_templates
from src.question_generation.utils.phases import PHASE_NAMES as _SHARED_PHASE_NAMES
from src.question_generation.utils.pair_balance import (
    build_index as _build_pair_index,
    cycle_targets as _cycle_targets,
    sample_pair as _sample_balanced_pair)
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset,
    encode_chunk,
    fill)
from src.question_generation.utils.time_series import (
    pick_fault_label)
from src.question_generation.utils.relevance import (
    is_enabled as relevance_enabled,
    load_specs as load_relevance_specs,
    relevance_report,
    sample_with_relevance,
    validate_relevance)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["aursad", "vorausad", "factorywave", "factorywave_kuka"]
SEVERITY_ORDER: Dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

# Precise per-fault ranking loaded from anomaly_ranking.json (fault_id → rank 1..N, higher = more severe)
_ANOMALY_RANKING: Dict[int, int] = {}

# Maps dataset name → machine_id as defined in machines.json
DATASET_MACHINE_ID: Dict[str, int] = {
    "aursad": 0,    # UR3e
    "vorausad": 2,  # Yu cobot
    "factorywave": 0,     # UR3e (FactoryWave real-robot recordings)
    "factorywave_kuka": 3,  # KUKA KR 10 R1100-2 (FactoryWave real-robot recordings)
}

_NO_ANOMALY_DESC = "No anomaly is present; the machine is operating nominally."

CONTEXT_MIN = 32
CONTEXT_MAX = 64

# Phase index → human-readable name per task. Defined in utils.phases so the
# prompt builder can render a legend using the exact wording that lands in the
# question text; re-exported here under its original name because the rest of
# this module (and existing question JSON) refers to it that way.
PHASE_NAMES = _SHARED_PHASE_NAMES

# Variable name → human-readable signal name
SIGNAL_DISPLAY_NAMES: Dict[str, str] = {}
# Build automatically for indexed signals
for _i in range(6):
    SIGNAL_DISPLAY_NAMES[f"feedback_pos_{_i}"] = f"the position of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"feedback_speed_{_i}"] = f"the velocity of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_pos_{_i}"] = f"the commanded position of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_speed_{_i}"] = f"the commanded velocity of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_acc_{_i}"] = f"the commanded acceleration of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"effort_current_{_i}"] = f"the motor current of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"effort_target_current_{_i}"] = f"the target current of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"effort_target_torque_{_i}"] = f"the motor torque of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"control_output_{_i}"] = f"the control output of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"joint_temp_{_i}"] = f"the temperature of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"joint_mode_{_i}"] = f"the mode of joint {_i}"
    SIGNAL_DISPLAY_NAMES[f"joint_voltage_{_i}"] = f"the voltage of joint {_i}"
for _i, _axis in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
    SIGNAL_DISPLAY_NAMES[f"setpoint_tcp_{_i}"] = f"the commanded TCP {_axis}"
    SIGNAL_DISPLAY_NAMES[f"setpoint_tcp_speed_{_i}"] = f"the commanded TCP speed {_axis}"
    SIGNAL_DISPLAY_NAMES[f"feedback_tcp_{_i}"] = f"the TCP {_axis}"
    SIGNAL_DISPLAY_NAMES[f"feedback_tcp_speed_{_i}"] = f"the TCP speed {_axis}"
for _i, _axis in enumerate(["x", "y", "z"]):
    SIGNAL_DISPLAY_NAMES[f"true_force_{_i}"] = f"the TCP force {_axis}"
    SIGNAL_DISPLAY_NAMES[f"true_force_{_i + 3}"] = f"the TCP torque {_axis}"
    SIGNAL_DISPLAY_NAMES[f"est_contact_force_{_i}"] = f"the estimated contact force {_axis}"
    SIGNAL_DISPLAY_NAMES[f"vibration_{_i}"] = f"the vibration {_axis}"
SIGNAL_DISPLAY_NAMES["gripper_command"] = "the gripper force"
SIGNAL_DISPLAY_NAMES["robot_mode"] = "the robot mode"
SIGNAL_DISPLAY_NAMES["safety_mode"] = "the safety mode"
SIGNAL_DISPLAY_NAMES["runtime_state"] = "the runtime state"
SIGNAL_DISPLAY_NAMES["speed_scaling"] = "the speed scaling"
SIGNAL_DISPLAY_NAMES["robot_current"] = "the robot current"
SIGNAL_DISPLAY_NAMES["main_voltage"] = "the main voltage"
SIGNAL_DISPLAY_NAMES["robot_voltage"] = "the robot voltage"
SIGNAL_DISPLAY_NAMES["tool_momentum"] = "the tool momentum"


def _phase_display_name(phase_raw: str, task: str = "pick_and_place") -> str:
    """Convert a phase index/name to a human-readable name."""
    task_phases = PHASE_NAMES.get(task, PHASE_NAMES.get("pick_and_place", {}))
    name = task_phases.get(str(phase_raw))
    if name:
        return name
    # Already a name string (not an index)
    return str(phase_raw).replace("_", " ")


def _signal_display_name(signal: str) -> str:
    """Convert a signal variable name to a human-readable name."""
    if signal in SIGNAL_DISPLAY_NAMES:
        return SIGNAL_DISPLAY_NAMES[signal]
    return signal.replace("_", " ")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _first_timestamp_ms(rows: List[Dict[str, Any]]) -> int:
    for row in rows:
        ts = row.get("timestamp_ms")
        try:
            if ts is not None:
                return int(float(ts))
        except (TypeError, ValueError):
            continue
    return 0


def normalize_timestamps(rows: List[Dict[str, Any]], base: int) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        ts = r.get("timestamp_ms")
        try:
            if ts is not None:
                r["timestamp_ms"] = int(float(ts)) - base
        except (TypeError, ValueError):
            pass
        out.append(r)
    return out


def sample_subseries(
    rows: List[Dict[str, Any]],
    min_len: int = CONTEXT_MIN,
    max_len: int = CONTEXT_MAX) -> Optional[Tuple[List[Dict[str, Any]], int]]:
    """Return (subseries, start_index) or None if the episode is too short."""
    n = len(rows)
    if n < min_len:
        return None
    length = random.randint(min_len, min(max_len, n))
    start = random.randint(0, n - length)
    return rows[start : start + length], start


def _episode_task(
    ep_path: Path,
    dataset: str,
    dataset_index: Dict[str, Dict[str, Any]]) -> str:
    """Resolve task for an episode: prefer per-episode metadata, fall back to dataset.json."""
    meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            task = meta.get("task")
            if task:
                return str(task)
        except Exception:
            pass
    return str(dataset_index.get(dataset, {}).get("task_id", ""))


def load_anomaly_ranking(path: Path) -> Dict[int, int]:
    """Load anomaly_ranking.json as {fault_id: rank}.

    Higher rank = more severe (the JSON lists least-to-most, so rank 1 is
    the least severe and rank N is the most severe).
    """
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load anomaly ranking from {path}: {exc}")
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("ranking_least_to_most_severe", [])
    return {
        int(entry["fault_id"]): int(entry["rank"])
        for entry in entries
        if isinstance(entry, dict) and "fault_id" in entry and "rank" in entry
    }


def get_severity_rank(fault_id: int, root_cause: Dict[str, Any]) -> int:
    """Return a numeric severity rank for a fault.

    Uses the precise per-fault ranking from anomaly_ranking.json if available,
    otherwise falls back to the coarse severity_levels mapping.
    """
    if _ANOMALY_RANKING and fault_id in _ANOMALY_RANKING:
        return _ANOMALY_RANKING[fault_id]
    levels = root_cause.get("severity_levels", ["none"])
    return max((SEVERITY_ORDER.get(s, 0) for s in levels), default=0)


def normalize_mc_option_id(value: str) -> str:
    text = str(value).strip()
    match = re.match(r"^(?:mc|l2_mc)_(\d+)$", text, re.IGNORECASE)
    if not match:
        return text
    return f"mc_{int(match.group(1)):03d}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_mc_option_lookup(path: Path, level: int) -> Dict[str, str]:
    """Load MC options as a map: id -> statement, filtered by level."""
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


def load_anomaly_lookup(path: Path) -> Dict[str, str]:
    """Load anomalies.json as {anomaly_name: description}."""
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load anomalies from {path}: {exc}")
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        item["anomaly_name"]: item["description"]
        for item in raw
        if isinstance(item, dict) and "anomaly_name" in item
    }


def load_dataset_index(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load dataset.json as {dataset_id: dataset_dict}."""
    try:
        raw = load_json(path)
    except Exception as exc:
        logger.warning(f"Could not load dataset index from {path}: {exc}")
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        item["dataset_id"]: item
        for item in raw
        if isinstance(item, dict) and "dataset_id" in item
    }


# ---------------------------------------------------------------------------
# Answer builders (one per template)
# ---------------------------------------------------------------------------


def build_anomaly_single_select(
    fault_label: int,
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
    fault_prior: Optional[Dict[int, float]] = None) -> Tuple[Dict[str, str], str]:
    """
    Build 4 single-select options + single letter answer for an
    anomaly-identification question.

    The 4 options are drawn from the canonical root_causes.json pool, plus
    a synthetic "No anomaly" entry added to that pool. The correct option
    (the root cause matching this episode's fault_label, or "No anomaly"
    for nominal episodes) is always one of the 4. ``anomaly_lookup`` is
    accepted for backward-compatible signature but no longer used.

    ``fault_prior`` weights the distractor draw by how often each fault is the
    correct answer. Without it the distractors come uniformly from the whole
    catalogue while the answer follows the corpus, so how often an option is
    correct given that it appears varies from 0.00 to 0.70 across anomalies.
    That let a model score 0.608 against a chance rate of 0.250 by learning
    which strings tend to win, without reading the time series at all. Drawing
    distractors from the same distribution as the answers pushes every option
    back to 0.25 and drops anomalies the corpus never exhibits, including the
    "no anomaly" entry on levels that only use faulty episodes.
    """
    NO_ANOMALY_FID = 0
    is_normal = fault_label == 0 or fault_label not in root_causes

    def _desc_for(rc: Dict[str, Any]) -> str:
        return rc.get("description") or rc.get("root_cause", "").replace("_", " ")

    # Pool: every root cause from root_causes.json + a "no anomaly" entry.
    pool: List[Tuple[int, str]] = []
    for fid, rc in root_causes.items():
        if not isinstance(fid, int) or fid <= 0:
            continue
        d = _desc_for(rc)
        if d:
            pool.append((fid, d))
    pool.append((NO_ANOMALY_FID, _NO_ANOMALY_DESC))

    if is_normal:
        correct_fid = NO_ANOMALY_FID
        correct_desc = _NO_ANOMALY_DESC
    else:
        correct_fid = fault_label
        correct_desc = _desc_for(root_causes[fault_label])

    distractors = [(fid, d) for (fid, d) in pool if fid != correct_fid]
    picked: List[Tuple[int, str]] = []
    if fault_prior:
        weighted = [(fid, d, float(fault_prior.get(fid, 0.0)))
                    for (fid, d) in distractors]
        weighted = [w for w in weighted if w[2] > 0]
        # Weighted draw without replacement: take three, removing each as it
        # is taken so an option cannot appear twice in one item.
        while weighted and len(picked) < 3:
            total = sum(w[2] for w in weighted)
            cut = random.random() * total
            run = 0.0
            for i, (fid, d, wt) in enumerate(weighted):
                run += wt
                if run >= cut:
                    picked.append((fid, d))
                    weighted.pop(i)
                    break
            else:
                picked.append((weighted[-1][0], weighted[-1][1]))
                weighted.pop()
    if len(picked) < 3:
        # Prior too sparse to fill the item; top up uniformly rather than
        # emitting an item with fewer than four options.
        taken = {fid for fid, _ in picked}
        rest = [(fid, d) for (fid, d) in distractors if fid not in taken]
        random.shuffle(rest)
        picked.extend(rest[: 3 - len(picked)])
    chosen: List[Tuple[str, bool]] = [(correct_desc, True)] + [
        (d, False) for (_, d) in picked[:3]
    ]
    random.shuffle(chosen)
    chosen = chosen[:4]

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {}
    answer = "A"
    for i, (desc, is_correct) in enumerate(chosen):
        options[labels[i]] = desc
        if is_correct:
            answer = labels[i]
    return options, answer


_COMPARATIVE_OPTION_ORDER = ["mc_020", "mc_022", "mc_023", "mc_026"]


def _uniform_window(
    rows: List[Dict[str, Any]],
    min_len: int,
    max_len: int) -> Optional[Tuple[List[Dict[str, Any]], int, str]]:
    """Pick a random subseries of length in ``[min_len, max_len]``.

    Returns (subseries, start_idx, sampler_tag) or None if the episode is
    too short. Used in L1 instead of ``sample_with_relevance`` because L1
    operates on nominal episodes only, anomaly relevance specs are not
    applicable, so we sample uniformly.
    """
    n = len(rows)
    if n < min_len:
        return None
    length = random.randint(min_len, min(max_len, n))
    start = random.randint(0, n - length)
    return rows[start:start + length], start, "uniform"


def pick_phase_isolation_candidates(
    rows: List[Dict[str, Any]],
    n: int = 3,
    min_phase_length: int = 3) -> List[Tuple[str, int, int]]:
    """Return up to ``n`` distinct phase candidates for a phase-isolation
    question.

    Sampling rule (per the design):

    * If the episode has ≥3 distinct phases, exclude the **first and last**
      phases of the task and sample inner ones.
    * If the episode has <3 phases (so excluding edges would leave nothing
      to sample), fall back to using ALL phases.
    * Each candidate must be at least ``min_phase_length`` timesteps so the
      window-start question is meaningfully discriminative.

    Returns up to ``n`` distinct phases (random choice without replacement);
    the caller drives one question per returned phase, multiplying yield
    per (template, episode) combo.
    """
    all_phases: List[Tuple[str, int, int]] = []
    current_phase = None
    phase_start = 0
    for i, row in enumerate(rows):
        p = row.get("task_phase")
        if p != current_phase:
            if current_phase is not None and str(current_phase) not in ("None", "none", ""):
                all_phases.append((str(current_phase), phase_start, i - phase_start))
            current_phase = p
            phase_start = i
    if current_phase is not None and str(current_phase) not in ("None", "none", ""):
        all_phases.append((str(current_phase), phase_start, len(rows) - phase_start))

    if len(all_phases) >= 3:
        pool = all_phases[1:-1]
    else:
        pool = all_phases
    pool = [t for t in pool if t[2] >= min_phase_length]
    if not pool:
        return []
    k = min(n, len(pool))
    return random.sample(pool, k)


def _modal_phase(rows: List[Dict[str, Any]]) -> Optional[str]:
    phases = [
        str(r.get("task_phase"))
        for r in rows
        if r.get("task_phase") not in (None, "None", "")
    ]
    if not phases:
        return None
    return Counter(phases).most_common(1)[0][0]


def build_comparative_multi_select(
    fault_a: int,
    fault_b: int,
    machine_id_a: int,
    machine_id_b: int,
    task_id_a: str,
    task_id_b: str,
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    mc_lookup: Dict[str, str]) -> Optional[Tuple[Dict[str, str], str]]:
    """Build fixed-order options + TFFT answer for template 3.

    Options are always presented in the order
    [mc_020, mc_022, mc_023, mc_026] (labels A..D) and the answer encodes
    whether each corresponding statement holds, independently:
      - mc_020: different robots
      - mc_022: different anomalous states
      - mc_023: different tasks
      - mc_026: same task but different modal phase

    Returns None when no statement is true (no meaningful change).
    """
    if not mc_lookup:
        return None

    option_ids = [normalize_mc_option_id(x) for x in _COMPARATIVE_OPTION_ORDER]
    statements = [mc_lookup.get(oid) for oid in option_ids]
    if not all(statements):
        return None

    different_robots = machine_id_a != machine_id_b
    different_anomalous_state = (fault_a == 0) != (fault_b == 0) or (
        fault_a != 0 and fault_b != 0 and fault_a != fault_b
    )
    different_tasks = task_id_a != task_id_b

    mode_a = _modal_phase(rows_a)
    mode_b = _modal_phase(rows_b)
    same_task_different_phases = (
        (not different_tasks)
        and mode_a is not None
        and mode_b is not None
        and mode_a != mode_b
    )

    flags = [
        different_robots,
        different_anomalous_state,
        different_tasks,
        same_task_different_phases,
    ]
    if not any(flags):
        return None

    labels = ["A", "B", "C", "D"]
    options: Dict[str, str] = {lbl: stmt for lbl, stmt in zip(labels, statements)}
    answer = "".join("T" if f else "F" for f in flags)
    return options, answer


def build_severity_ranking(
    segments: List[Tuple[List[Dict[str, Any]], int]],
    root_causes: Dict[int, Dict[str, Any]],
    important_features: Optional[List[str]] = None,
    min_chunk: int = 5,
    max_chunk: int = 7,
    relevance_specs: Optional[Dict[int, Dict[str, Any]]] = None,
    tasks: Optional[List[str]] = None) -> Optional[Tuple[Dict[str, str], str]]:
    """
    Build ranking options + answer string for template 5.
    Each option is a short encoded chunk from one episode filtered by important_features.
    Answer ranks options from most to least severe.

    When relevance_specs and tasks are provided, each chunk is anchored on the
    episode's fault-relevance spec so the ranking compares evidence-bearing
    segments, not arbitrary slices.
    """
    keep = set(important_features) if important_features else None
    labels = ["A", "B", "C", "D"]
    labeled: List[Tuple[str, str, int]] = []
    specs = relevance_specs or {}
    tasks = tasks or []

    for i, (rows, fault_label) in enumerate(segments[:4]):
        rc = root_causes.get(fault_label, root_causes.get(0, {}))
        srank = get_severity_rank(fault_label, rc)

        spec = specs.get(fault_label)
        task = tasks[i] if i < len(tasks) else ""
        chunk_result = sample_with_relevance(rows, fault_label, spec, task, min_chunk, max_chunk)
        if chunk_result is None:
            return None
        chunk, _, _ = chunk_result
        stripped = [
            {k: v for k, v in r.items() if k != "timestamp_ms" and (keep is None or k in keep)}
            for r in chunk
        ]
        encoded = encode_chunk(stripped)
        labeled.append((labels[i], encoded, srank))

    if len(labeled) < 4:
        return None

    sorted_by_severity = sorted(labeled, key=lambda x: x[2], reverse=True)
    answer = "".join(x[0] for x in sorted_by_severity)
    options = {label: encoded for label, encoded, _ in labeled}
    return options, answer


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------


def fill_template(
    template: Dict[str, Any],
    rows: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
    mc_option_lookup: Dict[str, str],
    rows_b: Optional[List[Dict[str, Any]]] = None,
    machine_id: int = -1,
    machine_id_b: int = -1,
    task_id: str = "",
    task_id_b: str = "",
    severity_segments: Optional[List[Tuple[List[Dict[str, Any]], int]]] = None,
    severity_relevance_specs: Optional[Dict[int, Dict[str, Any]]] = None,
    severity_tasks: Optional[List[str]] = None,
    phase_override: Optional[Tuple[str, int, int]] = None) -> Optional[Dict[str, Any]]:
    """
    Fill a Level 1 question template.

    Template IDs:
      1 - phase segmentation (numerical)
      2 - anomaly detection (single-select)
      3 - comparative change detection (multi-select, TFFT)
      5 - severity ranking
      6 - robot identification (single-select)

    For template 1, callers may pass ``phase_override`` (a ``(phase_name,
    start_idx, length)`` triple) to drive multiple questions per (template,
    episode) combo, each anchored on a different inner phase.
    """
    tid = template["id"]
    tmpl_text: str = template["template"]
    answer_format: Dict[str, Any] = template["answer_format"]

    fault_label = pick_fault_label(rows)
    options: Dict[str, Any] = {}
    answer: Any = None
    acceptance_bounds = None

    if tid == 1:
        if phase_override is not None:
            phase_name, phase_start_idx, phase_length = phase_override
        else:
            candidates = pick_phase_isolation_candidates(rows, n=1)
            if not candidates:
                return None
            phase_name, phase_start_idx, phase_length = candidates[0]
        window_length = phase_length + 5
        # Answer is the t= value the model sees, not the row index. Acceptance
        # is any t inside the t-window of the GT row's ±3 neighbors (asymmetric
        # bounds at the subseries edges).
        gt_t = int(round(float(rows[phase_start_idx]["timestamp_ms"])))
        low_idx = max(0, phase_start_idx - 3)
        high_idx = min(len(rows) - 1, phase_start_idx + 3)
        t_low = int(round(float(rows[low_idx]["timestamp_ms"])))
        t_high = int(round(float(rows[high_idx]["timestamp_ms"])))
        answer = gt_t
        _TASK_DISPLAY = {"pick_and_place": "pick-and-place", "screwing": "screwing", "peg_in_hole": "peg-in-hole"}
        task_display = _TASK_DISPLAY.get(task_id, task_id.replace("_", " ") if task_id else "manipulation")
        question = fill(tmpl_text, task=task_display, phase=_phase_display_name(phase_name, task_id), window_length=window_length)
        acceptance_bounds = {"min": t_low, "max": t_high}

    elif tid == 2:
        if not anomaly_lookup:
            return None
        options, answer = build_anomaly_single_select(fault_label, root_causes, anomaly_lookup)
        question = tmpl_text

    elif tid == 3:
        if rows_b is None:
            return None
        fault_b = pick_fault_label(rows_b)
        result = build_comparative_multi_select(
            fault_label, fault_b, machine_id, machine_id_b,
            task_id, task_id_b, rows, rows_b, mc_option_lookup)
        if result is None:
            return None
        options, answer = result
        question = tmpl_text

    elif tid == 5:
        if not severity_segments or len(severity_segments) < 4:
            return None
        important_features = template.get("important_features")
        result = build_severity_ranking(
            severity_segments, root_causes, important_features=important_features,
            relevance_specs=severity_relevance_specs,
            tasks=severity_tasks)
        if result is None:
            return None
        options, answer = result
        question = tmpl_text

    elif tid == 6:
        # Robot identification: answer is determined by machine_id
        robot_names = {
            0: "Universal Robots UR3e",
            2: "Agile Robots Yu 5 Industrial",
            3: "KUKA KR 10 R1100-2",
        }
        if machine_id not in robot_names:
            return None
        correct_name = robot_names[machine_id]
        fixed = answer_format.get("fixed_options", list(robot_names.values()))
        labels = ["A", "B", "C"][:len(fixed)]
        entries = list(fixed)
        random.shuffle(entries)
        options = {}
        answer = "A"
        for i, name in enumerate(entries):
            options[labels[i]] = name
            if name == correct_name:
                answer = labels[i]
        question = tmpl_text

    elif tid == 7:
        # Prediction: given the subseries, predict a signal value n steps ahead
        important_features = template.get("important_features", [])
        # Pick a signal that exists and has numeric values
        candidate_signals = [
            s for s in important_features
            if any(isinstance(row.get(s), (int, float)) for row in rows[-5:])
        ]
        if not candidate_signals:
            return None
        signal = random.choice(candidate_signals)
        # steps_ahead: how many rows beyond the given context we ask the model
        # to forecast. The prompt reports the horizon in milliseconds so that
        # it aligns with the `t=<ms>` axis of the visible time series rather
        # than being off by a factor of the sample period; the ms offset is
        # computed at generation time from the actual timestamps.
        steps_ahead = random.randint(1, 10)
        question = fill(tmpl_text, signal=_signal_display_name(signal), n=steps_ahead)
        answer = None  # set by generation loop (which also rewrites `n` in ms)
        acceptance_bounds = {"signal": signal, "steps_ahead": steps_ahead}

    else:
        logger.warning(f"Unknown template id: {tid}")
        return None

    return {
        "question": question,
        "answer_format": answer_format,
        "options": options,
        "answer": answer,
        "acceptance_bounds": acceptance_bounds,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


def generate_level1_questions(
    datasets_dir: Path,
    output_dir: Path,
    templates: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    anomaly_lookup: Dict[str, str],
    mc_option_lookup: Dict[str, str],
    dataset_index: Dict[str, Dict[str, Any]],
    relevance_specs: Optional[Dict[int, Dict[str, Any]]] = None,
    n: int = 100,
    seed: Optional[int] = None,
    datasets: Optional[List[str]] = None,
    enumerate_mode: bool = False,
    uploader: Optional[HfStreamUploader] = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    relevance_specs = relevance_specs or {}

    allowed = datasets if datasets else VALID_DATASETS
    by_dataset = discover_episodes_by_dataset(datasets_dir, allowed)
    episodes_by_dataset = {ds: paths for ds, paths in by_dataset.items() if paths}
    available_datasets = list(episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError(
            f"No usable datasets found under {datasets_dir / 'normalized_episodes'} "
            f"for datasets: {allowed}"
        )

    all_episode_paths: List[Tuple[str, Path]] = [
        (ds, ep) for ds, eps in episodes_by_dataset.items() for ep in eps
    ]

    # LRU-bounded so enumerate-mode runs (which touch every episode) don't OOM.
    episode_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    _EPISODE_CACHE_MAX = 50
    _normal_cache: Dict[str, bool] = {}

    def load_episode(path: Path) -> List[Dict[str, Any]]:
        key = str(path)
        if key in episode_cache:
            episode_cache.move_to_end(key)
            return episode_cache[key]
        raw = load_json(path)
        if isinstance(raw, dict):
            raw = raw.get("baseline", raw.get("flat", []))
        episode_cache[key] = raw
        while len(episode_cache) > _EPISODE_CACHE_MAX:
            episode_cache.popitem(last=False)
        return episode_cache[key]

    def is_normal_episode(rows: List[Dict[str, Any]], path: Path) -> bool:
        """Return True if the episode has no anomalies (fault_label is 0 or absent throughout)."""
        key = str(path)
        if key in _normal_cache:
            return _normal_cache[key]
        result = all(
            int(row.get("fault_label", 0) or 0) == 0
            for row in rows
            if row.get("fault_label") is not None
        )
        _normal_cache[key] = result
        return result

    def _episode_is_nominal_fast(path: Path) -> bool:
        """Cheap nominal check: prefer metadata, fall back to first-row peek.

        Avoids parsing/caching the full episode (multi-MB) when all we need is
        fault_label. fault_label is episode-wide in all current datasets, so the
        first row's value is authoritative.
        """
        key = str(path)
        if key in _normal_cache:
            return _normal_cache[key]
        meta_path = path.with_name(path.stem + "_metadata.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if "fault_id" in meta:
                    fid = meta.get("fault_id")
                    result = fid is None or int(float(fid)) == 0
                    _normal_cache[key] = result
                    return result
            except Exception:
                pass
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw = raw.get("baseline", raw.get("flat", []))
            if isinstance(raw, list) and raw:
                fl = raw[0].get("fault_label")
                result = fl is None or int(fl) == 0
            else:
                result = True
        except Exception:
            result = False
        _normal_cache[key] = result
        return result

    # Level 1 operates on nominal episodes only (no faults). Filter once so every
    # template branch, single, paired, or quadrupled, picks from the same pool.
    # The result is cached on disk because the first scan reads ~14k episode JSONs;
    # subsequent runs load the cache (a tiny file mapping path → bool) instead.
    nominal_cache_path = datasets_dir / ".level1_nominal_cache.json"
    cache: Dict[str, bool] = {}
    if nominal_cache_path.exists():
        try:
            cache = json.loads(nominal_cache_path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}
    _normal_cache.update({k: v for k, v in cache.items() if isinstance(v, bool)})

    all_paths = [p for paths in episodes_by_dataset.values() for p in paths]
    uncached = [p for p in all_paths if str(p) not in _normal_cache]
    if uncached:
        from concurrent.futures import ThreadPoolExecutor
        logger.info(f"First-pass nominal filter: scanning {len(uncached)} new episodes (cached: {len(all_paths) - len(uncached)}) ...")
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(_episode_is_nominal_fast, uncached))
        try:
            nominal_cache_path.write_text(json.dumps(_normal_cache), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Could not persist nominal cache: {exc}")
    else:
        logger.info(f"Loaded nominal-cache for {len(all_paths)} episodes from {nominal_cache_path.name}")

    nominal_episodes_by_dataset: Dict[str, List[Path]] = {}
    for ds, paths in episodes_by_dataset.items():
        nominals = [p for p in paths if _normal_cache.get(str(p), False)]
        if nominals:
            nominal_episodes_by_dataset[ds] = nominals
    available_datasets = list(nominal_episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError(
            f"No nominal episodes found under {datasets_dir / 'normalized_episodes'} "
            f"for datasets: {allowed}. Level 1 requires fault-free episodes."
        )
    # Template 3 compares two streams and asks whether their anomalous states
    # differ. Restricting every L1 template to nominal episodes made that
    # proposition False in all 2,407 released items, so it carried no
    # information at all. Keep the full pool for the comparison template while
    # the single-stream templates stay nominal-only.
    # ---- balanced pairing support for template 3 -------------------------
    _balance_targets = _cycle_targets(max(n * 4, 64))
    _pair_attr_cache: Dict[str, Dict[str, Any]] = {}
    _pair_pool: List[Dict[str, Any]] = []
    _pair_index: Dict[Any, List[Dict[str, Any]]] = {}
    _group_pairs_cache: Dict[Any, List[Any]] = {}
    _used_pairs: set = set()
    _phase_cache: Dict[str, Optional[str]] = {}

    def _pair_attrs(ds: str, ep_path: Path) -> Dict[str, Any]:
        """Cheap comparison attributes for one episode, read from metadata."""
        key = str(ep_path)
        cached = _pair_attr_cache.get(key)
        if cached is not None:
            return cached
        fault = 0
        robot = None
        meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
        if meta_path.exists():
            try:
                meta = load_json(meta_path) or {}
                raw = meta.get("fault_id") or meta.get("cf_fault_id") or 0
                fault = int(float(raw)) if raw else 0
                # The dataset folder is not the robot: FactoryWave keeps UR and
                # KUKA episodes side by side under one name, so keying identity
                # off the folder made "different robots" unsatisfiable.
                robot = meta.get("robot_type") or meta.get("machine_id")
            except Exception:
                fault, robot = 0, None
        attrs = {
            "key": key,
            "dataset": ds,
            "path": ep_path,
            "robot": str(robot).lower() if robot is not None else f"ds:{DATASET_MACHINE_ID.get(ds, -1)}",
            "task": _episode_task(ep_path, ds, dataset_index),
            "fault": fault,
        }
        _pair_attr_cache[key] = attrs
        return attrs

    def _episode_modal_phase(ep_path: Path) -> Optional[str]:
        key = str(ep_path)
        if key in _phase_cache:
            return _phase_cache[key]
        phase = None
        try:
            rows = load_episode(ep_path)
            if isinstance(rows, list) and rows:
                phase = _modal_phase(rows)
        except Exception:
            phase = None
        if len(_phase_cache) > 4096:
            _phase_cache.clear()
        _phase_cache[key] = phase
        return phase

    def _phases_differ(path_a: Path, path_b: Path) -> bool:
        """Proposition D holds only when the two runs sit in different phases."""
        pa, pb = _episode_modal_phase(path_a), _episode_modal_phase(path_b)
        return pa is not None and pb is not None and pa != pb

    def _machine_id_for(ds: str, ep_path: Path) -> int:
        """Machine id for one episode.

        FactoryWave records a UR3 and a KUKA under a single dataset name, so
        deriving the machine from the dataset made "different robots" false for
        every within-FactoryWave pair regardless of which arms were actually
        compared. Prefer the episode's own robot_type and fall back to the
        dataset mapping for sources that carry one robot throughout.
        """
        robot = str(_pair_attrs(ds, ep_path).get("robot") or "")
        by_robot = {"ur3": 0, "ur3e": 0, "yu": 2, "kuka": 3}
        for name, mid in by_robot.items():
            if robot.lower().startswith(name):
                return mid
        return DATASET_MACHINE_ID.get(ds, -1)

    comparison_episodes_by_dataset = {
        ds: list(paths) for ds, paths in episodes_by_dataset.items() if paths
    }
    episodes_by_dataset = nominal_episodes_by_dataset
    all_episode_paths = [
        (ds, ep) for ds, eps in episodes_by_dataset.items() for ep in eps
    ]
    logger.info(
        f"Filtered to nominal episodes: "
        + ", ".join(f"{ds}={len(p)}" for ds, p in episodes_by_dataset.items())
    )

    # Templates 2 (anomaly detection) and 5 (severity ranking) require faulty
    # episodes to be meaningful, disabled in the nominal-only regime.
    _DISABLED_TEMPLATE_IDS: set = {2, 5}
    usable_templates = [t for t in templates if t["id"] not in _DISABLED_TEMPLATE_IDS]
    if not usable_templates:
        raise ValueError("No usable templates for Level 1 generation.")

    generated = 0
    attempts = 0

    # Build a deterministic (template, ds, ep_path) iterator if --enumerate.
    # The outer loop walks every combo once; secondary episodes for paired
    # templates (tid 3) and group templates (tid 5) are still sampled at
    # random as before, since exhaustive enumeration there is combinatorial.
    enum_iter = None
    if enumerate_mode:
        enum_combos = [
            (t, ds, ep)
            for t in usable_templates
            for ds in available_datasets
            for ep in episodes_by_dataset[ds]
        ]
        enum_iter = iter(enum_combos)
        max_total_attempts = len(enum_combos)
        logger.info(f"[enumerate] {len(enum_combos)} (template, episode) combos to attempt; -n={n} caps output")
    else:
        max_total_attempts = n * 20

    while generated < n and attempts < max_total_attempts:
        attempts += 1
        if enum_iter is not None:
            try:
                template, _enum_ds, _enum_ep = next(enum_iter)
            except StopIteration:
                logger.info(f"[enumerate] all combos exhausted at {generated} questions")
                break
        else:
            template = random.choice(usable_templates)
            _enum_ds = None
            _enum_ep = None
        tid = template["id"]
        important_features = template.get("important_features")

        # ------------------------------------------------------------------
        # Templates 1, 2, 6: single episode, sampled subseries
        # (template 1 isolates a phase; the subseries must contain at least one
        # entirely-contained phase, otherwise we reject and try another episode, # see fill_template's tid==1 branch which returns None on no inner phase.)
        # ------------------------------------------------------------------
        if tid in (1, 2, 6):
            if _enum_ep is not None:
                ds = _enum_ds
                ep_path = _enum_ep
            else:
                ds = random.choice(available_datasets)
                ep_path = random.choice(episodes_by_dataset[ds])
            rows = load_episode(ep_path)
            if not isinstance(rows, list) or len(rows) < CONTEXT_MIN:
                continue

            ep_fault_id = pick_fault_label(rows)
            ep_task = _episode_task(ep_path, ds, dataset_index)
            sampled = _uniform_window(rows, CONTEXT_MIN, CONTEXT_MAX)
            if sampled is None:
                continue
            subseries, start_idx, sampler_tag = sampled
            subseries = normalize_timestamps(subseries, _first_timestamp_ms(subseries))

            if important_features:
                keep = set(important_features) | {"timestamp_ms"}
                context_rows = [{k: v for k, v in row.items() if k in keep} for row in subseries]
            else:
                context_rows = subseries
            context = build_context(context_rows)

            # Template 1: sample up to 3 distinct phases per (template, episode)
            # combo so each anomalous subseries yields multiple phase-isolation
            # questions. Each call to fill_template anchors on a different phase;
            # the rest of the sampling (subseries window, relevance) is shared.
            if tid == 1:
                phase_candidates = pick_phase_isolation_candidates(subseries, n=3)
                if not phase_candidates:
                    continue
                wrote_any = False
                for _phase in phase_candidates:
                    if generated >= n:
                        break
                    filled = fill_template(
                        template, subseries, root_causes, anomaly_lookup, mc_option_lookup,
                        machine_id=DATASET_MACHINE_ID.get(ds, -1),
                        phase_override=_phase)
                    if filled is None:
                        continue
                    item = {
                        "id": str(uuid.uuid4()),
                        "level": 1,
                        "template_id": tid,
                        "template_type": template["type"],
                        "hides": template.get("hides", []),
                        "question": filled["question"],
                        "options": filled["options"],
                        "answer": filled["answer"],
                        "acceptance_bounds": filled.get("acceptance_bounds"),
                        "provenance": {
                            "dataset": ds,
                            "episode": ep_path.stem,
                            "task": ep_task,
                            "subseries_start_index": start_idx,
                            "subseries_length": len(subseries),
                            "phase_name": _phase[0],
                            "phase_start_in_subseries": _phase[1],
                            "phase_length": _phase[2],
                            "sampler": sampler_tag,
                        },
                        "context": context,
                    }
                    out_path = output_dir / f"level1_{generated:04d}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(item, f, indent=2)
                    logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template 1, {ds}, phase={_phase[0]})")
                    generated += 1
                    if uploader is not None:
                        uploader.maybe_flush(generated)
                    wrote_any = True
                if not wrote_any:
                    continue
                continue  # tid==1 has its own write path

            filled = fill_template(
                template, subseries, root_causes, anomaly_lookup, mc_option_lookup,
                machine_id=DATASET_MACHINE_ID.get(ds, -1))
            if filled is None:
                continue

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset": ds,
                    "episode": ep_path.stem,
                    "task": ep_task,
                    "subseries_start_index": start_idx,
                    "subseries_length": len(subseries),
                    "sampler": sampler_tag,
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Template 7: prediction, sample subseries + future steps for answer
        # ------------------------------------------------------------------
        elif tid == 7:
            if _enum_ep is not None:
                ds = _enum_ds
                ep_path = _enum_ep
            else:
                ds = random.choice(available_datasets)
                ep_path = random.choice(episodes_by_dataset[ds])
            rows = load_episode(ep_path)
            if not isinstance(rows, list) or len(rows) < CONTEXT_MIN + 10:
                continue

            # Anchored sampling on the "prefix" (all rows except the last 10 reserved
            # for future values); this ensures the context still ends with room for
            # lookahead.
            prefix_rows = rows[:len(rows) - 10]
            ep_fault_id = pick_fault_label(rows)
            ep_task = _episode_task(ep_path, ds, dataset_index)
            sampled = _uniform_window(prefix_rows, CONTEXT_MIN, CONTEXT_MAX)
            if sampled is None:
                continue
            raw_sub, start_idx, sampler_tag = sampled
            context_len = len(raw_sub)
            subseries = normalize_timestamps(raw_sub, _first_timestamp_ms(raw_sub))

            filled = fill_template(
                template, subseries, root_causes, anomaly_lookup, mc_option_lookup,
                machine_id=DATASET_MACHINE_ID.get(ds, -1))
            if filled is None:
                continue

            # Get the actual future value for the answer
            bounds = filled.get("acceptance_bounds", {})
            signal = bounds.get("signal")
            steps_ahead = bounds.get("steps_ahead", 1)
            future_idx = start_idx + context_len + steps_ahead - 1
            if future_idx >= len(rows) or signal is None:
                continue
            future_val = rows[future_idx].get(signal)
            if future_val is None or not isinstance(future_val, (int, float)):
                continue
            filled["answer"] = round(float(future_val), 4)
            filled["acceptance_bounds"]["actual_value"] = filled["answer"]

            # Rewrite the forecast horizon in real milliseconds using the source
            # timestamps. The template originally rendered `T+{steps_ahead}ms`,
            # which is wrong because `steps_ahead` counts rows (~100ms apart at
            # the 10Hz resampled rate) rather than milliseconds.
            last_ctx_ts = rows[start_idx + context_len - 1].get("timestamp_ms")
            future_ts = rows[future_idx].get("timestamp_ms")
            if isinstance(last_ctx_ts, (int, float)) and isinstance(future_ts, (int, float)):
                n_ms = int(round(float(future_ts) - float(last_ctx_ts)))
                filled["question"] = re.sub(
                    r"T\+\d+ms",
                    f"T+{n_ms}ms",
                    filled.get("question", ""),
                    count=1)
                filled["acceptance_bounds"]["horizon_ms"] = n_ms

            if important_features:
                keep = set(important_features) | {"timestamp_ms"}
                context_rows = [{k: v for k, v in row.items() if k in keep} for row in subseries]
            else:
                context_rows = subseries
            context = build_context(context_rows)

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset": ds,
                    "episode": ep_path.stem,
                    "task": ep_task,
                    "subseries_start_index": start_idx,
                    "subseries_length": context_len,
                    "prediction_index": future_idx,
                    "sampler": sampler_tag,
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Template 3: two episodes (prefer different datasets)
        # ------------------------------------------------------------------
        elif tid == 3:
            # Balanced pairing: aim each item at one cell of
            # (different robot) x (different anomaly) x (different task) so no
            # proposition is guessable from its prior. See utils.pair_balance.
            _target = _balance_targets[generated % len(_balance_targets)]
            # Both episodes come from the sampler: choosing the primary first
            # and then hunting for a partner is what capped P(different robots)
            # at 0.30, since the minority robot is 15% of the pool.
            _cand_ds = list(comparison_episodes_by_dataset.keys()) or available_datasets
            if not _pair_index:
                _pair_pool.extend(
                    _pair_attrs(d, p)
                    for d in _cand_ds
                    for p in comparison_episodes_by_dataset.get(d, [])
                )
                _pair_index.update(_build_pair_index(_pair_pool))
            _picked = _sample_balanced_pair(
                _pair_index, _target,
                used=_used_pairs,
                phases_differ=lambda x, y: _phases_differ(x["path"], y["path"]),
                group_pairs_cache=_group_pairs_cache)
            if _picked is None:
                continue
            _a, _b = _picked
            ds_a, ep_a = _a["dataset"], _a["path"]
            ds_b, ep_b = _b["dataset"], _b["path"]
            rows_a = load_episode(ep_a)
            rows_b_raw = load_episode(ep_b)

            if not (isinstance(rows_a, list) and isinstance(rows_b_raw, list)):
                continue
            if len(rows_a) < CONTEXT_MIN or len(rows_b_raw) < CONTEXT_MIN:
                continue

            fault_a = pick_fault_label(rows_a)
            fault_b = pick_fault_label(rows_b_raw)
            task_a = _episode_task(ep_a, ds_a, dataset_index)
            task_b = _episode_task(ep_b, ds_b, dataset_index)

            # Proposition D is evaluated on the displayed windows, not on the
            # whole episodes, so a same-task pair whose episodes differ in
            # modal phase can still yield two windows sitting in the same
            # phase. Resample the windows until they disagree, otherwise the
            # cell that was asked for silently becomes a different one.
            _want_diff_phase = (tid == 3) and not _target[2]
            sampled_a = sampled_b = None
            for _win_try in range(24):
                sampled_a = _uniform_window(rows_a, CONTEXT_MIN, CONTEXT_MAX)
                sampled_b = _uniform_window(rows_b_raw, CONTEXT_MIN, CONTEXT_MAX)
                if sampled_a is None or sampled_b is None:
                    break
                if not _want_diff_phase:
                    break
                _pa = _modal_phase(sampled_a[0])
                _pb = _modal_phase(sampled_b[0])
                if _pa is not None and _pb is not None and _pa != _pb:
                    break
            else:
                continue
            if sampled_a is None or sampled_b is None:
                continue

            sub_a, start_a, sampler_a = sampled_a
            sub_b, start_b, sampler_b = sampled_b
            sub_a = normalize_timestamps(sub_a, _first_timestamp_ms(sub_a))
            sub_b = normalize_timestamps(sub_b, _first_timestamp_ms(sub_b))

            filled = fill_template(
                template, sub_a, root_causes, anomaly_lookup, mc_option_lookup,
                rows_b=sub_b,
                machine_id=_machine_id_for(ds_a, ep_a),
                machine_id_b=_machine_id_for(ds_b, ep_b),
                task_id=task_a,
                task_id_b=task_b)
            if filled is None:
                continue

            if important_features:
                keep = set(important_features) | {"timestamp_ms"}
                sub_a_ctx = [{k: v for k, v in row.items() if k in keep} for row in sub_a]
                sub_b_ctx = [{k: v for k, v in row.items() if k in keep} for row in sub_b]
            else:
                sub_a_ctx, sub_b_ctx = sub_a, sub_b

            context = {
                "series_a": build_context(sub_a_ctx),
                "series_b": build_context(sub_b_ctx),
            }

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "dataset_a": ds_a,
                    "machine_id_a": _machine_id_for(ds_a, ep_a),
                    "episode_a": ep_a.stem,
                    "subseries_start_a": start_a,
                    "dataset_b": ds_b,
                    "machine_id_b": _machine_id_for(ds_b, ep_b),
                    "episode_b": ep_b.stem,
                    "subseries_start_b": start_b,
                    "sampler_a": sampler_a,
                    "sampler_b": sampler_b,
                },
                "context": context,
            }

        # ------------------------------------------------------------------
        # Template 5: four episodes for severity ranking
        # ------------------------------------------------------------------
        elif tid == 5:
            if len(all_episode_paths) < 4:
                continue
            sampled_eps = random.sample(all_episode_paths, 4)

            segments: List[Tuple[List[Dict[str, Any]], int]] = []
            severity_tasks: List[str] = []
            valid = True
            for ds, ep_path in sampled_eps:
                ep_rows = load_episode(ep_path)
                if not isinstance(ep_rows, list) or len(ep_rows) < 5:
                    valid = False
                    break
                fl = pick_fault_label(ep_rows)
                segments.append((ep_rows, fl))
                severity_tasks.append(_episode_task(ep_path, ds, dataset_index))
            if not valid or len(segments) < 4:
                continue

            filled = fill_template(
                template, segments[0][0], root_causes, anomaly_lookup, mc_option_lookup,
                severity_segments=segments,
                severity_relevance_specs=relevance_specs,
                severity_tasks=severity_tasks)
            if filled is None:
                continue

            item = {
                "id": str(uuid.uuid4()),
                "level": 1,
                "template_id": tid,
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": filled["question"],
                "options": filled["options"],
                "answer": filled["answer"],
                "acceptance_bounds": filled.get("acceptance_bounds"),
                "provenance": {
                    "episodes": [
                        {"dataset": ds, "episode": ep.stem}
                        for ds, ep in sampled_eps
                    ],
                },
                "context": {},
            }

        else:
            continue

        out_path = output_dir / f"level1_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        prov = item.get("provenance", {})
        if "dataset" in prov:
            ds_label = prov["dataset"]
        elif "dataset_a" in prov and "dataset_b" in prov:
            ds_label = f"{prov['dataset_a']}+{prov['dataset_b']}"
        elif "episodes" in prov and isinstance(prov["episodes"], list):
            ds_label = "+".join(sorted({str(e.get("dataset", "?")) for e in prov["episodes"]}))
        else:
            ds_label = "?"
        logger.info(f"✓ [{generated + 1}/{n}] {out_path.name} (template {tid}, {ds_label})")
        generated += 1

        if uploader is not None:
            uploader.maybe_flush(generated)

    if uploader is not None:
        uploader.flush_remaining()

    if generated < n:
        logger.warning(f"Only generated {generated}/{n} questions after {attempts} attempts.")
    else:
        logger.info(f"Done: {generated} questions written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Level 1 (State Understanding) Q&A pairs."
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
        default=repo_root / "output" / "questions" / "level1",
        help="Output directory (default: <repo>/output/questions/level1)")
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate (cap; in --enumerate mode this is an upper bound, not a target)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to sample from (default: all). Choices: {VALID_DATASETS}")
    parser.add_argument(
        "--enumerate",
        dest="enumerate_mode",
        action="store_true",
        help="Walk every (template x episode) combination deterministically instead "
             "of random sampling. -n becomes an upper cap. Combinations whose "
             "episode does not satisfy the template's preconditions are skipped.")
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

    global _ANOMALY_RANKING
    _ANOMALY_RANKING = load_anomaly_ranking(
        args.datasets_dir / "labelling" / "rca" / "anomaly_ranking.json"
    )

    anomaly_lookup = load_anomaly_lookup(
        args.datasets_dir / "labelling" / "rca" / "anomalies.json"
    )
    mc_option_lookup = load_mc_option_lookup(
        args.datasets_dir / "mc_options" / "mc_options.json",
        level=1)
    dataset_index = load_dataset_index(args.datasets_dir / "labelling" / "dataset.json")

    relevance_specs = (
        load_relevance_specs(args.datasets_dir / "labelling" / "rca" / "relevance_specs.json")
        if relevance_enabled()
        else {}
    )
    if relevance_enabled():
        logger.info(f"Relevance-aware sampling enabled ({len(relevance_specs)} fault specs loaded)")
    else:
        logger.info("Relevance-aware sampling disabled (FB_RELEVANCE=0)")

    args.output.mkdir(parents=True, exist_ok=True)
    uploader = make_uploader_from_args(args, level=1, output_dir=args.output)

    generate_level1_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        anomaly_lookup=anomaly_lookup,
        mc_option_lookup=mc_option_lookup,
        dataset_index=dataset_index,
        relevance_specs=relevance_specs,
        n=args.n,
        seed=args.seed,
        datasets=args.datasets,
        enumerate_mode=args.enumerate_mode,
        uploader=uploader)


if __name__ == "__main__":
    main()
