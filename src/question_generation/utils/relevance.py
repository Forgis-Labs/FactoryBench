"""Fault-aware sub-series sampling.

Given an episode and its fault_id, picks a window where the fault's signature
is observable, per a declarative spec in
data/labelling/rca/relevance_specs.json.

Four localities:
  - global, uniform sampling (nominal or fault present throughout).
  - event, enforce min window length so the transient is likely captured.
  - phase_gated, window must overlap one of the target task_phase runs.
  - cumulative, enforce min window length, optional start-phase gate.

Kill switch: set FB_RELEVANCE=0 to restore uniform sampling everywhere.
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

Spec = Dict[str, Any]
Row = Dict[str, Any]
Window = Tuple[List[Row], int]
SampleResult = Tuple[List[Row], int, str]  # (subseries, start_index, sampler_tag)


def is_enabled() -> bool:
    return os.environ.get("FB_RELEVANCE", "1") != "0"


def load_specs(path: Path) -> Dict[int, Spec]:
    """Load relevance_specs.json → {fault_id: spec}. Merges `defaults` into specs."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning(f"Relevance specs not found at {path}; relevance-aware sampling disabled for all faults")
        return {}
    except Exception as exc:
        logger.warning(f"Could not load relevance specs ({exc}); using empty table")
        return {}

    defaults = raw.get("defaults") or {}
    out: Dict[int, Spec] = {}
    for key, value in (raw.get("specs") or {}).items():
        try:
            fid = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        merged = dict(defaults.get(value.get("locality", "global"), {}))
        merged.update(value)
        out[fid] = merged
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phase_of(row: Row) -> Optional[str]:
    p = row.get("task_phase")
    if p is None or p == "None" or p == "":
        return None
    return str(p)


def _phase_ids_for_task(spec: Spec, task: Optional[str], key: str) -> List[int]:
    pbt = spec.get(key) or {}
    if task and task in pbt:
        return list(pbt[task])
    if pbt and len(pbt) == 1:
        return list(next(iter(pbt.values())))
    return []


def _matching_indices(rows: List[Row], phase_ids: List[int]) -> List[int]:
    target = {str(int(p)) for p in phase_ids}
    return [i for i, r in enumerate(rows) if _phase_of(r) in target]


def _first_index_with_phase(rows: List[Row], phase_ids: List[int]) -> Optional[int]:
    target = {str(int(p)) for p in phase_ids}
    for i, r in enumerate(rows):
        if _phase_of(r) in target:
            return i
    return None


def _contiguous_runs(indices: List[int]) -> List[Tuple[int, int]]:
    """Group a sorted index list into (start, end_exclusive) runs."""
    if not indices:
        return []
    runs: List[Tuple[int, int]] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev + 1))
            start = prev = i
    runs.append((start, prev + 1))
    return runs


def _uniform(rows: List[Row], min_len: int, max_len: int) -> Optional[Window]:
    n = len(rows)
    if n < min_len:
        return None
    length = random.randint(min_len, min(max_len, n))
    start = random.randint(0, n - length)
    return rows[start:start + length], start


# ---------------------------------------------------------------------------
# Anchored samplers per locality
# ---------------------------------------------------------------------------


def _sample_event(rows: List[Row], spec: Spec, min_len: int, max_len: int) -> Optional[Window]:
    """Sample a window that ACTUALLY contains an event onset.

    Anchors on a randomly chosen row whose ``event`` token resolves to a
    non-zero, non-1 id (a real event, not the implicit "no event" / "task
    started" placeholder). Window length and exact start are still
    randomized, the constraint is only that the event row is inside the
    window.
    """
    from src.question_generation.utils.time_series import parse_event_id

    min_required = max(int(spec.get("min_window_length", min_len)), min_len)
    # When the caller requests a shorter window than the spec prefers (e.g. 5-7 row
    # severity-ranking chunks), honor the caller's upper bound rather than refusing.
    min_required = min(min_required, max_len)
    n = len(rows)
    if n < min_required:
        return None

    # Find rows that carry a real event token (not 0 = no event, not 1 = task start).
    event_indices: List[int] = []
    for i, r in enumerate(rows):
        eid = parse_event_id(r.get("event", 0))
        if eid not in (0, 1):
            event_indices.append(i)
    if not event_indices:
        return None

    anchor = random.choice(event_indices)
    length_hi = min(max_len, n)

    # Try a random length first; if no valid (start, length) exists for it,
    # fall back to the minimum length (which has the widest start range).
    for attempt_length in (random.randint(min_required, length_hi), min_required):
        # Window must contain the anchor: start <= anchor < start + length.
        min_start = max(0, anchor - attempt_length + 1)
        max_start = min(n - attempt_length, anchor)
        if min_start <= max_start:
            start = random.randint(min_start, max_start)
            return rows[start:start + attempt_length], start
    return None


def _sample_cumulative(
    rows: List[Row], spec: Spec, task: Optional[str], min_len: int, max_len: int) -> Optional[Window]:
    n = len(rows)
    min_required = max(int(spec.get("min_window_length", min_len)), min_len)
    min_required = min(min_required, max_len)
    if n < min_required:
        return None

    start_phase_ids = _phase_ids_for_task(spec, task, "start_phase_ids_by_task")
    start_floor = 0
    if start_phase_ids:
        idx = _first_index_with_phase(rows, start_phase_ids)
        if idx is not None:
            start_floor = idx
        # if the start phase isn't present, fall through with floor=0

    remaining = n - start_floor
    if remaining < min_required:
        # Cannot honor gate; fall back to unrestricted start (still meets min)
        start_floor = 0
        remaining = n

    length = random.randint(min_required, min(max_len, remaining))
    start = random.randint(start_floor, n - length)
    return rows[start:start + length], start


def _sample_phase_gated(
    rows: List[Row], spec: Spec, task: Optional[str], min_len: int, max_len: int) -> Optional[Window]:
    phase_ids = _phase_ids_for_task(spec, task, "phases_by_task")
    if not phase_ids:
        return None
    matches = _matching_indices(rows, phase_ids)
    if not matches:
        return None
    min_overlap = int(spec.get("min_overlap", 5))
    runs = _contiguous_runs(matches)
    random.shuffle(runs)

    n = len(rows)
    for run_start, run_end in runs:
        run_len = run_end - run_start
        desired_overlap = min(run_len, min_overlap)

        # Window length candidates
        length_lo = min_len
        length_hi = min(max_len, n)
        if length_lo > length_hi:
            continue
        length = random.randint(length_lo, length_hi)

        # Window start must satisfy:
        #   start <= run_end - desired_overlap   (window reaches into run)
        #   start + length >= run_start + desired_overlap  (enough overlap)
        #   0 <= start <= n - length
        min_start = max(0, run_start + desired_overlap - length)
        max_start = min(n - length, run_end - desired_overlap)
        if min_start > max_start:
            continue
        start = random.randint(min_start, max_start)
        return rows[start:start + length], start
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_with_relevance(
    rows: List[Row],
    fault_id: int,
    spec: Optional[Spec],
    task: Optional[str],
    min_len: int,
    max_len: int,
    fallback_to_uniform: bool = False) -> Optional[SampleResult]:
    """Sample a window honoring the fault's relevance spec.

    Returns (subseries, start_index, sampler_tag) or None if no valid window fits.
    When the relevance system is disabled or spec is missing/global, samples uniformly.

    When ``fallback_to_uniform`` is True and a strict locality (event /
    phase_gated / cumulative) cannot be honored, the sampler returns a
    uniform window tagged ``"uniform_fallback"`` instead of None. This is
    useful for templates where displaying the anomaly's signature is
    nice-to-have (e.g. phase isolation doesn't need anomaly evidence) but
    not required.
    """
    if not is_enabled() or not spec:
        win = _uniform(rows, min_len, max_len)
        return (win[0], win[1], "uniform") if win else None

    locality = spec.get("locality", "global")

    if locality == "global":
        win = _uniform(rows, min_len, max_len)
        return (win[0], win[1], "uniform") if win else None

    if locality == "event":
        win = _sample_event(rows, spec, min_len, max_len)
        if win is None and fallback_to_uniform:
            win = _uniform(rows, min_len, max_len)
            return (win[0], win[1], "uniform_fallback") if win else None
        return (win[0], win[1], "event") if win else None

    if locality == "cumulative":
        win = _sample_cumulative(rows, spec, task, min_len, max_len)
        if win is None and fallback_to_uniform:
            win = _uniform(rows, min_len, max_len)
            return (win[0], win[1], "uniform_fallback") if win else None
        return (win[0], win[1], "cumulative") if win else None

    if locality == "phase_gated":
        win = _sample_phase_gated(rows, spec, task, min_len, max_len)
        if win is None and fallback_to_uniform:
            win = _uniform(rows, min_len, max_len)
            return (win[0], win[1], "uniform_fallback") if win else None
        return (win[0], win[1], "phase_gated") if win else None

    logger.warning(f"Unknown locality '{locality}' for fault {fault_id}; using uniform")
    win = _uniform(rows, min_len, max_len)
    return (win[0], win[1], "uniform") if win else None


def validate_relevance(
    sub_rows: List[Row],
    spec: Optional[Spec],
    task: Optional[str]) -> bool:
    """Post-hoc check that a subseries satisfies the spec. Cheap safety net."""
    if not spec:
        return True
    locality = spec.get("locality", "global")
    if locality == "global":
        return True
    if locality in ("event", "cumulative"):
        min_required = int(spec.get("min_window_length", 0))
        return len(sub_rows) >= min_required
    if locality == "phase_gated":
        phase_ids = _phase_ids_for_task(spec, task, "phases_by_task")
        if not phase_ids:
            return False
        min_overlap = int(spec.get("min_overlap", 5))
        target = {str(int(p)) for p in phase_ids}
        overlap = sum(1 for r in sub_rows if _phase_of(r) in target)
        return overlap >= min_overlap
    return True


def relevance_report(
    sub_rows: List[Row],
    fault_id: int,
    spec: Optional[Spec],
    task: Optional[str],
    sampler: str) -> Dict[str, Any]:
    """Provenance-friendly summary of relevance state for a generated item."""
    report: Dict[str, Any] = {"fault_id": fault_id, "sampler": sampler}
    if not spec:
        report.update({"locality": None, "validated": True})
        return report
    locality = spec.get("locality", "global")
    report["locality"] = locality
    if locality == "phase_gated":
        phase_ids = _phase_ids_for_task(spec, task, "phases_by_task")
        if phase_ids:
            target = {str(int(p)) for p in phase_ids}
            report["phase_overlap"] = sum(1 for r in sub_rows if _phase_of(r) in target)
            report["target_phases"] = phase_ids
    report["validated"] = validate_relevance(sub_rows, spec, task)
    return report
