"""
Shared template-filling utilities for question generators across all levels.

Covers: safe string substitution, event description filling, signal picking,
chunk encoding/sampling, and context building.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.question_generation.utils.time_series import (
    encode_time_series,
    remove_feature,
    sort_feature_keys,
    strip_null_features,
)


# ---------------------------------------------------------------------------
# Safe string substitution
# ---------------------------------------------------------------------------


def fill(template_str: str, **kwargs: Any) -> str:
    """
    Safe template substitution via str.replace().
    Avoids str.format() pitfalls with literal braces (e.g. {T}, {T+n}).
    Automatically provides {Event} (sentence-start capitalised version) whenever
    the 'event' kwarg is supplied, so templates can use {Event} when the
    event description opens the sentence.
    """
    result = template_str
    for key, value in kwargs.items():
        result = result.replace(f"{{{key}}}", str(value))
    if "event" in kwargs:
        s = str(kwargs["event"])
        result = result.replace("{Event}", s[:1].upper() + s[1:] if s else "")
    return result


# ---------------------------------------------------------------------------
# Timestamp and signal helpers
# ---------------------------------------------------------------------------


def get_last_timestamp(rows: List[Dict[str, Any]]) -> int:
    """Return the last timestamp_ms value in the rows as an integer."""
    for row in reversed(rows):
        ts = row.get("timestamp_ms")
        if ts is not None:
            try:
                return int(float(ts))
            except (TypeError, ValueError):
                pass
    return 0


def get_numeric_signal_names(rows: List[Dict[str, Any]]) -> List[str]:
    """Return signal names that hold numeric values, excluding metadata columns."""
    exclude = {"timestamp_ms", "fault_label"}
    if not rows:
        return []
    return [
        key
        for key in sorted(rows[0].keys())
        if key not in exclude and isinstance(rows[0].get(key), (int, float, np.floating))
    ]


def pick_scalar_signal(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Pick a random numeric signal name from the first row."""
    names = get_numeric_signal_names(rows)
    return random.choice(names) if names else None


def pick_constrained_signal(
    rows: List[Dict[str, Any]],
    candidates: List[str],
) -> Optional[str]:
    """
    Pick a signal from candidates that is actually present and numeric in rows.
    Returns None if no candidate is available.
    """
    if not rows:
        return None
    available = {
        key
        for key in rows[0].keys()
        if isinstance(rows[0].get(key), (int, float, np.floating))
    }
    valid = [c for c in candidates if c in available]
    return random.choice(valid) if valid else None


def pick_joint_velocity_and_torque(
    rows: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    """Return a representative velocity and torque value from the last row."""
    if not rows:
        return None, None
    last_row = rows[-1]
    velocity = None
    for key in sorted(last_row.keys()):
        if "velocity" in key or "speed" in key:
            val = last_row.get(key)
            if isinstance(val, (int, float, np.floating)):
                velocity = round(float(val), 3)
                break
    torque = None
    for key in sorted(last_row.keys()):
        if "torque" in key or "current" in key:
            val = last_row.get(key)
            if isinstance(val, (int, float, np.floating)):
                torque = round(float(val), 3)
                break
    return velocity, torque


# ---------------------------------------------------------------------------
# Chunk sampling and encoding
# ---------------------------------------------------------------------------


def encode_chunk(rows: List[Dict[str, Any]]) -> str:
    """Encode a list of rows as a single compact string for display in options."""
    stripped = strip_null_features(rows)
    stripped = remove_feature(stripped, "fault_label")
    stripped = sort_feature_keys(stripped)
    if not stripped:
        return "[]"
    encoded, _ = encode_time_series(stripped)
    return " | ".join(encoded)


def sample_chunks(
    rows: List[Dict[str, Any]],
    n_chunks: int = 4,
    min_chunk: int = 5,
    max_chunk: int = 7,
) -> List[List[Dict[str, Any]]]:
    """
    Sample n_chunks non-overlapping contiguous windows of random length
    [min_chunk, max_chunk] from rows.  Returns fewer than n_chunks if the
    rows are not long enough.
    """
    available = len(rows)
    if available < n_chunks * min_chunk:
        return []
    occupied: set = set()
    chunks: List[List[Dict[str, Any]]] = []
    max_attempts = n_chunks * 50
    attempts = 0
    while len(chunks) < n_chunks and attempts < max_attempts:
        attempts += 1
        chunk_len = random.randint(min_chunk, min(max_chunk, available))
        if available < chunk_len:
            continue
        start = random.randint(0, available - chunk_len)
        span = set(range(start, start + chunk_len))
        if not span & occupied:
            chunks.append(rows[start : start + chunk_len])
            occupied |= span
    return chunks


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def build_context(subseries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the context dict attached to every generated question.
    Contains the encoded time series with all features inlined per row.
    """
    ts = strip_null_features(subseries)
    ts = sort_feature_keys(ts)
    ts = remove_feature(ts, "fault_label")
    encoded, acronym_mapping = encode_time_series(ts)

    ctx: Dict[str, Any] = {}
    ctx["time_series_format"] = {
        "description": (
            "Each row in time_series is one timestep encoded as "
            "'t=<timestamp>: acronym=value, ...'. "
            "Feature names use acronyms defined in provenance.feature_mapping."
        ),
        "acronym_mapping": acronym_mapping,
    }
    ctx["time_series"] = encoded
    return ctx


# ---------------------------------------------------------------------------
# Event description filling
# ---------------------------------------------------------------------------


def fill_event_description(
    event: Dict[str, Any],
    subseries: List[Dict[str, Any]],
    t: int,
    post_event_rows: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Fill an event's description template with values derived from the subseries.

    Supported variable types (from events.json):
      feature_i  → a random numeric signal name
      X          → signal value at the start of the subseries
      Y          → signal value at the end of the subseries
      delta      → |end - start|
      duration   → random integer 2–10
      rate       → (end - start) / len(subseries)
      x          → random payload weight
      T          → the last timestamp of the subseries
      L          → number of consecutive timesteps the event persists in post_event_rows

    If post_event_rows carries encoded event labels in the form
    "id_v1_v2_...", values are parsed and used directly (in variable order)
    for consistency between encoded events and generated question text.
    """
    desc: str = event["description"]
    variables: Dict[str, str] = event.get("variables", {})

    def _to_int_if_possible(value: str) -> Any:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    def _to_float_if_possible(value: str) -> Any:
        try:
            x = float(value)
            if x.is_integer():
                return int(x)
            return x
        except (TypeError, ValueError):
            return value

    def _parse_event_variable_values() -> Optional[Dict[str, Any]]:
        if not post_event_rows:
            return None
        raw_event = post_event_rows[0].get("event", 0)
        if raw_event in (None, 0, "0"):
            return None

        text = str(raw_event)
        underscore_idx = text.find("_")
        if underscore_idx < 0:
            return None

        params_str = text[underscore_idx + 1:]
        if not params_str:
            return None

        var_items = list(variables.items())
        if not var_items:
            return {}

        # Key=value format: "motor=motor_2;phase_offset_deg=9.87"
        if "=" in params_str:
            parsed_kv: Dict[str, Any] = {}
            for pair in params_str.split(";"):
                pair = pair.strip()
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                parsed_kv[k.strip()] = _to_float_if_possible(v.strip())

            result: Dict[str, Any] = {}
            for var_name, var_type in var_items:
                if var_name in parsed_kv:
                    val = parsed_kv[var_name]
                    result[var_name] = _to_int_if_possible(str(val)) if var_type == "integer" else val
            return result if result else None

        # Positional format (legacy): underscore-separated values after event_id
        parts = text.split("_")
        tokens = parts[1:]
        parsed: Dict[str, Any] = {}

        # Parse from right to left for numeric/integer vars, then assign the
        # remaining left-side tokens to string vars (e.g., feature_i).
        right = len(tokens)
        for var_name, var_type in reversed(var_items):
            if var_type in {"numeric", "integer"}:
                if right <= 0:
                    return None
                tok = tokens[right - 1]
                right -= 1
                if var_type == "integer":
                    parsed[var_name] = _to_int_if_possible(tok)
                else:
                    parsed[var_name] = _to_float_if_possible(tok)

        string_vars = [name for name, typ in var_items if typ == "string"]
        if string_vars:
            # Current event schema effectively has a single string variable
            # (feature_i). Join all remaining tokens to preserve underscores.
            for i, var_name in enumerate(string_vars):
                if i == 0:
                    parsed[var_name] = "_".join(tokens[:right])
                else:
                    parsed[var_name] = ""

        # Rebuild dict in template variable order for deterministic behavior.
        ordered: Dict[str, Any] = {}
        for var_name in variables.keys():
            if var_name in parsed:
                ordered[var_name] = parsed[var_name]
        return ordered

    encoded_values = _parse_event_variable_values()

    if encoded_values is not None:
        kwargs: Dict[str, Any] = {}
        for var_name in variables.keys():
            if var_name in encoded_values:
                kwargs[var_name] = encoded_values[var_name]

        # Keep T/L available even if not encoded explicitly in event labels
        # or missing in events.json variable schema.
        if ("T" in variables or "{T}" in desc) and "T" not in kwargs:
            kwargs["T"] = t
        if ("L" in variables or "{L}" in desc) and "L" not in kwargs:
            L = 0
            onset_val = post_event_rows[0].get("event", 0)
            for r in post_event_rows:
                if r.get("event", 0) == onset_val:
                    L += 1
                else:
                    break
            kwargs["L"] = L
        return fill(desc, **kwargs)

    feature_candidates: Optional[List[str]] = event.get("variable_constraints", {}).get("feature_i")
    if feature_candidates:
        signal = (
            pick_constrained_signal(subseries, feature_candidates)
            or pick_scalar_signal(subseries)
            or "joint_velocity_0"
        )
    else:
        signal = pick_scalar_signal(subseries) or "joint_velocity_0"

    start_val: Optional[float] = None
    end_val: Optional[float] = None
    for row in subseries:
        v = row.get(signal)
        if isinstance(v, (int, float, np.floating)):
            start_val = round(float(v), 2)
            break
    for row in reversed(subseries):
        v = row.get(signal)
        if isinstance(v, (int, float, np.floating)):
            end_val = round(float(v), 2)
            break

    kwargs: Dict[str, Any] = {}

    if "T" in variables or "{T}" in desc:
        kwargs["T"] = t

    if "feature_i" in variables:
        kwargs["feature_i"] = signal
    if "X" in variables:
        kwargs["X"] = start_val if start_val is not None else 0.0
    if "Y" in variables:
        kwargs["Y"] = end_val if end_val is not None else 0.0
    if "delta" in variables:
        kwargs["delta"] = round(abs((end_val or 0.0) - (start_val or 0.0)), 2)
    if "duration" in variables:
        kwargs["duration"] = random.randint(2, 10)
    if "rate" in variables:
        n = max(1, len(subseries))
        kwargs["rate"] = round(((end_val or 0.0) - (start_val or 0.0)) / n, 2)
    if "x" in variables:
        kwargs["x"] = random.choice([0.5, 1.0, 1.5, 2.0, 2.5])
    if "L" in variables or "{L}" in desc:
        L = 0
        if post_event_rows:
            onset_val = post_event_rows[0].get("event", 0)
            for r in post_event_rows:
                if r.get("event", 0) == onset_val:
                    L += 1
                else:
                    break
        kwargs["L"] = L

    return fill(desc, **kwargs)


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------


def discover_episodes_by_dataset(
    datasets_dir: Path,
    datasets: List[str],
) -> Dict[str, List[Path]]:
    """
    Return {dataset_name: [episode_path, ...]} for each dataset that has
    normalized episode JSON files under datasets_dir/normalized_episodes/<name>/.
    """
    by_dataset: Dict[str, List[Path]] = {}
    for ds in datasets:
        ep_dir = datasets_dir / "normalized_episodes" / ds
        if not ep_dir.exists():
            continue
        paths = [
            p
            for p in sorted(ep_dir.glob("*.json"))
            if not p.stem.endswith("_metadata")
        ]
        if paths:
            by_dataset[ds] = paths
    return by_dataset
