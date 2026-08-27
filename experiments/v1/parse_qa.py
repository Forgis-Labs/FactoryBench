"""Parse FactoryBench Q&A items for Chronos-2 forecasting.

For each of the 3 applicable templates (L1.7, L2.4, L2.5), extracts:
  - context: numpy array of shape (T, C) with the time-series context
  - target_channels: list of column names to forecast (1 for scalar, 6 for tensor)
  - horizon_steps: how many steps into the future to forecast
  - ground_truth: scalar or list[float] ground-truth answer
  - acceptance_bounds: margin(s) for scoring
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ChronosTask:
    """Everything Chronos-2 needs to answer one Q&A item."""

    qa_id: str
    level: int
    template_id: int
    answer_format: str  # "numerical" or "tensor"

    # Time-series context: dict of {channel_name: np.array}
    channels: Dict[str, np.ndarray] = field(repr=False)
    timestamps_ms: np.ndarray = field(repr=False)

    # What to forecast
    target_channels: List[str] = field(default_factory=list)
    horizon_steps: int = 0

    # Ground truth & scoring
    ground_truth: Any = None  # float or list[float]
    acceptance_bounds: Optional[Dict[str, Any]] = None

    # Original question text (for display)
    question: str = ""


# ---------------------------------------------------------------------------
# Reverse-parse encoded time-series strings
# ---------------------------------------------------------------------------

# Matches "key=value" pairs inside an encoded row
_KV_RE = re.compile(r"([a-zA-Z_]\w*)=([-\d.eE+]+)")


def parse_encoded_rows(
    encoded_rows: List[str],
    acronym_mapping: Dict[str, str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Parse text-encoded rows back into numeric arrays.

    Returns:
        timestamps_ms: (T,) array of timestamps
        channel_names: list of full feature names (sorted)
        data: (T, C) array of channel values
    """
    parsed = []
    timestamps = []
    for row_str in encoded_rows:
        kvs = dict(_KV_RE.findall(row_str))
        # Extract timestamp
        ts = float(kvs.pop("t", 0))
        timestamps.append(ts)
        # Map acronyms -> full names
        full_kvs = {}
        for acro, val_str in kvs.items():
            full_name = acronym_mapping.get(acro, acro)
            full_kvs[full_name] = float(val_str)
        parsed.append(full_kvs)

    if not parsed:
        return np.array([]), [], np.array([]).reshape(0, 0)

    # Get stable column order
    all_cols = sorted(parsed[0].keys())
    data = np.array([[row.get(c, np.nan) for c in all_cols] for row in parsed])
    return np.array(timestamps), all_cols, data


# ---------------------------------------------------------------------------
# Template-specific parsers
# ---------------------------------------------------------------------------


def _resolve_signal_column(
    display_name: str,
    acceptance_bounds: Dict[str, Any],
    channel_names: List[str],
) -> Optional[str]:
    """Resolve the actual column name from acceptance_bounds or display name."""
    # acceptance_bounds["signal"] is the raw column name (e.g., "feedback_pos_2")
    signal = acceptance_bounds.get("signal", "")
    if signal in channel_names:
        return signal
    # Fallback: try matching display name to channel names
    clean = display_name.lower().replace(" ", "_")
    for ch in channel_names:
        if clean in ch:
            return ch
    return None


def parse_l1_7(item: Dict[str, Any], channel_names: List[str]) -> Optional[dict]:
    """L1.7: 'Expected value of {signal} at T+{n}ms?'

    acceptance_bounds: {"signal": "feedback_pos_2", "steps_ahead": 5}
    Note: {n} in the question text is steps_ahead (not ms!) for L1.7.
    """
    bounds = item.get("acceptance_bounds") or {}
    signal = bounds.get("signal")
    steps_ahead = bounds.get("steps_ahead")

    if signal is None or steps_ahead is None:
        return None
    if signal not in channel_names:
        return None

    return {
        "target_channels": [signal],
        "horizon_steps": int(steps_ahead),
        "answer_format": "numerical",
    }


def parse_l2_4(item: Dict[str, Any], channel_names: List[str]) -> Optional[dict]:
    """L2.4: 'Robot exhibiting {anomaly}. Expected value of {signal} at T+{n}ms?'

    acceptance_bounds: {"signal": "feedback_pos_2", "std": 0.1, "margin": 0.075}
    {n} in question text is milliseconds. We need to compute steps from the
    timestamps in the context.
    """
    bounds = item.get("acceptance_bounds") or {}
    signal = bounds.get("signal")

    if signal is None or signal not in channel_names:
        return None

    # Extract n_ms from question text
    q = item.get("question", "")
    m = re.search(r"T\+(\d+)\s*ms", q)
    if not m:
        return None
    n_ms = int(m.group(1))

    return {
        "target_channels": [signal],
        "horizon_ms": n_ms,  # converted to steps later using timestamps
        "answer_format": "numerical",
    }


def parse_l2_5(item: Dict[str, Any], channel_names: List[str]) -> Optional[dict]:
    """L2.5: 'Expected values of {joint_signal} at T+{n}ms?' -> tensor [6]

    acceptance_bounds: {"signal": "feedback_pos", "std": [...], "margin": [...]}
    """
    bounds = item.get("acceptance_bounds") or {}
    base_signal = bounds.get("signal")  # e.g., "feedback_pos"

    if base_signal is None:
        return None

    # Build 6 channel names
    target_channels = [f"{base_signal}_{i}" for i in range(6)]
    if not all(ch in channel_names for ch in target_channels):
        return None

    # Extract n_ms from question text
    q = item.get("question", "")
    m = re.search(r"T\+(\d+)\s*ms", q)
    if not m:
        return None
    n_ms = int(m.group(1))

    return {
        "target_channels": target_channels,
        "horizon_ms": n_ms,
        "answer_format": "tensor",
    }


# ---------------------------------------------------------------------------
# Unified parser
# ---------------------------------------------------------------------------

PARSERS = {
    (1, 7): parse_l1_7,
    (2, 4): parse_l2_4,
    (2, 5): parse_l2_5,
}

APPLICABLE_TEMPLATES = {(1, 7), (2, 4), (2, 5)}


def ms_to_steps(n_ms: float, timestamps_ms: np.ndarray) -> int:
    """Convert a millisecond horizon to steps using the actual sample interval."""
    if len(timestamps_ms) < 2:
        return 1
    dt = np.median(np.diff(timestamps_ms))
    if dt <= 0:
        return 1
    return max(1, round(n_ms / dt))


def parse_qa_item(item: Dict[str, Any]) -> Optional[ChronosTask]:
    """Parse a single Q&A JSON item into a ChronosTask (or None if not applicable)."""
    level = item.get("level")
    tid = item.get("template_id")
    key = (level, tid)

    if key not in APPLICABLE_TEMPLATES:
        return None

    # Extract context
    context = item.get("context") or {}
    ts_encoded = context.get("time_series")
    ts_format = context.get("time_series_format") or {}
    acronym_mapping = ts_format.get("acronym_mapping") or {}

    if not ts_encoded or not isinstance(ts_encoded, list):
        return None

    timestamps_ms, channel_names, data = parse_encoded_rows(ts_encoded, acronym_mapping)
    if data.size == 0:
        return None

    # Run template-specific parser
    parser = PARSERS[key]
    parsed = parser(item, channel_names)
    if parsed is None:
        return None

    # Build channel dict
    channels = {ch: data[:, i] for i, ch in enumerate(channel_names)}

    # Compute horizon_steps
    if "horizon_steps" in parsed:
        horizon_steps = parsed["horizon_steps"]
    elif "horizon_ms" in parsed:
        horizon_steps = ms_to_steps(parsed["horizon_ms"], timestamps_ms)
    else:
        return None

    return ChronosTask(
        qa_id=item.get("id", "unknown"),
        level=level,
        template_id=tid,
        answer_format=parsed["answer_format"],
        channels=channels,
        timestamps_ms=timestamps_ms,
        target_channels=parsed["target_channels"],
        horizon_steps=horizon_steps,
        ground_truth=item.get("answer"),
        acceptance_bounds=item.get("acceptance_bounds"),
        question=item.get("question", ""),
    )


# ---------------------------------------------------------------------------
# Batch loader: filter a JSONL file to applicable items
# ---------------------------------------------------------------------------


def load_applicable_items(
    jsonl_path: Path,
    max_per_template: Optional[int] = None,
) -> List[ChronosTask]:
    """Load Q&A items from a JSONL file, keeping only Chronos-applicable ones."""
    counts: Dict[tuple, int] = {k: 0 for k in APPLICABLE_TEMPLATES}
    tasks: List[ChronosTask] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = (item.get("level"), item.get("template_id"))
            if key not in APPLICABLE_TEMPLATES:
                continue
            if max_per_template and counts[key] >= max_per_template:
                continue

            task = parse_qa_item(item)
            if task is not None:
                tasks.append(task)
                counts[key] += 1

            # Early exit if all templates are full
            if max_per_template and all(c >= max_per_template for c in counts.values()):
                break

    return tasks


if __name__ == "__main__":
    import sys

    # Quick test: parse a single JSONL and print stats
    if len(sys.argv) < 2:
        print("Usage: python parse_qa.py <path_to_test.jsonl> [max_per_template]")
        sys.exit(1)

    path = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    tasks = load_applicable_items(path, max_per_template=limit)

    print(f"Loaded {len(tasks)} tasks from {path.name}")
    for key in sorted(APPLICABLE_TEMPLATES):
        n = sum(1 for t in tasks if (t.level, t.template_id) == key)
        print(f"  L{key[0]}.{key[1]}: {n} items")

    if tasks:
        t = tasks[0]
        print(f"\nExample task:")
        print(f"  ID: {t.qa_id}")
        print(f"  Template: L{t.level}.{t.template_id} ({t.answer_format})")
        print(f"  Context: {len(t.timestamps_ms)} timesteps, {len(t.channels)} channels")
        print(f"  Target: {t.target_channels}")
        print(f"  Horizon: {t.horizon_steps} steps")
        print(f"  Ground truth: {t.ground_truth}")
        print(f"  Question: {t.question[:120]}...")
