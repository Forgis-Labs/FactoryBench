"""
Inject synthetic event labels into normalized episode JSON files.

Copies aursad / vorausad normalized episodes to test_aursad / test_vorausad,
adding an "event" field to every timestep row.  The field is 0 most of the
time; occasionally it becomes a random event ID (from events.json) and stays
at that value for a random number of consecutive timesteps.

Metadata files (*_metadata.json) are copied unchanged.

Usage:
    python -m src.data.data_normalization.inject_events [options]

    # with defaults (processes both aursad and vorausad):
    python -m src.data.data_normalization.inject_events --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_event_id(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        return 0
    head = s.split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return 0


def _to_number_or_text(value: Any) -> str:
    if isinstance(value, (int, float)):
        x = float(value)
        if x.is_integer():
            return str(int(x))
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _pick_feature_key(row: Dict[str, Any]) -> str:
    exclude = {"timestamp_ms", "fault_label", "event"}
    candidates = [
        k
        for k, v in row.items()
        if k not in exclude and isinstance(v, (int, float))
    ]
    if not candidates:
        return "feature_0"
    return random.choice(candidates)


def build_event_label(
    event_obj: Dict[str, Any],
    onset_row: Dict[str, Any],
    end_of_event_row: Dict[str, Any],
    run_len: int,
    prev_row: Optional[Dict[str, Any]] = None) -> str:
    event_id = int(event_obj["id"])
    variables: Dict[str, str] = event_obj.get("variables", {})
    feature_key = _pick_feature_key(onset_row)

    onset_ts = onset_row.get("timestamp_ms", 0)
    onset_val = onset_row.get(feature_key, 0)
    prev_val = prev_row.get(feature_key, onset_val) if prev_row else onset_val
    end_val = end_of_event_row.get(feature_key, onset_val)
    delta = abs(float(end_val) - float(prev_val)) if isinstance(end_val, (int, float)) and isinstance(prev_val, (int, float)) else 0.0
    rate = (float(end_val) - float(prev_val)) / max(1, run_len) if isinstance(end_val, (int, float)) and isinstance(prev_val, (int, float)) else 0.0

    payload_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    payload_idx = (event_id + run_len) % len(payload_values)
    payload = payload_values[payload_idx]

    parts: List[str] = [str(event_id)]
    for var_name in variables.keys():
        var_lower = var_name.lower()
        if var_name == "feature_i":
            val = feature_key
        elif var_name in {"X", "x"}:
            if event_obj.get("name", "").lower().startswith("payload"):
                val = payload
            else:
                val = prev_val
        elif var_name == "Y":
            # Y is taken at the final timestep of the current event segment.
            val = end_val
        elif var_name == "delta":
            val = delta
        elif var_name in {"duration", "L"}:
            val = run_len
        elif var_name == "rate":
            val = rate
        elif var_name in {"T", "t"} or "timestamp" in var_lower:
            val = onset_ts
        else:
            val = onset_ts if "time" in var_lower else 0
        parts.append(_to_number_or_text(val))

    return "_".join(parts)


# ---------------------------------------------------------------------------
# Event label generation
# ---------------------------------------------------------------------------


def make_event_sequence(
    n: int,
    event_ids: List[int],
    p_start: float,
    min_duration: int,
    max_duration: int) -> List[int]:
    """
    Generate a sequence of length n where each element is either 0 or an
    event ID.  Events are contiguous runs of a single ID; between events the
    value is always 0.

    p_start   - probability per timestep (while idle) of starting a new event
    min/max_duration - uniform range for event run length (timesteps)
    """
    seq: List[int] = []
    i = 0
    while i < n:
        remaining = n - i
        if remaining >= min_duration and random.random() < p_start:
            duration = random.randint(min_duration, min(max_duration, remaining))
            event_id = random.choice(event_ids)
            seq.extend([event_id] * duration)
            i += duration
        else:
            seq.append(0)
            i += 1
    return seq[:n]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process_episode(
    src: Path,
    dst: Path,
    event_ids: List[int],
    events_by_id: Dict[int, Dict[str, Any]],
    p_start: float,
    min_duration: int,
    max_duration: int) -> None:
    with src.open("r", encoding="utf-8") as f:
        rows: Any = json.load(f)

    if not isinstance(rows, list) or not rows:
        # unexpected format, copy verbatim
        shutil.copy2(src, dst)
        return

    seq = make_event_sequence(
        len(rows), event_ids, p_start, min_duration, max_duration
    )

    out: List[Dict[str, Any]] = [dict(r) for r in rows]

    i = 0
    while i < len(out):
        ev_id = parse_event_id(seq[i])
        if ev_id == 0:
            out[i]["event"] = "0"
            i += 1
            continue

        j = i
        while j < len(out) and parse_event_id(seq[j]) == ev_id:
            j += 1

        event_obj = events_by_id.get(ev_id)
        if event_obj is None:
            label = str(ev_id)
        else:
            onset_row = out[i]
            end_row = out[j - 1]
            prev_row = out[i - 1] if i > 0 else None
            label = build_event_label(event_obj, onset_row, end_row, j - i, prev_row)

        for k in range(i, j):
            out[k]["event"] = label
        i = j

    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def inject_events(
    normalized_dir: Path,
    events_path: Path,
    source_datasets: List[str],
    p_start: float,
    min_duration: int,
    max_duration: int,
    seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)

    with events_path.open("r", encoding="utf-8") as f:
        events = json.load(f)
    event_ids = [e["id"] for e in events]
    events_by_id = {int(e["id"]): e for e in events}
    logger.info(f"Event IDs: {event_ids}")

    for ds in source_datasets:
        src_dir = normalized_dir / ds
        dst_dir = normalized_dir / f"test_{ds}"

        if not src_dir.exists():
            logger.warning(f"Source directory not found, skipping: {src_dir}")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.glob("*.json"))
        logger.info(f"{ds}: {len(files)} files → {dst_dir}")

        for src_file in files:
            dst_file = dst_dir / src_file.name

            if src_file.stem.endswith("_metadata"):
                shutil.copy2(src_file, dst_file)
                logger.debug(f"  copied (metadata): {src_file.name}")
            else:
                process_episode(
                    src_file,
                    dst_file,
                    event_ids,
                    events_by_id,
                    p_start,
                    min_duration,
                    max_duration)
                logger.debug(f"  injected events:   {src_file.name}")

        logger.info(f"  Done: {ds}")


def reformat_existing_test_events(
    normalized_dir: Path,
    events_path: Path,
    test_datasets: List[str]) -> None:
    with events_path.open("r", encoding="utf-8") as f:
        events = json.load(f)
    events_by_id = {int(e["id"]): e for e in events}

    for ds in test_datasets:
        ds_dir = normalized_dir / ds
        if not ds_dir.exists():
            logger.warning(f"Test dataset directory not found, skipping: {ds_dir}")
            continue

        files = sorted(ds_dir.glob("*.json"))
        logger.info(f"{ds}: reformatting {len(files)} files")

        for path in files:
            if path.stem.endswith("_metadata"):
                continue

            try:
                with path.open("r", encoding="utf-8") as f:
                    rows: Any = json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping malformed JSON file: {path.name} ({exc})")
                continue
            if not isinstance(rows, list) or not rows:
                continue

            out: List[Dict[str, Any]] = [dict(r) for r in rows]
            i = 0
            while i < len(out):
                ev_id = parse_event_id(out[i].get("event", 0))
                if ev_id == 0:
                    out[i]["event"] = "0"
                    i += 1
                    continue

                j = i
                while j < len(out) and parse_event_id(out[j].get("event", 0)) == ev_id:
                    j += 1

                event_obj = events_by_id.get(ev_id)
                if event_obj is None:
                    label = str(ev_id)
                else:
                    onset_row = out[i]
                    end_row = out[j - 1]
                    prev_row = out[i - 1] if i > 0 else None
                    label = build_event_label(event_obj, onset_row, end_row, j - i, prev_row)

                for k in range(i, j):
                    out[k]["event"] = label
                i = j

            with path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add synthetic event labels to normalized episode JSON files."
    )
    repo_root = Path(__file__).resolve().parents[3]

    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=repo_root / "data" / "normalized_episodes",
        help="Root normalized episodes directory (default: <repo>/data/normalized_episodes)")
    parser.add_argument(
        "--events",
        type=Path,
        default=repo_root / "data" / "events" / "events.json",
        help="Path to events.json (default: <repo>/data/events/events.json)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aursad", "vorausad"],
        help="Source dataset names to process (default: aursad vorausad)")
    parser.add_argument(
        "--p-start",
        type=float,
        default=0.02,
        help="Probability per idle timestep of starting a new event (default: 0.02)")
    parser.add_argument(
        "--min-duration",
        type=int,
        default=5,
        help="Minimum event run length in timesteps (default: 5)")
    parser.add_argument(
        "--max-duration",
        type=int,
        default=30,
        help="Maximum event run length in timesteps (default: 30)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--reformat-existing-test-events",
        action="store_true",
        help=(
            "Reformat existing event values in test_* datasets to i_v1_v2_... "
            "without regenerating event placement"
        ))
    parser.add_argument(
        "--test-datasets",
        nargs="+",
        default=["test_aursad", "test_vorausad"],
        help="Test dataset names used with --reformat-existing-test-events")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    if args.reformat_existing_test_events:
        reformat_existing_test_events(
            normalized_dir=args.normalized_dir,
            events_path=args.events,
            test_datasets=args.test_datasets)
    else:
        inject_events(
            normalized_dir=args.normalized_dir,
            events_path=args.events,
            source_datasets=args.datasets,
            p_start=args.p_start,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            seed=args.seed)


if __name__ == "__main__":
    main()
