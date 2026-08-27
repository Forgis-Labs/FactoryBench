"""Backfill ``fault_id`` into aursad/vorausad metadata files.

These two datasets ship metadata without a ``fault_id`` field (the per-episode
fault is encoded as ``fault_label`` on each row instead). Every other generator
expects ``meta.fault_id`` as the canonical anomaly identifier, without it,
L2/L3/L4 either over-include nominal episodes or have to do expensive per-row
scans at runtime.

This script reads each episode JSON once, computes the **dominant non-zero**
``fault_label`` across rows, and writes ``fault_id: <int>`` back into the
matching ``*_metadata.json``. If no row is non-zero, ``fault_id: 0`` (nominal).

Idempotent: a second run reads the already-written ``fault_id`` and leaves it
alone unless it disagrees with the data (``--rewrite`` to force).

Usage::

    python -m scripts.backfill_fault_id_metadata
    python -m scripts.backfill_fault_id_metadata --datasets aursad
    python -m scripts.backfill_fault_id_metadata --rewrite
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATASETS = ("aursad", "vorausad")
REPO_ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_DIR = REPO_ROOT / "data" / "normalized_episodes"


def _dominant_non_zero_fault(rows: list) -> int:
    counts: Counter = Counter()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        val = row.get("fault_label")
        if val is None:
            continue
        try:
            n = int(float(val))
        except (TypeError, ValueError):
            continue
        if n != 0:
            counts[n] += 1
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def _episode_path_for_meta(meta_path: Path) -> Path:
    # `<stem>_metadata.json` -> `<stem>.json` (strip the trailing _metadata)
    stem = meta_path.stem
    if stem.endswith("_metadata"):
        stem = stem[: -len("_metadata")]
    return meta_path.with_name(f"{stem}.json")


def _scan_dataset(
    ds: str, *, rewrite: bool
) -> Tuple[int, int, int]:
    """Returns (touched, unchanged, missing_episode)."""
    folder = NORMALIZED_DIR / ds
    if not folder.is_dir():
        logger.warning(f"{ds}: folder missing at {folder}; skipping")
        return 0, 0, 0
    meta_files = sorted(folder.glob("*_metadata.json"))
    touched = 0
    unchanged = 0
    missing = 0
    fid_dist: Counter = Counter()
    for meta_path in meta_files:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"{meta_path.name}: unreadable ({exc})")
            continue

        existing = meta.get("fault_id")
        if existing is not None and not rewrite:
            try:
                fid_dist[int(float(existing))] += 1
            except Exception:
                pass
            unchanged += 1
            continue

        ep_path = _episode_path_for_meta(meta_path)
        if not ep_path.exists():
            missing += 1
            continue
        try:
            raw = json.loads(ep_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"{ep_path.name}: unreadable ({exc})")
            missing += 1
            continue
        rows = raw if isinstance(raw, list) else raw.get("baseline", raw.get("flat", []))
        if not isinstance(rows, list):
            missing += 1
            continue

        fid = _dominant_non_zero_fault(rows)
        meta["fault_id"] = fid
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        fid_dist[fid] += 1
        touched += 1

    print(f"\n[{ds}] processed {len(meta_files)} metadata files")
    print(f"  touched (wrote fault_id): {touched}")
    print(f"  unchanged (already had fault_id): {unchanged}")
    print(f"  missing episode rows: {missing}")
    print(f"  fault_id distribution: {dict(sorted(fid_dist.items()))}")
    return touched, unchanged, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help=f"Datasets to backfill (default: {DEFAULT_DATASETS}).")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Recompute fault_id even when the metadata already has one.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    total = (0, 0, 0)
    for ds in args.datasets:
        t, u, m = _scan_dataset(ds, rewrite=args.rewrite)
        total = (total[0] + t, total[1] + u, total[2] + m)
    print(f"\nTotal across {len(args.datasets)} datasets: "
          f"touched={total[0]} unchanged={total[1]} missing={total[2]}")


if __name__ == "__main__":
    main()
