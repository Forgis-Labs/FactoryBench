"""
Precompute counterfactual selection via KL divergence.

Reads raw ur_signals parquet files and the episode table, computes KL divergence
between each CF's pre-fault segment and its baseline, and saves the best CF per
baseline group to a JSON cache file.

The normalizer then reads this cache instead of recomputing KL every time.

Usage:
    python -m src.data.precompute_cf_selection -v
    python -m src.data.precompute_cf_selection --limit 20 -v
"""

import ast
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from src.data.factorywave_normalizer import (
    compute_kl_divergence,
    _find_fault_onset,
    _KL_SIGNAL_COLS)

logger = logging.getLogger(__name__)

TARGET_HZ = 10


def precompute_cf_selection(
    input_dir: Path,
    output_path: Path,
    limit: Optional[int] = None,
    tasks: Optional[List[str]] = None) -> None:
    """Precompute best CF per baseline and save to JSON cache."""

    # Load episode table
    ep_table = pd.read_parquet(input_dir / "episode.parquet")
    ep_table["_meta"] = ep_table["episode_metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or {})
    )
    ep_table["_task"] = ep_table["_meta"].apply(lambda m: m.get("task") if isinstance(m, dict) else None)
    ep_table["_condition"] = ep_table["_meta"].apply(lambda m: m.get("condition") if isinstance(m, dict) else None)

    if tasks:
        ep_table = ep_table[ep_table["_task"].isin(tasks)]

    ep_lookup = {str(row["id"]): row for _, row in ep_table.iterrows()}

    # Build CF groups
    cf_groups: Dict[str, List[str]] = {}
    for _, row in ep_table.iterrows():
        if row["_condition"] == "counterfactual" and pd.notna(row.get("cf_baseline_episode_id")):
            bl_id = str(row["cf_baseline_episode_id"])
            cf_id = str(row["id"])
            cf_groups.setdefault(bl_id, []).append(cf_id)

    logger.info(f"Episode table: {len(ep_lookup)} episodes, {len(cf_groups)} CF groups")

    # Collect all needed episode IDs
    needed_ids: set = set()
    groups_to_process = list(cf_groups.items())
    if limit:
        groups_to_process = groups_to_process[:limit + 50]
    for bl_id, cids in groups_to_process:
        needed_ids.add(bl_id)
        needed_ids.update(cids)

    # Find signal files
    signal_paths = []
    for subdir in ["ur_signals_10hz", "ur_signals_10hz_sub"]:
        p = input_dir / subdir / "data.parquet"
        if p.exists():
            signal_paths.append(p)
    if not signal_paths:
        for subdir in ["ur_signals_125hz"]:
            p = input_dir / subdir / "data.parquet"
            if p.exists():
                signal_paths.append(p)
    if not signal_paths:
        single = input_dir / "ur_signals.parquet"
        if single.exists():
            signal_paths.append(single)

    logger.info(f"Loading {len(needed_ids)} episodes from {len(signal_paths)} files...")

    # Load needed episodes, sort by time
    episode_dfs: Dict[str, pd.DataFrame] = {}
    for signal_path in signal_paths:
        pf = pq.ParquetFile(signal_path)
        for rg_idx in range(pf.metadata.num_row_groups):
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

    # Sort each episode by time
    for ep_id_str in episode_dfs:
        episode_dfs[ep_id_str] = episode_dfs[ep_id_str].sort_values("time").reset_index(drop=True)

    logger.info(f"Loaded {len(episode_dfs)} episodes")

    # Process CF groups
    results: Dict[str, Any] = {}
    processed = 0
    skipped = 0

    for baseline_id, cf_ids in cf_groups.items():
        if limit and processed >= limit:
            break

        if baseline_id not in episode_dfs:
            skipped += 1
            continue

        baseline_df = episode_dfs[baseline_id]

        candidates: List[Tuple[str, float, int]] = []
        for cf_id in cf_ids:
            if cf_id not in episode_dfs:
                continue
            cf_df = episode_dfs[cf_id]
            cf_row = ep_lookup.get(cf_id)
            if cf_row is None:
                continue

            # Find fault onset
            onset = _find_fault_onset(cf_df)
            if onset is None:
                raw_timestep = cf_row.get("cf_injection_timestep")
                if pd.notna(raw_timestep) and raw_timestep is not None:
                    onset = max(1, int(float(raw_timestep) / (500.0 / TARGET_HZ)))

            if onset is not None and onset >= 5 and onset < len(cf_df):
                # Has a pre-fault segment, compare pre-fault only
                n_rows = min(onset, len(baseline_df))
                kl = compute_kl_divergence(baseline_df, cf_df, n_rows)
                candidates.append((cf_id, kl, onset))
            else:
                # Fault present from start (config faults), compare full episodes
                n_rows = min(len(baseline_df), len(cf_df))
                if n_rows >= 5:
                    kl = compute_kl_divergence(baseline_df, cf_df, n_rows)
                    candidates.append((cf_id, kl, 0))

        if not candidates:
            skipped += 1
            continue

        best_cf_id, best_kl, best_onset = min(candidates, key=lambda x: x[1])
        cf_row = ep_lookup.get(best_cf_id)

        results[baseline_id] = {
            "best_cf_id": best_cf_id,
            "kl_divergence": round(best_kl, 6),
            "fault_onset_index": best_onset,
            "cf_fault_id": int(float(cf_row.get("cf_fault_id", 0))) if cf_row is not None and pd.notna(cf_row.get("cf_fault_id")) else None,
            "candidates_evaluated": len(candidates),
        }

        processed += 1
        if processed % 50 == 0:
            logger.info(f"  {processed} groups processed...")

    logger.info(f"Done: {processed} selected, {skipped} skipped")

    # Save cache
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute CF selection via KL divergence.")
    repo_root = Path(__file__).resolve().parents[2]

    parser.add_argument("--input", type=Path, default=repo_root / "data" / "factorywave" / "data")
    parser.add_argument("--output", type=Path, default=repo_root / "data" / "factorywave" / "cf_selection.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    try:
        precompute_cf_selection(args.input, args.output, args.limit, args.tasks)
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
