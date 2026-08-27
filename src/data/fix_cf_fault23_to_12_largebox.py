"""Remap CF episodes with fault_id 23 to fault_id 12 and switch the box metadata
to the large box (1.2 kg, [0.08, 0.139, 0.044]).

Scope: only the 99 counterfactual episodes (counterfactual in {1, 2, 3}) whose
fault_metadata.fault_id == 23. Baselines are NOT touched.

Touches:
  - data/factorywave/data/episode.parquet
      * fault_metadata.fault_id: 23 -> 12
      * cf_fault_id column: "23.0" / 23.0 -> "12.0" / 12.0 (preserves original dtype)
      * episode_metadata.weight_of_box: 0.6 -> 1.2
      * episode_metadata.position_of_box: "[0.056, 0.122, 0.035]" -> "[0.08, 0.139, 0.044]"
  - data/factorywave/cf_selection.json
      * cf_fault_id: 23 -> 12 (entries whose best CF belongs to the affected group)
  - data/normalized_episodes/factorywave/<baseline_id>_metadata.json
      * weight_of_box: 0.6 -> 1.2
      * position_of_box: set to "[0.08, 0.139, 0.044]" (added if missing)
      * cf_fault_id: set to 12
"""
import argparse
import ast
import json
import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FACTORYWAVE_DIR = REPO_ROOT / "data" / "factorywave"
DATA_DIR = FACTORYWAVE_DIR / "data"
PARQUET_PATH = DATA_DIR / "episode.parquet"
PARQUET_BACKUP = DATA_DIR / "episode.parquet.bak_before_cf23_to_12_largebox"
CF_SELECTION_PATH = FACTORYWAVE_DIR / "cf_selection.json"
CF_SELECTION_BACKUP = FACTORYWAVE_DIR / "cf_selection.json.bak_before_cf23_to_12_largebox"
NORM_DIR = REPO_ROOT / "data" / "normalized_episodes" / "factorywave"

LARGE_BOX_WEIGHT = 1.2
LARGE_BOX_POSITION = "[0.08, 0.139, 0.044]"
OLD_FAULT_ID = 23
NEW_FAULT_ID = 12


def _parse_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x:
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return {}
    return {}


def update_parquet(dry_run: bool) -> set[str]:
    print(f"Loading {PARQUET_PATH}...")
    ep = pd.read_parquet(PARQUET_PATH)

    fm_fault_id = ep["fault_metadata"].apply(
        lambda x: _parse_dict(x).get("fault_id")
    )
    mask = ep["counterfactual"].isin([1, 2, 3]) & (fm_fault_id == OLD_FAULT_ID)

    target = ep[mask]
    print(f"CF episodes matching fault_id={OLD_FAULT_ID}: {len(target)}")
    print(f"  counterfactual level distribution: "
          f"{target['counterfactual'].value_counts().to_dict()}")

    affected_cf_ids = set(target["id"].astype(str))
    if len(target) == 0:
        print("Nothing to update.")
        return affected_cf_ids

    cf_fault_id_dtype = ep["cf_fault_id"].dtype
    if pd.api.types.is_numeric_dtype(cf_fault_id_dtype):
        new_cf_fault_id_val = float(NEW_FAULT_ID)
    else:
        sample = target["cf_fault_id"].dropna().iloc[0] if not target["cf_fault_id"].dropna().empty else None
        new_cf_fault_id_val = f"{NEW_FAULT_ID}.0" if isinstance(sample, str) and sample.endswith(".0") else str(NEW_FAULT_ID)

    def update_fm(x):
        d = _parse_dict(x)
        d["fault_id"] = NEW_FAULT_ID
        return str(d)

    def update_em(x):
        d = _parse_dict(x)
        d["weight_of_box"] = LARGE_BOX_WEIGHT
        d["position_of_box"] = LARGE_BOX_POSITION
        return str(d)

    if dry_run:
        sample_before = ep.loc[mask].iloc[0]
        print("  Sample BEFORE:")
        print(f"    fault_metadata={sample_before['fault_metadata']}")
        print(f"    cf_fault_id={sample_before['cf_fault_id']}")
        print(f"    episode_metadata={sample_before['episode_metadata']}")
        print("  Sample AFTER (dry-run preview):")
        print(f"    fault_metadata={update_fm(sample_before['fault_metadata'])}")
        print(f"    cf_fault_id={new_cf_fault_id_val}")
        print(f"    episode_metadata={update_em(sample_before['episode_metadata'])}")
        return affected_cf_ids

    if not PARQUET_BACKUP.exists():
        print(f"Creating backup {PARQUET_BACKUP}...")
        shutil.copy2(PARQUET_PATH, PARQUET_BACKUP)
    else:
        print(f"Backup already present at {PARQUET_BACKUP} (skipping).")

    ep.loc[mask, "fault_metadata"] = ep.loc[mask, "fault_metadata"].apply(update_fm)
    ep.loc[mask, "episode_metadata"] = ep.loc[mask, "episode_metadata"].apply(update_em)
    ep.loc[mask, "cf_fault_id"] = new_cf_fault_id_val

    ep.to_parquet(PARQUET_PATH, index=False)
    print(f"Wrote updated parquet to {PARQUET_PATH}")
    return affected_cf_ids


def update_cf_selection(affected_cf_ids: set[str], dry_run: bool) -> set[str]:
    if not CF_SELECTION_PATH.exists():
        print(f"cf_selection.json not found at {CF_SELECTION_PATH}")
        return set()

    with open(CF_SELECTION_PATH) as f:
        cf_sel = json.load(f)

    baselines_to_update: set[str] = set()
    for bl_id, entry in cf_sel.items():
        best_cf = entry.get("best_cf_id")
        if entry.get("cf_fault_id") == OLD_FAULT_ID and best_cf in affected_cf_ids:
            baselines_to_update.add(bl_id)

    # Also catch entries whose cf_fault_id==23 but best_cf is outside affected_cf_ids
    # (shouldn't happen, but guard against it).
    extra = {bl for bl, e in cf_sel.items() if e.get("cf_fault_id") == OLD_FAULT_ID} - baselines_to_update
    if extra:
        print(f"WARNING: {len(extra)} cf_selection entries with cf_fault_id=23 but best_cf "
              f"not in affected CF set. They will still be updated.")
        baselines_to_update |= extra

    print(f"cf_selection.json entries to remap: {len(baselines_to_update)}")
    if not baselines_to_update:
        return baselines_to_update

    if dry_run:
        sample_bl = next(iter(baselines_to_update))
        print(f"  Sample BEFORE: {sample_bl}: {cf_sel[sample_bl]}")
        preview = dict(cf_sel[sample_bl], cf_fault_id=NEW_FAULT_ID)
        print(f"  Sample AFTER : {sample_bl}: {preview}")
        return baselines_to_update

    if not CF_SELECTION_BACKUP.exists():
        shutil.copy2(CF_SELECTION_PATH, CF_SELECTION_BACKUP)
        print(f"Backed up cf_selection.json to {CF_SELECTION_BACKUP}")

    for bl_id in baselines_to_update:
        cf_sel[bl_id]["cf_fault_id"] = NEW_FAULT_ID

    with open(CF_SELECTION_PATH, "w") as f:
        json.dump(cf_sel, f, indent=2)
    print(f"Wrote updated {CF_SELECTION_PATH}")
    return baselines_to_update


def update_normalized(baseline_ids: set[str], dry_run: bool) -> None:
    if not NORM_DIR.exists():
        print(f"Normalized dir not found: {NORM_DIR}")
        return

    updated = 0
    missing = 0
    for bl_id in baseline_ids:
        meta_path = NORM_DIR / f"{bl_id}_metadata.json"
        if not meta_path.exists():
            missing += 1
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        meta["weight_of_box"] = LARGE_BOX_WEIGHT
        meta["position_of_box"] = LARGE_BOX_POSITION
        meta["cf_fault_id"] = NEW_FAULT_ID

        if dry_run:
            if updated == 0:
                print(f"  Sample normalized AFTER (dry-run) for {bl_id}:")
                print(json.dumps(meta, indent=2))
            updated += 1
            continue

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        updated += 1

    print(f"Normalized metadata: updated={updated}, missing={missing}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    affected_cf_ids = update_parquet(dry_run=args.dry_run)
    print()
    baseline_ids = update_cf_selection(affected_cf_ids, dry_run=args.dry_run)
    print()
    update_normalized(baseline_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
