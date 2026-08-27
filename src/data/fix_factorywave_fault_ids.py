"""
Fix fault_ids for pick_and_place errored episodes by matching them in CHRONOLOGICAL ORDER
to Error_list.txt entries (ep_numbers ignored - only order + expected fault_id from section
header matter).

Strategy:
1. Parse pick_and_place section of Error_list.txt -> ordered list of expected_fault_ids
   (one entry per error line, regardless of error type).
2. Find all episodes with safety_mode != 1 in ur_signals.parquet.
3. Group consecutive bad episodes (same safety event) by position gap.
4. Order the groups chronologically.
5. Walk through both lists in parallel: N-th txt entry <-> N-th bad group.
6. For each bad group whose current fault_id != expected fault_id, update it.
"""
import ast
import shutil
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/factorywave/data")
PARQUET_PATH = DATA_DIR / "episode.parquet"
BACKUP_PATH = DATA_DIR / "episode.parquet.bak_before_fault_fix"


# ---------------------------------------------------------------------------
# Pick_and_place section of Error_list.txt, IN ORDER, ONE ENTRY PER LINE.
# Only fault_id from section header matters; ep_number and error_type are ignored.
# Entries with no EP number are skipped (uncertain ordering).
# Ends at the Counterfactual section (peg_in_hole/screwing not included).
# ---------------------------------------------------------------------------
PICK_AND_PLACE_EXPECTED_FAULT_IDS = (
    # 0.6kg normal (25 entries)
    [0] * 25 +
    # 0.3kg normal (26 entries)
    [0] * 26 +
    # 1.2kg normal (14 entries)
    [0] * 14 +
    # 1.2kg fault 25 (5)
    [25] * 5 +
    # 0.6kg fault 25 (5)
    [25] * 5 +
    # 0.3kg fault 25 (2)
    [25] * 2 +
    # 0.3kg fault 28 (5)
    [28] * 5 +
    # 0.6kg fault 28 (3)
    [28] * 3 +
    # 1.2kg fault 28 (3)
    [28] * 3 +
    # 1.2kg fault 22 (5)
    [22] * 5 +
    # 0.6kg fault 22 (4)
    [22] * 4 +
    # 0.3kg fault 22 (2)
    [22] * 2 +
    # 0.6kg fault 29 (5)
    [29] * 5 +
    # 0.3kg fault 29 (0 - "no errors")
    # 1.2kg fault 29 (5)
    [29] * 5 +
    # 1.2kg fault 11 (3)
    [11] * 3 +
    # 0.6kg fault 11 (3)
    [11] * 3 +
    # 0.3kg fault 11 (2)
    [11] * 2 +
    # 0.3kg fault 30 (6)
    [30] * 6 +
    # 0.6kg fault 30 (5)
    [30] * 5 +
    # 1.2kg fault 30 (4)
    [30] * 4 +
    # 1.2kg fault 31 (1)
    [31] * 1 +
    # 0.6kg fault 31 (1)
    [31] * 1 +
    # 0.3kg fault 31 (1)
    [31] * 1 +
    # BACKUP USED -> STARTING FROM 2772
    # 0.3kg fault 15 (3, but one has no EP) -> 2 with EP
    [15] * 2 +
    # 0.6kg fault 15 (2)
    [15] * 2 +
    # 1.2kg fault 15 (4)
    [15] * 4 +
    # 0.3kg fault 12 w0 (1)
    [12] * 1 +
    # 0.6kg fault 12 w0 (2)
    [12] * 2 +
    # 1.2kg fault 12 w0 (2)
    [12] * 2 +
    # 1.2kg fault 12 w0.5 (4)
    [12] * 4 +
    # 0.6kg fault 12 w0.5 (3)
    [12] * 3 +
    # 0.3kg fault 12 w0.5 (4)
    [12] * 4 +
    # 0.3kg fault 12 w1 (2)
    [12] * 2 +
    # 0.6kg fault 12 w1 (4)
    [12] * 4 +
    # 1.2kg fault 12 w1 (1)
    [12] * 1 +
    # 1.2kg fault 38 (3)
    [38] * 3 +
    # 0.6kg fault 38 (6)
    [38] * 6 +
    # 0.3kg fault 38 (2)
    [38] * 2 +
    # 0.3kg fault 10 (4)
    [10] * 4 +
    # 0.6kg fault 10 (5)
    [10] * 5 +
    # 1.2kg fault 10 (3)
    [10] * 3 +
    # 0.3kg fault 14 (2)
    [14] * 2 +
    # 0.6kg fault 14 (3)
    [14] * 3 +
    # 1.2kg fault 14 (4)
    [14] * 4 +
    # 0.3kg fault 8 (3, one has no EP) -> 2 with EP
    [8] * 2 +
    # 0.6kg fault 8 (2)
    [8] * 2 +
    # 1.2kg fault 8 (3)
    [8] * 3 +
    # 0.3kg fault 9 (4)
    [9] * 4 +
    # 0.6kg fault 9 (8)
    [9] * 8 +
    # 1.2kg fault 9 (1)
    [9] * 1 +
    # trajectory optimization 0.6kg (1 manual stop, fault_id=0 since it was a normal run)
    [0] * 1
)


def parse_fault_metadata(x):
    if isinstance(x, str) and x:
        return ast.literal_eval(x)
    return {}


def parse_episode_metadata(x):
    if isinstance(x, str) and x:
        return ast.literal_eval(x)
    return {}


def get_sorted_nonscrew(ep: pd.DataFrame) -> pd.DataFrame:
    ep = ep.copy()
    ep["fm"] = ep["fault_metadata"].apply(parse_fault_metadata)
    ep["em"] = ep["episode_metadata"].apply(parse_episode_metadata)
    ep["fault_id"] = ep["fm"].apply(lambda x: x.get("fault_id", 0) if isinstance(x, dict) else 0)
    ep["task"] = ep["em"].apply(lambda x: x.get("task", "") if isinstance(x, dict) else "")
    nonscrew = ep[ep["task"] != "screwing"].sort_values("created_at").reset_index(drop=True)
    nonscrew["ep_seq"] = range(1, len(nonscrew) + 1)
    return nonscrew


def group_bad_episodes(nonscrew: pd.DataFrame, max_position_gap: int = 1) -> list[dict]:
    """Group consecutive bad episodes (safety_mode > 1) into error events.

    Consecutive = position gap <= max_position_gap (default: adjacent only).
    Returns list of dicts with group info ordered chronologically.
    """
    sig = pd.read_parquet(DATA_DIR / "ur_signals.parquet", columns=["episode_id", "safety_mode"])
    max_safety = sig.groupby("episode_id")["safety_mode"].max().reset_index()
    max_safety.columns = ["id", "max_safety_mode"]
    nonscrew = nonscrew.merge(max_safety, on="id", how="left")

    bad = nonscrew[nonscrew["max_safety_mode"] > 1].sort_values("ep_seq").reset_index(drop=True)
    bad["position_gap"] = bad["ep_seq"].diff().fillna(99999)
    bad["event_group"] = (bad["position_gap"] > max_position_gap).cumsum()

    groups = []
    for gid, sub in bad.groupby("event_group"):
        groups.append({
            "group_id": int(gid),
            "first_seq": int(sub["ep_seq"].min()),
            "last_seq": int(sub["ep_seq"].max()),
            "episode_ids": sub["id"].tolist(),
            "fault_ids": sub["fault_id"].tolist(),
            "first_fault_id": int(sub.iloc[0]["fault_id"]),
        })
    return groups


def update_fault_metadata_fault_id(raw_str, new_fault_id: int) -> str:
    d = ast.literal_eval(raw_str) if isinstance(raw_str, str) and raw_str else {}
    d["fault_id"] = new_fault_id
    return str(d)


def update_episode_metadata_fault_id(raw_str, new_fault_id: int) -> str:
    if not raw_str or str(raw_str).strip() in ("{}", ""):
        return raw_str
    d = ast.literal_eval(raw_str)
    if isinstance(d, dict) and "fault_id" in d:
        d["fault_id"] = new_fault_id
    return str(d)


def main(dry_run: bool = False):
    print("Loading episode.parquet...")
    ep = pd.read_parquet(PARQUET_PATH)
    nonscrew = get_sorted_nonscrew(ep)

    print("Grouping consecutive bad episodes...")
    groups = group_bad_episodes(nonscrew, max_position_gap=1)
    print(f"Bad episode groups (chronological): {len(groups)}")
    print(f"Pick_and_place txt entries (chronological): {len(PICK_AND_PLACE_EXPECTED_FAULT_IDS)}")

    if len(groups) < len(PICK_AND_PLACE_EXPECTED_FAULT_IDS):
        print(f"WARNING: fewer bad groups ({len(groups)}) than txt entries ({len(PICK_AND_PLACE_EXPECTED_FAULT_IDS)}).")
        print("  This means some txt entries (likely manual_stops) did not trigger safety_mode!=1.")
        print("  In-order matching will cover only the first {} groups.".format(
            min(len(groups), len(PICK_AND_PLACE_EXPECTED_FAULT_IDS))))

    # Match in order, 1-to-1
    N = min(len(groups), len(PICK_AND_PLACE_EXPECTED_FAULT_IDS))
    mismatches = []
    matches_correct = 0
    for i in range(N):
        group = groups[i]
        expected = PICK_AND_PLACE_EXPECTED_FAULT_IDS[i]
        actual = group["first_fault_id"]
        if actual != expected:
            mismatches.append({
                "order_idx": i,
                "first_seq": group["first_seq"],
                "last_seq": group["last_seq"],
                "actual_fault_id": actual,
                "expected_fault_id": expected,
                "episode_ids": group["episode_ids"],
            })
        else:
            matches_correct += 1

    print(f"\nIn-order matches: {N}")
    print(f"  Correct: {matches_correct}")
    print(f"  Mismatches (will be fixed): {len(mismatches)}")

    if mismatches:
        print("\n=== MISMATCHES ===")
        for m in mismatches:
            print(f"  order={m['order_idx']:3d} seq={m['first_seq']}-{m['last_seq']} "
                  f"actual={m['actual_fault_id']} expected={m['expected_fault_id']} "
                  f"({len(m['episode_ids'])} episode(s))")

    if dry_run:
        print("\nDRY RUN - no changes written.")
        return mismatches

    if not mismatches:
        print("No mismatches to fix.")
        return mismatches

    # Apply fixes
    if not BACKUP_PATH.exists():
        print(f"\nCreating backup at {BACKUP_PATH}...")
        shutil.copy2(PARQUET_PATH, BACKUP_PATH)
    else:
        print(f"\nBackup exists at {BACKUP_PATH} (not overwriting).")

    ids_to_fix = {}
    for m in mismatches:
        for ep_id in m["episode_ids"]:
            ids_to_fix[ep_id] = m["expected_fault_id"]

    ep_fixed = ep.copy()
    fixed_count = 0
    for idx, row in ep_fixed.iterrows():
        if row["id"] not in ids_to_fix:
            continue
        new_fid = ids_to_fix[row["id"]]
        ep_fixed.at[idx, "fault_metadata"] = update_fault_metadata_fault_id(row["fault_metadata"], new_fid)
        ep_fixed.at[idx, "episode_metadata"] = update_episode_metadata_fault_id(row["episode_metadata"], new_fid)
        fixed_count += 1

    print(f"Fixed {fixed_count} episodes across {len(mismatches)} groups.")
    ep_fixed.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved updated episode.parquet.")

    return mismatches


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
