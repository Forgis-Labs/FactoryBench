"""
Analyze and fix fault_ids for episodes where safety_mode != 1.

The error list provides the recording order of safety events.
Episodes are sorted chronologically to match positions in the error list.
Screwing has its own sequential numbering; all others share one.
"""
import ast
import pandas as pd

DATA_DIR = "data/factorywave/data"

# ---------------------------------------------------------------------------
# Error list: (ep_number_approx, expected_fault_id, error_type)
# Parsed from Error_list.txt. Screwing uses separate numbering (is_screwing=True).
# ---------------------------------------------------------------------------
ERROR_LIST_NONSCREW = [
    # 0.6kg, pick_and_place, normal
    (8, 0, "out_of_bounds"),
    (21, 0, "out_of_bounds"),
    (37, 0, "out_of_bounds"),
    (41, 0, "out_of_bounds"),
    (53, 0, "self_collision"),
    (62, 0, "out_of_bounds"),
    (73, 0, "out_of_bounds"),
    (83, 0, "self_collision"),
    (91, 0, "out_of_bounds"),
    (102, 0, "out_of_bounds"),
    (107, 0, "out_of_bounds"),
    (126, 0, "out_of_bounds"),
    (143, 0, "out_of_bounds"),
    (149, 0, "self_collision"),
    (159, 0, "out_of_bounds"),
    (172, 0, "out_of_bounds"),
    (185, 0, "out_of_bounds"),
    (201, 0, "out_of_bounds"),
    (217, 0, "out_of_bounds"),
    (230, 0, "out_of_bounds"),
    (258, 0, "out_of_bounds"),
    (265, 0, "self_collision"),
    (269, 0, "out_of_bounds"),
    (294, 0, "out_of_bounds"),
    (333, 0, "out_of_bounds"),
    # 0.3kg, pick_and_place, normal
    (341, 0, "out_of_bounds"),
    (352, 0, "self_collision"),
    (380, 0, "out_of_bounds"),
    (388, 0, "out_of_bounds"),
    (396, 0, "out_of_bounds"),
    (409, 0, "out_of_bounds"),
    (412, 0, "out_of_bounds"),
    (448, 0, "out_of_bounds"),
    (484, 0, "out_of_bounds"),
    (485, 0, "out_of_bounds"),
    (504, 0, "out_of_bounds"),
    (517, 0, "self_collision"),
    (532, 0, "self_collision"),
    (618, 0, "manual_stop"),
    (623, 0, "out_of_bounds"),
    (628, 0, "out_of_bounds"),
    (640, 0, "self_collision"),
    (651, 0, "out_of_bounds"),
    (667, 0, "out_of_bounds"),
    (706, 0, "out_of_bounds"),
    (724, 0, "out_of_bounds"),
    (735, 0, "out_of_bounds"),
    (743, 0, "out_of_bounds"),
    (746, 0, "out_of_bounds"),
    (753, 0, "out_of_bounds"),
    (770, 0, "out_of_bounds"),
    # 1.2kg, pick_and_place, normal
    (773, 0, "out_of_bounds"),
    (788, 0, "out_of_bounds"),
    (801, 0, "out_of_bounds"),
    (812, 0, "out_of_bounds"),
    (815, 0, "joint_speed_limit"),
    (831, 0, "self_collision"),
    (840, 0, "self_collision"),
    (860, 0, "out_of_bounds"),
    (878, 0, "out_of_bounds"),
    (958, 0, "self_collision"),
    (963, 0, "out_of_bounds"),
    (1031, 0, "out_of_bounds"),
    (1056, 0, "out_of_bounds"),
    (1126, 0, "self_collision"),
    # 1.2kg, fault 25
    (1165, 25, "self_collision"),
    (1184, 25, "self_collision"),
    (1194, 25, "out_of_bounds"),
    (1212, 25, "out_of_bounds"),
    (1252, 25, "out_of_bounds"),
    # 0.6kg, fault 25
    (1268, 25, "out_of_bounds"),
    (1276, 25, "out_of_bounds"),
    (1279, 25, "self_collision"),
    (1312, 25, "out_of_bounds"),
    (1360, 25, "self_collision"),
    # 0.3kg, fault 25
    (1430, 25, "self_collision"),
    (1463, 25, "manual_stop"),
    # 0.3kg, fault 28
    (1475, 28, "out_of_bounds"),
    (1503, 28, "self_collision"),
    (1531, 28, "out_of_bounds"),
    (1555, 28, "out_of_bounds"),
    (1565, 28, "manual_stop"),
    # 0.6kg, fault 28
    (1589, 28, "out_of_bounds"),
    (1615, 28, "out_of_bounds"),
    (1670, 28, "manual_stop"),
    # 1.2kg, fault 28
    (1720, 28, "self_collision"),
    (1759, 28, "self_collision"),
    (1776, 28, "manual_stop"),
    # 1.2kg, fault 22
    (1788, 22, "self_collision"),
    (1835, 22, "self_collision"),
    (1845, 22, "self_collision"),
    (1869, 22, "self_collision"),
    (1882, 22, "manual_stop"),
    # 0.6kg, fault 22
    (1888, 22, "out_of_bounds"),
    (1907, 22, "self_collision"),
    (1941, 22, "self_collision"),
    (1988, 22, "manual_stop"),
    # 0.3kg, fault 22
    (2030, 22, "out_of_bounds"),
    (2075, 22, "manual_stop"),
    # 0.6kg, fault 29
    (2085, 29, "self_collision"),
    (2105, 29, "self_collision"),
    (2108, 29, "self_collision"),
    (2110, 29, "self_collision"),
    (2129, 29, "manual_stop"),
    # 0.3kg, fault 29: no errors (EP 2151)
    # 1.2kg, fault 29
    (2162, 29, "out_of_bounds"),
    (2182, 29, "self_collision"),
    (2190, 29, "out_of_bounds"),
    (2197, 29, "self_collision"),
    (2212, 29, "out_of_bounds"),
    # 1.2kg, fault 11
    (2236, 11, "out_of_bounds"),
    (2257, 11, "self_collision"),
    (2276, 11, "self_collision"),
    # 0.6kg, fault 11
    (2296, 11, "self_collision"),
    (2319, 11, "out_of_bounds"),
    (2340, 11, "manual_stop"),
    # 0.3kg, fault 11
    (2353, 11, "self_collision"),
    (2400, 11, "self_collision"),
    # 0.3kg, fault 30
    (2404, 30, "self_collision"),
    (2419, 30, "self_collision"),
    (2421, 30, "out_of_bounds"),
    (2441, 30, "self_collision"),
    (2442, 30, "self_collision"),
    (2461, 30, "self_collision"),
    # 0.6kg, fault 30
    (2464, 30, "self_collision"),
    (2492, 30, "self_collision"),
    (2516, 30, "out_of_bounds"),
    (2521, 30, "out_of_bounds"),
    (2527, 30, "manual_stop"),
    # 1.2kg, fault 30
    (2531, 30, "self_collision"),
    (2540, 30, "self_collision"),
    (2565, 30, "self_collision"),
    (2590, 30, "self_collision"),
    # fault 31 (approximate, no detailed errors given)
    (2620, 31, "current"),
    (2650, 31, "current"),
    (2680, 31, "current"),
    # BACKUP USED at 2772
    # 0.3kg, fault 15
    (2811, 15, "manual_stop"),
    # 0.6kg, fault 15
    (2827, 15, "self_collision"),
    (2857, 15, "manual_stop"),
    # 1.2kg, fault 15
    (2876, 15, "self_collision"),
    (2908, 15, "self_collision"),
    (2915, 15, "self_collision"),
    (2938, 15, "manual_stop"),
    # 0.3kg, fault 12 weight 0
    (2994, 12, "manual_stop"),
    # 0.6kg, fault 12 weight 0
    (2996, 12, "self_collision"),
    (3051, 12, "manual_stop"),
    # 1.2kg, fault 12 weight 0
    (3072, 12, "self_collision"),
    (3107, 12, "manual_stop"),
    # 1.2kg, fault 12 weight 0.5
    (3120, 12, "self_collision"),
    (3130, 12, "out_of_bounds"),
    (3153, 12, "self_collision"),
    (3160, 12, "self_collision"),
    # 0.6kg, fault 12 weight 0.5
    (3163, 12, "self_collision"),
    (3176, 12, "out_of_bounds"),
    (3212, 12, "self_collision"),
    # 0.3kg, fault 12 weight 0.5
    (3227, 12, "out_of_bounds"),
    (3242, 12, "self_collision"),
    (3243, 12, "self_collision"),
    (3265, 12, "manual_stop"),
    # 0.3kg, fault 12 weight 1
    (3297, 12, "self_collision"),
    (3320, 12, "manual_stop"),
    # 0.6kg, fault 12 weight 1
    (3325, 12, "out_of_bounds"),
    (3359, 12, "out_of_bounds"),
    (3372, 12, "out_of_bounds"),
    (3384, 12, "manual_stop"),
    # 1.2kg, fault 12 weight 1
    (3442, 12, "out_of_bounds"),
    # 1.2kg, fault 38
    (3449, 38, "self_collision"),
    (3488, 38, "self_collision"),
    (3491, 38, "self_collision"),
    # 0.6kg, fault 38
    (3498, 38, "self_collision"),
    (3501, 38, "self_collision"),
    (3505, 38, "out_of_bounds"),
    (3507, 38, "self_collision"),
    (3508, 38, "out_of_bounds"),
    (3542, 38, "manual_stop"),
    # 0.3kg, fault 38
    (3554, 38, "self_collision"),
    (3593, 38, "manual_stop"),
    # 0.3kg, fault 10
    (3603, 10, "self_collision"),
    (3610, 10, "out_of_bounds"),
    (3641, 10, "out_of_bounds"),
    (3658, 10, "manual_stop"),
    # 0.6kg, fault 10
    (3662, 10, "self_collision"),
    (3672, 10, "self_collision"),
    (3687, 10, "self_collision"),
    (3715, 10, "self_collision"),
    (3725, 10, "manual_stop"),
    # 1.2kg, fault 10
    (3731, 10, "self_collision"),
    (3751, 10, "self_collision"),
    (3793, 10, "manual_stop"),
    # 0.3kg, fault 14
    (3824, 14, "self_collision"),
    (3849, 14, "manual_stop"),
    # 0.6kg, fault 14
    (3887, 14, "self_collision"),
    (3893, 14, "self_collision"),
    (3915, 14, "manual_stop"),
    # 1.2kg, fault 14
    (3939, 14, "out_of_bounds"),
    (3947, 14, "self_collision"),
    (3958, 14, "self_collision"),
    (3983, 14, "manual_stop"),
    # 0.3kg, fault 8 (first error has no EP number)
    (None, 8, "out_of_bounds"),
    (4036, 8, "self_collision"),
    (4051, 8, "manual_stop"),
    # 0.6kg, fault 8
    (4063, 8, "self_collision"),
    (4114, 8, "manual_stop"),
    # 1.2kg, fault 8
    (4135, 8, "self_collision"),
    (4140, 8, "self_collision"),
    (4173, 8, "manual_stop"),
    # 0.3kg, fault 9
    (4190, 9, "self_collision"),
    (4211, 9, "collision"),
    (4222, 9, "self_collision"),
    (4228, 9, "manual_stop"),
    # 0.6kg, fault 9
    (4236, 9, "collision"),
    (4239, 9, "collision"),
    (4244, 9, "collision"),
    (4248, 9, "self_collision"),
    (4251, 9, "collision"),
    (4257, 9, "out_of_bounds"),
    (4270, 9, "collision"),
    (4285, 9, "manual_stop"),
    # 1.2kg, fault 9
    (4342, 9, "out_of_bounds"),
    # trajectory opt 0.6kg
    (4742, 0, "manual_stop"),
    # counterfactual
    (4874, 23, "stop"),
    (4974, 11, "stop"),
    (5074, 30, "stop"),
    (5178, 29, "stop"),
    # peg_in_hole normal
    (5474, 0, "collision"),
    (5500, 0, "stop"),
    # peg fault 28
    (5580, 28, "manual_stop"),
    # peg fault 22
    (5635, 22, "collision"),
    (5660, 22, "manual_stop"),
    # peg fault 12 weight 0
    (5740, 12, "manual_stop"),
    # peg fault 12 weight 0.5
    (5800, 12, "manual_stop"),
    # peg fault 12 weight 1
    (5860, 12, "manual_stop"),
    # peg fault 35
    (5920, 35, "collision"),
    (5950, 35, "manual_stop"),
    # peg fault 38
    (6021, 38, "manual_stop"),
    # peg fault 33
    (6100, 33, "manual_stop"),
    # peg fault 11
    (6160, 11, "manual_stop"),
    # peg fault 30
    (6220, 30, "manual_stop"),
    # peg fault 29
    (6280, 29, "manual_stop"),
    # peg fault 25
    (6360, 25, "manual_stop"),
    # peg fault 34
    (6440, 34, "manual_stop"),
    # peg fault 32
    (6520, 32, "manual_stop"),
    # peg fault 10
    (6552, 10, "manual_stop"),
    (6632, 10, "manual_stop"),
    # peg CF fault 11
    (6732, 11, "manual_stop"),
    # peg CF fault 30
    (6832, 30, "manual_stop"),
    # peg CF fault 32
    (6932, 32, "manual_stop"),
    # peg CF fault 34
    (7032, 34, "manual_stop"),
    # peg fault 36
    (7100, 36, "manual_stop"),
]

# Filter out entries with no EP number
ERROR_LIST_NONSCREW = [(ep, fid, etype) for ep, fid, etype in ERROR_LIST_NONSCREW if ep is not None]


def load_episodes_with_safety():
    ep = pd.read_parquet(f"{DATA_DIR}/episode.parquet")
    ep["fm"] = ep["fault_metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {}
    )
    ep["em"] = ep["episode_metadata"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {}
    )
    ep["fault_id"] = ep["fm"].apply(
        lambda x: x.get("fault_id", 0) if isinstance(x, dict) else 0
    )
    ep["task"] = ep["em"].apply(
        lambda x: x.get("task", "") if isinstance(x, dict) else ""
    )

    # Sort and assign sequential positions
    nonscrew = ep[ep["task"] != "screwing"].sort_values("created_at").reset_index(drop=True)
    nonscrew["ep_seq"] = range(1, len(nonscrew) + 1)

    # Load max safety_mode per episode
    sig = pd.read_parquet(f"{DATA_DIR}/ur_signals.parquet", columns=["episode_id", "safety_mode"])
    max_safety = sig.groupby("episode_id")["safety_mode"].max().reset_index()
    max_safety.columns = ["id", "max_safety_mode"]

    nonscrew = nonscrew.merge(max_safety, on="id", how="left")
    return nonscrew


def match_error_list_to_episodes(nonscrew: pd.DataFrame):
    """For each error list entry, find the nearest bad episode by position."""
    bad = nonscrew[nonscrew["max_safety_mode"] > 1].copy().sort_values("ep_seq")
    bad_seqs = bad["ep_seq"].tolist()

    results = []
    for ep_num, expected_fault_id, error_type in ERROR_LIST_NONSCREW:
        # Find closest bad episode to this EP number
        distances = [(abs(s - ep_num), s) for s in bad_seqs]
        distances.sort()
        nearest_seq = distances[0][1]
        nearest_dist = distances[0][0]

        row = bad[bad["ep_seq"] == nearest_seq].iloc[0]
        actual_fault_id = int(row["fault_id"])
        match = actual_fault_id == expected_fault_id

        results.append(
            {
                "error_list_ep": ep_num,
                "nearest_ep_seq": nearest_seq,
                "distance": nearest_dist,
                "episode_id": row["id"],
                "actual_fault_id": actual_fault_id,
                "expected_fault_id": expected_fault_id,
                "error_type": error_type,
                "match": match,
            }
        )

    return pd.DataFrame(results)


def main():
    print("Loading episodes and signal data...")
    nonscrew = load_episodes_with_safety()
    bad_count = (nonscrew["max_safety_mode"] > 1).sum()
    print(f"Total non-screwing episodes: {len(nonscrew)}, bad (safety_mode>1): {bad_count}")

    print("\nMatching error list entries to bad episodes...")
    results = match_error_list_to_episodes(nonscrew)

    mismatches = results[~results["match"]]
    print(f"\nTotal error list entries: {len(results)}")
    print(f"Matches (fault_id correct): {results['match'].sum()}")
    print(f"Mismatches (fault_id WRONG): {len(mismatches)}")

    if len(mismatches) > 0:
        print("\n=== MISMATCHES (episodes with wrong fault_id) ===")
        print(mismatches[
            ["error_list_ep", "nearest_ep_seq", "distance", "episode_id",
             "actual_fault_id", "expected_fault_id", "error_type"]
        ].to_string(index=False))

    # Also flag entries where the distance is suspiciously large
    large_dist = results[results["distance"] > 50]
    if len(large_dist) > 0:
        print(f"\n=== Entries with large position distance (>50) - uncertain match ===")
        print(large_dist[
            ["error_list_ep", "nearest_ep_seq", "distance", "actual_fault_id", "expected_fault_id"]
        ].to_string(index=False))

    return results, nonscrew


if __name__ == "__main__":
    results, nonscrew = main()
