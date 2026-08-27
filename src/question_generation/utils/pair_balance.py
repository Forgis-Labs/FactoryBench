"""Balanced episode pairing for the two-robot comparison templates.

L1.3 and L2.8 present two streams and ask four independent yes/no questions:
different robots (A), different anomalous states (B), different tasks (C), and
same task at a different phase (D). Both templates drew their pairs uniformly,
which left the labels badly skewed:

    L1.3   P(A)=0.503  P(B)=0.000  P(C)=0.689  P(D)=0.298
    L2.8   P(A)=0.300  P(B)=0.953  P(C)=0.617  P(D)=0.294

B was degenerate in both directions for the same reason: L1 pairs two healthy
episodes so the anomalous states can never differ, while L2 always makes the
first stream faulty so they almost always do. The rest is pool shape, with
7,553 UR3 episodes against 1,302 KUKA. Guessing the per-position majority
scored 0.723 and 0.744 against a chance of 0.500, and emitting one fixed string
scored 35.5% and 32.8% against 6.25%.

This module picks pairs by target cell instead. Callers cycle through the eight
cells of A x B x (C,D) so every proposition lands near 50/50.

D is not free: it can only hold when the tasks match, so P(D) <= P(not C) and
the two can never both be 0.5 unless every same-task pair also differs in
phase. That is the constraint encoded here: cells ask for either "different
tasks" (C true, D false) or "same task, different phase" (C false, D true),
never same task and same phase. Eight cells, all four marginals at 0.5, and
only eight of the sixteen answer strings reachable, so a fixed-string guess
caps at 12.5%.

Supply is not a limit. Over the 8,855-episode pool the tightest cell still has
684,308 distinct ordered pairs, against the few hundred per cell a balanced
template needs.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# (different_robot, different_anomaly, different_task); D is implied by C:
# C true -> D false, C false -> D true (same task, different phase required).
CELLS: Tuple[Tuple[bool, bool, bool], ...] = (
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
)


def anomaly_states_differ(fault_a: Any, fault_b: Any) -> bool:
    """Mirror of the template's own rule for proposition B.

    True when exactly one stream is anomalous, or both are but with different
    faults. Kept here so the sampler targets precisely what the scorer records.
    """
    try:
        fa = int(float(fault_a or 0))
        fb = int(float(fault_b or 0))
    except (TypeError, ValueError):
        return False
    if (fa == 0) != (fb == 0):
        return True
    return fa != 0 and fb != 0 and fa != fb


def cycle_targets(count: int, seed: Optional[int] = None) -> List[Tuple[bool, bool, bool]]:
    """A shuffled, evenly balanced list of target cells of length ``count``.

    Shuffled rather than round-robin so cell membership does not correlate with
    position in the output, which would reintroduce a shortcut for anything
    reading the files in order.
    """
    reps = (count + len(CELLS) - 1) // len(CELLS)
    targets = (list(CELLS) * reps)[:count]
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(targets)
    return targets


# --------------------------------------------------------------------------
# direct cell sampling
#
# Rejection sampling drew the primary episode before it knew the target, so a
# cell needing a cross-robot partner often had none within the retry budget and
# the item was dropped. With 1,302 KUKA episodes against 7,553 UR3 that pushed
# P(different robots) down to 0.30 however many retries were allowed. Indexing
# episodes by their comparison attributes and picking the *group pair* first
# removes the dependence on what the primary happened to be.
# --------------------------------------------------------------------------
def build_index(episodes: Sequence[Dict[str, Any]]) -> Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]]:
    """Group episodes by (robot, task, fault)."""
    index: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = {}
    for ep in episodes:
        index.setdefault((ep.get("robot"), ep.get("task"), ep.get("fault")), []).append(ep)
    return index


def _group_pairs_for(index, target) -> List[Tuple[Tuple, Tuple]]:
    """Every pair of groups whose attributes land in ``target``."""
    keys = list(index)
    out = []
    for ka in keys:
        for kb in keys:
            if (ka[0] != kb[0]) != target[0]:
                continue
            if anomaly_states_differ(ka[2], kb[2]) != target[1]:
                continue
            if (ka[1] != kb[1]) != target[2]:
                continue
            if ka == kb and len(index[ka]) < 2:
                continue
            out.append((ka, kb))
    return out


def sample_pair(
    index,
    target,
    used: Optional[set] = None,
    phases_differ: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    group_pairs_cache: Optional[Dict[Any, List]] = None,
    rng: Optional[random.Random] = None,
    max_tries: int = 200,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """An unused episode pair landing in ``target``, or None.

    Group pairs are chosen uniformly rather than in proportion to how many
    episodes they hold, so a cell is not dominated by whichever combination
    happens to be most common in the corpus. ``used`` records unordered pairs
    already emitted so no two items compare the same two episodes.
    """
    rng = rng or random
    if group_pairs_cache is not None:
        pairs = group_pairs_cache.get(target)
        if pairs is None:
            pairs = _group_pairs_for(index, target)
            group_pairs_cache[target] = pairs
    else:
        pairs = _group_pairs_for(index, target)
    if not pairs:
        return None
    want_same_task = not target[2]
    for _ in range(max_tries):
        ka, kb = pairs[rng.randrange(len(pairs))]
        ga, gb = index[ka], index[kb]
        a = ga[rng.randrange(len(ga))]
        b = gb[rng.randrange(len(gb))]
        if a.get("key") == b.get("key"):
            continue
        if used is not None:
            token = frozenset((a.get("key"), b.get("key")))
            if token in used:
                continue
        if want_same_task and phases_differ is not None and not phases_differ(a, b):
            continue
        if used is not None:
            used.add(frozenset((a.get("key"), b.get("key"))))
        return a, b
    return None
