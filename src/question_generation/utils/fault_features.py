"""Which channels a fault is actually diagnosable from.

Templates that name the anomaly in the prompt ("the sensor stream below is
from a robot exhibiting a TCP frame misconfiguration") only make sense if the
context carries the signals that fault shows up in. Nothing was checking that,
and the templates pick their features from a fixed ``important_features`` list
that does not vary with the fault, so an item could name a fault and then show
none of the channels it manifests in.

The mapping already exists in the labelling, spread over two files:
``root_causes.json`` gives each fault its ``possible_anomalies``, and
``anomalies.json`` gives each anomaly its ``relevant_features_from_schema``.
Composing them gives fault -> channels.

Worth noting because it is counterintuitive: a TCP frame misconfiguration
resolves to ``protective_stop_event`` and ``persistent_tracking_error``, whose
features are joint-space (position tracking, current, contact force). It has no
TCP channel among them. So "does this item show TCP data" is the wrong question
for those items even though the fault is named after the tool frame, and the
right one is whether the joint-space evidence is there.

Channel sets run large, up to 96 for a fault with four candidate anomalies,
against a 30-channel budget per item. ``select_features`` spends that budget
breadth-first across the fault's anomalies rather than depth-first, so every
anomaly the fault could present as keeps some evidence in the context instead
of the first one consuming the whole allowance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from src.question_generation.utils.mc_availability import expand_feature

MAX_CONTEXT_FEATURES = 30


def _entries(blob: Any) -> List[Dict[str, Any]]:
    if isinstance(blob, dict):
        return [v for v in blob.values() if isinstance(v, dict)]
    return [v for v in (blob or []) if isinstance(v, dict)]


def load_fault_feature_map(labelling_dir: Path) -> Dict[int, List[List[str]]]:
    """fault_id -> one channel group per anomaly the fault can present as.

    Groups are kept separate rather than merged so a caller can spread a
    limited feature budget across them.
    """
    try:
        root_causes = json.loads((labelling_dir / "root_causes.json").read_text(encoding="utf-8"))
        anomalies = json.loads((labelling_dir / "anomalies.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    by_name = {
        str(a.get("anomaly_name")): a
        for a in _entries(anomalies)
        if a.get("anomaly_name")
    }
    out: Dict[int, List[List[str]]] = {}
    for rc in _entries(root_causes):
        try:
            fid = int(rc.get("fault_id"))
        except (TypeError, ValueError):
            continue
        groups: List[List[str]] = []
        for name in rc.get("possible_anomalies") or []:
            anomaly = by_name.get(str(name))
            if not anomaly:
                continue
            channels: Set[str] = set()
            for spec in anomaly.get("relevant_features_from_schema") or []:
                expanded = expand_feature(spec)
                if expanded:
                    channels |= expanded
            if channels:
                groups.append(sorted(channels))
        if groups:
            out[fid] = groups
    return out


def required_channels(fault_id: int, fault_map: Dict[int, List[List[str]]]) -> Set[str]:
    """Every channel any of the fault's anomalies is read from."""
    out: Set[str] = set()
    for group in fault_map.get(int(fault_id), []):
        out |= set(group)
    return out


def coverage(fault_id: int, fault_map: Dict[int, List[List[str]]],
             channels: Set[str]) -> Optional[float]:
    """Fraction of the fault's anomalies with at least one channel present.

    None when the fault has no mapping, which the caller must not read as
    either pass or fail.
    """
    groups = fault_map.get(int(fault_id), [])
    if not groups:
        return None
    return sum(1 for g in groups if set(g) & channels) / len(groups)


def select_features(
    fault_id: int,
    fault_map: Dict[int, List[List[str]]],
    available: Iterable[str],
    baseline: Sequence[str] = (),
    limit: int = MAX_CONTEXT_FEATURES,
    reserve: Sequence[Sequence[str]] = (),
) -> List[str]:
    """Up to ``limit`` channels covering the fault, then ``reserve``, then padding.

    Three claims on one budget, in priority order:

    1. the fault's own anomaly groups, round-robin so each is represented;
    2. ``reserve``, a list of channel groups the item needs for something other
       than the fault, one channel taken per group, again round-robin;
    3. whatever of ``baseline`` still fits.

    Step 2 exists because covering only the fault starves the multiple-choice
    pool. The option filter drops any statement whose channels the context does
    not show, so a context holding nothing but the fault's diagnostic channels
    leaves too few answerable options to fill an item: measured on UR3 it left
    exactly three usable statement families against the four an item needs, and
    every UR3 item was being discarded. Spending a little of the budget on one
    channel per option family costs the fault almost nothing and keeps the
    item constructible.
    """
    have = {c for c in available}
    picked: List[str] = []
    seen: Set[str] = set()

    def _round_robin(groups: List[List[str]]) -> None:
        depth = 0
        while groups and len(picked) < limit:
            progressed = False
            for group in groups:
                if depth >= len(group):
                    continue
                progressed = True
                channel = group[depth]
                if channel not in seen:
                    seen.add(channel)
                    picked.append(channel)
                    if len(picked) >= limit:
                        return
            if not progressed:
                return
            depth += 1

    fault_groups = [[c for c in g if c in have]
                    for g in fault_map.get(int(fault_id), [])
                    if any(c in have for c in g)]
    reserve_groups = [[c for c in g if c in have]
                      for g in reserve
                      if any(c in have for c in g)]

    # One pass each before either is deepened. Letting the fault run to
    # exhaustion first spent the whole budget on it and left the reserve
    # nothing, which is the same starvation this argument exists to prevent.
    _round_robin([g[:1] for g in fault_groups])
    _round_robin([g[:1] for g in reserve_groups])
    _round_robin([[c for c in g if c not in seen] for g in fault_groups])
    _round_robin([[c for c in g if c not in seen] for g in reserve_groups])
    for channel in baseline:
        if len(picked) >= limit:
            break
        if channel in have and channel not in seen:
            seen.add(channel)
            picked.append(channel)
    return picked
