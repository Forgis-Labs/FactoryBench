"""Drop multiple-choice options whose evidence is not in the episode.

Every option in ``data/mc_options/mc_options.json`` declares the channels its
statement is about. Nothing was checking those against the episode the item is
built from, so an item could ask whether "robot current remains approximately
constant" while showing a context that carries no current channel at all. On
the released benchmark this affected 46--74% of the options on the four
multi-select templates (L2.2, L2.3, L3.2, L3.3), and every single L3.2 and
L3.3 item contained at least one such option.

Adding the missing channels is not a way out: measured over those items, the
median item would need 26 further channels and 24 of them do not exist in its
source dataset, so no item could be repaired that way. Filtering the option
pool is the only fix, and there is room for it: the smallest per-dataset pool
after filtering is 5 eligible options against the 4 an item needs.

Two details the catalogue forces on us:

* Placeholders are written both ways, ``{axis}`` in some entries and ``{i}``
  in others. Both are expanded here rather than corrected in place, so a
  consumer that only knew one spelling cannot silently under-match.
* A few entries name families that are alternatives rather than joint
  requirements (contact force can come from ``est_contact_force`` or from
  ``true_force``). ``require="any"`` treats the listed families as
  alternatives, ``require="all"`` demands every one. The generators use
  ``any``, which is the reading the verifiable rules imply.

An option whose ``required_features`` cannot be interpreted as channel names
at all (``mc_019`` lists the prose "tracking features") is treated as
ineligible: an unverifiable requirement cannot be checked against an episode,
and shipping it unchecked is what produced the defect in the first place.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

_AXIS_RANGE = range(6)
_CHANNEL_RE = re.compile(r"^[a-z_]+[0-9]?$")


def expand_feature(spec: str) -> Optional[Set[str]]:
    """Concrete channel names for one ``required_features`` entry.

    ``a|b`` means the entry is satisfied by either family, so the two expand
    into one set. That distinguishes a real alternative from two separate
    entries, which matters once an option asks for both of two things: the TCP
    tracking options need a command channel *and* a measured one, but either
    may be a pose or a speed.

    Returns None when the entry is prose rather than a channel specification,
    which the caller must treat as "cannot be verified".
    """
    spec = str(spec).strip()
    if "|" in spec:
        out: Set[str] = set()
        for part in spec.split("|"):
            expanded = expand_feature(part)
            if expanded is None:
                return None
            out |= expanded
        return out or None
    if "{axis}" in spec or "{i}" in spec:
        base = spec.replace("{axis}", "{}").replace("{i}", "{}")
        return {base.format(i) for i in _AXIS_RANGE}
    if _CHANNEL_RE.match(spec):
        return {spec}
    return None


def available_channels(rows: Sequence[Dict[str, Any]], sample: int = 24) -> Set[str]:
    """Channels carrying a non-null value anywhere in the first ``sample`` rows."""
    out: Set[str] = set()
    for row in list(rows)[:sample]:
        if isinstance(row, dict):
            out |= {k for k, v in row.items() if v is not None}
    return out


def option_is_answerable(
    option: Dict[str, Any],
    channels: Set[str],
    require: str = "any",
) -> bool:
    """Whether ``option``'s evidence is present in ``channels``.

    An option may override ``require`` with its own ``require`` field. A
    statement that compares two quantities is only answerable when both are
    shown, and reading its entries as alternatives admits it on the strength of
    half the evidence: every released item carrying a "command versus measured
    TCP motion" option showed the command side and none showed the measured
    one, because the command family alone satisfied ``any``.
    """
    specs = option.get("required_features") or []
    if not specs:
        return True
    require = str(option.get("require") or require)
    groups: List[Set[str]] = []
    for spec in specs:
        expanded = expand_feature(spec)
        if expanded is None:
            return False
        groups.append(expanded)
    if require == "all":
        return all(g & channels for g in groups)
    return any(g & channels for g in groups)


def answerable_option_ids(
    catalogue: Iterable[Dict[str, Any]],
    channels: Set[str],
    level: Optional[int] = None,
    require: str = "any",
) -> Set[str]:
    """Ids from ``catalogue`` whose evidence the episode actually carries."""
    keep: Set[str] = set()
    for option in catalogue:
        if not isinstance(option, dict) or not option.get("id"):
            continue
        if level is not None:
            usable = option.get("usable_levels")
            if usable and level not in usable:
                continue
        if option_is_answerable(option, channels, require=require):
            keep.add(str(option["id"]))
    return keep


def filter_lookup_by_availability(
    mc_option_lookup: Dict[str, str],
    catalogue: Iterable[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    level: Optional[int] = None,
    require: str = "any",
    minimum: int = 4,
) -> Dict[str, str]:
    """Restrict an id -> statement lookup to options this episode can support.

    Falls back to the unfiltered lookup when fewer than ``minimum`` options
    survive, so a thin episode degrades to the previous behaviour rather than
    failing to produce an item at all. Callers that would rather skip the item
    can compare the returned size against ``minimum`` themselves.
    """
    channels = available_channels(rows)
    keep = answerable_option_ids(catalogue, channels, level=level, require=require)
    filtered = {k: v for k, v in mc_option_lookup.items() if k in keep}
    return filtered if len(filtered) >= minimum else dict(mc_option_lookup)
