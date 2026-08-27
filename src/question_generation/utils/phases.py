"""Canonical task-phase vocabulary, shared by question generation and prompting.

FactoryBench episodes are segmented into task phases, and the benchmark refers
to those phases in two different surface forms that a tested model has to
reconcile on its own:

  * the ``task_phase`` signal in the context time series, which carries a bare
    integer label (``tph=3``);
  * the prose the question templates use (``the grasp of the object phase``).

Neither is self-describing. Before this module existed, a model was asked to
"isolate the grasp of the object phase" with no statement of what that phase
is, where it sits in the task, or how it relates to the integers it can see in
the series. That ambiguity is a property of the prompt, not of the model, and
it shows up as noise in the scores.

``build_phase_reference`` renders a legend joining the three names for each
phase, integer label, canonical name, question prose, with the one-line
description from ``data/labelling/tasks.json``. ``build_prompts_from_questions``
attaches it to any prompt whose question or context involves phases.

The legend is deliberately timing-free. It says what a phase *is* and where it
sits in the task's fixed order; it never says which timesteps of *this* episode
belong to it. That distinction matters because several templates (L2 template 6
in particular) ask the model to locate a phase boundary in the window, and the
answer would leak otherwise. Definitions are shared context; segmentations are
the thing under test.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# Phase index -> the wording the question templates use. Moved here verbatim
# from level1.py so generation and prompting read one dictionary instead of
# drifting apart; level1 imports it back under its original name. Do not
# reword these: they appear inside already-published question text.
PHASE_NAMES: Dict[str, Dict[str, str]] = {
    "pick_and_place": {
        "0": "approach to the object",
        "1": "descent to the object",
        "2": "pre-grasp pause",
        "3": "grasp of the object",
        "4": "lift of the object",
        "5": "transfer to the bin",
        "6": "descent to the bin",
        "7": "release of the object",
        "8": "retreat from the bin",
        "9": "return to home",
    },
    "screwing": {
        "0": "approach to the fastener",
        "1": "descent to the fastener",
        "2": "tightening of the fastener",
        "3": "disengagement from the fastener",
        "4": "retreat to a safe height",
        "5": "re-descent to the fastener",
        "6": "loosening of the fastener",
        "7": "re-engagement with the fastener",
        "8": "return to home",
    },
    "peg_in_hole": {
        "0": "approach to the hole",
        "1": "insertion of the peg",
        "2": "release of the peg",
        "3": "retreat from the hole",
        "4": "return to home",
        "5": "approach to the peg",
        "6": "grasp of the peg",
        "7": "lift of the peg",
        "8": "return to home",
    },
}

DEFAULT_TASKS_PATH = Path("data") / "labelling" / "tasks.json"

# Signal name and its acronym as emitted by utils.template.encode_chunk.
TASK_PHASE_SIGNAL = "task_phase"

# Questions that hide the phase vocabulary opt out via the standard `hides`
# mechanism, the same way `robot` and `gripper` already work. Nothing emits
# this today; it exists so a future "which phase is this?" template can be
# added without silently handing the model its own answer key.
HIDE_TOKEN = "task_phases"

_PHASE_WORD = re.compile(r"\bphases?\b", re.IGNORECASE)


@lru_cache(maxsize=8)
def load_tasks(tasks_path: str) -> Dict[str, Dict[str, Any]]:
    """Load ``tasks.json`` into ``{task_id: task_object}``.

    Returns an empty dict when the file is missing or malformed, callers treat
    an empty vocabulary as "emit no legend" rather than failing the build.
    """
    path = Path(tasks_path)
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}
    return {
        str(task["id"]): task
        for task in raw
        if isinstance(task, dict) and task.get("id")
    }


def resolve_task_from_episode(
    dataset: str,
    episode: str,
    episodes_root: Path) -> Optional[str]:
    """Look up an episode's task from its ``*_metadata.json`` sidecar.

    Mirrors ``level1._episode_task``, but reachable from the prompt builder,
    which sees only a question's provenance. This is what lets an existing
    question set gain phase definitions by re-running the prompt build, with no
    question regeneration.
    """
    if not dataset or not episode:
        return None
    meta_path = episodes_root / dataset / f"{episode}_metadata.json"
    if not meta_path.is_file():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    task = meta.get("task")
    return str(task) if task else None


def observed_phase_labels(context: Any) -> List[int]:
    """Integer ``task_phase`` values present in the rendered context series."""
    if not isinstance(context, dict):
        return []
    ts_format = context.get("time_series_format")
    if not isinstance(ts_format, dict):
        return []
    mapping = ts_format.get("acronym_mapping")
    if not isinstance(mapping, dict):
        return []
    acronym = next(
        (a for a, full in mapping.items() if full == TASK_PHASE_SIGNAL), None
    )
    if not acronym:
        return []
    labels = set()

    rows = context.get("time_series")
    if isinstance(rows, list):
        pattern = re.compile(rf"\b{re.escape(str(acronym))}=(-?\d+)")
        for row in rows:
            labels.update(int(m) for m in pattern.findall(str(row)))

    # A window that sits entirely inside one phase has task_phase hoisted out
    # of the rows into constant_features, so the label is still known.
    notes = context.get("notes")
    if isinstance(notes, dict):
        constants = notes.get("constant_features")
        if isinstance(constants, dict):
            try:
                labels.add(int(float(constants[TASK_PHASE_SIGNAL])))
            except (KeyError, TypeError, ValueError):
                pass

    return sorted(labels)


def resolve_task_from_context_labels(context: Any) -> Optional[str]:
    """Infer the task from the phase labels visible in the series.

    Mirrors ``src.data.factorywave_normalizer._infer_task_from_phases``: only
    pick_and_place has a phase 9, so seeing one is decisive. A maximum of 8 is
    shared by screwing and peg_in_hole and stays unresolved, emitting the
    wrong task's definitions would be worse than emitting none.
    """
    labels = observed_phase_labels(context)
    if labels and max(labels) >= 9:
        return "pick_and_place"
    return None


def resolve_task_from_question_text(question_text: str) -> Optional[str]:
    """Infer the task from phase prose that is unique to one task.

    Fallback for episodes whose metadata sidecar is missing. Only fires when
    the matched wording belongs to exactly one task, "return to home" appears
    in all three and resolves nothing, while "insertion of the peg" is
    unambiguous.
    """
    text = (question_text or "").lower()
    hits = {
        task_id
        for task_id, phases in PHASE_NAMES.items()
        if any(alias in text for alias in phases.values())
    }
    if len(hits) == 1:
        return hits.pop()

    # Narrow by the aliases that are themselves unique across tasks.
    alias_owner: Dict[str, set] = {}
    for task_id, phases in PHASE_NAMES.items():
        for alias in phases.values():
            alias_owner.setdefault(alias, set()).add(task_id)
    unique = {
        next(iter(owners))
        for alias, owners in alias_owner.items()
        if len(owners) == 1 and alias in text
    }
    return unique.pop() if len(unique) == 1 else None


# dataset.json marks factorywave as "mixed": its episodes span all three tasks,
# so the dataset-level answer is not a task and must not be used as one.
MIXED_TASK_SENTINEL = "mixed"


def resolve_task_from_dataset(
    dataset: str,
    dataset_index: Optional[Dict[str, Dict[str, Any]]]) -> Optional[str]:
    """Dataset-level task from ``dataset.json`` (aursad -> screwing, etc.).

    Returns None for datasets marked ``mixed``, which carry more than one task
    and can only be resolved per-episode.
    """
    if not dataset or not dataset_index:
        return None
    task = (dataset_index.get(dataset) or {}).get("task_id")
    if not task or str(task).strip().lower() == MIXED_TASK_SENTINEL:
        return None
    return str(task)


def resolve_task(
    question_item: Dict[str, Any],
    episodes_root: Path,
    dataset_index: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    """Best-effort task id for a question.

    Four sources, most to least specific: provenance, the episode's metadata
    sidecar, the dataset-level task in ``dataset.json``, and finally the phase
    prose in the question itself. The chain exists because no single source
    covers the corpus: factorywave is a mixed-task dataset that only the
    sidecar can disambiguate, while aursad and vorausad are single-task but
    ship no per-episode sidecars.
    """
    provenance = question_item.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}

    # Emitted by newer generator runs; cheapest and most reliable when present.
    task = provenance.get("task")
    if task and str(task).strip().lower() != MIXED_TASK_SENTINEL:
        return str(task)

    dataset = str(provenance.get("dataset") or "")
    # L1 comparison templates carry two episodes under _a / _b suffixes; either
    # resolves the same phase vocabulary, so take whichever is present.
    datasets_seen: List[str] = [dataset] if dataset else []
    for key in ("episode", "episode_a", "episode_b"):
        episode = provenance.get(key)
        if not episode:
            continue
        suffix = key[len("episode"):]  # "", "_a", "_b"
        ds = dataset or str(provenance.get(f"dataset{suffix}") or "")
        if ds and ds not in datasets_seen:
            datasets_seen.append(ds)
        resolved = resolve_task_from_episode(ds, str(episode), episodes_root)
        if resolved:
            return resolved

    for ds in datasets_seen:
        resolved = resolve_task_from_dataset(ds, dataset_index)
        if resolved:
            return resolved

    resolved = resolve_task_from_context_labels(question_item.get("context"))
    if resolved:
        return resolved

    return resolve_task_from_question_text(str(question_item.get("question", "")))


def context_mentions_task_phase(context: Any) -> bool:
    """True when the rendered context exposes the ``task_phase`` signal."""
    if not isinstance(context, dict):
        return False
    ts_format = context.get("time_series_format")
    if isinstance(ts_format, dict):
        mapping = ts_format.get("acronym_mapping")
        if isinstance(mapping, dict) and TASK_PHASE_SIGNAL in mapping.values():
            return True
    notes = context.get("notes")
    if isinstance(notes, dict):
        constants = notes.get("constant_features")
        if isinstance(constants, dict) and TASK_PHASE_SIGNAL in constants:
            return True
    return False


def needs_phase_reference(question_item: Dict[str, Any]) -> bool:
    """Whether this question's prompt should carry the phase legend.

    True when the question talks about phases, or when the context exposes the
    ``task_phase`` signal the model would otherwise have to decode blind.
    Keeping this conditional matters: the legend is ~10 lines and most L1
    prompts have no phase content at all.
    """
    if HIDE_TOKEN in set(question_item.get("hides") or []):
        return False
    if _PHASE_WORD.search(str(question_item.get("question", ""))):
        return True
    options = question_item.get("options")
    if isinstance(options, dict) and any(
        _PHASE_WORD.search(str(v)) for v in options.values()
    ):
        return True
    if isinstance(options, list) and any(_PHASE_WORD.search(str(v)) for v in options):
        return True
    return context_mentions_task_phase(question_item.get("context"))


def build_phase_reference(
    task_id: Optional[str],
    tasks: Dict[str, Dict[str, Any]],
    include_label_column: bool = True) -> str:
    """Render the phase legend for ``task_id``.

    One line per phase: integer label, canonical name, the question-template
    prose for the same phase, and the one-line description. The prose alias is
    only printed when it differs from the canonical name, so the block stays
    readable rather than repeating itself.

    Returns "" when the task is unknown or absent from ``tasks.json``, a
    missing legend is better than a guessed one.
    """
    if not task_id:
        return ""
    task = tasks.get(task_id)
    if not isinstance(task, dict):
        return ""
    phases = task.get("phases")
    if not isinstance(phases, list) or not phases:
        return ""

    aliases = PHASE_NAMES.get(task_id, {})
    task_name = str(task.get("name") or task_id)
    task_desc = str(task.get("description") or "").strip()

    lines: List[str] = []
    header = f"Task: {task_name}."
    if task_desc:
        header += f" {task_desc}"
    lines.append(header)
    lines.append(
        "The task runs through the following phases in this fixed order"
        + (
            f", labelled by the {TASK_PHASE_SIGNAL} signal:"
            if include_label_column
            else ":"
        )
    )

    for phase in phases:
        if not isinstance(phase, dict):
            continue
        label = phase.get("label")
        name = str(phase.get("name") or phase.get("id") or "").strip()
        description = str(phase.get("description") or "").strip()
        alias = aliases.get(str(label), "").strip()

        prefix = f"  {label} = " if include_label_column else "  - "
        entry = f"{prefix}{name}"
        # Only surface the alias when it adds something. "Insert" vs "insertion
        # of the peg" is worth stating; a duplicate is just noise.
        if alias and alias.lower() != name.lower():
            entry += f' (referred to in questions as "{alias}")'
        if description:
            entry += f": {description}"
        lines.append(entry)

    return "\n".join(lines)


def phase_reference_for_question(
    question_item: Dict[str, Any],
    episodes_root: Path,
    tasks_path: Path = DEFAULT_TASKS_PATH,
    dataset_index: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """Legend for ``question_item``, or "" when one is not warranted."""
    if not needs_phase_reference(question_item):
        return ""
    tasks = load_tasks(str(tasks_path))
    if not tasks:
        return ""
    task_id = resolve_task(question_item, episodes_root, dataset_index)
    return build_phase_reference(
        task_id,
        tasks,
        include_label_column=context_mentions_task_phase(question_item.get("context")))
