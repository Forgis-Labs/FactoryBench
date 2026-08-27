"""
Level 4 question generator: Open-Ended Reasoning.

Reads normalized episode JSON files and generates questions from four templates:
  1 - troubleshooting      : anomaly diagnosis + remediation steps (single episode subseries)
  2 - optimization         : throughput / parameter improvement suggestions (single episode subseries)
  3 - ranking by duration  : rank 4 trajectory-opt episodes shortest → longest
  4 - ranking by energy    : rank 4 trajectory-opt episodes most → least energy-efficient

Templates 1–2 use a random subseries of a single episode; ground truth is delegated
to an LLM-as-Judge pipeline.
Templates 3–4 use 4 full episodes that carry episode-level duration/energy metadata;
ground truth is the label permutation sorted by the metric (ascending).

Output: output/questions/level4/level4_{NNNN}.json

Usage:
    python -m src.question_generation.level4.level4 -n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.question_generation.utils.hf_streaming import (
    HfStreamUploader,
    add_streaming_args,
    make_uploader_from_args)
from src.question_generation.utils.io import load_events, load_json, load_root_causes, load_templates, load_ur3_mapping
from src.question_generation.utils.template import (
    build_context,
    discover_episodes_by_dataset)
from src.question_generation.utils.time_series import parse_event_id
from src.question_generation.utils.relevance import (
    is_enabled as relevance_enabled,
    load_specs as load_relevance_specs,
    relevance_report,
    sample_with_relevance,
    validate_relevance)

logger = logging.getLogger(__name__)

VALID_DATASETS = ["aursad", "vorausad", "factorywave", "factorywave_kuka"]
RANKING_LABELS = ["A", "B", "C", "D"]
RANKING_TEMPLATE_IDS = {3, 4}
RANKING_METRIC = {3: "duration", 4: "energy"}

CONTEXT_MIN = 32
CONTEXT_MAX = 64


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------



NO_ANOMALY_TEXT = (
    "No anomalous behavior detected in the sensor stream. "
    "The machine is operating normally; no remediation is required."
)


def _optimization_answer(fault_id: Optional[int], meta: Dict[str, Any]) -> Optional[str]:
    """The corrective action for one episode, or None when it is not derivable."""
    if fault_id == 22:
        configured, correct = meta.get("tcp_offset_configured"), meta.get("correct_tcp_offset")
        if configured is None or correct is None:
            return None
        return (f"The TCP offset is misconfigured at {configured} instead of the correct "
                f"{correct}. Update the TCP position offset in the installation settings "
                f"to {correct}.")
    if fault_id == 23:
        configured, correct = meta.get("payload_mass_configured"), meta.get("correct_payload_mass")
        if configured is None or correct is None:
            return None
        return (f"The payload mass is set to {configured} kg but the actual payload weighs "
                f"{correct} kg. Update the payload mass in the installation settings to "
                f"{correct} kg.")
    if fault_id == 28:
        configured, correct = meta.get("payload_cog_configured"), meta.get("correct_payload_cog")
        if configured is None or correct is None:
            return None
        return (f"The payload center of gravity is set to {configured} but the actual center "
                f"of gravity is {correct}. Update the payload CoG in the installation "
                f"settings to {correct}.")
    return None


def get_root_cause_for_subseries(
    subseries: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    events: List[Dict[str, Any]],
    all_rows: Optional[List[Dict[str, Any]]] = None,
    start_idx: int = 0) -> Dict[str, Any]:
    """
    Determine the anomaly status and root cause for a subseries.

    Event resolution (in priority order):
      1. Any row inside the subseries with a non-zero, non-1 event token.
      2. The most recent non-zero, non-1 event in rows *before* the subseries
         (subseries is in post-event territory).

    Returns a dict with:
      - anomaly_present (bool)
      - fault_label (int)
      - root_cause (str)
      - description (str)
      - event_id (int or None)
      - event_name (str or None)
      - event_context (str): "in_window" | "pre_window" | None
    """
    # --- Fault label: dominant non-zero label in subseries ---
    label_counts: Dict[int, int] = {}
    for row in subseries:
        fl = row.get("fault_label")
        try:
            fl_int = int(float(fl))
            label_counts[fl_int] = label_counts.get(fl_int, 0) + 1
        except (TypeError, ValueError):
            pass
    non_zero_labels = {k: v for k, v in label_counts.items() if k != 0}
    dominant_label = max(non_zero_labels, key=lambda k: non_zero_labels[k]) if non_zero_labels else 0

    # --- Event: first non-zero, non-1 event inside the subseries ---
    event_id_in_window: Optional[int] = None
    for row in subseries:
        ev_id = parse_event_id(row.get("event", 0))
        if ev_id not in (0, 1):
            event_id_in_window = ev_id
            break

    # --- Event: most recent non-zero, non-1 event in rows before the subseries ---
    event_id_pre_window: Optional[int] = None
    if event_id_in_window is None and all_rows is not None and start_idx > 0:
        for row in reversed(all_rows[:start_idx]):
            ev_id = parse_event_id(row.get("event", 0))
            if ev_id not in (0, 1):
                event_id_pre_window = ev_id
                break

    dominant_event_id = event_id_in_window if event_id_in_window is not None else event_id_pre_window
    event_context = (
        "in_window" if event_id_in_window is not None
        else "pre_window" if event_id_pre_window is not None
        else None
    )
    event_obj = next((e for e in events if e["id"] == dominant_event_id), None) if dominant_event_id else None

    if dominant_label == 0 and dominant_event_id is None:
        return {
            "anomaly_present": False,
            "fault_label": 0,
            "root_cause": "normal",
            "description": "Normal operation, no anomaly present.",
            "event_id": None,
            "event_name": None,
            "event_context": None,
        }

    # If the subseries has no non-zero fault_label, the window shows normal
    # operation regardless of event tokens nearby.
    if dominant_label == 0:
        return {
            "anomaly_present": False,
            "fault_label": 0,
            "root_cause": "normal",
            "description": "Normal operation, no anomaly present.",
            "event_id": None,
            "event_name": None,
            "event_context": None,
        }

    rc = root_causes.get(dominant_label, {})
    root_cause_key = rc.get("root_cause", f"fault_{dominant_label}")

    return {
        "anomaly_present": True,
        "fault_label": dominant_label,
        "root_cause": root_cause_key,
        "description": rc.get("description", ""),
        "event_id": dominant_event_id,
        "event_name": event_obj["name"] if event_obj else None,
        "event_context": event_context,
    }


def _extract_episode_meta(raw: Any) -> Dict[str, Any]:
    """Return the episode-level duration and energy from a raw episode dict."""
    if isinstance(raw, dict):
        return {"duration": raw.get("duration"), "energy": raw.get("energy")}
    return {"duration": None, "energy": None}


def _build_ranking_context(
    labeled_rows: List[Tuple[str, List[Dict[str, Any]]]],
    important_features: Optional[List[str]]) -> Dict[str, Any]:
    """
    Build a multi-stream context for ranking questions.

    Each episode is padded to the common maximum length (repeating the last row)
    then encoded independently with build_context.  The result is a dict
    {"streams": {"A": <context>, "B": <context>...}}.
    """
    max_len = max((len(rows) for _, rows in labeled_rows), default=0)
    keep = (set(important_features) | {"timestamp_ms"}) if important_features else None

    streams: Dict[str, Any] = {}
    for label, rows in labeled_rows:
        padded: List[Dict[str, Any]] = list(rows)
        if padded and len(padded) < max_len:
            padded += [dict(padded[-1])] * (max_len - len(padded))
        if keep and padded:
            padded = [{k: v for k, v in row.items() if k in keep} for row in padded]
        streams[label] = build_context(padded)

    return {"streams": streams}


def _try_generate_ranking_question(
    template: Dict[str, Any],
    episodes_by_dataset: Dict[str, List[Path]],
    available_datasets: List[str],
    raw_cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Attempt to build a ranking question (templates 3 and 4).

    Scans all available episodes for ones that carry the required metric
    (duration for template 3, energy for template 4) in their top-level
    metadata, then samples 4 and ranks them.

    Returns None if fewer than 4 eligible episodes exist.
    """
    metric_key = RANKING_METRIC[template["id"]]

    # Collect episodes that have the required metadata field
    candidates: List[Tuple[str, Path]] = []
    for ds in available_datasets:
        for path in episodes_by_dataset[ds]:
            key = str(path)
            if key not in raw_cache:
                raw_cache[key] = load_json(path)
            meta = _extract_episode_meta(raw_cache[key])
            if meta.get(metric_key) is not None:
                candidates.append((ds, path))

    if len(candidates) < 4:
        return None

    sampled = random.sample(candidates, 4)
    labels = list(RANKING_LABELS)
    random.shuffle(labels)

    labeled_rows: List[Tuple[str, List[Dict[str, Any]]]] = []
    metric_values: Dict[str, float] = {}
    provenance_episodes = []

    for (ds, path), label in zip(sampled, labels):
        raw = raw_cache[str(path)]
        if isinstance(raw, dict):
            rows = raw.get("baseline", raw.get("flat", []))
        else:
            rows = raw
        rows = rows if isinstance(rows, list) else []
        if not rows:
            return None
        metric_values[label] = float(_extract_episode_meta(raw)[metric_key])
        labeled_rows.append((label, rows))
        provenance_episodes.append({"dataset": ds, "episode": path.stem, "label": label})

    # Ground truth: labels sorted ascending by metric
    # (shortest duration for template 3; lowest energy = most efficient for template 4)
    answer = "".join(sorted(metric_values, key=lambda lbl: metric_values[lbl]))

    context = _build_ranking_context(labeled_rows, template.get("important_features"))

    return {
        "id": str(uuid.uuid4()),
        "level": 4,
        "template_id": template["id"],
        "template_type": template["type"],
                "hides": template.get("hides", []),
        "question": template["template"],
        "options": {},
        "answer": answer,
        "acceptance_bounds": None,
        "provenance": {"episodes": provenance_episodes},
        "context": context,
    }


def generate_level4_questions(
    datasets_dir: Path,
    output_dir: Path,
    templates: List[Dict[str, Any]],
    root_causes: Dict[int, Dict[str, Any]],
    events: List[Dict[str, Any]],
    n: int = 100,
    seed: Optional[int] = None,
    datasets: Optional[List[str]] = None,
    ur3_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
    relevance_specs: Optional[Dict[int, Dict[str, Any]]] = None,
    enumerate_mode: bool = False,
    uploader: Optional[HfStreamUploader] = None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    relevance_specs = relevance_specs or {}

    allowed = datasets if datasets else VALID_DATASETS
    by_dataset = discover_episodes_by_dataset(datasets_dir, allowed)
    if not by_dataset:
        raise FileNotFoundError(
            f"No normalized episode JSON files found under "
            f"{datasets_dir / 'normalized_episodes'} for datasets: {allowed}"
        )

    episodes_by_dataset = {ds: paths for ds, paths in by_dataset.items() if paths}
    available_datasets = list(episodes_by_dataset.keys())
    if not available_datasets:
        raise FileNotFoundError("No usable datasets found.")

    # LRU-bounded so enumerate-mode runs (which touch every episode) don't OOM.
    episode_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    _EPISODE_CACHE_MAX = 50
    raw_cache: Dict[str, Any] = {}  # only used by ranking templates (currently disabled)
    meta_cache: Dict[str, Tuple[Optional[int], Dict[str, Any]]] = {}

    _OPTIMIZATION_FAULTS = {22, 23, 28}

    def load_episode(path: Path) -> List[Dict[str, Any]]:
        key = str(path)
        if key in episode_cache:
            episode_cache.move_to_end(key)
            return episode_cache[key]
        raw = load_json(path)
        if isinstance(raw, dict):
            rows = raw.get("baseline", raw.get("flat", []))
        else:
            rows = raw
        episode_cache[key] = rows if isinstance(rows, list) else []
        while len(episode_cache) > _EPISODE_CACHE_MAX:
            episode_cache.popitem(last=False)
        return episode_cache[key]

    def load_meta(ep_path: Path) -> Tuple[Optional[int], Dict[str, Any]]:
        key = str(ep_path)
        if key not in meta_cache:
            meta_path = ep_path.with_name(ep_path.stem + "_metadata.json")
            fid = None
            meta: Dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = load_json(meta_path)
                    # Counterfactual episodes leave fault_id null and name the
                    # injected fault in cf_fault_id instead, with a fault_label
                    # of 0 on every row because the injection was never
                    # propagated back into them. Reading only fault_id made
                    # every such episode look healthy, which on a
                    # factorywave-only run turned all 6,461 troubleshooting
                    # items into "no anomalous behavior detected".
                    fid = meta.get("fault_id")
                    if fid is None:
                        fid = meta.get("cf_fault_id")
                    if fid is None and isinstance(meta.get("counterfactual"), dict):
                        fid = meta["counterfactual"].get("cf_fault_id")
                    if fid is not None:
                        fid = int(float(fid))
                except Exception:
                    pass
            meta_cache[key] = (fid, meta)
        return meta_cache[key]

    # Build per-template episode pools based on fault_id
    all_episodes: List[Tuple[str, Path]] = [
        (ds, p) for ds, paths in episodes_by_dataset.items() for p in paths
    ]
    # Torque is not the discriminating channel and requiring it was wrong.
    # effort_target_torque exists only on KUKA, where it turns out to be
    # measured current times a fixed per-joint gain rather than anything the
    # controller predicted, so requiring it selected exactly the slice that
    # cannot answer the question and excluded UR3, which does carry a genuine
    # model-side channel in effort_target_current. The rendering fix supplies
    # that pair instead; the pool just needs an optimization fault.
    optimization_episodes = [
        (ds, p) for ds, p in all_episodes
        if load_meta(p)[0] in _OPTIMIZATION_FAULTS
    ]
    troubleshooting_episodes = [
        (ds, p) for ds, p in all_episodes if load_meta(p)[0] not in _OPTIMIZATION_FAULTS
    ]

    # Hoisted out of the main loop so we can build the deterministic combo
    # list for --enumerate; previously these were recomputed per iteration.
    _DISABLED_TEMPLATE_IDS = {3, 4}
    usable = [t for t in templates if t["id"] not in _DISABLED_TEMPLATE_IDS]
    if not optimization_episodes:
        usable = [t for t in usable if t["id"] != 2]
    if not troubleshooting_episodes:
        usable = [t for t in usable if t["id"] != 1]
    if not usable:
        return

    enum_iter = None
    if enumerate_mode:
        enum_combos = []
        for t in usable:
            tid = t["id"]
            if tid == 2:
                pool = optimization_episodes
            elif tid == 1:
                pool = troubleshooting_episodes
            elif tid in RANKING_TEMPLATE_IDS:
                # Primary episode walks; partners stay random later.
                pool = all_episodes
            else:
                pool = []
            for ds, ep in pool:
                enum_combos.append((t, ds, ep))
        enum_iter = iter(enum_combos)
        max_total_attempts = len(enum_combos)
        logger.info(f"[enumerate] {len(enum_combos)} (template, episode) combos to attempt; -n={n} caps output")
    else:
        max_total_attempts = n * 20

    generated = 0
    attempts = 0

    while generated < n and attempts < max_total_attempts:
        attempts += 1

        if enum_iter is not None:
            try:
                template, sampled_dataset, ep_path = next(enum_iter)
            except StopIteration:
                logger.info(f"[enumerate] all combos exhausted at {generated} questions")
                break
        else:
            template = random.choice(usable)
            sampled_dataset = None
            ep_path = None

        # --- Templates 3 & 4: multi-episode ranking ---
        if template["id"] in RANKING_TEMPLATE_IDS:
            item = _try_generate_ranking_question(
                template, episodes_by_dataset, available_datasets, raw_cache
            )
            if item is None:
                continue

        # --- Templates 1 & 2: single-episode subseries ---
        else:
            if ep_path is None:
                if template["id"] == 2:
                    sampled_dataset, ep_path = random.choice(optimization_episodes)
                else:
                    sampled_dataset, ep_path = random.choice(troubleshooting_episodes)
            rows = load_episode(ep_path)

            if not rows:
                continue

            ep_fault_id, ep_meta = load_meta(ep_path)
            ep_task = str(ep_meta.get("task") or "")
            spec = relevance_specs.get(ep_fault_id) if ep_fault_id is not None else None

            sampled = sample_with_relevance(rows, ep_fault_id or 0, spec, ep_task, CONTEXT_MIN, CONTEXT_MAX)
            if sampled is None:
                continue
            subseries, start_idx, sampler_tag = sampled
            context_len = len(subseries)

            if not subseries:
                continue
            if not validate_relevance(subseries, spec, ep_task):
                continue

            answer = None
            root_cause = None

            if template["id"] == 1:
                rc_info = get_root_cause_for_subseries(
                    subseries, root_causes, events,
                    all_rows=rows, start_idx=start_idx)
                root_cause = rc_info.get("root_cause")
                if rc_info.get("anomaly_present"):
                    ur3_entry = (ur3_mapping or {}).get(root_cause, {})
                    if not ur3_entry.get("ur3_protocol"):
                        # No remediation protocol for this root cause (typically
                        # placeholder/undocumented faults like fault 6, 12), skip
                        # rather than ship an item with answer=null.
                        continue
                    answer = ur3_entry["ur3_protocol"]
                else:
                    answer = NO_ANOMALY_TEXT

            elif template["id"] == 2:
                # KUKA carries no genuine model-side channel: its
                # effort_target_torque is measured current times a fixed
                # per-joint gain, so a configuration error leaves no trace
                # anywhere on the page. Intervention testing put the
                # pose-conditioned response at +0.021 Nm/kg against the
                # -5.07 Nm/kg a real model-side channel would show. The slice
                # is unanswerable rather than merely hard, so it is excluded.
                if str(ep_meta.get("robot_type", "")).lower().startswith("kuka"):
                    continue
                assert ep_fault_id is not None
                rc_entry = root_causes.get(ep_fault_id, {})
                root_cause = rc_entry.get("root_cause", "")

                _correct = _optimization_answer(ep_fault_id, ep_meta)
                if not _correct:
                    continue
                answer = _correct


            important_features = template.get("important_features")
            context_subseries = subseries
            if important_features:
                keep = set(important_features) | {"timestamp_ms"}
                context_subseries = [
                    {k: v for k, v in row.items() if k in keep} for row in subseries
                ]
            context = build_context(context_subseries)

            item = {
                "id": str(uuid.uuid4()),
                "level": 4,
                "template_id": template["id"],
                "template_type": template["type"],
                "hides": template.get("hides", []),
                "question": template["template"],
                # Level 4 is free-form throughout. Both templates were briefly
                # rendered as lettered MCQs to make them deterministically
                # gradeable, which fixed grading by changing what the level
                # measures: "produce a diagnosis and a recovery procedure"
                # became "pick a letter", answerable at 1/6 by guessing. The
                # answer below is the reference protocol itself and the
                # three-judge rubric grades a free-text response against it.
                "options": {},
                "answer": answer,
                "root_cause": root_cause,
                "acceptance_bounds": None,
                "provenance": {
                    "dataset": sampled_dataset,
                    "episode": ep_path.stem,
                    "task": ep_task,
                    "subseries_start_index": start_idx,
                    "subseries_length": context_len,
                    "relevance": relevance_report(subseries, ep_fault_id or 0, spec, ep_task, sampler_tag),
                },
                "context": context,
            }

        out_path = output_dir / f"level4_{generated:04d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

        logger.info(
            f"✓ [{generated + 1}/{n}] {out_path.name} "
            f"(template {template['id']} '{template['type']}', {item.get('provenance', {}).get('dataset', '?')})"
        )
        generated += 1
        if uploader is not None:
            uploader.maybe_flush(generated)

    if uploader is not None:
        uploader.flush_remaining()

    if generated < n:
        logger.warning(f"Only generated {generated}/{n} questions after {attempts} attempts.")
    else:
        logger.info(f"Done: {generated} questions written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Level 4 (Open-Ended Reasoning) Q&A pairs."
    )
    repo_root = Path(__file__).resolve().parents[3]

    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=repo_root / "data",
        help="Root data directory (default: <repo>/data)")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "output" / "questions" / "level4",
        help="Output directory (default: <repo>/output/questions/level4)")
    parser.add_argument("-n", type=int, default=100, help="Number of questions to generate (cap; in --enumerate mode this is an upper bound, not a target)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--enumerate",
        dest="enumerate_mode",
        action="store_true",
        help="Walk every (template x episode) combination deterministically instead "
             "of random sampling. -n becomes an upper cap. Combinations whose "
             "episode does not satisfy the template's preconditions are skipped. "
             "For ranking templates (t3/t4) the primary episode walks; the other "
             "3 episodes per question are still sampled at random.")
    add_streaming_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to sample from (default: all). Choices: {VALID_DATASETS}")
    parser.add_argument(
        "--template-ids",
        type=int,
        nargs="+",
        default=None,
        help="Restrict generation to these template ids (default: all). Useful "
             "for regenerating a single template without rerunning the rest.")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    templates = load_templates(Path(__file__).with_name("question_template.json"))
    if getattr(args, "template_ids", None):
        wanted = set(args.template_ids)
        templates = [t for t in templates if t["id"] in wanted]
        if not templates:
            parser.error(f"No templates match --template-ids {sorted(wanted)}")

    root_causes = load_root_causes(args.datasets_dir / "labelling" / "rca" / "root_causes.json")
    events = load_events(args.datasets_dir / "labelling" / "events.json")
    # The file is root_cause_ur3_error_mapping.json, which is also the name
    # load_ur3_mapping documents. Pointing at the wrong one left ur3_mapping
    # None, and template 1 skips any anomalous item whose root cause has no
    # remediation protocol, so every faulty item was dropped and the whole
    # template collapsed to the canned "no anomalous behavior" answer.
    ur3_mapping_path = args.datasets_dir / "labelling" / "rca" / "root_cause_ur3_error_mapping.json"
    ur3_mapping = load_ur3_mapping(ur3_mapping_path) if ur3_mapping_path.exists() else None

    relevance_specs = (
        load_relevance_specs(args.datasets_dir / "labelling" / "rca" / "relevance_specs.json")
        if relevance_enabled() else {}
    )

    args.output.mkdir(parents=True, exist_ok=True)
    uploader = make_uploader_from_args(args, level=4, output_dir=args.output)

    generate_level4_questions(
        datasets_dir=args.datasets_dir,
        output_dir=args.output,
        templates=templates,
        root_causes=root_causes,
        events=events,
        n=args.n,
        seed=args.seed,
        relevance_specs=relevance_specs,
        datasets=args.datasets,
        ur3_mapping=ur3_mapping,
        enumerate_mode=args.enumerate_mode,
        uploader=uploader)


if __name__ == "__main__":
    main()
