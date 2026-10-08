"""Loader for the FactoryBench test split from the Hugging Face Hub."""
from __future__ import annotations

import json
import warnings
from typing import Iterable

from huggingface_hub import hf_hub_download

from .types import AnswerFormat, Item

REPO_ID = "Forgis/FactoryBench"
REPO_TYPE = "dataset"
VALID_LEVELS = (1, 2, 3, 4)
# The dataset is one flat file per level. The train/validation/test
# partition was retired upstream, so these names are accepted only for
# backward compatibility and all resolve to the full level.
RETIRED_SPLITS = ("train", "validation", "test")
VALID_SPLITS = RETIRED_SPLITS
# Number of items to take from each level for the deterministic mini split.
MINI_PER_LEVEL = 50


def _infer_answer_format(row: dict) -> AnswerFormat:
    options = row.get("options") or {}
    answer = row.get("answer")
    bounds = row.get("acceptance_bounds") or {}

    if options:
        if isinstance(answer, str):
            if len(answer) == 1:
                return AnswerFormat.SINGLE_LETTER_MCQ
            if len(answer) == 4:
                chars = set(answer.upper())
                if chars <= {"T", "F"}:
                    return AnswerFormat.FOUR_LETTER_TF
                if chars <= {"A", "B", "C", "D"}:
                    return AnswerFormat.FOUR_LETTER_RANKING
        # Unknown MCQ-like shape — treat as free-form.
        return AnswerFormat.FREE_FORM

    if not bounds:
        return AnswerFormat.FREE_FORM

    margin = bounds.get("margin")
    if isinstance(margin, list):
        return AnswerFormat.TENSOR_MARGIN
    if isinstance(margin, (int, float)):
        return AnswerFormat.SCALAR_MARGIN
    if "min" in bounds and "max" in bounds:
        return AnswerFormat.SCALAR_RANGE
    if "actual_value" in bounds:
        return AnswerFormat.SCALAR_EXACT
    return AnswerFormat.FREE_FORM


def _row_to_item(row: dict) -> Item:
    return Item(
        id=row["id"],
        level=int(row["level"]),
        template_id=int(row["template_id"]),
        template_type=row.get("template_type", ""),
        question=row["question"],
        options=row.get("options") or {},
        answer=row.get("answer"),
        acceptance_bounds=row.get("acceptance_bounds"),
        provenance=row.get("provenance") or {},
        context=row.get("context") or {},
        hides=row.get("hides") or [],
        root_cause=row.get("root_cause"),
        answer_format=_infer_answer_format(row),
    )


def _download_level(level: int, revision: str | None) -> str:
    """Download one level's JSONL. One flat file per level; there are no splits."""
    if level not in VALID_LEVELS:
        raise ValueError(f"level must be one of {VALID_LEVELS}, got {level!r}")
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=f"factorybench_qa/level_{level}.jsonl",
        repo_type=REPO_TYPE,
        revision=revision,
    )


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_split(
    level: int | str | Iterable[int | str],
    split: str = "test",
    revision: str | None = None,
    max_items: int | None = None,
) -> list[Item]:
    """Load items from the FactoryBench Hub repo.

    Args:
        level: A level number (1..4), the string ``"L1".."L4"``, ``"all"``,
            or an iterable of those.
        split: ``"mini"`` for the first ``MINI_PER_LEVEL`` items per level, or
            ``None`` for the whole level. ``"train"``/``"validation"``/``"test"``
            are accepted for backward compatibility and return the whole level
            with a ``DeprecationWarning``: the dataset no longer ships a
            train/validation/test partition.
        revision: Optional git revision / tag of the dataset repo.
        max_items: Optional per-level cap.
    """
    levels = _resolve_levels(level)

    if split == "mini":
        per_level_cap = MINI_PER_LEVEL if max_items is None else min(MINI_PER_LEVEL, max_items)
    else:
        if split in RETIRED_SPLITS:
            # Do not do this quietly: a caller asking for "test" and silently
            # receiving the whole level would get a surprise bill on a paid API.
            warnings.warn(
                f"split={split!r} is retired: FactoryBench no longer ships a "
                "train/validation/test partition, so the full level is returned. "
                "Pass max_items= to cap it, or split='mini' for a small sample.",
                DeprecationWarning, stacklevel=2)
        elif split is not None:
            raise ValueError(
                f"split must be 'mini', None, or one of {RETIRED_SPLITS}, got {split!r}")
        per_level_cap = max_items

    items: list[Item] = []
    for lvl in levels:
        path = _download_level(lvl, revision)
        for i, row in enumerate(_read_jsonl(path)):
            if per_level_cap is not None and i >= per_level_cap:
                break
            items.append(_row_to_item(row))
    return items


def _resolve_levels(level: int | str | Iterable[int | str]) -> list[int]:
    if isinstance(level, int):
        return [level]
    if isinstance(level, str):
        s = level.strip().lower()
        if s == "all":
            return list(VALID_LEVELS)
        if s.startswith("l") and s[1:].isdigit():
            return [int(s[1:])]
        if s.isdigit():
            return [int(s)]
        raise ValueError(f"Could not parse level={level!r}")
    # Iterable
    out: list[int] = []
    for v in level:
        out.extend(_resolve_levels(v))
    # De-dup, keep order.
    seen: set[int] = set()
    deduped = []
    for lvl in out:
        if lvl not in seen:
            seen.add(lvl)
            deduped.append(lvl)
    return deduped
