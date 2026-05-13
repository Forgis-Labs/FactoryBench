"""Per-item scoring and chance rates for FactoryBench answer formats."""
from __future__ import annotations

import json
import math
from typing import Any

from .types import AnswerFormat, Item

_SCALAR_EXACT_REL_TOL = 1e-2  # used when no margin is provided


def score_item(item: Item, parsed: Any) -> float:
    """Return a score in [0, 1] comparing ``parsed`` to ``item.answer``.

    For element-wise formats (4-letter TF, tensors), this returns the fraction
    of correctly-predicted elements. For all-or-nothing formats it returns
    0.0 or 1.0.
    """
    fmt = item.answer_format

    if fmt == AnswerFormat.SINGLE_LETTER_MCQ:
        return 1.0 if str(parsed).upper() == str(item.answer).upper() else 0.0

    if fmt == AnswerFormat.FOUR_LETTER_TF:
        truth = str(item.answer).upper()
        pred = str(parsed).upper()
        if len(truth) != 4 or len(pred) != 4:
            return 0.0
        return sum(a == b for a, b in zip(truth, pred)) / 4.0

    if fmt == AnswerFormat.FOUR_LETTER_RANKING:
        return 1.0 if str(parsed).upper() == str(item.answer).upper() else 0.0

    if fmt == AnswerFormat.SCALAR_RANGE:
        bounds = item.acceptance_bounds or {}
        lo, hi = float(bounds["min"]), float(bounds["max"])
        return 1.0 if lo <= float(parsed) <= hi else 0.0

    if fmt == AnswerFormat.SCALAR_MARGIN:
        bounds = item.acceptance_bounds or {}
        margin = float(bounds["margin"])
        truth = float(item.answer)
        return 1.0 if abs(float(parsed) - truth) <= margin else 0.0

    if fmt == AnswerFormat.SCALAR_EXACT:
        bounds = item.acceptance_bounds or {}
        truth = float(bounds.get("actual_value", item.answer))
        return 1.0 if math.isclose(float(parsed), truth, rel_tol=_SCALAR_EXACT_REL_TOL) else 0.0

    if fmt == AnswerFormat.TENSOR_MARGIN:
        bounds = item.acceptance_bounds or {}
        margins = bounds.get("margin")
        truth = item.answer
        if isinstance(truth, str):
            truth = json.loads(truth)
        if not isinstance(parsed, list) or not isinstance(truth, list):
            return 0.0
        if len(parsed) != len(truth):
            return 0.0
        if isinstance(margins, list) and len(margins) == len(truth):
            return sum(abs(p - t) <= m for p, t, m in zip(parsed, truth, margins)) / len(truth)
        # Scalar margin applied uniformly.
        m_val = float(margins) if isinstance(margins, (int, float)) else 0.0
        return sum(abs(p - t) <= m_val for p, t in zip(parsed, truth)) / len(truth)

    # free_form: not scored deterministically.
    return float("nan")


def chance_of(item: Item) -> float:
    """Random-baseline accuracy for the item's answer format."""
    fmt = item.answer_format
    if fmt == AnswerFormat.SINGLE_LETTER_MCQ:
        n = len(item.options or {})
        return 1.0 / n if n else 0.0
    if fmt == AnswerFormat.FOUR_LETTER_TF:
        return 0.5  # per-element TF chance
    if fmt == AnswerFormat.FOUR_LETTER_RANKING:
        return 1.0 / 24.0  # 4! permutations
    # Continuous / range formats: chance ~ 0.
    return 0.0


def chance_corrected(mean_acc: float, mean_chance: float) -> float:
    """Cohen's-kappa style chance correction."""
    denom = 1.0 - mean_chance
    if denom <= 0:
        return 0.0
    return (mean_acc - mean_chance) / denom
