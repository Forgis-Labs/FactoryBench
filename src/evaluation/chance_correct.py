"""Chance-correction for FactoryBench raw scores.

Maps a raw item score $s \\in [0,1]$ to a chance-corrected score
$\\tilde{s} = \\max(0, (s - E) / (1 - E))$, where $E$ is the expected
score under random guessing for that item's answer format.

After correction, $\\tilde{s} = 0$ corresponds to pure-chance performance
and $\\tilde{s} = 1$ to perfect performance for every format, so per-level
and cross-format averages are directly comparable.

Per-format chance levels:

  * single-select MCQ:  E = 1/k where k is the option count for the item
                        (the released set uses k=3 for L1 t6 / L2 t10 and
                        k=4 elsewhere).
  * multi-select MCQ:   E = 1/2  (each T/F slot matches gold w.p. 1/2 under
                        uniform guessing; expected fraction = 1/2 regardless
                        of length).
  * ranking:            E = 1/n where n is the permutation length. The
                        expected number of fixed points of a uniform random
                        permutation is exactly 1, so E = 1/n. Released set
                        uses n=4.
  * tensor:             E = 1/4 by construction; raw scoring uses the
                        three-level piecewise scorer (1 within m, 0.5 within
                        2m, 0 otherwise) with per-channel margin calibrated
                        to m_j = R_j/12, which yields E = 3m/R = 1/4. So
                        the same correction as single-select MCQ applies.
  * free-form:          NOT chance-corrected. The Level-4 rubric is already
                        on a {0, 0.5, 1} scale by construction and a model
                        emitting random text essentially never lands a
                        non-zero rubric score, so E = 0 and the correction
                        is a no-op. Free-form scores are returned unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _option_count(question: Optional[Dict[str, Any]]) -> int:
    """Return the number of options for a single-select item, defaulting to 4."""
    if not isinstance(question, dict):
        return 4
    opts = question.get("options")
    if isinstance(opts, dict) and opts:
        return len(opts)
    if isinstance(opts, (list, tuple)) and opts:
        return len(opts)
    return 4


def _permutation_length(question: Optional[Dict[str, Any]]) -> int:
    """Return the gold permutation length for a ranking item, defaulting to 4."""
    if isinstance(question, dict):
        ans = question.get("answer")
        if isinstance(ans, str) and ans:
            return len(ans.strip())
    return 4


def expected_chance_score(
    answer_format: str,
    question: Optional[Dict[str, Any]] = None,
) -> float:
    """Return E for the given format/item. Free-form returns 0 (no-op)."""
    if answer_format == "multiple_choice_single_select":
        k = _option_count(question)
        return 1.0 / max(1, k)
    if answer_format == "multiple_choice_multi_select":
        return 0.5
    if answer_format == "ranking":
        n = _permutation_length(question)
        return 1.0 / max(1, n)
    if answer_format == "tensor":
        # Calibrated per-channel margin m_j = R_j/12 yields E = 3m/R = 1/4
        # under the three-level piecewise scorer.
        return 0.25
    if answer_format == "numerical":
        # Single-scalar tensor branch; same calibration target.
        return 0.25
    if answer_format == "free_form":
        return 0.0
    return 0.0


def chance_correct(
    raw_score: Optional[float],
    answer_format: str,
    question: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Map a raw item score to its chance-corrected counterpart.

    Returns None if ``raw_score`` is None (ungraded item). Returns the raw
    score unchanged for ``free_form`` (which has E = 0 by construction).
    """
    if raw_score is None:
        return None
    if answer_format == "free_form":
        return float(raw_score)
    E = expected_chance_score(answer_format, question)
    if E >= 1.0 or math.isnan(E):
        # Item is uninformative (chance saturates to 1); treat any score >= 1 as 1.
        return 1.0 if float(raw_score) >= 1.0 else 0.0
    s = float(raw_score)
    corrected = (s - E) / (1.0 - E)
    if corrected < 0.0:
        return 0.0
    if corrected > 1.0:
        return 1.0
    return corrected
