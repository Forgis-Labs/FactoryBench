"""Parser for `ranking` answer format.

Ground truth is a permutation string of the option letters (typically 4 letters
A-D, e.g. "BDCA").

Per-sample score is **Kendall's tau-a normalized to [0, 1]**:
  - exact match           -> tau = +1 -> score = 1.0
  - random ordering       -> tau ~  0 -> score ~ 0.5
  - fully reversed order  -> tau = -1 -> score = 0.0
  - invalid (not a permutation of gt's letters) -> score = 0.0

The aggregate "exact-match rate" of the paper is recoverable from the per-sample
scores as `(score == 1.0).mean()` over the dataset.

The cascade prefers explicit cues (Answer:, bold) before falling back to
"last bare [A-D]{N} substring".
"""
from __future__ import annotations

import re
from typing import Any, ClassVar

from ..types import ParseResult
from .base import Parser
from .multiple_choice import _last_match, _strip_prompt_leak


class RankingParser(Parser):
    """ranking, extracts a permutation string of length len(ground_truth)."""

    answer_format: ClassVar[str] = "ranking"

    def parse(self, text: str, ground_truth: Any, **ctx: Any) -> ParseResult:
        gt = (str(ground_truth) if ground_truth is not None else "").strip().upper()
        if not gt or not re.fullmatch(r"[A-D]+", gt):
            return ParseResult(None, None, "unparseable", f"invalid ground_truth: {gt!r}")
        n = len(gt)

        raw = text or ""
        if not raw.strip():
            return ParseResult(None, None, "unparseable", "empty model response")

        cleaned = _strip_prompt_leak(raw).strip()

        # ---- Strict: text is exactly the permutation ----
        if re.fullmatch(rf"[A-Da-d]{{{n}}}", cleaned):
            return self._score(cleaned.upper(), gt, "strict")

        # ---- Lenient: Answer: BDCA cue ----
        cue_pat = re.compile(
            rf"(?:final\s+)?answer\s*[:=]\s*\*{{0,2}}\s*([A-Da-d]{{{n}}})\b",
            re.IGNORECASE)
        if (m := _last_match(cue_pat, cleaned)) is not None:
            return self._score(m.group(1).upper(), gt, "lenient")

        # ---- Lenient: bold permutation ----
        bold_pat = re.compile(rf"\*{{1,2}}\s*([A-Da-d]{{{n}}})\s*\*{{1,2}}")
        if (m := _last_match(bold_pat, cleaned)) is not None:
            return self._score(m.group(1).upper(), gt, "lenient")

        # ---- Lenient: last [A-D]{n} substring (final answer wins) ----
        bare_pat = re.compile(rf"\b([A-Da-d]{{{n}}})\b")
        if (m := _last_match(bare_pat, cleaned)) is not None:
            return self._score(m.group(1).upper(), gt, "lenient")

        return ParseResult(None, None, "unparseable", f"no [A-D]{{{n}}} substring found")

    @staticmethod
    def _score(parsed: str, gt: str, provenance) -> ParseResult:
        # Predictions that aren't valid permutations of gt's letters fail
        # closed: the model didn't even respect the answer set.
        if len(parsed) != len(gt) or set(parsed) != set(gt):
            return ParseResult(
                score=0.0, parsed=parsed, provenance=provenance,
                reason="not a valid permutation of ground-truth letters")
        tau = _kendall_tau(parsed, gt)
        score = (tau + 1.0) / 2.0
        return ParseResult(score=score, parsed=parsed, provenance=provenance)


def _kendall_tau(pred: str, gt: str) -> float:
    """Kendall's tau-a rank correlation between two permutations of the same items.

    For each of the C(n, 2) item pairs, count whether their relative order
    matches between `pred` and `gt`:
        tau = (n_concordant - n_discordant) / C(n, 2)

    Returns a value in [-1, +1]: +1 for exact match, -1 for fully reversed.
    Caller is responsible for ensuring `pred` is a valid permutation of `gt`.
    """
    n = len(gt)
    if n < 2:
        return 1.0
    rank_pred = {item: i for i, item in enumerate(pred)}
    concordant = discordant = 0
    # Iterate items in gt order so rank_gt[items[i]] < rank_gt[items[j]] holds
    # by construction; we only need to check sign in pred.
    for i in range(n):
        for j in range(i + 1, n):
            if rank_pred[gt[i]] < rank_pred[gt[j]]:
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) // 2
    return (concordant - discordant) / total
