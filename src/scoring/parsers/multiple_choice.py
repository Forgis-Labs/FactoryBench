"""Parsers for multiple-choice answer formats.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar

from ..types import ParseResult
from .base import Parser


_PROMPT_LEAK_RE = re.compile(r"\n\s*\+{3,}\s*prompts?/", re.IGNORECASE)


def _strip_prompt_leak(text: str) -> str:
    m = _PROMPT_LEAK_RE.search(text)
    return text[: m.start()] if m else text


# Strict: the entire response is just a letter (with optional whitespace).
_STRICT_RE = re.compile(r"^\s*([A-Da-d])\s*$")

# Lenient layer 1, explicit "Answer: X" cues, with optional markdown emphasis.
# Captures `Answer: B`, `**Answer: B**`, `Final Answer: B`, `Answer = B`, etc.
_ANSWER_CUE_RE = re.compile(
    r"(?:\*{1,2}\s*)?(?:final\s+)?answer\s*[:=]\s*\*{0,2}\s*([A-Da-d])\b",
    re.IGNORECASE)

# Lenient layer 2, "the correct/right answer is X" phrasing.
_IS_CUE_RE = re.compile(
    r"(?:correct|right)\s+answer\s+is\s*:?\s*\*{0,2}\s*([A-Da-d])\b",
    re.IGNORECASE)

# Lenient layer 3, standalone bolded letter, e.g. `**B**`.
_BOLD_LETTER_RE = re.compile(r"\*{1,2}\s*([A-Da-d])\s*\*{1,2}")

# Lenient layer 4, last non-empty line is just the letter
# (optionally bolded, optionally trailing period). Multiline.
_LINE_LETTER_RE = re.compile(
    r"^\s*\*{0,2}\s*([A-Da-d])\s*\*{0,2}\s*\.?\s*$",
    re.MULTILINE)

# Lenient layer 5 (last resort), bare [A-D] anywhere, with light unit-context filter.
_BARE_LETTER_RE = re.compile(r"\b([A-Da-d])\b")
# A letter is "unit-like" if it follows a number (e.g., "0.38 A", "5 N").
_UNIT_CONTEXT_RE = re.compile(r"\d\s*$")


def _last_match(pattern: re.Pattern[str], text: str) -> re.Match[str] | None:
    last = None
    for m in pattern.finditer(text):
        last = m
    return last


class MCMultiParser(Parser):
    """multiple_choice_multi_select, extracts a T/F string of fixed length."""

    answer_format: ClassVar[str] = "multiple_choice_multi_select"

    def parse(self, text: str, ground_truth: Any, **ctx: Any) -> ParseResult:
        gt = (str(ground_truth) if ground_truth is not None else "").strip().upper()
        if not gt or set(gt) - {"T", "F"}:
            return ParseResult(None, None, "unparseable", f"invalid ground_truth: {gt!r}")
        n = len(gt)

        raw = text or ""
        if not raw.strip():
            return ParseResult(None, None, "unparseable", "empty model response")

        cleaned = _strip_prompt_leak(raw).strip()

        # ---- Strict ----
        # example: "TFFT"
        if re.fullmatch(r"[TFtf]+", cleaned) and len(cleaned) == n:
            return self._score(cleaned.upper(), gt, "strict")

        # ---- Lenient: explicit Answer cue with TF string ----
        # example: "Answer: TFFT", "**Answer: TFFT**", "Final Answer = TFFT", etc.
        cue_pat = re.compile(
            rf"(?:final\s+)?answer\s*[:=]\s*\*{{0,2}}\s*([TFtf]{{{n}}})\b",
            re.IGNORECASE)
        m = _last_match(cue_pat, cleaned)
        if m is not None:
            return self._score(m.group(1).upper(), gt, "lenient")

        # ---- Lenient: bold TF string ----
        # example: "**TFFT**"
        bold_pat = re.compile(rf"\*{{1,2}}\s*([TFtf]{{{n}}})\s*\*{{1,2}}")
        m = _last_match(bold_pat, cleaned)
        if m is not None:
            return self._score(m.group(1).upper(), gt, "lenient")

        # ---- Lenient: first [TF]{n} substring (Mistral pattern) ----
        # Use FIRST not last because Mistral writes the answer up-front and
        # then explains; a later [TF]{n} would more likely be a hallucination.
        # example: "TFFT ... explanation ... TFFT"
        first = re.search(rf"\b([TFtf]{{{n}}})\b", cleaned)
        if first is not None:
            return self._score(first.group(1).upper(), gt, "lenient")

        return ParseResult(
            None, None, "unparseable",
            f"no [TF]{{{n}}} substring found"
        )

    @staticmethod
    def _score(parsed: str, gt: str, provenance) -> ParseResult:
        n = len(gt)
        n_correct = sum(p == g for p, g in zip(parsed, gt))
        if n_correct == n:
            score = 1.0
        elif n_correct >= n - 1:
            score = 0.5
        else:
            score = 0.0
        return ParseResult(score=score, parsed=parsed, provenance=provenance)


class MCSingleParser(Parser):
    """multiple_choice_single_select, extracts a single letter A-D."""

    answer_format: ClassVar[str] = "multiple_choice_single_select"

    def parse(self, text: str, ground_truth: Any, **ctx: Any) -> ParseResult:
        gt = (str(ground_truth) if ground_truth is not None else "").strip().upper()
        raw = text or ""

        if not raw.strip():
            return ParseResult(None, None, "unparseable", "empty model response")

        cleaned = _strip_prompt_leak(raw).strip()

        # ---- Strict ----
        if (m := _STRICT_RE.match(cleaned)) is not None:
            return self._score(m.group(1).upper(), gt, "strict")

        # ---- Lenient cascade ----
        # Take the LAST match in each layer: models typically end with the answer.
        for pat in (_ANSWER_CUE_RE, _IS_CUE_RE, _BOLD_LETTER_RE, _LINE_LETTER_RE):
            if (m := _last_match(pat, cleaned)) is not None:
                return self._score(m.group(1).upper(), gt, "lenient")

        # Last resort: last bare letter, skipping unit-like contexts.
        bare_candidates = [
            m for m in _BARE_LETTER_RE.finditer(cleaned)
            if not _UNIT_CONTEXT_RE.search(cleaned[max(0, m.start() - 5):m.start()])
        ]
        if bare_candidates:
            return self._score(bare_candidates[-1].group(1).upper(), gt, "lenient")

        return ParseResult(None, None, "unparseable", "no A-D letter found")

    @staticmethod
    def _score(parsed: str, gt: str, provenance) -> ParseResult:
        score = 1.0 if parsed == gt else 0.0
        return ParseResult(score=score, parsed=parsed, provenance=provenance)
