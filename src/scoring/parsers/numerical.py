"""Parser for `numerical` answer format.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar

from ..types import ParseResult
from .base import Parser
from .multiple_choice import _last_match, _strip_prompt_leak

# Default tolerance when no acceptance_bounds are provided.
ABSOLUTE_TOL = 1e-4

# Generic float pattern (handles scientific notation).
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Trailing unit suffixes commonly emitted by the models. The match is anchored
# at end-of-string so it only strips when the unit is the final token.
_UNIT_SUFFIX_RE = re.compile(
    r"\s*(?:"
    r"rad/s|rad|m/s\^?2|m/s|mm/s|cm/s|mm|cm|km|m|"
    r"ms|us|ns|s|min|h|"
    r"mA|kA|A|"
    r"Nm|N|"
    r"°C|°F|°|deg(?:rees?)?|"
    r"%|"
    r"kPa|MPa|Pa|"
    r"mV|kV|V|"
    r"mW|kW|W|"
    r"kJ|J|"
    r"kHz|MHz|Hz"
    r")\.?\s*$",
    re.IGNORECASE)

# Markdown emphasis around a value: **0.5**, *0.5*, `0.5`
_EMPHASIS_RE = re.compile(r"^\s*[\*`]+\s*(.*?)\s*[\*`]+\s*$", re.DOTALL)

# Cue patterns. Use re.DOTALL for tolerance to multiline answers.
_ANSWER_CUE_RE = re.compile(
    r"(?:final\s+)?answer\s*[:=]\s*\*{0,2}\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE)


def _strip_emphasis_and_unit(s: str) -> str:
    """Strip surrounding markdown emphasis and a trailing unit if present."""
    s = s.strip()
    if (m := _EMPHASIS_RE.match(s)) is not None:
        s = m.group(1).strip()
    s = _UNIT_SUFFIX_RE.sub("", s).strip()
    s = s.rstrip(".").strip()
    return s


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class NumericalParser(Parser):
    """numerical, extracts a single float."""

    answer_format: ClassVar[str] = "numerical"

    def parse(self, text: str, ground_truth: Any, **ctx: Any) -> ParseResult:
        try:
            gt = float(ground_truth)
        except (TypeError, ValueError):
            return ParseResult(None, None, "unparseable", f"invalid ground_truth: {ground_truth!r}")

        raw = text or ""
        if not raw.strip():
            return ParseResult(None, None, "unparseable", "empty model response")

        cleaned = _strip_prompt_leak(raw).strip()

        # ---- Strict: whole text is a float ----
        if (val := _try_float(cleaned)) is not None:
            return self._score(val, gt, ctx, "strict")

        # ---- Lenient: strip emphasis / unit / trailing period ----
        stripped = _strip_emphasis_and_unit(cleaned)
        if (val := _try_float(stripped)) is not None:
            return self._score(val, gt, ctx, "lenient")

        # ---- Lenient: explicit Answer: <num> cue (last match) ----
        if (m := _last_match(_ANSWER_CUE_RE, cleaned)) is not None:
            if (val := _try_float(m.group(1))) is not None:
                return self._score(val, gt, ctx, "lenient")

        # ---- Lenient: last float in the LAST non-empty line only ----
        # Deliberately do NOT scan the whole text: when the model refuses
        # ("I cannot determine..."), reasoning often contains sensor values
        # that would be picked up as false positives. Forcing the number to
        # appear on the final line keeps the parser conservative; chatty
        # answers without an explicit cue go to the judge instead.
        last_line = next(
            (ln for ln in reversed(cleaned.splitlines()) if ln.strip()),
            "")
        last_line_floats = _FLOAT_RE.findall(_strip_emphasis_and_unit(last_line))
        if last_line_floats:
            if (val := _try_float(last_line_floats[-1])) is not None:
                return self._score(val, gt, ctx, "lenient")

        return ParseResult(None, None, "unparseable", "no numeric value on last line")

    @staticmethod
    def _score(val: float, gt: float, ctx: dict[str, Any], provenance) -> ParseResult:
        bounds = ctx.get("acceptance_bounds") if ctx else None
        if bounds and "min" in bounds and "max" in bounds:
            score = float(float(bounds["min"]) <= val <= float(bounds["max"]))
        else:
            margin = (bounds or {}).get("margin")
            if margin is not None:
                score = float(abs(val - gt) <= margin)
            else:
                score = float(abs(val - gt) <= ABSOLUTE_TOL)
        return ParseResult(score=score, parsed=val, provenance=provenance)
