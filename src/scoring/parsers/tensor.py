"""Parser for `tensor` answer format.

Ground truth is a string of floats separated by underscores, e.g.
`-94.7_0.94_0.0` (3 components) or
`-0.27988_-0.343254_0.081161_2.072647_2.357377_0.01592` (6 components).

Also accepts list-style ground truth (`[-94.7, 0.94, 0.0]`) for compatibility
with the question generators that emit JSON arrays.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar

from ..types import ParseResult
from .base import Parser
from .multiple_choice import _last_match, _strip_prompt_leak
from .numerical import _strip_emphasis_and_unit, _try_float


# A tensor "chunk" candidate: at least two `_`-separated segments where each
# segment contains a number (possibly preceded by a variable-name prefix
# like "aat0="). The regex is greedy on length to capture maximal tensors.
_TENSOR_CHUNK_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*=)?-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:_(?:[A-Za-z_][A-Za-z0-9_]*=)?-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)+")

# JSON-array-shaped tensor: `[1, 2.5, 3]`
_TENSOR_BRACKET_RE = re.compile(r"\[\s*[^\[\]]*?-?\d[^\[\]]*?\]")
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_ANSWER_CUE_RE = re.compile(r"(?:final\s+)?answer\s*[:=]\s*", re.IGNORECASE)


def _parse_tensor_chunk(chunk: str) -> list[float] | None:
    """Split a candidate string by `_`, strip optional `name=` prefix per part,
    and cast each to float. Returns None on any failure."""
    parts = chunk.split("_")
    out: list[float] = []
    for p in parts:
        # Strip any leading variable-name prefix: "aat0=-9.65" -> "-9.65".
        after_eq = p.rsplit("=", 1)[-1]
        val = _try_float(after_eq)
        if val is None:
            return None
        out.append(val)
    return out


def _parse_ground_truth_tensor(gt: Any) -> list[float] | None:
    """Accept either an `_`-joined string or a JSON list of floats."""
    if isinstance(gt, list):
        out: list[float] = []
        for x in gt:
            v = _try_float(str(x))
            if v is None:
                return None
            out.append(v)
        return out or None
    s = (str(gt) if gt is not None else "").strip()
    if not s:
        return None
    if s.startswith("["):
        nums = _FLOAT_RE.findall(s)
        return [float(x) for x in nums] if nums else None
    return _parse_tensor_chunk(s)


def _extract_bracket_tensor(text: str, n: int) -> list[float] | None:
    """Pick the LAST `[...]` block whose number count matches `n`, or fall back
    to the LAST `n` floats anywhere in the text."""
    for seg in reversed(_TENSOR_BRACKET_RE.findall(text)):
        nums = _FLOAT_RE.findall(seg)
        if len(nums) == n:
            return [float(x) for x in nums]
    return None


class TensorParser(Parser):
    """tensor, extracts a list of floats separated by `_` or in `[...]` form."""

    answer_format: ClassVar[str] = "tensor"

    def parse(self, text: str, ground_truth: Any, **ctx: Any) -> ParseResult:
        gt_vals = _parse_ground_truth_tensor(ground_truth)
        if gt_vals is None or not gt_vals:
            return ParseResult(None, None, "unparseable", f"invalid ground_truth: {ground_truth!r}")
        n = len(gt_vals)

        raw = text or ""
        if not raw.strip():
            return ParseResult(None, None, "unparseable", "empty model response")

        cleaned = _strip_prompt_leak(raw).strip()

        # ---- Strict: whole text is a clean `_`-separated tensor of length n ----
        strict_vals = _parse_tensor_chunk(cleaned)
        if strict_vals is not None and len(strict_vals) == n:
            return self._score(strict_vals, gt_vals, ctx, "strict")

        # ---- Strict alt: whole text is a `[...]` array of length n ----
        if cleaned.startswith("[") and cleaned.endswith("]"):
            nums = _FLOAT_RE.findall(cleaned)
            if len(nums) == n:
                return self._score([float(x) for x in nums], gt_vals, ctx, "strict")

        # ---- Lenient: strip surrounding emphasis / unit, retry strict ----
        stripped = _strip_emphasis_and_unit(cleaned)
        stripped_vals = _parse_tensor_chunk(stripped)
        if stripped_vals is not None and len(stripped_vals) == n:
            return self._score(stripped_vals, gt_vals, ctx, "lenient")

        # ---- Lenient: explicit Answer cue followed by tensor ----
        cue = _last_match(_ANSWER_CUE_RE, cleaned)
        if cue is not None:
            after_cue = cleaned[cue.end():]
            after_cue_first_line = after_cue.splitlines()[0] if after_cue else ""
            after_cue_stripped = _strip_emphasis_and_unit(after_cue_first_line)
            cue_vals = _parse_tensor_chunk(after_cue_stripped)
            if cue_vals is not None and len(cue_vals) == n:
                return self._score(cue_vals, gt_vals, ctx, "lenient")
            # Also try a `[...]` block on the cue line.
            bracket_vals = _extract_bracket_tensor(after_cue_first_line, n)
            if bracket_vals is not None:
                return self._score(bracket_vals, gt_vals, ctx, "lenient")

        # ---- Lenient: last `[...]` array of length n anywhere in the text ----
        bracket_vals = _extract_bracket_tensor(cleaned, n)
        if bracket_vals is not None:
            return self._score(bracket_vals, gt_vals, ctx, "lenient")

        # ---- Lenient: last tensor-shaped chunk in the LAST non-empty line ----
        last_line = next(
            (ln for ln in reversed(cleaned.splitlines()) if ln.strip()),
            "")
        last_line_stripped = _strip_emphasis_and_unit(last_line)
        chunks = _TENSOR_CHUNK_RE.findall(last_line_stripped)
        for chunk in reversed(chunks):
            vals = _parse_tensor_chunk(chunk)
            if vals is not None and len(vals) == n:
                return self._score(vals, gt_vals, ctx, "lenient")

        return ParseResult(None, None, "unparseable", f"no tensor of length {n} found")

    @staticmethod
    def _score(parsed: list[float], gt: list[float], ctx: dict[str, Any], provenance) -> ParseResult:
        bounds = ctx.get("acceptance_bounds") if ctx else None
        margins = (bounds or {}).get("margin") if bounds else None
        n = len(gt)
        if margins and isinstance(margins, list) and len(margins) == n:
            n_correct = sum(abs(p - g) <= m for p, g, m in zip(parsed, gt, margins))
            score = n_correct / n
        else:
            # No bounds: element-wise equality (matches existing inline scoring).
            score = 1.0 if parsed == gt else 0.0
        return ParseResult(score=score, parsed=parsed, provenance=provenance)
