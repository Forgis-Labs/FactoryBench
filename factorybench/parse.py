"""Parse raw model outputs into typed values according to the item's answer format."""
from __future__ import annotations

import json
import re
from typing import Any

from .types import AnswerFormat, Item

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_FOUR_TF_RE = re.compile(r"[TF]{4}", re.IGNORECASE)
_FOUR_RANK_RE = re.compile(r"[ABCD]{4}", re.IGNORECASE)


class ParseError(ValueError):
    """Raised when a model output cannot be parsed into the expected format."""


def parse_output(raw: str, item: Item) -> Any:
    """Parse the model's raw string output for ``item``.

    Returns a typed value (``str``, ``float``, ``list[float]``) suitable for
    scoring, or raises :class:`ParseError` if the output cannot be parsed.
    """
    if raw is None:
        raise ParseError("model returned None")
    fmt = item.answer_format
    text = raw.strip()

    if fmt == AnswerFormat.SINGLE_LETTER_MCQ:
        return _parse_single_letter(text, item)
    if fmt == AnswerFormat.FOUR_LETTER_TF:
        m = _FOUR_TF_RE.search(text)
        if not m:
            raise ParseError(f"expected 4 T/F chars, got {text!r}")
        return m.group(0).upper()
    if fmt == AnswerFormat.FOUR_LETTER_RANKING:
        m = _FOUR_RANK_RE.search(text)
        if not m:
            raise ParseError(f"expected 4 chars from A-D, got {text!r}")
        seq = m.group(0).upper()
        if len(set(seq)) != 4:
            raise ParseError(f"ranking must be a permutation of A,B,C,D, got {seq}")
        return seq
    if fmt in (AnswerFormat.SCALAR_RANGE, AnswerFormat.SCALAR_MARGIN, AnswerFormat.SCALAR_EXACT):
        return _parse_scalar(text)
    if fmt == AnswerFormat.TENSOR_MARGIN:
        return _parse_tensor(text)
    if fmt == AnswerFormat.FREE_FORM:
        return text
    raise ParseError(f"unsupported answer_format {fmt!r}")


def _parse_single_letter(text: str, item: Item) -> str:
    valid = set(k.upper() for k in (item.options or {}).keys())
    m = _LETTER_RE.search(text)
    if m:
        letter = m.group(1).upper()
        if not valid or letter in valid:
            return letter
    # Fall back to first valid letter anywhere in the string.
    for ch in text.upper():
        if ch in valid:
            return ch
    raise ParseError(f"could not extract a letter from {text!r}")


def _parse_scalar(text: str) -> float:
    # Try the whole string first.
    try:
        return float(text)
    except ValueError:
        pass
    m = _NUMBER_RE.search(text)
    if not m:
        raise ParseError(f"could not extract a number from {text!r}")
    return float(m.group(0))


def _parse_tensor(text: str) -> list[float]:
    # Prefer a JSON-array decode.
    try:
        value = json.loads(text)
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            return [float(x) for x in value]
    except json.JSONDecodeError:
        pass
    nums = _NUMBER_RE.findall(text)
    if not nums:
        raise ParseError(f"could not extract numbers from {text!r}")
    return [float(x) for x in nums]
