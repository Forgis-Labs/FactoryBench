"""Canned-response adapter for testing the pipeline without API calls."""
from __future__ import annotations

import re

_OPTION_LETTER_RE = re.compile(r"\b([A-D])\b")


class MockAdapter:
    """Returns a deterministic, format-aware placeholder string.

    Useful for ``factorybench evaluate --model mock`` smoke tests:

    * MCQ-looking prompts return ``"A"``.
    * 4-letter TF prompts return ``"TFTF"``.
    * 4-letter ranking prompts return ``"ABCD"``.
    * JSON-array prompts return ``"[0,0,0,0,0,0]"``.
    * Numeric prompts return ``"0"``.
    """

    def predict(self, prompt: str) -> str:
        low = prompt.lower()
        if "ranking (ie. dcab)" in low or "four letter string indicating your ranking" in low:
            return "ABCD"
        if "tfft" in low or "4 letter string using f and t" in low:
            return "TFTF"
        if "json array" in low or "list of 6 numbers" in low:
            return "[0,0,0,0,0,0]"
        if "letter of the correct option" in low or _OPTION_LETTER_RE.search(prompt):
            # Only fall through to A if it looks MCQ-ish.
            if "options:" in low or "answer only with the letter" in low:
                return "A"
        return "0"
