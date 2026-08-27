"""LLM-as-judge for FactoryBench scoring.

Two distinct modes:

  * `score_free_form(...)`, the paper's §3.2 protocol for `free_form` answers:
    classify the model's answer as Wrong (0.0) / Neutral (0.5) / Correct (1.0)
    against a reference. Always invoked for free-form (no deterministic option).

  * `extract(...)`, escalation path for the deterministic formats. When the
    parser cascade returns `unparseable`, we ask the judge to extract the
    model's INTENDED answer in the canonical format (a letter, a TFTF string,
    a number, etc.). The caller then re-runs the parser on the extracted text
    so that the actual scoring stays deterministic.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .types import ParseResult

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_MAX_TOKENS = 256
# Cap the prediction sent to the judge to avoid blowing the input budget on
# multi-thousand-token chatty responses.
PREDICTION_MAX_CHARS = 8000


_FREE_FORM_PROMPT = """You are an expert evaluator for a robotics sensor-data Q&A benchmark.
Score a model's free-form answer against a reference answer.

Scoring (per the FactoryBench paper §3.2):
  1.0 (Correct):  semantically equivalent to the reference (correct direction, signal, magnitude, time window).
  0.5 (Neutral):  partially correct or imprecise but on-topic.
  0.0 (Wrong):    incorrect, irrelevant, or refuses to answer.

Respond with ONLY a JSON object on a single line, no markdown:
{{"score": <0.0|0.5|1.0>, "reason": "<one short sentence>"}}

Question:
{question}

Reference answer:
{reference}

Model answer:
{prediction}
"""


# Per-format extraction prompts. Each must instruct the judge to respond with
# ONLY the canonical answer string (or `NO_ANSWER`).
_EXTRACT_PROMPTS: dict[str, str] = {
    "multiple_choice_single_select": (
        "The model was asked a multiple-choice question with options A, B, C, D.\n"
        "Read the model's response below and identify its FINAL chosen letter "
        "(ignoring intermediate mentions of options used in reasoning).\n\n"
        "Respond with ONLY a single letter (A, B, C, or D), or `NO_ANSWER` "
        "if the model refused or gave no clear answer.\n\n"
        "Model response:\n---\n{prediction}\n---"
    ),
    "multiple_choice_multi_select": (
        "The model was asked a multiple-select question that requires a string "
        "of {n} characters where each is T (true) or F (false), in order.\n"
        "Read the response and reconstruct that {n}-character string.\n\n"
        "Respond with ONLY the {n}-character T/F string (e.g. `TFTF`), or "
        "`NO_ANSWER` if the model refused.\n\n"
        "Model response:\n---\n{prediction}\n---"
    ),
    "ranking": (
        "The model was asked to rank options as a permutation string of length "
        "{n} using the letters A-D (each appearing once).\n"
        "Identify the model's FINAL ranking string.\n\n"
        "Respond with ONLY the {n}-letter permutation (e.g. `BDCA`), or "
        "`NO_ANSWER` if the model refused.\n\n"
        "Model response:\n---\n{prediction}\n---"
    ),
    "numerical": (
        "The model was asked for a single numerical value.\n"
        "Identify its FINAL numerical answer.\n\n"
        "Respond with ONLY the number (e.g. `0.42` or `-1.5e-3`), or "
        "`NO_ANSWER` if the model refused.\n\n"
        "Model response:\n---\n{prediction}\n---"
    ),
    "tensor": (
        "The model was asked for a tensor of {n} numerical values, formatted "
        "with underscores between values (e.g. `0.5_-1.2_0.0`).\n"
        "Identify the model's FINAL tensor.\n\n"
        "Respond with ONLY the underscore-separated values, or `NO_ANSWER` "
        "if the model refused.\n\n"
        "Model response:\n---\n{prediction}\n---"
    ),
}


def _expected_length(answer_format: str, ground_truth: Any) -> int | None:
    if answer_format == "tensor":
        if isinstance(ground_truth, list):
            return len(ground_truth)
        gt = (str(ground_truth) if ground_truth is not None else "").strip()
        if not gt:
            return None
        if gt.startswith("["):
            return len(re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", gt))
        return len(gt.split("_"))
    gt = (str(ground_truth) if ground_truth is not None else "").strip()
    if not gt:
        return None
    if answer_format in ("multiple_choice_multi_select", "ranking"):
        return len(gt)
    return None


def _parse_json_verdict(raw: str) -> tuple[float | None, str]:
    """Parse {'score': float, 'reason': str} from a judge response. Tolerant
    of code fences and surrounding prose."""
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip leading ```json or ```
        text = text.split("```", 2)
        text = text[1] if len(text) >= 2 else ""
        text = text.removeprefix("json").strip()
        if "```" in text:
            text = text.split("```", 1)[0]
    # Try direct JSON.
    try:
        obj = json.loads(text)
        score = float(obj.get("score"))
        score = max(0.0, min(1.0, score))
        return score, str(obj.get("reason", ""))[:300]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fall back: scan for a 0/0.5/1 token.
    m = re.search(r"\b(1(?:\.0+)?|0\.5|0(?:\.0+)?)\b", text)
    if m:
        return float(m.group(1)), text[:300]
    return None, text[:300]


def _extract_canonical(raw: str) -> str | None:
    """Strip whitespace, code fences, and surrounding markdown. Return the
    first non-empty line, or None if the judge said NO_ANSWER."""
    text = (raw or "").strip()
    text = text.strip("`").strip()
    if "NO_ANSWER" in text.upper():
        return None
    # Take first non-empty line and strip simple emphasis.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    first_line = first_line.strip("*` ").strip()
    return first_line or None


class LLMJudge:
    """Single-model LLM judge. Defer client construction to first use so
    importing this module is cheap and tests don't accidentally hit the API.
    """

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._call = None

    def _ensure_call(self) -> None:
        if self._call is None:
            from src.evaluation.run_foundry_eval import call_model
            self._call = call_model

    # ---- Free-form scoring (always uses judge) ----
    def score_free_form(
        self, *, question: str | None, prediction: str, reference: Any) -> ParseResult:
        self._ensure_call()
        prompt = _FREE_FORM_PROMPT.format(
            question=(question or "(question text not available in trace)")[:6000],
            reference=str(reference)[:2000],
            prediction=str(prediction)[:PREDICTION_MAX_CHARS])
        try:
            raw, _body = self._call(self.model, prompt, self.max_tokens)
        except Exception as e:
            logger.warning("Judge call failed (free_form): %s", e)
            return ParseResult(None, None, "unparseable", f"judge call failed: {e}")

        score, reason = _parse_json_verdict(raw)
        if score is None:
            return ParseResult(None, None, "unparseable",
                               f"judge response unparseable: {raw[:200]!r}")
        return ParseResult(score=score, parsed=prediction, provenance="judge",
                           reason=f"judge: {reason}")

    # ---- Canonical-answer extraction (escalation for deterministic formats) ----
    def extract(
        self, *, answer_format: str, prediction: str, ground_truth: Any) -> tuple[str | None, str]:
        """Returns (extracted_canonical_answer, judge_reason).

        `extracted` is None when the format is unsupported or the judge said
        NO_ANSWER. The caller is responsible for re-running the matching
        parser on the extracted string.
        """
        self._ensure_call()
        tpl = _EXTRACT_PROMPTS.get(answer_format)
        if tpl is None:
            return None, f"no extract prompt for format {answer_format!r}"

        n = _expected_length(answer_format, ground_truth)
        prompt = tpl.format(prediction=str(prediction)[:PREDICTION_MAX_CHARS], n=n)
        try:
            raw, _body = self._call(self.model, prompt, self.max_tokens)
        except Exception as e:
            logger.warning("Judge call failed (extract %s): %s", answer_format, e)
            return None, f"judge call failed: {e}"

        extracted = _extract_canonical(raw)
        if extracted is None:
            return None, "judge: NO_ANSWER"
        return extracted, f"judge extracted: {extracted!r}"
