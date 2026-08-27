"""Orchestrator: dispatch a (prediction, ground_truth, answer_format) triple
to the right parser, and escalate to the LLM judge when the deterministic
cascade fails.

Policy:

  free_form              -> always judge (paper §3.2 free-form protocol)
  every other format     -> deterministic parser first; only escalate the
                            `unparseable` cases to the judge.

The judge ESCALATION for deterministic formats is "extract & re-parse": the
judge returns the model's intended answer in the canonical format, and we
re-run the matching parser on that string. This keeps scoring deterministic
end-to-end while only paying for the LLM call when we can't read the answer
ourselves.
"""
from __future__ import annotations

from typing import Any, Optional

from .judge import LLMJudge
from .parsers import (
    MCMultiParser,
    MCSingleParser,
    NumericalParser,
    Parser,
    RankingParser,
    TensorParser)
from .types import ParseResult


_PARSER_REGISTRY: dict[str, Parser] = {
    "multiple_choice_single_select": MCSingleParser(),
    "multiple_choice_multi_select": MCMultiParser(),
    "ranking": RankingParser(),
    "numerical": NumericalParser(),
    "tensor": TensorParser(),
}


def parse_only(
    *,
    answer_format: str,
    prediction: str,
    ground_truth: Any,
    acceptance_bounds: Optional[dict[str, Any]] = None) -> ParseResult:
    """Run the deterministic parser cascade only, never call a judge.

    Use this when free-form is handled separately (e.g. by an existing judge
    pipeline) and you only want the strict/lenient/unparseable verdict for
    deterministic formats.
    """
    parser = _PARSER_REGISTRY.get(answer_format)
    if parser is None:
        return ParseResult(None, None, "unparseable",
                           f"no parser for answer_format={answer_format!r}")
    ctx: dict[str, Any] = {}
    if acceptance_bounds:
        ctx["acceptance_bounds"] = acceptance_bounds
    return parser.parse(prediction, ground_truth, **ctx)


def score(
    *,
    answer_format: str,
    prediction: str,
    ground_truth: Any,
    question: Optional[str] = None,
    acceptance_bounds: Optional[dict[str, Any]] = None,
    judge: Optional[LLMJudge] = None) -> ParseResult:
    """Score a single prediction. Returns a ParseResult.

    `judge` is optional: when omitted, free-form trivially returns
    `unparseable` and deterministic-format failures stay unparseable.
    """
    # 1) Free-form: judge-only path.
    if answer_format == "free_form":
        if judge is None:
            return ParseResult(None, None, "unparseable",
                               "free_form requires LLM judge")
        return judge.score_free_form(
            question=question, prediction=prediction, reference=ground_truth)

    # 2) Deterministic formats.
    result = parse_only(
        answer_format=answer_format,
        prediction=prediction,
        ground_truth=ground_truth,
        acceptance_bounds=acceptance_bounds)
    if result.provenance != "unparseable":
        return result

    # 3) Escalate to judge if available.
    if judge is None:
        return result

    extracted, judge_reason = judge.extract(
        answer_format=answer_format,
        prediction=prediction,
        ground_truth=ground_truth)
    if extracted is None:
        # Judge declined, record verdict as "wrong" with judge provenance,
        # since the model effectively didn't answer.
        return ParseResult(score=0.0, parsed=None, provenance="judge",
                           reason=judge_reason)

    retry = parse_only(
        answer_format=answer_format,
        prediction=extracted,
        ground_truth=ground_truth,
        acceptance_bounds=acceptance_bounds)
    if retry.provenance == "unparseable":
        # Judge gave us something but parser still can't read it. Treat as
        # wrong with judge provenance to preserve the audit trail.
        return ParseResult(
            score=0.0, parsed=extracted, provenance="judge",
            reason=f"{judge_reason}; parser could not read judge output: {retry.reason}")
    return ParseResult(
        score=retry.score,
        parsed=retry.parsed,
        provenance="judge",
        reason=judge_reason)
