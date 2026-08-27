"""Three-judge ensemble with median-of-3 voting.

The judges are gpt-5.1-1 (foundry batch), claude-sonnet-4.6 (bedrock batch),
and deepseek-v3.2 (bedrock batch). Each is submitted as its own batch job;
``score_all()`` joins the three result sets by ``custom_id`` and applies
median voting.

Voting policy (recap):
  * Each judge returns a verdict in {0.0, 0.5, 1.0}.
  * 3 valid verdicts  → median (== "2-out-of-3 majority, with the middle
                        value when all disagree"). Always in-codomain.
  * 2 valid verdicts  → median (== mean for an even count, but with our
                        codomain {0, 0.5, 1} the mean of any two valid
                        values is itself in {0, 0.25, 0.5, 0.75, 1}; we
                        snap to the nearest half-step).
  * 1 valid verdict   → that verdict, flagged in reason.
  * 0 valid verdicts  → score=None, provenance=unparseable.
"""
from __future__ import annotations

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .batch import JudgeRequest, submit_judge_batch
from .templates import build_prompt, parse_verdict

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODELS: Tuple[str, ...] = (
    "gpt-5.1-1",
    "claude-sonnet-4.6",
    "deepseek-v3.2",
)


@dataclass
class JudgeItem:
    """One free-form item to score."""
    custom_id: str
    template_id: int
    question: str
    prediction: str
    reference: str
    root_cause: Optional[str] = None


@dataclass
class JudgeVote:
    score: Optional[float]
    reason: str


@dataclass
class EnsembleResult:
    custom_id: str
    final_score: Optional[float]
    reason: str
    votes: Dict[str, JudgeVote] = field(default_factory=dict)


def _snap_to_half(x: float) -> float:
    """Round to nearest half-step in [0, 1]."""
    snapped = round(x * 2) / 2
    return max(0.0, min(1.0, snapped))


def _aggregate(votes: Dict[str, JudgeVote]) -> Tuple[Optional[float], str]:
    """Median of the valid (non-None) votes, snapped to {0, 0.5, 1}."""
    valid = [(m, v) for m, v in votes.items() if v.score is not None]
    if not valid:
        return None, "all judges failed"
    scores = [v.score for _m, v in valid]
    if len(scores) == 1:
        m, v = valid[0]
        return v.score, f"only {m} returned a score: {v.reason}"
    med = statistics.median(scores)
    final = _snap_to_half(med)
    parts = [f"{m}={v.score}" for m, v in valid]
    return final, f"median({', '.join(parts)}) = {final}"


def _run_one_judge(
    model: str,
    items: List[JudgeItem],
    max_output_tokens: int,
    poll_interval: int,
    sync_concurrency: int = 0,
) -> Dict[str, JudgeVote]:
    """Submit one batch for ``model``, parse verdicts, return per-id votes."""
    requests_ = [
        JudgeRequest(
            custom_id=it.custom_id,
            prompt=build_prompt(
                template_id=it.template_id,
                question=it.question,
                prediction=it.prediction,
                reference=it.reference,
                root_cause=it.root_cause,
            ),
        )
        for it in items
    ]
    raw_by_id = submit_judge_batch(
        model=model,
        requests_=requests_,
        max_output_tokens=max_output_tokens,
        poll_interval=poll_interval,
        sync_concurrency=sync_concurrency,
    )
    out: Dict[str, JudgeVote] = {}
    for it in items:
        raw = raw_by_id.get(it.custom_id)
        if not raw:
            out[it.custom_id] = JudgeVote(score=None, reason="no response from judge")
            continue
        score, reason = parse_verdict(raw)
        out[it.custom_id] = JudgeVote(score=score, reason=reason)
    return out


def score_all(
    items: List[JudgeItem],
    *,
    judges: Tuple[str, ...] = DEFAULT_JUDGE_MODELS,
    max_output_tokens: int = 256,
    poll_interval: int = 30,
    parallel_judges: bool = True,
    sync_concurrency: int = 0,
) -> Dict[str, EnsembleResult]:
    """Score every item with each judge in ``judges`` (one batch per judge),
    then median-vote across judges. Returns ``{custom_id: EnsembleResult}``.
    """
    if not items:
        return {}

    per_judge: Dict[str, Dict[str, JudgeVote]] = {}

    if parallel_judges:
        with ThreadPoolExecutor(max_workers=len(judges)) as ex:
            futures = {
                ex.submit(
                    _run_one_judge, model, items, max_output_tokens, poll_interval, sync_concurrency,
                ): model
                for model in judges
            }
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    per_judge[model] = fut.result()
                except Exception as exc:
                    logger.error(f"[multi-judge] {model}: batch failed: {exc}")
                    per_judge[model] = {
                        it.custom_id: JudgeVote(score=None, reason=f"batch failed: {exc}")
                        for it in items
                    }
    else:
        for model in judges:
            try:
                per_judge[model] = _run_one_judge(
                    model, items, max_output_tokens, poll_interval, sync_concurrency,
                )
            except Exception as exc:
                logger.error(f"[multi-judge] {model}: batch failed: {exc}")
                per_judge[model] = {
                    it.custom_id: JudgeVote(score=None, reason=f"batch failed: {exc}")
                    for it in items
                }

    results: Dict[str, EnsembleResult] = {}
    for it in items:
        votes = {model: per_judge[model].get(
            it.custom_id, JudgeVote(score=None, reason="missing from batch")
        ) for model in judges}
        final, reason = _aggregate(votes)
        results[it.custom_id] = EnsembleResult(
            custom_id=it.custom_id,
            final_score=final,
            reason=reason,
            votes=votes,
        )
    return results
