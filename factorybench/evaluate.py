"""Top-level ``evaluate`` entrypoint."""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .cost import compute_cost_from_usage
from .data import load_split
from .judges import JudgePanel, parse_judges_flag, precheck_credentials
from .parse import ParseError, parse_output
from .prompt import render_prompt
from .registry import get_model
from .result import ItemResult, Result
from .score import chance_of, score_item
from .types import AnswerFormat, Item

L4_REQUIRES_JUDGES_MSG = (
    "Level 4 evaluation requires an LLM-as-judge ensemble. "
    "Pass judges=JudgePanel([...]) (Python) or --judges paper-default | <comma-list> (CLI)."
)


def evaluate(
    model: str | Any,
    level: int | str | Iterable[int | str] = "all",
    split: str = "test",
    revision: str | None = None,
    max_items: int | None = None,
    progress: bool = True,
    model_name: str | None = None,
    judges: JudgePanel | str | None = None,
    concurrency: int = 1,
    resume_from: str | Path | None = None,
    judge_cache_only: bool = False,
) -> Result:
    """Run a model over the FactoryBench test split and return a :class:`Result`.

    Args:
        model: Registered model name, built-in provider spec, or an object with
            ``predict(prompt: str) -> str``. ``predict_batch`` is used when defined.
        level: ``"L1"|"L2"|"L3"|"L4"|"all"`` (or numeric / iterable equivalent).
            ``"L4"`` requires ``judges``. ``"all"`` includes L4 only when ``judges``
            is provided -- otherwise it silently runs L1+L2+L3.
        split: ``"test"`` (default), ``"validation"``, ``"train"``, or ``"mini"``.
        revision: Optional dataset repo revision.
        max_items: Optional per-level item cap.
        progress: Show a tqdm progress bar.
        model_name: Override the name reported in the result. Defaults to the
            spec string or class name.
        judges: A :class:`factorybench.judges.JudgePanel`, or the CLI shorthand
            string (``"paper-default"`` or a comma-separated list). Required to
            score Level 4 items.
        concurrency: Number of parallel candidate-model calls (ThreadPoolExecutor).
            Defaults to ``1`` (sequential). Ignored when the model defines
            ``predict_batch``. Judges are still called sequentially within an item.
        resume_from: Path to a previously-saved Result JSON. Items whose ``id`` is
            present in that file AND were scored without a parse error are reused
            verbatim; everything else is re-run.
        judge_cache_only: If True, the judge panel refuses to make new API calls
            and returns only cached votes. Items missing a cached vote are reported
            with ``parse_error="judge_cache_miss"``. Lets you re-aggregate against a
            saved cache without re-billing.
    """
    panel = _resolve_judges(judges)
    if panel is not None and judge_cache_only:
        panel.cache_only = True
    levels = _resolve_levels_strict(level, have_judges=panel is not None)
    instance = get_model(model)

    items = load_split(level=levels, split=split, revision=revision, max_items=max_items)
    resolved_name = model_name or (model if isinstance(model, str) else type(instance).__name__)

    resumed = _load_resume(resume_from) if resume_from else {}
    to_run_indices: list[int] = []
    reused: dict[int, ItemResult] = {}
    for i, it in enumerate(items):
        prior = resumed.get(it.id)
        if prior is not None and _is_resumable(prior):
            reused[i] = prior
        else:
            to_run_indices.append(i)

    if progress and resumed:
        print(f"resume: reusing {len(reused)} item(s) from prior run; running {len(to_run_indices)}")

    prompts = [render_prompt(items[i]) for i in to_run_indices]

    t0 = time.perf_counter()
    raw_outputs, candidate_usages = _run_predict(instance, prompts, progress=progress, concurrency=concurrency)
    wall = timedelta(seconds=time.perf_counter() - t0)

    fresh_results: dict[int, ItemResult] = {}
    for i, raw in zip(to_run_indices, raw_outputs):
        fresh_results[i] = _score_one(items[i], raw, panel=panel)

    item_results = [reused.get(i) or fresh_results[i] for i in range(len(items))]

    # Roll up actual token usage + cost.
    candidate_model_id = _adapter_model_id(instance, fallback=resolved_name)
    tokens_used = _summarize_usage(
        candidate_model_id=candidate_model_id,
        candidate_usages=candidate_usages,
        panel=panel,
    )
    cost = compute_cost_from_usage(tokens_used) if tokens_used else 0.0

    return Result(
        model_name=resolved_name,
        items=item_results,
        wall_time=wall,
        judges=list(panel.judge_specs) if panel else [],
        tokens_used=tokens_used,
        cost=cost,
    )


def _adapter_model_id(instance: Any, *, fallback: str) -> str:
    """The price-table key for this adapter (e.g. ``gpt-5.1``, ``claude-opus-4-7``)."""
    return getattr(instance, "model_name", None) or fallback


def _summarize_usage(
    *,
    candidate_model_id: str,
    candidate_usages: list[dict | None],
    panel: JudgePanel | None,
) -> dict:
    """Aggregate per-call usage into a {candidate, judges} structure for Result.tokens_used."""
    cand_in = sum(int(u["input_tokens"]) for u in candidate_usages if u)
    cand_out = sum(int(u["output_tokens"]) for u in candidate_usages if u)
    summary = {
        "candidate": {
            "model": candidate_model_id,
            "input_tokens": cand_in,
            "output_tokens": cand_out,
            "calls": sum(1 for u in candidate_usages if u),
        },
        "judges": {},
    }
    if panel is not None:
        for spec, ledger in panel.usage_by_judge().items():
            summary["judges"][spec] = {
                # Underlying provider model (used for price-table lookup), not the spec name.
                "model": ledger.get("model") or spec,
                "input_tokens": ledger["input_tokens"],
                "output_tokens": ledger["output_tokens"],
                "calls": ledger["calls"],
            }
    return summary


def _resolve_judges(judges: JudgePanel | str | None) -> JudgePanel | None:
    if judges is None:
        return None
    if isinstance(judges, JudgePanel):
        return judges
    if isinstance(judges, str):
        return parse_judges_flag(judges)
    raise TypeError(f"judges must be JudgePanel, str, or None; got {type(judges).__name__}")


def _resolve_levels_strict(level: int | str | Iterable[int | str], *, have_judges: bool) -> list[int]:
    """Apply the L4 gating rules and return the concrete level list."""
    if isinstance(level, str) and level.strip().lower() == "all":
        return [1, 2, 3, 4] if have_judges else [1, 2, 3]

    from .data import _resolve_levels

    levels = _resolve_levels(level)
    if 4 in levels and not have_judges:
        raise ValueError(L4_REQUIRES_JUDGES_MSG)
    return levels


# --------------------------------------------------------------------------- #
# Prediction (sequential / concurrent / batched)
# --------------------------------------------------------------------------- #

def _run_predict(
    model: Any,
    prompts: list[str],
    *,
    progress: bool,
    concurrency: int = 1,
) -> tuple[list[str], list[dict | None]]:
    """Run the candidate model over ``prompts``.

    Returns ``(predictions, usages)``. ``usages[i]`` is the per-call usage dict
    (``{input_tokens, output_tokens, model}``) when the adapter implements
    ``predict_with_usage``, else ``None``. Order matches ``prompts``.
    """
    if not prompts:
        return [], []

    batch_fn = getattr(model, "predict_batch", None)
    if callable(batch_fn):
        outputs = batch_fn(prompts)
        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            raise RuntimeError("predict_batch must return a list with one entry per prompt")
        return [_safe_str(x) for x in outputs], [None] * len(prompts)

    call = _make_call(model)

    if concurrency <= 1:
        results: list[str] = []
        usages: list[dict | None] = []
        for p in tqdm(prompts, desc="evaluate", disable=not progress):
            text, usage = call(p)
            results.append(text)
            usages.append(usage)
        return results, usages

    return _run_predict_threaded(call, prompts, max_workers=concurrency, progress=progress)


def _make_call(model: Any):
    """Return a function ``(prompt) -> (text, usage_or_None)`` for this adapter."""
    fn = getattr(model, "predict_with_usage", None)
    if callable(fn):
        def call(prompt: str):
            text, usage = fn(prompt)
            return _safe_str(text), usage
        return call

    def call(prompt: str):
        return _safe_str(model.predict(prompt)), None
    return call


def _run_predict_threaded(
    call,
    prompts: list[str],
    *,
    max_workers: int,
    progress: bool,
) -> tuple[list[str], list[dict | None]]:
    """Order-preserving parallel ``predict`` via a thread pool."""
    results: list[str] = [""] * len(prompts)
    usages: list[dict | None] = [None] * len(prompts)
    bar = tqdm(total=len(prompts), desc=f"evaluate (x{max_workers})", disable=not progress)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(call, p): i for i, p in enumerate(prompts)}
            for fut in futures:
                idx = futures[fut]
                try:
                    text, usage = fut.result()
                except Exception as exc:
                    # Surface as an empty prediction; the scorer will record a
                    # parse error against this item rather than aborting the run.
                    text, usage = "", None
                    if progress:
                        bar.write(f"item {idx}: {type(exc).__name__}: {exc}")
                results[idx] = text
                usages[idx] = usage
                bar.update(1)
    finally:
        bar.close()
    return results, usages


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #

def _load_resume(path: str | Path) -> dict[str, ItemResult]:
    """Load a Result JSON saved by ``Result.save`` and index by item id."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, ItemResult] = {}
    for row in raw.get("items") or []:
        out[row["id"]] = ItemResult(
            id=row["id"],
            level=row["level"],
            template_id=row["template_id"],
            template_type=row["template_type"],
            answer_format=row["answer_format"],
            dataset=row.get("dataset"),
            fault_id=row.get("fault_id"),
            raw_output=row.get("raw_output"),
            parsed=row.get("parsed"),
            score=float(row.get("score")) if row.get("score") is not None else float("nan"),
            chance=float(row.get("chance", 0.0)),
            parse_error=row.get("parse_error"),
            judge_votes=row.get("judge_votes"),
        )
    return out


def _is_resumable(r: ItemResult) -> bool:
    """An item is resumable iff it was scored cleanly the previous run."""
    if r.parse_error is not None:
        return False
    if isinstance(r.score, float) and math.isnan(r.score):
        return False
    return True


# --------------------------------------------------------------------------- #
# Per-item scoring
# --------------------------------------------------------------------------- #

def _score_one(item: Item, raw: str, *, panel: JudgePanel | None = None) -> ItemResult:
    """Score one item. L1-L3 use the deterministic scorer; L4 uses ``panel``."""
    parsed: Any = None
    score = float("nan")
    chance = chance_of(item)
    parse_error: str | None = None
    judge_votes: list[dict] | None = None

    if item.answer_format == AnswerFormat.FREE_FORM:
        if panel is not None:
            l4 = panel.score(item, raw)
            score = l4.score
            judge_votes = [
                {"judge": v.judge, "score": v.score, "raw": v.raw, "parse_error": v.parse_error, "cached": v.cached}
                for v in l4.votes
            ]
            if not l4.valid_votes:
                # Distinguish cache-only misses from real model failures.
                if panel.cache_only and all(v.parse_error == "judge_cache_miss" for v in l4.votes):
                    parse_error = "judge_cache_miss"
                else:
                    parse_error = "all judges failed to return a valid 0/0.5/1 score"
        return ItemResult(
            id=item.id,
            level=item.level,
            template_id=item.template_id,
            template_type=item.template_type,
            answer_format=item.answer_format.value,
            dataset=item.dataset,
            fault_id=item.fault_id,
            raw_output=raw,
            parsed=None,
            score=score,
            chance=chance,
            parse_error=parse_error,
            judge_votes=judge_votes,
        )

    try:
        parsed = parse_output(raw, item)
    except ParseError as exc:
        parse_error = str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        parse_error = f"{type(exc).__name__}: {exc}"

    if parse_error is None:
        try:
            score = float(score_item(item, parsed))
        except Exception as exc:
            parse_error = f"scoring_error: {type(exc).__name__}: {exc}"
            score = float("nan")

    return ItemResult(
        id=item.id,
        level=item.level,
        template_id=item.template_id,
        template_type=item.template_type,
        answer_format=item.answer_format.value,
        dataset=item.dataset,
        fault_id=item.fault_id,
        raw_output=raw,
        parsed=parsed,
        score=score,
        chance=chance,
        parse_error=parse_error,
    )
