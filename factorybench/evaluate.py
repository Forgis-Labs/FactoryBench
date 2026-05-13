"""Top-level ``evaluate`` entrypoint."""
from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Iterable

from tqdm import tqdm

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
    """
    panel = _resolve_judges(judges)
    levels = _resolve_levels_strict(level, have_judges=panel is not None)
    instance = get_model(model)

    items = load_split(level=levels, split=split, revision=revision, max_items=max_items)
    prompts = [render_prompt(it) for it in items]

    resolved_name = model_name or (model if isinstance(model, str) else type(instance).__name__)

    t0 = time.perf_counter()
    raw_outputs = _run_predict(instance, prompts, progress=progress)
    wall = timedelta(seconds=time.perf_counter() - t0)

    item_results = [_score_one(it, raw, panel=panel, progress=progress) for it, raw in zip(items, raw_outputs)]

    return Result(
        model_name=resolved_name,
        items=item_results,
        wall_time=wall,
        judges=list(panel.judge_specs) if panel else [],
    )


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


def _run_predict(model: Any, prompts: list[str], *, progress: bool) -> list[str]:
    batch_fn = getattr(model, "predict_batch", None)
    if callable(batch_fn):
        outputs = batch_fn(prompts)
        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            raise RuntimeError("predict_batch must return a list with one entry per prompt")
        return [str(x) if x is not None else "" for x in outputs]

    iterator = tqdm(prompts, desc="evaluate", disable=not progress)
    return [_safe_str(model.predict(p)) for p in iterator]


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _score_one(item: Item, raw: str, *, panel: JudgePanel | None = None, progress: bool = True) -> ItemResult:
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
            # If every judge failed to parse, surface that as a parse_error.
            if not l4.valid_votes:
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
