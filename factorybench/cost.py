"""Cost preview for ``factorybench evaluate``.

Token counts use the ``len(text) / 4`` heuristic (+/-25%). Per-model prices
are a static snapshot table; override at runtime with :func:`set_price`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Iterable

from .data import load_split
from .judges import PAPER_DEFAULT_JUDGES, JudgePanel, RUBRIC_PROMPT, parse_judges_flag
from .prompt import render_prompt
from .types import AnswerFormat, Item


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #

_HEURISTIC_CHARS_PER_TOKEN = 4

# Output token estimates per item, by answer format.
_OUTPUT_TOKEN_ESTIMATE = {
    AnswerFormat.SINGLE_LETTER_MCQ:   2,
    AnswerFormat.FOUR_LETTER_TF:      4,
    AnswerFormat.FOUR_LETTER_RANKING: 4,
    AnswerFormat.SCALAR_RANGE:       10,
    AnswerFormat.SCALAR_MARGIN:      10,
    AnswerFormat.SCALAR_EXACT:       10,
    AnswerFormat.TENSOR_MARGIN:      40,
    AnswerFormat.FREE_FORM:         150,
}

# Each judge call outputs exactly one of "0", "0.5", "1".
_JUDGE_OUTPUT_TOKENS = 3


def estimate_tokens(text: str | None, model: str | None = None) -> int:
    """Heuristic token count: ``len(text) / 4``. Model argument is ignored."""
    if not text:
        return 0
    return max(1, len(text) // _HEURISTIC_CHARS_PER_TOKEN)


def estimate_output_tokens(item: Item) -> int:
    return _OUTPUT_TOKEN_ESTIMATE.get(item.answer_format, 50)


# --------------------------------------------------------------------------- #
# Price table ($/1M tokens). Static snapshot -- override via set_price.
# --------------------------------------------------------------------------- #

# Each entry: (input $/M tokens, output $/M tokens).
PRICES_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o":               (2.50,  10.00),
    "gpt-4o-mini":          (0.15,   0.60),
    "gpt-4.1":              (2.00,   8.00),
    "gpt-4.1-mini":         (0.40,   1.60),
    "gpt-4.1-nano":         (0.10,   0.40),
    "gpt-5":                (5.00,  40.00),
    "gpt-5.1":              (5.00,  40.00),
    # Anthropic
    "claude-opus-4":        (15.00,  75.00),
    "claude-opus-4-7":      (15.00,  75.00),
    "claude-sonnet-4":       (3.00,  15.00),
    "claude-sonnet-4-6":     (3.00,  15.00),
    "claude-sonnet-4.6":     (3.00,  15.00),
    "claude-haiku-4":        (0.80,   4.00),
    "claude-haiku-4-5":      (0.80,   4.00),
    # DeepSeek
    "deepseek-chat":         (0.27,   1.10),
    "deepseek-v3":           (0.27,   1.10),
    "deepseek-v3.2":         (0.27,   1.10),
}

DEFAULT_PRICE_PER_M_TOKENS: tuple[float, float] = (5.00, 15.00)


def set_price(model: str, *, input_per_m: float, output_per_m: float) -> None:
    """Override the price-per-1M-tokens for a given model string."""
    PRICES_PER_M_TOKENS[model] = (float(input_per_m), float(output_per_m))


def _price_of(model: str) -> tuple[float, float]:
    """Look up (input_per_m, output_per_m) for a model with prefix fallback."""
    if model in PRICES_PER_M_TOKENS:
        return PRICES_PER_M_TOKENS[model]
    norm = model.replace(".", "-")
    if norm in PRICES_PER_M_TOKENS:
        return PRICES_PER_M_TOKENS[norm]
    return DEFAULT_PRICE_PER_M_TOKENS


def _is_priced(model: str) -> bool:
    if model in PRICES_PER_M_TOKENS:
        return True
    return model.replace(".", "-") in PRICES_PER_M_TOKENS


# --------------------------------------------------------------------------- #
# Time estimation (very rough wall-clock baseline per call type)
# --------------------------------------------------------------------------- #

_SECONDS_PER_CANDIDATE_CALL = 3.0
_SECONDS_PER_JUDGE_CALL = 4.0


# --------------------------------------------------------------------------- #
# CostEstimate dataclass + the estimator
# --------------------------------------------------------------------------- #

@dataclass
class JudgeCostBreakdown:
    judge: str
    input_tokens: int
    output_tokens: int
    cost: float
    priced: bool  # False if we fell back to DEFAULT_PRICE_PER_M_TOKENS


@dataclass
class CostEstimate:
    """Order-of-magnitude cost preview for a planned evaluate() call."""

    n_items: int
    n_l4_items: int
    model: str
    model_input_tokens: int
    model_output_tokens: int
    model_cost: float
    model_priced: bool
    judges: list[JudgeCostBreakdown] = field(default_factory=list)
    judge_cost: float = 0.0
    total_cost: float = 0.0
    estimated_wall_time: timedelta = field(default_factory=lambda: timedelta(0))
    notes: list[str] = field(default_factory=list)
    # Always False in this version; set to True when tiktoken is wired in.
    precise_tokens: bool = False

    def format(self) -> str:
        """Return a human-readable preview string."""
        precision_tag = "heuristic tokens (+/-25%)"
        lines = [
            f"This will evaluate {self.model} on {self.n_items} item(s).",
            f"Estimated cost ({precision_tag}):",
            f"  Model calls ({self.model:<22s}): {_dollar(self.model_cost):>10s}"
            + ("" if self.model_priced else "  [no price table entry; default rate used]"),
        ]
        if self.judges:
            lines.append(f"  Judge calls ({len(self.judges)} judge(s)):             {_dollar(self.judge_cost):>10s}")
            for j in self.judges:
                marker = "" if j.priced else "  [default rate]"
                lines.append(f"    {j.judge:<24s}: {_dollar(j.cost):>10s}{marker}")
        lines.append(f"  Total:                              {_dollar(self.total_cost):>10s}")
        lines.append(f"Estimated wall time (sequential): {self.estimated_wall_time}")
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_l4_items": self.n_l4_items,
            "model": self.model,
            "model_input_tokens": self.model_input_tokens,
            "model_output_tokens": self.model_output_tokens,
            "model_cost": round(self.model_cost, 4),
            "model_priced": self.model_priced,
            "precise_tokens": self.precise_tokens,
            "judges": [
                {
                    "judge": j.judge,
                    "input_tokens": j.input_tokens,
                    "output_tokens": j.output_tokens,
                    "cost": round(j.cost, 4),
                    "priced": j.priced,
                }
                for j in self.judges
            ],
            "judge_cost": round(self.judge_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "estimated_wall_time_seconds": self.estimated_wall_time.total_seconds(),
            "notes": list(self.notes),
        }


def estimate_cost(
    model: str,
    level: str | int | Iterable = "all",
    split: str = "test",
    revision: str | None = None,
    max_items: int | None = None,
    judges: JudgePanel | str | list[str] | None = None,
) -> CostEstimate:
    """Estimate dollar cost + wall time of an :func:`evaluate` call."""
    panel = _coerce_panel(judges)
    have_judges = panel is not None

    if isinstance(level, str) and level.strip().lower() == "all":
        levels = [1, 2, 3, 4] if have_judges else [1, 2, 3]
    else:
        from .data import _resolve_levels
        levels = _resolve_levels(level)

    items = load_split(level=levels, split=split, revision=revision, max_items=max_items)
    return _estimate_for_items(items, model=model, panel=panel)


def _estimate_for_items(items: list[Item], *, model: str, panel: JudgePanel | None) -> CostEstimate:
    notes: list[str] = []
    notes.append(
        "token counts are heuristic; install tiktoken for exact counts "
        "(pip install \"factorybench[tokenizers]\")"
    )

    in_tokens = 0
    out_tokens = 0
    n_l4 = 0
    for it in items:
        prompt = render_prompt(it)
        in_tokens += estimate_tokens(prompt, model=model)
        out_tokens += estimate_output_tokens(it)
        if it.answer_format == AnswerFormat.FREE_FORM:
            n_l4 += 1

    model_priced = _is_priced(model)
    if not model_priced:
        notes.append(
            f"no price entry for {model!r}; using default "
            f"{DEFAULT_PRICE_PER_M_TOKENS[0]}/{DEFAULT_PRICE_PER_M_TOKENS[1]} $/M tokens"
        )
    model_cost = _dollars(in_tokens, out_tokens, _price_of(model))

    judges_breakdown: list[JudgeCostBreakdown] = []
    judge_total = 0.0
    if panel is not None and n_l4 > 0:
        for spec in panel.judge_specs:
            per_judge_in = 0
            per_judge_out = 0
            for it in items:
                if it.answer_format != AnswerFormat.FREE_FORM:
                    continue
                stand_in = str(it.answer or "")
                rendered = RUBRIC_PROMPT.format(
                    question=it.question.strip(),
                    reference=stand_in,
                    root_cause_block=(
                        f"\nREFERENCE ROOT CAUSE: {(it.root_cause or '').strip()}\n"
                        if (it.root_cause or "").strip() else ""
                    ),
                    prediction=stand_in,
                )
                per_judge_in += estimate_tokens(rendered, model=spec)
                per_judge_out += _JUDGE_OUTPUT_TOKENS

            priced = _is_priced(spec)
            cost = _dollars(per_judge_in, per_judge_out, _price_of(spec))
            judges_breakdown.append(JudgeCostBreakdown(
                judge=spec,
                input_tokens=per_judge_in,
                output_tokens=per_judge_out,
                cost=cost,
                priced=priced,
            ))
            judge_total += cost
            if not priced:
                notes.append(f"no price entry for judge {spec!r}; using default rate")

    candidate_seconds = len(items) * _SECONDS_PER_CANDIDATE_CALL
    judge_seconds = n_l4 * len(judges_breakdown) * _SECONDS_PER_JUDGE_CALL
    wall = timedelta(seconds=candidate_seconds + judge_seconds)

    return CostEstimate(
        n_items=len(items),
        n_l4_items=n_l4,
        model=model,
        model_input_tokens=in_tokens,
        model_output_tokens=out_tokens,
        model_cost=model_cost,
        model_priced=model_priced,
        judges=judges_breakdown,
        judge_cost=judge_total,
        total_cost=model_cost + judge_total,
        estimated_wall_time=wall,
        notes=notes,
        precise_tokens=False,
    )


def _dollars(in_tokens: int, out_tokens: int, price: tuple[float, float]) -> float:
    inp, out = price
    return (in_tokens / 1_000_000) * inp + (out_tokens / 1_000_000) * out


def _dollar(x: float) -> str:
    return f"${x:,.2f}"


def compute_cost_from_usage(tokens_used: dict | None) -> float:
    """Sum dollar cost from a Result-style ``tokens_used`` payload."""
    if not tokens_used:
        return 0.0
    total = 0.0
    cand = tokens_used.get("candidate") or {}
    if cand:
        total += _dollars(
            int(cand.get("input_tokens", 0) or 0),
            int(cand.get("output_tokens", 0) or 0),
            _price_of(cand.get("model") or ""),
        )
    for judge_stats in (tokens_used.get("judges") or {}).values():
        total += _dollars(
            int(judge_stats.get("input_tokens", 0) or 0),
            int(judge_stats.get("output_tokens", 0) or 0),
            _price_of(judge_stats.get("model") or ""),
        )
    return round(total, 6)


def _coerce_panel(judges) -> JudgePanel | None:
    if judges is None:
        return None
    if isinstance(judges, JudgePanel):
        return judges
    if isinstance(judges, str):
        return parse_judges_flag(judges)
    if isinstance(judges, (list, tuple)):
        return JudgePanel(list(judges))
    raise TypeError(f"judges must be JudgePanel, str, list, or None; got {type(judges).__name__}")
