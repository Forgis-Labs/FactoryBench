"""Cost preview for ``factorybench evaluate``.

Token counts are exact when ``tiktoken`` is installed (via the
``[tokenizers]`` extra), and fall back to a ``len(text) / 4`` heuristic
otherwise. The bundled per-model price table (``factorybench/prices.json``)
is a manual snapshot with a version stamp; override at runtime with
:func:`set_price`, drop a file at ``~/.config/factorybench/prices.json``, or
set the ``FB_PRICES`` env var.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Iterable

from .data import load_split
from .judges import PAPER_DEFAULT_JUDGES, JudgePanel, RUBRIC_PROMPT, parse_judges_flag
from .prompt import render_prompt
from .tokens import count_tokens as precise_count_tokens, has_tiktoken, is_precise
from .types import AnswerFormat, Item


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #

# Output token estimates per item, by answer format. L4 reference answers are
# multi-sentence; the rest are very short.
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

# Each judge call outputs exactly one of "0", "0.5", "1" so its output is tiny.
_JUDGE_OUTPUT_TOKENS = 3


def estimate_tokens(text: str | None, model: str | None = None) -> int:
    """Token count for cost previews.

    Uses :func:`factorybench.tokens.count_tokens`, which is exact when
    ``tiktoken`` is installed (via ``pip install "factorybench[tokenizers]"``)
    and falls back to a ``len(text) / 4`` heuristic otherwise. Pass ``model``
    to pick the most accurate encoder available.
    """
    return precise_count_tokens(text, model=model)


def estimate_output_tokens(item: Item) -> int:
    return _OUTPUT_TOKEN_ESTIMATE.get(item.answer_format, 50)


# --------------------------------------------------------------------------- #
# Price table ($/1M tokens). Snapshot -- override via set_price or FB_PRICES.
# --------------------------------------------------------------------------- #

# Each entry: (input $/M tokens, output $/M tokens).
# Bundled price snapshot lives at <package>/prices.json (with a version stamp).
# User overrides at ~/.config/factorybench/prices.json take precedence.
BUNDLED_PRICES_PATH = Path(__file__).parent / "prices.json"
USER_PRICES_PATH = Path.home() / ".config" / "factorybench" / "prices.json"

# Populated by _load_bundled_prices() at import time.
PRICES_PER_M_TOKENS: dict[str, tuple[float, float]] = {}
DEFAULT_PRICE_PER_M_TOKENS: tuple[float, float] = (5.00, 15.00)
# Where the *currently active* prices came from (set by _load_bundled_prices).
PRICES_METADATA: dict = {}


def _load_prices_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _apply_prices_payload(payload: dict, *, source: str) -> None:
    """Replace PRICES_PER_M_TOKENS with the table in ``payload`` (one source at a time)."""
    global DEFAULT_PRICE_PER_M_TOKENS, PRICES_METADATA
    PRICES_PER_M_TOKENS.clear()
    for model, rates in (payload.get("models") or {}).items():
        try:
            PRICES_PER_M_TOKENS[model] = (
                float(rates["input_per_m"]),
                float(rates["output_per_m"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    default = payload.get("default_rate")
    if isinstance(default, dict):
        try:
            DEFAULT_PRICE_PER_M_TOKENS = (
                float(default["input_per_m"]),
                float(default["output_per_m"]),
            )
        except (KeyError, TypeError, ValueError):
            pass
    PRICES_METADATA = {
        "source": source,
        "version": payload.get("version"),
        "updated_at": payload.get("updated_at"),
        "source_description": payload.get("source"),
    }


def _load_bundled_prices() -> None:
    """Load the bundled snapshot, then overlay any user override on top."""
    bundled = _load_prices_file(BUNDLED_PRICES_PATH)
    if bundled is None:
        # Last-resort hardcoded fallback so the module still imports.
        bundled = {"version": "fallback", "models": {}, "default_rate": {"input_per_m": 5.0, "output_per_m": 15.0}}
    _apply_prices_payload(bundled, source=str(BUNDLED_PRICES_PATH))

    user = _load_prices_file(USER_PRICES_PATH)
    if user is not None:
        # Merge: user keys override bundled; default_rate replaced only if user provides it.
        merged_models = dict(PRICES_PER_M_TOKENS)
        for model, rates in (user.get("models") or {}).items():
            try:
                merged_models[model] = (
                    float(rates["input_per_m"]),
                    float(rates["output_per_m"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        payload = {
            "version": user.get("version") or PRICES_METADATA.get("version"),
            "updated_at": user.get("updated_at") or PRICES_METADATA.get("updated_at"),
            "source": "bundled + user override",
            "models": {m: {"input_per_m": p[0], "output_per_m": p[1]} for m, p in merged_models.items()},
            "default_rate": user.get("default_rate") or {
                "input_per_m": DEFAULT_PRICE_PER_M_TOKENS[0],
                "output_per_m": DEFAULT_PRICE_PER_M_TOKENS[1],
            },
        }
        _apply_prices_payload(payload, source=f"{BUNDLED_PRICES_PATH} + {USER_PRICES_PATH}")


_load_bundled_prices()


def set_price(model: str, *, input_per_m: float, output_per_m: float) -> None:
    """Override the price-per-1M-tokens for a given model string."""
    PRICES_PER_M_TOKENS[model] = (float(input_per_m), float(output_per_m))


def _load_env_prices() -> None:
    """Read ``FB_PRICES`` from the environment.

    Format: comma-separated entries ``model:input_per_m/output_per_m``.
    Example: ``FB_PRICES="gpt-5.1:6.0/18.0,my-model:0.5/2.0"``.
    """
    raw = os.environ.get("FB_PRICES")
    if not raw:
        return
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        model, rates = entry.split(":", 1)
        try:
            inp, out = rates.split("/", 1)
            set_price(model.strip(), input_per_m=float(inp), output_per_m=float(out))
        except ValueError:
            continue


_load_env_prices()


def _price_of(model: str) -> tuple[float, float]:
    """Look up (input_per_m, output_per_m) for a model with prefix fallback."""
    if model in PRICES_PER_M_TOKENS:
        return PRICES_PER_M_TOKENS[model]
    # Try a normalized variant (dots -> dashes).
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
    # True when token counts came from a real tokenizer (tiktoken) rather than
    # the len(text) / 4 heuristic.
    precise_tokens: bool = False

    def format(self) -> str:
        """Return a human-readable preview string."""
        # "tiktoken-counted" not "exact": tiktoken is OpenAI's tokenizer; we use it
        # as a universal-ish proxy for non-OpenAI models (Claude / DeepSeek) too,
        # where it's close (~5-10%) but not literally the provider's tokenizer.
        precision_tag = "tiktoken-counted" if self.precise_tokens else "heuristic tokens (+/-25%)"
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
    """Estimate dollar cost + wall time of an :func:`evaluate` call.

    ``judges`` may be a :class:`JudgePanel`, a CLI shorthand string, or a list
    of judge specs.
    """
    panel = _coerce_panel(judges)
    have_judges = panel is not None

    # Mirror evaluate()'s level-resolution rules.
    if isinstance(level, str) and level.strip().lower() == "all":
        levels = [1, 2, 3, 4] if have_judges else [1, 2, 3]
    else:
        from .data import _resolve_levels
        levels = _resolve_levels(level)

    items = load_split(level=levels, split=split, revision=revision, max_items=max_items)
    return _estimate_for_items(items, model=model, panel=panel)


def _estimate_for_items(items: list[Item], *, model: str, panel: JudgePanel | None) -> CostEstimate:
    notes: list[str] = []

    # Candidate-model tokens.
    in_tokens = 0
    out_tokens = 0
    n_l4 = 0
    for it in items:
        prompt = render_prompt(it)
        in_tokens += precise_count_tokens(prompt, model=model)
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

    # Judge tokens (only on L4 items).
    judges_breakdown: list[JudgeCostBreakdown] = []
    judge_total = 0.0
    precise = has_tiktoken()
    if panel is not None and n_l4 > 0:
        # Build the per-item judge prompt without a real prediction; use the
        # reference answer as a stand-in for prediction length.
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
                per_judge_in += precise_count_tokens(rendered, model=spec)
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
                notes.append(
                    f"no price entry for judge {spec!r}; using default rate"
                )

    if not precise:
        notes.append(
            "tiktoken is unavailable; falling back to a len(text)/4 heuristic. "
            "tiktoken is a runtime dependency -- reinstall the package or run "
            "'pip install tiktoken' to restore tiktoken-counted previews."
        )

    # Time estimate (sequential baseline).
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
        precise_tokens=precise,
    )


def _dollars(in_tokens: int, out_tokens: int, price: tuple[float, float]) -> float:
    inp, out = price
    return (in_tokens / 1_000_000) * inp + (out_tokens / 1_000_000) * out


def _dollar(x: float) -> str:
    return f"${x:,.2f}"


def compute_cost_from_usage(tokens_used: dict | None) -> float:
    """Sum dollar cost from a :class:`Result`-style ``tokens_used`` payload.

    Shape::

        {
          "candidate": {"model": "...", "input_tokens": N, "output_tokens": N, "calls": N},
          "judges": {"<judge>": {"model": "...", "input_tokens": N, "output_tokens": N, "calls": N}, ...},
        }
    """
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
