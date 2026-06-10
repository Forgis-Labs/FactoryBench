"""Render a FactoryBench item into the final prompt string sent to a model."""
from __future__ import annotations

from .types import Item


def render_prompt(item: Item) -> str:
    """Render an Item into the canonical prompt.

    The prompt is assembled from, in order:
      1. The time-series format description (free-text).
      2. The acronym mapping (compact ``key=name`` listing).
      3. The encoded time-series rows.
      4. The question text (which already contains answer-format instructions).
      5. The MCQ options block, if any.
    """
    ctx = item.context or {}
    ts_fmt = ctx.get("time_series_format") or {}
    description = ts_fmt.get("description")
    acronyms: dict[str, str] = ts_fmt.get("acronym_mapping") or {}
    time_series: list[str] = ctx.get("time_series") or []

    parts: list[str] = []

    if description:
        parts.append(description.strip())

    if acronyms:
        acr_lines = ", ".join(f"{k}={v}" for k, v in acronyms.items())
        parts.append(f"Acronym mapping:\n{acr_lines}")

    if time_series:
        parts.append("Time series:\n" + "\n".join(time_series))

    parts.append(item.question.strip())

    if item.options:
        opt_lines = "\n".join(f"{k}: {v}" for k, v in item.options.items())
        parts.append(f"Options:\n{opt_lines}")

    return "\n\n".join(parts)
