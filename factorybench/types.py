from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerFormat(str, Enum):
    """Categorical answer format inferred from a row's options and acceptance_bounds.

    Each item is scored according to its format. ``free_form`` items (Level 4)
    require an LLM-as-judge and are not scored by this library.
    """

    SINGLE_LETTER_MCQ = "single_letter_mcq"
    FOUR_LETTER_TF = "four_letter_tf"
    FOUR_LETTER_RANKING = "four_letter_ranking"
    SCALAR_RANGE = "scalar_range"
    SCALAR_MARGIN = "scalar_margin"
    SCALAR_EXACT = "scalar_exact"
    TENSOR_MARGIN = "tensor_margin"
    FREE_FORM = "free_form"


@dataclass
class Item:
    """One QA item from the FactoryBench test split."""

    id: str
    level: int
    template_id: int
    template_type: str
    question: str
    options: dict[str, Any]
    answer: Any
    acceptance_bounds: dict[str, Any] | None
    provenance: dict[str, Any]
    context: dict[str, Any]
    hides: list[Any] = field(default_factory=list)
    root_cause: str | None = None
    answer_format: AnswerFormat = AnswerFormat.FREE_FORM

    @property
    def dataset(self) -> str | None:
        """Source dataset name from provenance (e.g. ``aursad``)."""
        return self.provenance.get("dataset") if self.provenance else None

    @property
    def fault_id(self) -> int | None:
        relevance = (self.provenance or {}).get("relevance") or {}
        return relevance.get("fault_id")
