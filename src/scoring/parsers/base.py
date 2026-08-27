from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..types import ParseResult


class Parser(ABC):
    """One parser per `answer_format`. Encapsulates the strict -> lenient cascade.

    The judge fallback lives in the orchestrator (cascade.py) so parsers stay
    pure: no API calls, fully deterministic, trivially unit-testable.
    """

    answer_format: ClassVar[str]

    @abstractmethod
    def parse(self, text: str, ground_truth: Any, **ctx: Any) -> ParseResult:
        """Try strict -> lenient. Return ParseResult.

        `ctx` carries format-specific extras (e.g. `acceptance_bounds` for
        numerical/tensor scoring). Parsers ignore keys they don't use.

        If strict and lenient both fail, return `score=None,
        provenance='unparseable'` and the caller decides whether to escalate
        to the LLM judge.
        """
        ...
