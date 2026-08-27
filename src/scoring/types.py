from dataclasses import dataclass
from typing import Any, Literal, Optional

Provenance = Literal["strict", "lenient", "judge", "unparseable"]


@dataclass
class ParseResult:
    """Outcome of running a parser on a model prediction.

    `score` is in [0, 1] when the answer was extractable, or None when no
    deterministic method could parse it (the orchestrator may then fall back
    to an LLM judge).

    `provenance` records which layer of the cascade produced the answer:
      - "strict":   prediction matched the expected format exactly
      - "lenient":  recovered via heuristics (cues, regex, last-line, etc.)
      - "judge":    LLM-as-judge verdict (only set by the judge wrapper)
      - "unparseable": no method recovered an answer
    """

    score: Optional[float]
    parsed: Any
    provenance: Provenance
    reason: Optional[str] = None