"""Post-hoc scoring of model predictions stored in Opik traces.

The scoring layer is independent from inference: given a `(prediction, ground_truth,
answer_format)` triple, each parser runs a strict -> lenient cascade and only
escalates to an LLM judge when deterministic methods fail.
"""
from .types import ParseResult, Provenance

__all__ = ["ParseResult", "Provenance"]