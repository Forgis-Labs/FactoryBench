"""Template-aware judge prompts for L4 free-form items.

Two templates score under explicit rubrics (rather than the generic
0/0.5/1 free-form protocol from the paper §3.2):

  * **Troubleshooting** (template_id=1):
      1.0  → root cause correctly identified AND remediation steps are
             mostly right (do not need to match the reference exactly).
      0.5  → root cause correctly identified, but remediation wrong/missing.
      0.0  → otherwise.

  * **Optimization** (template_id=2):
      1.0  → problematic parameter identified AND adjustment direction is
             correct (increase / decrease in the right direction).
      0.5  → problematic parameter identified, but direction wrong/missing.
      0.0  → otherwise.

Prompts return a single JSON line ``{"score": float, "reason": str}``. The
caller parses with ``parse_verdict()``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple


_PREDICTION_MAX_CHARS = 8000


_TROUBLESHOOTING_PROMPT = """You are scoring a model's troubleshooting answer for a robotics sensor-data benchmark.

Rubric (apply strictly, no other values allowed):
  1.0, the answer correctly identifies the underlying root cause AND the proposed remediation steps are mostly right. The steps do NOT need to match the reference exactly: minor wording differences, additional sensible checks, or omitted minor steps are all fine, as long as the overall remediation is on the right track and would plausibly resolve the issue.
  0.5, the answer correctly identifies the underlying root cause but the remediation is wrong, missing, or off-topic (e.g. proposes an unrelated fix, contradicts the reference, or refuses to give steps).
  0.0, the root cause is wrong or absent (model refused, identified the wrong fault, or stated "no anomaly" when one was present).

The known root cause is provided to you separately. An answer counts as identifying the root cause when it names the same physical phenomenon, equivalent paraphrases are fine, but a different fault category (e.g. saying "payload too heavy" when the root cause is "TCP misconfiguration") is wrong.

Respond with ONLY a JSON object on a single line, no markdown:
{{"score": <0.0|0.5|1.0>, "reason": "<one short sentence>"}}

Question:
{question}

Reference answer:
{reference}

Known root cause (canonical key):
{root_cause}

Model answer:
{prediction}
"""


_OPTIMIZATION_PROMPT = """You are scoring a model's optimization answer for a robotics sensor-data benchmark.

Rubric (apply strictly, no other values allowed):
  1.0, the answer identifies the problematic parameter AND proposes adjusting it in the correct direction (increase vs. decrease, matching the reference). Exact magnitudes do not matter; only direction matters.
  0.5, the answer identifies the problematic parameter but proposes the wrong direction, no direction, or contradictory directions.
  0.0, the problematic parameter is not identified, or the model recommends adjusting an unrelated parameter, or refuses to answer.

The reference answer states the correct parameter and its target value, from which the correct adjustment direction can be inferred relative to the configured value.

Respond with ONLY a JSON object on a single line, no markdown:
{{"score": <0.0|0.5|1.0>, "reason": "<one short sentence>"}}

Question:
{question}

Reference answer:
{reference}

Model answer:
{prediction}
"""


def build_prompt(
    *,
    template_id: int,
    question: str,
    prediction: str,
    reference: Any,
    root_cause: Optional[str] = None) -> str:
    """Assemble the per-template judge prompt. Raises for unsupported templates."""
    pred = (str(prediction) if prediction is not None else "")[:_PREDICTION_MAX_CHARS]
    ref = (str(reference) if reference is not None else "")[:2000]
    q = (question or "(question text not available)")[:6000]

    if template_id == 1:
        rc = (root_cause or "(unspecified, treat as 'normal/no anomaly')").strip()
        return _TROUBLESHOOTING_PROMPT.format(
            question=q, reference=ref, root_cause=rc, prediction=pred)
    if template_id == 2:
        return _OPTIMIZATION_PROMPT.format(
            question=q, reference=ref, prediction=pred)
    raise ValueError(f"no rubric prompt for template_id={template_id!r}")


def parse_verdict(raw: str) -> Tuple[Optional[float], str]:
    """Pull ``{score, reason}`` from a judge response. Tolerant of code fences
    and surrounding prose. Returns (None, raw_excerpt) when unparseable."""
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) >= 2 else ""
        text = text.removeprefix("json").strip()
        if "```" in text:
            text = text.split("```", 1)[0]
    try:
        obj = json.loads(text)
        score = float(obj.get("score"))
        if score not in (0.0, 0.5, 1.0):
            score = max(0.0, min(1.0, round(score * 2) / 2))
        return score, str(obj.get("reason", ""))[:300]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r"\b(1(?:\.0+)?|0\.5|0(?:\.0+)?)\b", text)
    if m:
        return float(m.group(1)), text[:300]
    return None, text[:300]
