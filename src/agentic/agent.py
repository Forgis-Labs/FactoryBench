"""ReAct-style agent loop with OpenAI function-calling.

One call = one question. The agent is instantiated with the four tool
objects (already bound to this question's time series and the shared
RAG index), plus a chat client from the same runner path used
by the zero-shot panel, so auth and cost accounting inherit for free.

Termination conditions (whichever fires first):
  * the model returns a final assistant message with no tool_calls,
  * the tool-call budget is exhausted (default 8 calls),
  * an unrecoverable error is raised (returned to the caller as an
    error payload).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


AGENT_SYSTEM_PROMPT = """You are an industrial-telemetry reasoning agent evaluating on FactoryBench.

You have access to four tools:

  * `signal_stats` - per-channel statistics (min, max, mean, std, p05, p95, derivative).
  * `forecast` - Chronos-Bolt (200M-param) forecaster for future value prediction.
  * `run_python` - sandboxed Python (numpy/scipy) with the item's time series bound as `ts` (dict of channel → np.ndarray). Great for windowed searches, derivatives, pattern matching.
  * `retrieve_manual` - RAG over vendor PDFs (UR3e, KUKA KR6/KR10, voraus AI) for machine-specific concepts and protocols.

HOW TO DECIDE WHICH TOOL (map from question shape to tool):

  QUESTION SHAPE                                        → TOOL
  --------------------------------------------------------------------------
  "expected value of <signal> at T+N ms/steps?"         → forecast(channel=<signal>, horizon=N)
     Answer with the returned `predicted_value_at_horizon` scalar directly.
     Do NOT reduce/average the full_median_series.

  "at which timestamp should the window begin?" /       → run_python
  "when does <event> begin?" (segment localization)        Use ts['<channel>'] and numpy.
     Typical recipe:
        v = np.abs(np.diff(ts['feedback_speed_0']))   # activity signal
        # find window of length W where activity best matches description
        onset_scores = np.array([v[i:i+W].sum() for i in range(len(v)-W)])
        answer = int(np.argmax(onset_scores))
     Return the integer timestamp answer.

  Root-cause / remediation / vendor-specific concept    → retrieve_manual
     Query with the machine name + observed symptom + specific noun.

  Simple identification / phase-reading / obvious       → DIRECT (no tool)
  visual comparison

RULES:
  1. If the question matches one of the shapes above, use the matching tool.
     A wrong tool call is worse than no tool call - never guess arguments.
  2. Max 6 tool calls per question. Do not loop; if a tool errors, try a
     different approach or answer directly.
  3. Final answer format is EXACT: a single letter, T/F string, ranking
     permutation, single number, or free-form protocol. No preamble, no
     units, no explanation unless the question is free-form. When the
     question says "Answer only with an integer or decimal number, nothing
     else" - do exactly that."""


class Agent:
    def __init__(
        self,
        client,
        model: str,
        tools: Sequence[Any],
        max_tool_calls: int = 6,
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = model
        self.tools_by_name = {t.NAME: t for t in tools}
        self.tool_specs = [t.spec() for t in tools]
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature

    def answer(self, prompt: str) -> Dict[str, Any]:
        """Run the ReAct loop on one prompt. Returns dict with keys:
        `answer`, `trace` (list of tool-call summaries), `usage`, `error`.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        trace: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        calls_used = 0
        last_error: Optional[str] = None

        while calls_used <= self.max_tool_calls:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_specs,
                    tool_choice="auto",
                )
            except Exception as exc:
                last_error = f"chat call failed: {type(exc).__name__}: {exc}"
                logger.warning(last_error)
                break

            u = getattr(resp, "usage", None)
            if u is not None:
                for k in total_usage:
                    total_usage[k] += int(getattr(u, k, 0) or 0)

            choice = resp.choices[0]
            msg = choice.message
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ] or None,
            })

            if not msg.tool_calls:
                # final answer
                return {
                    "answer": (msg.content or "").strip(),
                    "trace": trace,
                    "usage": total_usage,
                    "n_tool_calls": calls_used,
                    "error": None,
                }

            for tc in msg.tool_calls:
                calls_used += 1
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                tool = self.tools_by_name.get(name)
                if tool is None:
                    tool_result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        tool_result = tool(**args)
                    except Exception as exc:
                        tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                trace.append({"tool": name, "args": args, "result_repr": _truncate(repr(tool_result))})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": _serialise(tool_result),
                })
            # loop back for the model's next turn

        return {
            "answer": "",
            "trace": trace,
            "usage": total_usage,
            "n_tool_calls": calls_used,
            "error": last_error or f"tool-call budget exhausted ({self.max_tool_calls})",
        }


def _serialise(x: Any, cap: int = 8000) -> str:
    try:
        s = json.dumps(x, default=lambda o: str(o))
    except Exception:
        s = repr(x)
    return s if len(s) <= cap else s[:cap] + f"...[truncated {len(s) - cap} chars]"


def _truncate(s: str, cap: int = 400) -> str:
    return s if len(s) <= cap else s[:cap] + "…"


class BedrockAgent:
    """Same ReAct pipeline as `Agent`, but driven by AWS Bedrock Converse
    (Anthropic-native tool_use) instead of the OpenAI chat.completions API.

    Interface is deliberately identical: ``answer(prompt) -> {"answer",
    "trace", "usage", "n_tool_calls", "error"}``. Tool specs are the same
    OpenAI-flavoured dicts produced by every tool's ``spec()`` method -
    we translate the ``function`` / ``parameters`` fields into Bedrock's
    ``toolSpec`` / ``inputSchema`` on the fly, so the tools themselves
    don't have to know which backend is running them.
    """

    def __init__(
        self,
        client,        # boto3 bedrock-runtime client
        model: str,    # bedrock model id, e.g. "eu.anthropic.claude-sonnet-4-6"
        tools: Sequence[Any],
        max_tool_calls: int = 6,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        self.client = client
        self.model = model
        self.tools_by_name = {t.NAME: t for t in tools}
        self._oai_specs = [t.spec() for t in tools]
        self.tool_config = {
            "tools": [_oai_to_bedrock_tool(s) for s in self._oai_specs],
        }
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def answer(self, prompt: str) -> Dict[str, Any]:
        system_blocks = [{"text": AGENT_SYSTEM_PROMPT}]
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"text": prompt}]},
        ]
        trace: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        calls_used = 0
        last_error: Optional[str] = None

        while calls_used <= self.max_tool_calls:
            try:
                resp = self.client.converse(
                    modelId=self.model,
                    messages=messages,
                    system=system_blocks,
                    toolConfig=self.tool_config,
                    inferenceConfig={
                        "temperature": self.temperature,
                        "maxTokens":  self.max_output_tokens,
                    },
                )
            except Exception as exc:
                last_error = f"converse call failed: {type(exc).__name__}: {exc}"
                logger.warning(last_error)
                break

            u = resp.get("usage") or {}
            total_usage["prompt_tokens"]     += int(u.get("inputTokens", 0) or 0)
            total_usage["completion_tokens"] += int(u.get("outputTokens", 0) or 0)
            total_usage["total_tokens"]      += int(u.get("totalTokens", 0) or 0)

            out_msg = (resp.get("output") or {}).get("message") or {}
            content_blocks: List[Dict[str, Any]] = out_msg.get("content") or []
            stop_reason = resp.get("stopReason") or ""

            # Echo the assistant turn back into the conversation.
            messages.append({"role": "assistant", "content": content_blocks})

            tool_uses = [b for b in content_blocks if "toolUse" in b]
            if not tool_uses or stop_reason not in ("tool_use", "toolUse"):
                # Final answer - concatenate all text blocks in order.
                text = "".join(b.get("text") or "" for b in content_blocks if "text" in b).strip()
                return {
                    "answer": text,
                    "trace": trace,
                    "usage": total_usage,
                    "n_tool_calls": calls_used,
                    "error": None if text else "empty assistant response",
                }

            tool_result_blocks: List[Dict[str, Any]] = []
            for b in tool_uses:
                tu = b["toolUse"]
                name = tu.get("name")
                tuid = tu.get("toolUseId")
                args = tu.get("input") or {}
                calls_used += 1
                tool = self.tools_by_name.get(name)
                if tool is None:
                    tool_result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        tool_result = tool(**args)
                    except Exception as exc:
                        tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                trace.append({"tool": name, "args": args, "result_repr": _truncate(repr(tool_result))})
                tool_result_blocks.append({
                    "toolResult": {
                        "toolUseId": tuid,
                        "content":   [{"text": _serialise(tool_result)}],
                    }
                })

            messages.append({"role": "user", "content": tool_result_blocks})

        return {
            "answer": "",
            "trace": trace,
            "usage": total_usage,
            "n_tool_calls": calls_used,
            "error": last_error or f"tool-call budget exhausted ({self.max_tool_calls})",
        }


def _oai_to_bedrock_tool(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an OpenAI function spec into Bedrock Converse toolSpec."""
    fn = spec.get("function") or spec
    return {
        "toolSpec": {
            "name":        fn["name"],
            "description": fn.get("description") or "",
            "inputSchema": {"json": fn.get("parameters") or {"type": "object", "properties": {}}},
        }
    }
