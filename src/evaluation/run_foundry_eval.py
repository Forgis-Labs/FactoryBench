#!/usr/bin/env python3
"""Multi-endpoint evaluation for Microsoft Foundry models.

Reuses scoring, prompt loading, and ground-truth helpers from
``src.evaluation.test_gpt_5mini`` so evaluation output shape is identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import (
    DEFAULT_JUDGE_MODEL,
    FOUNDRY_MODELS,
    get_api_key_env,
    get_upstream_model_id)
from src.evaluation.test_gpt_5mini import (
    JUDGE_SYSTEM_PROMPT,
    _estimate_cost,
    _extract_output_text_from_responses_body,
    _to_dict,
    load_dotenv_file,
    load_json,
    load_prompt_entries,
    parse_llm_answer,
    save_json)

logger = logging.getLogger(__name__)


_TENSOR_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_TENSOR_BRACKET_RE = re.compile(r"\[\s*[^\[\]]*?-?\d[^\[\]]*?\]")


def _parse_tensor_answer(value: Any, expected_len: Optional[int] = None) -> list[float]:
    """Parse a tensor answer formatted as a JSON-like array, e.g. "[1, 2.5, 3]".

    Accepts surrounding whitespace, brackets, and trailing reasoning. When the
    response is prose with the answer at the end (e.g. claude), prefer the
    LAST ``[...]`` block; if no bracket-shaped block matches the expected
    length, fall back to the LAST ``expected_len`` numbers in the text. This
    avoids treating timestamps/setpoints from the reasoning as the answer.
    """
    if value is None:
        raise ValueError("empty tensor")
    s = str(value).strip()
    if not s:
        raise ValueError("empty tensor")

    if expected_len is not None:
        # Prefer the last bracket block whose number count matches.
        for seg in reversed(_TENSOR_BRACKET_RE.findall(s)):
            nums = _TENSOR_NUM_RE.findall(seg)
            if len(nums) == expected_len:
                return [float(x) for x in nums]
        # Fall back to the LAST expected_len numbers in the text.
        all_nums = _TENSOR_NUM_RE.findall(s)
        if len(all_nums) >= expected_len:
            return [float(x) for x in all_nums[-expected_len:]]

    matches = _TENSOR_NUM_RE.findall(s)
    if not matches:
        raise ValueError(f"no numeric tokens in {s!r}")
    return [float(x) for x in matches]


_MCMS_TF_CACHE: Dict[int, "re.Pattern[str]"] = {}


def _parse_mcms_answer(value: Any, expected_len: int) -> Optional[str]:
    """Extract a length-N T/F answer from a model response. Tolerates models
    that emit reasoning around the final TFTF-style token.
    """
    if value is None:
        return None
    s = str(value).strip().upper()
    if len(s) == expected_len and set(s) <= {"T", "F"}:
        return s
    pat = _MCMS_TF_CACHE.get(expected_len)
    if pat is None:
        pat = re.compile(r"\b([TF]{" + str(expected_len) + r"})\b")
        _MCMS_TF_CACHE[expected_len] = pat
    matches = pat.findall(s)
    return matches[-1] if matches else None


def _parse_numerical_answer(value: Any) -> float:
    """Parse a numerical answer; if the raw value isn't a float, fall back to
    the last numerical token in the text (e.g. claude wraps its answer like
    ``**1810**`` after multi-paragraph reasoning despite being asked for just
    a number).
    """
    if value is None:
        raise ValueError("empty numerical")
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    matches = _TENSOR_NUM_RE.findall(s)
    if not matches:
        raise ValueError(f"no numeric tokens in {s!r}")
    return float(matches[-1])


def resolve_api_key(model: Optional[str] = None) -> str:
    """Resolve the API key for ``model``.

    Per-model ``api_key_env`` (e.g. ``OPENROUTER_API_KEY`` for qwen on
    OpenRouter) wins; absent that, falls back to the Azure/OpenAI defaults so
    existing GPT-5.x configs keep working.
    """
    if model:
        env_key = get_api_key_env(model)
        if env_key:
            override = os.getenv(env_key)
            if override:
                return override
    key = (
        os.getenv("AZURE_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError(
            "Missing API key. Set AZURE_API_KEY (preferred) or AZURE_OPENAI_API_KEY in .env"
        )
    return key


def resolve_endpoint(model: str) -> Tuple[str, str, Optional[str]]:
    """Return (endpoint_url, api_style, api_version) for the given model."""
    cfg = FOUNDRY_MODELS.get(model)
    if cfg is None:
        raise ValueError(
            f"Unknown Foundry model: {model}. Supported: {list(FOUNDRY_MODELS.keys())}"
        )
    endpoint = (os.getenv(cfg["endpoint_env"]) or cfg["endpoint_default"]).rstrip("/").rstrip('"')
    api_version: Optional[str] = None
    if cfg.get("requires_api_version"):
        # Per-model env var wins (e.g. PROJECT_API_VERSION for Foundry project
        # endpoint), then model-level default, then generic env fallbacks.
        env_name = cfg.get("api_version_env")
        api_version = (
            (os.getenv(env_name) if env_name else None)
            or cfg.get("api_version_default")
            or os.getenv("AZURE_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
        )
    return endpoint, cfg["api_style"], api_version


def _openai_client(base_url: str, api_key: str, api_version: Optional[str] = None) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install with: pip install openai")
    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if api_version:
        kwargs["default_query"] = {"api-version": api_version}
    return OpenAI(**kwargs)


def call_openai_style(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_style: str = "openai",
    api_version: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Call an OpenAI-compatible endpoint (gpt-5.1, DeepSeek, Mistral)."""
    client = _openai_client(base_url, api_key, api_version=api_version)

    # Mistral on Azure doesn't support /responses and uses `max_tokens` on chat.completions.
    if api_style != "mistral":
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens)
            body = _to_dict(response)
            answer = _extract_output_text_from_responses_body(body)
            if answer:
                return answer, body
        except Exception as exc:
            logger.debug(f"responses.create failed for {model}: {exc}; falling back to chat.completions")

    completion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_style == "mistral":
        completion_kwargs["max_tokens"] = max_tokens
    else:
        completion_kwargs["max_completion_tokens"] = max_tokens

    try:
        response = client.chat.completions.create(**completion_kwargs)
    except TypeError as exc:
        if "max_completion_tokens" in str(exc):
            completion_kwargs.pop("max_completion_tokens", None)
            completion_kwargs["extra_body"] = {"max_completion_tokens": max_tokens}
            response = client.chat.completions.create(**completion_kwargs)
        else:
            raise
    body = _to_dict(response)
    answer = body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return answer, body


def call_anthropic_style(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    """Call Anthropic-native /messages endpoint (claude-haiku-4-5)."""
    r = requests.post(
        f"{base_url}/messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300)
    r.raise_for_status()
    body = r.json()

    answer_parts: list[str] = []
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            answer_parts.append(block.get("text", ""))
    answer = "".join(answer_parts)

    usage = body.get("usage", {}) or {}
    body["usage"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    body.setdefault("model", model)
    return answer, body


def call_model(model: str, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    api_key = resolve_api_key(model)
    endpoint, api_style, api_version = resolve_endpoint(model)
    upstream_id = get_upstream_model_id(model)

    if api_style == "anthropic":
        return call_anthropic_style(endpoint, api_key, upstream_id, prompt, max_tokens)
    return call_openai_style(
        endpoint, api_key, upstream_id, prompt, max_tokens,
        api_style=api_style, api_version=api_version)


def _is_judge_disabled(judge_model: Optional[str]) -> bool:
    """Return True when LLM-as-judge should be skipped (no API call, score=None).

    Triggered by env var FB_DISABLE_JUDGE=1, an empty/falsy --judge-model, or
    --judge-model=none/off/disabled (case-insensitive).
    """
    if os.environ.get("FB_DISABLE_JUDGE", "").strip() in ("1", "true", "yes", "on"):
        return True
    if not judge_model:
        return True
    return str(judge_model).strip().lower() in ("none", "off", "disabled", "no")


def foundry_llm_judge(
    question: str,
    prediction: str,
    reference: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 256) -> Tuple[float, str]:
    """Score a free-form prediction with the default judge model (gpt-5.1)."""
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model answer: {prediction}\n\n"
        "Please score the model answer (0-10) and provide a one-sentence justification."
    )
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_msg}"
    try:
        raw, _body = call_model(judge_model, full_prompt, max_tokens)
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        raw_score = float(parsed["score"])
        reason = str(parsed.get("reason", ""))
        normalised = max(0.0, min(1.0, raw_score / 10.0))
        return normalised, reason
    except Exception as e:
        return 0.0, f"Judge call failed: {e}"


def build_question_index(questions_dir: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Map filename stem -> full question payload (answer, template_type, options, acceptance_bounds...)."""
    index: Dict[str, Dict[str, Any]] = {}
    if questions_dir is None or not questions_dir.is_dir():
        return index
    for path in questions_dir.rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            index[path.stem] = payload
    return index


def infer_answer_format(q: Dict[str, Any]) -> str:
    """Infer an answer_format string from question payload fields.

    Question generators don't emit an explicit `answer_format`; we derive one
    from template_type / options / answer shape / level so score_prediction
    can dispatch correctly.
    """
    level = q.get("level")
    tid = q.get("template_id")
    template_type = str(q.get("template_type") or "").lower()
    options = q.get("options") or {}
    ans = q.get("answer")

    if level == 4:
        if tid in (3, 4) or "ranking" in template_type:
            return "ranking"
        return "free_form"

    if "ranking" in template_type:
        return "ranking"

    if isinstance(ans, str):
        s = ans.strip()
        s_upper = s.upper()
        if s and set(s_upper) <= {"T", "F"} and len(s_upper) >= 2:
            return "multiple_choice_multi_select"
        if len(s_upper) >= 3 and set(s_upper) <= set("ABCD"):
            return "ranking"
        if len(s) == 1 and s_upper in "ABCDEFGH" and isinstance(options, dict) and options:
            return "multiple_choice_single_select"
        if s.startswith("["):
            return "tensor"

    if isinstance(ans, list):
        return "tensor"

    if isinstance(ans, (int, float)):
        return "numerical"

    if isinstance(ans, str):
        try:
            float(ans)
            return "numerical"
        except ValueError:
            pass

    return "free_form"


def _normalized_acceptance_bounds(ab: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Accept `tolerance` (level 1) as a synonym for `margin` (level 2)."""
    if not isinstance(ab, dict):
        return ab
    if "margin" not in ab and "tolerance" in ab:
        out = dict(ab)
        out["margin"] = ab["tolerance"]
        return out
    return ab


def score_prediction(
    answer_format: str,
    prediction: Any,
    ground_truth: Any,
    acceptance_bounds: Optional[Dict[str, Any]],
    question_text: str,
    judge_model: str) -> Tuple[Optional[float], Optional[Tuple[float, str]]]:
    """Return (score, judge_result_or_none). Mirrors test_gpt_5mini scoring branches."""
    acceptance_bounds = _normalized_acceptance_bounds(acceptance_bounds)
    gt = ground_truth
    pred = prediction
    judge_result: Optional[Tuple[float, str]] = None
    score: Optional[float] = None

    try:
        if answer_format == "free_form":
            # Single-judge scoring removed; free-form items are graded post-hoc
            # by the 3-judge ensemble in scripts/score_replies_batch.py.
            score = None
            judge_result = None

        elif answer_format == "numerical":
            try:
                gt_val = round(float(gt), 4)
                pred_val = round(_parse_numerical_answer(pred), 4)
            except Exception:
                score = 0.0
            else:
                # Three-level piecewise for margin-bounded items (matches tensor branch).
                # min/max bounds remain binary because there is no native "2x margin" zone.
                if acceptance_bounds and "min" in acceptance_bounds and "max" in acceptance_bounds:
                    score = float(
                        float(acceptance_bounds["min"]) <= pred_val <= float(acceptance_bounds["max"])
                    )
                elif acceptance_bounds:
                    margin = float(acceptance_bounds.get("margin", 0))
                    d = abs(pred_val - gt_val)
                    if d <= margin:
                        score = 1.0
                    elif d <= 2 * margin:
                        score = 0.5
                    else:
                        score = 0.0
                else:
                    score = float(abs(pred_val - gt_val) < 1e-4)

        elif answer_format == "tensor":
            try:
                gt_vals = _parse_tensor_answer(gt)
                pred_vals = _parse_tensor_answer(pred, expected_len=len(gt_vals))
            except Exception:
                score = 0.0
            else:
                # Three-level piecewise scorer: 1 within margin, 0.5 within 2x margin, 0 otherwise.
                # Calibration intent: per-channel margin m_j = R_j/12 yields E = 3m/R = 1/4 under
                # uniform random in the channel's natural range, matching single-select MCQ chance.
                if acceptance_bounds and "margin" in acceptance_bounds:
                    margins = acceptance_bounds["margin"]
                    if len(gt_vals) == len(pred_vals) == len(margins):
                        n = len(gt_vals)
                        per_elem = []
                        for p, g, m in zip(pred_vals, gt_vals, margins):
                            d = abs(p - g)
                            if d <= m:
                                per_elem.append(1.0)
                            elif d <= 2 * m:
                                per_elem.append(0.5)
                            else:
                                per_elem.append(0.0)
                        score = sum(per_elem) / n if n > 0 else 0.0
                    else:
                        score = 0.0
                else:
                    score = float(gt_vals == pred_vals)

        elif answer_format == "multiple_choice_multi_select":
            gt_str = str(gt).strip().upper()
            pred_str = _parse_mcms_answer(pred, len(gt_str))
            # Previous scheme (commented): tiered 1.0 / 0.5 / 0.0 (all-correct,
            # off-by-one, otherwise zero). Now: positional fraction so each
            # correctly answered T/F position contributes 1/n.
            # if pred_str is not None and set(gt_str) <= {"T", "F"}:
            #     n = len(gt_str)
            #     n_correct = sum(g == p for g, p in zip(gt_str, pred_str))
            #     if n_correct == n:
            #         score = 1.0
            #     elif n_correct >= n - 1:
            #         score = 0.5
            #     else:
            #         score = 0.0
            # else:
            #     score = 0.0
            if pred_str is not None and set(gt_str) <= {"T", "F"}:
                n = len(gt_str)
                n_correct = sum(g == p for g, p in zip(gt_str, pred_str))
                score = n_correct / n
            else:
                score = 0.0

        elif answer_format == "multiple_choice_single_select":
            gt_str = str(gt).strip().upper()
            pred_letter = parse_llm_answer(str(pred).strip())
            score = 0.0 if pred_letter is None else float(gt_str == pred_letter)

        elif answer_format == "ranking":
            gt_str = str(gt).strip().upper()
            match = re.search(r"\b([A-D]{4})\b", str(pred).strip().upper())
            pred_str = match.group(1) if match else ""
            # Previous scheme (commented): exact match only (1.0 or 0.0).
            # Now: positional fraction so a near-miss like ABCD vs ABDC
            # gets credit for the 2 correctly-placed items.
            # score = float(gt_str == pred_str)
            if pred_str and len(pred_str) == len(gt_str):
                n = len(gt_str)
                n_correct = sum(g == p for g, p in zip(gt_str, pred_str))
                score = n_correct / n
            else:
                score = 0.0

        else:
            score = float(str(gt) == str(pred))
    except Exception:
        score = None

    return score, judge_result


def log_to_opik(
    qa_payload: Dict[str, Any],
    prompt_text: str,
    pred: Any,
    gt: Any,
    body: Dict[str, Any],
    model_name: str,
    eval_level: str,
    answer_format: str,
    question_type: str,
    score: Optional[float],
    judge_result: Optional[Tuple[float, str]],
    est_cost: float,
    usage_raw: Dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int) -> None:
    if not os.getenv("OPIK_API_KEY"):
        return
    try:
        import opik
        client_opik = opik.Opik(
            project_name=os.getenv("OPIK_PROJECT_NAME", "FactoryBench"),
            workspace=os.getenv("OPIK_WORKSPACE", "anon-workspace"))
        total_tokens = int(usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens))
        metadata_payload = qa_payload.get("metadata") or {}
        dataset_tag = metadata_payload.get("dataset")
        episode_tag = metadata_payload.get("episode")
        qa_pair_id = metadata_payload.get("qa_pair_id")

        opik_tags = [str(t) for t in [eval_level, question_type, dataset_tag, answer_format] if t]
        if answer_format and answer_format != "unknown":
            opik_tags.append(f"eval_{answer_format}")
        if model_name and model_name != "unknown":
            opik_tags.append(model_name)
        if qa_pair_id:
            opik_tags.append(f"template_{qa_pair_id}")

        opik_metadata = {
            "model": model_name,
            "answer_format": answer_format,
            "question_type": question_type,
            "qa_pair_id": qa_pair_id,
            "dataset": dataset_tag,
            "episode": episode_tag,
            "time_window": metadata_payload.get("time_window"),
            "correct_answer": gt,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "total_estimated_cost": f"${est_cost}",
            },
            "full_telemetry_context": {"text": prompt_text},
        }

        trace = client_opik.trace(
            name=f"factorybench_{eval_level}_{qa_pair_id}",
            input={"question": qa_payload.get("question"), "ground_truth": gt, "prediction": pred},
            output={"answer": pred, "raw_body": body},
            tags=opik_tags,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata=opik_metadata,
            total_estimated_cost=est_cost,
            model=body.get("model") or model_name)

        if score is not None:
            pred_str_repr = str(pred).strip() if pred is not None else "N/A"
            gt_str_repr = str(gt).strip() if gt is not None else "N/A"
            accuracy_reason = (
                judge_result[1] if (judge_result and judge_result[1])
                else f"Predicted: '{pred_str_repr}' | Ground truth: '{gt_str_repr}'"
            )
            trace.log_feedback_score(name="accuracy", value=float(score), reason=accuracy_reason)
            if judge_result is not None:
                trace.log_feedback_score(
                    name="llm_judge", value=judge_result[0], reason=judge_result[1] or accuracy_reason)
    except Exception as e:
        logger.warning(f"Opik logging failed: {e}")


def _build_openai_batch_jsonl(
    entries: list[Tuple[Path, str, int, str]],
    model: str,
    max_tokens: int) -> str:
    """JSONL body for OpenAI /v1/batches with /chat/completions endpoint."""
    from src.config import get_batch_deployment
    deployment = get_batch_deployment(model)
    lines = []
    for prompt_path, prompt_text, prompt_idx, custom_id in entries:
        body = {
            "model": deployment,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_completion_tokens": max_tokens,
        }
        lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/chat/completions",
            "body": body,
        }))
    return "\n".join(lines)


def run_openai_batch(
    entries: list[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    state: Dict[str, Any],
    lock: threading.Lock,
    total: int,
    cost_limit: float,
    poll_interval: int = 30) -> None:
    """Submit an OpenAI-style batch job, poll to completion, save results."""
    import tempfile
    import time

    base_url, _api_style, api_version = resolve_endpoint(model)
    api_key = resolve_api_key(model)
    client = _openai_client(base_url, api_key, api_version=api_version)

    upstream_id = get_upstream_model_id(model)
    jsonl_text = _build_openai_batch_jsonl(entries, upstream_id, max_output_tokens)
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    tf.write(jsonl_text)
    tf.close()

    try:
        with open(tf.name, "rb") as fb:
            file_obj = client.files.create(file=fb, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/chat/completions",
            completion_window="24h")
        logger.info(
            f"[batch] Submitted {model} batch={batch.id} with {len(entries)} requests"
        )
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass

    terminal = {"completed", "failed", "cancelled", "expired"}
    while True:
        batch = client.batches.retrieve(batch.id)
        rc = getattr(batch, "request_counts", None)
        rc_str = ""
        if rc is not None:
            rc_str = (
                f" | completed={getattr(rc, 'completed', 0)}"
                f"/{getattr(rc, 'total', 0)} "
                f"failed={getattr(rc, 'failed', 0)}"
            )
        logger.info(f"[batch] {model} batch={batch.id}: status={batch.status}{rc_str}")
        if batch.status in terminal:
            break
        time.sleep(poll_interval)

    if batch.status != "completed":
        logger.error(f"[batch] {model} batch={batch.id} ended status={batch.status}")
        for entry in entries:
            _finalize_failure(
                entry, f"batch ended with status={batch.status}",
                output_dir, state, lock, total)
        return

    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, Any] = {}

    if getattr(batch, "output_file_id", None):
        content = client.files.content(batch.output_file_id).read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        for line in content.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            cid = data.get("custom_id")
            if data.get("error"):
                errors[cid] = data["error"]
                continue
            resp = data.get("response") or {}
            if int(resp.get("status_code", 0)) == 200:
                results[cid] = resp.get("body") or {}
            else:
                errors[cid] = resp.get("body") or {"message": f"status={resp.get('status_code')}"}

    if getattr(batch, "error_file_id", None):
        content = client.files.content(batch.error_file_id).read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        for line in content.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            cid = data.get("custom_id", "unknown")
            errors[cid] = data.get("error", {"message": "unknown"})

    for entry in entries:
        _prompt_path, _prompt_text, _prompt_idx, custom_id = entry
        if custom_id in results:
            body = results[custom_id]
            answer = ""
            choices = body.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                answer = msg.get("content") or ""
            _finalize_success(
                entry, answer, body, model, output_dir,
                ground_truth_index, eval_level, judge_model,
                state, lock, total, cost_limit)
        elif custom_id in errors:
            err = errors[custom_id]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            _finalize_failure(entry, err_msg, output_dir, state, lock, total)
        else:
            _finalize_failure(entry, "missing from batch results", output_dir, state, lock, total)


def run_anthropic_batch(
    entries: list[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    state: Dict[str, Any],
    lock: threading.Lock,
    total: int,
    cost_limit: float,
    poll_interval: int = 30) -> None:
    """Submit an Anthropic batch job to /v1/messages/batches, poll, save."""
    import time

    base_url, _api_style, _ = resolve_endpoint(model)
    api_key = resolve_api_key(model)
    upstream_id = get_upstream_model_id(model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    requests_payload = []
    for prompt_path, prompt_text, _prompt_idx, custom_id in entries:
        requests_payload.append({
            "custom_id": custom_id,
            "params": {
                "model": upstream_id,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": prompt_text}],
            },
        })

    r = requests.post(
        f"{base_url}/messages/batches",
        headers=headers,
        json={"requests": requests_payload},
        timeout=300)
    r.raise_for_status()
    batch = r.json()
    batch_id = batch["id"]
    logger.info(f"[batch] Submitted {model} batch={batch_id} with {len(entries)} requests")

    while True:
        r = requests.get(
            f"{base_url}/messages/batches/{batch_id}",
            headers=headers,
            timeout=60)
        r.raise_for_status()
        batch = r.json()
        status = batch.get("processing_status")
        counts = batch.get("request_counts", {}) or {}
        logger.info(
            f"[batch] {model} batch={batch_id}: status={status} | "
            f"processing={counts.get('processing', 0)} "
            f"succeeded={counts.get('succeeded', 0)} "
            f"errored={counts.get('errored', 0)} "
            f"canceled={counts.get('canceled', 0)} "
            f"expired={counts.get('expired', 0)}"
        )
        if status == "ended":
            break
        time.sleep(poll_interval)

    results_url = batch.get("results_url")
    if not results_url:
        logger.error(f"[batch] {model} batch={batch_id} ended without results_url")
        for entry in entries:
            _finalize_failure(entry, "batch ended without results_url",
                              output_dir, state, lock, total)
        return

    r = requests.get(results_url, headers=headers, timeout=600)
    r.raise_for_status()

    results: Dict[str, Dict[str, Any]] = {}
    for line in r.text.strip().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        results[data.get("custom_id")] = data

    for entry in entries:
        _prompt_path, _prompt_text, _prompt_idx, custom_id = entry
        data = results.get(custom_id)
        if data is None:
            _finalize_failure(entry, "missing from batch results",
                              output_dir, state, lock, total)
            continue

        result = data.get("result") or {}
        rtype = result.get("type")
        if rtype == "succeeded":
            msg = result.get("message") or {}
            answer_parts: list[str] = []
            for block in msg.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    answer_parts.append(block.get("text", ""))
            answer = "".join(answer_parts)
            usage = msg.get("usage", {}) or {}
            msg["usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
            msg.setdefault("model", model)
            _finalize_success(
                entry, answer, msg, model, output_dir,
                ground_truth_index, eval_level, judge_model,
                state, lock, total, cost_limit)
        else:
            err = result.get("error") or result
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            _finalize_failure(entry, f"{rtype}: {err_msg}",
                              output_dir, state, lock, total)


MISTRAL_NATIVE_BASE_URL = "https://api.mistral.ai/v1"


def _build_mistral_batch_jsonl(
    entries: list[Tuple[Path, str, int, str]],
    max_tokens: int) -> str:
    """Mistral native batch JSONL: no `method`/`url` per line; endpoint/model set at job creation."""
    lines = []
    for _prompt_path, prompt_text, _prompt_idx, custom_id in entries:
        body = {
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
        }
        lines.append(json.dumps({"custom_id": custom_id, "body": body}))
    return "\n".join(lines)


def run_mistral_batch(
    entries: list[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    state: Dict[str, Any],
    lock: threading.Lock,
    total: int,
    cost_limit: float,
    poll_interval: int = 30) -> None:
    """Submit a batch to Mistral's native API (api.mistral.ai).

    Requires MISTRAL_API_KEY env var. The Azure Foundry project endpoint does
    not support batches; this routes to api.mistral.ai directly with the
    native batch shape (POST /v1/files, POST /v1/batch/jobs, polling).
    """
    import tempfile
    import time

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set. Mistral batch requires a direct "
            "api.mistral.ai key; Azure Foundry has no batch API for Mistral."
        )

    base_url = os.getenv("MISTRAL_BASE_URL", MISTRAL_NATIVE_BASE_URL).rstrip("/")
    native_model = os.getenv("MISTRAL_BATCH_MODEL", "mistral-large-latest")
    auth_headers = {"Authorization": f"Bearer {api_key}"}

    jsonl_text = _build_mistral_batch_jsonl(entries, max_output_tokens)
    tf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    tf.write(jsonl_text)
    tf.close()

    try:
        with open(tf.name, "rb") as fb:
            files = {"file": ("batch.jsonl", fb, "application/jsonl")}
            data = {"purpose": "batch"}
            r = requests.post(
                f"{base_url}/files",
                headers=auth_headers,
                files=files,
                data=data,
                timeout=300)
        r.raise_for_status()
        input_file_id = r.json()["id"]

        r = requests.post(
            f"{base_url}/batch/jobs",
            headers={**auth_headers, "Content-Type": "application/json"},
            json={
                "input_files": [input_file_id],
                "endpoint": "/v1/chat/completions",
                "model": native_model,
                "metadata": {"source": "FactoryBench"},
            },
            timeout=300)
        r.raise_for_status()
        job = r.json()
        job_id = job["id"]
        logger.info(
            f"[batch] Submitted Mistral batch={job_id} model={native_model} "
            f"with {len(entries)} requests"
        )
    finally:
        try:
            os.unlink(tf.name)
        except OSError:
            pass

    terminal = {"SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED"}
    while True:
        r = requests.get(
            f"{base_url}/batch/jobs/{job_id}",
            headers=auth_headers,
            timeout=60)
        r.raise_for_status()
        job = r.json()
        status = job.get("status")
        logger.info(
            f"[batch] Mistral batch={job_id}: status={status} | "
            f"total={job.get('total_requests', 0)} "
            f"succeeded={job.get('succeeded_requests', 0)} "
            f"failed={job.get('failed_requests', 0)}"
        )
        if status in terminal:
            break
        time.sleep(poll_interval)

    if status != "SUCCESS":
        logger.error(f"[batch] Mistral batch={job_id} ended status={status}")
        for entry in entries:
            _finalize_failure(
                entry, f"batch ended with status={status}",
                output_dir, state, lock, total)
        return

    output_file_id = job.get("output_file")
    if not output_file_id:
        logger.error(f"[batch] Mistral batch={job_id} succeeded but no output_file")
        for entry in entries:
            _finalize_failure(entry, "batch succeeded without output_file",
                              output_dir, state, lock, total)
        return

    r = requests.get(
        f"{base_url}/files/{output_file_id}/content",
        headers=auth_headers,
        timeout=600)
    r.raise_for_status()

    results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, Any] = {}
    for line in r.text.strip().splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        cid = data.get("custom_id")
        if cid is None:
            continue
        if data.get("error"):
            errors[cid] = data["error"]
            continue
        # Mistral wraps the response body similarly to OpenAI's batch results.
        resp = data.get("response") or {}
        body = resp.get("body") if isinstance(resp, dict) else None
        if body is None:
            # Alternative shape: the line itself is the chat completion body.
            body = {k: v for k, v in data.items() if k != "custom_id"}
        if body and body.get("choices"):
            results[cid] = body
        else:
            errors[cid] = body or {"message": "empty response"}

    error_file_id = job.get("error_file")
    if error_file_id:
        try:
            r = requests.get(
                f"{base_url}/files/{error_file_id}/content",
                headers=auth_headers,
                timeout=600)
            r.raise_for_status()
            for line in r.text.strip().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                cid = data.get("custom_id", "unknown")
                errors[cid] = data.get("error") or data
        except Exception as exc:
            logger.warning(f"[batch] Mistral error file fetch failed: {exc}")

    for entry in entries:
        _prompt_path, _prompt_text, _prompt_idx, custom_id = entry
        if custom_id in results:
            body = results[custom_id]
            answer = ""
            choices = body.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                answer = msg.get("content") or ""
            _finalize_success(
                entry, answer, body, model, output_dir,
                ground_truth_index, eval_level, judge_model,
                state, lock, total, cost_limit)
        elif custom_id in errors:
            err = errors[custom_id]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            _finalize_failure(entry, err_msg, output_dir, state, lock, total)
        else:
            _finalize_failure(entry, "missing from batch results",
                              output_dir, state, lock, total)


def _finalize_success(
    entry: Tuple[Path, str, int, str],
    answer: str,
    body: Dict[str, Any],
    model: str,
    output_dir: Path,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    state: Dict[str, Any],
    lock: threading.Lock,
    total: int,
    cost_limit: float) -> None:
    """Score, log to Opik, save the answer JSON, update counters."""
    prompt_path, prompt_text, prompt_idx, custom_id = entry
    out_path = output_dir / f"{custom_id}_answer.json"

    # ground_truth_index now holds the FULL question payload (not just the
    # answer) so we can read answer_format / template_type / acceptance_bounds
    # from it, prompt JSONs carry only {prompt, metadata}.
    qa_payload = ground_truth_index.get(prompt_path.stem) or {}
    if not isinstance(qa_payload, dict):
        qa_payload = {"answer": qa_payload}

    answer_format = infer_answer_format(qa_payload)
    question_type = str(qa_payload.get("template_type") or qa_payload.get("type") or "unknown")
    acceptance_bounds = qa_payload.get("acceptance_bounds")
    level_val = qa_payload.get("level")
    effective_eval_level = eval_level or (f"level_{level_val}" if level_val is not None else None)

    usage_raw = body.get("usage", {}) or {}
    prompt_tokens = int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0)
    completion_tokens = int(usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0)
    model_name = (body.get("model") or model).lower()
    est_cost = _estimate_cost(model_name, prompt_tokens, completion_tokens)

    gt = qa_payload.get("answer")
    score, judge_result = score_prediction(
        answer_format=answer_format,
        prediction=answer,
        ground_truth=gt,
        acceptance_bounds=acceptance_bounds,
        question_text=qa_payload.get("question", ""),
        judge_model=judge_model)

    log_to_opik(
        qa_payload=qa_payload,
        prompt_text=prompt_text,
        pred=answer,
        gt=gt,
        body=body,
        model_name=model_name,
        eval_level=effective_eval_level or "unknown",
        answer_format=answer_format,
        question_type=question_type,
        score=score,
        judge_result=judge_result,
        est_cost=est_cost,
        usage_raw=usage_raw,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens)

    save_json(out_path, {
        "custom_id": custom_id,
        "prompt_index": prompt_idx,
        "prompt_file": str(prompt_path),
        "prompt": prompt_text,
        "answer": answer,
        "ground_truth": gt,
        "score": score,
        "answer_format": answer_format,
        "question_type": question_type,
        "model": body.get("model") or model,
        "usage": body.get("usage"),
        "estimated_cost": est_cost,
        "raw_api_response": body,
    })
    with lock:
        state["total_cost"] += est_cost
        state["completed"] += 1
        done = state["completed"] + state["failed"] + state["skipped"]
        total_cost_now = state["total_cost"]
        if total_cost_now >= cost_limit and not state["stopped"]:
            state["stopped"] = True
            hit_limit = True
        else:
            hit_limit = False
    logger.info(
        f"✓ [{done}/{total}] {custom_id}"
        f" | cost: ${est_cost:.4f} | total: ${total_cost_now:.4f}"
    )
    if hit_limit:
        logger.warning(f"Cost limit ${cost_limit:.2f} reached (${total_cost_now:.4f}). No new tasks will start.")


def _finalize_failure(
    entry: Tuple[Path, str, int, str],
    error: Any,
    output_dir: Path,
    state: Dict[str, Any],
    lock: threading.Lock,
    total: int) -> None:
    prompt_path, prompt_text, prompt_idx, custom_id = entry
    fail_path = output_dir / f"{custom_id}_failed.json"
    save_json(fail_path, {
        "custom_id": custom_id,
        "prompt_index": prompt_idx,
        "prompt_file": str(prompt_path),
        "error": str(error),
    })
    with lock:
        state["failed"] += 1
        done = state["completed"] + state["failed"] + state["skipped"]
    logger.warning(f"✗ [{done}/{total}] Failed: {custom_id} ({error})")


def _process_entry(
    entry: Tuple[Path, str, int, str],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    state: Dict[str, Any],
    lock: threading.Lock,
    total: int,
    cost_limit: float) -> None:
    prompt_path, prompt_text, prompt_idx, custom_id = entry
    out_path = output_dir / f"{custom_id}_answer.json"
    fail_path = output_dir / f"{custom_id}_failed.json"

    if not overwrite and out_path.exists():
        with lock:
            state["skipped"] += 1
            done = state["completed"] + state["failed"] + state["skipped"]
        logger.info(f"- [{done}/{total}] Skipping existing result: {custom_id}")
        return

    if fail_path.exists():
        try:
            fail_path.unlink()
        except OSError:
            pass

    if len(prompt_text) // 4 > 900_000:
        with lock:
            state["skipped"] += 1
            done = state["completed"] + state["failed"] + state["skipped"]
        logger.warning(f"- [{done}/{total}] Skipping {custom_id}: prompt too large")
        return

    with lock:
        if state["stopped"]:
            state["skipped"] += 1
            return

    try:
        answer, body = call_model(model, prompt_text, max_output_tokens)
        _finalize_success(
            entry, answer, body, model, output_dir,
            ground_truth_index, eval_level, judge_model,
            state, lock, total, cost_limit)
    except Exception as exc:
        _finalize_failure(entry, exc, output_dir, state, lock, total)


class StrictBatchUnavailable(RuntimeError):
    """Raised in --strict-batch mode when the batch path cannot be used."""


def run_foundry_eval(
    entries: list[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    cost_limit: float,
    concurrency: int = 1,
    use_batch: bool = True,
    poll_interval: int = 30,
    strict_batch: bool = False) -> Tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {
        "completed": 0, "failed": 0, "skipped": 0,
        "total_cost": 0.0, "stopped": False,
    }
    lock = threading.Lock()
    total = len(entries)

    cfg = FOUNDRY_MODELS.get(model, {})
    supports_batch = bool(cfg.get("supports_batch", False))
    api_style = cfg.get("api_style", "openai")

    if strict_batch:
        if not use_batch:
            raise StrictBatchUnavailable(
                "--strict-batch is set but --no-batch was passed; choose one"
            )
        if not supports_batch:
            raise StrictBatchUnavailable(
                f"--strict-batch: model {model!r} does not declare supports_batch=True in src/config.py"
            )

    batch_fn = None
    skip_batch_reason: Optional[str] = None
    if use_batch and supports_batch:
        if api_style == "anthropic":
            endpoint_url, _, _ = resolve_endpoint(model)
            if "azure.com" in endpoint_url or "/anthropic/v" in endpoint_url:
                skip_batch_reason = (
                    "Azure's Anthropic proxy does not forward /messages/batches; "
                    "set an endpoint on api.anthropic.com to enable claude batch"
                )
            else:
                batch_fn = run_anthropic_batch
        else:
            # openai, deepseek, mistral, all OpenAI-compatible via /v1/batches.
            # Azure Foundry's project endpoint may not actually accept batch for
            # non-OpenAI models; if submission errors, the fallback kicks in
            # (unless strict_batch=True; then the exception propagates).
            batch_fn = run_openai_batch

    if strict_batch and batch_fn is None:
        raise StrictBatchUnavailable(
            f"--strict-batch: cannot route {model!r} to a batch endpoint"
            + (f", {skip_batch_reason}" if skip_batch_reason else "")
        )

    if use_batch and supports_batch and batch_fn is not None:
        pending: list[Tuple[Path, str, int, str]] = []
        for entry in entries:
            prompt_path, prompt_text, _prompt_idx, custom_id = entry
            out_path = output_dir / f"{custom_id}_answer.json"
            fail_path = output_dir / f"{custom_id}_failed.json"
            if not overwrite and out_path.exists():
                state["skipped"] += 1
                logger.info(f"- Skipping existing result: {custom_id}")
                continue
            if fail_path.exists():
                try:
                    fail_path.unlink()
                except OSError:
                    pass
            if len(prompt_text) // 4 > 900_000:
                state["skipped"] += 1
                logger.warning(f"- Skipping {custom_id}: prompt too large")
                continue
            pending.append(entry)

        if not pending:
            return state["completed"], state["failed"], state["skipped"]

        try:
            batch_fn(
                pending, model, output_dir, max_output_tokens,
                ground_truth_index, eval_level, judge_model,
                state, lock, total, cost_limit, poll_interval=poll_interval)
            return state["completed"], state["failed"], state["skipped"]
        except Exception as exc:
            if strict_batch:
                raise StrictBatchUnavailable(
                    f"--strict-batch: {model} batch submission failed ({exc}); "
                    f"refusing to fall back to sync"
                ) from exc
            logger.warning(
                f"[batch] {model} batch submission failed ({exc}); "
                f"falling back to concurrent sync."
            )
            # Batch path only writes on success; sync path below picks up pending.
            entries = pending
    elif skip_batch_reason:
        logger.info(f"[batch] {model}: skipping batch, {skip_batch_reason}")

    def _work(entry):
        _process_entry(
            entry, model, output_dir, max_output_tokens, overwrite,
            ground_truth_index, eval_level, judge_model,
            state, lock, total, cost_limit)

    if concurrency <= 1:
        for entry in entries:
            _work(entry)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_work, entry) for entry in entries]
            for _ in as_completed(futures):
                pass

    return state["completed"], state["failed"], state["skipped"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundry multi-model evaluation")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing prompt JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for replies")
    parser.add_argument("--questions", type=Path, required=True, help="Directory with Q&A ground truth JSONs")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(FOUNDRY_MODELS.keys()),
                        help="Foundry model to evaluate")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Model used for free-form scoring (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-judge entirely. Free-form items get score=None instead of being graded.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--total-prompts", type=int, default=None)
    parser.add_argument("--batch-number", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--eval-level", type=str, default=None)
    parser.add_argument("--cost-limit", type=float, default=20.0)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent API calls (default: 1)")
    parser.add_argument("--summary-file", type=Path, default=None,
                        help="Write a JSON summary {completed, failed, skipped} to this path")
    parser.add_argument("--use-batch", dest="use_batch", action="store_true", default=True,
                        help="Use provider batch APIs for models that support them (default)")
    parser.add_argument("--no-batch", dest="use_batch", action="store_false",
                        help="Disable batch APIs; force concurrent sync for all models")
    parser.add_argument("--strict-batch", action="store_true", default=False,
                        help="Require batch path: error out if the model can't use batch "
                             "or if batch submission fails. Never falls back to sync. "
                             "Use when you specifically want batch pricing/quotas (e.g. "
                             "GPT-5.1 globalbatch deployments) and would rather fail "
                             "loudly than silently spend on sync calls.")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Batch polling interval in seconds (default: 30)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    load_dotenv_file(args.env_file)

    endpoint, api_style, _api_version = resolve_endpoint(args.model)
    logger.info(f"Model={args.model} | api_style={api_style} | endpoint={endpoint}")

    # Full question payloads keyed by stem, needed so answer_format,
    # template_type, and acceptance_bounds flow into scoring.
    ground_truth_index = build_question_index(args.questions)
    logger.info(f"Loaded {len(ground_truth_index)} question payloads from {args.questions}")
    entries = load_prompt_entries(
        input_dir=args.input,
        total_prompts=args.total_prompts,
        batch_number=args.batch_number,
        batch_size=args.batch_size)
    if not entries:
        logger.error("No prompts to process")
        return

    effective_judge = "" if args.no_judge else args.judge_model
    if args.no_judge:
        logger.info("LLM-as-judge disabled (--no-judge); free-form items will get score=None")
    completed, failed, skipped = run_foundry_eval(
        entries=entries,
        model=args.model,
        output_dir=args.output_dir,
        max_output_tokens=args.max_output_tokens,
        overwrite=args.overwrite,
        ground_truth_index=ground_truth_index,
        eval_level=args.eval_level,
        judge_model=effective_judge,
        cost_limit=args.cost_limit,
        concurrency=args.concurrency,
        use_batch=args.use_batch,
        poll_interval=args.poll_interval,
        strict_batch=args.strict_batch)
    logger.info(
        f"Done. Completed={completed}, Failed={failed}, Skipped={skipped}, OutputDir={args.output_dir}"
    )
    if args.summary_file is not None:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_file, "w") as f:
            json.dump(
                {"completed": completed, "failed": failed, "skipped": skipped,
                 "model": args.model, "output_dir": str(args.output_dir)},
                f)


if __name__ == "__main__":
    main()
