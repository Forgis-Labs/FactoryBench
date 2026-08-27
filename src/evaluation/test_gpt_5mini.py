#!/usr/bin/env python3
"""
Generate answers for prompt files using direct OpenAI Responses API calls.

This script:
1. Loads pre-generated prompt JSON files from a folder
2. Calls the model per prompt (Responses endpoint)
3. Saves one output JSON per prompt with the generated answer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None

logger = logging.getLogger(__name__)


_TENSOR_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_TENSOR_BRACKET_RE = re.compile(r"\[\s*[^\[\]]*?-?\d[^\[\]]*?\]")


def _parse_tensor_answer(value: Any, expected_len: Optional[int] = None) -> list:
    """Parse a tensor answer; prefer the LAST ``[...]`` block matching the
    expected length, then fall back to the LAST ``expected_len`` numbers in
    the text. Avoids treating reasoning numbers (timestamps, intermediate
    values) as the answer when the model emits prose around its tensor.
    """
    if value is None:
        raise ValueError("empty tensor")
    s = str(value).strip()
    if not s:
        raise ValueError("empty tensor")

    if expected_len is not None:
        for seg in reversed(_TENSOR_BRACKET_RE.findall(s)):
            nums = _TENSOR_NUM_RE.findall(seg)
            if len(nums) == expected_len:
                return [float(x) for x in nums]
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
    """Parse a numerical answer; on float() failure, fall back to the last
    numerical token in the text (claude often wraps its answer like ``**1810**``
    after reasoning, despite being told to return only a number).
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


def load_dotenv_file(env_file: Path) -> None:
    if not env_file.exists() or not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key(cli_api_key: Optional[str]) -> str:
    if cli_api_key:
        return cli_api_key

    key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key. Provide --api-key or set OPENAI_API_KEY/AZURE_OPENAI_API_KEY in .env"
        )
    return key


def create_client_and_model(
    cli_api_key: Optional[str],
    cli_model: Optional[str]) -> tuple[Any, str, str]:
    """
    Returns (client, model, provider), where provider is 'openai' or 'azure'.
    """

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
    # Use AZURE_OPENAI_CHAT_DEPLOYMENT as the primary model/deployment variable for Azure
    azure_model = (
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )

    if azure_endpoint and (azure_api_key or cli_api_key):
        key = cli_api_key or azure_api_key
        client = AzureOpenAI(
            api_key=key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint)
        model = cli_model or azure_model or "gpt-5.1"
        return client, model, "azure"

    key = resolve_api_key(cli_api_key)
    client = OpenAI(api_key=key)
    model = os.getenv("OPENAI_MODEL") or cli_model or "gpt-5.1"
    return client, model, "openai"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_ground_truth_index(questions_dir: Optional[Path]) -> Dict[str, Any]:
    """Build a mapping of filename stem -> answer from a questions directory."""
    if questions_dir is None or not questions_dir.is_dir():
        return {}
    index: Dict[str, Any] = {}
    for path in questions_dir.rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and "answer" in payload:
            index[path.stem] = payload["answer"]
    logger.info(f"Loaded {len(index)} ground truth entries from {questions_dir}")
    return index


def load_prompt_entries(
    input_dir: Path,
    total_prompts: Optional[int] = None,
    batch_number: int = 0,
    batch_size: int = 1000) -> list[Tuple[Path, str, int, str]]:
    """
    Load pre-generated prompts and return selected slice entries.

    total_prompts:
      Total number of prompts to process across all batches. If None, processes all prompts.
    batch_number:
      Which batch this is (0-indexed), where each batch is batch_size prompts.

    Returns:
      List of tuples (prompt_path, prompt_text, global_index, custom_id).
    """
    prompt_files = sorted(input_dir.rglob("*.json"))
    if not prompt_files:
        logger.warning(f"No prompt JSON files found in {input_dir}")
        return []

    prompt_entries: list[tuple[Path, str]] = []
    for prompt_path in prompt_files:
        try:
            payload = load_json(prompt_path)
        except Exception:
            logger.warning(f"Skipping unreadable JSON: {prompt_path}")
            continue

        if not isinstance(payload, dict):
            logger.warning(f"Skipping non-object JSON: {prompt_path}")
            continue

        prompt_text = payload.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            logger.warning(f"Skipping JSON without 'prompt' text: {prompt_path}")
            continue

        prompt_entries.append((prompt_path, prompt_text.strip()))

    if not prompt_entries:
        logger.warning(f"No valid prompt entries found in {input_dir}")
        return []

    batch_start = batch_number * batch_size
    batch_end = (batch_number + 1) * batch_size

    if total_prompts is not None:
        prompt_entries = prompt_entries[:total_prompts]

    logger.info(f"Loaded {len(prompt_entries)} prompt entries from {input_dir}")

    selected_entries = prompt_entries[batch_start:batch_end]

    selected: list[Tuple[Path, str, int, str]] = []
    for prompt_idx, (prompt_path, prompt_text) in enumerate(selected_entries, start=batch_start):
        custom_id = f"{prompt_path.stem}_{prompt_idx}"
        selected.append((prompt_path, prompt_text, prompt_idx, custom_id))

    logger.info(f"Selected {len(selected)} prompts for processing")
    return selected


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def _extract_output_text_from_responses_body(body: Dict[str, Any]) -> str:
    """
    Responses API format: body["output"] is a list of items.
    We concatenate any output_text segments found inside message items.
    """
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a robotics sensor-data Q&A benchmark.
Your task is to score a model's free-form answer against a reference answer.

Scoring criteria (0-10):
  10 – Answer is semantically equivalent to the reference: correct direction of change, correct signal, correct magnitude range, correct time window.
   7 – Answer captures the main trend and signal correctly but is imprecise on magnitude or timing.
   4 – Answer mentions the right signal but the described behaviour is partially incorrect or vague.
   1 – Answer is on-topic but mostly incorrect or misleading.
   0 – Answer is completely wrong, irrelevant, or refuses to answer.

Respond ONLY with a JSON object in this exact format (no extra text):
{"score": <integer 0-10>, "reason": "<one sentence justification>"}"""


def llm_judge_score(
    client: Any,
    model: str,
    question: str,
    prediction: str,
    reference: str,
    max_tokens: int = 256) -> tuple[float, str]:
    """
    Ask the LLM to score `prediction` against `reference` for the given `question`.
    Returns (normalised_score 0.0–1.0, justification_string).
    Falls back to (0.0, error_message) on any failure.
    """
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model answer: {prediction}\n\n"
        "Please score the model answer (0-10) and provide a one-sentence justification."
    )
    try:
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                       {"role": "user",   "content": user_msg}],
                max_output_tokens=max_tokens)
            raw = _extract_output_text_from_responses_body(_to_dict(response))
        except Exception:
            # Fallback to Chat Completions
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=max_tokens)
            raw = _to_dict(response).get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse the JSON response from the judge
        raw = raw.strip()
        # Sometimes the model wraps in markdown code fences
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



def parse_llm_answer(text):
    # search last A,B,C,D and return it
    # the model use to answer at the end of the answer something like "The correct answer is C."
    matches = re.findall(r'\b([A-D])\b', text.upper())
    return matches[-1] if matches else None

def _estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a single call given token counts.

    Prices are USD per 1M tokens (input, output) sourced from each vendor's
    published list price (Azure-hosted prices match the vendor list for
    OpenAI, Anthropic and Mistral; open-weight models use the median Together /
    Fireworks rate at the matched parameter scale). The table is checked in
    order, so more-specific patterns must come before more-generic ones.
    Forecasting foundation models served from local checkpoints incur no API
    fee and are recorded at zero cost.
    """
    model_name = model_name.lower()

    pricing = [
        # OpenAI
        ("gpt-4o-mini",       0.15,   0.60),
        ("gpt-5.1",           1.25,  10.00),
        ("gpt-5-mini",        0.25,   2.00),
        ("gpt-5",             1.25,  10.00),
        ("gpt-4o",            2.50,  10.00),
        ("o1-mini",           3.00,  12.00),
        ("o1-preview",       15.00,  60.00),
        ("gpt-4-turbo",      10.00,  30.00),
        ("gpt-4",            30.00,  60.00),
        ("gpt-3.5-turbo",     0.50,   1.50),
        # Anthropic Claude (4.x family)
        ("claude-haiku-4-5",  1.00,   5.00),
        ("claude-haiku-4",    1.00,   5.00),
        ("claude-sonnet-4",   3.00,  15.00),
        ("claude-opus-4",    15.00,  75.00),
        ("claude-3.5-haiku",  0.80,   4.00),
        ("claude-3.5-sonnet", 3.00,  15.00),
        ("claude-3-opus",    15.00,  75.00),
        ("claude-haiku",      1.00,   5.00),
        ("claude-sonnet",     3.00,  15.00),
        ("claude-opus",      15.00,  75.00),
        ("haiku",             1.00,   5.00),
        # DeepSeek (cache-miss list price; cache-hit input is ~$0.07)
        ("deepseek-v3.1",     0.27,   1.10),
        ("deepseek-v3",       0.27,   1.10),
        ("deepseek",          0.27,   1.10),
        # Mistral
        ("mistral-large-3",   0.50,   1.50),
        ("mistral-large",     2.00,   6.00),
        ("mistral-medium",    0.40,   2.00),
        ("mistral-small",     0.20,   0.60),
        # Qwen. The 235B moved from a self-hosted Together endpoint to Vertex
        # Model Garden MaaS, which bills per token, the old blanket ("qwen",
        # 0, 0) rule silently reported $0 for every 235B call after that
        # migration. These are order-of-magnitude MaaS rates and are NOT
        # vendor-confirmed; check the Vertex pricing page before quoting them
        # in the paper. The 4B stays at zero because it is served from a
        # self-deployed endpoint billed by the GPU-hour, not per token.
        ("qwen-3-235b",       0.30,   1.20),
        ("qwen3-235b",        0.30,   1.20),
        ("qwen",              0.00,   0.00),
        # Time-series foundation models served from local checkpoints
        ("chronos",           0.00,   0.00),
        ("moirai",            0.00,   0.00),
        # Generic small-tier OpenAI-style fallback
        ("mini",              0.15,   0.60),
    ]
    for pattern, p_in, p_out in pricing:
        if pattern in model_name:
            return (prompt_tokens / 1_000_000) * p_in + (completion_tokens / 1_000_000) * p_out

    # Unknown model: log a warning and use a conservative default so the
    # cost-limit guard still trips at a sensible spend.
    logger.warning(
        "_estimate_cost: unknown model %r, using fallback rate $5/$15 per 1M tokens",
        model_name)
    return (prompt_tokens / 1_000_000) * 5.0 + (completion_tokens / 1_000_000) * 15.0


def run_direct_requests(
    entries: list[Tuple[Path, str, int, str]],
    client: Any,
    output_dir: Path,
    model: str,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    cost_limit: float = 40.0) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0
    skipped = 0
    total_cost = 0.0

    for prompt_path, prompt_text, prompt_idx, custom_id in entries:
        out_path = output_dir / f"{custom_id}_answer.json"
        fail_path = output_dir / f"{custom_id}_failed.json"

        if not overwrite and (out_path.exists() or fail_path.exists()):
            skipped += 1
            logger.info(f"- Skipping existing result: {custom_id}")
            continue


        # Token budget guard – skip prompts that are too large for the model
        estimated_tokens = len(prompt_text) // 4
        if estimated_tokens > 900_000:
            logger.warning(
                f"- Skipping {custom_id}: estimated {estimated_tokens} tokens exceeds budget"
            )
            skipped += 1
            continue

        # Load full Q&A object for scoring (not just answer)
        try:
            qa_payload = load_json(prompt_path)
        except Exception:
            qa_payload = {}

        answer_format = str(qa_payload.get("answer_format") or "unknown")
        question_type = str(qa_payload.get("type") or "unknown")
        acceptance_bounds = qa_payload.get("acceptance_bounds")

        # Derive eval level from the Q&A JSON if not provided via CLI
        level_val = qa_payload.get("level")
        effective_eval_level = eval_level or (f"level_{level_val}" if level_val is not None else None)

        try:
            try:
                response = client.responses.create(
                    model=model,
                    input=prompt_text,
                    max_output_tokens=max_output_tokens)
                body = _to_dict(response)
                answer = _extract_output_text_from_responses_body(body)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower() or not hasattr(client, "responses"):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        max_tokens=max_output_tokens)
                    body = _to_dict(response)
                    answer = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    raise e

            # --- Cost tracking ---
            usage_raw = body.get("usage", {}) or {}
            prompt_tokens: int = int(
                usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0
            )
            completion_tokens: int = int(
                usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0
            )
            model_name = (body.get("model") or model).lower()
            est_cost = _estimate_cost(model_name, prompt_tokens, completion_tokens)
            total_cost += est_cost

            # --- Scoring logic ---
            score = None
            llm_judge_result: tuple[float, str] | None = None
            gt = ground_truth_index.get(prompt_path.stem)
            pred = answer
            try:
                if answer_format == "free_form":
                    # LLM-as-a-judge: call the model to score the free-form answer
                    question_text = qa_payload.get("question", "")
                    ref_answer = str(gt) if gt is not None else ""
                    judge_score, judge_reason = llm_judge_score(
                        client=client,
                        model=model,
                        question=question_text,
                        prediction=str(pred) if pred is not None else "",
                        reference=ref_answer)
                    score = judge_score
                    llm_judge_result = (judge_score, judge_reason)
                elif answer_format in ("numerical", "tensor"):
                    # Numerical: float, Tensor: underscore-separated floats
                    if answer_format == "numerical":
                        try:
                            gt_val = round(float(gt), 4)
                            pred_val = round(_parse_numerical_answer(pred), 4)
                        except Exception:
                            score = 0
                        else:
                            # Three-level piecewise for margin-bounded items (mirrors tensor).
                            # min/max bounds stay binary (no native "2x margin" zone).
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
                                score = int(abs(pred_val - gt_val) < 1e-4)
                    elif answer_format == "tensor":
                        try:
                            gt_vals = _parse_tensor_answer(gt)
                            pred_vals = _parse_tensor_answer(pred, expected_len=len(gt_vals))
                        except Exception:
                            score = 0
                        else:
                            # Three-level piecewise: 1 within margin, 0.5 within 2x margin,
                            # 0 otherwise. With margins calibrated to R_j/12, the chance level
                            # is 1/4 under uniform random in the channel's natural range.
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
                    # Multi-select MCQ: string of T/F, e.g., TFFT
                    gt_str = str(gt).strip().upper()
                    pred_str = _parse_mcms_answer(pred, len(gt_str))
                    # Previous scheme (commented): tiered 1.0 / 0.5 / 0.0
                    # (all-correct, off-by-one, otherwise zero). Now:
                    # positional fraction so each correct T/F slot is 1/n.
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
                    # Single-select MCQ: ground truth is a letter like "C",
                    # model may answer "C", "C.", "C. some explanation", etc.
                    gt_str = str(gt).strip().upper()
                    pred_str = str(pred).strip()

                    pred_letter = parse_llm_answer(pred_str)
                    if pred_letter is None:
                        score = 0.0
                    else:
                        score = float(gt_str == pred_letter)

                elif answer_format == "ranking":
                    # Ranking: permutation of A-D, e.g., DCAB
                    gt_str = str(gt).strip().upper()
                    pred_raw = str(pred).strip().upper()

                    # Extract only alphabetic characters from the prediction
                    match = re.search(r'\b([A-D]{4})\b', pred_raw)
                    pred_str = match.group(1) if match else ""

                    # Previous scheme (commented): exact match only.
                    # Now: positional fraction.
                    # score = float(gt_str == pred_str)
                    if pred_str and len(pred_str) == len(gt_str):
                        n = len(gt_str)
                        n_correct = sum(g == p for g, p in zip(gt_str, pred_str))
                        score = n_correct / n
                    else:
                        score = 0.0

                else:
                    if answer_format == "llm_judge":
                        # Already handled above; this branch won't be reached
                        pass
                    else:
                        # Fallback: exact match
                        score = float(str(gt) == str(pred))
            except Exception:
                score = None

            # --- Opik Tracing ---
            try:
                if os.getenv("OPIK_API_KEY"):
                    import opik
                    client_opik = opik.Opik(
                        project_name=os.getenv("OPIK_PROJECT_NAME", "FactoryBench"),
                        workspace=os.getenv("OPIK_WORKSPACE", "anon-workspace")
                    )
                    
                    # prompt_tokens, completion_tokens, model_name, est_cost already computed above
                    total_tokens: int = int(usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens))

                    metadata_payload = qa_payload.get("metadata") or {}
                    dataset_tag = metadata_payload.get("dataset")
                    episode_tag = metadata_payload.get("episode")
                    qa_pair_id = metadata_payload.get("qa_pair_id")
                    
                    # Filter out None/empty values so Opik receives only real tags
                    opik_tags = [str(t) for t in [effective_eval_level, question_type, dataset_tag, answer_format] if t]

                    type_tag = metadata_payload.get("type")
                    model_tag = model_name

                    if type_tag and type_tag != "unknown":
                        opik_tags.append(str(type_tag))

                    if answer_format and answer_format != "unknown":
                        opik_tags.append(f"eval_{answer_format}")
                    
                    if model_tag and model_tag != "unknown":
                        opik_tags.append(model_tag)

                    if dataset_tag and dataset_tag != "unknown":
                        opik_tags.append(dataset_tag)

                    if qa_pair_id and qa_pair_id != "unknown":
                        opik_tags.append("template_" + str(qa_pair_id))


                    opik_metadata = {
                        "model": model_tag,
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
                            "total_estimated_cost": f"${est_cost}"
                        },
                        "full_telemetry_context": {"text": prompt_text}
                    }


                    trace = client_opik.trace(
                        name=f"factorybench_{effective_eval_level}_{qa_pair_id}",
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
                        model=body.get("model") or model_name
                    )
                    if score is not None:
                        pred_str_repr = str(pred).strip() if pred is not None else "N/A"
                        gt_str_repr   = str(gt).strip()   if gt   is not None else "N/A"
                        # For llm_judge questions, use the judge's reason as the primary accuracy reason
                        accuracy_reason = judge_reason if (llm_judge_result and judge_reason) else (
                            f"Predicted: '{pred_str_repr}' | "
                            f"Ground truth: '{gt_str_repr}'"
                        )

                        # Accuracy
                        trace.log_feedback_score(
                            name="accuracy",
                            value=float(score),
                            reason=accuracy_reason)

                        # LLM-as-a-judge score
                        if llm_judge_result is not None:
                            judge_value, judge_reason = llm_judge_result
                            trace.log_feedback_score(
                                name="llm_judge",
                                value=judge_value,
                                reason=judge_reason or accuracy_reason)

            except Exception as e:
                logger.warning(f"Opik logging failed: {e}")


            save_json(out_path, {
                "custom_id": custom_id,
                "prompt_index": prompt_idx,
                "prompt_file": str(prompt_path),
                "prompt": prompt_text,
                "answer": answer,
                "ground_truth": gt,
                "score": score,
                "llm_judge_score": llm_judge_result[0] if llm_judge_result else None,
                "llm_judge_reason": llm_judge_result[1] if llm_judge_result else None,
                "answer_format": answer_format,
                "question_type": question_type,
                "model": body.get("model"),
                "usage": body.get("usage"),
                "estimated_cost": est_cost if 'est_cost' in locals() else None,
                "raw_api_response": body,
            })
            completed += 1
            logger.info(
                f"✓ [{completed + failed + skipped}/{len(entries)}] Saved answer: {custom_id}"
                f" | cost this call: ${est_cost:.4f} | run total: ${total_cost:.4f}"
            )
            if total_cost >= cost_limit:
                logger.warning(
                    f"Cost limit of ${cost_limit:.2f} reached (${total_cost:.4f} spent). Stopping."
                )
                return completed, failed, skipped
        except Exception as exc:
            failed += 1
            save_json(fail_path, {
                "custom_id": custom_id,
                "prompt_index": prompt_idx,
                "prompt_file": str(prompt_path),
                "error": str(exc),
            })
            logger.warning(f"✗ [{completed + failed + skipped}/{len(entries)}] Failed: {custom_id} ({exc})")

    return completed, failed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers via direct OpenAI Responses API calls")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/prompts/level3"),
        help="Directory containing pre-generated prompt JSON files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/replies/level3"),
        help="Directory to save answer outputs")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (overrides .env)")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env)")
    parser.add_argument("--model", type=str, default=None, help="Model name (default from env or gpt-5.1)")
    parser.add_argument("--max-output-tokens", type=int, default=2000, help="Max output tokens")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result files")
    parser.add_argument(
        "--total-prompts",
        type=int,
        default=None,
        help="Total number of prompts to process across batches")
    parser.add_argument(
        "--batch-number",
        type=int,
        default=0,
        help="Which batch slice to process (0-indexed)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="How many prompts per batch slice")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Directory containing Q&A pair JSON files (for ground truth lookup)")
    parser.add_argument(
        "--eval-level",
        type=str,
        default=None,
        help="Evaluation level tag for Opik (e.g. level_1)")
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=20.0,
        help="Maximum USD spend per run (default: $20.00). Stops after the limit is reached.")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    load_dotenv_file(args.env_file)
    client, model, provider = create_client_and_model(args.api_key, args.model)
    logger.info(f"Using provider={provider}, model={model}")

    if args.questions is None:
        logger.error("--questions not provided; all prompts will be skipped. Pass the Q&A directory to enable ground truth lookup.")
        return

    ground_truth_index = build_ground_truth_index(args.questions)

    entries = load_prompt_entries(
        input_dir=args.input,
        total_prompts=args.total_prompts,
        batch_number=args.batch_number,
        batch_size=args.batch_size)

    if not entries:
        logger.error("No requests to process")
        return

    if args.total_prompts is not None:
        total_batches = (args.total_prompts + args.batch_size - 1) // args.batch_size
        logger.info(f"Processing batch {args.batch_number}/{total_batches - 1}")
        logger.info(f"This slice contains {len(entries)} prompts")

    completed, failed, skipped = run_direct_requests(
        entries=entries,
        client=client,
        output_dir=args.output_dir,
        model=model,
        max_output_tokens=args.max_output_tokens,
        overwrite=args.overwrite,
        ground_truth_index=ground_truth_index,
        eval_level=args.eval_level,
        cost_limit=args.cost_limit)
    logger.info(
        f"Done. Completed={completed}, Failed={failed}, Skipped={skipped}, OutputDir={args.output_dir}"
    )


if __name__ == "__main__":
    main()