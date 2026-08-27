#!/usr/bin/env python3
"""AWS evaluation entrypoint: dispatches Bedrock or SageMaker per model.

Provider routing is read from ``src.config``. Region is resolved per-model
(set via the model's ``region_env`` field) because availability differs:

  * Bedrock models  → batch via ``CreateModelInvocationJob`` (S3-in / S3-out),
                     fallback to sync ``invoke_model`` on validation errors.
                     Each model's ``bedrock`` / ``bedrock-runtime`` client is
                     constructed with the model-specific region.
  * SageMaker models → ``InvokeEndpointAsync`` against a pre-deployed JumpStart
                     endpoint configured for asynchronous inference and
                     scale-to-zero. Each call writes input to S3, returns an
                     output S3 location, which we poll until the result lands.

Reply JSON shape, ground-truth lookup, scoring, and Opik logging are reused
verbatim from ``run_foundry_eval`` so this module is drop-in compatible.

Required environment variables (full setup walkthrough in
``src/evaluation/aws-setup.md``):

  Per-model id + region (Bedrock):
    CLAUDE_SONNET_46_MODEL_ID   CLAUDE_SONNET_46_REGION
    MISTRAL_LARGE_3_MODEL_ID    MISTRAL_LARGE_3_REGION
    DEEPSEEK_V32_MODEL_ID       DEEPSEEK_V32_REGION

  Per-model endpoint + region (SageMaker async):
    QWEN_SAGEMAKER_ENDPOINT     QWEN_SAGEMAKER_REGION

  Shared:
    FB_S3_BUCKET                S3 bucket for batch + async I/O
    FB_S3_PREFIX                key prefix (default: factorybench/)
    BEDROCK_BATCH_ROLE_ARN      role for CreateModelInvocationJob
    SAGEMAKER_ROLE_ARN          role for SageMaker calls

Usage:
    python -m src.evaluation.run_aws_eval \\
        --input output/prompts/level1 \\
        --output-dir output/replies/level1/claude-sonnet-4_6 \\
        --questions output/questions/level1 \\
        --model claude-sonnet-4.6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    AWS_REGION_DEFAULT,
    BEDROCK_MODELS,
    DEFAULT_JUDGE_MODEL,
    SAGEMAKER_MODELS,
    get_provider)

# Reuse existing helpers (scoring / GT index / IO / reply finalisation)
from src.evaluation.run_foundry_eval import (
    _estimate_cost,
    build_question_index,
    infer_answer_format,
    score_prediction)

# Bedrock batch is 50% cheaper than on-demand inference. We apply this to the
# pre-flight estimator so the cost-limit gate compares apples-to-apples with
# the actual job pricing. Sync fallback uses on-demand rates (multiplier 1.0).
BEDROCK_BATCH_PRICE_MULTIPLIER: float = 0.5
BEDROCK_SYNC_PRICE_MULTIPLIER: float = 1.0
# Bedrock CreateModelInvocationJob requires at least this many records per
# job (AWS-side minimum, not configurable).
BEDROCK_BATCH_MIN_RECORDS: int = 100
# Rough char-to-token approximation used only for pre-flight cost estimation
# (real billing is done from the API's reported usage). 1 token ≈ 4 chars.
_CHARS_PER_TOKEN = 4
from src.evaluation.test_gpt_5mini import (
    load_dotenv_file,
    load_json,
    load_prompt_entries,
    save_json)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy boto3 import (so non-AWS users can still import this module)
# ---------------------------------------------------------------------------
def _boto3():
    try:
        import boto3  # type: ignore
        return boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AWS evaluation. Install with `pip install boto3`."
        ) from exc


def _model_region(cfg: Dict[str, Any]) -> str:
    """Resolve a model's region from its ``region_env`` field, falling back to
    ``AWS_REGION`` and finally the configured default. Per-model regions matter
    because Mistral Large 3 / DeepSeek V3.1 are not available in eu-central-1
    (Mistral has no EU region; DeepSeek lives in eu-west-2 / eu-north-1)."""
    env = cfg.get("region_env")
    if env:
        val = os.environ.get(env)
        if val:
            return val
    return os.environ.get("AWS_REGION") or AWS_REGION_DEFAULT


def _s3_bucket() -> str:
    bucket = os.environ.get("FB_S3_BUCKET")
    if not bucket:
        raise RuntimeError(
            "FB_S3_BUCKET env var is required for batch I/O (any S3 bucket the "
            "Bedrock/SageMaker role can read+write)."
        )
    return bucket


def _bucket_for_region(region: str) -> str:
    """Resolve the S3 bucket for batch I/O in a specific region.

    Bedrock batch (CreateModelInvocationJob) requires the input/output bucket
    to be in the SAME region as the Bedrock job, Mistral runs in us-west-2
    and DeepSeek in eu-west-2, so a single eu-central-1 bucket can't serve all
    three Bedrock models for batch.

    Resolution order:
      1. ``FB_S3_BUCKET_<REGION>`` (e.g. FB_S3_BUCKET_US_WEST_2)
      2. ``FB_S3_BUCKET`` (the cross-region default, used when its region matches)

    If only the global bucket is set and the model's region differs, batch
    submission will fail at AWS with a region-mismatch error and the dispatcher
    auto-falls-back to sync. To enable batch on every model, set one
    ``FB_S3_BUCKET_<REGION>`` per Bedrock region in use.
    """
    region_var = "FB_S3_BUCKET_" + region.upper().replace("-", "_")
    val = os.environ.get(region_var)
    if val:
        return val
    val = os.environ.get("FB_S3_BUCKET")
    if not val:
        raise RuntimeError(
            f"No S3 bucket configured for region {region!r}. Set either "
            f"{region_var} (recommended for batch) or the global FB_S3_BUCKET."
        )
    return val


def _s3_prefix() -> str:
    return os.environ.get("FB_S3_PREFIX", "factorybench/").rstrip("/") + "/"


def _estimate_prompt_tokens(prompt_text: str) -> int:
    """Cheap upper-ish estimate: 1 token per ~4 chars. Used only for pre-flight
    cost gating; final billed cost reflects the API's reported usage."""
    return max(1, len(prompt_text) // _CHARS_PER_TOKEN)


def estimate_batch_cost(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    max_output_tokens: int,
    price_multiplier: float) -> Tuple[float, int, int]:
    """Pre-flight worst-case cost in USD (sum over entries) plus token totals.

    Worst case = each entry uses its full prompt as input and produces
    ``max_output_tokens`` completions. The Bedrock batch discount is applied
    via ``price_multiplier``. Real cost will usually be lower because most
    answers are far shorter than max_output_tokens.
    """
    total_in = sum(_estimate_prompt_tokens(text) for _p, text, _i, _c in entries)
    total_out = max_output_tokens * len(entries)
    cost = _estimate_cost(model.lower(), total_in, total_out) * price_multiplier
    return cost, total_in, total_out


def _enforce_cost_limit(
    cost_limit: Optional[float],
    estimated: float,
    in_tokens: int,
    out_tokens: int,
    model: str,
    flow: str) -> None:
    """Raise RuntimeError if the pre-flight estimate exceeds ``cost_limit``."""
    logger.info(
        f"[{flow}] {model}: pre-flight estimate "
        f"~${estimated:.2f} (worst-case: {in_tokens:,} in + {out_tokens:,} out tokens)"
    )
    if cost_limit is not None and estimated > cost_limit:
        raise RuntimeError(
            f"Pre-flight cost estimate ${estimated:.2f} exceeds --cost-limit ${cost_limit:.2f} "
            f"for {model} ({flow}). Lower --max-output-tokens, reduce the input set, or "
            f"raise --cost-limit."
        )


def _resolve_bedrock_model_id(model_name: str) -> str:
    cfg = BEDROCK_MODELS[model_name]
    env = cfg.get("model_id_env")
    val = os.environ.get(env) if env else None
    if not val:
        raise RuntimeError(
            f"Set {env} to the Bedrock model id for {model_name!r} "
            f"(see src/evaluation/aws-setup.md for verified ids)."
        )
    return val


def _resolve_sagemaker_endpoint(model_name: str) -> str:
    """Resolve the SageMaker endpoint name for an async-inference model."""
    cfg = SAGEMAKER_MODELS[model_name]
    env = cfg.get("endpoint_env")
    val = os.environ.get(env) if env else None
    if not val:
        raise RuntimeError(
            f"Set {env} to the SageMaker endpoint name for {model_name!r} "
            f"(deploy the JumpStart model with Async Inference + scale-to-zero "
            f"and copy the endpoint name; see src/evaluation/aws-setup.md)."
        )
    return val


# ---------------------------------------------------------------------------
# Provider-specific request body builders (Bedrock direct invocation format)
# ---------------------------------------------------------------------------
def _bedrock_body_anthropic(prompt: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _bedrock_body_mistral(prompt: str, max_tokens: int) -> Dict[str, Any]:
    # Mistral on Bedrock uses the chat-completions style for Large 2/3.
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def _bedrock_body_deepseek(prompt: str, max_tokens: int) -> Dict[str, Any]:
    # DeepSeek on Bedrock Marketplace typically exposes an OpenAI-style chat API.
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def _build_bedrock_body(api_style: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    if api_style == "anthropic":
        return _bedrock_body_anthropic(prompt, max_tokens)
    if api_style == "mistral":
        return _bedrock_body_mistral(prompt, max_tokens)
    if api_style == "deepseek":
        return _bedrock_body_deepseek(prompt, max_tokens)
    raise ValueError(f"Unknown bedrock api_style: {api_style!r}")


# ---------------------------------------------------------------------------
# Provider-specific output extractors
# ---------------------------------------------------------------------------
def _extract_anthropic(out: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    chunks = out.get("content") or []
    text = "".join(c.get("text", "") for c in chunks if c.get("type") == "text")
    usage = out.get("usage", {}) or {}
    return text.strip(), {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0)),
    }


def _extract_chat(out: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """OpenAI-style chat output (Mistral / DeepSeek on Bedrock)."""
    choices = out.get("choices") or out.get("outputs") or []
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        text = msg.get("content") or first.get("text") or ""
    else:
        text = out.get("output_text") or ""
    usage = out.get("usage", {}) or {}
    return str(text).strip(), {
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _extract_tgi(out: Any) -> Tuple[str, Dict[str, Any]]:
    """HuggingFace TGI / vLLM output: list[{'generated_text': '...'}]."""
    if isinstance(out, list) and out:
        text = out[0].get("generated_text", "")
    elif isinstance(out, dict):
        text = out.get("generated_text") or ""
    else:
        text = str(out or "")
    return str(text).strip(), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _extract_output(api_style: str, raw: Any) -> Tuple[str, Dict[str, Any]]:
    if api_style == "anthropic":
        return _extract_anthropic(raw)
    if api_style in ("mistral", "deepseek", "openai-chat"):
        return _extract_chat(raw)
    if api_style == "tgi":
        return _extract_tgi(raw)
    raise ValueError(f"Unknown api_style for output extraction: {api_style!r}")


# ---------------------------------------------------------------------------
# Reply writer (mirrors run_foundry_eval._finalize_success)
# ---------------------------------------------------------------------------
def _write_reply(
    entry: Tuple[Path, str, int, str],
    answer: str,
    raw_body: Dict[str, Any],
    usage: Dict[str, Any],
    model: str,
    output_dir: Path,
    ground_truth_index: Dict[str, Any],
    judge_model: str) -> Tuple[float, Optional[float]]:
    prompt_path, prompt_text, prompt_idx, custom_id = entry
    out_path = output_dir / f"{custom_id}_answer.json"

    qa_payload = ground_truth_index.get(prompt_path.stem) or {}
    if not isinstance(qa_payload, dict):
        qa_payload = {"answer": qa_payload}

    answer_format = infer_answer_format(qa_payload)
    question_type = str(qa_payload.get("template_type") or qa_payload.get("type") or "unknown")
    acceptance_bounds = qa_payload.get("acceptance_bounds")
    gt = qa_payload.get("answer")

    score, judge_result = score_prediction(
        answer_format=answer_format,
        prediction=answer,
        ground_truth=gt,
        acceptance_bounds=acceptance_bounds,
        question_text=qa_payload.get("question", ""),
        judge_model=judge_model)

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    est_cost = _estimate_cost(model.lower(), prompt_tokens, completion_tokens)

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
        "model": model,
        "usage": usage,
        "estimated_cost": est_cost,
        "raw_api_response": raw_body,
    })
    return est_cost, score


def _write_failure(entry: Tuple[Path, str, int, str], error: str, output_dir: Path) -> None:
    prompt_path, _prompt_text, prompt_idx, custom_id = entry
    save_json(output_dir / f"{custom_id}_failed.json", {
        "custom_id": custom_id,
        "prompt_index": prompt_idx,
        "prompt_file": str(prompt_path),
        "error": error,
    })


# ---------------------------------------------------------------------------
# Bedrock batch flow
# ---------------------------------------------------------------------------
def _bedrock_role_arn() -> str:
    arn = os.environ.get("BEDROCK_BATCH_ROLE_ARN")
    if not arn:
        raise RuntimeError(
            "BEDROCK_BATCH_ROLE_ARN env var is required: an IAM role with "
            "bedrock:InvokeModel + s3:GetObject/PutObject on FB_S3_BUCKET."
        )
    return arn


def _upload_jsonl(s3, bucket: str, key: str, lines: List[Dict[str, Any]]) -> str:
    body = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    return f"s3://{bucket}/{key}"


def _download_s3_text(s3, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def _list_s3_keys(s3, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            keys.append(obj["Key"])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def run_bedrock_batch(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    judge_model: str,
    poll_interval: int = 30,
    cost_limit: Optional[float] = None) -> Tuple[int, int, int]:
    """Submit a Bedrock batch job, wait for completion, write replies.

    Returns (completed, failed, skipped). Raises on non-recoverable submission
    errors so the caller can decide whether to fall back to sync invocation.
    """
    cfg = BEDROCK_MODELS[model]
    api_style = cfg["api_style"]
    model_id = _resolve_bedrock_model_id(model)
    region = _model_region(cfg)
    bucket = _bucket_for_region(region)

    # AWS-side hard minimum on batch job size, fail fast instead of uploading
    # to S3, submitting, and waiting for "status=Failed".
    if len(entries) < BEDROCK_BATCH_MIN_RECORDS:
        raise RuntimeError(
            f"Bedrock batch requires at least {BEDROCK_BATCH_MIN_RECORDS} records "
            f"per job; got {len(entries)}. Use --no-batch (sync) for smaller runs, "
            f"or batch ≥{BEDROCK_BATCH_MIN_RECORDS} records at a time."
        )

    # Cost gate (pre-flight).
    est_cost, in_tok, out_tok = estimate_batch_cost(
        entries, model, max_output_tokens, BEDROCK_BATCH_PRICE_MULTIPLIER)
    _enforce_cost_limit(cost_limit, est_cost, in_tok, out_tok, model, "bedrock-batch")

    boto3 = _boto3()
    s3 = boto3.client("s3", region_name=region)
    bedrock = boto3.client("bedrock", region_name=region)
    logger.info(
        f"[bedrock-batch] {model}: region={region} model_id={model_id} bucket={bucket}"
    )

    # 1. Build input JSONL, one record per entry, recordId = custom_id.
    job_uid = uuid.uuid4().hex[:8]
    input_key = f"{_s3_prefix()}{model}/{job_uid}/input.jsonl"
    output_prefix = f"{_s3_prefix()}{model}/{job_uid}/output/"
    lines = []
    for entry in entries:
        _prompt_path, prompt_text, _idx, custom_id = entry
        lines.append({
            "recordId": custom_id,
            "modelInput": _build_bedrock_body(api_style, prompt_text, max_output_tokens),
        })
    input_uri = _upload_jsonl(s3, bucket, input_key, lines)
    output_uri = f"s3://{bucket}/{output_prefix}"
    logger.info(f"[bedrock-batch] {model}: uploaded {len(lines)} records to {input_uri}")

    # 2. Submit batch job.
    job_name = f"factorybench-{model.replace('.', '-').replace('_', '-')}-{job_uid}"
    job_resp = bedrock.create_model_invocation_job(
        jobName=job_name,
        roleArn=_bedrock_role_arn(),
        modelId=model_id,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": input_uri}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}})
    job_arn = job_resp["jobArn"]
    logger.info(f"[bedrock-batch] {model}: submitted job {job_arn}")

    # 3. Poll until terminal state.
    terminal = {"Completed", "Failed", "Stopped", "PartiallyCompleted", "Expired"}
    last_status = None
    while True:
        desc = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        status = desc.get("status")
        if status != last_status:
            logger.info(f"[bedrock-batch] {model}: status={status}")
            last_status = status
        if status in terminal:
            break
        time.sleep(poll_interval)

    if status not in ("Completed", "PartiallyCompleted"):
        raise RuntimeError(f"Bedrock batch job ended with status={status}: {desc.get('message')}")

    # 4. Find the output JSONL and parse it.
    out_keys = _list_s3_keys(s3, bucket, output_prefix)
    jsonl_keys = [k for k in out_keys if k.endswith(".jsonl.out") or k.endswith(".jsonl")]
    if not jsonl_keys:
        raise RuntimeError(f"No JSONL output found at {output_uri} (keys={out_keys[:5]})")
    by_record: Dict[str, Dict[str, Any]] = {}
    for key in jsonl_keys:
        text = _download_s3_text(s3, bucket, key)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = rec.get("recordId")
            if rid:
                by_record[rid] = rec

    # 5. Write reply JSONs in original order.
    completed = failed = 0
    for entry in entries:
        custom_id = entry[3]
        rec = by_record.get(custom_id)
        if rec is None:
            _write_failure(entry, "no record returned in bedrock batch output", output_dir)
            failed += 1
            continue
        err = rec.get("error")
        if err:
            _write_failure(entry, f"bedrock error: {err}", output_dir)
            failed += 1
            continue
        model_out = rec.get("modelOutput")
        if model_out is None:
            _write_failure(entry, "missing modelOutput in bedrock record", output_dir)
            failed += 1
            continue
        try:
            answer, usage = _extract_output(api_style, model_out)
            _write_reply(
                entry, answer, model_out, usage,
                model, output_dir, ground_truth_index, judge_model)
            completed += 1
        except Exception as exc:
            _write_failure(entry, f"reply parse error: {exc}", output_dir)
            failed += 1
    logger.info(f"[bedrock-batch] {model}: completed={completed} failed={failed}")
    return completed, failed, 0


# ---------------------------------------------------------------------------
# Bedrock sync fallback (one invoke per entry)
# ---------------------------------------------------------------------------
def run_bedrock_sync(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    judge_model: str,
    cost_limit: Optional[float] = None) -> Tuple[int, int, int]:
    cfg = BEDROCK_MODELS[model]
    api_style = cfg["api_style"]
    model_id = _resolve_bedrock_model_id(model)
    region = _model_region(cfg)

    # Pre-flight at on-demand rates so users see what the worst case would be.
    est_cost, in_tok, out_tok = estimate_batch_cost(
        entries, model, max_output_tokens, BEDROCK_SYNC_PRICE_MULTIPLIER)
    _enforce_cost_limit(cost_limit, est_cost, in_tok, out_tok, model, "bedrock-sync")

    boto3 = _boto3()
    runtime = boto3.client("bedrock-runtime", region_name=region)
    logger.info(f"[bedrock-sync] {model}: region={region} model_id={model_id}")

    completed = failed = 0
    total = len(entries)
    cumulative_cost = 0.0
    stopped = False
    for i, entry in enumerate(entries, 1):
        if stopped:
            failed += 1  # remaining entries unwritten, count as failed for accounting
            _write_failure(entry, f"cost limit ${cost_limit:.2f} reached", output_dir)
            continue
        _prompt_path, prompt_text, _idx, custom_id = entry
        body = _build_bedrock_body(api_style, prompt_text, max_output_tokens)
        try:
            resp = runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json")
            raw = json.loads(resp["body"].read())
            answer, usage = _extract_output(api_style, raw)
            est = _write_reply(entry, answer, raw, usage, model, output_dir, ground_truth_index, judge_model)[0]
            cumulative_cost += est
            completed += 1
            logger.info(
                f"[bedrock-sync] [{i}/{total}] {custom_id} | "
                f"cost: ${est:.4f} | cumulative: ${cumulative_cost:.4f}"
            )
            if cost_limit is not None and cumulative_cost >= cost_limit:
                logger.warning(
                    f"Cost limit ${cost_limit:.2f} reached at {custom_id} "
                    f"(${cumulative_cost:.4f}). Halting submissions."
                )
                stopped = True
        except Exception as exc:
            _write_failure(entry, str(exc), output_dir)
            failed += 1
            logger.warning(f"[bedrock-sync] [{i}/{total}] {custom_id} failed: {exc}")
    return completed, failed, 0


# ---------------------------------------------------------------------------
# SageMaker Async Inference flow
#
# Per src/evaluation/aws-setup.md the JumpStart endpoint is deployed in
# Asynchronous mode with auto-scaling to zero. Each request uploads its input
# JSON to S3, calls InvokeEndpointAsync (which returns immediately with an
# OutputLocation S3 URI), and polls that URI until the response object lands.
# ---------------------------------------------------------------------------
def _sagemaker_role_arn() -> str:
    arn = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not arn:
        raise RuntimeError(
            "SAGEMAKER_ROLE_ARN env var is required: an IAM role with "
            "sagemaker:InvokeEndpointAsync + S3 read/write on FB_S3_BUCKET."
        )
    return arn


def _build_sagemaker_request(api_style: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    if api_style == "tgi":
        return {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.0}}
    if api_style in ("openai-chat", "mistral", "deepseek"):
        return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    raise ValueError(f"Unknown SageMaker api_style: {api_style!r}")


def _s3_parse_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://")
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def _wait_for_s3_object(s3, bucket: str, key: str, timeout_s: int, poll_s: float = 2.0) -> bool:
    """Poll until an S3 object exists or timeout elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            time.sleep(poll_s)
    return False


def run_sagemaker_async(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    judge_model: str,
    poll_interval: int = 30,
    cost_limit: Optional[float] = None,
    per_request_timeout_s: int = 1800) -> Tuple[int, int, int]:
    """Submit Async Inference requests against a pre-deployed JumpStart endpoint.

    Each entry: upload one JSON input to S3 → call InvokeEndpointAsync →
    poll the returned OutputLocation S3 key until the result lands.
    """
    cfg = SAGEMAKER_MODELS[model]
    api_style = cfg["api_style"]
    endpoint_name = _resolve_sagemaker_endpoint(model)
    region = _model_region(cfg)
    bucket = _bucket_for_region(region)

    # SageMaker is per-instance-hour. The token-based estimate is a rough upper
    # bound (it understates the fixed instance cost when run-time is short)
    # but it does flag pathologically large jobs for free.
    est_cost, in_tok, out_tok = estimate_batch_cost(
        entries, model, max_output_tokens, BEDROCK_SYNC_PRICE_MULTIPLIER)
    _enforce_cost_limit(cost_limit, est_cost, in_tok, out_tok, model, "sagemaker-async")

    boto3 = _boto3()
    s3 = boto3.client("s3", region_name=region)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
    logger.info(f"[sagemaker-async] {model}: region={region} endpoint={endpoint_name}")

    job_uid = uuid.uuid4().hex[:8]
    input_prefix = f"{_s3_prefix()}{model}/{job_uid}/input/"
    output_prefix = f"{_s3_prefix()}{model}/{job_uid}/output/"

    # 1. Submit all requests in order; collect output S3 URIs.
    pending: List[Tuple[Tuple[Path, str, int, str], str]] = []
    completed = failed = 0
    for entry in entries:
        _prompt_path, prompt_text, _idx, custom_id = entry
        body = _build_sagemaker_request(api_style, prompt_text, max_output_tokens)
        in_key = f"{input_prefix}{custom_id}.json"
        try:
            s3.put_object(
                Bucket=bucket, Key=in_key,
                Body=json.dumps(body).encode("utf-8"),
                ContentType="application/json")
            resp = sm_runtime.invoke_endpoint_async(
                EndpointName=endpoint_name,
                InputLocation=f"s3://{bucket}/{in_key}",
                ContentType="application/json",
                Accept="application/json")
            out_uri = resp["OutputLocation"]
            pending.append((entry, out_uri))
        except Exception as exc:
            _write_failure(entry, f"async submit failed: {exc}", output_dir)
            failed += 1
            logger.warning(f"[sagemaker-async] submit failed for {custom_id}: {exc}")

    logger.info(f"[sagemaker-async] {model}: {len(pending)} requests in flight")

    # 2. Poll each output URI until it lands (or timeout).
    for entry, out_uri in pending:
        custom_id = entry[3]
        out_bucket, out_key = _s3_parse_uri(out_uri)
        if not _wait_for_s3_object(s3, out_bucket, out_key, timeout_s=per_request_timeout_s):
            _write_failure(entry, f"async timeout waiting for {out_uri}", output_dir)
            failed += 1
            logger.warning(f"[sagemaker-async] timeout: {custom_id}")
            continue
        try:
            raw_text = _download_s3_text(s3, out_bucket, out_key)
            raw = json.loads(raw_text)
            answer, usage = _extract_output(api_style, raw)
            _write_reply(entry, answer, raw, usage, model, output_dir, ground_truth_index, judge_model)
            completed += 1
        except Exception as exc:
            _write_failure(entry, f"async reply parse error: {exc}", output_dir)
            failed += 1
            logger.warning(f"[sagemaker-async] parse failed for {custom_id}: {exc}")

    logger.info(f"[sagemaker-async] {model}: completed={completed} failed={failed}")
    return completed, failed, 0


# Backwards-compatible alias (the previous Batch Transform implementation has
# been replaced by Async Inference per the verified deployment guide).
run_sagemaker_batch = run_sagemaker_async


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------
def run_aws_eval(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
    judge_model: str,
    use_batch: bool = True,
    poll_interval: int = 30,
    cost_limit: Optional[float] = None,
    strict_batch: bool = False) -> Tuple[int, int, int]:
    """Dispatch to the right AWS pipeline for ``model``.

    Returns (completed, failed, skipped). Mirrors run_foundry_eval's signature
    so run_pipeline.py can call either with the same arguments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip-existing logic (mirrors foundry path).
    pending: List[Tuple[Path, str, int, str]] = []
    skipped = 0
    for entry in entries:
        _prompt_path, _prompt_text, _idx, custom_id = entry
        if not overwrite and (output_dir / f"{custom_id}_answer.json").exists():
            skipped += 1
            continue
        fail_path = output_dir / f"{custom_id}_failed.json"
        if fail_path.exists():
            try:
                fail_path.unlink()
            except OSError:
                pass
        pending.append(entry)

    if not pending:
        return 0, 0, skipped

    provider = get_provider(model)
    if provider == "bedrock":
        if use_batch and BEDROCK_MODELS[model].get("supports_batch", False):
            try:
                done, fail, _ = run_bedrock_batch(
                    pending, model, output_dir, max_output_tokens,
                    ground_truth_index, judge_model,
                    poll_interval=poll_interval, cost_limit=cost_limit)
                return done, fail, skipped
            except Exception as exc:
                if strict_batch:
                    logger.error(f"[bedrock-batch] {model} batch failed and --strict-batch is set; not falling back.")
                    raise
                logger.warning(
                    f"[bedrock-batch] {model} batch failed ({exc}); falling back to sync."
                )
        done, fail, _ = run_bedrock_sync(
            pending, model, output_dir, max_output_tokens,
            ground_truth_index, judge_model, cost_limit=cost_limit)
        return done, fail, skipped

    if provider == "sagemaker":
        # SageMaker Async Inference: per-request S3 in/out against a deployed
        # endpoint (typically scale-to-zero JumpStart deploy).
        done, fail, _ = run_sagemaker_async(
            pending, model, output_dir, max_output_tokens,
            ground_truth_index, judge_model,
            poll_interval=poll_interval, cost_limit=cost_limit)
        return done, fail, skipped

    raise ValueError(
        f"Model {model!r} is not registered as a Bedrock or SageMaker model. "
        f"Use run_foundry_eval for Foundry-served models."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AWS evaluation (Bedrock + SageMaker)")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing prompt JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for replies")
    parser.add_argument("--questions", type=Path, required=True, help="Directory with Q&A ground truth JSONs")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(BEDROCK_MODELS.keys()) + list(SAGEMAKER_MODELS.keys()),
        help="AWS-served model to evaluate")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-judge entirely. Free-form items get score=None.")
    # Accepted for parity with the Foundry and GCP CLIs. run_pipeline passes it
    # to every backend unconditionally, so omitting it here made the AWS path
    # die on `unrecognized arguments` whenever it was driven by the pipeline.
    parser.add_argument("--eval-level", type=str, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-batch", dest="use_batch", action="store_true", default=True)
    parser.add_argument("--no-batch", dest="use_batch", action="store_false",
                        help="Bedrock only: force sync invocation per record.")
    parser.add_argument("--strict-batch", action="store_true",
                        help="If batch submission fails, error out instead of falling back "
                             "to sync. Useful for debugging batch-specific issues.")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Batch polling interval in seconds (default: 30)")
    parser.add_argument("--cost-limit", type=float, default=20.0,
                        help="USD pre-flight cost limit. Bedrock batch applies a "
                             "50%% discount in the estimate; sync uses on-demand "
                             "rates and stops mid-stream once the cap is hit. "
                             "Pass 0 to disable.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N prompts from --input (for smoke tests).")
    parser.add_argument("--batch-number", type=int, default=0,
                        help="0-indexed slice of size --batch-size to process. Used by the "
                             "test-set orchestrator to chunk a level across multiple batches.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Slice width for --batch-number (process items "
                             "[batch_number*batch_size : (batch_number+1)*batch_size]).")
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

    # Load .env so AWS_*, FB_S3_*, BEDROCK_BATCH_ROLE_ARN etc. are visible.
    if args.env_file and args.env_file.exists():
        load_dotenv_file(args.env_file)

    judge_model = "" if args.no_judge else args.judge_model

    entries = load_prompt_entries(
        args.input,
        batch_number=args.batch_number,
        batch_size=args.batch_size)
    if args.limit is not None and args.limit > 0:
        entries = entries[: args.limit]
        logger.info(f"Limited to first {len(entries)} prompts (--limit {args.limit})")
    ground_truth_index = build_question_index(args.questions)

    cost_limit = None if (args.cost_limit is not None and args.cost_limit <= 0) else args.cost_limit
    completed, failed, skipped = run_aws_eval(
        entries=entries,
        model=args.model,
        output_dir=args.output_dir,
        max_output_tokens=args.max_output_tokens,
        overwrite=args.overwrite,
        ground_truth_index=ground_truth_index,
        judge_model=judge_model,
        use_batch=args.use_batch,
        poll_interval=args.poll_interval,
        cost_limit=cost_limit,
        strict_batch=args.strict_batch)
    logger.info(f"Done. completed={completed} failed={failed} skipped={skipped}")

    if args.summary_file:
        save_json(args.summary_file, {
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "model": args.model,
            "output_dir": str(args.output_dir),
        })


if __name__ == "__main__":
    main()
