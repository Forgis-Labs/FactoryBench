"""Batch-only judge calls across foundry and bedrock providers.

Mirrors the inference batch pattern in ``scripts/eval_test_set.py`` and the
provider-specific helpers in ``src/evaluation/run_foundry_eval.py`` /
``run_aws_eval.py``: build the JSONL body in the provider's native shape,
submit, poll until terminal, parse results into ``{custom_id: response_text}``.

Sync is intentionally NOT supported here, the caller is expected to size
batches above each provider's minimum (Bedrock's hard 100-record floor).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    BEDROCK_MODELS,
    FOUNDRY_MODELS,
    get_provider,
    get_upstream_model_id,
    get_batch_deployment)

logger = logging.getLogger(__name__)

BEDROCK_BATCH_MIN_RECORDS: int = 100


@dataclass(frozen=True)
class JudgeRequest:
    """One judge call. ``custom_id`` is opaque to the batch layer; the caller
    uses it to correlate the response back to the source reply file."""
    custom_id: str
    prompt: str


# ---------------------------------------------------------------------------
# Foundry (Azure /v1/batches with /chat/completions endpoint)
# ---------------------------------------------------------------------------
def _submit_foundry_batch(
    model: str,
    requests_: List[JudgeRequest],
    max_output_tokens: int,
    poll_interval: int) -> Dict[str, str]:
    import httpx as _httpx
    from src.evaluation.run_foundry_eval import (
        resolve_api_key, resolve_endpoint, _openai_client)
    from openai import OpenAI as _OpenAI

    base_url, _api_style, api_version = resolve_endpoint(model)
    api_key = resolve_api_key(model)
    # Use a generous connect timeout (30s), Azure AI Foundry batch endpoint
    # can be slow to accept connections; the SDK default of 5s is too tight.
    _timeout = _httpx.Timeout(timeout=600.0, connect=30.0)
    _kwargs: dict = {"api_key": api_key, "base_url": base_url, "timeout": _timeout}
    if api_version:
        _kwargs["default_query"] = {"api-version": api_version}
    client = _OpenAI(**_kwargs)
    deployment = get_batch_deployment(model)

    lines = []
    for req in requests_:
        body = {
            "model": deployment,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_completion_tokens": max_output_tokens,
        }
        lines.append(json.dumps({
            "custom_id": req.custom_id,
            "method": "POST",
            "url": "/chat/completions",
            "body": body,
        }))
    jsonl_text = "\n".join(lines)

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
            f"[judge-batch][foundry] {model}: submitted batch={batch.id} "
            f"with {len(requests_)} requests"
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
        logger.info(
            f"[judge-batch][foundry] {model} batch={batch.id}: "
            f"status={batch.status}{rc_str}"
        )
        if batch.status in terminal:
            break
        time.sleep(poll_interval)

    if batch.status != "completed":
        raise RuntimeError(
            f"foundry batch ended with status={batch.status} "
            f"(model={model}, batch_id={batch.id})"
        )

    out: Dict[str, str] = {}
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
            if not cid or data.get("error"):
                continue
            resp = data.get("response") or {}
            if int(resp.get("status_code", 0)) != 200:
                continue
            body = resp.get("body") or {}
            choices = body.get("choices") or []
            if not choices:
                continue
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            if text:
                out[cid] = text
    return out


# ---------------------------------------------------------------------------
# Bedrock (CreateModelInvocationJob, S3 in/out)
# ---------------------------------------------------------------------------
def _submit_bedrock_batch(
    model: str,
    requests_: List[JudgeRequest],
    max_output_tokens: int,
    poll_interval: int) -> Dict[str, str]:
    if len(requests_) < BEDROCK_BATCH_MIN_RECORDS:
        raise RuntimeError(
            f"Bedrock batch requires ≥{BEDROCK_BATCH_MIN_RECORDS} records "
            f"(got {len(requests_)} for {model}). Combine more replies into "
            f"a single judge run, or score with foundry only."
        )

    from src.evaluation.run_aws_eval import (
        _boto3, _model_region, _bucket_for_region, _s3_prefix,
        _resolve_bedrock_model_id, _build_bedrock_body, _extract_output,
        _bedrock_role_arn, _upload_jsonl, _list_s3_keys, _download_s3_text)

    cfg = BEDROCK_MODELS[model]
    api_style = cfg["api_style"]
    model_id = _resolve_bedrock_model_id(model)
    region = _model_region(cfg)
    bucket = _bucket_for_region(region)

    boto3 = _boto3()
    s3 = boto3.client("s3", region_name=region)
    bedrock = boto3.client("bedrock", region_name=region)
    logger.info(
        f"[judge-batch][bedrock] {model}: region={region} "
        f"model_id={model_id} bucket={bucket}"
    )

    job_uid = uuid.uuid4().hex[:8]
    input_key = f"{_s3_prefix()}judge/{model}/{job_uid}/input.jsonl"
    output_prefix = f"{_s3_prefix()}judge/{model}/{job_uid}/output/"
    lines = [
        {
            "recordId": req.custom_id,
            "modelInput": _build_bedrock_body(api_style, req.prompt, max_output_tokens),
        }
        for req in requests_
    ]
    input_uri = _upload_jsonl(s3, bucket, input_key, lines)
    output_uri = f"s3://{bucket}/{output_prefix}"
    logger.info(
        f"[judge-batch][bedrock] {model}: uploaded {len(lines)} records to {input_uri}"
    )

    job_name = f"factorybench-judge-{model.replace('.', '-').replace('_', '-')}-{job_uid}"
    job_resp = bedrock.create_model_invocation_job(
        jobName=job_name,
        roleArn=_bedrock_role_arn(),
        modelId=model_id,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": input_uri}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}})
    job_arn = job_resp["jobArn"]
    logger.info(f"[judge-batch][bedrock] {model}: submitted job {job_arn}")

    terminal = {"Completed", "Failed", "Stopped", "PartiallyCompleted", "Expired"}
    last = None
    while True:
        desc = bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        status = desc.get("status")
        if status != last:
            logger.info(f"[judge-batch][bedrock] {model}: status={status}")
            last = status
        if status in terminal:
            break
        time.sleep(poll_interval)

    if status not in ("Completed", "PartiallyCompleted"):
        raise RuntimeError(
            f"bedrock judge batch ended with status={status}: {desc.get('message')}"
        )

    out_keys = _list_s3_keys(s3, bucket, output_prefix)
    jsonl_keys = [k for k in out_keys if k.endswith(".jsonl.out") or k.endswith(".jsonl")]
    if not jsonl_keys:
        raise RuntimeError(
            f"no JSONL output found at {output_uri} (keys={out_keys[:5]})"
        )

    out: Dict[str, str] = {}
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
            cid = rec.get("recordId")
            if not cid or rec.get("error"):
                continue
            model_out = rec.get("modelOutput")
            if model_out is None:
                continue
            try:
                answer, _usage = _extract_output(api_style, model_out)
            except Exception:
                continue
            if answer:
                out[cid] = answer
    return out


# ---------------------------------------------------------------------------
# Foundry sync (concurrent /chat/completions, fallback when batch is broken)
# ---------------------------------------------------------------------------
def _submit_foundry_sync(
    model: str,
    requests_: List[JudgeRequest],
    max_output_tokens: int,
    concurrency: int = 20) -> Dict[str, str]:
    """Call /chat/completions synchronously for each request, with thread-pool
    concurrency. Avoids the /files batch upload endpoint entirely."""
    import httpx as _httpx
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    from openai import OpenAI as _OpenAI
    from src.evaluation.run_foundry_eval import resolve_api_key, resolve_endpoint

    base_url, _api_style, api_version = resolve_endpoint(model)
    api_key = resolve_api_key(model)
    deployment = get_batch_deployment(model)
    _timeout = _httpx.Timeout(timeout=120.0, connect=30.0)
    _kwargs: dict = {"api_key": api_key, "base_url": base_url, "timeout": _timeout}
    if api_version:
        _kwargs["default_query"] = {"api-version": api_version}
    client = _OpenAI(**_kwargs)

    logger.info(
        f"[judge-sync][foundry] {model}: {len(requests_)} requests "
        f"with concurrency={concurrency}"
    )

    def _call(req: JudgeRequest) -> Tuple[str, Optional[str]]:
        try:
            # Try the newer Responses API first (gpt-5.x on Azure Foundry).
            resp = client.responses.create(
                model=deployment,
                input=req.prompt,
                max_output_tokens=max_output_tokens)
            from src.evaluation.run_foundry_eval import _extract_output_text_from_responses_body, _to_dict
            text = _extract_output_text_from_responses_body(_to_dict(resp)) or ""
            if text:
                return req.custom_id, text
        except Exception:
            pass
        try:
            # Fallback: chat.completions
            resp = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": req.prompt}],
                max_completion_tokens=max_output_tokens)
            text = resp.choices[0].message.content or ""
            return req.custom_id, text
        except Exception as exc:
            logger.warning(f"[judge-sync][foundry] {model}: {req.custom_id} failed: {exc}")
            return req.custom_id, None

    out: Dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_call, req): req for req in requests_}
        for fut in _as_completed(futures):
            cid, text = fut.result()
            done += 1
            if done % 100 == 0:
                logger.info(f"[judge-sync][foundry] {model}: {done}/{len(requests_)} done")
            if text:
                out[cid] = text
    logger.info(f"[judge-sync][foundry] {model}: completed {len(out)}/{len(requests_)} with responses")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def submit_judge_batch(
    model: str,
    requests_: List[JudgeRequest],
    max_output_tokens: int = 256,
    poll_interval: int = 30,
    sync_concurrency: int = 0) -> Dict[str, str]:
    """Submit ``requests_`` to ``model`` as a single batch and return
    ``{custom_id: response_text}`` once the job completes.

    Routes to foundry (gpt-5.1-1) or bedrock (sonnet-4.6, deepseek-v3.2) based
    on ``get_provider()``. Raises if the provider is unknown or if a Bedrock
    job is below the 100-record minimum.

    ``sync_concurrency > 0`` forces foundry models to use concurrent sync calls
    instead of the /v1/batches endpoint (useful when the batch upload is broken).
    """
    if not requests_:
        return {}
    provider = get_provider(model)
    if provider == "foundry":
        if model not in FOUNDRY_MODELS:
            raise ValueError(f"foundry model {model!r} not in FOUNDRY_MODELS")
        if sync_concurrency > 0:
            return _submit_foundry_sync(model, requests_, max_output_tokens, sync_concurrency)
        return _submit_foundry_batch(model, requests_, max_output_tokens, poll_interval)
    if provider == "bedrock":
        if model not in BEDROCK_MODELS:
            raise ValueError(f"bedrock model {model!r} not in BEDROCK_MODELS")
        return _submit_bedrock_batch(model, requests_, max_output_tokens, poll_interval)
    raise ValueError(f"unsupported provider for judge model {model!r}: {provider!r}")
