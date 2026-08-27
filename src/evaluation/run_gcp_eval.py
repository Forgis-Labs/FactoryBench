#!/usr/bin/env python3
"""GCP evaluation entrypoint: Vertex AI counterpart of ``run_aws_eval``.

Same three checkpoints the paper ran on Bedrock, re-pointed at Vertex AI.
Provider routing and per-model region / id resolution come from ``src.config``.
Reply JSON shape, ground-truth lookup, scoring and cost accounting are reused
verbatim from ``run_aws_eval`` / ``run_foundry_eval``, so this module is
drop-in compatible with ``run_pipeline.py``.

The three models do not share a serving surface, this is the one structural
thing that did not survive the migration intact:

  ``claude-sonnet-4.6``  MaaS. Anthropic Messages API over ``:rawPredict``.
                         Native Vertex batch prediction (GCS in / GCS out),
                         the analogue of Bedrock ``CreateModelInvocationJob``.
  ``deepseek-v3.2``      MaaS. OpenAI-compatible ``/endpoints/openapi`` route,
                         identical in shape to the qwen-3-235b path already in
                         ``run_foundry_eval``. Sync only, Vertex exposes no
                         batch surface for MaaS partner models.
  ``mistral-large-3``    NOT available as MaaS on Vertex, so it is not routed
                         here at all: ``src/config.py`` sends it to Azure AI
                         Foundry. Model Garden ships it only as a self-deploy
                         vLLM container (8xH200 or 8xB200) billing per
                         GPU-hour. If you do stand one up, set
                         ``MISTRAL_LARGE_3_VERTEX_ENDPOINT`` and give the model
                         ``provider: "vertex"`` with ``self_deployed: True``;
                         the ``vertex_raw_predict`` path below handles it.

Auth is Application Default Credentials, no long-lived API key. On a
workstation run ``gcloud auth application-default login``; on GCE / GKE /
Cloud Run / Batch the metadata server supplies it. The token helper is shared
with the Vertex path in ``run_foundry_eval`` so there is one refresh cache.

Required environment (full walkthrough in ``src/evaluation/gcp-setup.md``):

  Shared:
    GCP_PROJECT                 project id (default: anon-gcp-project)
    GCS_BUCKET                  bucket for batch I/O (Vertex batch only)
    GCS_PREFIX                  key prefix (default: factorybench/)

  Per-model overrides (all optional; defaults live in src/config.py):
    CLAUDE_SONNET_46_VERTEX_MODEL       CLAUDE_SONNET_46_VERTEX_REGION
    CLAUDE_SONNET_46_VERTEX_BATCH_REGION
    DEEPSEEK_V32_VERTEX_MODEL           DEEPSEEK_V32_VERTEX_REGION
    MISTRAL_LARGE_3_VERTEX_MODEL        MISTRAL_LARGE_3_VERTEX_REGION
    MISTRAL_LARGE_3_VERTEX_ENDPOINT     numeric endpoint id from the deploy

Usage:
    python -m src.evaluation.run_gcp_eval \\
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

import requests

from src.config import (
    DEFAULT_JUDGE_MODEL,
    GCP_PROJECT_DEFAULT,
    GCS_BUCKET_DEFAULT,
    GCS_PREFIX_DEFAULT,
    VERTEX_MODELS,
    get_provider,
    get_region,
    get_upstream_model_id,
    get_vertex_publisher)

# Shared helpers. run_aws_eval owns the reply writer, the pre-flight cost gate
# and the OpenAI/Anthropic response extractors; all are provider-agnostic and
# importing them keeps the two clouds' reply JSON byte-identical. Its boto3
# import is lazy, so pulling this in costs nothing on a machine without AWS.
from src.evaluation.run_aws_eval import (
    _enforce_cost_limit,
    _extract_anthropic,
    _extract_chat,
    _write_failure,
    _write_reply,
    estimate_batch_cost)
from src.evaluation.run_foundry_eval import (
    _vertex_access_token,
    build_question_index)
from src.evaluation.test_gpt_5mini import (
    load_dotenv_file,
    load_prompt_entries,
    save_json)

logger = logging.getLogger(__name__)


# Vertex batch prediction has no published blanket discount the way Bedrock
# batch does (50% off on-demand), so the pre-flight estimator charges batch at
# on-demand rates. That makes the GCP estimate conservative relative to the
# AWS one rather than optimistic, a cost gate should never under-predict.
VERTEX_BATCH_PRICE_MULTIPLIER: float = 1.0
VERTEX_SYNC_PRICE_MULTIPLIER: float = 1.0

# Vertex's Anthropic surface pins this literal; it is not the model version.
ANTHROPIC_VERTEX_VERSION: str = "vertex-2023-10-16"

# batchPredictionJobs is not served from the `global` endpoint. Claude's sync
# region default is `global`, so batch needs its own regional override.
VERTEX_BATCH_REGION_DEFAULT: str = "us-central1"

_TERMINAL_JOB_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}
_OK_JOB_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


# ---------------------------------------------------------------------------
# Lazy google-cloud-storage import (mirrors run_aws_eval._boto3)
# ---------------------------------------------------------------------------
def _gcs():
    try:
        from google.cloud import storage  # type: ignore
        return storage
    except ImportError as exc:
        raise ImportError(
            "google-cloud-storage is required for Vertex batch I/O. "
            "Install with `pip install google-cloud-storage`."
        ) from exc


# ---------------------------------------------------------------------------
# Project / region / endpoint resolution
# ---------------------------------------------------------------------------
def _project() -> str:
    return os.environ.get("GCP_PROJECT") or GCP_PROJECT_DEFAULT


def _api_host(location: str) -> str:
    """Vertex regional host. The `global` location uses the unprefixed host."""
    return (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )


def _batch_region(model: str) -> str:
    """Region for this model's batch prediction jobs.

    Separate from the sync region because ``batchPredictionJobs`` is not served
    from `global`, which is where Claude's sync traffic goes by default.
    """
    override = os.environ.get("CLAUDE_SONNET_46_VERTEX_BATCH_REGION")
    if override:
        return override
    region = get_region(model) or VERTEX_BATCH_REGION_DEFAULT
    return VERTEX_BATCH_REGION_DEFAULT if region == "global" else region


def _publisher_model_path(model: str) -> str:
    """``publishers/<pub>/models/<id>`` for a Model Garden model."""
    publisher = get_vertex_publisher(model)
    if not publisher:
        raise RuntimeError(f"{model!r} has no vertex_publisher in src/config.py")
    return f"publishers/{publisher}/models/{get_upstream_model_id(model)}"


def _self_deployed_endpoint(model: str) -> str:
    """Numeric endpoint id for a self-deployed model (Mistral Large 3)."""
    cfg = VERTEX_MODELS[model]
    env = cfg.get("endpoint_env")
    value = os.environ.get(env) if env else None
    if not value:
        raise RuntimeError(
            f"{model} is self-deployed on Vertex and has no endpoint yet. "
            f"Deploy it from Model Garden and put the resulting endpoint id in "
            f"{env}. Note this stands up a multi-GPU node that bills per hour "
            f"until you tear it down."
        )
    # Accept either a bare numeric id or a full resource path.
    return value.rstrip("/").split("/")[-1]


def _sync_url(model: str) -> str:
    """Full URL for one synchronous request against ``model``."""
    cfg = VERTEX_MODELS[model]
    api_style = cfg["api_style"]
    location = get_region(model) or VERTEX_BATCH_REGION_DEFAULT
    host = _api_host(location)
    project = _project()

    if api_style == "vertex_anthropic":
        return (
            f"https://{host}/v1/projects/{project}/locations/{location}/"
            f"{_publisher_model_path(model)}:rawPredict"
        )
    if api_style == "vertex_openai":
        # OpenAI-compatible MaaS route. Same shape the qwen-3-235b foundry
        # entry already uses, just built here instead of read from an env var.
        return (
            f"https://{host}/v1beta1/projects/{project}/locations/{location}/"
            f"endpoints/openapi/chat/completions"
        )
    if api_style == "vertex_raw_predict":
        return (
            f"https://{host}/v1/projects/{project}/locations/{location}/"
            f"endpoints/{_self_deployed_endpoint(model)}:rawPredict"
        )
    raise ValueError(f"Unknown vertex api_style: {api_style!r}")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
def _body_anthropic(prompt: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "anthropic_version": ANTHROPIC_VERTEX_VERSION,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }


def _body_openai(model: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": get_upstream_model_id(model),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def build_request_body(model: str, prompt: str, max_tokens: int) -> Dict[str, Any]:
    api_style = VERTEX_MODELS[model]["api_style"]
    if api_style == "vertex_anthropic":
        return _body_anthropic(prompt, max_tokens)
    if api_style in ("vertex_openai", "vertex_raw_predict"):
        return _body_openai(model, prompt, max_tokens)
    raise ValueError(f"Unknown vertex api_style: {api_style!r}")


def extract_output(api_style: str, raw: Any) -> Tuple[str, Dict[str, Any]]:
    if api_style == "vertex_anthropic":
        return _extract_anthropic(raw)
    if api_style in ("vertex_openai", "vertex_raw_predict"):
        return _extract_chat(raw)
    raise ValueError(f"Unknown api_style for output extraction: {api_style!r}")


# Vertex MaaS partner models are throttled on concurrent requests per project
# and answer 429 RESOURCE_EXHAUSTED under sustained sequential load, not just
# parallel load. Without backoff a 50-prompt run loses a fifth of its records
# to throttling. 503 is included because MaaS backends shed load that way too.
_RETRY_STATUS = {429, 500, 503, 504}
_RETRY_ATTEMPTS = 6
_RETRY_BASE_SLEEP = 4.0


def _post_with_retry(
    url: str,
    body: Dict[str, Any],
    timeout_s: int) -> "requests.Response":
    """POST with exponential backoff on throttling and transient 5xx.

    Honours Retry-After when the server sends it, otherwise backs off
    4s, 8s, 16s... Returns the final response; the caller still checks
    status, so a request that exhausts its retries surfaces the real error.
    """
    delay = _RETRY_BASE_SLEEP
    response = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        response = requests.post(url, headers=_auth_headers(), json=body, timeout=timeout_s)
        if response.status_code not in _RETRY_STATUS:
            return response
        if attempt == _RETRY_ATTEMPTS:
            break
        wait = delay
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                wait = max(wait, float(retry_after))
            except ValueError:
                pass
        logger.info(
            f"[vertex-sync] HTTP {response.status_code}; retrying in {wait:.0f}s "
            f"(attempt {attempt}/{_RETRY_ATTEMPTS})"
        )
        time.sleep(wait)
        delay *= 2
    return response  # type: ignore[return-value]


def _auth_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_vertex_access_token()}",
        "Content-Type": "application/json",
        # Required when authenticating with local ADC: billing/quota would
        # otherwise have no project to attach to and the call 403s.
        "x-goog-user-project": _project(),
    }


# ---------------------------------------------------------------------------
# GCS helpers (the S3 helpers' counterpart)
# ---------------------------------------------------------------------------
def _bucket_name() -> str:
    bucket = os.environ.get("GCS_BUCKET") or GCS_BUCKET_DEFAULT
    if not bucket:
        raise RuntimeError(
            "GCS_BUCKET env var is required for Vertex batch I/O (any bucket the "
            "caller can read+write, in the same region as the batch job)."
        )
    return bucket


def _gcs_prefix() -> str:
    return os.environ.get("GCS_PREFIX") or GCS_PREFIX_DEFAULT


def _upload_jsonl(bucket_name: str, blob_path: str, lines: List[Dict[str, Any]]) -> str:
    client = _gcs().Client(project=_project())
    blob = client.bucket(bucket_name).blob(blob_path)
    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
    blob.upload_from_string(payload, content_type="application/jsonl")
    return f"gs://{bucket_name}/{blob_path}"


def _list_blobs(bucket_name: str, prefix: str) -> List[str]:
    client = _gcs().Client(project=_project())
    return [b.name for b in client.list_blobs(bucket_name, prefix=prefix)]


def _download_text(bucket_name: str, blob_path: str) -> str:
    client = _gcs().Client(project=_project())
    return client.bucket(bucket_name).blob(blob_path).download_as_text()


# ---------------------------------------------------------------------------
# Vertex batch prediction (Anthropic only)
#
# The GCS-in / GCS-out analogue of Bedrock CreateModelInvocationJob. Input is
# one JSONL line per prompt carrying a custom_id we can join back on; output
# lands as one or more predictions JSONL files under the destination prefix.
# ---------------------------------------------------------------------------
def run_vertex_batch(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    judge_model: str,
    poll_interval: int = 30,
    cost_limit: Optional[float] = None) -> Tuple[int, int, int]:
    """Submit a Vertex batch prediction job, wait, write replies.

    Returns (completed, failed, skipped). Raises on non-recoverable submission
    errors so the caller can decide whether to fall back to sync.
    """
    cfg = VERTEX_MODELS[model]
    api_style = cfg["api_style"]
    if api_style != "vertex_anthropic":
        raise RuntimeError(
            f"{model} has no Vertex batch surface (api_style={api_style}). "
            f"Only the Anthropic publisher models support batchPredictionJobs."
        )

    est_cost, in_tok, out_tok = estimate_batch_cost(
        entries, model, max_output_tokens, VERTEX_BATCH_PRICE_MULTIPLIER)
    _enforce_cost_limit(cost_limit, est_cost, in_tok, out_tok, model, "vertex-batch")

    location = _batch_region(model)
    project = _project()
    bucket = _bucket_name()
    job_uid = uuid.uuid4().hex[:8]
    input_path = f"{_gcs_prefix()}{model}/{job_uid}/input.jsonl"
    output_prefix = f"{_gcs_prefix()}{model}/{job_uid}/output/"

    logger.info(
        f"[vertex-batch] {model}: location={location} "
        f"model={get_upstream_model_id(model)} bucket={bucket}"
    )

    # 1. Build and upload the input JSONL.
    lines = [
        {
            "custom_id": custom_id,
            "request": build_request_body(model, prompt_text, max_output_tokens),
        }
        for _prompt_path, prompt_text, _idx, custom_id in entries
    ]
    input_uri = _upload_jsonl(bucket, input_path, lines)
    output_uri = f"gs://{bucket}/{output_prefix}"
    logger.info(f"[vertex-batch] {model}: uploaded {len(lines)} records to {input_uri}")

    # 2. Submit.
    host = _api_host(location)
    jobs_url = (
        f"https://{host}/v1/projects/{project}/locations/{location}/batchPredictionJobs"
    )
    job_body = {
        "displayName": f"factorybench-{model.replace('.', '-').replace('_', '-')}-{job_uid}",
        "model": _publisher_model_path(model),
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [input_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": output_uri},
        },
    }
    resp = requests.post(jobs_url, headers=_auth_headers(), json=job_body, timeout=300)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"batchPredictionJobs.create failed: {resp.status_code} {resp.text[:500]}"
        )
    job = resp.json()
    job_name = job.get("name")
    logger.info(f"[vertex-batch] {model}: submitted {job_name}")

    # 3. Poll to a terminal state.
    last_state = None
    state = None
    detail: Dict[str, Any] = {}
    while True:
        got = requests.get(
            f"https://{host}/v1/{job_name}", headers=_auth_headers(), timeout=120
        )
        if got.status_code >= 400:
            raise RuntimeError(
                f"batchPredictionJobs.get failed: {got.status_code} {got.text[:300]}"
            )
        detail = got.json()
        state = detail.get("state")
        if state != last_state:
            logger.info(f"[vertex-batch] {model}: state={state}")
            last_state = state
        if state in _TERMINAL_JOB_STATES:
            break
        time.sleep(poll_interval)

    if state not in _OK_JOB_STATES:
        raise RuntimeError(
            f"Vertex batch job ended with state={state}: {detail.get('error')}"
        )

    # 4. Collect predictions. Vertex writes into a timestamped subdirectory
    #    beneath the prefix we asked for, so scan the whole prefix.
    blob_names = _list_blobs(bucket, output_prefix)
    jsonl_blobs = [b for b in blob_names if b.endswith(".jsonl") or b.endswith(".jsonl.out")]
    if not jsonl_blobs:
        raise RuntimeError(
            f"No JSONL predictions under {output_uri} (blobs={blob_names[:5]})"
        )
    by_record: Dict[str, Dict[str, Any]] = {}
    for blob_name in jsonl_blobs:
        for raw_line in _download_text(bucket, blob_name).splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rid = rec.get("custom_id") or rec.get("recordId")
            if rid:
                by_record[rid] = rec

    # 5. Write replies in the original order.
    completed = failed = 0
    for entry in entries:
        custom_id = entry[3]
        rec = by_record.get(custom_id)
        if rec is None:
            _write_failure(entry, "no record returned in vertex batch output", output_dir)
            failed += 1
            continue
        if rec.get("error") or rec.get("status"):
            _write_failure(
                entry, f"vertex error: {rec.get('error') or rec.get('status')}", output_dir
            )
            failed += 1
            continue
        model_out = rec.get("response") or rec.get("prediction")
        if model_out is None:
            _write_failure(entry, "missing response in vertex record", output_dir)
            failed += 1
            continue
        try:
            answer, usage = extract_output(api_style, model_out)
            _write_reply(
                entry, answer, model_out, usage,
                model, output_dir, ground_truth_index, judge_model)
            completed += 1
        except Exception as exc:
            _write_failure(entry, f"reply parse error: {exc}", output_dir)
            failed += 1
    logger.info(f"[vertex-batch] {model}: completed={completed} failed={failed}")
    return completed, failed, 0


# ---------------------------------------------------------------------------
# Sync path (one HTTP request per entry)
# ---------------------------------------------------------------------------
def run_vertex_sync(
    entries: List[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    ground_truth_index: Dict[str, Any],
    judge_model: str,
    cost_limit: Optional[float] = None,
    timeout_s: int = 900) -> Tuple[int, int, int]:
    cfg = VERTEX_MODELS[model]
    api_style = cfg["api_style"]

    est_cost, in_tok, out_tok = estimate_batch_cost(
        entries, model, max_output_tokens, VERTEX_SYNC_PRICE_MULTIPLIER)
    _enforce_cost_limit(cost_limit, est_cost, in_tok, out_tok, model, "vertex-sync")

    url = _sync_url(model)
    logger.info(f"[vertex-sync] {model}: POST {url}")

    completed = failed = 0
    total = len(entries)
    cumulative_cost = 0.0
    stopped = False
    for i, entry in enumerate(entries, 1):
        if stopped:
            _write_failure(entry, f"cost limit ${cost_limit:.2f} reached", output_dir)
            failed += 1
            continue
        _prompt_path, prompt_text, _idx, custom_id = entry
        body = build_request_body(model, prompt_text, max_output_tokens)
        try:
            resp = _post_with_retry(url, body, timeout_s)
            if resp.status_code >= 400:
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason} for {url}, body: {resp.text[:500]}"
                )
            raw = resp.json()
            answer, usage = extract_output(api_style, raw)
            est = _write_reply(
                entry, answer, raw, usage,
                model, output_dir, ground_truth_index, judge_model)[0]
            cumulative_cost += est
            completed += 1
            logger.info(
                f"[vertex-sync] [{i}/{total}] {custom_id} | "
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
            logger.warning(f"[vertex-sync] [{i}/{total}] {custom_id} failed: {exc}")
    return completed, failed, 0


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def run_gcp_eval(
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
    """Dispatch to the right Vertex flow for ``model``.

    Returns (completed, failed, skipped). Signature mirrors ``run_aws_eval``
    so ``run_pipeline.py`` can call either with the same arguments.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip-existing logic (identical to the AWS and Foundry paths).
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
    if provider != "vertex":
        raise ValueError(
            f"Model {model!r} is not registered as a Vertex model (provider={provider!r}). "
            f"Use run_foundry_eval for Foundry-served models, or run_aws_eval with "
            f"FB_INFERENCE_CLOUD=aws for the retired Bedrock routing."
        )

    if use_batch and VERTEX_MODELS[model].get("supports_batch", False):
        try:
            done, fail, _ = run_vertex_batch(
                pending, model, output_dir, max_output_tokens,
                ground_truth_index, judge_model,
                poll_interval=poll_interval, cost_limit=cost_limit)
            return done, fail, skipped
        except Exception as exc:
            if strict_batch:
                logger.error(
                    f"[vertex-batch] {model} batch failed and --strict-batch is set; "
                    f"not falling back."
                )
                raise
            logger.warning(
                f"[vertex-batch] {model} batch failed ({exc}); falling back to sync."
            )

    done, fail, _ = run_vertex_sync(
        pending, model, output_dir, max_output_tokens,
        ground_truth_index, judge_model, cost_limit=cost_limit)
    return done, fail, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GCP evaluation (Vertex AI)")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing prompt JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for replies")
    parser.add_argument("--questions", type=Path, required=True, help="Directory with Q&A ground truth JSONs")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(VERTEX_MODELS.keys()),
        help="Vertex-served model to evaluate")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-judge entirely. Free-form items get score=None.")
    # Accepted for parity with run_foundry_eval's CLI so run_pipeline can pass
    # the same argv to either backend. Only used as an Opik tag downstream.
    parser.add_argument("--eval-level", type=str, default=None)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-batch", dest="use_batch", action="store_true", default=True)
    parser.add_argument("--no-batch", dest="use_batch", action="store_false",
                        help="Force sync invocation per record (Claude only; the other "
                             "two Vertex models are sync-only regardless).")
    parser.add_argument("--strict-batch", action="store_true",
                        help="If batch submission fails, error out instead of falling back "
                             "to sync. Useful for debugging batch-specific issues.")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Batch polling interval in seconds (default: 30)")
    parser.add_argument("--cost-limit", type=float, default=20.0,
                        help="USD pre-flight cost limit. Vertex batch is estimated at "
                             "on-demand rates (no published batch discount); sync stops "
                             "mid-stream once the cap is hit. Pass 0 to disable. Note "
                             "this does NOT bound mistral-large-3, which bills per "
                             "GPU-hour for as long as its endpoint is up.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N prompts from --input (for smoke tests).")
    parser.add_argument("--batch-number", type=int, default=0,
                        help="0-indexed slice of size --batch-size to process.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Slice width for --batch-number.")
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s")

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
    completed, failed, skipped = run_gcp_eval(
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
