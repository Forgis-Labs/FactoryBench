"""Shared configuration for FactoryBench pipeline and evaluation.

Single source of truth for the model catalog (provider routing, endpoints,
batch support) and pipeline-wide defaults. Adding a new model is one dict edit
here and every pipeline/evaluation script picks it up automatically.

Providers:

  * ``foundry``, OpenAI-compatible HTTP endpoints (Azure AI Foundry for
                    GPT-5.x; direct OpenAI for the agentic baseline; Vertex
                    MaaS for the two Qwen models). The foundry path handles
                    per-model overrides for the base URL, API key env var, and
                    the upstream model id sent to ``chat.completions.create``.
  * ``vertex``, Google Cloud Vertex AI. Serves the three models that used
                    to run on AWS Bedrock. Native batch prediction (GCS-in /
                    GCS-out) for Anthropic; sync for the rest. Auth is ADC
                    OAuth, no long-lived key.
  * ``bedrock``, AWS Bedrock. Pre-migration routing, retained in
                    ``LEGACY_AWS_MODELS`` only. Reachable by setting
                    FB_INFERENCE_CLOUD=aws.
  * ``sagemaker``, AWS SageMaker Async Inference. Unused; the JumpStart Qwen
                    deploy was flaky and Qwen now runs on Vertex.

Each model declares its own region because availability differs per model. On
Vertex, Claude Sonnet 4.6 is served from ``global``, DeepSeek V3.2 MaaS only
from ``us-central1``, and Mistral Large 3 has no MaaS offering at all (it is a
self-deployed vLLM endpoint). See ``src/evaluation/gcp-setup.md`` for the
verified id / region table and ``src/evaluation/aws-setup.md`` for the retired
Bedrock one.

Model ids, regions and endpoints are resolved at runtime from environment
variables, the catalog ships without baking in values that change per cloud
project.
"""
from __future__ import annotations

from typing import Any, Dict, List


MODELS: Dict[str, Dict[str, Any]] = {
    # --- Azure Foundry (kept for OpenAI proxy; AWS does not host GPT-5.x) ---
    # Routed to OpenAI direct, not Azure Foundry. The ETH Foundry deployment
    # `gpt-5.1-1` still exists but rejects every inference call with
    # `400 The current operation is not allowed in this deployment`, on
    # chat/completions and /responses alike, streaming or not, with either auth
    # header. Parameter validation still fires (`max_tokens` is rejected in
    # favour of `max_completion_tokens`), which proves routing reaches the
    # model and the block is a deployment policy rather than a bad id.
    # `gpt-5.1-2025-11-13` on api.openai.com is the same pinned snapshot the
    # Azure deployment served, so the model under test is unchanged.
    # To go back to Azure once the deployment is re-enabled: set
    # endpoint_env/api_key_env to CHAT_ENDPOINT/AZURE_API_KEY and drop
    # model_id_default.
    "gpt-5.1-1": {
        "provider": "foundry",
        "endpoint_env": "GPT_5_1_BASE_URL",
        "endpoint_default": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model_id_env": "GPT_5_1_MODEL",
        "model_id_default": "gpt-5.1-2025-11-13",
        "api_style": "openai",
        "supports_batch": True,
        # Azure /v1/batches requires a deployment whose SKU is `globalbatch`
        # or `datazonebatch`. The default `gpt-5.1-1` deployment is
        # `GlobalStandard` (sync only) and rejects batch with HTTP 400. Set
        # GPT_5_1_BATCH_DEPLOYMENT to a separate batch-capable deployment
        # name; absent the env var, batch falls back to concurrent sync.
        "batch_deployment_env": "GPT_5_1_BATCH_DEPLOYMENT",
    },
    # Direct OpenAI route to the same underlying model. Used by the agentic
    # baseline because ETH's Foundry deployment blocks the sync tool-calling
    # path (`The current operation is not allowed in this deployment`), and
    # OpenAI's batch API is single-shot so tool-calling can't use it either.
    "gpt-5-openai": {
        "provider": "foundry",   # runner path (OpenAI-compatible HTTP)
        "endpoint_env": "OPENAI_BASE_URL",
        "endpoint_default": "https://api.openai.com/v1",
        "api_style": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model_id_default": "gpt-5",
        "supports_batch": False,   # we never use batch for agentic
    },

    # --- Google Cloud Vertex AI (migrated off AWS Bedrock, Aug 2026) ------
    # Same three checkpoints the paper evaluated on Bedrock, re-pointed at
    # Vertex. Publisher model ids verified against the live Model Garden
    # catalog for project `anon-gcp-project`, see src/evaluation/gcp-setup.md for
    # the listing command and the per-model serving notes.
    #
    # The three do NOT share one serving surface, which is the main structural
    # difference from Bedrock (where all three were serverless + batch):
    #
    #   claude-sonnet-4.6  MaaS, Anthropic Messages API over `:rawPredict`,
    #                      plus native Vertex batch prediction (GCS in/out).
    #   deepseek-v3.2      MaaS, OpenAI-compatible `/endpoints/openapi` route
    #                      (same shape as qwen-3-235b below). Sync only.
    #   mistral-large-3    NOT offered as MaaS on Vertex. Model Garden ships it
    #                      as a self-deploy vLLM container needing 8xH200 or
    #                      8xB200. Costs by the hour, not by the token.
    "claude-sonnet-4.6": {
        "provider": "vertex",
        "vertex_publisher": "anthropic",
        "model_id_env": "CLAUDE_SONNET_46_VERTEX_MODEL",
        "model_id_default": "claude-sonnet-4-6",
        "region_env": "CLAUDE_SONNET_46_VERTEX_REGION",
        "region_default": "global",
        "api_style": "vertex_anthropic",
        "supports_batch": True,
    },
    "deepseek-v3.2": {
        "provider": "vertex",
        "vertex_publisher": "deepseek-ai",
        "model_id_env": "DEEPSEEK_V32_VERTEX_MODEL",
        "model_id_default": "deepseek-ai/deepseek-v3.2-maas",
        "region_env": "DEEPSEEK_V32_VERTEX_REGION",
        # `global`, not us-central1. The Model Garden catalog lists this model
        # under us-central1, but the openapi serving route there returns
        # FAILED_PRECONDITION "not available in region 'us-central1'".
        # Verified by probing the region matrix: only `global` answers 200.
        "region_default": "global",
        "api_style": "vertex_openai",
        "supports_batch": False,
    },
    # Mistral Large 3 is served from Azure AI Foundry, not GCP. Vertex offers
    # it only as a self-deploy vLLM container on 8xH200 / 8xB200, which bills
    # per GPU-hour whether or not it is serving. The Foundry resource already
    # has a `Mistral-Large-3` deployment on the same endpoint as GPT-5.x, so
    # this is per-token and needs no new infrastructure. Full GCP parity would
    # mean standing that vLLM container up by hand; see gcp-setup.md section 6.
    "mistral-large-3": {
        "provider": "foundry",
        "endpoint_env": "CHAT_ENDPOINT",
        "endpoint_default": "https://<your-foundry-resource>.services.ai.azure.com/openai/v1",
        # `mistral` skips the /responses probe and sends max_tokens on
        # chat.completions, which is what this deployment accepts.
        "api_style": "mistral",
        "api_key_env": "AZURE_API_KEY",
        "model_id_env": "MISTRAL_LARGE_3_AZURE_MODEL",
        "model_id_default": "Mistral-Large-3",
        "supports_batch": False,
    },

    # --- Together AI (sync-only; OpenAI-compatible) ----------------------
    # Routed through the foundry path because Together exposes the same
    # /chat/completions API surface as Azure / OpenAI. Tried OpenRouter and
    # smaller Together variants first, none host qwen3 below 235B
    # serverless. Qwen/Qwen3-235B-A22B-Instruct-2507-tput is a 235B MoE
    # with 22B active per token; the `-tput` suffix is Together's reliable
    # serverless / throughput-tier marker.
    # Migrated to Vertex AI Model Garden MaaS after Together AI hit credit
    # exhaustion on the KUKA re-run. Same model variant (2507 instruct
    # release). Vertex MaaS is served via the global endpoint, per-token
    # billed, no dedicated GPU. Auth uses google-auth ADC (short-lived OAuth
    # token, refreshed per-call by resolve_api_key).
    "qwen-3-235b": {
        "provider": "foundry",
        "endpoint_env": "VERTEX_QWEN_BASE_URL",
        "endpoint_default": "https://aiplatform.googleapis.com/v1beta1/projects/anon-gcp-project/locations/global/endpoints/openapi",
        "api_style": "openai",
        "api_key_env": "VERTEX_OAUTH",  # sentinel: fetch via google-auth
        "model_id_env": "VERTEX_QWEN_MODEL",
        "model_id_default": "qwen/qwen3-235b-a22b-instruct-2507-maas",
        "supports_batch": False,
    },
    # Self-hosted on Vertex AI custom endpoint (vLLM OpenAI api_server on 1x T4).
    # Same underlying HF checkpoint as the paper's Together dedicated endpoint
    # (Qwen/Qwen3-4B-Instruct-2507). Vertex custom endpoints proxy requests via
    # `:rawPredict`, so this uses api_style=vertex_raw_predict which the runner
    # handles by POSTing the OpenAI payload to <endpoint_id>:rawPredict.
    # Endpoint ids are not stable: the original (4739621343144706048) was
    # deleted and this one was recreated by
    # scripts/gcp/deploy_qwen3_4b_vertex.py. Set VERTEX_QWEN4B_BASE_URL to
    # override after any redeploy rather than editing this default.
    "qwen-3-4b": {
        "provider": "foundry",
        "endpoint_env": "VERTEX_QWEN4B_BASE_URL",
        "endpoint_default": "https://us-central1-aiplatform.googleapis.com/v1/projects/000000000000/locations/us-central1/endpoints/7263396353076625408",
        "api_style": "vertex_raw_predict",
        "api_key_env": "VERTEX_OAUTH",
        "model_id_env": "VERTEX_QWEN4B_MODEL",
        "model_id_default": "qwen-3-4b",
        "supports_batch": False,
    },
}

# --- Legacy AWS catalog ---------------------------------------------------
# The pre-migration Bedrock routing for the same three slugs. Kept so
# ``src.evaluation.run_aws_eval`` still imports and can reproduce the
# published numbers, and so the migration is reversible without a git revert.
# NOT part of ``MODELS``, the live pipeline routes these slugs to Vertex.
# To fall back to AWS for a run, set FB_INFERENCE_CLOUD=aws (see
# ``resolve_provider`` below).
LEGACY_AWS_MODELS: Dict[str, Dict[str, Any]] = {
    "claude-sonnet-4.6": {
        "provider": "bedrock",
        "model_id_env": "CLAUDE_SONNET_46_MODEL_ID",        # eu.anthropic.claude-sonnet-4-6 (CRIS)
        "region_env": "CLAUDE_SONNET_46_REGION",            # eu-central-1
        "api_style": "anthropic",
        "supports_batch": True,
    },
    "mistral-large-3": {
        "provider": "bedrock",
        "model_id_env": "MISTRAL_LARGE_3_MODEL_ID",         # mistral.mistral-large-3-675b-instruct
        "region_env": "MISTRAL_LARGE_3_REGION",             # us-west-2 (no EU region yet)
        "api_style": "mistral",
        "supports_batch": True,
    },
    "deepseek-v3.2": {
        "provider": "bedrock",
        "model_id_env": "DEEPSEEK_V32_MODEL_ID",            # deepseek.v3.2
        "region_env": "DEEPSEEK_V32_REGION",                # eu-west-2 or eu-north-1
        "api_style": "deepseek",
        "supports_batch": True,
    },
}

# Per-provider views derived from the unified MODELS table.
FOUNDRY_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in MODELS.items() if cfg.get("provider") == "foundry"
}
VERTEX_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in MODELS.items() if cfg.get("provider") == "vertex"
}
BEDROCK_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in LEGACY_AWS_MODELS.items() if cfg.get("provider") == "bedrock"
}
SAGEMAKER_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in LEGACY_AWS_MODELS.items() if cfg.get("provider") == "sagemaker"
}

MODEL_NAMES: List[str] = list(MODELS.keys())
FOUNDRY_MODEL_NAMES: List[str] = list(FOUNDRY_MODELS.keys())
VERTEX_MODEL_NAMES: List[str] = list(VERTEX_MODELS.keys())
BEDROCK_MODEL_NAMES: List[str] = list(BEDROCK_MODELS.keys())
SAGEMAKER_MODEL_NAMES: List[str] = list(SAGEMAKER_MODELS.keys())


def _cloud_override() -> str:
    """Which cloud the Bedrock-era slugs should route to for this run.

    ``gcp`` (default) uses the Vertex entries in ``MODELS``. ``aws`` restores
    the pre-migration Bedrock routing from ``LEGACY_AWS_MODELS`` so the
    published numbers stay reproducible without editing this file.
    """
    import os
    return (os.getenv("FB_INFERENCE_CLOUD") or "gcp").strip().lower()


def get_model_config(model_name: str) -> "Dict[str, Any] | None":
    if _cloud_override() == "aws" and model_name in LEGACY_AWS_MODELS:
        return LEGACY_AWS_MODELS[model_name]
    return MODELS.get(model_name)


def get_provider(model_name: str) -> "str | None":
    cfg = get_model_config(model_name)
    return cfg.get("provider") if cfg else None


def get_vertex_publisher(model_name: str) -> "str | None":
    """Model Garden publisher namespace (``anthropic``, ``deepseek-ai``...)."""
    cfg = get_model_config(model_name) or {}
    return cfg.get("vertex_publisher")


def get_region(model_name: str) -> "str | None":
    """Resolve a model's serving region.

    ``region_env`` wins when populated, then the model's ``region_default``.
    Vertex models fall back to ``GCP_REGION`` then ``GCP_REGION_DEFAULT``;
    availability is per-model, so this is not one global setting (Claude is
    served from `global`, DeepSeek MaaS only from `us-central1`).
    """
    import os
    cfg = get_model_config(model_name) or {}
    env_key = cfg.get("region_env")
    if env_key:
        override = os.getenv(env_key)
        if override:
            return override
    if cfg.get("region_default"):
        return cfg["region_default"]
    if cfg.get("provider") == "vertex":
        return os.getenv("GCP_REGION") or GCP_REGION_DEFAULT
    return None


def get_upstream_model_id(model_name: str) -> str:
    """Upstream model id to send to the API.

    Lets the FactoryBench-side name diverge from the provider's model id, e.g. ``qwen-3-4b`` (FactoryBench) -> ``qwen/qwen3-4b`` (OpenRouter). When
    ``model_id_env`` is set and populated it wins; otherwise falls back to
    ``model_id_default``; otherwise the FactoryBench name itself.
    """
    import os
    cfg = get_model_config(model_name) or {}
    env_key = cfg.get("model_id_env")
    if env_key:
        override = os.getenv(env_key)
        if override:
            return override
    return cfg.get("model_id_default") or model_name


def get_api_key_env(model_name: str) -> "str | None":
    """Env var name to consult for this model's API key, if any.

    When a foundry-style model declares ``api_key_env``, that env var is
    checked before falling back to the default Azure/OpenAI keys. Lets
    multiple OpenAI-compatible providers (Azure, OpenRouter...) coexist
    without sharing one key.
    """
    cfg = get_model_config(model_name) or {}
    return cfg.get("api_key_env")


def get_batch_deployment(model_name: str) -> str:
    """Deployment name to send to Azure /v1/batches for ``model_name``.

    Returns the value of ``batch_deployment_env`` (when set on a foundry model
    AND the env var is populated), otherwise the model name itself. Lets a
    sync-only ``GlobalStandard`` deployment coexist with a separate
    ``globalbatch`` deployment used only by batch jobs.
    """
    import os
    cfg = get_model_config(model_name) or {}
    env_key = cfg.get("batch_deployment_env")
    if env_key:
        override = os.getenv(env_key)
        if override:
            return override
    return model_name


DEFAULT_JUDGE_MODEL: str = "gpt-5.1-1"

# AWS defaults (can be overridden per-call or via env). Only reached when
# FB_INFERENCE_CLOUD=aws puts the legacy Bedrock catalog back in play.
AWS_REGION_DEFAULT: str = "eu-central-1"  # Frankfurt

# GCP defaults (can be overridden per-call or via env).
# GCP_PROJECT / GCP_REGION override these; per-model region_env wins over both.
GCP_PROJECT_DEFAULT: str = "anon-gcp-project"
GCP_REGION_DEFAULT: str = "us-central1"
# Bucket for Vertex batch-prediction I/O, the GCS analogue of FB_S3_BUCKET.
# Must live in the same region as the batch job. Set via GCS_BUCKET.
GCS_BUCKET_DEFAULT: str = "anon-batch-io"
GCS_PREFIX_DEFAULT: str = "factorybench/"
