# AWS setup for factorybench

This guide walks through enabling the four reference models, creating IAM roles, populating `.env`, and running the first end-to-end Bedrock batch evaluation. **Every model id, region, and API behavior in this doc was re-verified against AWS docs at the time of writing, read the per-model notes carefully because not all of them work from `eu-central-1` directly.**

> **TL;DR:** Frankfurt as the source region works for Claude (via EU CRIS) and Mistral 7B / Large 2 (in-region). It does **not** work for Mistral Large 3 (no EU region yet) or DeepSeek V3.1 (London/Stockholm only in EU). Qwen 3.5 4B is a SageMaker JumpStart deploy.

---

## 0. One-time account setup

### Model access (no longer manual for most models)

As of October 15, 2025 AWS removed the manual "request model access" step for nearly all serverless foundation models, they're auto-enabled per account. **Anthropic models still require a one-time First-Time Usage form submission** in the Bedrock console (a use-case description box, typically auto-approved in under a minute). Some Marketplace models also still require explicit subscription.

Practical implication: skip the model-access dance for Mistral, DeepSeek, Meta, Qwen-on-Bedrock, etc. For Anthropic Claude Sonnet 4.6, fill in the First-Time Usage form once.

---

## 1. Per-model enablement

### 1.1 Claude Sonnet 4.6: `eu.anthropic.claude-sonnet-4-6`

**Verified id:** `eu.anthropic.claude-sonnet-4-6` (Geo CRIS profile, EU). The base id `anthropic.claude-sonnet-4-6` works only from a region where the model is hosted in-region; from `eu-central-1` you must use a CRIS profile.

**Why the `eu.` prefix:** the EU profile routes across `eu-north-1`, `eu-west-3`, `eu-south-1`, `eu-south-2`, `eu-west-1`, `eu-central-1`. If your SCPs deny any of those regions the call will fail with `AccessDeniedException` even if the source region is allowed. This is the most common first-time failure.

**Steps:**
1. Bedrock console → `eu-central-1` → **Model catalog** → **Anthropic, Claude Sonnet 4.6** → submit the First-Time Usage form.
2. Verify the profile resolves and capture the destination region list:
   ```bash
   aws bedrock get-inference-profile \
     --inference-profile-identifier eu.anthropic.claude-sonnet-4-6 \
     --region eu-central-1
   ```
3. The destination-region list in the response is the set you must allow in IAM.

**`.env`:**
```
CLAUDE_SONNET_46_MODEL_ID=eu.anthropic.claude-sonnet-4-6
CLAUDE_SONNET_46_REGION=eu-central-1
```

### 1.2 Mistral Large 3: `mistral.mistral-large-3-675b-instruct`

**Verified id:** `mistral.mistral-large-3-675b-instruct`. 675B-parameter MoE (41B active), 256K context, multimodal.

**Region issue:** at time of writing Mistral Large 3 is in `ap-northeast-1`, `ap-south-1`, `ap-southeast-2`, `sa-east-1`, `us-east-1`, `us-east-2`, `us-west-2`, plus a global CRIS profile. **No EU region yet.** Three options:

| Option | Configuration |
|---|---|
| Call cross-region from Frankfurt to `us-west-2` | `MISTRAL_REGION=us-west-2`, accept higher latency and US data residency |
| Use the global CRIS profile (if available for this model) | `global.mistral.mistral-large-3-675b-instruct`, verify with `aws bedrock list-inference-profiles --region us-west-2` |
| Substitute Mistral Large 2 (`mistral.mistral-large-2407-v1:0`) which **is** in `eu-central-1` | Different model, different evals |

If your project requires EU residency, you currently cannot use Mistral Large 3, pick option 3 and update the slug to `mistral-large-2`.

**`.env` (assuming option 1):**
```
MISTRAL_LARGE_3_MODEL_ID=mistral.mistral-large-3-675b-instruct
MISTRAL_LARGE_3_REGION=us-west-2
```

### 1.3 DeepSeek V3.1: fully managed, **not in Frankfurt**

**Verified id:** `deepseek.v3-1` (some docs render it as `deepseek.v3.1`, check the exact id in the model card's *Programmatic Access* section in your console; AWS has used both formats across pages).

**Region:** US West (Oregon), Asia Pacific (Tokyo), Asia Pacific (Mumbai), Europe (London `eu-west-2`), Europe (Stockholm `eu-north-1`). **Not in `eu-central-1`.**

**Subscription:** the original spec said "subscribe to the DeepSeek V3.1 listing in Bedrock Marketplace." That was true at launch. As of the September 19, 2025 update DeepSeek V3.1 is a **fully-managed serverless model** and access is auto-enabled per account, no Marketplace subscribe step. (Marketplace deployment still applies to the older R1 distill variants if those are what you actually want.)

**Steps:**
1. Open the Bedrock console in `eu-west-2` or `eu-north-1`.
2. Confirm the exact id from the model card:
   ```bash
   aws bedrock list-foundation-models --region eu-west-2 \
     --query "modelSummaries[?contains(modelId, 'deepseek')].modelId"
   ```
3. Be aware: tool calling via Converse is **not** yet supported for V3.1 on Bedrock. If your dispatcher needs tool use, this matters.

**`.env`:**
```
DEEPSEEK_V31_MODEL_ID=deepseek.v3-1
DEEPSEEK_V31_REGION=eu-west-2
```

### 1.4 Qwen 3.5 4B via SageMaker JumpStart

**Verified existence:** Qwen3.5-4B was added to SageMaker JumpStart in late April 2026, alongside Qwen3-Coder-Next, Qwen3-30B-A3B, Qwen3-30B-A3B-Thinking-2507, and Qwen3-Coder-30B-A3B-Instruct. It is described as supporting unified vision-language training and 201 languages. (Apologies for my earlier doubt, this release is recent enough that it slipped past my first check.)

**Steps:**
1. Open SageMaker Studio in a region that hosts the JumpStart catalog (Frankfurt `eu-central-1` works for JumpStart). Confirm the model is listed in your region, JumpStart catalog availability varies by region.
2. JumpStart → search "Qwen3.5-4B" → **Deploy**.
3. Choose an instance, for a 4B model, `ml.g5.xlarge` or `ml.g5.2xlarge` is appropriate. For a vision-language model expect higher memory needs; check the JumpStart page's instance recommendation.
4. **For cost:** in the deployment dialog, choose **Asynchronous Inference** (not real-time) and set min instances = 0 with auto-scaling, so the endpoint scales to zero between batches.
5. Note the resulting endpoint name.

**`.env`:**
```
QWEN_SAGEMAKER_ENDPOINT=qwen-3-5-4b-async-endpoint
QWEN_SAGEMAKER_REGION=eu-central-1
```

---

## 2. IAM roles

### 2.1 Bedrock batch role (`BEDROCK_BATCH_ROLE_ARN`)

**Trust policy:** principal `bedrock.amazonaws.com` for `sts:AssumeRole`.

**Critical for cross-region:** when invoking a CRIS profile, the IAM `Resource` must include both the **inference-profile ARN** AND every **destination foundation-model ARN** in every destination region. SCPs must not deny any destination region.

**Permissions policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvokeAndBatch",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:CreateModelInvocationJob",
        "bedrock:GetModelInvocationJob",
        "bedrock:ListModelInvocationJobs",
        "bedrock:StopModelInvocationJob"
      ],
      "Resource": [
        "arn:aws:bedrock:eu-central-1:*:inference-profile/eu.anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-north-1::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-west-3::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-south-1::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-south-2::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:eu-central-1::foundation-model/anthropic.claude-sonnet-4-6",
        "arn:aws:bedrock:us-west-2::foundation-model/mistral.mistral-large-3-675b-instruct",
        "arn:aws:bedrock:eu-west-2::foundation-model/deepseek.v3-1"
      ]
    },
    {
      "Sid": "S3InputOutput",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::anon-batch-io",
        "arn:aws:s3:::anon-batch-io/*"
      ]
    }
  ]
}
```

> Run `aws bedrock get-inference-profile --inference-profile-identifier eu.anthropic.claude-sonnet-4-6` and copy the destination ARNs from the `models` field directly, the EU profile's destination set may change as AWS adds regions.

**Batch inference quota gotcha:** Bedrock limits 10 concurrent batch jobs per model per region by default. If factorybench fans out across many models that's still fine; if it submits 11+ jobs against one model, the 11th will fail until one finishes.

### 2.2 SageMaker role (`SAGEMAKER_ROLE_ARN`)

**Trust policy:** principal `sagemaker.amazonaws.com`.

**Permissions policy** (factorybench is using async endpoints, but the spec calls for batch transform support too, both included):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SageMakerInferenceAndBatch",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTransformJob",
        "sagemaker:DescribeTransformJob",
        "sagemaker:StopTransformJob",
        "sagemaker:InvokeEndpoint",
        "sagemaker:InvokeEndpointAsync",
        "sagemaker:DescribeEndpoint",
        "sagemaker:DescribeEndpointConfig"
      ],
      "Resource": [
        "arn:aws:sagemaker:eu-central-1:*:endpoint/qwen-3-5-4b-async-endpoint",
        "arn:aws:sagemaker:eu-central-1:*:endpoint-config/qwen-3-5-4b-*",
        "arn:aws:sagemaker:eu-central-1:*:transform-job/factorybench-*"
      ]
    },
    {
      "Sid": "S3InputOutput",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::anon-batch-io",
        "arn:aws:s3:::anon-batch-io/*"
      ]
    },
    {
      "Sid": "ECRPullForJumpStartImage",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:CreateLogGroup",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    }
  ]
}
```

After deploying the JumpStart endpoint, tighten the ECR `*` to the specific image ARN visible in `DescribeEndpointConfig`.

**Capture both ARNs:**
```
BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::<account-id>:role/factorybench-bedrock-batch
SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/factorybench-sagemaker
```

---

## 3. Install boto3

```bash
pip install boto3
```

The dispatcher imports `boto3` lazily, so `python -m src.pipeline.run_pipeline --help` works without it. Any actual AWS call raises `ImportError` until installed.

Local credentials:
```bash
aws configure --profile factorybench
export AWS_PROFILE=factorybench
export AWS_REGION=eu-central-1
```

The profile needs `sts:AssumeRole` for both role ARNs.

---

## 4. First run

```bash
python -m src.pipeline.run_pipeline \
  --stages eval \
  --levels 1 \
  --models claude-sonnet-4.6 \
  --questions-dir output/questions \
  --prompts-dir "output/prompts/level{level}" \
  --replies-dir "output/replies/level{level}/{slug}"
```

This exercises the Bedrock batch path on Claude Sonnet 4.6. If anything's misconfigured, the dispatcher falls back to synchronous per-record invocation, the run still completes, slower, with `INFO`-level fallback warnings.

### Common first-run failures

| Symptom | Likely cause |
|---|---|
| `AccessDeniedException` on `CreateModelInvocationJob` | IAM `Resource` missing one of the EU CRIS destination foundation-model ARNs (most often `eu-south-2`) |
| `ValidationException: The provided ARN is invalid for the service region` | Source region of the `bedrock` client doesn't include the inference profile, or you passed a base model id where a profile is required |
| `ValidationException: model X not found` | Wrong region, Mistral Large 3 needs `us-west-2` etc., DeepSeek V3.1 needs `eu-west-2` / `eu-north-1` |
| `ServiceQuotaExceededException` on batch creation | Hit the 10-concurrent-jobs-per-model-per-region limit |
| Falls back to sync immediately, log says "model does not support batch" | Confirm in *Supported Regions and models for batch inference*, not every model supports `CreateModelInvocationJob` |
| `botocore.exceptions.NoCredentialsError` | `boto3` installed but profile/env credentials not resolved |

---

## Appendix: verified model id quick reference

| factorybench slug | Provider | Service | Source region | Id / endpoint | Notes |
|---|---|---|---|---|---|
| `claude-sonnet-4.6` | Anthropic | Bedrock | `eu-central-1` | `eu.anthropic.claude-sonnet-4-6` | EU CRIS profile; destination set includes `eu-south-2`, must allow in SCP |
| `mistral-large-3` | Mistral | Bedrock | `us-west-2` (no EU yet) | `mistral.mistral-large-3-675b-instruct` | 675B MoE, 41B active, 256K ctx |
| `deepseek-v3.1` | DeepSeek | Bedrock | `eu-west-2` or `eu-north-1` | `deepseek.v3-1` | Tool calling not supported on Bedrock Converse |
| `qwen-3.5-4b` | Alibaba | SageMaker JumpStart | `eu-central-1` | endpoint name from deploy | Vision-language; deploy with async + scale-to-zero |

## Appendix: changes from the original spec

| Original spec | Verified correction |
|---|---|
| `anthropic.claude-sonnet-4-6-*` in Frankfurt | Use `eu.anthropic.claude-sonnet-4-6` (CRIS profile) |
| `mistral.mistral-large-3-*` in Frankfurt | Mistral Large 3 not in EU, use `us-west-2` or fall back to Mistral Large 2 |
| Subscribe DeepSeek V3.1 in Marketplace | V3.1 is fully-managed serverless; auto-enabled, no subscribe step. Region must be `eu-west-2` / `eu-north-1` |
| Request access in Bedrock console for each model | Mostly obsolete since Oct 15, 2025; only Anthropic still requires First-Time Usage form |
| Qwen 3.5 4B in JumpStart | Confirmed, released April 2026 |
