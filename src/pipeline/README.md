# FactoryBench Pipelines

Unified end-to-end runner for FactoryBench: generate Q&A, build prompts,
evaluate against Microsoft Foundry models, and produce figures + Opik traces.

| Scripts | Purpose |
| --- | --- |
| [run_pipeline.py](run_pipeline.py) | Single entry point for levels 1-4 and all Foundry models. Use `--levels`, `--stages`, `--models` to scope the run. |
| [upload_qa_pairs.py](upload_qa_pairs.py) | Upload generated QA JSONs to a HF dataset repo under `<dataset_folder>/level_<N>/`. |

## Stages

`run_pipeline.py` executes the following stages (in order). Use `--stages` to
run a subset:

1. `generate`, Q&A generation (`src.question_generation.level{N}.level{N}`).
2. `fetch`, pull QA JSONs from a HF dataset repo into the local questions dir
   (mutually exclusive with `generate`). Requires `--hf-dataset-folder`.
3. `prompts`, prompt building (`src.question_generation.build_prompts_from_questions`).
4. `eval`, LLM evaluation (`src.evaluation.run_foundry_eval`), per model.

Default stages are `generate,prompts,eval`. Swap `generate` for `fetch`
to use Q&A pairs that are already uploaded to Hugging Face.

Outputs land under `datasets/{questions,prompts,replies}/level{N}_pipeline/`.
Replies are sharded by model slug (e.g. `.../gpt-5_1/`).

Post-run figures and aggregate analysis are produced manually via
[scripts/evaluate_opik_results.ipynb](../../scripts/evaluate_opik_results.ipynb).

The Knowledge Graph is only downloaded when the `prompts` stage is active.

## Examples of Usage

```bash
# Everything: all levels, all Foundry models, all stages
python -m src.pipeline.run_pipeline

# Only level 2 with all Foundry models and all stages
python -m src.pipeline.run_pipeline --levels 2

# Only generate Q&A + prompts (no model inference) for level 2, 10 questions per template and seed 42
python -m src.pipeline.run_pipeline --stages generate,prompts --levels 2 --test-mode --t 10 --seed 42

# Run inference against QA pairs already uploaded to HF (the authors/FactoryBench_QA_pairs)
# Uses factorynet_qa_260/level_1/ with two models. Skips generation as is already done and upload to HF
python -m src.pipeline.run_pipeline \
    --stages fetch,prompts,eval \
    --hf-dataset-folder factorynet_qa_260 \
    --levels 1 \
    --models gpt-5.1,claude-haiku-4-5
```

## Flags

- `--levels`, comma-separated levels to run (default `1,2,3`).
- `--stages`, comma-separated stages (default `generate,prompts,eval`).
- `--models`, comma-separated Foundry models (default: all from [src/config.py](../config.py)).
- `--judge-model`, LLM-as-judge for free-form scoring (default from `src/config.py`).
- `--cost-limit`, USD cap forwarded to each eval run.
- `--max-output-tokens`, forwarded to eval.
- `--dataset-repo`, HF source dataset (default `the authors/FactoryNet_Dataset`).
- `--seed`, `--test-mode`, `-n/--num-questions`, `-t/--questions-per-template`, forwarded to generators.

The Foundry model catalog (names, endpoints, API style) is defined in [src/config.py](../config.py). Add a model there and every pipeline/eval script picks it up automatically.

## Uploading generated QA pairs to Hugging Face

`upload_qa_pairs.py` uploads the JSON files from `datasets/questions/level{N}_pipeline/`
to an HF dataset repo (default `the authors/FactoryBench_QA_pairs`), organized as
`<dataset_folder>/level_<N>/`. All files are grouped into a single commit.


```bash
# Upload level 1 QA pairs to HF the folder datasets/questions/level1_pipeline into factorynet_qa_260 HF folder
python -m src.pipeline.upload_qa_pairs --input datasets/questions/level1_pipeline --dataset-folder factorynet_qa_260 --level 1
```

Use a different `--dataset-folder` (e.g. `factorywave_qa_260`) for the
FactoryWave dataset once it is normalized.

## Required `.env`

```bash
# Foundry routing (used by src/evaluation/run_foundry_eval.py)
AZURE_API_KEY="<your_api_key>"
CHAT_ENDPOINT="https://<your-foundry-resource>.services.ai.azure.com/openai/v1"
REASONING_ENDPOINT="https://<your-foundry-resource>.services.ai.azure.com/anthropic/v1"
PROJECT_ENDPOINT="https://<your-foundry-resource>.services.ai.azure.com/api/projects/<your-project>"

# Optional: separate Azure deployment for GPT-5.1 batch jobs.
# Azure /v1/batches rejects deployments on the `GlobalStandard` SKU (HTTP 400
# `invalid_deployment_type`). Create a second deployment of gpt-5.1 with SKU
# `globalbatch` (or `datazonebatch`) in Foundry, then set its name here. When
# unset, batch falls back to concurrent sync against the default deployment.
GPT_5_1_BATCH_DEPLOYMENT="gpt-5.1-batch"

# HuggingFace (datasets + KG)
HF_API_TOKEN="<your_hf_token>"

# Opik tracing
OPIK_API_KEY="<your_opik_key>"
OPIK_PROJECT_NAME="FactoryBench"
OPIK_WORKSPACE="anon-workspace"

# AWS routing (used by src/evaluation/run_aws_eval.py for Bedrock + SageMaker)
# Full setup walkthrough: src/evaluation/aws-setup.md
AWS_PROFILE="factorybench"                      # or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
FB_S3_BUCKET="anon-batch-io"             # any bucket the IAM roles can read+write
FB_S3_PREFIX="factorybench/"                    # optional, default 'factorybench/'

# Bedrock IAM
BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::<acct>:role/factorybench-bedrock-batch"

# Bedrock per-model id + region (regions differ; see aws-setup.md)
CLAUDE_SONNET_46_MODEL_ID="eu.anthropic.claude-sonnet-4-6"   # CRIS profile
CLAUDE_SONNET_46_REGION="eu-central-1"
MISTRAL_LARGE_3_MODEL_ID="mistral.mistral-large-3-675b-instruct"
MISTRAL_LARGE_3_REGION="us-west-2"                            # no EU region yet
DEEPSEEK_V32_MODEL_ID="deepseek.v3.2"
DEEPSEEK_V32_REGION="eu-west-2"                               # or eu-north-1

# SageMaker Async Inference (Qwen via JumpStart, scale-to-zero)
SAGEMAKER_ROLE_ARN="arn:aws:iam::<acct>:role/factorybench-sagemaker"
QWEN_SAGEMAKER_ENDPOINT="qwen-3-5-4b-async-endpoint"
QWEN_SAGEMAKER_REGION="eu-central-1"
```
