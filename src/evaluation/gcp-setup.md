# GCP setup for factorybench

This is the Vertex AI replacement for `aws-setup.md`. It covers the same three models the paper ran on Bedrock, re-pointed at Google Cloud, plus the provisioning, auth and teardown around them.

**Every model id and region below was verified against the live Model Garden catalog for project `anon-gcp-project` on 2026-08-02** using the listing command in section 1. Re-run it before trusting this doc: Model Garden ids change more often than Bedrock ones.

> **TL;DR:** Claude and DeepSeek migrate cleanly and get cheaper to operate (no batch role, no per-region bucket dance, ADC instead of long-lived keys). Mistral Large 3 does not migrate cleanly: Vertex has no managed offering, only a self-deploy 8-GPU vLLM endpoint billed by the hour. Decide what to do about Mistral before you start.

---

## 0. What changed from AWS

| | AWS (before) | GCP (after) |
|---|---|---|
| Claude Sonnet 4.6 | Bedrock serverless, EU CRIS profile, native batch | Vertex MaaS, `:rawPredict`, native batch prediction |
| DeepSeek V3.2 | Bedrock serverless, native batch | Vertex MaaS, OpenAI-compatible route, **sync only** |
| Mistral Large 3 | Bedrock serverless, native batch | **Self-deploy only**, 8xH200, hourly billing, sync only |
| Auth | IAM user + two assumed roles | ADC OAuth, no long-lived key |
| Batch I/O | S3, one bucket per model region | One regional GCS bucket |
| Batch discount | 50% off on-demand | none published |
| Data residency | EU achievable for Claude | **US only** for all three |

Two of those rows are regressions and you should decide about them deliberately rather than discover them mid-run. See section 6.

---

## 1. Verify the catalog first

Everything downstream depends on the publisher model ids being right. List them:

```bash
TOKEN=$(gcloud auth print-access-token)
for pub in anthropic deepseek-ai mistralai; do
  curl -s -H "Authorization: Bearer $TOKEN" \
       -H "x-goog-user-project: anon-gcp-project" \
    "https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/$pub/models" \
  | python -c "import sys,json;print('$pub:',[m['name'].split('/')[-1] for m in json.load(sys.stdin).get('publisherModels',[])])"
done
```

What that returned on 2026-08-02:

| factorybench slug | Publisher | Model Garden id | Region | Surface | Batch |
|---|---|---|---|---|---|
| `claude-sonnet-4.6` | `anthropic` | `claude-sonnet-4-6` | `global`, `us-central1` | Anthropic Messages over `:rawPredict` | yes |
| `deepseek-v3.2` | `deepseek-ai` | `deepseek-v3.2-maas` | **`global`** | OpenAI-compatible `/endpoints/openapi` | no |
| `mistral-large-3` | `mistralai` | `mistral-large-3` (`mistral-large-3-instruct-2512`) | self-deploy | vLLM container on your GPUs | no |

The metadata endpoint 404s in `europe-west1`, `europe-west4`, `europe-west9` and `us-east5` for all three. There is no EU option today.

**Catalog presence is not serving availability.** These are separate checks and they disagree for both MaaS models:

- DeepSeek V3.2 is *listed* under `us-central1`, but serving there returns `FAILED_PRECONDITION: not available in region 'us-central1'`. Probing the region matrix, only `global` answers 200. `region_default` is set accordingly.
- Claude Sonnet 4.6 *lists* fine at `global` and `us-central1`, but `:rawPredict` returns 404 in **every** region until the project accepts the Anthropic offer (section 3). The message says "not found or your project does not have access to it", which reads like a wrong model id and is actually a missing entitlement.

So verify serving, not just listing: run `preflight_check.py`, then send one cheap call per model.

---

## 2. Auth

Vertex uses Application Default Credentials. There is no API key to put in `.env`.

**Workstation:**
```bash
gcloud auth application-default login
gcloud config set project anon-gcp-project
```

`gcloud auth login` alone is **not** enough, the runner needs the *application-default* variant. If only the plain login is present, `src/evaluation/run_foundry_eval._vertex_access_token()` falls back to shelling out to `gcloud auth print-access-token`, which works but re-shells on every token refresh.

**Unattended (CI, Cloud Batch, GCE):** the metadata server supplies ADC automatically once the VM or job runs as the service account from section 3. No key file needed, and none should be created.

Every request also sends `x-goog-user-project`. Without it, local ADC calls 403 with "authenticating by using local Application Default Credentials ... without a quota project".

---

## 3. Provision

```bash
bash scripts/gcp/provision_inference.sh            # dry run, prints the plan
bash scripts/gcp/provision_inference.sh --apply    # create
```

Idempotent. It enables `aiplatform`, `storage` and `compute`, creates a regional GCS bucket with uniform bucket-level access, public access prevention and a 30-day lifecycle rule, creates the `factorybench-inference` service account, and grants:

- `roles/aiplatform.user` at project level (predict, rawPredict, batchPredictionJobs)
- `roles/storage.objectAdmin` scoped to the one bucket, not project-wide

That is the whole IAM surface. Compare with the two hand-written Bedrock/SageMaker policies in `aws-setup.md` and their cross-region CRIS destination-ARN list, which no longer has an equivalent.

### The one manual step

Anthropic models require a one-time Model Garden enablement (the publisher's terms acceptance) before `:rawPredict` will answer:

https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-sonnet-4-6?project=anon-gcp-project

`preflight_check.py` flags this: the model metadata carries a `requestAccess` action until it is done. DeepSeek MaaS is self-serve and needs nothing.

---

## 4. `.env`

```
GCP_PROJECT=anon-gcp-project
GCP_REGION=us-central1
GCS_BUCKET=anon-batch-io
GCS_PREFIX=factorybench/

# All optional: defaults live in src/config.py and are the verified values.
# CLAUDE_SONNET_46_VERTEX_MODEL=claude-sonnet-4-6
# CLAUDE_SONNET_46_VERTEX_REGION=global
# CLAUDE_SONNET_46_VERTEX_BATCH_REGION=us-central1
# DEEPSEEK_V32_VERTEX_MODEL=deepseek-ai/deepseek-v3.2-maas
# DEEPSEEK_V32_VERTEX_REGION=global

# Only if you self-deploy Mistral Large 3 on Vertex (section 6):
# MISTRAL_LARGE_3_VERTEX_ENDPOINT=<numeric endpoint id>
```

`CLAUDE_SONNET_46_VERTEX_BATCH_REGION` exists because `batchPredictionJobs` is not served from the `global` endpoint, and `global` is where Claude's sync traffic goes by default. Batch falls back to `us-central1`.

---

## 5. Preflight

```bash
python scripts/gcp/preflight_check.py
```

Metadata GETs only. It never sends a prompt, so it costs nothing and cannot be mistaken for an eval run. It checks ADC, the bucket, and each model's presence in the catalog at its configured region, and it tells you which models still need work. Expected output before the bucket exists and before any Mistral deploy:

```
[PASS] ADC token resolved (260 chars)
[FAIL] bucket gs://anon-batch-io does not exist
[PASS] claude-sonnet-4.6: anthropic/claude-sonnet-4-6 @ global (launchStage=GA, version=default)
       note: this publisher requires a one-time Model Garden enablement before rawPredict succeeds.
[PASS] deepseek-v3.2: deepseek-ai/deepseek-v3.2-maas @ us-central1 (launchStage=GA, version=001)
[WARN] mistral-large-3: self-deploy model with no endpoint (MISTRAL_LARGE_3_VERTEX_ENDPOINT unset)
```

---

## 6. Mistral Large 3: read this before running the full sweep

On Bedrock, `mistral.mistral-large-3-675b-instruct` was serverless and per-token. Vertex does not offer it as MaaS. Model Garden publishes only a self-deploy configuration:

```
image   us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20251205_0916_RC01
model   gs://vertex-model-garden-restricted-us/mistralai/Mistral-Large-3-675B-Instruct-2512
shape   a3-ultragpu-8g (8x H200 141GB)  or  a4-highgpu-8g (8x B200)
args    --tensor-parallel-size=8 --tokenizer_mode=mistral --config_format=mistral --load_format=mistral
```

Same weights as the Bedrock model, so results stay comparable. The cost model does not: an 8-GPU A3-Ultra node is on the order of tens of dollars per hour, billed from endpoint creation to deletion regardless of traffic. `--cost-limit` in the runner bounds *token* spend and does nothing about this.

**What we actually did:** none of the above. `mistral-large-3` routes to Azure AI Foundry, which already has a `Mistral-Large-3` deployment on the same endpoint as GPT-5.x. Per-token, no new infrastructure, no idle burn. See its entry in `src/config.py`.

The alternatives, if you want it on GCP anyway:

1. **Self-deploy from Model Garden, run, tear down the same day.** Cheapest way to keep it on Vertex. Set `MISTRAL_LARGE_3_VERTEX_ENDPOINT` to the endpoint id afterwards. You will likely need a quota increase for `custom_model_serving_a3_ultra_gpus` in the target region.
2. **Drop `mistral-large-3` from the GCP sweep** and cite the Bedrock numbers for it, noting the serving difference. Honest and free.
3. **Keep only Mistral on AWS** via `FB_INFERENCE_CLOUD=aws` (section 8). Preserves the numbers but keeps an AWS dependency alive.

A `scripts/gcp/deploy_mistral_large_3.py` used to automate option 1. It was removed: the Foundry route made it unnecessary, and it had stopped working when `mistral-large-3` moved to `provider: "foundry"`, since it looked itself up in `VERTEX_MODELS` and raised `KeyError` on every subcommand.

---

## 7. Running an evaluation

Identical to before. `run_pipeline.py` routes on provider, so nothing in the invocation changes:

```bash
python -m src.pipeline.run_pipeline \
  --stages eval \
  --levels 1 \
  --models claude-sonnet-4.6 \
  --questions-dir output/questions \
  --prompts-dir "output/prompts/level{level}" \
  --replies-dir "output/replies/level{level}/{slug}"
```

Or drive the backend directly:

```bash
python -m src.evaluation.run_gcp_eval \
  --input output/prompts/level1 \
  --output-dir output/replies/level1/claude-sonnet-4_6 \
  --questions output/questions/level1 \
  --model claude-sonnet-4.6 \
  --cost-limit 5
```

Claude uses batch prediction when the run has enough records and falls back to sync on any submission error (`--strict-batch` to make that fatal instead). DeepSeek and Mistral are sync-only; passing `--no-batch` for them is a no-op.

Unlike Bedrock, Vertex batch has **no minimum record count**, so the `BEDROCK_BATCH_MIN_RECORDS = 100` guard has no counterpart here. Small batches submit fine.

---

## 8. Going back to AWS

The Bedrock catalog is retained in `src/config.py` as `LEGACY_AWS_MODELS`. One env var restores it:

```bash
FB_INFERENCE_CLOUD=aws python -m src.pipeline.run_pipeline --stages eval ...
```

`get_model_config` then returns the Bedrock entry for those three slugs and `run_pipeline` routes to `run_aws_eval`. Use this to reproduce the published numbers, or to keep Mistral on Bedrock while everything else runs on Vertex.

---

## 9. Common first-run failures

| Symptom | Likely cause |
|---|---|
| `403 ... authenticating by using local Application Default Credentials` | Missing `x-goog-user-project`. The runner sets it; if you are curling by hand, add it. |
| `403` on Claude `:rawPredict` but preflight passes | The one-time Anthropic Model Garden enablement in section 3 has not been done. |
| `404` on a publisher model | Wrong region. DeepSeek MaaS is `us-central1` only; Claude is `global` / `us-central1`. |
| `batchPredictionJobs` 400s with an invalid-location error | Batch was pointed at `global`. Set `CLAUDE_SONNET_46_VERTEX_BATCH_REGION`. |
| `mistral-large-3 is self-deployed on Vertex and has no endpoint yet` | Expected until section 6 is done. Deploy it or drop it from `--models`. |
| Deploy fails mentioning quota | `custom_model_serving_a3_ultra_gpus` not granted in that region. |
| `google-cloud-storage is required for Vertex batch I/O` | `pip install -r requirements.txt`. |
| `429 RESOURCE_EXHAUSTED` partway through a run | MaaS partner models throttle per project even under sequential load. The sync path retries with backoff (4s, 8s, 16s...); reruns skip completed records, so re-running finishes the stragglers. |
| `FAILED_PRECONDITION: not available in region` on DeepSeek | Region must be `global`, not `us-central1`. See section 1. |
| ADC warning about a missing quota project | Harmless here, but `gcloud auth application-default set-quota-project anon-gcp-project` silences it. |

---

## Appendix: files this migration touches

| File | Role |
|---|---|
| `src/config.py` | Vertex catalog, `LEGACY_AWS_MODELS`, `FB_INFERENCE_CLOUD` switch, region/publisher resolvers |
| `src/evaluation/run_gcp_eval.py` | The runner. Batch + sync, three api styles |
| `src/pipeline/run_pipeline.py` | Routes `provider == "vertex"` to the new runner |
| `scripts/gcp/provision_inference.sh` | APIs, bucket, service account, IAM |
| `scripts/gcp/preflight_check.py` | Read-only verification |
| `scripts/gcp/deploy_qwen3_4b_vertex.py` | Stands up the self-hosted Qwen3-4B custom endpoint |
| `src/evaluation/run_aws_eval.py` | Unchanged behavior, retained for `FB_INFERENCE_CLOUD=aws` |
