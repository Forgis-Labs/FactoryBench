#!/usr/bin/env bash
# FactoryBench, provision the GCP side of the inference pipeline.
#
# Creates everything the Vertex evaluation path needs and nothing it doesn't:
# enabled APIs, a regional GCS bucket for batch I/O, a service account, and the
# minimum IAM. It does NOT deploy any model and does NOT run any inference.
#
# Idempotent: safe to re-run. Every create is guarded by an existence check.
#
# Usage:
#   bash scripts/gcp/provision_inference.sh                 # dry run, prints the plan
#   bash scripts/gcp/provision_inference.sh --apply         # actually create
#
# Env overrides:
#   GCP_PROJECT   (default: anon-gcp-project)
#   GCP_REGION    (default: us-central1)  region for the batch bucket
#   GCS_BUCKET    (default: anon-batch-io)
#   FB_SA_NAME    (default: factorybench-inference)

set -euo pipefail

PROJECT="${GCP_PROJECT:-anon-gcp-project}"
REGION="${GCP_REGION:-us-central1}"
BUCKET="${GCS_BUCKET:-anon-batch-io}"
SA_NAME="${FB_SA_NAME:-factorybench-inference}"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

# gcloud is gcloud.cmd on Windows; the extension-less file is a shell script
# that Windows cannot spawn. Same resolution order as run_foundry_eval.
GCLOUD="${GCLOUD_BIN:-}"
if [[ -z "$GCLOUD" ]]; then
  for c in gcloud.cmd gcloud; do
    if command -v "$c" >/dev/null 2>&1; then GCLOUD="$(command -v "$c")"; break; fi
  done
fi
if [[ -z "$GCLOUD" ]]; then
  # Default SDK install locations, checked when the launcher isn't on PATH
  # (common in Git Bash, where the Windows PATH entry points at gcloud.cmd
  # but the shell resolves the extension-less Unix script instead).
  for c in \
    "$LOCALAPPDATA/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd" \
    "/c/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd" \
    "$HOME/google-cloud-sdk/bin/gcloud"; do
    if [[ -x "$c" || -f "$c" ]]; then GCLOUD="$c"; break; fi
  done
fi
if [[ -z "$GCLOUD" ]]; then
  echo "ERROR: could not find gcloud. Set GCLOUD_BIN to its full path." >&2
  exit 1
fi

run() {
  if [[ $APPLY -eq 1 ]]; then
    echo "+ $*"
    "$@"
  else
    echo "  [dry-run] $*"
  fi
}

echo "=== FactoryBench GCP inference provisioning ==="
echo "project : $PROJECT"
echo "region  : $REGION"
echo "bucket  : gs://$BUCKET"
echo "sa      : $SA_EMAIL"
[[ $APPLY -eq 0 ]] && echo "mode    : DRY RUN (pass --apply to execute)"
echo

# --- 1. APIs --------------------------------------------------------------
# aiplatform: Vertex prediction + batchPredictionJobs
# storage:    batch I/O bucket
# compute:    required for self-deployed endpoints (Mistral Large 3)
echo "--- 1. Enabling APIs"
for api in aiplatform.googleapis.com storage.googleapis.com compute.googleapis.com; do
  if "$GCLOUD" services list --enabled --project "$PROJECT" --format="value(config.name)" 2>/dev/null | grep -qx "$api"; then
    echo "  already enabled: $api"
  else
    run "$GCLOUD" services enable "$api" --project "$PROJECT"
  fi
done
echo

# --- 2. Batch I/O bucket --------------------------------------------------
# Must be in the same region as the batch prediction job, hence a regional
# (not multi-region) bucket. Uniform bucket-level access so IAM is the only
# access surface. 30-day lifecycle: batch inputs/outputs are reproducible
# intermediates, not artifacts worth paying to retain.
echo "--- 2. GCS bucket for Vertex batch I/O"
if "$GCLOUD" storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  echo "  already exists: gs://$BUCKET"
else
  run "$GCLOUD" storage buckets create "gs://$BUCKET" \
    --project "$PROJECT" \
    --location "$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

LIFECYCLE_JSON="$(mktemp)"
cat > "$LIFECYCLE_JSON" <<'EOF'
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30}}]}
EOF
run "$GCLOUD" storage buckets update "gs://$BUCKET" --lifecycle-file="$LIFECYCLE_JSON"
echo

# --- 3. Service account ---------------------------------------------------
# For unattended runs (CI, Cloud Batch). Interactive workstation runs use ADC
# from `gcloud auth application-default login` and don't need this.
echo "--- 3. Service account"
if "$GCLOUD" iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
  echo "  already exists: $SA_EMAIL"
else
  run "$GCLOUD" iam service-accounts create "$SA_NAME" \
    --project "$PROJECT" \
    --display-name "FactoryBench inference"
fi
echo

# --- 4. IAM ---------------------------------------------------------------
# aiplatform.user  : predict / rawPredict / batchPredictionJobs
# storage.objectAdmin scoped to the one bucket, NOT project-wide storage admin.
echo "--- 4. IAM bindings"
run "$GCLOUD" projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:$SA_EMAIL" \
  --role "roles/aiplatform.user" \
  --condition=None

run "$GCLOUD" storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member "serviceAccount:$SA_EMAIL" \
  --role "roles/storage.objectAdmin"
echo

# --- 5. Model Garden access notes ----------------------------------------
cat <<EOF
--- 5. Manual steps that cannot be scripted

  a) Anthropic Claude Sonnet 4.6 requires a one-time Model Garden enablement
     (the publisher's terms acceptance). Open:
       https://console.cloud.google.com/vertex-ai/publishers/anthropic/model-garden/claude-sonnet-4-6?project=$PROJECT
     and click Enable. Until then rawPredict returns 403.

  b) DeepSeek V3.2 MaaS is self-serve and needs no enablement step.

  c) Mistral Large 3 has NO managed offering on Vertex. It only ships as a
     self-deploy vLLM container on 8xH200 or 8xB200, billing by the hour
     whether or not it serves a request, so FactoryBench routes this model to
     Azure AI Foundry instead (see src/config.py). If you want it on Vertex
     anyway, deploy it from Model Garden by hand and check quota first:
       $GCLOUD compute regions describe $REGION --project $PROJECT

--- Add to .env
GCP_PROJECT=$PROJECT
GCP_REGION=$REGION
GCS_BUCKET=$BUCKET
GCS_PREFIX=factorybench/
# MISTRAL_LARGE_3_VERTEX_ENDPOINT=<only if you self-deploy it on Vertex>

--- Verify without spending anything
python scripts/gcp/preflight_check.py
EOF

[[ $APPLY -eq 0 ]] && echo && echo "Dry run complete. Nothing was created. Re-run with --apply."
exit 0
