#!/usr/bin/env python3
"""Deploy / inspect / tear down Qwen3-4B-Instruct-2507 on Vertex AI.

The panel's lightweight open-weight reference point. Unlike Qwen3-235B, the 4B
has no Model Garden MaaS entry and no Model Garden self-deploy config either
(the `qwen/qwen3` publisher entry resolves to the 235B), so there is no
one-click `:deploy` path. It has to be stood up as a custom container: the
Model Garden vLLM image pulling `Qwen/Qwen3-4B-Instruct-2507` straight from
Hugging Face, fronted by an OpenAI-compatible route.

That is what `src/config.py` already expects, `qwen-3-4b` is declared with
`api_style: vertex_raw_predict`, which POSTs an OpenAI chat-completions payload
to `<endpoint>:rawPredict` and lets the container's own handler answer. The
previous endpoint (4739621343144706048) was deleted at some point; this script
recreates an equivalent one.

Cost is modest compared with the Mistral situation: one L4 or T4, order of
$0.35-0.85 per hour, and a 4B checkpoint loads in minutes rather than an hour.
It still bills continuously while up, so --create pairs with --delete and
nothing happens without --yes.

Context note: FactoryBench prompts run to ~13k tokens, so --max-model-len
defaults to 20480 rather than the container default. On a 16GB T4 that leaves
little KV headroom; L4 (24GB) is the default for that reason.

Usage:
    python scripts/gcp/deploy_qwen3_4b_vertex.py                 # plan only
    python scripts/gcp/deploy_qwen3_4b_vertex.py --create --yes
    python scripts/gcp/deploy_qwen3_4b_vertex.py --status
    python scripts/gcp/deploy_qwen3_4b_vertex.py --delete --yes

After --create, put the printed line in .env (config.py reads this var):
    VERTEX_QWEN4B_BASE_URL=<full endpoint URL>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import GCP_PROJECT_DEFAULT  # noqa: E402
from src.evaluation.run_foundry_eval import _vertex_access_token  # noqa: E402
from src.evaluation.test_gpt_5mini import load_dotenv_file  # noqa: E402

HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVED_NAME = "qwen-3-4b"
DISPLAY_NAME = "factorybench-qwen3-4b"
VLLM_IMAGE = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:20251205_0916_RC01"
)
CONTAINER_PORT = 8080

MACHINE_SHAPES = {
    # 24GB, comfortable for a 4B at 20k context. Default.
    "l4": {"machineType": "g2-standard-8", "acceleratorType": "NVIDIA_L4", "acceleratorCount": 1},
    # 16GB, what the original endpoint used. Cheaper, tighter on KV cache.
    "t4": {"machineType": "n1-standard-8", "acceleratorType": "NVIDIA_TESLA_T4", "acceleratorCount": 1},
    # 40GB. Roughly 4x the L4's bf16 throughput, which is what matters for the
    # Level-4 free-form items: those prompts run to ~22k tokens, and prefill on
    # an L4 took ~11 minutes per item, enough to make a 286-item run a 14-hour
    # job. Costs ~4x per hour but finishes ~4x sooner, so the total is a wash.
    "a100": {"machineType": "a2-highgpu-1g", "acceleratorType": "NVIDIA_TESLA_A100", "acceleratorCount": 1},
}

COST_WARNING = """
  COST: a T4 or L4 node is roughly $0.35-0.85 per hour and an A100 40GB is
  roughly $3.70; all bill from deployment until deletion regardless of traffic.
  Cheap next to an 8-GPU node, but not free, run --delete when the eval
  finishes.
"""


def _project() -> str:
    return os.environ.get("GCP_PROJECT") or GCP_PROJECT_DEFAULT


def _region() -> str:
    return os.environ.get("VERTEX_QWEN4B_REGION") or os.environ.get("GCP_REGION") or "us-central1"


def _host(region: str) -> str:
    return f"{region}-aiplatform.googleapis.com"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_vertex_access_token()}",
        "Content-Type": "application/json",
        "x-goog-user-project": _project(),
    }


def _wait_lro(region: str, op_name: str, what: str, timeout_s: int = 3600) -> dict:
    """Poll a long-running operation to completion."""
    url = f"https://{_host(region)}/v1/{op_name}"
    waited = 0
    while waited < timeout_s:
        r = requests.get(url, headers=_headers(), timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"{what}: operations.get {r.status_code} {r.text[:300]}")
        body = r.json()
        if body.get("done"):
            if body.get("error"):
                raise RuntimeError(f"{what} failed: {json.dumps(body['error'])[:500]}")
            return body.get("response", {})
        time.sleep(15)
        waited += 15
        if waited % 60 == 0:
            print(f"  ...{what} still running ({waited // 60} min)")
    raise TimeoutError(f"{what} did not finish within {timeout_s}s")


def cmd_status(project: str, region: str) -> int:
    r = requests.get(
        f"https://{_host(region)}/v1/projects/{project}/locations/{region}/endpoints",
        headers=_headers(), timeout=120)
    if r.status_code >= 400:
        print(f"endpoints.list failed: {r.status_code} {r.text[:300]}")
        return 1
    eps = [e for e in r.json().get("endpoints", []) if DISPLAY_NAME in (e.get("displayName") or "")]
    if not eps:
        print(f"No endpoint named {DISPLAY_NAME} in {project}/{region}. Nothing is running.")
        return 0
    for e in eps:
        eid = e["name"].split("/")[-1]
        deployed = e.get("deployedModels", [])
        print(f"endpoint {eid}  displayName={e.get('displayName')}  deployedModels={len(deployed)}")
        for d in deployed:
            spec = (d.get("dedicatedResources") or {}).get("machineSpec", {})
            print(f"    - {d.get('displayName')} id={d.get('id')} "
                  f"{spec.get('machineType')} {spec.get('acceleratorType')}x{spec.get('acceleratorCount')}")
        if deployed:
            print("  THIS IS BILLING NOW. Tear down: --delete --yes")
        print(f"  .env: VERTEX_QWEN4B_BASE_URL=https://{_host(region)}/v1/{e['name']}")
    return 0


def _find_by_display_name(base: str, collection: str) -> list:
    r = requests.get(f"{base}/{collection}", headers=_headers(), timeout=120)
    if r.status_code >= 400:
        return []
    return [x for x in r.json().get(collection, []) if DISPLAY_NAME in (x.get("displayName") or "")]


def cmd_redeploy(project: str, region: str, shape: str, apply: bool) -> int:
    """Deploy the already-uploaded Model onto the already-created Endpoint.

    Recovery path for a run whose upload and endpoint creation succeeded but
    whose deployModel never landed (or was later undeployed). Skips ~2 minutes
    of re-upload and, more importantly, avoids leaving a second orphaned
    endpoint/model pair behind every time a deploy is retried.
    """
    base = f"https://{_host(region)}/v1/projects/{project}/locations/{region}"
    endpoints = _find_by_display_name(base, "endpoints")
    models = _find_by_display_name(base, "models")
    if not endpoints or not models:
        print(f"Need an existing endpoint AND model named {DISPLAY_NAME}; "
              f"found endpoints={len(endpoints)} models={len(models)}. Use --create.")
        return 1

    endpoint, model = endpoints[0], models[0]
    endpoint_id = endpoint["name"].split("/")[-1]
    already = endpoint.get("deployedModels", [])
    if already:
        print(f"Endpoint {endpoint_id} already has {len(already)} deployed model(s). "
              f"Nothing to do, it is live (and billing).")
        return 0

    machine = MACHINE_SHAPES[shape]
    print(f"redeploy {model['name']}")
    print(f"      -> {endpoint['name']}")
    print(f"      on {machine['machineType']} {machine['acceleratorType']}x{machine['acceleratorCount']}")
    print(COST_WARNING)
    if not apply:
        print("Plan only. Nothing was deployed. Re-run with --redeploy --yes.")
        return 0

    body = {
        "deployedModel": {
            "model": model["name"],
            "displayName": DISPLAY_NAME,
            "dedicatedResources": {
                "machineSpec": machine, "minReplicaCount": 1, "maxReplicaCount": 1,
            },
        },
        "trafficSplit": {"0": 100},
    }
    r = requests.post(f"https://{_host(region)}/v1/{endpoint['name']}:deployModel",
                      headers=_headers(), json=body, timeout=600)
    if r.status_code >= 400:
        print(f"deployModel failed: {r.status_code} {r.text[:600]}")
        return 1
    print("deployModel accepted; this takes ~20 min for a 4B checkpoint.")
    try:
        _wait_lro(region, r.json()["name"], "deployModel")
    except Exception as exc:
        print(f"\n!! LOST TRACK OF THE DEPLOY: {type(exc).__name__}: {str(exc)[:300]}")
        print("!! A GPU node is probably running and BILLING. Recover with:")
        print("!!   python scripts/gcp/deploy_qwen3_4b_vertex.py --status")
        print("!!   python scripts/gcp/deploy_qwen3_4b_vertex.py --delete --yes")
        return 1

    print(f"\nLive. Add to .env:\n    VERTEX_QWEN4B_BASE_URL=https://{_host(region)}/v1/{endpoint['name']}")
    return 0


def cmd_create(project: str, region: str, shape: str, max_len: int, apply: bool) -> int:
    machine = MACHINE_SHAPES[shape]
    args = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host=0.0.0.0", f"--port={CONTAINER_PORT}",
        f"--model={HF_MODEL}",
        f"--served-model-name={SERVED_NAME}",
        f"--max-model-len={max_len}",
        "--gpu-memory-utilization=0.90",
        "--trust-remote-code",
    ]
    if shape == "t4":
        # T4 is pre-Ampere: no bf16 support in vLLM, must force fp16.
        args.append("--dtype=float16")

    model_body = {
        "model": {
            "displayName": DISPLAY_NAME,
            "containerSpec": {
                "imageUri": VLLM_IMAGE,
                "args": args,
                "ports": [{"containerPort": CONTAINER_PORT}],
                # rawPredict forwards the body verbatim to this route, which is
                # why config.py can send a plain OpenAI chat payload.
                "predictRoute": "/v1/chat/completions",
                "healthRoute": "/health",
            },
        }
    }

    print(f"1. models.upload  image={VLLM_IMAGE}")
    print(f"   args: {' '.join(args)}")
    print(f"2. endpoints.create  displayName={DISPLAY_NAME}")
    print(f"3. deployModel  {machine['machineType']} {machine['acceleratorType']}x{machine['acceleratorCount']}")
    print(COST_WARNING)
    if not apply:
        print("Plan only. Nothing was deployed. Re-run with --create --yes.")
        return 0

    base = f"https://{_host(region)}/v1/projects/{project}/locations/{region}"

    print("uploading model...")
    r = requests.post(f"{base}/models:upload", headers=_headers(), json=model_body, timeout=600)
    if r.status_code >= 400:
        print(f"models.upload failed: {r.status_code} {r.text[:600]}")
        return 1
    resp = _wait_lro(region, r.json()["name"], "models.upload")
    model_name = resp.get("model") or resp.get("name")
    print(f"  model: {model_name}")

    print("creating endpoint...")
    r = requests.post(f"{base}/endpoints", headers=_headers(),
                      json={"displayName": DISPLAY_NAME}, timeout=600)
    if r.status_code >= 400:
        print(f"endpoints.create failed: {r.status_code} {r.text[:600]}")
        return 1
    endpoint_name = _wait_lro(region, r.json()["name"], "endpoints.create").get("name")
    endpoint_id = str(endpoint_name).split("/")[-1]
    print(f"  endpoint: {endpoint_name}")

    print("deploying model to endpoint (this is the slow part)...")
    deploy_body = {
        "deployedModel": {
            "model": model_name,
            "displayName": DISPLAY_NAME,
            "dedicatedResources": {
                "machineSpec": machine,
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            },
        },
        "trafficSplit": {"0": 100},
    }
    r = requests.post(f"https://{_host(region)}/v1/{endpoint_name}:deployModel",
                      headers=_headers(), json=deploy_body, timeout=600)
    if r.status_code >= 400:
        print(f"deployModel failed: {r.status_code} {r.text[:600]}")
        if "quota" in r.text.lower():
            print("\nQuota denial: request `custom_model_serving_nvidia_l4_gpus` "
                  "(or the T4 equivalent) in this region under IAM & Admin -> Quotas.")
        return 1

    # Once deployModel is accepted the GPU node is being provisioned server-side
    # and starts billing regardless of what happens to this process. If polling
    # dies here (expired token, dropped network, Ctrl-C) the resources are
    # orphaned and silently accruing cost, so surface the teardown path loudly
    # instead of letting a stack trace be the only record.
    try:
        _wait_lro(region, r.json()["name"], "deployModel")
    except Exception as exc:
        print(f"\n!! LOST TRACK OF THE DEPLOY: {type(exc).__name__}: {str(exc)[:300]}")
        print("!! The deploy was ACCEPTED before this failure, so a GPU node is")
        print("!! probably running and BILLING right now. It will not stop on its own.")
        print(f"!!   endpoint : {endpoint_name}")
        print(f"!!   model    : {model_name}")
        print("!! Recover with:")
        print("!!   gcloud auth login && gcloud auth application-default login")
        print("!!   python scripts/gcp/deploy_qwen3_4b_vertex.py --status")
        print("!!   python scripts/gcp/deploy_qwen3_4b_vertex.py --delete --yes")
        print(f"!! Or in the console: https://console.cloud.google.com/vertex-ai/"
              f"online-prediction/locations/{region}/endpoints/{endpoint_id}"
              f"?project={project}")
        return 1

    print(f"\nLive. Add to .env:\n    VERTEX_QWEN4B_ENDPOINT_ID={endpoint_id}")
    print("Tear down when done:  python scripts/gcp/deploy_qwen3_4b_vertex.py --delete --yes")
    return 0


def cmd_delete(project: str, region: str, apply: bool) -> int:
    """Undeploy, delete the endpoint, then delete the uploaded Model."""
    base = f"https://{_host(region)}/v1/projects/{project}/locations/{region}"
    r = requests.get(f"{base}/endpoints", headers=_headers(), timeout=120)
    eps = [e for e in r.json().get("endpoints", []) if DISPLAY_NAME in (e.get("displayName") or "")] \
        if r.status_code < 400 else []

    for e in eps:
        name = e["name"]
        for d in e.get("deployedModels", []):
            print(f"undeploy {d.get('id')} from {name}")
            if apply:
                rr = requests.post(f"https://{_host(region)}/v1/{name}:undeployModel",
                                   headers=_headers(), json={"deployedModelId": d["id"]}, timeout=600)
                print(f"  -> {rr.status_code}")
                if rr.status_code < 400:
                    _wait_lro(region, rr.json()["name"], "undeployModel")
        print(f"delete endpoint {name}")
        if apply:
            rr = requests.delete(f"https://{_host(region)}/v1/{name}", headers=_headers(), timeout=600)
            print(f"  -> {rr.status_code}")

    rm = requests.get(f"{base}/models", headers=_headers(), timeout=120)
    models = [m for m in rm.json().get("models", []) if DISPLAY_NAME in (m.get("displayName") or "")] \
        if rm.status_code < 400 else []
    for m in models:
        print(f"delete model {m['name']}")
        if apply:
            rr = requests.delete(f"https://{_host(region)}/v1/{m['name']}", headers=_headers(), timeout=600)
            print(f"  -> {rr.status_code}")

    if not eps and not models:
        print("Nothing to delete.")
    if not apply:
        print("\nPlan only. Nothing was deleted. Re-run with --delete --yes.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_mutually_exclusive_group()
    a.add_argument("--create", action="store_true")
    a.add_argument("--status", action="store_true")
    a.add_argument("--delete", action="store_true")
    a.add_argument("--redeploy", action="store_true",
                   help="Deploy the existing Model onto the existing Endpoint (recovery path).")
    p.add_argument("--shape", choices=list(MACHINE_SHAPES), default="l4")
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--yes", action="store_true", help="Actually execute. Without it, plan only.")
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    args = p.parse_args()

    if args.env_file.exists():
        load_dotenv_file(args.env_file)
    project, region = _project(), _region()
    print(f"project={project} region={region}\n")

    if args.status:
        return cmd_status(project, region)
    if args.delete:
        return cmd_delete(project, region, args.yes)
    if args.redeploy:
        return cmd_redeploy(project, region, args.shape, args.yes)
    if args.create:
        return cmd_create(project, region, args.shape, args.max_model_len, args.yes)
    p.print_help()
    print(COST_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
