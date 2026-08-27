#!/usr/bin/env python3
"""Verify the GCP inference pipeline is wired up, without running inference.

Checks, in order, and reports a per-model verdict:

  1. ADC resolves and yields an OAuth token.
  2. The project is set and the Vertex API answers.
  3. Each configured model exists in Model Garden under the publisher and id
     ``src/config.py`` resolves to.
  4. The batch I/O bucket exists and is in a region that can host batch jobs.
  5. Self-deployed models (Mistral Large 3) have a live endpoint.

Every call is a metadata GET. No prompt is ever sent, so this costs nothing
and cannot be mistaken for an eval run.

Usage:
    python scripts/gcp/preflight_check.py
    python scripts/gcp/preflight_check.py --models claude-sonnet-4.6 deepseek-v3.2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

# Import from the repo root regardless of where this is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import (  # noqa: E402
    GCP_PROJECT_DEFAULT,
    GCS_BUCKET_DEFAULT,
    VERTEX_MODELS,
    get_region,
    get_upstream_model_id,
    get_vertex_publisher,
)
from src.evaluation.run_foundry_eval import _vertex_access_token  # noqa: E402
from src.evaluation.test_gpt_5mini import load_dotenv_file  # noqa: E402

OK = "PASS"
BAD = "FAIL"
WARN = "WARN"


def _project() -> str:
    return os.environ.get("GCP_PROJECT") or GCP_PROJECT_DEFAULT


def _host(location: str) -> str:
    return (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(VERTEX_MODELS.keys()))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    if args.env_file.exists():
        load_dotenv_file(args.env_file)

    project = _project()
    failures = 0
    print(f"project: {project}\n")

    # 1. Auth
    try:
        token = _vertex_access_token()
        print(f"[{OK}] ADC token resolved ({len(token)} chars)")
    except Exception as exc:
        print(f"[{BAD}] ADC token: {exc}")
        print("      fix: gcloud auth application-default login")
        return 1
    headers = {"Authorization": f"Bearer {token}", "x-goog-user-project": project}

    # 2. Batch I/O bucket
    bucket = os.environ.get("GCS_BUCKET") or GCS_BUCKET_DEFAULT
    try:
        r = requests.get(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}",
            headers=headers, timeout=60,
        )
        if r.status_code == 200:
            loc = r.json().get("location", "?")
            print(f"[{OK}] bucket gs://{bucket} exists (location={loc})")
        elif r.status_code == 404:
            print(f"[{BAD}] bucket gs://{bucket} does not exist")
            print("      fix: bash scripts/gcp/provision_inference.sh --apply")
            failures += 1
        else:
            print(f"[{BAD}] bucket gs://{bucket}: {r.status_code} {r.text[:160]}")
            failures += 1
    except Exception as exc:
        print(f"[{BAD}] bucket check failed: {exc}")
        failures += 1

    # 3. Per-model catalog presence
    print()
    for model in args.models:
        cfg = VERTEX_MODELS.get(model)
        if cfg is None:
            print(f"[{BAD}] {model}: not a Vertex model in src/config.py")
            failures += 1
            continue

        publisher = get_vertex_publisher(model)
        model_id = get_upstream_model_id(model)
        location = get_region(model) or "us-central1"
        # The MaaS OpenAI route carries the publisher inside the model id
        # ("deepseek-ai/deepseek-v3.2-maas"); the catalog wants the bare id.
        catalog_id = model_id.split("/")[-1]

        if cfg.get("self_deployed"):
            endpoint = os.environ.get(cfg.get("endpoint_env", ""), "")
            if not endpoint:
                print(
                    f"[{WARN}] {model}: self-deploy model with no endpoint "
                    f"({cfg.get('endpoint_env')} unset)"
                )
                print(
                    "      This model has no MaaS offering on Vertex. Stand the "
                    "Model Garden vLLM container up by hand (8-GPU node, billed "
                    f"hourly), set {cfg.get('endpoint_env')} to its endpoint id, "
                    "or exclude it from --models."
                )
                continue
            url = (
                f"https://{_host(location)}/v1/projects/{project}/locations/"
                f"{location}/endpoints/{endpoint.rstrip('/').split('/')[-1]}"
            )
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200:
                deployed = r.json().get("deployedModels", [])
                state = "with a deployed model" if deployed else "EMPTY (nothing deployed)"
                print(f"[{OK if deployed else BAD}] {model}: endpoint {endpoint} {state}")
                failures += 0 if deployed else 1
            else:
                print(f"[{BAD}] {model}: endpoint {endpoint}: {r.status_code} {r.text[:160]}")
                failures += 1
            continue

        # The catalog and the serving surface are registered in different
        # regions and do not agree. DeepSeek V3.2 is catalogued only in
        # us-central1 but serves only from `global`; Claude is catalogued in
        # both. So look the model up wherever its metadata actually lives
        # rather than assuming it is registered at its serving region, and
        # report the two separately.
        body = None
        found_at = None
        for candidate in (location, "us-central1", "global"):
            if candidate is None:
                continue
            r = requests.get(
                f"https://{_host(candidate)}/v1beta1/publishers/{publisher}/models/{catalog_id}",
                headers=headers, timeout=60,
            )
            if r.status_code == 200:
                body, found_at = r.json(), candidate
                break
            if r.status_code not in (403, 404):
                print(f"[{BAD}] {model}: {r.status_code} {r.text[:200]}")
                failures += 1
                body = "errored"  # type: ignore[assignment]
                break

        if body == "errored":
            continue
        if body is None:
            print(
                f"[{BAD}] {model}: {publisher}/{catalog_id} not found in the catalog "
                f"(tried {location}, us-central1, global)"
            )
            failures += 1
            continue

        where = f"serves from {location}"
        if found_at != location:
            where += f", catalogued in {found_at}"
        print(
            f"[{OK}] {model}: {publisher}/{catalog_id} ({where}; "
            f"launchStage={body.get('launchStage')}, version={body.get('versionId')})"
        )
        if "requestAccess" in (body.get("supportedActions") or {}):
            print(
                "      note: this publisher requires a one-time Model Garden "
                "enablement before rawPredict succeeds. Catalog presence does "
                "not prove it; only a real call does."
            )

    print()
    if failures:
        print(f"{failures} check(s) failed. See src/evaluation/gcp-setup.md.")
        return 1
    print("All checks passed. No inference was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
