"""Deploy Qwen/Qwen3-4B-Instruct-2507 to SageMaker Async Inference via JumpStart.

The model is in the JumpStart catalog as of April 2026 (verified in Studio UI:
v1.3.0, huggingface hub, task=Reasoning), so we deploy through ``JumpStartModel``
rather than building a HuggingFace TGI image from scratch. Two reasons:

  * The JumpStart container ships a pre-tuned vLLM build (chunked prefill,
    Hermes tool parser, batched rolling decode) that we'd otherwise re-do.
  * Bypasses the Studio UI's ``Register Model`` flow, which fails because the
    JumpStart spec ships ~19 env vars and Model Package Groups cap at 16.
    The direct ``deploy`` path has no such limit.

The endpoint runs on a single ml.g5.xlarge (24 GB VRAM, plenty for a 4B in fp16)
in **Async Inference** mode so the existing ``run_aws_eval.run_sagemaker_async``
dispatcher works unchanged. We then register an Application Auto Scaling target
with ``MinCapacity=0`` so the endpoint scales to zero when idle (cost = 0)
and wakes on demand from a CloudWatch alarm on ``HasBacklogWithoutCapacity``.

Usage:
    python scripts/deploy_qwen3_4b.py --list      # discover the JumpStart id
    python scripts/deploy_qwen3_4b.py             # deploy using $SAGEMAKER_ROLE_ARN
    python scripts/deploy_qwen3_4b.py --studio-user-profile anon-studio-user
                                                  # deploy using the Studio user
                                                  # profile's execution role
                                                  # (skips the custom IAM dance)
    python scripts/deploy_qwen3_4b.py --instance-type ml.g5.2xlarge

Required env (loaded from --env-file, default .env):
    FB_S3_BUCKET               S3 bucket for async I/O (must be in --region)
    SAGEMAKER_ROLE_ARN         IAM role per src/evaluation/aws-setup.md §2.2.
                               NOT required when --studio-user-profile is set, the Studio domain's execution role is used instead.
                               That role ships with AmazonSageMakerFullAccess
                               (incl. read on jumpstart-cache-prod-*), which is
                               what a custom factorybench-sagemaker role often
                               lacks until the JumpStart S3 statement is added.

After a successful deploy, append the printed pair to .env:
    QWEN3_4B_SAGEMAKER_ENDPOINT=<endpoint name>
    QWEN3_4B_SAGEMAKER_REGION=<region>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL_ID = "huggingface-reasoning-qwen3-4b-instruct-2507"
DEFAULT_INSTANCE = "ml.g5.xlarge"
DEFAULT_ENDPOINT = "qwen3-4b-test-endpoint"
DEFAULT_REGION = "eu-central-1"
DEFAULT_STUDIO_DOMAIN = "Students-Research-Lab"


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        sys.exit(f"ERROR: {name} env var is required (set it in .env).")
    return val


def resolve_studio_role(region: str, domain_name: str, user_profile: str) -> str:
    """Resolve the SageMaker Studio execution role ARN for ``user_profile``.

    Studio domains and user profiles each carry an execution role; the user
    profile's role takes precedence when set, otherwise the domain default
    applies. These roles ship with ``AmazonSageMakerFullAccess`` by default,
    which already grants ``s3:GetObject`` on ``jumpstart-cache-prod-*``, the
    exact permission a hand-rolled ``factorybench-sagemaker`` role typically
    lacks. Using this role lets the JumpStart deploy work without waiting on
    an IAM admin to extend the custom role.
    """
    import boto3

    sm = boto3.client("sagemaker", region_name=region)

    matching = [d for d in sm.list_domains().get("Domains", [])
                if d["DomainName"] == domain_name]
    if not matching:
        sys.exit(
            f"ERROR: no SageMaker Studio domain named {domain_name!r} in {region}. "
            f"Confirm the domain name in the SageMaker console (Frankfurt → "
            f"Amazon SageMaker AI → Domains) and pass it via --studio-domain."
        )
    domain_id = matching[0]["DomainId"]

    up = sm.describe_user_profile(DomainId=domain_id, UserProfileName=user_profile)
    role = up.get("UserSettings", {}).get("ExecutionRole")
    if role:
        return role

    domain = sm.describe_domain(DomainId=domain_id)
    role = domain.get("DefaultUserSettings", {}).get("ExecutionRole")
    if not role:
        sys.exit(
            f"ERROR: domain {domain_name!r} has no default execution role and "
            f"user profile {user_profile!r} doesn't override it. Set "
            f"SAGEMAKER_ROLE_ARN explicitly instead."
        )
    return role


def list_qwen_models(region: str) -> None:
    """Print JumpStart model ids matching 'qwen3' so the caller can verify
    --model-id before running an actual deploy."""
    from sagemaker.jumpstart.notebook_utils import list_jumpstart_models

    print(f"JumpStart catalog ({region}), qwen3 matches:")
    matches = [m for m in list_jumpstart_models(region=region) if "qwen3" in m.lower()]
    for m in matches:
        print(f"  {m}")
    if not matches:
        print("  (none, open JumpStart in Studio and copy the id from the model card)")


def deploy(
    *,
    model_id: str,
    instance_type: str,
    endpoint_name: str,
    region: str,
    s3_bucket: str,
    role_arn: str,
    autoscale: bool,
    max_instances: int) -> str:
    import boto3
    from sagemaker import Session
    from sagemaker.async_inference import AsyncInferenceConfig
    from sagemaker.jumpstart.model import JumpStartModel

    boto_sess = boto3.Session(region_name=region)
    sm_session = Session(boto_session=boto_sess)

    sm = boto_sess.client("sagemaker")
    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        sys.exit(
            f"ERROR: endpoint {endpoint_name!r} already exists in {region}. "
            f"Delete it first or pass --endpoint-name to use a different name:\n"
            f"  aws sagemaker delete-endpoint --endpoint-name {endpoint_name} --region {region}"
        )
    except sm.exceptions.ClientError as exc:
        if "Could not find endpoint" not in str(exc):
            raise

    print(f"Deploying {model_id}")
    print(f"  instance       : {instance_type}")
    print(f"  endpoint name  : {endpoint_name}")
    print(f"  region         : {region}")
    print(f"  async output   : s3://{s3_bucket}/factorybench/qwen3-4b/async-out/")
    print()

    # The JumpStart spec for Qwen3-4B targets ml.g5.12xlarge (4×A10G, 96 GB VRAM):
    # TP=4, MAX_MODEL_LEN=157286, MAX_NUM_BATCHED_TOKENS=314572, ROLLING_BATCH=128.
    # On ml.g5.xlarge (1×A10G, 24 GB) every one of those overcommits the GPU.
    # We rescale all four for single-GPU: TP=1, 32k context, 8k batched tokens,
    # rolling batch of 16, leaves headroom for the 8 GB of fp16 weights plus
    # KV cache. Bump GPU_MEMORY_UTILIZATION slightly so vLLM uses the breathing
    # room it now has.
    model = JumpStartModel(
        model_id=model_id,
        role=role_arn,
        sagemaker_session=sm_session,
        env={
            "OPTION_TENSOR_PARALLEL_DEGREE": "1",
            "TENSOR_PARALLEL_DEGREE": "1",
            "OPTION_MAX_MODEL_LEN": "32768",
            "OPTION_MAX_NUM_BATCHED_TOKENS": "8192",
            "OPTION_MAX_ROLLING_BATCH_SIZE": "16",
            "OPTION_GPU_MEMORY_UTILIZATION": "0.90",
        })
    async_cfg = AsyncInferenceConfig(
        output_path=f"s3://{s3_bucket}/factorybench/qwen3-4b/async-out/",
        max_concurrent_invocations_per_instance=4)
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        async_inference_config=async_cfg,
        accept_eula=True)
    print(f"Endpoint live: {predictor.endpoint_name}")

    if autoscale:
        configure_scale_to_zero(region, endpoint_name, max_instances=max_instances)
    return predictor.endpoint_name


def configure_scale_to_zero(region: str, endpoint_name: str, *, max_instances: int) -> None:
    """Register MinCapacity=0 + step-scaling on HasBacklogWithoutCapacity.

    Async endpoints can scale to zero, but waking from zero requires an explicit
    CloudWatch alarm wired to a step-scaling policy, target tracking alone
    won't trigger when there are no instances reporting metrics. We therefore
    set up two policies:
      * Step-scaling (wake-from-zero): +1 instance when the alarm fires.
      * Target tracking (steady state): keep ~5 queued requests per instance
        under continuous load.
    """
    import boto3

    asg = boto3.client("application-autoscaling", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    asg.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=max_instances)

    step_resp = asg.put_scaling_policy(
        PolicyName=f"{endpoint_name}-wake-from-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "Cooldown": 60,
            "MetricAggregationType": "Maximum",
            "StepAdjustments": [{"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1}],
        })

    cw.put_metric_alarm(
        AlarmName=f"{endpoint_name}-has-backlog-without-capacity",
        MetricName="HasBacklogWithoutCapacity",
        Namespace="AWS/SageMaker",
        Statistic="Maximum",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        EvaluationPeriods=2,
        DatapointsToAlarm=2,
        Period=60,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="missing",
        AlarmActions=[step_resp["PolicyARN"]])

    asg.put_scaling_policy(
        PolicyName=f"{endpoint_name}-target-backlog",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 5.0,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 600,
            "ScaleOutCooldown": 60,
        })

    print(f"Auto-scaling configured: min=0, max={max_instances} (scale-to-zero enabled)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help=f"JumpStart model id (default: {DEFAULT_MODEL_ID}). "
                             f"Run with --list to verify.")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE,
                        help=f"SageMaker instance type (default: {DEFAULT_INSTANCE})")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT,
                        help=f"Endpoint name (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--region", default=os.getenv("QWEN3_4B_SAGEMAKER_REGION", DEFAULT_REGION),
                        help=f"AWS region (default: $QWEN3_4B_SAGEMAKER_REGION or {DEFAULT_REGION})")
    parser.add_argument("--max-instances", type=int, default=2,
                        help="Auto-scaling max instance count (default: 2)")
    parser.add_argument("--no-autoscale", action="store_true",
                        help="Skip the scale-to-zero auto-scaling setup (endpoint stays at 1 instance)")
    parser.add_argument("--list", action="store_true",
                        help="List Qwen3 models in JumpStart for the given --region and exit")
    parser.add_argument("--studio-user-profile", default=None,
                        help="If set, resolve the SageMaker Studio user profile's execution "
                             "role and use it instead of $SAGEMAKER_ROLE_ARN. Studio roles "
                             "already grant the JumpStart S3 reads that a custom role often "
                             "lacks. Example: --studio-user-profile anon-studio-user.")
    parser.add_argument("--studio-domain", default=DEFAULT_STUDIO_DOMAIN,
                        help=f"SageMaker Studio domain that hosts --studio-user-profile "
                             f"(default: {DEFAULT_STUDIO_DOMAIN}).")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    if args.env_file.exists():
        load_dotenv(args.env_file)

    if args.list:
        list_qwen_models(args.region)
        return

    if args.studio_user_profile:
        role_arn = resolve_studio_role(args.region, args.studio_domain, args.studio_user_profile)
        print(f"Using Studio execution role from {args.studio_domain}/{args.studio_user_profile}:")
        print(f"  {role_arn}")
    else:
        role_arn = _require_env("SAGEMAKER_ROLE_ARN")
    s3_bucket = _require_env("FB_S3_BUCKET")

    endpoint_name = deploy(
        model_id=args.model_id,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint_name,
        region=args.region,
        s3_bucket=s3_bucket,
        role_arn=role_arn,
        autoscale=not args.no_autoscale,
        max_instances=args.max_instances)

    print()
    print("Append to .env:")
    print(f"  QWEN3_4B_SAGEMAKER_ENDPOINT={endpoint_name}")
    print(f"  QWEN3_4B_SAGEMAKER_REGION={args.region}")
    print()
    print("Smoke test (after .env updated):")
    print("  python -m src.evaluation.run_aws_eval \\")
    print("    --input output/prompts/level1 \\")
    print("    --output-dir output/replies/level1/qwen3-4b \\")
    print("    --questions output/questions/level1 \\")
    print("    --model qwen3-4b --limit 2")


if __name__ == "__main__":
    main()
