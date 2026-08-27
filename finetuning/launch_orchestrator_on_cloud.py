"""Run the full FactoryBench pipeline (baseline eval -> finetune -> finetuned
eval, per model) as a SageMaker Processing job, so the orchestration itself
runs in the cloud and survives your laptop disconnecting.

Architecture:
    you ──submit──▶ orchestrator processing job (ml.t3.medium, $0.04/h, ~hours)
                         │
                         ├─submit──▶ baseline eval (ml.g5.2xlarge, spot)  ─┐
                         ├─submit──▶ finetune     (ml.p5.48xlarge, spot)   │  per model
                         └─submit──▶ finetuned eval (ml.g5.2xlarge, spot) ─┘

The orchestrator just shells run_pipeline.py with whatever CLI args you pass
here. It uses the same IAM role as the training/eval jobs (the role inherits
sagemaker:CreateTrainingJob), so no extra setup.

Usage:
    # Full 4-model sweep, 5000 train samples
    python finetuning/launch_orchestrator_on_cloud.py --max_train_samples 5000

    # Just one model
    python finetuning/launch_orchestrator_on_cloud.py --model bearing_r32_qwen3_4b --max_train_samples 5000

    # Dry run, print plan, don't submit
    python finetuning/launch_orchestrator_on_cloud.py --max_train_samples 5000 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch.processing import PyTorchProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_dora_sweep import BUCKET, REGION, ROLE  # noqa: E402

FINETUNING_DIR = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    # Pipeline knobs, passed through to run_pipeline.py verbatim
    p.add_argument("--model", nargs="*", default=None)
    p.add_argument("--skip", nargs="*", default=None,
                   choices=["baseline", "train", "finetuned"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Eval-side cap.")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--no-spot", action="store_true",
                   help="Pass --no-spot to the inner pipeline.")
    p.add_argument("--quota-retry-seconds", type=int, default=None)
    p.add_argument("--quota-max-wait", type=int, default=None)

    # Orchestrator container knobs
    p.add_argument("--instance", default="ml.t3.medium",
                   help="Instance for the orchestrator itself (CPU-only is fine).")
    p.add_argument("--max-runtime-hours", type=int, default=72,
                   help="Hard cap on orchestrator runtime. Default 72h.")
    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    # Build the argv to forward to run_pipeline.py inside the container.
    fwd: list[str] = []
    if args.model:
        fwd += ["--model", *args.model]
    if args.skip:
        fwd += ["--skip", *args.skip]
    if args.smoke:
        fwd.append("--smoke")
    if args.epochs is not None:
        fwd += ["--epochs", str(args.epochs)]
    if args.max_samples is not None:
        fwd += ["--max_samples", str(args.max_samples)]
    if args.max_train_samples is not None:
        fwd += ["--max_train_samples", str(args.max_train_samples)]
    if args.max_val_samples is not None:
        fwd += ["--max_val_samples", str(args.max_val_samples)]
    if args.no_spot:
        fwd.append("--no-spot")
    if args.quota_retry_seconds is not None:
        fwd += ["--quota-retry-seconds", str(args.quota_retry_seconds)]
    if args.quota_max_wait is not None:
        fwd += ["--quota-max-wait", str(args.quota_max_wait)]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"fbench-orchestrator-{stamp}"

    print("=" * 72)
    print(f"Submitting orchestrator processing job")
    print(f"  job_name:       {job_name}")
    print(f"  instance:       {args.instance}")
    print(f"  max_runtime:    {args.max_runtime_hours}h")
    print(f"  source_dir:     {FINETUNING_DIR}")
    print(f"  forwarded args: {' '.join(fwd) if fwd else '<none>'}")
    print("=" * 72)

    if args.dry_run:
        print("[dry-run] not submitted.")
        return

    sess = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION),
        default_bucket=BUCKET)

    processor = PyTorchProcessor(
        framework_version="2.5.1",
        py_version="py311",
        role=ROLE,
        instance_count=1,
        instance_type=args.instance,
        base_job_name="fbench-orchestrator",
        sagemaker_session=sess,
        max_runtime_in_seconds=args.max_runtime_hours * 3600,
        volume_size_in_gb=30,
        tags=[
            {"Key": "Project", "Value": "factorybench"},
            {"Key": "Role",    "Value": "orchestrator"},
        ])

    # PyTorchProcessor uploads `source_dir` as a tarball; inside the container
    # it ends up at /opt/ml/processing/input/code/. `code=` is the entry script
    # relative to source_dir. `requirements.txt` at the top of source_dir is
    # auto-installed before code runs (we added one with sagemaker + boto3).
    processor.run(
        code="run_pipeline.py",
        source_dir=str(FINETUNING_DIR),
        arguments=fwd,
        job_name=job_name,
        wait=False,
        logs=False)

    print(f"\nOrchestrator launched: {job_name}")
    print(f"Tail logs:")
    print(f"  aws logs tail /aws/sagemaker/ProcessingJobs "
          f"--log-stream-name-prefix {job_name} --region {REGION} --follow")
    print(f"Status:")
    print(f"  aws sagemaker describe-processing-job --processing-job-name "
          f"{job_name} --region {REGION} --query "
          f"\"{{S:ProcessingJobStatus}}\" --output table")
    print(f"Stop:")
    print(f"  aws sagemaker stop-processing-job --processing-job-name "
          f"{job_name} --region {REGION}")


if __name__ == "__main__":
    main()
