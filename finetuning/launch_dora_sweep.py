"""Launch FactoryBench DoRA-on-DoRA fine-tuning for all 4 pretrained Shrike
checkpoints in parallel on SageMaker.

For each base checkpoint, we:
  1. Stage the .pt and its matching TS tokenizer .pt as SageMaker input channels
  2. Run train_factorybench.py which loads the wrapper, MERGES the original
     DoRA r=32 adapter into the base weights, then attaches a FRESH DoRA on
     top for FactoryBench finetuning.

Submits 4 jobs (one per checkpoint) with --wait False, so they all queue and
run in parallel on separate instances.

Usage:
    python finetuning/launch_dora_sweep.py            # all 4 on spot
    python finetuning/launch_dora_sweep.py --only bearing_r32 phase0_totem
    python finetuning/launch_dora_sweep.py --no-spot  # on-demand instead
    python finetuning/launch_dora_sweep.py --dry-run  # print plan, don't submit

Required environment (see finetuning/README.md):
    FB_SAGEMAKER_BUCKET     S3 bucket holding the checkpoints, tokenizers and
                            FactoryBench JSONLs, and receiving job output
    FB_SAGEMAKER_ROLE_ARN   SageMaker execution role ARN
    AWS_REGION              region of the bucket and the training jobs
    FB_SAGEMAKER_PREFIX     key prefix for the Shrike artefacts (default: shrike)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# -- Repo / S3 layout ----------------------------------------------------
# Source dir is always SIBLING of this file. Computing it as "parent of
# parent / finetuning / _factorybench_src" used to break inside the
# orchestrator processing container, where __file__ no longer lives under
# a "finetuning" directory.
SRC_DIR = Path(__file__).resolve().parent / "_factorybench_src"


def _require_env(name: str, what: str, example: str) -> str:
    """Read a required setting from the environment, or explain what is missing.

    The account, bucket and execution role that ran the published sweep were
    hardcoded here. They are private infrastructure, so they are now supplied
    by the caller. Every launcher in this directory imports these three names,
    so failing loudly at import is better than submitting a job into whatever
    account boto3 happens to resolve.
    """
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"  {what}\n"
            f'  export {name}="{example}"\n'
            "  See finetuning/README.md for the full environment."
        )
    return value


BUCKET = _require_env(
    "FB_SAGEMAKER_BUCKET",
    "S3 bucket holding the Shrike checkpoints, TS tokenizers and FactoryBench "
    "training JSONLs, and receiving training/eval output.",
    "my-sagemaker-bucket")
REGION = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "").strip()
if not REGION:
    raise SystemExit(
        "AWS_REGION is not set.\n"
        "  Region of the bucket above and of the SageMaker training jobs.\n"
        '  export AWS_REGION="us-east-2"'
    )
ROLE = _require_env(
    "FB_SAGEMAKER_ROLE_ARN",
    "SageMaker execution role ARN, needs read/write on the bucket above.",
    "arn:aws:iam::<acct>:role/service-role/AmazonSageMaker-ExecutionRole-<id>")
# Key prefix under which the Shrike checkpoints and tokenizers are staged.
S3_PREFIX = (os.environ.get("FB_SAGEMAKER_PREFIX") or "shrike").strip()

# Pre-staged base LLM weights (already on S3, used as the "llm" channel)
S3_LLM = {
    "Qwen/Qwen3-4B":   f"s3://{BUCKET}/{S3_PREFIX}/llm/Qwen3-4B/",
    "Qwen/Qwen3-1.7B": f"s3://{BUCKET}/{S3_PREFIX}/llm/Qwen3-1.7B/",
}

# Pre-staged FactoryBench training JSONLs
S3_FACTORYBENCH_DATA = f"s3://{BUCKET}/factorybench/data/"

# -- Per-checkpoint configs ---------------------------------------------
# Keys match what the eval launcher uses for "bearing_v3" / "bearing_v4",
# but renamed to reflect the local filenames the user gave us.
CHECKPOINTS = {
    "bearing_mixed_r32_qwen3_4b": {
        "base_ckpt": f"s3://{BUCKET}/{S3_PREFIX}/checkpoints/"
                     f"hyperion_v4_multi5bearing/best_model.pt",
        "checkpoint_type": "bearing",
        "tokenizer_type": "totem",
        # Bearing models were trained against the original 256-code TOTEM
        # (totem_clean.pt in the eval launcher's S3 map).
        "ts_tokenizer_s3": f"s3://{BUCKET}/{S3_PREFIX}/tokenizer/totem_clean.pt",
        "ts_tokenizer_channel": "totem_ckpt",
        "llm_id": "Qwen/Qwen3-4B",
        "instance": "ml.p5.48xlarge",
        "job_name": "fbench-dora-bearing-mixed-r32",
    },
    "bearing_r32_qwen3_4b": {
        "base_ckpt": f"s3://{BUCKET}/{S3_PREFIX}/checkpoints/hyperion_v3_r32_best.pt",
        "checkpoint_type": "bearing",
        "tokenizer_type": "totem",
        "ts_tokenizer_s3": f"s3://{BUCKET}/{S3_PREFIX}/tokenizer/totem_clean.pt",
        "ts_tokenizer_channel": "totem_ckpt",
        "llm_id": "Qwen/Qwen3-4B",
        "instance": "ml.p5.48xlarge",
        "job_name": "fbench-dora-bearing-r32",
    },
    "phase0_fsq_transformer": {
        "base_ckpt": f"s3://{BUCKET}/{S3_PREFIX}/checkpoints/"
                     f"shrike-phase0-fsqt-4B-v6/shrike-phase0-fsqt-4B-v6/phase0_best.pt",
        "checkpoint_type": "shrike",
        "tokenizer_type": "fsq_transformer",
        # Matches phase0_fsq_transformer_4B.yaml (`fsq_ckpt: fsq_transformer_625_best.pt`)
        "ts_tokenizer_s3": f"s3://{BUCKET}/{S3_PREFIX}/tokenizer/fsq_transformer_625_best.pt",
        "ts_tokenizer_channel": "fsq_ckpt",
        "llm_id": "Qwen/Qwen3-4B",
        "instance": "ml.p5.48xlarge",
        "job_name": "fbench-dora-phase0-fsqt",
    },
    "phase0_totem": {
        "base_ckpt": f"s3://{BUCKET}/{S3_PREFIX}/checkpoints/"
                     f"totem-phase0-training-10epochs-v3/totem-phase0-training-10epochs-v3/"
                     f"phase0_best.pt",
        "checkpoint_type": "shrike",
        "tokenizer_type": "totem",
        # Matches phase0_totem.yaml (`fsq_ckpt: totem_625_best.pt`, yes, the
        # field is misnamed in that config but it's the 625-code TOTEM).
        "ts_tokenizer_s3": f"s3://{BUCKET}/{S3_PREFIX}/tokenizer/totem_625_best.pt",
        "ts_tokenizer_channel": "totem_ckpt",
        # phase0_totem.yaml uses Qwen3-1.7B (smaller than the FSQ variant).
        "llm_id": "Qwen/Qwen3-1.7B",
        "instance": "ml.p5.48xlarge",
        "job_name": "fbench-dora-phase0-totem",
    },
}


# -- Defaults for training hyperparameters ------------------------------
# Same shape as launch_factorybench.py, kept identical so the two sweeps are
# directly comparable. The fresh DoRA targets attention + MLP (the FactoryBench
# default in train_factorybench.py); existing per-checkpoint DoRA surface is
# absorbed by merge_and_unload before the fresh adapter is attached.
TRAIN_DEFAULTS = {
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 8,
    "max_length": 32768,
    "lr": 2e-5,
    "lora_r": 16,
    "lora_alpha": 32,
    "use_dora": True,
    "patience": 2,
    "warmup_frac": 0.10,
    "levels": "1,2,3",
    # 0 = full split. Set to ~100 in --smoke for a fast end-to-end sanity run.
    "max_train_samples": 0,
    "max_val_samples": 0,
}


def build_estimator(name: str, cfg: dict, args: argparse.Namespace,
                    sess: sagemaker.Session) -> tuple[PyTorch, dict]:
    """Build a SageMaker PyTorch estimator + its input channels for one checkpoint."""
    hyperparameters = {
        # Training knobs
        "EPOCHS":      str(args.epochs),
        "BATCH_SIZE":  str(args.batch_size),
        "GRAD_ACCUM":  str(args.grad_accum),
        "MAX_LENGTH":  str(args.max_length),
        "LR":          str(args.lr),
        "LORA_R":      str(args.lora_r),
        "LORA_ALPHA":  str(args.lora_alpha),
        "USE_DORA":    "true" if args.use_dora else "false",
        "PATIENCE":    str(args.patience),
        "LEVELS":      args.levels,
        "WARMUP_FRAC": str(args.warmup_frac),
        "MAX_TRAIN_SAMPLES": str(getattr(args, "max_train_samples", 0)),
        "MAX_VAL_SAMPLES":   str(getattr(args, "max_val_samples", 0)),
        # Where the SageMaker channels mount on the container
        "LLM_ID":           "/opt/ml/input/data/llm",
        "BASE_CKPT":        "/opt/ml/input/data/base_ckpt",
        "CHECKPOINT_TYPE":  cfg["checkpoint_type"],
        "TOKENIZER_TYPE":   cfg["tokenizer_type"],
    }
    # Wire the TS tokenizer to the right env var based on which kind it is.
    if cfg["ts_tokenizer_channel"] == "totem_ckpt":
        hyperparameters["TOTEM_CKPT"] = "/opt/ml/input/data/totem_ckpt"
    else:
        hyperparameters["FSQ_CKPT"] = "/opt/ml/input/data/fsq_ckpt"

    environment = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "WARN",
    }

    spot_kwargs = {}
    if args.spot:
        spot_kwargs = {
            "use_spot_instances": True,
            "max_run":  3600 * 12,
            "max_wait": 3600 * 24,
        }

    estimator = PyTorch(
        entry_point="train_factorybench.py",
        source_dir=str(SRC_DIR),
        role=ROLE,
        instance_count=1,
        instance_type=cfg["instance"],
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        distribution={"torch_distributed": {"enabled": True}},
        checkpoint_s3_uri=f"s3://{BUCKET}/factorybench/checkpoints/{cfg['job_name']}",
        checkpoint_local_path="/opt/ml/checkpoints",
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=f"s3://{BUCKET}/factorybench/output",
        volume_size=300,
        tags=[
            {"Key": "Project", "Value": "factorybench"},
            {"Key": "Sweep",   "Value": "dora-on-dora"},
            {"Key": "Base",    "Value": name},
        ],
        **spot_kwargs)

    data_channels = {
        "llm":       S3_LLM[cfg["llm_id"]],
        "data":      S3_FACTORYBENCH_DATA,
        "base_ckpt": cfg["base_ckpt"],
        cfg["ts_tokenizer_channel"]: cfg["ts_tokenizer_s3"],
    }
    return estimator, data_channels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="Subset of checkpoint keys to launch (default: all 4). "
                        f"Choices: {list(CHECKPOINTS)}")
    p.add_argument("--no-spot", action="store_true",
                   help="Use on-demand instances instead of spot.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the launch plan but don't submit any jobs.")

    # Training hyperparameter overrides (same names as launch_factorybench.py)
    for k, v in TRAIN_DEFAULTS.items():
        if isinstance(v, bool):
            continue   # spot is handled separately
        kind: type = type(v) if not isinstance(v, str) else str
        p.add_argument(f"--{k}", type=kind, default=v)
    p.add_argument("--use_dora", type=lambda s: s.lower() == "true",
                   default=TRAIN_DEFAULTS["use_dora"])

    args = p.parse_args()
    args.spot = not args.no_spot

    keys = args.only or list(CHECKPOINTS)
    unknown = [k for k in keys if k not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"Unknown checkpoint(s): {unknown}. "
                         f"Valid: {list(CHECKPOINTS)}")

    print("=" * 72)
    print(f"FactoryBench DoRA-on-DoRA sweep, {len(keys)} job(s)")
    print(f"Spot:        {args.spot}")
    print(f"LoRA r/α:    {args.lora_r}/{args.lora_alpha}, DoRA={args.use_dora}")
    print(f"Epochs:      {args.epochs}  Batch: {args.batch_size}x{args.grad_accum}")
    print(f"Max length:  {args.max_length}, Levels: {args.levels}")
    print("=" * 72)

    boto_sess = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_sess, default_bucket=BUCKET)

    submitted: list[str] = []
    skipped: list[tuple[str, str]] = []  # (key, reason)
    for key in keys:
        cfg = CHECKPOINTS[key]
        print(f"\n--- {key} ---")
        print(f"  base_ckpt:    {cfg['base_ckpt']}")
        print(f"  ts_tokenizer: {cfg['ts_tokenizer_s3']}  ({cfg['ts_tokenizer_channel']})")
        print(f"  llm_id:       {cfg['llm_id']}  (channel: {S3_LLM[cfg['llm_id']]})")
        print(f"  instance:     {cfg['instance']}")
        print(f"  job_name:     {cfg['job_name']}")

        if args.dry_run:
            continue

        estimator, channels = build_estimator(key, cfg, args, sess)
        try:
            estimator.fit(inputs=channels, job_name=cfg["job_name"], wait=False)
            submitted.append(key)
            print(f"  -> launched: {estimator.latest_training_job.name}")
        except Exception as e:
            # Per-instance-type quota caps are the usual culprit. Keep going
            # so other checkpoints (potentially on different instance types,
            # or just smaller chance of hitting the cap) still get a shot.
            reason = type(e).__name__ + ": " + str(e).split("\n", 1)[0]
            print(f"  !! SKIP {key}: {reason}", flush=True)
            skipped.append((key, reason))

    if args.dry_run:
        print("\n[dry-run] no jobs submitted.")
        return

    print(f"\nSubmitted {len(submitted)} job(s).")
    if skipped:
        print(f"Skipped {len(skipped)} job(s):")
        for key, reason in skipped:
            print(f"  - {key}: {reason}")
        retry_keys = " ".join(k for k, _ in skipped)
        print("\nRetry the skipped ones once a slot frees up:")
        print(f"  python finetuning/launch_dora_sweep.py --only {retry_keys}")
    print(f"\nMonitor: https://{REGION}.console.aws.amazon.com/sagemaker/home"
          f"?region={REGION}#/jobs")


if __name__ == "__main__":
    main()
