"""Launch FactoryBench evaluation on SageMaker for all 4 pretrained Shrike
checkpoints, in baseline mode (no FactoryBench adapter) or in finetuned mode
(stacking the DoRA trained by launch_dora_sweep.py).

Workflow:

    # 1. Baseline eval, pretrained Shrike/Bearing checkpoints, no FactoryBench finetune
    python finetuning/launch_eval_sweep.py --mode baseline

    # 2. Launch finetuning (separate command)
    python finetuning/launch_dora_sweep.py

    # 3. After finetuning completes, eval the finetuned versions
    python finetuning/launch_eval_sweep.py --mode finetuned

Predictions land in:
    s3://{BUCKET}/factorybench/eval-output/{eval_job_name}/output/model.tar.gz
which contains level_X_predictions.jsonl + eval_summary.json. Download with:
    aws s3 cp <uri> model.tar.gz && tar xzf model.tar.gz
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

from launch_dora_sweep import (  # type: ignore[import-not-found]
    BUCKET, REGION, ROLE, S3_PREFIX,
    S3_LLM, S3_FACTORYBENCH_DATA,
    SRC_DIR, CHECKPOINTS)


# Inference only, one A10G GPU (24 GB) is plenty for both Qwen sizes at batch=1.
# We deliberately use the same instance type for both so a single g5.2xlarge
# spot quota slot covers every eval job (only one runs at a time anyway in
# the pipeline orchestrator). If you raise quota on g5.xlarge separately,
# switch the 1.7B entry back for slight cost savings.
EVAL_INSTANCE_BY_LLM = {
    "Qwen/Qwen3-4B":   "ml.g5.2xlarge",
    "Qwen/Qwen3-1.7B": "ml.g5.2xlarge",
}


# Default eval hyperparameters
EVAL_DEFAULTS = {
    "levels": "1,2,3,4",
    "split": "test",
    "max_samples": 0,          # 0 = all
    "max_input_length": 16384,
    "max_new_tokens": 200,
    "batch_size": 1,
}


def _adapter_s3_uri(job_name: str) -> str:
    """Where launch_dora_sweep.py wrote the trained adapter to.

    train_factorybench.py saves adapter/ to BOTH /opt/ml/model (gets tarred
    into model.tar.gz) AND /opt/ml/checkpoints/{job_name}/adapter/ (synced raw
    to S3 via checkpoint_s3_uri). We mount the raw-files copy so SageMaker
    doesn't have to untar anything.

    The doubled {job_name} is the SageMaker checkpoint-sync behavior: with
    checkpoint_local_path=/opt/ml/checkpoints and
    checkpoint_s3_uri=s3://.../checkpoints/{job_name}, the local subdir
    /opt/ml/checkpoints/{job_name}/X ends up at .../checkpoints/{job_name}/{job_name}/X.
    Same pattern as TEMPO's shrike-phase0-fsqt-4B-v6/shrike-phase0-fsqt-4B-v6/...
    """
    return f"s3://{BUCKET}/factorybench/checkpoints/{job_name}/{job_name}/adapter/"


_TIMESTAMP_RE = re.compile(r"-(\d{8}-\d{6})$")
_SM_JOB_NAME_MAX = 63


def build_estimator(name: str, cfg: dict, mode: str, args: argparse.Namespace,
                    sess: sagemaker.Session) -> tuple[PyTorch, dict, str]:
    """Build a SageMaker PyTorch estimator + input channels for one eval job.

    Returns (estimator, data_channels, eval_job_name). The job name includes
    a timestamp suffix so re-running the launcher never collides with prior
    submissions, SageMaker permanently reserves names once used.

    If cfg['job_name'] already ends with a -YYYYMMDD-HHMMSS timestamp
    (e.g. when called from run_pipeline.py which timestamps the training
    job name and reuses it for the eval cfg so the adapter URI matches),
    we strip and reuse that timestamp instead of stacking a second one, otherwise the final eval_job_name blows past SageMaker's 63-char limit.
    """
    job_suffix = "baseline" if mode == "baseline" else "finetuned"
    m = _TIMESTAMP_RE.search(cfg["job_name"])
    if m:
        base = cfg["job_name"][: m.start()]
        suffix = m.group(1)
    else:
        base = cfg["job_name"]
        suffix = getattr(args, "job_suffix", None) or \
            datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_job_name = f"{base}-eval-{job_suffix}-{suffix}"
    if len(eval_job_name) > _SM_JOB_NAME_MAX:
        # Shouldn't happen with current keys (max ~61 chars) but guard anyway.
        overflow = len(eval_job_name) - _SM_JOB_NAME_MAX
        base = base[:-overflow]
        eval_job_name = f"{base}-eval-{job_suffix}-{suffix}"
    instance = EVAL_INSTANCE_BY_LLM.get(cfg["llm_id"], "ml.g5.2xlarge")

    hyperparameters = {
        "LEVELS":           args.levels,
        "SPLIT":            args.split,
        "MAX_SAMPLES":      str(args.max_samples),
        "MAX_INPUT_LENGTH": str(args.max_input_length),
        "MAX_NEW_TOKENS":   str(args.max_new_tokens),
        "BATCH_SIZE":       str(args.batch_size),
        # Mirror the training-side channel naming so the entry script's
        # SM_CHANNEL_* lookups pick everything up automatically.
        "LLM_ID":          "/opt/ml/input/data/llm",
        "BASE_CKPT":       "/opt/ml/input/data/base_ckpt",
        "CHECKPOINT_TYPE": cfg["checkpoint_type"],
        "TOKENIZER_TYPE":  cfg["tokenizer_type"],
    }
    if cfg["ts_tokenizer_channel"] == "totem_ckpt":
        hyperparameters["TOTEM_CKPT"] = "/opt/ml/input/data/totem_ckpt"
    else:
        hyperparameters["FSQ_CKPT"] = "/opt/ml/input/data/fsq_ckpt"

    data_channels = {
        "llm":       S3_LLM[cfg["llm_id"]],
        "data":      S3_FACTORYBENCH_DATA,
        "base_ckpt": cfg["base_ckpt"],
        cfg["ts_tokenizer_channel"]: cfg["ts_tokenizer_s3"],
    }

    if mode == "finetuned":
        # When running standalone (after a completed training job), the
        # user can pass --train-job-name to point the adapter URI at a
        # specific training run. Otherwise fall back to cfg['job_name']
        # (which run_pipeline.py sets to the timestamped train job name).
        train_jn = getattr(args, "train_job_name", None) or cfg["job_name"]
        adapter_uri = _adapter_s3_uri(train_jn)
        data_channels["adapter"] = adapter_uri
        hyperparameters["ADAPTER_DIR"] = "/opt/ml/input/data/adapter"

    # --resume-from-job: mount a prior eval's predictions so this run skips
    # already-decoded samples. Tries the raw-files location (new resumable
    # code) first, then falls back to the output tarball.
    prior_job = getattr(args, "resume_from_job", None)
    if prior_job:
        # Prefer the raw-files path (cheap, no untar), fall back to tarball.
        ckpt_uri = f"s3://{BUCKET}/factorybench/eval-checkpoints/{prior_job}/"
        out_uri = f"s3://{BUCKET}/factorybench/eval-output/{prior_job}/output/model.tar.gz"
        # We can't probe S3 from here without an extra round-trip; just point
        # at the tarball path which always exists for completed jobs. The
        # eval entry script handles BOTH layouts inside the mounted dir.
        data_channels["prior_predictions"] = out_uri
        hyperparameters["PRIOR_PREDICTIONS_DIR"] = "/opt/ml/input/data/prior_predictions"

    environment = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "WARN",
    }

    spot_kwargs = {}
    if args.spot:
        # 24h hard cap (not 6h). Eval over the full FactoryBench test set
        # with max_new_tokens=200 is ~3-5h of real compute; once we add
        # spot-reclaim retries (which used to lose all progress before the
        # resume fix below), we need real headroom. 24h is still well under
        # SageMaker's 5-day spot max.
        spot_kwargs = {
            "use_spot_instances": True,
            "max_run":  3600 * 24,
            "max_wait": 3600 * 48,
        }

    estimator = PyTorch(
        entry_point="eval_factorybench.py",
        source_dir=str(SRC_DIR),
        role=ROLE,
        instance_count=1,
        instance_type=instance,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=sess,
        hyperparameters=hyperparameters,
        environment=environment,
        output_path=f"s3://{BUCKET}/factorybench/eval-output",
        # Persistent across spot reclaims: SageMaker syncs anything we write
        # under checkpoint_local_path to checkpoint_s3_uri as raw files (NOT
        # tarballed). On restart, it pulls those files back, so the eval
        # script can resume from where it left off instead of starting over.
        checkpoint_s3_uri=f"s3://{BUCKET}/factorybench/eval-checkpoints/{eval_job_name}",
        checkpoint_local_path="/opt/ml/checkpoints",
        volume_size=200,
        disable_profiler=True,
        tags=[
            {"Key": "Project", "Value": "factorybench"},
            {"Key": "Sweep",   "Value": f"eval-{mode}"},
            {"Key": "Base",    "Value": name},
        ],
        **spot_kwargs)
    return estimator, data_channels, eval_job_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "finetuned"], required=True)
    p.add_argument("--only", nargs="*", default=None,
                   help="Subset of checkpoint keys to eval (default: all). "
                        f"Choices: {list(CHECKPOINTS)}")
    p.add_argument("--no-spot", action="store_true",
                   help="Use on-demand instances instead of spot.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the launch plan but don't submit jobs.")
    p.add_argument("--wait", action="store_true",
                   help="Block until each job finishes before submitting the next. "
                        "Default is to fire-and-forget (all 4 run in parallel).")
    p.add_argument("--train-job-name", type=str, default=None,
                   help="(finetuned mode only, --only single checkpoint) Use this "
                        "exact training job name when computing the adapter S3 URI. "
                        "Lets you eval the adapter from a specific past training run.")
    p.add_argument("--resume-from-job", type=str, default=None,
                   help="Name of a prior eval job; its level_*_predictions.jsonl "
                        "(from tarball or eval-checkpoints) are mounted and the "
                        "new run skips already-decoded samples.")
    # Eval hyperparameter overrides
    for k, v in EVAL_DEFAULTS.items():
        kind: type = type(v) if not isinstance(v, str) else str
        p.add_argument(f"--{k}", type=kind, default=v)

    args = p.parse_args()
    args.spot = not args.no_spot

    keys = args.only or list(CHECKPOINTS)
    unknown = [k for k in keys if k not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"Unknown checkpoint(s): {unknown}. "
                         f"Valid: {list(CHECKPOINTS)}")

    print("=" * 72)
    print(f"FactoryBench Eval, mode={args.mode}, {len(keys)} job(s)")
    print(f"Spot:        {args.spot}, wait={args.wait}")
    print(f"Levels:      {args.levels}  Split: {args.split}")
    print(f"Gen budget:  max_input={args.max_input_length}, "
          f"max_new={args.max_new_tokens}, batch={args.batch_size}")
    print(f"Max samples: {args.max_samples or 'ALL'}")
    print("=" * 72)

    # In finetuned mode, surface the adapter location we'll be reading
    # from so the user can spot stale/missing artifacts before submitting.
    if args.mode == "finetuned":
        s3 = boto3.client("s3", region_name=REGION)
        for key in keys:
            uri = _adapter_s3_uri(CHECKPOINTS[key]["job_name"])
            bucket = uri.split("/", 3)[2]
            prefix = uri.split("/", 3)[3]
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
            n = resp.get("KeyCount", 0)
            status = f"{n} object(s)" if n else "EMPTY, adapter not yet synced"
            print(f"  adapter@{key}: {uri}  [{status}]")
        print()

    boto_sess = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_sess, default_bucket=BUCKET)

    submitted: list[tuple[str, str]] = []
    skipped: list[tuple[str, str, str]] = []   # (key, job_name, reason)
    for key in keys:
        cfg = CHECKPOINTS[key]
        print(f"\n--- {key}  (mode={args.mode}) ---")
        estimator, channels, eval_job_name = build_estimator(
            key, cfg, args.mode, args, sess)
        print(f"  job_name:   {eval_job_name}")
        print(f"  instance:   {estimator.instance_type}")
        for ch, uri in channels.items():
            print(f"  channel/{ch:<12} {uri}")
        if args.dry_run:
            continue

        try:
            estimator.fit(inputs=channels, job_name=eval_job_name, wait=args.wait,
                          logs="All" if args.wait else None)
            submitted.append((key, eval_job_name))
            print(f"  -> launched: {estimator.latest_training_job.name}")
        except Exception as e:
            # Most common: ResourceLimitExceeded (per-instance-type quota cap).
            # Don't let one quota-blocked job stop the rest of the sweep, # other checkpoints may use a different instance type that does
            # have headroom, and the user can retry the failed ones later
            # with --only after a quota increase or after another job ends.
            reason = type(e).__name__ + ": " + str(e).split("\n", 1)[0]
            print(f"  !! SKIP {key}: {reason}", flush=True)
            skipped.append((key, eval_job_name, reason))

    if args.dry_run:
        print("\n[dry-run] no jobs submitted.")
        return

    print(f"\nSubmitted {len(submitted)} job(s).")
    if skipped:
        print(f"Skipped {len(skipped)} job(s):")
        for key, job, reason in skipped:
            print(f"  - {key} ({job}): {reason}")
        print("\nTo retry just the skipped ones after a quota increase or "
              "after the running job ends:")
        retry_keys = " ".join(k for k, _, _ in skipped)
        print(f"  python finetuning/launch_eval_sweep.py --mode {args.mode} "
              f"--only {retry_keys}")
    print(f"\nMonitor: https://{REGION}.console.aws.amazon.com/sagemaker/home"
          f"?region={REGION}#/jobs")
    if submitted:
        print("\nResults (model.tar.gz) will appear at:")
        for key, job in submitted:
            print(f"  s3://{BUCKET}/factorybench/eval-output/{job}/output/model.tar.gz")


if __name__ == "__main__":
    main()
