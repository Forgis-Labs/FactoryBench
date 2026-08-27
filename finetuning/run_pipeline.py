"""Run the full FactoryBench cloud pipeline ONE model at a time.

For each selected checkpoint, runs three stages in sequence and blocks on
each until SageMaker reports completion:

    1. Baseline eval, pretrained Shrike/Bearing on FactoryBench test
    2. DoRA-on-DoRA finetune, train a fresh adapter on FactoryBench train
    3. Finetuned eval, same model with the new adapter stacked on top

Stages run sequentially within a model and models run sequentially across the
sweep, so at any moment only ONE SageMaker training job is consuming quota.
This is the right shape when your per-instance-type spot limit is 1.

Usage:
    # All 4 models, full pipeline (long-running, ~12-24h depending on quotas)
    python finetuning/run_pipeline.py

    # Just one model
    python finetuning/run_pipeline.py --model bearing_r32_qwen3_4b

    # Smoke test: 10 eval samples per level, 1 train epoch
    python finetuning/run_pipeline.py --model bearing_r32_qwen3_4b --smoke

    # Resume mid-pipeline after a crash (skip already-done stages)
    python finetuning/run_pipeline.py --model phase0_totem --skip baseline train
"""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

# Same dir as the existing launchers, import them to reuse the per-checkpoint
# config and the estimator builders, so there's exactly one source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import boto3
import sagemaker

from launch_dora_sweep import (  # noqa: E402
    CHECKPOINTS, BUCKET, REGION,
    TRAIN_DEFAULTS, build_estimator as build_train_estimator)
from launch_eval_sweep import (  # noqa: E402
    EVAL_DEFAULTS, build_estimator as build_eval_estimator)


STAGES = ("baseline", "train", "finetuned")


def _train_args(args: Namespace) -> Namespace:
    """Build a Namespace shaped like launch_dora_sweep.py's argparse output."""
    out = Namespace(**TRAIN_DEFAULTS)
    out.spot = args.spot
    if args.epochs is not None:
        out.epochs = args.epochs
    if args.smoke:
        # End-to-end sanity run: 1 epoch over 100 train / 20 val samples.
        # ~3-5 min of actual training instead of an hour, while still
        # exercising save/load/eval handoffs in stage 3.
        out.epochs = 1
        out.max_train_samples = 100
        out.max_val_samples = 20
    # Explicit CLI overrides win over --smoke defaults.
    if args.max_train_samples is not None:
        out.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        out.max_val_samples = args.max_val_samples
    return out


def _eval_args(args: Namespace) -> Namespace:
    """Build a Namespace shaped like launch_eval_sweep.py's argparse output."""
    out = Namespace(**EVAL_DEFAULTS)
    out.spot = args.spot
    if args.max_samples is not None:
        out.max_samples = args.max_samples
    if args.smoke and args.max_samples is None:
        out.max_samples = 10
    return out


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s"


def _run_stage(label: str, fit_callable) -> None:
    """Call fit_callable (which wraps estimator.fit(wait=True)) and time it."""
    t0 = time.time()
    print(f"\n>>> {label}, submitting", flush=True)
    fit_callable()
    print(f"<<< {label}, done in {_format_elapsed(time.time() - t0)}", flush=True)


def _submit_with_quota_retry(estimator, inputs: dict, job_name: str,
                             retry_seconds: int, max_wait_seconds: int) -> None:
    """estimator.fit() with auto-retry on ResourceLimitExceeded.

    SageMaker rejects the create-training-job call immediately if you're at
    your per-instance-type quota cap. Rather than failing the pipeline, we
    sleep for ``retry_seconds`` and try again, typically the blocking job
    will free its slot within that interval. Caps total wait at
    ``max_wait_seconds`` so a permanently-stuck quota doesn't hang forever.
    """
    t0 = time.time()
    attempt = 0
    while True:
        try:
            estimator.fit(inputs=inputs, job_name=job_name, wait=True, logs="All")
            return
        except Exception as e:
            msg = str(e)
            if "ResourceLimitExceeded" not in msg:
                # Anything else (auth, bad channel, etc.), let it bubble up.
                raise
            elapsed = time.time() - t0
            if elapsed >= max_wait_seconds:
                raise
            attempt += 1
            mins = retry_seconds // 60
            print(f"  [quota] full ({_format_elapsed(elapsed)} elapsed). "
                  f"Sleeping {mins} min before retry #{attempt}...", flush=True)
            time.sleep(retry_seconds)


def run_pipeline_for_model(key: str, cfg: dict, args: Namespace,
                           sess: sagemaker.Session) -> None:
    """Run the 3 stages for one checkpoint, blocking between each.

    All 3 SageMaker job names get the same timestamp suffix, so:
      * re-running the pipeline never collides with names from a prior run;
      * the finetuned-eval can find its adapter by deriving the S3 path from
        the (timestamped) training job name, they share the suffix.
    """
    # One timestamp per (model, pipeline invocation), flows into every stage.
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_run = dict(cfg)
    cfg_run["job_name"] = f"{cfg['job_name']}-{suffix}"

    print("\n" + "=" * 72)
    print(f"Pipeline: {key}")
    print(f"  base_ckpt:    {cfg['base_ckpt']}")
    print(f"  ts_tokenizer: {cfg['ts_tokenizer_s3']}")
    print(f"  llm_id:       {cfg['llm_id']}")
    print(f"  run id:       {cfg_run['job_name']}")
    print(f"  skip:         {sorted(args.skip)}")
    print("=" * 72, flush=True)

    eval_a = _eval_args(args)
    train_a = _train_args(args)
    t_model = time.time()
    retry_kw = dict(retry_seconds=args.quota_retry_seconds,
                    max_wait_seconds=args.quota_max_wait)

    # ---------------- Stage 1: baseline eval ----------------
    if "baseline" not in args.skip:
        def _fit_baseline():
            est, ch, jn = build_eval_estimator(key, cfg_run, "baseline",
                                               eval_a, sess)
            print(f"    instance:   {est.instance_type}")
            print(f"    job_name:   {jn}")
            _submit_with_quota_retry(est, ch, jn, **retry_kw)
        _run_stage(f"[{key}] 1/3 baseline eval", _fit_baseline)
    else:
        print(f"\n[skip] {key} stage 1 (baseline eval)")

    # ---------------- Stage 2: finetune ----------------
    if "train" not in args.skip:
        def _fit_train():
            est, ch = build_train_estimator(key, cfg_run, train_a, sess)
            print(f"    instance:   {est.instance_type}")
            print(f"    job_name:   {cfg_run['job_name']}")
            _submit_with_quota_retry(est, ch, cfg_run["job_name"], **retry_kw)
        _run_stage(f"[{key}] 2/3 finetune", _fit_train)
    else:
        print(f"\n[skip] {key} stage 2 (finetune)")

    # ---------------- Stage 3: finetuned eval ----------------
    # build_eval_estimator derives the adapter URI from cfg_run['job_name'],
    # which is the timestamped training name above, so the eval job picks up
    # the adapter that the training stage just wrote.
    if "finetuned" not in args.skip:
        def _fit_finetuned():
            est, ch, jn = build_eval_estimator(key, cfg_run, "finetuned",
                                               eval_a, sess)
            print(f"    instance:   {est.instance_type}")
            print(f"    job_name:   {jn}")
            _submit_with_quota_retry(est, ch, jn, **retry_kw)
        _run_stage(f"[{key}] 3/3 finetuned eval", _fit_finetuned)
    else:
        print(f"\n[skip] {key} stage 3 (finetuned eval)")

    print(f"\n[{key}] full pipeline done in "
          f"{_format_elapsed(time.time() - t_model)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="*", default=None,
                   help="Checkpoints to run (default: all 4, in order). "
                        f"Choices: {list(CHECKPOINTS)}")
    p.add_argument("--skip", nargs="*", default=[], choices=STAGES,
                   help="Stages to skip (applies to ALL selected models). "
                        "Useful for resuming after a crash.")
    p.add_argument("--no-spot", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run: 1 train epoch, --max_samples 10 for eval.")
    # Common overrides
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Eval-side cap (samples per level).")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Training-set cap (e.g. 5000 for a partial-data run).")
    p.add_argument("--max_val_samples", type=int, default=None,
                   help="Validation-set cap.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-model plan; don't submit anything.")
    # Quota-retry behavior, defaults are generous; tune if you know your
    # other jobs are short or you want to bail out faster.
    p.add_argument("--quota-retry-seconds", type=int, default=300,
                   help="Seconds to wait between retries when a submission "
                        "hits ResourceLimitExceeded (default: 300 = 5 min).")
    p.add_argument("--quota-max-wait", type=int, default=12 * 3600,
                   help="Max seconds to keep retrying one submission "
                        "(default: 12h). After this, the stage fails.")
    args = p.parse_args()
    args.spot = not args.no_spot

    keys = args.model or list(CHECKPOINTS)
    unknown = [k for k in keys if k not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"Unknown checkpoint(s): {unknown}. "
                         f"Valid: {list(CHECKPOINTS)}")

    print("=" * 72)
    print(f"FactoryBench full pipeline, {len(keys)} model(s), sequential")
    print(f"Models:  {keys}")
    print(f"Skip:    {sorted(args.skip) or '<none>'}")
    print(f"Spot:    {args.spot}")
    if args.smoke:
        print("Smoke:   ON  (1 epoch, max_samples=10)")
    if args.epochs is not None:
        print(f"Epochs:  {args.epochs}")
    if args.max_samples is not None:
        print(f"Samples: {args.max_samples}")
    print("=" * 72)

    if args.dry_run:
        for key in keys:
            print(f"\n--- plan for {key} ---")
            for stage in STAGES:
                tag = "SKIP" if stage in args.skip else "RUN "
                print(f"  [{tag}] {stage}")
        print("\n[dry-run] no jobs submitted.")
        return

    boto_sess = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto_sess, default_bucket=BUCKET)

    t_all = time.time()
    failed: list[tuple[str, str]] = []
    for i, key in enumerate(keys, start=1):
        print(f"\n\n############  model {i}/{len(keys)}: {key}  ############")
        try:
            run_pipeline_for_model(key, CHECKPOINTS[key], args, sess)
        except Exception as e:
            # Failure mid-pipeline: log and continue with the next model so
            # one bad checkpoint doesn't block the rest of the overnight run.
            reason = type(e).__name__ + ": " + str(e).split("\n", 1)[0]
            print(f"\n!! pipeline FAILED for {key}: {reason}", flush=True)
            failed.append((key, reason))

    print("\n" + "=" * 72)
    print(f"Sweep complete in {_format_elapsed(time.time() - t_all)}.")
    print(f"  succeeded: {len(keys) - len(failed)} / {len(keys)}")
    if failed:
        print("  failures:")
        for key, reason in failed:
            print(f"    - {key}: {reason}")
        retry = " ".join(k for k, _ in failed)
        print(f"\n  retry with:  python finetuning/run_pipeline.py --model {retry}")
    print("=" * 72)


if __name__ == "__main__":
    main()
