# FactoryBench finetuning

Does a time-series LLM, pretrained to reason over signals, do better on
FactoryBench than the same LLM without that pretraining, and does finetuning it
on FactoryBench help?

This directory answers that for **Shrike / BearingModel**, the authors's internal
time-series LLM: a Qwen3 backbone whose vocabulary is extended with discrete
`<ts_*>` codes produced by a VQ-VAE tokenizer over raw signal. Four pretrained
checkpoints are evaluated on FactoryBench as they are, then DoRA-finetuned on
FactoryBench and evaluated again. Everything runs on AWS SageMaker; the results
feed the baseline-versus-finetuned figures in the paper.

This is a **side experiment**, not part of the benchmark. Nothing under `src/`
imports anything here, and you do not need any of it to generate FactoryBench,
run the zero-shot LLM evaluation, or reproduce the main results table.

---

## Required environment

The AWS account, bucket and execution role are **not** in the source. Every
launcher imports them from `launch_dora_sweep.py`, which reads them from the
environment and exits with an explanatory message if any is unset.

| Variable | Meaning |
|---|---|
| `FB_SAGEMAKER_BUCKET` | S3 bucket holding the Shrike checkpoints, TS tokenizers and FactoryBench training JSONLs, and receiving job output. |
| `FB_SAGEMAKER_ROLE_ARN` | SageMaker execution role ARN. Needs read/write on that bucket. |
| `AWS_REGION` | Region of the bucket and of the jobs. `AWS_DEFAULT_REGION` is accepted as a fallback. |
| `FB_SAGEMAKER_PREFIX` | Key prefix for the Shrike artefacts. Optional, defaults to `shrike`. |

```bash
export FB_SAGEMAKER_BUCKET="my-sagemaker-bucket"
export FB_SAGEMAKER_ROLE_ARN="arn:aws:iam::<acct>:role/service-role/AmazonSageMaker-ExecutionRole-<id>"
export AWS_REGION="us-east-2"
```

The failure is deliberately loud and immediate. These scripts submit billable
GPU jobs, and boto3 with no explicit region and no explicit role will happily
resolve whatever profile you have active.

The bucket is expected to already hold, under `$FB_SAGEMAKER_PREFIX/`:
`checkpoints/` (the four `.pt` wrappers), `tokenizer/` (their matching TOTEM /
FSQ-Transformer `.pt` files), `llm/` (pre-staged Qwen3-4B and Qwen3-1.7B
weights), and under `factorybench/data/` the FactoryBench train/val/test
JSONLs. Staging those is a manual step; the launchers only reference them.

---

## The pipeline: 3 stages x 4 checkpoints

For each checkpoint, three SageMaker jobs run in order:

1. **Baseline eval.** Load the wrapper, merge its original DoRA (r=32) into the
   base weights, generate. This is how well the pretrained checkpoint already
   does on FactoryBench, before it has seen a single FactoryBench item.
2. **DoRA-on-DoRA finetune.** Same merge, then attach a *fresh* DoRA (r=16,
   alpha=32) on top and train it on the FactoryBench train split, levels 1-3.
   The original adapter is baked in, so only the new one moves.
3. **Finetuned eval.** Same wrapper, same merge, with the trained adapter
   stacked via `PeftModel.from_pretrained`.

The four checkpoints differ in wrapper class and in which tokenizer they were
pretrained against, which is why each carries its own tokenizer channel:

| Key | Wrapper | TS tokenizer | LLM |
|---|---|---|---|
| `bearing_mixed_r32_qwen3_4b` | `BearingModel` | TOTEM (256-code) | Qwen3-4B |
| `bearing_r32_qwen3_4b` | `BearingModel` | TOTEM (256-code) | Qwen3-4B |
| `phase0_fsq_transformer` | `Shrike` | FSQ-Transformer (625) | Qwen3-4B |
| `phase0_totem` | `Shrike` | TOTEM (625) | Qwen3-4B |

Training runs on `ml.p5.48xlarge`, eval on `ml.g5.2xlarge`. Both default to
spot. `run_pipeline.py` runs stages sequentially *within* a model and models
sequentially *across* the sweep, so exactly one training job holds quota at any
moment. That is the right shape when the per-instance-type spot limit is 1, and
it is why a full sweep takes on the order of 12-24 hours.

---

## Entry points

| File | What it does |
|---|---|
| `run_pipeline.py` | **Start here.** Drives all three stages for one or more checkpoints, blocking on each. `--skip` resumes mid-pipeline after a crash; `--smoke` runs 10 eval samples per level and 1 train epoch. |
| `launch_orchestrator_on_cloud.py` | Runs `run_pipeline.py` itself inside a SageMaker Processing job, so the long sweep survives your laptop closing. Uploads this directory as `source_dir`. |
| `launch_dora_sweep.py` | Stage 2 alone: submits the four finetuning jobs with `--wait False` so they queue in parallel. Also the single source of the per-checkpoint config and the AWS settings every other launcher imports. |
| `launch_eval_sweep.py` | Stages 1 and 3 alone: `--mode baseline` or `--mode finetuned`. |
| `score_predictions.py` | Re-scores a downloaded eval with the official cascade (`src.scoring.cascade`). **Use this, not the job's own numbers.** |
| `peek_predictions.py` | Prints sample predictions from a `level_*_predictions.jsonl`; `--wrong` shows only mismatches. |
| `plot_baseline_vs_finetuned.py` | Bar charts from the `*_summary.json` files the scorer writes. Owned by the paper-figure workstream. |

Under `_factorybench_src/`, which is the `source_dir` uploaded into the
training and eval containers:

| File | What it does |
|---|---|
| `train_factorybench.py` | The trainer. Loads a wrapper or a plain HF model, merges, attaches the fresh DoRA, trains. |
| `eval_factorybench.py` | The evaluator. Baseline or `--adapter_dir` mode. Reuses the trainer's prompt formatter directly, because train/eval prompt skew is silent and ruinous. |
| `ts_prompt.py` | Converts FactoryBench's `feature=value` text rows into the per-channel `<ts_*>` token prompt the checkpoints were pretrained on. |
| `shrike/` | **Vendored.** A partial, frozen copy of the internal Shrike repo, present only so the checkpoints deserialize. Read the header in `shrike/__init__.py` before touching it. |
| `requirements.txt` | Container manifest. SageMaker pip-installs it before the job runs. Not a project manifest. |

---

## Running it

```bash
# Everything, all four checkpoints (long)
python finetuning/run_pipeline.py

# One checkpoint, end to end
python finetuning/run_pipeline.py --model bearing_r32_qwen3_4b

# Prove the wiring works before spending anything real
python finetuning/run_pipeline.py --model bearing_r32_qwen3_4b --smoke

# Resume after a crash, skipping what already finished
python finetuning/run_pipeline.py --model phase0_totem --skip baseline train

# Hand the whole sweep to a Processing job and close the laptop
python finetuning/launch_orchestrator_on_cloud.py
```

Results land in
`s3://$FB_SAGEMAKER_BUCKET/factorybench/eval-output/<job>/output/model.tar.gz`,
containing one `level_N_predictions.jsonl` per level plus `eval_summary.json`.

```bash
aws s3 cp <uri> model.tar.gz && mkdir baseline_bm && tar xzf model.tar.gz -C baseline_bm
python finetuning/score_predictions.py baseline_bm/            # deterministic levels
python finetuning/score_predictions.py baseline_bm/ --judge    # + LLM judge for L4
```

### Score with `score_predictions.py`, not with `eval_summary.json`

`eval_factorybench.py` logs exact-match accuracy, which is the wrong metric for
most of FactoryBench: rankings want Kendall tau, multi-select wants per-bit
agreement, numerics want tolerance bands, and L4 free-form needs the three-judge
rubric. `eval_summary.json` is a liveness signal during a run, nothing more. Any
number that leaves this directory should come from `score_predictions.py`, which
wraps the same `src.scoring.cascade` the rest of the repository reports through.
