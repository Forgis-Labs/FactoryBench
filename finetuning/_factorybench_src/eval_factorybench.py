"""Evaluate a Shrike / BearingModel checkpoint (optionally with a trained
FactoryBench DoRA adapter on top) on the FactoryBench test split.

Two evaluation modes, controlled purely by whether ``--adapter_dir`` is set:

  1. Baseline (no adapter): loads the wrapper, merges its DoRA into the base,
     and generates from the merged model directly. This tells you how well
     the pretrained Shrike/Bearing checkpoint already does on FactoryBench
     before any FactoryBench-specific finetuning.

  2. Post-finetune (with adapter): loads the wrapper, merges its DoRA, then
     stacks the trained FactoryBench DoRA from ``--adapter_dir`` via
     ``PeftModel.from_pretrained``. Generates from the stacked model.

Prompt building MUST stay byte-identical to ``FactoryBenchDataset._format_chatml``
in train_factorybench.py, eval/train skew is silent and ruinous, so the
formatter is reused directly from that module.

Output (written to ``--output_dir``, which on SageMaker is /opt/ml/model
and therefore lands in model.tar.gz):

    level_1_predictions.jsonl  ─ one JSON record per test sample
    level_2_predictions.jsonl
    ...
    eval_summary.json          ─ exact-match accuracy per level + overall

Each prediction record matches the schema in eval_predictions/ in this repo:

    {"idx": int, "id": str, "level": int, "template_id": int,
     "template_type": str, "question_type": null, "question": str,
     "options": dict, "ground_truth": <any>, "answer": str,
     "n_pred_tokens": int}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import torch

# Reuse the training-time prompt builder so eval and train share one
# canonical wire format. The dataset class lives in the same source dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_factorybench import FactoryBenchDataset  # noqa: E402


def _env_opt(key: str) -> str | None:
    """Read SM_HP_<KEY> or <KEY> from env; return None for missing/empty/'none'."""
    v = os.environ.get(f"SM_HP_{key}", os.environ.get(key))
    if v is None or v == "" or v.lower() == "none":
        return None
    return v


def _resolve_pt(path: str | None) -> str | None:
    """If path is a dir, return the .pt inside (preferring 'best')."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.is_dir():
        pts = sorted(p.rglob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt file under {path}")
        best = [x for x in pts if "best" in x.name.lower()]
        return str(best[0] if best else pts[0])
    raise FileNotFoundError(f"Path does not exist: {path}")


def _seed_resume_dir(prior_channel: str | None, ckpt_root: Path) -> int:
    """Copy prior predictions into the resume checkpoint dir.

    `prior_channel` is a SageMaker channel mount that may contain either:
      * raw ``level_*_predictions.jsonl`` files (from a prior run that used
        the resumable code path → eval-checkpoints S3 prefix), or
      * a ``model.tar.gz`` (the output artifact from an older run) which we
        untar to find the JSONLs.

    Returns the number of files seeded into ``ckpt_root``.
    """
    if not prior_channel:
        return 0
    p = Path(prior_channel)
    if not p.exists():
        return 0
    ckpt_root.mkdir(parents=True, exist_ok=True)

    seeded = 0
    # Case 1: raw JSONLs in the channel root (or nested anywhere under it)
    for src in p.rglob("level_*_predictions.jsonl"):
        dst = ckpt_root / src.name
        if not dst.exists():
            import shutil
            shutil.copyfile(src, dst)
            seeded += 1

    # Case 2: tarball, extract any level_*.jsonl found inside
    if seeded == 0:
        for tar in list(p.glob("*.tar.gz")) + list(p.glob("*.tar")):
            try:
                with tarfile.open(tar, "r:*") as tf:
                    for m in tf.getmembers():
                        name = Path(m.name).name
                        if name.startswith("level_") and name.endswith("_predictions.jsonl"):
                            dst = ckpt_root / name
                            if dst.exists():
                                continue
                            f = tf.extractfile(m)
                            if f is None:
                                continue
                            with open(dst, "wb") as out:
                                out.write(f.read())
                            seeded += 1
            except (tarfile.TarError, OSError) as e:
                print(f"  [warn] couldn't extract prior tarball {tar}: {e}",
                      flush=True)

    if seeded:
        print(f"  [resume] seeded {seeded} prior prediction file(s) from "
              f"{prior_channel} -> {ckpt_root}", flush=True)
    return seeded


def _resolve_adapter_dir(path: str | None) -> str | None:
    """Resolve the adapter directory.

    A SageMaker channel may point at any of:
      - a directory containing adapter_config.json + adapter_model.safetensors
      - a directory containing nested subdirs (e.g. checkpoint dir with
        adapter/ as a child)
      - a model.tar.gz file (auto-extracted into a temp dir)

    Returns the absolute path to the dir that holds adapter_config.json.
    """
    if not path:
        return None
    p = Path(path)
    if p.is_file() and p.suffix in (".tar", ".gz") or p.name.endswith(".tar.gz"):
        # Auto-extract tarballs so callers can point at model.tar.gz directly.
        extract_dir = Path("/tmp") / f"adapter_extracted_{int(time.time())}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(p, "r:*") as tf:
            tf.extractall(extract_dir)
        p = extract_dir
    if not p.is_dir():
        raise FileNotFoundError(f"Adapter path is not a directory: {path}")
    # Find the dir that actually contains adapter_config.json.
    candidates = [p] + sorted(p.rglob("adapter_config.json"))
    for c in candidates:
        d = c.parent if c.is_file() else c
        if (d / "adapter_config.json").is_file():
            return str(d)
    raise FileNotFoundError(f"No adapter_config.json under {path}")


def _build_base_model(
    base_ckpt: str | None,
    checkpoint_type: str | None,
    tokenizer_type: str | None,
    totem_ckpt: str | None,
    fsq_ckpt: str | None,
    llm_id: str):
    """Load the base LLM (with original DoRA already merged) + tokenizer.

    Mirrors train_factorybench._load_base_with_shrike_loader so eval picks up
    exactly the same weights the training run started from.
    """
    if base_ckpt:
        ck = (checkpoint_type or "").lower()
        if ck == "bearing":
            from shrike.model.bearing import BearingModel
            if not totem_ckpt:
                raise ValueError("--totem_ckpt is required for bearing checkpoints")
            wrapper = BearingModel.from_pretrained(
                checkpoint_path=base_ckpt, totem_ckpt=totem_ckpt,
                llm_id=llm_id, device="cpu")
        elif ck == "shrike":
            from shrike.model.shrike import Shrike
            kwargs: dict = {"checkpoint_path": base_ckpt, "llm_id": llm_id,
                            "device": "cpu", "tokenizer_type": tokenizer_type}
            if totem_ckpt:
                kwargs["totem_ckpt"] = totem_ckpt
            if fsq_ckpt:
                kwargs["fsq_ckpt"] = fsq_ckpt
            wrapper = Shrike.from_pretrained(**kwargs)
        else:
            raise ValueError(f"--checkpoint_type must be shrike|bearing, got {ck!r}")

        print("  Merging existing DoRA into base LLM weights...", flush=True)
        merged = wrapper.llm.merge_and_unload()
        tokenizer = wrapper.tokenizer
        # KEEP the TS tokenizer alive, needed to encode FactoryBench
        # signals into <ts_*> codes (the format the model was pretrained on).
        wrapper.llm = None
        gc.collect()
        return merged, tokenizer, wrapper

    # Fallback: vanilla HF model (no Shrike/Bearing wrapper).
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_id, trust_remote_code=True)
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        llm_id, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation=attn_impl)
    return model, tokenizer, None


def _stack_adapter(base_model, adapter_dir: str):
    """Load a trained PEFT adapter on top of the already-merged base."""
    from peft import PeftModel
    print(f"  Loading FactoryBench adapter from {adapter_dir}...", flush=True)
    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)


def _generate_batch(model, tokenizer, prompts: list[str], device,
                    max_new_tokens: int, max_input_length: int):
    """Greedy-decode a batch of prompts. Stops on <|im_end|>."""
    enc = tokenizer(
        prompts,
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id)
    new_ids = gen[:, input_ids.shape[1]:]
    n_pred = [int((row != tokenizer.pad_token_id).sum().item()) for row in new_ids]
    texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    return [t.strip() for t in texts], n_pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm_id", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--levels", type=str, default=None,
                   help="Comma-separated, e.g. '1,2,3,4'")
    p.add_argument("--split", type=str, default=None,
                   choices=[None, "test", "validation"])
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap per-level samples (0 = all).")
    p.add_argument("--max_input_length", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    # Base model loading (mirrors train_factorybench.py)
    p.add_argument("--base_ckpt", type=str, default=None)
    p.add_argument("--checkpoint_type", type=str, default=None,
                   choices=[None, "shrike", "bearing"])
    p.add_argument("--tokenizer_type", type=str, default=None,
                   choices=[None, "totem", "fsq", "fsq_transformer",
                            "fsq_transformer_rope"])
    p.add_argument("--totem_ckpt", type=str, default=None)
    p.add_argument("--fsq_ckpt", type=str, default=None)
    # Trained FactoryBench adapter (empty => baseline eval)
    p.add_argument("--adapter_dir", type=str, default=None)
    args, _ = p.parse_known_args()

    # Resolve from env / SageMaker channels
    args.llm_id = args.llm_id or _env_opt("LLM_ID") or "Qwen/Qwen3-4B"
    args.data_dir = args.data_dir or os.environ.get("SM_CHANNEL_DATA") \
        or "shrike/data/factorybench"
    args.output_dir = args.output_dir or os.environ.get("SM_MODEL_DIR") \
        or "results/factorybench-eval"
    args.levels = args.levels or _env_opt("LEVELS") or "1,2,3,4"
    args.split = args.split or _env_opt("SPLIT") or "test"
    args.max_samples = args.max_samples if args.max_samples is not None \
        else int(_env_opt("MAX_SAMPLES") or 0)
    args.max_input_length = args.max_input_length or int(
        _env_opt("MAX_INPUT_LENGTH") or 16384)
    args.max_new_tokens = args.max_new_tokens or int(
        _env_opt("MAX_NEW_TOKENS") or 200)
    args.batch_size = args.batch_size or int(_env_opt("BATCH_SIZE") or 1)

    args.base_ckpt = args.base_ckpt or _env_opt("BASE_CKPT") \
        or os.environ.get("SM_CHANNEL_BASE_CKPT")
    args.checkpoint_type = args.checkpoint_type or _env_opt("CHECKPOINT_TYPE")
    args.tokenizer_type = args.tokenizer_type or _env_opt("TOKENIZER_TYPE")
    args.totem_ckpt = args.totem_ckpt or _env_opt("TOTEM_CKPT") \
        or os.environ.get("SM_CHANNEL_TOTEM_CKPT")
    args.fsq_ckpt = args.fsq_ckpt or _env_opt("FSQ_CKPT") \
        or os.environ.get("SM_CHANNEL_FSQ_CKPT")
    args.adapter_dir = args.adapter_dir or _env_opt("ADAPTER_DIR") \
        or os.environ.get("SM_CHANNEL_ADAPTER")

    args.base_ckpt = _resolve_pt(args.base_ckpt)
    args.totem_ckpt = _resolve_pt(args.totem_ckpt)
    args.fsq_ckpt = _resolve_pt(args.fsq_ckpt)
    args.adapter_dir = _resolve_adapter_dir(args.adapter_dir)

    # If the launcher mounted a prior eval's predictions, copy them into the
    # resume checkpoint dir BEFORE any level loop runs, the existing resume
    # logic will then naturally skip the already-done samples.
    prior_channel = os.environ.get("SM_CHANNEL_PRIOR_PREDICTIONS") \
        or _env_opt("PRIOR_PREDICTIONS_DIR")
    ckpt_root = Path(os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    _seed_resume_dir(prior_channel, ckpt_root)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FactoryBench Evaluation")
    print(f"  LLM:           {args.llm_id}")
    print(f"  Base ckpt:     {args.base_ckpt}")
    print(f"  Ckpt type:     {args.checkpoint_type}")
    print(f"  TS tokenizer:  {args.tokenizer_type} "
          f"(totem={args.totem_ckpt}, fsq={args.fsq_ckpt})")
    print(f"  Adapter:       {args.adapter_dir or '<none, baseline>'}")
    print(f"  Levels:        {levels}  (split={args.split})")
    print(f"  Max samples:   {args.max_samples or 'ALL'}")
    print(f"  Gen budget:    max_input={args.max_input_length}, "
          f"max_new={args.max_new_tokens}, batch={args.batch_size}")
    print("=" * 72)

    # ---------------------------------------------------------------- model
    print("\nLoading model...", flush=True)
    model, tokenizer, ts_wrapper = _build_base_model(
        base_ckpt=args.base_ckpt,
        checkpoint_type=args.checkpoint_type,
        tokenizer_type=args.tokenizer_type,
        totem_ckpt=args.totem_ckpt,
        fsq_ckpt=args.fsq_ckpt,
        llm_id=args.llm_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for generation, keeps the assistant continuation slot aligned
    # across a batch.
    tokenizer.padding_side = "left"
    # Left-TRUNCATE too: FactoryBench L2/L3 prompts can run 12-16k+ tokens
    # because they embed 4-5 options worth of feature dumps. The question and
    # the <|im_start|>assistant\n<think>...</think> cue both sit at the END
    # of the prompt. HF's default truncation_side="right" would drop them,
    # leaving the model staring at a half-finished options list, it'd just
    # keep generating more option-formatted text. Truncating from the LEFT
    # instead drops some early feature-dump rows but preserves the question
    # and the answer-time cue, which is what the model actually needs.
    tokenizer.truncation_side = "left"

    if args.adapter_dir:
        model = _stack_adapter(model, args.adapter_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  Model on {device}, dtype={next(model.parameters()).dtype}", flush=True)

    # ---------------------------------------------------------------- data
    data_dir = Path(args.data_dir)

    summary: dict = {"per_level": {}, "overall": {}}
    total_correct = 0
    total_seen = 0

    for level in levels:
        path = data_dir / f"level_{level}_{args.split}.jsonl"
        if not path.is_file():
            print(f"[skip] {path} not found", flush=True)
            continue

        samples: list[dict] = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
        if args.max_samples and args.max_samples > 0:
            samples = samples[: args.max_samples]
        print(f"\nLevel {level}: {len(samples)} samples", flush=True)

        # Build prompts using the same path as training:
        #  - if a TS wrapper is available, encode signals as <ts_*> codes
        #    (the format Shrike/BearingModel were pretrained on);
        #  - otherwise, dump raw text rows (vanilla-Qwen3 path).
        # Reusing the dataset's own builder keeps train/eval byte-aligned.
        if ts_wrapper is not None:
            from ts_prompt import build_ts_prompt
            def _builder(s):
                return build_ts_prompt(s, ts_wrapper)
        else:
            formatter = FactoryBenchDataset.__dict__["_format_chatml"]
            def _builder(s):
                return formatter(None, s)

        prompts, samples_keep = [], []
        for s in samples:
            try:
                prompt, _answer = _builder(s)
            except Exception as e:
                print(f"  [warn] formatter failed on id={s.get('id')}: {e}", flush=True)
                continue
            prompts.append(prompt)
            samples_keep.append(s)

        # Surface the prompt-length distribution so we can see at a glance
        # whether max_input_length is cutting samples off (after left-truncation,
        # cuts come from the FEATURE-DUMP head, not the question tail, but
        # losing the feature dump still hurts grounding).
        tok_lens = [len(tokenizer(p, add_special_tokens=False).input_ids)
                    for p in prompts]
        n_trunc = sum(1 for n in tok_lens if n > args.max_input_length)
        print(f"  prompt tokens: min={min(tok_lens)} med={sorted(tok_lens)[len(tok_lens)//2]} "
              f"max={max(tok_lens)}  truncated@{args.max_input_length}: "
              f"{n_trunc}/{len(prompts)}", flush=True)

        out_path = output_dir / f"level_{level}_predictions.jsonl"
        # SageMaker syncs /opt/ml/checkpoints/<...> to S3 as raw files across
        # spot reclaims, so we write predictions there too. On restart, we
        # rehydrate `done_ids` from this file and skip already-done samples,
        # so the eval RESUMES instead of starting over. Final tarball still
        # gets a copy under SM_MODEL_DIR for downstream tooling.
        ckpt_root = Path(os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints"))
        ckpt_root.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / f"level_{level}_predictions.jsonl"

        done_ids: set = set()
        if ckpt_path.is_file():
            try:
                with open(ckpt_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rid = json.loads(line).get("id")
                            if rid is not None:
                                done_ids.add(rid)
                        except json.JSONDecodeError:
                            # Last line could be a partial write from a kill;
                            # ignore it, we'll redo that one sample.
                            continue
                if done_ids:
                    print(f"  [resume] {len(done_ids)} samples already done "
                          f"from prior run, skipping", flush=True)
            except OSError:
                pass

        # Filter out done samples BEFORE the loop so progress reporting
        # reflects what's left to do.
        keep_idx = [i for i, s in enumerate(samples_keep)
                    if s.get("id") not in done_ids]
        prompts = [prompts[i] for i in keep_idx]
        samples_keep = [samples_keep[i] for i in keep_idx]
        if not prompts:
            print(f"  Level {level}: all {len(done_ids)} samples already done. "
                  f"Skipping.", flush=True)
            # Mirror checkpoint file into output_dir so the tarball is complete.
            try:
                import shutil
                shutil.copyfile(ckpt_path, out_path)
            except Exception:
                pass
            continue

        n_correct, n_seen = 0, 0
        t0 = time.time()
        # APPEND mode, new predictions go after the resumed ones.
        with open(ckpt_path, "a", buffering=1) as fout:
            for batch_start in range(0, len(prompts), args.batch_size):
                batch_prompts = prompts[batch_start: batch_start + args.batch_size]
                batch_samples = samples_keep[batch_start: batch_start + args.batch_size]
                try:
                    answers, n_pred = _generate_batch(
                        model, tokenizer, batch_prompts, device,
                        args.max_new_tokens, args.max_input_length)
                except torch.cuda.OutOfMemoryError:
                    print(f"  [OOM] batch starting at {batch_start} "
                          f", halving max_new_tokens and retrying", flush=True)
                    torch.cuda.empty_cache()
                    answers, n_pred = _generate_batch(
                        model, tokenizer, batch_prompts, device,
                        max(32, args.max_new_tokens // 2), args.max_input_length)

                for i, (s, ans, npt) in enumerate(zip(batch_samples, answers, n_pred)):
                    gt = s.get("answer")
                    rec = {
                        "idx": batch_start + i,
                        "id": s.get("id"),
                        "level": s.get("level", level),
                        "template_id": s.get("template_id"),
                        "template_type": s.get("template_type"),
                        "question_type": s.get("question_type"),
                        "question": s.get("question"),
                        "options": s.get("options", {}),
                        "ground_truth": gt,
                        "answer": ans,
                        "n_pred_tokens": npt,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    n_seen += 1
                    if gt is not None and ans.strip() == str(gt).strip():
                        n_correct += 1

                if batch_start % (50 * args.batch_size) == 0:
                    rate = n_seen / max(1e-9, time.time() - t0)
                    print(f"  [L{level}] {n_seen}/{len(prompts)} "
                          f"acc={n_correct/max(1,n_seen):.3f} "
                          f"({rate:.1f} samp/s)", flush=True)

        acc = n_correct / max(1, n_seen)
        # Count includes resumed samples for accurate per-level totals.
        # (n_correct only reflects this run's newly-decoded samples; resumed
        # ones already had their pred recorded but we don't re-tally them.)
        summary["per_level"][f"level_{level}"] = {
            "n": n_seen, "n_correct": n_correct, "exact_match_acc": acc,
            "resumed_from_ckpt": len(done_ids),
        }
        total_correct += n_correct
        total_seen += n_seen
        # Copy the resumable checkpoint file into the final tarball location.
        try:
            import shutil
            shutil.copyfile(ckpt_path, out_path)
        except Exception as e:
            print(f"  [warn] failed to copy ckpt -> output: {e}", flush=True)
        print(f"\n  Level {level}: {n_correct} new + {len(done_ids)} resumed = "
              f"{n_correct + len(done_ids)} total predictions, written to {out_path}",
              flush=True)

    if total_seen:
        summary["overall"] = {
            "n": total_seen,
            "n_correct": total_correct,
            "exact_match_acc": total_correct / total_seen,
        }
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
