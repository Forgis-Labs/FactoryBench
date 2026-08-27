"""Fine-tune Qwen3 (or a Shrike/BearingModel checkpoint) on FactoryBench with DoRA.

Two modes:

  1. Plain HF base model (default):
       --llm_id Qwen/Qwen3-4B
     Loads the LLM from HuggingFace / a local dir, attaches a fresh DoRA,
     and trains on FactoryBench text QA.

  2. Pretrained Shrike / BearingModel checkpoint:
       --base_ckpt /opt/ml/input/data/base_ckpt/best_model.pt \
       --checkpoint_type {shrike,bearing} \
       --tokenizer_type {totem,fsq,fsq_transformer,fsq_transformer_rope} \
       --totem_ckpt /opt/ml/input/data/ts_tok/totem_clean.pt   (TOTEM-style)
       --fsq_ckpt   /opt/ml/input/data/ts_tok/fsq_625.pt        (FSQ-style)
     Loads the checkpoint via the matching loader (which restores Qwen3 +
     extended <ts_*> vocab + the original DoRA r=32), MERGES the original
     DoRA into the base weights with `merge_and_unload`, then attaches a
     FRESH DoRA on top for FactoryBench finetuning.

The dataset / loss path is identical to the plain mode, FactoryBench
prompts are text-only, the extra <ts_*> tokens in the vocabulary are
unused at train time and just ride along in the embedding table.

Usage:
    uv run python train_factorybench.py \
        --llm_id Qwen/Qwen3-4B \
        --data_dir shrike/data/factorybench \
        --levels 1 2 3

    # DoRA-on-DoRA from a bearing checkpoint
    uv run python train_factorybench.py \
        --base_ckpt /path/to/bearing_r32_qwen3_4b.pt \
        --checkpoint_type bearing \
        --tokenizer_type totem \
        --totem_ckpt /path/to/totem_clean.pt \
        --llm_id Qwen/Qwen3-4B \
        --data_dir shrike/data/factorybench \
        --levels 1 2 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class FactoryBenchDataset(Dataset):
    """FactoryBench JSONL -> ChatML formatted training samples."""

    def __init__(self, jsonl_paths: list[str], tokenizer, max_length: int = 4096,
                 max_samples: int = 0, wrapper=None, max_channels: int = 64):
        """If ``wrapper`` is a Shrike or BearingModel instance, each sample's
        time-series rows are pushed through ``wrapper.tokenize_ts`` and encoded
        as ``<ts_start> <ts_*> <ts_end>`` blocks per channel, the canonical
        Shrike wire format. Otherwise we fall back to dumping the raw text
        rows in the prompt (the vanilla-Qwen3 path).
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.wrapper = wrapper
        self.max_channels = max_channels

        # Left-truncate: when a prompt+answer exceeds max_length, drop early
        # feature-dump rows rather than the answer at the tail. Without this,
        # oversize samples silently teach the model to predict NOTHING.
        try:
            tokenizer.truncation_side = "left"
        except Exception:
            pass

        for path in jsonl_paths:
            with open(path) as f:
                for line in f:
                    self.samples.append(json.loads(line))

        # Smoke-test knob, cap dataset BEFORE pre-tokenization so we don't
        # pay the tokenization cost on 40k+ samples just to do a 20-step run.
        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

        # Pre-tokenize all samples
        self.input_ids = []
        self.prompt_lengths = []

        if wrapper is not None:
            print(f"  Pre-tokenizing {len(self.samples)} samples "
                  f"(TS-token mode, max_channels={max_channels})...",
                  end=" ", flush=True)
        else:
            print(f"  Pre-tokenizing {len(self.samples)} samples (text-only)...",
                  end=" ", flush=True)

        for sample in self.samples:
            prompt, answer = self._build_prompt(sample)
            full_text = prompt + answer

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 truncation=True, max_length=max_length).input_ids

            self.input_ids.append(full_ids)
            self.prompt_lengths.append(len(prompt_ids))

        avg_len = np.mean([len(ids) for ids in self.input_ids])
        max_len = max(len(ids) for ids in self.input_ids)
        print(f"done (avg={avg_len:.0f}, max={max_len} tokens)")

    def _build_prompt(self, sample: dict) -> tuple[str, str]:
        """Dispatch to the TS-tokenized builder if we have a wrapper, else
        fall back to the original text-dump format."""
        if self.wrapper is not None:
            from ts_prompt import build_ts_prompt
            return build_ts_prompt(sample, self.wrapper,
                                   max_channels=self.max_channels)
        return self._format_chatml(sample)

    def _format_chatml(self, sample: dict) -> tuple[str, str]:
        """Format a FactoryBench sample as ChatML prompt + answer.

        MUST stay byte-identical to ``build_prompt`` in
        ``finetuning/eval_factorybench.py``, any drift retrains the model on
        a distribution that diverges from inference. The exact wire format is:

            <|im_start|>user
            Feature mapping: <first-10-acronyms>... (N total)

            Time series data:
            <rows>

            <question>[\n\nOptions:\n  k: v\n  ...]<|im_end|>
            <|im_start|>assistant
            <think>

            </think>

            <answer><|im_end|>

        The empty <think>...</think> block is the cue Qwen3 was trained to
        recognize as "skip reasoning, emit answer directly". Including it in
        training teaches the model to honour it at inference (where it is
        prefilled and the model continues from after the second \\n\\n).
        """
        question = sample["question"]
        context = sample.get("context", {}) or {}
        ts_format = context.get("time_series_format", {}) or {}
        ts_rows = context.get("time_series", []) or []
        acronym_map = ts_format.get("acronym_mapping", {}) or {}
        options = sample.get("options", {}) or {}

        parts = []
        if acronym_map:
            mapping_str = ", ".join(
                f"{k}={v}" for k, v in list(acronym_map.items())[:10]
            )
            if len(acronym_map) > 10:
                mapping_str += f"... ({len(acronym_map)} total)"
            parts.append(f"Feature mapping: {mapping_str}")

        if isinstance(ts_rows, list) and ts_rows:
            ts_str = "\n".join(ts_rows[:100])
            if len(ts_rows) > 100:
                ts_str += f"\n... ({len(ts_rows)} timesteps total)"
            parts.append(f"Time series data:\n{ts_str}")

        if options:
            # NB: do NOT sort or truncate, eval iterates raw dict order with
            # full values, and divergence here is a silent train/eval skew.
            opts_str = "\n".join(f"  {k}: {v}" for k, v in options.items())
            question = f"{question}\n\nOptions:\n{opts_str}"

        context_str = "\n\n".join(parts)
        user_content = f"{context_str}\n\n{question}" if context_str else question

        prompt = (
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        answer = f"{sample['answer']}<|im_end|>"

        return prompt, answer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "prompt_length": self.prompt_lengths[idx],
        }


def compute_loss(batch, model, tokenizer, device, max_length):
    """Compute causal LM loss from pre-tokenized batch."""
    pad_id = tokenizer.pad_token_id or 0
    max_len = min(max(len(s["input_ids"]) for s in batch), max_length)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, sample in enumerate(batch):
        ids = sample["input_ids"][:max_len]
        seq_len = len(ids)
        pl = sample["prompt_length"]

        input_ids[i:seq_len] = torch.tensor(ids, dtype=torch.long)
        attn_mask[i:seq_len] = 1
        if pl < seq_len:
            labels[i, pl:seq_len] = input_ids[i, pl:seq_len]

    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    labels = labels.to(device)

    return model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss


def _load_base_with_shrike_loader(
    base_ckpt: str,
    checkpoint_type: str,
    llm_id: str,
    tokenizer_type: str,
    totem_ckpt: str | None,
    fsq_ckpt: str | None):
    """Load a Shrike/BearingModel checkpoint and return (llm, tokenizer).

    The returned ``llm`` is a HF Qwen3ForCausalLM with the original DoRA
    already MERGED into the base weights (so a fresh DoRA can be stacked
    on top by the caller). The returned ``tokenizer`` is the LLM's text
    tokenizer extended with <ts_*> special tokens, kept so the embedding
    table sizes match the loaded weights. FactoryBench prompts never emit
    those token IDs, so the extra rows just ride along unused.
    """
    # Vendored shrike package (copied into source_dir at build time)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    ckpt_type = checkpoint_type.lower()
    if ckpt_type == "bearing":
        from shrike.model.bearing import BearingModel
        # BearingModel always uses TOTEM as its TS tokenizer.
        if not totem_ckpt:
            raise ValueError("--totem_ckpt is required for --checkpoint_type bearing")
        wrapper = BearingModel.from_pretrained(
            checkpoint_path=base_ckpt,
            totem_ckpt=totem_ckpt,
            llm_id=llm_id,
            device="cpu",     # accelerator moves it later
        )
    elif ckpt_type == "shrike":
        from shrike.model.shrike import Shrike
        kwargs: dict = {
            "checkpoint_path": base_ckpt,
            "llm_id": llm_id,
            "device": "cpu",
            "tokenizer_type": tokenizer_type,
        }
        if totem_ckpt:
            kwargs["totem_ckpt"] = totem_ckpt
        if fsq_ckpt:
            kwargs["fsq_ckpt"] = fsq_ckpt
        wrapper = Shrike.from_pretrained(**kwargs)
    else:
        raise ValueError(
            f"--checkpoint_type must be 'shrike' or 'bearing', got {checkpoint_type!r}"
        )

    # Merge the original DoRA into the base so we can stack a fresh adapter
    # on top. merge_and_unload returns the plain Qwen3ForCausalLM (no PEFT
    # wrapper) with adapter deltas folded into the base weights, the
    # standard PEFT pattern for "use this finetune as the new base".
    print("  Merging existing DoRA into base LLM weights...")
    merged_llm = wrapper.llm.merge_and_unload()
    tokenizer = wrapper.tokenizer

    # KEEP the wrapper around, its ts_tokenizer (TOTEM / FSQ-Transformer)
    # is what FactoryBenchDataset uses to encode signals as <ts_*> codes.
    # Free the now-redundant LLM inside the wrapper; merged_llm replaces it.
    wrapper.llm = None
    import gc; gc.collect()

    return merged_llm, tokenizer, wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--llm_id", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--use_dora", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--warmup_frac", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training-set size for smoke runs (default: all).")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Cap validation-set size for smoke runs (default: all).")
    # DoRA-on-DoRA: Shrike/BearingModel checkpoint loading.
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="Path (or SageMaker channel dir) holding a "
                             "Shrike/BearingModel .pt checkpoint. If set, "
                             "--checkpoint_type and the matching TS tokenizer "
                             "ckpt must also be provided.")
    parser.add_argument("--checkpoint_type", type=str, default=None,
                        choices=[None, "shrike", "bearing"])
    parser.add_argument("--tokenizer_type", type=str, default=None,
                        choices=[None, "totem", "fsq", "fsq_transformer",
                                 "fsq_transformer_rope"])
    parser.add_argument("--totem_ckpt", type=str, default=None)
    parser.add_argument("--fsq_ckpt", type=str, default=None)
    args, _ = parser.parse_known_args()

    # SageMaker passes hyperparameters as SM_HP_* env vars (uppercase)
    def _env(key, default):
        return os.environ.get(f"SM_HP_{key}", os.environ.get(key, str(default)))

    def _env_opt(key):
        """Like _env but returns None for missing / empty / literal 'None'."""
        v = os.environ.get(f"SM_HP_{key}", os.environ.get(key))
        if v is None or v == "" or v.lower() == "none":
            return None
        return v

    args.llm_id = args.llm_id or _env("LLM_ID", "Qwen/Qwen3-4B")
    args.data_dir = args.data_dir or os.environ.get("SM_CHANNEL_DATA", "shrike/data/factorybench")
    args.output_dir = args.output_dir or os.environ.get("SM_MODEL_DIR", "results/factorybench")
    args.lora_r = args.lora_r or int(_env("LORA_R", 32))
    args.lora_alpha = args.lora_alpha or int(_env("LORA_ALPHA", 64))
    args.use_dora = (args.use_dora or _env("USE_DORA", "true")).lower() == "true"
    args.batch_size = args.batch_size or int(_env("BATCH_SIZE", 2))
    args.grad_accum = args.grad_accum or int(_env("GRAD_ACCUM", 16))
    args.lr = args.lr or float(_env("LR", 2e-5))
    args.epochs = args.epochs or int(_env("EPOCHS", 5))
    args.patience = args.patience or int(_env("PATIENCE", 2))
    args.max_length = args.max_length or int(_env("MAX_LENGTH", 4096))
    args.warmup_frac = args.warmup_frac or float(_env("WARMUP_FRAC", 0.10))
    args.max_train_samples = args.max_train_samples if args.max_train_samples is not None \
        else int(_env("MAX_TRAIN_SAMPLES", 0))
    args.max_val_samples = args.max_val_samples if args.max_val_samples is not None \
        else int(_env("MAX_VAL_SAMPLES", 0))

    # DoRA-on-DoRA paths (all optional; resolved from SageMaker channels if present)
    args.base_ckpt = args.base_ckpt or _env_opt("BASE_CKPT") \
        or os.environ.get("SM_CHANNEL_BASE_CKPT")
    args.checkpoint_type = args.checkpoint_type or _env_opt("CHECKPOINT_TYPE")
    args.tokenizer_type = args.tokenizer_type or _env_opt("TOKENIZER_TYPE")
    args.totem_ckpt = args.totem_ckpt or _env_opt("TOTEM_CKPT") \
        or os.environ.get("SM_CHANNEL_TOTEM_CKPT")
    args.fsq_ckpt = args.fsq_ckpt or _env_opt("FSQ_CKPT") \
        or os.environ.get("SM_CHANNEL_FSQ_CKPT")

    # SageMaker channels mount a directory, not a file, if these point to a
    # directory, pick the .pt inside (preferring filenames containing "best").
    def _resolve_pt(path: str | None) -> str | None:
        if not path:
            return None
        p = Path(path)
        if p.is_file():
            return str(p)
        if p.is_dir():
            pts = sorted(p.rglob("*.pt"))
            if not pts:
                raise FileNotFoundError(f"No .pt file found under {path}")
            best = [x for x in pts if "best" in x.name.lower()]
            return str(best[0] if best else pts[0])
        raise FileNotFoundError(f"Path does not exist: {path}")

    args.base_ckpt = _resolve_pt(args.base_ckpt)
    args.totem_ckpt = _resolve_pt(args.totem_ckpt)
    args.fsq_ckpt = _resolve_pt(args.fsq_ckpt)

    # Parse levels (can be "1,2,3" string or list)
    if args.levels is None:
        args.levels = _env("LEVELS", "1,2,3")
    if isinstance(args.levels, str):
        args.levels = [int(x) for x in args.levels.split(",")]

    # Load YAML config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for section in cfg.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if hasattr(args, k) and v is not None:
                        setattr(args, k, v)
        if "train_levels" in cfg.get("data", {}):
            args.levels = cfg["data"]["train_levels"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FactoryBench Fine-Tuning")
    print(f"  Model: {args.llm_id}")
    if args.base_ckpt:
        print(f"  Base ckpt:    {args.base_ckpt}")
        print(f"  Ckpt type:    {args.checkpoint_type}")
        print(f"  TS tokenizer: {args.tokenizer_type} "
              f"(totem={args.totem_ckpt}, fsq={args.fsq_ckpt})")
    print(f"  Levels: {args.levels}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print("\nLoading model...")
    if args.base_ckpt:
        # DoRA-on-DoRA path: load wrapper, merge existing DoRA, return the
        # plain Qwen3 + extended-vocab tokenizer ready for a fresh adapter.
        if not args.checkpoint_type:
            raise ValueError("--checkpoint_type is required when --base_ckpt is set")
        if not args.tokenizer_type:
            # BearingModel is implicitly TOTEM, so default for convenience.
            args.tokenizer_type = "totem" if args.checkpoint_type == "bearing" else None
            if not args.tokenizer_type:
                raise ValueError("--tokenizer_type is required for shrike checkpoints")

        model, tokenizer, ts_wrapper = _load_base_with_shrike_loader(
            base_ckpt=args.base_ckpt,
            checkpoint_type=args.checkpoint_type,
            llm_id=args.llm_id,
            tokenizer_type=args.tokenizer_type,
            totem_ckpt=args.totem_ckpt,
            fsq_ckpt=args.fsq_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        # Vanilla path: load fresh from HF / local LLM dir.
        ts_wrapper = None
        tokenizer = AutoTokenizer.from_pretrained(args.llm_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2")

    # alpha defaults to 2*r if the caller didn't pass one explicitly (the
    # standard LoRA convention) but a CLI/env override now actually wins.
    effective_alpha = args.lora_alpha if args.lora_alpha else args.lora_r * 2
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=effective_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        use_dora=args.use_dora)
    print(f"  LoRA: r={args.lora_r} alpha={effective_alpha} use_dora={args.use_dora}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Load data
    print("\nLoading data...")
    data_dir = Path(args.data_dir)
    train_paths = [str(data_dir / f"level_{l}_train.jsonl") for l in args.levels]
    val_paths = [str(data_dir / f"level_{l}_validation.jsonl") for l in args.levels]

    train_ds = FactoryBenchDataset(train_paths, tokenizer, args.max_length,
                                   max_samples=args.max_train_samples,
                                   wrapper=ts_wrapper)
    val_ds = FactoryBenchDataset(val_paths, tokenizer, args.max_length,
                                 max_samples=args.max_val_samples,
                                 wrapper=ts_wrapper)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Accelerate
    from accelerate import Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16")
    device = accelerator.device

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: b, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: b, num_workers=4, pin_memory=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: min(1.0, s / warmup_steps) if s < warmup_steps
                            else max(0.1, 1.0 - (s - warmup_steps) / max(1, total_steps - warmup_steps)))

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler)
    val_loader = accelerator.prepare(val_loader)

    is_main = accelerator.is_main_process
    # Used only for state_dict() saves and final reporting, NEVER call its
    # forward directly inside the training loop, that bypasses DDP/FSDP and
    # mixed-precision wrappers. Use ``model`` (the prepared one) for forward.
    raw_model = accelerator.unwrap_model(model)

    # Save the (possibly vocab-extended) tokenizer up front so eval can
    # rebuild the embedding table without rerunning the Shrike loader.
    if is_main:
        try:
            tokenizer.save_pretrained(str(output_dir / "tokenizer"))
        except Exception as e:
            print(f"  [warn] tokenizer.save_pretrained failed: {e}", flush=True)

    # Training loop
    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            # DDP correctness: every rank MUST execute the same NCCL collectives
            # in the same order on every iteration. A `continue` here (e.g. on
            # OOM or NaN) would skip an all-reduce on one rank while the others
            # block on it, that's the NCCL watchdog timeout you'll see in the
            # logs ("WorkNCCL ... ran for 600099 ms before timing out").
            #
            # So instead of skipping the iteration, we let it run with a zero
            # "fake loss" anchored to a trainable parameter. The graph carries
            # zero gradient to every param, accelerator.backward fires
            # normally, the optimizer step is a no-op, ranks stay in lock-step.
            with accelerator.accumulate(model):
                bad_batch = False
                try:
                    loss = compute_loss(batch, model, tokenizer, device, args.max_length)
                    if torch.isnan(loss) or torch.isinf(loss):
                        bad_batch = True
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    bad_batch = True
                    loss = None

                if bad_batch:
                    # Build a zero loss that IS connected to trainable params
                    # so DDP sees a real gradient graph and runs all-reduce.
                    zero = None
                    for p in model.parameters():
                        if p.requires_grad:
                            zero = (p.sum() * 0.0)
                            break
                    loss = zero if zero is not None else torch.zeros(
                        (), device=device, requires_grad=True)
                    if is_main:
                        print(f"    [bad-batch] {batch_idx} masked to zero loss "
                              f"(preserves DDP sync)", flush=True)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running += loss.item()
            n_batches += 1

            if is_main and batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [{epoch}/{args.epochs}] batch {batch_idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f} lr={lr:.2e}", flush=True)

            # Mid-epoch checkpoint (synced to S3 by SageMaker)
            if is_main and batch_idx > 0 and batch_idx % 500 == 0:
                ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints"))
                job_name = os.environ.get("SAGEMAKER_JOB_NAME",
                           os.environ.get("SM_HP_SAGEMAKER_JOB_NAME", "local"))
                mid_dir = ckpt_dir / job_name
                mid_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "train_loss": running / n_batches,
                }, mid_dir / "mid_epoch.pt")
                print(f"    [checkpoint] batch {batch_idx} saved", flush=True)

        avg_train = running / max(n_batches, 1)

        # Validate (use prepared ``model`` for forward; autocast / DDP wrappers
        # apply identically to train and eval that way).
        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    val_item = compute_loss(batch, model, tokenizer, device, args.max_length).item()
                    # NaN guard: ``x != x`` is true iff x is NaN.
                    if val_item == val_item:
                        v_loss += val_item
                        v_n += 1
                except Exception:
                    continue
        avg_val = v_loss / max(v_n, 1)

        if is_main:
            print(f"  Epoch {epoch}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"(best={best_val:.4f}, patience={patience_left})")

            epoch_state = {
                "model_state": raw_model.state_dict(),
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
            }

            # last_model.pt, overwritten each epoch (lightweight resume target,
            # reflects most recent weights, used for crash recovery).
            torch.save(epoch_state, output_dir / "last_model.pt")

            ckpt_dir = Path(os.environ.get("CHECKPOINT_DIR", "/opt/ml/checkpoints"))
            job_name = os.environ.get("SAGEMAKER_JOB_NAME",
                       os.environ.get("SM_HP_SAGEMAKER_JOB_NAME", "local"))
            ckpt_job_dir = ckpt_dir / job_name
            ckpt_job_dir.mkdir(parents=True, exist_ok=True)

            # Save every epoch with val-loss in the filename so eval can
            # pick a specific snapshot by name.
            epoch_name = f"epoch{epoch}_val{avg_val:.4f}.pt"
            torch.save(epoch_state, ckpt_job_dir / epoch_name)
            print(f"  -> Saved {epoch_name} to S3 checkpoint dir")

            if avg_val < best_val - 1e-4:
                best_val = avg_val
                patience_left = args.patience
                torch.save(epoch_state, output_dir / "best_model.pt")
                torch.save(epoch_state, ckpt_job_dir / "best_model.pt")
                # PEFT-style adapter dump, saved to BOTH the model output dir
                # (gets tarred into model.tar.gz) AND the checkpoint dir
                # (synced to S3 as raw files, so eval can mount it as a
                # SageMaker channel without having to untar anything).
                try:
                    raw_model.save_pretrained(str(output_dir / "adapter"))
                    raw_model.save_pretrained(str(ckpt_job_dir / "adapter"))
                    # Tokenizer too, eval needs the extended <ts_*> vocab.
                    tokenizer.save_pretrained(str(ckpt_job_dir / "tokenizer"))
                except Exception as e:
                    print(f"  [warn] save_pretrained(adapter) failed: {e}", flush=True)
                print(f"  -> New best (val={avg_val:.4f})")
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

    if is_main:
        print(f"\nDone. Best val loss: {best_val:.4f}")
        print(f"Checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
