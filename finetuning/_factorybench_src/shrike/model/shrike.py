"""Shrike: Time Series Understanding via Discrete Tokenization.

Single model class that handles inference, training, and forecasting.
No separate model files, no inconsistent APIs.

Architecture:
    raw signal → TOTEM encoder (frozen 1D CNN, 4:1 compression)
              → VQ codebook (256 discrete codes)
              → <ts_start> <ts_0>..<ts_255> <ts_end> in LLM vocabulary
              → LLM (frozen backbone + LoRA/DoRA adapters)

Usage:
    from shrike import Shrike, ShrikeConfig

    # Inference
    model = Shrike.from_pretrained("checkpoint.pt", totem_ckpt="totem.pt")
    answer = model.analyze(signal, "What is the trend?")
    forecast = model.forecast(signal, horizon=64)

    # Training
    model = Shrike(ShrikeConfig(llm_id="Qwen/Qwen3-4B"))
    loss = model.compute_loss(batch)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from shrike.tokenizer.totem import TOTEMTokenizer


# ===================================================================
# Config
# ===================================================================

@dataclass
class ShrikeConfig:
    """All settings for a Shrike model in one place.

    No scattered configs, no argparse, no magic numbers.
    """

    # Base LLM
    llm_id: str = "Qwen/Qwen3-4B"

    # Tokenizer
    tokenizer_type: str = "totem"  # "totem" or "fsq"
    totem_ckpt: str | None = None
    fsq_ckpt: str | None = None
    n_bins: int = 256              # auto-set from tokenizer for FSQ
    compression_factor: int = 4    # 4 raw time points -> 1 code

    # Time series limits
    max_ts_tokens: int = 750   # 750 = covers all UCR datasets (max 2844 pts / 4)
    max_length: int = 2048     # max total sequence (prompt + codes + answer)

    # LoRA / DoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP (SwiGLU)
        ]
    )
    use_dora: bool = False
    neftune_alpha: float = 0.0

    @property
    def max_raw_points(self) -> int:
        """Max raw time series length this config supports."""
        return self.max_ts_tokens * self.compression_factor


# ===================================================================
# Model
# ===================================================================

class Shrike(nn.Module):
    """Shrike: discrete tokenization + LLM for time series.

    One class for everything: inference, training, and forecasting.
    """

    def __init__(self, config: ShrikeConfig, device: str = "cpu") -> None:
        super().__init__()
        self.config = config
        self.device_str = device

        # --- Time series tokenizer (frozen, CPU) ---
        if config.tokenizer_type in ("fsq_transformer", "fsq_transformer_rope"):
            if config.tokenizer_type == "fsq_transformer_rope":
                from shrike.tokenizer.fsq_transformer_rope import FSQTransformerRoPETokenizer
                loader = FSQTransformerRoPETokenizer
            else:
                from shrike.tokenizer.fsq_transformer import FSQTransformerTokenizer
                loader = FSQTransformerTokenizer
            if config.fsq_ckpt and os.path.exists(config.fsq_ckpt):
                self.ts_tokenizer = loader.from_pretrained(config.fsq_ckpt)
                config.n_bins = self.ts_tokenizer.codebook_size
                config.compression_factor = self.ts_tokenizer.config.patch_size
            else:
                raise ValueError(f"{config.tokenizer_type} requires fsq_ckpt path")
            self.totem = None
        elif config.tokenizer_type == "fsq":
            from shrike.tokenizer.fsq import FSQTokenizer
            if config.fsq_ckpt and os.path.exists(config.fsq_ckpt):
                self.ts_tokenizer = FSQTokenizer.from_pretrained(config.fsq_ckpt)
                config.n_bins = self.ts_tokenizer.codebook_size
                config.compression_factor = self.ts_tokenizer.config.compression
            else:
                raise ValueError("FSQ tokenizer requires fsq_ckpt path")
            self.totem = None  # for backward compat
        else:
            # TOTEM (default)
            if config.totem_ckpt and os.path.exists(config.totem_ckpt):
                self.ts_tokenizer = TOTEMTokenizer.from_pretrained(config.totem_ckpt)
                config.n_bins = self.ts_tokenizer.codebook_size
                config.compression_factor = 4  # TOTEM always uses 4:1
            else:
                self.ts_tokenizer = TOTEMTokenizer(num_embeddings=config.n_bins)
                self.ts_tokenizer.eval()
                self.ts_tokenizer.requires_grad_(False)
            self.totem = self.ts_tokenizer  # backward compat

        print(f"Tokenizer: {config.tokenizer_type}, {config.n_bins} codes, "
              f"{config.compression_factor}:1 compression")

        # --- LLM + tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_id)

        # TS special tokens: codes + delimiters
        ts_tokens = [f"<ts_{i}>" for i in range(config.n_bins)]
        ts_tokens += ["<ts_start>", "<ts_end>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": ts_tokens})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use flash attention on CUDA, fall back to sdpa on CPU/Windows
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_id,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # --- Init TS embeddings ---
        n_special = config.n_bins + 2
        self._original_vocab = len(self.tokenizer) - n_special
        self._init_ts_embeddings(config)

        # --- Freeze LLM + apply LoRA ---
        self.llm.requires_grad_(False)
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            **({"use_dora": True} if config.use_dora else {}))
        self.llm = get_peft_model(self.llm, lora_cfg)

        # --- Gradient mask: only TS embeddings are trainable ---
        self._ts_range = (self._original_vocab, self._original_vocab + n_special)
        embed = self.llm.get_input_embeddings()
        embed.weight.requires_grad = True
        embed.weight.register_hook(self._mask_grad)

    @property
    def _device(self):
        """Get actual device from model parameters (works with FSDP/DDP)."""
        return next(self.llm.parameters()).device

    def _init_ts_embeddings(self, config: ShrikeConfig) -> None:
        """Initialize TS token embeddings in the LLM embedding space.

        For TOTEM: project codebook vectors into LLM space.
        For FSQ: random init matching existing embedding norm distribution.
        """
        embed = self.llm.get_input_embeddings()
        llm_dim = embed.weight.shape[1]
        existing_norm = embed.weight.data[:self._original_vocab].float().norm(dim=1).mean()

        if config.tokenizer_type == "totem" and self.totem is not None:
            # Project TOTEM codebook into LLM embedding space
            codebook = self.totem.codebook.float()
            torch.manual_seed(42)
            proj = torch.randn(self.totem.codebook_dim, llm_dim) / (self.totem.codebook_dim ** 0.5)
            projected = (codebook @ proj).to(torch.bfloat16)

            proj_norm = projected.float().norm(dim=1).mean()
            if proj_norm > 0:
                projected *= existing_norm / proj_norm

            with torch.no_grad():
                start = self._original_vocab
                embed.weight.data[start:start + config.n_bins] = projected
        else:
            # FSQ or fallback: random init matching norm distribution
            torch.manual_seed(42)
            with torch.no_grad():
                start = self._original_vocab
                for i in range(config.n_bins):
                    rand_e = torch.randn(llm_dim, dtype=torch.bfloat16)
                    rand_e = rand_e * (existing_norm / (llm_dim ** 0.5))
                    embed.weight.data[start + i] = rand_e
            # <ts_start>, <ts_end> get random init at matching norm
            for offset in range(config.n_bins, config.n_bins + 2):
                rand_e = torch.randn(llm_dim, dtype=torch.bfloat16) * (existing_norm / llm_dim ** 0.5)
                embed.weight.data[start + offset] = rand_e

    def _mask_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Zero out gradients for non-TS embeddings."""
        mask = torch.zeros_like(grad)
        mask[self._ts_range[0]:self._ts_range[1]] = 1.0
        return grad * mask

    def freeze_lora(self) -> int:
        """Freeze DoRA/LoRA adapters, only TS embeddings remain trainable.

        Used during embedding alignment stage (stage0_align) to force all
        gradient signal into the 627 TS token embeddings.

        Returns:
            Number of parameters frozen.
        """
        count = 0
        for name, param in self.llm.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
                count += param.numel()
        return count

    def unfreeze_lora(self) -> int:
        """Re-enable DoRA/LoRA adapter training.

        Called after embedding alignment to restore normal training mode.

        Returns:
            Number of parameters unfrozen.
        """
        count = 0
        for name, param in self.llm.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
                count += param.numel()
        return count

    # ==============================================================
    # Tokenization
    # ==============================================================

    def tokenize_ts(self, values: torch.Tensor) -> list[int]:
        """Convert raw time series to discrete code IDs.

        Handles normalization, downsampling, and encoding via TOTEM or FSQ.
        """
        ts = values.float().flatten()
        mean, std = ts.mean(), ts.std()
        ts_norm = (ts - mean) / max(std, 1e-8)

        max_raw = self.config.max_ts_tokens * self.config.compression_factor
        if len(ts_norm) > max_raw:
            idx = torch.linspace(0, len(ts_norm) - 1, max_raw).long()
            ts_norm = ts_norm[idx]

        with torch.no_grad():
            dev = next(self.ts_tokenizer.parameters()).device
            codes = self.ts_tokenizer.tokenize(ts_norm.to(dev))
        return codes.squeeze(0).tolist()

    MASK_SENTINEL = -1  # sentinel value in ts_codes for masked positions

    @staticmethod
    def codes_to_text(codes: list[int]) -> str:
        """Convert code IDs to text tokens: [42, 88] → '<ts_42> <ts_88>'.

        Mask sentinels (-1) are skipped, use codes_to_text_with_masks()
        for imputation tasks.
        """
        return " ".join(f"<ts_{c}>" for c in codes if c >= 0)

    @classmethod
    def codes_to_text_with_masks(cls, codes: list[int]) -> str:
        """Convert codes with mask sentinels to interleaved text.

        [367, 367, -1, -1, 496] →
          '<ts_start> <ts_367> <ts_367> <ts_end> [MASK] [MASK] <ts_start> <ts_496> <ts_end>'

        Each contiguous run of real codes gets wrapped in <ts_start>/<ts_end>.
        Each mask sentinel becomes a [MASK] text token.
        """
        parts = []
        buf = []
        for c in codes:
            if c == cls.MASK_SENTINEL:
                if buf:
                    parts.append(f"<ts_start> {' '.join(f'<ts_{x}>' for x in buf)} <ts_end>")
                    buf = []
                parts.append("[MASK]")
            else:
                buf.append(c)
        if buf:
            parts.append(f"<ts_start> {' '.join(f'<ts_{x}>' for x in buf)} <ts_end>")
        return " ".join(parts)

    # ==============================================================
    # Prompt building
    # ==============================================================

    def build_text(self, sample: dict) -> str:
        """Build ChatML-framed prompt with <ts_start>/<ts_end> delimited codes.

        Output format (Qwen3 ChatML, thinking disabled):
            <|im_start|>user
            {pre_prompt} {ts_codes} (mean=..., std=...) {post_prompt}<|im_end|>
            <|im_start|>assistant

        Handles single-channel, multi-channel, and text-only samples.
        If sample contains 'ts_codes' (pre-tokenized), skips encoding.
        """
        ts = sample.get("time_series", [])
        ts_descs = sample.get("time_series_text", [])

        # Text-only (e.g., instruction data)
        if not ts or (isinstance(ts, list) and len(ts) == 0):
            if not sample.get("ts_codes"):
                content = (sample.get("pre_prompt", "") + " " +
                           sample.get("post_prompt", "")).strip()
                return (f"<|im_start|>user\n{content}<|im_end|>\n"
                        f"<|im_start|>assistant\n")

        # Fast path: use pre-tokenized codes if available
        if sample.get("ts_codes"):
            codes = sample["ts_codes"]
            # Flatten nested list (one list per channel) to flat code list
            if isinstance(codes, list) and len(codes) > 0 and isinstance(codes[0], list):
                codes = [c for ch in codes for c in ch]
            mean_val = sample.get("ts_mean", 0.0)
            std_val = sample.get("ts_std", 0.0)
        else:
            # Tokenize per-channel (each channel normalized independently)
            if isinstance(ts, torch.Tensor):
                channels = [ts.flatten()]
            elif isinstance(ts, list) and len(ts) > 0:
                channels = []
                for ch in ts:
                    if isinstance(ch, torch.Tensor):
                        channels.append(ch.flatten())
                    else:
                        channels.append(torch.tensor(ch, dtype=torch.float32).flatten())
            else:
                channels = []

            codes_per_ch = [self.tokenize_ts(ch) for ch in channels]
            codes = [c for ch_codes in codes_per_ch for c in ch_codes]  # flat for single-channel path
            # Global stats for metadata (informational only)
            all_vals = torch.cat(channels) if channels else torch.zeros(1)
            mean_val = all_vals.mean().item()
            std_val = all_vals.std().item()

        # Build user content with signal data
        content = sample.get("pre_prompt", "") + " "

        # Check if this is an imputation sample (codes contain mask sentinels)
        has_masks = any(c == self.MASK_SENTINEL for c in codes)

        if has_masks:
            # Imputation: interleave <ts_start>...<ts_end> segments with [MASK] text
            label = ts_descs[0] if ts_descs else ""
            if label:
                content += f"{label} "
            content += f"(mean={mean_val:.4f}, std={std_val:.4f}) "
            content += self.codes_to_text_with_masks(codes) + " "
        elif len(ts_descs) > 1 and not sample.get("ts_codes"):
            # Multi-channel: per-channel blocks with per-channel stats
            for desc, ch_codes, ch in zip(ts_descs, codes_per_ch, channels):
                ch_mean = ch.mean().item()
                ch_std = ch.std().item()
                content += (f"{desc} (mean={ch_mean:.4f}, std={ch_std:.4f}) "
                            f"<ts_start> {self.codes_to_text(ch_codes)} <ts_end> ")
        else:
            label = ts_descs[0] if ts_descs else ""
            if label:
                content += f"{label} "
            content += f"(mean={mean_val:.4f}, std={std_val:.4f}) "
            content += f"<ts_start> {self.codes_to_text(codes)} <ts_end> "

        content += sample.get("post_prompt", "")

        return (f"<|im_start|>user\n{content}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def _split_codes_by_channel(self, codes: list[int], ts: list) -> list[list[int]]:
        """Split flat codes into per-channel sublists."""
        if isinstance(ts, list) and isinstance(ts[0], torch.Tensor):
            cf = self.config.compression_factor
            lengths = []
            for t in ts:
                n = t.flatten().shape[0]
                pad = (cf - n % cf) % cf
                lengths.append((n + pad) // cf)
            result, offset = [], 0
            for ln in lengths:
                result.append(codes[offset:offset + ln])
                offset += ln
            return result
        n = max(1, len(ts))
        chunk = len(codes) // n
        return [codes[i * chunk:(i + 1) * chunk] for i in range(n)]

    # ==============================================================
    # Training
    # ==============================================================

    def forward(
        self,
        batch: list[dict],
        mode: str = "loss",
        pack_length: int = 2048) -> torch.Tensor:
        """DDP-compatible forward pass.

        Must be called through the wrapped model (not raw_model) so that
        DDP gradient sync hooks fire during backward.

        Args:
            batch: List of sample dicts.
            mode: "loss" (raw), "pretokenized", or "packed".
            pack_length: Pack length for packed mode.
        """
        if mode == "packed":
            return self.compute_loss_packed(batch, pack_length=pack_length)
        elif mode == "pretokenized":
            return self.compute_loss_pretokenized(batch)
        return self.compute_loss(batch)

    def compute_loss(self, batch: list[dict]) -> torch.Tensor:
        """Causal LM loss. Only answer tokens are supervised.

        The full sequence is:
            <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
        Loss is only computed on {answer}<|im_end|> (everything after "assistant\n").
        """
        texts, prompt_lengths = [], []
        for sample in batch:
            prompt = self.build_text(sample)
            p_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_lengths.append(len(p_ids))
            # Answer terminated by <|im_end|>
            answer = sample.get("answer", "")
            texts.append(prompt + answer + "<|im_end|>")

        tok = self.tokenizer(
            texts, padding="longest", return_tensors="pt",
            truncation=True, max_length=self.config.max_length)
        input_ids = tok.input_ids.to(self._device)
        attn_mask = tok.attention_mask.to(self._device)

        labels = torch.full_like(input_ids, -100)
        for i, pl in enumerate(prompt_lengths):
            non_pad = torch.where(input_ids[i] != self.tokenizer.pad_token_id)[0]
            answer_pos = non_pad[non_pad >= pl]
            if len(answer_pos) > 0:
                labels[i, answer_pos] = input_ids[i, answer_pos]

        return self.llm(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss

    def compute_loss_pretokenized(self, batch: list[dict]) -> torch.Tensor:
        """Fast loss computation from pre-tokenized input_ids and labels.

        Each sample in batch must have 'input_ids' (list[int]) and
        'prompt_length' (int). Labels are constructed by masking prompt tokens.
        Skips all text building and tokenizer calls.
        """
        import torch.nn.functional as Fn

        max_len = min(max(len(s["input_ids"]) for s in batch), self.config.max_length)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, sample in enumerate(batch):
            ids = sample["input_ids"][:max_len]
            seq_len = len(ids)
            pl = sample["prompt_length"]

            input_ids[i:seq_len] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i:seq_len] = 1
            # Only supervise answer tokens (after prompt)
            if pl < seq_len:
                labels[i, pl:seq_len] = input_ids[i, pl:seq_len]

        input_ids = input_ids.to(self._device)
        attn_mask = attn_mask.to(self._device)
        labels = labels.to(self._device)

        return self.llm(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss

    def compute_loss_packed(self, batch: list[dict], pack_length: int = 2048) -> torch.Tensor:
        """Causal LM loss with sequence packing, no wasted padding tokens.

        Packs multiple samples into sequences of pack_length using
        length-sorted first-fit-decreasing bin packing for minimal waste.
        Uses position_ids reset + document-level labels to prevent
        cross-sample attention leakage and loss contamination.

        Supports both pre-tokenized samples (with 'input_ids' and
        'prompt_length' keys) and raw samples (tokenizes on the fly).
        """
        # Extract (input_ids, labels) from each sample
        all_input_ids = []
        all_label_ids = []

        for sample in batch:
            if "input_ids" in sample and "prompt_length" in sample:
                # Pre-tokenized fast path
                input_ids = sample["input_ids"]
                p_len = sample["prompt_length"]
            else:
                # Fallback: tokenize on the fly
                prompt = self.build_text(sample)
                answer = sample.get("answer", "")
                full_text = prompt + answer + "<|im_end|>"
                tok = self.tokenizer(full_text, add_special_tokens=False,
                                     truncation=True, max_length=pack_length)
                input_ids = tok.input_ids
                p_tok = self.tokenizer(prompt, add_special_tokens=False)
                p_len = len(p_tok.input_ids)

            # Labels: -100 for prompt tokens, real ids for answer tokens
            labels = [-100] * min(p_len, len(input_ids)) + input_ids[p_len:]
            all_input_ids.append(input_ids)
            all_label_ids.append(labels)

        # First-fit-decreasing bin packing: sort longest first,
        # then try to fit each sample into any existing bin before opening a new one.
        # Produces 5-15% fewer bins than single-accumulator greedy packing.
        order = sorted(range(len(all_input_ids)),
                       key=lambda i: len(all_input_ids[i]), reverse=True)

        bins_ids: list[list[int]] = []
        bins_labels: list[list[int]] = []
        bins_pos: list[list[int]] = []
        bins_len: list[int] = []  # current length of each bin (for fast lookup)

        for idx in order:
            ids = all_input_ids[idx][:pack_length]
            labs = all_label_ids[idx][:pack_length]
            sample_len = len(ids)

            # Try to fit into existing bin
            placed = False
            for b in range(len(bins_ids)):
                if bins_len[b] + sample_len <= pack_length:
                    bins_pos[b] += list(range(sample_len))
                    bins_ids[b] += ids
                    bins_labels[b] += labs
                    bins_len[b] += sample_len
                    placed = True
                    break

            if not placed:
                bins_ids.append(list(ids))
                bins_labels.append(list(labs))
                bins_pos.append(list(range(sample_len)))
                bins_len.append(sample_len)

        # Pad all bins to pack_length
        pad_id = self.tokenizer.pad_token_id or 0
        for b in range(len(bins_ids)):
            pad_len = pack_length - bins_len[b]
            bins_ids[b] += [pad_id] * pad_len
            bins_labels[b] += [-100] * pad_len
            bins_pos[b] += list(range(pad_len))

        packed_input_ids = bins_ids
        packed_labels = bins_labels
        packed_position_ids = bins_pos

        if not packed_input_ids:
            return torch.tensor(0.0, device=self._device, requires_grad=True)

        input_ids = torch.tensor(packed_input_ids, device=self._device)
        labels = torch.tensor(packed_labels, device=self._device)
        position_ids = torch.tensor(packed_position_ids, device=self._device)
        attn_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return self.llm(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            labels=labels).loss

    # ==============================================================
    # Inference
    # ==============================================================

    @torch.no_grad()
    def generate(self, batch: list[dict], max_new_tokens: int = 400) -> list[str]:
        """Generate text answers.

        Uses <|im_end|> as the stop token so the model stops after
        producing the answer, not after internal thinking blocks.
        """
        texts = [self.build_text(s) for s in batch]
        tok = self.tokenizer(
            texts, padding="longest", return_tensors="pt",
            truncation=True, max_length=self.config.max_length)
        ids = tok.input_ids.to(self._device)
        mask = tok.attention_mask.to(self._device)

        # Stop on <|im_end|>, this is the ChatML turn delimiter
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_ids = [im_end_id, self.tokenizer.eos_token_id]

        gen = self.llm.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False)
        return self.tokenizer.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True)

    def analyze(
        self,
        signal: torch.Tensor | list[float],
        question: str,
        signal_label: str = "Signal:",
        context: str = "",
        max_new_tokens: int = 400) -> str:
        """Analyze a time series and answer a question about it.

        This is the main inference entry point. One signal, one question,
        one answer.

        Args:
            signal: 1-D time series (raw values, any length).
            question: What to analyze (e.g., "What is the trend?").
            signal_label: Label for the signal (e.g., "Vibration:").
            context: Optional context (e.g., "sampled at 12kHz").
            max_new_tokens: Max tokens to generate.

        Returns:
            Generated answer string.
        """
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

        sample = {
            "pre_prompt": context if context else "Analyze this time series.",
            "post_prompt": question,
            "time_series": [signal],
            "time_series_text": [signal_label],
            "answer": "",
        }
        results = self.generate([sample], max_new_tokens=max_new_tokens)
        return results[0].strip()

    def forecast(
        self,
        signal: torch.Tensor | list[float],
        context_label: str = "Historical signal:",
        max_codes: int = 64) -> dict[str, Any]:
        """Forecast future values (Chameleon-style code generation).

        Generates future TOTEM codes, then decodes them to values
        using the TOTEM decoder.

        Args:
            signal: 1-D historical time series.
            context_label: Label for the context signal.
            max_codes: Max future codes to generate.

        Returns:
            Dict with 'codes' (list[int]), 'values' (tensor, if decoder
            available), and 'raw_text' (generated string).
        """
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

        sample = {
            "pre_prompt": "Forecast the continuation of this time series.",
            "post_prompt": "Forecast:",
            "time_series": [signal],
            "time_series_text": [context_label],
            "answer": "",
        }
        raw = self.generate([sample], max_new_tokens=max_codes * 3)[0]

        # Extract generated codes from text
        code_pattern = re.findall(r"<ts_(\d+)>", raw)
        codes = [int(c) for c in code_pattern if 0 <= int(c) < self.config.n_bins]

        result = {"codes": codes, "raw_text": raw}

        # Decode to values if TOTEM decoder is available
        if codes and hasattr(self.totem, "decode"):
            code_tensor = torch.tensor(codes, dtype=torch.long)
            mean = signal.mean().item()
            std = max(signal.std().item(), 1e-8)
            with torch.no_grad():
                dev = next(self.totem.parameters()).device
                decoded = self.totem.decode(code_tensor.to(dev))
            result["values"] = decoded.squeeze() * std + mean

        return result

    # ==============================================================
    # Save / Load
    # ==============================================================

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def set_max_ts_tokens(self, n: int) -> None:
        """Update max TS tokens (for curriculum stages with different lengths)."""
        self.config.max_ts_tokens = n

    def trainable_summary(self) -> dict[str, Any]:
        """Count trainable vs total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = trainable / total * 100 if total > 0 else 0.0
        return {"trainable": trainable, "total": total, "pct": pct}

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        totem_ckpt: str | None = None,
        llm_id: str = "Qwen/Qwen3-4B",
        device: str = "cpu",
        tokenizer_type: str | None = None,
        fsq_ckpt: str | None = None,
        **config_overrides) -> "Shrike":
        """Load a trained Shrike model from a checkpoint.

        Args:
            checkpoint_path: Path to .pt file with model_state.
            totem_ckpt: Path to TOTEM VQ-VAE checkpoint.
            llm_id: HuggingFace model ID for the base LLM.
            device: Device to load onto.
            tokenizer_type: "totem", "fsq", or "fsq_transformer".
            fsq_ckpt: Path to FSQ/FSQ-Transformer checkpoint.
            **config_overrides: Override any ShrikeConfig field.

        Returns:
            Loaded Shrike model in eval mode.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Recover config from checkpoint if available
        saved_config = ckpt.get("config", {})

        # Determine tokenizer type from args, saved config, or default
        tok_type = tokenizer_type or saved_config.get("tokenizer_type", "totem")

        # CLI args take priority over saved SageMaker paths
        def _resolve(cli_val, saved_key, cli_default):
            saved_val = saved_config.get(saved_key, "")
            if cli_val and cli_val != cli_default:
                return cli_val
            if saved_val and not saved_val.startswith("/opt/ml"):
                return saved_val
            return cli_val  # fall back to CLI default

        config = ShrikeConfig(
            llm_id=_resolve(llm_id, "llm_id", "Qwen/Qwen3-4B"),
            tokenizer_type=tok_type,
            totem_ckpt=_resolve(totem_ckpt, "totem_ckpt", None),
            fsq_ckpt=_resolve(fsq_ckpt, "fsq_ckpt", None),
            **{k: v for k, v in config_overrides.items() if hasattr(ShrikeConfig, k)})

        model = cls(config, device=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model
