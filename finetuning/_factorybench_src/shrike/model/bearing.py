"""Bearing diagnosis model, loads NanoHyperion checkpoints.

The hyperion_v4 bearing model was trained with the NanoHyperion class
(attention-only LoRA, TOTEM 256 codes, Qwen3-4B). This module wraps
NanoHyperion with an `analyze()` method matching the Shrike API so
the qualitative experiments script works with either model class.

Usage:
    from shrike.model.bearing import BearingModel

    model = BearingModel.from_pretrained(
        "checkpoints/hyperion_v4_multi5bearing/best_model.pt",
        totem_ckpt="checkpoints/totem_clean_local.pt",
        llm_id="Qwen/Qwen3-4B",
        device="cuda")
    answer = model.analyze(signal, "Diagnose this bearing.", context="SKF 6205...")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from shrike.tokenizer.totem import TOTEMTokenizer


@dataclass
class BearingConfig:
    """Config matching the NanoHyperion setup used for hyperion_v4."""

    llm_id: str = "Qwen/Qwen3-4B"
    totem_ckpt: str | None = None
    n_bins: int = 256
    compression_factor: int = 4
    max_ts_tokens: int = 512
    max_length: int = 2048
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    use_dora: bool = True
    neftune_alpha: float = 0.0


class BearingModel(nn.Module):
    """NanoHyperion-compatible model for bearing diagnosis.

    Identical architecture to the original NanoHyperion class:
    TOTEM 256 VQ-VAE + Qwen3-4B + DoRA r=32 (attention-only).
    """

    def __init__(self, config: BearingConfig, device: str = "cuda") -> None:
        super().__init__()
        self.config = config
        self.device_str = device

        # --- TOTEM tokenizer (frozen, CPU) ---
        if config.totem_ckpt and os.path.exists(config.totem_ckpt):
            self.totem = TOTEMTokenizer.from_pretrained(config.totem_ckpt, device="cpu")
        else:
            self.totem = TOTEMTokenizer(num_embeddings=config.n_bins)
            self.totem.eval()
            self.totem.requires_grad_(False)

        # --- LLM + text tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_id)

        ts_tokens = [f"<ts_{i}>" for i in range(config.n_bins)]
        ts_tokens += ["<ts_start>", "<ts_end>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": ts_tokens})
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        # --- Init TS embeddings from TOTEM codebook ---
        llm_embed = self.llm.get_input_embeddings()
        llm_dim = llm_embed.weight.shape[1]
        n_special = config.n_bins + 2
        original_vocab = len(self.tokenizer) - n_special

        codebook = self.totem.codebook.float()
        totem_dim = self.totem.codebook_dim
        torch.manual_seed(42)
        proj = torch.randn(totem_dim, llm_dim) / (totem_dim ** 0.5)
        projected = (codebook @ proj).to(torch.bfloat16)

        existing_norm = llm_embed.weight.data[:original_vocab].float().norm(dim=1).mean()
        proj_norm = projected.float().norm(dim=1).mean()
        if proj_norm > 0:
            projected = projected * (existing_norm / proj_norm)

        with torch.no_grad():
            llm_embed.weight.data[original_vocab:original_vocab + config.n_bins] = projected
            for offset in range(config.n_bins, n_special):
                rand_e = torch.randn(llm_dim, dtype=torch.bfloat16) * (existing_norm / (llm_dim ** 0.5))
                llm_embed.weight.data[original_vocab + offset] = rand_e

        # --- Freeze LLM, apply LoRA ---
        self.llm.requires_grad_(False)

        lora_kwargs = {}
        if config.use_dora:
            lora_kwargs["use_dora"] = True

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            **lora_kwargs)
        self.llm = get_peft_model(self.llm, lora_config)

        # --- Gradient mask ---
        self._ts_start = original_vocab
        self._ts_end = original_vocab + n_special
        embed = self.llm.get_input_embeddings()
        embed.weight.requires_grad = True

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            mask[self._ts_start:self._ts_end] = 1.0
            return grad * mask

        embed.weight.register_hook(_mask_grad)

    # ── Tokenization ──────────────────────────────────────────────

    def tokenize_ts(self, ts_values: torch.Tensor) -> list[int]:
        """Tokenize a time series through the frozen TOTEM encoder."""
        ts = ts_values.float().flatten()
        mean = ts.mean()
        std = ts.std()
        if std < 1e-8:
            std = torch.tensor(1.0)
        ts_norm = (ts - mean) / std

        max_raw = self.config.max_ts_tokens * self.config.compression_factor
        if len(ts_norm) > max_raw:
            indices = torch.linspace(0, len(ts_norm) - 1, max_raw).long()
            ts_norm = ts_norm[indices]

        with torch.no_grad():
            totem_device = next(self.totem.parameters()).device
            token_ids = self.totem.tokenize(ts_norm.to(totem_device))
        return token_ids.squeeze(0).tolist()

    @staticmethod
    def _bins_to_text(bins: list[int]) -> str:
        return " ".join(f"<ts_{b}>" for b in bins)

    # ── Prompt building ───────────────────────────────────────────

    def _build_content(self, sample: dict) -> str:
        """Build the inner content (pre_prompt + ts + stats + post_prompt)."""
        ts = sample["time_series"]

        if isinstance(ts, list) and len(ts) == 0:
            text = sample.get("pre_prompt", "")
            if text:
                text += " "
            text += sample.get("post_prompt", "")
            return text

        if isinstance(ts, torch.Tensor):
            ts_flat = ts.flatten()
        elif isinstance(ts, list) and len(ts) > 0 and isinstance(ts[0], torch.Tensor):
            ts_flat = torch.cat(ts).flatten()
        else:
            ts_flat = torch.tensor(ts, dtype=torch.float32).flatten()

        bins = self.tokenize_ts(ts_flat)
        ts_text = self._bins_to_text(bins)

        mean_val = ts_flat.mean().item()
        std_val = ts_flat.std().item()

        ts_descs = sample.get("time_series_text", [])
        text = sample["pre_prompt"] + " "

        if len(ts_descs) == 1:
            text += f"{ts_descs[0]} <ts_start> {ts_text} <ts_end> "
        else:
            text += f"<ts_start> {ts_text} <ts_end> "

        text += f"(mean={mean_val:.4f}, std={std_val:.4f}) "
        text += sample["post_prompt"]
        return text

    def build_text(self, sample: dict, use_chatml: bool = False) -> str:
        """Build prompt, optionally wrapped in ChatML for Qwen3 think mode.

        Args:
            sample: Dict with time_series, pre_prompt, post_prompt.
            use_chatml: If True, wrap in ChatML tags so Qwen3 can use
                        its native <think> reasoning before answering.
        """
        content = self._build_content(sample)
        if use_chatml:
            return (
                f"<|im_start|>system\n"
                f"You are an expert vibration analyst. Think step by step about "
                f"the signal evidence before giving your diagnosis.<|im_end|>\n"
                f"<|im_start|>user\n{content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return content

    # ── Generation ────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        batch: list[dict],
        max_new_tokens: int = 400,
        use_chatml: bool = False) -> list[str]:
        texts = [self.build_text(s, use_chatml=use_chatml) for s in batch]
        tokenized = self.tokenizer(
            texts, padding="longest", return_tensors="pt",
            truncation=True, max_length=self.config.max_length)
        input_ids = tokenized.input_ids.to(self.device_str)
        attention_mask = tokenized.attention_mask.to(self.device_str)

        # In ChatML mode, stop on <|im_end|> (turn delimiter)
        if use_chatml:
            im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            eos_ids = [im_end_id, self.tokenizer.eos_token_id]
        else:
            eos_ids = self.tokenizer.eos_token_id

        gen_ids = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=self.tokenizer.pad_token_id)
        answer_ids = gen_ids[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)

    # ── analyze(), same API as Shrike ────────────────────────────

    def analyze(
        self,
        signal: torch.Tensor | list[float],
        question: str,
        signal_label: str = "Signal:",
        context: str = "",
        max_new_tokens: int = 400,
        use_chatml: bool = False) -> str:
        """Analyze a time series and answer a question about it.

        Args:
            use_chatml: If True, wrap prompt in ChatML with a system message
                        that encourages Qwen3's <think> reasoning mode.
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
        results = self.generate(
            [sample], max_new_tokens=max_new_tokens, use_chatml=use_chatml
        )
        return results[0].strip()

    # ── Load checkpoint ───────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        totem_ckpt: str | None = None,
        llm_id: str = "Qwen/Qwen3-4B",
        device: str = "cpu") -> "BearingModel":
        """Load a NanoHyperion/hyperion checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config = BearingConfig(
            llm_id=llm_id,
            totem_ckpt=totem_ckpt)

        model = cls(config, device=device)

        # Handle vocab size mismatch (tokenizer version drift)
        state = ckpt["model_state"]
        current_state = model.state_dict()

        for key in list(state.keys()):
            if key in current_state and state[key].shape != current_state[key].shape:
                saved_w = state[key]
                current_w = current_state[key]
                n = min(saved_w.shape[0], current_w.shape[0])
                current_w[:n] = saved_w[:n]
                print(f"  Patched {key}: {saved_w.shape[0]} -> {current_w.shape[0]} rows")
                state[key] = current_w

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys: {missing[:5]}")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}")

        model.to(device)
        model.eval()
        return model
