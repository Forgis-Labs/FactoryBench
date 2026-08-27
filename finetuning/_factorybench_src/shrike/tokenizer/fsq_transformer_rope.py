"""FSQ Transformer Tokenizer with RoPE, variable-length time series tokenization.

Drop-in replacement for FSQTransformerTokenizer that uses Rotary Position
Embeddings instead of learned absolute position embeddings. This allows
tokenizing signals of any length without retraining, a 128-point HAR signal
and a 2048-point bearing signal both get correct positional information.

Architecture is identical to fsq_transformer.py except:
  - Learned pos_embed replaced with RoPE applied to Q,K in attention
  - Custom TransformerEncoderLayer to inject RoPE into attention
  - No maximum sequence length (was capped at 1024 patches)

Usage:
    from shrike.tokenizer.fsq_transformer_rope import FSQTransformerRoPETokenizer

    tok = FSQTransformerRoPETokenizer(config)
    codes = tok.tokenize(signal)     # works for ANY signal length
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsq_transformer import FSQTransformerConfig, FSQLayer, PatchEmbedding, PatchReconstruct


# ══════════════════════════════════════════════════════════════════════════════
# RoPE Implementation
# ══════════════════════════════════════════════════════════════════════════════

def _build_rope_freqs(dim: int, max_len: int = 8192, theta: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor.

    Returns complex exponentials: e^(i * pos * freq) for each (position, dim_pair).
    Shape: (max_len, dim // 2) complex64.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # (max_len, dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a tensor.

    Args:
        x: (batch, n_heads, seq_len, head_dim), query or key
        freqs: (seq_len, head_dim // 2) complex, from _build_rope_freqs

    Returns:
        Rotated tensor, same shape as x.
    """
    # View as pairs of floats → complex
    b, h, s, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, h, s, d // 2, 2))
    # Broadcast freqs to (1, 1, seq_len, head_dim//2)
    freqs = freqs[:s].unsqueeze(0).unsqueeze(0)
    # Rotate and convert back to real
    rotated = torch.view_as_real(x_complex * freqs).reshape(b, h, s, d)
    return rotated.to(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Custom Attention with RoPE
# ══════════════════════════════════════════════════════════════════════════════

class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings.

    Standard MHA but applies RoPE to queries and keys before computing
    attention scores. Values are left unrotated (standard RoPE practice).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Precompute RoPE frequencies (registered as buffer, not parameter)
        freqs = _build_rope_freqs(self.head_dim)
        self.register_buffer("rope_freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, b, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        q = _apply_rope(q, self.rope_freqs)
        k = _apply_rope(k, self.rope_freqs)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.out_proj(out)


class RoPETransformerLayer(nn.Module):
    """Transformer encoder layer with RoPE attention.

    Pre-norm architecture (LayerNorm before attention and FFN),
    matching the original FSQTransformerTokenizer's convention.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RoPETransformerEncoder(nn.Module):
    """Stack of RoPE transformer layers."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════════════
# FSQ Transformer with RoPE
# ══════════════════════════════════════════════════════════════════════════════

class FSQTransformerRoPETokenizer(nn.Module):
    """Transformer-based FSQ tokenizer with Rotary Position Embeddings.

    Identical to FSQTransformerTokenizer except:
    - No learned positional embeddings (pos_embed removed)
    - RoPE applied inside attention layers to Q and K
    - Supports any input length without retraining

    Can be initialized from a pretrained FSQTransformerTokenizer checkpoint
    by loading matching weights and ignoring pos_embed.
    """

    def __init__(self, config: FSQTransformerConfig | None = None):
        super().__init__()
        if config is None:
            config = FSQTransformerConfig()
        self.config = config

        # Patch embedding (same as original)
        self.patch_embed = PatchEmbedding(config.patch_size, config.d_model)

        # RoPE Transformer encoder (replaces nn.TransformerEncoder + pos_embed)
        self.encoder = RoPETransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_encoder_layers,
            dropout=config.dropout)

        # Project to FSQ dimension
        self.to_fsq = nn.Linear(config.d_model, config.fsq_dim)

        # FSQ quantization
        self.fsq = FSQLayer(config.levels)

        # Project back from FSQ to d_model
        self.from_fsq = nn.Linear(config.fsq_dim, config.d_model)

        # RoPE Transformer decoder
        self.decoder = RoPETransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_decoder_layers,
            dropout=config.dropout)

        # Reconstruct patches
        self.patch_reconstruct = PatchReconstruct(config.patch_size, config.d_model)

    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode raw signal to FSQ codes.

        Args:
            x: (batch, seq_len) raw signal, ANY length (must be divisible by patch_size)

        Returns:
            quantized, indices, z_pre
        """
        h = self.patch_embed(x)  # (b, n_patches, d_model)
        # No positional embedding added, RoPE handles it inside attention
        h = self.encoder(h)
        z = self.to_fsq(h)
        quantized, indices = self.fsq(z)
        return quantized, indices, z

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized vectors back to signal."""
        h = self.from_fsq(quantized)
        # No positional embedding, RoPE inside decoder attention
        h = self.decoder(h)
        return self.patch_reconstruct(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        quantized, indices, z_pre = self.encode(x)
        recon = self.decode(quantized)
        return recon, indices, z_pre

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal to flat code IDs. Works for any signal length."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        _, indices, _ = self.encode(x)
        return self.fsq.codes_to_flat(indices)

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode flat code IDs back to signal."""
        indices = self.fsq.flat_to_codes(codes)
        half_width = self.fsq._levels / 2
        quantized_raw = indices.float() - half_width.unsqueeze(0).unsqueeze(0)
        quantized_norm = quantized_raw / (half_width.unsqueeze(0).unsqueeze(0) - 0.5)
        return self.decode(quantized_norm)

    def temporal_smoothness_loss(self, z_pre: torch.Tensor) -> torch.Tensor:
        """Penalize rapid changes in pre-quantization latents."""
        diff = z_pre[:, 1::] - z_pre[::-1:]
        return diff.abs().mean()

    def trainable_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: str = "cpu") -> "FSQTransformerRoPETokenizer":
        """Load from checkpoint.

        Can load from either:
        - A RoPE checkpoint (exact match)
        - An original FSQTransformerTokenizer checkpoint (ignores pos_embed,
          maps nn.TransformerEncoder weights to RoPETransformerEncoder)
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})

        config = FSQTransformerConfig(
            levels=cfg_dict.get("levels", [7, 7, 7, 7]),
            patch_size=cfg_dict.get("patch_size", 4),
            input_length=cfg_dict.get("input_length", 256),
            d_model=cfg_dict.get("d_model", 128),
            n_heads=cfg_dict.get("n_heads", 4),
            n_encoder_layers=cfg_dict.get("n_encoder_layers", 4),
            n_decoder_layers=cfg_dict.get("n_decoder_layers", 4),
            d_ff=cfg_dict.get("d_ff", 512),
            dropout=cfg_dict.get("dropout", 0.0),
            temporal_weight=cfg_dict.get("temporal_weight", 0.1))

        model = cls(config).to(device)

        state = ckpt.get("model_state", ckpt.get("state_dict", {}))

        # Try direct load first (RoPE checkpoint)
        try:
            model.load_state_dict(state, strict=True)
            print(f"FSQ-Transformer-RoPE loaded (exact): {config.codebook_size} codes, "
                  f"levels={config.levels}")
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model
        except RuntimeError:
            pass

        # Fallback: load from original FSQTransformerTokenizer checkpoint
        # Map old nn.TransformerEncoder keys to new RoPETransformerEncoder keys
        mapped = {}
        for k, v in state.items():
            new_k = k

            # Skip pos_embed (RoPE doesn't use it)
            if "pos_embed" in k:
                continue

            # Map encoder.layers.N.self_attn.in_proj_{weight,bias} → encoder.layers.N.attn.qkv.{weight,bias}
            if ".self_attn.in_proj_weight" in k:
                new_k = k.replace(".self_attn.in_proj_weight", ".attn.qkv.weight")
            elif ".self_attn.in_proj_bias" in k:
                new_k = k.replace(".self_attn.in_proj_bias", ".attn.qkv.bias")
            elif ".self_attn.out_proj." in k:
                new_k = k.replace(".self_attn.out_proj.", ".attn.out_proj.")
            # Map FFN: linear1 → ffn.0, linear2 → ffn.3
            elif ".linear1.weight" in k:
                new_k = k.replace(".linear1.weight", ".ffn.0.weight")
            elif ".linear1.bias" in k:
                new_k = k.replace(".linear1.bias", ".ffn.0.bias")
            elif ".linear2.weight" in k:
                new_k = k.replace(".linear2.weight", ".ffn.3.weight")
            elif ".linear2.bias" in k:
                new_k = k.replace(".linear2.bias", ".ffn.3.bias")
            # Map norms: norm1/norm2 stay the same name
            # Map encoder.norm → encoder.norm (final layernorm)

            mapped[new_k] = v

        # Load with strict=False (rope_freqs are buffers, not in checkpoint)
        missing, unexpected = model.load_state_dict(mapped, strict=False)

        # Filter out expected missing/unexpected keys
        missing_real = [k for k in missing if "rope_freqs" not in k]
        if missing_real:
            print(f"  WARNING: Missing keys: {missing_real[:5]}")
        if unexpected:
            print(f"  NOTE: Skipped keys from old checkpoint: {unexpected[:5]}")

        n_loaded = len(mapped) - len(unexpected)
        print(f"FSQ-Transformer-RoPE loaded (migrated from abs PE): {config.codebook_size} codes, "
              f"levels={config.levels}, {n_loaded} weights loaded")

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
